//! The exponential integral E₁(z), and Ei as `kurven.expr` continues it. The
//! port of `ExpInt.swift`.

use core::f64::consts::PI;

use crate::complex::Complex;
use crate::gamma::Gamma;
use crate::real;

pub enum ExpInt {}

impl ExpInt {
    /// E₁(z), cut along the negative real axis.
    ///
    /// The Maclaurin series where it does not cancel -- it alternates only
    /// through the imaginary direction, so its loss is e^(|z| - |Re z|) on
    /// the left, which is harmless near the negative real axis, exactly where
    /// the continued fraction converges slowest -- and the continued fraction
    /// `e^{-z}/(z + 1/(1 + 1/(z + 2/(1 + 2/(z + ...)))))` by modified Lentz
    /// everywhere else.
    pub fn e1(z: Complex) -> Complex {
        let a0 = z.magnitude();
        if a0 == 0.0 {
            return Complex::new(f64::INFINITY, 0.0);
        }
        if a0 <= 10.0 || (z.re < 0.0 && z.im.abs() <= 12.0 && a0 <= 60.0) {
            let mut total = Complex::ONE;
            let mut cr = Complex::ONE;
            for k in 1..400 {
                cr = -cr * f64::from(k) * z / f64::from((k + 1) * (k + 1));
                total += cr;
                if cr.magnitude() <= total.magnitude() * 1e-16 {
                    break;
                }
            }
            return -Gamma::EULER - Complex::log(z) + z * total;
        }
        let tiny = 1e-300;
        let mut f = z;
        let mut c = f;
        let mut d = Complex::ZERO;
        let mut k = 1;
        for i in 1..2000 {
            let a = f64::from(k);
            let b = if i % 2 == 1 { Complex::ONE } else { z };
            d = b + a * d;
            if d.re == 0.0 && d.im == 0.0 {
                d = Complex::real(tiny);
            }
            c = b + a / c;
            if c.re == 0.0 && c.im == 0.0 {
                c = Complex::real(tiny);
            }
            d = 1.0 / d;
            let delta = c * d;
            f *= delta;
            if i % 2 == 0 {
                k += 1;
            }
            if (delta - 1.0).magnitude() < 1e-16 {
                break;
            }
        }
        let mut result = Complex::exp(-z) / f;
        if z.re <= 0.0 && z.im == 0.0 {
            result -= Complex::new(0.0, if !real::sign_minus(z.im) { PI } else { -PI });
        }
        result
    }

    /// `expi(z) = -E₁(-z)`, the language's continuation of Ei -- including
    /// the ∓iπ it carries on the positive real axis, which is what the Python
    /// side evaluates and what the fixtures pin.
    pub fn expi(z: Complex) -> Complex {
        -ExpInt::e1(-z)
    }
}
