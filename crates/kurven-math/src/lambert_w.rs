//! The principal branch of the Lambert W function: scipy's algorithm, with a
//! tighter stopping tolerance than its ufunc default. The port of
//! `LambertW.swift`.

#![allow(clippy::excessive_precision)]

use core::f64::consts::E;
use std::sync::LazyLock;

use crate::complex::Complex;
use crate::gamma::Gamma;
use crate::real;

pub enum LambertW {}

static EXP_MINUS_ONE: LazyLock<f64> = LazyLock::new(|| real::exp(-1.0));

impl LambertW {
    /// W₀(z), to the default tolerance 1e-12 (Swift's default argument).
    pub fn w0(z: Complex) -> Complex {
        LambertW::w0_to(z, 1e-12)
    }

    pub fn w0_to(z: Complex, tolerance: f64) -> Complex {
        let exp_minus_one = *EXP_MINUS_ONE;
        if z.re == 0.0 && z.im == 0.0 {
            return Complex::ZERO;
        }
        if z.re == -exp_minus_one && z.im == 0.0 {
            return Complex::new(-1.0, 0.0);
        }
        let mut w: Complex;
        if (z + exp_minus_one).magnitude() < 0.3 {
            // Series about the branch point.
            let p = Complex::sqrt(2.0 * (E * z + 1.0));
            w = -1.0 + p - p * p / 3.0;
        } else if -1.0 < z.re && z.re < 1.5 && z.im.abs() < 1.0 && -2.5 * z.im.abs() - 0.2 < z.re {
            // Padé approximant about the origin.
            let num = [12.85106382978723404255, 12.34042553191489361902, 1.0];
            let den = [32.53191489361702127660, 14.34042553191489361702, 1.0];
            w = z * Gamma::eval_poly(&num, z) / Gamma::eval_poly(&den, z);
        } else {
            w = Complex::log(z);
            w = w - Complex::log(w);
        }
        // Halley's iteration, in the form that keeps its exponential bounded.
        if w.re >= 0.0 {
            for _ in 0..100 {
                let ew = Complex::exp(-w);
                let wewz = w - z * ew;
                let wn = w - wewz / (w + 1.0 - (w + 2.0) * wewz / (2.0 * w + 2.0));
                if (wn - w).magnitude() <= tolerance * wn.magnitude() {
                    return wn;
                }
                w = wn;
            }
        } else {
            for _ in 0..100 {
                let ew = Complex::exp(w);
                let wew = w * ew;
                let wewz = wew - z;
                let wn = w - wewz / (wew + ew - (w + 2.0) * wewz / (2.0 * w + 2.0));
                if (wn - w).magnitude() <= tolerance * wn.magnitude() {
                    return wn;
                }
                w = wn;
            }
        }
        w
    }
}
