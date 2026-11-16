//! The Jacobi elliptic functions sn, cn, dn of a complex argument. The port of
//! `Jacobi.swift`.
//!
//! The real ones are cephes' `ellpj` (scipy's `ellipj`): the arithmetic-
//! geometric mean, with the two ends of the parameter range handled by their
//! own expansions. The imaginary direction comes from the addition formulas
//! with the complementary parameter, exactly as `kurven.expr._jacobi` writes
//! them, so the two sides agree to rounding rather than to a tolerance.

use core::f64::consts::PI;

use crate::complex::Complex;
use crate::real;

pub enum Jacobi {}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Real {
    pub sn: f64,
    pub cn: f64,
    pub dn: f64,
    pub ph: f64,
}

/// sn, cn and dn at one complex point: Swift's `(sn:, cn:, dn:)` tuple.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Elliptic {
    pub sn: Complex,
    pub cn: Complex,
    pub dn: Complex,
}

/// Swift's `Double.ulp` of a finite value: the gap from its magnitude to the
/// next double up (the least subnormal at zero), and 2^971 at the largest
/// finite double, where there is no next one. NaN otherwise.
pub fn ulp(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    let a = x.abs();
    if a == f64::MAX {
        return a - a.next_down();
    }
    a.next_up() - a
}

impl Jacobi {
    /// The complete elliptic integral of the first kind, K(m): the quarter
    /// period of sn and cn along the real axis. `quarter_period(1 - m)` is
    /// K', the one along the imaginary axis.
    ///
    /// π / (2 AGM(1, √(1 − m))). The mean converges quadratically, so the loop
    /// runs until the two terms agree to the last bit -- five or six rounds
    /// for any m short of 1, where K diverges.
    pub fn quarter_period(m: f64) -> f64 {
        if !(0.0..1.0).contains(&m) {
            return if m == 1.0 { f64::INFINITY } else { f64::NAN };
        }
        let mut a = 1.0;
        let mut b = real::sqrt(1.0 - m);
        for _ in 0..64 {
            if a == b {
                continue;
            }
            (a, b) = (0.5 * (a + b), real::sqrt(a * b));
            if (a - b).abs() <= ulp(a) {
                break;
            }
        }
        PI / (2.0 * a)
    }

    /// `ellpj(u, m)` for 0 <= m <= 1.
    pub fn real(u: f64, m: f64) -> Real {
        if m < 0.0 || m > 1.0 || m.is_nan() {
            return Real { sn: f64::NAN, cn: f64::NAN, dn: f64::NAN, ph: f64::NAN };
        }
        if m < 1e-9 {
            let t = real::sin(u);
            let b = real::cos(u);
            let ai = 0.25 * m * (u - t * b);
            return Real { sn: t - ai * b, cn: b + ai * t, dn: 1.0 - 0.5 * m * t * t, ph: u - ai };
        }
        if m >= 0.9999999999 {
            let mut ai = 0.25 * (1.0 - m);
            let b = real::cosh(u);
            let t = real::tanh(u);
            let phi = 1.0 / b;
            let twon = b * real::sinh(u);
            let sn = t + ai * (twon - u) / (b * b);
            let ph = 2.0 * real::atan(real::exp(u)) - PI / 2.0 + ai * (twon - u) / b;
            ai *= t * phi;
            return Real { sn, cn: phi - ai * (twon - u), dn: phi + ai * (twon + u), ph };
        }
        // The mean's two tables, nine terms each, on the stack: this is
        // called once per evaluation of sn, and an evaluation is the unit
        // the contour refiner and the hidden-line traversal spend by the
        // million.
        let mut a = [1.0f64; 9];
        let mut c = [0.0f64; 9];
        c[0] = real::sqrt(m);
        let mut b = real::sqrt(1.0 - m);
        let mut twon = 1.0;
        let mut i = 0;
        while (c[i] / a[i]).abs() > 2.220446049250313e-16 {
            if i > 7 {
                break;
            }
            let ai = a[i];
            i += 1;
            c[i] = (ai - b) / 2.0;
            let t = real::sqrt(ai * b);
            a[i] = (ai + b) / 2.0;
            b = t;
            twon *= 2.0;
        }
        let mut phi = twon * a[i] * u;
        let mut bb = 0.0;
        while i > 0 {
            let t = c[i] * real::sin(phi) / a[i];
            bb = phi;
            phi = (real::asin(t) + phi) / 2.0;
            i -= 1;
        }
        let sn = real::sin(phi);
        let t = real::cos(phi);
        let dnfac = real::cos(phi - bb);
        let dn = if dnfac.abs() < 0.1 { real::sqrt(1.0 - m * sn * sn) } else { t / dnfac };
        Real { sn, cn: t, dn, ph: phi }
    }

    /// (sn, cn, dn)(z, m) for complex z.
    pub fn complex(z: Complex, m: f64) -> Elliptic {
        let u = Jacobi::real(z.re, m);
        let v = Jacobi::real(z.im, 1.0 - m);
        let (s, c, d) = (u.sn, u.cn, u.dn);
        let (s1, c1, d1) = (v.sn, v.cn, v.dn);
        let den = c1 * c1 + m * s * s * s1 * s1;
        let sn = Complex::new(s * d1, c * d * s1 * c1) / den;
        let cn = Complex::new(c * c1, -s * d * s1 * d1) / den;
        let dn = Complex::new(d * c1 * d1, -m * s * c * s1) / den;
        Elliptic { sn, cn, dn }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulp_is_swifts() {
        assert_eq!(ulp(1.0), f64::EPSILON);
        assert_eq!(ulp(-1.0), f64::EPSILON);
        assert_eq!(ulp(0.0), f64::from_bits(1));
        assert_eq!(ulp(f64::MAX), real::pow(2.0, 971.0));
        assert_eq!(ulp(1.5), f64::EPSILON);
        assert!(ulp(f64::INFINITY).is_nan());
    }
}
