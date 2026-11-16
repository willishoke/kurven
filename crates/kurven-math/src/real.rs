//! The transcendental functions of a real argument, the only way the engine
//! reaches them.
//!
//! Rust's `f64::sin` and its kin call the platform's libm on native targets
//! and a compiled-in one on wasm, and the two disagree in the last bit. A
//! drawing decided by a bisection on such a value can then fall the other
//! way, and native would no longer stand for the browser. So every engine
//! crate calls these, which call the `libm` crate, pure Rust, on every
//! target: native and wasm compute the same bits, and the check runs natively
//! for both. `clippy.toml` refuses the std spellings and
//! `scripts/check-web.sh` greps for them.
//!
//! The names are C's, as Swift's Foundation spells them (`log`, not `ln`),
//! so a call reads the same in the port as in `Sources/`.
//!
//! The four operations and `sqrt` are correctly rounded everywhere and need
//! no door; [`sqrt`] is here only so a module that imports this one has the
//! full set. The port does not try to match Darwin's libm, which the Swift
//! reference calls: against Swift, the comparison is to a tolerance.

#[inline]
pub fn sin(x: f64) -> f64 {
    libm::sin(x)
}

#[inline]
pub fn cos(x: f64) -> f64 {
    libm::cos(x)
}

#[inline]
pub fn tan(x: f64) -> f64 {
    libm::tan(x)
}

#[inline]
pub fn asin(x: f64) -> f64 {
    libm::asin(x)
}

#[inline]
pub fn acos(x: f64) -> f64 {
    libm::acos(x)
}

#[inline]
pub fn atan(x: f64) -> f64 {
    libm::atan(x)
}

/// The angle of (x, y), in (−π, π]; C's argument order, y first.
#[inline]
pub fn atan2(y: f64, x: f64) -> f64 {
    libm::atan2(y, x)
}

#[inline]
pub fn sinh(x: f64) -> f64 {
    libm::sinh(x)
}

#[inline]
pub fn cosh(x: f64) -> f64 {
    libm::cosh(x)
}

#[inline]
pub fn tanh(x: f64) -> f64 {
    libm::tanh(x)
}

#[inline]
pub fn asinh(x: f64) -> f64 {
    libm::asinh(x)
}

#[inline]
pub fn acosh(x: f64) -> f64 {
    libm::acosh(x)
}

#[inline]
pub fn atanh(x: f64) -> f64 {
    libm::atanh(x)
}

#[inline]
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}

#[inline]
pub fn exp2(x: f64) -> f64 {
    libm::exp2(x)
}

/// eˣ − 1, accurate where x is small.
#[inline]
pub fn expm1(x: f64) -> f64 {
    libm::expm1(x)
}

/// The natural logarithm.
#[inline]
pub fn log(x: f64) -> f64 {
    libm::log(x)
}

#[inline]
pub fn log2(x: f64) -> f64 {
    libm::log2(x)
}

#[inline]
pub fn log10(x: f64) -> f64 {
    libm::log10(x)
}

/// log(1 + x), accurate where x is small.
#[inline]
pub fn log1p(x: f64) -> f64 {
    libm::log1p(x)
}

#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}

/// √(x² + y²) without overflow or underflow on the way.
#[inline]
pub fn hypot(x: f64, y: f64) -> f64 {
    libm::hypot(x, y)
}

#[inline]
pub fn cbrt(x: f64) -> f64 {
    libm::cbrt(x)
}

#[inline]
pub fn erf(x: f64) -> f64 {
    libm::erf(x)
}

#[inline]
pub fn erfc(x: f64) -> f64 {
    libm::erfc(x)
}

/// C's `fmod`: x − n y with n = trunc(x / y), the sign of x. Exact.
#[inline]
pub fn fmod(x: f64, y: f64) -> f64 {
    libm::fmod(x, y)
}

/// C's `remainder`: x − n y with n the integer nearest x / y, ties to even.
/// Exact.
#[inline]
pub fn remainder(x: f64, y: f64) -> f64 {
    libm::remainder(x, y)
}

/// The magnitude of x with the sign of y. Exact.
#[inline]
pub fn copysign(x: f64, y: f64) -> f64 {
    libm::copysign(x, y)
}

/// Correctly rounded on every target, as IEEE 754 requires.
#[inline]
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values these functions return exactly, by IEEE 754 or by C's
    /// specification of the function.
    #[test]
    fn exact_values() {
        assert_eq!(sin(0.0), 0.0);
        assert!(sin(-0.0).is_sign_negative());
        assert_eq!(cos(0.0), 1.0);
        assert_eq!(tan(0.0), 0.0);
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp2(10.0), 1024.0);
        assert_eq!(log(1.0), 0.0);
        assert_eq!(log2(1024.0), 10.0);
        assert_eq!(log10(1000.0), 3.0);
        assert_eq!(pow(2.0, 10.0), 1024.0);
        assert_eq!(pow(0.0, 0.0), 1.0);
        assert_eq!(hypot(3.0, 4.0), 5.0);
        assert_eq!(cbrt(27.0), 3.0);
        assert_eq!(cbrt(-8.0), -2.0);
        assert_eq!(atan2(0.0, -1.0), core::f64::consts::PI);
        assert_eq!(atan2(-0.0, -1.0), -core::f64::consts::PI);
        assert_eq!(fmod(7.5, 2.0), 1.5);
        assert_eq!(fmod(-7.5, 2.0), -1.5);
        assert_eq!(remainder(7.5, 2.0), -0.5);
        assert_eq!(copysign(3.0, -0.0), -3.0);
        assert_eq!(sqrt(2.25), 1.5);
        assert_eq!(erf(0.0), 0.0);
        assert_eq!(erfc(0.0), 1.0);
        assert_eq!(expm1(0.0), 0.0);
        assert_eq!(log1p(0.0), 0.0);
    }

    /// Correctly rounded or within an ulp of the true value, at points whose
    /// true value is known.
    #[test]
    fn known_values_to_an_ulp() {
        use core::f64::consts::{E, FRAC_PI_2, FRAC_PI_4, LN_2};
        fn ulps(a: f64, b: f64) -> u64 {
            (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs()
        }
        assert!(ulps(atan(1.0), FRAC_PI_4) <= 1);
        assert!(ulps(asin(1.0), FRAC_PI_2) <= 1);
        assert!(ulps(acos(0.0), FRAC_PI_2) <= 1);
        assert!(ulps(exp(1.0), E) <= 1);
        assert!(ulps(log(2.0), LN_2) <= 1);
        assert!(ulps(sinh(LN_2), 0.75) <= 1);
        assert!(ulps(cosh(LN_2), 1.25) <= 1);
        assert!(ulps(tanh(LN_2), 0.6) <= 1);
        assert!(ulps(asinh(0.75), LN_2) <= 1);
        assert!(ulps(acosh(1.25), LN_2) <= 1);
        assert!(ulps(atanh(0.6), LN_2) <= 1);
    }

    #[test]
    fn domains() {
        assert!(log(-1.0).is_nan());
        assert_eq!(log(0.0), f64::NEG_INFINITY);
        assert!(sqrt(-1.0).is_nan());
        assert!(acosh(0.5).is_nan());
        assert_eq!(exp(1000.0), f64::INFINITY);
        assert_eq!(exp(-1000.0), 0.0);
    }
}
