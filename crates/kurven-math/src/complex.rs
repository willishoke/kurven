//! A complex number in double precision, with the elementary functions on the
//! principal branches numpy uses. The port of `Complex.swift`.
//!
//! The frontend evaluates the expression language itself, so every function
//! the language offers has to exist here, and has to agree with numpy and
//! scipy at every point of the plane -- including *on* the branch cuts, where
//! the sign of a zero imaginary part decides which side of the cut a value
//! comes from. The formulas below are written component by component where
//! that matters (`1 - z*z` is not `Complex(1 - (x*x - y*y), -2*x*y)` under
//! IEEE rules), and `tests/fixtures/expr` holds them to numpy's answers on a
//! grid that runs along the real axis at exactly `im = 0`.
//!
//! The functions are associated functions, `Complex::exp(z)`, as Swift's are
//! statics (`Complex.exp(z)`); called that way they never spell a method of
//! f64's, which `scripts/check-web.sh` refuses outside `real`.

use core::f64::consts::PI;
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::format::description;
use crate::real;

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };
    pub const ONE: Complex = Complex { re: 1.0, im: 0.0 };
    pub const I: Complex = Complex { re: 0.0, im: 1.0 };

    #[inline]
    pub const fn new(re: f64, im: f64) -> Complex {
        Complex { re, im }
    }

    /// A real number as a complex one, `Complex(x)` in Swift.
    #[inline]
    pub const fn real(re: f64) -> Complex {
        Complex { re, im: 0.0 }
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    #[inline]
    pub fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }

    #[inline]
    pub fn conjugate(self) -> Complex {
        Complex::new(self.re, -self.im)
    }

    /// |z|, without intermediate overflow.
    #[inline]
    pub fn magnitude(self) -> f64 {
        real::hypot(self.re, self.im)
    }

    /// arg z in (-pi, pi], numpy's `angle`.
    #[inline]
    pub fn argument(self) -> f64 {
        real::atan2(self.im, self.re)
    }

    #[inline]
    pub fn squared_magnitude(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// `i * z`, with the signed zeros `-imag(z) + i real(z)` carries -- the
    /// form the C library uses to reduce atan to atanh, and what keeps the two
    /// on the same side of their cuts.
    #[inline]
    pub fn times_i(self) -> Complex {
        Complex::new(-self.im, self.re)
    }

    #[inline]
    pub fn times_minus_i(self) -> Complex {
        Complex::new(self.im, -self.re)
    }

    // MARK: elementary functions

    pub fn exp(z: Complex) -> Complex {
        let e = real::exp(z.re);
        if z.im == 0.0 {
            return Complex::new(e, z.im);
        }
        Complex::new(e * real::cos(z.im), e * real::sin(z.im))
    }

    pub fn log(z: Complex) -> Complex {
        Complex::new(real::log(z.magnitude()), z.argument())
    }

    /// C99 `csqrt`: the principal branch, cut along the negative real axis,
    /// with the side decided by the sign of the zero.
    pub fn sqrt(z: Complex) -> Complex {
        if z.re == 0.0 && z.im == 0.0 {
            return Complex::new(0.0, z.im);
        }
        let t = real::sqrt((z.re.abs() + z.magnitude()) / 2.0);
        if z.re >= 0.0 {
            return Complex::new(t, z.im / (2.0 * t));
        }
        Complex::new(z.im.abs() / (2.0 * t), real::copysign(t, z.im))
    }

    pub fn sin(z: Complex) -> Complex {
        Complex::new(real::sin(z.re) * real::cosh(z.im), real::cos(z.re) * real::sinh(z.im))
    }

    pub fn cos(z: Complex) -> Complex {
        Complex::new(real::cos(z.re) * real::cosh(z.im), -real::sin(z.re) * real::sinh(z.im))
    }

    pub fn sinh(z: Complex) -> Complex {
        Complex::new(real::sinh(z.re) * real::cos(z.im), real::cosh(z.re) * real::sin(z.im))
    }

    pub fn cosh(z: Complex) -> Complex {
        Complex::new(real::cosh(z.re) * real::cos(z.im), real::sinh(z.re) * real::sin(z.im))
    }

    /// `(sinh 2x + i sin 2y) / (cosh 2x + cos 2y)`, saturating far from the
    /// imaginary axis where cosh would overflow.
    pub fn tanh(z: Complex) -> Complex {
        if z.re.abs() > 350.0 {
            return Complex::new(real::copysign(1.0, z.re), real::copysign(0.0, real::sin(2.0 * z.im)));
        }
        let d = real::cosh(2.0 * z.re) + real::cos(2.0 * z.im);
        Complex::new(real::sinh(2.0 * z.re) / d, real::sin(2.0 * z.im) / d)
    }

    pub fn tan(z: Complex) -> Complex {
        Complex::tanh(z.times_i()).times_minus_i()
    }

    /// `-i log(iz + sqrt(1 - z^2))`.
    pub fn asin(z: Complex) -> Complex {
        let one_minus_square = Complex::new(1.0 - (z.re * z.re - z.im * z.im), -2.0 * z.re * z.im);
        Complex::log(z.times_i() + Complex::sqrt(one_minus_square)).times_minus_i()
    }

    pub fn acos(z: Complex) -> Complex {
        let a = Complex::asin(z);
        Complex::new(PI / 2.0 - a.re, -a.im)
    }

    pub fn atanh(z: Complex) -> Complex {
        let plus = Complex::new(1.0 + z.re, z.im);
        let minus = Complex::new(1.0 - z.re, -z.im);
        let d = Complex::log(plus) - Complex::log(minus);
        Complex::new(d.re / 2.0, d.im / 2.0)
    }

    pub fn atan(z: Complex) -> Complex {
        Complex::atanh(z.times_i()).times_minus_i()
    }

    /// `log(z + sqrt(z^2 + 1))`.
    pub fn asinh(z: Complex) -> Complex {
        let square_plus_one = Complex::new(z.re * z.re - z.im * z.im + 1.0, 2.0 * z.re * z.im);
        Complex::log(z + Complex::sqrt(square_plus_one))
    }

    /// `log(z + sqrt(z + 1) sqrt(z - 1))` -- the product of two roots rather
    /// than the root of a product, which is what puts the cut on (-inf, 1).
    pub fn acosh(z: Complex) -> Complex {
        Complex::log(z + Complex::sqrt(Complex::new(z.re + 1.0, z.im)) * Complex::sqrt(Complex::new(z.re - 1.0, z.im)))
    }

    /// `a^b`. A small integer power is repeated multiplication, as numpy does
    /// it, so `z^2` at a real z is exactly real rather than `exp(2 log z)`.
    pub fn pow(a: Complex, b: Complex) -> Complex {
        if b.re == 0.0 && b.im == 0.0 {
            return Complex::ONE;
        }
        if a.re == 0.0 && a.im == 0.0 {
            if b.im == 0.0 && b.re > 0.0 {
                return Complex::ZERO;
            }
            return Complex::new(f64::INFINITY, 0.0);
        }
        if b.im == 0.0 && b.re == b.re.round() && b.re.abs() < 100.0 {
            return Complex::integer_power(a, b.re as i64);
        }
        Complex::exp(b * Complex::log(a))
    }

    pub(crate) fn integer_power(a: Complex, n: i64) -> Complex {
        if n < 0 {
            return Complex::ONE / Complex::integer_power(a, -n);
        }
        let mut result = Complex::ONE;
        let mut base = a;
        let mut k = n;
        while k > 0 {
            if k & 1 == 1 {
                result *= base;
            }
            k >>= 1;
            if k > 0 {
                base = base * base;
            }
        }
        result
    }
}

/// `(re+imj)`, each part as Swift's `Double.description` writes it.
impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign = if self.im.is_sign_negative() { "-" } else { "+" };
        write!(f, "({}{}{}j)", description(self.re), sign, description(self.im.abs()))
    }
}

// MARK: arithmetic

impl Neg for Complex {
    type Output = Complex;
    #[inline]
    fn neg(self) -> Complex {
        Complex::new(-self.re, -self.im)
    }
}

impl Add for Complex {
    type Output = Complex;
    #[inline]
    fn add(self, b: Complex) -> Complex {
        Complex::new(self.re + b.re, self.im + b.im)
    }
}

impl Sub for Complex {
    type Output = Complex;
    #[inline]
    fn sub(self, b: Complex) -> Complex {
        Complex::new(self.re - b.re, self.im - b.im)
    }
}

impl Mul for Complex {
    type Output = Complex;
    #[inline]
    fn mul(self, b: Complex) -> Complex {
        let a = self;
        Complex::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
    }
}

impl Div for Complex {
    type Output = Complex;
    /// Smith's algorithm, as numpy divides.
    #[inline]
    fn div(self, b: Complex) -> Complex {
        let a = self;
        let abs_br = b.re.abs();
        let abs_bi = b.im.abs();
        if abs_br >= abs_bi {
            if abs_br == 0.0 && abs_bi == 0.0 {
                // x/0: numpy gives inf/nan components; the sampler pushes any
                // non-finite value to its ceiling, so the exact pattern is moot.
                return Complex::new(a.re / abs_br, a.im / abs_bi);
            }
            let r = b.im / b.re;
            let d = b.re + b.im * r;
            return Complex::new((a.re + a.im * r) / d, (a.im - a.re * r) / d);
        }
        let r = b.re / b.im;
        let d = b.re * r + b.im;
        Complex::new((a.re * r + a.im) / d, (a.im * r - a.re) / d)
    }
}

impl Add<f64> for Complex {
    type Output = Complex;
    #[inline]
    fn add(self, b: f64) -> Complex {
        Complex::new(self.re + b, self.im)
    }
}

impl Add<Complex> for f64 {
    type Output = Complex;
    #[inline]
    fn add(self, b: Complex) -> Complex {
        Complex::new(self + b.re, b.im)
    }
}

impl Sub<f64> for Complex {
    type Output = Complex;
    #[inline]
    fn sub(self, b: f64) -> Complex {
        Complex::new(self.re - b, self.im)
    }
}

impl Sub<Complex> for f64 {
    type Output = Complex;
    #[inline]
    fn sub(self, b: Complex) -> Complex {
        Complex::new(self - b.re, -b.im)
    }
}

impl Mul<f64> for Complex {
    type Output = Complex;
    #[inline]
    fn mul(self, b: f64) -> Complex {
        Complex::new(self.re * b, self.im * b)
    }
}

impl Mul<Complex> for f64 {
    type Output = Complex;
    #[inline]
    fn mul(self, b: Complex) -> Complex {
        Complex::new(self * b.re, self * b.im)
    }
}

impl Div<f64> for Complex {
    type Output = Complex;
    #[inline]
    fn div(self, b: f64) -> Complex {
        Complex::new(self.re / b, self.im / b)
    }
}

impl Div<Complex> for f64 {
    type Output = Complex;
    #[inline]
    fn div(self, b: Complex) -> Complex {
        Complex::new(self, 0.0) / b
    }
}

impl AddAssign for Complex {
    #[inline]
    fn add_assign(&mut self, b: Complex) {
        *self = *self + b;
    }
}

impl SubAssign for Complex {
    #[inline]
    fn sub_assign(&mut self, b: Complex) {
        *self = *self - b;
    }
}

impl MulAssign for Complex {
    #[inline]
    fn mul_assign(&mut self, b: Complex) {
        *self = *self * b;
    }
}

impl DivAssign for Complex {
    #[inline]
    fn div_assign(&mut self, b: Complex) {
        *self = *self / b;
    }
}

impl MulAssign<f64> for Complex {
    #[inline]
    fn mul_assign(&mut self, b: f64) {
        *self = *self * b;
    }
}

impl DivAssign<f64> for Complex {
    #[inline]
    fn div_assign(&mut self, b: f64) {
        *self = *self / b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_zeros_pick_the_side_of_the_cut() {
        let above = Complex::sqrt(Complex::new(-4.0, 0.0));
        let below = Complex::sqrt(Complex::new(-4.0, -0.0));
        assert_eq!(above, Complex::new(0.0, 2.0));
        assert_eq!(below, Complex::new(0.0, -2.0));
        assert_eq!(Complex::log(Complex::new(-1.0, -0.0)).im, -PI);
    }

    #[test]
    fn a_small_integer_power_is_multiplication() {
        let z = Complex::real(3.0);
        assert_eq!(Complex::pow(z, Complex::real(2.0)), Complex::real(9.0));
        assert_eq!(Complex::pow(Complex::ZERO, Complex::real(2.0)), Complex::ZERO);
        assert_eq!(Complex::pow(Complex::ZERO, Complex::real(-1.0)).re, f64::INFINITY);
        assert_eq!(Complex::pow(z, Complex::ZERO), Complex::ONE);
    }

    #[test]
    fn description_is_swifts() {
        assert_eq!(Complex::new(1.0, -0.5).to_string(), "(1.0-0.5j)");
        assert_eq!(Complex::new(0.0, 1e-5).to_string(), "(0.0+1e-05j)");
    }
}
