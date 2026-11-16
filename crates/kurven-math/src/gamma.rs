//! The gamma family: log Γ, Γ, 1/Γ and ψ, for complex arguments; and sin(πz),
//! cos(πz) with exact zeros. The port of `Gamma.swift`.
//!
//! `log_gamma` is scipy's algorithm (Hare, "Computing the principal branch of
//! log-Gamma", 1997): Stirling's series where it converges, a Taylor series
//! around 1 and 2, the reflection formula on the left, and otherwise a
//! recurrence that counts how many times the shifted product crosses the
//! negative real axis so the branch stays the principal one. Γ and 1/Γ are its
//! exponentials, which is also how scipy computes them, so the two agree to
//! the last few bits rather than to a tolerance.

// Swift's literals as it writes them: more digits than a double holds, and
// γ spelled out where Rust has `EULER_GAMMA` (the same double).
#![allow(clippy::excessive_precision, clippy::approx_constant)]

use core::f64::consts::PI;
use std::sync::LazyLock;

use crate::complex::Complex;
use crate::real;

pub enum Gamma {}

const SMALL_X: f64 = 7.0;
const SMALL_Y: f64 = 7.0;
const TAYLOR_RADIUS: f64 = 0.2;
static LOG_PI: LazyLock<f64> = LazyLock::new(|| real::log(PI));
static HALF_LOG_2PI: LazyLock<f64> = LazyLock::new(|| 0.5 * real::log(2.0 * PI));

/// B_{2n} / (2n (2n - 1)), highest power of 1/z^2 first.
const STIRLING: [f64; 8] = [
    -2.955065359477124183e-2,
    6.4102564102564102564e-3,
    -1.9175269175269175269e-3,
    8.4175084175084175084e-4,
    -5.952380952380952381e-4,
    7.9365079365079365079e-4,
    -2.7777777777777777778e-3,
    8.3333333333333333333e-2,
];

/// log Γ(1 + x) = -γ x + Σ (-1)^k ζ(k)/k x^k, k = 2...23, highest first.
const TAYLOR: [f64; 23] = [
    -0.04347826605304026,
    0.04545455629320467,
    -0.04761907033014223,
    0.05000004769810169,
    -0.05263167937961666,
    0.055555767627403614,
    -0.05882397865868458,
    0.06250095514121304,
    -0.06666870588242046,
    0.07143294629536134,
    -0.0769325164113522,
    0.083353840546109,
    -0.09095401714582904,
    0.10009945751278179,
    -0.11133426586956469,
    0.12550966952474304,
    -0.14404989676884614,
    0.16955717699740822,
    -0.207385551028674,
    0.27058080842778454,
    -0.40068563438653143,
    0.8224670334241132,
    -0.5772156649015329,
];

/// B_{2k} / (2k), k = 1...8.
const DIGAMMA_ASYMPTOTIC: [f64; 8] =
    [1.0 / 12.0, -1.0 / 120.0, 1.0 / 252.0, -1.0 / 240.0, 1.0 / 132.0, -691.0 / 32760.0, 1.0 / 12.0, -3617.0 / 8160.0];

impl Gamma {
    pub const EULER: f64 = 0.5772156649015329;

    #[inline(always)]
    pub(crate) fn eval_poly(coefficients: &[f64], z: Complex) -> Complex {
        let mut r = Complex::ZERO;
        for &c in coefficients {
            r = r * z + c;
        }
        r
    }

    fn stirling_series(z: Complex) -> Complex {
        let rz = 1.0 / z;
        let rzz = rz / z;
        (z - 0.5) * Complex::log(z) - z + *HALF_LOG_2PI + rz * Gamma::eval_poly(&STIRLING, rzz)
    }

    fn taylor_series(z: Complex) -> Complex {
        let w = z - 1.0;
        w * Gamma::eval_poly(&TAYLOR, w)
    }

    /// Requires Im z >= 0 (not a negative zero).
    fn recurrence(z: Complex) -> Complex {
        let mut signflips = 0;
        let mut sb = false;
        let mut shiftprod = z;
        let mut w = Complex::new(z.re + 1.0, z.im);
        while w.re <= SMALL_X {
            shiftprod *= w;
            let nsb = shiftprod.im.is_sign_negative();
            if nsb && !sb {
                signflips += 1;
            }
            sb = nsb;
            w = Complex::new(w.re + 1.0, w.im);
        }
        Gamma::stirling_series(w) - Complex::log(shiftprod) - Complex::new(0.0, f64::from(signflips) * 2.0 * PI)
    }

    /// True at 0, -1, -2, ...: where Γ has its poles.
    pub fn is_pole(z: Complex) -> bool {
        z.im == 0.0 && z.re <= 0.0 && z.re == z.re.floor()
    }

    /// The principal branch of log Γ. NaN at the poles.
    pub fn log_gamma(z: Complex) -> Complex {
        if Gamma::is_pole(z) {
            return Complex::new(f64::NAN, f64::NAN);
        }
        if z.re > SMALL_X || z.im.abs() > SMALL_Y {
            return Gamma::stirling_series(z);
        }
        if (z - 1.0).magnitude() <= TAYLOR_RADIUS {
            return Gamma::taylor_series(z);
        }
        if (z - 2.0).magnitude() <= TAYLOR_RADIUS {
            return Complex::log(z - 1.0) + Gamma::taylor_series(z - 1.0);
        }
        if z.re < 0.1 {
            let tmp = real::copysign(2.0 * PI, z.im) * (0.5 * z.re + 0.25).floor();
            return Complex::new(*LOG_PI, tmp) - Complex::log(Trig::sin_pi_complex(z)) - Gamma::log_gamma(1.0 - z);
        }
        if !z.im.is_sign_negative() {
            return Gamma::recurrence(z);
        }
        Gamma::recurrence(z.conjugate()).conjugate()
    }

    /// Γ(z); infinite at the poles rather than NaN, because a landscape is a
    /// picture of where they are.
    pub fn gamma(z: Complex) -> Complex {
        if Gamma::is_pole(z) {
            return Complex::new(f64::INFINITY, 0.0);
        }
        Complex::exp(Gamma::log_gamma(z))
    }

    /// 1/Γ(z), entire: exactly zero at the poles of Γ.
    pub fn rgamma(z: Complex) -> Complex {
        if Gamma::is_pole(z) {
            return Complex::ZERO;
        }
        Complex::exp(-Gamma::log_gamma(z))
    }

    /// ψ(z) = Γ'(z)/Γ(z): reflection onto Re z >= 1/2, recurrence out to
    /// |z| >= 16, then the asymptotic series.
    pub fn digamma(z: Complex) -> Complex {
        if Gamma::is_pole(z) {
            return Complex::new(f64::INFINITY, 0.0);
        }
        let mut res = Complex::ZERO;
        let mut w = z;
        if w.re < 0.5 {
            res -= PI * Trig::cos_pi_complex(w) / Trig::sin_pi_complex(w);
            w = 1.0 - w;
        }
        while w.magnitude() < 16.0 {
            res -= 1.0 / w;
            w += Complex::ONE;
        }
        let rz = 1.0 / w;
        let rzz = rz * rz;
        let mut tail = Complex::ZERO;
        let mut p = rzz;
        for c in DIGAMMA_ASYMPTOTIC {
            tail += c * p;
            p *= rzz;
        }
        res + Complex::log(w) - 0.5 * rz - tail
    }
}

/// sin(πz) and cos(πz) with exact zeros at the integers and half-integers.
///
/// The real reductions are scipy's, including the detail that `cos_pi` is a
/// *positive* zero at every half-integer: that zero, multiplied by sinh(πy),
/// is the imaginary part of sin(πz) on the lines Re z = n + 1/2, and its sign
/// picks the branch of the logarithm in `log_gamma`'s reflection formula.
///
/// Swift overloads `sinPi` and `cosPi` on Double and Complex; the complex
/// ones are `sin_pi_complex` and `cos_pi_complex` here.
pub enum Trig {}

impl Trig {
    pub fn sin_pi(x: f64) -> f64 {
        let mut s = 1.0;
        let mut r = real::fmod(x, 2.0);
        if r < 0.0 {
            r += 2.0;
        }
        if r > 1.0 {
            r -= 1.0;
            s = -1.0;
        }
        if r > 0.5 {
            r = 1.0 - r;
        }
        if r == 0.5 {
            return s;
        }
        s * real::sin(PI * r)
    }

    pub fn cos_pi(x: f64) -> f64 {
        let mut r = real::fmod(x.abs(), 2.0);
        let mut s = 1.0;
        if r > 1.0 {
            r -= 1.0;
            s = -1.0;
        }
        if r == 0.5 {
            return 0.0;
        }
        s * real::sin(PI * (0.5 - r))
    }

    pub fn sin_pi_complex(z: Complex) -> Complex {
        let piy = PI * z.im;
        Complex::new(Trig::sin_pi(z.re) * real::cosh(piy), Trig::cos_pi(z.re) * real::sinh(piy))
    }

    pub fn cos_pi_complex(z: Complex) -> Complex {
        let piy = PI * z.im;
        Complex::new(Trig::cos_pi(z.re) * real::cosh(piy), -Trig::sin_pi(z.re) * real::sinh(piy))
    }
}
