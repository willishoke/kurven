//! The Riemann zeta function of a complex argument: `kurven.expr`'s algorithm,
//! point by point. The port of `Zeta.swift`.
//!
//! Borwein's alternating series on Re s >= 1/2, with the order chosen from
//! |Im s| so a disc around the origin does not pay the ζ drawing's price; the
//! functional equation below the critical line; and Euler-Maclaurin at the
//! points where Borwein's own factor `1 - 2^(1-s)` vanishes (σ = 1,
//! t = 2πk/ln 2), where the quotient would otherwise be 0/0.

use core::f64::consts::PI;
use std::sync::LazyLock;

use crate::complex::Complex;
use crate::gamma::Gamma;
use crate::real;
use crate::swift;

pub enum Zeta {}

static LOG_GROWTH: LazyLock<f64> = LazyLock::new(|| real::log(3.0 + real::sqrt(8.0)));

/// B_{2k} for k = 0...12.
const BERNOULLI_EVEN: [f64; 13] = [
    1.0,
    1.0 / 6.0,
    -1.0 / 30.0,
    1.0 / 42.0,
    -1.0 / 30.0,
    5.0 / 66.0,
    -691.0 / 2730.0,
    7.0 / 6.0,
    -3617.0 / 510.0,
    43867.0 / 798.0,
    -174611.0 / 330.0,
    854513.0 / 138.0,
    -236364091.0 / 2730.0,
];

/// What the port answers where Swift's `Int(...)` traps (`swift::int`).
const NAN: Complex = Complex::new(f64::NAN, f64::NAN);

/// The orders `order_for_height` can ask for.
const ORDERS_LO: usize = 16;
const ORDERS_HI: usize = 200;

/// The series' weights `±(d_k - d_n) / d_n`, k < n, for every order, laid
/// end to end: order n's start at `offset(n)`.
///
/// Made once and read from every thread at once. (In Swift they were a
/// dictionary behind a lock, and every evaluation took the lock and retained
/// the table it read: with a landscape sampled and refined on every core,
/// that was the cores queueing for one lock and one reference count, and ζ
/// ran on one and a half of eighteen.)
static WEIGHTS: LazyLock<Vec<f64>> = LazyLock::new(|| {
    let mut buffer = vec![0.0; offset(ORDERS_HI + 1)];
    for n in ORDERS_LO..=ORDERS_HI {
        let d = Zeta::borwein_coefficients(n);
        for k in 0..n {
            let mut w = (d[k] - d[n]) / d[n];
            if k % 2 == 1 {
                w = -w;
            }
            buffer[offset(n) + k] = w;
        }
    }
    buffer
});

/// ln(k + 1) for every term of the longest series.
static LOGS: LazyLock<Vec<f64>> = LazyLock::new(|| (0..ORDERS_HI).map(|k| real::log((k + 1) as f64)).collect());

/// Where order n's weights start: the orders below it, end to end.
fn offset(n: usize) -> usize {
    let lo = ORDERS_LO;
    (n - lo) * (n + lo - 1) / 2
}

impl Zeta {
    /// `d_k`, k = 0...n, by the ratio between consecutive terms: the closed
    /// form is a sum of factorials that overflows long before the n this needs.
    pub(crate) fn borwein_coefficients(n: usize) -> Vec<f64> {
        let mut d = vec![0.0; n + 1];
        let nf = n as f64;
        let mut term = 1.0 / nf;
        let mut total = 0.0;
        for (j, dj) in d.iter_mut().enumerate() {
            total += term;
            *dj = nf * total;
            term *= 4.0 * ((n + j) as f64) * ((n - j) as f64) / (((2 * j + 1) * (2 * j + 2)) as f64);
        }
        d
    }

    /// How many terms the error bound needs at this height:
    /// `|γ_n(s)| ≲ 3(1 + 2|t|)e^{π|t|/2}(3+√8)^{-n}`. None at a NaN height,
    /// where Swift's `Int(...)` traps.
    fn order_for_height(t: f64) -> Option<usize> {
        let need = (PI * t / 2.0 + real::log(3.0 * (1.0 + 2.0 * t)) + 15.0 * real::log(10.0)) / *LOG_GROWTH;
        let clamped = swift::min(swift::max(need.ceil(), ORDERS_LO as f64), ORDERS_HI as f64);
        swift::int(clamped).map(|n| n as usize)
    }

    #[allow(clippy::needless_range_loop)]
    fn euler_maclaurin(s: Complex, terms: usize) -> Complex {
        // Swift's `max(10, Int(abs(s.im)) + 10)`, which traps past 2^63.
        let Some(cut) = swift::int(s.im.abs()).and_then(|t| t.checked_add(10)) else { return NAN };
        let cut = cut.max(10) as usize;
        let mut total = Complex::ZERO;
        for k in 1..cut {
            total += Complex::exp(-s * real::log(k as f64));
        }
        let c = cut as f64;
        total += Complex::pow(Complex::real(c), 1.0 - s) / (s - 1.0) + 0.5 * Complex::pow(Complex::real(c), -s);
        let mut product = s;
        let mut factorial = 1.0;
        for j in 1..=terms {
            factorial *= ((2 * j - 1) * (2 * j)) as f64;
            total +=
                BERNOULLI_EVEN[j] / factorial * product * Complex::pow(Complex::real(c), -s - ((2 * j - 1) as f64));
            product = product * (s + ((2 * j - 1) as f64)) * (s + ((2 * j) as f64));
        }
        total
    }

    /// ζ on Re s >= 1/2.
    fn half_plane(s: Complex) -> Complex {
        let Some(n) = Zeta::order_for_height(s.im.abs()) else { return NAN };
        let w = &WEIGHTS[offset(n)..offset(n) + n];
        let logs = &LOGS[..n];
        let mut total = Complex::ZERO;
        for k in 0..n {
            total += w[k] * Complex::exp(-s * logs[k]);
        }
        let eta = 1.0 - Complex::pow(Complex::real(2.0), 1.0 - s);
        if eta.magnitude() < 1e-5 {
            return Zeta::euler_maclaurin(s, 12);
        }
        -total / eta
    }

    pub fn zeta(s: Complex) -> Complex {
        if s.re == 1.0 && s.im == 0.0 {
            return Complex::new(f64::INFINITY, 0.0);
        }
        if s.re == 0.0 && s.im == 0.0 {
            return Complex::new(-0.5, 0.0);
        }
        if s.re >= 0.5 {
            return Zeta::half_plane(s);
        }
        // The functional equation: 2^s π^(s-1) sin(πs/2) Γ(1-s) ζ(1-s).
        Complex::pow(Complex::real(2.0), s)
            * Complex::pow(Complex::real(PI), s - 1.0)
            * Complex::sin(PI * s / 2.0)
            * Gamma::gamma(1.0 - s)
            * Zeta::half_plane(1.0 - s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Swift built from `Sources/KurvenMath` exits on SIGTRAP for ζ at
    /// (nan, nan) and (0.3, nan): `Int(...)` of a NaN order. The port answers
    /// NaN there, and is Swift's elsewhere.
    #[test]
    fn where_swift_traps_the_port_answers_nan() {
        assert!(Zeta::zeta(Complex::new(f64::NAN, f64::NAN)).is_nan());
        assert!(Zeta::zeta(Complex::new(0.3, f64::NAN)).is_nan());
        assert!(Zeta::zeta(Complex::new(f64::NAN, 0.0)).is_nan());
        assert!(Zeta::euler_maclaurin(Complex::new(1.0, 1e19), 12).is_nan());
    }
}
