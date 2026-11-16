//! The Faddeeva function w(z) = e^{-z²} erfc(-iz), and erf, erfc and erfi
//! through it. The port of `Erf.swift`.
//!
//! Weideman's rational approximation ("Computation of the complex error
//! function", 1994) with 64 terms: a single polynomial in `(L + iz)/(L - iz)`,
//! accurate to a few ulps across the upper half-plane, whose coefficients are
//! a discrete Fourier transform computed once, in a scalar prototype, and
//! written down here. The lower half-plane follows from
//! `w(-z) = 2 e^{-z²} - w(z)`.

use core::f64::consts::PI;
use std::sync::LazyLock;

use crate::complex::Complex;
use crate::real;

pub enum Faddeeva {}

const L: f64 = 6.727171322029716;
static SQRT_PI: LazyLock<f64> = LazyLock::new(|| real::sqrt(PI));

/// Polynomial coefficients, highest power first.
const COEFFICIENTS: [f64; 64] = [
    0.0,
    -9.055037244324897e-17,
    -1.1653209979568358e-16,
    -1.1716769801049914e-16,
    1.2109620530784368e-16,
    -1.2605218122250793e-17,
    -2.18212715384761e-16,
    -2.2104253016558195e-16,
    -1.0824968888340203e-16,
    -2.2944355346394034e-16,
    -8.964015185669267e-17,
    -2.124476077622484e-17,
    -3.357657519909759e-17,
    2.3581813318296436e-17,
    -6.8871462743752275e-18,
    -4.376637422574319e-17,
    7.417563524085893e-17,
    5.389074992350545e-17,
    7.793604873008101e-17,
    -5.193078949160759e-17,
    2.384529504876383e-16,
    6.186473856582468e-16,
    -7.657088729755898e-16,
    -4.2506362862487974e-15,
    -2.0190295016094956e-16,
    3.299454337156331e-14,
    5.909242235017462e-14,
    -1.549385220783347e-13,
    -7.920822247734917e-13,
    -3.939344590984745e-13,
    5.832585737473097e-12,
    1.7501502908887984e-11,
    -6.4707349345185925e-12,
    -1.7560607958960022e-10,
    -4.533915068936149e-10,
    2.4434786854343034e-10,
    5.186955616565486e-09,
    1.592681388741325e-08,
    7.435710708457646e-09,
    -1.3610261250028284e-07,
    -6.650424120776997e-07,
    -1.5547722782660322e-06,
    -7.564244111822494e-08,
    1.7901801586212798e-05,
    0.00010227006798923578,
    0.0003962745103981355,
    0.0012549788049982572,
    0.003460207948107515,
    0.008565381413176008,
    0.019380399024538295,
    0.040552846529580175,
    0.07911655067602572,
    0.14477859973586424,
    0.24963969994535562,
    0.4070443030398735,
    0.6293868343374367,
    0.9249760252638086,
    1.2944377517175158,
    1.727506085787117,
    2.20125657128641,
    2.680732639559084,
    3.1224481894020366,
    3.4804961039850415,
    3.7141697931977027,
];

impl Faddeeva {
    /// w(z) for Im z >= 0.
    fn upper(z: Complex) -> Complex {
        let iz = z.times_i();
        let lm = L - iz;
        let big = (L + iz) / lm;
        let mut p = Complex::ZERO;
        for c in COEFFICIENTS {
            p = p * big + c;
        }
        2.0 * p / (lm * lm) + (1.0 / *SQRT_PI) / lm
    }

    pub fn w(z: Complex) -> Complex {
        if !z.im.is_sign_negative() {
            return Faddeeva::upper(z);
        }
        2.0 * Complex::exp(-(z * z)) - Faddeeva::upper(-z)
    }

    /// erfc(z) = e^{-z²} w(iz) on Re z >= 0, and 2 - erfc(-z) beyond.
    pub fn erfc(z: Complex) -> Complex {
        if !z.re.is_sign_negative() {
            return Complex::exp(-(z * z)) * Faddeeva::upper(z.times_i());
        }
        2.0 - Faddeeva::erfc(-z)
    }

    fn erf_series(z: Complex) -> Complex {
        let z2 = z * z;
        let mut term = z;
        let mut total = z;
        let mut k = 0;
        loop {
            k += 1;
            term *= -z2 / f64::from(k);
            let d = term / f64::from(2 * k + 1);
            total += d;
            if d.magnitude() <= 2.220446049250313e-16 * total.magnitude() {
                break;
            }
        }
        total * (2.0 / *SQRT_PI)
    }

    /// erf(z): the Maclaurin series inside the unit disc, where `1 - erfc`
    /// would lose the leading digits, and the complement outside.
    pub fn erf(z: Complex) -> Complex {
        if z.magnitude() < 1.0 {
            return Faddeeva::erf_series(z);
        }
        1.0 - Faddeeva::erfc(z)
    }

    /// erfi(z) = -i erf(iz).
    pub fn erfi(z: Complex) -> Complex {
        Faddeeva::erf(z.times_i()).times_minus_i()
    }
}
