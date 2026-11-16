//! Bessel functions of real order and complex argument: J, Y, I, K, and the
//! Airy functions through them. The port of `Bessel.swift`.
//!
//! Two engines, each used only where it is accurate, and each holding scipy's
//! AMOS-based answers to about 1e-11 across the windows the landscapes use
//! (`tests/fixtures/expr` pins the numbers):
//!
//! - **J and Y** (`cylinder`): the power series under |z| = 12 or when the
//!   order exceeds the argument -- its cancellation is e^(|z| - |Im z|),
//!   tempered by 1/Γ(ν + k + 1) for large orders -- and otherwise the Hankel
//!   expansions at the fractional part of the order, carried up by the
//!   three-term recurrence in the one direction each solution is stable in.
//!   Everything is reduced to the first quadrant first: conjugation for the
//!   lower half-plane, the continuation formulas for the left.
//! - **I and K** (`modified`): Temme's series under |z| = 2 and Steed's
//!   continued fraction beyond, on Re z > 0 (Thompson and Barnett 1987, the
//!   complex form of Numerical Recipes' `bessik`), with the continuation
//!   formulas for the left half-plane and the J/Y engine on the imaginary
//!   axis itself, where the fraction cannot converge.

use core::f64::consts::PI;
use std::sync::LazyLock;

use crate::complex::Complex;
use crate::gamma::{Gamma, Trig};
use crate::real;

pub enum Bessel {}

const EPS: f64 = 2.220446049250313e-16;
const TINY: f64 = 1e-300;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Cylinder {
    pub j: Complex,
    pub y: Complex,
    pub h1: Complex,
    pub h2: Complex,
}

/// I and K at one point: Swift's `(i:, k:)` tuple.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Modified {
    pub i: Complex,
    pub k: Complex,
}

/// Swift's `Int(x)`, which traps on a double that is not finite or does not
/// fit: `None` there. Where Swift would trap, the port answers NaN
/// (`crates/FINDINGS.md`); Rust's `as` would saturate instead, and the loops
/// counted by the result would run for ever.
fn swift_int(x: f64) -> Option<i64> {
    if (-9223372036854775808.0..9223372036854775808.0).contains(&x) { Some(x as i64) } else { None }
}

const NAN: Complex = Complex::new(f64::NAN, f64::NAN);
const NAN_CYLINDER: Cylinder = Cylinder { j: NAN, y: NAN, h1: NAN, h2: NAN };

/// The two quotients of `gamma_pair`, and the two reciprocals.
struct GammaPair {
    g1: f64,
    g2: f64,
    gp: f64,
    gm: f64,
}

impl Bessel {
    // MARK: J and Y

    /// Σ (-1)^k (z/2)^{2k+ν} / (k! Γ(ν+k+1)); ν not a negative integer.
    fn j_series(nu: f64, z: Complex) -> Complex {
        let half = z / 2.0;
        let mut term = Complex::pow(half, Complex::real(nu)) * Gamma::rgamma(Complex::real(nu + 1.0));
        let mut total = term;
        let q = -(half * half);
        let mut k = 0;
        loop {
            k += 1;
            term *= q / (f64::from(k) * (nu + f64::from(k)));
            total += term;
            if term.magnitude() <= EPS * total.magnitude() || k > 500 {
                break;
            }
        }
        total
    }

    fn harmonic(n: i64) -> f64 {
        let mut h = 0.0;
        for k in 1..=n {
            h += 1.0 / k as f64;
        }
        h
    }

    fn factorial(n: i64) -> f64 {
        let mut f = 1.0;
        for k in 2..=n {
            f *= k as f64;
        }
        f
    }

    /// Y_n for integer n >= 0, DLMF 10.8.1.
    fn y_series_integer(n: i64, z: Complex) -> Complex {
        let half = z / 2.0;
        let h2 = half * half;
        let mut total = Complex::ZERO;
        if n > 0 {
            let mut p = Complex::pow(half, Complex::real(-n as f64));
            for k in 0..n {
                total -= Bessel::factorial(n - k - 1) / Bessel::factorial(k) * p;
                p *= h2;
            }
        }
        total /= PI;
        let jn = Bessel::j_series(n as f64, z);
        total += (2.0 / PI) * Complex::log(half) * jn;
        let mut p = Complex::pow(half, Complex::real(n as f64)) / Bessel::factorial(n);
        let mut k = 0i64;
        let mut s = Complex::ZERO;
        loop {
            let psi = (Bessel::harmonic(k) - Gamma::EULER) + (Bessel::harmonic(n + k) - Gamma::EULER);
            let d = psi * p;
            s += d;
            if (d.magnitude() <= EPS * s.magnitude() && k > 2) || k > 500 {
                break;
            }
            k += 1;
            p *= -h2 / (k as f64 * (n + k) as f64);
        }
        total - s / PI
    }

    /// The Hankel expansions, summed to their smallest term.
    fn hankel_asymptotic(nu: f64, z: Complex) -> (Complex, Complex) {
        let mu = 4.0 * nu * nu;
        let mut s1 = Complex::ONE;
        let mut s2 = Complex::ONE;
        let mut term = Complex::ONE;
        let mut ik = Complex::ONE;
        let mut smallest = 1.0;
        let mut k = 0;
        while k < 200 {
            k += 1;
            let odd = f64::from(2 * k - 1);
            term *= (mu - odd * odd) / (8.0 * f64::from(k) * z);
            ik = ik.times_i();
            let mag = term.magnitude();
            if mag > smallest {
                break;
            }
            smallest = mag;
            s1 += ik * term;
            s2 += ik.conjugate() * term;
            if mag < 1e-15 {
                break;
            }
        }
        let omega = z - nu * PI / 2.0 - PI / 4.0;
        let pref = Complex::sqrt(2.0 / (PI * z));
        let e = Complex::exp(omega.times_i());
        (pref * e * s1, pref * s2 / e)
    }

    /// J, Y, H1, H2 for ν >= 0 in the closed first quadrant, z != 0.
    fn cylinder_first_quadrant(nu: f64, z: Complex) -> Cylinder {
        let a = z.magnitude();
        if a >= 12.0 && nu <= a {
            let Some(nl) = swift_int(nu) else { return NAN_CYLINDER };
            let mu = nu - nl as f64;
            let (mut h1a, h2a) = Bessel::hankel_asymptotic(mu, z);
            if nl == 0 {
                return Cylinder { j: (h1a + h2a) / 2.0, y: ((h1a - h2a) / 2.0).times_minus_i(), h1: h1a, h2: h2a };
            }
            let (mut h1b, _) = Bessel::hankel_asymptotic(mu + 1.0, z);
            let jmu = (h1a + h2a) / 2.0;
            // H1 forward: the dominant solution below |z| in the upper
            // half-plane, and dominant above it everywhere.
            for k in 1..=nl {
                (h1a, h1b) = (h1b, (2.0 * (mu + k as f64) / z) * h1b - h1a);
            }
            // J by Miller's backward recurrence, scaled to the accurate J_mu.
            let Some(n) = swift_int(a).and_then(|a| nl.max(a).checked_add(40)) else { return NAN_CYLINDER };
            let mut jp = Complex::ZERO;
            let mut jc = Complex::real(TINY);
            let mut jnu = Complex::ZERO;
            for k in (1..=n).rev() {
                let jm = (2.0 * (mu + k as f64) / z) * jc - jp;
                (jp, jc) = (jc, jm);
                if k - 1 == nl {
                    jnu = jc;
                }
                if jc.magnitude() > 1e200 {
                    jp /= 1e200;
                    jc /= 1e200;
                    jnu /= 1e200;
                }
            }
            jnu *= jmu / jc;
            let y = (h1a - jnu).times_minus_i();
            return Cylinder { j: jnu, y, h1: h1a, h2: 2.0 * jnu - h1a };
        }
        let j = Bessel::j_series(nu, z);
        let n = nu.round();
        let y = if nu == n {
            let Some(n) = swift_int(n) else { return NAN_CYLINDER };
            Bessel::y_series_integer(n, z)
        } else {
            let jm = Bessel::j_series(-nu, z);
            (j * Trig::cos_pi(nu) - jm) / Trig::sin_pi(nu)
        };
        Cylinder { j, y, h1: j + y.times_i(), h2: j - y.times_i() }
    }

    /// J_ν, Y_ν and the two Hankel functions anywhere on the principal sheet.
    pub fn cylinder(nu: f64, z: Complex) -> Cylinder {
        if z.re == 0.0 && z.im == 0.0 {
            let j = if nu == 0.0 {
                Complex::ONE
            } else if nu > 0.0 || nu == nu.round() {
                Complex::ZERO
            } else {
                Complex::new(f64::INFINITY, 0.0)
            };
            let inf = Complex::new(f64::NEG_INFINITY, 0.0);
            return Cylinder { j, y: inf, h1: inf, h2: Complex::new(f64::INFINITY, 0.0) };
        }
        if nu < 0.0 {
            // J_{-ν} = cos(νπ) J_ν - sin(νπ) Y_ν ; Y_{-ν} = sin(νπ) J_ν + cos(νπ) Y_ν.
            let c = Bessel::cylinder(-nu, z);
            let co = Trig::cos_pi(-nu);
            let si = Trig::sin_pi(-nu);
            let j = co * c.j - si * c.y;
            let y = si * c.j + co * c.y;
            return Cylinder { j, y, h1: j + y.times_i(), h2: j - y.times_i() };
        }
        if z.re < 0.0 {
            // z = z' e^{±iπ}: DLMF 10.11.
            let zr = -z;
            let s: f64 = if z.im.is_sign_negative() { -1.0 } else { 1.0 };
            let c = Bessel::cylinder(nu, zr);
            let co = Trig::cos_pi(nu);
            let rot = Complex::exp(Complex::new(0.0, s * nu * PI));
            let j = rot * c.j;
            let y = c.y / rot + (2.0 * s * co) * c.j.times_i();
            let (h1, h2) = if s > 0.0 {
                (-(c.h2 / rot), rot * c.h1 + (2.0 * co) * c.h2)
            } else {
                (rot * c.h2 + (2.0 * co) * c.h1, -(c.h1 / rot))
            };
            return Cylinder { j, y, h1, h2 };
        }
        if z.im.is_sign_negative() {
            let c = Bessel::cylinder_first_quadrant(nu, z.conjugate());
            return Cylinder { j: c.j.conjugate(), y: c.y.conjugate(), h1: c.h2.conjugate(), h2: c.h1.conjugate() };
        }
        Bessel::cylinder_first_quadrant(nu, z)
    }

    pub fn j(nu: f64, z: Complex) -> Complex {
        Bessel::cylinder(nu, z).j
    }

    pub fn y(nu: f64, z: Complex) -> Complex {
        Bessel::cylinder(nu, z).y
    }

    // MARK: I and K

    /// `(1/Γ(1-μ) - 1/Γ(1+μ))/(2μ)` and `(1/Γ(1-μ) + 1/Γ(1+μ))/2`, plus the
    /// two reciprocals; the first by its Taylor series near μ = 0, where the
    /// quotient would cancel.
    fn gamma_pair(mu: f64) -> GammaPair {
        let gp = Gamma::rgamma(Complex::real(1.0 + mu)).re;
        let gm = Gamma::rgamma(Complex::real(1.0 - mu)).re;
        let g1 = if mu.abs() < 1e-3 {
            let c4 = -0.0420026350340952;
            -Gamma::EULER - c4 * mu * mu
        } else {
            (gm - gp) / (2.0 * mu)
        };
        GammaPair { g1, g2: (gm + gp) / 2.0, gp, gm }
    }

    /// I_ν, K_ν for ν >= 0 and Re z > 0.
    fn modified_right_half(nu: f64, z: Complex) -> Modified {
        let maxit = 20000;
        let Some(nl) = swift_int(nu + 0.5) else { return Modified { i: NAN, k: NAN } };
        let mu = nu - nl as f64;
        let mu2 = mu * mu;
        let xi = 1.0 / z;
        let xi2 = 2.0 * xi;

        // CF1 for I'_ν / I_ν by modified Lentz.
        let mut b = xi2 * nu;
        let mut h = nu * xi;
        if h.magnitude() < TINY {
            h = Complex::real(TINY);
        }
        let mut d = Complex::ZERO;
        let mut c = h;
        for _ in 1..=maxit {
            b += xi2;
            let bd = b + d;
            d = if bd.re == 0.0 && bd.im == 0.0 { Complex::real(1.0 / TINY) } else { 1.0 / bd };
            c = b + 1.0 / c;
            let delta = c * d;
            h *= delta;
            if (delta - 1.0).magnitude() < EPS {
                break;
            }
        }
        let mut ril = Complex::real(TINY);
        let mut ripl = h * ril;
        let ril1 = ril;
        let mut fact = nu * xi;
        for _ in 0..nl.max(0) {
            let ritemp = fact * ril + ripl;
            fact -= xi;
            ripl = fact * ritemp + ril;
            ril = ritemp;
        }
        let f = ripl / ril;

        let rkmu: Complex;
        let rk1: Complex;
        if z.magnitude() < 2.0 {
            // Temme's series.
            let x2 = z / 2.0;
            let pimu = PI * mu;
            let fact1 = if pimu.abs() < EPS { 1.0 } else { pimu / real::sin(pimu) };
            let dlog = -Complex::log(x2);
            let mut e = mu * dlog;
            let fact2 = if e.magnitude() < EPS { Complex::ONE } else { Complex::sinh(e) / e };
            let g = Bessel::gamma_pair(mu);
            let mut ff = fact1 * (g.g1 * Complex::cosh(e) + g.g2 * fact2 * dlog);
            let mut total = ff;
            e = Complex::exp(e);
            let mut p = 0.5 * e / g.gp;
            let mut q = 0.5 / (e * g.gm);
            let mut cc = Complex::ONE;
            let dd = x2 * x2;
            let mut sum1 = p;
            let mut i = 0;
            loop {
                i += 1;
                let di = f64::from(i);
                ff = (di * ff + p + q) / (di * di - mu2);
                cc *= dd / di;
                p /= di - mu;
                q /= di + mu;
                let delta = cc * ff;
                total += delta;
                sum1 += cc * (p - di * ff);
                if delta.magnitude() < total.magnitude() * EPS || i > maxit {
                    break;
                }
            }
            rkmu = total;
            rk1 = sum1 * xi2;
        } else {
            // Steed's CF2.
            let mut bb = 2.0 * (1.0 + z);
            let mut dd = 1.0 / bb;
            let mut hh = dd;
            let mut delh = dd;
            let mut q1 = Complex::ZERO;
            let mut q2 = Complex::ONE;
            let a1 = 0.25 - mu2;
            let mut q = Complex::real(a1);
            let mut cc = Complex::real(a1);
            let mut a = -a1;
            let mut s = 1.0 + q * delh;
            for i in 2..=maxit {
                a -= f64::from(2 * (i - 1));
                cc = -a * cc / f64::from(i);
                let qnew = (q1 - bb * q2) / a;
                q1 = q2;
                q2 = qnew;
                q += cc * qnew;
                bb += Complex::real(2.0);
                dd = 1.0 / (bb + a * dd);
                delh = (bb * dd - 1.0) * delh;
                hh += delh;
                let dels = q * delh;
                s += dels;
                if dels.magnitude() < s.magnitude() * EPS {
                    break;
                }
            }
            hh = a1 * hh;
            rkmu = Complex::sqrt(PI / (2.0 * z)) * Complex::exp(-z) / s;
            rk1 = rkmu * (mu + z + 0.5 - hh) * xi;
        }
        // Wronskian: I_μ = (1/z) / (K_{μ+1} + K_μ (f - μ/z)).
        let rimu = xi / (rk1 + rkmu * (f - mu * xi));
        let inu = rimu * (ril1 / ril);
        let mut rkmup = rkmu;
        let mut rkp = rk1;
        for i in 1..=nl {
            let rktemp = (mu + i as f64) * xi2 * rkp + rkmup;
            rkmup = rkp;
            rkp = rktemp;
        }
        Modified { i: inu, k: rkmup }
    }

    /// I and K on and near the imaginary axis, through J and Y:
    /// DLMF 10.27.6 and 10.27.8.
    fn modified_via_cylinder(nu: f64, z: Complex) -> Modified {
        if !(z.re < 0.0 && z.im > 0.0) {
            let c = Bessel::cylinder(nu, z.times_i());
            let rot = Complex::exp(Complex::new(0.0, -nu * PI / 2.0));
            return Modified { i: rot * c.j, k: (Complex::new(0.0, PI / 2.0) / rot) * c.h1 };
        }
        let c = Bessel::cylinder(nu, z.times_minus_i());
        let rot = Complex::exp(Complex::new(0.0, nu * PI / 2.0));
        Modified { i: rot * c.j, k: -(Complex::new(0.0, PI / 2.0) / rot) * c.h2 }
    }

    /// I_ν and K_ν anywhere on the principal sheet.
    pub fn modified(nu: f64, z: Complex) -> Modified {
        if z.re == 0.0 && z.im == 0.0 {
            let i = if nu == 0.0 {
                Complex::ONE
            } else if nu > 0.0 || nu == nu.round() {
                Complex::ZERO
            } else {
                Complex::new(f64::INFINITY, 0.0)
            };
            return Modified { i, k: Complex::new(f64::INFINITY, 0.0) };
        }
        if nu < 0.0 {
            // I_{-ν} = I_ν + (2/π) sin(νπ) K_ν ; K_{-ν} = K_ν.
            let m = Bessel::modified(-nu, z);
            return Modified { i: m.i + (2.0 / PI) * Trig::sin_pi(-nu) * m.k, k: m.k };
        }
        if z.re == 0.0 {
            return Bessel::modified_via_cylinder(nu, z);
        }
        if z.re < 0.0 {
            // I(z' e^{±iπ}) = e^{±iνπ} I(z') ; K(z' e^{±iπ}) = e^{∓iνπ} K(z') ∓ iπ I(z').
            let zr = -z;
            let s: f64 = if z.im.is_sign_negative() { -1.0 } else { 1.0 };
            let m = Bessel::modified(nu, zr);
            let rot = Complex::exp(Complex::new(0.0, s * nu * PI));
            return Modified { i: rot * m.i, k: m.k / rot - Complex::new(0.0, s * PI) * m.i };
        }
        if z.re.abs() < 0.25 * z.magnitude() && z.magnitude() >= 2.0 {
            return Bessel::modified_via_cylinder(nu, z);
        }
        Bessel::modified_right_half(nu, z)
    }

    pub fn i(nu: f64, z: Complex) -> Complex {
        Bessel::modified(nu, z).i
    }

    pub fn k(nu: f64, z: Complex) -> Complex {
        Bessel::modified(nu, z).k
    }
}

/// Ai and Bi: the Maclaurin series inside |z| = 2, the Bessel forms
/// (DLMF 9.6) for |arg z| < 2π/3, and the rotation identities (9.2.10,
/// 9.2.12) in the remaining sector, where both rotated points fall back into
/// the Bessel one.
pub enum Airy {}

const AI0: f64 = 0.3550280538878172;
const AIP0: f64 = -0.2588194037928068;
const BI0: f64 = 0.6149266274460007;
const BIP0: f64 = 0.4482883573538264;
static OMEGA: LazyLock<Complex> = LazyLock::new(|| Complex::exp(Complex::new(0.0, 2.0 * PI / 3.0)));

/// Ai and Bi at one point: Swift's `(ai:, bi:)` tuple.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct AiryPair {
    pub ai: Complex,
    pub bi: Complex,
}

impl Airy {
    fn series(z: Complex) -> AiryPair {
        let z3 = z * z * z;
        let mut f = Complex::ONE;
        let mut g = z;
        let mut tf = Complex::ONE;
        let mut tg = z;
        let mut k = 0;
        loop {
            k += 1;
            tf *= z3 / f64::from((3 * k - 1) * (3 * k));
            tg *= z3 / f64::from((3 * k) * (3 * k + 1));
            f += tf;
            g += tg;
            if tf.magnitude() + tg.magnitude() <= 2.220446049250313e-16 * (f.magnitude() + g.magnitude()) || k > 200 {
                break;
            }
        }
        AiryPair { ai: AI0 * f + AIP0 * g, bi: BI0 * f + BIP0 * g }
    }

    fn bessel(z: Complex) -> AiryPair {
        let zeta = (2.0 / 3.0) * Complex::pow(z, Complex::real(1.5));
        let root = Complex::sqrt(z / 3.0);
        let k = Bessel::k(1.0 / 3.0, zeta);
        let ip = Bessel::i(1.0 / 3.0, zeta);
        let im = Bessel::i(-1.0 / 3.0, zeta);
        AiryPair { ai: root * k / PI, bi: root * (im + ip) }
    }

    pub fn airy(z: Complex) -> AiryPair {
        if z.magnitude() <= 2.0 {
            return Airy::series(z);
        }
        if z.argument().abs() < 2.0 * PI / 3.0 - 1e-9 {
            return Airy::bessel(z);
        }
        let omega = *OMEGA;
        let a1 = Airy::bessel(omega * z).ai;
        let a2 = Airy::bessel(z / omega).ai;
        let ai = -(omega * a1 + a2 / omega);
        let bi = Complex::exp(Complex::new(0.0, PI / 6.0)) * a1 + Complex::exp(Complex::new(0.0, -PI / 6.0)) * a2;
        AiryPair { ai, bi }
    }

    pub fn ai(z: Complex) -> Complex {
        Airy::airy(z).ai
    }

    pub fn bi(z: Complex) -> Complex {
        Airy::airy(z).bi
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Where Swift's `Int(x)` traps -- an infinite or enormous order, or an
    /// infinite argument on the Hankel path with an order of one or more --
    /// the port answers NaN (measured against Swift built from
    /// `Sources/KurvenMath`: each of these exits on SIGTRAP there).
    #[test]
    fn where_swift_traps_the_port_answers_nan() {
        let one = Complex::real(1.0);
        let inf = Complex::real(f64::INFINITY);
        for nu in [f64::INFINITY, 1e300] {
            assert!(Bessel::j(nu, one).is_nan() && Bessel::y(nu, one).is_nan(), "J, Y of order {nu}");
        }
        assert!(Bessel::i(f64::INFINITY, one).is_nan() && Bessel::k(f64::INFINITY, one).is_nan());
        for nu in [2.7, -2.3] {
            assert!(Bessel::j(nu, inf).is_nan() && Bessel::y(nu, inf).is_nan(), "J, Y of order {nu} at infinity");
        }
    }
}
