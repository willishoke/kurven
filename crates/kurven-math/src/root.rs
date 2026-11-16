//! Zeros of a real function of one variable. The port of `Root.swift`.

use crate::swift;

pub enum Root {}

impl Root {
    /// A zero of `f` between `a` and `b`, whose values `fa` and `fb` there
    /// differ in sign, by Brent's method (R. P. Brent, *Algorithms for
    /// Minimization without Derivatives*, 1973, ch. 4).
    ///
    /// Each step tries inverse quadratic interpolation through the last three
    /// points, or a secant step through the last two, and keeps it only when
    /// it lands inside the bracket and shrinks it fast enough; otherwise it
    /// bisects. So it converges wherever bisection would, and superlinearly
    /// where f is smooth. This is the iteration of scipy's `brentq`.
    ///
    /// Stops when the bracket is narrower than `xtol` plus a few ulps of the
    /// zero, or when |f| ≤ `ftol`. Returns None when the ends do not bracket a
    /// zero, or when f is not finite at a point the iteration asks for; after
    /// `max_iterations` steps, the best point so far. Swift's defaults are
    /// `ftol` 0 and `max_iterations` 100; [`Root::brent_default`] takes them.
    #[allow(clippy::too_many_arguments)]
    pub fn brent(
        mut f: impl FnMut(f64) -> f64,
        a: f64,
        b: f64,
        fa: f64,
        fb: f64,
        xtol: f64,
        ftol: f64,
        max_iterations: usize,
    ) -> Option<f64> {
        let rtol = 4.0 * f64::EPSILON;
        let (mut xpre, mut xcur, mut fpre, mut fcur) = (a, b, fa, fb);
        if !(fpre.is_finite() && fcur.is_finite()) {
            return None;
        }
        if fpre == 0.0 {
            return Some(xpre);
        }
        if fcur == 0.0 {
            return Some(xcur);
        }
        if fpre.is_sign_negative() == fcur.is_sign_negative() {
            return None;
        }
        // The other end of the bracket, and the last two step lengths.
        let (mut xblk, mut fblk, mut spre, mut scur) = (0.0, 0.0, 0.0, 0.0);
        for _ in 0..max_iterations {
            if fpre != 0.0 && fcur != 0.0 && fpre.is_sign_negative() != fcur.is_sign_negative() {
                xblk = xpre;
                fblk = fpre;
                spre = xcur - xpre;
                scur = spre;
            }
            // Keep xcur the better of the two ends.
            if fblk.abs() < fcur.abs() {
                xpre = xcur;
                xcur = xblk;
                xblk = xpre;
                fpre = fcur;
                fcur = fblk;
                fblk = fpre;
            }
            let delta = (xtol + rtol * xcur.abs()) / 2.0;
            let sbis = (xblk - xcur) / 2.0;
            if fcur == 0.0 || sbis.abs() < delta || fcur.abs() <= ftol {
                return Some(xcur);
            }
            if spre.abs() > delta && fcur.abs() < fpre.abs() {
                let stry = if xpre == xblk {
                    -fcur * (xcur - xpre) / (fcur - fpre)
                } else {
                    let dpre = (fpre - fcur) / (xpre - xcur);
                    let dblk = (fblk - fcur) / (xblk - xcur);
                    -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
                };
                if 2.0 * stry.abs() < swift::min(spre.abs(), 3.0 * sbis.abs() - delta) {
                    spre = scur;
                    scur = stry;
                } else {
                    spre = sbis;
                    scur = sbis;
                }
            } else {
                spre = sbis;
                scur = sbis;
            }
            xpre = xcur;
            fpre = fcur;
            xcur += if scur.abs() > delta {
                scur
            } else if sbis > 0.0 {
                delta
            } else {
                -delta
            };
            fcur = f(xcur);
            if !fcur.is_finite() {
                return None;
            }
        }
        Some(xcur)
    }

    /// [`Root::brent`] with Swift's defaults: `ftol` 0, 100 iterations.
    pub fn brent_default(f: impl FnMut(f64) -> f64, a: f64, b: f64, fa: f64, fb: f64, xtol: f64) -> Option<f64> {
        Root::brent(f, a, b, fa, fb, xtol, 0.0, 100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::real;

    /// `refinementTests`' first suite in `Sources/kurven-test/Expressions.swift`.
    #[test]
    fn brent_converges_where_bisection_would_and_faster() {
        let mut calls = 0;
        // Dottie's number: bisection needs forty halvings for 1e-12.
        let dottie = 0.7390851332151607;
        let x = Root::brent_default(
            |x| {
                calls += 1;
                real::cos(x) - x
            },
            0.0,
            1.0,
            1.0,
            real::cos(1.0) - 1.0,
            1e-12,
        );
        assert!(x.is_some_and(|x| (x - dottie).abs() <= 1e-12) && calls <= 10, "cos x = x: {x:?} after {calls}");
        // A bracket like exp(1/z)'s beside its essential singularity: -1 at
        // one end, 1e76 at the other, where interpolation lands beside the
        // small end every time and only bisection makes progress.
        let steep = |x: f64| real::exp(250.0 * (x - 0.3)) - 1.0;
        let mut calls = 0;
        let y = Root::brent_default(
            |x| {
                calls += 1;
                steep(x)
            },
            0.0,
            1.0,
            steep(0.0),
            steep(1.0),
            1e-12,
        );
        assert!(y.is_some_and(|y| (y - 0.3).abs() <= 1e-11) && calls <= 120, "-1 against 1e76: {y:?} after {calls}");
        assert_eq!(Root::brent_default(|x| x, 0.0, 1.0, 0.0, 1.0, 1e-12), Some(0.0), "a zero at an end is that end");
        assert_eq!(Root::brent_default(|x| x * x + 1.0, -1.0, 1.0, 2.0, 2.0, 1e-12), None, "ends of one sign");
        assert_eq!(
            Root::brent_default(|x| if x < 0.5 { -1.0 } else { f64::NAN }, 0.0, 1.0, -1.0, 1.0, 1e-12),
            None,
            "a value that is not finite ends the search"
        );
    }
}
