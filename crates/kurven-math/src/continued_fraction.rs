//! The continued fraction of a number, and the fractions it is nearly. The
//! port of `ContinuedFraction.swift`.
//!
//! A linear curve at an irrational slope never closes, but it nearly closes at
//! every convergent `p/q` of the slope's continued fraction: after `q` turns
//! it is back within `1/q` of a turn of where it began. Those denominators
//! are where a trajectory drawn for `q` turns cuts its torus most evenly
//! (the three-gap theorem), and those fractions are the linear curves a
//! quasi-periodic orbit would lock to. Both are what a slider should rest on.

pub enum ContinuedFraction {}

/// A fraction `p/q` in lowest terms.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Convergent {
    pub p: i64,
    pub q: i64,
}

impl Convergent {
    pub fn new(p: i64, q: i64) -> Convergent {
        Convergent { p, q }
    }

    pub fn value(self) -> f64 {
        self.p as f64 / self.q as f64
    }
}

impl ContinuedFraction {
    /// The partial quotients `[a₀; a₁, a₂, …]` of `x`, at most `count` of
    /// them, stopping when the remainder is zero to within `tolerance` -- a
    /// fraction's expansion ends -- or the next quotient would be
    /// meaningless at double precision. Swift's defaults are `count` 12 and
    /// `tolerance` 1e-12.
    ///
    /// Panics on a number that is not finite, as Swift's precondition traps.
    pub fn quotients(x: f64, count: usize, tolerance: f64) -> Vec<i64> {
        assert!(x.is_finite(), "a continued fraction wants a finite number");
        let mut out = Vec::new();
        let mut r = x;
        for _ in 0..count {
            let a = r.floor();
            if a.abs() >= 1e15 {
                break;
            }
            out.push(a as i64);
            let frac = r - a;
            if frac <= tolerance * x.abs().max(1.0) {
                break;
            }
            r = 1.0 / frac;
        }
        out
    }

    /// The convergents of `x` with denominators up to `max_denominator`,
    /// smallest first: `p/q = [a₀; a₁, …, aₖ]` for each `k`. Swift's defaults
    /// are `max_denominator` 1,000,000 and `tolerance` 1e-12.
    pub fn convergents(x: f64, max_denominator: i64, tolerance: f64) -> Vec<Convergent> {
        let mut out = Vec::new();
        let (mut p0, mut q0, mut p1, mut q1) = (1i64, 0i64, 0i64, 1i64); // h₋₁/k₋₁ and h₋₂/k₋₂
        for a in ContinuedFraction::quotients(x, 40, tolerance) {
            let (p, q) = (a * p0 + p1, a * q0 + q1);
            if q > max_denominator {
                break;
            }
            out.push(Convergent::new(p, q));
            (p1, q1, p0, q0) = (p0, q0, p, q);
        }
        out
    }

    /// `[a₀; a₁, a₂, …]`, as it is written.
    pub fn describe(quotients: &[i64], ellipsis: bool) -> String {
        let Some(first) = quotients.first() else { return "[]".to_string() };
        let rest = quotients[1..].iter().map(i64::to_string).collect::<Vec<_>>().join(", ");
        format!(
            "[{first}{}{}]",
            if rest.is_empty() { String::new() } else { format!("; {rest}") },
            if ellipsis { ", …" } else { "" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `continuedFractionTests` in `Sources/kurven-test/Dynamics.swift`.
    #[test]
    fn the_fractions_a_number_is_nearly() {
        fn names(c: &[Convergent]) -> Vec<String> {
            c.iter().map(|c| format!("{}/{}", c.p, c.q)).collect()
        }
        let pi = core::f64::consts::PI;
        assert_eq!(names(&ContinuedFraction::convergents(pi, 200, 1e-12)), ["3/1", "22/7", "333/106", "355/113"]);
        assert_eq!(ContinuedFraction::quotients(pi, 5, 1e-12), [3, 7, 15, 1, 292]);
        assert_eq!(ContinuedFraction::quotients(0.4, 12, 1e-12), [0, 2, 2]);
        assert_eq!(names(&ContinuedFraction::convergents(0.4, 1_000_000, 1e-12)), ["0/1", "1/2", "2/5"]);
        let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
        assert!(ContinuedFraction::quotients(phi, 8, 1e-12).iter().all(|&a| a == 1));
        assert_eq!(
            names(&ContinuedFraction::convergents(phi, 13, 1e-12)),
            ["1/1", "2/1", "3/2", "5/3", "8/5", "13/8", "21/13"]
        );
        assert_eq!(ContinuedFraction::describe(&[0, 2, 2, 1, 2, 5], true), "[0; 2, 2, 1, 2, 5, …]");
        assert_eq!(ContinuedFraction::describe(&[3], false), "[3]");
        assert_eq!(ContinuedFraction::convergents(2.5, 1, 1e-12), [Convergent::new(2, 1)]);
    }
}
