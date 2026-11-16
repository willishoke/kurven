//! Bisection to the last bit: the port of `Sources/KurvenCore/Bisection.swift`.

use crate::simd::Vec2;

/// Bisection to the last bit: the bracket from `a`, where `holds` is true,
/// to `b`, where it is not, halved sixty times at its midpoint
/// 0.5 * (a + b), each time keeping the half whose ends still disagree.
/// Sixty halvings take a bracket a grid cell wide past the last bit of its
/// ends. The caller checks that the ends disagree; the answer is the final
/// bracket, `holds` true at its first end and false at its second, and most
/// callers take its midpoint.
///
/// A sign change is a `holds` that compares with the sign at `a`:
/// `bisect(a, b, |t| (f(t) > 0.0) == (f(a) > 0.0))`, with f(a) computed once.
pub fn bisect(a: f64, b: f64, holds: impl FnMut(f64) -> bool) -> (f64, f64) {
    bisect_by(a, b, |a, b| 0.5 * (a + b), holds)
}

/// `bisect` along the segment between two points.
pub fn bisect_points(a: Vec2, b: Vec2, holds: impl FnMut(Vec2) -> bool) -> (Vec2, Vec2) {
    bisect_by(a, b, |a, b| 0.5 * (a + b), holds)
}

pub(crate) fn bisect_by<V: Copy>(a: V, b: V, midpoint: impl Fn(V, V) -> V, mut holds: impl FnMut(V) -> bool) -> (V, V) {
    let (mut a, mut b) = (a, b);
    for _ in 0..60 {
        let m = midpoint(a, b);
        if holds(m) {
            a = m;
        } else {
            b = m;
        }
    }
    (a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bisects_to_the_last_bit() {
        let (lo, hi) = bisect(0.0, 2.0, |t| t * t < 2.0);
        assert!(lo * lo < 2.0 && hi * hi >= 2.0);
        assert!(hi - lo <= 2.0 * f64::EPSILON * 2.0);
        let (p, q) = bisect_points(Vec2::new(0.0, 0.0), Vec2::new(4.0, 2.0), |v| v.x < 1.0);
        assert!(p.x < 1.0 && q.x >= 1.0 && (q - p).length() < 1e-15);
    }
}
