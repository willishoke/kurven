//! The deck group of a torus: the port of `Sources/KurvenCore/Cover.swift`.

use crate::manifest::Domain;
use crate::simd::Vec2;
use crate::spaces::{DomainSpace, P2, ParamSpace};

/// The deck group of a torus: translations by the two periods of a rectangle.
///
/// A doubly periodic function lives on the plane, and the plane is where its
/// curves are computed: sampled, contoured, placed on f, carried to its zeros and
/// poles. The torus is the quotient of the plane by the lattice of periods,
/// and the rectangle a drawing draws is one fundamental domain of that
/// quotient. It is not a chart of the torus, and its edges are not edges of
/// anything: a zero of sn on the real axis is a zero in the interior of the
/// plane, with the whole fan of phase lines round it, and the curves round it are
/// computed there, on a window of the plane that reaches past the rectangle
/// on every side (`PeriodicTorus::drawing`). What the rectangle decides is which
/// copy of each curve is kept.
///
/// Typed accordingly. `DomainSpace` is the plane, the cover; `ParamSpace` is
/// the rectangle, the quotient; `canonical` is the passage from the first to
/// the second, and it applies the group. A function on the torus has no
/// seam. A drawing of it has one, and this is where it is drawn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeriodLattice {
    /// The fundamental rectangle, `[real.lo, real.hi) × [imag.lo, imag.hi)`.
    pub cell: Domain,
}

impl PeriodLattice {
    pub fn new(cell: Domain) -> PeriodLattice {
        PeriodLattice { cell }
    }

    /// The translation of the group that brings `p` into the rectangle.
    fn shift(&self, p: P2<DomainSpace>) -> Vec2 {
        let (real, imag) = (self.cell.real, self.cell.imag);
        Vec2::new(
            -((p.x() - real.lo) / real.length()).floor() * real.length(),
            -((p.y() - imag.lo) / imag.length()).floor() * imag.length(),
        )
    }

    /// A run of the plane carried into the rectangle by one translation of
    /// the group: the one that brings its middle vertex inside. One
    /// translation for the whole run rather than one per vertex, so that a
    /// run cut to the rectangle (`clipped`) keeps an end on the far edge
    /// *on* the far edge, where it coincides, on the torus, with the start of
    /// its continuation from the near one.
    pub fn canonical(&self, run: &[P2<DomainSpace>]) -> Vec<P2<ParamSpace>> {
        if run.is_empty() {
            return Vec::new();
        }
        let s = self.shift(run[run.len() / 2]);
        run.iter().map(|p| P2::new(p.x() + s.x, p.y() + s.y)).collect()
    }

    /// `clipped` with the chord's crossing kept as it is (Swift's default
    /// `onEdge`).
    pub fn clipped_plain(&self, path: &[P2<DomainSpace>], slack: f64) -> Vec<Vec<P2<DomainSpace>>> {
        self.clipped(path, slack, |_, _, c| c)
    }

    /// The stretches of a path that lie in the rectangle, each cut where the
    /// path crosses an edge, with the crossing itself for its end vertex.
    ///
    /// Half-open: a vertex on the far edge of either axis belongs to the next
    /// copy of the rectangle. Curves lying along a seam -- sn's lines of phase
    /// ±π/2 run the length of its imaginary period -- is therefore kept once,
    /// from the copy on the near edge. `slack` is the rounding allowed in that
    /// decision, a small fraction of a cell.
    ///
    /// The crossing is the chord's unless `on_edge` improves it: it is given
    /// the inside vertex, the outside vertex and the chord's crossing of the
    /// edge, and returns the crossing to use, so a caller with the function
    /// can solve the level along the edge. The copy of the curve one period
    /// over, from which the continuation comes, was computed from the same
    /// sample positions and agrees with it to rounding either way.
    pub fn clipped(
        &self,
        path: &[P2<DomainSpace>],
        slack: f64,
        mut on_edge: impl FnMut(P2<DomainSpace>, P2<DomainSpace>, P2<DomainSpace>) -> P2<DomainSpace>,
    ) -> Vec<Vec<P2<DomainSpace>>> {
        let (real, imag) = (self.cell.real, self.cell.imag);
        // The rectangle as clipped, shifted down by the slack on both
        // axes, so that the inside test and the crossing agree on where the
        // edge is: a zero or pole lying on the far edge to rounding is
        // outside, and the run to it ends on the edge within the slack of
        // it, not a whole cell short.
        let lo = (real.lo - slack, imag.lo - slack);
        let hi = (real.hi - slack, imag.hi - slack);
        let inside = |p: P2<DomainSpace>| p.x() >= lo.0 && p.x() < hi.0 && p.y() >= lo.1 && p.y() < hi.1;
        // Where the segment from `a`, inside, toward `b`, outside, first
        // reaches an edge of the rectangle: `b` itself when it lies on the
        // edge. None when the segment reaches none, or reaches one at `a`,
        // which happens only when `a` is on the edge already and the run
        // ends at `a` as it is. The point is put on the rectangle's own
        // edge, not the shifted one, so that a crossing of the far edge and
        // its continuation from the near edge are one point of the torus
        // whatever the slack.
        let crossing = |a: P2<DomainSpace>, b: P2<DomainSpace>| -> Option<P2<DomainSpace>> {
            let mut t = f64::INFINITY;
            let mut reached: Option<(usize, f64)> = None;
            let axes = [
                (a.x(), b.x(), [(lo.0, real.lo), (hi.0, real.hi)]),
                (a.y(), b.y(), [(lo.1, imag.lo), (hi.1, imag.hi)]),
            ];
            for (axis, (pa, pb, edges)) in axes.into_iter().enumerate() {
                for (edge, exact) in edges {
                    if pa != pb && (pa - edge) * (pb - edge) <= 0.0 {
                        let s = (edge - pa) / (pb - pa);
                        if s < t {
                            t = s;
                            reached = Some((axis, exact));
                        }
                    }
                }
            }
            let (axis, exact) = reached?;
            if !(t > 0.0 && t <= 1.0) {
                return None;
            }
            let x = a.x() + (b.x() - a.x()) * t;
            let y = a.y() + (b.y() - a.y()) * t;
            Some(if axis == 0 { P2::new(exact, y) } else { P2::new(x, exact) })
        };
        let mut out: Vec<Vec<P2<DomainSpace>>> = Vec::new();
        let mut run: Vec<P2<DomainSpace>> = Vec::new();
        let mut previous: Option<(P2<DomainSpace>, bool)> = None;
        for &p in path {
            let now = inside(p);
            if now {
                if let Some((q, false)) = previous
                    && let Some(c) = crossing(p, q)
                {
                    run.push(on_edge(p, q, c));
                }
                run.push(p);
            } else if let Some((q, true)) = previous {
                if let Some(c) = crossing(q, p) {
                    run.push(on_edge(q, p, c));
                }
                if run.len() >= 2 {
                    out.push(std::mem::take(&mut run));
                }
                run.clear();
            }
            previous = Some((p, now));
        }
        // A closed path that begins inside and leaves is cut at its first
        // vertex as well as at the edges; the stretch that returns to that
        // vertex continues into the stretch that left it.
        if run.len() >= 2 {
            if path.first() == path.last() && !out.is_empty() && out[0].first() == path.first() {
                run.extend_from_slice(&out[0][1..]);
                out[0] = run;
            } else {
                out.push(run);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Interval;

    fn lattice() -> PeriodLattice {
        PeriodLattice::new(Domain::new(Interval::new(0.0, 4.0), Interval::new(0.0, 2.0)))
    }

    #[test]
    fn a_run_is_carried_in_by_its_middle_vertex() {
        let run = [P2::new(4.5, 2.5), P2::new(5.0, 3.0), P2::new(5.5, 3.5)];
        let c = lattice().canonical(&run);
        assert_eq!(c, vec![P2::new(0.5, 0.5), P2::new(1.0, 1.0), P2::new(1.5, 1.5)]);
    }

    #[test]
    fn a_path_across_an_edge_ends_on_the_edge_and_a_closed_one_is_joined() {
        let l = lattice();
        let across = [P2::new(3.0, 1.0), P2::new(5.0, 1.0)];
        assert_eq!(l.clipped_plain(&across, 0.0), vec![vec![P2::new(3.0, 1.0), P2::new(4.0, 1.0)]]);
        // A loop from inside out across the right edge and back: one stretch
        // from the edge round to the edge, its first vertex not a cut.
        let lp = [P2::new(3.0, 0.5), P2::new(5.0, 0.5), P2::new(5.0, 1.5), P2::new(3.0, 1.5), P2::new(3.0, 0.5)];
        let out = l.clipped_plain(&lp, 0.0);
        assert_eq!(out, vec![vec![P2::new(4.0, 1.5), P2::new(3.0, 1.5), P2::new(3.0, 0.5), P2::new(4.0, 0.5)]]);
    }
}
