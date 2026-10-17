import Foundation

/// The deck group of a torus: translations by the two periods of a rectangle.
///
/// A doubly periodic function lives on the plane, and the plane is where its
/// ink is computed: sampled, contoured, placed on f, carried to its zeros and
/// poles. The torus is the quotient of the plane by the lattice of periods,
/// and the rectangle a plate draws is one fundamental domain of that
/// quotient. It is not a chart of the torus, and its edges are not edges of
/// anything: a zero of sn on the real axis is a zero in the interior of the
/// plane, with the whole fan of phase lines round it, and the ink round it is
/// computed there, on a window of the plane that reaches past the rectangle
/// on every side (`PeriodicTorus.plate`). What the rectangle decides is which
/// copy of each curve is kept.
///
/// Typed accordingly. `DomainSpace` is the plane, the cover; `ParamSpace` is
/// the rectangle, the quotient; `canonical` is the passage from the first to
/// the second, and it applies the group. A function on the torus has no
/// seam. A drawing of it has one, and this is where it is drawn.
public struct PeriodLattice: Sendable, Equatable {
    /// The fundamental rectangle, `[real.lo, real.hi) × [imag.lo, imag.hi)`.
    public let cell: Domain

    public init(cell: Domain) { self.cell = cell }

    /// The translation of the group that brings `p` into the rectangle.
    func shift(bringing p: P2<DomainSpace>) -> SIMD2<Double> {
        SIMD2(-((p.x - cell.real.lo) / cell.real.length).rounded(.down) * cell.real.length,
              -((p.y - cell.imag.lo) / cell.imag.length).rounded(.down) * cell.imag.length)
    }

    /// A run of the plane carried into the rectangle by one translation of
    /// the group: the one that brings its middle vertex inside. One
    /// translation for the whole run rather than one per vertex, so that a
    /// run cut to the rectangle (`clipped`) keeps an end on the far edge
    /// *on* the far edge, where it coincides, on the torus, with the start of
    /// its continuation from the near one.
    public func canonical(_ run: [P2<DomainSpace>]) -> [P2<ParamSpace>] {
        guard !run.isEmpty else { return [] }
        let s = shift(bringing: run[run.count / 2])
        return run.map { P2<ParamSpace>($0.x + s.x, $0.y + s.y) }
    }

    /// The stretches of a path that lie in the rectangle, each cut where the
    /// path crosses an edge, with the crossing itself for its end vertex.
    ///
    /// Half-open: a vertex on the far edge of either axis belongs to the next
    /// copy of the rectangle. Ink lying along a seam -- sn's lines of phase
    /// ±π/2 run the length of its imaginary period -- is therefore kept once,
    /// from the copy on the near edge. `slack` is the rounding allowed in that
    /// decision, a small fraction of a cell.
    ///
    /// The crossing is the chord's unless `onEdge` improves it: it is given
    /// the inside vertex, the outside vertex and the chord's crossing of the
    /// edge, and returns the crossing to use, so a caller with the function
    /// can solve the level along the edge. The copy of the curve one period
    /// over, from which the continuation comes, was computed from the same
    /// sample positions and agrees with it to rounding either way.
    public func clipped(_ path: [P2<DomainSpace>], slack: Double,
                        onEdge: (P2<DomainSpace>, P2<DomainSpace>, P2<DomainSpace>) -> P2<DomainSpace>
                            = { _, _, c in c }) -> [[P2<DomainSpace>]] {
        // The rectangle as clipped, shifted down by the slack on both
        // axes, so that the inside test and the crossing agree on where the
        // edge is: a zero or pole lying on the far edge to rounding is
        // outside, and the run to it ends on the edge within the slack of
        // it, not a whole cell short.
        let lo = (x: cell.real.lo - slack, y: cell.imag.lo - slack)
        let hi = (x: cell.real.hi - slack, y: cell.imag.hi - slack)
        func inside(_ p: P2<DomainSpace>) -> Bool {
            p.x >= lo.x && p.x < hi.x && p.y >= lo.y && p.y < hi.y
        }
        /// Where the segment from `a`, inside, toward `b`, outside, first
        /// reaches an edge of the rectangle: `b` itself when it lies on the
        /// edge. Nil when the segment reaches none, or reaches one at `a`,
        /// which happens only when `a` is on the edge already and the run
        /// ends at `a` as it is. The point is put on the rectangle's own
        /// edge, not the shifted one, so that a crossing of the far edge and
        /// its continuation from the near edge are one point of the torus
        /// whatever the slack.
        func crossing(from a: P2<DomainSpace>, toward b: P2<DomainSpace>) -> P2<DomainSpace>? {
            var t = Double.infinity
            var reached: (axis: Int, exact: Double)?
            let axes = [(a.x, b.x, [(lo.x, cell.real.lo), (hi.x, cell.real.hi)]),
                        (a.y, b.y, [(lo.y, cell.imag.lo), (hi.y, cell.imag.hi)])]
            for (axis, (pa, pb, edges)) in axes.enumerated() {
                for (edge, exact) in edges where pa != pb && (pa - edge) * (pb - edge) <= 0 {
                    let s = (edge - pa) / (pb - pa)
                    if s < t { t = s; reached = (axis, exact) }
                }
            }
            guard let reached, t > 0, t <= 1 else { return nil }
            let x = a.x + (b.x - a.x) * t, y = a.y + (b.y - a.y) * t
            return reached.axis == 0 ? P2(reached.exact, y) : P2(x, reached.exact)
        }
        var out: [[P2<DomainSpace>]] = []
        var run: [P2<DomainSpace>] = []
        var previous: (p: P2<DomainSpace>, inside: Bool)?
        for p in path {
            let now = inside(p)
            if now {
                if let q = previous, !q.inside, let c = crossing(from: p, toward: q.p) {
                    run.append(onEdge(p, q.p, c))
                }
                run.append(p)
            } else if let q = previous, q.inside {
                if let c = crossing(from: q.p, toward: p) { run.append(onEdge(q.p, p, c)) }
                if run.count >= 2 { out.append(run) }
                run = []
            }
            previous = (p, now)
        }
        // A closed path that begins inside and leaves is cut at its first
        // vertex as well as at the edges; the stretch that returns to that
        // vertex continues into the stretch that left it.
        if run.count >= 2 {
            if path.first == path.last, out.count >= 1, out[0].first == path.first {
                out[0] = run + out[0].dropFirst()
            } else {
                out.append(run)
            }
        }
        return out
    }
}
