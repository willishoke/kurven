import Foundation
import Dispatch
import KurvenCore
import KurvenMath

/// Contours placed by the function rather than by the grid.
///
/// Marching squares puts a contour vertex where *linear interpolation* between
/// two samples crosses the level, which is off the true level set by about
/// h² |f''| / |f'| -- nothing where f is nearly linear across a cell, and
/// growing like 1/r toward a pole or a zero, which is exactly where the plates
/// are densest. The Python pipeline answered that with a second, finer grid
/// in rectangles around the steep places, because there every extra sample
/// was scipy time. Here the function is a call, so the answer is to ask it:
///
///  1. **Snap.** Every vertex lies on a grid edge whose two samples bracket the
///     level. A one-dimensional root find along that edge, against f itself,
///     moves the vertex to the exact crossing. Regula falsi with the Illinois
///     modification, half a dozen evaluations.
///  2. **Subdivide.** Between two snapped vertices the contour may curve away
///     from the chord. The residual at the chord's midpoint, divided by the
///     gradient there, estimates how far off the chord is; where that exceeds
///     a fraction of a cell, a new vertex is solved for along the normal and
///     the two halves are checked in turn. Density ends up proportional to
///     curvature, continuously, and with no seams to stitch because there is
///     only one contour pass.
///
/// The grid still decides which contours exist and roughly where; f decides
/// exactly where. The lift stays the grid's, so the ink stays on the drawn
/// surface -- see `Surface.derive`.
///
/// Phase contours wrap. Where arg f passes from π to -π, marching squares on
/// the wrapped grid interpolates straight through the jump and emits a crossing
/// for *every* level between: a bundle of spurious segments along each wrap
/// line, in the Python plates as much as here (a third of the gamma plate's
/// phase vertices, measured). Two samples cannot tell such a vertex from a
/// real one beside a zero, where the phase turns a full circle within a cell
/// and one edge can legitimately span more than π of it. So the phase is
/// tracked through the edge's midpoint, each half the short way round, and a
/// vertex is a crossing only where that path passes the level (`bracket`);
/// otherwise it is dropped and the path split there. Residuals of the phase
/// are taken the short way round throughout, so a level near π is as near to
/// a sample at -π as it is. A chord midpoint whose residual is a jump is
/// likewise left alone rather than solved toward the wrong branch.
///
/// Every phase meets at a zero of f, and a phase contour ends there. The grid
/// stops it at the last cell edge before the zero; the refiner, which has f,
/// carries each run whose end lies within a cell or so of a zero -- found by
/// Newton from the end -- to the zero itself (`ended(atZeros:)`).
public struct ContourRefiner: Sendable {
    /// f itself, for the roots the phase contours end at.
    public let value: @Sendable (Complex) -> Complex
    public let magnitude: @Sendable (Complex) -> Double
    public let phase: @Sendable (Complex) -> Double
    /// arg(-f): the phase turned a half turn, whose zero level is f's cut.
    public let turnedPhase: @Sendable (Complex) -> Double
    public let grid: Domain
    public let width: Int
    public let height: Int
    /// Subdivide while the chord is estimated to be further than this fraction
    /// of a cell from the level set.
    public var tolerance: Double
    /// How many times one grid-level segment may be halved.
    public var maxDepth: Int

    public init(_ compiled: KurvenMath.Expression.Compiled, domain: Domain,
                width: Int, height: Int, tolerance: Double = 0.02, maxDepth: Int = 5) {
        self.value = { z in compiled(z) }
        self.magnitude = { z in
            let v = compiled(z)
            return v.isFinite ? min(v.magnitude, NativeLandscape.huge) : NativeLandscape.huge
        }
        self.phase = { z in
            let v = compiled(z)
            return v.isFinite ? v.argument : 0
        }
        self.turnedPhase = { z in
            let v = compiled(z)
            return v.isFinite ? (-v).argument : 0
        }
        self.grid = domain; self.width = width; self.height = height
        self.tolerance = tolerance; self.maxDepth = maxDepth
    }

    var dx: Double { grid.real.length / Double(max(width - 1, 1)) }
    var dy: Double { grid.imag.length / Double(max(height - 1, 1)) }
    /// The cell size the tolerances are measured against.
    public var cell: Double { max(abs(dx), abs(dy)) }

    /// An angle brought into [-π, π]: the short way round.
    static func wrapped(_ angle: Double) -> Double {
        angle - (angle / (2 * .pi)).rounded() * 2 * .pi
    }

    /// How far a sample is from the level: for the phase, the short way round.
    func residual(_ sample: Double, level: Double, cut: Bool) -> Double {
        cut ? Self.wrapped(sample - level) : sample - level
    }

    /// Whether a domain point is inside the grid's window, to rounding: the
    /// zeros of a function sampled on a half-plane sit on its edge.
    func inWindow(_ p: P2<DomainSpace>) -> Bool {
        let r = grid.real, i = grid.imag, slack = 1e-9 * cell
        return p.x >= min(r.lo, r.hi) - slack && p.x <= max(r.lo, r.hi) + slack
            && p.y >= min(i.lo, i.hi) - slack && p.y <= max(i.lo, i.hi) + slack
    }

    func field(_ f: ContourField) -> @Sendable (Complex) -> Double {
        f == .magnitude ? magnitude : phase
    }

    /// The scalar field and level a contour is solved on. A phase level of
    /// ±π is the cut, which `Surface.derive` contours as the zero level of
    /// arg(-f) (`Contour.cut`), and is placed on that.
    func field(_ f: ContourField, level: Double) -> (g: @Sendable (Complex) -> Double, level: Double) {
        if f == .phase && Contour.isCut(level) { return (turnedPhase, 0) }
        return (field(f), level)
    }

    /// `Surface.derive`'s hook, and `Surface.crest`'s and
    /// `Heightfield.foldLines`'s: the contours placed, |f| for the crest's
    /// minima and the zeros' heights, and the zeros the fold lines are
    /// carried to. The magnitude is the one the grid was sampled with,
    /// clamped at `NativeLandscape.huge` beside a pole.
    public var refine: ContourRefine {
        ContourRefine(contours: { field, level, paths in
                          self.refine(field: field, level: level, paths)
                      },
                      magnitude: { p in self.magnitude(Complex(p.x, p.y)) },
                      zero: { p in self.zero(near: p) })
    }

    public func refine(field: ContourField, level: Double,
                       _ paths: [[P2<DomainSpace>]]) -> [[P2<DomainSpace>]] {
        let (g, level) = self.field(field, level: level)
        let cut = field == .phase
        var out = [[[P2<DomainSpace>]]](repeating: [], count: paths.count)
        out.withUnsafeMutableBufferPointer { buffer in
            // Each iteration writes only its own slot, so sharing is safe.
            nonisolated(unsafe) let base = buffer.baseAddress!
            DispatchQueue.concurrentPerform(iterations: paths.count) { i in
                base[i] = refine(paths[i], level: level, g: g, cut: cut)
            }
        }
        return out.flatMap { $0 }
    }

    // MARK: one path

    /// One grid path, refined: possibly several, where wrap vertices were
    /// dropped.
    func refine(_ path: [P2<DomainSpace>], level: Double,
                g: @Sendable (Complex) -> Double, cut: Bool) -> [[P2<DomainSpace>]] {
        guard path.count >= 2 else { return [] }
        let closed = path.first == path.last
        var snapped = path.map { snap($0, level: level, g: g, cut: cut) }
        if closed { snapped[snapped.count - 1] = snapped[0] }
        var runs: [[P2<DomainSpace>]] = []
        var out: [P2<DomainSpace>] = []
        out.reserveCapacity(snapped.count * 2)
        var previous: P2<DomainSpace>?
        for vertex in snapped {
            guard let vertex else {
                if out.count >= 2 { runs.append(out) }
                out = []
                previous = nil
                continue
            }
            if let previous {
                subdivide(previous, vertex, level: level, g: g, cut: cut, depth: 0, into: &out)
            }
            out.append(vertex)
            previous = vertex
        }
        if out.count >= 2 { runs.append(out) }
        return cut ? runs.map { ended(atZeros: $0, level: level, g: g) } : runs
    }

    /// A phase run carried to the zero of f it ends beside, at either end.
    ///
    /// Every phase meets at a zero, so a phase contour that stops within a
    /// cell or so of one was always going there: the grid's marching squares
    /// put its last vertex on the last cell edge it crossed. The zero is found
    /// from the end by Newton on f, accepted when the iteration converges
    /// without leaving the end's neighbourhood and lands on a value a million
    /// times smaller than the end's, and the new segment is subdivided like
    /// any other. A run that already ends on a zero, or that is closed, is
    /// left alone.
    func ended(atZeros run: [P2<DomainSpace>], level: Double,
               g: @Sendable (Complex) -> Double) -> [P2<DomainSpace>] {
        guard run.count >= 2, run.first != run.last else { return run }
        /// The zero this end leads to, when the chord to it lies on the level
        /// set -- its midpoint's phase within an eighth of a turn of the level,
        /// which a different contour's end beside the same zero fails -- and
        /// the run does not already end there from the other side, as a stub
        /// in a pit a cell wide would.
        func zero(for end: P2<DomainSpace>, other: P2<DomainSpace>) -> P2<DomainSpace>? {
            guard let z = self.zero(near: end) else { return nil }
            let apart = ((z.x - other.x) * (z.x - other.x) + (z.y - other.y) * (z.y - other.y)).squareRoot()
            guard apart > 1e-9 * cell else { return nil }
            let mid = Complex((z.x + end.x) / 2, (z.y + end.y) / 2)
            let r = residual(g(mid), level: level, cut: true)
            return r.isFinite && abs(r) <= .pi / 4 ? z : nil
        }
        var out = run
        if let z = zero(for: out[0], other: out[out.count - 1]) {
            var lead: [P2<DomainSpace>] = []
            subdivide(z, out[0], level: level, g: g, cut: true, depth: 0, into: &lead)
            out = [z] + lead + out
        }
        if let z = zero(for: out[out.count - 1], other: out[0]) {
            subdivide(out[out.count - 1], z, level: level, g: g, cut: true, depth: 0, into: &out)
            out.append(z)
        }
        return out
    }

    /// How far from a run's end a zero is looked for, in cells. The grid's
    /// contour can stop two cells short where the zero's own cells pair
    /// their crossings the other way.
    static let reach = 3.0

    /// A zero of f within `reach` cells of `p`, or nil.
    func zero(near p: P2<DomainSpace>) -> P2<DomainSpace>? {
        var z = Complex(p.x, p.y)
        let f0 = value(z)
        guard f0.isFinite, f0.magnitude > 0 else { return nil }
        let h = 1e-6 * cell
        for _ in 0..<40 {
            let fz = value(z)
            guard fz.isFinite else { return nil }
            let d = (value(z + h) - value(z - h)) / (2 * h)
            guard d.isFinite, d.squaredMagnitude > 0 else { return nil }
            let step = fz / d
            guard step.isFinite else { return nil }
            z = z - step
            guard Complex(z.re - p.x, z.im - p.y).magnitude <= Self.reach * cell else { return nil }
            if step.magnitude <= 1e-12 * cell {
                let at = value(z)
                guard at.isFinite, at.magnitude <= 1e-6 * f0.magnitude else { return nil }
                let q = P2<DomainSpace>(z.re, z.im)
                return inWindow(q) ? q : nil
            }
        }
        return nil
    }

    /// The sub-edge of `a -> b` on which the level is crossed, with the
    /// residuals at its ends, of opposite sign or zero; nil when it is not
    /// crossed.
    ///
    /// For a field without a cut that is the whole edge when the residuals
    /// differ in sign. For the phase, the two samples alone cannot say: a
    /// difference of more than π is a wrap when the samples straddle the cut,
    /// and a genuine turn when the edge passes beside a zero. So the phase is
    /// tracked through the edge's midpoint, each half the short way round --
    /// a straight edge subtends less than a half turn from a simple zero, so
    /// each half is unambiguous -- and the level is crossed on whichever half
    /// that path passes a multiple of 2π of its residual.
    func bracket(_ a: P2<DomainSpace>, _ b: P2<DomainSpace>, level: Double,
                 g: @Sendable (Complex) -> Double, cut: Bool)
        -> (a: P2<DomainSpace>, b: P2<DomainSpace>, ra: Double, rb: Double)? {
        let pa = g(Complex(a.x, a.y)), pb = g(Complex(b.x, b.y))
        guard pa.isFinite, pb.isFinite else { return nil }
        guard cut else {
            let ra = pa - level, rb = pb - level
            return ra == 0 || rb == 0 || ra.sign != rb.sign ? (a, b, ra, rb) : nil
        }
        let m = P2<DomainSpace>((a.x + b.x) / 2, (a.y + b.y) / 2)
        let pm = g(Complex(m.x, m.y))
        guard pm.isFinite else { return nil }
        let ra = Self.wrapped(pa - level)
        let rm = ra + Self.wrapped(pm - pa)
        let rb = rm + Self.wrapped(pb - pm)
        if let k = Self.turn(between: ra, rm) { return (a, m, ra - k, rm - k) }
        if let k = Self.turn(between: rm, rb) { return (m, b, rm - k, rb - k) }
        return nil
    }

    /// The multiple of 2π between two residuals, if there is one.
    static func turn(between u: Double, _ v: Double) -> Double? {
        let lo = min(u, v), hi = max(u, v)
        let k = (lo / (2 * .pi)).rounded(.up) * 2 * .pi
        return k <= hi ? k : nil
    }

    /// The exact crossing on the grid edge this vertex sits on; nil for a
    /// vertex on a phase wrap, which is not a crossing of anything.
    func snap(_ p: P2<DomainSpace>, level: Double,
              g: @Sendable (Complex) -> Double, cut: Bool) -> P2<DomainSpace>? {
        let fx = (p.x - grid.real.lo) / dx
        let fy = (p.y - grid.imag.lo) / dy
        let onColumn = abs(fx - fx.rounded()) < 1e-6
        let onRow = abs(fy - fy.rounded()) < 1e-6
        let a: P2<DomainSpace>, b: P2<DomainSpace>
        if onRow && !onColumn {
            // A horizontal edge: between the two columns either side.
            let i = Int(fx.rounded(.down))
            guard i >= 0, i + 1 < width else { return p }
            a = P2(grid.real.lo + Double(i) * dx, p.y)
            b = P2(grid.real.lo + Double(i + 1) * dx, p.y)
        } else if onColumn && !onRow {
            let j = Int(fy.rounded(.down))
            guard j >= 0, j + 1 < height else { return p }
            a = P2(p.x, grid.imag.lo + Double(j) * dy)
            b = P2(p.x, grid.imag.lo + Double(j + 1) * dy)
        } else if onColumn && onRow {
            // On a node. Marching squares puts a vertex there when the sample
            // equals the level, which is right, and also when the far sample
            // is the cap: exp(1/z) beside its essential singularity has a
            // node at 1e-109 next to one at 1e12, and the interpolated
            // crossing is 1e-12 of the way along the edge, which rounds onto
            // the node where f is flat and nowhere near the level. The edge is
            // not recorded, but if exactly one of the node's four edges
            // brackets the level, that is the one.
            let i = Int(fx.rounded()), j = Int(fy.rounded())
            let gp = residual(g(Complex(p.x, p.y)), level: level, cut: cut)
            guard gp.isFinite, !(cut && abs(gp) > .pi / 2) else { return p }
            let (gx, gy) = gradient(g, at: p)
            let slope = (gx * gx + gy * gy).squareRoot()
            guard slope > 0, slope.isFinite, abs(gp) / slope > 1e-2 * tolerance * cell else { return p }
            var bracketing: [P2<DomainSpace>] = []
            for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let ni = i + di, nj = j + dj
                guard ni >= 0, ni < width, nj >= 0, nj < height else { continue }
                let n = P2<DomainSpace>(grid.real.lo + Double(ni) * dx, grid.imag.lo + Double(nj) * dy)
                if bracket(p, n, level: level, g: g, cut: cut) != nil { bracketing.append(n) }
            }
            guard bracketing.count == 1 else { return p }
            a = p; b = bracketing[0]
        } else {
            return p       // nowhere a marching-squares vertex can be
        }
        guard let edge = bracket(a, b, level: level, g: g, cut: cut) else {
            let ra = residual(g(Complex(a.x, a.y)), level: level, cut: cut)
            let rb = residual(g(Complex(b.x, b.y)), level: level, cut: cut)
            guard ra.isFinite, rb.isFinite else { return p }
            // The float32 grid saw a crossing that f, in double, does not:
            // the level set passes within rounding of a node. The nearer
            // endpoint is the honest position. For the phase, anything else
            // is a wrap, which is not a crossing of anything.
            if cut && min(abs(ra), abs(rb)) > 1e-9 { return nil }
            return abs(ra) <= abs(rb) ? a : b
        }
        let (a1, b1) = (edge.a, edge.b)
        var ga = edge.ra, gb = edge.rb
        if ga == 0 { return a1 }
        if gb == 0 { return b1 }
        // Illinois regula falsi on t in [0, 1] along the edge, which cannot
        // fail to converge on a bracketed root; sixty iterations is far more
        // than it ever takes -- except against a bracket like exp(1/z)'s
        // beside its essential singularity, 1e-109 at one end and the 1e12
        // cap at the other, where every secant step lands beside the small
        // end and sixty halvings of 1e12 still do not reach the crossing.
        // Two steps in a row on the same side and the next is a bisection,
        // which halves the bracket whatever the values are.
        var ta = 0.0, tb = 1.0
        var side = 0, stuck = 0
        var t = (p.x - a1.x) * (b1.x - a1.x) + (p.y - a1.y) * (b1.y - a1.y)
        t /= max((b1.x - a1.x) * (b1.x - a1.x) + (b1.y - a1.y) * (b1.y - a1.y), .leastNormalMagnitude)
        t = min(max(t, 0), 1)
        var q = p
        for _ in 0..<60 {
            q = P2<DomainSpace>(a1.x + t * (b1.x - a1.x), a1.y + t * (b1.y - a1.y))
            let gq = residual(g(Complex(q.x, q.y)), level: level, cut: cut)
            if !gq.isFinite { return p }
            if abs(gq) <= 1e-13 * max(abs(level), 1) { return q }
            if gq.sign == ga.sign {
                ta = t; ga = gq
                if side == -1 { gb /= 2; stuck += 1 } else { stuck = 0 }
                side = -1
            } else {
                tb = t; gb = gq
                if side == 1 { ga /= 2; stuck += 1 } else { stuck = 0 }
                side = 1
            }
            if tb - ta < 1e-12 { return q }
            if stuck >= 2 {
                t = (ta + tb) / 2; stuck = 0
            } else {
                t = tb - gb * (tb - ta) / (gb - ga)
                if !(t > ta && t < tb) { t = (ta + tb) / 2 }
            }
        }
        return q
    }

    /// The estimated distance from the chord `a-b` to the level set, and the
    /// point on the level set nearest the chord's midpoint, when it is worth
    /// inserting.
    func subdivide(_ a: P2<DomainSpace>, _ b: P2<DomainSpace>, level: Double,
                   g: @Sendable (Complex) -> Double, cut: Bool, depth: Int,
                   into out: inout [P2<DomainSpace>]) {
        guard depth < maxDepth else { return }
        let mid = P2<DomainSpace>((a.x + b.x) / 2, (a.y + b.y) / 2)
        let r = residual(g(Complex(mid.x, mid.y)), level: level, cut: cut)
        guard r.isFinite else { return }
        if cut && abs(r) > .pi / 2 { return }
        let chord = ((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y)).squareRoot()
        guard chord > 1e-9 * cell else { return }
        let (gx, gy) = gradient(g, at: mid)
        let slope = (gx * gx + gy * gy).squareRoot()
        guard slope > 0, slope.isFinite else { return }
        let error = abs(r) / slope
        guard error > tolerance * cell else { return }
        // Solve along the unit normal to the chord, starting from the Newton
        // step, secant thereafter; give up rather than wander further than a
        // chord's length from the midpoint.
        let nx = -(b.y - a.y) / chord, ny = (b.x - a.x) / chord
        let gn = gx * nx + gy * ny
        guard abs(gn) > 1e-3 * slope else { return }
        var t0 = 0.0, r0 = r
        var t1 = -r / gn
        var found: P2<DomainSpace>?
        var best: (q: P2<DomainSpace>, r: Double)?
        // Accept only a converged point: an unconverged one would be a vertex
        // off the level set, which is what this is here to remove. Converged
        // means within a hundredth of the tolerance by the estimate the
        // measurement uses, residual over the gradient at the point itself.
        // A residual at rounding is not enough on its own: where the level
        // set crosses itself at a critical point (|cn| = 1 at 0, |sin| = 1 at
        // π/2, 1/(z³ - 1) = 1 at 0 with f'' = 0 too) the residual along the
        // normal has a multiple root, so it is tiny well before the point is
        // close, and the secant converges only linearly, so it may never
        // reach rounding at all. The gradient is the part that shrinks
        // toward a saddle, and the estimate knows it.
        func converged(_ q: P2<DomainSpace>, _ r: Double) -> Bool {
            let (gx, gy) = gradient(g, at: q)
            let s = (gx * gx + gy * gy).squareRoot()
            return s > 0 && s.isFinite && abs(r) / s <= 1e-2 * tolerance * cell
        }
        for _ in 0..<24 {
            if abs(t1) > chord { break }
            let q = P2<DomainSpace>(mid.x + t1 * nx, mid.y + t1 * ny)
            let r1 = residual(g(Complex(q.x, q.y)), level: level, cut: cut)
            guard r1.isFinite, !(cut && abs(r1) > .pi / 2) else { break }
            if best == nil || abs(r1) < abs(best!.r) { best = (q, r1) }
            let stalled = abs(t1 - t0) < 1e-10 * cell
            if abs(r1) <= 1e-13 * max(abs(level), 1) || stalled {
                if converged(q, r1) { found = q }
                if found != nil || stalled { break }
            }
            let denominator = r1 - r0
            guard denominator != 0 else { break }
            let t2 = t1 - r1 * (t1 - t0) / denominator
            t0 = t1; r0 = r1; t1 = t2
        }
        if found == nil, let best, converged(best.q, best.r) { found = best.q }
        guard let point = found else { return }
        subdivide(a, point, level: level, g: g, cut: cut, depth: depth + 1, into: &out)
        out.append(point)
        subdivide(point, b, level: level, g: g, cut: cut, depth: depth + 1, into: &out)
    }

    /// The gradient of g at p, by central differences at a step that is small
    /// against the cell and large against rounding.
    func gradient(_ g: @Sendable (Complex) -> Double, at p: P2<DomainSpace>) -> (Double, Double) {
        let step = 1e-4 * cell
        return ((g(Complex(p.x + step, p.y)) - g(Complex(p.x - step, p.y))) / (2 * step),
                (g(Complex(p.x, p.y + step)) - g(Complex(p.x, p.y - step))) / (2 * step))
    }

    // MARK: measuring

    /// The distance of each vertex from the level set, in cells: the residual
    /// over the gradient, first order. What the refinement is trying to make
    /// small, measured against f rather than against a finer grid. Capped at
    /// a cell: the estimate is first order, and where f is flat (a grid
    /// vertex left on a node beside exp(1/z)'s essential singularity, where
    /// |f| is 1e-109) it runs to 1e106 and says only "far".
    public func positionErrors(_ path: [P2<DomainSpace>], field: ContourField,
                               level: Double) -> [Double] {
        let (g, level) = self.field(field, level: level)
        return path.map { p in
            let r = residual(g(Complex(p.x, p.y)), level: level, cut: field == .phase)
            let (gx, gy) = gradient(g, at: p)
            let slope = (gx * gx + gy * gy).squareRoot()
            guard r.isFinite, slope > 0, slope.isFinite else { return .nan }
            if field == .phase && abs(r) > .pi / 2 { return .nan }
            return min(abs(r) / slope / cell, 1)
        }
    }

    /// Where a chord departs from the level set between two vertices: the
    /// midpoint error, in cells, for each segment.
    public func chordErrors(_ path: [P2<DomainSpace>], field: ContourField,
                            level: Double) -> [Double] {
        guard path.count >= 2 else { return [] }
        let mids = zip(path, path.dropFirst()).map {
            P2<DomainSpace>(($0.x + $1.x) / 2, ($0.y + $1.y) / 2)
        }
        return positionErrors(mids, field: field, level: level)
    }
}

public extension NativeLandscape {
    /// A refiner for the function behind a landscape, at its grid.
    static func refiner(for expression: String, domain: Domain,
                        shape: (nReal: Int, nImag: Int)) throws -> ContourRefiner {
        ContourRefiner(try KurvenMath.Expression.compile(expression), domain: domain,
                       width: shape.nReal, height: shape.nImag)
    }

    /// A refiner for a bundle that records its expression -- one written by
    /// `build`, or by the Python service for a landscape.
    static func refiner(for bundle: KurvenBundle) -> ContourRefiner? {
        guard bundle.manifest.provenance.example == "function" else { return nil }
        let expression: String
        if case .some(.string(let s)) = bundle.manifest.provenance.params["expression"] {
            expression = s
        } else {
            expression = bundle.manifest.provenance.function
        }
        let h = bundle.surface.height
        return try? refiner(for: expression, domain: bundle.manifest.domain,
                            shape: (h.width, h.height))
    }
}

// MARK: - the benchmark

public extension NativeLandscape {
    /// One contour layer, derived with and without refinement, timed and
    /// measured.
    struct RefinementReport: Sendable {
        public var layer: String
        public var field: ContourField
        public var levels: Int
        public var gridVertices: Int
        public var refinedVertices: Int
        /// Grid vertices that sat on a phase wrap and were dropped.
        public var wrapVertices: Int
        public var gridSeconds: Double
        public var refineSeconds: Double
        /// Vertex error in cells: mean, 95th percentile, max -- before and after.
        public var gridVertex: (mean: Double, p95: Double, max: Double)
        public var refinedVertex: (mean: Double, p95: Double, max: Double)
        /// Midpoint-of-chord error in cells, before and after.
        public var gridChord: (mean: Double, p95: Double, max: Double)
        public var refinedChord: (mean: Double, p95: Double, max: Double)
    }

    static func summarize(_ values: [Double]) -> (mean: Double, p95: Double, max: Double) {
        let finite = values.filter(\.isFinite).sorted()
        guard !finite.isEmpty else { return (0, 0, 0) }
        let mean = finite.reduce(0, +) / Double(finite.count)
        let p95 = finite[min(Int(Double(finite.count - 1) * 0.95), finite.count - 1)]
        return (mean, p95, finite[finite.count - 1])
    }

    /// Sample the request, then for every described contour layer: contour the
    /// grid, refine, and measure both against f.
    static func benchmarkRefinement(_ request: LandscapeRequest,
                                    tolerance: Double = 0.02, maxDepth: Int = 5) throws
        -> (bundle: KurvenBundle, reports: [RefinementReport]) {
        let bundle = try build(request, refine: false)
        let h = bundle.surface.height
        var refiner = try refiner(for: request.expression, domain: bundle.manifest.domain,
                                  shape: (h.width, h.height))
        refiner.tolerance = tolerance
        refiner.maxDepth = maxDepth
        let clock = ContinuousClock()
        var reports: [RefinementReport] = []
        for spec in bundle.manifest.layers {
            guard case .contour(let field, let levels, _, _) = spec.source else { continue }
            let grid: Grid2D<Float>
            switch field {
            case .magnitude: grid = bundle.surface.height
            case .phase: guard let p = bundle.surface.phase else { continue }; grid = p
            }
            var contoured: [(level: Double, paths: [[P2<DomainSpace>]])] = []
            let gridTime = clock.measure {
                // As `Surface.derive` contours them under a refiner: the cut
                // once, as the zero level of the turned phase.
                let plain = field == .phase ? levels.filter { !Contour.isCut($0) } : levels
                contoured = Contour.levels(of: grid, plain)
                if plain.count < levels.count { contoured.append((.pi, Contour.cut(of: grid))) }
            }
            var refined: [(level: Double, paths: [[P2<DomainSpace>]])] = []
            let refineTime = clock.measure {
                refined = contoured.map { ($0.level, refiner.refine(field: field, level: $0.level, $0.paths)) }
            }
            func errors(_ set: [(level: Double, paths: [[P2<DomainSpace>]])])
                -> (vertex: [Double], chord: [Double], count: Int) {
                var v: [Double] = [], c: [Double] = [], n = 0
                for (level, paths) in set {
                    for path in paths {
                        v += refiner.positionErrors(path, field: field, level: level)
                        c += refiner.chordErrors(path, field: field, level: level)
                        n += path.count
                    }
                }
                return (v, c, n)
            }
            let before = errors(contoured), after = errors(refined)
            var wraps = 0
            if field == .phase {
                for (level, paths) in contoured {
                    let (g, level) = refiner.field(field, level: level)
                    for path in paths where path.count >= 2 {
                        for p in path where refiner.snap(p, level: level, g: g, cut: true) == nil {
                            wraps += 1
                        }
                    }
                }
            }
            reports.append(RefinementReport(
                layer: spec.name, field: field, levels: levels.count,
                gridVertices: before.count, refinedVertices: after.count, wrapVertices: wraps,
                gridSeconds: Double(gridTime.components.seconds)
                    + Double(gridTime.components.attoseconds) / 1e18,
                refineSeconds: Double(refineTime.components.seconds)
                    + Double(refineTime.components.attoseconds) / 1e18,
                gridVertex: summarize(before.vertex), refinedVertex: summarize(after.vertex),
                gridChord: summarize(before.chord), refinedChord: summarize(after.chord)))
        }
        return (bundle, reports)
    }
}
