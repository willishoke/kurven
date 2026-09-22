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
/// phase vertices, measured). A vertex whose two edge samples differ by more
/// than π is on a wrap, not on the level set; the refiner drops it and splits
/// the path there. A chord midpoint whose residual is a jump is likewise left
/// alone rather than solved toward the wrong branch.
public struct ContourRefiner: Sendable {
    public let magnitude: @Sendable (Complex) -> Double
    public let phase: @Sendable (Complex) -> Double
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
        self.magnitude = { z in
            let v = compiled(z)
            return v.isFinite ? min(v.magnitude, NativeLandscape.huge) : NativeLandscape.huge
        }
        self.phase = { z in
            let v = compiled(z)
            return v.isFinite ? v.argument : 0
        }
        self.grid = domain; self.width = width; self.height = height
        self.tolerance = tolerance; self.maxDepth = maxDepth
    }

    var dx: Double { grid.real.length / Double(max(width - 1, 1)) }
    var dy: Double { grid.imag.length / Double(max(height - 1, 1)) }
    /// The cell size the tolerances are measured against.
    public var cell: Double { max(abs(dx), abs(dy)) }

    func field(_ f: ContourField) -> @Sendable (Complex) -> Double {
        f == .magnitude ? magnitude : phase
    }

    /// `Surface.derive`'s hook.
    public var refine: ContourRefine {
        { field, level, paths in self.refine(field: field, level: level, paths) }
    }

    public func refine(field: ContourField, level: Double,
                       _ paths: [[P2<DomainSpace>]]) -> [[P2<DomainSpace>]] {
        let g = self.field(field)
        let cut = field == .phase
        var out = [[[P2<DomainSpace>]]](repeating: [], count: paths.count)
        out.withUnsafeMutableBufferPointer { buffer in
            let base = buffer.baseAddress!
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
        return runs
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
        } else {
            return p       // on a node, or nowhere a marching-squares vertex can be
        }
        var ga = g(Complex(a.x, a.y)) - level
        var gb = g(Complex(b.x, b.y)) - level
        if cut && abs(ga - gb) > .pi { return nil }
        guard ga.isFinite, gb.isFinite else { return p }
        if ga == 0 { return a }
        if gb == 0 { return b }
        // The float32 grid saw a crossing that f, in double, does not: the
        // level set passes within rounding of a node. The nearer endpoint is
        // the honest position.
        guard ga.sign != gb.sign else { return abs(ga) <= abs(gb) ? a : b }
        // Illinois regula falsi on t in [0, 1] along the edge, which cannot
        // fail to converge on a bracketed root; sixty iterations is far more
        // than it ever takes.
        var ta = 0.0, tb = 1.0
        var side = 0
        var t = (p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y)
        t /= max((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y), .leastNormalMagnitude)
        t = min(max(t, 0), 1)
        var q = p
        for _ in 0..<60 {
            q = P2<DomainSpace>(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y))
            let gq = g(Complex(q.x, q.y)) - level
            if !gq.isFinite { return p }
            if abs(gq) <= 1e-13 * max(abs(level), 1) { return q }
            if gq.sign == ga.sign {
                ta = t; ga = gq
                if side == -1 { gb /= 2 }
                side = -1
            } else {
                tb = t; gb = gq
                if side == 1 { ga /= 2 }
                side = 1
            }
            if tb - ta < 1e-12 { return q }
            t = tb - gb * (tb - ta) / (gb - ga)
            if !(t > ta && t < tb) { t = (ta + tb) / 2 }
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
        let r = g(Complex(mid.x, mid.y)) - level
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
        for _ in 0..<24 {
            if abs(t1) > chord { break }
            let q = P2<DomainSpace>(mid.x + t1 * nx, mid.y + t1 * ny)
            let r1 = g(Complex(q.x, q.y)) - level
            guard r1.isFinite, !(cut && abs(r1) > .pi / 2) else { break }
            if best == nil || abs(r1) < abs(best!.r) { best = (q, r1) }
            // Accept only a converged point: an unconverged one would be a
            // vertex off the level set, which is what this is here to remove.
            if abs(r1) <= 1e-13 * max(abs(level), 1)
                || (abs(t1 - t0) < 1e-10 * cell && abs(r1) < abs(r)) {
                found = q; break
            }
            let denominator = r1 - r0
            guard denominator != 0 else { break }
            let t2 = t1 - r1 * (t1 - t0) / denominator
            t0 = t1; r0 = r1; t1 = t2
        }
        // Where the level set crosses itself at a critical point (|cn| = 1 at
        // z = 0, |sin| = 1 at π/2) the residual along the normal has a double
        // root: the secant converges only linearly and never reaches rounding.
        // The chord across the saddle would be left a quarter cell off, so the
        // nearest iterate is accepted if it is within a hundredth of the
        // tolerance by the estimate the measurement uses -- the gradient at
        // the point itself, which is what shrinks toward a saddle.
        if found == nil, let best {
            let (bx, by) = gradient(g, at: best.q)
            let s = (bx * bx + by * by).squareRoot()
            if s > 0, s.isFinite, abs(best.r) / s <= 1e-2 * tolerance * cell { found = best.q }
        }
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
    /// small, measured against f rather than against a finer grid.
    public func positionErrors(_ path: [P2<DomainSpace>], field: ContourField,
                               level: Double) -> [Double] {
        let g = self.field(field)
        return path.map { p in
            let r = g(Complex(p.x, p.y)) - level
            let (gx, gy) = gradient(g, at: p)
            let slope = (gx * gx + gy * gy).squareRoot()
            guard r.isFinite, slope > 0, slope.isFinite else { return .nan }
            if field == .phase && abs(r) > .pi / 2 { return .nan }
            return abs(r) / slope / cell
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
            let gridTime = clock.measure { contoured = Contour.levels(of: grid, levels) }
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
                let g = refiner.field(field)
                for (level, paths) in contoured {
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
