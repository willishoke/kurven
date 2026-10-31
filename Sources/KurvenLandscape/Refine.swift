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
///     moves the vertex to the exact crossing, by Brent's method (`Root.brent`).
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
/// Every phase meets at a zero of f and at a pole, and a phase contour ends
/// at one or the other. The grid stops it at the last cell edge before; the
/// refiner, which has f, carries each run whose end lies within a cell or so
/// of a zero or a pole -- found by Newton from the end, on f or on 1/f -- to
/// the point itself (`ended(atSingularPoints:)`, `trace`).
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
    /// Whether the ink is drawn at height |f|, as a landscape's is, so that a
    /// phase chord is also split where its height sags from the chord's
    /// (`subdivide`). Ink on a parametric surface has no height to sag in.
    public var lifted: Bool
    /// The height above which the ink is cut away, past which no line is
    /// followed or seeded: set for the duration of a
    /// `refine(field:level:_:points:ceiling:)` call from its ceiling.
    var ceiling: Double = .infinity

    /// The expression f was compiled from, for a fork to compile again.
    let node: KurvenMath.Expression.Node

    public init(_ compiled: KurvenMath.Expression.Compiled, domain: Domain,
                width: Int, height: Int, tolerance: Double = 0.02, maxDepth: Int = 5,
                lifted: Bool = true) {
        self.lifted = lifted
        self.node = compiled.node
        self.value = { z in compiled(z) }
        self.magnitude = { z in
            let v = compiled(z)
            return v.isFinite ? min(v.magnitude, NativeLandscape.huge) : NativeLandscape.huge
        }
        // The phase is undefined where f is not finite and where it has
        // underflowed to zero: the argument of 0 + 0i is a number, but not
        // one of f's. NaN there, which no bracket, snap or chord accepts,
        // so beside a singularity the lines end where f's phase does.
        self.phase = { z in
            let v = compiled(z)
            return v.isFinite && v.magnitude > 0 ? v.argument : .nan
        }
        self.turnedPhase = { z in
            let v = compiled(z)
            return v.isFinite && v.magnitude > 0 ? (-v).argument : .nan
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

    /// `Surface.derive`'s hook, and `Surface.topEdge`'s and
    /// `Heightfield.foldCurves`'s: the contours placed, |f| for the top edge's
    /// minima and the zeros' heights, and the zeros the fold curves are
    /// carried to. The magnitude is the one the grid was sampled with,
    /// clamped at `NativeLandscape.huge` beside a pole.
    ///
    /// Remembering (`ContourRefine.remembering`): a bundle carries this hook
    /// through every restyle of its grids, and a restyle asks again for
    /// curves it has already placed.
    public var refine: ContourRefine {
        hook(points: singularPoints()).remembering()
    }

    /// The hook over these singular points, forking into a refiner of its
    /// own: f compiled again from its tree, so the fork's closures share
    /// nothing with these (`ContourRefine.fork`).
    func hook(points: [SingularPoint]) -> ContourRefine {
        let node = node, grid = grid, width = width, height = height
        let tolerance = tolerance, maxDepth = maxDepth, lifted = lifted
        return ContourRefine(contours: { field, level, paths, ceiling in
                                 self.refine(field: field, level: level, paths, points: points,
                                             ceiling: ceiling)
                             },
                             magnitude: { p in self.magnitude(Complex(p.x, p.y)) },
                             zero: { p in self.zero(near: p) },
                             tolerance: tolerance,
                             fork: {
                                 ContourRefiner(KurvenMath.Expression.Compiled(node: node), domain: grid,
                                                width: width, height: height, tolerance: tolerance,
                                                maxDepth: maxDepth, lifted: lifted).hook(points: points)
                             })
    }

    /// The grid's paths at one level of one field, placed on f; and, for the
    /// magnitude with the zeros at hand, the rings around each zero that the
    /// grid could not see (`seededRings`).
    ///
    /// A phase path is placed only where it can be drawn: along the
    /// stretches with a vertex at or under `ceiling`, the height above which
    /// its ink is cut away (`ContourRefine.Contours`). A magnitude line is
    /// level, and wholly below the truncation or wholly over it.
    public func refine(field: ContourField, level theta: Double,
                       _ paths: [[P2<DomainSpace>]], points: [SingularPoint] = [],
                       ceiling: Double = .infinity) -> [[P2<DomainSpace>]] {
        let (g, level) = self.field(field, level: theta)
        let cut = field == .phase
        let ceiling = cut ? ceiling : .infinity
        // Nothing is followed or seeded above this call's ceiling: a copy of
        // the refiner carries it.
        guard self.ceiling == ceiling else {
            var under = self
            under.ceiling = ceiling
            return under.refine(field: field, level: theta, paths, points: points, ceiling: ceiling)
        }
        // Every path snapped and subdivided into its pieces.
        var pieced = [[Piece]](repeating: [], count: paths.count)
        pieced.withUnsafeMutableBufferPointer { buffer in
            // Each iteration writes only its own slot, so sharing is safe.
            nonisolated(unsafe) let base = buffer.baseAddress!
            DispatchQueue.concurrentPerform(iterations: paths.count) { i in
                base[i] = pieces(of: paths[i], level: level, g: g, cut: cut, ceiling: ceiling)
            }
        }
        let all = pieced.flatMap { $0 }
        guard cut else {
            var placed = all.map(\.run)
            let zeros = points.filter { $0.order > 0 }.map(\.point)
            if field == .magnitude, !zeros.isEmpty {
                placed += seededRings(level: level, zeros: zeros, existing: placed)
            }
            return placed
        }
        // The fan at each zero and pole, from the point's normal form: its
        // rays at this level, each followed from the point out to the
        // handoff radius. The grid's pieces are cut back from those discs
        // and the rays spliced on, so the ink nearest a singular point is
        // f's, with as many rays as the point's order says, whatever the
        // grid resolved there.
        let rays = points.indices.flatMap { seededRays(from: $0, of: points, theta: theta,
                                                       level: level, g: g, ceiling: ceiling) }
        let seeded = Set(rays.filter(\.handoff).map(\.point))
        let discs = seeded.map { (index: $0, center: points[$0].point) }
        let radius = Self.handoff * cell
        let stopping: @Sendable (P2<DomainSpace>) -> Int? = { p in
            discs.first { Self.apart(p, $0.center) < radius }?.index
        }
        // A phase piece is followed on f beyond its ends, to the zero or
        // pole it runs into, to the window's edge, or to a disc. First once
        // over: a piece lying along another is a copy of a stretch of it,
        // and following it would copy the rest.
        let kept = drawnOnce(all.flatMap { trimmed($0, awayFrom: discs) }, run: \.run)
        var followed = [Piece](repeating: Piece(run: [], ends: (false, false)), count: kept.count)
        followed.withUnsafeMutableBufferPointer { buffer in
            nonisolated(unsafe) let base = buffer.baseAddress!
            DispatchQueue.concurrentPerform(iterations: kept.count) { i in
                let piece = kept[i]
                base[i] = traced(Piece(run: ended(atSingularPoints: piece.run, level: level, g: g,
                                                  ends: piece.ends),
                                       ends: piece.ends, met: piece.met),
                                 level: level, g: g, stopping: stopping)
            }
        }
        let joined = spliced(rays, onto: followed, points: points, level: level, g: g,
                             stopping: stopping)
        // And once more: two pieces of one curve, split where a chord could
        // not be placed, have each been followed over the other's ground.
        return drawnOnce(joined, run: { $0 })
    }

    /// A run of placed vertices; whether each end is the curve's own, to be
    /// followed on f and carried to a zero or pole, or a cut at the ceiling;
    /// and, for an end cut at the handoff disc of a singular point, which
    /// point, for its ray to be spliced on.
    struct Piece: Sendable {
        var run: [P2<DomainSpace>]
        var ends: (start: Bool, end: Bool)
        var met: (start: Int?, end: Int?) = (nil, nil)
    }

    /// The runs drawn once: a run every vertex of which lies within a tenth
    /// of a cell of another run -- a longer one, or one as long and earlier
    /// -- is a copy of a stretch of it, and is dropped.
    ///
    /// A stretch comes to be placed twice in two ways. Through the cell of a
    /// pole, marching squares links the ray at a level into the bundle of
    /// spurious crossings along the cut, and beside the pole those crossings
    /// lie on edges the ray crosses too, so the snap puts them on the ray.
    /// And a chord that could not be placed splits a run in two, after which
    /// each half is followed on f over the other's ground. Distinct curves at
    /// one level meet only at a zero, a pole or a saddle, so a run lying
    /// wholly along another is never a curve of its own.
    func drawnOnce<T>(_ items: [T], run: (T) -> [P2<DomainSpace>]) -> [T] {
        guard items.count > 1 else { return items }
        let runs = items.map(run)
        let near = 0.1 * cell
        func key(_ p: P2<DomainSpace>) -> SIMD2<Int> {
            SIMD2(Int((p.x / cell).rounded(.down)), Int((p.y / cell).rounded(.down)))
        }
        // Every segment, binned by the cells its ends fall in. A segment is
        // at most about a cell long, so the three-by-three block of cells
        // round a vertex holds every segment within a tenth of a cell of it.
        var segments: [SIMD2<Int>: [(run: Int, at: Int)]] = [:]
        for (i, r) in runs.enumerated() {
            for k in 0..<max(r.count - 1, 0) {
                let a = key(r[k]), b = key(r[k + 1])
                segments[a, default: []].append((i, k))
                if b != a { segments[b, default: []].append((i, k)) }
            }
        }
        func distance(_ p: P2<DomainSpace>, _ a: P2<DomainSpace>, _ b: P2<DomainSpace>) -> Double {
            let abx = b.x - a.x, aby = b.y - a.y
            let l2 = abx * abx + aby * aby
            let t = l2 > 0 ? min(max(((p.x - a.x) * abx + (p.y - a.y) * aby) / l2, 0), 1) : 0
            let qx = a.x + t * abx - p.x, qy = a.y + t * aby - p.y
            return (qx * qx + qy * qy).squareRoot()
        }
        func covered(_ i: Int) -> Bool {
            let r = runs[i]
            return r.allSatisfy { p in
                let c = key(p)
                for dx in -1...1 {
                    for dy in -1...1 {
                        for s in segments[SIMD2(c.x + dx, c.y + dy)] ?? [] where s.run != i {
                            let other = runs[s.run]
                            guard other.count > r.count || (other.count == r.count && s.run < i)
                            else { continue }
                            if distance(p, other[s.at], other[s.at + 1]) <= near { return true }
                        }
                    }
                }
                return false
            }
        }
        return items.indices.filter { !covered($0) }.map { items[$0] }
    }

    // MARK: rings the grid cannot see

    /// The zeros of f in the window. Every node that is a local minimum of
    /// |f| among its neighbours is taken to the zero beside it by Newton:
    /// |f| has no local minimum but a zero (the maximum principle), so the
    /// interior hides none, and a node on the window's edge that is a
    /// minimum without a zero fails Newton's test. Found once per refiner.
    func zeros() -> [P2<DomainSpace>] {
        rooted(extremes().minima, by: zero(near:))
    }

    static func apart(_ a: P2<DomainSpace>, _ b: P2<DomainSpace>) -> Double {
        ((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)).squareRoot()
    }

    // MARK: the normal form at a zero or pole

    /// A zero or pole of f with its normal form there: f(z) ≈ coefficient ·
    /// (z − point)^order, the order negative at a pole.
    ///
    /// This is the chart in which the ink round the point is trivial. In
    /// w = f(z) a line of constant phase θ is the ray arg w = θ and a line
    /// of constant magnitude is a circle, so near the point the drawing is
    /// read off the normal form and carried back by Newton on f, and the
    /// grid, which cannot see inside the cell the point lies in, is not
    /// consulted there. For order k there are k rays at every level,
    /// leaving at the angles (θ − arg coefficient + 2πj) / k, and the same
    /// formula with k negative gives a pole's.
    public struct SingularPoint: Sendable {
        public let point: P2<DomainSpace>
        public let order: Int
        public let coefficient: Complex
    }

    /// The zeros and poles of f in the window, each with its order and
    /// leading coefficient. The order is the linear curve of arg f round a
    /// small circle about the point, small against the distance to any
    /// other; the coefficient is the mean of f(z) / (z − point)^order round
    /// a smaller one. A point whose linear curve does not agree with the way it
    /// was found, or round which f is not finite and nonzero, is left out,
    /// and its lines are then carried onto it as before (`ended`).
    public func singularPoints() -> [SingularPoint] {
        let found = zeros().map { ($0, 1) } + poles().map { ($0, -1) }
        var out: [SingularPoint] = []
        for (i, (p, sign)) in found.enumerated() {
            var nearest = Double.infinity
            for (j, (q, _)) in found.enumerated() where j != i { nearest = min(nearest, Self.apart(p, q)) }
            let r = min(0.5 * cell, 0.25 * nearest)
            guard r > 0 else { continue }
            let c = Complex(p.x, p.y)
            let n = 64
            var total = 0.0, previous: Double?, sound = true
            for k in 0...n {
                let t = 2 * Double.pi * Double(k) / Double(n)
                let v = value(c + Complex(r * cos(t), r * sin(t)))
                guard v.isFinite, v.magnitude > 0 else { sound = false; break }
                if let previous { total += Self.wrapped(v.argument - previous) }
                previous = v.argument
            }
            guard sound else { continue }
            let order = Int((total / (2 * .pi)).rounded())
            guard order != 0, (order > 0) == (sign > 0) else { continue }
            let rc = min(1e-2 * cell, 0.5 * r)
            var sum = Complex(0), count = 0
            for k in 0..<8 {
                let t = 2 * Double.pi * (Double(k) + 0.5) / 8
                let w = Complex(rc * cos(t), rc * sin(t))
                let v = value(c + w)
                guard v.isFinite, v.magnitude > 0 else { continue }
                var power = Complex(1)
                for _ in 0..<abs(order) { power = power * w }
                let a = order > 0 ? v / power : v * power
                guard a.isFinite else { continue }
                sum = sum + a; count += 1
            }
            guard count > 0 else { continue }
            out.append(SingularPoint(point: p, order: order,
                                     coefficient: Complex(sum.re / Double(count), sum.im / Double(count))))
        }
        return out
    }

    /// The poles of f in the window: every node that is the greatest of |f|
    /// among its neighbours, taken to the pole beside it by Newton on 1/f.
    /// |f| has no local maximum (the maximum principle), so a node that is
    /// one is beside a pole or on the window's edge, where Newton's test
    /// sorts them.
    func poles() -> [P2<DomainSpace>] {
        rooted(extremes().maxima, by: pole(near:))
    }

    func rooted(_ nodes: [P2<DomainSpace>],
                by find: (P2<DomainSpace>) -> P2<DomainSpace>?) -> [P2<DomainSpace>] {
        var found: [P2<DomainSpace>] = []
        for node in nodes {
            guard let z = find(node),
                  !found.contains(where: { Self.apart($0, z) <= 1e-6 * cell }) else { continue }
            found.append(z)
        }
        return found
    }

    /// The grid's nodes that are the least, and the greatest, of |f| among
    /// their neighbours. |f| is clamped at `huge` beside a pole, so a pole's
    /// node is a greatest one.
    func extremes() -> (minima: [P2<DomainSpace>], maxima: [P2<DomainSpace>]) {
        guard width >= 2, height >= 2 else { return ([], []) }
        var values = [Double](repeating: .infinity, count: width * height)
        values.withUnsafeMutableBufferPointer { buffer in
            nonisolated(unsafe) let base = buffer.baseAddress!
            DispatchQueue.concurrentPerform(iterations: height) { j in
                for i in 0..<width {
                    base[j * width + i] = magnitude(Complex(grid.real.lo + Double(i) * dx,
                                                            grid.imag.lo + Double(j) * dy))
                }
            }
        }
        var minima: [P2<DomainSpace>] = [], maxima: [P2<DomainSpace>] = []
        for j in 0..<height {
            for i in 0..<width {
                let v = values[j * width + i]
                guard v.isFinite else { continue }
                var least = true, greatest = true
                for dj in -1...1 {
                    for di in -1...1 where di != 0 || dj != 0 {
                        let ni = i + di, nj = j + dj
                        guard ni >= 0, ni < width, nj >= 0, nj < height else { continue }
                        let u = values[nj * width + ni]
                        if u < v { least = false }
                        if u > v { greatest = false }
                    }
                }
                let node = P2<DomainSpace>(grid.real.lo + Double(i) * dx, grid.imag.lo + Double(j) * dy)
                if least { minima.append(node) }
                if greatest, v > 0 { maxima.append(node) }
            }
        }
        return (minima, maxima)
    }

    /// How far from a singular point its rays are followed before the
    /// grid's ink takes over, in cells: the handoff radius. The same
    /// distance a run's end is looked for a zero or pole from, so a piece
    /// cut back from the disc is never also carried onto its point.
    static let handoff = reach

    /// One ray of a singular point's fan at a level, from the point out to
    /// the handoff radius; `handoff` is false when it ended before that, on
    /// the window's edge, at the ceiling or on another singular point, and
    /// is then complete as it stands.
    struct Ray: Sendable {
        var run: [P2<DomainSpace>]
        let point: Int
        var handoff: Bool
    }

    /// The rays at level `theta` of the singular point `index` of `points`,
    /// read off its normal form: each started half a cell out along its
    /// direction, moved onto the level set, joined to the point by a chord
    /// placed like any other, and followed on f to the handoff radius.
    /// Under a finite ceiling a pole has no rays, its fan being cut away
    /// with everything over the truncation, and a zero's ray ends where it climbs
    /// over the ceiling, complete as it stands.
    func seededRays(from index: Int, of points: [SingularPoint], theta: Double, level: Double,
                    g: @Sendable (Complex) -> Double, ceiling: Double) -> [Ray] {
        let s = points[index]
        guard s.order != 0, s.order > 0 || !ceiling.isFinite else { return [] }
        let z0 = s.point, radius = Self.handoff * cell
        var out: [Ray] = []
        for j in 0..<abs(s.order) {
            let phi = (theta - s.coefficient.argument + 2 * .pi * Double(j)) / Double(s.order)
            let r0 = 0.5 * cell
            let q0 = P2<DomainSpace>(z0.x + r0 * cos(phi), z0.y + r0 * sin(phi))
            guard inWindow(q0), let p1 = onto(q0, level: level, g: g, within: 0.25 * cell) else { continue }
            var chord: [[P2<DomainSpace>]] = [[]]
            subdivide(z0, p1, level: level, g: g, cut: true, depth: 0, into: &chord)
            guard chord.count == 1 else { continue }
            var ray = Ray(run: [z0] + chord[0] + [p1], point: index, handoff: false)
            let (tail, _) = trace(from: p1, before: z0, level: level, g: g,
                                  stopping: { p in Self.apart(p, z0) >= radius ? index : nil })
            ray.run += tail
            let tip = ray.run[ray.run.count - 1]
            ray.handoff = Self.apart(tip, z0) >= radius && !(magnitude(Complex(tip.x, tip.y)) > ceiling)
            out.append(ray)
        }
        return out
    }

    /// The piece cut back from the handoff discs: its vertices inside one
    /// are dropped, and an end so made is marked with the point it meets,
    /// for that point's ray to be spliced on. A piece that traversing
    /// squares led through a singular point's cell, from one ray onto
    /// another, comes apart there into the two.
    func trimmed(_ piece: Piece, awayFrom discs: [(index: Int, center: P2<DomainSpace>)]) -> [Piece] {
        guard !discs.isEmpty else { return [piece] }
        let radius = Self.handoff * cell
        func inside(_ p: P2<DomainSpace>) -> Int? {
            discs.first { Self.apart(p, $0.center) < radius }?.index
        }
        var out: [Piece] = []
        var run: [P2<DomainSpace>] = []
        var follow = piece.ends.start, met = piece.met.start
        for p in piece.run {
            if let k = inside(p) {
                if run.count >= 2 { out.append(Piece(run: run, ends: (follow, false), met: (met, k))) }
                run = []; follow = false; met = k
            } else {
                run.append(p)
            }
        }
        if run.count >= 2 { out.append(Piece(run: run, ends: (follow, piece.ends.end), met: (met, piece.met.end))) }
        return out
    }

    /// The rays spliced onto the runs that meet their discs, and every ray
    /// nothing meets followed on from its handoff.
    ///
    /// A run's end inside a point's disc lies on one of the point's rays,
    /// the one leaving at the nearest angle; the two are joined by a chord
    /// placed like any other, so the ink is one curve from the point out.
    /// A ray no run meets -- the grid drew nothing along it, as it may on a
    /// coarse grid -- is followed on f to wherever it ends, and if that
    /// is another point's disc it is spliced there in turn. An end that no
    /// ray fits is carried onto the point as before.
    func spliced(_ rays: [Ray], onto runs: [Piece], points: [SingularPoint], level: Double,
                 g: @Sendable (Complex) -> Double,
                 stopping: @escaping @Sendable (P2<DomainSpace>) -> Int?) -> [[P2<DomainSpace>]] {
        var used = [Bool](repeating: false, count: rays.count)
        /// The unused handoff ray of point `k` leaving toward `end`, by angle.
        func ray(of k: Int, toward end: P2<DomainSpace>) -> Int? {
            let center = points[k].point
            let angle = atan2(end.y - center.y, end.x - center.x)
            var best: (index: Int, off: Double)?
            for (i, r) in rays.enumerated() where r.point == k && r.handoff && !used[i] {
                let tip = r.run[r.run.count - 1]
                let off = abs(Self.wrapped(atan2(tip.y - center.y, tip.x - center.x) - angle))
                if best == nil || off < best!.off { best = (i, off) }
            }
            guard let best, best.off <= .pi / Double(2 * abs(points[k].order)) else { return nil }
            return best.index
        }
        /// The chord from `a` to `b`, with its interior vertices, when it can be placed.
        func chord(_ a: P2<DomainSpace>, _ b: P2<DomainSpace>) -> [P2<DomainSpace>]? {
            var runs: [[P2<DomainSpace>]] = [[]]
            subdivide(a, b, level: level, g: g, cut: true, depth: 0, into: &runs)
            return runs.count == 1 ? runs[0] : nil
        }
        /// `run` with the ray of point `k` spliced onto its start, or carried onto the point.
        func headed(_ run: [P2<DomainSpace>], by k: Int) -> [P2<DomainSpace>] {
            if let i = ray(of: k, toward: run[0]), let bridge = chord(rays[i].run[rays[i].run.count - 1], run[0]) {
                used[i] = true
                return rays[i].run + bridge + run
            }
            let center = points[k].point
            if let bridge = chord(center, run[0]) { return [center] + bridge + run }
            return run
        }
        var out: [[P2<DomainSpace>]] = []
        for piece in runs {
            guard piece.run.count >= 2 else { continue }
            var run = piece.run
            if let k = piece.met.start { run = headed(run, by: k) }
            if let k = piece.met.end { run = headed(run.reversed(), by: k).reversed() }
            out.append(run)
        }
        // Rays nothing met, followed on; one that reaches another disc is
        // spliced there, to a ray not yet used.
        for i in rays.indices where !used[i] {
            var run = rays[i].run
            if rays[i].handoff {
                used[i] = true
                let (tail, met) = trace(from: run[run.count - 1], before: run[run.count - 2],
                                        level: level, g: g, stopping: stopping)
                run += tail
                if let k = met, k != rays[i].point { run = headed(run.reversed(), by: k).reversed() }
            }
            out.append(run)
        }
        return out
    }

    /// How far round `z` the path turns, in radians: ±2π for a loop round it.
    static func linearCurves(_ path: [P2<DomainSpace>], around z: P2<DomainSpace>) -> Double {
        var total = 0.0
        for (a, b) in zip(path, path.dropFirst()) {
            let ta = atan2(a.y - z.y, a.x - z.x), tb = atan2(b.y - z.y, b.x - z.x)
            total += wrapped(tb - ta)
        }
        return total
    }

    /// The rings at one level that the grid cannot see: around a zero, at a
    /// level below |f| at every corner of the cell the zero lies in, no node
    /// lies inside the loop and marching squares draws nothing, though the
    /// loop exists -- around a simple zero every such level is one closed
    /// curve. The grid decides which contours exist only where it can see
    /// them; at a zero, f decides. Each ring is seeded from the zero as a
    /// circle of radius level / |f'|, every vertex moved onto the level along
    /// its ray by bisection, the loop subdivided to the tolerance like any
    /// other, and cut to the window where it leaves it, each crossing solved
    /// along the window's edge. A level the grid did see -- a path already
    /// turning round the zero -- is left to the grid.
    func seededRings(level: Double, zeros: [P2<DomainSpace>],
                     existing: [[P2<DomainSpace>]]) -> [[P2<DomainSpace>]] {
        guard level > 0, level.isFinite else { return [] }
        var out: [[P2<DomainSpace>]] = []
        for z0 in zeros {
            let fx = (z0.x - grid.real.lo) / dx, fy = (z0.y - grid.imag.lo) / dy
            let i = min(max(Int(fx.rounded(.down)), 0), width - 2)
            let j = min(max(Int(fy.rounded(.down)), 0), height - 2)
            var lowest = Double.infinity
            for (ci, cj) in [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)] {
                lowest = min(lowest, magnitude(Complex(grid.real.lo + Double(ci) * dx, grid.imag.lo + Double(cj) * dy)))
            }
            guard level < lowest else { continue }
            guard !existing.contains(where: { abs(Self.linearCurves($0, around: z0)) > .pi / 2 }) else { continue }
            let c = Complex(z0.x, z0.y)
            let h = 1e-6 * cell
            let d = (value(c + h) - value(c - h)) / (2 * h)
            guard d.isFinite, d.magnitude > 0 else { continue }
            let r0 = level / d.magnitude
            guard r0.isFinite, r0 < 2 * cell else { continue }
            let n = 24
            var loop: [P2<DomainSpace>] = []
            for k in 0..<n {
                let theta = 2 * Double.pi * Double(k) / Double(n)
                let dir = Complex(cos(theta), sin(theta))
                func excess(_ r: Double) -> Double { magnitude(c + r * dir) - level }
                // Bracket the crossing along the ray from the zero, where
                // |f| is below the level, outward to where it is above.
                var lo = 0.0, hi = r0
                var tries = 0
                while excess(hi) <= 0, tries < 12 { hi *= 2; tries += 1 }
                guard excess(hi) > 0, hi <= 4 * cell else { loop = []; break }
                for _ in 0..<60 {
                    let m = 0.5 * (lo + hi)
                    if excess(m) > 0 { hi = m } else { lo = m }
                }
                let r = 0.5 * (lo + hi)
                loop.append(P2(c.re + r * dir.re, c.im + r * dir.im))
            }
            guard loop.count == n else { continue }
            loop.append(loop[0])
            var refined: [[P2<DomainSpace>]] = [[loop[0]]]
            for (a, b) in zip(loop, loop.dropFirst()) {
                subdivide(a, b, level: level, g: magnitude, cut: false, depth: 0, into: &refined)
                refined[refined.count - 1].append(b)
            }
            if refined.count == 1 {
                out += clippedToWindow(refined[0], level: level)
            } else {
                // Broken where a chord could not be placed: its pieces, open.
                for piece in refined where piece.count >= 2 { out += clippedToWindow(piece, level: level, closed: false) }
            }
        }
        return out
    }

    /// A loop cut to the window: the runs of it inside, each ending where
    /// it crosses the window's edge, the crossing solved on the edge itself
    /// so that it lies on the level as the grid's own edge vertices do. A
    /// loop wholly inside is returned closed.
    func clippedToWindow(_ loop: [P2<DomainSpace>], level: Double, closed: Bool = true) -> [[P2<DomainSpace>]] {
        let inside = loop.map { inWindow($0) }
        if inside.allSatisfy({ $0 }) { return [loop] }
        /// The level's crossing of the window's edge between an inside vertex
        /// and an outside one.
        func crossing(_ a: P2<DomainSpace>, _ b: P2<DomainSpace>) -> P2<DomainSpace>? {
            var lo = a, hi = b
            for _ in 0..<60 {
                let m = P2<DomainSpace>(0.5 * (lo.x + hi.x), 0.5 * (lo.y + hi.y))
                if inWindow(m) { lo = m } else { hi = m }
            }
            let p = lo
            // Along whichever edge it is on, solve |f| = level by bisection
            // between the chord point and a step either way.
            let r = grid.real, im = grid.imag
            let onRow = abs(p.y - im.lo) <= 1e-6 * cell || abs(p.y - im.hi) <= 1e-6 * cell
            let onColumn = abs(p.x - r.lo) <= 1e-6 * cell || abs(p.x - r.hi) <= 1e-6 * cell
            guard onRow || onColumn else { return p }
            func at(_ s: Double) -> P2<DomainSpace> { onRow ? P2(p.x + s, p.y) : P2(p.x, p.y + s) }
            func excess(_ s: Double) -> Double { magnitude(Complex(at(s).x, at(s).y)) - level }
            var slo = -1e-3 * cell, shi = 1e-3 * cell
            var tries = 0
            while (excess(slo) > 0) == (excess(shi) > 0), tries < 10 { slo *= 2; shi *= 2; tries += 1 }
            guard (excess(slo) > 0) != (excess(shi) > 0) else { return p }
            var elo = excess(slo)
            for _ in 0..<60 {
                let m = 0.5 * (slo + shi)
                let em = excess(m)
                if (em > 0) == (elo > 0) { slo = m; elo = em } else { shi = m }
            }
            return at(0.5 * (slo + shi))
        }
        // A closed loop: start the walk at an outside vertex so no run is
        // split in two. An open run is walked from its first vertex.
        let n = closed ? loop.count - 1 : loop.count
        guard n >= 2, let start = closed ? inside.firstIndex(of: false) : 0 else { return [loop] }
        var runs: [[P2<DomainSpace>]] = []
        var run: [P2<DomainSpace>] = []
        for step in 0...(closed ? n : n - 1) {
            let k = (start + step) % n
            let v = loop[k]
            if inside[k] {
                if run.isEmpty, step > 0, let c = crossing(v, loop[(start + step - 1) % n]) { run.append(c) }
                run.append(v)
            } else if !run.isEmpty {
                if let c = crossing(run[run.count - 1], v) { run.append(c) }
                if run.count >= 2 { runs.append(run) }
                run = []
            }
        }
        if run.count >= 2 { runs.append(run) }
        return runs
    }

    // MARK: one path

    /// One grid path, placed: possibly several pieces, where wrap vertices
    /// were dropped or a chord could not be placed, and only its stretches
    /// that reach under `ceiling`.
    ///
    /// A segment is placed when either of its grid vertices has |f| at or
    /// under the ceiling, so the vertex just over the truncation is placed too and
    /// the cut at the truncation edge (`Surface.truncationCrossing`) is the one it always was.
    /// A stretch of segments wholly over it is left out: its ink is cut away
    /// whatever its vertices are. The ends this makes are not ends of the
    /// curve, so they are not followed on f (`traced`) or carried to a zero.
    func pieces(of path: [P2<DomainSpace>], level: Double,
                g: @Sendable (Complex) -> Double, cut: Bool,
                ceiling: Double = .infinity) -> [Piece] {
        guard path.count >= 2 else { return [] }
        guard ceiling.isFinite else { return placed(path, level: level, g: g, cut: cut) }
        let n = path.count - 1
        // Not over: a vertex where |f| is not a number is the snap's to test.
        let low = path.map { p in !(magnitude(Complex(p.x, p.y)) > ceiling) }
        let wanted = (0..<n).map { low[$0] || low[$0 + 1] }
        if !wanted.contains(false) { return placed(path, level: level, g: g, cut: cut) }
        guard wanted.contains(true) else { return [] }
        var vertices = path, want = wanted
        let closed = path.first == path.last
        if closed {
            // Walked from the start of a stretch, so none is split in two.
            let s = (0..<n).first { wanted[$0] && !wanted[($0 + n - 1) % n] }!
            vertices = (0...n).map { path[(s + $0) % n] }
            want = (0..<n).map { wanted[(s + $0) % n] }
        }
        var out: [Piece] = []
        var k = 0
        while k < n {
            guard want[k] else { k += 1; continue }
            var e = k
            while e < n && want[e] { e += 1 }
            out += placed(Array(vertices[k...e]), level: level, g: g, cut: cut,
                          followStart: !closed && k == 0, followEnd: !closed && e == n)
            k = e
        }
        return out
    }

    /// A run of grid vertices placed on f, in pieces. `followStart` and
    /// `followEnd` say whether the path's own ends are the curve's, to be
    /// followed on f and carried to a zero or pole, or a cut at the ceiling;
    /// an end made by a split is always the curve's.
    func placed(_ path: [P2<DomainSpace>], level: Double,
                g: @Sendable (Complex) -> Double, cut: Bool,
                followStart: Bool = true, followEnd: Bool = true) -> [Piece] {
        guard path.count >= 2 else { return [] }
        let closed = path.first == path.last
        var snapped = path.map { snap($0, level: level, g: g, cut: cut) }
        if closed { snapped[snapped.count - 1] = snapped[0] }
        // The runs: a new one begins at a dropped wrap vertex and wherever a
        // chord could not be placed on the level set (`subdivide`), so that
        // nothing is drawn between two vertices the curve does not join.
        var runs: [[P2<DomainSpace>]] = [[]]
        var previous: P2<DomainSpace>?
        for vertex in snapped {
            guard let vertex else {
                runs.append([])
                previous = nil
                continue
            }
            if let previous {
                subdivide(previous, vertex, level: level, g: g, cut: cut, depth: 0, into: &runs)
            }
            runs[runs.count - 1].append(vertex)
            previous = vertex
        }
        // Which runs begin at the path's first vertex and end at its last:
        // only those ends can be cuts at the ceiling.
        let headed = snapped.first! != nil, tailed = snapped.last! != nil
        return runs.enumerated()
            .filter { $1.count >= 2 }
            .map { i, run in
                Piece(run: run, ends: (start: followStart || i != 0 || !headed,
                                       end: followEnd || i != runs.count - 1 || !tailed))
            }
    }

    /// A phase run followed on f beyond either end that lies inside the
    /// window and on no zero. Marching squares ends a run where it cannot
    /// bracket the level on a grid edge: beside an essential singularity
    /// the phase turns many times within a cell, and the lines stopped a
    /// cell short of the origin they run into. From the last vertex the
    /// grid placed, the curve is followed on f: a step along the
    /// tangent, corrected onto the level set along the gradient, kept only
    /// when the chord to it can be placed (`subdivide`) and halved
    /// otherwise, until the curve leaves the window, where it ends on the
    /// edge, or the step has shrunk to nothing at a singularity.
    func traced(_ piece: Piece, level: Double, g: @Sendable (Complex) -> Double,
                stopping: (@Sendable (P2<DomainSpace>) -> Int?)? = nil) -> Piece {
        var piece = piece
        guard piece.run.count >= 2, piece.run.first != piece.run.last else { return piece }
        func interior(_ p: P2<DomainSpace>) -> Bool {
            let r = grid.real, i = grid.imag, slack = 1e-6 * cell
            return p.x > min(r.lo, r.hi) + slack && p.x < max(r.lo, r.hi) - slack
                && p.y > min(i.lo, i.hi) + slack && p.y < max(i.lo, i.hi) - slack
        }
        /// On a zero or a pole, where the line is already at its end.
        func onSingularPoint(_ p: P2<DomainSpace>) -> Bool {
            let u = magnitude(Complex(p.x, p.y))
            return !(u.isFinite && u > 1e-9 && u < NativeLandscape.huge)
        }
        let n = piece.run.count
        if piece.ends.end, interior(piece.run[n - 1]), !onSingularPoint(piece.run[n - 1]) {
            let (tail, met) = trace(from: piece.run[n - 1], before: piece.run[n - 2], level: level,
                                    g: g, stopping: stopping)
            piece.run += tail
            piece.met.end = met
        }
        if piece.ends.start, interior(piece.run[0]), !onSingularPoint(piece.run[0]) {
            let (head, met) = trace(from: piece.run[0], before: piece.run[1], level: level,
                                    g: g, stopping: stopping)
            piece.run = head.reversed() + piece.run
            piece.met.start = met
        }
        return piece
    }

    /// The vertices the curve through `p` continues with, away from `before`.
    func trace(from p0: P2<DomainSpace>, before: P2<DomainSpace>, level: Double,
               g: @Sendable (Complex) -> Double,
               stopping: (@Sendable (P2<DomainSpace>) -> Int?)? = nil)
        -> (run: [P2<DomainSpace>], met: Int?) {
        var out: [P2<DomainSpace>] = []
        var p = p0, prev = before
        var h = 0.25 * cell
        let hMin = 1e-7 * cell, hMax = 0.5 * cell
        var met: Int?
        var ended = false
        let u0 = magnitude(Complex(p0.x, p0.y))
        // |f| the last time a zero or pole was looked for from the line.
        var marker = u0
        /// Onto the zero or pole `z`, within `reach` of `p`, when the chord
        /// to it can be placed: every phase line through either ends on it.
        func carried(to z: P2<DomainSpace>?) -> Bool {
            guard let z, Self.apart(z, p) <= Self.reach * cell else { return false }
            var runs: [[P2<DomainSpace>]] = [[]]
            subdivide(p, z, level: level, g: g, cut: true, depth: 0, into: &runs)
            guard runs.count == 1 else { return false }
            out += runs[0]; out.append(z)
            return true
        }
        for _ in 0..<40000 {
            // A trace ends on a zero or a pole, on the window's edge, or
            // where its step has shrunk to nothing. The bound is a safety
            // net, not a budget: a line the grid lost beside its zero is
            // followed the whole way to its pole.
            guard out.count < 20000 else { break }
            // Into the handoff disc of a singular point, where its ray takes
            // over (`spliced`).
            if let stopping, let k = stopping(p) { met = k; ended = true; break }
            // Over the ceiling, where the ink is cut away: the vertex just
            // over it is kept, for the cut at the truncation edge to be solved on.
            if magnitude(Complex(p.x, p.y)) > ceiling { ended = true; break }
            // Approaching a zero or a pole, the line ends on it rather than
            // stepping through it onto another ray, and rather than creeping
            // up to it: |f| doubles with every halving of the distance to a
            // simple pole and halves with it to a simple zero, so each time
            // |f| has doubled or halved since the last look, the point is
            // looked for, and the line is carried onto it from two cells out.
            let u = magnitude(Complex(p.x, p.y))
            if u <= marker / 2 {
                marker = u
                if carried(to: zero(near: p)) { ended = true; break }
            } else if u >= 2 * marker {
                marker = u
                if carried(to: pole(near: p)) { ended = true; break }
            }
            if u > 1e6 { break }
            let (gx, gy) = gradient(g, at: p)
            let slope = (gx * gx + gy * gy).squareRoot()
            guard slope > 0, slope.isFinite else { break }
            var tx = -gy / slope, ty = gx / slope
            if tx * (p.x - prev.x) + ty * (p.y - prev.y) < 0 { tx = -tx; ty = -ty }
            var placed: [P2<DomainSpace>]?
            var next: P2<DomainSpace>?
            while h >= hMin {
                let q0 = P2<DomainSpace>(p.x + h * tx, p.y + h * ty)
                if let q = onto(q0, level: level, g: g, within: 0.5 * h) {
                    var runs: [[P2<DomainSpace>]] = [[]]
                    subdivide(p, q, level: level, g: g, cut: true, depth: 0, into: &runs)
                    if runs.count == 1 { placed = runs[0]; next = q; break }
                }
                h *= 0.5
            }
            guard let q = next, let between = placed else { break }
            if !inWindow(q) {
                // Ended on the window's edge: the chord's crossing of the
                // edge, then the level solved along the edge itself.
                var lo = p, hi = q
                for _ in 0..<60 {
                    let m = P2<DomainSpace>(0.5 * (lo.x + hi.x), 0.5 * (lo.y + hi.y))
                    if inWindow(m) { lo = m } else { hi = m }
                }
                let e = lo
                let r = grid.real, im = grid.imag
                let onRow = abs(e.y - im.lo) <= 1e-6 * cell || abs(e.y - im.hi) <= 1e-6 * cell
                func at(_ t: Double) -> P2<DomainSpace> { onRow ? P2(e.x + t, e.y) : P2(e.x, e.y + t) }
                func res(_ t: Double) -> Double { residual(g(Complex(at(t).x, at(t).y)), level: level, cut: true) }
                var slo = -1e-3 * cell, shi = 1e-3 * cell, tries = 0
                while (res(slo) > 0) == (res(shi) > 0), tries < 8 { slo *= 2; shi *= 2; tries += 1 }
                var end: P2<DomainSpace>?
                if (res(slo) > 0) != (res(shi) > 0), abs(res(slo)) < .pi / 2, abs(res(shi)) < .pi / 2 {
                    var rlo = res(slo)
                    for _ in 0..<60 {
                        let m = 0.5 * (slo + shi)
                        let rm = res(m)
                        if (rm > 0) == (rlo > 0) { slo = m; rlo = rm } else { shi = m }
                    }
                    end = at(0.5 * (slo + shi))
                }
                // Only an end on the level, that the surface has a height
                // at, reached by a chord that can be placed: a line that
                // meets the edge where the level is not crossed along it,
                // or runs into a singularity on the edge, stops at its last
                // vertex rather than ending off the level or jumping to the truncation.
                if let end {
                    var runs: [[P2<DomainSpace>]] = [[]]
                    subdivide(p, end, level: level, g: g, cut: true, depth: 0, into: &runs)
                    let rEnd = residual(g(Complex(end.x, end.y)), level: level, cut: true)
                    if rEnd.isFinite, abs(rEnd) <= 1e-9, magnitude(Complex(end.x, end.y)).isFinite, runs.count == 1 {
                        out += runs[0]
                        out.append(end)
                    }
                }
                _ = r
                ended = true
                break
            }
            out += between
            out.append(q)
            prev = p; p = q
            if between.isEmpty { h = min(h * 1.5, hMax) }
        }
        // Stopped inside the window: on the zero or pole beside it, if any.
        if !ended { _ = carried(to: zero(near: p)) || carried(to: pole(near: p)) }
        return (out, met)
    }

    /// `q0` moved onto the level set along the gradient, by Newton; nil when
    /// it does not converge within `reach` of where it started.
    func onto(_ q0: P2<DomainSpace>, level: Double, g: @Sendable (Complex) -> Double,
              within reach: Double) -> P2<DomainSpace>? {
        var q = q0
        for _ in 0..<30 {
            let r = residual(g(Complex(q.x, q.y)), level: level, cut: true)
            guard r.isFinite else { return nil }
            let (gx, gy) = gradient(g, at: q)
            let s2 = gx * gx + gy * gy
            guard s2 > 0, s2.isFinite else { return nil }
            let step = r / s2
            q = P2(q.x - step * gx, q.y - step * gy)
            guard Self.apart(q, q0) <= reach else { return nil }
            if abs(step) * s2.squareRoot() <= 1e-12 * cell { return q }
        }
        return nil
    }

    /// A phase run carried to the zero or pole of f it ends beside, at
    /// either end.
    ///
    /// Every phase meets at a zero and at a pole, so a phase contour that
    /// stops within a cell or so of one was always going there: the grid's
    /// marching squares put its last vertex on the last cell edge it crossed.
    /// The point is found from the end by Newton, on f for a zero and on 1/f
    /// for a pole, accepted when the iteration converges without leaving the
    /// end's neighbourhood and lands on a value a million times smaller than
    /// the end's, and the new segment is subdivided like any other. A run
    /// that already ends on one, or that is closed, is left alone.
    func ended(atSingularPoints run: [P2<DomainSpace>], level: Double,
               g: @Sendable (Complex) -> Double,
               ends: (start: Bool, end: Bool) = (true, true)) -> [P2<DomainSpace>] {
        guard run.count >= 2, run.first != run.last else { return run }
        /// The zero or pole this end leads to, when the chord to it lies on
        /// the level set -- its midpoint's phase within an eighth of a turn
        /// of the level, which a different contour's end beside the same
        /// point fails -- and the run does not already end there from the
        /// other side, as a stub in a pit a cell wide would.
        func zero(for end: P2<DomainSpace>, other: P2<DomainSpace>) -> P2<DomainSpace>? {
            guard let z = self.zero(near: end) ?? self.pole(near: end) else { return nil }
            let apart = ((z.x - other.x) * (z.x - other.x) + (z.y - other.y) * (z.y - other.y)).squareRoot()
            guard apart > 1e-9 * cell else { return nil }
            let mid = Complex((z.x + end.x) / 2, (z.y + end.y) / 2)
            let r = residual(g(mid), level: level, cut: true)
            return r.isFinite && abs(r) <= .pi / 4 ? z : nil
        }
        var out = run
        if ends.start, let z = zero(for: out[0], other: out[out.count - 1]) {
            var lead: [[P2<DomainSpace>]] = [[]]
            subdivide(z, out[0], level: level, g: g, cut: true, depth: 0, into: &lead)
            // Carried only when the whole chord to the zero could be placed.
            if lead.count == 1 { out = [z] + lead[0] + out }
        }
        if ends.end, let z = zero(for: out[out.count - 1], other: out[0]) {
            var tail: [[P2<DomainSpace>]] = [[]]
            subdivide(out[out.count - 1], z, level: level, g: g, cut: true, depth: 0, into: &tail)
            if tail.count == 1 { out += tail[0]; out.append(z) }
        }
        return out
    }

    /// How far from a run's end a zero is looked for, in cells. The grid's
    /// contour can stop two cells short where the zero's own cells pair
    /// their crossings the other way.
    static let reach = 3.0

    /// A zero of f within `reach` cells of `p`, or nil.
    public func zero(near p: P2<DomainSpace>) -> P2<DomainSpace>? {
        root(near: p, of: value)
    }

    /// A pole of f within `reach` cells of `p`, or nil: a zero of 1/f, found
    /// the same way. Newton on 1/f is well conditioned where Newton on f is
    /// not, since 1/f is small and smooth beside a pole.
    public func pole(near p: P2<DomainSpace>) -> P2<DomainSpace>? {
        root(near: p, of: reciprocal)
    }

    /// 1/f: zero where f is not finite, which is the pole itself, and not
    /// finite where f is zero.
    func reciprocal(_ z: Complex) -> Complex {
        let v = value(z)
        guard v.isFinite else { return Complex(0) }
        return v.magnitude > 0 ? Complex(1) / v : Complex(Double.infinity)
    }

    /// A zero of `g` within `reach` cells of `p`, by Newton from `p`, or nil:
    /// accepted when the iteration converges without leaving that
    /// neighbourhood and lands on a value a million times smaller than at
    /// `p`, so that a step toward a far-off root, or a stall at a critical
    /// point, is not mistaken for one.
    func root(near p: P2<DomainSpace>, of g: (Complex) -> Complex) -> P2<DomainSpace>? {
        var z = Complex(p.x, p.y)
        let g0 = g(z)
        guard g0.isFinite, g0.magnitude > 0 else { return nil }
        let h = 1e-6 * cell
        for _ in 0..<40 {
            let gz = g(z)
            guard gz.isFinite else { return nil }
            let d = (g(z + h) - g(z - h)) / (2 * h)
            guard d.isFinite, d.squaredMagnitude > 0 else { return nil }
            let step = gz / d
            guard step.isFinite else { return nil }
            z = z - step
            guard Complex(z.re - p.x, z.im - p.y).magnitude <= Self.reach * cell else { return nil }
            if step.magnitude <= 1e-12 * cell {
                let at = g(z)
                guard at.isFinite, at.magnitude <= 1e-6 * g0.magnitude else { return nil }
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
            // is the truncation: exp(1/z) beside its essential singularity has a
            // node at 1e-109 next to one at 1e12, and the interpolated
            // crossing is 1e-12 of the way along the edge, which rounds onto
            // the node where f is flat and nowhere near the level. The edge is
            // not recorded, but if exactly one of the node's four edges
            // brackets the level, that is the one.
            let i = Int(fx.rounded()), j = Int(fy.rounded())
            let gp = residual(g(Complex(p.x, p.y)), level: level, cut: cut)
            guard gp.isFinite else { return nil }
            guard !(cut && abs(gp) > .pi / 2) else { return p }
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
            // An end where the field has no value -- a node on a zero, or
            // where f is not finite: the vertex was placed by a sample that
            // meant nothing, and is dropped; a phase line is carried onto
            // the zero from its last placed vertex instead.
            guard ra.isFinite, rb.isFinite else { return nil }
            // The float32 grid saw a crossing that f, in double, does not:
            // the level set passes within rounding of a node. The nearer
            // endpoint is the honest position. For the phase, anything else
            // is a wrap, which is not a crossing of anything.
            if cut && min(abs(ra), abs(rb)) > 1e-9 { return nil }
            return abs(ra) <= abs(rb) ? a : b
        }
        let (a1, b1) = (edge.a, edge.b)
        // Brent's method on t in [0, 1] along the edge. A bracket like
        // exp(1/z)'s beside its essential singularity, 1e-109 at one end and
        // the 1e12 truncation at the other, defeats interpolation -- every step lands
        // beside the small end -- and there it bisects.
        func at(_ t: Double) -> P2<DomainSpace> {
            P2(a1.x + t * (b1.x - a1.x), a1.y + t * (b1.y - a1.y))
        }
        func r(_ t: Double) -> Double {
            let q = at(t)
            return residual(g(Complex(q.x, q.y)), level: level, cut: cut)
        }
        // An edge can cross the level more than once. The crossing meant is
        // the one marching squares put the vertex beside, so the bracket is
        // first split at the vertex, keeping the half that changes sign.
        var (ta, tb, ra, rb) = (0.0, 1.0, edge.ra, edge.rb)
        var t0 = (p.x - a1.x) * (b1.x - a1.x) + (p.y - a1.y) * (b1.y - a1.y)
        t0 /= max((b1.x - a1.x) * (b1.x - a1.x) + (b1.y - a1.y) * (b1.y - a1.y), .leastNormalMagnitude)
        if t0 > 0, t0 < 1, ra != 0, rb != 0 {
            let r0 = r(t0)
            guard r0.isFinite else { return p }
            if r0 == 0 { return at(t0) }
            if r0.sign == ra.sign { (ta, ra) = (t0, r0) } else { (tb, rb) = (t0, r0) }
        }
        guard let t = Root.brent(r, ta, tb, fa: ra, fb: rb,
                                 xtol: 1e-12, ftol: 1e-13 * max(abs(level), 1)) else { return p }
        return at(t)
    }

    /// The estimated distance from the chord `a-b` to the level set, and the
    /// point on the level set nearest the chord's midpoint, when it is worth
    /// inserting.
    /// The vertices placed between `a` and `b` go on the last run of
    /// `runs`. A chord that cannot be placed on the level set -- its
    /// midpoint's value is not finite or, on a phase line, more than a
    /// quarter turn off the level, so the chord spans a jump of the phase;
    /// the depth limit is reached with the error still above the tolerance;
    /// or no point of the level set is found along the chord's normal --
    /// breaks the run there, so that nothing is drawn between two vertices
    /// the curve does not join. Beside an essential singularity the grid
    /// brackets nothing and marching squares joins vertices of different
    /// branches; those chords used to be drawn as they were, as hooks and
    /// zigzags. A chord whose midpoint lies on a critical point of the
    /// field, where the level set crosses itself, has nothing to measure
    /// by and is kept.
    func subdivide(_ a: P2<DomainSpace>, _ b: P2<DomainSpace>, level: Double,
                   g: @Sendable (Complex) -> Double, cut: Bool, depth: Int,
                   into runs: inout [[P2<DomainSpace>]]) {
        func unplaced() { runs.append([]) }
        let mid = P2<DomainSpace>((a.x + b.x) / 2, (a.y + b.y) / 2)
        var r = residual(g(Complex(mid.x, mid.y)), level: level, cut: cut)
        guard r.isFinite else { return unplaced() }
        let chord = ((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y)).squareRoot()
        guard chord > 1e-9 * cell else { return }
        let (gx, gy) = gradient(g, at: mid)
        let slope = (gx * gx + gy * gy).squareRoot()
        guard slope > 0, slope.isFinite else { return }
        var error = abs(r) / slope
        if cut {
            // A phase level repeats every turn: beside an essential
            // singularity the circles of one level crowd together, and a
            // chord from one circle to the next can have its midpoint on a
            // third, with a residual that says nothing. The phase is read
            // at eight points along the chord and unwrapped from one to
            // the next; a jump of more than a quarter turn between
            // neighbours is a chord the samples cannot follow, and a
            // residual that does not return to the level at the far end
            // is a chord that has wound round. Along a chord the samples
            // can follow, its departure from the level set is the largest
            // residual over the gradient at any sample, so a chord that
            // follows a small circle to the tolerance is placed although
            // the phase turns steeply beside it.
            // The ends themselves are not read: a chord's end may be a zero,
            // where the phase is undefined.
            var unwrapped = 0.0
            for k in 1...7 {
                let t = Double(k) / 8
                let q = P2<DomainSpace>(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y))
                let rq = residual(g(Complex(q.x, q.y)), level: level, cut: true)
                guard rq.isFinite else { return unplaced() }
                var d = rq - unwrapped
                while d > .pi { d -= 2 * .pi }
                while d < -.pi { d += 2 * .pi }
                guard abs(d) <= .pi / 2 else { return unplaced() }
                unwrapped += d
                guard abs(unwrapped) <= .pi / 2 else { return unplaced() }
                if k == 4 { r = unwrapped }
                let (qx, qy) = gradient(g, at: q)
                let s = (qx * qx + qy * qy).squareRoot()
                if s > 0, s.isFinite { error = max(error, abs(unwrapped) / s) }
            }
        }
        // A phase line is a curve in space: across the plane it is placed to
        // the tolerance, but on a steep graph its height between two
        // vertices is not the chord's, and in projection the chords kink.
        // So a chord is also split where |f| at its midpoint departs from
        // the mean of |f| at its ends by more than the tolerance. A
        // magnitude line is level and has no height to depart in. Nor has a
        // chord with a pole at its end: the height there is whatever the
        // truncation's is, and no halving brings the midpoint's height to the mean
        // of a finite one and `huge`, so the test would refuse every chord
        // that carries a line onto its pole.
        var sag = 0.0
        if cut, lifted {
            let ua = magnitude(Complex(a.x, a.y)), ub = magnitude(Complex(b.x, b.y))
            let um = magnitude(Complex(mid.x, mid.y))
            if ua.isFinite, ub.isFinite, um.isFinite,
               ua < NativeLandscape.huge, ub < NativeLandscape.huge {
                sag = abs(um - 0.5 * (ua + ub))
            }
        }
        guard error > tolerance * cell || sag > tolerance * cell else { return }
        guard depth < maxDepth else { return unplaced() }
        // Solve along the unit normal to the chord, never further than a
        // chord's length from the midpoint. Where the level curve crosses
        // itself at a critical point (|cn| = 1 at 0, |sin| = 1 at π/2,
        // 1/(z³ - 1) = 1 at 0 with f'' = 0 too) the residual along the
        // normal has a multiple root, which an iteration that does not
        // bracket creeps toward linearly and may stop short of.
        let nx = -(b.y - a.y) / chord, ny = (b.x - a.x) / chord
        let gn = gx * nx + gy * ny
        guard abs(gn) > 1e-3 * slope else { return }
        func along(_ t: Double) -> P2<DomainSpace> { P2(mid.x + t * nx, mid.y + t * ny) }
        func res(_ t: Double) -> Double {
            let q = along(t)
            return residual(g(Complex(q.x, q.y)), level: level, cut: cut)
        }
        // So first look for a sign change: step toward the Newton step,
        // doubling until the residual changes sign or the step reaches the
        // chord's length.
        var reach = abs(r / gn)
        let toward: Double = r / gn > 0 ? -1 : 1
        var end: (t: Double, r: Double)?
        while true {
            let t = toward * min(reach, chord)
            let rt = res(t)
            guard rt.isFinite, !(cut && abs(rt) > .pi / 2) else { break }
            if rt == 0 || rt.sign != r.sign { end = (t, rt); break }
            guard reach < chord else { break }
            reach *= 2
        }
        var found: P2<DomainSpace>?
        if let end {
            // A bracket, and Brent's method within it, whose multiplicity
            // bisection does not care about. The bracket is the certificate:
            // the level curve crosses the normal within xtol of the point.
            // The residual over the gradient could not say so at a saddle,
            // where the gradient vanishes at the very point sought. A phase
            // residual also changes sign across a wrap, so a point whose
            // residual is near a half turn is refused.
            if let t = Root.brent(res, 0, end.t, fa: r, fb: end.r, xtol: 1e-10 * cell),
               !(cut && abs(res(t)) > .pi / 2) {
                found = along(t)
            }
        } else {
            // No sign change: the residual only touches zero, as it does
            // along a line through a simple saddle (|cn| - 1 ~ y²/2 up the
            // imaginary axis), and there is nothing to bracket. Newton's
            // step, then the secant, accepting only a converged point: within
            // a hundredth of the tolerance by the estimate the measurement
            // uses, residual over the gradient at the point itself.
            func converged(_ q: P2<DomainSpace>, _ r: Double) -> Bool {
                let (gx, gy) = gradient(g, at: q)
                let s = (gx * gx + gy * gy).squareRoot()
                return s > 0 && s.isFinite && abs(r) / s <= 1e-2 * tolerance * cell
            }
            var t0 = 0.0, r0 = r
            var t1 = -r / gn
            var best: (q: P2<DomainSpace>, r: Double)?
            for _ in 0..<24 {
                if abs(t1) > chord { break }
                let q = along(t1)
                let r1 = res(t1)
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
        }
        guard let point = found else { return }
        subdivide(a, point, level: level, g: g, cut: cut, depth: depth + 1, into: &runs)
        runs[runs.count - 1].append(point)
        subdivide(point, b, level: level, g: g, cut: cut, depth: depth + 1, into: &runs)
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
    /// small, measured against f rather than against a finer grid. Truncated at
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
    /// `build`.
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
