import Foundation
import simd

/// Deriving the hatching: the ink that shades a cut face and a truncated top.
///
/// `kurven/hatch.py` is the definition of all four kinds and this is the
/// consumer's implementation of it; `tests/fixtures/hatch` holds the two to each
/// other on the same grid, the same perimeter and the same caps. Where they
/// necessarily differ is the *height*: Python has f and evaluates it, this has
/// the grid and interpolates it, so a stroke's top sits at the grid's crest
/// rather than the function's. That difference is the grid's interpolation
/// error, half a cell's rise at most -- and a whole cell's rise where a zero of
/// f lies between two samples, since the grid's chord over a pit bottoms out at
/// the nearer sample. So with a refiner in play the crest gains a vertex at
/// every minimum of |f| between two nodes, solved on f, and the hatch stands
/// under it (`crest(of:tiles:refine:)`). The fixtures compare the grid's
/// answer, with no refiner.
///
/// Why this is here at all, rather than a list of strokes in the bundle: the cap
/// is a slider. Moving it moves every plateau, which moves every cap stroke and
/// every rim and re-cuts every wall stroke at the new crest. Shipping the answer
/// would make the slider a round trip to Python; shipping the question makes it
/// a few milliseconds of this.
public extension Surface {
    /// What the hatching needs beyond the grids: where the walls stand, what
    /// the footprint is, and how the landscape is tiled.
    struct HatchContext: Sendable {
        public var perimeter: BoundaryPerimeter?
        public var region: Region
        public var tiles: [Affine2]

        public init(perimeter: BoundaryPerimeter?, region: Region = .full,
                    tiles: [Affine2] = [.identity]) {
            self.perimeter = perimeter; self.region = region; self.tiles = tiles
        }

        /// The context a manifest describes. The wall kinds hatch the perimeter
        /// the walls are *derived* from, referenced rather than repeated -- so a
        /// bundle whose walls are a dumped mesh has no perimeter to hatch, and
        /// says so by deriving nothing.
        public init(_ occluder: Occluder) {
            if case .perimeter(let p, _) = occluder.walls {
                perimeter = p
            } else {
                perimeter = nil
            }
            region = occluder.region
            tiles = occluder.tiles
        }
    }

    /// Any described layer's ink: a contour family, or a hatching.
    ///
    /// The one place that knows which kinds of description exist, so a reader,
    /// a re-derivation after an edit and a bake all produce the same strokes
    /// for the same spec.
    func ink(_ spec: LayerSpec, occluder: Occluder,
             refine: ContourRefine? = nil) -> PolylineSet<WorldSpace> {
        if let hatched = hatch(spec.source, in: HatchContext(occluder), refine: refine) {
            return hatched
        }
        return derive(spec.source, policy: spec.heightPolicy,
                      region: occluder.region, tiles: occluder.tiles, refine: refine)
    }

    /// Any of the four hatching sources, as world-space polylines. `nil` for a
    /// source this does not own, so a caller can use it as the test as well as
    /// the derivation. With a refiner, the cut faces take their heights from f.
    func hatch(_ source: LayerSource, in context: HatchContext,
               refine: ContourRefine? = nil) -> PolylineSet<WorldSpace>? {
        var paths: [[P3<WorldSpace>]]
        switch source {
        case .file, .contour, .parameterLines, .winding, .foldLines, .trajectory, .edges:
            return nil
        case .wallHatch(let edges, let spacing, let pitch, let trim, let base, let top):
            guard let perimeter = context.perimeter else { return .empty }
            paths = wallHatch(perimeter, edges: edges, spacing: spacing, pitch: pitch,
                              trim: trim, base: base, topOffset: top,
                              tiles: context.tiles, refine: refine)
        case .wallOutline(let edges, let pitch, let base):
            guard let perimeter = context.perimeter else { return .empty }
            paths = wallOutline(perimeter, edges: edges, pitch: pitch, base: base,
                                tiles: context.tiles, refine: refine)
        case .capHatch(let axis, let spacing, let tiled):
            paths = capHatch(axis: axis, spacing: spacing, refine: refine)
            if tiled { paths = Surface.replicate(paths, context.tiles) }
        case .capOutline(let tiled):
            paths = capOutline(tiles: context.tiles, refine: refine)
            if tiled { paths = Surface.replicate(paths, context.tiles) }
        }
        return PolylineSet(paths: Surface.clip(paths, to: context.region))
    }
}

// MARK: - the shared rules

extension Surface {
    /// Samples along an edge of this length, hatched at this spacing.
    ///
    /// `floor(L / spacing + 0.5)` rather than `.rounded()`: Swift rounds a half
    /// away from zero and Python rounds it to even, and an edge that is an exact
    /// multiple of the spacing would otherwise be hatched differently by the two
    /// sides of the contract.
    static func strokeCount(length: Double, spacing: Double) -> Int {
        guard spacing.isFinite, spacing > 0 else { return 2 }
        return max(2, Int((length / spacing + 0.5).rounded(.down)) + 1)
    }

    /// `Mesh.wallCurtain`'s parameterization, which is not `a + (b - a)t` in
    /// floating point: the hatch has to stand on the wall, not a rounding error
    /// away from it.
    static func point(on edge: PerimeterEdge, at t: Double) -> P2<DomainSpace> {
        P2(edge.start.x * (1 - t) + edge.end.x * t,
           edge.start.y * (1 - t) + edge.end.y * t)
    }

    /// One vertical stroke, subdivided every `pitch`.
    ///
    /// Not a two-point segment. The bake clips per vertex, so a stroke whose
    /// middle is behind a ridge and whose ends are not would be drawn straight
    /// through it.
    static func vertical(x: Double, y: Double, base: Double, top: Double,
                         pitch: Double) -> [P3<WorldSpace>]? {
        guard top > base else { return nil }
        var steps = 1
        if pitch.isFinite, pitch > 0 {
            steps = max(1, Int(((top - base) / pitch).rounded(.up)))
        }
        return (0...steps).map { j in
            P3(x, y, base + (top - base) * Double(j) / Double(steps))
        }
    }

    /// Split each path at the region boundary, keeping the runs inside -- the
    /// rule the contours already follow, and for the same reason: keeping the
    /// survivors as one polyline welds a stroke's far side to its near side
    /// across ground the region removed.
    static func clip(_ paths: [[P3<WorldSpace>]], to region: Region)
        -> [[P3<WorldSpace>]] {
        guard case .inside = region else { return paths }
        var out: [[P3<WorldSpace>]] = []
        for path in paths {
            var run: [P3<WorldSpace>] = []
            for v in path {
                if region.contains(v.xy) {
                    run.append(v)
                } else {
                    if run.count >= 2 { out.append(run) }
                    run = []
                }
            }
            if run.count >= 2 { out.append(run) }
        }
        return out
    }

    static func replicate(_ paths: [[P3<WorldSpace>]], _ tiles: [Affine2])
        -> [[P3<WorldSpace>]] {
        tiles.flatMap { tile in paths.map { $0.map { tile($0) } } }
    }
}

// MARK: - cut faces

extension Surface {
    /// The crest of one wall: the surface's profile along the edge, sampled
    /// once per cell at the grid's nodes and capped, with a vertex wherever it
    /// reaches the cap between two nodes and, with a refiner, wherever it has
    /// a minimum between two nodes.
    ///
    /// Without a refiner the cap crossing is solved on the grid's heights
    /// linearly, as the rim is contoured (`kurven.hatch._cap_crossing`,
    /// operation for operation), so the crest meets the rim where the rim
    /// ends rather than a sample later, and everything else is the grid's.
    ///
    /// With one, the crest is f's own profile along the edge: the nodes at
    /// f's height, the cap crossing bisected on f along the edge (the rim,
    /// refined, ends on the same bisection: `capOutline`), a vertex at each
    /// minimum of |f| between two nodes (`withMinima`: the grid has no
    /// sample at a zero of f, so its chord across a pit bottoms out at the
    /// nearer node, a whole cell's rise above the floor the phase lines fan
    /// down to), and between any two of these the profile subdivided until
    /// no chord departs from f by more than the refiner's tolerance
    /// (`subdividedProfile`). The rings are placed on f, the fold lines are
    /// traced on f and the ink is judged against f, so the crest has to be
    /// f's too, or the rings end on a fold the crest parts from.
    func crest(of edge: PerimeterEdge, tiles: [Affine2],
               refine: ContourRefine?) -> [P3<WorldSpace>] {
        let n = max(edge.density, 2)
        let magnitudeOfF = refine?.magnitude
        var crest: [P3<WorldSpace>] = []
        crest.reserveCapacity(n)
        var previous: (p: P2<DomainSpace>, excess: Double)?
        // The nodes along the edge; with the function, where a banded cap
        // steps between two nodes, the step's two vertices as well, on
        // either side of the boundary at its own band's height -- the
        // boundary itself belongs to the band beyond it, so the near side
        // is a rounding step short of it.
        var stations: [P2<DomainSpace>] = (0..<n).map { Surface.point(on: edge, at: Double($0) / Double(n - 1)) }
        if magnitudeOfF != nil, case .realBands(let bands, _) = caps {
            var withSteps: [P2<DomainSpace>] = []
            for (a, b) in zip(stations, stations.dropFirst()) {
                withSteps.append(a)
                for band in bands {
                    let x = band.below
                    guard (a.x < x && b.x >= x) || (b.x < x && a.x >= x), b.x != a.x else { continue }
                    let t = (x - a.x) / (b.x - a.x)
                    let y = a.y + (b.y - a.y) * t
                    let near = P2<DomainSpace>(a.x < x ? x.nextDown : x, y), far = P2<DomainSpace>(a.x < x ? x : x.nextDown, y)
                    withSteps.append(near); withSteps.append(far)
                }
            }
            withSteps.append(stations[stations.count - 1])
            stations = withSteps
        }
        for p in stations {
            let u = magnitudeOfF.map { $0(inTile(p, tiles: tiles)) } ?? magnitude(at: p, tiles: tiles)
            let cap = caps.height(atX: p.x)
            let excess = u - cap
            if let a = previous, excess.isFinite, a.excess.isFinite, a.p.x != p.x || a.p.y != p.y {
                let here: (p: P2<DomainSpace>, excess: Double) = (p, excess)
                var under: (p: P2<DomainSpace>, excess: Double)?
                var over: (p: P2<DomainSpace>, excess: Double)?
                if a.excess <= 0 && excess > 0 { under = a; over = here }
                if excess <= 0 && a.excess > 0 { under = here; over = a }
                if let under, let over {
                    if let magnitudeOfF {
                        if let q = capCrossing(from: under.p, under.excess, to: over.p, over.excess,
                                               tiles: tiles, magnitude: magnitudeOfF) {
                            crest.append(P3(q.x, q.y, caps.height(atX: q.x)))
                        }
                    } else {
                        let t = under.excess / (under.excess - over.excess)
                        if t > 0, t < 1 {
                            let x = under.p.x + (over.p.x - under.p.x) * t
                            let y = under.p.y + (over.p.y - under.p.y) * t
                            crest.append(P3(x, y, caps.height(atX: x)))
                        }
                    }
                }
            }
            crest.append(P3(p.x, p.y, min(u, cap)))
            previous = (p, excess)
        }
        guard let refine, let magnitude = refine.magnitude else { return crest }
        return subdividedProfile(withMinima(crest, tiles: tiles, magnitude: magnitude),
                                 tiles: tiles, magnitude: magnitude, tolerance: refine.tolerance)
    }

    /// The grid's spacing, the larger of the two directions: what the
    /// refiner's tolerance is measured against.
    var cell: Double {
        max(abs(height.domain.real.length) / Double(max(height.width - 1, 1)),
            abs(height.domain.imag.length) / Double(max(height.height - 1, 1)))
    }

    /// Where |f| crosses the cap on the straight segment from `a` to `b`,
    /// given the excess over the cap at each end: bisection on f to the last
    /// bit, nil when both ends lie on one side. One routine for the crest
    /// and the rim's ends, so the two meet to rounding.
    func capCrossing(from a: P2<DomainSpace>, _ ea: Double, to b: P2<DomainSpace>, _ eb: Double,
                     tiles: [Affine2], magnitude: ContourRefine.Magnitude) -> P2<DomainSpace>? {
        guard ea.isFinite, eb.isFinite, (ea > 0) != (eb > 0) else { return nil }
        func excess(_ p: P2<DomainSpace>) -> Double {
            magnitude(inTile(p, tiles: tiles)) - caps.height(atX: p.x)
        }
        var lo = SIMD2(a.x, a.y), hi = SIMD2(b.x, b.y)
        var elo = ea
        for _ in 0..<60 {
            let m = 0.5 * (lo + hi)
            let em = excess(P2(m.x, m.y))
            guard em.isFinite else { return nil }
            if (em > 0) == (elo > 0) { lo = m; elo = em } else { hi = m }
        }
        let m = 0.5 * (lo + hi)
        return P2(m.x, m.y)
    }

    /// The crest with f's profile between its vertices: each chord's
    /// midpoint is taken onto the profile where the chord departs from it
    /// by more than `tolerance` cells, and the halves likewise, five deep.
    /// The crest is a function of the distance along its straight edge, so
    /// the vertices keep their order.
    func subdividedProfile(_ crest: [P3<WorldSpace>], tiles: [Affine2],
                           magnitude: ContourRefine.Magnitude, tolerance: Double) -> [P3<WorldSpace>] {
        guard crest.count >= 2 else { return crest }
        let cell = self.cell
        func profile(_ p: P2<DomainSpace>) -> Double {
            min(magnitude(inTile(p, tiles: tiles)), caps.height(atX: p.x))
        }
        func subdivide(_ a: P3<WorldSpace>, _ b: P3<WorldSpace>, depth: Int,
                       into out: inout [P3<WorldSpace>]) {
            guard depth < 5, a.z.isFinite, b.z.isFinite else { return }
            // The two vertices of a banded cap's step stand at one point.
            guard abs(a.x - b.x) + abs(a.y - b.y) > 1e-9 * cell else { return }
            let mid = P2<DomainSpace>(0.5 * (a.x + b.x), 0.5 * (a.y + b.y))
            let z = profile(mid)
            guard z.isFinite, abs(z - 0.5 * (a.z + b.z)) > tolerance * cell else { return }
            let v = P3<WorldSpace>(mid.x, mid.y, z)
            subdivide(a, v, depth: depth + 1, into: &out)
            out.append(v)
            subdivide(v, b, depth: depth + 1, into: &out)
        }
        var out: [P3<WorldSpace>] = [crest[0]]
        for (a, b) in zip(crest, crest.dropFirst()) {
            subdivide(a, b, depth: 0, into: &out)
            out.append(b)
        }
        return out
    }

    /// The crest with a vertex at every minimum of |f| that falls between two
    /// of its vertices. The edge is straight, so the crest is a function of
    /// the distance along it and the new vertices keep that order.
    func withMinima(_ crest: [P3<WorldSpace>], tiles: [Affine2],
                    magnitude: ContourRefine.Magnitude) -> [P3<WorldSpace>] {
        guard crest.count >= 3 else { return crest }
        // The edge lies in the domain: world x, y are domain x, y here.
        func dom(_ v: P3<WorldSpace>) -> P2<DomainSpace> { P2(v.x, v.y) }
        func profile(_ p: P2<DomainSpace>) -> Double {
            min(magnitude(inTile(p, tiles: tiles)), caps.height(atX: p.x))
        }
        func between(_ a: P2<DomainSpace>, _ b: P2<DomainSpace>, _ t: Double) -> P2<DomainSpace> {
            P2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
        }
        func apart(_ a: P2<DomainSpace>, _ b: P2<DomainSpace>) -> Double {
            ((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y)).squareRoot()
        }
        var out: [P3<WorldSpace>] = [crest[0]]
        for i in 1..<(crest.count - 1) {
            let v = crest[i]
            out.append(v)
            guard v.z.isFinite, v.z <= crest[i - 1].z, v.z <= crest[i + 1].z,
                  v.z < crest[i - 1].z || v.z < crest[i + 1].z else { continue }
            let a = dom(crest[i - 1]), b = dom(crest[i + 1])
            var lo = 0.0, hi = 1.0
            let phi = (5.0.squareRoot() - 1) / 2
            var c = hi - phi * (hi - lo), d = lo + phi * (hi - lo)
            var fc = profile(between(a, b, c)), fd = profile(between(a, b, d))
            for _ in 0..<90 {
                if fc < fd { hi = d; d = c; fd = fc; c = hi - phi * (hi - lo); fc = profile(between(a, b, c)) }
                else { lo = c; c = d; fc = fd; d = lo + phi * (hi - lo); fd = profile(between(a, b, d)) }
                if hi - lo < 1e-15 { break }
            }
            let q = between(a, b, (lo + hi) / 2)
            let z = profile(q)
            guard z.isFinite, z < v.z - 1e-12 * max(abs(v.z), 1),
                  apart(q, dom(v)) > 1e-12 * apart(a, b) else { continue }
            // Two nodes of equal height either side of a dip both lead here;
            // the vertex is added once.
            guard !out.suffix(2).contains(where: { apart(dom($0), q) <= 1e-9 * apart(a, b) })
            else { continue }
            let vertex = P3<WorldSpace>(q.x, q.y, z)
            // Before the node or after it, by where it lies along the edge.
            let before = (q.x - a.x) * (b.x - a.x) + (q.y - a.y) * (b.y - a.y)
                       < (v.x - a.x) * (b.x - a.x) + (v.y - a.y) * (b.y - a.y)
            if before { out.insert(vertex, at: out.count - 1) } else { out.append(vertex) }
        }
        out.append(crest[crest.count - 1])
        return out
    }

    /// The crest's height at a point of its edge, by linear interpolation
    /// between its vertices: what a hatch stroke rises to. Without a refiner
    /// that is the grid's own interpolation along the edge, which
    /// `height(at:tiles:)` computes bit for bit as `kurven.hatch` does and the
    /// fixtures hold it to; with one, the crest has vertices the grid does
    /// not, and the strokes stand under them.
    func crestHeight(of crest: [P3<WorldSpace>], at p: P2<DomainSpace>) -> Double? {
        guard let a0 = crest.first, let b0 = crest.last, crest.count >= 2 else { return nil }
        let dx = b0.x - a0.x, dy = b0.y - a0.y
        let length = dx * dx + dy * dy
        guard length > 0 else { return nil }
        func along(_ x: Double, _ y: Double) -> Double { ((x - a0.x) * dx + (y - a0.y) * dy) / length }
        let s = along(p.x, p.y)
        for (a, b) in zip(crest, crest.dropFirst()) {
            let sa = along(a.x, a.y), sb = along(b.x, b.y)
            guard sa <= s, s <= sb else { continue }
            let t = sb > sa ? (s - sa) / (sb - sa) : 0
            return a.z + (b.z - a.z) * t
        }
        return nil
    }

    func wallHatch(_ perimeter: BoundaryPerimeter, edges: [Int], spacing: Double,
                   pitch: Double, trim: Bool, base: Double, topOffset: Double,
                   tiles: [Affine2], refine: ContourRefine? = nil) -> [[P3<WorldSpace>]] {
        var out: [[P3<WorldSpace>]] = []
        for index in edges where index >= 0 && index < perimeter.edges.count {
            let edge = perimeter.edges[index]
            let length = simd_length(edge.end.v - edge.start.v)
            let n = Surface.strokeCount(length: length, spacing: spacing)
            let range = trim ? Array(1..<max(n - 1, 1)) : Array(0..<n)
            let crest = refine?.magnitude == nil ? nil : crest(of: edge, tiles: tiles, refine: refine)
            for k in range {
                let p = Surface.point(on: edge, at: Double(k) / Double(max(n - 1, 1)))
                let top = (crest.flatMap { crestHeight(of: $0, at: p) }
                           ?? height(at: p, tiles: tiles)) + topOffset
                if let stroke = Surface.vertical(x: p.x, y: p.y, base: base,
                                                 top: top, pitch: pitch) {
                    out.append(stroke)
                }
            }
        }
        return out
    }

    /// The crest and foot of every wall, and a post at every corner.
    func wallOutline(_ perimeter: BoundaryPerimeter, edges: [Int], pitch: Double,
                     base: Double, tiles: [Affine2],
                     refine: ContourRefine? = nil) -> [[P3<WorldSpace>]] {
        var out: [[P3<WorldSpace>]] = []
        var corners: [P2<WorldSpace>] = []
        for index in edges where index >= 0 && index < perimeter.edges.count {
            let edge = perimeter.edges[index]
            let n = max(edge.density, 2)
            out.append(crest(of: edge, tiles: tiles, refine: refine))
            out.append((0..<n).map { k in
                let p = Surface.point(on: edge, at: Double(k) / Double(n - 1))
                return P3(p.x, p.y, base)
            })
            // Deduplicated by exact coordinate, so a closed traversal stands one
            // post at each corner rather than two.
            for corner in [edge.start, edge.end] where !corners.contains(corner) {
                corners.append(corner)
            }
        }
        for corner in corners {
            let top = height(at: P2<DomainSpace>(corner.x, corner.y), tiles: tiles)
            if let post = Surface.vertical(x: corner.x, y: corner.y, base: base,
                                           top: top, pitch: pitch) {
                out.append(post)
            }
        }
        return out
    }
}

// MARK: - truncated tops

extension Surface {
    /// Where the cap changes across `[lo, hi]`, as the intervals it makes. The
    /// cap is constant on each `[a, b)`.
    func capIntervals(lo: Double, hi: Double) -> [(Double, Double)] {
        var edges = [lo]
        if case .realBands(let bands, _) = caps {
            for band in bands where band.below > lo && band.below < hi {
                edges.append(band.below)
            }
        }
        edges = Array(Set(edges)).sorted() + [hi]
        return zip(edges, edges.dropFirst()).filter { $1 > $0 }.map { ($0, $1) }
    }

    func capHatch(axis: KeepAxis, spacing: Double, refine: ContourRefine? = nil) -> [[P3<WorldSpace>]] {
        guard spacing.isFinite, spacing > 0 else { return [] }
        let grid = height
        let across = axis == .real ? grid.domain.imag : grid.domain.real
        let alongInterval = axis == .real ? grid.domain.real : grid.domain.imag
        let count = axis == .real ? grid.width : grid.height
        let samples = (0..<count).map { i in
            axis == .real ? grid.position(x: i, y: 0).x : grid.position(x: 0, y: i).y
        }

        var out: [[P3<WorldSpace>]] = []
        // Anchored to whole multiples of the spacing rather than to the domain,
        // so dragging the domain slides the landscape under a fixed ruling
        // instead of re-ruling it.
        let first = Int((across.lo / spacing).rounded(.up))
        let last = Int((across.hi / spacing).rounded(.down))
        guard first <= last else { return out }
        for k in first...last {
            let c = Double(k) * spacing
            guard c > across.lo, c < across.hi else { continue }
            // The cap varies with Re and with nothing else: a line along the
            // real axis crosses the bands and is solved piece by piece, and one
            // along the imaginary axis stays in a single band for its length.
            let intervals = axis == .real
                ? capIntervals(lo: alongInterval.lo, hi: alongInterval.hi)
                : [(alongInterval.lo, alongInterval.hi)]
            for (a, b) in intervals {
                let cap = caps.height(atX: axis == .real ? a : c)
                guard cap.isFinite else { continue }
                var coord = [a]
                coord.append(contentsOf: samples.filter { $0 > a && $0 < b })
                coord.append(b)
                // With the function at hand the strokes are ruled on f and
                // end where |f| reaches the cap along them, bisected: the
                // lattice's chords lie above f on a convex flank, so its cap
                // region reaches past the rim drawn on f by a fraction of a
                // cell, and a stroke ruled on it poked out past the rim.
                func excessAt(_ t: Double) -> Double {
                    let p = axis == .real ? P2<DomainSpace>(t, c) : P2<DomainSpace>(c, t)
                    if let f = refine?.magnitude {
                        let u = f(p)
                        return (u.isNaN ? .infinity : u) - cap
                    }
                    return magnitude(at: p) - cap
                }
                let excess = coord.map(excessAt)
                let runs = refine?.magnitude == nil
                    ? Surface.runsAtOrAbove(coord, excess)
                    : Surface.runsAtOrAbove(coord, excess) { lo, hi in Surface.bisected(lo, hi, excessAt) }
                for run in runs {
                    out.append(run.map { t in
                        axis == .real ? P3<WorldSpace>(t, c, cap)
                                      : P3<WorldSpace>(c, t, cap)
                    })
                }
            }
        }
        return out
    }

    /// The stretches where `excess >= 0`, with their ends solved linearly.
    ///
    /// `t = e0 / (e0 - e1)` is where the same linear interpolation puts the rim
    /// contour, so a stroke ends exactly on the rim rather than at the last
    /// sample inside it.
    static func runsAtOrAbove(_ coord: [Double], _ excess: [Double],
                              crossing: ((Double, Double) -> Double)? = nil) -> [[Double]] {
        var runs: [[Double]] = []
        var start: Int?
        for i in 0...coord.count {
            let inside = i < coord.count && excess[i] >= 0
            if inside, start == nil { start = i }
            if !inside, let a = start {
                var line: [Double] = []
                if a > 0 {
                    if let crossing {
                        line.append(crossing(coord[a - 1], coord[a]))
                    } else {
                        let (e0, e1) = (excess[a - 1], excess[a])
                        let t = e0 != e1 ? e0 / (e0 - e1) : 0
                        line.append(coord[a - 1] + t * (coord[a] - coord[a - 1]))
                    }
                }
                line.append(contentsOf: coord[a..<i])
                if i < coord.count {
                    if let crossing {
                        line.append(crossing(coord[i - 1], coord[i]))
                    } else {
                        let (e0, e1) = (excess[i - 1], excess[i])
                        let t = e0 != e1 ? e0 / (e0 - e1) : 0
                        line.append(coord[i - 1] + t * (coord[i] - coord[i - 1]))
                    }
                }
                if line.count >= 2 { runs.append(line) }
                start = nil
            }
        }
        return runs
    }

    /// The coordinate between `lo` and `hi` where `f` changes sign, bisected
    /// to the last bit; `lo` when it does not.
    static func bisected(_ lo: Double, _ hi: Double, _ f: (Double) -> Double) -> Double {
        var a = lo, b = hi
        var fa = f(a)
        guard (fa < 0) != (f(b) < 0) else { return lo }
        for _ in 0..<60 {
            let m = 0.5 * (a + b)
            let fm = f(m)
            if (fm < 0) == (fa < 0) { a = m; fa = fm } else { b = m }
        }
        return 0.5 * (a + b)
    }

    /// |f| - cap, as a grid, in float32 -- the same subtraction the Python side
    /// performs, on the same float32 samples, so the rim lands in the same
    /// place. The rim is this grid's zero contour.
    func capExcess() -> Grid2D<Float>? {
        let grid = height
        var values = grid.values
        var anyFinite = false
        for x in 0..<grid.width {
            let cap = Float(caps.height(atX: grid.position(x: x, y: 0).x))
            guard cap.isFinite else {
                for y in 0..<grid.height { values[y * grid.width + x] = -.infinity }
                continue
            }
            anyFinite = true
            for y in 0..<grid.height {
                values[y * grid.width + x] = grid[x, y] - cap
            }
        }
        guard anyFinite else { return nil }
        return Grid2D(width: grid.width, height: grid.height, domain: grid.domain,
                      values: values)
    }

    /// The rim of the cap: the zero level of the grid's excess over the cap,
    /// at the cap's height. With a refiner it is the contour |f| = cap
    /// placed on f, as the rings are, and an end of it on the domain's edge
    /// is re-solved by the crest's own bisection along that edge
    /// (`capCrossing`), so the crest meets the rim to rounding. A banded
    /// cap's rim is placed band by band at each band's own level
    /// (`bandedRim`), with the top of the wall where the cap steps.
    func capOutline(tiles: [Affine2] = [.identity], refine: ContourRefine? = nil) -> [[P3<WorldSpace>]] {
        guard let excess = capExcess() else { return [] }
        var out: [[P3<WorldSpace>]] = []
        for (_, lines) in Contour.levels(of: excess, [0.0]) {
            if let refine, let magnitude = refine.magnitude {
                switch caps {
                case .uniform(let cap) where cap.isFinite:
                    for line in refine.contours(.magnitude, cap, lines) {
                        var line = line
                        for i in Set([0, line.count - 1]) where line.count >= 2 {
                            if let q = onEdgeCapCrossing(near: line[i], tiles: tiles, magnitude: magnitude) {
                                line[i] = q
                            }
                        }
                        out.append(line.map { P3($0.x, $0.y, cap) })
                    }
                    continue
                case .realBands(let bands, let beyond):
                    out += bandedRim(lines, bands: bands, beyond: beyond, tiles: tiles,
                                     refine: refine, magnitude: magnitude)
                    continue
                default:
                    break
                }
            }
            for line in lines {
                out.append(line.map { P3($0.x, $0.y, caps.height(atX: $0.x)) })
            }
        }
        return out
    }

    /// The rim of a banded cap, on f. Within a band the rim is the contour
    /// |f| = that band's cap: the grid's rim lines are split by band, each
    /// run placed by the refiner at its own level, and a run's end beside a
    /// band's boundary is solved onto the boundary, where |f| reaches the
    /// cap along it. Where the cap steps the surface has a wall, and its
    /// top is the rim there: along the boundary, at the lesser cap where
    /// |f| is below it and at f's own height up to the greater cap, from
    /// one end of the lesser band's rim to the other. The grid's vertices
    /// in the cells that straddle a boundary are interpolated between two
    /// caps and mean nothing; they are dropped, and the boundary's own
    /// pieces take their place.
    func bandedRim(_ lines: [[P2<DomainSpace>]], bands: [RealBand], beyond: Double, tiles: [Affine2],
                   refine: ContourRefine, magnitude: ContourRefine.Magnitude) -> [[P3<WorldSpace>]] {
        let boundaries = bands.map(\.below).sorted()
        let g = height
        let dx = abs(g.domain.real.length) / Double(max(g.width - 1, 1))
        let dy = abs(g.domain.imag.length) / Double(max(g.height - 1, 1))
        let cell = self.cell
        let tolerance = refine.tolerance * cell
        func mag(_ p: P2<DomainSpace>) -> Double { magnitude(inTile(p, tiles: tiles)) }
        /// The boundary a point is beside, within a column either way.
        func boundary(beside x: Double) -> Double? {
            boundaries.first { abs(x - $0) <= 1.0001 * dx }
        }
        /// The point on the boundary x = b where |f| = cap, nearest in y to
        /// `near`: bracketed by walking out from it, then bisected.
        func ontoBoundary(_ b: Double, cap: Double, near: P2<DomainSpace>) -> P2<DomainSpace>? {
            func excess(_ y: Double) -> Double { mag(P2(b, y)) - cap }
            let step = dy / 8
            var lo = near.y, hi = near.y
            var elo = excess(lo), ehi = elo
            guard elo.isFinite else { return nil }
            var found = false
            for k in 1...24 {
                let up = near.y + Double(k) * step, down = near.y - Double(k) * step
                let eUp = excess(up), eDown = excess(down)
                if eUp.isFinite, (eUp > 0) != (elo > 0) { lo = near.y + Double(k - 1) * step; elo = excess(lo); hi = up; ehi = eUp; found = true; break }
                if eDown.isFinite, (eDown > 0) != (elo > 0) { hi = near.y - Double(k - 1) * step; ehi = excess(hi); lo = down; elo = eDown; found = true; break }
            }
            guard found, (elo > 0) != (ehi > 0) else { return nil }
            for _ in 0..<60 {
                let m = 0.5 * (lo + hi)
                let em = excess(m)
                if (em > 0) == (elo > 0) { lo = m; elo = em } else { hi = m; ehi = em }
            }
            return P2(b, 0.5 * (lo + hi))
        }

        var out: [[P3<WorldSpace>]] = []
        /// The ends of refined runs that lie on a boundary, by boundary:
        /// the cap they belong to and the point.
        var endsOnBoundary: [Double: [(cap: Double, point: P2<DomainSpace>)]] = [:]
        for line in lines {
            // Runs of one band's vertices, the straddling cells' dropped.
            var runs: [(cap: Double, points: [P2<DomainSpace>])] = []
            var broken = true
            for v in line {
                if boundary(beside: v.x) != nil { broken = true; continue }
                let c = caps.height(atX: v.x)
                if !broken, let last = runs.last, last.cap == c {
                    runs[runs.count - 1].points.append(v)
                } else {
                    runs.append((c, [v]))
                }
                broken = false
            }
            // A closed line whose first and last runs are the same band and
            // unbroken between is one run.
            if line.first == line.last, runs.count >= 2, runs[0].cap == runs[runs.count - 1].cap,
               boundary(beside: line[0].x) == nil {
                let last = runs.removeLast()
                runs[0].points = last.points.dropLast() + runs[0].points
            }
            for run in runs where run.points.count >= 2 && run.cap.isFinite {
                for piece in refine.contours(.magnitude, run.cap, [run.points]) where piece.count >= 2 {
                    var piece = piece
                    for i in [0, piece.count - 1] {
                        if let q = onEdgeCapCrossing(near: piece[i], tiles: tiles, magnitude: magnitude) {
                            piece[i] = q
                        } else if let b = boundary(beside: piece[i].x) ?? boundaries.first(where: { abs(piece[i].x - $0) <= 2 * dx }),
                                  let q = ontoBoundary(b, cap: run.cap, near: piece[i]) {
                            if i == 0 { piece.insert(q, at: 0) } else { piece.append(q) }
                            endsOnBoundary[b, default: []].append((run.cap, q))
                        }
                    }
                    out.append(piece.map { P3($0.x, $0.y, run.cap) })
                }
            }
        }
        // The top of each step wall: between the two ends of the lesser
        // band's rim on the boundary, nearest each other in y, at f's
        // height capped by the greater band.
        for (b, ends) in endsOnBoundary {
            let left = caps.height(atX: b.nextDown), right = caps.height(atX: b)
            let lesser = min(left, right), greater = max(left, right)
            guard lesser.isFinite, lesser < greater else { continue }
            let lows = ends.filter { $0.cap == lesser }.map(\.point).sorted { $0.y < $1.y }
            var k = 0
            while k + 1 < lows.count {
                let a = lows[k], c = lows[k + 1]
                k += 2
                func top(_ y: Double) -> Double { min(mag(P2(b, y)), greater) }
                var curve: [P3<WorldSpace>] = [P3(b, a.y, top(a.y))]
                func subdivide(_ y0: Double, _ z0: Double, _ y1: Double, _ z1: Double, depth: Int) {
                    guard depth < 7, abs(y1 - y0) > 1e-9 * cell else { return }
                    let ym = 0.5 * (y0 + y1), zm = top(ym)
                    guard zm.isFinite, abs(zm - 0.5 * (z0 + z1)) > tolerance else { return }
                    subdivide(y0, z0, ym, zm, depth: depth + 1)
                    curve.append(P3(b, ym, zm))
                    subdivide(ym, zm, y1, z1, depth: depth + 1)
                }
                subdivide(a.y, top(a.y), c.y, top(c.y), depth: 0)
                curve.append(P3(b, c.y, top(c.y)))
                out.append(curve)
            }
        }
        return out
    }

    /// For a rim vertex on the domain's edge, the cap crossing between the
    /// two grid nodes beside it on that edge, by the crest's bisection; nil
    /// for a vertex off the edge or with no crossing beside it.
    func onEdgeCapCrossing(near p: P2<DomainSpace>, tiles: [Affine2],
                           magnitude: ContourRefine.Magnitude) -> P2<DomainSpace>? {
        let g = height
        let eps = 1e-9 * cell
        let f = g.index(of: p)
        let onRow = abs(p.y - g.domain.imag.lo) <= eps || abs(p.y - g.domain.imag.hi) <= eps
        let onColumn = abs(p.x - g.domain.real.lo) <= eps || abs(p.x - g.domain.real.hi) <= eps
        guard onRow || onColumn else { return nil }
        func excess(_ q: P2<DomainSpace>) -> Double {
            magnitude(inTile(q, tiles: tiles)) - caps.height(atX: q.x)
        }
        let a: P2<DomainSpace>, b: P2<DomainSpace>
        if onRow {
            let j = abs(p.y - g.domain.imag.lo) <= eps ? 0 : g.height - 1
            let i = min(max(Int(f.x.rounded(.down)), 0), g.width - 2)
            a = g.position(x: i, y: j); b = g.position(x: i + 1, y: j)
        } else {
            let i = abs(p.x - g.domain.real.lo) <= eps ? 0 : g.width - 1
            let j = min(max(Int(f.y.rounded(.down)), 0), g.height - 2)
            a = g.position(x: i, y: j); b = g.position(x: i, y: j + 1)
        }
        return capCrossing(from: a, excess(a), to: b, excess(b), tiles: tiles, magnitude: magnitude)
    }
}
