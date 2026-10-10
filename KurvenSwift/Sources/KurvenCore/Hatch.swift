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
            paths = capHatch(axis: axis, spacing: spacing)
            if tiled { paths = Surface.replicate(paths, context.tiles) }
        case .capOutline(let tiled):
            paths = capOutline()
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
    /// The cap crossing is solved on the grid's heights linearly, as the rim
    /// is contoured (`kurven.hatch._cap_crossing`, operation for operation),
    /// so the crest meets the rim where the rim ends rather than a sample
    /// later. The minimum is solved on f: the grid has no sample at a zero of
    /// f, so its chord across a pit bottoms out at the nearer node, a whole
    /// cell's rise above the floor the phase lines fan down to -- on a plate's
    /// narrowest pit a quarter of the cap's height. A node lower than both
    /// its neighbours has the profile's minimum somewhere between them;
    /// golden-section search on |f| over that span finds it to rounding, and
    /// where it is not the node itself it is a vertex, at a zero on the edge
    /// the floor. Everything else stays the grid's, so the crest coincides
    /// with the rim, the fold lines and the lifted contours as before.
    func crest(of edge: PerimeterEdge, tiles: [Affine2],
               refine: ContourRefine?) -> [P3<WorldSpace>] {
        let n = max(edge.density, 2)
        var crest: [P3<WorldSpace>] = []
        crest.reserveCapacity(n)
        var previous: (p: P2<DomainSpace>, excess: Double)?
        for k in 0..<n {
            let p = Surface.point(on: edge, at: Double(k) / Double(n - 1))
            let u = magnitude(at: p, tiles: tiles)
            let cap = caps.height(atX: p.x)
            let excess = u - cap
            if let a = previous, excess.isFinite, a.excess.isFinite {
                let here: (p: P2<DomainSpace>, excess: Double) = (p, excess)
                var under: (p: P2<DomainSpace>, excess: Double)?
                var over: (p: P2<DomainSpace>, excess: Double)?
                if a.excess <= 0 && excess > 0 { under = a; over = here }
                if excess <= 0 && a.excess > 0 { under = here; over = a }
                if let under, let over {
                    let t = under.excess / (under.excess - over.excess)
                    if t > 0, t < 1 {
                        let x = under.p.x + (over.p.x - under.p.x) * t
                        let y = under.p.y + (over.p.y - under.p.y) * t
                        crest.append(P3(x, y, caps.height(atX: x)))
                    }
                }
            }
            crest.append(P3(p.x, p.y, min(u, cap)))
            previous = (p, excess)
        }
        guard let magnitude = refine?.magnitude else { return crest }
        return withMinima(crest, tiles: tiles, magnitude: magnitude)
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

    func capHatch(axis: KeepAxis, spacing: Double) -> [[P3<WorldSpace>]] {
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
                let excess = coord.map { t -> Double in
                    let p = axis == .real ? P2<DomainSpace>(t, c) : P2<DomainSpace>(c, t)
                    return magnitude(at: p) - cap
                }
                for run in Surface.runsAtOrAbove(coord, excess) {
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
    static func runsAtOrAbove(_ coord: [Double], _ excess: [Double]) -> [[Double]] {
        var runs: [[Double]] = []
        var start: Int?
        for i in 0...coord.count {
            let inside = i < coord.count && excess[i] >= 0
            if inside, start == nil { start = i }
            if !inside, let a = start {
                var line: [Double] = []
                if a > 0 {
                    let (e0, e1) = (excess[a - 1], excess[a])
                    let t = e0 != e1 ? e0 / (e0 - e1) : 0
                    line.append(coord[a - 1] + t * (coord[a] - coord[a - 1]))
                }
                line.append(contentsOf: coord[a..<i])
                if i < coord.count {
                    let (e0, e1) = (excess[i - 1], excess[i])
                    let t = e0 != e1 ? e0 / (e0 - e1) : 0
                    line.append(coord[i - 1] + t * (coord[i] - coord[i - 1]))
                }
                if line.count >= 2 { runs.append(line) }
                start = nil
            }
        }
        return runs
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

    func capOutline() -> [[P3<WorldSpace>]] {
        guard let excess = capExcess() else { return [] }
        var out: [[P3<WorldSpace>]] = []
        for (_, lines) in Contour.levels(of: excess, [0.0]) {
            for line in lines {
                out.append(line.map { P3($0.x, $0.y, caps.height(atX: $0.x)) })
            }
        }
        return out
    }
}
