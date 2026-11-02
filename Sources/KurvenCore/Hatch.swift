import Foundation
import simd

/// Deriving the hatching: the curves that shade a wall and a plateau.
///
/// `kurven/hatch.py` is the definition of all four kinds and this is the
/// consumer's implementation of it; `tests/fixtures/hatch` holds the two to each
/// other on the same grid, the same perimeter and the same truncations. Where they
/// necessarily differ is the *height*: Python has f and evaluates it, this has
/// the grid and interpolates it, so a stroke's top sits at the grid's top edge
/// rather than the function's. That difference is the grid's interpolation
/// error, half a cell's rise at most -- and a whole cell's rise where a zero of
/// f lies between two samples, since the grid's chord over a pit bottoms out at
/// the nearer sample. So with a refiner in play the top edge gains a vertex at
/// every minimum of |f| between two nodes, solved on f, and the hatch stands
/// under it (`topEdge(of:tiles:refine:)`). The fixtures compare the grid's
/// answer, with no refiner.
///
/// Why this is here at all, rather than a list of strokes in the bundle: the truncation
/// is a slider. Moving it moves every plateau, which moves every plateau hatch line and
/// every truncation edge and re-cuts every wall stroke at the new top edge. Shipping the answer
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

    /// Any described layer's curves: a contour family, or a hatching.
    ///
    /// The one place that knows which kinds of description exist, so a reader,
    /// a re-derivation after an edit and a bake all produce the same strokes
    /// for the same spec.
    func curves(_ spec: LayerSpec, occluder: Occluder,
                refine: ContourRefine? = nil) -> PolylineSet<WorldSpace> {
        if let hatched = hatch(spec.source, in: HatchContext(occluder), refine: refine) {
            return hatched
        }
        return derive(spec.source, policy: spec.heightPolicy,
                      region: occluder.region, tiles: occluder.tiles, refine: refine)
    }

    /// The heightfield's hatchings and the edges of its walls and plateaus,
    /// as world-space polylines. `nil` for a source this does not own, so a
    /// caller can use it as the test as well as the derivation. With a
    /// refiner, the walls take their heights from f.
    func hatch(_ source: LayerSource, in context: HatchContext,
               refine: ContourRefine? = nil) -> PolylineSet<WorldSpace>? {
        var paths: [[P3<WorldSpace>]]
        switch source {
        case .file, .contour, .coordinateCurves, .linearCurves, .foldCurves, .trajectory, .edges,
             .hatching(_, .solid):
            return nil
        case .hatching(let hatching, .walls(let edges, let trim, let base, let top)):
            guard let perimeter = context.perimeter else { return .empty }
            paths = wallHatchLines(perimeter, edges: edges, spacing: hatching.spacing,
                                   step: hatching.step, trim: trim, base: base, topOffset: top,
                                   tiles: context.tiles, refine: refine)
        case .hatching(let hatching, .plateaus(let tiled)):
            // Decoding holds the angle to an axis; one built in code at any
            // other draws nothing.
            guard let axis = DomainAxis(angle: hatching.angle) else { return .empty }
            paths = plateauHatchLines(axis: axis, spacing: hatching.spacing, refine: refine)
            if tiled { paths = Surface.replicate(paths, context.tiles) }
        case .wallEdges(let edges, let step, let base):
            guard let perimeter = context.perimeter else { return .empty }
            paths = wallEdges(perimeter, edges: edges, step: step, base: base,
                              tiles: context.tiles, refine: refine)
        case .truncationEdge(let tiled):
            paths = truncationEdge(tiles: context.tiles, refine: refine)
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
        point(from: edge.start, to: edge.end, at: t)
    }

    /// The foot `t` of the way along a wall from `a` to `b`, as
    /// `a(1 - t) + bt`, but with a vertical edge's x its own: interpolated,
    /// a(1 - t) + at lands a rounding either side of a, and a wall standing
    /// on a band boundary would read the band on whichever side its foot
    /// fell rather than its footprint's.
    static func point(from a: P2<WorldSpace>, to b: P2<WorldSpace>, at t: Double) -> P2<DomainSpace> {
        P2(a.x == b.x ? a.x : a.x * (1 - t) + b.x * t, a.y * (1 - t) + b.y * t)
    }

    /// One vertical stroke, subdivided every `step`.
    ///
    /// Not a two-point segment. The bake clips per vertex, so a stroke whose
    /// middle is behind a ridge and whose ends are not would be drawn straight
    /// through it.
    static func vertical(x: Double, y: Double, base: Double, top: Double,
                         step: Double) -> [P3<WorldSpace>]? {
        guard top > base else { return nil }
        var steps = 1
        if step.isFinite, step > 0 {
            steps = max(1, Int(((top - base) / step).rounded(.up)))
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
    /// The top edge of one wall: the surface's profile along the edge, sampled
    /// once per cell at the grid's nodes and truncated, with a vertex wherever it
    /// reaches the truncation between two nodes and, with a refiner, wherever it has
    /// a minimum between two nodes.
    ///
    /// Without a refiner the truncation crossing is solved on the grid's heights
    /// linearly, as the truncation edge is contoured (as the Python original
    /// solved it, operation for operation), so the top edge meets the truncation edge where the truncation edge
    /// ends rather than a sample later, and everything else is the grid's.
    ///
    /// With one, the top edge is f's own profile along the edge: the nodes at
    /// f's height, the truncation crossing bisected on f along the edge (the truncation edge,
    /// refined, ends on the same bisection: `truncationEdge`), a vertex at each
    /// minimum of |f| between two nodes (`withMinima`: the grid has no
    /// sample at a zero of f, so its chord across a pit bottoms out at the
    /// nearer node, a whole cell's rise above the floor the phase lines fan
    /// down to), and between any two of these the profile subdivided until
    /// no chord departs from f by more than the refiner's tolerance
    /// (`subdividedProfile`). The rings are placed on f, the fold curves are
    /// traced on f and the curves are tested against f, so the top edge has to be
    /// f's too, or the rings end on a fold the top edge parts from.
    ///
    /// The top edge reads the truncation from the footprint's side, `inward`
    /// the edge's normal into it (`Surface.footprintSide`): a wall standing
    /// on a band boundary reads the band its footprint is in, all along, and
    /// a wall ending on one reads its own band at the end.
    func topEdge(of edge: PerimeterEdge, inward: SIMD2<Double>, tiles: [Affine2],
                 refine: ContourRefine?) -> [P3<WorldSpace>] {
        let n = max(edge.density, 2)
        let magnitudeOfF = refine?.magnitude
        let along = edge.end.v - edge.start.v
        // The side the truncation is read from at a station `t` along the
        // edge, nil for one between the ends; at world x, as the top edge
        // reads the truncation.
        func side(_ t: Double?) -> Truncation.Side {
            Surface.footprintSide(at: t ?? 0.5, along: along, inward: inward, back: .identity)
        }
        let between = side(nil)
        var topEdge: [P3<WorldSpace>] = []
        topEdge.reserveCapacity(n)
        var previous: (p: P2<DomainSpace>, excess: Double)?
        // The nodes along the edge; with the function, where the edge meets a
        // riser between two nodes, two vertices there as well, one either
        // side of the boundary at its own band's height -- the boundary
        // itself belongs to the band beyond it, so the near side is a
        // rounding step short of it.
        var stations: [(p: P2<DomainSpace>, t: Double?)] = (0..<n).map {
            let t = Double($0) / Double(n - 1)
            return (Surface.point(on: edge, at: t), t)
        }
        if magnitudeOfF != nil, case .realBands(let bands, _) = truncation {
            // The boundaries inside the domain: a band that starts on its
            // right edge starts beyond it.
            let real = domain.real
            let boundaries = bands.map(\.below).filter { $0 > min(real.lo, real.hi) && $0 < max(real.lo, real.hi) }
            var withRisers: [(p: P2<DomainSpace>, t: Double?)] = []
            for (a, b) in zip(stations, stations.dropFirst()) {
                withRisers.append(a)
                for x in boundaries {
                    let (a, b) = (a.p, b.p)
                    guard (a.x < x && b.x >= x) || (b.x < x && a.x >= x), b.x != a.x else { continue }
                    let t = (x - a.x) / (b.x - a.x)
                    let y = a.y + (b.y - a.y) * t
                    let near = P2<DomainSpace>(a.x < x ? x.nextDown : x, y), far = P2<DomainSpace>(a.x < x ? x : x.nextDown, y)
                    withRisers.append((near, nil)); withRisers.append((far, nil))
                }
            }
            withRisers.append(stations[stations.count - 1])
            stations = withRisers
        }
        for (p, t) in stations {
            let u = magnitudeOfF.map { $0(inTile(p, tiles: tiles)) } ?? magnitude(at: p, tiles: tiles)
            let truncationHeight = self.truncationHeight(atX: p.x, from: side(t))
            let excess = u - truncationHeight
            if let a = previous, excess.isFinite, a.excess.isFinite, a.p.x != p.x || a.p.y != p.y {
                let here: (p: P2<DomainSpace>, excess: Double) = (p, excess)
                var under: (p: P2<DomainSpace>, excess: Double)?
                var over: (p: P2<DomainSpace>, excess: Double)?
                if a.excess <= 0 && excess > 0 { under = a; over = here }
                if excess <= 0 && a.excess > 0 { under = here; over = a }
                if let under, let over {
                    if let magnitudeOfF {
                        if let q = truncationCrossing(from: under.p, under.excess, to: over.p, over.excess,
                                                      tiles: tiles, magnitude: magnitudeOfF, from: between) {
                            topEdge.append(P3(q.x, q.y, self.truncationHeight(atX: q.x, from: between)))
                        }
                    } else {
                        let t = under.excess / (under.excess - over.excess)
                        if t > 0, t < 1 {
                            let x = under.p.x + (over.p.x - under.p.x) * t
                            let y = under.p.y + (over.p.y - under.p.y) * t
                            topEdge.append(P3(x, y, self.truncationHeight(atX: x, from: between)))
                        }
                    }
                }
            }
            topEdge.append(P3(p.x, p.y, min(u, truncationHeight)))
            previous = (p, excess)
        }
        guard let refine, let magnitude = refine.magnitude else { return topEdge }
        return subdividedProfile(withMinima(topEdge, tiles: tiles, magnitude: magnitude, from: between),
                                 tiles: tiles, magnitude: magnitude, tolerance: refine.tolerance, from: between)
    }

    /// The grid's spacing, the larger of the two directions: what the
    /// refiner's tolerance is measured against.
    var cell: Double {
        max(abs(height.domain.real.length) / Double(max(height.width - 1, 1)),
            abs(height.domain.imag.length) / Double(max(height.height - 1, 1)))
    }

    /// How near the domain's edge a truncation edge's end is taken to lie
    /// on it: a billionth of a cell.
    var edgeTolerance: Double { 1e-9 * cell }

    /// Where |f| crosses the truncation on the straight segment from `a` to `b`,
    /// given the excess over the truncation at each end: bisection on f to the last
    /// bit, nil when both ends lie on one side. One routine for the top edge
    /// and the truncation edge's ends, so the two meet to rounding. On a band
    /// boundary the truncation is read from `side`.
    func truncationCrossing(from a: P2<DomainSpace>, _ ea: Double, to b: P2<DomainSpace>, _ eb: Double,
                            tiles: [Affine2], magnitude: ContourRefine.Magnitude,
                            from side: Truncation.Side = .right) -> P2<DomainSpace>? {
        guard ea.isFinite, eb.isFinite, (ea > 0) != (eb > 0) else { return nil }
        func excess(_ p: P2<DomainSpace>) -> Double {
            magnitude(inTile(p, tiles: tiles)) - truncationHeight(atX: p.x, from: side)
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

    /// The top edge with f's profile between its vertices: each chord's
    /// midpoint is taken onto the profile where the chord departs from it
    /// by more than `tolerance` cells, and the halves likewise, five deep.
    /// The top edge is a function of the distance along its straight edge, so
    /// the vertices keep their order. On a band boundary the truncation is
    /// read from `side`.
    func subdividedProfile(_ topEdge: [P3<WorldSpace>], tiles: [Affine2],
                           magnitude: ContourRefine.Magnitude, tolerance: Double,
                           from side: Truncation.Side = .right) -> [P3<WorldSpace>] {
        guard topEdge.count >= 2 else { return topEdge }
        let cell = self.cell
        func profile(_ p: P2<DomainSpace>) -> Double {
            min(magnitude(inTile(p, tiles: tiles)), truncationHeight(atX: p.x, from: side))
        }
        func subdivide(_ a: P3<WorldSpace>, _ b: P3<WorldSpace>, depth: Int,
                       into out: inout [P3<WorldSpace>]) {
            guard depth < 5, a.z.isFinite, b.z.isFinite else { return }
            // The two vertices either side of a riser stand at one point.
            guard abs(a.x - b.x) + abs(a.y - b.y) > 1e-9 * cell else { return }
            let mid = P2<DomainSpace>(0.5 * (a.x + b.x), 0.5 * (a.y + b.y))
            let z = profile(mid)
            guard z.isFinite, abs(z - 0.5 * (a.z + b.z)) > tolerance * cell else { return }
            let v = P3<WorldSpace>(mid.x, mid.y, z)
            subdivide(a, v, depth: depth + 1, into: &out)
            out.append(v)
            subdivide(v, b, depth: depth + 1, into: &out)
        }
        var out: [P3<WorldSpace>] = [topEdge[0]]
        for (a, b) in zip(topEdge, topEdge.dropFirst()) {
            subdivide(a, b, depth: 0, into: &out)
            out.append(b)
        }
        return out
    }

    /// The top edge with a vertex at every minimum of |f| that falls between two
    /// of its vertices. The edge is straight, so the top edge is a function of
    /// the distance along it and the new vertices keep that order. On a band
    /// boundary the truncation is read from `side`.
    func withMinima(_ topEdge: [P3<WorldSpace>], tiles: [Affine2],
                    magnitude: ContourRefine.Magnitude, from side: Truncation.Side = .right) -> [P3<WorldSpace>] {
        guard topEdge.count >= 3 else { return topEdge }
        // The edge lies in the domain: world x, y are domain x, y here.
        func dom(_ v: P3<WorldSpace>) -> P2<DomainSpace> { P2(v.x, v.y) }
        func profile(_ p: P2<DomainSpace>) -> Double {
            min(magnitude(inTile(p, tiles: tiles)), truncationHeight(atX: p.x, from: side))
        }
        func between(_ a: P2<DomainSpace>, _ b: P2<DomainSpace>, _ t: Double) -> P2<DomainSpace> {
            P2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
        }
        func apart(_ a: P2<DomainSpace>, _ b: P2<DomainSpace>) -> Double {
            ((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y)).squareRoot()
        }
        var out: [P3<WorldSpace>] = [topEdge[0]]
        for i in 1..<(topEdge.count - 1) {
            let v = topEdge[i]
            out.append(v)
            guard v.z.isFinite, v.z <= topEdge[i - 1].z, v.z <= topEdge[i + 1].z,
                  v.z < topEdge[i - 1].z || v.z < topEdge[i + 1].z else { continue }
            let a = dom(topEdge[i - 1]), b = dom(topEdge[i + 1])
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
        out.append(topEdge[topEdge.count - 1])
        return out
    }

    /// The top edge's height at a point of its edge, by linear interpolation
    /// between its vertices: what a hatch stroke rises to. Without a refiner
    /// that is the grid's own interpolation along the edge, which
    /// `height(at:tiles:)` computes bit for bit as `kurven.hatch` does and the
    /// fixtures hold it to; with one, the top edge has vertices the grid does
    /// not, and the strokes stand under them.
    func topEdgeHeight(of topEdge: [P3<WorldSpace>], at p: P2<DomainSpace>) -> Double? {
        guard let a0 = topEdge.first, let b0 = topEdge.last, topEdge.count >= 2 else { return nil }
        let dx = b0.x - a0.x, dy = b0.y - a0.y
        let length = dx * dx + dy * dy
        guard length > 0 else { return nil }
        func along(_ x: Double, _ y: Double) -> Double { ((x - a0.x) * dx + (y - a0.y) * dy) / length }
        let s = along(p.x, p.y)
        for (a, b) in zip(topEdge, topEdge.dropFirst()) {
            let sa = along(a.x, a.y), sb = along(b.x, b.y)
            guard sa <= s, s <= sb else { continue }
            let t = sb > sa ? (s - sa) / (sb - sa) : 0
            return a.z + (b.z - a.z) * t
        }
        return nil
    }

    func wallHatchLines(_ perimeter: BoundaryPerimeter, edges: [Int], spacing: Double,
                        step: Double, trim: Bool, base: Double, topOffset: Double,
                        tiles: [Affine2], refine: ContourRefine? = nil) -> [[P3<WorldSpace>]] {
        var out: [[P3<WorldSpace>]] = []
        for index in edges where index >= 0 && index < perimeter.edges.count {
            let edge = perimeter.edges[index], inward = perimeter.inward(index)
            let length = simd_length(edge.end.v - edge.start.v)
            let n = Surface.strokeCount(length: length, spacing: spacing)
            let range = trim ? Array(1..<max(n - 1, 1)) : Array(0..<n)
            let topEdge = refine?.magnitude == nil ? nil : topEdge(of: edge, inward: inward, tiles: tiles, refine: refine)
            for k in range {
                let t = Double(k) / Double(max(n - 1, 1))
                let p = Surface.point(on: edge, at: t)
                let top = (topEdge.flatMap { topEdgeHeight(of: $0, at: p) }
                           ?? wallHeight(at: p, t: t, from: edge.start, to: edge.end, inward: inward, tiles: tiles))
                    + topOffset
                if let stroke = Surface.vertical(x: p.x, y: p.y, base: base,
                                                 top: top, step: step) {
                    out.append(stroke)
                }
            }
        }
        return out
    }

    /// The top and bottom edges of every wall, and a vertical edge at every corner.
    ///
    /// A corner's vertical edge is where two walls meet, up to the lower of
    /// their tops. They differ only on a band boundary, where one wall
    /// stands on it, reading its footprint's band, and the other leaves it
    /// in its own band, and a riser ends at the corner, in the plane of the
    /// wall on the boundary: the edge rises to the top of the wall that
    /// leaves, past the other's between it and the riser, or short of it,
    /// where the riser and the wall on the boundary are one plane.
    func wallEdges(_ perimeter: BoundaryPerimeter, edges: [Int], step: Double,
                   base: Double, tiles: [Affine2],
                   refine: ContourRefine? = nil) -> [[P3<WorldSpace>]] {
        var out: [[P3<WorldSpace>]] = []
        var corners: [P2<WorldSpace>] = []
        for index in edges where index >= 0 && index < perimeter.edges.count {
            let edge = perimeter.edges[index]
            let n = max(edge.density, 2)
            out.append(topEdge(of: edge, inward: perimeter.inward(index), tiles: tiles, refine: refine))
            out.append((0..<n).map { k in
                let p = Surface.point(on: edge, at: Double(k) / Double(n - 1))
                return P3(p.x, p.y, base)
            })
            // Deduplicated by exact coordinate, so a closed traversal stands one
            // vertical edge at each corner rather than two.
            for corner in [edge.start, edge.end] where !corners.contains(corner) {
                corners.append(corner)
            }
        }
        for corner in corners {
            // Every wall ending at the corner, and its top there; those that
            // leave the line x = corner.x, where any does.
            let tops = perimeter.edges.indices.flatMap { k -> [(top: Double, leaves: Bool)] in
                let e = perimeter.edges[k]
                return [(e.start, 0.0), (e.end, 1.0)].filter { $0.0 == corner }.map { _, t in
                    (wallHeight(at: P2(corner.x, corner.y), t: t, from: e.start, to: e.end,
                                inward: perimeter.inward(k), tiles: tiles), e.start.x != e.end.x)
                }
            }
            let leaving = tops.filter(\.leaves).map(\.top)
            let top = (leaving.count == 1 ? leaving : tops.map(\.top)).max()
                ?? height(at: P2<DomainSpace>(corner.x, corner.y), tiles: tiles)
            if let verticalEdge = Surface.vertical(x: corner.x, y: corner.y, base: base,
                                                   top: top, step: step) {
                out.append(verticalEdge)
            }
        }
        return out
    }
}

// MARK: - plateaus

extension Surface {
    func plateauHatchLines(axis: DomainAxis, spacing: Double,
                           refine: ContourRefine? = nil) -> [[P3<WorldSpace>]] {
        guard spacing.isFinite, spacing > 0 else { return [] }
        let grid = height
        let across = axis == .real ? grid.domain.imag : grid.domain.real
        let alongInterval = axis == .real ? grid.domain.real : grid.domain.imag
        let count = axis == .real ? grid.width : grid.height
        let samples = (0..<count).map { i in
            axis == .real ? grid.position(x: i, y: 0).x : grid.position(x: 0, y: i).y
        }

        var out: [[P3<WorldSpace>]] = []
        // Divided across the domain the way `wallHatchLines` divides an edge: the
        // spacing asks for a count, and the strokes are spread to fit it. The
        // hatching then meets the plateau's edges with whole gaps, and each stroke
        // stands where the walls that run across it stand theirs, so it turns
        // the corner down the wall instead of missing it.
        let n = Surface.strokeCount(length: across.hi - across.lo, spacing: spacing)
        for k in 1..<max(n - 1, 1) {
            let c = across.lo + (across.hi - across.lo) * Double(k) / Double(n - 1)
            // The truncation varies with Re and with nothing else: a line along the
            // real axis crosses the bands and is solved piece by piece, and one
            // along the imaginary axis stays in a single band for its length.
            let intervals = axis == .real
                ? truncation.bands(from: alongInterval.lo, to: alongInterval.hi).map { ($0.lo, $0.hi) }
                : [(alongInterval.lo, alongInterval.hi)]
            for (a, b) in intervals {
                let truncationHeight = self.truncationHeight(atX: axis == .real ? a : c)
                guard truncationHeight.isFinite else { continue }
                var coord = [a]
                coord.append(contentsOf: samples.filter { $0 > a && $0 < b })
                coord.append(b)
                // With the function at hand the strokes are ruled on f and
                // end where |f| reaches the truncation along them, bisected: the
                // grid's chords lie above f on a convex graph, so its truncation
                // region reaches past the truncation edge drawn on f by a fraction of a
                // cell, and a stroke ruled on it poked out past the truncation edge.
                func excessAt(_ t: Double) -> Double {
                    let p = axis == .real ? P2<DomainSpace>(t, c) : P2<DomainSpace>(c, t)
                    if let f = refine?.magnitude {
                        let u = f(p)
                        return (u.isNaN ? .infinity : u) - truncationHeight
                    }
                    return magnitude(at: p) - truncationHeight
                }
                let excess = coord.map(excessAt)
                let runs = refine?.magnitude == nil
                    ? Surface.runsAtOrAbove(coord, excess)
                    : Surface.runsAtOrAbove(coord, excess) { lo, hi in Surface.bisected(lo, hi, excessAt) }
                for run in runs {
                    out.append(run.map { t in
                        axis == .real ? P3<WorldSpace>(t, c, truncationHeight)
                                      : P3<WorldSpace>(c, t, truncationHeight)
                    })
                }
            }
        }
        return out
    }

    /// The stretches where `excess >= 0`, with their ends solved linearly.
    ///
    /// `t = e0 / (e0 - e1)` is where the same linear interpolation puts the truncation edge
    /// contour, so a stroke ends exactly on the truncation edge rather than at the last
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
        let flo = f(lo)
        guard (flo < 0) != (f(hi) < 0) else { return lo }
        let (a, b) = bisect(lo, hi) { (f($0) < 0) == (flo < 0) }
        return 0.5 * (a + b)
    }

    /// |f| - truncation, as a grid, in float32 -- the same subtraction the Python side
    /// performs, on the same float32 samples, so the truncation edge lands in the same
    /// place. The truncation edge is this grid's zero contour.
    func truncationExcess() -> Grid2D<Float>? {
        let grid = height
        var values = grid.values
        var anyFinite = false
        for x in 0..<grid.width {
            let truncationHeight = Float(self.truncationHeight(atX: grid.position(x: x, y: 0).x))
            guard truncationHeight.isFinite else {
                for y in 0..<grid.height { values[y * grid.width + x] = -.infinity }
                continue
            }
            anyFinite = true
            for y in 0..<grid.height {
                values[y * grid.width + x] = grid[x, y] - truncationHeight
            }
        }
        guard anyFinite else { return nil }
        return Grid2D(width: grid.width, height: grid.height, domain: grid.domain,
                      values: values)
    }

    /// The truncation edge, as the graph's trim loops draw it
    /// (`graphTrimLoops(tiles:refine:)`).
    func truncationEdge(tiles: [Affine2] = [.identity], refine: ContourRefine? = nil) -> [[P3<WorldSpace>]] {
        graphTrimLoops(tiles: tiles, refine: refine).truncationEdge
    }

    /// The graph's trim loops, and the truncation edge drawn on them.
    ///
    /// The truncation edge is the zero level of the grid's excess over the
    /// truncation, at the truncation height. With a refiner it is the contour
    /// |f| = h(x) placed on f, as the rings are, and an end of it on the
    /// domain's edge is re-solved by the top edge's own bisection along that
    /// edge (`truncationCrossing`), so the top edge meets the truncation edge
    /// to rounding. A banded truncation edge is placed band by band at each
    /// band's own level (`bandedTruncationEdge`), with the top of each riser.
    ///
    /// The loops are the domain's edge and the plateaus' loops: the
    /// truncation edge, closed along the domain's edge where a plateau
    /// reaches it. Without a refiner they are the excess grid's, one set for
    /// all the bands. With one they are placed band by band, and each band's
    /// are closed along the band's own edges as well, across the risers: the
    /// top of a riser lies on the lower band's loops, and where both sides
    /// of a band boundary are plateau, both bands' loops run along it. The
    /// loops bound the graph even-odd, a stretch passed twice bounding
    /// nothing. The truncation edge lies on them, riser tops included; a
    /// closing stretch is where the domain's or the band's edge is plateau,
    /// read on the excess grid without a refiner and on f with one. Each
    /// loop ends where it begins; `open` counts the chains that could not
    /// be closed, at an end off the edges with no other end there, which
    /// are among the loops as they are.
    public func graphTrimLoops(tiles: [Affine2] = [.identity], refine: ContourRefine? = nil)
        -> (loops: [[P2<DomainSpace>]], truncationEdge: [[P3<WorldSpace>]], open: Int) {
        let g = height
        let corner0 = g.position(x: 0, y: 0), corner1 = g.position(x: g.width - 1, y: g.height - 1)
        let rectangle = PlateauLoops.Rectangle(x0: min(corner0.x, corner1.x), x1: max(corner0.x, corner1.x),
                                               y0: min(corner0.y, corner1.y), y1: max(corner0.y, corner1.y))
        let domainEdge = rectangle.corners.map(\.p) + [rectangle.corners[0].p]
        guard let excess = truncationExcess() else { return ([domainEdge], [], 0) }
        var out: [[P3<WorldSpace>]] = []
        var plateaus: [PlateauLoops] = []
        for (_, lines) in Contour.levels(of: excess, [0.0]) {
            if let refine, let magnitude = refine.magnitude {
                @Sendable func plateau(_ truncationHeight: Double) -> @Sendable (P2<DomainSpace>) -> Bool {
                    { p in !(magnitude(inTile(p, tiles: tiles)) < truncationHeight) }
                }
                switch truncation {
                case .uniform(let truncationHeight) where truncationHeight.isFinite:
                    var whole = PlateauLoops(rectangle: rectangle, plateau: plateau(truncationHeight))
                    for line in refine.contours(.magnitude, truncationHeight, lines, .infinity) {
                        var line = line
                        for i in Set([0, line.count - 1]) where line.count >= 2 {
                            if let q = onBoundaryTruncationCrossing(near: line[i], tiles: tiles, magnitude: magnitude) {
                                line[i] = q
                            }
                        }
                        out.append(line.map { P3($0.x, $0.y, truncationHeight) })
                        whole.pieces.append(line)
                    }
                    plateaus.append(whole)
                    continue
                case .realBands(let bands, let beyond):
                    var pieces: [Int: [[P2<DomainSpace>]]] = [:]
                    out += bandedTruncationEdge(lines, bands: bands, beyond: beyond, tiles: tiles,
                                                refine: refine, magnitude: magnitude, pieces: &pieces)
                    let real = domain.real
                    let (lo, hi) = (min(real.lo, real.hi), max(real.lo, real.hi))
                    for span in truncation.bands(from: lo, to: hi) where span.height.isFinite {
                        var band = PlateauLoops(
                            rectangle: PlateauLoops.Rectangle(x0: span.lo == lo ? rectangle.x0 : span.lo,
                                                              x1: span.hi == hi ? rectangle.x1 : span.hi,
                                                              y0: rectangle.y0, y1: rectangle.y1),
                            plateau: plateau(span.height))
                        // Every placed run lies in the domain under a finite
                        // truncation, off the band boundaries, so its band
                        // has a span here.
                        band.pieces = pieces[span.band] ?? []
                        plateaus.append(band)
                    }
                    continue
                default:
                    break
                }
            }
            var whole = PlateauLoops(rectangle: rectangle, plateau: { excess.sample($0) > 0 })
            for line in lines {
                out.append(line.map { P3($0.x, $0.y, truncationHeight(atX: $0.x)) })
                whole.pieces.append(line)
            }
            plateaus.append(whole)
        }
        let closed = plateaus.map { $0.loops(eps: edgeTolerance) }
        return ([domainEdge] + closed.flatMap(\.loops), out, closed.reduce(0) { $0 + $1.open })
    }

    /// The truncation edge of a banded truncation, on f. Within a band the truncation edge is the contour
    /// |f| = that band's truncation: the grid's truncation edge lines are split by band, each
    /// run placed by the refiner at its own level, and a run's end beside a
    /// band's boundary is solved onto the boundary, where |f| reaches the
    /// truncation along it. Where the truncation steps the surface has a riser, and its
    /// top is the truncation edge there: along the boundary, at the lesser truncation where
    /// |f| is below it and at f's own height up to the greater truncation, from
    /// one end of the lesser band's truncation edge to the other. The grid's vertices
    /// in the cells that straddle a boundary are interpolated between two
    /// truncation heights and mean nothing; they are dropped, and the boundary's own
    /// pieces take their place.
    ///
    /// Each placed run, as drawn but on the ground, goes into `pieces`
    /// under its band, as `band(atX:from:)` numbers it; the risers'
    /// tops do not.
    func bandedTruncationEdge(_ lines: [[P2<DomainSpace>]], bands: [RealBand], beyond: Double, tiles: [Affine2],
                              refine: ContourRefine, magnitude: ContourRefine.Magnitude,
                              pieces: inout [Int: [[P2<DomainSpace>]]]) -> [[P3<WorldSpace>]] {
        // The boundaries inside the domain: one on its edge has no riser,
        // the band beyond the far edge starting outside the domain.
        let real = domain.real
        let boundaries = bands.map(\.below).filter { $0 > min(real.lo, real.hi) && $0 < max(real.lo, real.hi) }.sorted()
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
        /// The point on the boundary x = b where |f| = h(x), nearest in y to
        /// `near`: bracketed by walking out from it, then bisected.
        func ontoBoundary(_ b: Double, truncationHeight: Double, near: P2<DomainSpace>) -> P2<DomainSpace>? {
            func excess(_ y: Double) -> Double { mag(P2(b, y)) - truncationHeight }
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
            let bracket = bisect(lo, hi) { (excess($0) > 0) == (elo > 0) }
            return P2(b, 0.5 * (bracket.holds + bracket.fails))
        }

        var out: [[P3<WorldSpace>]] = []
        /// The ends of refined runs that lie on a boundary, by boundary:
        /// the truncation they belong to and the point.
        var endsOnBoundary: [Double: [(truncationHeight: Double, point: P2<DomainSpace>)]] = [:]
        for line in lines {
            // Runs of one band's vertices, the straddling cells' dropped.
            var runs: [(truncationHeight: Double, points: [P2<DomainSpace>])] = []
            var broken = true
            for v in line {
                if boundary(beside: v.x) != nil { broken = true; continue }
                let c = truncationHeight(atX: v.x)
                if !broken, let last = runs.last, last.truncationHeight == c {
                    runs[runs.count - 1].points.append(v)
                } else {
                    runs.append((c, [v]))
                }
                broken = false
            }
            // A closed line whose first and last runs are the same band and
            // unbroken between is one run.
            if line.first == line.last, runs.count >= 2, runs[0].truncationHeight == runs[runs.count - 1].truncationHeight,
               boundary(beside: line[0].x) == nil {
                let last = runs.removeLast()
                runs[0].points = last.points.dropLast() + runs[0].points
            }
            for run in runs where run.points.count >= 2 && run.truncationHeight.isFinite {
                for piece in refine.contours(.magnitude, run.truncationHeight, [run.points], .infinity) where piece.count >= 2 {
                    var piece = piece
                    // The far end's index read after the near end has had
                    // its point on the boundary put before it.
                    for atStart in [true, false] {
                        let i = atStart ? 0 : piece.count - 1
                        if let q = onBoundaryTruncationCrossing(near: piece[i], tiles: tiles, magnitude: magnitude) {
                            piece[i] = q
                        } else if let b = boundary(beside: piece[i].x) ?? boundaries.first(where: { abs(piece[i].x - $0) <= 2 * dx }),
                                  let q = ontoBoundary(b, truncationHeight: run.truncationHeight, near: piece[i]) {
                            if atStart { piece.insert(q, at: 0) } else { piece.append(q) }
                            endsOnBoundary[b, default: []].append((run.truncationHeight, q))
                        }
                    }
                    out.append(piece.map { P3($0.x, $0.y, run.truncationHeight) })
                    if let band = self.band(atX: run.points[0].x) { pieces[band, default: []].append(piece) }
                }
            }
        }
        // The top of each riser: between the two ends of the lesser
        // band's truncation edge on the boundary, nearest each other in y, at f's
        // height truncated by the greater band. In order along x, then y.
        for b in endsOnBoundary.keys.sorted() {
            let ends = endsOnBoundary[b] ?? []
            let left = truncation.height(atX: b, from: .left), right = truncation.height(atX: b, from: .right)
            let lesser = min(left, right), greater = max(left, right)
            guard lesser.isFinite, lesser < greater else { continue }
            let lows = ends.filter { $0.truncationHeight == lesser }.map(\.point).sorted { $0.y < $1.y }
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

    /// For a truncation edge vertex on the domain's edge, the truncation crossing between the
    /// two grid nodes beside it on that edge, by the top edge's bisection; nil
    /// for a vertex off the edge or with no crossing beside it.
    func onBoundaryTruncationCrossing(near p: P2<DomainSpace>, tiles: [Affine2],
                                      magnitude: ContourRefine.Magnitude) -> P2<DomainSpace>? {
        let g = height
        let eps = edgeTolerance
        let f = g.index(of: p)
        let onRow = abs(p.y - g.domain.imag.lo) <= eps || abs(p.y - g.domain.imag.hi) <= eps
        let onColumn = abs(p.x - g.domain.real.lo) <= eps || abs(p.x - g.domain.real.hi) <= eps
        guard onRow || onColumn else { return nil }
        func excess(_ q: P2<DomainSpace>) -> Double {
            magnitude(inTile(q, tiles: tiles)) - truncationHeight(atX: q.x)
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
        return truncationCrossing(from: a, excess(a), to: b, excess(b), tiles: tiles, magnitude: magnitude)
    }
}

// MARK: - the graph's trim loops

/// One set of plateaus' loops as they are closed: the truncation edge's
/// pieces over a rectangle, the domain or one band of it, joined end to end
/// along the rectangle's edge wherever the plateau holds it.
struct PlateauLoops {
    struct Rectangle {
        let x0: Double, x1: Double, y0: Double, y1: Double

        var perimeter: Double { 2 * (x1 - x0) + 2 * (y1 - y0) }
        /// The corners, counter-clockwise from (x0, y0), with the distance
        /// round the edge to each.
        var corners: [(s: Double, p: P2<DomainSpace>)] {
            let w = x1 - x0, h = y1 - y0
            return [(0, P2(x0, y0)), (w, P2(x1, y0)), (w + h, P2(x1, y1)), (2 * w + h, P2(x0, y1))]
        }

        /// The distance round the edge, counter-clockwise from (x0, y0), of
        /// a point within `eps` of it; nil for one off it.
        func distance(_ p: P2<DomainSpace>, eps: Double) -> Double? {
            let w = x1 - x0, h = y1 - y0
            guard p.x >= x0 - eps, p.x <= x1 + eps, p.y >= y0 - eps, p.y <= y1 + eps else { return nil }
            if abs(p.y - y0) <= eps { return min(max(p.x - x0, 0), w) }
            if abs(p.x - x1) <= eps { return w + min(max(p.y - y0, 0), h) }
            if abs(p.y - y1) <= eps { return w + h + min(max(x1 - p.x, 0), w) }
            if abs(p.x - x0) <= eps { return 2 * w + h + min(max(y1 - p.y, 0), h) }
            return nil
        }

        /// The point of the edge a distance round it.
        func point(_ s: Double) -> P2<DomainSpace> {
            let w = x1 - x0, h = y1 - y0
            let s = s.truncatingRemainder(dividingBy: perimeter)
            if s < w { return P2(x0 + s, y0) }
            if s < w + h { return P2(x1, y0 + (s - w)) }
            if s < 2 * w + h { return P2(x1 - (s - w - h), y1) }
            return P2(x0, y1 - (s - 2 * w - h))
        }
    }

    /// The rectangle the pieces are closed along.
    let rectangle: Rectangle
    /// Whether a point of the rectangle's edge is plateau.
    let plateau: @Sendable (P2<DomainSpace>) -> Bool
    var pieces: [[P2<DomainSpace>]] = []

    /// The pieces joined into loops: a closed piece is a loop of its own; an
    /// open one ending on the rectangle's edge is joined to the next end
    /// round it, through the corners between, wherever the plateau holds the
    /// stretch between, read at its middle; and pieces meeting end to end
    /// are joined. With no end on the edge the whole edge is a loop where it
    /// is plateau. Each loop ends where it begins; a chain that cannot be
    /// closed, at an end off the edge with nothing there, is kept as it is,
    /// and counted in `open`.
    func loops(eps: Double) -> (loops: [[P2<DomainSpace>]], open: Int) {
        var loops: [[P2<DomainSpace>]] = [], open = 0
        var edges: [[P2<DomainSpace>]] = []
        for piece in pieces where piece.count >= 2 {
            if piece.first == piece.last { loops.append(piece) } else { edges.append(piece) }
        }
        var ends: [(s: Double, p: P2<DomainSpace>)] = []
        for piece in edges {
            for p in [piece[0], piece[piece.count - 1]] {
                if let s = rectangle.distance(p, eps: eps) { ends.append((s, p)) }
            }
        }
        ends.sort { ($0.s, $0.p.x, $0.p.y) < ($1.s, $1.p.x, $1.p.y) }
        let perimeter = rectangle.perimeter
        if ends.isEmpty {
            if plateau(rectangle.point(0.5 * (rectangle.x1 - rectangle.x0))) {
                loops.append(rectangle.corners.map(\.p) + [rectangle.corners[0].p])
            }
        } else {
            for k in ends.indices {
                let a = ends[k], next = ends[(k + 1) % ends.count]
                let sb = k + 1 < ends.count ? next.s : next.s + perimeter
                guard sb > a.s, plateau(rectangle.point(0.5 * (a.s + sb))) else { continue }
                let between = (rectangle.corners + rectangle.corners.map { ($0.s + perimeter, $0.p) })
                    .filter { $0.s > a.s && $0.s < sb }.map(\.p)
                edges.append([a.p] + between + [next.p])
            }
        }
        // Chained end to end, each edge once, in the order they come.
        var at: [P2<DomainSpace>: [Int]] = [:]
        for (i, e) in edges.enumerated() {
            at[e[0], default: []].append(i)
            at[e[e.count - 1], default: []].append(i)
        }
        var used = [Bool](repeating: false, count: edges.count)
        for i in edges.indices where !used[i] {
            used[i] = true
            var chain = edges[i]
            while chain.first != chain.last,
                  let j = at[chain[chain.count - 1]]?.first(where: { !used[$0] }) {
                used[j] = true
                let e = edges[j]
                chain += (e[0] == chain[chain.count - 1] ? e : e.reversed()).dropFirst()
            }
            // Open at the far end: carried back from the near one as far as it goes.
            while chain.first != chain.last, let j = at[chain[0]]?.first(where: { !used[$0] }) {
                used[j] = true
                let e = edges[j]
                chain = (e[e.count - 1] == chain[0] ? e : e.reversed()).dropLast() + chain
            }
            loops.append(chain)
            if chain.first != chain.last { open += 1 }
        }
        return (loops, open)
    }
}
