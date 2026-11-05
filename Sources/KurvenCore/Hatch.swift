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
            // Cut to the footprint already, the risers' tops exactly.
            return PolylineSet(paths: drawnTruncationEdge(tiles: context.tiles, tiled: tiled,
                                                          region: context.region, refine: refine))
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
    /// The top edge reads the truncation as the curtain does
    /// (`wallFoot`): at each point's own x, through the tile under it,
    /// from the footprint's side, `inward` the edge's normal into it
    /// (`Surface.footprintSide`). A wall standing on a band boundary reads the
    /// band its footprint is in, all along, and a wall ending on one reads its
    /// own band at the end.
    func topEdge(of edge: PerimeterEdge, inward: SIMD2<Double>, tiles: [Affine2],
                 refine: ContourRefine?) -> [P3<WorldSpace>] {
        let n = max(edge.density, 2)
        let magnitudeOfF = refine?.magnitude
        let along = edge.end.v - edge.start.v
        /// A point of the edge: `t` of the way along it, nil between the
        /// ends; or where it meets a riser, the riser's boundary in the
        /// domain and the side of it the truncation is read from; or where it
        /// crosses a tile boundary, the tile it is read through on one side.
        typealias Station = (p: P2<DomainSpace>, t: Double?, riser: (x: Double, side: Truncation.Side)?, back: Affine2?)
        /// |f| over a station and the truncation there, both read through
        /// the tile under it, or the station's own.
        func read(_ station: Station) -> (u: Double, truncationHeight: Double) {
            let foot = wallFoot(at: station.p, t: station.t ?? 0.5, along: along, inward: inward, tiles: tiles,
                                through: station.back)
            let truncationHeight = station.riser.map { self.truncationHeight(atX: $0.x, from: $0.side) } ?? foot.truncationHeight
            return (magnitudeOfF.map { $0(foot.point) } ?? magnitude(at: foot.point), truncationHeight)
        }
        /// The surface over a point between the stations, truncated.
        func profile(_ p: P2<DomainSpace>) -> Double {
            let (u, truncationHeight) = read((p, nil, nil, nil))
            return min(u, truncationHeight)
        }
        var topEdge: [P3<WorldSpace>] = []
        topEdge.reserveCapacity(n)
        var previous: (p: P2<DomainSpace>, excess: Double)?
        // The nodes along the edge; with the function, where the edge meets a
        // riser between two nodes, two vertices there as well, one either
        // side of the boundary at its own band's height -- the boundary
        // itself belongs to the band beyond it, so the near side is a
        // rounding step short of it.
        var stations: [Station] = (0..<n).map {
            let t = Double($0) / Double(n - 1)
            return (Surface.point(on: edge, at: t), t, nil, nil)
        }
        // Where the edge crosses a tile boundary between two nodes and the
        // copies' tops there differ (`stands(_:over:)`), two stations at the
        // crossing, each read through the tile on its side
        // (`tileBoundaryCrossings`): the top steps there, down a riser, as at
        // a band boundary. One at an end of the segment, which is read
        // through the other already. Where the tops agree, none: the top
        // runs on across, as it would on one copy.
        if tiles.count > 1 {
            var withCrossings: [Station] = []
            for (a, b) in zip(stations, stations.dropFirst()) {
                withCrossings.append(a)
                for crossing in tileBoundaryCrossings(of: edge, from: a.t ?? 0, to: b.t ?? 1, inward: inward, tiles: tiles) {
                    let p = Surface.point(on: edge, at: crossing.t)
                    let near: Station = (p, crossing.t, nil, crossing.near), far: Station = (p, crossing.t, nil, crossing.far)
                    let (zn, zf) = (read(near), read(far))
                    let (tn, tf) = (min(zn.u, zn.truncationHeight), min(zf.u, zf.truncationHeight))
                    guard Surface.stands(tn, over: tf) || Surface.stands(tf, over: tn) else { continue }
                    if crossing.t != a.t { withCrossings.append(near) }
                    if crossing.t != b.t { withCrossings.append(far) }
                }
            }
            withCrossings.append(stations[stations.count - 1])
            stations = withCrossings
        }
        if magnitudeOfF != nil, case .realBands(let bands, _) = truncation {
            // The boundaries inside the domain, a band that starts on its
            // right edge starting beyond it, that the edge crosses in the
            // domain, through the tile under the crossing: one it ends on it
            // meets in its own band (`Surface.footprintSide`).
            let real = domain.real
            let boundaries = bands.map(\.below).filter { $0 > min(real.lo, real.hi) && $0 < max(real.lo, real.hi) }
            var withRisers: [Station] = []
            for (a, b) in zip(stations, stations.dropFirst()) {
                withRisers.append(a)
                let backs = [a, b].map { $0.back ?? wallTile(at: $0.p, t: $0.t ?? 0.5, along: along, inward: inward, tiles: tiles).back }
                var met: [(s: Double, near: Station, far: Station)] = []
                for back in backs.indices.filter({ $0 == 0 || backs[$0] != backs[0] }).map({ backs[$0] }) {
                    guard let forward = back.inverse else { continue }
                    let qa = back(P3(a.p.x, a.p.y, 0)), qb = back(P3(b.p.x, b.p.y, 0))
                    // An end of the edge on a boundary, a rounding off it
                    // where it came back through a tile.
                    let xa = a.t == 0 ? wallFootX(qa.x, at: 0, along: along, back: back) : qa.x
                    let xb = b.t == 1 ? wallFootX(qb.x, at: 1, along: along, back: back) : qb.x
                    for x in boundaries {
                        guard (xa < x && xb >= x) || (xb < x && xa >= x), xb != xa,
                              !(a.t == 0 && xa == x), !(b.t == 1 && xb == x) else { continue }
                        let s = (x - xa) / (xb - xa)
                        let y = qa.y + (qb.y - qa.y) * s
                        // Met where this tile is the one under it.
                        let crossing = forward(P3(x, y, 0))
                        guard backs[0] == backs[1]
                                || wallTile(at: P2(crossing.x, crossing.y), t: 0.5, along: along, inward: inward, tiles: tiles).back == back,
                              !met.contains(where: { $0.s == s }) else { continue }
                        // Carried forward, on the edge: a vertical or a
                        // horizontal edge keeps its own x or y, which a
                        // turned tile carries a rounding off.
                        func onEdge(_ q: P3<WorldSpace>) -> P2<DomainSpace> {
                            P2(a.p.x == b.p.x ? a.p.x : q.x, a.p.y == b.p.y ? a.p.y : q.y)
                        }
                        let near = forward(P3(xa < x ? x.nextDown : x, y, 0)), far = forward(P3(xa < x ? x : x.nextDown, y, 0))
                        let (nearSide, farSide): (Truncation.Side, Truncation.Side) = xa < x ? (.left, .right) : (.right, .left)
                        met.append((s, (onEdge(near), nil, (x, nearSide), nil), (onEdge(far), nil, (x, farSide), nil)))
                    }
                }
                for crossing in met.sorted(by: { $0.s < $1.s }) {
                    withRisers.append(crossing.near); withRisers.append(crossing.far)
                }
            }
            withRisers.append(stations[stations.count - 1])
            stations = withRisers
        }
        for station in stations {
            let p = station.p
            let (u, truncationHeight) = read(station)
            let excess = u - truncationHeight
            if let a = previous, excess.isFinite, a.excess.isFinite, a.p.x != p.x || a.p.y != p.y {
                let here: (p: P2<DomainSpace>, excess: Double) = (p, excess)
                var under: (p: P2<DomainSpace>, excess: Double)?
                var over: (p: P2<DomainSpace>, excess: Double)?
                if a.excess <= 0 && excess > 0 { under = a; over = here }
                if excess <= 0 && a.excess > 0 { under = here; over = a }
                if let under, let over {
                    if magnitudeOfF != nil {
                        if let q = truncationCrossing(from: under.p, under.excess, to: over.p, over.excess,
                                                      excess: { let r = read(($0, nil, nil, nil)); return r.u - r.truncationHeight }) {
                            topEdge.append(P3(q.x, q.y, read((q, nil, nil, nil)).truncationHeight))
                        }
                    } else {
                        let t = under.excess / (under.excess - over.excess)
                        if t > 0, t < 1 {
                            let x = under.p.x + (over.p.x - under.p.x) * t
                            let y = under.p.y + (over.p.y - under.p.y) * t
                            topEdge.append(P3(x, y, read((P2(x, y), nil, nil, nil)).truncationHeight))
                        }
                    }
                }
            }
            topEdge.append(P3(p.x, p.y, min(u, truncationHeight)))
            previous = (p, excess)
        }
        guard let refine, refine.magnitude != nil else { return topEdge }
        return subdividedProfile(withMinima(topEdge, profile: profile), profile: profile, tolerance: refine.tolerance)
    }

    /// Where a wall's edge crosses a tile boundary between `a` and `b` of the
    /// way along it, in order, each with the tiles either side: the tile
    /// under the foot at `a` (`wallTile`), its copy of the domain left where
    /// the edge carried back through it leaves the rectangle, and the tile
    /// under the edge just beyond, until the one under the foot at `b`. A
    /// foot on a tile boundary read through the tile on its footprint's side
    /// meets it at that foot. Nothing for one tile, or a stretch under one.
    func tileBoundaryCrossings(of edge: PerimeterEdge, from a: Double, to b: Double, inward: SIMD2<Double>,
                       tiles: [Affine2]) -> [(t: Double, near: Affine2, far: Affine2)] {
        guard tiles.count > 1 else { return [] }
        let along = edge.end.v - edge.start.v, length = simd_length(along)
        guard length > 0 else { return [] }
        let d = domain
        let (xlo, xhi) = (min(d.real.lo, d.real.hi), max(d.real.lo, d.real.hi))
        let (ylo, yhi) = (min(d.imag.lo, d.imag.hi), max(d.imag.lo, d.imag.hi))
        let pb = Surface.point(on: edge, at: b)
        var back = wallTile(at: Surface.point(on: edge, at: a), t: a, along: along, inward: inward, tiles: tiles).back
        let last = wallTile(at: pb, t: b, along: along, inward: inward, tiles: tiles).back
        var t = a
        var out: [(t: Double, near: Affine2, far: Affine2)] = []
        while back != last, out.count < tiles.count {
            let pa = Surface.point(on: edge, at: t)
            let qa = back(P3(pa.x, pa.y, 0)), qb = back(P3(pb.x, pb.y, 0))
            // Where the segment leaves the rectangle; at `b` where it does
            // not, the foot there on the boundary.
            var s = 1.0
            for (u, v, lo, hi) in [(qa.x, qb.x, xlo, xhi), (qa.y, qb.y, ylo, yhi)] where v != u {
                if v > hi { s = min(s, (hi - u) / (v - u)) }
                if v < lo { s = min(s, (lo - u) / (v - u)) }
            }
            let crossing = s >= 1 ? b : t + (b - t) * max(s, 0)
            let p = Surface.point(on: edge, at: crossing)
            let next = crossing == b ? last : tile(under: P2(p.v + edgeTolerance * along / length), tiles: tiles).back
            guard next != back else { break }
            out.append((crossing, back, next))
            back = next; t = crossing
        }
        return out
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
    /// and the truncation edge's ends, so the two meet to rounding. `excess`
    /// is |f| less the truncation at a point of the segment, as the caller
    /// reads it: the top edge through the tile under the point, from the
    /// footprint's side.
    func truncationCrossing(from a: P2<DomainSpace>, _ ea: Double, to b: P2<DomainSpace>, _ eb: Double,
                            excess: (P2<DomainSpace>) -> Double) -> P2<DomainSpace>? {
        guard ea.isFinite, eb.isFinite, (ea > 0) != (eb > 0) else { return nil }
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
    /// the vertices keep their order. `profile` is the truncated surface
    /// over a point of the edge, as the top edge reads it.
    func subdividedProfile(_ topEdge: [P3<WorldSpace>], profile: (P2<DomainSpace>) -> Double,
                           tolerance: Double) -> [P3<WorldSpace>] {
        guard topEdge.count >= 2 else { return topEdge }
        let cell = self.cell
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
    /// the distance along it and the new vertices keep that order. `profile`
    /// is the truncated surface over a point of the edge, as the top edge
    /// reads it.
    func withMinima(_ topEdge: [P3<WorldSpace>], profile: (P2<DomainSpace>) -> Double) -> [P3<WorldSpace>] {
        guard topEdge.count >= 3 else { return topEdge }
        // A vertex's point on the edge.
        func dom(_ v: P3<WorldSpace>) -> P2<DomainSpace> { P2(v.x, v.y) }
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
            // are not parallel to the imaginary axis in the domain, through
            // the tile the wall reads its foot there through, where any is not.
            let tops = perimeter.edges.indices.flatMap { k -> [(top: Double, leaves: Bool)] in
                let e = perimeter.edges[k], along = e.end.v - e.start.v
                return [(e.start, 0.0), (e.end, 1.0)].filter { $0.0 == corner }.map { _, t in
                    let back = wallTile(at: P2(corner.x, corner.y), t: t, along: along, inward: perimeter.inward(k), tiles: tiles).back
                    return (wallHeight(at: P2(corner.x, corner.y), t: t, from: e.start, to: e.end,
                                       inward: perimeter.inward(k), tiles: tiles),
                            !Surface.isParallelToImaginaryAxis(along, back: back))
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
    /// (`graphTrimLoops(refine:)`).
    func truncationEdge(refine: ContourRefine? = nil) -> [[P3<WorldSpace>]] {
        graphTrimLoops(refine: refine).truncationEdge
    }

    /// The truncation edge as its layer draws it: on every tile when
    /// `tiled`, and in place otherwise, cut to the footprint, each riser's
    /// top kept only where the riser stands, where the footprint holds both
    /// sides of its boundary, read a hair either side of it where the copy
    /// stands, as `risers(tiles:region:refine:)` reads it. Where the
    /// footprint holds one side there is no riser: a wall on the boundary
    /// rises to the top on that side, and a top drawn there would stand over
    /// the wall, or along its top edge.
    ///
    /// The placed pieces are cut at their vertices, as every layer is
    /// (`clip(_:to:)`). A riser's top is cut exactly, where the footprint's
    /// perimeter crosses the two lines either side of its boundary, its
    /// height read there: a flat top has a vertex at each end and nothing
    /// between, and cut at its vertices it went missing where the riser
    /// stands, or ran across a notch where none does.
    ///
    /// Tiled, the risers' tops at the tile boundaries follow, where those
    /// risers stand (`tileBoundaryRiserTops(tiles:region:refine:)`).
    func drawnTruncationEdge(tiles: [Affine2], tiled: Bool, region: Region,
                             refine: ContourRefine?) -> [[P3<WorldSpace>]] {
        let trim = graphTrimLoops(refine: refine)
        let pieces = trim.truncationEdge.dropLast(trim.riserTops.count)
        let side = 1e-9 * cell
        let magnitude = refine?.magnitude
        let atTileBoundaries = tiled ? tileBoundaryRiserTops(tiles: tiles, region: region, refine: refine) : []
        return (tiled ? tiles.map(Optional.some) : [nil]).flatMap { tile -> [[P3<WorldSpace>]] in
            func placed(_ v: P3<WorldSpace>) -> P3<WorldSpace> { tile.map { $0(v) } ?? v }
            var paths = Surface.clip(pieces.map { $0.map(placed) }, to: region)
            guard case .inside(let perimeter) = region else {
                return paths + trim.riserTops.map { $0.top.map(placed) }
            }
            for (b, top) in trim.riserTops {
                guard let first = top.first, let last = top.last, first.y < last.y else { continue }
                let greater = max(truncation.height(atX: b, from: .left), truncation.height(atX: b, from: .right))
                func height(_ y: Double) -> Double {
                    guard let magnitude else { return first.z }
                    let u = magnitude(P2(b, y))
                    return min(u.isNaN ? .infinity : u, greater)
                }
                // Where the perimeter crosses the lines a hair either side
                // of the boundary, in the copy's place: w(y) = o + y d.
                var stops = [first.y, last.y]
                for x in [b - side, b + side] {
                    let o = placed(P3(x, 0, 0)).xy.v, d = placed(P3(x, 1, 0)).xy.v - o
                    for edge in perimeter.edges {
                        let p = edge.start.v, e = edge.end.v - edge.start.v
                        let det = e.x * d.y - e.y * d.x
                        guard det != 0 else { continue }
                        let r = p - o
                        let y = (e.x * r.y - e.y * r.x) / det, s = (d.x * r.y - d.y * r.x) / det
                        if s >= 0, s <= 1, y > first.y, y < last.y { stops.append(y) }
                    }
                }
                stops.sort()
                func holds(_ y: Double) -> Bool {
                    region.contains(placed(P3(b - side, y, 0)).xy) && region.contains(placed(P3(b + side, y, 0)).xy)
                }
                // The stretches between the stops the footprint holds on
                // both sides, run together, each drawn from its first stop to
                // its last through the top's vertices between.
                var stretches: [(lo: Double, hi: Double)] = []
                for (a, c) in zip(stops, stops.dropFirst()) where c > a && holds(0.5 * (a + c)) {
                    if let last = stretches.last, last.hi == a { stretches[stretches.count - 1].hi = c } else { stretches.append((a, c)) }
                }
                for (lo, hi) in stretches {
                    let start = lo == first.y ? first : P3<WorldSpace>(b, lo, height(lo))
                    let end = hi == last.y ? last : P3<WorldSpace>(b, hi, height(hi))
                    paths.append(([start] + top.filter { $0.y > lo && $0.y < hi } + [end]).map(placed))
                }
            }
            return paths
        } + atTileBoundaries
    }

    /// The tops of the risers at the tile boundaries
    /// (`tileBoundaryRisers(tiles:region:refine:)`) where they stand
    /// (`tileBoundaryStretches`), at the higher of the two copies' tops, each
    /// read on the tile boundary through its tile (`tileBoundaryTop`), in
    /// world space. A top has a vertex at each of the stretch's stops and,
    /// with a refiner, its chords are subdivided until none departs from the
    /// top by more than the refiner's tolerance, as a riser's top at a band
    /// boundary is.
    func tileBoundaryRiserTops(tiles: [Affine2], region: Region, refine: ContourRefine?) -> [[P3<WorldSpace>]] {
        let magnitude = faceMagnitude(refine)
        var out: [[P3<WorldSpace>]] = []
        for boundary in tileBoundaries(tiles: tiles) {
            guard let first = tiles[boundary.first].inverse, let second = tiles[boundary.second].inverse else { continue }
            func tops(_ w: P2<WorldSpace>) -> [Double] {
                [first, second].map { tileBoundaryTop(w, back: $0, magnitude: magnitude) }
            }
            let stretches = tileBoundaryStretches(boundary, backs: [first, second], region: region, tops: tops,
                                          excess: { q in magnitude(q) - self.truncationHeight(atX: q.x) })
            for stretch in stretches {
                var curve: [P3<WorldSpace>] = []
                func vertex(_ s: Double, _ z: Double) -> P3<WorldSpace> {
                    let w = boundary.point(s)
                    return P3(w.x, w.y, z)
                }
                func top(_ s: Double) -> Double { tops(boundary.point(s)).max() ?? .nan }
                for (k, stop) in stretch.enumerated() {
                    let z = tops(stop.read).max() ?? .nan
                    let at = P3<WorldSpace>(stop.at.x, stop.at.y, z)
                    if k > 0, let tolerance = refine?.tolerance, refine?.magnitude != nil {
                        let a = stretch[k - 1].s, za = curve[curve.count - 1].z
                        func subdivide(_ s0: Double, _ z0: Double, _ s1: Double, _ z1: Double, depth: Int) {
                            guard depth < 7, s1 - s0 > 1e-9 * cell else { return }
                            let sm = 0.5 * (s0 + s1), zm = top(sm)
                            guard zm.isFinite, abs(zm - 0.5 * (z0 + z1)) > tolerance * cell else { return }
                            subdivide(s0, z0, sm, zm, depth: depth + 1)
                            curve.append(vertex(sm, zm))
                            subdivide(sm, zm, s1, z1, depth: depth + 1)
                        }
                        subdivide(a, za, stop.s, z, depth: 0)
                    }
                    curve.append(at)
                }
                if curve.count >= 2 { out.append(curve) }
            }
        }
        return out
    }

    /// The stretches of a tile boundary a riser stands on: where the
    /// footprint holds both sides, read a hair either side of the tile
    /// boundary, and one copy's top stands over the other's
    /// (`stands(_:over:)`), `tops` reading each copy's over a point of the
    /// tile boundary, `backs` the two tiles' inverses. Each stretch is its
    /// stops in order: the distance `s` along the tile boundary, the point
    /// `at` it is, and the point its tops are read at, at its ends a hair
    /// inside it, since a copy's top can step there, or the footprint end. The
    /// stops: each copy's grid
    /// nodes along the tile boundary (every `step`-th); where the footprint's
    /// perimeter crosses the lines either side of the tile boundary, exactly,
    /// on the edge itself where the edge or the boundary runs along an axis;
    /// where a copy's top steps, at a band boundary its side of the domain
    /// crosses; where a copy's `excess` over its truncation changes sign; and
    /// where the copies' tops cross, or the riser stops standing otherwise,
    /// bisected to the last bit. Stops nearer than a billionth of a cell are
    /// one, a step or the footprint's end kept over the others: a band
    /// boundary on one copy's grid node lands a rounding from the other
    /// copy's node there. A stretch runs on through a crossing of the tops
    /// and breaks where a copy's top steps.
    func tileBoundaryStretches(_ boundary: TileBoundary, backs: [Affine2], region: Region, step: Int = 1,
                               tops: (P2<WorldSpace>) -> [Double], excess: (P2<DomainSpace>) -> Double)
        -> [[(s: Double, at: P2<WorldSpace>, read: P2<WorldSpace>)]] {
        let side = 1e-9 * cell, nudge = 1e-9 * cell
        let length = boundary.length, n = boundary.normal
        let g = height
        func held(_ s: Double) -> Bool {
            let w = boundary.point(s).v
            return region.contains(P2(w - side * n)) && region.contains(P2(w + side * n))
        }
        func stands(_ s: Double) -> Bool {
            let z = tops(boundary.point(s))
            return held(s) && (Surface.stands(z[0], over: z[1]) || Surface.stands(z[1], over: z[0]))
        }
        func copy(_ k: Int, _ s: Double) -> P2<DomainSpace> {
            let w = boundary.point(s)
            return ontoDomain(backs[k](P3(w.x, w.y, 0)).xy)
        }
        // Each stop with what it is: a grid node, or bisected, 0; where the
        // footprint ends, 1; where a copy's top steps, 2.
        var stops: [(s: Double, kind: Int, at: P2<WorldSpace>?)] = [(0, 0, nil), (length, 0, nil)]
        for k in backs.indices {
            let q0 = copy(k, 0), q1 = copy(k, length)
            func along(_ c: Double, _ c0: Double, _ c1: Double) -> Double? {
                let s = length * (c - c0) / (c1 - c0)
                return s > 0 && s < length ? s : nil
            }
            if q0.x != q1.x {
                for i in stride(from: 0, to: g.width, by: max(step, 1)) {
                    if let s = along(g.position(x: i, y: 0).x, q0.x, q1.x) { stops.append((s, 0, nil)) }
                }
                if case .realBands(let bands, _) = truncation {
                    for b in bands { if let s = along(b.below, q0.x, q1.x) { stops.append((s, 2, nil)) } }
                }
            }
            if q0.y != q1.y {
                for j in stride(from: 0, to: g.height, by: max(step, 1)) {
                    if let s = along(g.position(x: 0, y: j).y, q0.y, q1.y) { stops.append((s, 0, nil)) }
                }
            }
        }
        // Where the perimeter crosses the lines either side of the tile
        // boundary, o + s u, at the edge's p + r e.
        if case .inside(let perimeter) = region {
            let u = (boundary.end.v - boundary.start.v) / length
            for o in [boundary.start.v - side * n, boundary.start.v + side * n] {
                for edge in perimeter.edges {
                    let p = edge.start.v, e = edge.end.v - edge.start.v
                    let det = e.x * u.y - e.y * u.x
                    guard det != 0 else { continue }
                    let r = p - o
                    var s = (e.x * r.y - e.y * r.x) / det
                    let t = (u.x * r.y - u.y * r.x) / det
                    guard t >= 0, t <= 1 else { continue }
                    // Along an axis, the crossing's coordinate there is the
                    // edge's or the boundary's own, not a ratio a rounding
                    // off it.
                    var at: P2<WorldSpace>?
                    if u.x == 0 {
                        let y = e.y == 0 ? p.y : p.y + t * e.y
                        at = P2(boundary.start.x, y); s = (y - boundary.start.y) / u.y
                    } else if u.y == 0 {
                        let x = e.x == 0 ? p.x : p.x + t * e.x
                        at = P2(x, boundary.start.y); s = (x - boundary.start.x) / u.x
                    }
                    if s > 0, s < length { stops.append((s, 1, at)) }
                }
            }
        }
        /// The stops in order, those nearer than `nudge` one, of the
        /// greater kind.
        func merged(_ stops: [(s: Double, kind: Int, at: P2<WorldSpace>?)]) -> [(s: Double, kind: Int, at: P2<WorldSpace>?)] {
            var out: [(s: Double, kind: Int, at: P2<WorldSpace>?)] = []
            for stop in stops.sorted(by: { $0.s < $1.s }) {
                if let last = out.last, stop.s - last.s <= nudge {
                    if stop.kind > last.kind { out[out.count - 1] = stop }
                } else {
                    out.append(stop)
                }
            }
            return out
        }
        let sorted = merged(stops).map(\.s)
        var more: [(s: Double, kind: Int, at: P2<WorldSpace>?)] = []
        for (a, c) in zip(sorted, sorted.dropFirst()) {
            let (ia, ic) = (a + nudge, c - nudge)
            guard ic > ia else { continue }
            for k in backs.indices {
                let ea = excess(copy(k, ia)), ec = excess(copy(k, ic))
                if ea.isFinite, ec.isFinite, (ea > 0) != (ec > 0) {
                    let bracket = bisect(ia, ic) { (excess(copy(k, $0)) > 0) == (ea > 0) }
                    more.append((0.5 * (bracket.holds + bracket.fails), 0, nil))
                }
            }
            let (da, dc) = (tops(boundary.point(ia)), tops(boundary.point(ic)))
            if (da[0] - da[1]).isFinite, (dc[0] - dc[1]).isFinite, (da[0] > da[1]) != (dc[0] > dc[1]) {
                let bracket = bisect(ia, ic) { s in let z = tops(boundary.point(s)); return (z[0] > z[1]) == (da[0] > da[1]) }
                more.append((0.5 * (bracket.holds + bracket.fails), 0, nil))
            } else if stands(ia) != stands(ic) {
                let standing = stands(ia)
                let bracket = bisect(ia, ic) { stands($0) == standing }
                more.append((0.5 * (bracket.holds + bracket.fails), 0, nil))
            }
        }
        let all = merged(merged(stops) + more)
        var stretches: [[(s: Double, at: P2<WorldSpace>, read: P2<WorldSpace>)]] = []
        var open = false
        for (a, c) in zip(all, all.dropFirst()) {
            guard stands(0.5 * (a.s + c.s)) else { open = false; continue }
            // Read from within at the stretch's ends; where it runs on, a
            // stop between is read where it is.
            let start = (s: a.s, at: a.at ?? boundary.point(a.s), read: boundary.point(a.s + nudge))
            let end = (s: c.s, at: c.at ?? boundary.point(c.s), read: boundary.point(c.s - nudge))
            if open, a.kind != 2 {
                stretches[stretches.count - 1][stretches[stretches.count - 1].count - 1].read = boundary.point(a.s)
                stretches[stretches.count - 1].append(end)
            } else {
                stretches.append([start, end])
            }
            open = true
        }
        return stretches
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
    /// are among the loops as they are. `riserTops` are the risers' tops,
    /// which end `truncationEdge`, each with its boundary.
    public func graphTrimLoops(refine: ContourRefine? = nil)
        -> (loops: [[P2<DomainSpace>]], truncationEdge: [[P3<WorldSpace>]],
            riserTops: [(at: Double, top: [P3<WorldSpace>])], open: Int) {
        let g = height
        let corner0 = g.position(x: 0, y: 0), corner1 = g.position(x: g.width - 1, y: g.height - 1)
        let rectangle = PlateauLoops.Rectangle(x0: min(corner0.x, corner1.x), x1: max(corner0.x, corner1.x),
                                               y0: min(corner0.y, corner1.y), y1: max(corner0.y, corner1.y))
        let domainEdge = rectangle.corners.map(\.p) + [rectangle.corners[0].p]
        guard let excess = truncationExcess() else { return ([domainEdge], [], [], 0) }
        var out: [[P3<WorldSpace>]] = []
        var riserTops: [(at: Double, top: [P3<WorldSpace>])] = []
        var plateaus: [PlateauLoops] = []
        for (_, lines) in Contour.levels(of: excess, [0.0]) {
            if let refine, let magnitude = refine.magnitude {
                @Sendable func plateau(_ truncationHeight: Double) -> @Sendable (P2<DomainSpace>) -> Bool {
                    { p in !(magnitude(p) < truncationHeight) }
                }
                switch truncation {
                case .uniform(let truncationHeight) where truncationHeight.isFinite:
                    var whole = PlateauLoops(rectangle: rectangle, plateau: plateau(truncationHeight))
                    var placed = refine.contours(.magnitude, truncationHeight, lines, .infinity)
                    placed += refine.rings?(truncationHeight, placed) ?? []
                    for line in placed {
                        var line = line
                        for i in Set([0, line.count - 1]) where line.count >= 2 {
                            if let q = onBoundaryTruncationCrossing(near: line[i], magnitude: magnitude) {
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
                    out += bandedTruncationEdge(lines, bands: bands, beyond: beyond,
                                                refine: refine, magnitude: magnitude, pieces: &pieces,
                                                riserTops: &riserTops)
                    out += riserTops.map(\.top)
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
        return ([domainEdge] + closed.flatMap(\.loops), out, riserTops, closed.reduce(0) { $0 + $1.open })
    }

    /// The truncation edge of a banded truncation, on f. Within a band the truncation edge is the contour
    /// |f| = that band's truncation: the grid's truncation edge lines are split by band, each
    /// run placed by the refiner at its own level, and a run's end beside a
    /// band's boundary is solved onto the boundary, where |f| reaches the
    /// truncation along it. Where the truncation steps the surface has a riser, and its
    /// top is the truncation edge there: along the boundary, at f's own height up to the
    /// greater truncation, wherever the lesser band's plateau holds the boundary, from one
    /// end of that band's truncation edge on it to the next, or to the domain's edge. The grid's vertices
    /// in the cells that straddle a boundary are interpolated between two
    /// truncation heights and mean nothing; they are dropped, and the boundary's own
    /// pieces take their place. But where a plateau comes within a column of a boundary
    /// and turns back, |f| reaching the band's truncation nowhere on the boundary near
    /// either side of the gap, the dropped vertices are that band's own truncation edge:
    /// they are placed with the runs either side, at its level, so long as the placed run
    /// keeps to the band.
    ///
    /// Each placed run, as drawn but on the ground, goes into `pieces`
    /// under its band, as `band(atX:from:)` numbers it; the risers'
    /// tops go into `riserTops` instead, each with its boundary, and are not
    /// among the curves returned.
    func bandedTruncationEdge(_ lines: [[P2<DomainSpace>]], bands: [RealBand], beyond: Double,
                              refine: ContourRefine, magnitude: ContourRefine.Magnitude,
                              pieces: inout [Int: [[P2<DomainSpace>]]],
                              riserTops: inout [(at: Double, top: [P3<WorldSpace>])]) -> [[P3<WorldSpace>]] {
        // The boundaries inside the domain: one on its edge has no riser,
        // the band beyond the far edge starting outside the domain.
        let real = domain.real
        let boundaries = bands.map(\.below).filter { $0 > min(real.lo, real.hi) && $0 < max(real.lo, real.hi) }.sorted()
        let g = height
        let dx = abs(g.domain.real.length) / Double(max(g.width - 1, 1))
        let dy = abs(g.domain.imag.length) / Double(max(g.height - 1, 1))
        let cell = self.cell
        let tolerance = refine.tolerance * cell
        func mag(_ p: P2<DomainSpace>) -> Double { magnitude(p) }
        /// The boundary a point is beside, within a column either way.
        func boundary(beside x: Double) -> Double? {
            boundaries.first { abs(x - $0) <= 1.0001 * dx }
        }
        /// The point on the boundary x = b where |f| = h(x), nearest in y to
        /// `near`: bracketed by walking out from it, then bisected.
        /// It walks three rows either way, or as far as `across` reaches,
        /// the dropped vertices of the gap the end is beside: a plateau that
        /// crosses the boundary within that stretch meets it there.
        func ontoBoundary(_ b: Double, truncationHeight: Double, near: P2<DomainSpace>,
                          across: [P2<DomainSpace>] = []) -> P2<DomainSpace>? {
            func excess(_ y: Double) -> Double { mag(P2(b, y)) - truncationHeight }
            let step = dy / 8
            let reach = across.map { abs($0.y - near.y) + dy }.max() ?? 0
            var lo = near.y, hi = near.y
            var elo = excess(lo), ehi = elo
            guard elo.isFinite else { return nil }
            var found = false
            for k in 1...max(24, Int((reach / step).rounded(.up))) {
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
            // Runs of one band's vertices, the straddling cells' dropped:
            // after each run, the vertices dropped there and the boundaries
            // they lie beside, and before the first, `head`.
            var runs: [(truncationHeight: Double, points: [P2<DomainSpace>])] = []
            var gaps: [(beside: Set<Double>, points: [P2<DomainSpace>])] = []
            var head: [P2<DomainSpace>] = []
            var broken = true
            for v in line {
                if let b = boundary(beside: v.x) {
                    if runs.isEmpty { head.append(v) } else {
                        gaps[gaps.count - 1].beside.insert(b); gaps[gaps.count - 1].points.append(v)
                    }
                    broken = true
                    continue
                }
                let c = truncationHeight(atX: v.x)
                if !broken, let last = runs.last, last.truncationHeight == c {
                    runs[runs.count - 1].points.append(v)
                } else {
                    runs.append((c, [v])); gaps.append(([], []))
                }
                broken = false
            }
            let closed = line.first == line.last
            // A closed line whose first and last runs are the same band and
            // unbroken between is one run.
            if closed, runs.count >= 2, runs[0].truncationHeight == runs[runs.count - 1].truncationHeight,
               boundary(beside: line[0].x) == nil {
                let last = runs.removeLast()
                gaps.removeLast()
                runs[0].points = last.points.dropLast() + runs[0].points
            }
            // On a closed line the last gap runs on through the head, its
            // first vertex the line's last, to the first run.
            if closed, !head.isEmpty, !runs.isEmpty {
                for b in head.compactMap({ boundary(beside: $0.x) }) { gaps[gaps.count - 1].beside.insert(b) }
                gaps[gaps.count - 1].points += head.dropFirst()
            }
            // A gap is a stretch of the band's own truncation edge, not a
            // crossing of the boundary, where the runs either side are the
            // same band and |f| is under that band's truncation all along
            // the boundary across the gap, from a row below its lowest point
            // to a row above its highest, read every eighth of a row: the
            // plateau comes within a column of the boundary and turns back.
            // Then the dropped vertices are the band's, and are placed with
            // the runs, on f, at its level.
            func turnsBack(at b: Double, below truncationHeight: Double, across points: [P2<DomainSpace>]) -> Bool {
                let ys = points.map(\.y)
                guard let low = ys.min(), let high = ys.max() else { return false }
                let lo = low - dy, hi = high + dy
                let steps = max(Int(((hi - lo) / (dy / 8)).rounded(.up)), 1)
                return (0...steps).allSatisfy { i in
                    mag(P2(b, lo + (hi - lo) * Double(i) / Double(steps))) < truncationHeight
                }
            }
            let n = runs.count
            let joins = (0..<n).map { k -> Bool in
                let j = k + 1 < n ? k + 1 : 0
                guard closed || k + 1 < n, let b = gaps[k].beside.first, gaps[k].beside.count == 1,
                      runs[k].truncationHeight == runs[j].truncationHeight, runs[k].truncationHeight.isFinite,
                      let end = runs[k].points.last, let start = runs[j].points.first else { return false }
                return turnsBack(at: b, below: runs[k].truncationHeight, across: [end] + gaps[k].points + [start])
            }
            // The runs joined across those gaps, each with the runs it was
            // made of; in order from the first run, or on a closed line whose
            // last gap joins, from the run after the last gap that does not.
            var placed: [(truncationHeight: Double, points: [P2<DomainSpace>], from: [Int], joined: Bool)] = []
            if closed, n > 0, joins.allSatisfy({ $0 }) {
                var points = (0..<n).flatMap { runs[$0].points + gaps[$0].points }
                points.append(points[0])
                placed = [(runs[0].truncationHeight, points, Array(0..<n), true)]
            } else {
                let first = closed && joins.last == true ? (joins.lastIndex(of: false)! + 1) % n : 0
                for m in 0..<n {
                    let k = (first + m) % n
                    if m > 0, joins[(k + n - 1) % n] {
                        placed[placed.count - 1].points += runs[k].points
                        placed[placed.count - 1].from.append(k)
                    } else {
                        placed.append((runs[k].truncationHeight, runs[k].points, [k], false))
                    }
                    if joins[k], m + 1 < n {
                        placed[placed.count - 1].points += gaps[k].points
                        placed[placed.count - 1].joined = true
                    }
                }
            }
            // The dropped vertices either side of a placed run, which its
            // ends are carried across onto the boundary.
            func gap(before k: Int) -> [P2<DomainSpace>] {
                closed ? gaps[(k + n - 1) % n].points : k == 0 ? head : gaps[k - 1].points
            }
            for run in placed where run.points.count >= 2 && run.truncationHeight.isFinite {
                let band = self.band(atX: run.points[0].x)
                var across = (start: gap(before: run.from[0]), end: gaps[run.from[run.from.count - 1]].points)
                /// Points placed on f, with the rings at their level.
                func placed(_ points: [P2<DomainSpace>]) -> [[P2<DomainSpace>]] {
                    let placed = refine.contours(.magnitude, run.truncationHeight, [points], .infinity)
                    return placed + (refine.rings?(run.truncationHeight, placed) ?? [])
                }
                var refined = placed(run.points)
                // Joined, it must come back placed, in its band, on its
                // level; where it does not, its runs are placed apart, as
                // they were, without the gaps.
                if run.joined, refined.isEmpty || !refined.joined().allSatisfy({
                    self.band(atX: $0.x) == band && abs(mag($0) - run.truncationHeight) <= 1e-6 * run.truncationHeight
                }) {
                    refined = run.from.flatMap { placed(runs[$0].points) }
                    across = ([], [])
                }
                for (index, piece) in refined.enumerated() where piece.count >= 2 {
                    var piece = piece
                    // A closed piece has no ends. The far end's index read
                    // after the near end has had its point on the boundary
                    // put before it.
                    for atStart in [true, false] where piece.first != piece.last {
                        let i = atStart ? 0 : piece.count - 1
                        if let q = onBoundaryTruncationCrossing(near: piece[i], magnitude: magnitude) {
                            piece[i] = q
                        } else if let b = boundary(beside: piece[i].x) ?? boundaries.first(where: { abs(piece[i].x - $0) <= 2 * dx }),
                                  let q = ontoBoundary(b, truncationHeight: run.truncationHeight, near: piece[i],
                                                       across: atStart ? (index == 0 ? across.start : [])
                                                                       : (index == refined.count - 1 ? across.end : [])) {
                            if atStart { piece.insert(q, at: 0) } else { piece.append(q) }
                            endsOnBoundary[b, default: []].append((run.truncationHeight, q))
                        }
                    }
                    out.append(piece.map { P3($0.x, $0.y, run.truncationHeight) })
                    if let band { pieces[band, default: []].append(piece) }
                }
            }
        }
        // The top of each riser, at f's height truncated by the greater
        // band: along the boundary wherever the lesser band's plateau holds
        // it, from one end of that band's truncation edge on it to the next,
        // or to the domain's edge, read at the middle of each stretch
        // between. In order along x, then y.
        let corner0 = g.position(x: 0, y: 0), corner1 = g.position(x: g.width - 1, y: g.height - 1)
        let (y0, y1) = (min(corner0.y, corner1.y), max(corner0.y, corner1.y))
        for b in boundaries {
            let left = truncation.height(atX: b, from: .left), right = truncation.height(atX: b, from: .right)
            let lesser = min(left, right), greater = max(left, right)
            guard lesser.isFinite, lesser < greater else { continue }
            let lows = (endsOnBoundary[b] ?? []).filter { $0.truncationHeight == lesser }.map(\.point.y).sorted()
            let stops = [y0] + lows.filter { $0 > y0 && $0 < y1 } + [y1]
            for (a, c) in zip(stops, stops.dropFirst()) where c > a && !(mag(P2(b, 0.5 * (a + c))) < lesser) {
                let a = P2<DomainSpace>(b, a), c = P2<DomainSpace>(b, c)
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
                riserTops.append((b, curve))
            }
        }
        return out
    }

    /// For a truncation edge vertex on the domain's edge, the truncation crossing between the
    /// two grid nodes beside it on that edge, by the top edge's bisection; nil
    /// for a vertex off the edge or with no crossing beside it.
    func onBoundaryTruncationCrossing(near p: P2<DomainSpace>,
                                      magnitude: ContourRefine.Magnitude) -> P2<DomainSpace>? {
        let g = height
        let eps = edgeTolerance
        let f = g.index(of: p)
        let onRow = abs(p.y - g.domain.imag.lo) <= eps || abs(p.y - g.domain.imag.hi) <= eps
        let onColumn = abs(p.x - g.domain.real.lo) <= eps || abs(p.x - g.domain.real.hi) <= eps
        guard onRow || onColumn else { return nil }
        func excess(_ q: P2<DomainSpace>) -> Double {
            magnitude(q) - truncationHeight(atX: q.x)
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
        return truncationCrossing(from: a, excess(a), to: b, excess(b), excess: excess)
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
