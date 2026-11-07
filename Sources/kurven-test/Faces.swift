import Foundation
import simd
import KurvenCore
import KurvenLandscape

// MARK: - faces
//
// A solid's faces tile its boundary: a point of it lies in exactly one face,
// or on a crease two of them share, and that face's map returns it; a point
// off it lies in none. What the boundary is comes from the drawing, never
// from the faces, on every tile: the footprint as visibility and the mesh
// test it, at the point in space (`Region.contains`); the top as the curves
// are tested against it, read through the tile under the point
// (`Heightfield.surfaceHeight(at:)`); a wall's top as its curtain draws it
// (`Mesh.wallCurtain`), or with f as its top edge does (the `wallEdges`
// layer); a riser as the drop between the top on either side of a band's
// boundary; and the normal as the facing reads it
// (`Heightfield.graphNormal(at:)`).

/// The parameter at which `copy` of `face` passes through `q`, if it does.
/// The graph and the plateaus lie over the domain, so a point's parameter is
/// its (x, y); every other face of a landscape or a polyhedron is flat with
/// an affine map, and a point's parameter is its projection onto the face's
/// plane. The map returning the point is what decides.
private func parameter(of q: P3<WorldSpace>, on face: Face, _ copy: Face.Copy) -> P2<ParamSpace>? {
    guard let back = copy.tile.inverse else { return nil }
    let local = back(q).v
    let p: P2<ParamSpace>
    switch face.id {
    case .graph, .plateau:
        p = P2(local.x, local.y)
    default:
        let p0 = P2<ParamSpace>(face.domain.u.range.lo, face.domain.v.range.lo)
        let jet = face.map(p0)
        let r = local - (jet.position - jet.du * p0.x - jet.dv * p0.y)
        let a = simd_dot(jet.du, jet.du), b = simd_dot(jet.du, jet.dv), c = simd_dot(jet.dv, jet.dv)
        let det = a * c - b * b
        guard det > 0 else { return nil }
        let ru = simd_dot(jet.du, r), rv = simd_dot(jet.dv, r)
        p = P2((c * ru - b * rv) / det, (a * rv - b * ru) / det)
    }
    let tolerance = 1e-9 * max(1, simd_length(local))
    guard copy.contains(p), simd_length(face.map(p).position - local) <= tolerance else { return nil }
    return p
}

/// Every copy of a face that holds a point, with the point's parameter there.
private func holders(of q: P3<WorldSpace>, in faces: [Face]) -> [(face: Face, copy: Face.Copy, p: P2<ParamSpace>)] {
    faces.flatMap { face in
        face.copies.compactMap { copy in parameter(of: q, on: face, copy).map { (face, copy, $0) } }
    }
}

/// The outward normal of a copy at a parameter, in space: zero where the
/// face bounds no solid.
private func outwardNormal(_ face: Face, _ copy: Face.Copy, _ p: P2<ParamSpace>) -> SIMD3<Double> {
    copy.place(face.map(p)).normal * (copy.outward ?? 0)
}

/// The tile under a world point as the drawing finds it: the one that
/// carries it back nearest the domain, inside it where any does, and the
/// point carried back (`Surface.tile(under:tiles:)`).
private func carriedBack(_ w: SIMD2<Double>, _ l: Landscape) -> (q: SIMD2<Double>, back: Affine2) {
    guard l.tiles.count > 1 else { return (w, .identity) }
    let real = l.surface.domain.real, imag = l.surface.domain.imag
    var best: (q: SIMD2<Double>, back: Affine2) = (w, .identity), nearest = Double.infinity
    for tile in l.tiles {
        guard let back = tile.inverse else { continue }
        let q = back(P3(w.x, w.y, 0)).xy.v
        let d = max(min(real.lo, real.hi) - q.x, 0) + max(q.x - max(real.lo, real.hi), 0)
            + max(min(imag.lo, imag.hi) - q.y, 0) + max(q.y - max(imag.lo, imag.hi), 0)
        if d < nearest { nearest = d; best = (q, back) }
        if d == 0 { break }
    }
    return best
}

/// Edge `k`'s curtain as `Mesh.walls(of:surface:base:tiles:)` builds it:
/// its top's vertices, and all of them, the top's then the bottom's.
private func curtain(_ perimeter: BoundaryPerimeter, _ k: Int, surface: Surface, base: Double,
                     tiles: [Affine2]) -> (top: [P3<WorldSpace>], all: [P3<WorldSpace>]) {
    let e = perimeter.edges[k]
    let all = Mesh.wallCurtain(from: e.start, to: e.end, inward: perimeter.inward(k), samples: e.density,
                               surface: surface, base: base, tiles: tiles).vertices
    return (Array(all.prefix(all.count / 2)), all)
}

/// Where a chord on the ground from `a` to `b` passes from one tile to
/// another, found by bisection on the tile under it a hair `inward`, and
/// whether `top`, read a hair either side of that point along the chord,
/// differs by more than it moves there: a drop between two copies.
private func dropAcross(_ a: SIMD2<Double>, _ b: SIMD2<Double>, inward: SIMD2<Double>, _ l: Landscape,
                        top: (SIMD2<Double>) -> Double) -> (at: SIMD2<Double>, drops: Bool) {
    // In from the wall by far less than along it, so a read either side
    // stays on its side of the boundary whatever the angle between.
    let size = max(abs(l.surface.domain.real.length), abs(l.surface.domain.imag.length)), e = 1e-9 * size, i = 1e-3 * e * inward
    func back(_ t: Double) -> Affine2 { carriedBack(a + (b - a) * t + i, l).back }
    let first = back(1.0 / 32)
    let bracket = bisect(1.0 / 32, 31.0 / 32) { back($0) == first }
    let at = a + (b - a) * (0.5 * (bracket.holds + bracket.fails)), d = simd_normalize(b - a)
    let before = top(at - e * d + i), after = top(at + e * d + i)
    let slide = max(abs(top(at - 2 * e * d + i) - before), abs(top(at + 2 * e * d + i) - after))
    return (at, abs(before - after) > 4 * slide + 1e-9 * max(1, abs(before)))
}

/// Whether a point of a tile boundary, `n` the boundary's normal, lies in
/// the footprint: held a hair either side of the boundary, or on the
/// perimeter, exactly where the edge runs along an axis; where it slants,
/// within ten times that hair, since the riser ends where a line a hair
/// either side of the boundary crosses it.
private func inFootprint(_ p: SIMD2<Double>, _ region: Region, n: SIMD2<Double>, size: Double) -> Bool {
    guard case .inside(let perimeter) = region else { return true }
    let side = 1e-9 * size
    if region.contains(P2(p - side * n)) && region.contains(P2(p + side * n)) { return true }
    return perimeter.edges.contains { e in
        let a = e.start.v, b = e.end.v
        if a.x == b.x { return p.x == a.x && p.y >= min(a.y, b.y) && p.y <= max(a.y, b.y) }
        if a.y == b.y { return p.y == a.y && p.x >= min(a.x, b.x) && p.x <= max(a.x, b.x) }
        let d = b - a, t = max(0, min(1, simd_dot(p - a, d) / simd_length_squared(d)))
        return simd_length(p - (a + t * d)) <= 10 * side
    }
}

/// A landscape as the faces are built from it, and as the drawing reads it.
private struct Landscape {
    let name: String
    let surface: Surface
    let tiles: [Affine2]
    let region: Region
    let walls: Walls
    let refine: ContourRefine?
    let faces: [Face]
    /// Whether two copies meet at a tile boundary with different tops where the
    /// footprint holds both sides: the solid drops there.
    var dropsAtATileBoundary = false
    var heightfield: Heightfield {
        Heightfield(surface: surface, occluder: .empty, tiles: tiles, region: region, step: 1,
                    refine: refine, walls: walls)
    }
    /// Whether any sample of the grid is under the truncation: a landscape
    /// all plateau has no graph to contour or clip.
    var hasGraph: Bool {
        let g = surface.height
        return (0..<g.width).contains { i in
            let h = surface.truncationHeight(atX: g.position(x: i, y: 0).x)
            return (0..<g.height).contains { Double(g[i, $0]) < h }
        }
    }

    init(_ name: String, surface: Surface, tiles: [Affine2] = [.identity], region: Region = .full,
         walls: Walls = .none, refine: ContourRefine? = nil, faces: [Face]? = nil) {
        self.name = name; self.surface = surface; self.tiles = tiles; self.region = region
        self.walls = walls; self.refine = refine
        self.faces = faces ?? surface.faces(tiles: tiles, region: region, walls: walls, refine: refine)
    }
}

/// What the gate counts on one landscape: points asked, those in the wrong
/// number of faces, those whose face turns inward, those on the graph whose
/// normal is not the facing's, and the first failure of each kind.
private struct Tally {
    var sampled = 0, wrong = 0, inward = 0, onGraph = 0, normals = 0, skewed = 0, stations = 0, offTop = 0
    var boundaryPoints = 0, drops = 0
    var examples: [String: String] = [:]
    mutating func fail(_ kind: String, _ what: String) { if examples[kind] == nil { examples[kind] = what } }
}

private func tiling(_ l: Landscape) -> Tally {
    let surface = l.surface, truncation = surface.truncation, heightfield = l.heightfield
    let onF = l.refine?.magnitude != nil
    let real = surface.domain.real, imag = surface.domain.imag
    let (lo, hi) = (min(real.lo, real.hi), max(real.lo, real.hi))
    let size = max(abs(real.length), abs(imag.length))
    let cell = SIMD2(abs(real.length) / Double(surface.height.width - 1),
                     abs(imag.length) / Double(surface.height.height - 1))
    var t = Tally()

    /// `q` lies in `expected` faces, or within `slack` of that: one on a
    /// crease, two at a corner where three faces meet. Where it lies in one,
    /// `out` says whether that face's normal points out.
    func expect(_ q: P3<WorldSpace>, _ expected: Int, slack: Int = 0,
                out: ((Face, SIMD3<Double>) -> Bool)? = nil) {
        t.sampled += 1
        let found = holders(of: q, in: l.faces)
        if abs(found.count - expected) > slack {
            t.wrong += 1
            t.fail("wrong", "\(q.v) in \(found.map(\.face.id)), not \(expected) face(s)")
            return
        }
        guard let out, found.count == 1, let (face, copy, p) = found.first else { return }
        if !out(face, outwardNormal(face, copy, p)) {
            t.inward += 1
            t.fail("inward", "\(face.id) at \(q.v) faces in")
        }
    }

    /// How many tiles' images of the domain hold a point of the ground. Only
    /// a point one tile holds is asked: the drawing reads a point two hold,
    /// on a tile boundary or an overlap, through whichever comes first.
    func cover(_ w: P3<WorldSpace>) -> Int {
        l.tiles.filter { tile in
            guard let back = tile.inverse else { return false }
            let p = back(w)
            return (lo...hi).contains(p.x)
                && (min(imag.lo, imag.hi)...max(imag.lo, imag.hi)).contains(p.y)
        }.count
    }

    /// |f| at a point of the domain, untruncated: where the graph goes on
    /// above the truncation, and how near a point is to the truncation edge.
    func magnitude(_ x: Double, _ y: Double) -> Double {
        let p = P2<DomainSpace>(x, y)
        let u = l.refine?.magnitude?(p) ?? surface.magnitude(at: p)
        return u.isNaN ? .infinity : u
    }

    // The top: over every point of the footprint, at the height the curves
    // are tested against, the graph or a plateau, on every tile; nothing
    // just above it or just under it, nor over the ground the footprint
    // leaves out. Outward is up, and on the graph it is the facing's normal.
    for tile in l.tiles {
        for j in 0..<47 {
            for i in 0..<83 {
                let x = real.lo + real.length * (Double(i) + 0.37) / 83
                let y = imag.lo + imag.length * (Double(j) + 0.61) / 47
                let ground = tile(P3(x, y, 0))
                guard cover(ground) == 1 else { continue }
                let z = heightfield.surfaceHeight(at: ground.xy)
                guard z.isFinite else { continue }
                let held = l.region.contains(ground.xy)
                let m = magnitude(x, y), h = truncation.height(atX: x)
                let onCrease = abs(m - h) <= 1e-9 * max(1, abs(h))
                expect(P3(ground.x, ground.y, z), held ? 1 : 0, slack: onCrease ? 1 : 0) { face, n in
                    guard n.z > 0 else { return false }
                    if face.id == .graph { t.onGraph += 1 }
                    // The facing's differences reach a cell out along the
                    // grid's axes as the tile turns them, and where they
                    // reach ground another tile holds, or is nearest, they
                    // read that one, across a tile boundary the copies need
                    // not meet smoothly at.
                    let reach = [(tile.a * cell.x, tile.c * cell.x), (-tile.a * cell.x, -tile.c * cell.x),
                                 (tile.b * cell.y, tile.d * cell.y), (-tile.b * cell.y, -tile.d * cell.y)]
                    if face.id == .graph, reach.allSatisfy({ r in
                        let w = P3<WorldSpace>(ground.x + r.0, ground.y + r.1, 0)
                        return cover(w) <= 1 && carriedBack(w.xy.v, l).back == tile.inverse
                    }) {
                        t.normals += 1
                        let facing = onF ? heightfield.graphNormal(at: ground.xy) : heightfield.normal(at: ground.xy)
                        // Directions: a copy's du × dv is the normal scaled
                        // by its tile's determinant.
                        let off = simd_length(simd_normalize(n) - simd_normalize(facing))
                        if !(off <= 1e-9) {
                            t.skewed += 1
                            t.fail("skewed", "the graph's normal at \(ground.v) is \(n), the facing's \(facing)")
                        }
                    }
                    return true
                }
                let off = 1e-3 * max(1, abs(z))
                expect(P3(ground.x, ground.y, z + off), 0)
                expect(P3(ground.x, ground.y, z - off), 0)
                // The sheet that is not drawn here, over it: the graph above
                // a plateau, a plateau's height above the graph.
                let over = max(m, h)
                if over.isFinite, !onCrease { expect(P3(ground.x, ground.y, over), 0) }
            }
        }
    }

    // The band boundaries inside the domain, and one on its far edge, which
    // is no riser's: the band beyond starts outside the domain.
    var boundaries: [Double] = []
    var farEdge = false
    if case .realBands(let bands, _) = truncation {
        boundaries = Array(Set(bands.map(\.below).filter { $0 > lo && $0 < hi })).sorted()
        farEdge = bands.contains { $0.below == hi }
    }
    let side = 1e-9 * size

    // The walls: from the base to the top their curtain draws, or with f
    // their top edge, along each edge of the perimeter, a crease where two
    // meet; nothing under the base or over the top. Outward leaves the
    // perimeter. And the top is the solid's, read just beside the foot in
    // the footprint, so no wall stands over the solid as a fin nor short of
    // it: beside is along the wall's inward normal, and at the wall's ends
    // along the wall as well, so a foot on a band boundary is read on its
    // footprint's side, and an end on one in the wall's own band.
    var wallFeet: [(a: P2<WorldSpace>, b: P2<WorldSpace>)] = []
    if case .perimeter(let perimeter, let base) = l.walls {
        let corners = perimeter.edges.map(\.start)
        let eps = 1e-6 * size
        let area = corners.indices.reduce(0.0) { a, k in
            let p = corners[k], q = corners[(k + 1) % corners.count]
            return a + p.x * q.y - q.x * p.y
        }
        // The curtain as the drawing builds it, at 41 stations an edge.
        let dense = BoundaryPerimeter(edges: perimeter.edges.map { PerimeterEdge(start: $0.start, end: $0.end, density: 41) })
        let tops: [[P3<WorldSpace>]] = perimeter.edges.indices.map { k in
            if onF {
                let context = Surface.HatchContext(perimeter: perimeter, region: .full, tiles: l.tiles)
                let drawn = surface.hatch(.wallEdges(edges: [k], step: 1, base: base), in: context, refine: l.refine)
                return drawn.map { Array($0[path: 0]) } ?? []
            }
            return curtain(dense, k, surface: surface, base: base, tiles: l.tiles).top
        }
        // Where two walls meet with different tops, at a corner a riser
        // ends at, the higher stands over the lower.
        var cornerTop: [P2<WorldSpace>: Double] = [:]
        for (edge, stations) in zip(perimeter.edges, tops) {
            for v in stations where v.xy == edge.start || v.xy == edge.end {
                cornerTop[v.xy] = max(cornerTop[v.xy] ?? -.infinity, v.z)
            }
        }
        for (k, edge) in perimeter.edges.enumerated() {
            wallFeet.append((edge.start, edge.end))
            let stations = tops[k]
            let d = simd_normalize(edge.end.v - edge.start.v)
            let inward = area > 0 ? SIMD2(-d.y, d.x) : SIMD2(d.y, -d.x)
            for v in stations {
                let top = v.z
                guard top.isFinite, top > base else { continue }
                let atStart = simd_length(v.xy.v - edge.start.v) < 1e-12, atEnd = simd_length(v.xy.v - edge.end.v) < 1e-12
                // Where the wall crosses a band boundary on f its top steps:
                // two vertices at one point, each at its own band's height.
                // Between the two the wall stands on the boundary's line, a
                // crease with the riser where there is one, and is read either
                // side of the line, by a rounding.
                let atFoot = stations.filter { simd_length($0.xy.v - v.xy.v) <= 1e-9 * size }.map(\.z)
                let lower = atFoot.min() ?? top
                // Where the wall crosses a tile boundary, two copies meet: a
                // riser there meets the wall in a crease, and at a copy's top
                // the copy's face meets both, three faces at a point.
                let crossesTileBoundary = carriedBack(v.xy.v + 1e-6 * size * d, l).back != carriedBack(v.xy.v - 1e-6 * size * d, l).back
                for j in 0..<7 {
                    let z = base + (top - base) * (Double(j) + 0.5) / 7
                    let atACopysTop = crossesTileBoundary && atFoot.contains { abs($0 - z) <= 1e-9 * max(1, abs(z)) }
                    let slack = atACopysTop ? 2 : atStart || atEnd || z > lower || crossesTileBoundary ? 1 : 0
                    expect(P3(v.x, v.y, z), 1, slack: slack) { face, n in
                        // At a corner, the other wall there is asked from its
                        // own stations.
                        guard case .wall(let e) = face.id, e == k || !(atStart || atEnd) else { return true }
                        // Asked a little way in from a corner, where either
                        // side of the wall may be the other wall's.
                        let w = v.xy.v + (atStart ? d : atEnd ? -d : .zero) * (10 * eps)
                        let h = simd_normalize(SIMD2(n.x, n.y)) * eps
                        return !pointInPolygon(P2(w + h), corners) && pointInPolygon(P2(w - h), corners)
                    }
                }
                // Nothing over the top, nor over the solid inside the footprint
                // within a few times the distance a face is matched to a point
                // at (`parameter(of:on:_:)`), read along the inward normal, and
                // at an end along the wall too, never on the wall's line: a
                // riser stands a hair inside a wall that leaves its boundary at
                // a slant, and a read on the line of a wall on a boundary is
                // the band across it.
                let match = 4e-9 * max(1, simd_length(v.v))
                let near = ([inward] + (atStart ? [inward + d] : atEnd ? [inward - d] : [])).map { v.xy.v + match * $0 }
                    .filter { l.region.contains(P2($0)) }.map { heightfield.surfaceHeight(at: P2($0)) }.filter(\.isFinite)
                let over = max(atFoot.max() ?? top, cornerTop[v.xy] ?? top, near.max() ?? top)
                expect(P3(v.x, v.y, over + 1e-3 * max(1, abs(over)) + 0.1 * (over - base)), 0)
                expect(P3(v.x, v.y, base - 0.1 * (top - base)), 0)
                t.stations += 1
                // Where it steps, the solid's top either side along the wall,
                // on its line: the boundary crosses it at a slant, and a step
                // in as well can land both reads on one side. Elsewhere beside
                // the foot in the band the wall reads there: a billionth along
                // `toward`, or half the foot's distance from a band boundary
                // it is nearer than that, so the read stays in the wall's
                // band where the wall runs a hair off the boundary; and at an
                // end on a boundary of a wall that leaves it, across the
                // boundary into the band the wall runs on in, its own.
                var towards = atFoot.count == 1 ? [inward + (atStart ? d : atEnd ? -d : .zero)] : [d, -d]
                var reach = side
                if atFoot.count == 1 {
                    let back = carriedBack(v.xy.v + side * towards[0], l).back
                    let q = back(P3(v.x, v.y, 0)).xy.v
                    let along = SIMD2(back.a * d.x + back.b * d.y, back.c * d.x + back.d * d.y)
                    if let b = boundaries.min(by: { abs(q.x - $0) < abs(q.x - $1) }) {
                        let apart = abs(q.x - b)
                        if apart > 1e-12 * size, apart < side {
                            reach = 0.5 * apart / simd_length(towards[0])
                        } else if apart <= 1e-12 * size, atStart || atEnd, abs(along.x) > 1e-12 * simd_length(along),
                                  let forward = back.inverse {
                            let own = (atStart ? along.x : -along.x) > 0 ? 1.0 : -1.0
                            towards = [SIMD2(forward.a * own, forward.c * own)]
                        }
                    }
                }
                let reads = towards.map { toward -> (solid: Double, slide: Double) in
                    let solid = heightfield.surfaceHeight(at: P2(v.xy.v + reach * toward))
                    return (solid, abs(heightfield.surfaceHeight(at: P2(v.xy.v + 2 * reach * toward)) - solid))
                }
                // Not where the wall crosses a tile boundary, which two
                // copies need not meet smoothly at: a vertex there is read
                // through either.
                if reads.allSatisfy(\.solid.isFinite), !crossesTileBoundary,
                   !reads.contains(where: { abs(top - $0.solid) <= 4 * $0.slide + 1e-9 * max(1, abs($0.solid)) }) {
                    t.offTop += 1
                    t.fail("offTop", "wall \(k)'s top at \(v.xy.v) is \(top), the solid's beside it \(reads.map(\.solid))")
                }
            }
        }
    }
    /// Whether a wall stands on a point of the ground, as near as a face is
    /// matched to a point (`parameter(of:on:_:)`).
    func wallStands(_ w: P2<WorldSpace>) -> Bool {
        wallFeet.contains { a, b in
            let d = b.v - a.v
            let s = max(0, min(1, simd_dot(w.v - a.v, d) / simd_length_squared(d)))
            return simd_length(w.v - (a.v + s * d)) <= 4e-9 * max(1, simd_length(w.v))
        }
    }

    // The risers: where the footprint holds both sides of a band boundary,
    // the solid drops from the top on one side to the top on the other,
    // read just either side of it: every point between in one face, a
    // riser; nothing over the higher, nor inside the solid under the lower.
    // Outward is toward the lower side. Where it holds one side and a wall
    // stands on the boundary, the wall rises to the top on that side and no
    // higher, and there is no riser: nothing over that top, nor up to the
    // top across the boundary.
    for b in boundaries {
        for tile in l.tiles {
            for j in 0..<47 {
                let y = imag.lo + imag.length * (Double(j) + 0.53) / 47
                let foot = tile(P3(b, y, 0))
                /// The top just off the boundary, `dx` along x; nil off the
                /// domain, or off the footprint where it must be `held`.
                func top(_ dx: Double, held: Bool = true) -> Double? {
                    let x = b + dx
                    guard x >= lo, x <= hi else { return nil }
                    let w = tile(P3(x, y, 0))
                    guard cover(w) == 1, !held || l.region.contains(w.xy) else { return nil }
                    let z = heightfield.surfaceHeight(at: w.xy)
                    return z.isFinite ? z : nil
                }
                let left = top(-side), right = top(side)
                // How far f alone moves across the boundary, which is no drop.
                let slide = abs(magnitude(b + side, y) - magnitude(b - side, y))
                switch (left, right) {
                case let (l?, r?):
                    let span = (lo: min(l, r), hi: max(l, r)), lowerSide: Double = l > r ? 1 : -1
                    let outward = SIMD2(tile.a * lowerSide, tile.c * lowerSide)
                    if span.hi - span.lo > 4 * slide + 1e-9 * max(1, abs(span.hi)) {
                        for k in 0..<7 {
                            let z = span.lo + (span.hi - span.lo) * (Double(k) + 0.5) / 7
                            expect(P3(foot.x, foot.y, z), 1) { face, n in
                                guard case .riser = face.id else { return true }
                                return simd_dot(SIMD2(n.x, n.y), outward) > 0
                            }
                        }
                    }
                    expect(P3(foot.x, foot.y, span.hi + 1e-3 * max(1, abs(span.hi))), 0)
                    if !wallStands(foot.xy) { expect(P3(foot.x, foot.y, span.lo - 1e-3 * max(1, abs(span.lo))), 0) }
                case let (h?, nil), let (nil, h?):
                    guard wallStands(foot.xy) else { continue }
                    expect(P3(foot.x, foot.y, h + 1e-3 * max(1, abs(h))), 0)
                    if let across = left == nil ? top(-side, held: false) : top(side, held: false),
                       across - h > 4 * slide + 1e-9 * max(1, abs(across)) {
                        expect(P3(foot.x, foot.y, 0.5 * (h + across)), 0)
                    }
                default:
                    continue
                }
            }
        }
    }

    // The tile boundaries, where two copies meet edge to edge, found on the
    // drawing: a point of a side of a tile's image whose ground a billionth
    // of the domain either side one tile holds each, two different ones.
    // Where the footprint holds both sides and the top read there, through
    // the tile under each, differs by more than f alone moves, the solid
    // drops: every point between in one face, a riser facing the lower side;
    // nothing over the higher, nor inside the solid under the lower. Where
    // they agree, nothing over the top. Where it holds one side and a wall
    // stands on the tile boundary, the wall rises to that side's top and no
    // higher: nothing over it, nor up to the top across the boundary.
    let (ylo, yhi) = (min(imag.lo, imag.hi), max(imag.lo, imag.hi))
    let domainCorners = [SIMD2(lo, ylo), SIMD2(hi, ylo), SIMD2(hi, yhi), SIMD2(lo, yhi)]
    for tile in l.tiles where tile.inverse != nil && l.tiles.count > 1 {
        for e in 0..<4 {
            let a = tile(P3(domainCorners[e].x, domainCorners[e].y, 0)).xy.v
            let b = tile(P3(domainCorners[(e + 1) % 4].x, domainCorners[(e + 1) % 4].y, 0)).xy.v
            let u = simd_normalize(b - a), n = SIMD2(-u.y, u.x)
            for j in 0..<47 {
                let w = a + (b - a) * ((Double(j) + 0.53) / 47)
                /// The top `side` off the tile boundary toward `sign`, and
                /// the tile under it; nil where no one tile holds the ground,
                /// or off the footprint where it must be `held`.
                func top(_ sign: Double, held: Bool = true) -> (z: Double, back: Affine2, slide: Double)? {
                    let g = P3<WorldSpace>(w.x + sign * side * n.x, w.y + sign * side * n.y, 0)
                    guard cover(g) == 1, !held || l.region.contains(g.xy) else { return nil }
                    let z = heightfield.surfaceHeight(at: g.xy)
                    let further = heightfield.surfaceHeight(at: P2(w + 2 * sign * side * n))
                    return z.isFinite ? (z, carriedBack(g.xy.v, l).back, abs(further - z)) : nil
                }
                guard let l0 = top(-1, held: false), let r0 = top(1, held: false), l0.back != r0.back else { continue }
                let foot = P2<WorldSpace>(w)
                switch (top(-1), top(1)) {
                case let (left?, right?):
                    t.boundaryPoints += 1
                    let span = (lo: min(left.z, right.z), hi: max(left.z, right.z))
                    let towardLower = left.z < right.z ? -n : n
                    if span.hi - span.lo > 4 * max(left.slide, right.slide) + 1e-9 * max(1, abs(span.hi)) {
                        t.drops += 1
                        for k in 0..<7 {
                            let z = span.lo + (span.hi - span.lo) * (Double(k) + 0.5) / 7
                            expect(P3(w.x, w.y, z), 1) { face, normal in
                                guard case .tileBoundaryRiser = face.id else { return false }
                                return simd_dot(SIMD2(normal.x, normal.y), towardLower) > 0
                            }
                        }
                        if !wallStands(foot) { expect(P3(w.x, w.y, span.lo - 1e-3 * max(1, abs(span.lo))), 0) }
                    }
                    expect(P3(w.x, w.y, span.hi + 1e-3 * max(1, abs(span.hi))), 0)
                case let (h?, nil), let (nil, h?):
                    guard wallStands(foot) else { continue }
                    expect(P3(w.x, w.y, h.z + 1e-3 * max(1, abs(h.z))), 0)
                    let across = h.back == l0.back ? r0.z : l0.z
                    if across - h.z > 4 * h.slide + 1e-9 * max(1, abs(across)) {
                        expect(P3(w.x, w.y, 0.5 * (h.z + across)), 0)
                    }
                default:
                    continue
                }
            }
        }
    }
    // A band boundary on the far edge: the landscape ends with the band
    // before it. With no wall on the edge nothing stands there, no riser up
    // from the band beyond to the top inside; a wall there is asked above.
    if farEdge {
        for tile in l.tiles {
            for j in 0..<47 {
                let y = imag.lo + imag.length * (Double(j) + 0.53) / 47
                let foot = tile(P3(hi, y, 0)), inside = tile(P3(hi - side, y, 0))
                guard cover(foot) == 1, !wallStands(foot.xy), l.region.contains(inside.xy) else { continue }
                let top = heightfield.surfaceHeight(at: inside.xy)
                let beyond = min(magnitude(hi, y), truncation.height(atX: hi))
                if top.isFinite, top - beyond > 1e-6 * max(1, abs(top)) {
                    expect(P3(foot.x, foot.y, 0.5 * (top + beyond)), 0)
                }
            }
        }
    }
    return t
}

/// The landscapes the faces suites hold to the drawing: the contract
/// fixtures, the hatch fixture under its truncations, banded on notched
/// footprints, with a band boundary on its far edge, in four reflected tiles,
/// and Γ banded on f.
private func faceLandscapes() throws -> [Landscape] {
    var landscapes: [Landscape] = []
    // The contract fixtures: two tiles, the second a reflection; mesh
    // walls, a footprint, and bands that lie beyond the domain. Through
    // the scene where the bundle has a camera, so the walls are the ones a
    // bundle's scene carries.
    for url in Fixtures.contractBundles {
        let bundle = try KurvenBundle.read(at: url)
        let o = bundle.manifest.occluder
        let geometry: SceneGeometry = bundle.manifest.presets.first.map {
            Scene(bundle: bundle, preset: $0).geometry
        } ?? .heightfield(Heightfield(surface: bundle.surface, occluder: .empty, tiles: o.tiles,
                                      region: o.region, step: 1, refine: bundle.refine, walls: o.walls))
        landscapes.append(Landscape(url.lastPathComponent, surface: bundle.surface, tiles: o.tiles,
                                    region: o.region, walls: o.walls, refine: bundle.refine,
                                    faces: geometry.faces))
    }
    // The hatch fixture under its two truncations, its bands inside the
    // domain, so it has risers.
    let index = try Fixtures.json("hatch/index.json")
    let shape = try index.value("shape", "hatch").object("hatch.shape")
    let grid = Grid2D(width: try shape.int("nx", "hatch.shape"), height: try shape.int("ny", "hatch.shape"),
                      domain: try Domain(json: index.value("domain", "hatch")),
                      values: try NPY.read(contentsOf: Fixtures.url("hatch/height.npy")).floats())
    let perimeter = try BoundaryPerimeter(json: index.value("perimeter", "hatch"))
    let truncations = try index.value("truncation", "hatch").object("hatch.truncations")
    for name in ["uniform", "bands"] {
        let surface = Surface(height: grid, phase: nil,
                              truncation: try Truncation(json: truncations.value(name, "hatch.truncations")))
        landscapes.append(Landscape("the hatch fixture, \(name)", surface: surface,
                                    walls: .perimeter(perimeter, base: 0)))
    }
    // Banded on footprints with notches cut from the front and walls on
    // them: one clear of the bands' boundaries, and two whose sides stand
    // on the boundaries, under bands low enough that |f| rises through
    // them there. Between them they hold the footprint on the lower
    // band's side of a boundary and on the higher's, left of it and right.
    let banded = Surface(height: grid, phase: nil,
                         truncation: try Truncation(json: truncations.value("bands", "hatch.truncations")))
    let low = Surface(height: grid, phase: nil,
                      truncation: .realBands([RealBand(below: -1, height: 0.3), RealBand(below: 0.4, height: 0.7)],
                                             beyond: 0.45))
    func footprint(_ corners: [(Double, Double)], density: Int = 9) -> BoundaryPerimeter {
        BoundaryPerimeter(edges: corners.indices.map { k in
            let a = corners[k], b = corners[(k + 1) % corners.count]
            return PerimeterEdge(start: P2(a.0, a.1), end: P2(b.0, b.1), density: density)
        })
    }
    let notches: [(String, Surface, BoundaryPerimeter)] = [
        ("bands, on a notch from Re z = 0.5 to 1.5", banded,
         footprint([(-2, -1), (0.5, -1), (0.5, 0), (1.5, 0), (1.5, -1), (2.5, -1), (2.5, 1.5), (-2, 1.5)])),
        ("low bands, on a notch from one boundary to the other", low,
         footprint([(-2, -1), (-1, -1), (-1, 0.6), (0.4, 0.6), (0.4, -1), (2.5, -1), (2.5, 1.5), (-2, 1.5)])),
        ("low bands, on notches from the left edge to one boundary and from the other to 1.5", low,
         footprint([(-1, -1), (0.4, -1), (0.4, 0.6), (1.5, 0.6), (1.5, -1), (2.5, -1), (2.5, 1.5), (-2, 1.5),
                    (-2, 0.6), (-1, 0.6)])),
    ]
    for (name, surface, notched) in notches {
        landscapes.append(Landscape("the hatch fixture, \(name)", surface: surface,
                                    region: .inside(notched), walls: .perimeter(notched, base: -0.5)))
    }
    // A band's boundary on the domain's far edge, Re z = 2.5: the band
    // beyond starts outside the domain, far below the band before it, which
    // the right wall reads.
    let edgeBand = Surface(height: grid, phase: nil,
                           truncation: .realBands([RealBand(below: -1, height: 1.2), RealBand(below: 2.5, height: 2.5)],
                                                  beyond: 0.05))
    landscapes.append(Landscape("the hatch fixture, a band boundary on the far edge", surface: edgeBand,
                                walls: .perimeter(perimeter, base: 0)))
    landscapes.append(Landscape("the hatch fixture, a band boundary on the far edge, no walls", surface: edgeBand))
    // A wall on the image of the boundary Re z = 0.4 in a second tile,
    // reflected x ↦ 5 - x, where it stands at 4.6, or turned a quarter, where
    // it runs along Im z = 0.7: the foot carried back lands a rounding off
    // 0.4, and the wall reads its footprint's band all the same, on either
    // side of it.
    let reflected = Affine2(a: -1, b: 0, tx: 5, c: 0, d: 1, ty: 0)
    let turned = Affine2(a: 0, b: -1, tx: 10, c: 1, d: 0, ty: 0.3)
    // The same two written from an angle, as a manifest that turns a tile
    // by one would write them: cos(π/2) is 6.1e-17, not 0, and sin(π)
    // 1.2e-16, so a wall on the boundary's image carries back a rounding off
    // standing on it.
    let reflectedByAngle = Affine2(a: cos(.pi), b: sin(.pi), tx: 5, c: sin(.pi), d: -cos(.pi), ty: 0)
    let turnedByAngle = Affine2(a: cos(.pi / 2), b: -sin(.pi / 2), tx: 10, c: sin(.pi / 2), d: cos(.pi / 2), ty: 0.3)
    for (name, tile, corners) in [
        ("reflected, the footprint on the higher band's side", reflected, [(4.6, -1.0), (5.9, -1.0), (5.9, 1.5), (4.6, 1.5)]),
        ("reflected, the footprint on the lower band's side", reflected, [(3.0, -1.0), (4.6, -1.0), (4.6, 1.5), (3.0, 1.5)]),
        ("turned, the footprint on the higher band's side", turned, [(8.6, -0.5), (10.9, -0.5), (10.9, 0.7), (8.6, 0.7)]),
        ("turned, the footprint on the lower band's side", turned, [(8.6, 0.7), (10.9, 0.7), (10.9, 2.7), (8.6, 2.7)]),
        ("reflected by an angle, the footprint on the higher band's side", reflectedByAngle,
         [(4.6, -1.0), (5.9, -1.0), (5.9, 1.5), (4.6, 1.5)]),
        ("reflected by an angle, the footprint on the lower band's side", reflectedByAngle,
         [(3.0, -1.0), (4.6, -1.0), (4.6, 1.5), (3.0, 1.5)]),
        ("turned by an angle, the footprint on the higher band's side", turnedByAngle,
         [(8.6, -0.5), (10.9, -0.5), (10.9, 0.7), (8.6, 0.7)]),
        ("turned by an angle, the footprint on the lower band's side", turnedByAngle,
         [(8.6, 0.7), (10.9, 0.7), (10.9, 2.7), (8.6, 2.7)]),
    ] {
        let walled = footprint(corners)
        landscapes.append(Landscape("the hatch fixture, low bands, a wall on a boundary's image in a tile \(name)",
                                    surface: low, tiles: [.identity, tile], region: .inside(walled),
                                    walls: .perimeter(walled, base: -0.5)))
    }
    // A second tile sheared and stretched, (x, y) ↦ (x + y/2 + 5, x/5 + 2y):
    // the facing's slope comes back through the transpose of the tile's
    // inverse, which is not the tile.
    landscapes.append(Landscape("the hatch fixture, uniform, a second tile sheared and stretched",
                                surface: Surface(height: grid, phase: nil, truncation: .uniform(2)),
                                tiles: [.identity, Affine2(a: 1, b: 0.5, tx: 5, c: 0.2, d: 2, ty: 0)]))
    // The notch from one boundary to the other in a tile turned a quarter,
    // the footprint the notched one's image: the walls on the boundaries'
    // images run along Re z, and at a corner where a riser ends the wall
    // that leaves the boundary is the one along Im z.
    let notched = [(-2.0, -1.0), (-1, -1), (-1, 0.6), (0.4, 0.6), (0.4, -1), (2.5, -1), (2.5, 1.5), (-2, 1.5)]
        .map { turned(P3($0.0, $0.1, 0)) }.map { ($0.x, $0.y) }
    landscapes.append(Landscape("the hatch fixture, low bands, on a notch from one boundary to the other, in a tile turned",
                                surface: low, tiles: [.identity, turned], region: .inside(footprint(notched)),
                                walls: .perimeter(footprint(notched), base: -0.5)))
    // Reflected across both of the domain's far edges into four tiles,
    // three of them reflected or turned: under the low bands, on a
    // footprint across all four with walls on it; and uniform, on all
    // of them.
    let reflections = [Affine2.identity, Affine2(a: -1, b: 0, tx: 5, c: 0, d: 1, ty: 0),
                       Affine2(a: 1, b: 0, tx: 0, c: 0, d: -1, ty: 3), Affine2(a: -1, b: 0, tx: 5, c: 0, d: -1, ty: 3)]
    let across = footprint([(-1.5, -0.5), (5.5, -0.8), (6.6, 1.2), (6.2, 3.6), (1.0, 3.3), (-1.8, 2.0)])
    landscapes.append(Landscape("the hatch fixture, low bands, in four tiles, on a footprint across them",
                                surface: low, tiles: reflections, region: .inside(across),
                                walls: .perimeter(across, base: -0.5)))
    landscapes.append(Landscape("the hatch fixture, uniform, in four tiles",
                                surface: Surface(height: grid, phase: nil, truncation: .uniform(2)),
                                tiles: reflections))
    // A wall on the boundary between two tiles, the second beside the
    // first, (x, y) ↦ (x + 1, y), or turned a quarter, (x, y) ↦ (1 + y, 1 -
    // x): |f| = 3(2 Re z - 1)² + 1/5 on the unit square, at 1 | 2 across Re z
    // = 1/2, so where the tile boundary is the first tile's right edge the
    // band there is 2 and the second's 1 or its own. On a footprint either
    // side of the tile boundary, the wall on it reads the tile the footprint
    // is in.
    let parabolaGrid = Grid2D(width: 21, height: 21, domain: Domain(real: Interval(lo: 0, hi: 1), imag: Interval(lo: 0, hi: 1)),
                          values: (0..<441).map { k in Float(3 * pow(2 * Double(k % 21) / 20 - 1, 2) + 0.2) })
    let parabola = Surface(height: parabolaGrid, phase: nil, truncation: .realBands([RealBand(below: 0.5, height: 1)], beyond: 2))
    for (tileName, tile) in [("beside it", Affine2(a: 1, b: 0, tx: 1, c: 0, d: 1, ty: 0)),
                             ("turned", Affine2(a: 0, b: 1, tx: 1, c: -1, d: 0, ty: 1))] {
        for (side, corners) in [("second", [(1.0, 0.2), (1.8, 0.2), (1.8, 0.8), (1.0, 0.8)]),
                                ("first", [(0.2, 0.2), (1.0, 0.2), (1.0, 0.8), (0.2, 0.8)])] {
            let walled = footprint(corners)
            landscapes.append(Landscape("a wall on the tile boundary with a tile \(tileName), the footprint in the \(side)",
                                        surface: parabola, tiles: [.identity, tile], region: .inside(walled),
                                        walls: .perimeter(walled, base: 0)))
        }
    }
    // A corner on the tile boundary where a riser ends: |f| = 10 on the unit
    // square at 1 | 2 across Re z = 1/2, the second tile turned, (x, y) ↦ (1
    // + y, 1 - x), a footprint wholly in it with a wall down the tile
    // boundary from (1, 1/2). There the second tile's riser, along Im z = 1/2
    // in space, ends, and both walls leave its boundary in that tile, while
    // through the first the wall on the tile boundary runs along Re z = 1:
    // the corner's edge rises to 2, the band below the riser.
    let unit = Domain(real: Interval(lo: 0, hi: 1), imag: Interval(lo: 0, hi: 1))
    let flat = Surface(height: Grid2D(width: 21, height: 21, domain: unit, values: [Float](repeating: 10, count: 441)),
                       phase: nil, truncation: .realBands([RealBand(below: 0.5, height: 1)], beyond: 2))
    let cornered = footprint([(1, 0.5), (1, 0.1), (1.8, 0.1), (1.8, 0.9), (1.5, 0.9)])
    landscapes.append(Landscape("|f| = 10 at 1 | 2, a tile turned, a corner on the tile boundary where its riser ends",
                                surface: flat, tiles: [.identity, Affine2(a: 0, b: 1, tx: 1, c: -1, d: 0, ty: 1)],
                                region: .inside(cornered), walls: .perimeter(cornered, base: 0)))
    // A wall leaving the boundary Re z = 1/2 at a slope of 1e-8 against the
    // imaginary axis, the footprint to its right: it runs in the band left
    // of the boundary, and its top is that band's, 1, all along, its end on
    // the boundary too, where a riser up to 2 stands a hair inside it.
    let slope = 1e-8
    let leaving = footprint([(0.5, 0.9), (0.5 - 0.8 * slope, 0.1), (0.9, 0.1), (0.9, 0.9)])
    landscapes.append(Landscape("|f| = 10 at 1 | 2, a wall leaving the boundary at a slope of 1e-8",
                                surface: flat, region: .inside(leaving), walls: .perimeter(leaving, base: 0)))
    // Γ under its banded truncation, on f: the faces read the function.
    let request = LandscapeRequest(
        expression: "gamma(z)",
        domain: Domain(real: Interval(lo: -4.5, hi: 4.5), imag: Interval(lo: -2.5, hi: 2.5)),
        resolution: 160,
        truncation: .realBands([RealBand(below: -3.5, height: 2.5), RealBand(below: -2.5, height: 3),
                                RealBand(below: -1.5, height: 4), RealBand(below: 0.5, height: 5.5)],
                               beyond: 4))
    let gamma = try NativeLandscape.build(request, refine: true)
    let scene = Scene(bundle: gamma, preset: gamma.manifest.presets[0])
    // A tower on f, higher left of its boundary, on a footprint notched
    // below Im z = 0 left of the boundary, walls on it: the riser's top
    // stands only over Im z > 0, where the footprint holds both sides.
    let tower = try NativeLandscape.build(
        LandscapeRequest(expression: "1/(z - 0.5)", domain: Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1)),
                         resolution: 32, truncation: .realBands([RealBand(below: 0.5, height: 4)], beyond: 2)),
        refine: true)
    let halfNotched = footprint([(0.5, -1), (1.5, -1), (1.5, 1), (-0.5, 1), (-0.5, 0), (0.5, 0)])
    landscapes.append(Landscape("a tower banded at 4 | 2, on a footprint holding one side below Im z = 0, on f",
                                surface: tower.surface, region: .inside(halfNotched),
                                walls: .perimeter(halfNotched, base: -0.5), refine: tower.refine))
    // The tower in two tiles, the second reflected across the right edge,
    // x ↦ 3 - x, or turned a quarter, (x, y) ↦ (5/2 - y, x), on a footprint
    // across both with walls on it that cross the boundary near the pole,
    // in the first tile and in the second, where its image is Re z = 5/2 or
    // Im z = 1/2: |f| is over the lower band's height there, and the walls'
    // tops step. One of them passes a fifth of a unit from the pole's image,
    // where |f| rises over the higher band's height too, and its top meets
    // the plateau.
    for (name, tile, corners) in [
        ("reflected", Affine2(a: -1, b: 0, tx: 3, c: 0, d: 1, ty: 0),
         [(-0.3, -0.2), (2.9, -0.2), (3.2, 0.35), (0.1, 0.3)]),
        ("turned", Affine2(a: 0, b: -1, tx: 2.5, c: 1, d: 0, ty: 0),
         [(-0.3, -0.3), (2.2, -0.3), (2.7, -0.25), (2.7, 1.2), (2.25, 1.25), (1.6, 0.3), (0.1, 0.3)]),
    ] {
        let across = footprint(corners)
        landscapes.append(Landscape("a tower banded at 4 | 2 in two tiles, the second \(name), walls across the boundary in both, on f",
                                    surface: tower.surface, tiles: [.identity, tile], region: .inside(across),
                                    walls: .perimeter(across, base: -0.5), refine: tower.refine))
    }
    // The tower in a second tile turned by an angle, 35° or 20°, written
    // from cos and sin, on a footprint wholly in it whose right wall is
    // vertical in space and crosses the boundary's image: the wall's feet
    // keep its x where its top steps, though the tile carries the step's
    // point a rounding off it.
    for (degrees, tx, x) in [(35.0, 13.0, 13.21), (20.0, 7.1, 7.37)] {
        let (c, s) = (cos(degrees * .pi / 180), sin(degrees * .pi / 180))
        let walled = footprint([(x - 0.4, -0.5), (x, -0.5), (x, 1.0), (x - 0.4, 1.0)])
        landscapes.append(Landscape("a tower banded at 4 | 2 in a tile turned by \(Int(degrees))°, a vertical wall across the boundary, on f",
                                    surface: tower.surface, tiles: [.identity, Affine2(a: c, b: -s, tx: tx, c: s, d: c, ty: 0)],
                                    region: .inside(walled), walls: .perimeter(walled, base: -0.5), refine: tower.refine))
    }
    // A tower at 0.5 | 2 in a tile turned a quarter, (x, y) ↦ (5/2 - y, x), a
    // wall across the tile boundary at Re z = 1.5: between the two stations
    // either side of the tile boundary, carried back through the second tile,
    // the wall crosses the boundary's image at a point the first tile holds,
    // where it meets no riser.
    let lowTower = try NativeLandscape.build(
        LandscapeRequest(expression: "1/(z - 0.5)", domain: Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1)),
                         resolution: 32, truncation: .realBands([RealBand(below: 0.5, height: 0.5)], beyond: 2)),
        refine: true)
    let acrossTheBoundary = BoundaryPerimeter(edges: [(0.95, 0.0), (1.95, 1.0), (1.3, 0.95), (0.6, 0.2)].indices.map { k in
        let c = [(0.95, 0.0), (1.95, 1.0), (1.3, 0.95), (0.6, 0.2)], a = c[k], b = c[(k + 1) % c.count]
        return PerimeterEdge(start: P2(a.0, a.1), end: P2(b.0, b.1), density: 8)
    })
    landscapes.append(Landscape("a tower banded at 0.5 | 2 in a tile turned a quarter, a wall across the tile boundary, on f",
                                surface: lowTower.surface, tiles: [.identity, Affine2(a: 0, b: -1, tx: 2.5, c: 1, d: 0, ty: 0)],
                                region: .inside(acrossTheBoundary), walls: .perimeter(acrossTheBoundary, base: -0.5), refine: lowTower.refine))
    // Copies that meet at a tile boundary with different tops, the second
    // tile turned. On the grid, |f| = 3(2 Re z - 1)² + 1/5 at 1 | 2 across Re
    // z = 1/2, the second tile (x, y) ↦ (1 + y, 1 - x): on the tile boundary
    // the first copy is at 2, the band over its right edge, and the second
    // steps from its graph to 1 and to 2 along its bottom edge; on a
    // footprint across the tile boundary, walls on it, and on the whole of
    // both tiles. On f, the reviewer's tower, 1/(z - 1.4) at 0.5 | 1 across
    // Re z = 1.45, the second tile (x, y) ↦ (2.5 + y, 0.5 - x), meeting the
    // first along Re z = 1.5: the first copy on its right edge at 1, the
    // second along its bottom edge under 1/2; on a footprint across the tile
    // boundary and on one in the second tile alone, with a wall down the
    // boundary.
    let turnedBeside = Affine2(a: 0, b: 1, tx: 1, c: -1, d: 0, ty: 1)
    let acrossTheTileBoundary = footprint([(0.2, 0.2), (1.8, 0.2), (1.8, 0.8), (0.2, 0.8)])
    var boundaryDrops = [
        Landscape("copies meeting with different tops, a tile turned, on a footprint across the tile boundary, on the grid",
                  surface: parabola, tiles: [.identity, turnedBeside], region: .inside(acrossTheTileBoundary),
                  walls: .perimeter(acrossTheTileBoundary, base: 0)),
        Landscape("copies meeting with different tops, a tile turned, on the grid",
                  surface: parabola, tiles: [.identity, turnedBeside]),
    ]
    let reviewersTower = try NativeLandscape.build(
        LandscapeRequest(expression: "1/(z - 1.4)", domain: Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1)),
                         resolution: 32, truncation: .realBands([RealBand(below: 1.45, height: 0.5)], beyond: 1)),
        refine: true)
    let turnedAt = Affine2(a: 0, b: 1, tx: 2.5, c: -1, d: 0, ty: 0.5)
    for (name, corners) in [("across the tile boundary", [(0.2, -0.5), (2.8, -0.5), (2.8, 0.5), (0.2, 0.5)]),
                            ("in the second tile, a wall down the tile boundary", [(1.5, -0.5), (2.8, -0.5), (2.8, 0.5), (1.5, 0.5)])] {
        let walled = footprint(corners, density: 17)
        boundaryDrops.append(Landscape("the reviewer's tower at 0.5 | 1, a tile turned, on a footprint \(name), on f",
                                   surface: reviewersTower.surface, tiles: [.identity, turnedAt], region: .inside(walled),
                                   walls: .perimeter(walled, base: -0.5), refine: reviewersTower.refine))
    }
    for k in boundaryDrops.indices where !boundaryDrops[k].name.contains("in the second tile") { boundaryDrops[k].dropsAtATileBoundary = true }
    landscapes += boundaryDrops
    // The higher copy's top steps along a tile boundary: |f| = 10 at
    // 1 | 2 | 1/2 across Re z = 1/2 and 0.9, the turned copy's bottom edge
    // stepping at 0.9, which is a grid node of the first copy's right edge,
    // and the riser between them ends there. The two stops land a rounding
    // apart, one read on the boundary itself.
    let stepped = Surface(height: Grid2D(width: 21, height: 21, domain: unit, values: [Float](repeating: 10, count: 441)),
                          phase: nil, truncation: .realBands([RealBand(below: 0.5, height: 1), RealBand(below: 0.9, height: 2)],
                                                             beyond: 0.5))
    var more = [Landscape("|f| = 10 at 1 | 2 | 1/2, a tile turned, the higher copy stepping along the tile boundary at a grid node",
                          surface: stepped, tiles: [.identity, turnedBeside])]
    // Three tiles in a row, the middle turned, on a footprint whose edges
    // run along Im z = 0.2 and 0.8, grid rows, across both tile boundaries:
    // the riser and its top end on the footprint, not a rounding outside it.
    // And the same footprint at 0.23 and 0.77, off the grid's rows, where the
    // riser ends between two nodes.
    let beside = { (x: Double) in Affine2(a: 1, b: 0, tx: x, c: 0, d: 1, ty: 0) }
    for (lo, hi) in [(0.2, 0.8), (0.23, 0.77)] {
        let row = footprint([(0.2, lo), (2.8, lo), (2.8, hi), (0.2, hi)])
        more.append(Landscape("three tiles in a row, the middle turned, on a footprint from Im z = \(lo) to \(hi), on the grid",
                              surface: parabola, tiles: [.identity, turnedBeside, beside(2)], region: .inside(row),
                              walls: .perimeter(row, base: 0)))
    }
    // The reviewer's footprint, from Im z = 0.12, in the cell where the
    // copies' tops cross, at 0.113: the riser ends on the footprint, not
    // where the tops meet outside it.
    let fromTheCrossing = footprint([(0.2, 0.12), (1.8, 0.12), (1.8, 0.8), (0.2, 0.8)])
    more.append(Landscape("copies meeting with different tops, a tile turned, a footprint from Im z = 0.12 where the tops cross, on the grid",
                          surface: parabola, tiles: [.identity, turnedBeside], region: .inside(fromTheCrossing),
                          walls: .perimeter(fromTheCrossing, base: 0)))
    for k in more.indices { more[k].dropsAtATileBoundary = true }
    // Four tiles, each the first moved by a translation, which is not its
    // own inverse, on a footprint over all four with walls on it: a drop
    // at Re z = 1, none at Im z = 1, and where a wall crosses the drop, a
    // plateau, the riser and the wall meet in one point.
    let square = footprint([(0.2, 0.2), (1.8, 0.2), (1.8, 1.8), (0.2, 1.8)])
    let translations = [Affine2.identity, beside(1), Affine2(a: 1, b: 0, tx: 0, c: 0, d: 1, ty: 1),
                        Affine2(a: 1, b: 0, tx: 1, c: 0, d: 1, ty: 1)]
    more.append(Landscape("four tiles by translation, on a footprint over all four, on the grid", surface: parabola,
                          tiles: translations, region: .inside(square), walls: .perimeter(square, base: 0)))
    more.append(Landscape("four tiles by translation, listed the other way, on the grid", surface: parabola,
                          tiles: translations.reversed()))
    more[more.count - 2].dropsAtATileBoundary = true
    more[more.count - 1].dropsAtATileBoundary = true
    // Copies whose tops agree but for rounding: |f| = 1/|z - (1/2 + i/2)|,
    // softened, on the unit square, with a second tile turned a quarter or
    // a half about the first's right edge, written from cos and sin, so a
    // point of the tile boundary comes back through either tile a rounding
    // apart. No riser stands, and none is drawn.
    let round = Surface(height: Grid2D(width: 21, height: 21, domain: unit, values: (0..<441).map { k in
        let x = Double(k % 21) / 20, y = Double(k / 21) / 20
        return Float(1 / ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + 0.01).squareRoot())
    }), phase: nil, truncation: .uniform(100))
    let (c4, s4, c2, s2) = (cos(Double.pi / 2), sin(Double.pi / 2), cos(Double.pi), sin(Double.pi))
    for (name, tile) in [("a quarter", Affine2(a: c4, b: s4, tx: 1, c: -s4, d: c4, ty: 1)),
                         ("a half", Affine2(a: c2, b: -s2, tx: 1 - c2 + 0.5 * s2, c: s2, d: c2, ty: 0.5 - s2 - 0.5 * c2))] {
        more.append(Landscape("|f| round about the centre, a tile turned \(name) from cos and sin, on the grid", surface: round,
                              tiles: [.identity, tile]))
    }
    // Copies whose tops differ by a millionth: |f| = 1 + x(1 - x) on the unit
    // square, its first column a millionth higher, with a second tile
    // beside the first. A drop a thousand times the rounding `stands`
    // forgives is a riser all the same.
    let millionth = Surface(height: Grid2D(width: 21, height: 21, domain: unit, values: (0..<441).map { k in
        let x = Double(k % 21) / 20
        return Float(1 + x * (1 - x) + (k % 21 == 0 ? 1e-6 : 0))
    }), phase: nil, truncation: .uniform(100))
    var differing = Landscape("copies whose tops differ by a millionth at a tile boundary, the second beside the first, on the grid",
                              surface: millionth, tiles: [.identity, beside(1)])
    differing.dropsAtATileBoundary = true
    more.append(differing)
    // On f, a pole a tenth outside the first copy's right edge, 1/(z - 1.6)
    // under 20, its top along the tile boundary peaking at 10 between two
    // grid nodes, where the turned copy's is under 1: the riser's top is f's
    // profile there, subdivided, not the chord between the nodes.
    let pole = try NativeLandscape.build(
        LandscapeRequest(expression: "1/(z - 1.6)", domain: Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1)),
                         resolution: 32, truncation: .uniform(20)),
        refine: true)
    var peaked = Landscape("a pole a tenth outside the tile boundary, a tile turned, on f", surface: pole.surface,
                           tiles: [.identity, Affine2(a: 0, b: 1, tx: 2.5, c: -1, d: 0, ty: 0.5)], refine: pole.refine)
    peaked.dropsAtATileBoundary = true
    more.append(peaked)
    landscapes += more
    let o = gamma.manifest.occluder
    landscapes.append(Landscape("Γ, banded, on f", surface: gamma.surface, tiles: o.tiles, region: o.region,
                                walls: o.walls, refine: gamma.refine, faces: scene.geometry.faces))
    return landscapes
}


/// Two poles either side of Re z = 1/2 and a zero between them on it.
private let holeBesideABoundary = "1/(z - (0.4804 - 0.0550i)) + 1/(z - (0.5196 + 0.0550i))"
/// Two poles above and below a zero just left of Re z = 1/2.
private let ringBesideABoundary = "1/(z - (0.482 + 0.0853333i)) + 1/(z - (0.482 - 0.2146667i))"

// MARK: - the graph's trim loops
//
// The graph's trim loops are its truncation edge, closed along the domain's
// edge and across the risers, and a copy's add the footprint where it stands.
// Held to the drawing and to membership: the truncation edge as the layer
// draws it lies on them, on every tile; a point the loops put inside, even-odd,
// and the footprint holds, is one the copy's `contains` holds; and a contour on
// a copy, cut where it crosses the copy's loops, is kept exactly where the copy
// holds it, read sixteen times a segment, its runs ending on the loops.

/// Polylines as segments, bucketed so that a point is met only with those
/// near it: the distance to them within a radius, and even-odd membership by
/// a ray along +x.
private struct Segments {
    let a: [SIMD2<Double>], b: [SIMD2<Double>]
    let lo: SIMD2<Double>, size: Double, columns: Int, rows: Int
    let cells: [[Int]]
    /// By rows alone, for the ray: each segment once in a row it spans.
    let strips: [[Int]]

    init(_ paths: [[SIMD2<Double>]], cell size: Double) {
        var a: [SIMD2<Double>] = [], b: [SIMD2<Double>] = []
        for path in paths { for (p, q) in zip(path, path.dropFirst()) where p != q { a.append(p); b.append(q) } }
        var lo = SIMD2(Double.infinity, .infinity), hi = -lo
        for (p, q) in zip(a, b) { lo = simd_min(lo, simd_min(p, q)); hi = simd_max(hi, simd_max(p, q)) }
        if a.isEmpty { lo = .zero; hi = .zero }
        self.a = a; self.b = b; self.lo = lo; self.size = size
        columns = max(1, min(4096, Int(((hi.x - lo.x) / size).rounded(.down)) + 1))
        rows = max(1, min(4096, Int(((hi.y - lo.y) / size).rounded(.down)) + 1))
        var cells = [[Int]](repeating: [], count: columns * rows), strips = [[Int]](repeating: [], count: rows)
        for k in a.indices {
            let l = simd_min(a[k], b[k]), h = simd_max(a[k], b[k])
            let i0 = Self.index(l.x, lo.x, size, columns), i1 = Self.index(h.x, lo.x, size, columns)
            let j0 = Self.index(l.y, lo.y, size, rows), j1 = Self.index(h.y, lo.y, size, rows)
            for j in j0...j1 {
                strips[j].append(k)
                for i in i0...i1 { cells[j * columns + i].append(k) }
            }
        }
        self.cells = cells; self.strips = strips
    }

    static func index(_ x: Double, _ lo: Double, _ size: Double, _ count: Int) -> Int {
        let f = ((x - lo) / size).rounded(.down)
        return f < 0 ? 0 : f >= Double(count) ? count - 1 : Int(f)
    }

    /// The distance from `p` to the nearest segment, or infinity where none
    /// lies within `r`.
    func distance(_ p: SIMD2<Double>, within r: Double) -> Double {
        guard !a.isEmpty else { return .infinity }
        let i0 = Self.index(p.x - r, lo.x, size, columns), i1 = Self.index(p.x + r, lo.x, size, columns)
        let j0 = Self.index(p.y - r, lo.y, size, rows), j1 = Self.index(p.y + r, lo.y, size, rows)
        var best = Double.infinity
        for j in j0...j1 { for i in i0...i1 { for k in cells[j * columns + i] {
            let d = b[k] - a[k]
            let t = max(0, min(1, simd_dot(p - a[k], d) / simd_length_squared(d)))
            best = min(best, simd_length(p - (a[k] + t * d)))
        } } }
        return best <= r ? best : .infinity
    }

    /// Whether the segments wind round `p` an odd number of times: a ray
    /// along +x crosses an odd number of them, each half open in y.
    func inside(_ p: SIMD2<Double>) -> Bool {
        guard !a.isEmpty, p.y >= lo.y, p.y <= lo.y + Double(rows) * size else { return false }
        var inside = false
        for k in strips[Self.index(p.y, lo.y, size, rows)] {
            let a = self.a[k], b = self.b[k]
            guard (a.y > p.y) != (b.y > p.y) else { continue }
            if p.x < a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y) { inside.toggle() }
        }
        return inside
    }
}

/// What the trim-loop suite counts on one landscape's graph.
private struct TrimTally {
    var open = 0, drawn = 0, drawnOff = 0, worstDrawn = 0.0
    var asked = 0, misplaced = 0, worstMisplaced = 0.0
    var runs = 0, endsOff = 0, worstEnd = 0.0, samples = 0, held = 0, misclipped = 0
    var riserAsked = 0, riserMissing = 0
    var examples: [String: [String]] = [:]
    mutating func note(_ kind: String, _ what: String) {
        if examples[kind, default: []].count < 3 { examples[kind, default: []].append(what) }
    }
    func examples(_ kind: String) -> String { examples[kind, default: []].joined(separator: "; ") }
}

/// The graph's trim loops held to the drawing, to membership, and as the
/// cut of a contour, on every copy. Membership by the loops is held to
/// `contains` except within `margin` cells of them: they are the truncation
/// edge as drawn, chords between vertices on |f| = h(x), or the excess
/// grid's marching squares, and `contains` is |f| < h(x) exactly, or the
/// grid's bilinear surface under it. Nor, without a refiner, in a cell whose
/// two columns read different truncation heights: the excess grid
/// interpolates between them there, and its truncation edge with it, while
/// the surface steps at the band's boundary. A contour's clip is held to the
/// loops themselves, which decide it, to a billionth of a cell.
private func trimLoops(_ l: Landscape, margin: Double) -> TrimTally? {
    guard let graph = l.faces.first, graph.id == .graph, let loops = graph.trimLoops else { return nil }
    let surface = l.surface
    let real = surface.domain.real, imag = surface.domain.imag
    let cell = max(abs(real.length) / Double(surface.height.width - 1), abs(imag.length) / Double(surface.height.height - 1))
    let tolerance = 1e-9 * cell
    let own = Segments(loops.map { $0.map(\.v) }, cell: cell)
    let grid = surface.height
    let columns = (0..<grid.width).map { grid.position(x: $0, y: 0).x }
    let straddled = (0..<(grid.width - 1)).filter { i in
        surface.truncation.height(atX: columns[i]) != surface.truncation.height(atX: columns[i + 1])
    }
    func straddles(_ p: SIMD2<Double>) -> Bool {
        guard l.refine?.magnitude == nil else { return false }
        return straddled.contains { i in
            let a = min(columns[i], columns[i + 1]), b = max(columns[i], columns[i + 1])
            return p.x >= a && p.x <= b
        }
    }
    var t = TrimTally()
    // The open chains as `graphTrimLoops` counts them, and the face's loops
    // its, with that many open.
    let reported = surface.graphTrimLoops(refine: l.refine)
    t.open = reported.open
    if reported.loops.map({ $0.map(\.v) }) != loops.map({ $0.map(\.v) })
        || loops.filter({ $0.first != $0.last }).count != reported.open {
        t.open += 1
        t.note("drawn", "the face's loops are not graphTrimLoops', or not as many open")
    }

    // On f, each riser's top drawn wherever the lower band's plateau holds
    // its boundary: read along each boundary inside the domain, a point
    // where |f| clears the lesser height lies on a top drawn along it, and
    // one where |f| is under it on none.
    if let magnitude = l.refine?.magnitude, case .realBands(let bands, _) = surface.truncation {
        let (lo, hi) = (min(real.lo, real.hi), max(real.lo, real.hi))
        let tops = reported.truncationEdge.filter { p in p.count >= 2 && p.allSatisfy { $0.x == p[0].x } }
        for b in Set(bands.map(\.below)).filter({ $0 > lo && $0 < hi }).sorted() {
            let left = surface.truncation.height(atX: b.nextDown), right = surface.truncation.height(atX: b)
            let lesser = min(left, right)
            guard lesser.isFinite, lesser < max(left, right) else { continue }
            for j in 0..<97 {
                let y = imag.lo + imag.length * (Double(j) + 0.5) / 97
                let u = magnitude(P2(b, y))
                guard abs(u - lesser) > 1e-3 * lesser else { continue }
                t.riserAsked += 1
                let drawn = tops.contains { p in
                    p[0].x == b && y >= p.map(\.y).min()! && y <= p.map(\.y).max()!
                }
                if drawn != (u > lesser) {
                    t.riserMissing += 1
                    t.note("riser", "(\(b), \(y)), |f| = \(u) against \(lesser), \(drawn ? "a top drawn" : "no top")")
                }
            }
        }
    }

    // The truncation edge as the layer draws it, replicated and cut to the
    // footprint: each vertex on the loops, carried back by some tile.
    let context = Surface.HatchContext(perimeter: nil, region: l.region, tiles: l.tiles)
    if let drawn = surface.hatch(.truncationEdge(tiled: true), in: context, refine: l.refine) {
        for v in drawn.vertices {
            t.drawn += 1
            let d = l.tiles.compactMap { $0.inverse }.map { own.distance($0(v).xy.v, within: cell) }.min() ?? .infinity
            if !(d <= tolerance) {
                t.drawnOff += 1; t.worstDrawn = max(t.worstDrawn, d)
                t.note("drawn", "drawn vertex \(v.v) \(d.isFinite ? "\(d / cell)" : "over a") cell(s) off the loops")
            }
        }
    }

    for copy in graph.copies {
        // Membership: even-odd over the face's loops, and the footprint where
        // the copy stands.
        for j in 0..<47 {
            for i in 0..<83 {
                let p = SIMD2(real.lo + real.length * (Double(i) + 0.37) / 83, imag.lo + imag.length * (Double(j) + 0.61) / 47)
                guard own.distance(p, within: margin * cell).isInfinite, !straddles(p) else { continue }
                t.asked += 1
                let byLoops = own.inside(p) && l.region.contains(copy.tile(P3(p.x, p.y, 0)).xy)
                if byLoops != copy.contains(P2(p)) {
                    t.misplaced += 1
                    t.worstMisplaced = max(t.worstMisplaced, own.distance(p, within: 10 * cell) / cell)
                    t.note("placed", "\(p) on tile \(copy.tile): the loops say \(byLoops), contains \(!byLoops)")
                }
            }
        }

        // A contour on the copy: lines of x and of y across the rectangle,
        // and level sets of |f| about the truncation, which curve and run
        // along the truncation edge.
        guard let copyLoops = copy.trimLoops else { continue }
        let cut = Segments(copyLoops.map { $0.map(\.v) }, cell: cell)
        let footprint = l.region
        func held(_ q: SIMD2<Double>) -> Bool {
            own.inside(q) && footprint.contains(copy.tile(P3(q.x, q.y, 0)).xy)
        }
        var families: [(ScalarField, [Double])] = [
            (ScalarField { $0.x }, (1...13).map { real.lo + real.length * (Double($0) - 0.29) / 13 }),
            (ScalarField { $0.y }, (1...9).map { imag.lo + imag.length * (Double($0) - 0.43) / 9 }),
        ]
        let heights = Set(surface.truncation.bands(from: min(real.lo, real.hi), to: max(real.lo, real.hi))
            .map(\.height).filter(\.isFinite)).sorted()
        if let magnitude = surface.field(of: .magnitude, refine: l.refine), !heights.isEmpty {
            families.append((magnitude, heights.flatMap { h in [0.6, 0.93, 1.07, 1.4].map { $0 * h } }))
        }
        for (field, levels) in families {
            let whole = Contour.levels(of: field, levels, over: graph.domain)
            let clipped = graph.contours(of: field, levels, on: copy)
            let paths = whole.flatMap(\.paths), runs = clipped.flatMap(\.paths)
            let pathEnds = Set(paths.flatMap { [$0[0], $0[$0.count - 1]] })
            let kept = Segments(runs.map { $0.map(\.v) }, cell: cell)
            t.runs += runs.count
            for run in runs {
                for e in [run[0], run[run.count - 1]] where !pathEnds.contains(e) {
                    let d = cut.distance(e.v, within: cell)
                    if !(d <= tolerance) {
                        t.endsOff += 1; t.worstEnd = max(t.worstEnd, d / cell)
                        t.note("clipped", "a run ends at \(e.v), \(d.isFinite ? "\(d / cell)" : "over a") cell(s) off the loops")
                    }
                }
            }
            for path in paths {
                for (a, b) in zip(path, path.dropFirst()) {
                    for k in 0..<16 {
                        let q = a.v + (b.v - a.v) * ((Double(k) + 0.5) / 16)
                        guard cut.distance(q, within: tolerance).isInfinite else { continue }
                        t.samples += 1
                        if held(q) { t.held += 1 }
                        let onRun = kept.distance(q, within: cell) <= tolerance
                        if onRun != held(q) {
                            t.misclipped += 1
                            t.note("clipped", "\(q) on tile \(copy.tile) \(onRun ? "kept" : "cut away"), the loops say otherwise")
                        }
                    }
                }
            }
        }
    }
    return t
}

func faceTests() {
    Check.suite("faces: a landscape's faces tile its boundary, and each face's map returns its points") {
        let landscapes = try faceLandscapes()

        for l in landscapes {
            guard !l.faces.isEmpty else {
                Check.expect(l.surface.height.width < 2 || l.surface.height.height < 2,
                             "\(l.name): no faces, for a grid too small to span its domain")
                continue
            }
            let t = tiling(l)
            let kinds = Dictionary(grouping: l.faces.map(\.id)) { id -> String in
                switch id {
                case .graph: "graph"
                case .plateau: "plateaus"
                case .riser, .tileBoundaryRiser: "risers"
                case .wall: "walls"
                case .surface, .solid: "other"
                }
            }.mapValues(\.count).sorted { $0.key < $1.key }.map { "\($0.value) \($0.key)" }.joined(separator: ", ")
            Check.expect(t.wrong == 0 && t.inward == 0 && t.skewed == 0 && t.sampled > 10_000
                            && (t.normals > 300 || t.onGraph == 0),
                         "\(l.name): \(kinds), \(l.tiles.count) tile(s); every point of the boundary in one face, "
                            + "none off it, each face turned out, the graph's normal the facing's",
                         "\(t.wrong) in the wrong number, \(t.inward) facing in, \(t.skewed) of \(t.normals) normals off, "
                            + "of \(t.sampled); \(t.drops) of \(t.boundaryPoints) tile boundary points with a drop: "
                            + t.examples.filter { $0.key != "offTop" }.sorted { $0.key < $1.key }.map(\.value).joined(separator: "; "))
            if l.dropsAtATileBoundary {
                Check.expect(t.drops > 30, "\(l.name): the solid drops at a tile boundary, and a riser stands there",
                             "\(t.drops) of \(t.boundaryPoints) tile boundary points with a drop")
            }
            if case .perimeter = l.walls {
                Check.expect(t.offTop == 0 && t.stations > 50,
                             "\(l.name): every wall rises to the solid's top beside it in the footprint, no higher and no lower",
                             "\(t.offTop) of \(t.stations) stations off: \(t.examples["offTop"] ?? "")")
            }
        }
        // `Surface.faces` lists them in `FaceID`'s order.
        Check.expect(landscapes.allSatisfy { l in zip(l.faces, l.faces.dropFirst()).allSatisfy { $0.id <= $1.id } },
                     "the faces come in FaceID's order: graph, plateaus, risers at band boundaries and at tile boundaries, walls")
    }

    Check.suite("walls: a wall on a band boundary reads its footprint's band, and the far edge the band before it") {
        let landscapes = try faceLandscapes()
        for l in landscapes {
            guard case .perimeter(let perimeter, let base) = l.walls else { continue }
            let dense = BoundaryPerimeter(edges: perimeter.edges.map { PerimeterEdge(start: $0.start, end: $0.end, density: 41) })
            let context = Surface.HatchContext(perimeter: dense, region: .full, tiles: l.tiles)
            // A vertical wall's feet stand on its own x, exactly: in the
            // curtain, its top and bottom edges and its hatching. A foot a
            // rounding off a band boundary reads the band across it.
            let vertical = perimeter.edges.indices.filter { perimeter.edges[$0].start.x == perimeter.edges[$0].end.x }
            if !vertical.isEmpty {
                var feet = 0, off: [String] = []
                for k in vertical {
                    let x = perimeter.edges[k].start.x
                    var vertices = curtain(dense, k, surface: l.surface, base: base, tiles: l.tiles).all
                    if let drawn = l.surface.hatch(.wallEdges(edges: [k], step: 1, base: base), in: context, refine: l.refine) {
                        vertices += drawn[path: 0] + drawn[path: 1]
                    }
                    let strokes = Hatching(angle: HatchedFaces.wallAngle, spacing: 0.02)
                    vertices += l.surface.hatch(.hatching(strokes, faces: .walls(edges: [k], trim: false, base: base, topOffset: 0)),
                                                in: context, refine: l.refine)?.vertices ?? []
                    for v in vertices {
                        feet += 1
                        if v.x != x, off.count < 3 { off.append("\(v.x) on Re z = \(x)") } else if v.x != x { off.append("") }
                    }
                }
                Check.expect(feet > 200 && off.isEmpty, "\(l.name): every foot of a vertical wall stands on its own x",
                             "\(off.count) of \(feet) off: \(off.prefix(3).joined(separator: ", "))")
            }
            // The top edge reads the truncation as the curtain does: at each
            // foot's own x, through the tile under it, from the footprint's
            // side. Without f the two stand on the same stations and agree
            // at every one, to the bit. With f the top edge is f's, and is
            // the wall face's top at every vertex but the two of a step,
            // which stand at one point; and wherever the tile under the wall
            // carries a band boundary across it, it steps there: no two
            // consecutive vertices lie either side of the boundary apart.
            if l.refine == nil {
                var stations = 0, wrong: [String] = []
                for k in perimeter.edges.indices {
                    guard let drawn = l.surface.hatch(.wallEdges(edges: [k], step: 1, base: base), in: context) else { continue }
                    let top = Array(drawn[path: 0])
                    // Two at a point where the wall crosses a tile boundary,
                    // in order.
                    var seen: [P2<WorldSpace>: Int] = [:]
                    for c in curtain(dense, k, surface: l.surface, base: base, tiles: l.tiles).top {
                        stations += 1
                        let at = top.filter { $0.xy == c.xy }, index = seen[c.xy, default: 0]
                        seen[c.xy] = index + 1
                        guard index < at.count else { wrong.append("no vertex at \(c.xy.v)"); continue }
                        let v = at[index]
                        if v.z != c.z { wrong.append("\(c.xy.v) at \(v.z), the curtain at \(c.z)") }
                    }
                }
                Check.expect(stations >= 41 * perimeter.edges.count && wrong.isEmpty,
                             "\(l.name): the top edge is the curtain's top at every station",
                             "\(wrong.count) of \(stations) off: \(wrong.prefix(3).joined(separator: "; "))")
            } else if case .realBands(let bands, _) = l.surface.truncation {
                let walls = l.faces.filter { if case .wall = $0.id { true } else { false } }
                let real = l.surface.domain.real
                let boundaries = bands.map(\.below).filter { $0 > min(real.lo, real.hi) && $0 < max(real.lo, real.hi) }
                let cell = max(abs(real.length) / Double(l.surface.height.width - 1),
                               abs(l.surface.domain.imag.length) / Double(l.surface.height.height - 1))
                var vertices = 0, crossings = 0, wrong: [String] = []
                var steps = [Int](repeating: 0, count: l.tiles.count)
                for (k, edge) in perimeter.edges.enumerated() {
                    guard let drawn = l.surface.hatch(.wallEdges(edges: [k], step: 1, base: base), in: context, refine: l.refine),
                          let face = walls.first(where: { $0.id == .wall(edge: k) }) else { continue }
                    let top = Array(drawn[path: 0])
                    func apart(_ a: P3<WorldSpace>, _ b: P3<WorldSpace>) -> Bool { simd_length(a.xy.v - b.xy.v) > 1e-9 * cell }
                    let d = simd_normalize(edge.end.v - edge.start.v), size = max(abs(real.length), abs(l.surface.domain.imag.length))
                    /// Whether the wall crosses a tile boundary at a point.
                    func onTileBoundary(_ v: P3<WorldSpace>) -> Bool {
                        carriedBack(v.xy.v + 1e-6 * size * d, l).back != carriedBack(v.xy.v - 1e-6 * size * d, l).back
                    }
                    for (i, v) in top.enumerated() {
                        guard (i == 0 || apart(top[i - 1], v)) && (i == top.count - 1 || apart(v, top[i + 1])) else { continue }
                        vertices += 1
                        let s = simd_length(v.xy.v - edge.start.v), z = v.z, dz = 1e-9 * max(1, abs(z))
                        let copy = face.copies[0]
                        if !(copy.contains(P2(s, z - dz)) && !copy.contains(P2(s, z + dz))) {
                            wrong.append("\(v.xy.v) at \(z), not the wall's top")
                        }
                    }
                    // Under one tile: across a tile boundary between two, the x
                    // carried back jumps.
                    for (a, c) in zip(top, top.dropFirst()) {
                        let (qa, back) = carriedBack(a.xy.v, l), qc = carriedBack(c.xy.v, l)
                        guard qc.back == back else { continue }
                        let (xa, xc) = (qa.x, qc.q.x)
                        for b in boundaries where min(xa, xc) < b - 1e-9 * cell && max(xa, xc) > b + 1e-9 * cell {
                            crossings += 1
                            wrong.append("a ramp across \(b) from \(a.v) to \(c.v)")
                        }
                        // Two vertices at one point on a boundary: where the
                        // wall meets a riser, at the two bands' heights.
                        guard let tile = l.tiles.firstIndex(where: { $0.inverse == back }) else { continue }
                        if !apart(a, c), boundaries.contains(where: { abs(xa - $0) <= 1e-9 * cell }) { steps[tile] += 1 }
                        // And it steps nowhere else but at a tile boundary,
                        // where two copies meet.
                        else if !apart(a, c), a.z != c.z, !onTileBoundary(a) {
                            wrong.append("a step at \(a.v), on no band boundary in the tile under it, nor on a tile boundary")
                        }
                    }
                }
                Check.expect(vertices > 100 && wrong.isEmpty,
                             "\(l.name): the top edge on f is the wall face's top, and steps where the tile under it crosses a boundary",
                             "\(wrong.count) wrong of \(vertices) vertices, \(crossings) ramps, steps by tile \(steps): "
                                + wrong.prefix(3).joined(separator: "; "))
            }
            // A corner's vertical edge, where two walls meet, rises to the
            // lower of their tops and on to the higher beside the riser that
            // ends there, unless the higher wall stands on that riser's
            // boundary, in one plane with it: to the top of the wall that
            // leaves the boundary, each wall's leaving read through the tile
            // under the footprint beside the corner, a hair in from it and
            // along the wall, since on a tile boundary the corner is two
            // tiles'. Read off the walls' own top edges, on the grid, as the
            // corners are.
            guard l.refine == nil,
                  let drawn = l.surface.hatch(.wallEdges(edges: Array(perimeter.edges.indices), step: 1, base: base),
                                              in: Surface.HatchContext(perimeter: perimeter, region: .full, tiles: l.tiles))
            else { continue }
            var corners = 0, wrong: [String] = []
            for (k, edge) in perimeter.edges.enumerated() {
                let corner = edge.start
                let previous = (k + perimeter.edges.count - 1) % perimeter.edges.count
                let here = drawn[path: 2 * k], before = drawn[path: 2 * previous]
                let size = max(abs(l.surface.domain.real.length), abs(l.surface.domain.imag.length))
                /// Whether wall `e` runs along Re z = const through the tile
                /// under the footprint beside its end at the corner.
                func alongX(_ e: Int, atStart: Bool) -> Bool {
                    let d = perimeter.edges[e].end.v - perimeter.edges[e].start.v
                    let into = simd_normalize(d) * (atStart ? 1 : -1)
                    let back = carriedBack(corner.v + 1e-9 * size * (perimeter.inward(e) + into), l).back
                    let carried = SIMD2(back.a * d.x + back.b * d.y, back.c * d.x + back.d * d.y)
                    return abs(carried.x) <= 1e-12 * simd_length(carried)
                }
                let walls = [(top: here[here.startIndex].z, alongX: alongX(k, atStart: true)),
                             (top: before[before.endIndex - 1].z, alongX: alongX(previous, atStart: false))]
                let leaving = walls.filter { !$0.alongX }
                let expected = (leaving.count == 1 ? leaving : walls).map(\.top).max()!
                guard expected > base else { continue }
                let edges = (2 * perimeter.edges.count..<drawn.count).map { drawn[path: $0] }
                guard let vertical = edges.first(where: { $0.allSatisfy { $0.xy == corner } }) else {
                    wrong.append("no edge at \(corner.v)"); continue
                }
                corners += 1
                let top = vertical[vertical.endIndex - 1].z
                // The edge's vertices are spaced from the base, a rounding
                // off its top.
                if !(abs(top - expected) <= 1e-12 * max(1, abs(expected))) { wrong.append("\(corner.v) to \(top), not \(expected)") }
            }
            Check.expect(corners >= perimeter.edges.count / 2 && wrong.isEmpty,
                         "\(l.name): each corner's edge rises to the walls' tops there, to the leaving wall's beside a riser",
                         "\(wrong.count) of \(corners): \(wrong.prefix(3).joined(separator: "; "))")
        }
        // On the tile boundary where the turned tile's riser ends, the
        // corner's edge reaches 2, the band below the riser in that tile.
        if let l = landscapes.first(where: { $0.name.hasSuffix("a corner on the tile boundary where its riser ends") }),
           case .perimeter(let perimeter, let base) = l.walls,
           let drawn = l.surface.hatch(.wallEdges(edges: Array(perimeter.edges.indices), step: 1, base: base),
                                       in: Surface.HatchContext(perimeter: perimeter, region: .full, tiles: l.tiles)) {
            let corner = (2 * perimeter.edges.count..<drawn.count).map { drawn[path: $0] }
                .first { $0.allSatisfy { $0.xy == P2(1, 0.5) } }
            let top = corner.map { $0[$0.endIndex - 1].z }
            Check.expect(top.map { abs($0 - 2) <= 1e-12 } ?? false,
                         "\(l.name): the corner's edge at (1, 1/2) rises to 2", "to \(top.map { "\($0)" } ?? "nothing")")
        }
        // A wall leaving the boundary at a slant of 1e-8 runs in the band
        // left of it, its end on the boundary too: its curtain reads 1 all
        // along.
        if let l = landscapes.first(where: { $0.name.hasSuffix("a wall leaving the boundary at a slope of 1e-8") }),
           case .perimeter(let perimeter, let base) = l.walls {
            let dense = BoundaryPerimeter(edges: perimeter.edges.map { PerimeterEdge(start: $0.start, end: $0.end, density: 41) })
            let tops = curtain(dense, 0, surface: l.surface, base: base, tiles: l.tiles).top.map(\.z)
            Check.expect(tops.allSatisfy { $0 == 1 }, "\(l.name): its curtain reads 1 all along",
                         "\(tops.filter { $0 != 1 }.count) of 41 stations not 1: \(Set(tops).sorted())")
        }

        // A wall that crosses a band boundary steps there: on f its top edge
        // is f's profile under the truncation at each vertex's own x, read
        // from the right as the band starts there, and wherever two
        // consecutive vertices lie in different bands they stand at one
        // point, the step a vertical segment, not a ramp. On walls rising,
        // falling and level across the boundary, the higher band on either
        // side.
        for (left, right) in [(0.5, 1.0), (1.0, 0.5)] {
            let tower = try NativeLandscape.build(
                LandscapeRequest(expression: "1/(z - 0.5)", domain: Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1)),
                                 resolution: 32, truncation: .realBands([RealBand(below: 0.5, height: left)], beyond: right)),
                refine: true)
            guard let magnitude = tower.refine?.magnitude else { continue }
            for (slope, front) in [("rising", [(-0.5, -0.9), (1.5, -0.6)]), ("falling", [(-0.5, -0.6), (1.5, -0.9)]),
                                   ("level", [(-0.5, -0.75), (1.5, -0.75)])] {
                let corners = front + [(1.5, 1.0), (-0.5, 1.0)]
                let perimeter = BoundaryPerimeter(edges: corners.indices.map { k in
                    let a = corners[k], b = corners[(k + 1) % corners.count]
                    return PerimeterEdge(start: P2(a.0, a.1), end: P2(b.0, b.1), density: 17)
                })
                let drawn = tower.surface.hatch(.wallEdges(edges: [0], step: 1, base: 0),
                                                in: Surface.HatchContext(perimeter: perimeter), refine: tower.refine)
                let top = drawn.map { Array($0[path: 0]) } ?? []
                var wrong: [String] = []
                for v in top where !(abs(v.z - min(magnitude(P2(v.x, v.y)), v.x < 0.5 ? left : right)) <= 1e-9) {
                    wrong.append("(\(v.x), \(v.z))")
                }
                var steps = 0
                for (a, c) in zip(top, top.dropFirst()) where (a.x < 0.5) != (c.x < 0.5) {
                    steps += 1
                    if !(simd_length(a.xy.v - c.xy.v) <= 1e-12) { wrong.append("a ramp from (\(a.x), \(a.z)) to (\(c.x), \(c.z))") }
                }
                Check.expect(!top.isEmpty && steps == 1 && wrong.isEmpty,
                             "a wall \(slope) across a boundary at \(left) | \(right), on f: its top edge steps at Re z = 1/2, with no ramp",
                             "\(steps) crossings, \(wrong.count) wrong: \(wrong.prefix(3).joined(separator: ", "))")
            }
        }

        // A wall whose stations either side of a tile boundary lie under
        // different tiles, with a band boundary between them in each: a
        // tower, 1/(z - 1.4), at 0.5 | 1 across Re z = 1.45, in two tiles,
        // the second reflected across the right edge, x ↦ 3 - x, so the
        // boundary's image is Re z = 1.55; a wall along Im z = -1/2 from 0.2
        // to 2.8 at density 4, its second station at 1.07 and its third at
        // 1.93. On f its top edge steps at 1.45 and at 1.55, and between its
        // vertices it is the solid's top a hair inside the footprint, read
        // every 32nd of each chord, a chord under two tiles included.
        let straddlingTower = try NativeLandscape.build(
            LandscapeRequest(expression: "1/(z - 1.4)", domain: Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1)),
                             resolution: 32, truncation: .realBands([RealBand(below: 1.45, height: 0.5)], beyond: 1)),
            refine: true)
        let straddled = BoundaryPerimeter(edges: [(0.2, -0.5), (2.8, -0.5), (2.8, 0.5), (0.2, 0.5)].indices.map { k in
            let c = [(0.2, -0.5), (2.8, -0.5), (2.8, 0.5), (0.2, 0.5)], a = c[k], b = c[(k + 1) % c.count]
            return PerimeterEdge(start: P2(a.0, a.1), end: P2(b.0, b.1), density: 4)
        })
        let straddling = Landscape("a tower at 0.5 | 1 in two tiles, the second reflected, a wall across the tile boundary at density 4",
                                   surface: straddlingTower.surface, tiles: [.identity, Affine2(a: -1, b: 0, tx: 3, c: 0, d: 1, ty: 0)],
                                   region: .inside(straddled), walls: .perimeter(straddled, base: -0.5), refine: straddlingTower.refine)
        if let drawn = straddling.surface.hatch(.wallEdges(edges: [0], step: 1, base: -0.5),
                                                in: Surface.HatchContext(perimeter: straddled, region: .full, tiles: straddling.tiles),
                                                refine: straddling.refine), let tolerance = straddling.refine?.tolerance {
            let top = Array(drawn[path: 0]), h = straddling.heightfield
            let cell = 2.0 / 31
            let steps = zip(top, top.dropFirst()).filter { simd_length($0.0.xy.v - $0.1.xy.v) <= 1e-12 && $0.0.z != $0.1.z }
                .map { ($0.0.x * 1e4).rounded() / 1e4 }
            var samples = 0, worst = 0.0, off: [String] = []
            for (a, c) in zip(top, top.dropFirst()) where simd_length(a.xy.v - c.xy.v) > 1e-12 {
                for k in 1..<32 {
                    let t = Double(k) / 32, w = a.xy.v + (c.xy.v - a.xy.v) * t
                    let solid = h.surfaceHeight(at: P2(w.x, w.y + 1e-9))
                    guard solid.isFinite else { continue }
                    samples += 1
                    let gap = abs(a.z + (c.z - a.z) * t - solid)
                    worst = max(worst, gap)
                    if gap > 2 * tolerance * cell { off.append(String(format: "x %.4f: %.4f off", w.x, gap)) }
                }
            }
            Check.expect(steps == [1.45, 1.55] && samples > 100 && off.isEmpty,
                         "\(straddling.name), on f: its top edge steps at 1.45 and 1.55, and follows the solid between",
                         "steps at \(steps), \(off.count) of \(samples) points off, worst \(worst): \(off.prefix(3).joined(separator: "; "))")
        }

        // Where a wall crosses a tile boundary and the copies' tops differ,
        // its top edge on f steps: no chord of it lies under two tiles across
        // a drop, and where the crossing is, two vertices at one point, each
        // the solid's top a hair along the wall on its own side, read through
        // the tile under it. Where the tops agree it does not step. Between its
        // vertices it is the solid's top a hair inside the footprint, read
        // every 32nd of each chord, within four times the refiner's
        // tolerance: the profile is subdivided where a chord's midpoint parts
        // from it, five deep, and beside a pole a chord parts from it by more
        // between.
        for l in landscapes where l.refine?.magnitude != nil && l.tiles.count > 1 {
            guard case .perimeter(let perimeter, let base) = l.walls, let tolerance = l.refine?.tolerance else { continue }
            let h = l.heightfield, real = l.surface.domain.real, imag = l.surface.domain.imag
            let size = max(abs(real.length), abs(imag.length))
            let cell = max(abs(real.length) / Double(l.surface.height.width - 1), abs(imag.length) / Double(l.surface.height.height - 1))
            let context = Surface.HatchContext(perimeter: perimeter, region: .full, tiles: l.tiles)
            var samples = 0, steps = 0, drops = 0, worst = 0.0, wrong: [String] = []
            for (k, edge) in perimeter.edges.enumerated() {
                guard let drawn = l.surface.hatch(.wallEdges(edges: [k], step: 1, base: base), in: context, refine: l.refine) else { continue }
                let top = Array(drawn[path: 0])
                let d = simd_normalize(edge.end.v - edge.start.v), inward = perimeter.inward(k)
                func solid(_ w: SIMD2<Double>) -> Double { h.surfaceHeight(at: P2(w + 1e-9 * size * inward)) }
                for (a, c) in zip(top, top.dropFirst()) {
                    if simd_length(a.xy.v - c.xy.v) > 1e-12 * size {
                        let near = a.xy.v + (c.xy.v - a.xy.v) / 32, far = a.xy.v + (c.xy.v - a.xy.v) * 31 / 32
                        if carriedBack(near + 1e-9 * size * inward, l).back != carriedBack(far + 1e-9 * size * inward, l).back,
                           dropAcross(a.xy.v, c.xy.v, inward: inward, l, top: { h.surfaceHeight(at: P2($0)) }).drops {
                            wrong.append("a chord from \(a.v) to \(c.v) across a drop between two tiles")
                        }
                        for j in 1..<32 {
                            let t = Double(j) / 32, w = a.xy.v + (c.xy.v - a.xy.v) * t
                            let z = solid(w)
                            guard z.isFinite else { continue }
                            samples += 1
                            let gap = abs(a.z + (c.z - a.z) * t - z)
                            worst = max(worst, gap)
                            if gap > 4 * tolerance * cell { wrong.append(String(format: "(%.4f, %.4f) %.4f off the solid", w.x, w.y, gap)) }
                        }
                    } else if carriedBack(a.xy.v + 1e-6 * size * d, l).back != carriedBack(a.xy.v - 1e-6 * size * d, l).back {
                        steps += 1
                        // Along the wall, in from it by far less, so each read
                        // stays on its side of the boundary.
                        let e = 1e-9 * size
                        func along(_ t: Double) -> Double { h.surfaceHeight(at: P2(a.xy.v + t * e * d + 1e-3 * e * inward)) }
                        let before = along(-1), after = along(1)
                        let slide = max(abs(along(-2) - before), abs(along(2) - after))
                        if abs(before - after) > 4 * slide + 1e-9 * max(1, abs(before)) { drops += 1 } else {
                            wrong.append("a step at \(a.xy.v), where the copies' tops agree")
                        }
                        for (v, z) in [(a, before), (c, after)] where !(abs(v.z - z) <= 4 * slide + 1e-9 * max(1, abs(z))) {
                            wrong.append("at the tile boundary \(v.xy.v) \(v.z), the solid on its side \(z)")
                        }
                    }
                }
            }
            Check.expect(samples > 300 && wrong.isEmpty && (!l.dropsAtATileBoundary || drops > 0),
                         "\(l.name): the top edge steps where a wall crosses a tile boundary, and follows the solid between",
                         "\(steps) steps at a tile boundary, \(drops) down a drop, \(wrong.count) wrong of \(samples) points, worst \(worst): "
                            + wrong.prefix(3).joined(separator: "; "))
        }

        // Where a wall crosses a tile boundary and the copies' tops agree, it
        // draws as on one copy: cn(z, 0.64) over [-K, K] × [0, K'] on f, a
        // second tile reflected across Re z = K or moved 2K along, |cn| the
        // same either way, on a footprint across both with walls on it. No
        // two vertices of a top edge stand at one point, and with the
        // reflection, whose copies read one point of the domain, the
        // curtain has no sample past its density.
        let K = 1.9953027776647292, Kp = 1.7507538029157523
        let cn = try NativeLandscape.build(
            LandscapeRequest(expression: "cn(z, 0.64)", domain: Domain(real: Interval(lo: -K, hi: K), imag: Interval(lo: 0, hi: Kp)),
                             resolution: 100, truncation: .uniform(4)),
            refine: true)
        let acrossCn = BoundaryPerimeter(edges: [(-K + 0.3, 0.3), (3 * K - 0.3, 0.3), (3 * K - 0.3, Kp - 0.3), (-K + 0.3, Kp - 0.3)].indices.map { k in
            let c = [(-K + 0.3, 0.3), (3 * K - 0.3, 0.3), (3 * K - 0.3, Kp - 0.3), (-K + 0.3, Kp - 0.3)], a = c[k], b = c[(k + 1) % c.count]
            return PerimeterEdge(start: P2(a.0, a.1), end: P2(b.0, b.1), density: 17)
        })
        for (name, tile, reflected) in [("reflected", Affine2(a: -1, b: 0, tx: 2 * K, c: 0, d: 1, ty: 0), true),
                                        ("moved 2K along", Affine2(a: 1, b: 0, tx: 2 * K, c: 0, d: 1, ty: 0), false)] {
            let tiles = [Affine2.identity, tile]
            let context = Surface.HatchContext(perimeter: acrossCn, region: .full, tiles: tiles)
            var vertices = 0, doubled: [String] = [], extra = 0
            for k in acrossCn.edges.indices {
                guard let drawn = cn.surface.hatch(.wallEdges(edges: [k], step: 1, base: 0), in: context, refine: cn.refine) else { continue }
                let top = Array(drawn[path: 0])
                vertices += top.count
                for (a, c) in zip(top, top.dropFirst()) where a.xy == c.xy { doubled.append("\(a.xy.v)") }
                if reflected { extra += curtain(acrossCn, k, surface: cn.surface, base: 0, tiles: tiles).top.count - 17 }
            }
            Check.expect(vertices > 100 && doubled.isEmpty && extra == 0,
                         "cn(z, 0.64), a second tile \(name), walls across both, on f: a wall across the tile boundary, the tops agreeing, draws as on one copy",
                         "\(doubled.count) points with two vertices, \(extra) curtain samples past the density: \(doubled.prefix(2).joined(separator: "; "))")
        }

        // A riser's top is drawn only where the riser stands: on f, every
        // segment of a top the truncation edge draws along a band boundary
        // lies on a riser at its middle, and where the footprint holds both
        // sides and |f| clears the lower band's height, a top is drawn. Also
        // on a flat top, two vertices, 10 exp(-2z) at 2 | 4: on a footprint
        // holding both sides only above Im z = 0, and on one notched across
        // the boundary over Im z in (-0.3, 0.3).
        let flat = try NativeLandscape.build(
            LandscapeRequest(expression: "10*exp(-2*z)", domain: Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1)),
                             resolution: 32, truncation: .realBands([RealBand(below: 0.2, height: 2)], beyond: 4)),
            refine: true)
        let flatTops = [
            ("both sides only above Im z = 0", [(0.2, -1.2), (1.7, -1.2), (1.7, 1.2), (-0.7, 1.2), (-0.7, 0.0), (0.2, 0.0)]),
            ("notched across the boundary over Im z in (-0.3, 0.3)",
             [(-0.7, -1.2), (1.7, -1.2), (1.7, 1.2), (-0.7, 1.2), (-0.7, 0.3), (0.2, 0.3), (0.2, -0.3), (-0.7, -0.3)]),
        ].map { name, corners in
            let notched = BoundaryPerimeter(edges: corners.indices.map { k in
                let a = corners[k], b = corners[(k + 1) % corners.count]
                return PerimeterEdge(start: P2(a.0, a.1), end: P2(b.0, b.1), density: 9)
            })
            return Landscape("a flat riser top, 10 exp(-2z) at 2 | 4, on a footprint \(name), on f",
                             surface: flat.surface, region: .inside(notched), refine: flat.refine)
        }
        for l in landscapes + flatTops where l.refine?.magnitude != nil && l.tiles.count == 1 {
            guard case .realBands(let bands, _) = l.surface.truncation,
                  let magnitude = l.refine?.magnitude,
                  let drawn = l.surface.hatch(.truncationEdge(tiled: true),
                                              in: Surface.HatchContext(perimeter: nil, region: l.region, tiles: l.tiles),
                                              refine: l.refine) else { continue }
            let real = l.surface.domain.real, imag = l.surface.domain.imag
            let (lo, hi) = (min(real.lo, real.hi), max(real.lo, real.hi))
            let boundaries = Set(bands.map(\.below)).filter { $0 > lo && $0 < hi }.sorted()
            let risers = l.faces.filter { if case .riser = $0.id { true } else { false } }
            let side = 1e-9 * max(abs(real.length), abs(imag.length))
            var stray: [String] = [], missing: [String] = [], asked = 0
            let tops = (0..<drawn.count).map { Array(drawn[path: $0]) }.filter { p in
                p.count >= 2 && boundaries.contains(p[0].x) && p.allSatisfy { $0.x == p[0].x }
            }
            // Asked under the chord, between the riser's foot and the lower
            // end: a chord of a curved top runs over the top itself.
            for (a, c) in tops.flatMap({ zip($0, $0.dropFirst()) }) {
                let y = 0.5 * (a.y + c.y)
                let held = risers.contains { face in
                    let p = P2<ParamSpace>(y, 0.5 * (face.domain.v.range.lo + min(a.z, c.z)))
                    return face.map(p).position.x == a.x && face.copies.contains { $0.contains(p) }
                }
                if !held { stray.append("(\(a.x), \(y))") }
            }
            for b in boundaries {
                let lesser = min(l.surface.truncation.height(atX: b.nextDown), l.surface.truncation.height(atX: b))
                guard lesser.isFinite else { continue }
                for j in 0..<97 {
                    let y = imag.lo + imag.length * (Double(j) + 0.5) / 97
                    guard l.region.contains(P2(b - side, y)), l.region.contains(P2(b + side, y)),
                          magnitude(P2(b, y)) > lesser * (1 + 1e-3) else { continue }
                    asked += 1
                    if !tops.contains(where: { p in p[0].x == b && y >= p.map(\.y).min()! && y <= p.map(\.y).max()! }) {
                        missing.append("(\(b), \(y))")
                    }
                }
            }
            guard !boundaries.isEmpty else { continue }
            Check.expect(stray.isEmpty && missing.isEmpty,
                         "\(l.name): the truncation edge draws a riser's top only where the riser stands, and all of it there",
                         "\(stray.count) vertices on no riser, \(missing.count) of \(asked) points with no top: "
                            + (stray.prefix(2) + missing.prefix(2)).joined(separator: ", "))
        }

        // A riser's top at a tile boundary is drawn only where the riser
        // stands, at the higher top, and all of it there: tiled, every
        // segment of the truncation edge drawn along a side of a tile's image
        // that another tile's image meets has, at its middle, the footprint
        // either side, a drop between the tops read a hair either side, and
        // the higher top; and at every point of such a side with a drop, a
        // segment passes over it at the higher top. On f the top is
        // subdivided to the refiner's tolerance; on the grid each copy's top
        // is linear between the vertices.
        for l in landscapes where l.tiles.count > 1 {
            guard let drawn = l.surface.hatch(.truncationEdge(tiled: true),
                                              in: Surface.HatchContext(perimeter: nil, region: l.region, tiles: l.tiles),
                                              refine: l.refine) else { continue }
            let real = l.surface.domain.real, imag = l.surface.domain.imag
            let size = max(abs(real.length), abs(imag.length)), side = 1e-9 * size
            let cell = max(abs(real.length) / Double(l.surface.height.width - 1), abs(imag.length) / Double(l.surface.height.height - 1))
            let bound = l.refine.map { 4 * $0.tolerance * cell } ?? 1e-9 * size
            let h = l.heightfield
            let (lo, hi) = (min(real.lo, real.hi), max(real.lo, real.hi)), (ylo, yhi) = (min(imag.lo, imag.hi), max(imag.lo, imag.hi))
            let corners = [SIMD2(lo, ylo), SIMD2(hi, ylo), SIMD2(hi, yhi), SIMD2(lo, yhi)]
            // The sides of the tiles' images.
            let sides: [(a: SIMD2<Double>, b: SIMD2<Double>)] = l.tiles.flatMap { tile in
                (0..<4).map { e in (tile(P3(corners[e].x, corners[e].y, 0)).xy.v, tile(P3(corners[(e + 1) % 4].x, corners[(e + 1) % 4].y, 0)).xy.v) }
            }
            /// The drop a point of a side stands over: the tops a hair
            /// either side, through the tile under each, where the footprint
            /// holds both and one tile holds each, two different ones; nil
            /// where there is no tile boundary, or no drop.
            func drop(_ w: SIMD2<Double>, _ n: SIMD2<Double>) -> (lo: Double, hi: Double)? {
                let reads = [-1.0, 1.0].map { sign -> (z: Double, back: Affine2, slide: Double)? in
                    let g = w + sign * side * n
                    let covering = l.tiles.filter { tile in
                        guard let back = tile.inverse else { return false }
                        let q = back(P3(g.x, g.y, 0))
                        return q.x >= lo && q.x <= hi && q.y >= ylo && q.y <= yhi
                    }.count
                    guard covering == 1, l.region.contains(P2(g)) else { return nil }
                    let z = h.surfaceHeight(at: P2(g))
                    return z.isFinite ? (z, carriedBack(g, l).back, abs(h.surfaceHeight(at: P2(w + 2 * sign * side * n)) - z)) : nil
                }
                guard let a = reads[0], let b = reads[1], a.back != b.back,
                      abs(a.z - b.z) > 4 * max(a.slide, b.slide) + 1e-9 * max(1, max(a.z, b.z)) else { return nil }
                return (min(a.z, b.z), max(a.z, b.z))
            }
            let paths = (0..<drawn.count).map { Array(drawn[path: $0]) }
            func onSide(_ v: SIMD2<Double>, _ s: (a: SIMD2<Double>, b: SIMD2<Double>)) -> Bool {
                let d = s.b - s.a, t = max(0, min(1, simd_dot(v - s.a, d) / simd_length_squared(d)))
                return simd_length(v - (s.a + t * d)) <= 1e-9 * size
            }
            var segments = 0, stray: [String] = [], asked = 0, missing: [String] = []
            var along: [(a: P3<WorldSpace>, c: P3<WorldSpace>)] = []
            for path in paths {
                guard let s = sides.first(where: { s in path.allSatisfy { onSide($0.xy.v, s) } }) else { continue }
                let u = simd_normalize(s.b - s.a), n = SIMD2(-u.y, u.x)
                // Its ends in the footprint, on its perimeter at the most.
                for e in [path[0], path[path.count - 1]] where !inFootprint(e.xy.v, l.region, n: n, size: size) {
                    stray.append("an end at \(e.xy.v) off the footprint")
                }
                for (a, c) in zip(path, path.dropFirst()) where simd_length(a.xy.v - c.xy.v) > 1e-12 * size {
                    let m = 0.5 * (a.xy.v + c.xy.v)
                    guard drop(m, n) != nil || l.region.contains(P2(m - side * n)) && l.region.contains(P2(m + side * n)) else {
                        stray.append("\(m) off the footprint"); continue
                    }
                    segments += 1
                    along.append((a, c))
                    guard let span = drop(m, n) else { stray.append("\(m) over no drop"); continue }
                    if !(abs(0.5 * (a.z + c.z) - span.hi) <= bound) { stray.append("\(m) at \(0.5 * (a.z + c.z)), the higher top \(span.hi)") }
                }
            }
            for s in sides {
                let u = simd_normalize(s.b - s.a), n = SIMD2(-u.y, u.x)
                for j in 0..<47 {
                    let w = s.a + (s.b - s.a) * ((Double(j) + 0.53) / 47)
                    guard let span = drop(w, n) else { continue }
                    asked += 1
                    let over = along.contains { a, c in
                        let d = c.xy.v - a.xy.v, t = simd_dot(w - a.xy.v, d) / simd_length_squared(d)
                        return t >= 0 && t <= 1 && simd_length(w - (a.xy.v + t * d)) <= 1e-9 * size
                            && abs(a.z + (c.z - a.z) * t - span.hi) <= bound
                    }
                    if !over { missing.append("\(w), to \(span.hi)") }
                }
            }
            Check.expect(stray.isEmpty && missing.isEmpty && (!l.dropsAtATileBoundary || (asked > 30 && segments > 10)),
                         "\(l.name): the truncation edge draws a riser's top at a tile boundary where the riser stands, at the higher top, and all of it there",
                         "\(segments) segments along a tile boundary, \(stray.count) stray; \(missing.count) of \(asked) points with no top: "
                            + (stray.prefix(2) + missing.prefix(2)).joined(separator: "; "))
        }

        // A band boundary on the domain's far edge draws as one a cell beyond
        // it: the band beyond starts outside the domain, and the landscape
        // ends with the band before. The depth pass's grid and truncation-edge
        // mesh, the walls' curtain, top edges and hatching, the truncation
        // edge, the graph's trim loops and the plateaus' hatching, each the
        // same; on the grid and on f.
        func moved(_ truncation: Truncation, from hi: Double, to x: Double) -> Truncation {
            guard case .realBands(let bands, let beyond) = truncation else { return truncation }
            return .realBands(bands.map { RealBand(below: $0.below == hi ? x : $0.below, height: $0.height) }, beyond: beyond)
        }
        var pairs: [(String, Surface, Surface, Walls, ContourRefine?, ContourRefine?)] = []
        if let edge = landscapes.first(where: { $0.name == "the hatch fixture, a band boundary on the far edge" }) {
            let s = edge.surface, hi = max(s.domain.real.lo, s.domain.real.hi)
            let cell = abs(s.domain.real.length) / Double(s.height.width - 1)
            pairs.append((edge.name, s, Surface(height: s.height, phase: s.phase, truncation: moved(s.truncation, from: hi, to: hi + cell)),
                          edge.walls, nil, nil))
        }
        let towerDomain = Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1))
        let towerWalls = BoundaryPerimeter(edges: [(-0.5, -1.0), (1.5, -1.0), (1.5, 1.0), (-0.5, 1.0)].indices.map { k in
            let c = [(-0.5, -1.0), (1.5, -1.0), (1.5, 1.0), (-0.5, 1.0)]
            let a = c[k], b = c[(k + 1) % c.count]
            return PerimeterEdge(start: P2(a.0, a.1), end: P2(b.0, b.1), density: 33)
        })
        let onEdge = Truncation.realBands([RealBand(below: 0.5, height: 2), RealBand(below: 1.5, height: 4)], beyond: 1)
        let tower = try NativeLandscape.build(LandscapeRequest(expression: "1/(z - 0.5)", domain: towerDomain, resolution: 32,
                                                               truncation: onEdge), refine: true)
        let past = try NativeLandscape.build(LandscapeRequest(expression: "1/(z - 0.5)", domain: towerDomain, resolution: 32,
                                                              truncation: moved(onEdge, from: 1.5, to: 1.5 + 2.0 / 31)), refine: true)
        pairs.append(("a tower banded at 2 | 4 to the far edge and 1 beyond, on f", tower.surface, past.surface,
                      .perimeter(towerWalls, base: 0), tower.refine, past.refine))
        // 1/(z³ - 1) on the cubic preset's window, Re z in [-1.3, 1.6], whose
        // grid's last column a rounding past 1.6 reads its truncation there.
        let window = Domain(real: Interval(lo: -1.3, hi: 1.6), imag: Interval(lo: -1.2, hi: 1.5))
        let windowWalls = BoundaryPerimeter(edges: [(-1.3, -1.2), (1.6, -1.2), (1.6, 1.5), (-1.3, 1.5)].indices.map { k in
            let c = [(-1.3, -1.2), (1.6, -1.2), (1.6, 1.5), (-1.3, 1.5)]
            let a = c[k], b = c[(k + 1) % c.count]
            return PerimeterEdge(start: P2(a.0, a.1), end: P2(b.0, b.1), density: 33)
        })
        let cubicBands = Truncation.realBands([RealBand(below: 0.2, height: 2.5), RealBand(below: 1.6, height: 5)], beyond: 0.3)
        let cubic = try NativeLandscape.build(LandscapeRequest(expression: "1/(z^3 - 1)", domain: window, resolution: 64,
                                                               truncation: cubicBands), refine: true)
        let cubicCell = 2.9 / Double(cubic.surface.height.width - 1)
        let cubicPast = try NativeLandscape.build(LandscapeRequest(expression: "1/(z^3 - 1)", domain: window, resolution: 64,
                                                                   truncation: moved(cubicBands, from: 1.6, to: 1.6 + cubicCell)),
                                                  refine: true)
        let lastColumn = cubic.surface.height.position(x: cubic.surface.height.width - 1, y: 0).x
        Check.expect(lastColumn > 1.6, "the cubic window's last column lies a rounding past Re z = 1.6", "\(lastColumn)")
        pairs.append(("1/(z³ - 1) at 2.5 | 5 to the far edge at 1.6 and 0.3 beyond, its last column past it, on f",
                      cubic.surface, cubicPast.surface, .perimeter(windowWalls, base: 0), cubic.refine, cubicPast.refine))
        for (name, s, t, walls, refine, refinePast) in pairs {
            guard case .perimeter(let perimeter, let base) = walls else { continue }
            let context = Surface.HatchContext(perimeter: perimeter, region: .full, tiles: [.identity])
            let edges = Array(perimeter.edges.indices)
            let layers: [(String, LayerSource)] = [
                ("the wall edges", .wallEdges(edges: edges, step: 0.1, base: base)),
                ("the wall hatching", .hatching(Hatching(angle: HatchedFaces.wallAngle, spacing: 0.05),
                                                 faces: .walls(edges: edges, trim: false, base: base, topOffset: 0))),
                ("the truncation edge", .truncationEdge(tiled: false)),
                ("the plateaus' hatching along Re z", .hatching(Hatching(angle: 0, spacing: 0.05), faces: .plateaus(tiled: false))),
                ("the plateaus' hatching along Im z", .hatching(Hatching(angle: .pi / 2, spacing: 0.05), faces: .plateaus(tiled: false))),
            ]
            var differ: [String] = []
            if s.clamped().values != t.clamped().values { differ.append("the depth pass's grid") }
            if Mesh.truncationEdgeMesh(of: s, step: 1, region: .full, tiles: [.identity])
                != Mesh.truncationEdgeMesh(of: t, step: 1, region: .full, tiles: [.identity]) {
                differ.append("the truncation-edge mesh")
            }
            if Mesh.walls(of: perimeter, surface: s, base: base) != Mesh.walls(of: perimeter, surface: t, base: base) {
                differ.append("the curtain")
            }
            for (what, layer) in layers
            where s.hatch(layer, in: context, refine: refine) != t.hatch(layer, in: context, refine: refinePast) {
                differ.append(what)
            }
            let loops = s.graphTrimLoops(refine: refine), loopsPast = t.graphTrimLoops(refine: refinePast)
            if loops.loops != loopsPast.loops || loops.open != 0 { differ.append("the graph's trim loops") }
            Check.expect(differ.isEmpty, "\(name): drawn as with the band boundary a cell past the far edge",
                         "differ: \(differ.joined(separator: ", "))")
        }
    }

    Check.suite("depth: the depth pass's walls step where they cross a tile boundary, and a riser fills the drop between two copies") {
        // The curtain of a wall that crosses a tile boundary where the copies'
        // tops differ steps there, as its top edge does: no chord of its top
        // lies under two tiles across a drop. Where they agree it does not
        // step: no two of its samples at one point. And
        // where two copies meet with different tops, the riser strip the
        // depth pass draws (`Mesh.tileBoundaryRisers`) covers the drop
        // between the tops the depth pass draws a hair either side
        // (`Heightfield.drawnHeight`), read at seven heights at points along
        // each side of a tile's image that another tile's image meets; and
        // every triangle of it stands over a drop, its centroid between the
        // two tops. On the grid.
        for l in try faceLandscapes() where l.tiles.count > 1 && l.refine == nil {
            let real = l.surface.domain.real, imag = l.surface.domain.imag
            let size = max(abs(real.length), abs(imag.length)), side = 1e-9 * size
            let h = l.heightfield
            var chords = 0, ramps: [String] = []
            if case .perimeter(let perimeter, let base) = l.walls {
                for k in perimeter.edges.indices {
                    let top = curtain(perimeter, k, surface: l.surface, base: base, tiles: l.tiles).top
                    let inward = perimeter.inward(k)
                    let d = simd_normalize(perimeter.edges[k].end.v - perimeter.edges[k].start.v)
                    for (a, c) in zip(top, top.dropFirst()) where simd_length(a.xy.v - c.xy.v) <= 1e-12 * size {
                        // Read along the wall either side.
                        if !dropAcross(a.xy.v - 1e-6 * size * d, a.xy.v + 1e-6 * size * d, inward: inward, l,
                                       top: { h.surfaceHeight(at: P2($0)) }).drops {
                            ramps.append("a step at \(a.xy.v) where the copies' tops agree")
                        }
                    }
                    for (a, c) in zip(top, top.dropFirst()) where simd_length(a.xy.v - c.xy.v) > 1e-12 * size {
                        chords += 1
                        let near = a.xy.v + (c.xy.v - a.xy.v) / 32, far = a.xy.v + (c.xy.v - a.xy.v) * 31 / 32
                        if carriedBack(near + side * inward, l).back != carriedBack(far + side * inward, l).back,
                           dropAcross(a.xy.v, c.xy.v, inward: inward, l, top: { h.surfaceHeight(at: P2($0)) }).drops {
                            ramps.append("\(a.v) to \(c.v)")
                        }
                    }
                }
            }
            let mesh = Mesh.tileBoundaryRisers(of: l.surface, step: 1, region: l.region, tiles: l.tiles)
            let triangles = mesh.triangles.map { t in (0..<3).map { mesh.vertices[Int(t[$0])].v } }
            /// Whether a point lies on a triangle of the strip, which is
            /// vertical: in its plane and inside it.
            func covered(_ p: SIMD3<Double>) -> Bool {
                triangles.contains { t in
                    let n = simd_cross(t[1] - t[0], t[2] - t[0])
                    guard simd_length(n) > 0, abs(simd_dot(p - t[0], n)) <= 1e-9 * simd_length(n) * max(1, simd_length(p)) else { return false }
                    let edges = (0..<3).map { simd_dot(simd_cross(t[($0 + 1) % 3] - t[$0], p - t[$0]), n) }
                    return edges.allSatisfy { $0 >= -1e-9 * simd_length_squared(n) } || edges.allSatisfy { $0 <= 1e-9 * simd_length_squared(n) }
                }
            }
            let (lo, hi) = (min(real.lo, real.hi), max(real.lo, real.hi)), (ylo, yhi) = (min(imag.lo, imag.hi), max(imag.lo, imag.hi))
            let corners = [SIMD2(lo, ylo), SIMD2(hi, ylo), SIMD2(hi, yhi), SIMD2(lo, yhi)]
            func cover(_ g: SIMD2<Double>) -> Int {
                l.tiles.filter { tile in
                    guard let back = tile.inverse else { return false }
                    let q = back(P3(g.x, g.y, 0))
                    return q.x >= lo && q.x <= hi && q.y >= ylo && q.y <= yhi
                }.count
            }
            /// The drawn drop at a point of a tile's side: the depth pass's
            /// top a hair either side, where one tile holds each, two
            /// different ones, and the footprint both; nil without a drop.
            func drop(_ w: SIMD2<Double>, _ n: SIMD2<Double>) -> (lo: Double, hi: Double)? {
                let reads = [-1.0, 1.0].map { sign -> (z: Double, back: Affine2, slide: Double)? in
                    let g = w + sign * side * n
                    guard cover(g) == 1, l.region.contains(P2(g)) else { return nil }
                    let z = h.drawnHeight(at: P2(g))
                    return z.isFinite ? (z, carriedBack(g, l).back, abs(h.drawnHeight(at: P2(w + 2 * sign * side * n)) - z)) : nil
                }
                guard let a = reads[0], let b = reads[1], a.back != b.back,
                      abs(a.z - b.z) > 4 * max(a.slide, b.slide) + 1e-9 * max(1, max(a.z, b.z)) else { return nil }
                return (min(a.z, b.z), max(a.z, b.z))
            }
            var asked = 0, uncovered: [String] = []
            for tile in l.tiles {
                for e in 0..<4 {
                    let a = tile(P3(corners[e].x, corners[e].y, 0)).xy.v, b = tile(P3(corners[(e + 1) % 4].x, corners[(e + 1) % 4].y, 0)).xy.v
                    let u = simd_normalize(b - a), n = SIMD2(-u.y, u.x)
                    for j in 0..<47 {
                        let w = a + (b - a) * ((Double(j) + 0.53) / 47)
                        guard let span = drop(w, n) else { continue }
                        for k in 0..<7 {
                            asked += 1
                            let z = span.lo + (span.hi - span.lo) * (Double(k) + 0.5) / 7
                            if !covered(SIMD3(w.x, w.y, z)) { uncovered.append("(\(w.x), \(w.y), \(z))") }
                        }
                    }
                }
            }
            var stray: [String] = []
            for t in triangles {
                let c = (t[0] + t[1] + t[2]) / 3
                let u = simd_normalize(SIMD2(t[1].x - t[0].x, t[1].y - t[0].y) + SIMD2(t[2].x - t[0].x, t[2].y - t[0].y))
                let n = SIMD2(-u.y, u.x)
                // Every corner in the footprint, on its perimeter at the most.
                for v in t where !inFootprint(SIMD2(v.x, v.y), l.region, n: n, size: size) {
                    stray.append("a corner at (\(v.x), \(v.y)) off the footprint")
                }
                guard let span = drop(SIMD2(c.x, c.y), n), c.z >= span.lo - 1e-9, c.z <= span.hi + 1e-9 else {
                    stray.append("(\(c.x), \(c.y), \(c.z))"); continue
                }
            }
            guard chords > 0 || asked > 0 || !triangles.isEmpty else { continue }
            Check.expect(ramps.isEmpty && uncovered.isEmpty && stray.isEmpty && (!l.dropsAtATileBoundary || (asked > 200 && triangles.count > 10)),
                         "\(l.name): the curtain steps at a tile boundary, and the riser strip covers each drop and nothing else",
                         "\(ramps.count) of \(chords) curtain chords under two tiles, \(uncovered.count) of \(asked) points of a drop "
                            + "uncovered, \(stray.count) of \(triangles.count) triangles over no drop: "
                            + (ramps.prefix(1) + uncovered.prefix(2) + stray.prefix(2)).joined(separator: "; "))
        }
    }

    Check.suite("depth: a bundle's occluder holds the risers at its tile boundaries") {
        // The reviewer's tower, a bundle on f, its manifest given a second
        // tile turned beside the first: the occluder the depth pass draws
        // (`KurvenBundle.occluder()`) has every triangle of the riser strip.
        let bundle = try NativeLandscape.build(
            LandscapeRequest(expression: "1/(z - 1.4)", domain: Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1)),
                             resolution: 32, truncation: .realBands([RealBand(below: 1.45, height: 0.5)], beyond: 1)),
            refine: true)
        var manifest = bundle.manifest
        manifest.occluder.tiles = [.identity, Affine2(a: 0, b: 1, tx: 2.5, c: -1, d: 0, ty: 0.5)]
        let tiled = KurvenBundle(url: bundle.url, manifest: manifest, surface: bundle.surface, layers: bundle.layers,
                                 wallMesh: bundle.wallMesh, refine: bundle.refine)
        let o = manifest.occluder
        let strip = Mesh.tileBoundaryRisers(of: tiled.surface, step: o.step, region: o.region, tiles: o.tiles)
        let occluder = tiled.occluder()
        func triangles(_ m: Mesh<WorldSpace>) -> Set<[SIMD3<Double>]> {
            Set(m.triangles.map { t in (0..<3).map { m.vertices[Int(t[$0])].v } })
        }
        let held = triangles(occluder)
        let missing = triangles(strip).filter { !held.contains($0) }
        Check.expect(strip.triangles.count > 10 && missing.isEmpty,
                     "the tower's occluder, a tile turned beside it, holds the riser strip at the tile boundary",
                     "\(missing.count) of \(strip.triangles.count) strip triangles missing")
    }

    Check.suite("visibility: a line of sight that passes a tile boundary under the higher copy's top is hidden, whichever tile comes first") {
        // |f| = max(1/5, 2 - 20(1 - Re z)) on the unit square, steep up to 2
        // at its right edge, the second tile turned, (x, y) ↦ (1 + y, 1 - x),
        // whose bottom edge meets the first's right edge at 1/5: a riser 1.8
        // high faces the second copy. A point on the second copy a hair from
        // the tile boundary, its line of sight toward -Re z rising 80/3 a
        // unit, passes the tile boundary at 1 and is over the first copy
        // again a column on: it is hidden, the riser in its way, with either
        // tile listed first. A point 1.3 higher passes over the riser and is
        // seen. On the grid.
        let unit = Domain(real: Interval(lo: 0, hi: 1), imag: Interval(lo: 0, hi: 1))
        let steep = Surface(height: Grid2D(width: 21, height: 21, domain: unit,
                                           values: (0..<441).map { k in Float(max(0.2, 2 - 20 * (1 - Double(k % 21) / 20))) }),
                            phase: nil, truncation: .uniform(10))
        let turned = Affine2(a: 0, b: 1, tx: 1, c: -1, d: 0, ty: 1)
        let toward = simd_normalize(SIMD3(-1.0, 0, 80.0 / 3))
        let r0 = SIMD3(0.0, 1, 0), r1 = simd_normalize(simd_cross(toward, r0))
        let view = Transform<WorldSpace, ViewSpace>(rows: (r0, r1, toward))
        for (name, tiles) in [("the identity first", [Affine2.identity, turned]), ("the turned tile first", [turned, .identity])] {
            let h = Heightfield(surface: steep, occluder: .empty, tiles: tiles, region: .full, step: 1)
            let behind = h.visibleCurves(PolylineSet(paths: [[P3(1.03, 0.5, 0.2), P3(1.03, 0.45, 0.2)]]), view: view, depthTolerance: 1e-6)
            let over = h.visibleCurves(PolylineSet(paths: [[P3(1.03, 0.5, 1.5), P3(1.03, 0.45, 1.5)]]), view: view, depthTolerance: 1e-6)
            Check.expect(behind.vertices.isEmpty && over.vertices.count == 2,
                         "\(name): a point behind the riser is hidden, and one over it seen",
                         "\(behind.vertices.count) of 2 behind seen, \(over.vertices.count) of 2 over seen")
        }
    }

    Check.suite("folds: on tiles under a banded truncation, a fold runs on the graph wherever the facing turns, read through the tile") {
        // The truncation under a fold is read at the point's own x through
        // the tile under it, as the surface is: every vertex of a fold lies
        // on the graph, |f| under the truncation there, or on its
        // truncation edge; and wherever the facing changes sign between two
        // points of the graph in the footprint, well under the truncation,
        // a fold passes between them.
        // A tower, 1/(z - 1/2), banded on either side of its pole, 4 | 2 and
        // 2 | 4, in two tiles, the second reflected across the right edge or
        // turned a quarter, (x, y) ↦ (5/2 - y, x): the truncation under the
        // second differs from the one at the world x by a band about the
        // pole's image. On f and on the grid.
        var landscapes: [Landscape] = []
        let domain = Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1))
        let reflected = Affine2(a: -1, b: 0, tx: 3, c: 0, d: 1, ty: 0), turned = Affine2(a: 0, b: -1, tx: 2.5, c: 1, d: 0, ty: 0)
        for (left, right, tile, name, onF) in [(4.0, 2.0, reflected, "reflected", true), (2, 4, turned, "turned", true),
                                               (2, 4, reflected, "reflected", false), (4, 2, turned, "turned", false)] {
            let tower = try NativeLandscape.build(
                LandscapeRequest(expression: "1/(z - 0.5)", domain: domain, resolution: 48,
                                 truncation: .realBands([RealBand(below: 0.5, height: left)], beyond: right)),
                refine: onF)
            landscapes.append(Landscape("a tower banded at \(Int(left)) | \(Int(right)) in two tiles, the second \(name), \(onF ? "on f" : "on the grid")",
                                        surface: tower.surface, tiles: [.identity, tile], refine: tower.refine))
        }
        let camera = Camera.oblique(ObliqueProjection(shear: 0, xAngle: -55, zAngle: 30, flipX: false, yScale: nil))
        // The direction lines of sight travel: the kernel of the view's first
        // two rows, away from the eye.
        let m = camera.view.m.columns
        let rows = (SIMD3(m.0.x, m.1.x, m.2.x), SIMD3(m.0.y, m.1.y, m.2.y), SIMD3(m.0.z, m.1.z, m.2.z))
        var sight = simd_normalize(simd_cross(rows.0, rows.1))
        if simd_dot(rows.2, sight) > 0 { sight = -sight }
        for l in landscapes {
            let h = l.heightfield, surface = l.surface
            let real = surface.domain.real, imag = surface.domain.imag
            let cell = max(abs(real.length) / Double(surface.height.width - 1), abs(imag.length) / Double(surface.height.height - 1))
            guard case .realBands(let bands, _) = surface.truncation else { continue }
            let boundaries = bands.map(\.below)
            func magnitude(_ q: SIMD2<Double>) -> Double {
                let u = l.refine?.magnitude?(P2(q.x, q.y)) ?? surface.magnitude(at: P2(q.x, q.y))
                return u.isNaN ? .infinity : u
            }
            let folds = h.foldCurves(view: camera.view)
            var vertices = 0, off: [String] = []
            for v in folds.vertices {
                let q = carriedBack(v.xy.v, l).q
                guard !boundaries.contains(where: { abs(q.x - $0) <= 1e-9 * cell }) else { continue }
                vertices += 1
                let top = surface.truncationHeight(atX: q.x)
                if !(magnitude(q) <= top * (1 + 1e-6)) { off.append("\(v.v), |f| \(magnitude(q)) over \(top)") }
            }
            // The facing on a grid of each tile's image, and the folds near
            // each sign change.
            let traced = Segments((0..<folds.count).map { folds[path: $0].map(\.xy.v) }, cell: cell)
            func facing(_ w: SIMD2<Double>) -> Double {
                let n = l.refine?.magnitude != nil ? h.graphNormal(at: P2(w)) : h.normal(at: P2(w))
                return -simd_dot(n, sight) / simd_length(n)
            }
            var turns = 0, missing: [String] = []
            for tile in l.tiles {
                let (nx, ny) = (161, 97)
                func point(_ i: Int, _ j: Int) -> (p: SIMD2<Double>, w: SIMD2<Double>) {
                    let p = SIMD2(real.lo + real.length * (Double(i) + 0.37) / Double(nx), imag.lo + imag.length * (Double(j) + 0.61) / Double(ny))
                    return (p, tile(P3(p.x, p.y, 0)).xy.v)
                }
                /// On the graph well under the truncation, away from a band
                /// boundary, the domain's edge and the footprint's, and held
                /// by this tile alone.
                func clear(_ p: SIMD2<Double>, _ w: SIMD2<Double>) -> Bool {
                    let reach = 2 * cell
                    guard p.x - reach > min(real.lo, real.hi), p.x + reach < max(real.lo, real.hi),
                          p.y - reach > min(imag.lo, imag.hi), p.y + reach < max(imag.lo, imag.hi),
                          !boundaries.contains(where: { abs(p.x - $0) <= reach }),
                          [SIMD2(reach, 0), SIMD2(-reach, 0), SIMD2(0, reach), SIMD2(0, -reach), .zero]
                              .allSatisfy({ l.region.contains(P2(w + $0)) }),
                          carriedBack(w, l).back == tile.inverse else { return false }
                    return magnitude(p) < 0.98 * surface.truncationHeight(atX: p.x)
                }
                for j in 0..<ny {
                    var previous: (p: SIMD2<Double>, w: SIMD2<Double>, facing: Double)?
                    for i in 0..<nx {
                        let (p, w) = point(i, j)
                        let here = (p, w, facing(w))
                        defer { previous = here }
                        guard let a = previous, (a.facing > 0) != (here.2 > 0), clear(a.p, a.w), clear(p, w),
                              clear(0.5 * (a.p + p), 0.5 * (a.w + w)) else { continue }
                        turns += 1
                        let mid = 0.5 * (a.w + w), reach = simd_length(w - a.w) + cell
                        if traced.distance(mid, within: reach).isInfinite { missing.append("\(mid)") }
                    }
                }
            }
            Check.expect(vertices > 50 && off.isEmpty && turns > 20 && missing.isEmpty,
                         "\(l.name): every fold vertex on the graph, and a fold wherever the facing turns on it",
                         "\(off.count) of \(vertices) vertices on a plateau, \(missing.count) of \(turns) turns with no fold: "
                            + (off.prefix(2) + missing.prefix(2)).joined(separator: "; "))
        }
    }

    Check.suite("folds: a fold that runs into a zero of f ends on it, on every tile") {
        // A cone, 10(z - 0.3), over Im z in [0, 1], its apex on the domain's
        // edge, in three tiles, the second reflected across Re z = 1 and the
        // third turned a quarter, (x, y) ↦ (4 - y, x): its outline runs down
        // a generator into the apex in each copy, and each copy's fold is
        // carried onto the apex's image there, at its height, the floor,
        // where placed alone it stops a hair short.
        let cone = try NativeLandscape.build(
            LandscapeRequest(expression: "10*(z - 0.3)", domain: Domain(real: Interval(lo: -1, hi: 1), imag: Interval(lo: 0, hi: 1)),
                             resolution: 30, truncation: .uniform(30)),
            refine: true)
        let tiles = [Affine2.identity, Affine2(a: -1, b: 0, tx: 2, c: 0, d: 1, ty: 0), Affine2(a: 0, b: -1, tx: 4, c: 1, d: 0, ty: 0)]
        let h = Heightfield(surface: cone.surface, occluder: .empty, tiles: tiles, region: .full, step: 1, refine: cone.refine)
        // Under cameras turned every quarter, so each copy shows the apex's
        // outline under some: every end near an apex is on it, and every
        // tile has one.
        var report: [String] = [], carried = [Int](repeating: 0, count: tiles.count)
        for zAngle in [30.0, 120.0, -150.0, -60.0] {
            let camera = Camera.oblique(ObliqueProjection(shear: 0, xAngle: -55, zAngle: zAngle, flipX: false, yScale: nil))
            let folds = h.foldCurves(view: camera.view)
            let ends = (0..<folds.count).flatMap { k -> [P3<WorldSpace>] in let r = folds[path: k]; return [r[r.startIndex], r[r.endIndex - 1]] }
            for (k, tile) in tiles.enumerated() {
                let apex = tile(P3(0.3, 0, 0)).xy.v
                let near = ends.filter { simd_length($0.xy.v - apex) < 0.2 }
                let onIt = near.filter { simd_length($0.xy.v - apex) <= 1e-6 && abs($0.z) <= 1e-6 }
                carried[k] += onIt.count
                if onIt.count != near.count {
                    report.append("tile \(k) at \(Int(zAngle))°: \(onIt.count) of \(near.count) ends near the apex on it, nearest "
                                  + "\(near.map { simd_length($0.xy.v - apex) }.min() ?? .infinity)")
                }
            }
        }
        Check.expect(report.isEmpty && carried.allSatisfy { $0 > 0 },
                     "the cone's folds end on the apex on every tile, at its height",
                     "ends on an apex by tile \(carried): \(report.prefix(3).joined(separator: "; "))")
    }

    Check.suite("folds: every chord of a fold lies within the refiner's tolerance of the fold, on every tile") {
        // The midpoint of each chord, on the graph under the truncation and
        // off the domain's edge, is within the refiner's tolerance of the
        // fold: the nearest sign change of the facing along the chord's
        // normal, bisected, read through the tile under each point as the
        // facing is. On f, in two tiles, the second reflected, turned a
        // quarter or beside the first; a fold's chords on a second tile are
        // as fine as on the first.
        let domain = Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1))
        let camera = Camera.oblique(ObliqueProjection(shear: 0, xAngle: -55, zAngle: 30, flipX: false, yScale: nil))
        let m = camera.view.m.columns
        let rows = (SIMD3(m.0.x, m.1.x, m.2.x), SIMD3(m.0.y, m.1.y, m.2.y), SIMD3(m.0.z, m.1.z, m.2.z))
        var sight = simd_normalize(simd_cross(rows.0, rows.1))
        if simd_dot(rows.2, sight) > 0 { sight = -sight }
        for (expression, resolution) in [("2*z*z*z - z", 20), ("1/(z - 0.5) + 0.3*z*z", 24), ("1/(z - 0.5)", 16)] {
            let bundle = try NativeLandscape.build(
                LandscapeRequest(expression: expression, domain: domain, resolution: resolution, truncation: .uniform(5)),
                refine: true)
            guard let refine = bundle.refine, let magnitude = refine.magnitude else { continue }
            let surface = bundle.surface
            let cell = max(abs(domain.real.length) / Double(surface.height.width - 1),
                           abs(domain.imag.length) / Double(surface.height.height - 1))
            for (name, tile) in [("reflected", Affine2(a: -1, b: 0, tx: 3, c: 0, d: 1, ty: 0)),
                                 ("turned", Affine2(a: 0, b: -1, tx: 2.5, c: 1, d: 0, ty: 0)),
                                 ("beside", Affine2(a: 1, b: 0, tx: 2, c: 0, d: 1, ty: 0))] {
                let l = Landscape("\(expression) at res \(resolution), a second tile \(name)", surface: surface,
                                  tiles: [.identity, tile], refine: refine)
                let h = l.heightfield
                func facing(_ w: SIMD2<Double>) -> Double {
                    let n = h.graphNormal(at: P2(w))
                    return -simd_dot(n, sight) / simd_length(n)
                }
                let folds = h.foldCurves(view: camera.view)
                var chords = [0, 0], worst = [0.0, 0.0], off: [String] = []
                for k in 0..<folds.count {
                    let run = Array(folds[path: k])
                    for (a, c) in zip(run, run.dropFirst()) {
                        let mid = 0.5 * (a.xy.v + c.xy.v), length = simd_length(c.xy.v - a.xy.v)
                        guard length > 1e-9 * cell else { continue }
                        let (q, back) = carriedBack(mid, l)
                        guard q.x > domain.real.lo + cell, q.x < domain.real.hi - cell,
                              q.y > domain.imag.lo + cell, q.y < domain.imag.hi - cell,
                              magnitude(P2(q.x, q.y)) < 0.9 * surface.truncationHeight(atX: q.x) else { continue }
                        let n = SIMD2(-(c.y - a.y), c.x - a.x) / length
                        let f0 = facing(mid)
                        var distance = Double.infinity
                        for sign in [1.0, -1.0] {
                            var lo = 0.0, hi = 0.5 * cell * sign
                            guard (facing(mid + hi * n) > 0) != (f0 > 0) else { continue }
                            for _ in 0..<60 {
                                let t = 0.5 * (lo + hi)
                                if (facing(mid + t * n) > 0) == (f0 > 0) { lo = t } else { hi = t }
                            }
                            distance = min(distance, abs(0.5 * (lo + hi)))
                        }
                        let t = back == .identity ? 0 : 1
                        chords[t] += 1
                        worst[t] = max(worst[t], distance / cell)
                        if !(distance <= refine.tolerance * cell) {
                            off.append(String(format: "tile %d at (%.4f, %.4f), %.3f cells", t, mid.x, mid.y, distance / cell))
                        }
                    }
                }
                Check.expect(chords.allSatisfy { $0 > 30 } && off.isEmpty,
                             "\(l.name), on f: every chord of a fold within \(refine.tolerance) cells of the fold, on both tiles",
                             "\(off.count) of \(chords) chords off, worst \(worst.map { String(format: "%.4f", $0) }) cells: "
                                + off.prefix(3).joined(separator: "; "))
            }
        }
    }

    Check.suite("faces: the graph's trim loops and its truncation edge are the same in whatever order the tiles are listed") {
        // Points of the domain are read on the domain, not through the tile
        // under them: on the domain's edge a second tile's image begins, and
        // listed first it was the tile under them. On f: 1/(z - 0.95), cut at
        // 3, its truncation edge ending on the right edge, a second tile
        // beside it there; a tower banded at 2 | 4, its riser's top, a second
        // tile beside it or reflected across the right edge.
        let pole = try NativeLandscape.build(
            LandscapeRequest(expression: "1/(z - 0.95)", domain: Domain(real: Interval(lo: 0, hi: 1), imag: Interval(lo: -0.5, hi: 0.5)),
                             resolution: 40, truncation: .uniform(3)),
            refine: true)
        let tower = try NativeLandscape.build(
            LandscapeRequest(expression: "1/(z - 0.5)", domain: Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1)),
                             resolution: 32, truncation: .realBands([RealBand(below: 0.5, height: 2)], beyond: 4)),
            refine: true)
        for (name, bundle, tile) in [
            ("1/(z - 0.95) cut at 3, a tile beside it", pole, Affine2(a: 1, b: 0, tx: 1, c: 0, d: 1, ty: 0)),
            ("a tower banded at 2 | 4, a tile beside it", tower, Affine2(a: 1, b: 0, tx: 2, c: 0, d: 1, ty: 0)),
            ("a tower banded at 2 | 4, a tile reflected across its right edge", tower, Affine2(a: -1, b: 0, tx: 3, c: 0, d: 1, ty: 0)),
        ] {
            func drawn(_ tiles: [Affine2]) -> (loops: [[P2<ParamSpace>]], edge: [SIMD3<Double>]) {
                let graph = bundle.surface.faces(tiles: tiles, refine: bundle.refine).first
                let edge = bundle.surface.hatch(.truncationEdge(tiled: true), in: Surface.HatchContext(perimeter: nil, tiles: tiles),
                                                refine: bundle.refine)?.vertices.map(\.v) ?? []
                return (graph?.trimLoops ?? [], edge.sorted { ($0.x, $0.y, $0.z) < ($1.x, $1.y, $1.z) })
            }
            let first = drawn([.identity, tile]), second = drawn([tile, .identity])
            let open = second.loops.filter { $0.first != $0.last }.count + first.loops.filter { $0.first != $0.last }.count
            Check.expect(!first.loops.isEmpty && first.loops.map { $0.map(\.v) } == second.loops.map { $0.map(\.v) }
                            && open == 0 && !first.edge.isEmpty && first.edge == second.edge,
                         "\(name): the trim loops, closed, and the truncation edge on every tile, the identity listed first or second",
                         "\(open) loops open, loops \(first.loops.map { $0.map(\.v) } == second.loops.map { $0.map(\.v) } ? "the same" : "differ"), "
                            + "\(first.edge.count) and \(second.edge.count) vertices drawn, \(first.edge == second.edge ? "the same" : "different")")
        }
    }

    Check.suite("restriction: a contour's copy is cut to the footprint where it lands, exactly") {
        // On tiles whose footprint is not symmetric under the tiling,
        // reflected and turned: every kept piece of a copy lies in the
        // footprint where the copy stands, ending on its perimeter or where
        // the contour ends, and the copy is kept wherever it is in the
        // footprint, read sixteen times a segment. With a band of Im z as
        // well, read on the domain the copy came from. And an untiled layer,
        // its one copy in place, wherever the footprint cuts the domain.
        let landscapes = try faceLandscapes().filter { l in
            guard case .inside = l.region, l.hasGraph else { return false }
            return true
        }
        for l in landscapes {
          for tiled in [true, false] {
            let copies = tiled ? l.tiles : [.identity]
            // Untiled, only where the footprint cuts the domain, holding a
            // twentieth of it and leaving a twentieth.
            if !tiled {
                let d = l.surface.domain
                let held = (0..<400).filter { k in
                    l.region.contains(P2(d.real.lo + d.real.length * (Double(k % 20) + 0.5) / 20,
                                         d.imag.lo + d.imag.length * (Double(k / 20) + 0.5) / 20))
                }.count
                guard held >= 20, held <= 380 else { continue }
            } else if l.tiles.count == 1 {
                continue
            }
            guard case .inside(let perimeter) = l.region else { continue }
            let surface = l.surface
            let real = surface.domain.real, imag = surface.domain.imag
            let size = max(abs(real.length), abs(imag.length))
            let cell = max(abs(real.length) / Double(surface.height.width - 1), abs(imag.length) / Double(surface.height.height - 1))
            let tolerance = 1e-9 * size
            let finite = surface.height.values.filter(\.isFinite).map(Double.init).sorted()
            let (lo, hi) = (finite[finite.count / 10], min(finite[9 * finite.count / 10], 6))
            let levels = (1...7).map { lo + (hi - lo) * Double($0) / 8 }
            let band = (lo: imag.lo + 0.3 * imag.length, hi: imag.lo + 0.7 * imag.length)
            let outline = Segments([perimeter.edges.map(\.start.v) + [perimeter.edges[0].start.v]], cell: cell)
            func cover(_ w: SIMD2<Double>) -> Int {
                l.tiles.filter { tile in
                    guard let back = tile.inverse else { return false }
                    let q = back(P3(w.x, w.y, 0))
                    return q.x >= min(real.lo, real.hi) && q.x <= max(real.lo, real.hi)
                        && q.y >= min(imag.lo, imag.hi) && q.y <= max(imag.lo, imag.hi)
                }.count
            }
            for (what, restriction, banded) in [
                ("the footprint", Restriction.footprint, false),
                ("the footprint and a band of Im z", .intersection([.footprint, .band(axis: .imag, lo: band.lo, hi: band.hi)]), true),
            ] {
                func derived(_ r: Restriction) -> PolylineSet<WorldSpace> {
                    surface.derive(.contour(field: .magnitude, levels: levels, restriction: r, tiled: tiled),
                                   policy: .magnitude, region: l.region, tiles: l.tiles, refine: l.refine)
                }
                let whole = derived(.all), kept = derived(restriction)
                guard whole.count % copies.count == 0 else {
                    Check.expect(false, "\(l.name), \(what): the whole contours come a copy per tile"); continue
                }
                let perCopy = whole.count / copies.count
                let wholeVertices = Set(whole.vertices.map(\.xy))
                let runs = (0..<kept.count).map { kept[path: $0].map(\.xy.v) }
                let pieces = Segments(runs, cell: cell)
                var outside: [String] = [], endsOff: [String] = [], misclipped: [String] = []
                var inSamples = 0, outSamples = 0
                for run in runs {
                    for (a, b) in zip(run, run.dropFirst()) {
                        for k in 0...16 {
                            let q = a + (b - a) * (Double(k) / 16)
                            if !l.region.contains(P2(q)), !(outline.distance(q, within: cell) <= tolerance) {
                                outside.append("\(q)")
                            }
                        }
                    }
                    for e in [run[0], run[run.count - 1]] where !wholeVertices.contains(P2(e)) {
                        if !(outline.distance(e, within: cell) <= tolerance) { endsOff.append("\(e)") }
                    }
                }
                for c in 0..<whole.count {
                    let tile = copies[c / perCopy]
                    guard let back = tile.inverse else { continue }
                    let path = whole[path: c].map(\.xy.v)
                    for (a, b) in zip(path, path.dropFirst()) {
                        let ya = back(P3(a.x, a.y, 0)).y, yb = back(P3(b.x, b.y, 0)).y
                        let inBand = { (y: Double) in !banded || (y > band.lo && y < band.hi) }
                        // Not where a vertex lies on the band's edge, which
                        // carried back and forth is a rounding either side.
                        let onEdge = { (y: Double) in banded && min(abs(y - band.lo), abs(y - band.hi)) <= tolerance }
                        guard inBand(ya) == inBand(yb), !onEdge(ya), !onEdge(yb) else { continue }
                        for k in 0..<16 {
                            let q = a + (b - a) * ((Double(k) + 0.5) / 16)
                            guard outline.distance(q, within: cell).isInfinite || outline.distance(q, within: cell) > 1e-6 * size,
                                  !tiled || cover(q) == 1 else { continue }
                            let expected = l.region.contains(P2(q)) && inBand(ya)
                            if expected { inSamples += 1 } else { outSamples += 1 }
                            let onPiece = pieces.distance(q, within: cell) <= tolerance
                            if onPiece != expected { misclipped.append("\(q) on tile \(c / perCopy) \(onPiece ? "kept" : "cut away")") }
                        }
                    }
                }
                Check.expect(inSamples > 200 && outSamples > 200 && outside.isEmpty && endsOff.isEmpty && misclipped.isEmpty,
                             "\(l.name), \(tiled ? "tiled" : "untiled"), kept to \(what): each copy cut to the footprint where it lands, exactly",
                             "\(outside.count) points kept outside, \(endsOff.count) of \(2 * runs.count) ends off the perimeter, "
                                + "\(misclipped.count) of \(inSamples) in and \(outSamples) out misclipped: "
                                + (outside.prefix(1) + endsOff.prefix(1) + misclipped.prefix(2)).joined(separator: "; "))
            }
          }
        }
    }

    Check.suite("faces: the graph's trim loops are its truncation edge, closed, and a contour on it is cut where it crosses them") {
        var landscapes = try faceLandscapes().filter(\.hasGraph)
        // A tower cut at 2 left of Re z = 1/2 and at 4 beyond, on f: the top
        // of its riser runs past where the graph meets it, flat at 4 between
        // |Im z| = 1/4, along the higher plateau.
        let tower = try NativeLandscape.build(
            LandscapeRequest(expression: "1/(z - 0.5)", domain: Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1)),
                             resolution: 32, truncation: .realBands([RealBand(below: 0.5, height: 2)], beyond: 4)),
            refine: true)
        landscapes.append(Landscape("a tower banded at 2 | 4, on f", surface: tower.surface, refine: tower.refine))
        // The same over Im [-0.3, 1]: its riser's top runs from the domain's
        // edge, where the band at 2 has no truncation edge to end.
        let cutTower = try NativeLandscape.build(
            LandscapeRequest(expression: "1/(z - 0.5)", domain: Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -0.3, hi: 1)),
                             resolution: 32, truncation: .realBands([RealBand(below: 0.5, height: 2)], beyond: 4)),
            refine: true)
        landscapes.append(Landscape("a tower banded at 2 | 4 over Im [-0.3, 1], on f", surface: cutTower.surface,
                                    refine: cutTower.refine))
        // 1/sin(πz) with a boundary through each pole from 1 to 6, banded at
        // 2 and 4 by turns: six risers, and at each boundary a pole's
        // plateau ending on it from either side.
        let poles = try NativeLandscape.build(
            LandscapeRequest(expression: "1/sin(pi*z)", domain: Domain(real: Interval(lo: -0.5, hi: 6.5), imag: Interval(lo: -1, hi: 1)),
                             resolution: 224,
                             truncation: .realBands((1...6).map { RealBand(below: Double($0), height: $0 % 2 == 1 ? 2 : 4) }, beyond: 2)),
            refine: true)
        landscapes.append(Landscape("1/sin(πz) banded at 2 | 4 by turns through its poles, on f", surface: poles.surface,
                                    refine: poles.refine))
        // A pole's plateau across a band boundary into a band too high for a
        // plateau of its own: its truncation edge ends on the boundary from
        // the one side, and its riser's top runs between the two ends.
        let across = try NativeLandscape.build(
            LandscapeRequest(expression: "1/(z - 0.4)", domain: Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1)),
                             resolution: 32, truncation: .realBands([RealBand(below: 0.5, height: 2)], beyond: 20)),
            refine: true)
        landscapes.append(Landscape("1/(z - 0.4) banded at 2 | 20, its plateau across the boundary, on f", surface: across.surface,
                                    refine: across.refine))
        // A pole's plateau that turns back a column short of a band boundary,
        // on a coarse grid and on a fine one.
        for (expression, resolution) in [("1/(z + 0.03)", 32), ("1/(z + 0.01)", 200)] {
            let pole = try NativeLandscape.build(
                LandscapeRequest(expression: expression, domain: Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1)),
                                 resolution: resolution, truncation: .realBands([RealBand(below: 0.5, height: 2)], beyond: 4)),
                refine: true)
            landscapes.append(Landscape("\(expression) at res \(resolution) banded at 2 | 4, a column short of the boundary, on f",
                                        surface: pole.surface, refine: pole.refine))
        }
        // The reviewer's tower in one tile: its band at 1, between Re z = 1.45
        // and the far edge at 1.5, is narrower than a column.
        let reviewersTower = try NativeLandscape.build(
            LandscapeRequest(expression: "1/(z - 1.4)", domain: Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1)),
                             resolution: 32, truncation: .realBands([RealBand(below: 1.45, height: 0.5)], beyond: 1)),
            refine: true)
        landscapes.append(Landscape("the reviewer's tower at 0.5 | 1, on f", surface: reviewersTower.surface,
                                    refine: reviewersTower.refine))
        // A hole in the plateau within a column of a band boundary, round a
        // zero on it: seen by the grid's row through the zero, and seen by
        // none, where it is the refiner's ring.
        for (re, im) in [((-0.3, 1.5), (-0.5, 0.5)), ((0.3, 1.0), (-1.0, 0.5))] {
            let hole = try NativeLandscape.build(
                LandscapeRequest(expression: holeBesideABoundary, domain: Domain(real: Interval(lo: re.0, hi: re.1), imag: Interval(lo: im.0, hi: im.1)),
                                 resolution: 24, truncation: .realBands([RealBand(below: 0.5, height: 1.5)], beyond: 1.8)),
                refine: true)
            landscapes.append(Landscape("a hole round a zero on a band boundary at 1.5 | 1.8 over Re \(re), Im \(im), on f",
                                        surface: hole.surface, refine: hole.refine))
        }
        // A zero's ring at 1.8 reaching over the boundary from the band
        // beside, a cut cell drawing an arc of it.
        let ring = try NativeLandscape.build(
            LandscapeRequest(expression: ringBesideABoundary, domain: Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1)),
                             resolution: 16, truncation: .realBands([RealBand(below: 0.5, height: 1.5)], beyond: 1.8)),
            refine: true)
        landscapes.append(Landscape("a zero's ring over a band boundary from the band beside, at 1.5 | 1.8, on f",
                                    surface: ring.surface, refine: ring.refine))
        // Uniform on f: 1/(z³ - 1), its three towers cut at 5.
        let cubic = try NativeLandscape.build(
            LandscapeRequest(expression: "1/(z^3 - 1)", domain: Domain(real: Interval(lo: -1.3, hi: 1.6), imag: Interval(lo: -1.2, hi: 1.5)),
                             resolution: 120, truncation: .uniform(5)),
            refine: true)
        landscapes.append(Landscape("1/(z³ - 1) truncated at 5, on f", surface: cubic.surface, refine: cubic.refine))
        // Two tiles side by side, the second a translation, which is not its
        // own inverse as a reflection is, on a footprint across both.
        if let hatch = landscapes.first(where: { $0.name == "the hatch fixture, low bands, in four tiles, on a footprint across them" }) {
            let besides = [Affine2.identity, Affine2(a: 1, b: 0, tx: 4.5, c: 0, d: 1, ty: 0)]
            let across = BoundaryPerimeter(edges: [(-1.5, -0.6), (6.4, -0.9), (6.1, 1.3), (-1.7, 1.4)].indices.map { k in
                let c = [(-1.5, -0.6), (6.4, -0.9), (6.1, 1.3), (-1.7, 1.4)]
                let a = c[k], b = c[(k + 1) % c.count]
                return PerimeterEdge(start: P2(a.0, a.1), end: P2(b.0, b.1), density: 9)
            })
            landscapes.append(Landscape("the hatch fixture, low bands, in two tiles side by side, on a footprint across them",
                                        surface: hatch.surface, tiles: besides, region: .inside(across)))
        }
        for l in landscapes {
            // How far the loops and `contains` may part: measured, not
            // bounded. The refiner places each chord of the truncation edge
            // with its midpoint's distance from |f| = h(x), estimated to first
            // order, within 0.02 cells (`ContourRefine.tolerance`), which
            // bounds neither the estimate's error nor the chord away from its
            // midpoint: 0.015 cells on Γ, 0.043 on the tower. Without a
            // refiner the loops are marching squares' chords under a bilinear
            // surface that curves within a cell: 0.16 on the hatch fixture.
            // The clip is held to the loops, which decide it, with no margin.
            let margin = l.refine == nil ? 0.25 : 0.05
            guard let t = trimLoops(l, margin: margin) else {
                Check.expect(l.faces.isEmpty, "\(l.name): the graph has trim loops"); continue
            }
            if t.riserAsked > 0 {
                Check.expect(t.riserMissing == 0,
                             "\(l.name): each riser's top is drawn where the lower band's plateau holds its boundary, and only there",
                             "\(t.riserMissing) of \(t.riserAsked) wrong: \(t.examples("riser"))")
            }
            Check.expect(t.open == 0 && t.drawnOff == 0,
                         "\(l.name): the graph's trim loops are closed, and the truncation edge drawn lies on them",
                         "\(t.open) open, \(t.drawnOff) of \(t.drawn) drawn vertices off, worst \(t.worstDrawn): \(t.examples("drawn"))")
            Check.expect(t.asked > 1000 && t.misplaced == 0,
                         "\(l.name): a point the loops and the footprint hold is in the graph, and one they do not is out",
                         "\(t.misplaced) of \(t.asked) disagree, worst \(t.worstMisplaced) cells from the loops: \(t.examples("placed"))")
            // A footprint that holds none of the graph keeps no run of it.
            Check.expect((t.runs > 0 || t.held == 0) && t.endsOff == 0 && t.samples > 200 && t.misclipped == 0,
                         "\(l.name): lines of x and y and level sets of |f| on each copy are cut where they cross its loops, "
                            + "and kept where the loops and the footprint hold them",
                         "\(t.endsOff) of \(t.runs) runs end off the loops (worst \(t.worstEnd) cells), \(t.misclipped) of \(t.samples) "
                            + "samples misclipped: \(t.examples("clipped"))")
        }
    }

    Check.suite("faces: a polyhedron's faces are its loops, and membership agrees with the convex pieces") {
        let solids: [(String, Solid)] = Solid.Platonic.allCases.map { ($0.rawValue, Solid.platonic($0)) }
            + [("the picture frame", Solid.frame(outer: 2, inner: 1, depth: 0.8)), ("Császár's torus", .csaszar)]
        for (name, s) in solids {
            let faces = SceneGeometry.solid(s).faces
            var mismatches = 0, untiled = 0, inward = 0, inside = 0
            for (i, face) in faces.enumerated() {
                let frame = s.frames[i]
                let copy = face.copies[0]
                func plane(_ k: Int) -> SIMD2<Double> { frame.coordinates(s.vertices[k].v) }
                let pieces = s.faces[i].map { (piece: $0, corners: s.pieces[$0].map(plane)) }
                func crossings(_ p: SIMD2<Double>, _ polygon: [SIMD2<Double>]) -> Int {
                    polygon.indices.filter { k in
                        let a = polygon[k], b = polygon[(k + 1) % polygon.count]
                        guard (a.y > p.y) != (b.y > p.y) else { return false }
                        return p.x < a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y)
                    }.count
                }
                func near(_ p: SIMD2<Double>, _ polygon: [SIMD2<Double>]) -> Bool {
                    polygon.indices.contains { k in
                        let a = polygon[k], b = polygon[(k + 1) % polygon.count]
                        let t = max(0, min(1, simd_dot(p - a, b - a) / simd_length_squared(b - a)))
                        return simd_length(p - (a + t * (b - a))) < 1e-6
                    }
                }
                // The partition suite's grid (`partitionReport`): each point in
                // the face or not, and in as many pieces; points near a side
                // are left out.
                let all = s.loops[i].joined().map(plane)
                let lo = all.reduce(all[0], simd_min), hi = all.reduce(all[0], simd_max)
                for a in 0..<61 { for b in 0..<61 {
                    let p = lo + (hi - lo) * SIMD2(Double(a) + 0.37, Double(b) + 0.61) / 61
                    if pieces.contains(where: { near(p, $0.corners) }) { continue }
                    let owners = pieces.filter { crossings(p, $0.corners) % 2 == 1 }
                    let inFace = copy.contains(P2(p))
                    if owners.count != (inFace ? 1 : 0) { mismatches += 1 }
                    guard let owner = owners.first, owners.count == 1 else { continue }
                    // A point inside the face is in it and in no other face,
                    // and the face's map turns out of the solid there.
                    inside += 1
                    let ids = holders(of: P3<WorldSpace>(frame.point(p)), in: faces).map(\.face.id)
                    if ids != [.solid(face: i)] { untiled += 1 }
                    if simd_dot(outwardNormal(face, copy, P2(p)), s.normals[owner.piece]) <= 0 { inward += 1 }
                } }
            }
            Check.expect(faces.count == s.loops.count && faces.map(\.id) == s.loops.indices.map { .solid(face: $0) }
                            && faces.allSatisfy { $0.copies.count == 1 && $0.copies[0].tile == .identity },
                         "\(name): a face per loop list, in order, each in one place", "\(faces.count) faces")
            Check.expect(mismatches == 0 && untiled == 0 && inward == 0 && inside > 1000,
                         "\(name): membership by the loops agrees with the pieces, and each point is in its face alone",
                         "\(mismatches) disagree, \(untiled) in another face too, \(inward) facing in, of \(inside) inside")
        }
    }

    Check.suite("faces: the graph's trim loops close a plateau that turns back a column short of a band boundary") {
        // A pole's plateau that comes within a column of a band boundary
        // without crossing it, |f| under the truncation all along the
        // boundary: in the cell the boundary cuts, contoured over the
        // plateau's own band, its line turns back short of the boundary, so
        // the plateau's loop closes, in the drawing as in the loops. On a
        // coarse grid, and on a fine one with the plateau nearer the
        // boundary.
        for (expression, resolution) in [("1/(z + 0.03)", 32), ("1/(z + 0.01)", 200)] {
            let pole = try NativeLandscape.build(
                LandscapeRequest(expression: expression, domain: Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1)),
                                 resolution: resolution, truncation: .realBands([RealBand(below: 0.5, height: 2)], beyond: 4)),
                refine: true)
            let trim = pole.surface.graphTrimLoops(refine: pole.refine)
            let open = trim.loops.filter { $0.first != $0.last }
            Check.expect(trim.open == 0 && open.isEmpty,
                         "\(expression) at res \(resolution): its plateau's loop, a column short of the boundary at Re z = 1/2, is closed",
                         "\(trim.open) counted, \(open.count) open: \(open.map { "\($0[0].v) to \($0[$0.count - 1].v)" })")
        }
    }

    Check.suite("faces: a hole within a column of a band boundary is drawn in each band, at its own height, once") {
        // Round the zero at 1/2 on the boundary Re z = 1/2, at 1.5 | 1.8, the
        // plateau has a hole a sixteenth of a column across, |f| ≈ 590 |z - 1/2|:
        // a half disc of radius 0.0026 at 1.5 left of the boundary and one of
        // 0.0031 at 1.8 right of it. On a window whose row through the zero
        // sees it on the boundary, and on two whose grid sees nothing there,
        // where it is the refiner's ring round the zero, seeded once a band
        // and cut on the boundary, not once a run of the truncation edge, at
        // that run's level, across it. On [-0.5, 1.5] x [-1, 1], where the
        // zero is at the centre of a cell with the poles, and no node is
        // the least of |f| beside it. And with the zero a thousandth off the
        // boundary, the poles moved with it, where the ring's chords cross
        // the boundary between their vertices, and each cut is solved on it.
        let offTheBoundary = "1/(z - (0.5206 + 0.0650i)) + 1/(z - (0.4814 - 0.0450i))"
        for (expression, re, im) in [(holeBesideABoundary, (-0.3, 1.5), (-0.5, 0.5)), (holeBesideABoundary, (0.3, 1.0), (-1.0, 0.5)),
                                     (holeBesideABoundary, (0.0, 1.0), (-0.5, 0.5)), (holeBesideABoundary, (-0.5, 1.5), (-1.0, 1.0)),
                                     (offTheBoundary, (0.0, 1.0), (-0.5, 0.5))] {
            let built = try NativeLandscape.build(
                LandscapeRequest(expression: expression, domain: Domain(real: Interval(lo: re.0, hi: re.1), imag: Interval(lo: im.0, hi: im.1)),
                                 resolution: 24, truncation: .realBands([RealBand(below: 0.5, height: 1.5)], beyond: 1.8)),
                refine: true)
            guard let magnitude = built.refine?.magnitude else { Check.expect(false, "Re \(re), Im \(im): a refiner"); continue }
            let trim = built.surface.graphTrimLoops(refine: built.refine)
            let hole = trim.truncationEdge.dropLast(trim.riserTops.count)
                .filter { p in p.allSatisfy { abs($0.x - 0.5) < 0.05 && abs($0.y) < 0.05 } }
            let left = hole.filter { p in p.allSatisfy { $0.x <= 0.5 && $0.z == 1.5 } }
            let right = hole.filter { p in p.allSatisfy { $0.x >= 0.5 && $0.z == 1.8 } }
            let onBoundary = hole.allSatisfy { p in p.first != p.last && p[0].x == 0.5 && p[p.count - 1].x == 0.5 }
            let onLevel = hole.joined().allSatisfy { abs(magnitude(P2($0.x, $0.y)) - $0.z) <= 1e-6 * $0.z }
            Check.expect(hole.count == 2 && left.count == 1 && right.count == 1 && onBoundary && onLevel && trim.open == 0,
                         "\(expression == offTheBoundary ? "the zero off the boundary, " : "")Re \(re), Im \(im): the hole's edge is one half at 1.5 left of the boundary and one at 1.8 right of it, "
                            + "each ending on the boundary, on its level",
                         "\(hole.count) pieces: " + hole.map { p in
                            String(format: "x %.5f to %.5f at %@%@", p.map(\.x).min()!, p.map(\.x).max()!,
                                   Set(p.map { "\($0.z)" }).sorted().joined(separator: ","), p.first == p.last ? ", closed" : "")
                         }.joined(separator: "; ") + ", \(trim.open) open")
        }
    }

    Check.suite("faces: a zero's ring is drawn once where a cell a band boundary cuts drew an arc of it") {
        // A zero at Re z = 0.482, a column's sixth left of the boundary
        // Re z = 1/2 at res 16, between poles 0.15 above and below it, at
        // 1.5 | 1.8: its hole at 1.8, of radius 0.021, reaches over the
        // boundary, and the cut cell's part right of it draws that arc, which
        // turns too little round the zero for the refiner to leave it its
        // ring. The ring cut to the band drew it again, and the arc and the
        // ring between nearly the same ends left a loop open.
        let built = try NativeLandscape.build(
            LandscapeRequest(expression: ringBesideABoundary, domain: Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1)),
                             resolution: 16, truncation: .realBands([RealBand(below: 0.5, height: 1.5)], beyond: 1.8)),
            refine: true)
        let trim = built.surface.graphTrimLoops(refine: built.refine)
        let arcs = trim.truncationEdge.dropLast(trim.riserTops.count)
            .filter { p in p.allSatisfy { $0.x >= 0.5 && $0.z == 1.8 && abs($0.x - 0.482) < 0.05 && abs($0.y + 0.0647) < 0.05 } }
        Check.expect(trim.open == 0 && arcs.count == 1 && arcs.allSatisfy { $0[0].x == 0.5 && $0[$0.count - 1].x == 0.5 },
                     "the arc right of the boundary is drawn once, from the boundary to the boundary, and the loops close",
                     "\(arcs.count) arcs: \(arcs.map { "\($0.count) vertices" }), \(trim.open) open")
    }

    Check.suite("faces: a zero's ring beside a saddle is drawn, though the edge round another zero passes near it") {
        // A simple zero at (0.650, 0.014) and a double zero at (0.769, -0.212),
        // the saddle between them at (0.698, -0.064) where |f| is 1.503, at
        // 1.5 | 4 across Re z = 0.7214, res 24: the grid sees the double
        // zero's hole and not the simple zero's. The double zero's edge
        // passes within a vertex of the simple zero's ring, a different
        // branch of the level set, and the ring is drawn all the same.
        let built = try NativeLandscape.build(
            LandscapeRequest(expression: "353.319490401*(z - (0.650093325017 + 0.014383779377i))*(z - (0.769125500952 - 0.211731000630i))^2"
                                + "/(z - (0.536334390878 - 0.621521184752i))",
                             domain: Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1)), resolution: 24,
                             truncation: .realBands([RealBand(below: 0.721440736242, height: 1.5)], beyond: 4)),
            refine: true)
        guard let magnitude = built.refine?.magnitude else { Check.expect(false, "a refiner"); return }
        let trim = built.surface.graphTrimLoops(refine: built.refine)
        let cell = 2.0 / 23
        let loops = Segments(trim.loops.map { $0.map(\.v) }, cell: cell)
        // Round the simple zero, clear of the saddle: the loops hold the
        // hole, |f| < 1.5, and only it, on a lattice a seventh of a cell
        // across, away from them.
        var asked = 0, wrong: [String] = []
        for i in 0..<10 {
            for j in 0..<9 {
                let p = SIMD2(0.6 + (Double(i) + 0.37) * cell / 7, -0.045 + (Double(j) + 0.61) * cell / 7)
                guard loops.distance(p, within: 0.06 * cell).isInfinite else { continue }
                asked += 1
                if loops.inside(p) != (magnitude(P2(p)) < 1.5) { wrong.append("\(p)") }
            }
        }
        Check.expect(asked > 50 && wrong.isEmpty && trim.open == 0, "round the simple zero the loops hold its hole, and only it",
                     "\(wrong.count) of \(asked) wrong: \(wrong.prefix(3).joined(separator: "; ")), \(trim.open) open")
    }

    Check.suite("faces: one ring is drawn round two zeros in one hole") {
        // Zeros at (0.307, -0.144) and (0.348, -0.146), half a cell apart at
        // res 24, under 4 right of Re z = 0.1141: the hole |f| < 4 holds
        // both, and the grid sees neither. Each zero's ring is the same
        // curve; drawn from both, the loop was there twice and held nothing.
        let built = try NativeLandscape.build(
            LandscapeRequest(expression: "4119.1074405*(z - (0.307498198308-0.144345961027i))*(z - (0.348393843985-0.145720969539i))"
                                + "/(z - (0.764249524783+0.504053378065i))",
                             domain: Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1)), resolution: 24,
                             truncation: .realBands([RealBand(below: 0.114094111388, height: 2)], beyond: 4)),
            refine: true)
        guard let magnitude = built.refine?.magnitude else { Check.expect(false, "a refiner"); return }
        let trim = built.surface.graphTrimLoops(refine: built.refine)
        let cell = 2.0 / 23
        let loops = Segments(trim.loops.map { $0.map(\.v) }, cell: cell)
        var asked = 0, wrong: [String] = []
        for i in 0..<9 {
            for j in 0..<7 {
                let p = SIMD2(0.28 + (Double(i) + 0.37) * cell / 7, -0.18 + (Double(j) + 0.61) * cell / 7)
                guard loops.distance(p, within: 0.06 * cell).isInfinite else { continue }
                asked += 1
                if loops.inside(p) != (magnitude(P2(p)) < 4) { wrong.append("\(p)") }
            }
        }
        let rings = trim.truncationEdge.filter { $0.allSatisfy { abs($0.x - 0.33) < 0.05 && abs($0.y + 0.145) < 0.05 } }
        Check.expect(asked > 40 && wrong.isEmpty && rings.count == 1 && trim.open == 0, "the loops hold the hole round both zeros, drawn once",
                     "\(rings.count) rings, \(wrong.count) of \(asked) wrong: \(wrong.prefix(3).joined(separator: "; ")), \(trim.open) open")
    }

    Check.suite("faces: a ring's part in a band is left out where the band's edge already joins its ends") {
        // Zeros at (0.725, 0.146) and (0.826, 0.076) beside the saddle between
        // them, under 3 | 4 across Re z = 0.7606, res 24. Right of the
        // boundary the cut cell's line joins the two points where the hole
        // round the second zero meets it, by a chord across the hole; the
        // grid sees nothing beyond. That zero's ring, cut to the band, joins
        // the same two points, and the band's loop, met there by both, was
        // left open.
        let built = try NativeLandscape.build(
            LandscapeRequest(expression: "595.849573804*(z - (0.724529786873+0.145657568132i))*(z - (0.825622995430+0.075733727046i))"
                                + "/(z - (0.294022667147+0.587508313898i))",
                             domain: Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1)), resolution: 24,
                             truncation: .realBands([RealBand(below: 0.760571784330, height: 3)], beyond: 4)),
            refine: true)
        let trim = built.surface.graphTrimLoops(refine: built.refine)
        let open = trim.loops.filter { $0.first != $0.last }
        Check.expect(trim.open == 0 && open.isEmpty, "the band's loops close",
                     "\(trim.open) counted, \(open.count) open: \(open.map { "\($0[0].v) to \($0[$0.count - 1].v)" })")
    }

    Check.suite("faces: a chord is taken past a zero only off its own run, and within its band") {
        // Beside a saddle the grid's line round two zeros cuts across between
        // them, and past the lower zero lies an arc the line already runs
        // along: at 2 | 4 across Re z = 0.4114, zeros at (0.564, 0.278) and
        // (0.550, 0.205). Round one zero, at 2 | 2.5 across Re z = 0.5330,
        // the grid's loop is four vertices, and past the zero lies the arc
        // its far chord stands for: zero at (0.840, -0.068). A chord placed
        // there doubled the line back over itself, and the loops dropped a
        // stretch of the hole: the lattice within it, a fourteenth of a cell
        // across, away from the loops. And at 2 | 2.5 across Re z = 0.7120,
        // past a zero by the boundary the band at 2.5 found its level left
        // of it, in the band at 2: every vertex of each band's truncation
        // edge lies in its band.
        let domain = Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1))
        let cell = 2.0 / 23
        for (name, expression, bands, beyond, x, y) in [
            ("the lower zero's hole", "948.115004635*(z - (0.563659351434+0.277678080119i))*(z - (0.550062724850+0.205189440392i))"
                + "/(z - (0.856142777298+0.772831571274i))", [RealBand(below: 0.411407161624, height: 2)], 4.0,
             (0.52, 0.585), (0.183, 0.212)),
            ("the hole of a four-vertex loop", "416.937178124*(z - (0.840040450901-0.068301508238i))*(z - (0.716014763340-0.121761607551i))"
                + "/(z - (0.113910517434+0.575219223905i))", [RealBand(below: 0.532950907488, height: 2)], 2.5,
             (0.81, 0.87), (-0.11, -0.035)),
        ] {
            let built = try NativeLandscape.build(
                LandscapeRequest(expression: expression, domain: domain, resolution: 24, truncation: .realBands(bands, beyond: beyond)),
                refine: true)
            guard let magnitude = built.refine?.magnitude else { Check.expect(false, "\(name): a refiner"); continue }
            let trim = built.surface.graphTrimLoops(refine: built.refine)
            let loops = Segments(trim.loops.map { $0.map(\.v) }, cell: cell)
            let h = built.surface.truncation.height(atX: x.0)
            var asked = 0, wrong: [String] = []
            for i in 0...Int((x.1 - x.0) / cell * 14) {
                for j in 0...Int((y.1 - y.0) / cell * 14) {
                    let p = SIMD2(x.0 + Double(i) * cell / 14, y.0 + Double(j) * cell / 14)
                    guard loops.distance(p, within: 0.06 * cell).isInfinite else { continue }
                    asked += 1
                    if loops.inside(p) != (magnitude(P2(p)) < h) { wrong.append("\(p)") }
                }
            }
            Check.expect(asked > 20 && wrong.isEmpty, "\(name): the loops hold it",
                         "\(wrong.count) of \(asked) wrong: \(wrong.prefix(3).joined(separator: "; "))")
        }
        let b = 0.711993911036
        let built = try NativeLandscape.build(
            LandscapeRequest(expression: "1703.90688736*(z - (0.732858379215+0.011300545905i))*(z - (0.811553422942-0.149386716774i))^2"
                                + "/(z - (0.183946050532+0.373413325442i))",
                             domain: domain, resolution: 24, truncation: .realBands([RealBand(below: b, height: 2)], beyond: 2.5)),
            refine: true)
        let trim = built.surface.graphTrimLoops(refine: built.refine)
        let outside = trim.truncationEdge.dropLast(trim.riserTops.count).joined()
            .filter { v in v.z == 2.5 ? v.x < b : v.x > b }
        Check.expect(trim.open == 0 && outside.isEmpty, "each band's truncation edge keeps to its band",
                     "\(outside.count) vertices outside: \(outside.prefix(3).map { "\($0.v)" }), \(trim.open) open")
    }

    Check.suite("faces: the truncation edge in a cell a band boundary cuts keeps to |f| = h, within the refiner's tolerance") {
        // Each chord with its midpoint within a column of a boundary, its
        // midpoint off the level by the residual over the gradient, in
        // cells, the refiner's own measure. Round a zero on the boundary
        // Re z = 0.5223 whose hole at 4, of radius 0.04, is wider than the
        // part right of it, so the arc leaves the cut cell where the grid
        // beside it cannot see it: three vertices, 0.45 cells off, before
        // the chords were taken onto f. Beside a pole within a chord across
        // its plateau, at 3 | 4 | 1 across Re z = 0.833 and 1.3, where |f|
        // falls away from the chord on the side the curve is not. Where a
        // pole's plateau at 3, right of Re z = -0.4238, crosses that boundary
        // between two rows and comes back, the line broken on the boundary.
        // The hole round a zero on a boundary that is a column, between two
        // poles within a cell of it, and round one on a boundary that is not,
        // a pole in its cell: the grid's nodes see neither zero. Beside a
        // zero whose hole joins the graph past the boundary, a chord left of
        // it, along whose normal |f| turns back short of the level toward
        // Newton's step and reaches it the other way, past the zero. And the
        // hole and the ring above, and the reviewer's tower.
        let mixed = "(z - (1.1934990849887146 - 0.20536860263116311i))*(z - (1.2234990849887146 - 0.2253686026311631i))"
            + "/((z - (1.2524804507563325 - 0.078069596745760084i))*(z - (1.399997924755525 + 0.40148084601005929i))^2)"
        let domain = Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1))
        for (name, expression, domain, resolution, bands, beyond) in [
            ("a zero on the boundary, its hole wider than the part", "1/(z - (0.59724288035304796 - 0.13390328401912382i)) + 1/(z - (0.44728054417818569 - 0.37915093303657482i))",
             domain, 24, [RealBand(below: 0.5222617122656168, height: 1.5)], 4.0),
            ("a pole within a chord", mixed, domain, 16, [RealBand(below: 0.8333333333333333, height: 3), RealBand(below: 1.3001429547193402, height: 4)], 1.0),
            ("a plateau leaving its band between two rows", "1/(z - (-0.095946946980314418 - 0.0047827152763062486i))",
             domain, 16, [RealBand(below: -0.42375125887220094, height: 1.2)], 3.0),
            ("a zero on a column between two poles", "1/(z - (0.59076180115334842-0.093833748934804229i)) + 1/(z - (0.5145013567413883-0.13573974784644616i))",
             domain, 20, [RealBand(below: 0.5526315789473684, height: 3)], 1.8),
            ("a zero in a pole's cell", "1/(z - (0.42240022182367271-0.40227584782117776i)) + 1/(z - (0.37140972304490555-0.48709752678493168i))",
             domain, 20, [RealBand(below: 0.3969049724342891, height: 3)], 4),
            ("a chord beside a zero", "(z - (0.40691152265074942-0.15224404800258995i))"
                + "/((z - (0.62359753483716163-0.13770458918473533i))*(z - (0.57445433013745983-0.14044013384631204i)))",
             domain, 16, [RealBand(below: 0.45369534601865147, height: 1.5)], 4),
            ("the hole round a zero on the boundary", holeBesideABoundary, Domain(real: Interval(lo: -0.3, hi: 1.5), imag: Interval(lo: -0.5, hi: 0.5)),
             24, [RealBand(below: 0.5, height: 1.5)], 1.8),
            ("a zero's ring and a cut cell's arc of it", ringBesideABoundary, domain, 16, [RealBand(below: 0.5, height: 1.5)], 1.8),
            ("the reviewer's tower", "1/(z - 1.4)", domain, 32, [RealBand(below: 1.45, height: 0.5)], 1.0),
        ] {
            let built = try NativeLandscape.build(
                LandscapeRequest(expression: expression, domain: domain, resolution: resolution, truncation: .realBands(bands, beyond: beyond)),
                refine: true)
            guard let refine = built.refine, let magnitude = refine.magnitude else { Check.expect(false, "\(name): a refiner"); continue }
            let surface = built.surface, g = surface.height
            let dx = abs(g.domain.real.length) / Double(g.width - 1)
            let cell = max(dx, abs(g.domain.imag.length) / Double(g.height - 1))
            let trim = surface.graphTrimLoops(refine: refine)
            var chords = 0, worst = 0.0
            for piece in trim.truncationEdge.dropLast(trim.riserTops.count) {
                for (a, b) in zip(piece, piece.dropFirst()) {
                    let m = P2<DomainSpace>(0.5 * (a.x + b.x), 0.5 * (a.y + b.y))
                    guard bands.contains(where: { abs(m.x - $0.below) <= dx }) else { continue }
                    let e = 1e-4 * cell
                    let gx = (magnitude(P2(m.x + e, m.y)) - magnitude(P2(m.x - e, m.y))) / (2 * e)
                    let gy = (magnitude(P2(m.x, m.y + e)) - magnitude(P2(m.x, m.y - e))) / (2 * e)
                    chords += 1
                    worst = max(worst, abs(magnitude(m) - a.z) / (gx * gx + gy * gy).squareRoot() / cell)
                }
            }
            Check.expect(chords > 1 && worst <= refine.tolerance, "\(name): every chord within a column of a boundary keeps to its level",
                         String(format: "%d chords, worst %.4f cells off", chords, worst))
            // And the loops hold the graph there, |f| under the truncation,
            // read on a lattice a seventh of a cell across, away from them.
            let loops = Segments(trim.loops.map { $0.map(\.v) }, cell: cell)
            let real = surface.domain.real, imag = surface.domain.imag
            var asked = 0, wrong: [String] = []
            for band in bands {
                for i in 0..<14 {
                    for j in 0...Int(abs(imag.length) / cell * 7) {
                        let p = SIMD2(band.below - dx + (Double(i) + 0.37) * dx / 7, min(imag.lo, imag.hi) + (Double(j) + 0.61) * cell / 7)
                        guard p.x > min(real.lo, real.hi), p.x < max(real.lo, real.hi), p.y < max(imag.lo, imag.hi),
                              loops.distance(p, within: 0.06 * cell).isInfinite else { continue }
                        asked += 1
                        if loops.inside(p) != (magnitude(P2(p)) < surface.truncation.height(atX: p.x)) { wrong.append("\(p)") }
                    }
                }
            }
            Check.expect(asked > 200 && wrong.isEmpty, "\(name): within a column of a boundary the loops hold |f| < h(x), and only it",
                         "\(wrong.count) of \(asked) wrong: \(wrong.prefix(3).joined(separator: "; "))")
        }
    }

    Check.suite("faces: a band narrower than a column, between a boundary and the far edge, has its plateau's loop") {
        // The reviewer's tower, 1/(z - 1.4) at 0.5 | 1 across Re z = 1.45, the
        // far edge at 1.5 and the column before it at 1.435: the band at 1
        // has no column of its own. Its plateau, where |z - 1.4| ≤ 1, is one
        // loop, and the points between the boundary and the edge are in the
        // graph where the loops say, away from them, read on a lattice a
        // fifth of a column across.
        let tower = try NativeLandscape.build(
            LandscapeRequest(expression: "1/(z - 1.4)", domain: Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1)),
                             resolution: 32, truncation: .realBands([RealBand(below: 1.45, height: 0.5)], beyond: 1)),
            refine: true)
        let trim = tower.surface.graphTrimLoops(refine: tower.refine)
        let own = trim.loops.filter { loop in loop.first == loop.last && loop.allSatisfy { $0.x >= 1.45 } }
        Check.expect(trim.open == 0 && own.count == 1 && (own.first.map { $0.map(\.y).max()! - $0.map(\.y).min()! } ?? 0) > 1.99,
                     "the narrow band's plateau is one closed loop, from Im z = -0.9987 to 0.9987",
                     "\(own.count) loops in the band, \(trim.open) open")
        guard let graph = tower.surface.faces(tiles: [.identity], region: .full, walls: .none, refine: tower.refine).first,
              graph.id == .graph, let loops = graph.trimLoops else { Check.expect(false, "the graph has trim loops"); return }
        let cell = 2.0 / 31
        let segments = Segments(loops.map { $0.map(\.v) }, cell: cell)
        var asked = 0, wrong = 0, example = ""
        for i in 0..<5 {
            for j in 0..<160 {
                let p = SIMD2(1.45 + 0.05 * (Double(i) + 0.5) / 5, -1 + 2 * (Double(j) + 0.5) / 160)
                guard segments.distance(p, within: 0.05 * cell).isInfinite else { continue }
                asked += 1
                if segments.inside(p) != graph.copies[0].contains(P2(p)) {
                    wrong += 1
                    if example.isEmpty { example = "\(p): the loops say \(segments.inside(p))" }
                }
            }
        }
        Check.expect(asked > 600 && wrong == 0, "between the boundary and the far edge, a point is in the graph where its loops hold it",
                     "\(wrong) of \(asked) disagree \(example)")
    }

    Check.suite("faces: a pole beside a band boundary, at every resolution, has its truncation edge on its own band's level") {
        // Poles 1/(z - c) along Re c from -0.2 to 1.2, Im c wandering within
        // 0.13 of the axis, beside the boundary Re z = 1/2 under 2 | 4 and
        // 4 | 2, at six resolutions: their plateaus cross the boundary, end
        // on it, or turn back short of it within a column. Every vertex of
        // the placed truncation edge, off the boundary, is in the band at its
        // own x and on |f| = that band's height, and the graph's loops close.
        // Open chains: 237 on main 1ae3d87 over the 492, and four on f6ea42f,
        // at the coarsest grids, where the higher band's plateau turns back
        // a column short of the boundary into the lower band's line.
        let domain = Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1))
        var cases = 0, wrong = 0, off = 0, examples: [String] = [], open: [String: Int] = [:]
        for (left, right) in [(2.0, 4.0), (4.0, 2.0)] {
            for resolution in [24, 32, 48, 64, 100, 200] {
                for k in 0..<41 {
                    let c = -0.2 + 1.4 * Double(k) / 40, cy = 0.13 * sin(Double(k))
                    let pole = try NativeLandscape.build(
                        LandscapeRequest(expression: "1/(z - (\(c) + \(cy)i))", domain: domain, resolution: resolution,
                                         truncation: .realBands([RealBand(below: 0.5, height: left)], beyond: right)),
                        refine: true)
                    guard let magnitude = pole.refine?.magnitude else { continue }
                    let trim = pole.surface.graphTrimLoops(refine: pole.refine)
                    let name = String(format: "%g|%g res %d c %.2f", left, right, resolution, c)
                    cases += 1
                    if trim.open > 0 { open[name] = trim.open }
                    for v in trim.truncationEdge.dropLast(trim.riserTops.count).joined() where v.x != 0.5 {
                        let h = v.x < 0.5 ? left : right
                        if v.z != h { wrong += 1; if examples.count < 3 { examples.append("\(name): (\(v.x), \(v.y)) at \(v.z)") } }
                        if !(abs(magnitude(P2(v.x, v.y)) - v.z) <= 1e-6 * v.z) {
                            off += 1
                            if examples.count < 3 { examples.append("\(name): (\(v.x), \(v.y)) |f| \(magnitude(P2(v.x, v.y)))") }
                        }
                    }
                }
            }
        }
        Check.expect(cases == 492 && wrong == 0 && off == 0,
                     "every vertex of 492 poles' truncation edges lies in the band at its x, on |f| = that band's height",
                     "\(wrong) in the wrong band, \(off) off the level: \(examples.joined(separator: "; "))")
        Check.expect(open.isEmpty, "and none of their loops is open",
                     "open: \(open.sorted { $0.key < $1.key }.map { "\($0.key): \($0.value)" })")
    }

    Check.suite("faces: a polyhedron face's trim loops are its loops, and a contour cut at them is its hatching") {
        // Hatching clips each line to the face's convex pieces, half-plane by
        // half-plane, and joins the intervals across the diagonals; a contour
        // of the coordinate across the hatching's angle, cut where it crosses
        // the face's loops, is the same lines. At an angle and a spacing that
        // put no line on a vertex or along a side.
        let solids: [(String, Solid)] = Solid.Platonic.allCases.map { ($0.rawValue, Solid.platonic($0)) }
            + [("the picture frame", Solid.frame(outer: 2, inner: 1, depth: 0.8)), ("Császár's torus", .csaszar)]
        for (name, s) in solids {
            var lines = 0, unmatched = 0, worst = 0.0, closed = true
            for (i, face) in SceneGeometry.solid(s).faces.enumerated() {
                let frame = s.frames[i], copy = face.copies[0]
                guard let loops = copy.trimLoops else { closed = false; continue }
                closed = closed && loops.count == s.loops[i].count && loops.allSatisfy { $0.first == $0.last }
                let (u, v) = face.domain
                let extent = simd_length(SIMD2(u.range.length, v.range.length))
                let hatching = Hatching(angle: 0.37, spacing: extent / 23.7)
                let per = SIMD2(-sin(hatching.angle), cos(hatching.angle))
                // The hatch lines by level, in frame coordinates.
                var drawn: [Int: [(SIMD2<Double>, SIMD2<Double>)]] = [:]
                for line in s.hatchLines(hatching, onFace: i) {
                    let a = frame.coordinates(line.path[0].v), b = frame.coordinates(line.path[line.path.count - 1].v)
                    drawn[Int((simd_dot(a, per) / hatching.spacing).rounded()), default: []].append((a, b))
                }
                let ks = drawn.keys.sorted()
                guard let first = ks.first, let last = ks.last else { continue }
                let levels = (first...last).map { Double($0) * hatching.spacing }
                let field = ScalarField { simd_dot($0.v, per) }
                for (level, runs) in face.contours(of: field, levels, on: copy) {
                    var expected = drawn[Int((level / hatching.spacing).rounded()), default: []]
                    lines += runs.count
                    for run in runs {
                        let a = run[0].v, b = run[run.count - 1].v
                        // The same segment, either way round.
                        let match = expected.indices.min { j, k in
                            let (p, q) = expected[j], (r, t) = expected[k]
                            return min(simd_length(p - a) + simd_length(q - b), simd_length(p - b) + simd_length(q - a))
                                < min(simd_length(r - a) + simd_length(t - b), simd_length(r - b) + simd_length(t - a))
                        }
                        guard let j = match else { unmatched += 1; continue }
                        let (p, q) = expected.remove(at: j)
                        worst = max(worst, min(max(simd_length(p - a), simd_length(q - b)),
                                               max(simd_length(p - b), simd_length(q - a))) / extent)
                    }
                    unmatched += expected.count
                }
            }
            Check.expect(closed && lines > 50 && unmatched == 0 && worst < 1e-12,
                         "\(name): each face's trim loops are its loops, closed, and its lines cut at them are its hatching",
                         "\(closed ? "" : "loops not its own; ")\(unmatched) of \(lines) unmatched, ends off by \(worst) of the extent")
        }
    }

    Check.suite("faces: a parametric surface is one face, its whole parameter rectangle") {
        let torus = ParametricSurface.torus(major: 3, minor: 1, samples: (u: 40, v: 20))
        let faces = SceneGeometry.parametric(torus).faces
        var agree = faces.count == 1 && faces.first?.id == .surface && faces.first?.copies.count == 1
            && faces.first?.copies.first?.outward == torus.outward
        if let face = faces.first, let copy = face.copies.first {
            for k in 0..<50 {
                // Unwrapped coordinates too: the axes are periodic.
                let p = P2<ParamSpace>(Double(k) * 0.37 - 4, Double(k) * 0.53 - 6)
                agree = agree && copy.contains(p) && copy.place(face.map(p)) == torus.map(p)
            }
        }
        Check.expect(agree, "the torus is one face, everywhere in it, mapped by the surface's own map")
    }
}
