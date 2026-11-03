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

/// A landscape as the faces are built from it, and as the drawing reads it.
private struct Landscape {
    let name: String
    let surface: Surface
    let tiles: [Affine2]
    let region: Region
    let walls: Walls
    let refine: ContourRefine?
    let faces: [Face]
    var heightfield: Heightfield {
        Heightfield(surface: surface, occluder: .empty, tiles: tiles, region: region, step: 1,
                    refine: refine, walls: walls)
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
    var sampled = 0, wrong = 0, inward = 0, normals = 0, skewed = 0, stations = 0, offTop = 0
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
    /// on a seam or an overlap, through whichever comes first.
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
                    // The facing's differences reach a cell out along the
                    // grid's axes as the tile turns them, and where they
                    // reach ground another tile holds, or is nearest, they
                    // read that one, across a seam the copies need not meet
                    // smoothly at.
                    let reach = [(tile.a * cell.x, tile.c * cell.x), (-tile.a * cell.x, -tile.c * cell.x),
                                 (tile.b * cell.y, tile.d * cell.y), (-tile.b * cell.y, -tile.d * cell.y)]
                    if face.id == .graph, reach.allSatisfy({ r in
                        let w = P3<WorldSpace>(ground.x + r.0, ground.y + r.1, 0)
                        return cover(w) <= 1 && carriedBack(w.xy.v, l).back == tile.inverse
                    }) {
                        t.normals += 1
                        let facing = onF ? heightfield.graphNormal(at: ground.xy) : heightfield.normal(at: ground.xy)
                        let off = simd_length(n - facing) / simd_length(facing)
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
        let curtain = onF ? nil : Mesh.walls(of: dense, surface: surface, base: base, tiles: l.tiles)
        let tops: [[P3<WorldSpace>]] = perimeter.edges.indices.map { k in
            if onF {
                let context = Surface.HatchContext(perimeter: perimeter, region: .full, tiles: l.tiles)
                let drawn = surface.hatch(.wallEdges(edges: [k], step: 1, base: base), in: context, refine: l.refine)
                return drawn.map { Array($0[path: 0]) } ?? []
            }
            return curtain.map { Array($0.vertices[(82 * k)..<(82 * k + 41)]) } ?? []
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
                for j in 0..<7 {
                    let z = base + (top - base) * (Double(j) + 0.5) / 7
                    expect(P3(v.x, v.y, z), 1, slack: atStart || atEnd || z > lower ? 1 : 0) { face, n in
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
                let over = max(atFoot.max() ?? top, cornerTop[v.xy] ?? top)
                expect(P3(v.x, v.y, over + 1e-3 * max(1, abs(over)) + 0.1 * (over - base)), 0)
                expect(P3(v.x, v.y, base - 0.1 * (top - base)), 0)
                t.stations += 1
                // Where it steps, the solid's top either side along the wall,
                // on its line: the boundary crosses it at a slant, and a step
                // in as well can land both reads on one side.
                let towards = atFoot.count == 1 ? [inward + (atStart ? d : atEnd ? -d : .zero)] : [d, -d]
                let reads = towards.map { toward -> (solid: Double, slide: Double) in
                    let solid = heightfield.surfaceHeight(at: P2(v.xy.v + side * toward))
                    return (solid, abs(heightfield.surfaceHeight(at: P2(v.xy.v + 2 * side * toward)) - solid))
                }
                if reads.allSatisfy(\.solid.isFinite),
                   !reads.contains(where: { abs(top - $0.solid) <= 4 * $0.slide + 1e-9 * max(1, abs($0.solid)) }) {
                    t.offTop += 1
                    t.fail("offTop", "wall \(k)'s top at \(v.xy.v) is \(top), the solid's beside it \(reads.map(\.solid))")
                }
            }
        }
    }
    /// Whether a wall stands on a point of the ground.
    func wallStands(_ w: P2<WorldSpace>) -> Bool {
        wallFeet.contains { a, b in
            let d = b.v - a.v
            let s = max(0, min(1, simd_dot(w.v - a.v, d) / simd_length_squared(d)))
            return simd_length(w.v - (a.v + s * d)) <= 1e-12 * max(1, size)
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
    func footprint(_ corners: [(Double, Double)]) -> BoundaryPerimeter {
        BoundaryPerimeter(edges: corners.indices.map { k in
            let a = corners[k], b = corners[(k + 1) % corners.count]
            return PerimeterEdge(start: P2(a.0, a.1), end: P2(b.0, b.1), density: 9)
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
    // A wall on a seam between two tiles, the second beside the first,
    // (x, y) ↦ (x + 1, y), or turned a quarter, (x, y) ↦ (1 + y, 1 - x):
    // |f| = 3(2 Re z - 1)² + 1/5 on the unit square, at 1 | 2 across Re z =
    // 1/2, so where the seam is the first tile's right edge the band there
    // is 2 and the second's 1 or its own. On a footprint either side of the
    // seam, the wall on it reads the tile the footprint is in.
    let seamGrid = Grid2D(width: 21, height: 21, domain: Domain(real: Interval(lo: 0, hi: 1), imag: Interval(lo: 0, hi: 1)),
                          values: (0..<441).map { k in Float(3 * pow(2 * Double(k % 21) / 20 - 1, 2) + 0.2) })
    let seamed = Surface(height: seamGrid, phase: nil, truncation: .realBands([RealBand(below: 0.5, height: 1)], beyond: 2))
    for (tileName, tile) in [("beside it", Affine2(a: 1, b: 0, tx: 1, c: 0, d: 1, ty: 0)),
                             ("turned", Affine2(a: 0, b: 1, tx: 1, c: -1, d: 0, ty: 1))] {
        for (side, corners) in [("second", [(1.0, 0.2), (1.8, 0.2), (1.8, 0.8), (1.0, 0.8)]),
                                ("first", [(0.2, 0.2), (1.0, 0.2), (1.0, 0.8), (0.2, 0.8)])] {
            let walled = footprint(corners)
            landscapes.append(Landscape("a wall on the seam with a tile \(tileName), the footprint in the \(side)",
                                        surface: seamed, tiles: [.identity, tile], region: .inside(walled),
                                        walls: .perimeter(walled, base: 0)))
        }
    }
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
    let o = gamma.manifest.occluder
    landscapes.append(Landscape("Γ, banded, on f", surface: gamma.surface, tiles: o.tiles, region: o.region,
                                walls: o.walls, refine: gamma.refine, faces: scene.geometry.faces))
    return landscapes
}


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
    var runs = 0, endsOff = 0, worstEnd = 0.0, samples = 0, misclipped = 0
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
    let reported = surface.graphTrimLoops(tiles: l.tiles, refine: l.refine)
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
                case .riser: "risers"
                case .wall: "walls"
                case .surface, .solid: "other"
                }
            }.mapValues(\.count).sorted { $0.key < $1.key }.map { "\($0.value) \($0.key)" }.joined(separator: ", ")
            Check.expect(t.wrong == 0 && t.inward == 0 && t.skewed == 0 && t.sampled > 10_000 && t.normals > 300,
                         "\(l.name): \(kinds), \(l.tiles.count) tile(s); every point of the boundary in one face, "
                            + "none off it, each face turned out, the graph's normal the facing's",
                         "\(t.wrong) in the wrong number, \(t.inward) facing in, \(t.skewed) of \(t.normals) normals off, "
                            + "of \(t.sampled): \(t.examples.filter { $0.key != "offTop" }.sorted { $0.key < $1.key }.map(\.value).joined(separator: "; "))")
            if case .perimeter = l.walls {
                Check.expect(t.offTop == 0 && t.stations > 50,
                             "\(l.name): every wall rises to the solid's top beside it in the footprint, no higher and no lower",
                             "\(t.offTop) of \(t.stations) stations off: \(t.examples["offTop"] ?? "")")
            }
        }
        // `Surface.faces` lists them in `FaceID`'s order.
        Check.expect(landscapes.allSatisfy { l in zip(l.faces, l.faces.dropFirst()).allSatisfy { $0.id <= $1.id } },
                     "the faces come in FaceID's order: graph, plateaus, risers, walls")
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
                let curtain = Mesh.walls(of: dense, surface: l.surface, base: base, tiles: l.tiles)
                var feet = 0, off: [String] = []
                for k in vertical {
                    let x = perimeter.edges[k].start.x
                    var vertices = Array(curtain.vertices[(82 * k)..<(82 * (k + 1))])
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
                let curtain = Mesh.walls(of: dense, surface: l.surface, base: base, tiles: l.tiles)
                var stations = 0, wrong: [String] = []
                for k in perimeter.edges.indices {
                    guard let drawn = l.surface.hatch(.wallEdges(edges: [k], step: 1, base: base), in: context) else { continue }
                    let top = Array(drawn[path: 0])
                    for c in curtain.vertices[(82 * k)..<(82 * k + 41)] {
                        stations += 1
                        guard let v = top.first(where: { $0.xy == c.xy }) else { wrong.append("no vertex at \(c.xy.v)"); continue }
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
                var steps = [Int](repeating: 0, count: l.tiles.count), held = steps
                for (k, edge) in perimeter.edges.enumerated() {
                    guard let drawn = l.surface.hatch(.wallEdges(edges: [k], step: 1, base: base), in: context, refine: l.refine),
                          let face = walls.first(where: { $0.id == .wall(edge: k) }) else { continue }
                    let top = Array(drawn[path: 0])
                    func apart(_ a: P3<WorldSpace>, _ b: P3<WorldSpace>) -> Bool { simd_length(a.xy.v - b.xy.v) > 1e-9 * cell }
                    for (i, v) in top.enumerated() {
                        guard (i == 0 || apart(top[i - 1], v)) && (i == top.count - 1 || apart(v, top[i + 1])) else { continue }
                        vertices += 1
                        let s = simd_length(v.xy.v - edge.start.v), z = v.z, dz = 1e-9 * max(1, abs(z))
                        let copy = face.copies[0]
                        if !(copy.contains(P2(s, z - dz)) && !copy.contains(P2(s, z + dz))) {
                            wrong.append("\(v.xy.v) at \(z), not the wall's top")
                        }
                    }
                    // Under one tile: across a seam between two, the x
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
                        held[tile] += 1
                        if !apart(a, c), boundaries.contains(where: { abs(xa - $0) <= 1e-9 * cell }) { steps[tile] += 1 }
                    }
                }
                // On the tiled towers the walls meet a riser on every tile
                // they stand on.
                Check.expect(vertices > 100 && wrong.isEmpty && (l.tiles.count == 1 || zip(steps, held).allSatisfy { $1 == 0 || $0 > 0 }),
                             "\(l.name): the top edge on f is the wall face's top, and steps where the tile under it crosses a boundary",
                             "\(wrong.count) wrong of \(vertices) vertices, \(crossings) ramps, steps by tile \(steps): "
                                + wrong.prefix(3).joined(separator: "; "))
            }
            // A corner's vertical edge, where two walls meet, rises to the
            // lower of their tops and on to the higher beside the riser that
            // ends there, unless the higher wall stands on that riser's
            // boundary, in one plane with it: to the top of the wall that
            // leaves the boundary, the boundary and the leaving read through
            // the tile under the corner. Read off the walls' own top edges,
            // on the grid, as the corners are.
            guard l.refine == nil,
                  let drawn = l.surface.hatch(.wallEdges(edges: Array(perimeter.edges.indices), step: 1, base: base),
                                              in: Surface.HatchContext(perimeter: perimeter, region: .full, tiles: l.tiles))
            else { continue }
            var corners = 0, wrong: [String] = []
            for (k, edge) in perimeter.edges.enumerated() {
                let corner = edge.start
                let previous = (k + perimeter.edges.count - 1) % perimeter.edges.count
                let here = drawn[path: 2 * k], before = drawn[path: 2 * previous]
                let back = carriedBack(corner.v, l).back
                func alongX(_ e: PerimeterEdge) -> Bool {
                    let d = e.end.v - e.start.v
                    let carried = SIMD2(back.a * d.x + back.b * d.y, back.c * d.x + back.d * d.y)
                    return abs(carried.x) <= 1e-12 * simd_length(carried)
                }
                let walls = [(top: here[here.startIndex].z, alongX: alongX(edge)),
                             (top: before[before.endIndex - 1].z, alongX: alongX(perimeter.edges[previous]))]
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

    Check.suite("restriction: a contour's copy is cut to the footprint where it lands, exactly") {
        // On tiles whose footprint is not symmetric under the tiling,
        // reflected and turned: every kept piece of a copy lies in the
        // footprint where the copy stands, ending on its perimeter or where
        // the contour ends, and the copy is kept wherever it is in the
        // footprint, read sixteen times a segment. With a band of Im z as
        // well, read on the domain the copy came from. And an untiled layer,
        // its one copy in place, wherever the footprint cuts the domain.
        let landscapes = try faceLandscapes().filter { l in
            guard case .inside = l.region else { return false }
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
        var landscapes = try faceLandscapes()
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
            Check.expect(t.runs > 0 && t.endsOff == 0 && t.samples > 200 && t.misclipped == 0,
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
        // boundary: the banded truncation edge drops the vertices beside the
        // boundary, and with nothing on it to join the ends to they are
        // placed at the plateau's own level instead, so the plateau's loop
        // closes, in the drawing as in the loops. On a coarse grid, and on a
        // fine one with the plateau nearer the boundary.
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

    Check.suite("faces: a pole beside a band boundary, at every resolution, has its truncation edge on its own band's level") {
        // Poles 1/(z - c) along Re c from -0.2 to 1.2, Im c wandering within
        // 0.13 of the axis, beside the boundary Re z = 1/2 under 2 | 4 and
        // 4 | 2, at six resolutions: their plateaus cross the boundary, end
        // on it, or turn back short of it within a column. Every vertex of
        // the placed truncation edge, off the boundary, is in the band at its
        // own x and on |f| = that band's height. Open chains: 237 on main
        // 1ae3d87 over the 492; four remain, at the coarsest grids, where
        // the higher band's plateau turns back a column short of the
        // boundary and the grid's line runs through the dropped cells into
        // the lower band's plateau, which crosses it, so no run of its own
        // band lies across the gap to join it to.
        let domain = Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1))
        let knownOpen: Set<String> = ["2|4 res 24 c 0.78", "2|4 res 32 c 0.78", "4|2 res 24 c 0.22", "4|2 res 32 c 0.22"]
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
        Check.expect(Set(open.keys).isSubset(of: knownOpen) && open.values.allSatisfy { $0 == 1 },
                     "and no more open chains than the four known, a plateau turning back into another band's line",
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
