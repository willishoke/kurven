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
    var sampled = 0, wrong = 0, inward = 0, skewed = 0
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
                    // The facing's differences reach a cell out, and where
                    // they reach ground two tiles hold they read the first,
                    // which need not be this copy's.
                    let reach = [(cell.x, 0.0), (-cell.x, 0), (0, cell.y), (0, -cell.y)]
                    if face.id == .graph, reach.allSatisfy({ cover(P3(ground.x + $0.0, ground.y + $0.1, 0)) <= 1 }) {
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

    // The band boundaries in the domain, the far end included.
    var boundaries: [Double] = []
    if case .realBands(let bands, _) = truncation {
        boundaries = Array(Set(bands.map(\.below).filter { $0 >= lo && $0 <= hi })).sorted()
    }
    func onBoundary(_ x: Double) -> Bool { boundaries.contains { abs(x - $0) <= 1e-12 * max(1, abs($0)) } }

    // The walls: from the base to the top their curtain draws, or with f
    // their top edge, along each edge of the perimeter, a crease where two
    // meet; nothing under the base or over the top. Outward leaves the
    // perimeter.
    var wallFeet: [(a: P2<WorldSpace>, b: P2<WorldSpace>)] = []
    if case .perimeter(let perimeter, let base) = l.walls {
        let corners = perimeter.edges.map(\.start)
        let eps = 1e-6 * size
        for (k, edge) in perimeter.edges.enumerated() {
            wallFeet.append((edge.start, edge.end))
            let stations: [P3<WorldSpace>]
            if onF {
                let context = Surface.HatchContext(perimeter: perimeter, region: .full, tiles: l.tiles)
                let drawn = surface.hatch(.wallEdges(edges: [k], step: 1, base: base), in: context, refine: l.refine)
                stations = drawn.map { Array($0[path: 0]) } ?? []
            } else {
                let curtain = Mesh.wallCurtain(from: edge.start, to: edge.end, samples: 41, surface: surface,
                                               base: base, tiles: l.tiles)
                stations = Array(curtain.vertices.prefix(41))
            }
            // A wall standing on a band's boundary is asked below, at the
            // boundary's own x: the curtain's feet along it land a rounding
            // either side of it, and read the band there.
            for v in stations where !onBoundary(v.x) {
                let top = v.z
                guard top.isFinite, top > base else { continue }
                let corner = simd_length(v.xy.v - edge.start.v) < 1e-12 || simd_length(v.xy.v - edge.end.v) < 1e-12
                for k in 0..<7 {
                    let z = base + (top - base) * (Double(k) + 0.5) / 7
                    expect(P3(v.x, v.y, z), 1, slack: corner ? 1 : 0) { face, n in
                        guard case .wall = face.id else { return true }
                        let h = simd_normalize(SIMD2(n.x, n.y)) * eps
                        return !pointInPolygon(P2(v.x + h.x, v.y + h.y), corners)
                            && pointInPolygon(P2(v.x - h.x, v.y - h.y), corners)
                    }
                }
                expect(P3(v.x, v.y, top + 1e-3 * max(1, abs(top)) + 0.1 * (top - base)), 0)
                expect(P3(v.x, v.y, base - 0.1 * (top - base)), 0)
            }
        }
    }
    /// The top of a wall standing on a point of the ground, as its curtain
    /// reads it, or nil where none stands.
    func wallTop(_ w: P2<WorldSpace>) -> Double? {
        guard case .perimeter(_, let base) = l.walls else { return nil }
        let stands = wallFeet.contains { a, b in
            let d = b.v - a.v
            let s = max(0, min(1, simd_dot(w.v - a.v, d) / simd_length_squared(d)))
            return simd_length(w.v - (a.v + s * d)) <= 1e-12 * max(1, size)
        }
        return stands ? max(heightfield.surfaceHeight(at: w), base) : nil
    }

    // The risers: at a band boundary the solid drops from the top on one
    // side to the top on the other, read just either side of it; where only
    // one side is the solid and a wall stands on the boundary, from the top
    // on that side to the wall's top. Every point between is in one face, a
    // riser or the wall, the two meeting on a crease; nothing over the
    // higher, nor inside the solid under the lower. Outward is toward the
    // lower side.
    let side = 1e-9 * size
    for b in boundaries {
        for tile in l.tiles {
            for j in 0..<47 {
                let y = imag.lo + imag.length * (Double(j) + 0.53) / 47
                let foot = tile(P3(b, y, 0))
                /// The top just off the boundary, `dx` along x; nil off the
                /// domain or the footprint.
                func top(_ dx: Double) -> Double? {
                    let x = b + dx
                    guard x >= lo, x <= hi else { return nil }
                    let w = tile(P3(x, y, 0))
                    guard cover(w) == 1, l.region.contains(w.xy) else { return nil }
                    let z = heightfield.surfaceHeight(at: w.xy)
                    return z.isFinite ? z : nil
                }
                let left = top(-side), right = top(side)
                // How far f alone moves across the boundary, which is no drop.
                let slide = abs(magnitude(b + side, y) - magnitude(b - side, y))
                let wall = wallTop(foot.xy)
                let span: (lo: Double, hi: Double), lowerSide: Double
                switch (left, right, wall) {
                case let (l?, r?, nil):
                    span = (min(l, r), max(l, r)); lowerSide = l > r ? 1 : -1
                case let (l?, nil, w?):
                    span = (min(w, l), max(w, l)); lowerSide = 1
                case let (nil, r?, w?):
                    span = (min(w, r), max(w, r)); lowerSide = -1
                default:
                    continue
                }
                let outward = SIMD2(tile.a * lowerSide, tile.c * lowerSide)
                if span.hi - span.lo > 4 * slide + 1e-9 * max(1, abs(span.hi)) {
                    for k in 0..<7 {
                        let z = span.lo + (span.hi - span.lo) * (Double(k) + 0.5) / 7
                        expect(P3(foot.x, foot.y, z), 1, slack: wall.map { abs(z - $0) < 1e-9 ? 1 : 0 } ?? 0) { face, n in
                            guard case .riser = face.id else { return true }
                            return simd_dot(SIMD2(n.x, n.y), outward) > 0
                        }
                    }
                }
                expect(P3(foot.x, foot.y, span.hi + 1e-3 * max(1, abs(span.hi))), 0)
                if wall == nil { expect(P3(foot.x, foot.y, span.lo - 1e-3 * max(1, abs(span.lo))), 0) }
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
    // beyond starts on the right wall, far below the band before it.
    let edgeBand = Surface(height: grid, phase: nil,
                           truncation: .realBands([RealBand(below: -1, height: 1.2), RealBand(below: 2.5, height: 2.5)],
                                                  beyond: 0.05))
    landscapes.append(Landscape("the hatch fixture, a band boundary on the far edge", surface: edgeBand,
                                walls: .perimeter(perimeter, base: 0)))
    landscapes.append(Landscape("the hatch fixture, a band boundary on the far edge, no walls", surface: edgeBand))
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
    var runs = 0, endsOff = 0, worstEnd = 0.0, samples = 0, misclipped = 0, worstMisclipped = 0.0
    var examples: [String: [String]] = [:]
    mutating func note(_ kind: String, _ what: String) {
        if examples[kind, default: []].count < 3 { examples[kind, default: []].append(what) }
    }
    func examples(_ kind: String) -> String { examples[kind, default: []].joined(separator: "; ") }
}

/// The graph's trim loops held to the drawing, to membership, and as the
/// cut of a contour, on every copy. Membership is not asked within `margin`
/// cells of the loops: they are the truncation edge as drawn, chords between
/// vertices on |f| = h(x), or the excess grid's marching squares, and
/// `contains` is |f| < h(x) exactly, or the grid's bilinear surface under
/// it. Nor, without a refiner, in a cell whose two columns read different
/// truncation heights: the excess grid interpolates between them there, and
/// its truncation edge with it, while the surface steps at the band's
/// boundary.
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
    t.open = loops.filter { $0.first != $0.last }.count

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

        // A contour on the copy: lines of x and of y across the rectangle.
        guard let copyLoops = copy.trimLoops else { continue }
        let cut = Segments(copyLoops.map { $0.map(\.v) }, cell: cell)
        let rectangle = graph.domain
        let families: [(ScalarField, [Double])] = [
            (ScalarField { $0.x }, (1...13).map { real.lo + real.length * (Double($0) - 0.29) / 13 }),
            (ScalarField { $0.y }, (1...9).map { imag.lo + imag.length * (Double($0) - 0.43) / 9 }),
        ]
        for (field, levels) in families {
            let whole = Contour.levels(of: field, levels, over: rectangle)
            let clipped = Contour.levels(of: field, levels, over: rectangle, contains: copy.contains, trimLoops: copyLoops)
            let paths = whole.flatMap(\.paths), runs = clipped.flatMap(\.paths)
            let pathEnds = Set(paths.flatMap { [$0[0], $0[$0.count - 1]] })
            let kept = Segments(runs.map { $0.map(\.v) }, cell: cell)
            t.runs += runs.count
            for run in runs {
                for e in [run[0], run[run.count - 1]] where !pathEnds.contains(e) && !straddles(e.v) {
                    let d = cut.distance(e.v, within: cell)
                    if !(d <= tolerance) {
                        t.endsOff += 1; t.worstEnd = max(t.worstEnd, d)
                        t.note("clipped", "a run ends at \(e.v), \(d.isFinite ? "\(d / cell)" : "over a") cell(s) off the loops")
                    }
                }
            }
            for path in paths {
                for (a, b) in zip(path, path.dropFirst()) {
                    for k in 0..<16 {
                        let q = a.v + (b.v - a.v) * ((Double(k) + 0.5) / 16)
                        guard cut.distance(q, within: margin * cell).isInfinite, !straddles(q) else { continue }
                        t.samples += 1
                        let onRun = kept.distance(q, within: cell) <= tolerance
                        if onRun != copy.contains(P2(q)) {
                            t.misclipped += 1
                            t.worstMisclipped = max(t.worstMisclipped, cut.distance(q, within: 10 * cell) / cell)
                            t.note("clipped", "\(q) on tile \(copy.tile) \(onRun ? "kept" : "cut away"), contains says otherwise")
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
            Check.expect(t.wrong == 0 && t.inward == 0 && t.skewed == 0 && t.sampled > 10_000,
                         "\(l.name): \(kinds), \(l.tiles.count) tile(s); every point of the boundary in one face, "
                            + "none off it, each face turned out, the graph's normal the facing's",
                         "\(t.wrong) in the wrong number, \(t.inward) facing in, \(t.skewed) normals off, "
                            + "of \(t.sampled): \(t.examples.sorted { $0.key < $1.key }.map(\.value).joined(separator: "; "))")
        }
        // `Surface.faces` lists them in `FaceID`'s order.
        Check.expect(landscapes.allSatisfy { l in zip(l.faces, l.faces.dropFirst()).allSatisfy { $0.id <= $1.id } },
                     "the faces come in FaceID's order: graph, plateaus, risers, walls")
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
            // Measured: within 0.015 cells on Γ and 0.043 on the tower, the
            // refined truncation edge's chords; 0.16 on the hatch fixture,
            // marching squares' chords under a bilinear surface that curves
            // within a cell.
            let margin = l.refine == nil ? 0.25 : 0.05
            guard let t = trimLoops(l, margin: margin) else {
                Check.expect(l.faces.isEmpty, "\(l.name): the graph has trim loops"); continue
            }
            Check.expect(t.open == 0 && t.drawnOff == 0,
                         "\(l.name): the graph's trim loops are closed, and the truncation edge drawn lies on them",
                         "\(t.open) open, \(t.drawnOff) of \(t.drawn) drawn vertices off, worst \(t.worstDrawn): \(t.examples("drawn"))")
            Check.expect(t.asked > 1000 && t.misplaced == 0,
                         "\(l.name): a point the loops and the footprint hold is in the graph, and one they do not is out",
                         "\(t.misplaced) of \(t.asked) disagree, worst \(t.worstMisplaced) cells from the loops: \(t.examples("placed"))")
            Check.expect(t.runs > 0 && t.endsOff == 0 && t.samples > 200 && t.misclipped == 0,
                         "\(l.name): lines of x and y on each copy are cut where they cross its loops, and kept where it holds them",
                         "\(t.endsOff) of \(t.runs) runs end off the loops (worst \(t.worstEnd)), \(t.misclipped) of \(t.samples) samples misclipped "
                            + "(worst \(t.worstMisclipped) cells from the loops): \(t.examples("clipped"))")
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
                for (level, runs) in Contour.levels(of: field, levels, over: face.domain,
                                                    contains: copy.contains, trimLoops: loops) {
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
