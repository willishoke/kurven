import Foundation
import simd
import KurvenCore
import KurvenLandscape

// MARK: - faces
//
// A solid's faces tile its boundary: a point of the drawn surface lies in
// exactly one face, or on a boundary two of them share, and that face's map
// returns it; a point off the surface lies in none. The drawn surface is
// sampled here from the landscape's own definition -- min(|f|, h(x)) over
// the footprint, the walls up to their top edge, the risers between two
// bands' heights -- and not from the faces, which are what is held to it.

/// The parameter at which `face`, moved by `tile`, passes through `q`, if
/// it does. The graph and the plateaus lie over the domain, so a point's
/// parameter is its (x, y); every other face of a landscape or a polyhedron
/// is flat with an affine map, and a point's parameter is its projection
/// onto the face's plane. The map returning the point is what decides.
private func parameter(of q: P3<WorldSpace>, on face: Face, tile: Affine2) -> P2<ParamSpace>? {
    guard let back = tile.inverse else { return nil }
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
    guard face.contains(p), simd_length(face.map(p).position - local) <= tolerance else { return nil }
    return p
}

/// The faces a point lies in, by id: a face copied to several tiles is one
/// face however many of its copies hold the point.
private func faceIDs(holding q: P3<WorldSpace>, in faces: [Face]) -> [FaceID] {
    faces.filter { face in face.instances.contains { parameter(of: q, on: face, tile: $0) != nil } }.map(\.id)
}

/// A landscape as the faces are built from it.
private struct Landscape {
    let name: String
    let surface: Surface
    let tiles: [Affine2]
    let region: Region
    let walls: Walls
    let refine: ContourRefine?
    let faces: [Face]
}

/// Points of the drawn surface and points off it, sampled from the
/// landscape's own definition, against its faces: how many were asked,
/// those in the wrong number of faces, those whose face is turned inward,
/// and an example of the first wrong.
private func tiling(_ l: Landscape) -> (sampled: Int, wrong: Int, inward: Int, example: String) {
    let surface = l.surface, truncation = surface.truncation
    func magnitude(_ x: Double, _ y: Double) -> Double {
        let p = P2<DomainSpace>(x, y)
        let u = l.refine?.magnitude?(p) ?? surface.magnitude(at: p)
        return u.isNaN ? .infinity : u
    }
    let real = surface.domain.real, imag = surface.domain.imag
    let size = max(abs(real.length), abs(imag.length))
    var sampled = 0, wrong = 0, inward = 0
    var example = ""
    /// `q` lies in `expected` faces, or within one of that where it is on
    /// a boundary; when it lies in one, that face's outward side is `out`.
    func expect(_ q: P3<WorldSpace>, _ expected: Int, boundary: Bool = false,
                out: ((FaceID, SIMD3<Double>) -> Bool)? = nil) {
        sampled += 1
        let ids = faceIDs(holding: q, in: l.faces)
        if ids.count != expected && !(boundary && abs(ids.count - expected) <= 1) {
            wrong += 1
            if example.isEmpty { example = "\(q.v) in \(ids), not \(expected) face(s)" }
            return
        }
        guard let out, ids.count == 1, let face = l.faces.first(where: { $0.id == ids[0] }),
              let p = parameter(of: q, on: face, tile: .identity) else { return }
        let jet = face.map(p)
        if !out(face.id, simd_cross(jet.du, jet.dv) * (face.outward ?? 0)) {
            inward += 1
            if example.isEmpty { example = "\(face.id) at \(q.v) faces in" }
        }
    }

    /// How many tiles cover a world point of the ground: where two do, as
    /// the contract fixture's overlap, a point over the one tile's surface
    /// can be on the other's, and only points on the surface are asked.
    func cover(_ q: P3<WorldSpace>) -> Int {
        l.tiles.filter { tile in
            guard let back = tile.inverse else { return false }
            let p = back(q)
            return (min(real.lo, real.hi)...max(real.lo, real.hi)).contains(p.x)
                && (min(imag.lo, imag.hi)...max(imag.lo, imag.hi)).contains(p.y)
        }.count
    }

    // The top: the graph or a plateau over every point of the footprint, at
    // min(|f|, h(x)), on every tile; nothing just above it or just under it,
    // nor anywhere off the footprint. Outward is up.
    for tile in l.tiles {
        for j in 0..<47 {
            for i in 0..<83 {
                let x = real.lo + real.length * (Double(i) + 0.37) / 83
                let y = imag.lo + imag.length * (Double(j) + 0.61) / 47
                let m = magnitude(x, y), h = truncation.height(atX: x)
                let z = min(m, h)
                guard z.isFinite else { continue }
                let inFootprint = l.region.contains(P2(x, y))
                let onCrease = abs(m - h) <= 1e-9 * max(1, abs(h))
                expect(tile(P3(x, y, z)), inFootprint ? 1 : 0, boundary: onCrease,
                       out: tile == .identity ? { _, n in n.z > 0 } : nil)
                guard cover(tile(P3(x, y, 0))) == 1 else { continue }
                let off = 1e-3 * max(1, abs(z))
                expect(tile(P3(x, y, z + off)), 0)
                expect(tile(P3(x, y, z - off)), 0)
                // The sheet that is not drawn here, over it: the graph above
                // a plateau, a plateau's height above the graph.
                let over = max(m, h)
                if over.isFinite, !onCrease { expect(tile(P3(x, y, over)), 0) }
            }
        }
    }

    // The walls: each from the base to the top edge along its perimeter
    // edge, nothing over the top edge or under the base. Outward leaves the
    // perimeter. Read on the identity, as the fixtures with walls have one
    // tile.
    if case .perimeter(let perimeter, let base) = l.walls, l.tiles == [.identity] {
        let corners = perimeter.edges.map(\.start)
        let eps = 1e-6 * size
        for edge in perimeter.edges {
            for i in 0..<31 {
                let t = (Double(i) + 0.4) / 31
                let x = edge.start.x + (edge.end.x - edge.start.x) * t
                let y = edge.start.y + (edge.end.y - edge.start.y) * t
                let top = min(magnitude(x, y), truncation.height(atX: x))
                guard top.isFinite, top > base else { continue }
                for k in 0..<7 {
                    let z = base + (top - base) * (Double(k) + 0.5) / 7
                    expect(P3(x, y, z), 1) { _, n in
                        let h = simd_normalize(SIMD2(n.x, n.y)) * eps
                        return !pointInPolygon(P2(x + h.x, y + h.y), corners)
                            && pointInPolygon(P2(x - h.x, y - h.y), corners)
                    }
                }
                expect(P3(x, y, top + 1e-3 * max(1, abs(top)) + 0.1 * (top - base)), 0)
                expect(P3(x, y, base - 0.1 * (top - base)), 0)
            }
        }
    }

    // The risers: where the truncation steps between two bands, from the
    // lesser height up to |f| or the greater, whichever is lower; nothing
    // over that or under the lesser. Outward is toward the lower band.
    let bands = truncation.bands(from: min(real.lo, real.hi), to: max(real.lo, real.hi))
    for (left, right) in zip(bands, bands.dropFirst()) where left.height != right.height {
        let b = left.hi
        let lesser = min(left.height, right.height), greater = max(left.height, right.height)
        let eps = 1e-6 * size
        for tile in l.tiles {
            for j in 0..<47 {
                let y = imag.lo + imag.length * (Double(j) + 0.53) / 47
                guard l.region.contains(P2(b, y)) else { continue }
                let top = min(magnitude(b, y), greater)
                guard top.isFinite else { continue }
                if top > lesser {
                    for k in 0..<7 {
                        let z = lesser + (top - lesser) * (Double(k) + 0.5) / 7
                        expect(tile(P3(b, y, z)), 1, out: tile == .identity ? { _, n in
                            truncation.height(atX: b + eps * n.x) < truncation.height(atX: b - eps * n.x)
                        } : nil)
                    }
                }
                expect(tile(P3(b, y, max(top, lesser) + 1e-3 * max(1, abs(top)))), 0)
                expect(tile(P3(b, y, lesser - 1e-3 * max(1, abs(lesser)))), 0)
            }
        }
    }
    return (sampled, wrong, inward, example)
}

func faceTests() {
    Check.suite("faces: a landscape's faces tile its boundary, and each face's map returns its points") {
        var landscapes: [Landscape] = []
        // The contract fixtures: tiles, mesh walls, a footprint, and bands
        // that lie beyond the domain. Through the scene where the bundle has
        // a camera, so the walls are the ones a bundle's scene carries.
        for url in Fixtures.contractBundles {
            let bundle = try KurvenBundle.read(at: url)
            let o = bundle.manifest.occluder
            let geometry: SceneGeometry = bundle.manifest.presets.first.map {
                Scene(bundle: bundle, preset: $0).geometry
            } ?? .heightfield(Heightfield(surface: bundle.surface, occluder: .empty, tiles: o.tiles,
                                          region: o.region, step: 1, refine: bundle.refine, walls: o.walls))
            landscapes.append(Landscape(name: url.lastPathComponent, surface: bundle.surface, tiles: o.tiles,
                                        region: o.region, walls: o.walls, refine: bundle.refine,
                                        faces: geometry.faces))
        }
        // The hatch fixture under its two truncations, its bands inside the
        // domain, so it has risers; and banded again on a footprint with a
        // notch cut from its front, its walls standing on the notch.
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
            let walls = Walls.perimeter(perimeter, base: 0)
            landscapes.append(Landscape(name: "the hatch fixture, \(name)", surface: surface, tiles: [.identity],
                                        region: .full, walls: walls, refine: nil,
                                        faces: surface.faces(walls: walls)))
        }
        let notched = BoundaryPerimeter(edges: zip(
            [(-2.0, -1.0), (0.5, -1.0), (0.5, 0.0), (1.5, 0.0), (1.5, -1.0), (2.5, -1.0), (2.5, 1.5), (-2.0, 1.5)],
            [(0.5, -1.0), (0.5, 0.0), (1.5, 0.0), (1.5, -1.0), (2.5, -1.0), (2.5, 1.5), (-2.0, 1.5), (-2.0, -1.0)]
        ).map { PerimeterEdge(start: P2($0.0, $0.1), end: P2($1.0, $1.1), density: 9) })
        let banded = Surface(height: grid, phase: nil,
                             truncation: try Truncation(json: truncations.value("bands", "hatch.truncations")))
        landscapes.append(Landscape(
            name: "the hatch fixture, bands, on a notched footprint", surface: banded, tiles: [.identity],
            region: .inside(notched), walls: .perimeter(notched, base: -0.5), refine: nil,
            faces: banded.faces(region: .inside(notched), walls: .perimeter(notched, base: -0.5))))
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
        landscapes.append(Landscape(name: "Γ, banded, on f", surface: gamma.surface, tiles: o.tiles,
                                    region: o.region, walls: o.walls, refine: gamma.refine,
                                    faces: scene.geometry.faces))
        // Under the truncation the graph's derivatives are the slope the
        // facing is read from.
        if let h = scene.heightfield, let graph = scene.geometry.faces.first, graph.id == .graph {
            var worst = 0.0, compared = 0
            for j in 0..<31 { for i in 0..<53 {
                let p = P2<ParamSpace>(-4.5 + 9 * (Double(i) + 0.37) / 53, -2.5 + 5 * (Double(j) + 0.61) / 31)
                let jet = graph.map(p)
                guard graph.contains(p), jet.position.z < 0.9 * gamma.surface.truncation.height(atX: p.x) else { continue }
                let n = simd_cross(jet.du, jet.dv), facing = h.normal(at: P2(p.x, p.y))
                worst = max(worst, simd_length(n - facing) / simd_length(facing))
                compared += 1
            } }
            Check.expect(worst < 1e-12 && compared > 500, "Γ: the graph's du × dv is the normal the facing reads",
                         "worst \(worst) of \(compared)")
        }

        for l in landscapes {
            guard !l.faces.isEmpty else {
                Check.expect(l.surface.height.width < 2 || l.surface.height.height < 2,
                             "\(l.name): no faces, for a grid too small to span its domain")
                continue
            }
            let (sampled, wrong, inward, example) = tiling(l)
            let kinds = Dictionary(grouping: l.faces.map(\.id)) { id -> String in
                switch id {
                case .graph: "graph"
                case .plateau: "plateaus"
                case .riser: "risers"
                case .wall: "walls"
                case .surface, .solid: "other"
                }
            }.mapValues(\.count).sorted { $0.key < $1.key }.map { "\($0.value) \($0.key)" }.joined(separator: ", ")
            Check.expect(wrong == 0 && inward == 0 && sampled > 10_000,
                         "\(l.name): \(kinds); every point on the surface in one face, every point off it in none",
                         "\(wrong) in the wrong number, \(inward) facing in, of \(sampled) \(example)")
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
                    let holders = pieces.filter { crossings(p, $0.corners) % 2 == 1 }
                    let inFace = face.contains(P2(p))
                    if holders.count != (inFace ? 1 : 0) { mismatches += 1 }
                    guard let holder = holders.first, holders.count == 1 else { continue }
                    // A point inside the face is in it and in no other face,
                    // and the face's map turns out of the solid there.
                    inside += 1
                    let q = P3<WorldSpace>(frame.point(p))
                    let ids = faceIDs(holding: q, in: faces)
                    if ids != [.solid(face: i)] { untiled += 1 }
                    let jet = face.map(P2(p))
                    if simd_dot(simd_cross(jet.du, jet.dv) * (face.outward ?? 0), s.normals[holder.piece]) <= 0 {
                        inward += 1
                    }
                } }
            }
            Check.expect(faces.count == s.loops.count && faces.map(\.id) == s.loops.indices.map { .solid(face: $0) },
                         "\(name): a face per loop list, in order", "\(faces.count) faces")
            Check.expect(mismatches == 0 && untiled == 0 && inward == 0 && inside > 1000,
                         "\(name): membership by the loops agrees with the pieces, and each point is in its face alone",
                         "\(mismatches) disagree, \(untiled) in another face too, \(inward) facing in, of \(inside) inside")
        }
    }

    Check.suite("faces: a parametric surface is one face, its whole parameter rectangle") {
        let torus = ParametricSurface.torus(major: 3, minor: 1, samples: (u: 40, v: 20))
        let faces = SceneGeometry.parametric(torus).faces
        var agree = faces.count == 1 && faces.first?.id == .surface && faces.first?.outward == torus.outward
        if let face = faces.first {
            for k in 0..<50 {
                // Unwrapped coordinates too: the axes are periodic.
                let p = P2<ParamSpace>(Double(k) * 0.37 - 4, Double(k) * 0.53 - 6)
                agree = agree && face.contains(p) && face.map(p) == torus.map(p)
            }
        }
        Check.expect(agree, "the torus is one face, everywhere in it, mapped by the surface's own map")
    }
}
