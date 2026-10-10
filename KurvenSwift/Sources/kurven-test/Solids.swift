import Foundation
import simd
import KurvenCore
import KurvenMetal
import KurvenBake

// MARK: - the solids
//
// A polyhedron's visibility is the resolution test in its purest form: an
// edge is seen where one of its faces faces the eye and nothing stands in
// front, decided per edge by two signs and, on a body that can stand in its
// own way, by casting the edge against the faces in closed form. Nothing
// here has a pixel in it, so a bake at 64 is the bake at 4096 to the bit --
// and the convex case is held to a rule simpler still, the two signs alone.

/// Whether two triangles cross: an edge of one passes through the open
/// interior of the other. Triangles sharing an edge cannot cross; ones
/// sharing a vertex are tested away from it.
func trianglesCross(_ a: [SIMD3<Double>], _ b: [SIMD3<Double>], shared: Int) -> Bool {
    guard shared < 2 else { return false }
    func through(_ p: SIMD3<Double>, _ q: SIMD3<Double>, _ t: [SIMD3<Double>]) -> Bool {
        let n = simd_cross(t[1] - t[0], t[2] - t[0])
        let d = q - p
        let den = simd_dot(n, d)
        guard abs(den) > 1e-12 else { return false }
        let s = simd_dot(n, t[0] - p) / den
        guard s > 1e-9, s < 1 - 1e-9 else { return false }
        let x = p + s * d
        let v0 = t[1] - t[0], v1 = t[2] - t[0], v2 = x - t[0]
        let d00 = simd_dot(v0, v0), d01 = simd_dot(v0, v1), d11 = simd_dot(v1, v1)
        let d20 = simd_dot(v2, v0), d21 = simd_dot(v2, v1)
        let den2 = d00 * d11 - d01 * d01
        let v = (d11 * d20 - d01 * d21) / den2, w = (d00 * d21 - d01 * d20) / den2
        return v > 1e-9 && w > 1e-9 && 1 - v - w > 1e-9
    }
    for k in 0..<3 where through(a[k], a[(k + 1) % 3], b) { return true }
    for k in 0..<3 where through(b[k], b[(k + 1) % 3], a) { return true }
    return false
}

func solidTests() {
    Check.suite("solids: the five Platonic solids come out of their coordinates alone") {
        let expected: [(Solid.Platonic, v: Int, e: Int, f: Int, sides: Int)] = [
            (.tetrahedron, 4, 6, 4, 3), (.cube, 8, 12, 6, 4), (.octahedron, 6, 12, 8, 3),
            (.dodecahedron, 20, 30, 12, 5), (.icosahedron, 12, 30, 20, 3),
        ]
        for (kind, v, e, f, sides) in expected {
            let s = Solid.platonic(kind, circumradius: 1)
            let lengths = s.edges.map { simd_length(s.vertices[$0.a].v - s.vertices[$0.b].v) }
            let regular = (lengths.max()! - lengths.min()!) < 1e-9
                && s.faces.allSatisfy { $0.count == sides }
                && s.vertices.allSatisfy { abs(simd_length($0.v) - 1) < 1e-9 }
            Check.expect(s.vertices.count == v && s.edges.count == e && s.faces.count == f
                         && s.eulerCharacteristic == 2 && s.isConvex && regular,
                         "\(kind.rawValue): \(v) vertices, \(e) edges, \(f) \(sides)-gons, χ = 2, convex, regular",
                         "\(s.vertices.count) vertices, \(s.edges.count) edges, \(s.faces.count) faces")
        }
    }

    Check.suite("solids: on a convex solid the sign test is the whole test") {
        // Every piece drawn is a whole edge, and the edges drawn are exactly
        // the ones with a face toward the eye -- under every camera,
        // including ones a degree off edge-on to a face, where a single
        // sign decides a whole pentagon's worth of edges.
        for kind in Solid.Platonic.allCases {
            let s = Solid.platonic(kind, circumradius: 1).resting(on: 0)
            var cameras = 0, whole = true, bySigns = true, drawnSome = false, undecided = 0
            for (_, view) in stressCameras() {
                cameras += 1
                let d = -view.sightLine
                let drawn = s.visibleEdges(view: view)
                // An edge with a face exactly edge-on -- a cube square to the
                // camera -- lies on the silhouette, and its sign is rounding;
                // whether it is drawn is not a question with an answer.
                func decidable(_ e: Solid.Edge) -> Bool {
                    abs(simd_dot(s.normals[e.left], d)) > 1e-9 && abs(simd_dot(s.normals[e.right], d)) > 1e-9
                }
                let wanted = Set(s.edges.filter {
                    decidable($0) && (simd_dot(s.normals[$0.left], d) > 0 || simd_dot(s.normals[$0.right], d) > 0)
                })
                undecided += s.edges.filter { !decidable($0) }.count
                var got = Set<Solid.Edge>()
                for i in 0..<drawn.count {
                    let p = drawn[path: i]
                    guard let e = s.edges.first(where: { e in
                        let a = s.vertices[e.a].v, b = s.vertices[e.b].v
                        let len = simd_length(b - a)
                        let t0 = simd_dot(p.first!.v - a, b - a) / (len * len)
                        let t1 = simd_dot(p.last!.v - a, b - a) / (len * len)
                        return simd_length(p.first!.v - (a + t0 * (b - a))) < 1e-9
                            && simd_length(p.last!.v - (a + t1 * (b - a))) < 1e-9
                            && t0 >= -1e-9 && t1 <= 1 + 1e-9
                    }) else { whole = false; continue }
                    guard decidable(e) else { continue }
                    if simd_length(p.first!.v - s.vertices[e.a].v) > 1e-12
                        || simd_length(p.last!.v - s.vertices[e.b].v) > 1e-12 { whole = false }
                    got.insert(e)
                }
                if got != wanted { bySigns = false }
                if drawn.count > 0 { drawnSome = true }
            }
            Check.expect(whole && bySigns && drawnSome,
                         "\(kind.rawValue): under \(cameras) cameras every drawn piece is a whole edge, and the set is the sign test's",
                         undecided > 0 ? "\(undecided) edge-on cases set aside" : "")
        }
    }

    Check.suite("solids: the drawing is the same at every resolution, to the bit") {
        let s = Solid.platonic(.dodecahedron, circumradius: 2).resting(on: 0)
        let camera = Camera.plate(PlateProjection(shear: 0.3, xAngle: -55, zAngle: -70, flipX: false, yScale: nil))
        let layer = Layer(spec: LayerSpec(name: "edges", role: .outline, source: .edges,
                                          width: 0.5, heightPolicy: .surface), paths: s.edgePaths)
        let scene = Scene(solid: s, layers: [layer], camera: camera)
        let renderer = try MetalRenderer()
        let coarse = try renderer.bake(scene, options: BakeOptions(resolution: 64))
        let fine = try renderer.bake(scene, options: BakeOptions(resolution: 4096))
        Check.expect(coarse.strokes.layers[0].paths == fine.strokes.layers[0].paths
                     && coarse.strokes.pathCount > 10,
                     "a dodecahedron bakes the same strokes at 64 px and 4096 px",
                     "\(coarse.strokes.pathCount) edges drawn")
        // The depth pass drew the solid, for the depth output's sake.
        var covered = 0
        for r in 0..<coarse.depth.frame.rows { for c in 0..<coarse.depth.frame.cols
            where coarse.depth.isCovered(row: r, col: c) { covered += 1 } }
        Check.expect(covered > 64 * 64 / 5, "and the depth pass rasterized its faces",
                     "\(covered) of \(64 * 64) pixels covered")
    }

    Check.suite("solids: the Császár polyhedron is the seven-vertex torus, and its edges are cast") {
        let s = Solid.csaszar
        Check.expect(s.vertices.count == 7 && s.edges.count == 21 && s.faces.count == 14
                     && s.eulerCharacteristic == 0,
                     "7 vertices, 21 edges, 14 triangles, χ = 0")
        var pairs = Set<SIMD2<Int>>()
        for e in s.edges { pairs.insert(SIMD2(e.a, e.b)) }
        Check.expect(pairs.count == 21, "every pair of vertices is an edge: K₇ on the torus")
        Check.expect(!s.isConvex, "it is not convex")
        // Embedded: no two faces cross.
        var crossings = 0
        for i in 0..<s.faces.count {
            for j in (i + 1)..<s.faces.count {
                let shared = Set(s.faces[i]).intersection(s.faces[j]).count
                if trianglesCross(s.faces[i].map { s.vertices[$0].v }, s.faces[j].map { s.vertices[$0].v },
                                  shared: shared) { crossings += 1 }
            }
        }
        Check.expect(crossings == 0, "no two faces cross: the surface is embedded")
        // The abstract triangulation: vertices mod 7, faces {i, i+1, i+3}
        // and {i, i+2, i+3}, under some relabelling.
        func abstractFaces() -> Set<[Int]> {
            var out = Set<[Int]>()
            for i in 0..<7 {
                out.insert([i, (i + 1) % 7, (i + 3) % 7].sorted())
                out.insert([i, (i + 2) % 7, (i + 3) % 7].sorted())
            }
            return out
        }
        let faceSet = Set(s.faces.map { $0.sorted() })
        var relabelled = false
        var perm = Array(0..<7)
        func permute(_ k: Int) {
            if relabelled { return }
            if k == 7 {
                let image = Set(abstractFaces().map { f in f.map { perm[$0] }.sorted() })
                if image == faceSet { relabelled = true }
                return
            }
            for i in k..<7 {
                perm.swapAt(k, i); permute(k + 1); perm.swapAt(k, i)
            }
        }
        permute(0)
        Check.expect(relabelled, "its faces are the K₇ triangulation {i, i+1, i+3}, {i, i+2, i+3} mod 7")

        // The sign test is not sufficient here: under some cameras an edge
        // that passes it runs partly behind a face, and the cast splits it.
        // Every piece the cast draws is held to a brute-force sampling of
        // the edge, point by point against every face.
        let mesh = s.mesh
        var split = 0, cameras = 0, compared = 0, wrong = 0
        var example = ""
        for (name, view) in stressCameras() {
            cameras += 1
            let d = -view.sightLine
            let drawn = s.visibleEdges(view: view)
            var pieces: [Solid.Edge: [(Double, Double)]] = [:]
            for i in 0..<drawn.count {
                let p = drawn[path: i]
                for e in s.edges {
                    let a = s.vertices[e.a].v, b = s.vertices[e.b].v
                    let len = simd_length(b - a)
                    let t0 = simd_dot(p.first!.v - a, b - a) / (len * len)
                    let t1 = simd_dot(p.last!.v - a, b - a) / (len * len)
                    let off = simd_length(p.first!.v - (a + t0 * (b - a))) + simd_length(p.last!.v - (a + t1 * (b - a)))
                    if off < 1e-9, t0 >= -1e-9, t1 <= 1 + 1e-9 {
                        pieces[e, default: []].append((t0, t1))
                        if t0 > 1e-9 || t1 < 1 - 1e-9 { split += 1 }
                        break
                    }
                }
            }
            for e in s.edges {
                let passes = simd_dot(s.normals[e.left], d) > 0 || simd_dot(s.normals[e.right], d) > 0
                let a = s.vertices[e.a].v, b = s.vertices[e.b].v
                let runs = pieces[e] ?? []
                for k in 0..<200 {
                    let t = (Double(k) + 0.5) / 200
                    // Within a hair of a piece's end the truth is at a
                    // boundary; elsewhere it is decided.
                    if runs.contains(where: { abs(t - $0.0) < 1e-6 || abs(t - $0.1) < 1e-6 }) { continue }
                    let p = a + t * (b - a)
                    let seenByCast = runs.contains { t > $0.0 && t < $0.1 }
                    // The brute force: a hit on any face but the edge's own
                    // two, which contain the point.
                    var others = mesh
                    _ = others
                    let own = Set([e.left, e.right])
                    var hit = false
                    for (f, face) in s.faces.enumerated() where !own.contains(f) {
                        let tri = Mesh<WorldSpace>(vertices: face.map { s.vertices[$0] },
                                                   triangles: [SIMD3(0, 1, 2)])
                        if firstHit(tri, from: p, along: d, after: 1e-9) != nil { hit = true; break }
                    }
                    let truth = passes && !hit
                    compared += 1
                    if truth != seenByCast {
                        wrong += 1
                        if example.isEmpty {
                            example = String(format: "e.g. %@ edge %d-%d t=%.3f truth %@", name, e.a, e.b, t,
                                             truth ? "seen" : "hidden")
                        }
                    }
                }
            }
        }
        Check.expect(split > 0, "under \(cameras) cameras some edges are drawn in pieces: the sign test alone would be wrong",
                     "\(split) pieces are parts of edges")
        Check.expect(wrong == 0 && compared > 10_000,
                     "and every piece agrees with a point-by-point ray cast against the faces",
                     "\(wrong) of \(compared) sampled points disagree \(example)")
    }
}
