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
// own way, by testing the edge for occlusion by the pieces in closed form. Nothing
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
                && s.faces.count == s.pieces.count && s.pieces.allSatisfy { $0.count == sides }
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
            let s = Solid.platonic(kind, circumradius: 1).resting(onFace: 0)
            var cameras = 0, whole = true, bySigns = true, drawnSome = false, undecided = 0
            for (_, view) in stressCameras() {
                cameras += 1
                let d = -view.projectionDirection
                let drawn = s.visibleEdges(view: view)
                // An edge with a face exactly edge-on -- a cube square to the
                // camera -- lies on the outline, and its sign is rounding;
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
        let s = Solid.platonic(.dodecahedron, circumradius: 2).resting(onFace: 0)
        let camera = Camera.oblique(ObliqueProjection(shear: 0.3, xAngle: -55, zAngle: -70, flipX: false, yScale: nil))
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

    Check.suite("solids: the Császár polyhedron is the seven-vertex torus, and its edges are tested for occlusion") {
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
        for i in 0..<s.pieces.count {
            for j in (i + 1)..<s.pieces.count {
                let shared = Set(s.pieces[i]).intersection(s.pieces[j]).count
                if trianglesCross(s.pieces[i].map { s.vertices[$0].v }, s.pieces[j].map { s.vertices[$0].v },
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
        let faceSet = Set(s.pieces.map { $0.sorted() })
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
        // that passes it runs partly behind a face, and occlusion splits it.
        // Every span drawn is held to a brute-force sampling of
        // the edge, point by point against every face.
        var split = 0, cameras = 0, compared = 0, wrong = 0
        var example = ""
        for (name, view) in stressCameras() {
            cameras += 1
            let d = -view.projectionDirection
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
                    let seenByOcclusion = runs.contains { t > $0.0 && t < $0.1 }
                    // The brute force: a hit on any face but the edge's own
                    // two, which contain the point.
                    let own = Set([e.left, e.right])
                    var hit = false
                    for (f, face) in s.pieces.enumerated() where !own.contains(f) {
                        let tri = Mesh<WorldSpace>(vertices: face.map { s.vertices[$0] },
                                                   triangles: [SIMD3(0, 1, 2)])
                        if firstHit(tri, from: p, along: d, after: 1e-9) != nil { hit = true; break }
                    }
                    let truth = passes && !hit
                    compared += 1
                    if truth != seenByOcclusion {
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
                     "and every span agrees with a point-by-point ray cast against the faces",
                     "\(wrong) of \(compared) sampled points disagree \(example)")
    }

    Check.suite("solids: the picture frame is a torus of ten faces, two of them with a hole") {
        let s = Solid.frame(outer: 4, inner: 2, depth: 1)
        let holes = s.loops.reduce(0) { $0 + $1.count - 1 }
        Check.expect(s.vertices.count == 16 && s.edgePaths.count == 24 && s.faces.count == 10 && holes == 2
                     && s.eulerCharacteristic == 0,
                     "16 vertices, 24 edges, 10 faces, 2 holes: 16 − 24 + 10 − 2 = 0, one shell of genus one",
                     "\(s.vertices.count) vertices, \(s.edgePaths.count) edges, \(s.faces.count) faces, \(holes) holes")
        Check.expect(!s.isConvex, "it is not convex")
        // The annuli are cut into convex pieces; the walls are one piece each.
        let sizes = s.faces.map(\.count)
        Check.expect(sizes.dropFirst(2).allSatisfy { $0 == 1 } && sizes.prefix(2).allSatisfy { $0 >= 4 },
                     "each wall is one piece, and each annulus at least four: a square hole has four reflex corners",
                     "pieces per face \(sizes)")
        let partition = partitionReport(s)
        Check.expect(partition.isEmpty, "the pieces are convex, tile their faces, and the partition counts close",
                     partition)
        // Every diagonal joins two pieces of one face, and every edge two
        // pieces of different faces.
        var sorted = true
        for (k, e) in s.edges.enumerated() {
            let same = s.faceIndex(ofPiece: e.left) == s.faceIndex(ofPiece: e.right)
            if same != s.isDiagonal[k] { sorted = false }
        }
        Check.expect(sorted, "a diagonal lies within a face and an edge between two")
    }

    Check.suite("solids: a face with holes is cut into convex pieces that tile it") {
        // Prisms over faces a convex partition has to work for: reflex
        // corners, several holes, a hole near the outer loop, straight corners.
        let square = [SIMD2(-2.0, -2.0), SIMD2(2.0, -2.0), SIMD2(2.0, 2.0), SIMD2(-2.0, 2.0)]
        func box(_ x0: Double, _ y0: Double, _ x1: Double, _ y1: Double) -> [SIMD2<Double>] {
            [SIMD2(x0, y0), SIMD2(x0, y1), SIMD2(x1, y1), SIMD2(x1, y0)]   // clockwise: a hole
        }
        // Five teeth: four slots cut down from the top, eight reflex corners.
        var comb = [SIMD2(0.0, 0.0), SIMD2(9.0, 0.0), SIMD2(9.0, 3.0)]
        for x in [8.0, 6, 4, 2] { comb += [SIMD2(x, 3), SIMD2(x, 1), SIMD2(x - 1, 1), SIMD2(x - 1, 3)] }
        comb.append(SIMD2(0, 3))
        let star = (0..<10).map { k -> SIMD2<Double> in
            let r = k % 2 == 0 ? 3.0 : 1.4, a = Double(k) * .pi / 5
            return SIMD2(r * cos(a), r * sin(a))
        }
        let shapes: [(String, [[SIMD2<Double>]])] = [
            ("an L", [[SIMD2(0, 0), SIMD2(3, 0), SIMD2(3, 1), SIMD2(1, 1), SIMD2(1, 3), SIMD2(0, 3)]]),
            ("a comb", [comb]),
            ("a square with two holes", [square, box(-1.5, -1, -0.5, 1), box(0.5, -1, 1.5, 1)]),
            ("a square with a hole by its outer loop", [square, box(1, -1.9, 1.9, 1.9)]),
            ("a star with a triangular hole",
             [star, [SIMD2(0.5, 0), SIMD2(-0.3, -0.4), SIMD2(-0.3, 0.4)]]),
            ("a square with straight corners", [
                [SIMD2(-2, -2), SIMD2(0, -2), SIMD2(2, -2), SIMD2(2, 0), SIMD2(2, 2), SIMD2(0, 2), SIMD2(-2, 2), SIMD2(-2, 0)],
                [SIMD2(-1, -1), SIMD2(-1, 0), SIMD2(-1, 1), SIMD2(1, 1), SIMD2(1, -1), SIMD2(0, -1)],
            ]),
        ]
        for (name, loops) in shapes {
            let s = prism(loops, depth: 0.5)
            let holes = loops.count - 1
            let pieces = s.faces[0].count
            Check.expect(s.eulerCharacteristic == 2 - 2 * holes,
                         "\(name): the prism over it closes, a solid of genus \(holes)",
                         "χ = \(s.eulerCharacteristic)")
            let partition = partitionReport(s)
            Check.expect(partition.isEmpty, "\(name): \(pieces) convex pieces tile the face, no vertex added",
                         partition)
        }
    }

    Check.suite("solids: on the frame no diagonal is drawn, and the back's inner edges are seen through the hole in part") {
        let s = Solid.frame(outer: 4, inner: 2, depth: 1)
        var onDiagonal = 0, offEdges = 0, cameras = 0, compared = 0, wrong = 0, throughHole = 0
        var example = ""
        for (name, view) in stressCameras() {
            cameras += 1
            let d = -view.projectionDirection
            let drawn = s.visibleEdges(view: view)
            var spans: [Solid.Edge: [(Double, Double)]] = [:]
            for i in 0..<drawn.count {
                let p = drawn[path: i]
                guard let (k, e) = edgeUnder(p.first!.v, p.last!.v, of: s) else { offEdges += 1; continue }
                if s.isDiagonal[k] { onDiagonal += 1 }
                spans[e, default: []].append(edgeFractions(p.first!.v, p.last!.v, e, s))
            }
            // The back's inner edges -- 4-5, 5-6, 6-7, 4-7 -- lie at the
            // bottom of the hole; from above they show only through the hole,
            // and only in part: the near side of the front cuts them.
            for e in s.edges where e.a >= 4 && e.a < 8 && e.b >= 4 && e.b < 8 {
                for (t0, t1) in spans[e] ?? [] where t0 > 1e-9 || t1 < 1 - 1e-9 { throughHole += 1 }
            }
            // An edge with a face exactly edge-on -- a wall of the frame
            // square to the camera -- lies on the outline, and whether it
            // is drawn is decided by rounding: set aside, as for the Platonic
            // solids.
            for (k, e) in s.edges.enumerated() where !s.isDiagonal[k]
                && abs(simd_dot(s.normals[e.left], d)) > 1e-9 && abs(simd_dot(s.normals[e.right], d)) > 1e-9 {
                let (w, c, ex) = occlusionAgreement(s, edge: e, spans: spans[e] ?? [], toEye: d, camera: name)
                wrong += w; compared += c
                if example.isEmpty { example = ex }
            }
        }
        Check.expect(onDiagonal == 0 && offEdges == 0,
                     "under \(cameras) cameras everything drawn lies on an edge, and nothing on a diagonal",
                     "\(onDiagonal) spans on a diagonal, \(offEdges) on no side at all")
        Check.expect(throughHole > 0, "the back's inner edges are drawn in part: seen through the hole, cut by the front",
                     "\(throughHole) spans")
        Check.expect(wrong == 0 && compared > 10_000,
                     "and every span agrees with a point-by-point ray cast against the pieces",
                     "\(wrong) of \(compared) sampled points disagree \(example)")
    }

    Check.suite("solids: a curve on a face breaks where it goes out of sight, and is not joined across") {
        // A frame 2 deep, seen from low over -y along (0, 2, -1): picture
        // (x, y + 2z), depth 2z - y, every coordinate exact. The far wall of
        // the hole, y = 1, is hidden below z = 1 by the near half of the
        // frame, whose top inner edge projects onto it exactly there.
        let s = Solid.frame(outer: 4, inner: 2, depth: 2)
        let view = Transform<WorldSpace, ViewSpace>(rows: (SIMD3(1, 0, 0), SIMD3(0, 1, 2), SIMD3(0, -1, 2)))
        let occlusion = s.occlusion(view: view)
        guard let wall = s.faces.indices.first(where: {
            let f = s.face($0)
            return simd_distance(f.normal, SIMD3(0, -1, 0)) < 1e-12 && abs(f.origin.y - 1) < 1e-12
        }) else { Check.expect(false, "the frame has a far inner wall"); return }
        // Down the wall to the line of sight's edge, on into the hidden
        // part, across it, and back up: seen, hidden, hidden, hidden, seen.
        let path: [P3<WorldSpace>] = [P3(-0.5, 1, 2), P3(-0.5, 1, 1), P3(-0.5, 1, 0.5),
                                      P3(0.5, 1, 0.5), P3(0.5, 1, 1), P3(0.5, 1, 2)]
        let drawn = s.visibleCurves([path], onFace: wall, occlusion: occlusion)
        Check.expect(drawn == [Array(path[0...1]), Array(path[4...5])],
                     "a path on the far wall that dips out of sight is drawn as its two seen ends",
                     "\(drawn.map { $0.map(\.v) })")
    }
}

// MARK: - partition

/// A prism over a face in the plane `z = 0`, `depth` deep: the face's loops
/// as the plane gives them (the outer counter-clockwise, the holes
/// clockwise), its top, its bottom, and a wall along every run of sides in
/// one line -- one face, with the straight corners in its loop.
func prism(_ loops: [[SIMD2<Double>]], depth: Double) -> Solid {
    var vertices: [P3<WorldSpace>] = []
    var top: [[Int]] = [], bottom: [[Int]] = []
    var walls: [[[Int]]] = []
    for loop in loops {
        let base = vertices.count, n = loop.count
        vertices += loop.map { P3($0.x, $0.y, depth) } + loop.map { P3($0.x, $0.y, 0) }
        top.append((0..<n).map { base + $0 })
        bottom.append((0..<n).reversed().map { base + n + $0 })
        func straight(_ k: Int) -> Bool {
            let a = loop[(k + n - 1) % n], b = loop[k], c = loop[(k + 1) % n]
            let u = b - a, v = c - b
            return abs(u.x * v.y - u.y * v.x) < 1e-12 && simd_dot(u, v) > 0
        }
        let start = (0..<n).first { !straight($0) }!
        var run = [start]
        for j in 1...n {
            let k = (start + j) % n
            run.append(k)
            guard !straight(k) else { continue }
            // Along the bottom, then back along the top.
            walls.append([run.map { base + n + $0 } + run.reversed().map { base + $0 }])
            run = [k]
        }
    }
    return Solid(vertices: vertices, faces: [top, bottom] + walls)
}

/// What is wrong with a solid's partition, or nothing: per face, every piece
/// is convex and uses only the face's vertices, the pieces' areas add up to
/// the face's, no point of the face lies in two pieces or in none, and the
/// diagonals number pieces − 1 + holes, which is Euler's formula for a disc
/// with holes cut into discs.
func partitionReport(_ s: Solid) -> String {
    var problems: [String] = []
    for i in s.faces.indices {
        let face = s.face(i)
        let loops = s.loops[i]
        func plane(_ k: Int) -> SIMD2<Double> { face.coordinates(s.vertices[k].v) }
        func area(_ polygon: [Int]) -> Double {
            (0..<polygon.count).reduce(0) { a, k in
                let p = plane(polygon[k]), q = plane(polygon[(k + 1) % polygon.count])
                return a + (p.x * q.y - q.x * p.y) / 2
            }
        }
        let own = Set(loops.joined())
        var pieceArea = 0.0
        for piece in face.pieces.map({ s.pieces[$0] }) {
            pieceArea += area(piece)
            if !Set(piece).isSubset(of: own) || Set(piece).count != piece.count {
                problems.append("face \(i): piece \(piece) is not made of the face's vertices")
            }
            for k in 0..<piece.count {
                let a = plane(piece[k]), b = plane(piece[(k + 1) % piece.count]), c = plane(piece[(k + 2) % piece.count])
                let turn = (b - a).x * (c - b).y - (b - a).y * (c - b).x
                if turn < -1e-12 { problems.append("face \(i): piece \(piece) turns right at \(piece[(k + 1) % piece.count])") }
            }
        }
        let faceArea = loops.reduce(0) { $0 + area($1) }
        if abs(pieceArea - faceArea) > 1e-9 * max(1, faceArea) {
            problems.append("face \(i): the pieces cover \(pieceArea), the face is \(faceArea)")
        }
        // A grid of points, each in the face (even-odd over the loops) or
        // not, and in as many pieces; points near a side are left out.
        func near(_ p: SIMD2<Double>, _ polygon: [Int]) -> Bool {
            (0..<polygon.count).contains { k in
                let a = plane(polygon[k]), b = plane(polygon[(k + 1) % polygon.count])
                let t = max(0, min(1, simd_dot(p - a, b - a) / simd_length_squared(b - a)))
                return simd_length(p - (a + t * (b - a))) < 1e-6
            }
        }
        func crossings(_ p: SIMD2<Double>, _ polygon: [Int]) -> Int {
            (0..<polygon.count).filter { k in
                let a = plane(polygon[k]), b = plane(polygon[(k + 1) % polygon.count])
                guard (a.y > p.y) != (b.y > p.y) else { return false }
                return p.x < a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y)
            }.count
        }
        let all = loops.joined().map(plane)
        let lo = all.reduce(SIMD2(Double.infinity, .infinity)) { simd_min($0, $1) }
        let hi = all.reduce(SIMD2(-Double.infinity, -.infinity)) { simd_max($0, $1) }
        let pieces = face.pieces.map { s.pieces[$0] }
        var mismatches = 0
        for a in 0..<61 { for b in 0..<61 {
            let p = lo + (hi - lo) * SIMD2(Double(a) + 0.37, Double(b) + 0.61) / 61
            if pieces.contains(where: { near(p, $0) }) { continue }
            let inFace = loops.reduce(0) { $0 + crossings(p, $1) } % 2 == 1
            let inPieces = pieces.filter { crossings(p, $0) % 2 == 1 }.count
            if inPieces != (inFace ? 1 : 0) { mismatches += 1 }
        } }
        if mismatches > 0 { problems.append("face \(i): \(mismatches) sample points in the wrong number of pieces") }
        let diagonals = s.edges.indices.filter { s.isDiagonal[$0] && s.faceIndex(ofPiece: s.edges[$0].left) == i }.count
        if diagonals != face.pieces.count - 1 + (loops.count - 1) {
            problems.append("face \(i): \(diagonals) diagonals for \(face.pieces.count) pieces and \(loops.count - 1) holes")
        }
    }
    return problems.prefix(4).joined(separator: "; ")
}

// MARK: - oracles

/// The edge of `s` that the segment `a -> b` lies along, with its index.
func edgeUnder(_ a: SIMD3<Double>, _ b: SIMD3<Double>, of s: Solid) -> (Int, Solid.Edge)? {
    for (k, e) in s.edges.enumerated() {
        let p = s.vertices[e.a].v, q = s.vertices[e.b].v
        let len2 = simd_length_squared(q - p)
        let t0 = simd_dot(a - p, q - p) / len2, t1 = simd_dot(b - p, q - p) / len2
        if simd_length(a - (p + t0 * (q - p))) < 1e-9, simd_length(b - (p + t1 * (q - p))) < 1e-9,
           t0 >= -1e-9, t1 <= 1 + 1e-9 {
            return (k, e)
        }
    }
    return nil
}

func edgeFractions(_ a: SIMD3<Double>, _ b: SIMD3<Double>, _ e: Solid.Edge, _ s: Solid) -> (Double, Double) {
    let p = s.vertices[e.a].v, q = s.vertices[e.b].v
    let len2 = simd_length_squared(q - p)
    return (simd_dot(a - p, q - p) / len2, simd_dot(b - p, q - p) / len2)
}

/// A piece as triangles, fanned, for the brute-force ray cast.
func pieceMesh(_ s: Solid, _ f: Int) -> Mesh<WorldSpace> {
    let piece = s.pieces[f]
    return Mesh(vertices: piece.map { s.vertices[$0] },
                triangles: (1..<(piece.count - 1)).map { SIMD3(0, Int32($0), Int32($0 + 1)) })
}

/// Whether a point of the solid's surface, lying in the pieces `own`, is
/// seen along `toEye`: one of its own pieces faces the eye and no other
/// piece is hit by the ray from it. The brute force every occlusion test is held to.
func rayVisible(_ s: Solid, at p: SIMD3<Double>, own: Set<Int>, toEye d: SIMD3<Double>) -> Bool {
    guard own.contains(where: { simd_dot(s.normals[$0], d) > 0 }) else { return false }
    for f in s.pieces.indices where !own.contains(f) {
        if firstHit(pieceMesh(s, f), from: p, along: d, after: 1e-9) != nil { return false }
    }
    return true
}

/// Point-by-point agreement of an edge's drawn spans with the ray cast:
/// disagreements, comparisons, and an example of the first disagreement.
func occlusionAgreement(_ s: Solid, edge e: Solid.Edge, spans: [(Double, Double)],
                        toEye d: SIMD3<Double>, camera: String) -> (Int, Int, String) {
    let a = s.vertices[e.a].v, b = s.vertices[e.b].v
    var wrong = 0, compared = 0
    var example = ""
    for k in 0..<200 {
        let t = (Double(k) + 0.5) / 200
        // Within a hair of a span's end the truth is at a boundary;
        // elsewhere it is decided.
        if spans.contains(where: { abs(t - $0.0) < 1e-6 || abs(t - $0.1) < 1e-6 }) { continue }
        let p = a + t * (b - a)
        let seenByOcclusion = spans.contains { t > $0.0 && t < $0.1 }
        let truth = rayVisible(s, at: p, own: [e.left, e.right], toEye: d)
        compared += 1
        if truth != seenByOcclusion {
            wrong += 1
            if example.isEmpty {
                example = String(format: "e.g. %@ edge %d-%d t=%.3f truth %@", camera, e.a, e.b, t,
                                 truth ? "seen" : "hidden")
                    + " " + rayReport(s, at: p, own: [e.left, e.right], toEye: d)
                    + " spans " + spans.map { String(format: "[%.4f, %.4f]", $0.0, $0.1) }.joined(separator: " ")
            }
        }
    }
    return (wrong, compared, example)
}

/// Why the ray cast says what it says: the facing of the own pieces and
/// every other piece hit, with its parameter.
func rayReport(_ s: Solid, at p: SIMD3<Double>, own: Set<Int>, toEye d: SIMD3<Double>) -> String {
    let facing = own.sorted().map { String(format: "f%d·d=%.3f", $0, simd_dot(s.normals[$0], d)) }
    var hits: [String] = []
    for f in s.pieces.indices where !own.contains(f) {
        if let t = firstHit(pieceMesh(s, f), from: p, along: d, after: 1e-9) {
            hits.append(String(format: "f%d@%.4f", f, t))
        }
    }
    return String(format: "p=(%.4f,%.4f,%.4f) d=(%.3f,%.3f,%.3f) ", p.x, p.y, p.z, d.x, d.y, d.z)
        + facing.joined(separator: " ") + " hits " + hits.joined(separator: " ")
}
