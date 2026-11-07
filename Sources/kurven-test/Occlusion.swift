import Foundation
import simd
import KurvenCore

// MARK: - the occlusion tests
//
// Each kind of solid's occlusion test, asked as a scene asks it, through
// `OcclusionTest`, and held to a brute-force cast of the line of sight
// against the same solid: a polyhedron's pieces as triangles, a landscape's
// drawn triangles, the round torus's tube walked in small steps. Where a
// nudge across the line of sight changes the cast's answer, the line grazes
// the solid and rounding decides; those points are left out.

func occlusionTests() {
    Check.suite("occlusion: each kind of solid's test is a cast of the line of sight") {
        // Two directions across the line of sight, for the nudges.
        func across(_ d: SIMD3<Double>) -> [SIMD3<Double>] {
            let a = simd_normalize(simd_cross(d, abs(d.z) < 0.9 ? SIMD3(0, 0, 1) : SIMD3(1, 0, 0)))
            let b = simd_cross(d, a)
            return [a, -a, b, -b]
        }

        // A polyhedron: points on every piece, which its own face's pieces
        // stand in front of by nothing, and points anywhere in its box.
        for (name, s) in [("cube", Solid.platonic(.cube)),
                          ("frame", Solid.frame(outer: 2, inner: 1, depth: 0.6)),
                          ("Császár", Solid.csaszar)] {
            let meshes = s.pieces.indices.map { pieceMesh(s, $0) }
            let scale = s.vertices.map { simd_length($0.v) }.max()!
            let lo = s.vertices.map(\.v).reduce(SIMD3(repeating: .infinity), simd_min)
            let hi = s.vertices.map(\.v).reduce(SIMD3(repeating: -.infinity), simd_max)
            let rng = SplitMix(seed: 0x0CC1)
            // (point, the pieces of the face it lies on, two directions in that face)
            var points: [(p: SIMD3<Double>, own: Set<Int>, plane: [SIMD3<Double>])] = []
            for f in s.pieces.indices {
                let piece = s.pieces[f].map { s.vertices[$0].v }
                let face = s.faceIndex(ofPiece: f)
                let own = Set(s.pieces.indices.filter { s.faceIndex(ofPiece: $0) == face })
                let u = simd_normalize(piece[1] - piece[0]), v = simd_normalize(simd_cross(s.normals[f], u))
                for _ in 0..<12 {
                    let k = 1 + Int(rng.next(0, Double(piece.count - 2) - 1e-9))
                    var a = rng.next(0, 1), b = rng.next(0, 1)
                    if a + b > 1 { (a, b) = (1 - a, 1 - b) }
                    points.append((piece[0] + a * (piece[k] - piece[0]) + b * (piece[k + 1] - piece[0]), own, [u, -u, v, -v]))
                }
            }
            for _ in 0..<200 {
                points.append((SIMD3(rng.next(lo.x, hi.x), rng.next(lo.y, hi.y), rng.next(lo.z, hi.z)), [], []))
            }
            var compared = 0, wrong = 0, undecided = 0, seen = 0
            var example = ""
            for (camera, view) in stressCameras() {
                let test = SceneGeometry.solid(s).occlusionTest(view: view, depthTolerance: 0)!
                let d = -view.projectionDirection
                for (p, own, plane) in points {
                    func cast(_ q: SIMD3<Double>) -> Bool {
                        meshes.indices.allSatisfy { own.contains($0) || firstHit(meshes[$0], from: q, along: d, after: 1e-9 * scale) == nil }
                    }
                    // On a face seen edge-on the line of sight runs in it,
                    // along whatever it meets there.
                    if let f = own.first, abs(simd_dot(s.normals[f], d)) < 1e-6 { undecided += 1; continue }
                    let truth = cast(p)
                    guard (own.isEmpty ? across(d) : plane).allSatisfy({ cast(p + 1e-7 * scale * $0) == truth }) else {
                        undecided += 1; continue
                    }
                    compared += 1
                    if truth { seen += 1 }
                    if test.isClear(P3(p)) != truth {
                        wrong += 1
                        if example.isEmpty {
                            example = String(format: " e.g. %@ at (%.3f, %.3f, %.3f): the cast says %@", camera, p.x, p.y, p.z, truth ? "clear" : "hidden")
                        }
                    }
                }
            }
            Check.expect(wrong == 0 && compared > 5000 && seen > compared / 10 && seen < compared,
                         "\(name): the point form of occlusion by the pieces is the cast, on the faces and off them",
                         "\(wrong) of \(compared) disagree, \(seen) clear, \(undecided) grazing\(example)")
        }

        // A landscape: points over its drawn surface, with no depth tolerance,
        // which outside the solid is the plain question of whether the line
        // of sight meets the triangles.
        let field = randomHeightfield(cells: 4, seed: 0xC0FFEE, truncation: .uniform(2.2))
        let mesh = drawnTriangles(field, base: -1)
        let d = field.surface.domain
        let rng = SplitMix(seed: 0x0CC2)
        var points: [SIMD3<Double>] = []
        while points.count < 400 {
            let p = P2<WorldSpace>(rng.next(d.real.lo, d.real.hi), rng.next(d.imag.lo, d.imag.hi))
            let floor = field.drawnHeight(at: p) + 0.01
            points.append(SIMD3(p.x, p.y, rng.next(floor, max(floor, 3.5))))
        }
        var compared = 0, wrong = 0, undecided = 0, seen = 0
        for (_, view) in stressCameras() {
            let test = SceneGeometry.heightfield(field).occlusionTest(view: view, depthTolerance: 0)!
            let toward = -view.projectionDirection
            for p in points {
                func cast(_ q: SIMD3<Double>) -> Bool { firstHit(mesh, from: q, along: toward, after: 1e-10) == nil }
                let truth = cast(p)
                guard across(toward).allSatisfy({ cast(p + 1e-7 * $0) == truth }) else { undecided += 1; continue }
                compared += 1
                if truth { seen += 1 }
                if test.isClear(P3(p)) != truth { wrong += 1 }
            }
        }
        Check.expect(wrong == 0 && compared > 10_000 && seen > compared / 10 && seen < compared,
                     "a landscape's grid traversal is the cast against its drawn triangles",
                     "\(wrong) of \(compared) disagree, \(seen) clear, \(undecided) grazing")

        // The round torus: points on it facing the eye and in its ball outside
        // it, the line of sight walked a ten-thousandth of the tube at a time.
        let (R, r) = (2.0, 1.0)
        let torus = ParametricSurface.torus(major: R, minor: r, samples: (64, 32))
        func clearance(_ q: SIMD3<Double>) -> Double {
            let rho = (q.x * q.x + q.y * q.y).squareRoot()
            return ((rho - R) * (rho - R) + q.z * q.z).squareRoot() - r
        }
        var torusPoints: [SIMD3<Double>] = []
        let trng = SplitMix(seed: 0x0CC3)
        while torusPoints.count < 300 {
            let q = SIMD3(trng.next(-3.2, 3.2), trng.next(-3.2, 3.2), trng.next(-1.2, 1.2))
            if clearance(q) > 0.01 { torusPoints.append(q) }
        }
        compared = 0; wrong = 0; undecided = 0; seen = 0
        for view in [testCamera(elevation: -60, azimuth: 30), testCamera(elevation: -20, azimuth: 70),
                     testCamera(elevation: -35, azimuth: -20, shear: 0.5)] {
            let test = SceneGeometry.parametric(torus).occlusionTest(view: view, depthTolerance: 0)!
            let toward = -view.projectionDirection
            var onSurface: [SIMD3<Double>] = []
            while onSurface.count < 200 {
                let c = P2<ParamSpace>(trng.next(0, 2 * .pi), trng.next(0, 2 * .pi))
                if let f = torus.facing(c, sight: view.projectionDirection), f > 0 { onSurface.append(torus.map(c).position) }
            }
            for q in torusPoints + onSurface {
                var t = 1e-5 * r, least = Double.infinity, clear = true
                while t < 10 {
                    let gap = clearance(q + t * toward)
                    least = min(least, gap)
                    if gap < 0 { clear = false; break }
                    t += 1e-4 * r
                }
                guard abs(least) > 1e-6 * r else { undecided += 1; continue }
                compared += 1
                if clear { seen += 1 }
                if test.isClear(P3(q)) != clear { wrong += 1 }
            }
        }
        Check.expect(wrong == 0 && compared > 1000 && seen > compared / 10 && seen < compared,
                     "the round torus's sphere tracing is the walk up the line of sight",
                     "\(wrong) of \(compared) disagree, \(seen) clear, \(undecided) grazing")

        // A surface with no tubular neighbourhood has no occlusion test: the
        // coordinate image stands in for one.
        let bare = ParametricSurface(u: torus.u, v: torus.v, encloses: true, map: torus.map)
        Check.expect(SceneGeometry.parametric(bare).occlusionTest(view: testCamera(elevation: -60, azimuth: 30),
                                                                   depthTolerance: 0) == nil,
                     "a parametric surface with no tubular neighbourhood has none")
    }
}
