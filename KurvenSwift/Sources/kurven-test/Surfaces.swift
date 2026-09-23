import Foundation
import simd
import KurvenCore
import KurvenMetal
import Metal

// MARK: - parametric surfaces and visibility by surface coordinate
//
// Nothing here has a Python oracle, and nothing needs one: the torus of
// revolution has exact visibility. A point is hidden when the ray from it
// toward the eye meets the torus again, and along that ray the torus's
// implicit equation is a quartic with a known root at the point itself --
// so a cubic, whose real roots are bracketed by its critical points.

/// Whether a point of the torus of revolution `(R, r)` about z is seen from
/// direction `eye` (unit, pointing from the point toward the viewer).
///
/// Along `p + s e` the torus is `F = (|x|² + R² − r²)² − 4R²(x² + y²) = 0`,
/// a monic quartic in `s` whose constant term vanishes because `p` is on the
/// torus. Dividing by `s` leaves a cubic `G`; the point is hidden exactly when
/// `G` has a root at some `s > 0` inside the torus's bounding sphere.
func torusSees(_ p: SIMD3<Double>, major R: Double, minor r: Double,
               eye e: SIMD3<Double>) -> Bool {
    let B = 2 * simd_dot(p, e)
    let C = simd_length_squared(p) + R * R - r * r
    let a2 = e.x * e.x + e.y * e.y
    let b2 = 2 * (p.x * e.x + p.y * e.y)
    let k2 = B * B + 2 * C - 4 * R * R * a2
    let k1 = 2 * B * C - 4 * R * R * b2
    func G(_ s: Double) -> Double { ((s + 2 * B) * s + k2) * s + k1 }
    let eps = 1e-9 * (R + r), far = 2.5 * (R + r)
    // G' = 3s² + 4Bs + k2: its real roots split [eps, far] into monotone
    // pieces, and a monotone piece holds a root exactly when its ends differ
    // in sign.
    var cuts = [eps]
    let disc = 16 * B * B - 12 * k2
    if disc > 0 {
        for s in [(-4 * B - disc.squareRoot()) / 6, (-4 * B + disc.squareRoot()) / 6]
        where s > eps && s < far { cuts.append(s) }
    }
    cuts.append(far)
    cuts.sort()
    for (a, b) in zip(cuts, cuts.dropFirst()) {
        let ga = G(a), gb = G(b)
        if ga == 0 || gb == 0 || (ga > 0) != (gb > 0) { return false }
    }
    return true
}

/// Elevation and azimuth as the plate cameras have them, with an optional
/// oblique shear in front, so the view is affine but not orthonormal.
func testCamera(elevation: Double, azimuth: Double,
                shear: Double = 0) -> Transform<WorldSpace, ViewSpace> {
    let s = simd_double3x3(rows: [SIMD3(1, shear, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1)])
    let m = rotationX(Angle(degrees: elevation)) * rotationZ(Angle(degrees: azimuth)) * s
    let t = m.transpose
    return Transform(rows: (t.columns.0, t.columns.1, t.columns.2))
}

/// The view-space bounds of a surface's lattice.
func viewBounds(_ s: ParametricSurface,
                _ view: Transform<WorldSpace, ViewSpace>) -> AABB<ViewSpace> {
    AABB(s.positions.map { view($0) })!
}

/// Hold a visibility test to the exact ray-cast at 20000 seeded points of the
/// torus `(R, r)`.
///
/// A disagreement is excused only within two pixels of a place where the
/// truth itself changes -- a fold, or the edge of something in front -- found
/// by asking the ray-cast at the eight neighbours two pixels away in each
/// parameter direction. The depth test at a 1% margin is scored on the same
/// points, for comparison.
func compareWithRayCast(_ vis: SurfaceVisibility, major R: Double, minor r: Double,
                        view: Transform<WorldSpace, ViewSpace>)
    -> (wrong: Int, unexplained: Int, depthWrong: Int, example: String)
{
    let surface = vis.surface, image = vis.image, frame = image.frame
    let eye = -vis.sight
    let px = frame.unitsPerPixel.max()
    let rng = SplitMix(seed: 0x5EED)
    var wrong = 0, unexplained = 0, depthWrong = 0
    var example = ""
    for _ in 0..<20_000 {
        let c = P2<ParamSpace>(rng.next(0, 2 * .pi), rng.next(0, 2 * .pi))
        let jet = surface.map(c)
        let q = view(P3(jet.position))
        let truth = torusSees(jet.position, major: R, minor: r, eye: eye)
        if (q.z + 0.01 * (R + r) > image.depth.depth(under: q.xy)) != truth { depthWrong += 1 }
        guard vis.isVisible(q, at: c) != truth else { continue }
        wrong += 1
        let speed = { (d: SIMD3<Double>) -> Double in
            let a = view(P3(jet.position)), b = view(P3(jet.position + d))
            return simd_length(SIMD2(b.x - a.x, b.y - a.y))
        }
        let hu = min(2 * px / max(speed(jet.du), 1e-12), 0.3)
        let hv = min(2 * px / max(speed(jet.dv), 1e-12), 0.3)
        var near = false
        for (su, sv) in [(1.0, 0.0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)] {
            let p = surface.map(P2(c.x + su * hu, c.y + sv * hv)).position
            if torusSees(p, major: R, minor: r, eye: eye) != truth { near = true; break }
        }
        if !near {
            unexplained += 1
            if example.isEmpty {
                example = String(format: "e.g. (%.4f, %.4f) truth %@", c.x, c.y,
                                 truth ? "seen" : "hidden")
            }
        }
    }
    return (wrong, unexplained, depthWrong, example)
}

func surfaceTests() {
    Check.suite("surfaces: ink can carry where on a surface it lies") {
        let a: [P3<WorldSpace>] = [P3(0, 0, 0), P3(1, 0, 0), P3(2, 0, 0)]
        let ca: [P2<ParamSpace>] = [P2(0, 0), P2(0.5, 0), P2(1, 0)]
        let set = PolylineSet(paths: [a, [P3(9, 9, 9)]], coords: [ca, [P2(9, 9)]])
        Check.expect(set.count == 1 && set.coords?.count == 3,
                     "a singleton path drops its coordinates with it")
        let moved = set.mapped(Transform<WorldSpace, ViewSpace>.identity)
        Check.expect(moved.coords == set.coords, "a camera moves the ink and not its coordinates")
        Check.expect(PolylineSet(paths: [a]).coords == nil,
                     "ink without coordinates says so rather than inventing them")
    }

    Check.suite("surfaces: which side is outside is computed, not guessed") {
        let t = ParametricSurface.torus(major: 2, minor: 1, samples: (64, 32))
        let out = t.map(P2(0, 0)).normal * (t.outward ?? 0)
        Check.expect(t.outward != nil && out.x > 0,
                     "the torus's outward normal at its outer equator points away from the axis",
                     "outward \(t.outward.map { "\($0)" } ?? "nil")")
        // The same torus with the tube angle run backwards: the parametrization's
        // normal flips, and the outward one must not.
        let flipped = ParametricSurface(u: .angle(samples: 64), v: .angle(samples: 32),
                                        encloses: true) { c in
            var j = t.map(P2(c.x, -c.y)); j.dv = -j.dv; return j
        }
        let outFlipped = flipped.map(P2(0, 0)).normal * (flipped.outward ?? 0)
        Check.expect(flipped.outward == -(t.outward ?? 0) && outFlipped.x > 0,
                     "reversing the parametrization reverses the sign and keeps the outside")
        let horn = ParametricSurface.torus(major: 1, minor: 1, samples: (64, 32))
        Check.expect(horn.outward == nil,
                     "a horn torus passes through itself, so it bounds nothing and culls nothing")
    }

    Check.suite("surfaces: the sight line is the camera's kernel, not the gradient of depth") {
        let view = testCamera(elevation: -55, azimuth: 30, shear: 0.5)
        let d = view.sightLine
        let p = P3<WorldSpace>(0.3, -0.7, 0.2)
        let a = view(p), b = view(P3(target: p, plus: d))
        Check.expect(simd_length(SIMD2(b.x - a.x, b.y - a.y)) < 1e-12 && b.z < a.z,
                     "travelling it leaves view x and y alone and moves away from the eye",
                     "Δxy \(simd_length(SIMD2(b.x - a.x, b.y - a.y))), Δz \(b.z - a.z)")
        let c = view.m.columns
        let gradient = -simd_normalize(SIMD3(c.0.z, c.1.z, c.2.z))
        let angle = acos(min(1, simd_dot(gradient, d))) * 180 / .pi
        Check.expect(angle > 1, "under a shear the two differ, which is why it matters",
                     String(format: "%.1f°", angle))
    }

    let torus = ParametricSurface.torus(major: 2, minor: 1, samples: (384, 192))

    Check.suite("surfaces: the rasterized coordinates land where their depth and pixel say") {
        let view = testCamera(elevation: -60, azimuth: 30)
        let frame = DepthFrame(covering: viewBounds(torus, view), resolution: 512)
        let image = SurfaceImage.rasterize(torus, view: view, frame: frame)
        var dz = 0.0, dxy = 0.0, covered = 0
        for r in 0..<frame.rows {
            for c in 0..<frame.cols {
                guard let q = image.coordinate(row: r, col: c) else { continue }
                covered += 1
                let p = view(P3(torus.map(q).position))
                let s = frame.coordinate(row: r, col: c)
                dz = max(dz, abs(p.z - image.depth[r, c]))
                dxy = max(dxy, simd_length(SIMD2(p.x - s.x, p.y - s.y)))
            }
        }
        // The lattice is a chord approximation of the torus: its sagitta at
        // 384 x 192 is (R + r)(1 − cos(π/384)) ≈ 1e-4 around the axis.
        Check.expect(covered > frame.rows * frame.cols / 5, "the torus covers the frame",
                     "\(covered) pixels")
        Check.expect(dz < 1e-3, "a pixel's coordinate maps back to its depth", "max \(dz)")
        Check.expect(dxy < 1e-3, "and to its own sample point", "max \(dxy)")
    }

    Check.suite("surfaces: visibility by coordinate agrees with the exact ray-cast") {
        // Two tori -- one with room in the hole, one fat enough that most of
        // it is hidden behind itself -- under cameras from nearly overhead to
        // nearly edge-on, one of them sheared as the plates are.
        let cameras: [(String, Transform<WorldSpace, ViewSpace>)] = [
            ("elevation 60°", testCamera(elevation: -60, azimuth: 30)),
            ("nearly overhead", testCamera(elevation: -85, azimuth: 10)),
            ("nearly edge-on", testCamera(elevation: -10, azimuth: 40)),
            ("sheared", testCamera(elevation: -35, azimuth: -20, shear: 0.5)),
        ]
        let tori = [(2.0, 1.0), (1.25, 1.0)]
        for (R, r) in tori {
            let surface = ParametricSurface.torus(major: R, minor: r, samples: (384, 192))
            for (name, view) in cameras {
                let frame = DepthFrame(covering: viewBounds(surface, view), resolution: 800)
                let image = SurfaceImage.rasterize(surface, view: view, frame: frame)
                let vis = SurfaceVisibility(surface: surface, image: image, view: view)
                let (wrong, unexplained, depthWrong, example) =
                    compareWithRayCast(vis, major: R, minor: r, view: view)
                let n = 20_000
                let label = "R/r = \(R / r), \(name)"
                Check.expect(unexplained == 0,
                             "\(label): every disagreement is within two pixels of a visibility edge",
                             "\(wrong) of \(n) disagree, \(unexplained) unexplained \(example)"
                             + " (depth test: \(depthWrong))")
            }
        }
    }

    Check.suite("surfaces: a segment across a fold is cut on the fold") {
        let view = testCamera(elevation: -35, azimuth: -20, shear: 0.5)
        let frame = DepthFrame(covering: viewBounds(torus, view), resolution: 256)
        let vis = SurfaceVisibility(surface: torus,
                                    image: SurfaceImage.rasterize(torus, view: view, frame: frame),
                                    view: view)
        let rng = SplitMix(seed: 7)
        var found = 0, worst = 0.0, straddles = true
        while found < 200 {
            let a = P2<ParamSpace>(rng.next(0, 2 * .pi),
                                   rng.next(0, 2 * .pi))
            let b = P2<ParamSpace>(a.x + rng.next(-0.2, 0.2),
                                   a.y + rng.next(-0.2, 0.2))
            guard let t = vis.foldCrossing(from: a, to: b) else { continue }
            found += 1
            let at = { (s: Double) in P2<ParamSpace>(a.v + (b.v - a.v) * s) }
            worst = max(worst, abs(vis.facing(at(t)) ?? 1))
            let before = vis.facing(at(max(t - 1e-9, 0)))!, after = vis.facing(at(min(t + 1e-9, 1)))!
            if (before > 0) == (after > 0) { straddles = false }
        }
        Check.expect(worst < 1e-10, "the cut is where the surface is edge-on",
                     "max |cos| \(worst) over \(found) segments")
        Check.expect(straddles, "and it faces the eye on exactly one side of the cut")
    }

    Check.suite("surfaces: clipped circles on a torus have the ray-cast's visible length") {
        for (name, view) in [("elevation 60°", testCamera(elevation: -60, azimuth: 30)),
                             ("sheared", testCamera(elevation: -35, azimuth: -20, shear: 0.5))] {
            let frame = DepthFrame(covering: viewBounds(torus, view), resolution: 1000)
            let vis = SurfaceVisibility(surface: torus,
                                        image: SurfaceImage.rasterize(torus, view: view, frame: frame),
                                        view: view)
            // Twenty-four circles each way round, 1500 vertices apiece.
            var paths: [[P3<ViewSpace>]] = [], coords: [[P2<ParamSpace>]] = []
            let k = 1500
            for line in 0..<24 {
                let fixed = 2 * Double.pi * (Double(line) + 0.5) / 24
                for around in [true, false] {
                    var p: [P3<ViewSpace>] = [], c: [P2<ParamSpace>] = []
                    for i in 0...k {
                        let s = 2 * Double.pi * Double(i) / Double(k)
                        let q = around ? P2<ParamSpace>(s, fixed) : P2<ParamSpace>(fixed, s)
                        c.append(q); p.append(view(P3(torus.map(q).position)))
                    }
                    paths.append(p); coords.append(c)
                }
            }
            let clipped = HiddenLine.clip(PolylineSet(paths: paths, coords: coords), on: vis)
            // The truth, by the midpoint of every segment of a far finer
            // division of the same circles.
            var truth = 0.0
            let fine = 20_000
            for line in 0..<24 {
                let fixed = 2 * Double.pi * (Double(line) + 0.5) / 24
                for around in [true, false] {
                    func at(_ s: Double) -> P2<ParamSpace> {
                        around ? P2(s, fixed) : P2(fixed, s)
                    }
                    for i in 0..<fine {
                        let s0 = 2 * Double.pi * Double(i) / Double(fine)
                        let s1 = 2 * Double.pi * Double(i + 1) / Double(fine)
                        let mid = torus.map(at(0.5 * (s0 + s1))).position
                        guard torusSees(mid, major: 2, minor: 1, eye: -vis.sight) else { continue }
                        let a = view(P3(torus.map(at(s0)).position))
                        let b = view(P3(torus.map(at(s1)).position))
                        truth += simd_length(SIMD2(b.x - a.x, b.y - a.y))
                    }
                }
            }
            let error = abs(clipped.inkLength - truth) / truth
            Check.expect(error < 0.002, "\(name): visible ink length matches the ray-cast",
                         String(format: "%.4f vs %.4f, %.3f%% off", clipped.inkLength, truth,
                                100 * error))
        }
    }
}

func surfaceGPUTests() {
    // The GPU pass is held to the CPU rasterizer as the depth pass is held to
    // the Python Z-buffer: the same triangles, sampled at the same lattice
    // points, and differing only where a rule of rasterization is allowed to
    // -- a sample exactly on an edge, which the CPU claims for both triangles
    // and Metal for one.
    Check.suite("metal: the GPU coordinate pass is the CPU rasterizer's") {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Check.expect(false, "a Metal device exists"); return
        }
        let renderer = try MetalRenderer(device: device)
        for (R, r) in [(2.0, 1.0), (1.25, 1.0)] {
            let surface = ParametricSurface.torus(major: R, minor: r, samples: (384, 192))
            for (name, view) in [("elevation 60°", testCamera(elevation: -60, azimuth: 30)),
                                 ("sheared", testCamera(elevation: -35, azimuth: -20, shear: 0.5))] {
                let label = "R/r = \(R / r), \(name)"
                let frame = DepthFrame(covering: viewBounds(surface, view), resolution: 800)
                let gpu = try renderer.renderSurface(surface, view: view, frame: frame)
                let cpu = SurfaceImage.rasterize(surface, view: view, frame: frame)

                // Coverage and depth: any disagreement sits on an edge -- of
                // the drawing, or of a sheet in front of another -- so some
                // neighbouring pixel of the CPU image is empty or far away.
                let span = (viewBounds(surface, view).size.z)
                var covered = 0, strays = 0, onEdge = 0, worstDepth = 0.0
                for row in 0..<frame.rows {
                    for col in 0..<frame.cols {
                        let a = gpu.depth.isCovered(row: row, col: col)
                        let b = cpu.depth.isCovered(row: row, col: col)
                        if a || b { covered += 1 }
                        let differs = a != b
                            || (a && abs(gpu.depth[row, col] - cpu.depth[row, col]) > 1e-4 * span)
                        if a && b && !differs {
                            worstDepth = max(worstDepth, abs(gpu.depth[row, col] - cpu.depth[row, col]))
                        }
                        guard differs else { continue }
                        strays += 1
                        var edge = false
                        for dr in -1...1 { for dc in -1...1 {
                            let rr = row + dr, cc = col + dc
                            guard rr >= 0, rr < frame.rows, cc >= 0, cc < frame.cols else {
                                edge = true; continue
                            }
                            if !cpu.depth.isCovered(row: rr, col: cc)
                                || abs(cpu.depth[rr, cc] - cpu.depth[row, col]) > 0.05 * span {
                                edge = true
                            }
                        } }
                        if edge { onEdge += 1 }
                    }
                }
                Check.expect(strays == onEdge && strays * 200 < covered,
                             "\(label): coverage and depth differ only on edges",
                             "\(strays) of \(covered) pixels differ, \(strays - onEdge) off an edge;"
                             + " elsewhere depth within \(worstDepth)")

                // The coordinate at every pixel names a point of the surface
                // at that pixel's depth: it is the front-most point, which is
                // the whole of its job.
                var worst = 0.0
                for row in 0..<frame.rows {
                    for col in 0..<frame.cols {
                        guard let c = gpu.coordinate(row: row, col: col) else { continue }
                        let p = view(P3(surface.map(c).position))
                        worst = max(worst, abs(p.z - gpu.depth[row, col]))
                    }
                }
                Check.expect(worst < 2e-3, "\(label): each coordinate is the front-most point",
                             "worst depth mismatch \(worst)")

                let vis = SurfaceVisibility(surface: surface, image: gpu, view: view)
                let (wrong, unexplained, depthWrong, example) =
                    compareWithRayCast(vis, major: R, minor: r, view: view)
                Check.expect(unexplained == 0,
                             "\(label): and visibility read from it agrees with the ray-cast",
                             "\(wrong) of 20000 disagree, \(unexplained) unexplained \(example)"
                             + " (depth test: \(depthWrong))")
            }
        }
    }
}
