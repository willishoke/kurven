import Foundation
import simd
import KurvenCore
import KurvenMetal
import KurvenBake
import KurvenLandscape
import KurvenMath
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

/// The real roots of a polynomial in `[a, b]`, coefficients highest first.
///
/// The critical points -- the roots of the derivative, found the same way --
/// split the interval into pieces on which the polynomial is monotone, and a
/// monotone piece holds a root exactly when its ends differ in sign; bisection
/// finds it. No tolerance decides whether a root exists, only where it is.
func realRoots(_ c: [Double], in a: Double, _ b: Double) -> [Double] {
    let n = c.count - 1
    func p(_ x: Double) -> Double { c.reduce(0) { $0 * x + $1 } }
    guard n >= 1 else { return [] }
    if n == 1 {
        let x = -c[1] / c[0]
        return x >= a && x <= b ? [x] : []
    }
    let derivative = (0..<n).map { c[$0] * Double(n - $0) }
    let cuts = [a] + realRoots(derivative, in: a, b) + [b]
    var roots: [Double] = []
    for (lo, hi) in zip(cuts, cuts.dropFirst()) {
        var l = lo, h = hi
        let pl = p(l), ph = p(h)
        if pl == 0 { roots.append(l); continue }
        guard (pl > 0) != (ph > 0) || ph == 0 else { continue }
        for _ in 0..<200 where h - l > 0 {
            let m = 0.5 * (l + h)
            if (p(m) > 0) == (pl > 0) { l = m } else { h = m }
        }
        roots.append(0.5 * (l + h))
    }
    return roots
}

/// Whether a point *off* the torus, just outside it, is seen from `eye`: the
/// full quartic along the sight line, which has no root at the point itself.
/// This is the question a point on a fold needs asked, since there the sight
/// line is tangent and the cubic of `torusSees` has a double root at zero.
func torusSeesFromOutside(_ p: SIMD3<Double>, major R: Double, minor r: Double,
                          eye e: SIMD3<Double>) -> Bool {
    let B = 2 * simd_dot(p, e)
    let C = simd_length_squared(p) + R * R - r * r
    let a2 = e.x * e.x + e.y * e.y
    let b2 = 2 * (p.x * e.x + p.y * e.y)
    let c2 = p.x * p.x + p.y * p.y
    let quartic = [1, 2 * B, B * B + 2 * C - 4 * R * R * a2,
                   2 * B * C - 4 * R * R * b2, C * C - 4 * R * R * c2]
    precondition(quartic[4] > 0, "the point is not outside the torus")
    return realRoots(quartic, in: 0, 2.5 * (R + r) + simd_length(p)).isEmpty
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

/// The visible length of ink on the torus `(R, r)`, by the exact ray-cast:
/// every segment divided `k` ways, each piece kept when its midpoint is seen.
func rayCastInkLength(_ ink: PolylineSet<WorldSpace>, on surface: ParametricSurface,
                      major R: Double, minor r: Double,
                      view: Transform<WorldSpace, ViewSpace>, k: Int = 8) -> Double {
    let eye = -view.sightLine
    let coords = ink.coords!
    var total = 0.0
    for path in 0..<ink.count {
        for i in ink.offsets[path]..<(ink.offsets[path + 1] - 1) {
            let a = coords[i].v, b = coords[i + 1].v
            for j in 0..<k {
                let at = { (t: Double) in surface.map(P2(a + (b - a) * t)).position }
                let t0 = Double(j) / Double(k), t1 = Double(j + 1) / Double(k)
                guard torusSees(at(0.5 * (t0 + t1)), major: R, minor: r, eye: eye) else { continue }
                let p = view(P3(at(t0))), q = view(P3(at(t1)))
                total += simd_length(SIMD2(q.x - p.x, q.y - p.y))
            }
        }
    }
    return total
}

func surfaceBakeTests() {
    Check.suite("bake: a parametric scene bakes its ink by where it lies on the surface") {
        let renderer = try MetalRenderer()
        let torus = ParametricSurface.torus(major: 2, minor: 1, samples: (512, 256))
        let ink = torus.parameterLines(counts: (24, 12), resolution: 2048)
        let spec = LayerSpec(name: "lines", role: .scaffold, source: .parameterLines(u: 24, v: 12),
                             width: 0.3, heightPolicy: .surface)
        // A plate camera, shear and all, as a real plate would have it.
        let camera = Camera.plate(PlateProjection(shear: 0.5, xAngle: -50, zAngle: 25,
                                                  flipX: false, yScale: nil))
        let scene = Scene(surface: torus, layers: [Layer(spec: spec, paths: ink)],
                          camera: camera, margin: 0.03)
        let whole = try renderer.bake(scene, options: BakeOptions(resolution: 2400, tiles: 1))
        let truth = rayCastInkLength(ink, on: torus, major: 2, minor: 1, view: camera.view)
        let error = abs(whole.strokes.inkLength - truth) / truth
        Check.expect(whole.surface != nil && error < 0.002,
                     "the baked plate carries the ray-cast's visible length",
                     String(format: "%.4f vs %.4f, %.3f%% off", whole.strokes.inkLength, truth,
                            100 * error))

        // Tiling splits the render and never the clip, and depth and
        // coordinates are stitched by the same code over the same sub-frames.
        let split = try renderer.bake(scene, options: BakeOptions(resolution: 2400, tiles: 3))
        let same = zip(whole.strokes.layers, split.strokes.layers).allSatisfy {
            $0.paths == $1.paths
        }
        let moved = zip(whole.surface!.coords, split.surface!.coords).filter { a, b in
            !(a.x.isNaN && b.x.isNaN) && (a.x.isNaN != b.x.isNaN || simd_length(a - b) > 1e-3)
        }.count
        Check.expect(same, "a tiled bake draws what a single-pass bake draws",
                     "\(moved) of \(whole.surface!.coords.count) pixel coordinates differ")

        // A winding is judged the same way, by the same coordinate image:
        // ink the surface passes never saw, placed from the lattice rather
        // than the map, still carries the ray-cast's visible length.
        let winding = torus.winding(slope: 2.0 / 5, turns: 5, count: 3)
        let wound = try renderer.bake(
            scene.drawing([Layer(spec: LayerSpec(name: "winding", role: .scaffold,
                                                 source: .winding(slope: 0.4, turns: 5, count: 3),
                                                 width: 0.3, heightPolicy: .surface),
                                 paths: winding)]),
            options: BakeOptions(resolution: 2400, tiles: 1))
        let windingTruth = rayCastInkLength(winding, on: torus, major: 2, minor: 1,
                                            view: camera.view)
        let windingError = abs(wound.strokes.inkLength - windingTruth) / windingTruth
        Check.expect(windingError < 0.002, "a winding bakes to the ray-cast's visible length too",
                     String(format: "%.4f vs %.4f, %.3f%% off", wound.strokes.inkLength,
                            windingTruth, 100 * windingError))

        // The same lines without coordinates are judged by depth, at the
        // scene's margin -- still a drawing, and a measurably worse one.
        let bare = Layer(spec: spec, paths: PolylineSet(vertices: ink.vertices, offsets: ink.offsets))
        let byDepth = try renderer.bake(scene.drawing([bare]),
                                        options: BakeOptions(resolution: 2400, tiles: 1))
        let depthError = abs(byDepth.strokes.inkLength - truth) / truth
        Check.expect(byDepth.strokes.pathCount > 0 && depthError > error,
                     "ink without coordinates falls back to the depth test",
                     String(format: "%.3f%% off by depth, %.3f%% by coordinate",
                            100 * depthError, 100 * error))
    }
}

func foldTests() {
    Check.suite("surfaces: fold lines lie on the folds, and are seen where the ray-cast sees them") {
        let renderer = try MetalRenderer()
        for (R, r) in [(2.0, 1.0), (1.25, 1.0)] {
            let torus = ParametricSurface.torus(major: R, minor: r, samples: (512, 256))
            for (name, xAngle, shear) in [("elevation 50°", -50.0, 0.0),
                                          ("sheared", -30.0, 0.5)] {
                let camera = Camera.plate(PlateProjection(shear: shear, xAngle: xAngle, zAngle: 25,
                                                          flipX: false, yScale: nil))
                let sight = camera.view.sightLine
                let folds = torus.foldLines(sight: sight)
                var worst = 0.0
                for c in folds.coords! {
                    let n = torus.map(c).normal
                    worst = max(worst, abs(simd_dot(n, sight)) / simd_length(n))
                }
                let label = "R/r = \(R / r), \(name)"
                Check.expect(folds.count > 0 && worst < 1e-9,
                             "\(label): every vertex is edge-on", "max |cos| \(worst)")

                let spec = LayerSpec(name: "folds", role: .outline, source: .foldLines,
                                     width: 0.6, heightPolicy: .surface)
                let scene = Scene(surface: torus, layers: [Layer(spec: spec, paths: .empty)],
                                  camera: camera, margin: 0)
                let baked = try renderer.bake(scene, options: BakeOptions(resolution: 2400))
                // On a fold the sight line is tangent to the torus, so the
                // ray-cast is asked a hair outside it, where it has an answer.
                let eye = -sight
                var truth = 0.0
                let coords = folds.coords!
                for path in 0..<folds.count {
                    for i in folds.offsets[path]..<(folds.offsets[path + 1] - 1) {
                        let mid = torus.map(P2(0.5 * (coords[i].v + coords[i + 1].v)))
                        let out = simd_normalize(mid.normal) * (torus.outward ?? 1)
                        guard torusSeesFromOutside(mid.position + 1e-7 * (R + r) * out,
                                                   major: R, minor: r, eye: eye) else { continue }
                        let a = camera.view(folds.vertices[i]), b = camera.view(folds.vertices[i + 1])
                        truth += simd_length(SIMD2(b.x - a.x, b.y - a.y))
                    }
                }
                // What the front-most test cannot see is a fold running on
                // behind its own sheet past a cusp: there the sheet in front
                // is the fold's own neighbourhood for the first couple of
                // pixels. So the bound is a few pixels per cusp, not zero.
                let px = baked.depth.frame.unitsPerPixel.max()
                let excess = baked.strokes.inkLength - truth
                Check.expect(abs(excess) < 16 * px,
                             "\(label): the visible fold length is the ray-cast's, to a few pixels",
                             String(format: "%.4f vs %.4f, %+.1f px", baked.strokes.inkLength,
                                    truth, excess / px))
            }
        }
    }
}


func periodicTorusTests() {
    Check.suite("math: the quarter periods are the complete elliptic integral") {
        Check.expect(abs(Jacobi.quarterPeriod(0) - .pi / 2) < 1e-15, "K(0) = π/2")
        // K(1/2) = Γ(1/4)² / (4√π), to the last digit.
        Check.expect(abs(Jacobi.quarterPeriod(0.5) - 1.854_074_677_301_371_9) < 4e-16,
                     "K(1/2) = Γ(1/4)² / 4√π", "\(Jacobi.quarterPeriod(0.5))")
        let m = 0.64
        let K = Jacobi.quarterPeriod(m), Kp = Jacobi.quarterPeriod(1 - m)
        Check.expect(abs(Jacobi.real(K, m).sn - 1) < 1e-14, "sn(K) = 1")
        let rng = SplitMix(seed: 11)
        var worst = 0.0
        for _ in 0..<200 {
            let z = Complex(rng.next(-3, 3), rng.next(-1, 1))
            let s = Jacobi.complex(z, m).sn
            for shifted in [Complex(z.re + 4 * K, z.im), Complex(z.re, z.im + 2 * Kp)] {
                let t = Jacobi.complex(shifted, m).sn
                worst = max(worst, (t - s).magnitude / max(1, s.magnitude))
            }
        }
        Check.expect(worst < 1e-11, "sn has periods 4K and 2iK'", "worst \(worst)")
    }

    Check.suite("surfaces: a doubly periodic function drawn on its torus") {
        let m = 0.64
        let plate = try PeriodicTorus.jacobiSN(modulus: m)
        var worst = 0.0, off = 0.0, vertices = 0, singular = 0
        for layer in plate.layers {
            guard case .contour(let field, let levels, _, _) = layer.spec.source else { continue }
            let coords = layer.paths.coords!
            for (k, c) in coords.enumerated() {
                vertices += 1
                let f = Jacobi.complex(Complex(c.x, c.y), m).sn
                // Every phase contour ends at a zero or a pole, where arg f
                // has no value; its last vertex there is on every level.
                if field == .phase, f.magnitude < 1e-4 || f.magnitude > 1e4 {
                    singular += 1; continue
                }
                let value = field == .magnitude ? f.magnitude : f.argument
                let level = levels.min { abs($0 - value) < abs($1 - value) }!
                let scale = field == .magnitude ? max(level, 1) : 1
                worst = max(worst, abs(value - level) / scale)
                off = max(off, simd_length(layer.paths.vertices[k].v
                                           - plate.surface.map(c).position))
            }
        }
        Check.expect(vertices > 10_000 && worst < 1e-6,
                     "every contour vertex is on its level of sn",
                     "\(vertices) vertices, worst \(worst);"
                     + " \(singular) phase ends at a zero or pole not asked")
        Check.expect(off < 1e-12, "and sits on the torus where its coordinate says", "\(off)")

        let renderer = try MetalRenderer()
        let camera = Camera.plate(PlateProjection(shear: 0.5, xAngle: -50, zAngle: 25,
                                                  flipX: false, yScale: nil))
        let contours = plate.layers.filter { if case .contour = $0.spec.source { true } else { false } }
        let baked = try renderer.bake(Scene(surface: plate.surface, layers: contours,
                                            camera: camera, margin: 0),
                                      options: BakeOptions(resolution: 2400))
        let truth = contours.reduce(0.0) {
            $0 + rayCastInkLength($1.paths, on: plate.surface, major: 2, minor: 1,
                                  view: camera.view, k: 4)
        }
        let error = abs(baked.strokes.inkLength - truth) / truth
        Check.expect(error < 0.003, "and it is seen where the ray-cast sees it",
                     String(format: "%.4f vs %.4f, %.3f%% off", baked.strokes.inkLength, truth,
                            100 * error))
    }
}

func surfacePreviewTests() {
    // The preview asks the bake's two questions per fragment rather than per
    // vertex, so the two may differ within a pixel of wherever visibility
    // changes and nowhere else. Held to that: the baked strokes, rasterized
    // into the preview's own pixels, against the preview's ink.
    Check.suite("preview: a parametric surface draws what its bake draws") {
        let renderer = try MetalRenderer()
        let torus = ParametricSurface.torus(major: 2, minor: 1, samples: (512, 256))
        let lines = Layer(spec: LayerSpec(name: "lines", role: .scaffold,
                                          source: .parameterLines(u: 24, v: 12),
                                          width: 0.3, heightPolicy: .surface),
                          paths: torus.parameterLines(counts: (24, 12), resolution: 2048))
        let folds = Layer(spec: LayerSpec(name: "folds", role: .outline, source: .foldLines,
                                          width: 0.3, heightPolicy: .surface),
                          paths: .empty)
        let style = PlateStyle(shear: 0.5, flipX: false, yScale: nil)
        let orbit = Orbit(target: P3(0, 0, 0), azimuth: Angle(degrees: 25),
                          elevation: Angle(degrees: -50), style: style)
        let viewport = Viewport(width: 900, height: 700)
        let scene = Scene(surface: torus, layers: [lines, folds], camera: orbit.camera, margin: 0)
        guard let bounds = scene.quickBounds() else {
            Check.expect(false, "the torus has bounds"); return
        }
        let navigator = Navigator(orbit: orbit, framing: .fitting(bounds, in: viewport))
        let target = try renderer.makePreviewTarget(viewport)
        try renderer.renderPreview(scene, navigator: navigator, viewport: viewport,
                                   options: PreviewOptions(slopeScale: 0), into: target)
        // Any visible mark, as `compare_preview.py` counts ink.
        func inkMask(_ bgra: [UInt8]) -> [Bool] {
            (0..<(bgra.count / 4)).map { k -> Bool in
                let r = 299 * Int(bgra[4 * k + 2]), g = 587 * Int(bgra[4 * k + 1])
                let b = 114 * Int(bgra[4 * k])
                return r + g + b < 224_000
            }
        }
        let preview = inkMask(try PNG.bgra(target))

        let baked = try renderer.bake(scene.looking(navigator.camera),
                                      options: BakeOptions(resolution: 3000))
        let frame = navigator.framing.frame(viewport)
        var bake = [Bool](repeating: false, count: preview.count)
        for layer in baked.strokes.layers {
            for path in layer.paths.paths {
                for (a, b) in zip(path, path.dropFirst()) {
                    let steps = max(Int(simd_length(b.v - a.v) / frame.unitsPerPixel.max() * 4), 1)
                    for j in 0...steps {
                        let t = Double(j) / Double(steps)
                        let i = frame.index(of: P2<ViewSpace>(a.x + (b.x - a.x) * t,
                                                              a.y + (b.y - a.y) * t))
                        guard i.x >= 0, i.x < viewport.height, i.y >= 0, i.y < viewport.width
                        else { continue }
                        bake[i.x * viewport.width + i.y] = true
                    }
                }
            }
        }
        func near(_ mask: [Bool], _ k: Int, _ r: Int) -> Bool {
            let row = k / viewport.width, col = k % viewport.width
            for dr in -r...r { for dc in -r...r {
                let rr = row + dr, cc = col + dc
                if rr >= 0, rr < viewport.height, cc >= 0, cc < viewport.width,
                   mask[rr * viewport.width + cc] { return true }
            } }
            return false
        }
        let baked1 = bake.indices.filter { bake[$0] }
        let shown = baked1.filter { near(preview, $0, 1) }.count
        let inked = preview.indices.filter { preview[$0] }
        let backed = inked.filter { near(bake, $0, 2) }.count
        let recall = Double(shown) / Double(max(baked1.count, 1))
        let precision = Double(backed) / Double(max(inked.count, 1))
        Check.expect(baked1.count > 5000 && recall > 0.99,
                     "the ink the bake keeps, the preview draws",
                     String(format: "%.2f%% of %d baked pixels", 100 * recall, baked1.count))
        Check.expect(precision > 0.99, "and the ink the preview draws, the bake keeps",
                     String(format: "%.2f%% of %d preview pixels", 100 * precision, inked.count))

        // The control: the same lines drawn without a visibility test. If the
        // measure could not tell this from the real thing it would prove
        // nothing.
        var open = lines.spec; open.clipped = false
        try renderer.renderPreview(scene.drawing([Layer(spec: open, paths: lines.paths), folds]),
                                   navigator: navigator, viewport: viewport,
                                   options: PreviewOptions(slopeScale: 0), into: target)
        let unclipped = inkMask(try PNG.bgra(target))
        let all = unclipped.indices.filter { unclipped[$0] }
        let stray = Double(all.filter { !near(bake, $0, 2) }.count) / Double(max(all.count, 1))
        Check.expect(stray > 0.2, "while the same lines drawn unclipped visibly are not",
                     String(format: "%.1f%% of %d pixels have no baked stroke near", 100 * stray,
                            all.count))
    }
}

func windingTests() {
    Check.suite("surfaces: a winding is a straight line on the flat torus, placed on the lattice") {
        let torus = ParametricSurface.torus(major: 2, minor: 1, samples: (512, 256))
        let turn = 2 * Double.pi

        // The interpolator returns the lattice at its own points, and is
        // within a lattice cell's sag of the map between them.
        var exact = true, sag = 0.0
        for j in stride(from: 0, to: 256, by: 17) {
            for i in stride(from: 0, to: 512, by: 13) {
                let c = P2<ParamSpace>(torus.u.coordinate(i), torus.v.coordinate(j))
                exact = exact
                    && simd_length(torus.interpolate(c).v - torus.position(i, j).v) < 1e-12
                let mid = P2<ParamSpace>(c.x + 0.5 * torus.u.spacing + 7 * turn,
                                         c.y + 0.5 * torus.v.spacing - 3 * turn)
                sag = max(sag, simd_length(torus.interpolate(mid).v - torus.map(mid).position))
            }
        }
        Check.expect(exact, "the lattice interpolates to itself at its own points")
        Check.expect(sag < 1e-3 && sag > 0, "and to the map between them, whole periods away",
                     String(format: "%.2e off", sag))

        // A rational slope closes after its denominator's turns, and stops
        // there: one closed (p, q) curve, straight in coordinates.
        let closed = torus.winding(slope: 2.0 / 5, turns: 24)
        let steps = 512
        Check.expect(closed.count == 1 && closed.vertices.count == 5 * steps + 1,
                     "slope 2/5 closes after five turns, a vertex per lattice cell",
                     "\(closed.count) paths, \(closed.vertices.count) vertices")
        let ends = simd_length(closed.vertices[0].v - closed.vertices[closed.vertices.count - 1].v)
        Check.expect(ends < 1e-9, "and its last vertex is its first",
                     String(format: "%.1e apart", ends))
        let coords = closed.coords!
        let straight = coords.allSatisfy { abs($0.y - 0.4 * $0.x) < 1e-9 }
        let rises = coords[coords.count - 1].x - coords[0].x
        Check.expect(straight && abs(rises - 5 * turn) < 1e-9,
                     "its coordinates run straight, v = 2/5 u, over five turns of u")
        let placed = zip(closed.vertices, coords)
            .map { simd_length($0.v - torus.map($1).position) }.max()!
        Check.expect(placed < 1e-3, "every vertex lies on the drawn surface at its coordinate",
                     String(format: "%.2e from the map", placed))

        // An irrational slope never closes: all the turns asked for, broken
        // into paths of at most eight turns that share their seam vertex,
        // coordinates continuous across it by whole periods.
        let open = torus.winding(slope: 2.0.squareRoot() - 1, turns: 20)
        let oc = open.coords!
        var seams = true
        for k in 1..<open.count {
            let end = open.offsets[k] - 1, start = open.offsets[k]
            let du = (oc[end].x - oc[start].x) / turn, dv = (oc[end].y - oc[start].y) / turn
            seams = seams && open.vertices[end] == open.vertices[start]
                && abs(du - du.rounded()) < 1e-9 && abs(dv - dv.rounded()) < 1e-9
        }
        let spans = (0..<open.count).map { k in
            (oc[open.offsets[k + 1] - 1].x - oc[open.offsets[k]].x) / turn
        }
        let far = simd_length(open.vertices[0].v - open.vertices[open.vertices.count - 1].v)
        Check.expect(open.count == 3 && spans.allSatisfy { $0 <= 8 + 1e-9 }
                     && abs(spans.reduce(0, +) - 20) < 1e-9,
                     "√2 − 1 runs its twenty turns in three paths of at most eight",
                     "\(open.count) paths spanning \(spans.map { String(format: "%.2f", $0) })")
        Check.expect(seams && far > 0.1, "which share their seam vertex and never close",
                     String(format: "ends %.3f apart", far))

        // Strands start evenly spaced along v; slope zero is the v-lines.
        let strands = torus.winding(slope: 0, turns: 3, count: 4)
        let sc = strands.coords!
        let starts = (0..<strands.count).map { sc[strands.offsets[$0]].y }
        Check.expect(strands.count == 4 && starts == [0, 0.25, 0.5, 0.75].map { $0 * turn }
                     && sc.allSatisfy { $0.x <= turn + 1e-9 },
                     "four strands of slope zero are four lines of constant v, one turn each")

        // The steeper axis sets the vertex count: slope 4 walks the tube
        // four times per turn and gets a vertex per v-cell of that.
        let steep = torus.winding(slope: 4, turns: 1)
        Check.expect(steep.vertices.count == 4 * 256 + 1,
                     "slope four is sampled once per v-cell", "\(steep.vertices.count) vertices")

        let source = LayerSource.winding(slope: 0.4, turns: 24, count: 3)
        Check.expect((try? LayerSource(json: source.json)) == source,
                     "a winding layer source round-trips through the manifest")
    }
}
