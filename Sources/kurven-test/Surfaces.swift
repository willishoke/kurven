import Foundation
import simd
import KurvenCore
import KurvenMetal
import KurvenBake
import KurvenDynamics
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
/// full quartic along the line of sight, which has no root at the point itself.
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

/// The view-space bounds of a surface's grid.
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
/// parameter direction. The depth test at a 1% depth tolerance is scored on the same
/// points, for comparison.
func compareWithRayCast(_ vis: SurfaceVisibility, major R: Double, minor r: Double,
                        view: Transform<WorldSpace, ViewSpace>)
    -> (wrong: Int, unexplained: Int, depthWrong: Int, example: String)
{
    let surface = vis.surface, image = vis.image!, frame = image.frame
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

    Check.suite("surfaces: the line of sight is the camera's kernel, not the gradient of depth") {
        let view = testCamera(elevation: -55, azimuth: 30, shear: 0.5)
        let d = view.projectionDirection
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
        // The grid is a chord approximation of the torus: its sagitta at
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
                // The round torus bounds a solid sphere tracing can ask directly,
                // and then there is no pixel to be within.
                Check.expect(vis.tracesSpheres && wrong == 0,
                             "\(label): by sphere tracing, no disagreement at all",
                             "\(wrong) of \(n) disagree")
            }
        }
    }

    Check.suite("surfaces: sphere tracing on a displaced torus agrees with a fine walk up the line of sight") {
        // The torus in relief under a smooth bump field, truncated at 1: the
        // solid sphere tracing is held to is the tubular neighbourhood's own point test,
        // walked up the line of sight in steps of a ten-thousandth of the
        // solid, which is as fine as the oracle gets without being the
        // sphere tracing. A disagreement is excused where the walk's least gap is
        // within rounding of zero: a line grazing the surface, which no
        // test with a tolerance decides the same way twice.
        let R = 2.0, r = 1.0
        let base = ParametricSurface.torus(major: R, minor: r, samples: (256, 128))
        let relief = base.displaced(by: { c in min(0.5 * (1 + cos(3 * c.x) * cos(2 * c.y)) + 0.3, 1) },
                                    amplitude: 0.4, ceiling: 1, encloses: true)
        Check.expect(relief.neighbourhood != nil, "the displaced torus keeps the torus's foot")
        for (name, view) in [("elevation 60°", testCamera(elevation: -60, azimuth: 30)),
                             ("low, over the far wall", testCamera(elevation: -80, azimuth: 40)),
                             ("sheared", testCamera(elevation: -35, azimuth: -20, shear: 0.5))] {
            let frame = DepthFrame(covering: viewBounds(relief, view), resolution: 64)
            let vis = SurfaceVisibility(surface: relief,
                                        image: SurfaceImage.rasterize(relief, view: view, frame: frame),
                                        view: view)
            let e = relief.neighbourhood!
            let d = -view.projectionDirection
            let rng = SplitMix(seed: 0xC0FFEE)
            var wrong = 0, unexplained = 0, tested = 0, seen = 0
            var example = ""
            while tested < 1500 {
                let c = P2<ParamSpace>(rng.next(0, 2 * .pi), rng.next(0, 2 * .pi))
                guard let f = vis.facing(c), f > 0 else { continue }
                tested += 1
                let q = relief.map(c).position
                let traversal = vis.isClear(q)
                if traversal { seen += 1 }
                // The walk: every sample from just off the surface to past
                // the ball, and the least gap it saw.
                let step = 1e-4 * e.radius
                var t = 1e-5 * e.radius, least = Double.infinity, clear = true
                while t < 2 * e.radius + simd_length(q) {
                    let (coord, distance) = e.foot(q + t * d)
                    let gap = distance - e.normalOffset(coord)
                    least = min(least, gap)
                    if gap < -1e-9 * e.radius { clear = false; break }
                    t += step
                }
                guard traversal != clear else { continue }
                wrong += 1
                if abs(least) > 1e-6 * e.radius {
                    unexplained += 1
                    if example.isEmpty {
                        example = String(format: "e.g. (%.4f, %.4f) sphere tracing %@, walk %@, least gap %.2e",
                                         c.x, c.y, traversal ? "clear" : "hidden", clear ? "clear" : "hidden", least)
                    }
                }
            }
            Check.expect(seen > tested / 4 && seen < tested,
                         "\(name): the sample has both seen and hidden points", "\(seen) of \(tested) seen")
            Check.expect(unexplained == 0,
                         "\(name): sphere tracing and the walk agree except where the line grazes",
                         "\(wrong) of \(tested) disagree, \(unexplained) unexplained \(example)")
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
    // the Python Z-buffer: the same triangles, sampled at the same grid
    // points, and differing only where a rule of rasterization is allowed to
    // -- a sample exactly on an edge, which the CPU claims for both triangles
    // and Metal for one.
    Check.suite("metal: the GPU coordinate pass is the CPU rasterizer's") {
        guard let device = Check.gpu() else { return }
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
    let eye = -view.projectionDirection
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
        let ink = torus.coordinateCurves(counts: (24, 12), resolution: 2048)
        let spec = LayerSpec(name: "lines", role: .scaffold, source: .coordinateCurves(u: 24, v: 12),
                             width: 0.3, heightPolicy: .surface)
        // A plate camera, shear and all, as a real plate would have it.
        let camera = Camera.plate(PlateProjection(shear: 0.5, xAngle: -50, zAngle: 25,
                                                  flipX: false, yScale: nil))
        let scene = Scene(surface: torus, layers: [Layer(spec: spec, paths: ink)],
                          camera: camera, depthTolerance: 0.03)
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

        // A linear curve is tested the same way, by the same coordinate image:
        // ink the surface passes never saw, placed from the grid rather
        // than the map, still carries the ray-cast's visible length.
        let linearCurves = torus.linearCurves(slope: 2.0 / 5, turns: 5, count: 3)
        let wound = try renderer.bake(
            scene.drawing([Layer(spec: LayerSpec(name: "linearCurves", role: .scaffold,
                                                 source: .linearCurves(slope: 0.4, turns: 5, count: 3),
                                                 width: 0.3, heightPolicy: .surface),
                                 paths: linearCurves)]),
            options: BakeOptions(resolution: 2400, tiles: 1))
        let linearCurveTruth = rayCastInkLength(linearCurves, on: torus, major: 2, minor: 1,
                                            view: camera.view)
        let linearCurveError = abs(wound.strokes.inkLength - linearCurveTruth) / linearCurveTruth
        Check.expect(linearCurveError < 0.002, "a linear curve bakes to the ray-cast's visible length too",
                     String(format: "%.4f vs %.4f, %.3f%% off", wound.strokes.inkLength,
                            linearCurveTruth, 100 * linearCurveError))

        // The same lines without coordinates are tested by depth, at the
        // scene's depth tolerance -- still a drawing, and a measurably worse one.
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
    Check.suite("surfaces: fold curves lie on the folds, and are seen where the ray-cast sees them") {
        let renderer = try MetalRenderer()
        for (R, r) in [(2.0, 1.0), (1.25, 1.0)] {
            let torus = ParametricSurface.torus(major: R, minor: r, samples: (512, 256))
            for (name, xAngle, shear) in [("elevation 50°", -50.0, 0.0),
                                          ("sheared", -30.0, 0.5)] {
                let camera = Camera.plate(PlateProjection(shear: shear, xAngle: xAngle, zAngle: 25,
                                                          flipX: false, yScale: nil))
                let sight = camera.view.projectionDirection
                let folds = torus.foldCurves(sight: sight)
                var worst = 0.0
                for c in folds.coords! {
                    let n = torus.map(c).normal
                    worst = max(worst, abs(simd_dot(n, sight)) / simd_length(n))
                }
                let label = "R/r = \(R / r), \(name)"
                Check.expect(folds.count > 0 && worst < 1e-9,
                             "\(label): every vertex is edge-on", "max |cos| \(worst)")

                let spec = LayerSpec(name: "folds", role: .outline, source: .foldCurves,
                                     width: 0.6, heightPolicy: .surface)
                let scene = Scene(surface: torus, layers: [Layer(spec: spec, paths: .empty)],
                                  camera: camera, depthTolerance: 0)
                let baked = try renderer.bake(scene, options: BakeOptions(resolution: 2400))
                // On a fold the line of sight is tangent to the torus, so the
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

func creaseTests() {
    Check.suite("surfaces: a truncation edge is a crease, drawn once and seen where either face is") {
        // sn in relief, truncated and floored, so that each pole is a chimney
        // whose flat top meets its wall in a crease: the plate the gaps
        // were found on, two periods each way.
        let base = try PeriodicTorus.jacobiSN(modulus: 0.64, resolution: 300, grid: 512,
                                              truncated: true, turn: 0.125)
            .repeated(real: 2, imag: 2, grid: 1024)
        let height = base.height!, floor = 1.5
        let plate = PeriodicTorus.Plate(surface: base.surface, layers: base.layers,
                                        height: { max(height($0) - floor, 0) },
                                        ceiling: base.ceiling - floor, radii: base.radii)
            .relief(amplitude: 0.45, reach: 1)
        let surface = plate.surface, ceiling = plate.ceiling
        let camera = Camera.plate(PlateProjection(shear: 0, xAngle: -55, zAngle: 30,
                                                  flipX: false, yScale: nil))
        let sight = camera.view.projectionDirection
        let h = { (c: P2<ParamSpace>) in max(height(c) - floor, 0) }
        let cell = min(surface.u.spacing, surface.v.spacing)
        // The four points a hair either way along each axis.
        func beside(_ c: P2<ParamSpace>, _ t: Double) -> [P2<ParamSpace>] {
            [P2(c.x + t, c.y), P2(c.x - t, c.y), P2(c.x, c.y + t), P2(c.x, c.y - t)]
        }

        // The facing changes sign across the truncation edge as well as through a fold,
        // and only the second is a fold: no piece of fold runs along the
        // plateau's edge, truncated one side and not the other at both its
        // ends. A fold that meets the truncation edge ends on it, below.
        let truncationEdgeLayer = plate.layers.first { $0.spec.name == "truncation edge" }!
        let folds = surface.foldCurves(sight: sight, creases: Scene(surface: surface, layers: [truncationEdgeLayer], camera: camera, depthTolerance: 0).creases)
        func onTruncationEdge(_ c: P2<ParamSpace>) -> Bool {
            let hs = beside(c, 1e-4 * cell).map(h)
            return hs.contains { $0 >= ceiling } && hs.contains { $0 < ceiling }
        }
        var along = 0
        for path in 0..<folds.count {
            let fc = folds.coords!
            for i in folds.offsets[path]..<(folds.offsets[path + 1] - 1) where onTruncationEdge(fc[i]) && onTruncationEdge(fc[i + 1]) {
                along += 1
            }
        }
        Check.expect(folds.count > 0 && along == 0, "no fold is drawn along a crease",
                     "\(along) of \(folds.coords!.count - folds.count) fold segments along a truncation edge")
        // Nor beside one: where the grid saw the sign change across a
        // crease and the map finds no zero within a cell, there is no fold
        // to move the vertex onto, and it is not drawn where it was.
        let worst = folds.coords!.map { abs(surface.facing($0, sight: sight) ?? 0) }.max() ?? 0
        Check.expect(worst < 1e-6, "every fold vertex on the relief is edge-on",
                     String(format: "max |cos| %.1e", worst))

        // Where a fold meets a truncation edge, it ends on the truncation edge. The wall or the truncation
        // turns edge-on between two truncation edge vertices where its facing a hair
        // off the truncation edge changes sign, and the fold starts there; on a steep
        // wall its whole visible length can lie within a grid cell.
        let truncationEdgeCoords = plate.layers.first { $0.spec.name == "truncation edge" }!.paths
        func side(_ c: P2<ParamSpace>, truncationHeight: Bool) -> Double {
            let e = 1e-3 * cell
            let g = SIMD2(h(P2(c.x + e, c.y)) - h(P2(c.x - e, c.y)), h(P2(c.x, c.y + e)) - h(P2(c.x, c.y - e)))
            let n = simd_length(g) > 0 ? g / simd_length(g) : SIMD2(1, 0)
            return surface.facing(P2(c.v + (truncationHeight ? 0.02 : -0.02) * cell * n), sight: sight) ?? 0
        }
        var farthest = 0.0, junctions = 0
        for path in 0..<truncationEdgeCoords.count {
            let rc = truncationEdgeCoords.coords!
            for i in truncationEdgeCoords.offsets[path]..<(truncationEdgeCoords.offsets[path + 1] - 1) {
                for truncationHeight in [false, true] {
                    let fa = side(rc[i], truncationHeight: truncationHeight), fb = side(rc[i + 1], truncationHeight: truncationHeight)
                    guard (fa > 0) != (fb > 0) else { continue }
                    let t = fa / (fa - fb)
                    let j = P2<ParamSpace>(rc[i].v + t * (rc[i + 1].v - rc[i].v))
                    let nearest = folds.coords!.map { simd_length($0.v - j.v) / cell }.min() ?? .infinity
                    farthest = max(farthest, nearest)
                    junctions += 1
                }
            }
        }
        Check.expect(junctions > 0 && farthest < 0.1, "a fold that meets a truncation edge ends on it",
                     String(format: "%d junctions, farthest fold end %.3f cells off", junctions, farthest))

        // And where it meets it in sight, the truncation edge's seen piece ends where the
        // fold's does: a cut in the truncation edge is bisected on the truncation edge, not on the
        // chord between two of its vertices, which crosses the truncation.
        let sphereTracer = SurfaceVisibility(sphereTracing: surface, view: camera.view)
        let seenTruncationEdge = HiddenLine.clip(truncationEdgeLayer.paths.mapped(camera.view), on: sphereTracer,
                                      onFolds: true, onCrease: true)
        let seenFolds = HiddenLine.clip(folds.mapped(camera.view), on: sphereTracer, onFolds: true)
        func ends(_ p: PolylineSet<PlateSpace>) -> [SIMD2<Double>] {
            (0..<p.count).flatMap { i in
                [p.vertices[p.offsets[i]], p.vertices[p.offsets[i + 1] - 1]].map { SIMD2($0.x, $0.y) }
            }
        }
        let foldEnds = ends(seenFolds)
        let meetings = ends(seenTruncationEdge).compactMap { e in foldEnds.map { simd_length($0 - e) }.min().flatMap { $0 < 0.01 ? $0 : nil } }
        Check.expect(!meetings.isEmpty && meetings.max()! < 0.001,
                     "where a fold meets the truncation edge in sight, the two seen pieces meet",
                     String(format: "%d meetings, widest %.5f", meetings.count, meetings.max() ?? .nan))

        // The truncation edge is a convex edge: seen where the truncation or the wall beside
        // it faces the eye and nothing stands in front. The normal on the
        // crease itself is half of each, and says nothing.
        let truncationEdge = plate.layers.first { $0.spec.name == "truncation edge" }!
        let sphereTracing = SurfaceVisibility(sphereTracing: surface, view: camera.view)
        // Asked at the truncation edge's vertices, which are on the crease: a chord's
        // midpoint is inside the plateau, where only the truncation is beside it.
        let coords = truncationEdge.paths.coords!
        let seen = coords.map { c in
            beside(c, 1e-3 * cell).contains { (surface.facing($0, sight: sight) ?? 0) > 0 }
                && sphereTracing.isClear(surface.map(c).position)
        }
        // A chord whose ends disagree counts half, and is the reference's
        // own uncertainty: the bake finds where along it the verdict turns.
        var truth = 0.0, slack = 0.0
        for path in 0..<truncationEdge.paths.count {
            for i in truncationEdge.paths.offsets[path]..<(truncationEdge.paths.offsets[path + 1] - 1) where seen[i] || seen[i + 1] {
                let a = camera.view(truncationEdge.paths.vertices[i]), b = camera.view(truncationEdge.paths.vertices[i + 1])
                let length = simd_length(SIMD2(b.x - a.x, b.y - a.y))
                if seen[i] && seen[i + 1] { truth += length } else { truth += 0.5 * length; slack += 0.5 * length }
            }
        }
        let scene = Scene(surface: surface, layers: [truncationEdge], camera: camera, depthTolerance: 0)
        let baked = try MetalRenderer().bake(scene, options: BakeOptions(resolution: 2000))
        let px = baked.depth.frame.unitsPerPixel.max()
        let excess = baked.strokes.inkLength - truth
        Check.expect(truth > 0 && abs(excess) < slack + 2 * px,
                     "the baked truncation edge is the convex edge's visible length, to its turning chords",
                     String(format: "%.4f vs %.4f, %+.1f px of %.1f", baked.strokes.inkLength, truth,
                            excess / px, slack / px + 2))
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
        var worstWhere = ""
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
                if abs(value - level) / scale > worst {
                    worstWhere = String(format: "%@ at (%.5f, %.5f) |f| %.3e value %.4f level %.4f", layer.spec.name as NSString, c.x, c.y, f.magnitude, value, level)
                }
                worst = max(worst, abs(value - level) / scale)
                off = max(off, simd_length(layer.paths.vertices[k].v
                                           - plate.surface.map(c).position))
            }
        }
        Check.expect(vertices > 10_000 && worst < 1e-6,
                     "every contour vertex is on its level of sn",
                     "\(vertices) vertices, worst \(worst) \(worstWhere);"
                     + " \(singular) phase ends at a zero or pole not asked")
        Check.expect(off < 1e-12, "and sits on the torus where its coordinate says", "\(off)")

        let renderer = try MetalRenderer()
        let camera = Camera.plate(PlateProjection(shear: 0.5, xAngle: -50, zAngle: 25,
                                                  flipX: false, yScale: nil))
        let contours = plate.layers.filter { if case .contour = $0.spec.source { true } else { false } }
        let baked = try renderer.bake(Scene(surface: plate.surface, layers: contours,
                                            camera: camera, depthTolerance: 0),
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

    // The torus has no seam; the rectangle it is drawn from has one, and sn
    // puts both its zeros and one pole on it. Every line is computed on the
    // plane and cut to the rectangle, so a line may end in only three places:
    // on a zero, on a pole, or on the rectangle's edge, where its
    // continuation begins at the same point of the torus.
    Check.suite("surfaces: every line on the sn torus ends on a zero, a pole, or its continuation across the seam") {
        let m = 0.64
        let plate = try PeriodicTorus.jacobiSN(modulus: m)
        let u = plate.surface.u.range, v = plate.surface.v.range
        let onEdge = 1e-7
        var loose: [String] = []
        var seam: [(text: String, at: SIMD3<Double>)] = []
        var zeros = 0, poles = 0
        for layer in plate.layers {
            guard case .contour = layer.spec.source, let coords = layer.paths.coords else { continue }
            let offsets = layer.paths.offsets
            for i in 0..<(offsets.count - 1) {
                let run = coords[offsets[i]..<offsets[i + 1]]
                guard let a = run.first, let b = run.last, a != b else { continue }
                for c in [a, b] {
                    let f = Jacobi.complex(Complex(c.x, c.y), m).sn
                    if f.magnitude < 1e-6 { zeros += 1; continue }
                    if !f.isFinite || f.magnitude > 1e6 { poles += 1; continue }
                    let text = String(format: "%@ (%.5f, %.5f) |f| %.3g",
                                      layer.spec.name as NSString, c.x, c.y, f.magnitude)
                    if abs(c.x - u.lo) < onEdge || abs(c.x - u.hi) < onEdge
                        || abs(c.y - v.lo) < onEdge || abs(c.y - v.hi) < onEdge {
                        seam.append((text, plate.surface.map(c).position))
                    } else {
                        loose.append(text)
                    }
                }
            }
        }
        var unmatched: [String] = []
        for (k, end) in seam.enumerated()
        where !seam.indices.contains(where: { $0 != k && simd_length(seam[$0].at - end.at) < 1e-8 }) {
            unmatched.append(end.text)
        }
        Check.expect(zeros > 0 && poles > 0, "phase lines end on the zeros and on the poles",
                     "\(zeros) ends on zeros, \(poles) on poles")
        Check.expect(loose.isEmpty, "no line ends in the open",
                     "\(loose.count) do: \(loose.prefix(6).joined(separator: "; "))")
        Check.expect(!seam.isEmpty && unmatched.isEmpty,
                     "every end on the seam meets its continuation on the torus, to 1e-8",
                     "\(seam.count) ends on the seam, \(unmatched.count) unmatched"
                     + (unmatched.isEmpty ? "" : ": " + unmatched.prefix(6).joined(separator: "; ")))
    }

    // sn² has periods 2K and 2iK', double zeros and double poles. At a
    // double point every level has two rays, which the grid cannot
    // resolve inside the point's cell: the fan is read off the normal form,
    // f(z) ≈ a (z − z₀)^±2, and its rays are what the grid's lines are
    // spliced onto. The rectangle is shifted so that one zero and one pole
    // are interior, where every ray of the fan ends on the point itself.
    Check.suite("surfaces: a double zero and a double pole each hold two rays of every level") {
        let m = 0.64
        let K = Jacobi.quarterPeriod(m), Kp = Jacobi.quarterPeriod(1 - m)
        let plate = try PeriodicTorus.plate("sn(z, \(m))^2", origin: Complex(-K / 2, -Kp / 2),
                                            periods: (2 * K, 2 * Kp))
        let u = plate.surface.u.range, v = plate.surface.v.range
        var atZero = 0, atPole = 0, levels = 0
        var loose: [String] = []
        var seam: [(text: String, at: SIMD3<Double>)] = []
        for layer in plate.layers {
            guard case .contour(let field, let list, _, _) = layer.spec.source, field == .phase,
                  let coords = layer.paths.coords else { continue }
            // The cut, ±π, is drawn once.
            levels += list.filter { $0 > -.pi + 1e-9 }.count
            let offsets = layer.paths.offsets
            for i in 0..<(offsets.count - 1) {
                let run = coords[offsets[i]..<offsets[i + 1]]
                guard let a = run.first, let b = run.last, a != b else { continue }
                for c in [a, b] {
                    if hypot(c.x, c.y) < 1e-6 { atZero += 1; continue }
                    if hypot(c.x, c.y - Kp) < 1e-6 { atPole += 1; continue }
                    let f = Jacobi.complex(Complex(c.x, c.y), m).sn
                    let f2 = f.magnitude * f.magnitude
                    if f2 < 1e-6 || !f.isFinite || f2 > 1e6 { continue }
                    let text = String(format: "%@ (%.5f, %.5f) |f| %.3g",
                                      layer.spec.name as NSString, c.x, c.y, f2)
                    if abs(c.x - u.lo) < 1e-7 || abs(c.x - u.hi) < 1e-7
                        || abs(c.y - v.lo) < 1e-7 || abs(c.y - v.hi) < 1e-7 {
                        seam.append((text, plate.surface.map(c).position))
                    } else {
                        loose.append(text)
                    }
                }
            }
        }
        let unmatched = seam.indices.filter { k in
            !seam.indices.contains { $0 != k && simd_length(seam[$0].at - seam[k].at) < 1e-8 }
        }
        Check.expect(atZero == 2 * levels && atPole == 2 * levels,
                     "two rays of each of the \(levels) levels end on the double zero and on the double pole",
                     "\(atZero) ends on the zero, \(atPole) on the pole")
        Check.expect(loose.isEmpty, "and no line ends in the open",
                     "\(loose.count) do: \(loose.prefix(6).joined(separator: "; "))")
        Check.expect(!seam.isEmpty && unmatched.isEmpty,
                     "and every end on the seam meets its continuation, to 1e-8",
                     "\(seam.count) ends on the seam, \(unmatched.count) unmatched")
    }
}

func surfacePreviewTests() {
    // At rest the preview draws a sphere-traced surface's ink as the bake tests
    // it. A draft asks the bake's two questions per fragment rather than per
    // vertex, so the two may differ within a pixel of wherever visibility
    // changes and nowhere else. Both held to that: the baked strokes,
    // rasterized into the preview's own pixels, against the preview's ink.
    Check.suite("preview: a parametric surface draws what its bake draws") {
        let renderer = try MetalRenderer()
        let torus = ParametricSurface.torus(major: 2, minor: 1, samples: (512, 256))
        let lines = Layer(spec: LayerSpec(name: "lines", role: .scaffold,
                                          source: .coordinateCurves(u: 24, v: 12),
                                          width: 0.3, heightPolicy: .surface),
                          paths: torus.coordinateCurves(counts: (24, 12), resolution: 2048))
        let folds = Layer(spec: LayerSpec(name: "folds", role: .outline, source: .foldCurves,
                                          width: 0.3, heightPolicy: .surface),
                          paths: .empty)
        let style = PlateStyle(shear: 0.5, flipX: false, yScale: nil)
        let orbit = Orbit(target: P3(0, 0, 0), azimuth: Angle(degrees: 25),
                          elevation: Angle(degrees: -50), style: style)
        let viewport = Viewport(width: 900, height: 700)
        let scene = Scene(surface: torus, layers: [lines, folds], camera: orbit.camera, depthTolerance: 0)
        guard let bounds = scene.quickBounds() else {
            Check.expect(false, "the torus has bounds"); return
        }
        let navigator = Navigator(orbit: orbit, framing: .fitting(bounds, in: viewport))
        let target = try renderer.makePreviewTarget(viewport)
        // Any visible mark counts as ink: a stroke narrower than a pixel is
        // drawn lighter, not narrower.
        func inkMask(_ bgra: [UInt8]) -> [Bool] {
            (0..<(bgra.count / 4)).map { k -> Bool in
                let r = 299 * Int(bgra[4 * k + 2]), g = 587 * Int(bgra[4 * k + 1])
                let b = 114 * Int(bgra[4 * k])
                return r + g + b < 224_000
            }
        }

        let baked = try renderer.bake(scene.looking(navigator.camera),
                                      options: BakeOptions(resolution: 3000))
        let frame = navigator.framing.frame(viewport)
        var bake = [Bool](repeating: false, count: viewport.width * viewport.height)
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
        for draft in [false, true] {
            let when = draft ? "a draft" : "at rest"
            try renderer.renderPreview(scene, navigator: navigator, viewport: viewport,
                                       options: PreviewOptions(slopeScale: 0, draft: draft),
                                       into: target)
            let preview = inkMask(try PNG.bgra(target))
            let baked1 = bake.indices.filter { bake[$0] }
            let shown = baked1.filter { near(preview, $0, 1) }.count
            let inked = preview.indices.filter { preview[$0] }
            let backed = inked.filter { near(bake, $0, 2) }.count
            let recall = Double(shown) / Double(max(baked1.count, 1))
            let precision = Double(backed) / Double(max(inked.count, 1))
            Check.expect(baked1.count > 5000 && recall > 0.99,
                         "\(when): the ink the bake keeps, the preview draws",
                         String(format: "%.2f%% of %d baked pixels", 100 * recall, baked1.count))
            Check.expect(precision > 0.99, "\(when): and the ink the preview draws, the bake keeps",
                         String(format: "%.2f%% of %d preview pixels", 100 * precision, inked.count))
        }

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

func linearCurveTests() {
    Check.suite("surfaces: a linear curve is a straight line on the flat torus, placed on the grid") {
        let torus = ParametricSurface.torus(major: 2, minor: 1, samples: (512, 256))
        let turn = 2 * Double.pi

        // The interpolator returns the grid at its own points, and is
        // within a grid cell's sag of the map between them.
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
        Check.expect(exact, "the grid interpolates to itself at its own points")
        Check.expect(sag < 1e-3 && sag > 0, "and to the map between them, whole periods away",
                     String(format: "%.2e off", sag))

        // Its normals interpolate to within a cell's turn of the map's:
        // on this grid the tube turns 360°/256 a cell, so under a degree.
        let normals = torus.gridNormals()
        var tilt = 0.0
        for j in stride(from: 0, to: 256, by: 5) {
            for i in stride(from: 0, to: 512, by: 7) {
                let c = P2<ParamSpace>(torus.u.coordinate(i) + 0.3 * torus.u.spacing - 2 * turn,
                                       torus.v.coordinate(j) + 0.7 * torus.v.spacing + 5 * turn)
                let n = SIMD3<Double>(torus.interpolateNormal(c, normals: normals))
                let m = simd_normalize(torus.map(c).normal)
                tilt = max(tilt, acos(min(1, simd_dot(n, m))) * 180 / .pi)
            }
        }
        Check.expect(tilt < 1, "and its normals to within a degree of the map's, whole periods away",
                     String(format: "worst tilt %.3f°", tilt))

        // A rational slope closes after its denominator's turns, and stops
        // there: one closed (p, q) curve, straight in coordinates.
        let closed = torus.linearCurves(slope: 2.0 / 5, turns: 24)
        let steps = 512
        Check.expect(closed.count == 1 && closed.vertices.count == 5 * steps + 1,
                     "slope 2/5 closes after five turns, a vertex per grid cell",
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
        let open = torus.linearCurves(slope: 2.0.squareRoot() - 1, turns: 20)
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
        let strands = torus.linearCurves(slope: 0, turns: 3, count: 4)
        let sc = strands.coords!
        let starts = (0..<strands.count).map { sc[strands.offsets[$0]].y }
        Check.expect(strands.count == 4 && starts == [0, 0.25, 0.5, 0.75].map { $0 * turn }
                     && sc.allSatisfy { $0.x <= turn + 1e-9 },
                     "four strands of slope zero are four lines of constant v, one turn each")

        // The steeper axis sets the vertex count: slope 4 walks the tube
        // four times per turn and gets a vertex per v-cell of that.
        let steep = torus.linearCurves(slope: 4, turns: 1)
        Check.expect(steep.vertices.count == 4 * 256 + 1,
                     "slope four is sampled once per v-cell", "\(steep.vertices.count) vertices")

        let source = LayerSource.linearCurves(slope: 0.4, turns: 24, count: 3)
        Check.expect((try? LayerSource(json: source.json)) == source,
                     "a linear curve layer source round-trips through the manifest")
    }
}

func edgeOnPreviewTests() {
    // sn on its torus seen from half a degree off its equatorial plane, the
    // window's camera when this was found: the outline's long edges run
    // along the top and bottom of the tube, where the front-most coordinate
    // at a pixel flips between the near sheet and the far one. Visible per
    // fragment against the coordinate image, those edges drew as dashes.
    // At rest the window draws what sphere tracing keeps, as the bake does, and
    // the draft is held to showing the difference, or this proves nothing.
    Check.suite("preview: a torus nearly edge-on keeps its outline whole at rest") {
        let plate = try SurfacePlate.build(
            SurfaceRequest(.sn(modulus: 0.64, major: 2, minor: 1, grid: 200), grid: 600))
        var projection = SurfaceRequest.projection
        projection.xAngle = -89.5
        projection.zAngle = 118.2
        let scene = plate.scene(camera: Camera.plate(projection))
        let viewport = Viewport(width: 1400, height: 900)
        guard let bounds = scene.quickBounds() else {
            Check.expect(false, "the torus has bounds"); return
        }
        let navigator = Navigator(orbit: Orbit(matching: projection),
                                  framing: .fitting(bounds, in: viewport))
        let renderer = try MetalRenderer()
        let target = try renderer.makePreviewTarget(viewport)
        let frame = navigator.framing.frame(viewport)

        // The folds alone: the contours meet the outline all along it, and
        // ink beside a gap would fill it.
        guard let folds = scene.layers.firstIndex(where: {
            if case .foldCurves = $0.spec.source { true } else { false }
        }) else {
            Check.expect(false, "the plate draws its folds"); return
        }
        let baked = try renderer.bake(scene.looking(navigator.camera),
                                      options: BakeOptions(resolution: 3000))
        var bake = [Bool](repeating: false, count: viewport.width * viewport.height)
        do {
            let layer = baked.strokes.layers[folds]
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
        let baked1 = bake.indices.filter { bake[$0] }
        func recall(draft: Bool) throws -> Double {
            try renderer.renderPreview(scene, navigator: navigator, viewport: viewport,
                                       options: PreviewOptions(visibleLayers: [folds], slopeScale: 0,
                                                               draft: draft),
                                       into: target)
            let bgra = try PNG.bgra(target)
            func inked(_ row: Int, _ col: Int) -> Bool {
                guard row >= 0, row < viewport.height, col >= 0, col < viewport.width else { return false }
                let k = row * viewport.width + col
                return 299 * Int(bgra[4 * k + 2]) + 587 * Int(bgra[4 * k + 1])
                    + 114 * Int(bgra[4 * k]) < 224_000
            }
            let shown = baked1.filter { k in
                let row = k / viewport.width, col = k % viewport.width
                return (-1...1).contains { dr in (-1...1).contains { dc in inked(row + dr, col + dc) } }
            }.count
            return Double(shown) / Double(max(baked1.count, 1))
        }
        let rest = try recall(draft: false), draft = try recall(draft: true)
        Check.expect(baked1.count > 2000 && rest > 0.999,
                     "at rest the preview draws all of the outline the bake keeps",
                     String(format: "%.3f%% of %d baked pixels", 100 * rest, baked1.count))
        Check.expect(draft < rest - 0.002,
                     "while the draft, tested per fragment, breaks it",
                     String(format: "the draft draws %.3f%%", 100 * draft))
    }
}
