import Foundation
import simd
import KurvenCore
import KurvenMetal
import KurvenBake
import KurvenLandscape

// MARK: - the depth pass against an exact ray cast

/// What the depth buffer records at a pixel is the nearest point of the drawn
/// solid along that pixel's sight line: the heightfield's triangles, capped,
/// and the wall curtains that close it at the domain's edge. All of that is
/// one solid with a predicate -- a world point is inside when it lies over the
/// domain, above the walls' base and under the capped surface -- and a sight
/// line's first point inside it is the depth, to whatever precision one cares
/// to bisect. Casting that ray through every pixel is the reference the Metal
/// pass is held to here, with no second rasterizer anywhere in the loop.
///
/// The surface is interpolated the way the pass draws it: each cell as two
/// triangles split from (1,0) to (0,1), and the rim pieces making the result
/// min(interpolated, cap) exactly (Caps.swift holds them to that). What is
/// left to differ is float32 rounding, the side of a triangle edge a pixel
/// centre rounds to, and a sliver thinner than the march step. The
/// tolerances are the parametric surfaces' (Surfaces.swift).
struct HeightfieldSolid {
    let grid: Grid2D<Float>
    let caps: Caps
    let base: Double
    let origin: SIMD2<Double>
    let step: SIMD2<Double>

    init(surface: Surface, base: Double) {
        grid = surface.height
        caps = surface.caps
        self.base = base
        let d = grid.domain
        origin = SIMD2(d.real.lo, d.imag.lo)
        step = SIMD2(d.real.length / Double(max(grid.width - 1, 1)),
                     d.imag.length / Double(max(grid.height - 1, 1)))
    }

    /// The world box the solid lies in.
    var box: AABB<WorldSpace> {
        let d = grid.domain
        var top = -Double.infinity
        for y in 0..<grid.height {
            for x in 0..<grid.width {
                top = max(top, min(Double(grid[x, y]), caps.height(atX: origin.x + Double(x) * step.x)))
            }
        }
        return AABB(lo: SIMD3(min(d.real.lo, d.real.hi), min(d.imag.lo, d.imag.hi), base),
                    hi: SIMD3(max(d.real.lo, d.real.hi), max(d.imag.lo, d.imag.hi), top))
    }

    /// The drawn surface's height over a point, or nil off the domain.
    func height(at x: Double, _ y: Double) -> Double? {
        let u = (x - origin.x) / step.x, v = (y - origin.y) / step.y
        guard u >= 0, v >= 0, u <= Double(grid.width - 1), v <= Double(grid.height - 1) else {
            return nil
        }
        let i = min(Int(u), grid.width - 2), j = min(Int(v), grid.height - 2)
        let fu = u - Double(i), fv = v - Double(j)
        let h00 = Double(grid[i, j]), h10 = Double(grid[i + 1, j])
        let h01 = Double(grid[i, j + 1]), h11 = Double(grid[i + 1, j + 1])
        let h = fu + fv <= 1
            ? h00 + fu * (h10 - h00) + fv * (h01 - h00)
            : h11 + (1 - fu) * (h01 - h11) + (1 - fv) * (h10 - h11)
        return min(h, caps.height(atX: x))
    }

    func contains(_ p: SIMD3<Double>) -> Bool {
        guard let h = height(at: p.x, p.y) else { return false }
        return p.z >= base && p.z <= h
    }

    /// The view depth of the solid under a pixel: the largest view z along
    /// the sight line at which the line is inside the solid, or nil when the
    /// line misses it. `at` is the world point at view depth zero and
    /// `direction` the world step per unit of view depth.
    func depth(at w0: SIMD3<Double>, direction d: SIMD3<Double>, cell: Double) -> Double? {
        // Clip the line to the solid's box, so the march starts where there
        // could be anything to hit.
        let box = box
        var sLo = -Double.infinity, sHi = Double.infinity
        for k in 0..<3 {
            if abs(d[k]) < 1e-15 {
                if w0[k] < box.lo[k] || w0[k] > box.hi[k] { return nil }
                continue
            }
            let a = (box.lo[k] - w0[k]) / d[k], b = (box.hi[k] - w0[k]) / d[k]
            sLo = max(sLo, min(a, b)); sHi = min(sHi, max(a, b))
        }
        guard sLo <= sHi, sHi.isFinite, sLo.isFinite else { return nil }
        // March down from the near end in steps that move a quarter of a
        // cell over the plane, or a quarter of the height span down it.
        let planar = (d.x * d.x + d.y * d.y).squareRoot()
        let vertical = abs(d.z)
        var ds = Double.infinity
        if planar > 0 { ds = min(ds, 0.25 * cell / planar) }
        if vertical > 0 { ds = min(ds, 0.25 * max(box.hi.z - box.lo.z, cell) / vertical) }
        guard ds.isFinite, ds > 0 else { return nil }
        var s = sHi + 1e-9
        var outside = s
        guard !contains(w0 + s * d) else { return s }
        while s > sLo - ds {
            s -= ds
            if contains(w0 + s * d) {
                // Bisect the entry between the last point outside and this one.
                var lo = s, hi = outside
                for _ in 0..<50 {
                    let mid = (lo + hi) / 2
                    if contains(w0 + mid * d) { lo = mid } else { hi = mid }
                }
                return lo
            }
            outside = s
        }
        return nil
    }
}

func depthTests() {
    Check.suite("depth: the Metal pass agrees with an exact ray cast of the heightfield") {
        // A landscape with every kind of feature the pass draws: a plateau
        // under the cap, cut faces along the edges, pits down to the floor at
        // the zeros, and a long slope.
        let request = LandscapeRequest(preset: Catalog.native.preset("rgamma")!, resolution: 240)
        let bundle = try NativeLandscape.build(request)
        let preset = bundle.manifest.presets[0]
        let scene = Scene(bundle: bundle, preset: preset)
        let heightfield = scene.heightfield!
        Check.expect(heightfield.tiles.count == 1 && heightfield.region == .full,
                     "the landscape is one tile over its whole domain")

        let resolution = 300
        let renderer = try MetalRenderer()
        let baked = try renderer.bake(scene, options: BakeOptions(resolution: resolution))
        let depth = baked.depth
        let frame = depth.frame

        let solid = HeightfieldSolid(surface: heightfield.surface, base: 0)
        let cell = max(abs(solid.step.x), abs(solid.step.y))
        let toWorld = scene.camera.view.inverse
        let direction = toWorld(P3<ViewSpace>(0, 0, 1)).v - toWorld(P3<ViewSpace>(0, 0, 0)).v

        var reference = [Double?](repeating: nil, count: frame.rows * frame.cols)
        let clock = ContinuousClock()
        let elapsed = clock.measure {
            reference.withUnsafeMutableBufferPointer { buffer in
                // Each row writes only its own slots, so sharing is safe.
                nonisolated(unsafe) let base = buffer.baseAddress!
                DispatchQueue.concurrentPerform(iterations: frame.rows) { row in
                    for col in 0..<frame.cols {
                        let p = frame.coordinate(row: row, col: col)
                        let w0 = toWorld(P3<ViewSpace>(p.x, p.y, 0)).v
                        base[row * frame.cols + col] =
                            solid.depth(at: w0, direction: direction, cell: cell)
                    }
                }
            }
        }

        var covered = 0, shared = 0, onlyMetal = 0, onlyRay = 0
        var zLo = Double.infinity, zHi = -Double.infinity
        for (k, r) in reference.enumerated() {
            if let r { zLo = min(zLo, r); zHi = max(zHi, r) }
            let m = depth.isCovered(row: k / frame.cols, col: k % frame.cols)
            if m { covered += 1 }
            switch (m, r != nil) {
            case (true, true): shared += 1
            case (true, false): onlyMetal += 1
            case (false, true): onlyRay += 1
            case (false, false): break
            }
        }
        let span = zHi - zLo
        var differing = 0, worst = 0.0, total = 0.0
        for (k, r) in reference.enumerated() {
            guard let r, depth.isCovered(row: k / frame.cols, col: k % frame.cols) else { continue }
            let d = abs(depth[k / frame.cols, k % frame.cols] - r)
            total += d
            worst = max(worst, d)
            if d > 1e-3 * span { differing += 1 }
        }
        let pixels = frame.rows * frame.cols
        Check.expect(covered > pixels / 4, "the pass covered a good part of the frame",
                     "\(covered) of \(pixels) pixels")
        Check.expect(shared > 0 && Double(onlyMetal + onlyRay) <= 0.01 * Double(pixels),
                     "the two agree on which pixels the solid is under, to 1%",
                     "\(onlyMetal) only Metal, \(onlyRay) only the ray cast, of \(pixels)")
        Check.expect(Double(differing) <= 0.02 * Double(max(shared, 1)),
                     "and on its depth there, to 0.1% of the span on 98% of them",
                     String(format: "%d of %d differ; mean |Δ| %.2e, worst %.2e, span %.3f",
                            differing, shared, total / Double(max(shared, 1)), worst, span))
        Check.expect(total / Double(max(shared, 1)) < 1e-3 * span,
                     "the mean disagreement is below a thousandth of the span",
                     String(format: "%.2e over %.3f", total / Double(max(shared, 1)), span))
        Check.expect(elapsed < .seconds(20), "the ray cast ran in reasonable time", "\(elapsed)")
    }
}
