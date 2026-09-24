import Foundation
import simd
import KurvenCore

// MARK: - the truncated top, where the drawn surface and the ink meet

/// The cap rim: the pieces `Mesh.capRim` adds to the depth pass, and the
/// vertex `Surface.derive` carries a contour to.
func capTests() {
    /// A spire on a coarse grid: `h = 3 - 4 r²`, so the cap at 2 cuts it in a
    /// disc of radius one half, two cells across. Every cell round the rim
    /// straddles it.
    let n = 9
    let domain = Domain(real: Interval(lo: -1, hi: 1), imag: Interval(lo: -1, hi: 1))
    func spireHeight(_ x: Double, _ y: Double) -> Double { 3 - 4 * (x * x + y * y) }
    /// Sample `k` of an `m x m` grid over [-1, 1]², rows indexing imag.
    func at(_ k: Int, _ m: Int) -> (x: Double, y: Double) {
        (-1 + 2 * Double(k % m) / Double(m - 1), -1 + 2 * Double(k / m) / Double(m - 1))
    }
    let grid = Grid2D(width: n, height: n, domain: domain,
                      values: (0..<(n * n)).map { k -> Float in
                          let p = at(k, n)
                          return Float(spireHeight(p.x, p.y))
                      })
    let cap = 2.0
    let surface = Surface(height: grid, phase: nil, caps: .uniform(cap))

    Check.suite("cap rim: the pieces are the clamped surface, cell for cell") {
        let rim = Mesh.capRim(of: surface, step: 1, region: .full, tiles: [.identity])
        Check.expect(!rim.isEmpty, "a capped spire has a rim", "\(rim.triangles.count) triangles")
        Check.expect(Mesh.capRim(of: Surface(height: grid, phase: nil, caps: Caps.none),
                                 step: 1, region: .full, tiles: [.identity]).isEmpty,
                     "and an uncapped one has none")

        // The rasterizer's own triangle through a point: corners clamped, then
        // interpolated. And what the samples describe: interpolated, then
        // clamped. Drawn under a MAX blend with the rim pieces, the first must
        // become the second.
        func plane(_ a: SIMD3<Double>, _ b: SIMD3<Double>, _ c: SIMD3<Double>,
                   at p: SIMD2<Double>) -> Double? {
            let d = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y)
            let u = ((b.y - c.y) * (p.x - c.x) + (c.x - b.x) * (p.y - c.y)) / d
            let v = ((c.y - a.y) * (p.x - c.x) + (a.x - c.x) * (p.y - c.y)) / d
            let w = 1 - u - v
            let eps = 1e-9
            guard u >= -eps, v >= -eps, w >= -eps else { return nil }
            return u * a.z + v * b.z + w * c.z
        }
        var rng = SplitMix(seed: 11)
        var worst = 0.0
        var tested = 0
        var straddled = 0
        let cell = 2.0 / Double(n - 1)
        for _ in 0..<4000 {
            let p = SIMD2(rng.next(-1, 1), rng.next(-1, 1))
            let i = min(Int((p.x + 1) / cell), n - 2), j = min(Int((p.y + 1) / cell), n - 2)
            let x0 = -1 + Double(i) * cell, y0 = -1 + Double(j) * cell
            func corner(_ dx: Int, _ dy: Int) -> SIMD3<Double> {
                SIMD3(x0 + Double(dx) * cell, y0 + Double(dy) * cell,
                      Double(grid[i + dx, j + dy]))
            }
            let c00 = corner(0, 0), c10 = corner(1, 0), c01 = corner(0, 1), c11 = corner(1, 1)
            let corners = [c00, c10, c01, c11]
            let straddles = corners.contains { $0.z > cap } && corners.contains { $0.z < cap }
            // The rasterizer's diagonal: (00, 10, 01) and (10, 11, 01).
            let tri: [SIMD3<Double>]
            if plane(c00, c10, c01, at: p) != nil { tri = [c00, c10, c01] } else { tri = [c10, c11, c01] }
            let expected = min(plane(tri[0], tri[1], tri[2], at: p)!, cap)
            func clamp(_ v: SIMD3<Double>) -> SIMD3<Double> { SIMD3(v.x, v.y, min(v.z, cap)) }
            var drawn = plane(clamp(tri[0]), clamp(tri[1]), clamp(tri[2]), at: p)!
            for t in rim.triangles {
                let a = rim.vertices[Int(t.x)].v, b = rim.vertices[Int(t.y)].v, c = rim.vertices[Int(t.z)].v
                if let z = plane(a, b, c, at: p) { drawn = max(drawn, z) }
            }
            worst = max(worst, abs(drawn - expected))
            tested += 1
            if straddles { straddled += 1 }
        }
        Check.expect(straddled > 200, "the sample covers the rim cells", "\(straddled) of \(tested)")
        Check.expect(worst < 1e-9, "MAX(rasterized cell, rim pieces) is min(interpolated, cap) everywhere",
                     String(format: "worst %.1e over %d points", worst, tested))

        // Every piece lies on the clamped surface: on the cap, or on its cell's
        // interpolated height, and never above the cap.
        let over = rim.vertices.filter { $0.z > cap + 1e-12 }.count
        Check.expect(over == 0, "no piece rises above the cap", "\(over) vertices do")
    }

    Check.suite("cap rim: the lattice is the rasterizer's") {
        // Per tile, once; outside the region, not at all.
        let one = Mesh.capRim(of: surface, step: 1, region: .full, tiles: [.identity])
        let shifted = Affine2(a: 1, b: 0, tx: 5, c: 0, d: 1, ty: 0)
        let two = Mesh.capRim(of: surface, step: 1, region: .full, tiles: [.identity, shifted])
        Check.expect(two.triangles.count == 2 * one.triangles.count,
                     "two tiles draw the rim twice", "\(two.triangles.count) vs \(one.triangles.count)")
        Check.expect(two.vertices.map(\.x).max()! > 4,
                     "the second copy is where its tile puts it")
        let far = BoundaryPerimeter(edges: [
            PerimeterEdge(start: P2(3, 3), end: P2(4, 3), density: 2),
            PerimeterEdge(start: P2(4, 3), end: P2(4, 4), density: 2),
            PerimeterEdge(start: P2(4, 4), end: P2(3, 4), density: 2),
            PerimeterEdge(start: P2(3, 4), end: P2(3, 3), density: 2),
        ])
        Check.expect(Mesh.capRim(of: surface, step: 1, region: .inside(far),
                                 tiles: [.identity]).isEmpty,
                     "a region that excludes the spire has no rim in it")
        // Decimated by two, the lattice is 5 x 5 over the same domain, and the
        // rim is cut on those cells: coarser, and still under the cap.
        let coarse = Mesh.capRim(of: surface, step: 2, region: .full, tiles: [.identity])
        let coarseXs = Set(coarse.vertices.map { ($0.x * 1e9).rounded() / 1e9 })
        let lattice = Set((0..<5).map { (-1.0 + Double($0) * 0.5) })
        let onLattice = coarse.vertices.filter { v in
            lattice.contains(where: { abs($0 - v.x) < 1e-9 }) || lattice.contains(where: { abs($0 - v.y) < 1e-9 })
        }.count
        Check.expect(!coarse.isEmpty && onLattice > 0 && coarse.triangles.count < one.triangles.count,
                     "step 2 cuts the rim on the decimated lattice",
                     "\(coarse.triangles.count) triangles, \(coarseXs.count) distinct x")
    }

    Check.suite("cap crossing: a contour is carried to the cap, and only by a refiner") {
        // A ramp `2x + 3` under a cap of 3.5, so the cap is crossed at
        // x = 1/4; and a phase grid equal to y, whose zero contour is the
        // line y = 0 straight up the ramp.
        let m = 5
        let d = Domain(real: Interval(lo: -1, hi: 1), imag: Interval(lo: -1, hi: 1))
        let ramp = Grid2D(width: m, height: m, domain: d, values: (0..<(m * m)).map { k in
            Float(2 * at(k, m).x + 3)
        })
        let phase = Grid2D(width: m, height: m, domain: d, values: (0..<(m * m)).map { k in
            Float(at(k, m).y)
        })
        let rampCap = 3.5
        let s = Surface(height: ramp, phase: phase, caps: .uniform(rampCap))
        let source = LayerSource.contour(field: .phase, levels: [0], keep: .belowCap, tiled: false)
        let identity: ContourRefine = { _, _, paths in paths }

        let plain = s.derive(source, policy: .magnitude, region: .full, tiles: [.identity])
        let carried = s.derive(source, policy: .magnitude, region: .full, tiles: [.identity],
                               refine: identity)
        Check.expect(plain.count == 1 && carried.count == 1, "one run each",
                     "\(plain.count) and \(carried.count)")
        let plainTop = plain.vertices.map(\.z).max() ?? -1
        let carriedTop = carried.vertices.map(\.z).max() ?? -1
        Check.expect(abs(plainTop - 3) < 1e-6, "from the grid alone the run ends at its last sample under the cap",
                     "top z = \(plainTop)")
        Check.expect(abs(carriedTop - rampCap) < 1e-12, "with a refiner it ends on the cap exactly",
                     "top z = \(carriedTop)")
        let rimVertex = carried.vertices.first { abs($0.z - rampCap) < 1e-12 }
        Check.expect(rimVertex.map { abs($0.x - 0.25) < 1e-9 && abs($0.y) < 1e-9 } ?? false,
                     "and that vertex is on the rim, where the interpolated height is the cap",
                     rimVertex.map { "(\($0.x), \($0.y))" } ?? "no rim vertex")
        Check.expect(carried.vertices.count == plain.vertices.count + 1,
                     "one vertex was added and none moved",
                     "\(carried.vertices.count) vs \(plain.vertices.count)")
        Check.expect(carried.vertices.allSatisfy { $0.z <= rampCap + 1e-12 },
                     "nothing over the cap survives")

        // A layer that does not keep below the cap is not touched.
        let free = LayerSource.contour(field: .phase, levels: [0], keep: .all, tiled: false)
        let a = s.derive(free, policy: .magnitude, region: .full, tiles: [.identity])
        let b = s.derive(free, policy: .magnitude, region: .full, tiles: [.identity], refine: identity)
        Check.expect(a.vertices.count == b.vertices.count && a.vertices.map(\.z).max() == b.vertices.map(\.z).max(),
                     "a layer that keeps everything gains no cap vertex")
    }
}
