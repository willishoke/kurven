import Foundation
import simd
import Synchronization
import KurvenCore

/// Contours of a `ScalarField` on a parameter rectangle, held to closed forms.

/// The Hausdorff distance between two polylines in parameter coordinates,
/// taken on the flat torus where an axis has a period: each vertex of one to
/// the nearest point of the other, both ways.
private func hausdorff(_ a: [P2<ParamSpace>], _ b: [P2<ParamSpace>],
                       periods: (u: Double?, v: Double?)) -> Double {
    func directed(_ from: [P2<ParamSpace>], _ to: [P2<ParamSpace>]) -> Double {
        var worst = 0.0
        for p in from {
            var best = Double.infinity
            for (s, t) in zip(to, to.dropFirst()) {
                var q = p.v
                if let pu = periods.u { q.x -= pu * ((q.x - s.x) / pu).rounded() }
                if let pv = periods.v { q.y -= pv * ((q.y - s.y) / pv).rounded() }
                let d = t.v - s.v
                let k = simd_length_squared(d) > 0 ? min(max(simd_dot(q - s.v, d) / simd_length_squared(d), 0), 1) : 0
                best = min(best, simd_length(q - (s.v + k * d)))
            }
            worst = max(worst, best)
        }
        return worst
    }
    return max(directed(a, b), directed(b, a))
}

func scalarFieldTests() {
    Check.suite("fields: the contours of u and v on the torus are its coordinate curves") {
        let torus = ParametricSurface.torus(major: 3, minor: 1, samples: (u: 100, v: 50))
        let (u, v) = (torus.u, torus.v)
        let curves = torus.coordinateCurves(counts: (36, 18), resolution: 720)
        let uLevels = (0..<36).map { u.range.lo + u.range.length * Double($0) / 36 }
        let vLevels = (0..<18).map { v.range.lo + v.range.length * Double($0) / 18 }
        let us = Contour.levels(of: ScalarField(period: u.range.length) { $0.x }, uLevels, over: (u, v))
        let vs = Contour.levels(of: ScalarField(period: v.range.length) { $0.y }, vLevels, over: (u, v))
        let contours = (us + vs).map(\.paths)
        Check.expect(contours.allSatisfy { $0.count == 1 } && contours.count == curves.count,
                     "one closed path per level, as many as the coordinate curves",
                     "\(contours.map(\.count).reduce(0, +)) paths vs \(curves.count)")
        var worst = 0.0
        for (k, paths) in contours.enumerated() where paths.count == 1 {
            worst = max(worst, hausdorff(paths[0], Array(curves.coords(path: k)!),
                                         periods: (u.period, v.period)))
        }
        Check.expect(worst < 1e-12 * 2 * .pi, "each lies on its coordinate curve, to 1e-12 of the period",
                     "Hausdorff \(worst)")
        // Each is closed across the seam by marching squares itself, the
        // crossing there found once: its last vertex is its first carried a
        // period round, to the bit. Two crossings found apart on the two
        // sides of the seam, as an open path from seam to seam would end on,
        // agree only to rounding.
        let unclosed = contours.enumerated().filter { k, paths in
            guard let path = paths.first, let first = path.first, let last = path.last else { return true }
            let round = k < 36 ? SIMD2(0, v.range.length) : SIMD2(u.range.length, 0)
            return last.v != first.v + round && last.v != first.v - round
        }.count
        Check.expect(unclosed == 0, "each ends on its first vertex a period round",
                     "\(unclosed) of \(contours.count) do not")

        // A smooth field with no period: x² + y² on a square, whose level
        // 1/4 is the circle of radius 1/2.
        let square = (u: ParamAxis(Interval(lo: -1, hi: 1), periodic: false, samples: 41),
                      v: ParamAxis(Interval(lo: -1, hi: 1), periodic: false, samples: 41))
        let circle = Contour.levels(of: ScalarField { $0.x * $0.x + $0.y * $0.y }, [0.25], over: square)[0].paths
        let off = circle.joined().map { abs(simd_length($0.v) - 0.5) }.max() ?? .infinity
        Check.expect(circle.count == 1 && circle[0].first == circle[0].last && off < 1e-12,
                     "a contour of x² + y² lies on its circle, to 1e-12", "\(circle.count) paths, \(off) off")
    }

    Check.suite("fields: a contour closes across the seam") {
        // cos u + cos v = 1.2 is a loop round the rectangle's corner, which
        // the two seams cut into four pieces of the rectangle.
        let torus = (u: ParamAxis.angle(samples: 64), v: ParamAxis.angle(samples: 48))
        let field = ScalarField { cos($0.x) + cos($0.y) }
        let paths = Contour.levels(of: field, [1.2], over: torus)[0].paths
        Check.expect(paths.count == 1, "one path", "\(paths.count)")
        if let loop = paths.first {
            let off = loop.map { abs(cos($0.x) + cos($0.y) - 1.2) }.max() ?? .infinity
            let step = zip(loop, loop.dropFirst()).map { simd_length($1.v - $0.v) }.max() ?? 0
            let cell = 2 * Double.pi / 48
            Check.expect(loop.first == loop.last && step < 2 * cell,
                         "closed, its coordinates continuous across both seams",
                         "longest step \(step / cell) cells")
            Check.expect(loop.contains { $0.x < 0 } || loop.contains { $0.x > .pi },
                         "and running round the corner, not along the edges")
            Check.expect(off < 1e-12, "on the level set", "\(off)")
        }
    }

    Check.suite("fields: a level through grid vertices is drawn once") {
        // Seventy-two samples round u: every level of thirty-six is a column
        // of the grid, to rounding, and the first is the seam.
        let torus = (u: ParamAxis.angle(samples: 72), v: ParamAxis.angle(samples: 30))
        let levels = (0..<36).map { 2 * Double.pi * Double($0) / 36 }
        let contours = Contour.levels(of: ScalarField(period: 2 * .pi) { $0.x }, levels, over: torus)
        let counts = contours.map(\.paths.count)
        Check.expect(counts.allSatisfy { $0 == 1 }, "one path at each level",
                     "\(counts.filter { $0 != 1 }.count) levels with \(Set(counts).sorted()) paths")
        var off = 0.0
        for c in contours { for p in c.paths.joined() { off = max(off, abs(remainder(p.x - c.level, 2 * .pi))) } }
        Check.expect(off < 1e-14, "on its level", "\(off)")

        // x + y = 1 on a grid of tenths runs corner to corner through the
        // vertices of the anti-diagonal.
        let square = (u: ParamAxis(Interval(lo: 0, hi: 1), periodic: false, samples: 11),
                      v: ParamAxis(Interval(lo: 0, hi: 1), periodic: false, samples: 11))
        let line = Contour.levels(of: ScalarField { $0.x + $0.y }, [1], over: square)[0].paths
        let ends = line.first.map { [$0[0], $0[$0.count - 1]].map { min(simd_length($0.v - SIMD2(0, 1)),
                                                                   simd_length($0.v - SIMD2(1, 0))) } }
        Check.expect(line.count == 1 && (ends?.max() ?? 1) < 1e-15
                        && line[0].allSatisfy { abs($0.x + $0.y - 1) < 1e-15 },
                     "a diagonal through vertices is one path from corner to corner",
                     "\(line.count) paths")
    }

    Check.suite("fields: a level that touches nothing draws nothing") {
        let square = (u: ParamAxis(Interval(lo: -1, hi: 1), periodic: false, samples: 21),
                      v: ParamAxis(Interval(lo: -1, hi: 1), periodic: false, samples: 21))
        let bowl = ScalarField { $0.x * $0.x + $0.y * $0.y }
        let outside = Contour.levels(of: bowl, [-1, 2.5], over: square)
        Check.expect(outside.allSatisfy { $0.paths.isEmpty }, "a level beyond the field's range")

        // A circle-valued field over [-0.3, 0.3] has nothing at π, where the
        // difference from the level jumps by a period wherever the field is 0.
        let torus = (u: ParamAxis.angle(samples: 40), v: ParamAxis.angle(samples: 20))
        let small = ScalarField(period: 2 * .pi) { 0.3 * sin($0.x) }
        let opposite = Contour.levels(of: small, [.pi], over: torus)[0].paths
        Check.expect(opposite.isEmpty, "a circle-valued level opposite every value",
                     "\(opposite.count) paths")
    }

    Check.suite("fields: a field with a routine of its own is handed the grid's contours") {
        // Samples of x² + y² on a field whose value is something else: the
        // routine is handed the samples' contours, and its answer returned.
        let n = 41
        let domain = Domain(real: Interval(lo: -1, hi: 1), imag: Interval(lo: -1, hi: 1))
        var values = [Float](repeating: 0, count: n * n)
        for j in 0..<n { for i in 0..<n {
            let x = -1 + 2 * Double(i) / Double(n - 1), y = -1 + 2 * Double(j) / Double(n - 1)
            values[j * n + i] = Float(x * x + y * y)
        } }
        let samples = Grid2D(width: n, height: n, domain: domain, values: values)
        let field = ScalarField(samples: samples) { level, paths in
            paths.map { $0.map { P2($0.x + level, $0.y) } }
        }
        let square = (u: ParamAxis(Interval(lo: -1, hi: 1), periodic: false, samples: n),
                      v: ParamAxis(Interval(lo: -1, hi: 1), periodic: false, samples: n))
        let got = Contour.levels(of: field, [0.25], over: square)[0].paths
        let want = Contour.lines(of: samples, level: 0.25).map { $0.map { P2<ParamSpace>($0.x + 0.25, $0.y) } }
        Check.expect(got == want, "the grid's contours, as the routine places them",
                     "\(got.count) paths vs \(want.count)")

        // arg z round the origin, handed on as the samples' contours: a
        // circle-valued field's routine is handed its cut at π, where the
        // samples pass from π to -π, and no crossing along the cut at a
        // level beside it.
        let m = 40
        var phases = [Float](repeating: 0, count: m * m)
        for j in 0..<m { for i in 0..<m {
            let x = -1 + 2 * Double(i) / Double(m - 1), y = -1 + 2 * Double(j) / Double(m - 1)
            phases[j * m + i] = Float(atan2(y, x))
        } }
        let phase = ScalarField(samples: Grid2D(width: m, height: m, domain: domain, values: phases),
                                period: 2 * .pi) { _, paths in paths }
        let window = (u: ParamAxis(Interval(lo: -1, hi: 1), periodic: false, samples: m),
                      v: ParamAxis(Interval(lo: -1, hi: 1), periodic: false, samples: m))
        let cut = Contour.levels(of: phase, [.pi], over: window)[0].paths
        let across = cut.joined().map { abs($0.y) }.max() ?? .infinity
        Check.expect(cut.count == 1 && cut[0].allSatisfy { $0.x < 0 } && across < 1e-6,
                     "the level π is one path along the cut", "\(cut.count) paths, \(across) off the axis")
        // Linear along an edge, the ray is off its angle by a chord's error,
        // which is large only within a few cells of the origin.
        let ray = Contour.levels(of: phase, [3], over: window)[0].paths
        let turn = ray.joined().filter { simd_length($0.v) > 0.2 }
            .map { abs(remainder(atan2($0.y, $0.x) - 3, 2 * .pi)) }.max() ?? .infinity
        Check.expect(ray.count == 1 && turn < 1e-2, "the level 3 is one ray, with nothing along the cut",
                     "\(ray.count) paths, \(turn) off its angle")
    }

    Check.suite("fields: arg f hands the refiner the grid's own contours, and the cut at ±π") {
        // |z| and arg z on a grid that keeps off the origin. The refiner here
        // moves nothing, and says what it was asked: the field's routine is
        // handed what the grid has, so what comes back is what it was handed.
        let m = 40
        let domain = Domain(real: Interval(lo: -1, hi: 1), imag: Interval(lo: -1, hi: 1))
        var moduli = [Float](repeating: 0, count: m * m), phases = moduli
        for j in 0..<m { for i in 0..<m {
            let x = -1 + 2 * Double(i) / Double(m - 1), y = -1 + 2 * Double(j) / Double(m - 1)
            moduli[j * m + i] = Float((x * x + y * y).squareRoot())
            phases[j * m + i] = Float(atan2(y, x))
        } }
        let surface = Surface(height: Grid2D(width: m, height: m, domain: domain, values: moduli),
                              phase: Grid2D(width: m, height: m, domain: domain, values: phases),
                              truncation: .uniform(0.8))
        let asked = Mutex([String]())
        let unmoved = ContourRefine(contours: { field, level, paths, ceiling in
            asked.withLock { $0.append("\(field) \(level) \(ceiling)") }
            return paths
        })
        func inParameters(_ paths: [[P2<DomainSpace>]]) -> [[P2<ParamSpace>]] {
            paths.map { $0.map { P2<ParamSpace>($0.v) } }
        }
        let phase = surface.phase!
        let levels = Surface.levels([-.pi, 3, .pi], of: .phase)
        Check.expect(levels == [3, .pi], "the cut is asked once, after the rest", "\(levels)")

        // No period: at a level beside the cut the refiner is handed the
        // grid's contours whole, crossings along the wrap included, which
        // only it can tell from a level set beside a zero.
        let field = surface.field(.phase, refine: unmoved)!
        let placed = Contour.levels(of: field, levels, over: surface.rectangle)
        let grid = inParameters(Contour.levels(of: phase, [3])[0].paths)
        let alongWrap = placed[0].paths.joined().filter { $0.x < -0.2 && abs($0.y) < 1.0 / Double(m - 1) }
        Check.expect(placed[0].paths == grid && !alongWrap.isEmpty,
                     "the level 3 is the grid's, with its crossings along the wrap",
                     "\(placed[0].paths.count) paths vs \(grid.count), \(alongWrap.count) along the wrap")
        let cut = inParameters(Contour.cut(of: phase))
        Check.expect(placed[1].paths == cut && !cut.isEmpty, "the level π is the cut, `Contour.cut`",
                     "\(placed[1].paths.count) paths vs \(cut.count)")
        let bare = Contour.levels(of: surface.field(.phase, refine: nil)!, levels, over: surface.rectangle)
        Check.expect(bare[0].paths == grid && bare[1].paths.isEmpty,
                     "without a refiner the grid's contours, and no cut",
                     "\(bare[0].paths.count) and \(bare[1].paths.count) paths")

        // A landscape layer is its field's contours at the levels it asks
        // for, lifted; the refiner is handed the field's tag and the
        // layer's ceiling, a tenth over the truncation for curves lifted to
        // |f| and kept below it.
        asked.withLock { $0 = [] }
        let layer = surface.derive(.contour(field: .phase, levels: [-.pi, 3, .pi], restriction: .all, tiled: false),
                                   policy: .level, region: .full, tiles: [.identity], refine: unmoved)
        let lifted = placed.flatMap { level, paths in
            paths.filter { $0.count >= 2 }.map { $0.map { P3<WorldSpace>($0.x, $0.y, level) } }
        }
        Check.expect((0..<layer.count).map { Array(layer[path: $0]) } == lifted,
                     "a phase layer is the field's contours, lifted at their levels",
                     "\(layer.count) paths vs \(lifted.count)")
        _ = surface.derive(.contour(field: .magnitude, levels: [0.5], restriction: .belowTruncation, tiled: false),
                           policy: .magnitude, region: .full, tiles: [.identity], refine: unmoved)
        let calls = asked.withLock { $0 }
        Check.expect(calls.sorted() == ["magnitude 0.5 \(1.1 * 0.8)", "phase 3.0 inf", "phase \(Double.pi) inf"],
                     "placed under the field's tag and the layer's ceiling", "\(calls)")
    }

    Check.suite("fields: a contour is ended where it leaves the trimmed domain") {
        let square = (u: ParamAxis(Interval(lo: -1, hi: 1), periodic: false, samples: 41),
                      v: ParamAxis(Interval(lo: -1, hi: 1), periodic: false, samples: 41))
        let disc: @Sendable (P2<ParamSpace>) -> Bool = { simd_length($0.v) <= 1 }
        let chord = Contour.levels(of: ScalarField { $0.x }, [0.5], over: square, contains: disc)[0].paths
        let ends = chord.first.map { [$0[0], $0[$0.count - 1]].map { abs(simd_length($0.v) - 1) } }
        Check.expect(chord.count == 1 && (ends?.max() ?? 1) < 1e-12,
                     "a chord of the disc ends on its circle", "\(chord.count) runs, \(ends ?? [])")

        // A closed loop cut by a half-plane is one arc, on either side,
        // whether the loop begins on the cut, inside or outside: the ring
        // begins at its lowest point, (0, -1/2).
        let ring = ScalarField { $0.x * $0.x + $0.y * $0.y }
        let cell = 2.0 / 40
        for cut in [0.0, 0.2, -0.2] {
            for (name, side) in [("right", 1.0), ("left", -1.0)] {
                let arcs = Contour.levels(of: ring, [0.25], over: square,
                                          contains: { side * ($0.x - cut) >= 0 })[0].paths
                let ends = arcs.first.map { [$0[0], $0[$0.count - 1]].map { abs($0.x - cut) } }
                let kept = arcs.joined().allSatisfy { side * ($0.x - cut) >= 0 }
                let step = arcs.joined().isEmpty ? 0 : arcs.map { zip($0, $0.dropFirst())
                    .map { simd_length($1.v - $0.v) }.max() ?? 0 }.max()!
                Check.expect(arcs.count == 1 && (ends?.max() ?? 1) < 1e-12 && kept && step < 2 * cell,
                             "the \(name) of x = \(cut) on a ring is one arc, ended on the cut",
                             "\(arcs.count) arcs, ends \(ends ?? []) off, longest step \(step / cell) cells")
            }
        }

        // A closed contour across the seam whose closing vertex lies a period
        // past its first, where the domain may answer otherwise by rounding:
        // v = 1 on the torus, from u = 0 to 2π, where sin 2u is 0 and -4.9e-16.
        // Kept where sin 2u >= 0 it is two quarters, not one run with a chord
        // across the quarter between; so too where the loop begins inside
        // the domain, or outside it.
        let torus = (u: ParamAxis.angle(samples: 40), v: ParamAxis.angle(samples: 30))
        let quarters: [(String, @Sendable (Double) -> Double)] = [
            ("sin 2u >= 0, the loop beginning on its boundary", { sin(2 * $0) }),
            ("cos 2u >= 0, beginning inside", { cos(2 * $0) }),
            ("-cos 2u >= 0, beginning outside", { -cos(2 * $0) }),
        ]
        for (name, g) in quarters {
            let runs = Contour.levels(of: ScalarField(period: 2 * .pi) { $0.y }, [1], over: torus,
                                      contains: { g($0.x) >= 0 })[0].paths
            let length = runs.map { zip($0, $0.dropFirst()).map { simd_length($1.v - $0.v) }.reduce(0, +) }
                .reduce(0, +)
            let ends = runs.flatMap { [$0[0], $0[$0.count - 1]] }.map { abs(g($0.x)) }.max() ?? .infinity
            // A boundary that falls on a vertex ends the run there, once.
            let repeated = runs.contains { zip($0, $0.dropFirst()).contains { $0 == $1 } }
            Check.expect(runs.count == 2 && abs(length - .pi) < 1e-9 && ends < 1e-14 && !repeated,
                         "v = 1 kept where \(name) is two quarters, ended on their boundaries",
                         "\(runs.count) runs, \(length / .pi) π long, ends \(ends) off\(repeated ? ", a vertex repeated" : "")")
        }
    }
}
