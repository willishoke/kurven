import Foundation
import simd
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

    Check.suite("fields: a contour is ended on the trim") {
        let square = (u: ParamAxis(Interval(lo: -1, hi: 1), periodic: false, samples: 41),
                      v: ParamAxis(Interval(lo: -1, hi: 1), periodic: false, samples: 41))
        let disc: @Sendable (P2<ParamSpace>) -> Bool = { simd_length($0.v) <= 1 }
        let chord = Contour.levels(of: ScalarField { $0.x }, [0.5], over: square, contains: disc)[0].paths
        let ends = chord.first.map { [$0[0], $0[$0.count - 1]].map { abs(simd_length($0.v) - 1) } }
        Check.expect(chord.count == 1 && (ends?.max() ?? 1) < 1e-12,
                     "a chord of the disc ends on its circle", "\(chord.count) runs, \(ends ?? [])")

        // A closed loop cut by a half-plane is one arc, on either side, from
        // wherever the loop happened to begin.
        let ring = ScalarField { $0.x * $0.x + $0.y * $0.y }
        for (name, side) in [("right", 1.0), ("left", -1.0)] {
            let arcs = Contour.levels(of: ring, [0.25], over: square,
                                      contains: { side * $0.x >= 0 })[0].paths
            let ends = arcs.first.map { [$0[0], $0[$0.count - 1]].map { abs($0.x) } }
            Check.expect(arcs.count == 1 && (ends?.max() ?? 1) < 1e-12,
                         "the \(name) half of a ring is one arc, ended on the cut",
                         "\(arcs.count) arcs, \(ends ?? [])")
        }
    }
}
