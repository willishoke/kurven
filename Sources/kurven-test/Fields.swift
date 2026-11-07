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
    }

    Check.suite("fields: arg f hands the refiner the grid's own contours, and the branch cut at ±π") {
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
        let unmoved = ContourRefine(contours: { field, level, paths, ceiling, _ in
            asked.withLock { $0.append("\(field) \(level) \(ceiling)") }
            return paths
        }, rings: nil)
        func inParameters(_ paths: [[P2<DomainSpace>]]) -> [[P2<ParamSpace>]] {
            paths.map { $0.map { P2<ParamSpace>($0.v) } }
        }
        let phase = surface.phase!
        let levels = ContourField.phase.levels([-.pi, 3, .pi])
        Check.expect(levels == [3, .pi], "the branch cut is asked once, after the rest", "\(levels)")

        // No period: at a level beside the branch cut the refiner is handed the
        // grid's contours whole, crossings along the wrap included, which
        // only it can tell from a level set beside a zero.
        let rectangle = surface.parameterRectangle!
        let field = surface.field(of: .phase, refine: unmoved)!
        let placed = Contour.levels(of: field, levels, over: rectangle)
        let grid = inParameters(Contour.levels(of: phase, [3])[0].paths)
        let alongWrap = placed[0].paths.joined().filter { $0.x < -0.2 && abs($0.y) < 1.0 / Double(m - 1) }
        Check.expect(placed[0].paths == grid && !alongWrap.isEmpty,
                     "the level 3 is the grid's, with its crossings along the wrap",
                     "\(placed[0].paths.count) paths vs \(grid.count), \(alongWrap.count) along the wrap")
        let cut = inParameters(Contour.cut(of: phase))
        Check.expect(placed[1].paths == cut && !cut.isEmpty, "the level π is the branch cut, `Contour.cut`",
                     "\(placed[1].paths.count) paths vs \(cut.count)")
        let bare = Contour.levels(of: surface.field(of: .phase, refine: nil)!, levels, over: rectangle)
        Check.expect(bare[0].paths == grid && bare[1].paths.isEmpty,
                     "without a refiner the grid's contours, and no branch cut",
                     "\(bare[0].paths.count) and \(bare[1].paths.count) paths")

        // A landscape layer is its field's contours at the levels it asks
        // for, lifted; the refiner is handed the field's tag and the
        // layer's ceiling, a tenth over the truncation for curves lifted to
        // |f| and kept below it.
        asked.withLock { $0 = [] }
        let layer = surface.derive(.contour(field: .phase, levels: [-.pi, 3, .pi], restriction: .all, faces: .graph(tiled: false)),
                                   policy: .level, region: .full, tiles: [.identity], refine: unmoved)
        let lifted = placed.flatMap { level, paths in
            paths.filter { $0.count >= 2 }.map { $0.map { P3<WorldSpace>($0.x, $0.y, level) } }
        }
        Check.expect((0..<layer.count).map { Array(layer[path: $0]) } == lifted,
                     "a phase layer is the field's contours, lifted at their levels",
                     "\(layer.count) paths vs \(lifted.count)")
        _ = surface.derive(.contour(field: .magnitude, levels: [0.5], restriction: .belowTruncation, faces: .graph(tiled: false)),
                           policy: .magnitude, region: .full, tiles: [.identity], refine: unmoved)
        let calls = asked.withLock { $0 }
        Check.expect(calls.sorted() == ["magnitude 0.5 \(1.1 * 0.8)", "phase 3.0 inf", "phase \(Double.pi) inf"],
                     "placed under the field's tag and the layer's ceiling", "\(calls)")
    }

    Check.suite("fields: a contour is cut where it crosses a trim loop, and kept where the loops hold it even-odd") {
        let square = (u: ParamAxis(Interval(lo: -1, hi: 1), periodic: false, samples: 41),
                      v: ParamAxis(Interval(lo: -1, hi: 1), periodic: false, samples: 41))
        /// A regular polygon, as a trim loop, and membership in it, even-odd.
        func polygon(_ c: SIMD2<Double>, _ r: Double, _ n: Int) -> [P2<ParamSpace>] {
            let corners = (0..<n).map { k in P2<ParamSpace>(c + r * SIMD2(cos(2 * .pi * Double(k) / Double(n)),
                                                                         sin(2 * .pi * Double(k) / Double(n)))) }
            return corners + [corners[0]]
        }
        @Sendable func inside(_ p: P2<ParamSpace>, _ loop: [P2<ParamSpace>]) -> Bool {
            var inside = false
            for (a, b) in zip(loop, loop.dropFirst()) where (a.y > p.y) != (b.y > p.y) {
                if p.x < a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y) { inside.toggle() }
            }
            return inside
        }
        func distance(_ p: P2<ParamSpace>, _ loop: [P2<ParamSpace>]) -> Double {
            zip(loop, loop.dropFirst()).map { a, b in
                let d = b.v - a.v, t = max(0, min(1, simd_dot(p.v - a.v, d) / simd_length_squared(d)))
                return simd_length(p.v - (a.v + t * d))
            }.min() ?? .infinity
        }

        let disc = polygon(.zero, 1, 720)
        let chord = Contour.levels(of: ScalarField { $0.x }, [0.5], over: square, trimLoops: [disc])[0].paths
        let ends = chord.first.map { [$0[0], $0[$0.count - 1]].map { distance($0, disc) } }
        Check.expect(chord.count == 1 && (ends?.max() ?? 1) < 1e-15,
                     "a chord of the disc ends on its loop", "\(chord.count) runs, \(ends ?? [])")

        // A hole narrower than a step, between two of the line's vertices:
        // x = 0.31 crosses rows a twentieth apart, at Im 0 and 0.05, and the
        // hole spans 0.005 to 0.045 between them. The square's edge is the
        // outer loop.
        let edge = [P2<ParamSpace>(-1, -1), P2(1, -1), P2(1, 1), P2(-1, 1), P2(-1, -1)]
        let hole = polygon(SIMD2(0.31, 0.025), 0.02, 64)
        let split = Contour.levels(of: ScalarField { $0.x }, [0.31], over: square, trimLoops: [edge, hole])[0].paths
        let holeEnds = split.flatMap { [$0[0], $0[$0.count - 1]] }.filter { abs($0.y) < 0.5 }
        let length = split.map { zip($0, $0.dropFirst()).map { simd_length($1.v - $0.v) }.reduce(0, +) }.reduce(0, +)
        let across = zip(hole, hole.dropFirst()).compactMap { a, b -> Double? in
            guard (a.x - 0.31) * (b.x - 0.31) <= 0, a.x != b.x else { return nil }
            return a.y + (0.31 - a.x) * (b.y - a.y) / (b.x - a.x)
        }
        let gap = (across.max() ?? 0) - (across.min() ?? 0)
        Check.expect(split.count == 2 && holeEnds.count == 2 && holeEnds.allSatisfy { distance($0, hole) < 1e-15 }
                        && abs(length - (2 - gap)) < 1e-12,
                     "x = 0.31 on a 41 × 41 square with a hole of radius 0.02 at (0.31, 0.025) is two runs, ended on the hole",
                     "\(split.count) runs, ends \(holeEnds.map { distance($0, hole) }), \(length) long against \(2 - gap)")

        // Where the loops are chords of a circle, the loops decide, not the
        // circle: every run of 400 lines across a 24-gon hole ends on it.
        let gon = polygon(SIMD2(0, 0.013), 0.3, 24)
        var gonRuns = 0, gonOff = 0.0
        for k in 0..<400 {
            let runs = Contour.levels(of: ScalarField { $0.x }, [-0.3 + 0.6 * (Double(k) + 0.5) / 400], over: square,
                                      trimLoops: [edge, gon])[0].paths
            for r in runs {
                gonRuns += 1
                for e in [r[0], r[r.count - 1]] where abs(e.y) < 0.99 { gonOff = max(gonOff, distance(e, gon)) }
            }
        }
        Check.expect(gonRuns >= 800 && gonOff < 1e-15, "lines across a polygonal hole end on it, every one",
                     "\(gonRuns) runs, ends up to \(gonOff) off")

        // A boundary that runs along the line, given as a cut with the
        // membership it bounds: the line is cut where the stretch begins and
        // ends, not at the vertices either side. Met only along the line, as
        // the stretch has no side that crosses it.
        let along: [[P2<ParamSpace>]] = [[P2(0.31, 0.012), P2(0.31, 0.088)]]
        let aside = Contour.levels(of: ScalarField { $0.x }, [0.31], over: square,
                                   contains: { !($0.y > 0.012 && $0.y < 0.088) }, cuts: along)[0].paths
        let asideEnds = aside.flatMap { [$0[0].y, $0[$0.count - 1].y] }.filter { abs($0) < 0.5 }.sorted()
        Check.expect(aside.count == 2 && asideEnds == [0.012, 0.088],
                     "a boundary along the line cuts it where the boundary begins and ends", "\(aside.count) runs, ends \(asideEnds)")

        // A closed loop cut by a half-plane, bounded by a rectangle on the
        // cut, is one arc, on either side, whether the loop begins on the
        // cut, inside or outside: the ring begins at its lowest point,
        // (0, -1/2).
        let ring = ScalarField { $0.x * $0.x + $0.y * $0.y }
        let cell = 2.0 / 40
        for cut in [0.0, 0.2, -0.2] {
            for (name, side) in [("right", 1.0), ("left", -1.0)] {
                let half = [P2<ParamSpace>(cut, -2), P2(cut, 2), P2(cut + 3 * side, 2), P2(cut + 3 * side, -2), P2(cut, -2)]
                let arcs = Contour.levels(of: ring, [0.25], over: square, trimLoops: [half])[0].paths
                let ends = arcs.first.map { [$0[0], $0[$0.count - 1]].map { abs($0.x - cut) } }
                // Within a rounding of the cut an end may lie either side.
                let kept = arcs.allSatisfy { $0.dropFirst().dropLast().allSatisfy { side * ($0.x - cut) >= 0 } }
                let step = arcs.joined().isEmpty ? 0 : arcs.map { zip($0, $0.dropFirst())
                    .map { simd_length($1.v - $0.v) }.max() ?? 0 }.max()!
                Check.expect(arcs.count == 1 && (ends?.max() ?? 1) < 1e-15 && kept && step < 2 * cell,
                             "the \(name) of x = \(cut) on a ring is one arc, ended on the cut",
                             "\(arcs.count) arcs, ends \(ends ?? []) off, longest step \(step / cell) cells")
            }
        }

        // A closed contour across the seam whose closing vertex lies a period
        // past its first, cut where its membership changes, through its
        // vertices: v = 1 on the torus, from u = 0 to 2π, where sin 2u is 0
        // and -4.9e-16. Kept where sin 2u >= 0 it is two quarters, not one
        // run with a chord across the quarter between; so too where the loop
        // begins inside the domain, or outside it. On a torus no loop bounds
        // anything even-odd, so the domain is `contains`, and the cuts are
        // lines of u, closed a period along v and met a period along u too.
        let torus = (u: ParamAxis.angle(samples: 40), v: ParamAxis.angle(samples: 30))
        let quarters: [(String, @Sendable (Double) -> Double, [Double])] = [
            ("sin 2u >= 0, the loop beginning on its boundary", { sin(2 * $0) }, [0, 0.5, 1, 1.5]),
            ("cos 2u >= 0, beginning inside", { cos(2 * $0) }, [0.25, 0.75, 1.25, 1.75]),
            ("-cos 2u >= 0, beginning outside", { -cos(2 * $0) }, [0.25, 0.75, 1.25, 1.75]),
        ]
        for (name, g, zeros) in quarters {
            let lines: [[P2<ParamSpace>]] = zeros.map { [P2($0 * .pi, 0), P2($0 * .pi, 2 * .pi)] }
            let runs = Contour.levels(of: ScalarField(period: 2 * .pi) { $0.y }, [1], over: torus,
                                      contains: { g($0.x) >= 0 }, cuts: lines)[0].paths
            let length = runs.map { zip($0, $0.dropFirst()).map { simd_length($1.v - $0.v) }.reduce(0, +) }
                .reduce(0, +)
            let ends = runs.flatMap { [$0[0], $0[$0.count - 1]] }.map { abs(g($0.x)) }.max() ?? .infinity
            // A boundary that falls on a vertex ends the run there, once.
            let repeated = runs.contains { zip($0, $0.dropFirst()).contains { $0 == $1 } }
            Check.expect(runs.count == 2 && abs(length - .pi) < 1e-9 && ends < 1e-14 && !repeated,
                         "v = 1 kept where \(name) is two quarters, ended on their boundaries",
                         "\(runs.count) runs, \(length / .pi) π long, ends \(ends) off\(repeated ? ", a vertex repeated" : "")")
        }

        // The same lines a quarter cell off the vertices, so each cuts a
        // segment: u = π/2 + π/80 and so on, met a period along at the seam.
        let shifted: [Double] = [0.5, 1, 1.5, 2].map { (k: Double) in k * .pi + .pi / 80 }
        let offLines: [[P2<ParamSpace>]] = shifted.map { [P2($0, 0), P2($0, 2 * .pi)] }
        let offRuns = Contour.levels(of: ScalarField(period: 2 * .pi) { $0.y }, [1], over: torus,
                                     contains: { sin(2 * ($0.x - .pi / 80)) >= 0 }, cuts: offLines)[0].paths
        func offLine(_ x: Double) -> Double { shifted.map { abs(remainder(x - $0, 2 * .pi)) }.min()! }
        let offEnds = offRuns.flatMap { [$0[0].x, $0[$0.count - 1].x] }.map(offLine).max() ?? .infinity
        Check.expect(offRuns.count == 2 && offEnds < 1e-14,
                     "and cut between vertices, a quarter cell off them, it is two quarters ended on the lines",
                     "\(offRuns.count) runs, ends \(offEnds) off")
    }
}
