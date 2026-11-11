import Foundation
import simd
import KurvenCore
import KurvenLandscape

/// The plain routine's chords and cuts on the graph, held to closed forms: a
/// contour cut by the truncation ends on the truncation edge, not on a chord
/// of it, and no vertex stands over the truncation; a chord between two
/// vertices stays within the tolerance of its level set, and of the graph.
func chordTests() {
    let square = Domain(real: Interval(lo: -2, hi: 2), imag: Interval(lo: -2, hi: 2))
    func landscape(_ expression: String, resolution: Int, truncation: Truncation) throws -> KurvenBundle {
        try NativeLandscape.build(LandscapeRequest(expression: expression, domain: square, resolution: resolution,
                                                   truncation: truncation, layers: []), refine: true)
    }
    func onDomainEdge(_ p: P3<WorldSpace>) -> Bool { abs(abs(p.x) - 2) <= 1e-12 || abs(abs(p.y) - 2) <= 1e-12 }

    Check.suite("chords: a contour on a truncated graph ends on the truncation edge, and no vertex stands over it") {
        // 1/(z² − 1) cut at 2/3: the plateau is the Cassini oval |z² − 1| <
        // 3/2, a peanut, convex at its ends and pinched at its waist, so the
        // truncation edge's chords lie over the plateau at the ends and over
        // the graph at the waist. A cut on a chord stood over the plateau at
        // the ends and short of the edge at the waist.
        let h = 2.0 / 3
        let built = try landscape("1/(z^2 - 1)", resolution: 32, truncation: .uniform(h))
        let surface = built.surface, refine = built.refine
        let context = Surface.HatchContext(built.manifest.occluder)
        func f(_ p: P3<WorldSpace>) -> Double {
            1 / hypot(p.x * p.x - p.y * p.y - 1, 2 * p.x * p.y)
        }
        let light = SIMD3<Double>(-0.4, -0.6, 0.7)
        let lines = [-1.5, -1.2, -0.5, -0.05, 0.03, 0.1, 0.4, 1.3]
        // Each family, and how many of its ends must lie inside the domain:
        // the height's level sets under h are closed curves, and meet no
        // edge; a half-Lambert piece also ends where its width changes.
        let families: [(String, [[P3<WorldSpace>]], ends: Int?)] = [
            ("u", surface.contours(of: .u, lines, on: .graph(tiled: false),
                                   restriction: .all, in: context, refine: refine), 16),
            ("v", surface.contours(of: .v, [-0.6, 0.2, 0.5], on: .graph(tiled: false),
                                   restriction: .all, in: context, refine: refine), 6),
            ("the height", surface.contours(of: .height, [0.4, 0.6], on: .graph(tiled: false),
                                            restriction: .all, in: context, refine: refine), 0),
            ("isophotes", surface.contours(of: .isophote, [-0.5, -0.25, 0, 0.25, 0.5, 0.75], on: .graph(tiled: false),
                                           restriction: .all, in: context, refine: refine, light: light), 10),
            ("hatching", surface.graphHatching(Hatching(angle: 0, spacing: 0.1), tiled: false, in: context,
                                               refine: refine).flatMap(\.paths), 30),
            ("half-Lambert hatching", surface.graphHatching(
                Hatching(angle: .pi / 2, spacing: 0.1, tone: .halfLambert(light: light), widthSteps: 4),
                tiled: false, in: context, refine: refine).flatMap(\.paths), nil),
        ]
        for (name, paths, wanted) in families {
            var over = 0.0, offEdge = 0.0, offLine = 0.0, ends = 0, vertices = 0
            for path in paths {
                for p in path {
                    vertices += 1
                    over = max(over, max(f(p), p.z) - h)
                }
                // A line of u is cut on its own line: its ends are where
                // Re z = c meets |z² − 1| = 3/2.
                if name == "u" { offLine = max(offLine, path.map { p in lines.map { abs(p.x - $0) }.min()! }.max()!) }
                guard wanted != nil, path[0] != path[path.count - 1] else { continue }
                for end in [path[0], path[path.count - 1]] where !onDomainEdge(end) {
                    ends += 1
                    offEdge = max(offEdge, abs(f(end) - h))
                }
            }
            Check.expect(over <= 1e-12 * h && offEdge <= 1e-12 * h && offLine <= 1e-15 && ends >= wanted ?? 0,
                         "\(name): no vertex over the truncation" + (wanted == nil ? "" : ", every end inside the domain on |f| = 2/3"),
                         String(format: "%d vertices, worst %.1e over; %d ends, worst %.1e off", vertices, over, ends, offEdge))
        }
    }

    Check.suite("chords: the plain routine's chords stay within its tolerance of the level set, and of the graph") {
        let tolerance = Contour.chordTolerance
        // The height on the graph of z, a cone: its level sets are circles
        // |z| = r, and the gradient of |z| has unit length, so a midpoint's
        // distance from its circle is ||m| − r|, which the routine estimates
        // exactly.
        let cone = try landscape("z", resolution: 16, truncation: .none)
        let cell = 4.0 / 15
        let radii = [0.4, 0.7, 1.3, 1.9]
        var vertexOff = 0.0, chordOff = 0.0, chords = 0
        for (k, r) in radii.enumerated() {
            for path in cone.surface.contours(of: .height, [r], on: .graph(tiled: false), restriction: .all,
                                              in: Surface.HatchContext(cone.manifest.occluder), refine: cone.refine) {
                vertexOff = max(vertexOff, path.map { abs(hypot($0.x, $0.y) - radii[k]) }.max()!)
                for (a, b) in zip(path, path.dropFirst()) {
                    chords += 1
                    chordOff = max(chordOff, abs(hypot(0.5 * (a.x + b.x), 0.5 * (a.y + b.y)) - r))
                }
            }
        }
        Check.expect(vertexOff <= 1e-12 && chordOff <= tolerance * cell && chords > 40,
                     "the height on the graph of z: every chord's midpoint within \(tolerance) of a cell of its circle",
                     String(format: "%d chords, worst %.4f of a cell; vertices off by %.1e", chords, chordOff / cell, vertexOff))

        // The graph of z², |f| = x² + y²: a line of u = c lies on the parabola
        // z = c² + y², and a chord from y₁ to y₂ sags (y₁ − y₂)²/4 off it,
        // a fifteenth of a cell across a whole cell. Lit along z, n · l =
        // 1/√(1 + 4r²), so the isophote at c is the circle r = √(1/c² − 1)/2.
        let bowl = try landscape("z^2", resolution: 16, truncation: .none)
        let context = Surface.HatchContext(bowl.manifest.occluder)
        var sag = 0.0, closedSag = 0.0
        chords = 0
        for path in bowl.surface.contours(of: .u, [-1.3, -0.5, 0.2, 0.9], on: .graph(tiled: false), restriction: .all,
                                          in: context, refine: bowl.refine) {
            for (a, b) in zip(path, path.dropFirst()) {
                chords += 1
                let m = 0.5 * (a.v + b.v)
                sag = max(sag, abs(m.z - (m.x * m.x + m.y * m.y)))
                closedSag = max(closedSag, (a.y - b.y) * (a.y - b.y) / 4)
            }
        }
        Check.expect(sag <= tolerance * cell && abs(sag - closedSag) <= 1e-12 && chords > 40,
                     "a line of u on the graph of z²: every chord's midpoint within \(tolerance) of a cell of the graph",
                     String(format: "%d chords, worst sag %.4f of a cell", chords, sag / cell))
        let levels = [0.3, 0.5, 0.7, 0.9]
        var isophoteOff = 0.0
        chords = 0
        let graph = bowl.surface.graph(refine: bowl.refine)!
        for (k, paths) in graph.isophotes(light: SIMD3(0, 0, 1), levels, on: graph.copies[0]).map(\.paths).enumerated() {
            let r = (1 / (levels[k] * levels[k]) - 1).squareRoot() / 2
            for path in paths {
                for (a, b) in zip(path, path.dropFirst()) {
                    chords += 1
                    isophoteOff = max(isophoteOff, abs(simd_length(0.5 * (a.v + b.v)) - r))
                }
            }
        }
        Check.expect(isophoteOff <= tolerance * cell && chords > 40,
                     "the isophotes of the graph of z² lit along z: every chord's midpoint within \(tolerance) of a cell of its circle",
                     String(format: "%d chords, worst %.4f of a cell", chords, isophoteOff / cell))
    }

    Check.suite("chords: cuts on a domain whose edge the loops follow as chords: joined, carried and read between vertices") {
        // The unit disk, |p| − 1 its excess, and an octagon inscribed in its
        // circle, turned so a chord's midpoint lies at angle 0: the loops
        // follow the circle as chords, which lie inside it, a gap between
        // each chord and the arc over it.
        let axis = ParamAxis(Interval(lo: -2, hi: 2), periodic: false, samples: 32)
        let octagon = (0...8).map { k in
            P2<ParamSpace>(cos((Double(k % 8) + 0.5) * .pi / 4), sin((Double(k % 8) + 0.5) * .pi / 4))
        }
        let excess: @Sendable (P2<ParamSpace>) -> Double = { simd_length($0.v) - 1 }
        func clip(_ path: [SIMD2<Double>], loops: [[P2<ParamSpace>]]? = nil) -> [[P2<ParamSpace>]] {
            Contour.clipped([(0, [path.map(P2<ParamSpace>.init)])], over: (axis, axis), trimLoops: loops ?? [octagon],
                            excess: excess)[0].paths
        }
        func circle(_ c: SIMD2<Double>, _ r: Double, from phi: Double) -> [SIMD2<Double>] {
            (0...400).map { k in c + r * SIMD2(cos(phi + 2 * .pi * Double(k % 400) / 400), sin(phi + 2 * .pi * Double(k % 400) / 400)) }
        }
        /// The worst excess of a run's vertices, and of its ends off 0.
        func worst(_ runs: [[P2<ParamSpace>]]) -> (over: Double, ends: Double) {
            (runs.joined().map { excess($0) }.max() ?? -1,
             runs.filter { $0[0] != $0[$0.count - 1] }.flatMap { [$0[0], $0[$0.count - 1]] }.map { abs(excess($0)) }.max() ?? 0)
        }
        // Out through the top chord, along the gap over it, and back: one run.
        let through: [SIMD2<Double>] = [SIMD2(-0.5, 0), SIMD2(-0.2, 0.95), SIMD2(0.2, 0.95), SIMD2(0.5, 0)]
        let joined = clip(through)
        // The same closed, beginning in the gap.
        let loop: [SIMD2<Double>] = [SIMD2(0, 0.95), SIMD2(0.2, 0.95), SIMD2(0.5, 0), SIMD2(0, -0.5), SIMD2(-0.5, 0),
                                     SIMD2(-0.2, 0.95), SIMD2(0, 0.95)]
        let joinedRound = clip(loop)
        // Round the closed one, the run begins where it was cut, on the
        // path, and ends there again.
        let roundKept = joinedRound.count == 1 && joinedRound[0][0] == joinedRound[0][joinedRound[0].count - 1]
            && Set(loop.map { [$0.x, $0.y] }).isSubset(of: Set(joinedRound[0].map { [$0.x, $0.y] }))
            && joinedRound[0].count <= loop.count + 1
        Check.expect(joined.count == 1 && joined[0].map(\.v) == through && roundKept,
                     "a path the loops cut where it runs between a chord and the circle is one run, across a closed path's start too",
                     "\(joined.map(\.count)) and \(joinedRound.map(\.count)) vertices")
        // A circle round (1.2, 0) beginning in the gap: its arc inside the
        // disk is one run, ending on the circle at both ends, carried round
        // the path's start to reach it.
        let carried = clip(circle(SIMD2(1.2, 0), 0.3, from: .pi - acos(0.25 / 0.3)))
        // One round (1.3, 0) whose arc inside the disk lies wholly in the
        // gap, crossing no chord.
        let inGap = clip(circle(SIMD2(1.3, 0), 0.35, from: 0))
        Check.expect(carried.count == 1 && inGap.count == 1 && worst(carried + inGap).over < 0
                        && worst(carried + inGap).ends <= 1e-15,
                     "an arc ends on the circle, carried round a closed path's start, and one wholly in a gap is kept",
                     String(format: "%d and %d runs, ends off the circle by %.1e", carried.count, inGap.count,
                            worst(carried + inGap).ends))
        // A chord between two vertices outside the disk through its middle
        // keeps its middle: the domain read at the midpoint, between the
        // vertices, as a square's loops hold it all.
        let square: [P2<ParamSpace>] = [P2(-1.5, -1.5), P2(1.5, -1.5), P2(1.5, 1.5), P2(-1.5, 1.5), P2(-1.5, -1.5)]
        let across = clip([SIMD2(-1.2, 0), SIMD2(1.2, 0)], loops: [square])
        Check.expect(across.count == 1 && across[0].allSatisfy { $0.y == 0 } && abs(across[0][0].x + 1) <= 1e-15
                        && abs(across[0][across[0].count - 1].x - 1) <= 1e-15,
                     "a chord through the domain between two vertices outside it keeps the stretch inside, ended on the circle",
                     "\(across.map { $0.map(\.v) })")
    }

    Check.suite("chords: on the graph of sin z, every isophote cut on the truncation edge, every chord near its level, no run split") {
        // |sin z|, and the graph's unit normal in closed form: for analytic
        // f, ∇|f| = (Re f̄f′, −Im f̄f′)/|f|.
        func f(_ p: SIMD2<Double>) -> (re: Double, im: Double) { (sin(p.x) * cosh(p.y), cos(p.x) * sinh(p.y)) }
        func magnitude(_ p: SIMD2<Double>) -> Double { let w = f(p); return hypot(w.re, w.im) }
        func lit(_ p: SIMD2<Double>, _ l: SIMD3<Double>) -> Double {
            let w = f(p), d = (re: cos(p.x) * cosh(p.y), im: -sin(p.x) * sinh(p.y)), m = hypot(w.re, w.im)
            let g = SIMD2(w.re * d.re + w.im * d.im, -(w.re * d.im - w.im * d.re)) / m
            return simd_dot(SIMD3(-g.x, -g.y, 1) / (1 + simd_length_squared(g)).squareRoot(), simd_normalize(l))
        }
        let window = Domain(real: Interval(lo: -5, hi: 5), imag: Interval(lo: -1.5, hi: 1.5))
        func onEdge(_ p: P2<ParamSpace>) -> Bool { abs(abs(p.x) - 5) <= 1e-12 || abs(abs(p.y) - 1.5) <= 1e-12 }
        /// Pairs of runs where one ends where the next begins: one curve split.
        func abutting(_ runs: [[P2<ParamSpace>]]) -> Int {
            let starts = Set(runs.map { [$0[0].x, $0[0].y] })
            return runs.filter { $0[0] != $0[$0.count - 1] && starts.contains([$0[$0.count - 1].x, $0[$0.count - 1].y]) }.count
        }
        let h = 1.5
        // At 40 samples a cut lies within 1e-4 of the next vertex, where its
        // own crossing rounds past the loops' filter.
        let coarse = try NativeLandscape.build(LandscapeRequest(expression: "sin(z)", domain: window, resolution: 40,
                                                                truncation: .uniform(h), layers: []), refine: true)
        let graph = coarse.surface.graph(refine: coarse.refine)!
        let light = SIMD3(cos(Double.pi / 8) * cos(1.1), sin(Double.pi / 8) * cos(1.1), sin(1.1))
        var over = 0.0, offEdge = 0.0, ends = 0, split = 0
        for (_, paths) in graph.isophotes(light: light, stride(from: -0.95, through: 0.95, by: 0.05).map { $0 }, on: graph.copies[0]) {
            split += abutting(paths)
            for path in paths {
                over = max(over, path.map { magnitude($0.v) - h }.max()!)
                guard path[0] != path[path.count - 1] else { continue }
                for end in [path[0], path[path.count - 1]] where !onEdge(end) {
                    ends += 1
                    offEdge = max(offEdge, abs(magnitude(end.v) - h))
                }
            }
        }
        Check.expect(over <= 1e-12 * h && offEdge <= 1e-12 * h && ends > 50 && split == 0,
                     "every isophote of sin z at 40 samples is under h, each cut inside the domain on |sin z| = 3/2, none split",
                     String(format: "%d ends, worst %.1e off; worst %.1e over; %d split", ends, offEdge, over, split))
        // At 32 samples under a low light, a chord beside an extremum of n · l,
        // where Newton's step points away from the level set across it.
        let fine = try NativeLandscape.build(LandscapeRequest(expression: "sin(z)", domain: window, resolution: 32,
                                                              truncation: .uniform(h), layers: []), refine: true)
        let lowGraph = fine.surface.graph(refine: fine.refine)!
        let low = SIMD3<Double>(0.8, 0.1, 0.5), cell = 10.0 / 31
        // A vertex is held to the field it was placed on, n · l through the
        // graph's own jet, whose normal is f's by central differences; a
        // chord's midpoint to the closed form, the distance estimated as the
        // residual over the gradient, as the routine estimates it, so to
        // within a few percent of the tolerance. A cut on the truncation
        // edge lies on its chord, not on the level. At a zero of f the graph
        // is a cone, whose isophotes all meet at its apex, where n · l jumps:
        // a chord within a quarter of a cell of one is not measured.
        let lowCopy = lowGraph.copies[0]
        var chordOff = 0.0, vertexOff = 0.0, chords = 0, worstAt = SIMD2<Double>(0, 0)
        for (c, paths) in lowGraph.isophotes(light: low, [0.3, 0.6, 0.75, 0.9], on: lowCopy) {
            for path in paths {
                for v in path where abs(magnitude(v.v) - h) > 1e-9 {
                    let n = lowGraph.outwardNormal(at: v, on: lowCopy)!
                    vertexOff = max(vertexOff, abs(simd_dot(n, simd_normalize(low)) - c))
                }
                for (a, b) in zip(path, path.dropFirst()) {
                    chords += 1
                    let m = 0.5 * (a.v + b.v), step = 1e-6
                    guard hypot(m.x - (m.x / .pi).rounded() * .pi, m.y) > 0.25 * cell else { continue }
                    let g = SIMD2(lit(m + SIMD2(step, 0), low) - lit(m - SIMD2(step, 0), low),
                                  lit(m + SIMD2(0, step), low) - lit(m - SIMD2(0, step), low)) / (2 * step)
                    let off = abs(lit(m, low) - c) / simd_length(g)
                    if off > chordOff { chordOff = off; worstAt = m }
                }
            }
        }
        Check.expect(chordOff <= Contour.chordTolerance * cell * 1.05 && vertexOff <= 1e-9 && chords > 200,
                     "every isophote of sin z at 32 samples under a low light: each vertex on its level, each chord's midpoint near it",
                     String(format: "%d chords, worst %.4f of a cell at (%.4f, %.4f); vertices off by %.1e",
                            chords, chordOff / cell, worstAt.x, worstAt.y, vertexOff))
        // Γ in bands, its closed isophote round a pole cut by the truncation
        // alone: one run, not two meeting where the path began.
        let gamma = try NativeLandscape.build(
            LandscapeRequest(expression: "gamma(z)", domain: Domain(real: Interval(lo: -4.5, hi: 4.5), imag: Interval(lo: -2.5, hi: 2.5)),
                             resolution: 120,
                             truncation: .realBands([RealBand(below: -3.5, height: 2.5), RealBand(below: -2.5, height: 3),
                                                     RealBand(below: -1.5, height: 4), RealBand(below: 0.5, height: 5.5)], beyond: 4),
                             layers: []), refine: true)
        let gammaGraph = gamma.surface.graph(refine: gamma.refine)!
        let gammaSplit = gammaGraph.isophotes(light: SIMD3(-0.4, -0.6, 0.7), [-0.6, -0.3, 0.1, 0.5, 0.85],
                                              on: gammaGraph.copies[0]).map { abutting($0.paths) }
        Check.expect(gammaSplit.allSatisfy { $0 == 0 }, "no isophote of Γ in bands is split where its path began",
                     "runs abutting per level \(gammaSplit)")
    }
}
