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
}
