import Foundation
import simd
import KurvenCore
import KurvenLandscape

/// Contours of a face's coordinates and height on a landscape's faces, held
/// to closed forms, and what a manifest refuses of them.
func faceFieldTests() {
    // 1/(z - 0.45) over Re [-1/2, 3/2], Im [-1, 1], cut at 2 left of Re z =
    // 1/2 and 4 beyond: |f| = 1/r for r = |z - 0.45|. The graph is where 1/r
    // is under the band's height: r > 1/2 left of 1/2, r > 1/4 beyond. The
    // plateaus are the discs inside, at 2 and 4; the riser at Re z = 1/2
    // stands where 1/r > 2 there, |Im z| < √(1/4 - 1/400), from 2 up to
    // min(1/r, 4). The front wall stands on Im z = -1 under 1/√((x - 0.45)² + 1).
    let window = Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1))
    let built: KurvenBundle
    do {
        built = try NativeLandscape.build(
            LandscapeRequest(expression: "1/(z - 0.45)", domain: window, resolution: 48,
                             truncation: .realBands([RealBand(below: 0.5, height: 2)], beyond: 4)),
            refine: true)
    } catch {
        Check.suite("face fields: the tower builds") { Check.expect(false, "the tower builds", "\(error)") }
        return
    }
    let surface = built.surface, refine = built.refine
    let context = Surface.HatchContext(built.manifest.occluder)
    let g = surface.height
    let cell = max(window.real.length / Double(g.width - 1), window.imag.length / Double(g.height - 1))
    /// How far a run's end may lie from the closed form: the edges it is cut
    /// at are placed on f to the refiner's tolerance.
    let tolerance = 0.02 * cell
    func h(_ x: Double) -> Double { x < 0.5 ? 2 : 4 }
    func r(_ x: Double, _ y: Double) -> Double { hypot(x - 0.45, y) }
    func paths(_ field: ContourField, _ levels: [Double], _ faces: FaceSelector) -> [[P3<WorldSpace>]] {
        surface.contours(of: field, levels, on: faces, restriction: .all, in: context, refine: refine)
    }
    /// The stretches of [lo, hi] where `inside` holds, by bisection to the last bit.
    func stretches(_ lo: Double, _ hi: Double, _ inside: (Double) -> Bool) -> [(Double, Double)] {
        let n = 4000
        var out: [(Double, Double)] = []
        var start: Double? = inside(lo) ? lo : nil
        for k in 1...n {
            let a = lo + (hi - lo) * Double(k - 1) / Double(n), b = lo + (hi - lo) * Double(k) / Double(n)
            guard inside(a) != inside(b) else { continue }
            let at = bisect(a, b) { inside($0) == inside(a) }.holds
            if let s = start { out.append((s, at)); start = nil } else { start = at }
        }
        if let s = start { out.append((s, hi)) }
        return out
    }
    /// Each path's extent along one coordinate, in order, those that meet
    /// joined: a line across a band boundary is two plateaus' runs.
    func spans(_ ps: [[P3<WorldSpace>]], along: (P3<WorldSpace>) -> Double) -> [(Double, Double)] {
        let each = ps.map { p in (min(along(p[0]), along(p[p.count - 1])), max(along(p[0]), along(p[p.count - 1]))) }
            .sorted { $0.0 < $1.0 }
        var out: [(Double, Double)] = []
        for s in each {
            if let last = out.last, s.0 == last.1 { out[out.count - 1].1 = s.1 } else { out.append(s) }
        }
        return out
    }
    func near(_ a: [(Double, Double)], _ b: [(Double, Double)], _ within: Double) -> Bool {
        a.count == b.count && zip(a, b).allSatisfy { abs($0.0 - $1.0) <= within && abs($0.1 - $1.1) <= within }
    }
    func show(_ s: [(Double, Double)]) -> String { s.map { String(format: "[%.5f, %.5f]", $0.0, $0.1) }.joined(separator: " ") }

    Check.suite("face fields: on the graph, u and v are lines within the trimmed domain, and the height a circle of |f|") {
        for c in [-0.25, 0.25, 1.0] {
            let ps = paths(.u, [c], .graph(tiled: false))
            let off = ps.joined().map { abs($0.x - c) + abs($0.z - 1 / r($0.x, $0.y)) }.max() ?? .infinity
            let wanted = stretches(-1, 1) { y in 1 / r(c, y) < h(c) }
            let got = spans(ps) { $0.y }
            Check.expect(off <= 1e-9 && near(got, wanted, tolerance),
                         "u = \(c): the line Re z = \(c) at |f|, where |f| is under h",
                         String(format: "off by %.1e; ", off) + "\(show(got)) vs \(show(wanted))")
        }
        for c in [-0.6, 0.2] {
            let ps = paths(.v, [c], .graph(tiled: false))
            let off = ps.joined().map { abs($0.y - c) + abs($0.z - 1 / r($0.x, $0.y)) }.max() ?? .infinity
            let wanted = stretches(-0.5, 1.5) { x in 1 / r(x, c) < h(x) }
            let got = spans(ps) { $0.x }
            Check.expect(off <= 1e-9 && near(got, wanted, tolerance),
                         "v = \(c): the line Im z = \(c) at |f|, where |f| is under h, cut at the band boundary's drop too",
                         String(format: "off by %.1e; ", off) + "\(show(got)) vs \(show(wanted))")
        }
        let ring = paths(.height, [1.5], .graph(tiled: false))
        let off = ring.joined().map { abs(r($0.x, $0.y) - 1 / 1.5) + abs($0.z - 1.5) }.max() ?? .infinity
        let closed = ring.count == 1 && ring[0].first == ring[0].last
        Check.expect(closed && off <= 1e-9, "the height 3/2: the circle |z - 0.45| = 2/3 at 3/2, closed",
                     "\(ring.count) paths, " + String(format: "off by %.1e", off))
    }

    Check.suite("face fields: on a plateau, u and v are lines at its height where |f| reaches it, and the height has none") {
        for c in [0.2, -0.1] {
            let ps = paths(.v, [c], .plateaus(tiled: false))
            // A run ends on the band boundary at its own band's height.
            let exact = ps.joined().allSatisfy { $0.y == c && ($0.x == 0.5 ? $0.z == 2 || $0.z == 4 : $0.z == h($0.x)) }
            let wanted = stretches(-0.5, 1.5) { x in 1 / r(x, c) >= h(x) }
            let got = spans(ps) { $0.x }
            Check.expect(exact && near(got, wanted, 1e-9),
                         "v = \(c): the line Im z = \(c) at h, on the level and the height exactly, ending where |f| = h",
                         "\(show(got)) vs \(show(wanted))")
        }
        for c in [0.3, 0.6] {
            let ps = paths(.u, [c], .plateaus(tiled: false))
            let exact = ps.joined().allSatisfy { $0.x == c && $0.z == h(c) }
            let half = (1 / (h(c) * h(c)) - (c - 0.45) * (c - 0.45)).squareRoot()
            let got = spans(ps) { $0.y }
            Check.expect(exact && near(got, [(-half, half)], 1e-9),
                         "u = \(c): the line Re z = \(c) at h, across the disc 1/r ≥ h",
                         "\(show(got)) vs ±\(half)")
        }
        Check.expect(paths(.height, [2, 4], .plateaus(tiled: false)).isEmpty, "the height is constant on a plateau: no contour")
    }

    Check.suite("face fields: on a wall, u is vertical from the base to the top edge, and the height horizontal under it") {
        let front = FaceSelector.walls(edges: [0], endLines: false, base: 0, topOffset: 0)
        func top(_ x: Double) -> Double { 1 / r(x, -1) }
        for s in [0.3, 1.1] {
            let ps = paths(.u, [s], front)
            let x = -0.5 + s
            let ok = ps.count == 1 && ps[0].count == 2 && ps[0].allSatisfy { abs($0.x - x) <= 1e-12 && $0.y == -1 }
            let ends = ps.first.map { ($0.map(\.z).min()!, $0.map(\.z).max()!) } ?? (.nan, .nan)
            Check.expect(ok && ends.0 == 0 && abs(ends.1 - top(x)) <= tolerance,
                         "u = \(s): the vertical at Re z = \(x) from the base to |f|",
                         "\(ps.map(\.count)) vertices, from \(ends.0) to \(ends.1), the top \(top(x))")
        }
        let ps = paths(.height, [0.8], front)
        let level = ps.joined().allSatisfy { abs($0.z - 0.8) <= 1e-12 && $0.y == -1 }
        let wanted = stretches(-0.5, 1.5) { top($0) >= 0.8 }
        let got = spans(ps) { $0.x }
        Check.expect(level && near(got, wanted, tolerance), "the height 0.8: horizontal, where the top edge stands over it",
                     "\(show(got)) vs \(show(wanted))")
        Check.expect(paths(.v, [0.8], front).map { $0.map(\.z) } == ps.map { $0.map(\.z) },
                     "v is the height on a wall")
    }

    Check.suite("face fields: on a riser, u is vertical from its bottom to its top, and the height horizontal under its top") {
        let risers = FaceSelector.risers(tiled: false, endLines: false)
        func lower(_ p: P3<WorldSpace>) -> Bool { p.x < 0.5 && 0.5 - p.x < 2e-9 * cell }
        // s runs along Re z = 1/2 from Im z = -1: s = 1 is Im z = 0, where
        // the riser stands from 2 to 4.
        let ps = paths(.u, [1], risers)
        let ok = ps.count == 1 && ps[0].allSatisfy { lower($0) && abs($0.y) <= 1e-12 }
        let ends = ps.first.map { ($0.map(\.z).min()!, $0.map(\.z).max()!) } ?? (.nan, .nan)
        Check.expect(ok && ends == (2, 4), "u = 1: the vertical at Im z = 0, from the lower plateau to the higher, a hair onto the lower side",
                     "\(ps.count) paths, from \(ends.0) to \(ends.1)")
        let level = paths(.height, [3], risers)
        let flat = level.joined().allSatisfy { lower($0) && abs($0.z - 3) <= 1e-12 }
        let wanted = stretches(-1, 1) { y in 1 / r(0.5, y) >= 3 }
        let got = spans(level) { $0.y }
        Check.expect(flat && near(got, wanted, tolerance), "the height 3: horizontal, where the riser's top stands over it",
                     "\(show(got)) vs \(show(wanted))")
        Check.expect(paths(.height, [1.5, 4.5], risers).isEmpty, "a level under the riser or over it: none")
    }

    Check.suite("face fields: a manifest refuses a field a face does not expose, and hatch-line placement on a contour") {
        func decode(_ source: LayerSource) throws { _ = try LayerSource(json: source.json) }
        for faces in [FaceSelector.plateaus(tiled: false), .walls(edges: [0], endLines: false, base: 0, topOffset: 0),
                      .risers(tiled: false, endLines: false)] {
            for field in [ContourField.magnitude, .phase] {
                Check.expectThrows("refuses \(field.rawValue) on the \(faces.name)") {
                    try decode(.contour(field: field, levels: [1], restriction: .all, faces: faces))
                }
            }
        }
        for faces in [FaceSelector.surface, .solid] {
            Check.expectThrows("refuses a contour on a \(faces.name)'s faces: a manifest is a landscape's") {
                try decode(.contour(field: .u, levels: [1], restriction: .all, faces: faces))
            }
        }
        Check.expectThrows("refuses a contour on walls with end lines") {
            try decode(.contour(field: .u, levels: [1], restriction: .all,
                                faces: .walls(edges: [0], endLines: true, base: 0, topOffset: 0)))
        }
        Check.expectThrows("refuses hatching on the graph, which is curved") {
            try decode(.hatching(Hatching(angle: 0, spacing: 0.1), faces: .graph(tiled: false)))
        }
        var accepted = 0
        for faces in [FaceSelector.graph(tiled: true), .plateaus(tiled: false),
                      .walls(edges: [1, 2], endLines: false, base: 0, topOffset: 0), .risers(tiled: true, endLines: false)] {
            for field in [ContourField.u, .v, .height] {
                let source = LayerSource.contour(field: field, levels: [0.5], restriction: .all, faces: faces)
                if (try? LayerSource(json: source.json)) == source { accepted += 1 }
            }
        }
        Check.expect(accepted == 12, "u, v and the height on the graph, the plateaus, walls and risers round-trip",
                     "\(accepted) of 12")
    }
}
