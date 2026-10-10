import Foundation
import simd
import KurvenCore
import KurvenLandscape

// MARK: - the crest of a cut face, placed on f

/// The wall outline's crest between the grid's nodes: the grid's chord without
/// a refiner, f's own profile with one, down to the zero the phase lines fan to.
func crestTests() {
    /// The crest of the front edge (y at the domain's low imaginary bound) of
    /// a bundle's wall outline: the one stroke there whose height varies.
    func frontCrest(_ bundle: KurvenBundle) throws -> [P3<WorldSpace>] {
        let y0 = bundle.manifest.domain.imag.lo
        let outline = try bundle.layer("wall_outline").paths
        var best: [P3<WorldSpace>] = []
        for i in 0..<outline.count {
            let p = Array(outline[path: i])
            guard p.allSatisfy({ abs($0.y - y0) < 1e-9 }), p.count >= 2,
                  Set(p.map { $0.z }).count > 1, p.count > best.count else { continue }
            best = p
        }
        return best
    }

    Check.suite("crest to the zero: the refined crest gains the minimum of |f| between two nodes") {
        // f(z) = z over Re [-1, 1], Im [0, 1] on an 8 x 8 grid: the zero at 0
        // lies on the front edge between the nodes at -1/7 and 1/7, where the
        // grid's chord bottoms out at 1/7 and |z| itself reaches the floor.
        let domain = Domain(real: Interval(lo: -1, hi: 1), imag: Interval(lo: 0, hi: 1))
        let request = LandscapeRequest(expression: "z", domain: domain, resolution: 8,
                                       caps: .uniform(10))
        let plain = try NativeLandscape.build(request, refine: false)
        let refined = try NativeLandscape.build(request, refine: true)
        let shape = refined.manifest.height.shape
        Check.expect(shape.nx == 8 && shape.ny == 8, "the grid is 8 x 8", "\(shape.nx) x \(shape.ny)")

        let chord = try frontCrest(plain), crest = try frontCrest(refined)
        let chordLow = chord.min { $0.z < $1.z }!, crestLow = crest.min { $0.z < $1.z }!
        Check.expect(abs(chordLow.z - 1.0 / 7) < 1e-6 && abs(abs(chordLow.x) - 1.0 / 7) < 1e-9,
                     "from the grid alone the crest bottoms out at the node beside the zero",
                     String(format: "lowest (%.4f, %.4f)", chordLow.x, chordLow.z))
        Check.expect(abs(crestLow.z) < 1e-9 && abs(crestLow.x) < 1e-9,
                     "with a refiner it reaches the zero, at the zero",
                     String(format: "lowest (%.2e, %.2e)", crestLow.x, crestLow.z))
        Check.expect(crest.count > chord.count, "vertices were added and none removed",
                     "\(chord.count) -> \(crest.count)")
        // The nodes are where they were, the zero is the one new vertex, and
        // the vertices run along the edge in order.
        let nodesKept = chord.allSatisfy { v in crest.contains { abs($0.x - v.x) < 1e-12 && abs($0.z - v.z) < 1e-12 } }
        Check.expect(nodesKept, "the grid's nodes are still vertices, at the grid's heights")
        Check.expect(crest.count == chord.count + 1, "and the zero is the one vertex added",
                     "\(chord.count) -> \(crest.count)")
        let ordered = zip(crest, crest.dropFirst()).allSatisfy { $0.x < $1.x }
                   || zip(crest, crest.dropFirst()).allSatisfy { $0.x > $1.x }
        Check.expect(ordered, "the vertices run along the edge in order",
                     crest.map { String(format: "%.3f", $0.x) }.joined(separator: " "))

        // The phase lines end at the zero, at the zero's height: the fan meets
        // the crest on the floor rather than a cell's rise above it.
        func ends(_ b: KurvenBundle, _ name: String) throws -> [P3<WorldSpace>] {
            let paths = try b.layer(name).paths
            return (0..<paths.count).flatMap { [paths[path: $0].first!, paths[path: $0].last!] }
                .filter { abs($0.x) < 1e-6 && abs($0.y) < 1e-6 }
        }
        let fan = try ends(refined, "ang_major") + ends(refined, "ang_minor")
        Check.expect(fan.count >= 4, "the refined phase lines end on the zero", "\(fan.count) ends")
        Check.expect(fan.allSatisfy { abs($0.z) < 1e-9 }, "at height zero",
                     String(format: "highest %.2e", fan.map(\.z).max() ?? -1))
        // The grid's own phase lines reach x = 0 too, by interpolating the
        // phase across the zero, and are lifted to the chord over it.
        let plainFan = try ends(plain, "ang_major") + ends(plain, "ang_minor")
        Check.expect(!plainFan.isEmpty && plainFan.allSatisfy { abs($0.z - 1.0 / 7) < 1e-6 },
                     "where the grid's own phase lines hang a cell's rise above the floor",
                     String(format: "%d ends at z = %.4f", plainFan.count, plainFan.first?.z ?? -1))

        // The hatch of the front wall stands under the crest as drawn: on the
        // V down to the zero, not on the chord over it.
        func tops(_ b: KurvenBundle) throws -> [(x: Double, z: Double)] {
            let hatch = try b.layer("wall_hatch").paths
            return (0..<hatch.count).compactMap { i in
                let p = hatch[path: i]
                guard p.allSatisfy({ abs($0.y) < 1e-9 }) else { return nil }
                return (p.first!.x, p.map(\.z).max()!)
            }
        }
        func onCrest(_ tops: [(x: Double, z: Double)], _ crest: [P3<WorldSpace>]) -> Double {
            tops.map { t in
                let (a, b) = zip(crest, crest.dropFirst()).first { min($0.x, $1.x) - 1e-12 <= t.x && t.x <= max($0.x, $1.x) + 1e-12 }!
                let u = b.x == a.x ? 0 : (t.x - a.x) / (b.x - a.x)
                return abs(t.z - (a.z + (b.z - a.z) * u))
            }.max() ?? 1
        }
        let refinedTops = try tops(refined), plainTops = try tops(plain)
        Check.expect(!refinedTops.isEmpty && onCrest(refinedTops, crest) < 1e-9,
                     "the front wall's hatch strokes rise to the refined crest",
                     String(format: "%d strokes, worst %.1e", refinedTops.count, onCrest(refinedTops, crest)))
        Check.expect(onCrest(plainTops, chord) < 1e-6, "and to the grid's without a refiner",
                     String(format: "worst %.1e", onCrest(plainTops, chord)))
        let inTheV = refinedTops.filter { abs($0.x) < 1.0 / 7 }
        Check.expect(!inTheV.isEmpty && inTheV.allSatisfy { $0.z < 1.0 / 7 - 1e-9 },
                     "so the strokes in the pit stop under the V, not at the chord", "\(inTheV.count) strokes")
    }

    Check.suite("crest to the zero: the fold lines run down to the zero as well") {
        // A steeper cone, 10|z|, so that its far flank turns away from the
        // plate camera: its silhouettes are two generators running into the
        // apex, which the lattice loses a cell short of the floor.
        let domain = Domain(real: Interval(lo: -1, hi: 1), imag: Interval(lo: 0, hi: 1))
        let request = LandscapeRequest(expression: "10*z", domain: domain, resolution: 8,
                                       caps: .uniform(20))
        let plain = try NativeLandscape.build(request, refine: false)
        let refined = try NativeLandscape.build(request, refine: true)
        func ends(_ b: KurvenBundle) -> [P3<WorldSpace>] {
            let scene = Scene(bundle: b, preset: b.manifest.presets[0])
            let folds = scene.heightfield!.foldLines(view: scene.camera.view)
            return (0..<folds.count).flatMap { [folds[path: $0].first!, folds[path: $0].last!] }
        }
        let carried = ends(refined).filter { abs($0.x) < 1e-9 && abs($0.y) < 1e-9 }
        Check.expect(carried.count >= 1 && carried.allSatisfy { abs($0.z) < 1e-9 },
                     "with a refiner a fold ends at the zero, on the floor", "\(carried.count) ends there")
        let lowest = ends(plain).map(\.z).min() ?? -1
        Check.expect(lowest > 0.05, "which the lattice's own folds stop short of",
                     String(format: "lowest end at z = %.3f", lowest))
    }

    Check.suite("crest to the zero: the cap crossings stay the grid's, so the crest meets the rim") {
        // 1/z over Re [-1, 1], Im [0.1, 1.1] under a cap of 2: the front edge
        // crosses the cap twice, at x = ±sqrt(0.24). The rim is contoured on
        // the grid, so the crest's crossing is solved on the grid too, and the
        // two coincide; f's own crossing is a few hundredths to one side.
        let domain = Domain(real: Interval(lo: -1, hi: 1), imag: Interval(lo: 0.1, hi: 1.1))
        let request = LandscapeRequest(expression: "1/z", domain: domain, resolution: 8,
                                       caps: .uniform(2))
        let refined = try NativeLandscape.build(request, refine: true)
        let crest = try frontCrest(refined)
        let rim = try refined.layer("cap_outline").paths
        var rimEnds: [P3<WorldSpace>] = []
        for i in 0..<rim.count {
            let p = rim[path: i]
            for end in [p.first!, p.last!] where abs(end.y - 0.1) < 1e-9 { rimEnds.append(end) }
        }
        Check.expect(rimEnds.count == 2, "the rim ends twice on the front edge", "\(rimEnds.count)")
        var missed = 0.0
        for end in rimEnds {
            missed = max(missed, crest.map { simd_length($0.v - end.v) }.min() ?? .infinity)
        }
        Check.expect(missed < 1e-9, "and the crest has a vertex at each end", String(format: "worst gap %.1e", missed))
        let trueCrossing = 0.24.squareRoot()
        let gridCrossing = rimEnds.map { abs(abs($0.x) - trueCrossing) }.max()!
        Check.expect(gridCrossing > 1e-3, "which is the grid's crossing, not f's",
                     String(format: "%.4f from f's", gridCrossing))
        // 1/z has no minimum on this edge but at its ends, so nothing else is
        // added: the crest is the nodes and the two crossings.
        let nodes = refined.manifest.height.shape.nx
        Check.expect(crest.count == nodes + 2, "the crest is the nodes and the two crossings", "\(crest.count) vertices")
        Check.expect(crest.allSatisfy { $0.z <= 2 + 1e-12 }, "and none rises above the cap")
    }
}
