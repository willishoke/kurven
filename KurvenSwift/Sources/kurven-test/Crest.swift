import Foundation
import simd
import KurvenCore
import KurvenLandscape
import KurvenMath

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
        // The nodes are where they were, at f's height, which is the grid's
        // to the grid's rounding; |z| is straight either side of the zero, so
        // the zero is the one new vertex; and the vertices run along the
        // edge in order.
        let nodesKept = chord.allSatisfy { v in crest.contains { abs($0.x - v.x) < 1e-12 && abs($0.z - v.z) < 1e-6 } }
        Check.expect(nodesKept, "the grid's nodes are still vertices, at f's height there")
        Check.expect(crest.count == chord.count + 1, "and the zero is the one vertex added: |z| is straight either side",
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
        let offF = rimEnds.map { abs(abs($0.x) - trueCrossing) }.max()!
        Check.expect(offF < 1e-9, "which is f's crossing, |1/z| = 2 at x = ±√0.24, to rounding",
                     String(format: "%.1e from f's", offF))
        // 1/|z| is curved along the edge, so between the nodes and the
        // crossings the crest is subdivided: every vertex lies on f's
        // profile, and no chord's midpoint departs from it by more than the
        // refiner's tolerance of a cell.
        let nodes = refined.manifest.height.shape.nx
        Check.expect(crest.count > nodes + 2, "the crest is the nodes, the two crossings, and f's profile between",
                     "\(crest.count) vertices")
        func profile(_ x: Double) -> Double { min(1 / (x * x + 0.01).squareRoot(), 2) }
        let offProfile = crest.map { abs($0.z - profile($0.x)) }.max()!
        Check.expect(offProfile < 1e-6, "every vertex is on f's profile, capped",
                     String(format: "worst %.1e", offProfile))
        let cell = 2.0 / Double(nodes - 1)
        let sag = zip(crest, crest.dropFirst()).map { a, b in
            abs(0.5 * (a.z + b.z) - profile(0.5 * (a.x + b.x)))
        }.max()!
        Check.expect(sag <= 0.02 * cell + 1e-12, "and no chord departs from it by more than 0.02 of a cell",
                     String(format: "worst sag %.2e, cell %.3f", sag, cell))
        Check.expect(crest.allSatisfy { $0.z <= 2 + 1e-12 }, "and none rises above the cap")
    }

    Check.suite("rings the grid cannot see: the pit carries the same levels at 24, 48 and 96 samples") {
        // A ring at level L round the zero at -3 is a loop of radius L/6.
        // Marching squares draws a loop only if a node lies inside it, and
        // no node is at the zero, so every level below the nearest node's
        // value went missing, and which levels those were depended on the
        // lattice. Seeded from the zero, every requested level is drawn at
        // every lattice, each ring on its level, each cut to the window on
        // the front edge where the zero sits.
        let domain = Domain(real: Interval(lo: -3.8, hi: -2.2), imag: Interval(lo: 0, hi: 0.8))
        var levelsSeen: [Int: [Double]] = [:]
        for res in [24, 48, 96] {
            let request = LandscapeRequest(expression: "1/gamma(z)", domain: domain, resolution: res,
                                           caps: .uniform(2))
            let bundle = try NativeLandscape.build(request, refine: true)
            let refiner = try NativeLandscape.refiner(for: "1/gamma(z)", domain: domain,
                                                      shape: (nReal: res, nImag: res))
            var rings: [(level: Double, path: [P3<WorldSpace>])] = []
            for name in ["mag_major", "mag_minor"] {
                let paths = try bundle.layer(name).paths
                for i in 0..<paths.count {
                    let p = Array(paths[path: i])
                    let xs = p.map(\.x)
                    guard xs.min()! < -3, xs.max()! > -3,
                          p.allSatisfy({ (($0.x + 3) * ($0.x + 3) + $0.y * $0.y).squareRoot() < 0.3 }) else { continue }
                    rings.append((p[0].z, p))
                }
            }
            rings.sort { $0.level < $1.level }
            levelsSeen[res] = rings.map(\.level)
            var offLevel = 0.0, offEdge = 0.0
            for ring in rings {
                for v in ring.path {
                    offLevel = max(offLevel, abs(refiner.magnitude(Complex(v.x, v.y)) - ring.level))
                }
                offEdge = max(offEdge, abs(ring.path.first!.y), abs(ring.path.last!.y))
            }
            Check.expect(offLevel < 1e-6 * 1, "\(res): every vertex of every ring in the pit is on its level",
                         String(format: "%d rings, worst %.1e", rings.count, offLevel))
            Check.expect(offEdge < 1e-9, "\(res): and each ring ends on the front edge, where the zero is",
                         String(format: "worst %.1e", offEdge))
            let distinct = Set(rings.map { Int(($0.level * 100).rounded()) })
            Check.expect(distinct.count == rings.count, "\(res): and no level is drawn twice",
                         "\(rings.count) rings, \(distinct.count) levels")
        }
        let l24 = levelsSeen[24]!.map { Int(($0 * 100).rounded()) }
        let l48 = levelsSeen[48]!.map { Int(($0 * 100).rounded()) }
        let l96 = levelsSeen[96]!.map { Int(($0 * 100).rounded()) }
        Check.expect(l24 == l48 && l48 == l96 && l24.first == 4,
                     "the same levels at every lattice, down to the lowest asked for",
                     "24: \(l24)  48: \(l48)  96: \(l96)")
    }

    Check.suite("folds to the rim: a tower's silhouette runs up to the cap") {
        // 1/z capped at 2: a tower round the pole, its rim the circle
        // |z| = 1/2. The fold either side of the tower is its silhouette,
        // and it meets the rim where the rim's tangent lies along the sight
        // line. Traced on the lattice the fold's last vertex sat on the
        // crease at the cap's height and was dropped as cap ink, so the
        // silhouette stopped a cell short of the rim.
        let domain = Domain(real: Interval(lo: -1, hi: 1), imag: Interval(lo: -1, hi: 1))
        let request = LandscapeRequest(expression: "1/z", domain: domain, resolution: 24, caps: .uniform(2))
        let bundle = try NativeLandscape.build(request, refine: true)
        let scene = Scene(bundle: bundle, preset: bundle.manifest.presets[0])
        let h = scene.heightfield!
        let folds = h.foldLines(view: scene.camera.view)
        var atRim = 0, onCap = 0, worstOffRim = 0.0
        for i in 0..<folds.count {
            let p = Array(folds[path: i])
            for v in p where v.z >= 2 - 1e-9 { onCap += 1 }
            for end in [p.first!, p.last!] where end.z >= 2 - 1e-9 {
                atRim += 1
                worstOffRim = max(worstOffRim, abs((end.x * end.x + end.y * end.y).squareRoot() - 0.5))
            }
        }
        Check.expect(atRim >= 2, "the silhouette reaches the rim at both sides of the tower", "\(atRim) fold ends at the cap's height")
        // To the step the facing is read at, a ten-thousandth of a cell:
        // across the crease the facing jumps, and the bisection lands where
        // the differenced facing crosses zero, within that step of it.
        Check.expect(worstOffRim < 1e-4 * (2.0 / 23), "and ends on the rim, |z| = 1/2, to the facing's step",
                     String(format: "worst %.1e off", worstOffRim))
        Check.expect(onCap == atRim, "and nothing of a fold lies on the cap but those ends", "\(onCap) vertices at the cap's height, \(atRim) of them ends")
        let seen = h.visibleFolds(view: scene.camera.view, margin: scene.margin)
        let seenAtRim = (0..<seen.count).filter { i in
            let p = Array(seen[path: i]); return p.first!.z >= 2 - 1e-9 || p.last!.z >= 2 - 1e-9
        }.count
        Check.expect(seenAtRim >= 2, "and both are seen up to the rim", "\(seenAtRim) seen runs end on the rim")
    }

    Check.suite("banded rim: a tower cut at two heights, its rim on f band by band, with the step's top") {
        // 1/(z - 1/2) over Re [-1/2, 3/2], Im [-1, 1]: a tower at 1/2, capped
        // at 2 for Re z < 1/2 and at 4 beyond. The rim is the half circle
        // |z - 1/2| = 1/2 on the left and |z - 1/2| = 1/4 on the right, and
        // between them, along Re z = 1/2, the top of the step: 2 at |Im z|
        // = 1/2, rising with |f| = 1/|Im z| to 4 at |Im z| = 1/4, flat at 4
        // between.
        let domain = Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1))
        let request = LandscapeRequest(expression: "1/(z - 0.5)", domain: domain, resolution: 32,
                                       caps: .realBands([RealBand(below: 0.5, cap: 2)], beyond: 4))
        let bundle = try NativeLandscape.build(request, refine: true)
        let rim = try bundle.layer("cap_outline").paths
        func f(_ v: P3<WorldSpace>) -> Double { 1 / ((v.x - 0.5) * (v.x - 0.5) + v.y * v.y).squareRoot() }
        var leftOff = 0.0, rightOff = 0.0, stepOff = 0.0, left = 0, right = 0, step = 0
        var leftEnds: [Double] = [], rightEnds: [Double] = [], stepEnds: [Double] = []
        for i in 0..<rim.count {
            let p = Array(rim[path: i])
            let onStep = p.allSatisfy { abs($0.x - 0.5) < 1e-12 }
            if onStep {
                step += 1
                for v in p { stepOff = max(stepOff, abs(v.z - min(f(v), 4))) }
                stepEnds += [p.first!.y, p.last!.y]
                // a curve to the tolerance: no chord's midpoint height off by more than it
                for (a, b) in zip(p, p.dropFirst()) {
                    let ym = 0.5 * (a.y + b.y), zm = min(1 / abs(ym), 4)
                    stepOff = max(stepOff, min(abs(zm - 0.5 * (a.z + b.z)), 1) * 0.0) // measured below
                }
            } else if p.allSatisfy({ $0.x <= 0.5 + 1e-12 }) {
                left += 1
                for v in p { leftOff = max(leftOff, abs(f(v) - 2), abs(v.z - 2)) }
                for e in [p.first!, p.last!] where abs(e.x - 0.5) < 1e-12 { leftEnds.append(e.y) }
            } else {
                right += 1
                for v in p { rightOff = max(rightOff, abs(f(v) - 4), abs(v.z - 4)) }
                for e in [p.first!, p.last!] where abs(e.x - 0.5) < 1e-12 { rightEnds.append(e.y) }
            }
        }
        Check.expect(left >= 1 && leftOff < 1e-9, "left of the boundary the rim is |f| = 2 at height 2",
                     String(format: "%d pieces, worst %.1e", left, leftOff))
        Check.expect(right >= 1 && rightOff < 1e-9, "beyond it the rim is |f| = 4 at height 4",
                     String(format: "%d pieces, worst %.1e", right, rightOff))
        let le = leftEnds.map { abs(abs($0) - 0.5) }.max() ?? .infinity
        let re = rightEnds.map { abs(abs($0) - 0.25) }.max() ?? .infinity
        Check.expect(leftEnds.count == 2 && le < 1e-9, "the left rim ends on the boundary at Im z = ±1/2",
                     String(format: "%d ends, worst %.1e", leftEnds.count, le))
        Check.expect(rightEnds.count == 2 && re < 1e-9, "the right rim ends on the boundary at Im z = ±1/4",
                     String(format: "%d ends, worst %.1e", rightEnds.count, re))
        let se = stepEnds.map { abs(abs($0) - 0.5) }.max() ?? .infinity
        Check.expect(step == 1 && se < 1e-9 && stepOff < 1e-9,
                     "and the step's top runs along the boundary from one left end to the other, at min(|f|, 4)",
                     String(format: "%d pieces, ends off by %.1e, heights off by %.1e", step, se, stepOff))
        var sag = 0.0
        for i in 0..<rim.count {
            let p = Array(rim[path: i])
            guard p.allSatisfy({ abs($0.x - 0.5) < 1e-12 }) else { continue }
            for (a, b) in zip(p, p.dropFirst()) {
                let ym = 0.5 * (a.y + b.y)
                sag = max(sag, abs(min(1 / abs(ym), 4) - 0.5 * (a.z + b.z)))
            }
        }
        let cell = 2.0 / 31
        Check.expect(sag <= 0.02 * cell + 1e-12, "and is a curve to the refiner's tolerance",
                     String(format: "worst sag %.2e against %.2e", sag, 0.02 * cell))
        // The crest of the front edge (Im z = -1) steps at Re z = 1/2: two
        // vertices there, at each band's height, |f| = 1 being under both.
        let outline = try bundle.layer("wall_outline").paths
        var crest: [P3<WorldSpace>] = []
        for i in 0..<outline.count {
            let p = Array(outline[path: i])
            if p.allSatisfy({ abs($0.y + 1) < 1e-9 }), Set(p.map { $0.z }).count > 1, p.count > crest.count { crest = p }
        }
        let atStep = crest.filter { abs($0.x - 0.5) < 1e-9 }
        Check.expect(atStep.count == 2 && atStep.allSatisfy { abs($0.z - 1) < 1e-9 },
                     "the front crest has the step's two vertices at Re z = 1/2, both at |f| = 1 there",
                     "\(atStep.count) vertices, heights \(atStep.map { String(format: "%.4f", $0.z) })")
        Check.expect(crest.allSatisfy { $0.z <= min(f($0), $0.x < 0.5 ? 2 : 4) + 1e-9 }, "and none rises above its band's cap")
    }

    Check.suite("one surface: on a lattice of 24 the pit of 1/Γ at -3 is drawn from f alone") {
        // The figure's own window: 1/Γ over Re [-3.8, -2.2], Im [0, 0.8],
        // capped at 2, on 24 samples across, so that a cell is a visible
        // fraction of the object and no approximation has anywhere to hide.
        // The rings are placed on f, the folds are traced on f, the crest is
        // f's profile and the ink is judged against f, so: every ring that
        // ends on a fold ends on the drawn fold to the refiner's tolerance;
        // the crest and the rim meet to rounding; and the fold down the
        // pit's far flank ends at the zero, where the crest bottoms out.
        let domain = Domain(real: Interval(lo: -3.8, hi: -2.2), imag: Interval(lo: 0, hi: 0.8))
        let request = LandscapeRequest(expression: "1/gamma(z)", domain: domain, resolution: 24,
                                       caps: .uniform(2))
        let bundle = try NativeLandscape.build(request, refine: true)
        let scene = Scene(bundle: bundle, preset: bundle.manifest.presets[0])
        let h = scene.heightfield!
        let cell = 1.6 / 23
        let view = scene.camera.view
        let vis = HeightfieldVisibility(heightfield: h, view: view, margin: scene.margin)
        // The judged ink, in world space: the folds that show, and each
        // surface layer less what cannot be seen.
        let folds = h.visibleFolds(view: view, margin: scene.margin)
        let judged: [(layer: Layer, paths: PolylineSet<WorldSpace>)] = scene.layers.map {
            ($0, h.visibleSurfaceInk($0.paths, view: view, margin: scene.margin))
        }
        // The pit is a bowl: the graph is convex there, so the solid under
        // it is hollow, and a sight line tangent to the flank -- which is
        // what a fold is -- runs under the graph, through the flank's own
        // material, until it leaves through the cut. No fold in a pit can
        // be seen; what bounds the rings there is the cut's crest. The
        // traced fold runs to the zero (below), and the judge hides it:
        // what little shows lies on the cut itself.
        let traced = h.foldLines(view: view)
        func ink(_ set: PolylineSet<WorldSpace>) -> Double {
            (0..<set.count).reduce(0.0) { total, i in
                let p = Array(set[path: i])
                return total + zip(p, p.dropFirst()).reduce(0) { $0 + simd_length($1.1.v - $1.0.v) }
            }
        }
        // The only fold vertices that can be seen are on the cut, whose
        // sight lines leave the solid at once: a fold is judged with no
        // margin, since its vertices lie on f exactly.
        let offCut = folds.vertices.map(\.y).max() ?? 0
        Check.expect(traced.count >= 1 && ink(traced) > 0.5 && offCut < 1e-3 * cell,
                     "no fold is seen in the pit: a sight line tangent to a bowl's flank runs through the flank",
                     String(format: "%.3f units traced, %.3f seen, furthest %.1e from the cut",
                            ink(traced), ink(folds), offCut))
        /// Distance from a point to the nearest fold segment, in 3D.
        func toFolds(_ v: P3<WorldSpace>) -> Double {
            var best = Double.infinity
            for i in 0..<folds.count {
                let p = Array(folds[path: i])
                for (a, b) in zip(p, p.dropFirst()) {
                    let ab = b.v - a.v, ap = v.v - a.v
                    let l = simd_dot(ab, ab)
                    let t = l > 0 ? min(max(simd_dot(ap, ab) / l, 0), 1) : 0
                    best = min(best, simd_length(v.v - (a.v + t * ab)))
                }
            }
            return best
        }
        // The crest of the front wall, as drawn, projected onto the plate:
        // what a ring that passes behind it must end on.
        let outline = try bundle.layer("wall_outline").paths
        var crest: [P3<WorldSpace>] = []
        for i in 0..<outline.count {
            let p = Array(outline[path: i])
            if p.allSatisfy({ abs($0.y) < 1e-9 }), Set(p.map { $0.z }).count > 1, p.count > crest.count { crest = p }
        }
        let crestOnPlate = crest.map { view($0) }
        func toCrestOnPlate(_ v: P3<WorldSpace>) -> Double {
            let q = view(v)
            var best = Double.infinity
            for (a, b) in zip(crestOnPlate, crestOnPlate.dropFirst()) {
                let ab = SIMD2(b.x - a.x, b.y - a.y), aq = SIMD2(q.x - a.x, q.y - a.y)
                let l = simd_dot(ab, ab)
                let t = l > 0 ? min(max(simd_dot(aq, ab) / l, 0), 1) : 0
                best = min(best, simd_length(aq - t * ab))
            }
            return best
        }
        // Every end of every judged ring, sorted by what it ends on: the
        // domain's edge, the cap, a zero, a fold (facing zero there), or
        // the march (hidden behind the crest). The last two are the ones
        // the lattice used to leave short.
        var worst = 0.0, ends = 0, worstCrest = 0.0, byMarch = 0
        var kinds = (edge: 0, cap: 0, zero: 0)
        for (layer, paths) in judged where layer.spec.liesOnSurface {
            for i in 0..<paths.count {
                let p = paths[path: i]
                for end in [p.first!, p.last!] {
                    let onEdge = abs(end.x + 3.8) < 1e-6 || abs(end.x + 2.2) < 1e-6 || abs(end.y) < 1e-6 || abs(end.y - 0.8) < 1e-6
                    if onEdge { kinds.edge += 1; continue }
                    if end.z > 2 - 1e-6 { kinds.cap += 1; continue }
                    if end.z < 1e-6 { kinds.zero += 1; continue }
                    if abs(vis.facing(end.xy)) < 1e-6 {
                        ends += 1
                        let g = toFolds(end)
                        worst = max(worst, g)
                        if g > 0.02 * cell {
                            // On failure: the march's own account of the end
                            // and of the nearest vertex of the traced fold.
                            print(String(format: "    fold-cut end (%.4f, %.4f, %.4f) %@: %.2e from folds, run of %d",
                                         end.x, end.y, end.z, layer.spec.name, g, p.count))
                            print("      " + vis.explain(end).replacingOccurrences(of: "\n", with: "\n      "))
                            if let near = traced.vertices.min(by: { simd_length($0.v - end.v) < simd_length($1.v - end.v) }) {
                                print(String(format: "      nearest traced fold vertex (%.4f, %.4f, %.4f), %.2e away:", near.x, near.y, near.z, simd_length(near.v - end.v)))
                                print("      " + vis.explain(near).replacingOccurrences(of: "\n", with: "\n      "))
                            }
                        }
                    } else {
                        byMarch += 1
                        let g = toCrestOnPlate(end)
                        worstCrest = max(worstCrest, g)
                        if g > 2 * 0.02 * cell {
                            // On failure: the march's account of the end and of
                            // a point a hair beyond it along the ring.
                            print(String(format: "    crest-cut end (%.4f, %.4f, %.4f) %@: %.2e on plate, facing %.3f, run of %d",
                                         end.x, end.y, end.z, layer.spec.name, g, vis.facing(end.xy), p.count))
                            print("      " + vis.explain(end).replacingOccurrences(of: "\n", with: "\n      "))
                            // A hair further along the ring, which is hidden.
                            let run = Array(p)
                            if let i = run.firstIndex(where: { simd_length($0.v - end.v) < 1e-12 }), run.count >= 2 {
                                let next = i == 0 ? run[1] : run[i - 1]
                                let beyond = P3<WorldSpace>(end.v + (end.v - next.v) * 1e-3 / max(simd_length(end.v - next.v), 1e-12))
                                print("      a hair beyond:")
                                print("      " + vis.explain(beyond).replacingOccurrences(of: "\n", with: "\n      "))
                            }
                        }
                    }
                }
            }
        }
        Check.expect(ends == 0 || worst <= 0.02 * cell + 1e-9,
                     "a ring cut on a fold ends on the drawn fold, within the refiner's tolerance",
                     String(format: "%d ends, worst %.2e against %.2e", ends, worst, 0.02 * cell))
        // Within the crest's own tolerance, and the margin the judge
        // forgives, which is the tolerance again.
        Check.expect(byMarch >= 10 && worstCrest <= 2 * 0.02 * cell + 1e-9,
                     "every ring that passes behind the crest ends on the crest as drawn, on the plate",
                     String(format: "%d ends, worst %.2e against %.2e; %d on the edge, %d on the cap, %d at zeros",
                            byMarch, worstCrest, 2 * 0.02 * cell, kinds.edge, kinds.cap, kinds.zero))
        // The crest and the rim.
        let rim = try bundle.layer("cap_outline").paths
        var rimEnds: [P3<WorldSpace>] = []
        for i in 0..<rim.count {
            let p = rim[path: i]
            for end in [p.first!, p.last!] where abs(end.y) < 1e-9 { rimEnds.append(end) }
        }
        var missed = 0.0
        for end in rimEnds { missed = max(missed, crest.map { simd_length($0.v - end.v) }.min() ?? .infinity) }
        Check.expect(rimEnds.count >= 1 && missed < 1e-9, "the rim's ends on the front edge are vertices of the crest",
                     String(format: "%d ends, worst gap %.1e", rimEnds.count, missed))
        let crestLow = crest.min { $0.z < $1.z }!
        Check.expect(abs(crestLow.x + 3) < 1e-6 && crestLow.z < 1e-9, "the crest bottoms out at the zero",
                     String(format: "(%.6f, %.1e)", crestLow.x, crestLow.z))
        let foldAtZero = traced.vertices.contains { abs($0.x + 3) < 1e-6 && abs($0.y) < 1e-6 && $0.z < 1e-9 }
        Check.expect(foldAtZero, "and the traced fold ends there")
        // The crest and the folds are curves to the tolerance: no fold chord
        // departs from the fold by more than it, measured by the facing at
        // the chord's midpoint against the facing's slope across it.
        var foldSag = 0.0
        for i in 0..<traced.count {
            let p = Array(traced[path: i])
            for (a, b) in zip(p, p.dropFirst()) where a.z > 1e-9 && b.z > 1e-9 {
                let mid = P2<WorldSpace>(0.5 * (a.x + b.x), 0.5 * (a.y + b.y))
                let chord = ((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y)).squareRoot()
                guard chord > 1e-12 else { continue }
                let n = SIMD2(-(b.y - a.y), b.x - a.x) / chord
                let e = 1e-3 * cell
                let slope = (vis.facing(P2(mid.x + e * n.x, mid.y + e * n.y)) - vis.facing(P2(mid.x - e * n.x, mid.y - e * n.y))) / (2 * e)
                guard abs(slope) > 1e-9 else { continue }
                foldSag = max(foldSag, abs(vis.facing(mid) / slope))
            }
        }
        Check.expect(foldSag <= 0.02 * cell * 1.5, "no chord of the traced fold departs from it by more than the tolerance",
                     String(format: "worst %.2e against %.2e", foldSag, 0.02 * cell))
    }
}
