import Foundation
import simd
import KurvenCore
import KurvenLandscape
import KurvenMath

// MARK: - the top edge of a cut face, placed on f

/// The wall outline's top edge between the grid's nodes: the grid's chord without
/// a refiner, f's own profile with one, down to the zero the phase lines fan to.
func topEdgeTests() {
    /// The top edge of the front edge (y at the domain's low imaginary bound) of
    /// a bundle's wall outline: the one stroke there whose height varies.
    func frontTopEdge(_ bundle: KurvenBundle) throws -> [P3<WorldSpace>] {
        let y0 = bundle.manifest.domain.imag.lo
        let outline = try bundle.layer("wall_edges").paths
        var best: [P3<WorldSpace>] = []
        for i in 0..<outline.count {
            let p = Array(outline[path: i])
            guard p.allSatisfy({ abs($0.y - y0) < 1e-9 }), p.count >= 2,
                  Set(p.map { $0.z }).count > 1, p.count > best.count else { continue }
            best = p
        }
        return best
    }

    Check.suite("top edge to the zero: the refined top edge gains the minimum of |f| between two nodes") {
        // f(z) = z over Re [-1, 1], Im [0, 1] on an 8 x 8 grid: the zero at 0
        // lies on the front edge between the nodes at -1/7 and 1/7, where the
        // grid's chord bottoms out at 1/7 and |z| itself reaches the floor.
        let domain = Domain(real: Interval(lo: -1, hi: 1), imag: Interval(lo: 0, hi: 1))
        let request = LandscapeRequest(expression: "z", domain: domain, resolution: 8,
                                       truncation: .uniform(10))
        let plain = try NativeLandscape.build(request, refine: false)
        let refined = try NativeLandscape.build(request, refine: true)
        let shape = refined.manifest.height.shape
        Check.expect(shape.nx == 8 && shape.ny == 8, "the grid is 8 x 8", "\(shape.nx) x \(shape.ny)")

        let chord = try frontTopEdge(plain), topEdge = try frontTopEdge(refined)
        let chordLow = chord.min { $0.z < $1.z }!, topEdgeLow = topEdge.min { $0.z < $1.z }!
        Check.expect(abs(chordLow.z - 1.0 / 7) < 1e-6 && abs(abs(chordLow.x) - 1.0 / 7) < 1e-9,
                     "from the grid alone the top edge bottoms out at the node beside the zero",
                     String(format: "lowest (%.4f, %.4f)", chordLow.x, chordLow.z))
        Check.expect(abs(topEdgeLow.z) < 1e-9 && abs(topEdgeLow.x) < 1e-9,
                     "with a refiner it reaches the zero, at the zero",
                     String(format: "lowest (%.2e, %.2e)", topEdgeLow.x, topEdgeLow.z))
        Check.expect(topEdge.count > chord.count, "vertices were added and none removed",
                     "\(chord.count) -> \(topEdge.count)")
        // The nodes are where they were, at f's height, which is the grid's
        // to the grid's rounding; |z| is straight either side of the zero, so
        // the zero is the one new vertex; and the vertices run along the
        // edge in order.
        let nodesKept = chord.allSatisfy { v in topEdge.contains { abs($0.x - v.x) < 1e-12 && abs($0.z - v.z) < 1e-6 } }
        Check.expect(nodesKept, "the grid's nodes are still vertices, at f's height there")
        Check.expect(topEdge.count == chord.count + 1, "and the zero is the one vertex added: |z| is straight either side",
                     "\(chord.count) -> \(topEdge.count)")
        let ordered = zip(topEdge, topEdge.dropFirst()).allSatisfy { $0.x < $1.x }
                   || zip(topEdge, topEdge.dropFirst()).allSatisfy { $0.x > $1.x }
        Check.expect(ordered, "the vertices run along the edge in order",
                     topEdge.map { String(format: "%.3f", $0.x) }.joined(separator: " "))

        // The phase lines end at the zero, at the zero's height: the fan meets
        // the top edge on the floor rather than a cell's rise above it.
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

        // The hatch of the front wall stands under the top edge as drawn: on the
        // V down to the zero, not on the chord over it.
        func tops(_ b: KurvenBundle) throws -> [(x: Double, z: Double)] {
            let hatch = try b.layer("wall_hatch").paths
            return (0..<hatch.count).compactMap { i in
                let p = hatch[path: i]
                guard p.allSatisfy({ abs($0.y) < 1e-9 }) else { return nil }
                return (p.first!.x, p.map(\.z).max()!)
            }
        }
        func onTopEdge(_ tops: [(x: Double, z: Double)], _ topEdge: [P3<WorldSpace>]) -> Double {
            tops.map { t in
                let (a, b) = zip(topEdge, topEdge.dropFirst()).first { min($0.x, $1.x) - 1e-12 <= t.x && t.x <= max($0.x, $1.x) + 1e-12 }!
                let u = b.x == a.x ? 0 : (t.x - a.x) / (b.x - a.x)
                return abs(t.z - (a.z + (b.z - a.z) * u))
            }.max() ?? 1
        }
        let refinedTops = try tops(refined), plainTops = try tops(plain)
        Check.expect(!refinedTops.isEmpty && onTopEdge(refinedTops, topEdge) < 1e-9,
                     "the front wall's hatch strokes rise to the refined top edge",
                     String(format: "%d strokes, worst %.1e", refinedTops.count, onTopEdge(refinedTops, topEdge)))
        Check.expect(onTopEdge(plainTops, chord) < 1e-6, "and to the grid's without a refiner",
                     String(format: "worst %.1e", onTopEdge(plainTops, chord)))
        let inTheV = refinedTops.filter { abs($0.x) < 1.0 / 7 }
        Check.expect(!inTheV.isEmpty && inTheV.allSatisfy { $0.z < 1.0 / 7 - 1e-9 },
                     "so the strokes in the pit stop under the V, not at the chord", "\(inTheV.count) strokes")
    }

    Check.suite("top edge to the zero: the fold curves run down to the zero as well") {
        // A steeper cone, 10|z|, so that its far graph turns away from the
        // plate camera: its outlines are two generators running into the
        // apex, which the grid loses a cell short of the floor.
        let domain = Domain(real: Interval(lo: -1, hi: 1), imag: Interval(lo: 0, hi: 1))
        let request = LandscapeRequest(expression: "10*z", domain: domain, resolution: 8,
                                       truncation: .uniform(20))
        let plain = try NativeLandscape.build(request, refine: false)
        let refined = try NativeLandscape.build(request, refine: true)
        func ends(_ b: KurvenBundle) -> [P3<WorldSpace>] {
            let scene = Scene(bundle: b, preset: b.manifest.presets[0])
            let folds = scene.heightfield!.foldCurves(view: scene.camera.view)
            return (0..<folds.count).flatMap { [folds[path: $0].first!, folds[path: $0].last!] }
        }
        let carried = ends(refined).filter { abs($0.x) < 1e-9 && abs($0.y) < 1e-9 }
        Check.expect(carried.count >= 1 && carried.allSatisfy { abs($0.z) < 1e-9 },
                     "with a refiner a fold ends at the zero, on the floor", "\(carried.count) ends there")
        let lowest = ends(plain).map(\.z).min() ?? -1
        Check.expect(lowest > 0.05, "which the grid's own folds stop short of",
                     String(format: "lowest end at z = %.3f", lowest))
    }

    Check.suite("top edge to the zero: the truncation crossings stay the grid's, so the top edge meets the truncation edge") {
        // 1/z over Re [-1, 1], Im [0.1, 1.1] under a truncation of 2: the front edge
        // crosses the truncation twice, at x = ±sqrt(0.24). The truncation edge is contoured on
        // the grid, so the top edge's crossing is solved on the grid too, and the
        // two coincide; f's own crossing is a few hundredths to one side.
        let domain = Domain(real: Interval(lo: -1, hi: 1), imag: Interval(lo: 0.1, hi: 1.1))
        let request = LandscapeRequest(expression: "1/z", domain: domain, resolution: 8,
                                       truncation: .uniform(2))
        let refined = try NativeLandscape.build(request, refine: true)
        let topEdge = try frontTopEdge(refined)
        let truncationEdge = try refined.layer("truncation_edge").paths
        var truncationEdgeEnds: [P3<WorldSpace>] = []
        for i in 0..<truncationEdge.count {
            let p = truncationEdge[path: i]
            for end in [p.first!, p.last!] where abs(end.y - 0.1) < 1e-9 { truncationEdgeEnds.append(end) }
        }
        Check.expect(truncationEdgeEnds.count == 2, "the truncation edge ends twice on the front edge", "\(truncationEdgeEnds.count)")
        var missed = 0.0
        for end in truncationEdgeEnds {
            missed = max(missed, topEdge.map { simd_length($0.v - end.v) }.min() ?? .infinity)
        }
        Check.expect(missed < 1e-9, "and the top edge has a vertex at each end", String(format: "worst gap %.1e", missed))
        let trueCrossing = 0.24.squareRoot()
        let offF = truncationEdgeEnds.map { abs(abs($0.x) - trueCrossing) }.max()!
        Check.expect(offF < 1e-9, "which is f's crossing, |1/z| = 2 at x = ±√0.24, to rounding",
                     String(format: "%.1e from f's", offF))
        // 1/|z| is curved along the edge, so between the nodes and the
        // crossings the top edge is subdivided: every vertex lies on f's
        // profile, and no chord's midpoint departs from it by more than the
        // refiner's tolerance of a cell.
        let nodes = refined.manifest.height.shape.nx
        Check.expect(topEdge.count > nodes + 2, "the top edge is the nodes, the two crossings, and f's profile between",
                     "\(topEdge.count) vertices")
        func profile(_ x: Double) -> Double { min(1 / (x * x + 0.01).squareRoot(), 2) }
        let offProfile = topEdge.map { abs($0.z - profile($0.x)) }.max()!
        Check.expect(offProfile < 1e-6, "every vertex is on f's profile, truncated",
                     String(format: "worst %.1e", offProfile))
        let cell = 2.0 / Double(nodes - 1)
        let sag = zip(topEdge, topEdge.dropFirst()).map { a, b in
            abs(0.5 * (a.z + b.z) - profile(0.5 * (a.x + b.x)))
        }.max()!
        Check.expect(sag <= 0.02 * cell + 1e-12, "and no chord departs from it by more than 0.02 of a cell",
                     String(format: "worst sag %.2e, cell %.3f", sag, cell))
        Check.expect(topEdge.allSatisfy { $0.z <= 2 + 1e-12 }, "and none rises above the truncation")
    }

    Check.suite("rings the grid cannot see: the pit carries the same levels at 24, 48 and 96 samples") {
        // A ring at level L round the zero at -3 is a loop of radius L/6.
        // Marching squares draws a loop only if a node lies inside it, and
        // no node is at the zero, so every level below the nearest node's
        // value went missing, and which levels those were depended on the
        // grid. Seeded from the zero, every requested level is drawn at
        // every grid, each ring on its level, each cut to the window on
        // the front edge where the zero sits.
        let domain = Domain(real: Interval(lo: -3.8, hi: -2.2), imag: Interval(lo: 0, hi: 0.8))
        var levelsSeen: [Int: [Double]] = [:]
        for res in [24, 48, 96] {
            let request = LandscapeRequest(expression: "1/gamma(z)", domain: domain, resolution: res,
                                           truncation: .uniform(2))
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
                     "the same levels at every grid, down to the lowest asked for",
                     "24: \(l24)  48: \(l48)  96: \(l96)")
    }

    Check.suite("folds to the truncation edge: a tower's outline runs up to the truncation") {
        // 1/z truncated at 2: a tower round the pole, its truncation edge the circle
        // |z| = 1/2. The fold either side of the tower is its outline,
        // and it meets the truncation edge where the truncation edge's tangent lies along the sight
        // line. Traced on the grid the fold's last vertex sat on the
        // crease at the truncation height and was dropped as truncation ink, so the
        // outline stopped a cell short of the truncation edge.
        let domain = Domain(real: Interval(lo: -1, hi: 1), imag: Interval(lo: -1, hi: 1))
        let request = LandscapeRequest(expression: "1/z", domain: domain, resolution: 24, truncation: .uniform(2))
        let bundle = try NativeLandscape.build(request, refine: true)
        let scene = Scene(bundle: bundle, preset: bundle.manifest.presets[0])
        let h = scene.heightfield!
        let folds = h.foldCurves(view: scene.camera.view)
        var atTruncationEdge = 0, onPlateau = 0, worstOffTruncationEdge = 0.0
        for i in 0..<folds.count {
            let p = Array(folds[path: i])
            for v in p where v.z >= 2 - 1e-9 { onPlateau += 1 }
            for end in [p.first!, p.last!] where end.z >= 2 - 1e-9 {
                atTruncationEdge += 1
                worstOffTruncationEdge = max(worstOffTruncationEdge, abs((end.x * end.x + end.y * end.y).squareRoot() - 0.5))
            }
        }
        Check.expect(atTruncationEdge >= 2, "the outline reaches the truncation edge at both sides of the tower", "\(atTruncationEdge) fold ends at the truncation height")
        // To the step the facing is read at, a ten-thousandth of a cell:
        // across the crease the facing jumps, and the bisection lands where
        // the differenced facing crosses zero, within that step of it.
        Check.expect(worstOffTruncationEdge < 1e-4 * (2.0 / 23), "and ends on the truncation edge, |z| = 1/2, to the facing's step",
                     String(format: "worst %.1e off", worstOffTruncationEdge))
        Check.expect(onPlateau == atTruncationEdge, "and nothing of a fold lies on the plateau but those ends", "\(onPlateau) vertices at the truncation height, \(atTruncationEdge) of them ends")
        let seen = h.visibleFolds(view: scene.camera.view, depthTolerance: scene.depthTolerance)
        let seenAtTruncationEdge = (0..<seen.count).filter { i in
            let p = Array(seen[path: i]); return p.first!.z >= 2 - 1e-9 || p.last!.z >= 2 - 1e-9
        }.count
        Check.expect(seenAtTruncationEdge >= 2, "and both are seen up to the truncation edge", "\(seenAtTruncationEdge) seen runs end on the truncation edge")
    }

    Check.suite("folds arrive along the fold: a tower's outline meets the truncation edge where the graph turns edge-on") {
        // (z - 1)/(z + 1) over Re [-2.1, 2.1], Im [0, 1.4], truncated at 3, on a
        // grid of 24: the pole at -1 sits on the front edge, and the tower's
        // right graph carries a outline up to the truncation edge. Traced on the
        // truncated surface the fold's last vertex was where a grid edge
        // crossed the crease, a cell from where the fold meets the truncation edge, and
        // the straight chord joining them cut across the truncation's projection.
        // Traced on the graph the fold runs on below the truncation and is ended
        // at its own crossing of |f| = h(x), so it arrives along itself.
        let domain = Domain(real: Interval(lo: -2.1, hi: 2.1), imag: Interval(lo: 0, hi: 1.4))
        let request = LandscapeRequest(expression: "(z-1)/(z+1)", domain: domain, resolution: 24, truncation: .uniform(3))
        let bundle = try NativeLandscape.build(request, refine: true)
        let scene = Scene(bundle: bundle, preset: bundle.manifest.presets[0])
        let h = scene.heightfield!
        let cell = 4.2 / 23
        func f(_ x: Double, _ y: Double) -> Double {
            ((x - 1) * (x - 1) + y * y).squareRoot() / ((x + 1) * (x + 1) + y * y).squareRoot()
        }
        // The line of sight, as the kernel of the plate projection.
        let o = scene.camera.plateCoordinates(of: P3(0, 0, 0))
        let cols = [P3<WorldSpace>(1, 0, 0), P3(0, 1, 0), P3(0, 0, 1)].map { scene.camera.plateCoordinates(of: $0) }
        let row0 = SIMD3(cols[0].x - o.x, cols[1].x - o.x, cols[2].x - o.x)
        let row1 = SIMD3(cols[0].y - o.y, cols[1].y - o.y, cols[2].y - o.y)
        let sight = simd_normalize(simd_cross(row0, row1))
        /// The graph's facing: f's own slope, untruncated.
        func graphFacing(_ x: Double, _ y: Double) -> Double {
            let s = 1e-6
            let n = SIMD3(-(f(x + s, y) - f(x - s, y)) / (2 * s), -(f(x, y + s) - f(x, y - s)) / (2 * s), 1)
            return -simd_dot(n, sight) / simd_length(n)
        }
        let folds = h.foldCurves(view: scene.camera.view)
        var truncationEdgeEnds = 0, worstFacing = 0.0, worstOffTruncationEdge = 0.0, underTruncation = 0, longestLastChord = 0.0, offFold = 0.0
        for i in 0..<folds.count {
            let p = Array(folds[path: i])
            for v in p where v.z >= 3 - 1e-9 {
                truncationEdgeEnds += 1
                worstFacing = max(worstFacing, abs(graphFacing(v.x, v.y)))
                worstOffTruncationEdge = max(worstOffTruncationEdge, abs(f(v.x, v.y) - 3))
            }
            for v in p where v.z < 3 - 1e-9 {
                if f(v.x, v.y) > 3 { underTruncation += 1 }
                offFold = max(offFold, abs(graphFacing(v.x, v.y)))
            }
            for (a, b) in zip(p, p.dropFirst()) where a.z >= 3 - 1e-9 || b.z >= 3 - 1e-9 {
                longestLastChord = max(longestLastChord, simd_length(a.v - b.v))
            }
        }
        Check.expect(truncationEdgeEnds >= 1, "a fold ends on the truncation edge", "\(truncationEdgeEnds) truncation edge ends")
        Check.expect(worstOffTruncationEdge < 1e-9, "on |f| = 3 to the last bit", String(format: "worst %.1e off", worstOffTruncationEdge))
        Check.expect(worstFacing < 1e-6, "where the graph turns edge-on, not where a grid edge crosses the crease",
                     String(format: "graph facing %.1e at a truncation edge end", worstFacing))
        Check.expect(offFold < 1e-6, "and every graph vertex is on the fold", String(format: "facing %.1e", offFold))
        Check.expect(underTruncation == 0, "no fold vertex lies below the truncation", "\(underTruncation) do")
        Check.expect(longestLastChord < 0.25 * cell, "the last chord to the truncation edge is a subdivided piece of the fold, not a cell-long jump",
                     String(format: "%.3f cells", longestLastChord / cell))
        let seen = h.visibleFolds(view: scene.camera.view, depthTolerance: scene.depthTolerance)
        let reaches = (0..<seen.count).contains { i in
            let p = Array(seen[path: i])
            let top = max(p.first!.z, p.last!.z), bottom = min(p.first!.z, p.last!.z)
            return top >= 3 - 1e-9 && bottom < 2.2
        }
        Check.expect(reaches, "and the outline is seen in one piece from the graph up to the truncation edge")
        // The truncation's hatch is ruled on f: no stroke reaches past the truncation edge.
        let hatch = try bundle.layer("plateau_hatch").paths
        var past = 0, endsOff = 0.0, strokes = 0
        for i in 0..<hatch.count {
            let p = Array(hatch[path: i]); strokes += 1
            for v in p where f(v.x, v.y) < 3 - 1e-9 { past += 1 }
            for end in [p.first!, p.last!] where abs(end.x) < 2.1 - 1e-9 && end.y > 1e-9 && end.y < 1.4 - 1e-9 {
                endsOff = max(endsOff, abs(f(end.x, end.y) - 3))
            }
        }
        Check.expect(strokes > 0 && past == 0, "no plateau hatching vertex lies outside the truncation edge drawn on f", "\(past) of \(strokes) strokes' vertices")
        Check.expect(endsOff < 1e-9, "and every stroke ends on |f| = 3", String(format: "worst %.1e off", endsOff))
    }

    Check.suite("a top edge comes out from behind a tower where the traversal turns, not at the next node") {
        // tan(z) over Re [-0.7, 2.3], Im [0, 1], truncated at 2.5, on a grid of
        // 24: the tower at pi/2 hides a stretch of the back edge's top edge.
        // The top edge is concave there, so the chord between two of its
        // vertices lies under it, in the back wall, which faces away: a
        // run's end bisected along the chord found nothing admitted and the
        // top edge began at the next node, half a cell past where it comes
        // into view beside the outline's cusp.
        let domain = Domain(real: Interval(lo: -0.7, hi: 2.3), imag: Interval(lo: 0, hi: 1.0))
        let request = LandscapeRequest(expression: "tan(z)", domain: domain, resolution: 24, truncation: .uniform(2.5))
        let bundle = try NativeLandscape.build(request, refine: true)
        let scene = Scene(bundle: bundle, preset: bundle.manifest.presets[0])
        let h = scene.heightfield!
        let view = scene.camera.view
        let cell = 3.0 / 23
        let topEdge = try bundle.layer("wall_edges").paths
        let drawn = h.visibleWallInk(topEdge, view: view, depthTolerance: scene.depthTolerance)
        // the run of the back top edge that begins right of the tower
        var start: P3<WorldSpace>?
        for i in 0..<drawn.count {
            let p = Array(drawn[path: i])
            guard p.allSatisfy({ abs($0.y - 1.0) < 1e-9 }), p.first!.z > 0.01 else { continue }
            let left = p.min { $0.x < $1.x }!
            if left.x > 1.5 && left.x < 2.0 { start = left }
        }
        Check.expect(start != nil, "the back top edge's run right of the tower is drawn")
        if let s = start {
            let vis = HeightfieldVisibility(heightfield: h, view: view, depthTolerance: scene.depthTolerance)
            func at(_ x: Double) -> P3<WorldSpace> { P3(x, 1.0, h.surfaceHeight(at: P2(x, 1.0))) }
            let node = (s.x + 0.7) / cell
            Check.expect(abs(node - node.rounded()) > 1e-3, "and begins between nodes", String(format: "at x = %.4f, node %.3f", s.x, node))
            Check.expect(!vis.isClear(at(s.x - 1e-4 * cell)) && vis.isClear(at(s.x + 1e-4 * cell)),
                         "where the top edge comes into view, to a ten-thousandth of a cell",
                         String(format: "x = %.5f: hidden before %@, clear after %@", s.x,
                                vis.isClear(at(s.x - 1e-4 * cell)) ? "no" : "yes", vis.isClear(at(s.x + 1e-4 * cell)) ? "yes" : "no"))
            Check.expect(abs(s.z - h.surfaceHeight(at: P2(s.x, s.y))) < 1e-9, "on the top edge", String(format: "%.2e under f", h.surfaceHeight(at: P2(s.x, s.y)) - s.z))
        }
    }

    Check.suite("a chord the refiner cannot place is not drawn") {
        // exp(1/z) over Re [-1.1, 0.4], Im [0, 0.75], truncated at 3, on a grid
        // of 24: an essential singularity at the origin, on the front edge.
        // The circles of one phase level crowd into it, so marching squares
        // joins vertices of different circles, and the phase line along the
        // real axis jumps from the floor to the truncation at the origin. Those
        // chords used to be drawn as they were: hooks and zigzags beside the
        // origin, and a line up the cliff.
        let domain = Domain(real: Interval(lo: -1.1, hi: 0.4), imag: Interval(lo: 0, hi: 0.75))
        let request = LandscapeRequest(expression: "exp(1/z)", domain: domain, resolution: 24, truncation: .uniform(3))
        let bundle = try NativeLandscape.build(request, refine: true)
        let cell = 1.5 / 23
        func f(_ x: Double, _ y: Double) -> Complex { Complex.exp(1 / Complex(x, y)) }
        var chords = 0, strayed = 0, jumps = 0, worstStray = 0.0
        var examples: [String] = []
        for (spec, layer) in zip(bundle.manifest.layers, bundle.layers) where spec.name.hasPrefix("ang") {
            for i in 0..<layer.paths.count {
                let p = Array(layer.paths[path: i])
                for (a, b) in zip(p, p.dropFirst()) {
                    chords += 1
                    // the chord's departure from the level set at its quarter
                    // points: the phase residual against the level its far end
                    // sits on, over the gradient of the phase there, in cells
                    let level = f(b.x, b.y).argument
                    for t in [0.25, 0.5, 0.75] {
                        let q = (a.x + t * (b.x - a.x), a.y + t * (b.y - a.y))
                        var r = f(q.0, q.1).argument - level
                        while r > .pi { r -= 2 * .pi }; while r < -.pi { r += 2 * .pi }
                        let s = 1e-5 * cell
                        func ph(_ x: Double, _ y: Double) -> Double { f(x, y).argument }
                        var dx = ph(q.0 + s, q.1) - ph(q.0 - s, q.1), dy = ph(q.0, q.1 + s) - ph(q.0, q.1 - s)
                        while dx > .pi { dx -= 2 * .pi }; while dx < -.pi { dx += 2 * .pi }
                        while dy > .pi { dy -= 2 * .pi }; while dy < -.pi { dy += 2 * .pi }
                        let slope = (dx * dx + dy * dy).squareRoot() / (2 * s)
                        let off = slope > 0 && slope.isFinite ? abs(r) / slope / cell : abs(r)
                        if off > 0.021 {
                            strayed += 1
                            if examples.count < 4 { examples.append(String(format: "(%.5f, %.5f, %.3f)->(%.5f, %.5f, %.3f) t %.2f off %.3f cells r %.2f", a.x, a.y, a.z, b.x, b.y, b.z, t, off, r)) }
                        }
                        worstStray = max(worstStray, off)
                    }
                    // no chord climbs the cliff: a jump in height of more than a cell's worth of slope
                    if abs(a.z - b.z) > 1.5 { jumps += 1; examples.append(String(format: "jump (%.5f, %.5f, %.3f)->(%.5f, %.5f, %.3f)", a.x, a.y, a.z, b.x, b.y, b.z)) }
                }
            }
        }
        Check.expect(chords > 100, "phase chords are drawn beside the singularity", "\(chords) chords")
        Check.expect(strayed == 0, "and every one of them lies on the level set to the tolerance at its quarter points",
                     String(format: "%d quarter points off, worst %.3f cells  %@", strayed, worstStray, examples.joined(separator: "; ") as NSString))
        Check.expect(jumps == 0, "and none climbs the cliff at the origin", "\(jumps) chords rise by more than 1.5  \(examples.joined(separator: "; "))")
        // the rings all pass through the origin and are placed: every ring vertex is on its level
        var ringVertices = 0, off = 0
        for (spec, layer) in zip(bundle.manifest.layers, bundle.layers) where spec.name.hasPrefix("mag") {
            for i in 0..<layer.paths.count {
                let p = Array(layer.paths[path: i])
                for v in p where v.x * v.x + v.y * v.y > 1e-12 {
                    ringVertices += 1
                    if abs(f(v.x, v.y).magnitude - v.z) > 1e-6 * max(1, v.z) { off += 1 }
                }
            }
        }
        Check.expect(ringVertices > 100 && off == 0, "every ring vertex off the origin is on its level", "\(off) of \(ringVertices) off")
        // The phase lines are followed on f to the origin: marching squares
        // ends them a cell short, where it can no longer bracket a phase
        // that turns many times within a cell; from there each is traced
        // along its circle until f's phase runs out, which is where exp(1/z)
        // underflows a hair from the origin.
        var interiorEnds = 0, farEnds = 0, worstEnd = 0.0
        for (spec, layer) in zip(bundle.manifest.layers, bundle.layers) where spec.name.hasPrefix("ang") {
            for i in 0..<layer.paths.count {
                let p = Array(layer.paths[path: i])
                for e in [p.first!, p.last!] {
                    let onEdge = abs(e.x + 1.1) < 1e-6 || abs(e.x - 0.4) < 1e-6 || abs(e.y) < 1e-6 || abs(e.y - 0.75) < 1e-6
                    let onTruncationEdge = f(e.x, e.y).magnitude >= 3 - 1e-3
                    guard !onEdge, !onTruncationEdge, e.x < 0.1, e.y < 0.2 else { continue }
                    interiorEnds += 1
                    let d = (e.x * e.x + e.y * e.y).squareRoot()
                    worstEnd = max(worstEnd, d)
                    if d > 0.1 * cell { farEnds += 1 }
                }
            }
        }
        Check.expect(interiorEnds >= 10 && farEnds == 0, "and the phase lines beside the origin run to within a tenth of a cell of it",
                     String(format: "%d ends beside it, %d further off, worst %.4f cells", interiorEnds, farEnds, worstEnd / cell))
    }

    Check.suite("banded truncation edge: a tower cut at two heights, its truncation edge on f band by band, with the step's top") {
        // 1/(z - 1/2) over Re [-1/2, 3/2], Im [-1, 1]: a tower at 1/2, truncated
        // at 2 for Re z < 1/2 and at 4 beyond. The truncation edge is the half circle
        // |z - 1/2| = 1/2 on the left and |z - 1/2| = 1/4 on the right, and
        // between them, along Re z = 1/2, the top of the step: 2 at |Im z|
        // = 1/2, rising with |f| = 1/|Im z| to 4 at |Im z| = 1/4, flat at 4
        // between.
        let domain = Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1))
        let request = LandscapeRequest(expression: "1/(z - 0.5)", domain: domain, resolution: 32,
                                       truncation: .realBands([RealBand(below: 0.5, height: 2)], beyond: 4))
        let bundle = try NativeLandscape.build(request, refine: true)
        let truncationEdge = try bundle.layer("truncation_edge").paths
        func f(_ v: P3<WorldSpace>) -> Double { 1 / ((v.x - 0.5) * (v.x - 0.5) + v.y * v.y).squareRoot() }
        var leftOff = 0.0, rightOff = 0.0, stepOff = 0.0, left = 0, right = 0, step = 0
        var leftEnds: [Double] = [], rightEnds: [Double] = [], stepEnds: [Double] = []
        for i in 0..<truncationEdge.count {
            let p = Array(truncationEdge[path: i])
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
        Check.expect(left >= 1 && leftOff < 1e-9, "left of the boundary the truncation edge is |f| = 2 at height 2",
                     String(format: "%d pieces, worst %.1e", left, leftOff))
        Check.expect(right >= 1 && rightOff < 1e-9, "beyond it the truncation edge is |f| = 4 at height 4",
                     String(format: "%d pieces, worst %.1e", right, rightOff))
        let le = leftEnds.map { abs(abs($0) - 0.5) }.max() ?? .infinity
        let re = rightEnds.map { abs(abs($0) - 0.25) }.max() ?? .infinity
        Check.expect(leftEnds.count == 2 && le < 1e-9, "the left truncation edge ends on the boundary at Im z = ±1/2",
                     String(format: "%d ends, worst %.1e", leftEnds.count, le))
        Check.expect(rightEnds.count == 2 && re < 1e-9, "the right truncation edge ends on the boundary at Im z = ±1/4",
                     String(format: "%d ends, worst %.1e", rightEnds.count, re))
        let se = stepEnds.map { abs(abs($0) - 0.5) }.max() ?? .infinity
        Check.expect(step == 1 && se < 1e-9 && stepOff < 1e-9,
                     "and the step's top runs along the boundary from one left end to the other, at min(|f|, 4)",
                     String(format: "%d pieces, ends off by %.1e, heights off by %.1e", step, se, stepOff))
        var sag = 0.0
        for i in 0..<truncationEdge.count {
            let p = Array(truncationEdge[path: i])
            guard p.allSatisfy({ abs($0.x - 0.5) < 1e-12 }) else { continue }
            for (a, b) in zip(p, p.dropFirst()) {
                let ym = 0.5 * (a.y + b.y)
                sag = max(sag, abs(min(1 / abs(ym), 4) - 0.5 * (a.z + b.z)))
            }
        }
        let cell = 2.0 / 31
        Check.expect(sag <= 0.02 * cell + 1e-12, "and is a curve to the refiner's tolerance",
                     String(format: "worst sag %.2e against %.2e", sag, 0.02 * cell))
        // The top edge of the front edge (Im z = -1) steps at Re z = 1/2: two
        // vertices there, at each band's height, |f| = 1 being under both.
        let outline = try bundle.layer("wall_edges").paths
        var topEdge: [P3<WorldSpace>] = []
        for i in 0..<outline.count {
            let p = Array(outline[path: i])
            if p.allSatisfy({ abs($0.y + 1) < 1e-9 }), Set(p.map { $0.z }).count > 1, p.count > topEdge.count { topEdge = p }
        }
        let atStep = topEdge.filter { abs($0.x - 0.5) < 1e-9 }
        Check.expect(atStep.count == 2 && atStep.allSatisfy { abs($0.z - 1) < 1e-9 },
                     "the front top edge has the step's two vertices at Re z = 1/2, both at |f| = 1 there",
                     "\(atStep.count) vertices, heights \(atStep.map { String(format: "%.4f", $0.z) })")
        Check.expect(topEdge.allSatisfy { $0.z <= min(f($0), $0.x < 0.5 ? 2 : 4) + 1e-9 }, "and none rises above its band's truncation")
    }

    Check.suite("a window's ink is tested to its pixel: every end within the resolution asked for") {
        // The preview places the ends of tested ink to a quarter of its pixel
        // (`Scene.resolution`), and a bake to the last bit. The two drawings
        // differ only where a run ends, and there by less than what was
        // asked for, measured in the view plane: a point of either lies
        // within it of the other. A run that is itself shorter than that --
        // a vertex a hair in front of where the fold passes behind, drawn to
        // the last bit as a speck of no length -- is not drawn at all.
        let request = LandscapeRequest(preset: Catalog.native.preset("cn")!, resolution: 240)
        let bundle = try NativeLandscape.build(request, refine: true)
        let scene = Scene(bundle: bundle, preset: bundle.manifest.presets[0])
        let h = scene.heightfield!
        let view = scene.camera.view
        let bounds = scene.quickBounds()!
        let pixel = Framing.fitting(bounds, in: Viewport(width: 1600, height: 1000)).unitsPerPixel
        let resolution = 0.25 * pixel
        func onScreen(_ set: PolylineSet<WorldSpace>) -> [[SIMD2<Double>]] {
            (0..<set.count).map { i in
                set.vertices[set.offsets[i]..<set.offsets[i + 1]].map { let q = view($0); return SIMD2(q.x, q.y) }
            }
        }
        func distance(_ p: SIMD2<Double>, _ runs: [[SIMD2<Double>]]) -> Double {
            var best = Double.infinity
            for run in runs {
                for (a, b) in zip(run, run.dropFirst()) {
                    let ab = b - a, l2 = simd_dot(ab, ab)
                    let t = l2 > 0 ? max(0, min(1, simd_dot(p - a, ab) / l2)) : 0
                    best = min(best, simd_length(a + t * ab - p))
                }
            }
            return best
        }
        func length(_ run: [SIMD2<Double>]) -> Double {
            zip(run, run.dropFirst()).reduce(0) { $0 + simd_length($1.1 - $1.0) }
        }
        var worst = 0.0, longestDropped = 0.0
        var fewer = 0
        var pairs = [(h.visibleFolds(view: view, depthTolerance: scene.depthTolerance),
                      h.visibleFolds(view: view, depthTolerance: scene.depthTolerance, resolution: resolution))]
        for layer in scene.layers where layer.spec.source.isWallInk && layer.spec.clipped {
            pairs.append((h.visibleWallInk(layer.paths, view: view, depthTolerance: scene.depthTolerance),
                          h.visibleWallInk(layer.paths, view: view, depthTolerance: scene.depthTolerance,
                                           resolution: resolution)))
        }
        for (exact, coarse) in pairs {
            let a = onScreen(exact), b = onScreen(coarse)
            for run in a {
                let far = run.map { distance($0, b) }.max() ?? 0
                if far > resolution, length(run) <= resolution {
                    longestDropped = max(longestDropped, length(run))
                } else {
                    worst = max(worst, far)
                }
            }
            for run in b { for p in run { worst = max(worst, distance(p, a)) } }
            fewer += exact.vertices.count - coarse.vertices.count
        }
        Check.expect(worst <= resolution * (1 + 1e-9),
                     "each drawing lies within a quarter pixel of the other",
                     String(format: "worst %.3f of a pixel", worst / pixel))
        Check.expect(fewer > 0, "and the ends a pixel cannot show are not placed", "\(fewer) fewer vertices")
        Check.expect(longestDropped <= resolution, "and what is not drawn is a speck",
                     String(format: "longest %.4f of a pixel", longestDropped / pixel))
    }

    Check.suite("one surface: on a grid of 24 the pit of 1/Γ at -3 is drawn from f alone") {
        // The figure's own window: 1/Γ over Re [-3.8, -2.2], Im [0, 0.8],
        // truncated at 2, on 24 samples across, so that a cell is a visible
        // fraction of the object and no approximation has anywhere to hide.
        // The rings are placed on f, the folds are traced on f, the top edge is
        // f's profile and the ink is tested against f, so: every ring that
        // ends on a fold ends on the drawn fold to the refiner's tolerance;
        // the top edge and the truncation edge meet to rounding; and the fold down the
        // pit's far graph ends at the zero, where the top edge bottoms out.
        let domain = Domain(real: Interval(lo: -3.8, hi: -2.2), imag: Interval(lo: 0, hi: 0.8))
        let request = LandscapeRequest(expression: "1/gamma(z)", domain: domain, resolution: 24,
                                       truncation: .uniform(2))
        let bundle = try NativeLandscape.build(request, refine: true)
        let scene = Scene(bundle: bundle, preset: bundle.manifest.presets[0])
        let h = scene.heightfield!
        let cell = 1.6 / 23
        let view = scene.camera.view
        let vis = HeightfieldVisibility(heightfield: h, view: view, depthTolerance: scene.depthTolerance)
        // The tested ink, in world space: the folds that show, and each
        // surface layer less what cannot be seen.
        let folds = h.visibleFolds(view: view, depthTolerance: scene.depthTolerance)
        let visible: [(layer: Layer, paths: PolylineSet<WorldSpace>)] = scene.layers.map {
            ($0, h.visibleSurfaceInk($0.paths, view: view, depthTolerance: scene.depthTolerance))
        }
        // The pit is a bowl: the graph is convex there, so the solid under
        // it is hollow, and a line of sight tangent to the graph -- which is
        // what a fold is -- runs under the graph, through the graph's own
        // material, until it leaves through the cut. No fold in a pit can
        // be seen; what bounds the rings there is the cut's top edge. The
        // traced fold runs to the zero (below), and the test hides it:
        // what little shows lies on the cut itself.
        let traced = h.foldCurves(view: view)
        func ink(_ set: PolylineSet<WorldSpace>) -> Double {
            (0..<set.count).reduce(0.0) { total, i in
                let p = Array(set[path: i])
                return total + zip(p, p.dropFirst()).reduce(0) { $0 + simd_length($1.1.v - $1.0.v) }
            }
        }
        // The only fold vertices that can be seen are on the cut, whose
        // lines of sight leave the solid at once: a fold is tested with no
        // depth tolerance, since its vertices lie on f exactly.
        let offCut = folds.vertices.map(\.y).max() ?? 0
        Check.expect(traced.count >= 1 && ink(traced) > 0.5 && offCut < 1e-3 * cell,
                     "no fold is seen in the pit: a line of sight tangent to a bowl's graph runs through the graph",
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
        // The top edge of the front wall, as drawn, projected onto the plate:
        // what a ring that passes behind it must end on.
        let outline = try bundle.layer("wall_edges").paths
        var topEdge: [P3<WorldSpace>] = []
        for i in 0..<outline.count {
            let p = Array(outline[path: i])
            if p.allSatisfy({ abs($0.y) < 1e-9 }), Set(p.map { $0.z }).count > 1, p.count > topEdge.count { topEdge = p }
        }
        let topEdgeOnPage = topEdge.map { view($0) }
        func toTopEdgeOnPage(_ v: P3<WorldSpace>) -> Double {
            let q = view(v)
            var best = Double.infinity
            for (a, b) in zip(topEdgeOnPage, topEdgeOnPage.dropFirst()) {
                let ab = SIMD2(b.x - a.x, b.y - a.y), aq = SIMD2(q.x - a.x, q.y - a.y)
                let l = simd_dot(ab, ab)
                let t = l > 0 ? min(max(simd_dot(aq, ab) / l, 0), 1) : 0
                best = min(best, simd_length(aq - t * ab))
            }
            return best
        }
        // Every end of every tested ring, sorted by what it ends on: the
        // domain's edge, the truncation, a zero, a fold (facing zero there), or
        // the traversal (hidden behind the top edge). The last two are the ones
        // the grid used to leave short.
        var worst = 0.0, ends = 0, worstTopEdge = 0.0, byTraversal = 0
        var kinds = (edge: 0, truncationHeight: 0, zero: 0)
        for (layer, paths) in visible where layer.spec.liesOnSurface {
            for i in 0..<paths.count {
                let p = paths[path: i]
                for end in [p.first!, p.last!] {
                    let onEdge = abs(end.x + 3.8) < 1e-6 || abs(end.x + 2.2) < 1e-6 || abs(end.y) < 1e-6 || abs(end.y - 0.8) < 1e-6
                    if onEdge { kinds.edge += 1; continue }
                    if end.z > 2 - 1e-6 { kinds.truncationHeight += 1; continue }
                    if end.z < 1e-6 { kinds.zero += 1; continue }
                    if abs(vis.facing(end.xy)) < 1e-6 {
                        ends += 1
                        let g = toFolds(end)
                        worst = max(worst, g)
                        if g > 0.02 * cell {
                            // On failure: the traversal's own account of the end
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
                        byTraversal += 1
                        let g = toTopEdgeOnPage(end)
                        worstTopEdge = max(worstTopEdge, g)
                        if g > 2 * 0.02 * cell {
                            // On failure: the traversal's account of the end and of
                            // a point a hair beyond it along the ring.
                            print(String(format: "    top edge-cut end (%.4f, %.4f, %.4f) %@: %.2e on plate, facing %.3f, run of %d",
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
        // Within the top edge's own tolerance, and the depth tolerance the test
        // forgives, which is the tolerance again.
        Check.expect(byTraversal >= 10 && worstTopEdge <= 2 * 0.02 * cell + 1e-9,
                     "every ring that passes behind the top edge ends on the top edge as drawn, on the plate",
                     String(format: "%d ends, worst %.2e against %.2e; %d on the edge, %d on the plateau, %d at zeros",
                            byTraversal, worstTopEdge, 2 * 0.02 * cell, kinds.edge, kinds.truncationHeight, kinds.zero))
        // The top edge and the truncation edge.
        let truncationEdge = try bundle.layer("truncation_edge").paths
        var truncationEdgeEnds: [P3<WorldSpace>] = []
        for i in 0..<truncationEdge.count {
            let p = truncationEdge[path: i]
            for end in [p.first!, p.last!] where abs(end.y) < 1e-9 { truncationEdgeEnds.append(end) }
        }
        var missed = 0.0
        for end in truncationEdgeEnds { missed = max(missed, topEdge.map { simd_length($0.v - end.v) }.min() ?? .infinity) }
        Check.expect(truncationEdgeEnds.count >= 1 && missed < 1e-9, "the truncation edge's ends on the front edge are vertices of the top edge",
                     String(format: "%d ends, worst gap %.1e", truncationEdgeEnds.count, missed))
        let topEdgeLow = topEdge.min { $0.z < $1.z }!
        Check.expect(abs(topEdgeLow.x + 3) < 1e-6 && topEdgeLow.z < 1e-9, "the top edge bottoms out at the zero",
                     String(format: "(%.6f, %.1e)", topEdgeLow.x, topEdgeLow.z))
        let foldAtZero = traced.vertices.contains { abs($0.x + 3) < 1e-6 && abs($0.y) < 1e-6 && $0.z < 1e-9 }
        Check.expect(foldAtZero, "and the traced fold ends there")
        // The top edge and the folds are curves to the tolerance: no fold chord
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
