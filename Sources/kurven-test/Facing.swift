import Foundation
import simd
import KurvenCore
import KurvenMetal
import KurvenBake
import KurvenLandscape

// MARK: - ink on a heightfield is hidden where the surface faces away
//
// A Gaussian ridge along y, seen from a plate camera whose line of sight has a
// component across it: the near graph faces the eye, the steep part of the
// far graph faces away, and the far graph's lower slope faces the eye again. Ink run
// straight over the ridge on the surface must be cut at the two folds, found
// on the exact gradient; with nothing in front of it, the depth test alone
// keeps every vertex, which is the leak this test exists to close.

func facingTests() {
    Check.suite("facing: ink on a heightfield is hidden where the surface turns away") {
        let amplitude = 3.0, width = 0.5
        func height(_ x: Double) -> Double { amplitude * exp(-x * x / (width * width)) }
        func slope(_ x: Double) -> Double { -2 * x / (width * width) * height(x) }
        let nx = 801, ny = 21
        let domain = Domain(real: Interval(lo: -2, hi: 2), imag: Interval(lo: 0, hi: 1))
        var values = [Float](repeating: 0, count: nx * ny)
        for j in 0..<ny {
            for i in 0..<nx {
                values[j * nx + i] = Float(height(-2 + 4 * Double(i) / Double(nx - 1)))
            }
        }
        let grid = Grid2D(width: nx, height: ny, domain: domain, values: values)
        let surface = Surface(height: grid, phase: nil, truncation: .none)
        let field = Heightfield(surface: surface, occluder: Mesh(vertices: [], triangles: []),
                                tiles: [.identity], step: 1)

        // A camera whose line of sight crosses the ridge: of the two plate
        // turns, the one that looks more along x.
        let cameras = [0.0, 90.0].map {
            Camera.plate(PlateProjection(shear: 0, xAngle: -55, zAngle: $0, flipX: false, yScale: nil))
        }
        let camera = cameras.max { abs($0.view.projectionDirection.x) < abs($1.view.projectionDirection.x) }!
        let sight = camera.view.projectionDirection
        Check.expect(abs(sight.x) > 0.3, "the camera looks across the ridge",
                     String(format: "sight (%.2f, %.2f, %.2f)", sight.x, sight.y, sight.z))

        // The normal by differences agrees with the analytic gradient.
        var worst = 0.0
        for x in stride(from: -1.5, through: 1.5, by: 0.05) {
            let n = field.normal(at: P2<WorldSpace>(x, 0.5))
            worst = max(worst, abs(n.x + slope(x)) / max(1, abs(slope(x))))
        }
        Check.expect(worst < 1e-3, "the surface normal is the gradient, read off the grid",
                     String(format: "worst relative slope error %.1e", worst))

        // Nothing in front of anything: the depth test keeps every vertex.
        let ink = PolylineSet<WorldSpace>(paths: [stride(from: -2.0, through: 2.0, by: 0.01).map {
            P3<WorldSpace>($0, 0.5, height($0))
        }])
        let projected = ink.mapped(camera.view)
        let scene = Scene(surface: surface, occluder: Mesh(vertices: [], triangles: []),
                          tiles: [.identity], step: 1, layers: [], camera: camera, depthTolerance: 0.02)
        guard let bounds = scene.viewBounds() else {
            Check.expect(false, "the ridge has bounds"); return
        }
        let frame = DepthFrame(covering: bounds, resolution: 100)
        let empty = DepthImage(frame: frame,
                               values: [Float](repeating: -.infinity, count: frame.rows * frame.cols),
                               empty: -.infinity)
        let byDepth = HiddenLine.clip(projected, against: empty, depthTolerance: 0.02)
        // An infinite depth tolerance disarms the traversal: the facing alone decides.
        let cut = field.visibleSurfaceInk(ink, view: camera.view, depthTolerance: .infinity).mapped(camera.view)
        Check.expect(byDepth.count == 1 && byDepth.vertices.count == ink.vertices.count,
                     "tested by depth alone, the whole path is kept")

        // Where the surface turns edge-on: facing = -(n · sight)/|n| = 0,
        // with n = (-h', 0, 1), so h' = sight.z / sight.x.
        let critical = sight.z / sight.x
        var folds: [Double] = []
        var previous = -2.0
        for x in stride(from: -1.99, through: 2.0, by: 0.01) {
            if (slope(previous) - critical > 0) != (slope(x) - critical > 0) {
                var lo = previous, hi = x
                for _ in 0..<60 {
                    let mid = 0.5 * (lo + hi)
                    if (slope(lo) - critical > 0) == (slope(mid) - critical > 0) { lo = mid } else { hi = mid }
                }
                folds.append(0.5 * (lo + hi))
            }
            previous = x
        }
        Check.expect(folds.count == 2, "the far graph has two folds: steep enough to turn away, then not",
                     "folds at \(folds.map { String(format: "%.3f", $0) })")
        guard folds.count == 2, cut.count == 2 else {
            Check.expect(cut.count == 2, "the ink is cut into two runs, one each side of the hidden stretch",
                         "\(cut.count) runs"); return
        }
        let ends = [cut.vertices[cut.offsets[1] - 1], cut.vertices[cut.offsets[1]]]
        let expected = folds.map { camera.view(P3<WorldSpace>($0, 0.5, height($0))) }
        let miss = zip(ends, expected).map { simd_length($0.v - $1.v) }.max()!
        Check.expect(cut.count == 2 && miss < 2e-3,
                     "the ink is cut into two runs, ending and resuming on the exact folds",
                     String(format: "fold vertices within %.1e of the analytic folds", miss))
        let kept = (cut.offsets[1] - cut.offsets[0]) + (cut.offsets[2] - cut.offsets[1])
        Check.expect(kept < ink.vertices.count && kept > ink.vertices.count / 2,
                     "and the stretch between them, facing away, is gone",
                     "\(ink.vertices.count - kept) of \(ink.vertices.count) vertices hidden")
        // With the traversal armed, the far lower slope -- facing the eye again, but
        // behind the ridge from where the eye is -- goes too: the line of sight
        // from it passes under the top edge.
        let visible = field.visibleSurfaceInk(ink, view: camera.view, depthTolerance: 0.02).mapped(camera.view)
        let lastKept = visible.vertices.last.map { simd_length($0.v - expected[0].v) } ?? .infinity
        Check.expect(visible.count == 1 && lastKept < 2e-3,
                     "tested by the solid as well, only the near graph remains, up to the first fold",
                     "\(visible.count) runs, last vertex \(String(format: "%.1e", lastKept)) from the fold")
    }
}

// MARK: - the fold curves of a heightfield
//
// The same ridge: its folds are the two lines x = const where h'(x) is the
// line of sight's slope, straight along y. `Heightfield.foldCurves` must find
// both, nothing else, and put every vertex on them to rounding. Then a real
// landscape: every fold vertex is where the surface faces exactly edge-on,
// none sits on a plateau, and the pits have their far edges.

func foldCurveTests() {
    Check.suite("folds: a heightfield's fold curves are where it turns edge-on") {
        let amplitude = 3.0, width = 0.5
        func height(_ x: Double) -> Double { amplitude * exp(-x * x / (width * width)) }
        func slope(_ x: Double) -> Double { -2 * x / (width * width) * height(x) }
        let nx = 801, ny = 21
        let domain = Domain(real: Interval(lo: -2, hi: 2), imag: Interval(lo: 0, hi: 1))
        var values = [Float](repeating: 0, count: nx * ny)
        for j in 0..<ny {
            for i in 0..<nx {
                values[j * nx + i] = Float(height(-2 + 4 * Double(i) / Double(nx - 1)))
            }
        }
        let surface = Surface(height: Grid2D(width: nx, height: ny, domain: domain, values: values),
                              phase: nil, truncation: .none)
        let field = Heightfield(surface: surface, occluder: Mesh(vertices: [], triangles: []),
                                tiles: [.identity], step: 1)
        let cameras = [0.0, 90.0].map {
            Camera.plate(PlateProjection(shear: 0, xAngle: -55, zAngle: $0, flipX: false, yScale: nil))
        }
        let camera = cameras.max { abs($0.view.projectionDirection.x) < abs($1.view.projectionDirection.x) }!
        let sight = camera.view.projectionDirection
        let critical = sight.z / sight.x
        var folds: [Double] = []
        var previous = -2.0
        for x in stride(from: -1.99, through: 2.0, by: 0.01) {
            if (slope(previous) - critical > 0) != (slope(x) - critical > 0) {
                var lo = previous, hi = x
                for _ in 0..<60 {
                    let mid = 0.5 * (lo + hi)
                    if (slope(lo) - critical > 0) == (slope(mid) - critical > 0) { lo = mid } else { hi = mid }
                }
                folds.append(0.5 * (lo + hi))
            }
            previous = x
        }
        let traced = field.foldCurves(view: camera.view)
        Check.expect(traced.count == 2, "two fold curves, one per analytic fold",
                     "\(traced.count) lines for folds at \(folds.map { String(format: "%.3f", $0) })")
        var worst = 0.0
        for v in traced.vertices {
            worst = max(worst, folds.map { abs(v.x - $0) }.min() ?? .infinity)
        }
        Check.expect(worst < 1e-3, "every vertex lies on one of them",
                     String(format: "worst %.1e in x", worst))
        for i in 0..<traced.count {
            let ys = traced[path: i].map(\.y)
            Check.expect((ys.min() ?? 1) < 0.05 && (ys.max() ?? 0) > 0.95,
                         "and each runs the whole width of the ridge",
                         String(format: "y from %.2f to %.2f", ys.min() ?? 0, ys.max() ?? 0))
        }
        let grid = field.foldCurves(view: camera.view, refined: false)
        var gridWorst = 0.0
        for v in grid.vertices {
            gridWorst = max(gridWorst, folds.map { abs(v.x - $0) }.min() ?? .infinity)
        }
        Check.expect(grid.count == traced.count && gridWorst < 4.0 / Double(nx - 1),
                     "read off the grid alone, the same lines within a cell",
                     String(format: "worst %.1e in x", gridWorst))
    }

    Check.suite("folds: a landscape's pits have their far edges") {
        let preset = Catalog.native.preset("rgamma")!
        let bundle = try NativeLandscape.build(LandscapeRequest(preset: preset, resolution: 240))
        let scene = Scene(bundle: bundle, preset: bundle.manifest.presets[0])
        let h = scene.heightfield!
        Check.expect(scene.layers.contains { if case .foldCurves = $0.spec.source { return true }; return false },
                     "a landscape describes its fold curves")
        let folds = h.foldCurves(view: scene.camera.view)
        let ink = (0..<folds.count).reduce(0.0) { total, i in
            let p = folds[path: i]
            return total + zip(p, p.dropFirst()).reduce(0) { $0 + simd_length($1.1.v - $1.0.v) }
        }
        Check.expect(folds.count > 0 && ink > 1, "it has fold curves", String(format: "%d lines, %.2f units", folds.count, ink))
        guard scene.viewBounds() != nil else { Check.expect(false, "the plate has bounds"); return }
        let visibility = HeightfieldVisibility(heightfield: h, view: scene.camera.view, depthTolerance: 0.02)
        // Every vertex but a run's end carried to a zero of f, which is on
        // the floor, where the grid's facing is its chord's.
        // Every vertex but a run's end carried to a zero (on the floor) or
        // to the truncation edge (on the crease, where the facing jumps).
        let truncationAt = h.surface.truncation
        let offFold = folds.vertices.filter { $0.z > 1e-9 && $0.z < truncationAt.height(atX: $0.x) - 1e-9 }
            .map { ($0, abs(visibility.facing(P2($0.x, $0.y)))) }
        let worstOff = offFold.max { $0.1 < $1.1 }
        let edgeOn = worstOff?.1 ?? .infinity
        let where_ = worstOff.map { String(format: " at (%.4f, %.4f, %.4f), truncation %.3f, |f| %.4f", $0.0.x, $0.0.y, $0.0.z,
                                           h.surface.truncation.height(atX: $0.0.x), h.surface.magnitude(at: P2<DomainSpace>($0.0.x, $0.0.y))) } ?? ""
        let offCount = offFold.filter { $0.1 > 1e-6 }.count
        Check.expect(edgeOn < 1e-6, "every vertex is where the surface faces exactly edge-on",
                     String(format: "worst |facing| %.1e", edgeOn) + where_ + ", \(offCount) of \(offFold.count) off")
        let atZeros = folds.vertices.filter { $0.z <= 1e-9 }.count
        Check.expect(atZeros >= 2, "and the pits' outlines run down to their zeros on the front edge",
                     "\(atZeros) fold ends on the floor")
        // Nothing on a plateau but the one vertex where a fold up a graph meets
        // the truncation edge, which ends its run.
        let truncationHeight = h.surface.truncation
        var interiorOnPlateau = 0, endsOnTruncationEdge = 0
        for i in 0..<folds.count {
            let p = Array(folds[path: i])
            for (k, v) in p.enumerated() where v.z >= truncationHeight.height(atX: v.x) - 1e-9 {
                if k == 0 || k == p.count - 1 { endsOnTruncationEdge += 1 } else { interiorOnPlateau += 1 }
            }
        }
        Check.expect(interiorOnPlateau == 0, "and none is on a plateau but a run's end on the truncation edge",
                     "\(interiorOnPlateau) interior vertices on a plateau, \(endsOnTruncationEdge) ends on a truncation edge")
        // The pit at -3: its far edge is a fold running down to the floor.
        let pit = folds.vertices.filter { abs($0.x + 3) < 0.4 && $0.y < 0.6 }
        Check.expect((pit.map(\.z).min() ?? .infinity) < 0.5,
                     "the pit at -3 has a fold reaching down its wall",
                     String(format: "lowest fold vertex at z = %.3f", pit.map(\.z).min() ?? .infinity))
        // The bake draws them: tested by depth, not by facing.
        let baked = try MetalRenderer().bake(scene, options: BakeOptions(resolution: 400))
        let drawn = baked.strokes.layers.last!.paths
        Check.expect(drawn.vertices.count > 0, "the bake draws the fold layer",
                     "\(drawn.count) strokes")
    }
}

// MARK: - ink in a cut face is tested by the wall it lies in
//
// The plate camera sees the right wall of rgamma nearly edge-on, and the
// depth buffer erased most of its hatch, leaving spots. Visible by the wall
// instead: every stroke on a wall that faces the eye is kept whole, every
// stroke on a wall that faces away is dropped whole, a corner vertical edge stays
// while either of its walls shows, and the bake draws exactly that.

func wallInkTests() {
    Check.suite("walls: the hatch and outline of a cut face are tested by the wall") {
        let preset = Catalog.native.preset("rgamma")!
        let bundle = try NativeLandscape.build(LandscapeRequest(preset: preset, resolution: 240))
        let scene = Scene(bundle: bundle, preset: bundle.manifest.presets[0])
        let h = scene.heightfield!
        let sight = scene.camera.view.projectionDirection
        let d = h.surface.domain
        let box = (lo: SIMD2(min(d.real.lo, d.real.hi), min(d.imag.lo, d.imag.hi)),
                   hi: SIMD2(max(d.real.lo, d.real.hi), max(d.imag.lo, d.imag.hi)))
        // Which walls face the eye, by their outward normals: left, right,
        // front, back.
        let faces = [SIMD3<Double>(-1, 0, 0), SIMD3(1, 0, 0), SIMD3(0, -1, 0), SIMD3(0, 1, 0)]
            .map { -simd_dot($0, sight) > 0 }
        Check.expect(faces.contains(true) && faces.contains(false),
                     "the plate camera sees some walls and not others",
                     "left \(faces[0]) right \(faces[1]) front \(faces[2]) back \(faces[3])")
        func wallsOf(_ p: P3<WorldSpace>) -> [Int] {
            let slack = 1e-6
            var on: [Int] = []
            if abs(p.x - box.lo.x) <= slack { on.append(0) }
            if abs(p.x - box.hi.x) <= slack { on.append(1) }
            if abs(p.y - box.lo.y) <= slack { on.append(2) }
            if abs(p.y - box.hi.y) <= slack { on.append(3) }
            return on
        }
        func length(_ p: [P3<WorldSpace>]) -> Double {
            zip(p, p.dropFirst()).reduce(0) { $0 + simd_length($1.1.v - $1.0.v) }
        }
        func length(_ s: PolylineSet<WorldSpace>) -> Double {
            (0..<s.count).reduce(0.0) { $0 + length(Array(s[path: $1])) }
        }
        // The hatch lies in the face: whole strokes on facing walls, nothing
        // on the others.
        let hatch = scene.layers.first { $0.spec.name == "wall_hatch" }!
        let keptHatch = h.visibleWallInk(hatch.paths, view: scene.camera.view, depthTolerance: scene.depthTolerance)
        var wanted = 0.0
        for i in 0..<hatch.paths.count {
            let p = Array(hatch.paths[path: i])
            if p.allSatisfy({ v in wallsOf(v).contains { faces[$0] } }) { wanted += length(p) }
        }
        Check.expect(abs(length(keptHatch) - wanted) < 1e-9 * max(wanted, 1),
                     "wall_hatch: what is kept is every stroke on a wall that faces the eye",
                     String(format: "%.3f of %.3f units, %d of %d paths", length(keptHatch),
                            length(hatch.paths), keptHatch.count, hatch.paths.count))
        let strays = keptHatch.vertices.filter { v in !wallsOf(v).contains { faces[$0] } }
        let strayNote = strays.prefix(3).map { v in
            String(format: "(%.3f, %.3f, %.4f) f %.4f grid %.4f walls %@", v.x, v.y, v.z,
                   h.surfaceHeight(at: P2(v.x, v.y)), h.surface.height(at: P2<DomainSpace>(v.x, v.y)),
                   wallsOf(v).description)
        }.joined(separator: "; ")
        Check.expect(strays.isEmpty, "wall_hatch: and nothing on a wall that faces away",
                     "\(strays.count) stray vertices " + strayNote)
        if let stray = strays.first {
            let visibility = HeightfieldVisibility(heightfield: h, view: scene.camera.view, depthTolerance: scene.depthTolerance)
            for i in 0..<hatch.paths.count {
                let p = Array(hatch.paths[path: i])
                guard p.contains(where: { simd_length($0.v - stray.v) < 1e-12 }) else { continue }
                print("    the stroke: " + p.map { String(format: "%.4f", $0.z) }.joined(separator: " "))
                for j in 0..<keptHatch.count {
                    let q = Array(keptHatch[path: j])
                    if q.contains(where: { simd_length($0.v - stray.v) < 1e-12 }) {
                        print("    kept as: " + q.map { String(format: "(%.3f, %.3f, %.5f)", $0.x, $0.y, $0.z) }.joined(separator: " "))
                    }
                }
                print("    " + visibility.explain(stray).replacingOccurrences(of: "\n", with: "\n    "))
                break
            }
        }

        // The outline: a top edge is the surface's edge and shows over any wall
        // where the surface does; a bottom edge shows only on a facing wall.
        let outline = scene.layers.first { $0.spec.name == "wall_edges" }!
        let keptOutline = h.visibleWallInk(outline.paths, view: scene.camera.view, depthTolerance: scene.depthTolerance)
        func isTopEdge(_ p: [P3<WorldSpace>]) -> Bool {
            p.allSatisfy { abs($0.z - h.surfaceHeight(at: P2($0.x, $0.y))) < 1e-9 * max(1, abs($0.z)) }
                && !p.allSatisfy { $0.z == 0 }
        }
        var topEdgeWanted = [Double](repeating: 0, count: 4), topEdgeGot = [Double](repeating: 0, count: 4)
        var bottomEdgeHidden = true
        for i in 0..<outline.paths.count {
            let p = Array(outline.paths[path: i])
            guard let wall = wallsOf(p[0]).first(where: { w in p.allSatisfy { wallsOf($0).contains(w) } }) else { continue }
            if isTopEdge(p) { topEdgeWanted[wall] += length(p) }
        }
        for i in 0..<keptOutline.count {
            let p = Array(keptOutline[path: i])
            guard let wall = wallsOf(p[0]).first(where: { w in p.allSatisfy { wallsOf($0).contains(w) } }) else { continue }
            if isTopEdge(p) {
                topEdgeGot[wall] += length(p)
            } else if p.allSatisfy({ $0.z == 0 }), !faces[wall] {
                bottomEdgeHidden = false
            }
        }
        Check.expect(zip(topEdgeGot, topEdgeWanted).allSatisfy { $0 >= 0.95 * $1 },
                     "wall_outline: every wall's top edge is kept, the far edge included",
                     "kept " + zip(topEdgeGot, topEdgeWanted).map { String(format: "%.1f/%.1f", $0, $1) }.joined(separator: " "))
        Check.expect(bottomEdgeHidden, "wall_outline: and no foot of a wall that faces away")

        // Where the truncation edge of a plateau ends at the front edge -- the notch of a
        // pit -- the top edge has a vertex on it, so the two meet.
        let truncationEdge = scene.layers.first { $0.spec.name == "truncation_edge" }!.paths
        var truncationEdgeEnds: [P3<WorldSpace>] = []
        for i in 0..<truncationEdge.count {
            let p = truncationEdge[path: i]
            for end in [p.first!, p.last!] where abs(end.y - box.lo.y) <= 1e-9 { truncationEdgeEnds.append(end) }
        }
        let topEdgeVertices = outline.paths.vertices.filter { abs($0.y - box.lo.y) <= 1e-9 }
        var missed = 0.0
        for end in truncationEdgeEnds {
            let nearest = topEdgeVertices.map { simd_length($0.v - end.v) }.min() ?? .infinity
            missed = max(missed, nearest)
        }
        Check.expect(truncationEdgeEnds.count >= 2 && missed < 1e-9,
                     "the top edge meets the truncation edge where the truncation edge ends, at every notch",
                     String(format: "%d truncation edge ends on the front edge, worst gap %.1e", truncationEdgeEnds.count, missed))
        // The bake draws the tested ink as it is: the hatch on the edge-on
        // right wall is whole, not spotty.
        let baked = try MetalRenderer().bake(scene, options: BakeOptions(resolution: 400))
        let hatchIndex = scene.layers.firstIndex { $0.spec.name == "wall_hatch" }!
        let drawn = baked.strokes.layers[hatchIndex].paths
        let visible = h.visibleWallInk(scene.layers[hatchIndex].paths, view: scene.camera.view,
                                      depthTolerance: scene.depthTolerance)
        Check.expect(drawn.count == visible.count && drawn.vertices.count == visible.vertices.count,
                     "the bake draws the wall hatch exactly as tested, stroke for stroke",
                     "\(drawn.count) strokes drawn, \(visible.count) tested")
    }
}
