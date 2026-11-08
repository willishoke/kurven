import Foundation
import simd
import KurvenCore
import KurvenMetal
import KurvenBake
import KurvenLandscape

// MARK: - curves on a heightfield are hidden where the surface faces away
//
// A Gaussian ridge along y, seen from an oblique camera whose line of sight has a
// component across it: the near graph faces the eye, the steep part of the
// far graph faces away, and the far graph's lower slope faces the eye again. Curves run
// straight over the ridge on the surface must be cut at the two folds, found
// on the exact gradient; with nothing in front of it, the depth test alone
// keeps every vertex, which is the leak this test exists to close.

func facingTests() {
    Check.suite("facing: curves on a heightfield is hidden where the surface turns away") {
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
                                tiles: [.identity], step: 1, walls: .none)

        // A camera whose line of sight crosses the ridge: of the two drawing
        // turns, the one that looks more along x.
        let cameras = [0.0, 90.0].map {
            Camera.oblique(ObliqueProjection(shear: 0, xAngle: -55, zAngle: $0, flipX: false, yScale: nil))
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
        let curves = PolylineSet<WorldSpace>(paths: [stride(from: -2.0, through: 2.0, by: 0.01).map {
            P3<WorldSpace>($0, 0.5, height($0))
        }])
        let projected = curves.mapped(camera.view)
        let scene = Scene(surface: surface, occluder: Mesh(vertices: [], triangles: []),
                          tiles: [.identity], step: 1, layers: [], camera: camera, depthTolerance: 0.02,
                          walls: .none)
        guard let bounds = scene.viewBounds() else {
            Check.expect(false, "the ridge has bounds"); return
        }
        let frame = DepthFrame(covering: bounds, resolution: 100)
        let empty = DepthImage(frame: frame,
                               values: [Float](repeating: -.infinity, count: frame.rows * frame.cols),
                               empty: -.infinity)
        let byDepth = HiddenLine.clip(projected, against: empty, depthTolerance: 0.02)
        // An infinite depth tolerance disarms the traversal: the facing alone decides.
        let cut = field.visibleCurves(curves, on: .graph, view: camera.view, depthTolerance: .infinity).mapped(camera.view)
        Check.expect(byDepth.count == 1 && byDepth.vertices.count == curves.vertices.count,
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
            Check.expect(cut.count == 2, "the curves are cut into two runs, one each side of the hidden stretch",
                         "\(cut.count) runs"); return
        }
        let ends = [cut.vertices[cut.offsets[1] - 1], cut.vertices[cut.offsets[1]]]
        let expected = folds.map { camera.view(P3<WorldSpace>($0, 0.5, height($0))) }
        let miss = zip(ends, expected).map { simd_length($0.v - $1.v) }.max()!
        Check.expect(cut.count == 2 && miss < 2e-3,
                     "the curves are cut into two runs, ending and resuming on the exact folds",
                     String(format: "fold vertices within %.1e of the analytic folds", miss))
        let kept = (cut.offsets[1] - cut.offsets[0]) + (cut.offsets[2] - cut.offsets[1])
        Check.expect(kept < curves.vertices.count && kept > curves.vertices.count / 2,
                     "and the stretch between them, facing away, is gone",
                     "\(curves.vertices.count - kept) of \(curves.vertices.count) vertices hidden")
        // With the traversal armed, the far lower slope -- facing the eye again, but
        // behind the ridge from where the eye is -- goes too: the line of sight
        // from it passes under the top edge.
        let visible = field.visibleCurves(curves, on: .graph, view: camera.view, depthTolerance: 0.02).mapped(camera.view)
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
                                tiles: [.identity], step: 1, walls: .none)
        let cameras = [0.0, 90.0].map {
            Camera.oblique(ObliqueProjection(shear: 0, xAngle: -55, zAngle: $0, flipX: false, yScale: nil))
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
        let curves = (0..<folds.count).reduce(0.0) { total, i in
            let p = folds[path: i]
            return total + zip(p, p.dropFirst()).reduce(0) { $0 + simd_length($1.1.v - $1.0.v) }
        }
        Check.expect(folds.count > 0 && curves > 1, "it has fold curves", String(format: "%d lines, %.2f units", folds.count, curves))
        guard scene.viewBounds() != nil else { Check.expect(false, "the drawing has bounds"); return }
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

// MARK: - curves on a wall are tested by the wall they lie on
//
// The oblique camera sees the right wall of rgamma nearly edge-on, and the
// depth buffer erased most of its hatch, leaving spots. Visible by the wall
// instead: every stroke on a wall that faces the eye is kept whole, every
// stroke on a wall that faces away is dropped whole, a corner vertical edge stays
// while either of its walls shows, and the bake draws exactly that.

func wallCurveTests() {
    Check.suite("walls: the hatching and edges of a cut face are tested by the wall") {
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
                     "the oblique camera sees some walls and not others",
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
        let keptHatch = h.visibleWallCurves(hatch.paths, view: scene.camera.view, depthTolerance: scene.depthTolerance)
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

        // The edges: a top edge is the surface's edge and shows over any wall
        // where the surface does; a bottom edge shows only on a facing wall.
        let wallEdges = scene.layers.first { $0.spec.name == "wall_edges" }!
        let keptEdges = h.visibleWallCurves(wallEdges.paths, view: scene.camera.view, depthTolerance: scene.depthTolerance)
        func isTopEdge(_ p: [P3<WorldSpace>]) -> Bool {
            p.allSatisfy { abs($0.z - h.surfaceHeight(at: P2($0.x, $0.y))) < 1e-9 * max(1, abs($0.z)) }
                && !p.allSatisfy { $0.z == 0 }
        }
        var topEdgeWanted = [Double](repeating: 0, count: 4), topEdgeGot = [Double](repeating: 0, count: 4)
        var bottomEdgeHidden = true
        for i in 0..<wallEdges.paths.count {
            let p = Array(wallEdges.paths[path: i])
            guard let wall = wallsOf(p[0]).first(where: { w in p.allSatisfy { wallsOf($0).contains(w) } }) else { continue }
            if isTopEdge(p) { topEdgeWanted[wall] += length(p) }
        }
        for i in 0..<keptEdges.count {
            let p = Array(keptEdges[path: i])
            guard let wall = wallsOf(p[0]).first(where: { w in p.allSatisfy { wallsOf($0).contains(w) } }) else { continue }
            if isTopEdge(p) {
                topEdgeGot[wall] += length(p)
            } else if p.allSatisfy({ $0.z == 0 }), !faces[wall] {
                bottomEdgeHidden = false
            }
        }
        Check.expect(zip(topEdgeGot, topEdgeWanted).allSatisfy { $0 >= 0.95 * $1 },
                     "wall_edges: every wall's top edge is kept, the far edge included",
                     "kept " + zip(topEdgeGot, topEdgeWanted).map { String(format: "%.1f/%.1f", $0, $1) }.joined(separator: " "))
        Check.expect(bottomEdgeHidden, "wall_edges: and no bottom edge of a wall that faces away")

        // Where the truncation edge of a plateau ends at the front edge -- the notch of a
        // pit -- the top edge has a vertex on it, so the two meet.
        let truncationEdge = scene.layers.first { $0.spec.name == "truncation_edge" }!.paths
        var truncationEdgeEnds: [P3<WorldSpace>] = []
        for i in 0..<truncationEdge.count {
            let p = truncationEdge[path: i]
            for end in [p.first!, p.last!] where abs(end.y - box.lo.y) <= 1e-9 { truncationEdgeEnds.append(end) }
        }
        let topEdgeVertices = wallEdges.paths.vertices.filter { abs($0.y - box.lo.y) <= 1e-9 }
        var missed = 0.0
        for end in truncationEdgeEnds {
            let nearest = topEdgeVertices.map { simd_length($0.v - end.v) }.min() ?? .infinity
            missed = max(missed, nearest)
        }
        Check.expect(truncationEdgeEnds.count >= 2 && missed < 1e-9,
                     "the top edge meets the truncation edge where the truncation edge ends, at every notch",
                     String(format: "%d truncation edge ends on the front edge, worst gap %.1e", truncationEdgeEnds.count, missed))
        // The bake draws the tested curves as it is: the hatch on the edge-on
        // right wall is whole, not spotty.
        let baked = try MetalRenderer().bake(scene, options: BakeOptions(resolution: 400))
        let hatchIndex = scene.layers.firstIndex { $0.spec.name == "wall_hatch" }!
        let drawn = baked.strokes.layers[hatchIndex].paths
        let visible = h.visibleWallCurves(scene.layers[hatchIndex].paths, view: scene.camera.view,
                                      depthTolerance: scene.depthTolerance)
        Check.expect(drawn.count == visible.count && drawn.vertices.count == visible.vertices.count,
                     "the bake draws the wall hatch exactly as tested, stroke for stroke",
                     "\(drawn.count) strokes drawn, \(visible.count) tested")
    }
}

// MARK: - a wall is its face, wherever it stands
//
// Which walls a point lies in is read off the walls' faces, not off the
// tiles' boxes: a wall on a notch in the footprint stands on no box's edge,
// and a box's edge where two tiles meet is no wall's. With the traversal
// disarmed, the facing alone decides, and a stroke is kept exactly when a
// wall it stands on faces the eye.

func wallFaceTests() {
    Check.suite("walls: a stroke on a wall is tested by the wall it stands on, on a notch and at a tile boundary") {
        let index = try Fixtures.json("hatch/index.json")
        let shape = try index.value("shape", "hatch").object("hatch.shape")
        let grid = Grid2D(width: try shape.int("nx", "hatch.shape"), height: try shape.int("ny", "hatch.shape"),
                          domain: try Domain(json: index.value("domain", "hatch")),
                          values: try NPY.read(contentsOf: Fixtures.url("hatch/height.npy")).floats())
        let surface = Surface(height: grid, phase: nil, truncation: .uniform(2))
        func perimeter(_ corners: [(Double, Double)]) -> BoundaryPerimeter {
            BoundaryPerimeter(edges: corners.indices.map { k in
                let a = corners[k], b = corners[(k + 1) % corners.count]
                return PerimeterEdge(start: P2(a.0, a.1), end: P2(b.0, b.1), density: 9)
            })
        }
        let box = perimeter([(-2, -1), (2.5, -1), (2.5, 1.5), (-2, 1.5)])
        let notched = perimeter([(-2, -1), (0.5, -1), (0.5, 0), (1.5, 0), (1.5, -1), (2.5, -1), (2.5, 1.5), (-2, 1.5)])
        let clockwise = perimeter([(-2, 1.5), (2.5, 1.5), (2.5, -1), (1.5, -1), (1.5, 0), (0.5, 0), (0.5, -1), (-2, -1)])
        let layouts: [(name: String, walls: BoundaryPerimeter, region: Region, tiles: [Affine2])] = [
            ("a notch cut from the front", notched, .inside(notched), [.identity]),
            ("the notched footprint walked clockwise", clockwise, .inside(clockwise), [.identity]),
            // The first tile's walls, and a second tile beside it: the box
            // edge they meet on is the first tile's right wall alone.
            ("a second tile beside the first", box, .full, [.identity, Affine2(a: 1, b: 0, tx: 4.5, c: 0, d: 1, ty: 0)]),
        ]
        for layout in layouts {
            let h = Heightfield(surface: surface, occluder: .empty, tiles: layout.tiles, region: layout.region,
                                step: 1, walls: .perimeter(layout.walls, base: 0))
            let context = Surface.HatchContext(perimeter: layout.walls, region: layout.region, tiles: layout.tiles)
            // The walls' strokes, from the floor: where the second tile's top
            // stands over the first's at the box edge they meet on, a riser
            // stands over the wall there, and its hatching, a hair onto the
            // lower side, rises from its bottom (`riserFaceTests`).
            let both = surface.hatch(.wallHatching(edges: Array(layout.walls.edges.indices), spacing: 0.1),
                                     in: context)!
            let hatch = PolylineSet(paths: (0..<both.count).map { Array(both[path: $0]) }.filter { $0[0].z == 0 })
            // Each edge's outward normal, from the perimeter's turn.
            let area = layout.walls.signedArea
            let normals = layout.walls.edges.map { e -> SIMD3<Double> in
                let d = simd_normalize(e.end.v - e.start.v)
                return (area < 0 ? -1 : 1) * SIMD3(d.y, -d.x, 0)
            }
            func on(_ k: Int, _ v: P3<WorldSpace>) -> Bool {
                let e = layout.walls.edges[k]
                let along = e.end.v - e.start.v, r = SIMD2(v.x, v.y) - e.start.v
                let t = simd_dot(r, along) / simd_dot(along, along)
                return t > -1e-9 && t < 1 + 1e-9 && simd_length(r - t * along) < 1e-9
            }
            for (name, view) in [("from the left", testCamera(elevation: -55, azimuth: -60)),
                                 ("from the right", testCamera(elevation: -55, azimuth: 150))] {
                let sight = view.projectionDirection
                let kept = h.onGrid.visibleWallCurves(hatch, view: view, depthTolerance: .infinity)
                var wanted = 0, facingAway = 0
                for i in 0..<hatch.count {
                    let p = Array(hatch[path: i])
                    let walls = layout.walls.edges.indices.filter { k in p.allSatisfy { on(k, $0) } }
                    if walls.contains(where: { -simd_dot(normals[$0], sight) > 0 }) { wanted += 1 } else { facingAway += 1 }
                }
                let strays = kept.vertices.filter { v in
                    !layout.walls.edges.indices.contains { on($0, v) && -simd_dot(normals[$0], sight) > 0 }
                }
                Check.expect(kept.count == wanted && strays.isEmpty && facingAway > 0,
                             "\(layout.name), seen \(name): every stroke on a wall that faces the eye, and none on one that faces away",
                             "\(kept.count) of \(hatch.count) strokes kept, \(wanted) wanted, \(facingAway) facing away, \(strays.count) stray vertices")
            }
        }
    }

    // Where a wall's line of sight meets terrain -- the next copy, from a
    // wall on a tile boundary, or the far side of a notch -- the terrain hides
    // the wall's hatch from its bottom up. What it hides of the stroke at a
    // point of the wall is everything below the greatest of h(q(t)) - t d_z
    // along the line of sight, q(t) its footprint and h the drawn height
    // there: the highest the terrain stands over the line, carried back down
    // it. The walk reads that a five-hundredth of a unit at a time, then a
    // hundred-thousandth around every sample within reach of the greatest;
    // h is linear over each triangle, so the reach is its steepest slope
    // over the coarse step. Each stroke on a wall that faces the eye, tested
    // from its two ends and at 256 vertices, keeps one run from there to its
    // top, under every camera of the stress orbit. A run that starts within a
    // thousandth of the top is a stub where the top edge, which is on the
    // surface, is forgiven the depth tolerance's reach, and counts as nothing
    // kept.
    //
    // Across a notch, one disagreement is allowed, and counted: the
    // traversal keeps more than the walk where the terrain stands highest
    // over the line at the footprint's edge, since it does not visit the
    // line's crossings of the footprint's perimeter, and a crossing it visits
    // on the perimeter is inside or out by the even-odd rule's rounding.
    Check.suite("walls: a stroke is hidden up to where the walk up the line of sight meets terrain, on abutting copies and across a notch") {
        func perimeter(_ corners: [P2<WorldSpace>]) -> BoundaryPerimeter {
            BoundaryPerimeter(edges: corners.indices.map {
                PerimeterEdge(start: corners[$0], end: corners[($0 + 1) % corners.count], density: 9)
            })
        }
        var layouts: [(name: String, h: Heightfield, walls: BoundaryPerimeter)] = []
        for seed: UInt64 in [3, 17, 41] {
            // A tile, the next copy to its right and the next above it: two of
            // its walls stand on tile boundaries.
            let small = randomHeightfield(cells: 8, seed: seed, truncation: Truncation.none).surface
            let d = small.domain
            let box = perimeter([P2(d.real.lo, d.imag.lo), P2(d.real.hi, d.imag.lo),
                                 P2(d.real.hi, d.imag.hi), P2(d.real.lo, d.imag.hi)])
            let tiles = [Affine2.identity, Affine2(a: 1, b: 0, tx: d.real.length, c: 0, d: 1, ty: 0),
                         Affine2(a: 1, b: 0, tx: 0, c: 0, d: 1, ty: d.imag.length)]
            layouts.append(("seed \(seed), abutting copies",
                            Heightfield(surface: small, occluder: .empty, tiles: tiles, step: 1,
                                        walls: .perimeter(box, base: 0)), box))
            // A footprint with a notch cut into its far side, walled all round,
            // the notch along grid lines: inside a cell a wall's top edge
            // follows the surface's interpolated height, not the drawn
            // triangles, and the traversal hides it at its start.
            let fine = randomHeightfield(cells: 16, seed: seed, truncation: Truncation.none).surface
            let notched = perimeter([P2(-1, -1.5), P2(1, -1.5), P2(1, 0.5), P2(0.25, 0.5),
                                     P2(0.25, -0.5), P2(-0.25, -0.5), P2(-0.25, 0.5), P2(-1, 0.5)])
            layouts.append(("seed \(seed), a notch",
                            Heightfield(surface: fine, occluder: .empty, tiles: [.identity], region: .inside(notched),
                                        step: 1, walls: .perimeter(notched, base: 0)), notched))
        }
        for layout in layouts {
            let h = layout.h, dom = h.surface.domain
            let context = Surface.HatchContext(perimeter: layout.walls, region: h.region, tiles: h.tiles)
            let hatch = h.surface.hatch(.wallHatching(edges: Array(layout.walls.edges.indices), spacing: 0.1,
                                                      endLines: false, base: 0, topOffset: 0), in: context)!
            let dense = PolylineSet(paths: (0..<hatch.count).map { i -> [P3<WorldSpace>] in
                let p = Array(hatch[path: i]), a = p[0], b = p[p.count - 1]
                return (0...256).map { j in j == 256 ? b : P3(a.v + (b.v - a.v) * (Double(j) / 256)) }
            })
            // Each edge's outward normal, from the perimeter's turn.
            let area = layout.walls.signedArea
            let normals = layout.walls.edges.map { e -> SIMD2<Double> in
                let along = simd_normalize(e.end.v - e.start.v)
                return (area < 0 ? -1 : 1) * SIMD2(along.y, -along.x)
            }
            func wall(_ at: SIMD2<Double>) -> Int? {
                layout.walls.edges.indices.first { k in
                    let e = layout.walls.edges[k]
                    let along = e.end.v - e.start.v, r = at - e.start.v
                    let t = simd_dot(r, along) / simd_dot(along, along)
                    return t > -1e-9 && t < 1 + 1e-9 && simd_length(r - t * along) < 1e-9
                }
            }
            /// The drawn height over a point of the terrain, in the footprint
            /// and on a copy of the domain; nil elsewhere.
            func terrain(_ q: SIMD2<Double>) -> Double? {
                let w = P2<WorldSpace>(q.x, q.y)
                guard h.region.contains(w), h.tiles.contains(where: { tile in
                    guard let back = tile.inverse else { return false }
                    let u = back(P3<WorldSpace>(q.x, q.y, 0))
                    return u.x >= dom.real.lo && u.x <= dom.real.hi && u.y >= dom.imag.lo && u.y <= dom.imag.hi
                }) else { return nil }
                return h.drawnHeight(at: w)
            }
            var highest = 0.0, steepest = 0.0
            let g = h.surface.height
            let dx = dom.real.length / Double(g.width - 1), dy = dom.imag.length / Double(g.height - 1)
            for j in 0..<g.height {
                for i in 0..<g.width {
                    let z = Double(g[i, j])
                    highest = max(highest, z)
                    if i + 1 < g.width { steepest = max(steepest, abs(Double(g[i + 1, j]) - z) / dx) }
                    if j + 1 < g.height { steepest = max(steepest, abs(Double(g[i, j + 1]) - z) / dy) }
                }
            }
            var compared = 0, partly = 0, wrong = 0, explained = 0
            var example = ""
            for (name, view) in stressCameras() {
                let d = -view.projectionDirection, dxy = SIMD2(d.x, d.y)
                /// The height below which terrain hides the stroke at `at`,
                /// or its base; and where on the line's footprint the terrain
                /// stands highest over it.
                func hiddenBelow(_ at: SIMD2<Double>, base: Double) -> (height: Double, at: SIMD2<Double>?) {
                    let coarse = 2e-3, fine = 1e-5
                    let reach = 2 * (2.0.squareRoot() * steepest * simd_length(dxy) + abs(d.z)) * coarse
                    func value(_ t: Double) -> Double? { terrain(at + t * dxy).map { $0 - t * d.z } }
                    var samples: [(t: Double, value: Double)] = []
                    var best = base, tBest: Double?, t = fine
                    while highest - t * d.z > best - reach, t * simd_length(dxy) < 10 {
                        if let v = value(t) { samples.append((t, v)); if v > best { best = v; tBest = t } }
                        t += coarse
                    }
                    for s in samples where s.value > best - reach {
                        var u = max(s.t - coarse, fine)
                        while u < s.t + coarse {
                            if let v = value(u), v > best { best = v; tBest = u }
                            u += fine
                        }
                    }
                    return (best, tBest.map { at + $0 * dxy })
                }
                // Each kept run, by the stroke it lies on.
                func runs(_ set: PolylineSet<WorldSpace>) -> [SIMD2<Double>: [(lo: Double, hi: Double)]] {
                    var out: [SIMD2<Double>: [(lo: Double, hi: Double)]] = [:]
                    for i in 0..<set.count {
                        let p = Array(set[path: i])
                        out[SIMD2(p[0].x, p[0].y), default: []].append((p[0].z, p[p.count - 1].z))
                    }
                    return out
                }
                let two = runs(h.visibleWallCurves(hatch, view: view, depthTolerance: 0.02))
                let many = runs(h.visibleWallCurves(dense, view: view, depthTolerance: 0.02))
                for i in 0..<hatch.count {
                    let p = Array(hatch[path: i]), at = SIMD2(p[0].x, p[0].y)
                    let base = p[0].z, top = p[p.count - 1].z
                    // A wall seen edge-on is left out: its line of sight runs along
                    // the domain's edge, in and out of the terrain by rounding.
                    guard let k = wall(at), simd_dot(normals[k], dxy) > 1e-6 else { continue }
                    let (below, highestAt) = hiddenBelow(at, base: base)
                    compared += 1
                    let atEdge = layout.walls.edges.contains { e in
                        guard let q = highestAt, h.region != .full else { return false }
                        let along = e.end.v - e.start.v
                        let t = min(max(simd_dot(q - e.start.v, along) / simd_dot(along, along), 0), 1)
                        return simd_length(q - e.start.v - t * along) < 1e-4
                    }
                    if below > base + 1e-3, below < top - 1e-3 { partly += 1 }
                    for (kind, kept) in [("two ends", two[at] ?? []), ("256 vertices", many[at] ?? [])] {
                        let runs = kept.filter { $0.lo < top - 1e-3 }
                        let agree = below < top - 1e-3
                            ? runs.count == 1 && abs(runs[0].lo - below) < 1e-3 && abs(runs[0].hi - top) < 1e-3
                            : runs.isEmpty
                        if !agree {
                            if atEdge, (runs.first?.lo ?? top) < below { explained += 1; continue }
                            wrong += 1
                            if example.isEmpty {
                                example = String(format: "; e.g. %@, %@, at (%.3f, %.3f) hidden below %.4f, kept %@", name, kind,
                                                 at.x, at.y, below, kept.map { String(format: "%.4f to %.4f", $0.lo, $0.hi) }.joined(separator: ", "))
                            }
                        }
                    }
                }
            }
            Check.expect(wrong == 0 && partly > 10,
                         "\(layout.name): a stroke on a wall that faces the eye keeps what the walk sees",
                         "\(wrong) disagree, of \(compared) strokes each tested twice; \(explained) at the footprint's edge; \(partly) partly hidden\(example)")
        }
    }
}

// MARK: - curves on a riser are tested by the riser
//
// A riser is a vertical face of the solid, as a wall is: a hatching that
// selects the risers hatches it, and the truncation edge's layer draws its
// bottom edge with its top. Both stand a hair onto the lower side, and a point on them is tested
// by the riser's facing and then by the traversal, forgiven nothing.

/// A perimeter through the corners, in order.
private func riserPerimeter(_ corners: [(Double, Double)]) -> BoundaryPerimeter {
    BoundaryPerimeter(edges: corners.indices.map { k in
        let a = corners[k], b = corners[(k + 1) % corners.count]
        return PerimeterEdge(start: P2(a.0, a.1), end: P2(b.0, b.1), density: 9)
    })
}

func riserFaceTests() {
    Check.suite("risers: a riser's hatching and bottom edge lie on it, from its bottom to its top, where it stands") {
        // 1/(z - 1/2) over Re [-1/2, 3/2], Im [-1, 1], at 2 left of Re z = 1/2
        // and 4 beyond: the riser stands where |f| = 1/|Im z| reaches 2 on the
        // boundary, |Im z| ≤ 1/2, from 2 up to min(1/|Im z|, 4). Hatched at
        // 0.07, the boundary is 29 whole gaps across the domain, as the
        // plateaus divide it, so a line stands at each Im z = -1 + 2k/29 the
        // riser holds, none at its ends. On a footprint with a notch cut
        // from Im z = 0.2 back it stands to 0.2, and on one that ends at the
        // boundary nowhere: the wall there rises to the top. And 1/(z -
        // 0.45) cut at 4, in two tiles side by side, the second translated
        // by 2: at Re z = 3/2 the second's top, 1/|0.95 - i Im z|, stands
        // over the first's, 1/|1.05 + i Im z|, all along the tile boundary,
        // and the riser between them faces the first.
        let window = Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1))
        let tower = try NativeLandscape.build(
            LandscapeRequest(expression: "1/(z - 0.5)", domain: window, resolution: 48,
                             truncation: .realBands([RealBand(below: 0.5, height: 2)], beyond: 4)),
            refine: true)
        let pair = try NativeLandscape.build(
            LandscapeRequest(expression: "1/(z - 0.45)", domain: window, resolution: 48, truncation: .uniform(4)),
            refine: true)
        let box = riserPerimeter([(-0.5, -1), (1.5, -1), (1.5, 1), (-0.5, 1)])
        let notched = riserPerimeter([(-0.5, -1), (1.5, -1), (1.5, 1), (0.7, 1), (0.7, 0.2), (0.3, 0.2), (0.3, 1), (-0.5, 1)])
        let half = riserPerimeter([(-0.5, -1), (0.5, -1), (0.5, 1), (-0.5, 1)])
        let side = Affine2(a: 1, b: 0, tx: 2, c: 0, d: 1, ty: 0)
        func towerTop(_ y: Double) -> Double { min(1 / abs(y), 4) }
        let cases: [(name: String, built: KurvenBundle, walls: BoundaryPerimeter, region: Region, tiles: [Affine2],
                     at: Double, stands: (lo: Double, hi: Double)?, bottom: (Double) -> Double, top: (Double) -> Double)] = [
            ("a tower at 2 | 4", tower, box, .full, [.identity], 0.5, (-0.5, 0.5), { _ in 2 }, towerTop),
            ("a tower at 2 | 4, a notch cut from Im z = 0.2 back", tower, notched, .inside(notched), [.identity], 0.5,
             (-0.5, 0.2), { _ in 2 }, towerTop),
            ("a tower at 2 | 4, the footprint ending at the boundary", tower, half, .inside(half), [.identity], 0.5,
             nil, { _ in 2 }, towerTop),
            ("two tiles side by side meeting with different tops", pair, box, .full, [.identity, side], 1.5, (-1, 1),
             { 1 / hypot(1.05, $0) }, { 1 / hypot(0.95, $0) }),
        ]
        for c in cases {
            let surface = c.built.surface
            guard let refine = c.built.refine else { Check.expect(false, "\(c.name): a refiner"); continue }
            let g = surface.height
            let cell = max(abs(g.domain.real.length) / Double(g.width - 1), abs(g.domain.imag.length) / Double(g.height - 1))
            let tolerance = refine.tolerance * cell
            let tiled = c.tiles.count > 1
            let context = Surface.HatchContext(perimeter: c.walls, region: c.region, tiles: c.tiles)
            let sources: [LayerSource] = [.wallHatching(edges: Array(c.walls.edges.indices), spacing: 0.07),
                                          .riserHatching(spacing: 0.07, tiled: tiled)]
            let hatch = PolylineSet<WorldSpace>(paths: sources.flatMap { source in
                let lines = surface.hatch(source, in: context, refine: refine)!
                return (0..<lines.count).map { Array(lines[path: $0]) }
            })
            let edges = surface.hatch(.truncationEdge(tiled: tiled), in: context, refine: refine)!
            /// A billionth of a cell onto the lower side of the riser's line,
            /// Re z = `at`, which is to its left in all four.
            func onLowerSide(_ v: P3<WorldSpace>) -> Bool { v.x < c.at && c.at - v.x < 1e-9 * cell * 2 }
            /// Exactly on a wall: on the line of a perimeter edge, all of
            /// whose edges run along an axis.
            func onWall(_ v: P3<WorldSpace>) -> Bool {
                c.walls.edges.contains { e in
                    e.start.x == e.end.x ? v.x == e.start.x && v.y >= min(e.start.y, e.end.y) && v.y <= max(e.start.y, e.end.y)
                                         : v.y == e.start.y && v.x >= min(e.start.x, e.end.x) && v.x <= max(e.start.x, e.end.x)
                }
            }
            let strokes = (0..<hatch.count).map { Array(hatch[path: $0]) }
            let risers = strokes.filter { $0.allSatisfy(onLowerSide) }
            let strays = strokes.filter { p in !p.allSatisfy(onWall) && !p.allSatisfy(onLowerSide) }
            let n = 30
            let gaps: [Double] = (1..<(n - 1)).map { k -> Double in -1 + 2 * Double(k) / Double(n - 1) }
            let wanted: [Double] = c.stands.map { s in gaps.filter { $0 > s.lo && $0 < s.hi } } ?? []
            var off: [String] = []
            for p in risers where !(p.count == 2 && p[0].y == p[1].y
                                     && abs(p[0].z - c.bottom(p[0].y)) <= tolerance && abs(p[1].z - c.top(p[0].y)) <= tolerance) {
                off.append(String(format: "Im z = %.4f from %.6f to %.6f", p[0].y, p[0].z, p[p.count - 1].z))
            }
            let at = risers.map { $0[0].y }.sorted()
            let placed = at.count == wanted.count && zip(at, wanted).allSatisfy { abs($0 - $1) <= 1e-12 }
            Check.expect(placed && off.isEmpty && strays.isEmpty,
                         "\(c.name): a hatch line at each whole gap the riser holds, from its bottom to its top, a hair onto the lower side",
                         "\(risers.count) lines, \(wanted.count) wanted, \(off.count) not from bottom to top, \(strays.count) on neither a wall nor the riser"
                            + (off.isEmpty ? "" : ": " + off.prefix(3).joined(separator: ", ")))
            let bottoms = (0..<edges.count).map { Array(edges[path: $0]) }.filter { $0.allSatisfy(onLowerSide) }
            let bottomOff = bottoms.joined().map { abs($0.z - c.bottom($0.y)) }.max() ?? 0
            let ends = bottoms.map { ($0[0].y, $0[$0.count - 1].y) }
            let spans = c.stands.map { s in ends.count == 1 && abs(ends[0].0 - s.lo) <= 1e-9 && abs(ends[0].1 - s.hi) <= 1e-9 }
                ?? ends.isEmpty
            Check.expect(spans && bottomOff <= tolerance,
                         "\(c.name): its bottom edge runs where it stands, at the lower top, a hair onto the lower side",
                         "\(bottoms.count) bottoms, from \(ends.map { String(format: "%.6f to %.6f", $0.0, $0.1) }), "
                            + String(format: "off the lower top by %.2e", bottomOff))
        }
    }

    Check.suite("risers: only a layer that selects the risers hatches them, in place or on every tile") {
        // The tower at 2 | 4 as a bundle draws it: a wall hatching on all
        // four walls, the risers' hatching in place, and the truncation edge
        // in place. A riser's line is counted over every hatching layer by
        // where it stands: in place, a hair left of Re z = 1/2; on a second
        // tile beside the first, of 5/2; at the tile boundary between, of 3/2.
        let window = Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1))
        let built = try NativeLandscape.build(
            LandscapeRequest(expression: "1/(z - 0.45)", domain: window, resolution: 48,
                             truncation: .realBands([RealBand(below: 0.5, height: 2)], beyond: 4)),
            refine: true)
        let g = built.surface.height
        let cell = max(window.real.length / Double(g.width - 1), window.imag.length / Double(g.height - 1))
        /// The hatching's lines on each riser, under `manifest`.
        func risers(_ manifest: Manifest) throws -> (inPlace: Int, second: Int, between: Int) {
            let bundle = KurvenBundle(url: built.url, manifest: manifest, surface: built.surface, refine: built.refine)
            let feet = bundle.layers.filter { if case .hatching = $0.spec.source { true } else { false } }.flatMap { layer in
                (0..<layer.paths.count).map { layer.paths[path: $0].first! }
            }
            func on(_ x: Double) -> Int { feet.filter { x - $0.x > 0.5e-9 * cell && x - $0.x < 2e-9 * cell }.count }
            return (on(0.5), on(2.5), on(1.5))
        }
        func with(_ edit: (inout Manifest) -> Void) -> Manifest {
            var m = built.manifest
            edit(&m)
            return m
        }
        func walls(_ edges: [Int]) -> (inout Manifest) -> Void {
            { m in
                for k in m.layers.indices {
                    if case .hatching(let hatching, .walls(_, let endLines, let base, let top)) = m.layers[k].source {
                        m.layers[k].source = .hatching(hatching, faces: .walls(edges: edges, endLines: endLines, base: base, topOffset: top))
                    }
                }
            }
        }
        func risersTiled(_ tiled: Bool?) -> (inout Manifest) -> Void {
            { m in
                m.layers = m.layers.compactMap { spec in
                    guard case .hatching(let hatching, .risers(_, let endLines)) = spec.source else { return spec }
                    guard let tiled else { return nil }
                    var spec = spec
                    spec.source = .hatching(hatching, faces: .risers(tiled: tiled, endLines: endLines))
                    return spec
                }
            }
        }
        func beside(_ m: inout Manifest) {
            m.occluder.tiles = [.identity, Affine2(a: 1, b: 0, tx: 2, c: 0, d: 1, ty: 0)]
        }
        let all = try risers(built.manifest)
        Check.expect(all.inPlace > 30 && all.second == 0 && all.between == 0,
                     "every wall and the risers hatched, in place: the riser, in place", "\(all)")
        let none = try risers(with(risersTiled(nil)))
        Check.expect(none.inPlace == 0 && none.second == 0 && none.between == 0,
                     "every wall hatched and no layer on the risers: no riser", "\(none)")
        for edges in [[0, 2], [1], []] {
            let some = try risers(with(walls(edges)))
            Check.expect(some.inPlace == all.inPlace, "walls \(edges) hatched: the same riser lines", "\(some)")
        }
        let edgeless = try risers(with { m in m.layers.removeAll { if case .truncationEdge = $0.source { true } else { false } } })
        Check.expect(edgeless.inPlace == all.inPlace, "no truncation edge drawn: the same riser lines", "\(edgeless)")
        let untiled = try risers(with { m in beside(&m) })
        Check.expect(untiled.inPlace == all.inPlace && untiled.second == 0 && untiled.between == 0,
                     "two tiles, the risers hatched in place: the riser in place, none on the second tile or between them",
                     "\(untiled)")
        let tiled = try risers(with { m in beside(&m); risersTiled(true)(&m) })
        Check.expect(tiled.inPlace == all.inPlace && tiled.second == all.inPlace && tiled.between > 30,
                     "two tiles, the risers hatched on every tile: the riser on both, and the one between them", "\(tiled)")
    }

    // Where a riser faces away, the solid on its higher side stands between
    // its hatching and bottom edge and the eye, a hair along the line of
    // sight; where it faces the eye, what stands in front hides them from the
    // bottom up. What the solid hides of a vertical line at a point is
    // everything below the greatest of h(q(t)) - t d_z along the line of
    // sight, q(t) its footprint and h the solid's top there: min(|f|, h(x))
    // on the copy under it, the greater where two meet. A point is seen when
    // that is not above it. The walk reads that a five-hundredth of a unit at
    // a time, then a hundred-thousandth around each greatest within reach of
    // the best, as the walls' walk does (`wallFaceTests`). A hatch line,
    // tested from its two ends and at 256 vertices, keeps one run from there
    // to its top, or none; each vertex of a bottom edge is kept where the walk
    // sees it and dropped where it is under the solid by a thousandth, under
    // every camera of the stress orbit. A riser seen edge-on is left out, as
    // a wall is: its line of sight runs along it, a hair from its plane. On a
    // tower, and mirrored, on Γ with three risers, and on the tower in two tiles, the
    // second beside the first and then turned, both copies' risers and the
    // one at the tile boundary between; and on the grid, on random terrain in
    // abutting copies, where the traversal's skin is the drawing's depth
    // tolerance and a riser can be lower than it.
    Check.suite("risers: a riser's hatching and bottom edge are hidden up to where the walk up the line of sight meets the solid") {
        let window = Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1))
        let tower = LandscapeRequest(expression: "1/(z - 0.45)", domain: window, resolution: 100,
                                     truncation: .realBands([RealBand(below: 0.5, height: 2)], beyond: 4))
        // The tower mirrored, the higher band on the left: the traversal's
        // sample on a band boundary reads the band beyond it, here the
        // lower, and where |f| falls under a bottom's height within a column
        // of the boundary, no sample finds the higher side's solid in front
        // of the bottom; the riser's facing hides it.
        let mirrored = LandscapeRequest(expression: "1/(z - 0.55)", domain: window, resolution: 100,
                                        truncation: .realBands([RealBand(below: 0.5, height: 4)], beyond: 2))
        let gamma = LandscapeRequest(expression: "gamma(z)", domain: Domain(real: Interval(lo: -4.5, hi: 4.5), imag: Interval(lo: -2.5, hi: 2.5)),
                                     resolution: 180,
                                     truncation: .realBands([RealBand(below: -2.5, height: 0.5), RealBand(below: -1.5, height: 1),
                                                             RealBand(below: 0.5, height: 1.5)], beyond: 2.5))
        let side = Affine2(a: 1, b: 0, tx: 2, c: 0, d: 1, ty: 0), turned = Affine2(a: 0, b: -1, tx: 2.5, c: 1, d: 0, ty: -0.5)
        typealias Line = (o: SIMD2<Double>, n: SIMD2<Double>)
        var landscapes: [(name: String, surface: Surface, refine: ContourRefine?, tiles: [Affine2], tileBoundaries: [Line],
                          spacing: Double)] = []
        let onF: [(String, LandscapeRequest, [Affine2], [Line], Double)] = [
            ("a tower at 2 | 4", tower, [.identity], [], 0.025),
            ("a tower at 4 | 2, the higher band on the left", mirrored, [.identity], [], 0.025),
            ("Γ at 0.5 | 1 | 1.5 | 2.5", gamma, [.identity], [], 0.1),
            ("a tower at 2 | 4 in two tiles side by side", tower, [.identity, side], [(SIMD2(1.5, 0), SIMD2(1, 0))], 0.05),
            ("a tower at 2 | 4 in two tiles, the second turned", tower, [.identity, turned], [(SIMD2(1.5, 0), SIMD2(1, 0))], 0.05),
        ]
        for (name, request, tiles, lines, spacing) in onF {
            let built = try NativeLandscape.build(request, refine: true)
            landscapes.append((name, built.surface, built.refine, tiles, lines, spacing))
        }
        // On the grid the traversal forgives a point on the surface the
        // drawing's depth tolerance, a skin a riser's own curves are not
        // forgiven: random terrain on a tile and the next copies to its right
        // and above it, as the walls' walk has it, with risers all along the
        // two tile boundaries, some lower than the skin.
        for seed: UInt64 in [3, 17] {
            let terrain = randomHeightfield(cells: 8, seed: seed, truncation: Truncation.none).surface
            let d = terrain.domain
            landscapes.append(("seed \(seed), abutting copies, on the grid", terrain, nil,
                               [.identity, Affine2(a: 1, b: 0, tx: d.real.length, c: 0, d: 1, ty: 0),
                                Affine2(a: 1, b: 0, tx: 0, c: 0, d: 1, ty: d.imag.length)],
                               [(SIMD2(d.real.hi, 0), SIMD2(1, 0)), (SIMD2(0, d.imag.hi), SIMD2(0, 1))], 0.1))
        }
        var partly = 0
        for l in landscapes {
            let refine = l.refine, magnitude = l.refine?.magnitude
            let surface = l.surface, dom = surface.domain, g = surface.height
            let (lo, hi) = (min(dom.real.lo, dom.real.hi), max(dom.real.lo, dom.real.hi))
            let (ylo, yhi) = (min(dom.imag.lo, dom.imag.hi), max(dom.imag.lo, dom.imag.hi))
            let box = riserPerimeter([(lo, ylo), (hi, ylo), (hi, yhi), (lo, yhi)])
            let h = Heightfield(surface: surface, occluder: .empty, tiles: l.tiles, region: .full, step: 1,
                                refine: refine, walls: .perimeter(box, base: 0))
            let context = Surface.HatchContext(perimeter: box, region: .full, tiles: l.tiles)
            let cell = max(abs(dom.real.length) / Double(g.width - 1), abs(dom.imag.length) / Double(g.height - 1))
            // The lines the risers stand on, in place: the band boundaries
            // the truncation steps at inside the domain, on every copy, and
            // the tile boundaries.
            var lines: [Line] = l.tileBoundaries
            if case .realBands(let bands, _) = surface.truncation {
                for b in bands.map(\.below) where b > lo && b < hi {
                    // Square to the copy's boundary: the gradient of the
                    // domain's x, carried back.
                    for tile in l.tiles {
                        if let back = tile.inverse { lines.append((tile(P3(b, 0, 0)).xy.v, simd_normalize(SIMD2(back.a, back.b)))) }
                    }
                }
            }
            /// The riser a point stands a hair onto a side of, a billionth
            /// of a cell, as its curves stand, not a rounding: its line's
            /// normal.
            func riser(_ v: P3<WorldSpace>) -> SIMD2<Double>? {
                lines.first { line in
                    let d = abs(simd_dot(SIMD2(v.x, v.y) - line.o, line.n))
                    return d > 0.5e-9 * cell && d < 2e-9 * cell
                }?.n
            }
            func onRiser(_ v: P3<WorldSpace>) -> Bool { riser(v) != nil }
            let hatch = surface.hatch(.riserHatching(spacing: l.spacing, tiled: true), in: context, refine: refine)!
            let edges = surface.hatch(.truncationEdge(tiled: true), in: context, refine: refine)!
            let strokes = (0..<hatch.count).map { Array(hatch[path: $0]) }.filter { $0.allSatisfy(onRiser) }
            let bottoms = (0..<edges.count).map { Array(edges[path: $0]) }.filter { $0.allSatisfy(onRiser) }
            let ends = PolylineSet(paths: strokes)
            let dense = PolylineSet(paths: strokes.map { p in
                (0...256).map { j in j == 256 ? p[1] : P3<WorldSpace>(p[0].v + (p[1].v - p[0].v) * (Double(j) / 256)) }
            })
            let bottomSet = PolylineSet(paths: bottoms)
            /// The solid's top over a point of the plane: on the copy under
            /// it, the greater where two meet; nil off every copy.
            func top(_ q: SIMD2<Double>) -> Double? {
                var best: Double?
                for tile in l.tiles {
                    guard let back = tile.inverse else { continue }
                    let p = back(P3<WorldSpace>(q.x, q.y, 0))
                    guard p.x >= lo, p.x <= hi, p.y >= ylo, p.y <= yhi else { continue }
                    // On the grid, the drawn triangles, through the tile
                    // under the point.
                    guard let magnitude else { return h.drawnHeight(at: P2(q.x, q.y)) }
                    let u = magnitude(P2(p.x, p.y))
                    best = max(best ?? -.infinity, min(u.isNaN ? .infinity : u, surface.truncationHeight(atX: p.x)))
                }
                return best
            }
            // The steepest the truncated top rises between two samples in
            // one band, and the highest it stands.
            var highest = 0.0, steepest = 0.0
            let dx = dom.real.length / Double(g.width - 1), dy = dom.imag.length / Double(g.height - 1)
            func truncated(_ i: Int, _ j: Int) -> Double {
                min(Double(g[i, j]), surface.truncationHeight(atX: g.position(x: i, y: 0).x))
            }
            for j in 0..<g.height {
                for i in 0..<g.width {
                    let z = truncated(i, j), x = g.position(x: i, y: 0).x
                    guard z.isFinite else { continue }
                    highest = max(highest, z)
                    if i + 1 < g.width, surface.truncationHeight(atX: g.position(x: i + 1, y: 0).x) == surface.truncationHeight(atX: x) {
                        steepest = max(steepest, abs(truncated(i + 1, j) - z) / abs(dx))
                    }
                    if j + 1 < g.height { steepest = max(steepest, abs(truncated(i, j + 1) - z) / abs(dy)) }
                }
            }
            let extent = Double(l.tiles.count) * max(abs(dom.real.length), abs(dom.imag.length)) * 2
            var strokesCompared = 0, whole = 0, kept = 0, wrong = 0
            var vertices = 0, seen = 0, hidden = 0, wrongVertices = 0
            var example = ""
            for (name, view) in stressCameras() {
                let d = -view.projectionDirection, dxy = SIMD2(d.x, d.y)
                /// The height below which the solid hides a vertical line at
                /// `at`, or `base`.
                func hiddenBelow(_ at: SIMD2<Double>, base: Double) -> Double {
                    let coarse = 2e-3, fine = 1e-5
                    let reach = 2 * (2 * 2.0.squareRoot() * steepest * simd_length(dxy) + abs(d.z)) * coarse
                    func value(_ t: Double) -> Double? { top(at + t * dxy).map { $0 - t * d.z } }
                    var samples: [(t: Double, value: Double)] = []
                    var best = base, t = fine
                    while highest - t * d.z > best - reach, t * simd_length(dxy) < extent {
                        if let v = value(t) { samples.append((t, v)); best = max(best, v) }
                        t += coarse
                    }
                    // Around each greatest of the samples within reach of the
                    // best, finely: a riser's step lies between two samples.
                    for (k, s) in samples.enumerated() where s.value > best - reach
                        && (k == 0 || s.value >= samples[k - 1].value) && (k + 1 == samples.count || s.value >= samples[k + 1].value) {
                        var u = max(s.t - coarse, fine)
                        while u < s.t + coarse {
                            if let v = value(u), v > best { best = v }
                            u += fine
                        }
                    }
                    return best
                }
                func runs(_ set: PolylineSet<WorldSpace>) -> [SIMD2<Double>: [(lo: Double, hi: Double)]] {
                    var out: [SIMD2<Double>: [(lo: Double, hi: Double)]] = [:]
                    for i in 0..<set.count {
                        let p = Array(set[path: i])
                        out[SIMD2(p[0].x, p[0].y), default: []].append((p[0].z, p[p.count - 1].z))
                    }
                    return out
                }
                // As the risers' hatching layer and the truncation edge's
                // test them (`SceneGeometry.face(of:)`).
                let two = runs(h.visibleCurves(ends, on: .riser, view: view, depthTolerance: 0.02))
                let many = runs(h.visibleCurves(dense, on: .riser, view: view, depthTolerance: 0.02))
                func edgeOn(_ v: P3<WorldSpace>) -> Bool { riser(v).map { abs(simd_dot($0, dxy)) <= 1e-6 } ?? true }
                for p in strokes where !edgeOn(p[0]) {
                    let at = SIMD2(p[0].x, p[0].y), base = p[0].z, top = p[1].z
                    let below = hiddenBelow(at, base: base)
                    strokesCompared += 1
                    if below > base + 1e-3, below < top - 1e-3 { partly += 1 }
                    if below >= top - 1e-3 { whole += 1 } else { kept += 1 }
                    for (kind, got) in [("two ends", two[at] ?? []), ("256 vertices", many[at] ?? [])] {
                        let runs = got.filter { $0.lo < top - 1e-3 }
                        let agree = below < top - 1e-3
                            ? runs.count == 1 && abs(runs[0].lo - below) < 1e-3 && abs(runs[0].hi - top) < 1e-3
                            : runs.isEmpty
                        if !agree {
                            wrong += 1
                            if example.isEmpty {
                                example = String(format: "; e.g. %@, %@, a line at (%.4f, %.4f) hidden below %.4f, kept %@", name, kind,
                                                 at.x, at.y, below, got.map { String(format: "%.4f to %.4f", $0.lo, $0.hi) }.joined(separator: ", "))
                            }
                        }
                    }
                }
                let keptBottoms = Set(h.visibleCurves(bottomSet, on: nil, view: view, depthTolerance: 0.02).vertices.map(\.v))
                for v in bottoms.joined() where !edgeOn(v) {
                    // Where the riser has no height its bottom is its top,
                    // and a vertex there seen alone is no run to draw; nor
                    // where it ends on the window's edge, from which a line
                    // of sight leaves the solid at once, and its neighbours'
                    // do not.
                    let across = riser(v).map { n in [-4.0, 4.0].compactMap { top(SIMD2(v.x, v.y) + $0 * 1e-9 * cell * n) }.max() ?? v.z } ?? v.z
                    let onEdge = l.tiles.compactMap(\.inverse).contains { back in
                        let q = back(v)
                        return abs(q.y - ylo) <= 1e-12 || abs(q.y - yhi) <= 1e-12 || abs(q.x - lo) <= 1e-12 || abs(q.x - hi) <= 1e-12
                    }
                    guard across > v.z + 1e-3, !onEdge else { continue }
                    let over = hiddenBelow(SIMD2(v.x, v.y), base: v.z) - v.z
                    vertices += 1
                    if over > 1e-3 {
                        hidden += 1
                        if keptBottoms.contains(v.v) {
                            wrongVertices += 1
                            if example.isEmpty { example = String(format: "; e.g. %@, a bottom's vertex at (%.4f, %.4f, %.4f) under the solid by %.4f, kept", name, v.x, v.y, v.z, over) }
                        }
                    } else if over <= 1e-7 {
                        seen += 1
                        if !keptBottoms.contains(v.v) {
                            wrongVertices += 1
                            if example.isEmpty { example = String(format: "; e.g. %@, a bottom's vertex at (%.4f, %.4f, %.4f) in view, dropped", name, v.x, v.y, v.z) }
                        }
                    }
                }
            }
            Check.expect(wrong == 0 && wrongVertices == 0 && whole > 100 && kept > 100 && seen > 100 && hidden > 100,
                         "\(l.name): a riser's hatching and bottom edge keep what the walk up the line of sight sees",
                         "\(wrong) of \(strokesCompared) lines disagree, each tested twice (\(whole) hidden whole, \(kept) kept); "
                            + "\(wrongVertices) of \(vertices) bottom vertices (\(seen) seen, \(hidden) hidden)\(example)")
        }
        Check.expect(partly > 10, "and some lines are hidden in part, from the bottom up", "\(partly) partly hidden")
    }

    // A line of sight that crosses a riser's line between the two tops
    // there, over the lower and under the higher, passes through the solid,
    // however soon it comes out on the higher side. The walk up the line of
    // sight (`SolidWalk`) reads the greatest of h(q(t)) - z(t), as above, and
    // the solid closed at each riser's line it crosses: at the higher top.
    // Under every camera of the stress orbit, every vertex of the plateau
    // hatching, the contours and the wall edges that is kept, and whose line
    // of sight crosses a riser's line below the highest top, is under the
    // solid by no more than the depth tolerance; and none dropped is in
    // plain view of the walk, under the solid by no more than a millionth,
    // on a face turned to the eye: a plateau seen from above, the graph
    // facing the eye by a margin, a wall's top edge. On the dip, two
    // poles at 0.49 ± 0.1i whose truncation steps from 0.8 to 4 at Re z =
    // 1/2, where a line of sight passes into the higher band a hair beyond
    // the boundary and under its top, between two grid columns; on the tower
    // in two tiles, the second beside the first and then turned; and on Γ
    // with three risers.
    Check.suite("risers: no stroke is drawn through a riser, and none is hidden that the walk up the line of sight sees") {
        let window = Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1))
        let dip = LandscapeRequest(expression: "1/(z - 0.49 - 0.1*i) + 1/(z - 0.49 + 0.1*i)",
                                   domain: Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1.3)), resolution: 60,
                                   truncation: .realBands([RealBand(below: 0.5, height: 0.8)], beyond: 4))
        let tower = LandscapeRequest(expression: "1/(z - 0.45)", domain: window, resolution: 60,
                                     truncation: .realBands([RealBand(below: 0.5, height: 2)], beyond: 4))
        let gamma = LandscapeRequest(expression: "gamma(z)", domain: Domain(real: Interval(lo: -4.5, hi: 4.5), imag: Interval(lo: -2.5, hi: 2.5)),
                                     resolution: 120,
                                     truncation: .realBands([RealBand(below: -2.5, height: 0.5), RealBand(below: -1.5, height: 1),
                                                             RealBand(below: 0.5, height: 1.5)], beyond: 2.5))
        let side = Affine2(a: 1, b: 0, tx: 2, c: 0, d: 1, ty: 0), turned = Affine2(a: 0, b: -1, tx: 2.5, c: 1, d: 0, ty: -0.5)
        let cases: [(name: String, request: LandscapeRequest, tiles: [Affine2])] = [
            ("the dip at 0.8 | 4", dip, [.identity]),
            ("a tower at 2 | 4 in two tiles side by side", tower, [.identity, side]),
            ("a tower at 2 | 4 in two tiles, the second turned", tower, [.identity, turned]),
            ("Γ at 0.5 | 1 | 1.5 | 2.5", gamma, [.identity]),
        ]
        let depthTolerance = 0.02
        for c in cases {
            let built = try NativeLandscape.build(c.request, refine: true)
            guard let magnitude = built.refine?.magnitude else { Check.expect(false, "\(c.name): a refiner"); continue }
            // On two tiles the walls stand round both copies, and every layer
            // is drawn on both.
            var manifest = built.manifest
            manifest.occluder.tiles = c.tiles
            if c.tiles.count > 1 { manifest.occluder.walls = .perimeter(riserPerimeter([(-0.5, -1), (3.5, -1), (3.5, 1), (-0.5, 1)]), base: 0) }
            for k in manifest.layers.indices where c.tiles.count > 1 {
                switch manifest.layers[k].source {
                case .contour(let field, let levels, let restriction, _):
                    manifest.layers[k].source = .contour(field: field, levels: levels, restriction: restriction, faces: .graph(tiled: true))
                case .hatching(let hatching, .plateaus): manifest.layers[k].source = .hatching(hatching, faces: .plateaus(tiled: true))
                case .hatching(let hatching, .risers(_, let endLines)):
                    manifest.layers[k].source = .hatching(hatching, faces: .risers(tiled: true, endLines: endLines))
                case .truncationEdge: manifest.layers[k].source = .truncationEdge(tiled: true)
                default: break
                }
            }
            let bundle = KurvenBundle(url: built.url, manifest: manifest, surface: built.surface, refine: built.refine)
            let h = Heightfield(surface: bundle.surface, occluder: bundle.occluder(), tiles: c.tiles,
                                region: manifest.occluder.region, step: manifest.occluder.step, refine: bundle.refine,
                                walls: manifest.occluder.walls)
            let walk = SolidWalk(surface: bundle.surface, magnitude: magnitude, tiles: c.tiles)
            let layers: [(layer: Layer, face: FaceKind)] = try [("plateau_hatch", FaceKind.plateau), ("wall_edges", .wall)].map {
                (try bundle.layer($0.0), $0.1)
            } + bundle.layers.filter { $0.spec.role == .magnitude || $0.spec.role == .phase }.map { ($0, .graph) }
            var kept = 0, under = 0, dropped = 0, seen = 0
            var worst = 0.0, underExample = "", seenExample = ""
            for (name, view) in stressCameras() {
                let d = -view.projectionDirection
                for (layer, face) in layers {
                    let visible = h.visibleCurves(layer.paths, on: face, view: view, depthTolerance: depthTolerance)
                    // The curves' own vertices, not the ends of runs bisected
                    // to where the verdict turns: a line through a riser's
                    // corner at the domain's edge is in the solid on one side
                    // of it and clear on the other.
                    let keptSet = Set(visible.vertices.map(\.v))
                    let keptVertices = Array(keptSet.intersection(layer.paths.vertices.map(\.v))).map { P3<WorldSpace>($0) }
                    let depths = Heightfield.parallelMap(keptVertices, chunk: 64) { p -> Double? in
                        // A wall seen edge-on is left out: its line of sight
                        // runs along it, in and out of the solid by rounding.
                        let ts = walk.crossings(p, d).filter { $0 >= 1e-7 }
                        return ts.isEmpty || walk.edgeOn(p, d) ? nil : walk.depth(p, d, crossings: ts)
                    }
                    for (p, depth) in zip(keptVertices, depths) {
                        guard let depth else { continue }
                        kept += 1
                        if depth > worst { worst = depth }
                        if depth > depthTolerance {
                            under += 1
                            if underExample.isEmpty {
                                underExample = String(format: "; e.g. %@, %@, (%.4f, %.4f, %.4f) under the solid by %.4f, kept",
                                                      name, layer.spec.name, p.x, p.y, p.z, depth)
                            }
                        }
                    }
                    // What is dropped and faces the eye: a plateau seen from
                    // above, the graph facing the eye by a margin, off the
                    // truncation edge, and a wall's top edge off the domain's
                    // corners. Not a vertex on or a hair from a riser's line,
                    // which the traversal and the walk each read from their
                    // own side.
                    let candidates = Array(Set(layer.paths.vertices.map(\.v)).subtracting(keptSet)).map { P3<WorldSpace>($0) }
                    let verdicts = Heightfield.parallelMap(candidates, chunk: 64) { p -> Double? in
                        let all = walk.crossings(p, d)
                        guard !all.isEmpty, !walk.besideRiser(p) else { return nil }
                        switch face {
                        case .plateau: guard d.z > 1e-3 else { return nil }
                        case .graph: guard walk.graphFacing(p, d).map({ $0 > 0.05 }) ?? false else { return nil }
                        default:
                            guard let top = walk.top(SIMD2(p.x, p.y)), abs(p.z - top) <= 1e-9 * max(1, abs(top)),
                                  !walk.atCorner(p) else { return nil }
                        }
                        return walk.depth(p, d, crossings: all)
                    }
                    for (p, depth) in zip(candidates, verdicts) {
                        guard let depth else { continue }
                        dropped += 1
                        if depth <= 1e-6 {
                            seen += 1
                            if seenExample.isEmpty {
                                seenExample = String(format: "; e.g. %@, %@, (%.4f, %.4f, %.4f) in plain view, dropped", name, layer.spec.name, p.x, p.y, p.z)
                            }
                        }
                    }
                }
            }
            Check.expect(under == 0 && seen == 0 && kept > 1000 && dropped > 100,
                         "\(c.name): what is drawn and what is dropped where a line of sight crosses a riser agree with the walk",
                         String(format: "%d of %d kept vertices under the solid by more than the depth tolerance, worst %.4f%@; ",
                                under, kept, worst, underExample)
                            + "\(seen) of \(dropped) dropped on a face turned to the eye in plain view\(seenExample)")
        }
    }
}

/// The walk up a line of sight over a landscape on its tiles, for the riser
/// suites: the solid's top, closed, h(q) = min(|f|, h(x)) on the copy under q,
/// the greater where two meet and the greater band's height on a band
/// boundary, and how far under it a point lies along its line of sight.
private struct SolidWalk: Sendable {
    let surface: Surface
    let magnitude: ContourRefine.Magnitude
    let backs: [Affine2]
    let x: (lo: Double, hi: Double), y: (lo: Double, hi: Double)
    /// The band boundaries the truncation steps at, each with the greater
    /// height either side.
    let steps: [(b: Double, greater: Double)]
    /// The copies' bounding box in the plane.
    let box: (lo: SIMD2<Double>, hi: SIMD2<Double>)
    /// The highest the truncated top stands, and the steepest it rises
    /// between two samples in one band.
    let highest: Double, steepest: Double

    init(surface: Surface, magnitude: @escaping ContourRefine.Magnitude, tiles: [Affine2]) {
        self.surface = surface
        self.magnitude = magnitude
        backs = tiles.compactMap(\.inverse)
        let dom = surface.domain, g = surface.height
        x = (min(dom.real.lo, dom.real.hi), max(dom.real.lo, dom.real.hi))
        y = (min(dom.imag.lo, dom.imag.hi), max(dom.imag.lo, dom.imag.hi))
        let spans = surface.truncation.bands(from: x.lo, to: x.hi)
        steps = zip(spans, spans.dropFirst()).compactMap { l, r in l.height != r.height ? (l.hi, max(l.height, r.height)) : nil }
        var lo = SIMD2(Double.infinity, .infinity), hi = -lo
        for tile in tiles {
            for (cx, cy) in [(x.lo, y.lo), (x.hi, y.lo), (x.lo, y.hi), (x.hi, y.hi)] {
                let w = tile(P3<WorldSpace>(cx, cy, 0)).xy.v
                lo = simd_min(lo, w); hi = simd_max(hi, w)
            }
        }
        box = (lo, hi)
        var highest = 0.0, steepest = 0.0
        let dx = dom.real.length / Double(g.width - 1), dy = dom.imag.length / Double(g.height - 1)
        func truncated(_ i: Int, _ j: Int) -> Double {
            min(Double(g[i, j]), surface.truncationHeight(atX: g.position(x: i, y: 0).x))
        }
        for j in 0..<g.height {
            for i in 0..<g.width {
                let z = truncated(i, j), xi = g.position(x: i, y: 0).x
                guard z.isFinite else { continue }
                highest = max(highest, z)
                if i + 1 < g.width, surface.truncationHeight(atX: g.position(x: i + 1, y: 0).x) == surface.truncationHeight(atX: xi) {
                    steepest = max(steepest, abs(truncated(i + 1, j) - z) / abs(dx))
                }
                if j + 1 < g.height { steepest = max(steepest, abs(truncated(i, j + 1) - z) / abs(dy)) }
            }
        }
        self.highest = highest; self.steepest = steepest
    }

    /// Whether a point of the domain lies in it, to a rounding.
    func inDomain(_ p: P3<WorldSpace>, _ e: Double = 1e-12) -> Bool {
        p.x >= x.lo - e && p.x <= x.hi + e && p.y >= y.lo - e && p.y <= y.hi + e
    }

    /// The solid's top over a point of the plane, closed; nil off every copy.
    func top(_ q: SIMD2<Double>) -> Double? {
        var best: Double?
        for back in backs {
            let p = back(P3<WorldSpace>(q.x, q.y, 0))
            guard inDomain(p) else { continue }
            let px = min(max(p.x, x.lo), x.hi), py = min(max(p.y, y.lo), y.hi)
            let u = magnitude(P2(px, py))
            let h = steps.first { abs($0.b - px) <= 1e-12 }?.greater ?? surface.truncationHeight(atX: px)
            best = max(best ?? -.infinity, min(u.isNaN ? .infinity : u, h))
        }
        return best
    }

    /// How squarely the graph faces along `d` at a point of it, through the
    /// copy it lies on: the normal by differences either side, the lesser of
    /// the two, so that a crease faces neither way; nil off every copy or on
    /// a plateau.
    func graphFacing(_ p: P3<WorldSpace>, _ d: SIMD3<Double>) -> Double? {
        guard let back = backs.first(where: { inDomain($0(p), 1e-9) }) else { return nil }
        let e = 1e-6
        func height(_ q: SIMD2<Double>) -> Double? {
            let w = back(P3<WorldSpace>(q.x, q.y, 0))
            guard inDomain(w, 1e-9) else { return nil }
            let u = magnitude(P2(w.x, w.y))
            return u < surface.truncationHeight(atX: w.x) - 1e-6 ? u : nil
        }
        let at = SIMD2(p.x, p.y)
        guard let u = height(at) else { return nil }
        var least = Double.infinity
        for s in [-1.0, 1.0] {
            guard let ux = height(at + SIMD2(s * e, 0)), let uy = height(at + SIMD2(0, s * e)) else { return nil }
            let n = SIMD3(-(ux - u) / (s * e), -(uy - u) / (s * e), 1)
            least = min(least, simd_dot(n, d) / simd_length(n))
        }
        return least
    }

    /// Whether `p` lies on the domain's edge on a copy, a wall, and its line
    /// of sight runs along it.
    func edgeOn(_ p: P3<WorldSpace>, _ d: SIMD3<Double>) -> Bool {
        backs.contains { back in
            let o = back(p), r = SIMD2(back.a * d.x + back.b * d.y, back.c * d.x + back.d * d.y)
            let e = 1e-9
            return inDomain(o, e) && ((abs(o.x - x.lo) <= e || abs(o.x - x.hi) <= e) && abs(r.x) <= 1e-6
                                      || (abs(o.y - y.lo) <= e || abs(o.y - y.hi) <= e) && abs(r.y) <= 1e-6)
        }
    }

    /// Whether `p` stands at a corner of the domain on a copy, where its top
    /// edge meets a vertical edge and the walls' facing has the say.
    func atCorner(_ p: P3<WorldSpace>) -> Bool {
        backs.contains { back in
            let o = back(p), e = 1e-9
            return (abs(o.x - x.lo) <= e || abs(o.x - x.hi) <= e) && (abs(o.y - y.lo) <= e || abs(o.y - y.hi) <= e)
        }
    }

    /// Whether `p` lies on or a hair from a riser's line, a millionth: a
    /// band boundary on a copy, or a copy's edge where another meets it. The
    /// traversal and the walk each read such a point from their own side.
    func besideRiser(_ p: P3<WorldSpace>) -> Bool {
        let e = 1e-6
        let on = backs.filter { inDomain($0(p), e) }
        return on.contains { back in
            let o = back(p)
            return steps.contains { abs(o.x - $0.b) <= e }
                || on.count > 1 && (abs(o.x - x.lo) <= e || abs(o.x - x.hi) <= e || abs(o.y - y.lo) <= e || abs(o.y - y.hi) <= e)
        }
    }

    /// Where the line of sight from `p` along `d` crosses a riser's line
    /// below the highest top: a band boundary on a copy, or the edge of a
    /// copy where another meets it.
    func crossings(_ p: P3<WorldSpace>, _ d: SIMD3<Double>) -> [Double] {
        var out: [Double] = []
        for back in backs {
            let o = back(P3<WorldSpace>(p.x, p.y, 0))
            let r = SIMD2(back.a * d.x + back.b * d.y, back.c * d.x + back.d * d.y)
            var lines: [(axis: Int, level: Double, edge: Bool)] = steps.map { (0, $0.b, false) }
            lines += [(0, x.lo, true), (0, x.hi, true), (1, y.lo, true), (1, y.hi, true)]
            for line in lines where abs(r[line.axis]) > 1e-15 {
                let t = (line.level - (line.axis == 0 ? o.x : o.y)) / r[line.axis]
                guard t > 0, p.z + t * d.z < highest else { continue }
                let w = P3<WorldSpace>(o.x + t * r.x, o.y + t * r.y, 0)
                guard inDomain(w, 1e-9) else { continue }
                if line.edge {
                    let q = P3<WorldSpace>(p.x + t * d.x, p.y + t * d.y, 0)
                    guard backs.filter({ inDomain($0(q), 1e-9) }).count > 1 else { continue }
                }
                out.append(t)
            }
        }
        return out
    }

    /// How far under the solid `p` lies along its line of sight `d`: the
    /// greatest of h(q(t)) - z(t), less p.z, read a five-hundredth of a unit
    /// at a time from a hundred-thousandth on, then a hundred-thousandth at
    /// a time around each greatest within reach of the best, and at each of
    /// `crossings`, where the closed top is the higher side's. A riser's step
    /// lies between two samples, and a line through its top corner is in the
    /// solid for less than any step.
    func depth(_ p: P3<WorldSpace>, _ d: SIMD3<Double>, crossings: [Double]) -> Double {
        let at = SIMD2(p.x, p.y), dxy = SIMD2(d.x, d.y)
        let coarse = 2e-3, fine = 1e-5
        let reach = 2 * (2 * 2.0.squareRoot() * steepest * simd_length(dxy) + abs(d.z)) * coarse
        // Where the line's footprint leaves the copies' bounding box.
        var end = Double.infinity
        for axis in 0..<2 where abs(dxy[axis]) > 1e-15 {
            end = min(end, ((dxy[axis] > 0 ? box.hi[axis] : box.lo[axis]) - at[axis]) / dxy[axis])
        }
        func value(_ t: Double) -> Double? { top(at + t * dxy).map { $0 - t * d.z } }
        var samples: [(t: Double, value: Double)] = []
        var best = p.z, t = fine
        while highest - t * d.z > best - reach, t <= end + coarse {
            if let v = value(t) { samples.append((t, v)); best = max(best, v) }
            t += coarse
        }
        for (k, s) in samples.enumerated() where s.value > best - reach
            && (k == 0 || s.value >= samples[k - 1].value) && (k + 1 == samples.count || s.value >= samples[k + 1].value) {
            var u = max(s.t - coarse, fine)
            while u < s.t + coarse {
                if let v = value(u), v > best { best = v }
                u += fine
            }
        }
        for t in crossings { if let v = value(t), v > best { best = v } }
        return best - p.z
    }
}

// MARK: - curves on a plateau are tested by the plateau
//
// A plateau is flat and faces up: seen from above its hatching is tested by
// the traversal alone, and seen from below, behind the solid it bounds, none
// of it shows, whatever the traversal says.

func plateauCurveTests() {
    Check.suite("plateaus: the hatching on a plateau is tested by the plateau") {
        let preset = Catalog.native.preset("rgamma")!
        let bundle = try NativeLandscape.build(LandscapeRequest(preset: preset, resolution: 240))
        let scene = Scene(bundle: bundle, preset: bundle.manifest.presets[0])
        let h = scene.heightfield!
        let hatch = scene.layers.first { $0.spec.name == "plateau_hatch" }!
        Check.expect(scene.geometry.face(of: hatch.spec) == .plateau && hatch.paths.count > 0,
                     "the plateau hatching lies on the plateaus", "\(hatch.paths.count) lines")
        let above = scene.camera.view
        let onPlateaus = h.visibleCurves(hatch.paths, on: .plateau, view: above, depthTolerance: scene.depthTolerance)
        let alone = h.visibleCurves(hatch.paths, on: nil, view: above, depthTolerance: scene.depthTolerance)
        Check.expect(onPlateaus.count > 0 && onPlateaus.vertices == alone.vertices && onPlateaus.offsets == alone.offsets,
                     "seen from above, what the traversal alone keeps", "\(onPlateaus.count) runs")
        let below = testCamera(elevation: 120, azimuth: 30)
        Check.expect(below.projectionDirection.z > 0, "a camera under the landscape looks up")
        // With the traversal disarmed, on the grid, so that the facing alone
        // decides.
        let fromBelow = h.onGrid.visibleCurves(hatch.paths, on: .plateau, view: below, depthTolerance: .infinity)
        let disarmed = h.onGrid.visibleCurves(hatch.paths, on: nil, view: below, depthTolerance: .infinity)
        Check.expect(fromBelow.count == 0 && disarmed.count > 0, "seen from below, nothing, by the facing alone",
                     "\(fromBelow.count) runs; the disarmed traversal alone keeps \(disarmed.count)")
    }
}

// MARK: - a vertical stroke on a wall is its two ends
//
// A wall's hatch lines and its corners' vertical edges had a vertex every step
// of height, a vertex a cell up to |f|; untruncated, exp(10z) on a grid of 16
// put fourteen million on the walls, and exp(z²) and sin(30z) asked for more
// memory than there is. No vertex between the ends is needed: the solid is
// everything under the surface and every line of sight rises toward the eye,
// so what is hidden of a vertical stroke is one piece from its bottom up, and
// the bake bisects to where it ends.

func wallStrokeTests() {
    Check.suite("walls: a vertical stroke is its two ends, however high |f| stands") {
        // Untruncated, at sizes the subdivision every step survives, so that
        // this cannot itself run out of memory: hundreds of thousands of
        // vertices on the walls there, where two a stroke is dozens.
        let cases: [(expression: String, half: Double, spacing: Double)] = [
            ("exp(10*z)", 1, 0.5), ("exp(z^2)", 3, 1.5), ("sin(30*z)", 0.3, 0.15),
        ]
        for c in cases {
            let domain = Domain(real: Interval(lo: -c.half, hi: c.half), imag: Interval(lo: -c.half, hi: c.half))
            let bundle = try NativeLandscape.build(LandscapeRequest(
                expression: c.expression, domain: domain, resolution: 8, truncation: Truncation.none,
                spacing: c.spacing))
            let g = bundle.surface.height
            let cell = max(domain.real.length / Double(g.width - 1), domain.imag.length / Double(g.height - 1))
            let highest = g.values.map(Double.init).max() ?? 0
            Check.expect(highest > 1000 * cell, "\(c.expression): the walls rise a thousand cells and more",
                         String(format: "max |f| %.4g, a cell %.3g", highest, cell))
            // The budget reads the number of strokes and nothing of |f|: two
            // vertices a hatch line, and two a corner's vertical edge.
            let hatch = bundle.layers.first { $0.spec.name == "wall_hatch" }!.paths
            Check.expect(hatch.count > 0 && hatch.vertices.count <= 2 * hatch.count,
                         "\(c.expression): the wall hatch has two vertices a line",
                         "\(hatch.vertices.count) vertices on \(hatch.count) lines")
            let edges = bundle.layers.first { $0.spec.name == "wall_edges" }!.paths
            let verticals = (0..<edges.count).map { Array(edges[path: $0]) }.filter { p in
                p.allSatisfy { $0.x == p[0].x && $0.y == p[0].y }
            }
            let corners = verticals.reduce(0) { $0 + $1.count }
            Check.expect(verticals.count == 4 && corners <= 2 * verticals.count,
                         "\(c.expression): and each corner's vertical edge two",
                         "\(corners) vertices on \(verticals.count) vertical edges")
        }
    }

    Check.suite("walls: every camera navigation reaches sees from above the floor's plane") {
        var lowest = Double.infinity
        for elevation in stride(from: -Orbit.elevationLimit.degrees, through: Orbit.elevationLimit.degrees, by: 0.5) {
            for azimuth in stride(from: 0.0, to: 360, by: 15) {
                var orbit = Orbit(matching: NativeLandscape.projection)
                orbit.elevation = Angle(degrees: elevation); orbit.azimuth = Angle(degrees: azimuth)
                lowest = min(lowest, -orbit.camera.view.projectionDirection.z)
            }
        }
        Check.expect(lowest > 0, "a line of sight rises toward the eye", "least rise \(lowest)")
    }
}
