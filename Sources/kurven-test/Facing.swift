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
                                tiles: [.identity], step: 1)

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
        let cut = field.visibleSurfaceCurves(curves, view: camera.view, depthTolerance: .infinity).mapped(camera.view)
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
        let visible = field.visibleSurfaceCurves(curves, view: camera.view, depthTolerance: 0.02).mapped(camera.view)
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
            let hatch = surface.hatch(.wallHatching(edges: Array(layout.walls.edges.indices), spacing: 0.1, step: 0.1),
                                      in: context)!
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
