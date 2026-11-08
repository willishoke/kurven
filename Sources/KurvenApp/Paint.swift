import Foundation
import KurvenCore
import KurvenLandscape
import KurvenDynamics

/// What painting edits: the layers, a parametric surface's request, which
/// draws some of them and carries the function, and the level counts the
/// sliders set. One value, so an edit is one step to undo, and undoing it
/// puts back exactly what a save wrote before it.
struct Paint: Equatable {
    var layers: [LayerSpec]
    var request: SurfaceRequest?
    var levelCounts: [String: Int]
}

/// Why a paint edit was refused.
struct PaintRefusal: Error, CustomStringConvertible {
    let description: String
}

/// A curve the Add menu offers on a face: a layer as the catalog would draw
/// it there, at the catalog's levels or spacing.
struct PaintChoice: Identifiable, Equatable {
    /// How `--add` names it: the layer's name.
    var id: String { spec.name }
    let label: String
    let spec: LayerSpec
}

/// Painting curves on the picked face: the "Curves on" section's model.
///
/// A layer names the faces it lies on by a face selector (`FaceSelector`),
/// which names a kind of face, not one face: the graph, every plateau,
/// listed walls, every riser, the surface, every face of a solid. So a
/// layer painted on one plateau lies on all of them, and one painted on a
/// wall lies on that wall, which a wall's selector can name alone. Every
/// edit goes through `paintEdit` or a drag, so each is one undo step.
@MainActor
extension Document {
    // MARK: - the state

    /// What is painted now: the layers last asked for, which a restyle in
    /// flight is still drawing, and the request the controls say.
    var paint: Paint? {
        switch state {
        case .ready(.landscape):
            Paint(layers: styling?.layers ?? [], request: nil, levelCounts: levelCounts)
        case .ready(.surface(let bundle)):
            Paint(layers: surfaceStyled ?? bundle.manifest.layers, request: surface ?? bundle.manifest.surface,
                  levelCounts: levelCounts)
        case .ready(.polyhedron):
            Paint(layers: polyhedronLayers ?? [], request: nil, levelCounts: levelCounts)
        case .empty, .loading, .failed:
            nil
        }
    }

    /// Paint `next`: the layers restyled from what is built, a request edit
    /// built as the surface's controls build one, the levels drawn again.
    func setPaint(_ next: Paint) {
        guard let current = paint, next != current else { return }
        let levelsMoved = next.levelCounts != current.levelCounts
        levelCounts = next.levelCounts
        switch state {
        case .ready(.landscape):
            if next.layers != current.layers { restyle { $0.layers = next.layers } }
        case .ready(.surface):
            if let request = next.request, request != current.request {
                surfaceStyled = next.layers
                surface = request
                surfaceEdited(draft: false)
            } else if next.layers != current.layers {
                restyleSurface(next.layers)
            }
        case .ready(.polyhedron):
            if next.layers != current.layers { restylePolyhedron { $0 = next.layers } }
        case .empty, .loading, .failed:
            return
        }
        if levelsMoved { rebuildCurves() }
    }

    /// One edit, one undo step, named in the Edit menu.
    func paintEdit(_ name: String, _ change: (inout Paint) throws -> Void) rethrows {
        guard let before = paint else { return }
        var after = before
        try change(&after)
        guard after != before else { return }
        registerUndo(from: before, to: after, name)
        setPaint(after)
    }

    /// A slider taken hold of: what to undo back to when it is let go.
    func beginPaintDrag() {
        paintBeforeDrag = paint
    }

    /// A slider let go: the whole drag is one undo step.
    func endPaintDrag(_ name: String) {
        defer { paintBeforeDrag = nil }
        guard let before = paintBeforeDrag, let after = paint, before != after else { return }
        registerUndo(from: before, to: after, name)
    }

    /// ⌘Z paints `before` again, and ⌘⇧Z `after`.
    private func registerUndo(from before: Paint, to after: Paint, _ name: String) {
        guard let undoManager else { return }
        undoManager.registerUndo(withTarget: self) { document in
            MainActor.assumeIsolated {
                document.registerUndo(from: after, to: before, name)
                document.setPaint(before)
            }
        }
        undoManager.setActionName(name)
    }

    /// A different document: nothing done to the last one can be undone on
    /// it, and nothing of it is picked.
    func forgetEdits() {
        undoManager?.removeAllActions(withTarget: self)
        paintBeforeDrag = nil
        paintStatus = nil
        picked = nil
        selection = nil
    }

    // MARK: - the layers on a face

    /// The indices into `paint.layers` of the layers that lie on the picked
    /// face. A layer whose selector is not tiled lies only on the copy in
    /// place, on the identity tile; one on listed walls, only on those.
    func paintedLayers(on pick: FacePick) -> [Int] {
        guard let paint, let scene else { return [] }
        return paint.layers.indices.filter { lies(paint.layers[$0], on: pick, scene.geometry) }
    }

    private func lies(_ spec: LayerSpec, on pick: FacePick, _ geometry: SceneGeometry) -> Bool {
        // The fold curves are the graph's or the surface's, though their
        // image is the outline and no face holds them.
        if case .foldCurves = spec.source { return pick.face.kind == .graph || pick.face.kind == .surface }
        guard geometry.face(of: spec) == pick.face.kind else { return false }
        switch spec.source {
        case .contour(_, _, _, .walls(let edges, _, _, _)), .hatching(_, .walls(let edges, _, _, _)),
             .wallEdges(let edges, _):
            guard case .wall(let edge) = pick.face else { return false }
            return edges.contains(edge)
        case .contour(_, _, _, let faces), .hatching(_, let faces):
            return Self.tiled(faces) ?? true || onIdentityTile(pick)
        default:
            return true
        }
    }

    private static func tiled(_ faces: FaceSelector) -> Bool? {
        switch faces {
        case .graph(let tiled), .plateaus(let tiled), .risers(let tiled, _): tiled
        case .walls, .surface, .solid: nil
        }
    }

    private func onIdentityTile(_ pick: FacePick) -> Bool {
        guard let tiles = bundle?.manifest.occluder.tiles, tiles.indices.contains(pick.copy) else { return true }
        return tiles[pick.copy] == .identity
    }

    /// Whether what is painted on a landscape lands on every tile.
    private var paintsEveryTile: Bool { (bundle?.manifest.occluder.tiles.count ?? 1) > 1 }

    /// "Curves on" what: the faces a layer added here lies on.
    func paintTitle(_ pick: FacePick) -> String {
        switch pick.face {
        case .graph: "the graph"
        case .plateau: "the plateaus"
        case .riser, .tileBoundaryRiser: "the risers"
        case .wall(let edge): "wall \(edge)"
        case .surface: "the surface"
        case .solid: "the solid's faces"
        }
    }

    /// What the section says about where its layers lie, since a selector
    /// names a kind of face and not the one picked.
    func paintCaption(_ pick: FacePick) -> String {
        let count = scene.map { faces(of: $0).filter { $0.id.kind == pick.face.kind }.count } ?? 0
        let tiles = paintsEveryTile ? ", on every tile" : ""
        switch pick.face {
        case .graph:
            return "Picked the graph. A layer added here lies on the graph\(tiles)."
        case .plateau(let band):
            return "Picked plateau \(band). A layer added here lies on "
                + (count == 1 ? "the whole plateau, wherever |f| reaches the truncation" : "all \(count) plateaus")
                + "\(tiles)."
        case .riser, .tileBoundaryRiser:
            return "Picked \(pick). A layer added here lies on " + (count == 1 ? "it" : "all \(count) risers")
                + "\(tiles)."
        case .wall(let edge):
            return "Picked wall \(edge). A layer added here lies on wall \(edge) alone; "
                + "a layer listed may lie on other walls too."
        case .surface:
            return "Picked the surface, which is one face."
        case .solid(let face):
            return "Picked face \(face). Hatching added here lies on all \(count) faces of the solid."
        }
    }

    // MARK: - adding

    /// What the Add menu offers on the picked face: a contour of each field
    /// the face exposes, hatching on a flat face, the fold curves where
    /// there are none, each as the catalog draws it -- |f| and arg f at the
    /// catalog's levels, hatching at its spacing -- and a coordinate or the
    /// height at seven levels across the face. Hatching is not laid on a
    /// curved face, and a polyhedron's faces take hatching alone.
    func paintChoices(on pick: FacePick) -> [PaintChoice] {
        guard let paint, let scene, let face = faces(of: scene).first(where: { $0.id == pick.face }) else { return [] }
        let has = { (source: LayerSource) in paint.layers.contains { $0.source == source } }
        var choices: [PaintChoice] = []
        func contours(_ fields: [ContourField], on faces: FaceSelector, heightPolicy: HeightPolicy = .surface) {
            for field in fields where faces.unsupported(field) == nil {
                let levels = Self.levels(of: field, on: face)
                guard levels.count >= 2 else { continue }
                choices.append(PaintChoice(
                    label: "Contours of \(Self.words(field))",
                    spec: LayerSpec(name: "\(faces.name)_\(field.rawValue)", role: .scaffold,
                                    source: .contour(field: field, levels: levels, restriction: .all, faces: faces),
                                    width: 0.2, heightPolicy: heightPolicy)))
            }
        }
        func catalog(_ spec: LayerSpec, _ label: String) {
            choices.append(PaintChoice(label: label, spec: spec))
        }

        switch state {
        case .ready(.landscape(let bundle)):
            let m = bundle.manifest
            let tiled = paintsEveryTile
            // The ceiling the build chose the levels under: the truncation's,
            // or with none, read off the samples as the build reads it.
            let ceiling = LandscapeStyle.ceiling(of: m.truncation)
                ?? NativeLandscape.ceiling(of: m.truncation, magnitudes: bundle.surface.height.values.map(Double.init))
            let defaults = NativeLandscape.defaultLayers(m.domain, truncation: m.truncation, ceiling: ceiling,
                                                         spacing: nil, phase: bundle.surface.phase != nil)
            func template(_ name: String) -> LayerSpec? { defaults.first { $0.name == name } }
            switch pick.face {
            case .graph:
                for (name, label) in [("mag_major", "|f| at the major levels"), ("mag_minor", "|f| at the minor levels"),
                                      ("ang_major", "arg f at the major levels"), ("ang_minor", "arg f at the minor levels")] {
                    guard var spec = template(name),
                          case .contour(let field, let levels, let restriction, _) = spec.source else { continue }
                    spec.source = .contour(field: field, levels: levels, restriction: restriction, faces: .graph(tiled: tiled))
                    catalog(spec, label)
                }
                contours([.u, .v, .height], on: .graph(tiled: tiled))
                if !has(.foldCurves), let folds = template("folds") { catalog(folds, "Fold curves") }
            case .plateau:
                if var spec = template("plateau_hatch"), case .hatching(let hatching, _) = spec.source {
                    spec.source = .hatching(hatching, faces: .plateaus(tiled: tiled))
                    catalog(spec, "Hatching along Re")
                }
                contours([.u, .v], on: .plateaus(tiled: tiled))
            case .wall(let edge):
                if var spec = template("wall_hatch"), case .hatching(let hatching, _) = spec.source {
                    spec.source = .hatching(hatching, faces: .walls(edges: [edge], endLines: false, base: 0, topOffset: 0))
                    catalog(spec, "Hatching")
                }
                contours([.u, .v, .height], on: .walls(edges: [edge], endLines: false, base: 0, topOffset: 0))
            case .riser, .tileBoundaryRiser:
                if let wall = template("wall_hatch"), case .hatching(let hatching, _) = wall.source {
                    var spec = wall
                    spec.name = "riser_hatch"
                    spec.source = .riserHatching(spacing: hatching.spacing, tiled: tiled)
                    catalog(spec, "Hatching")
                }
                contours([.u, .v, .height], on: .risers(tiled: tiled, endLines: false))
            case .surface, .solid:
                break
            }
        case .ready(.surface):
            guard let request = paint.request else { break }
            if request.paintsFunction {
                let labels = ["mag_major": "|f| at the major levels", "mag_minor": "|f| at the minor levels",
                              "ang_major": "arg f at the major levels", "ang_minor": "arg f at the minor levels",
                              "truncation edge": "The truncation edge"]
                for spec in PeriodicTorus.contours(ceiling: 3, truncated: request.truncatesFunction) {
                    catalog(spec, labels[spec.name] ?? spec.name)
                }
            }
            contours([.u, .v, .height], on: .surface)
            if request.lines == nil {
                catalog(LayerSpec(name: "lines", role: .scaffold, source: .coordinateCurves(u: 36, v: 18),
                                  width: request.style.lines, heightPolicy: .surface), "Coordinate curves")
            }
            if request.style.folds == 0 {
                catalog(Self.folds(SurfaceRequest.Style().folds), "Fold curves")
            }
        case .ready(.polyhedron(let bundle)):
            let width = paint.layers.first { $0.source == .edges }.map { $0.width / 2 } ?? 0.3
            let reach = bundle.solid.frames.map(\.extent).max() ?? 1
            catalog(LayerSpec(name: "hatching", role: .scaffold,
                              source: .hatching(Hatching(angle: .pi / 4, spacing: reach / 25), faces: .solid),
                              width: width, heightPolicy: .surface),
                    "Hatching")
        case .empty, .loading, .failed:
            break
        }
        return choices
    }

    /// Add `choice` to the picked face's layers: before the outline drawn
    /// last, under a name of its own. A surface's coordinate curves and fold
    /// curves are its request's, which the edit sets too.
    func add(_ choice: PaintChoice) {
        paintEdit("Add \(choice.label)") { paint in
            var spec = choice.spec
            let names = Set(paint.layers.map(\.name))
            if names.contains(spec.name), !SurfaceDrawing.describesRequest(spec) || paint.request == nil {
                spec.name = (2...).lazy.map { "\(spec.name) \($0)" }.first { !names.contains($0) }!
            }
            if var request = paint.request, SurfaceDrawing.describesRequest(spec) {
                switch spec.source {
                case .coordinateCurves(let u, let v): request.lines = SIMD2(u, v)
                case .foldCurves: request.style.folds = spec.width
                default: break
                }
                paint.request = request
            }
            let outline = paint.layers.reversed().prefix { $0.role == .outline }.count
            paint.layers.insert(spec, at: spec.role == .outline ? paint.layers.count : paint.layers.count - outline)
        }
        paintStatus = nil
    }

    /// Take a layer off: one of the face's, by its index in `paint.layers`.
    /// A surface's coordinate curves, linear curve and fold curves go from
    /// its request with it; a forced torus's trajectory is the torus's own
    /// and stays.
    func removeLayer(at index: Int) throws {
        guard let spec = paint?.layers[safe: index] else { return }
        if case .trajectory = spec.source {
            throw PaintRefusal(description: "the trajectory is the forced torus's own: set its turns instead")
        }
        paintEdit("Remove \(spec.name)") { paint in
            if var request = paint.request, SurfaceDrawing.describesRequest(spec) {
                switch spec.source {
                case .coordinateCurves: request.lines = nil
                case .linearCurves: request.linearCurves = nil
                case .foldCurves: request.style.folds = 0
                default: break
                }
                paint.request = request
            }
            paint.layers.remove(at: index)
            paint.levelCounts.removeValue(forKey: spec.name)
        }
        paintStatus = nil
    }

    /// Paint a function on the surface (`SurfaceRequest.painting(_:periods:)`):
    /// its |f| and arg f, whose contours can then be added. Refused, saying
    /// why, and nothing changes.
    func addFunction(_ expression: String, periods: SIMD2<Double>) throws {
        guard let request = paint?.request else {
            throw FunctionRefusal("a function is painted on a parametric surface")
        }
        let next = try request.painting(expression, periods: periods)
        paintEdit("Function \(expression)") { $0.request = next }
        paintStatus = nil
    }

    // MARK: - sliders

    /// Set a layer's level count as its slider does: a step of a drag, or,
    /// with none under way, an edit of its own.
    func paintLevels(_ count: Int, ofLayer name: String, draft: Bool) {
        guard let index = layerSpecs.firstIndex(where: { $0.name == name }) else { return }
        if paintBeforeDrag == nil {
            paintEdit("Levels of \(name)") { $0.levelCounts[name] = max(count, 2) }
        } else {
            setLevelCount(count, forLayer: index, draft: draft)
        }
    }

    /// Set a hatching's strokes as its slider does.
    func paintStrokes(_ count: Int, ofLayer name: String) {
        guard let index = paint?.layers.firstIndex(where: { $0.name == name }) else { return }
        if paintBeforeDrag == nil {
            beginPaintDrag()
            setHatchCount(count, ofLayer: index)
            endPaintDrag("Strokes of \(name)")
        } else {
            setHatchCount(count, ofLayer: index)
        }
    }

    /// Put a layer's levels back to its own.
    func paintResetLevels(ofLayer name: String) {
        paintEdit("Reset Levels of \(name)") { $0.levelCounts.removeValue(forKey: name) }
    }

    /// Run the plateaus' hatching along an axis.
    func paintAxis(_ axis: DomainAxis) {
        beginPaintDrag()
        setPlateauHatchAxis(axis)
        endPaintDrag("Hatching along \(axis == .real ? "Re" : "Im")")
    }

    // MARK: - defaults

    private static func folds(_ width: Double) -> LayerSpec {
        LayerSpec(name: "folds", role: .outline, source: .foldCurves, width: width, heightPolicy: .surface)
    }

    private static func words(_ field: ContourField) -> String {
        switch field {
        case .magnitude: "|f|"
        case .phase: "arg f"
        case .u: "u"
        case .v: "v"
        case .height: "the height"
        }
    }

    /// Seven levels evenly across a field's range on a face, its ends left
    /// out: a coordinate's from the face's parameter rectangle, the height's
    /// from the face sampled where its first copy holds it.
    static func levels(of field: ContourField, on face: Face) -> [Double] {
        let (u, v) = face.domain
        var lo = Double.infinity, hi = -Double.infinity
        switch field {
        case .u: (lo, hi) = (u.range.lo, u.range.hi)
        case .v: (lo, hi) = (v.range.lo, v.range.hi)
        case .height, .magnitude, .phase:
            let n = 32
            for j in 0...n {
                for i in 0...n {
                    let p = P2<ParamSpace>(u.range.lo + u.range.length * Double(i) / Double(n),
                                           v.range.lo + v.range.length * Double(j) / Double(n))
                    guard face.copies.first?.contains(p) ?? true else { continue }
                    let z = face.map(p).position.z
                    guard z.isFinite else { continue }
                    lo = min(lo, z); hi = max(hi, z)
                }
            }
        }
        guard lo.isFinite, hi.isFinite, hi > lo else { return [] }
        return (1...7).map { lo + (hi - lo) * Double($0) / 8 }
    }
}
