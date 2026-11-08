import AppKit
import KurvenCore

/// Picking a face: a click reads a world point off the preview's depth
/// buffer, `SceneGeometry.face(at:within:faces:)` names the face under it,
/// and its boundary is drawn over the scene in the accent colour, through the
/// stroke path the curves take, tested as they are.
extension Document {
    /// Pick the face under a point of space, the point a click reads off the
    /// preview (`MetalRenderer.worldPoint(atPixel:navigator:viewport:)`), or
    /// clear the pick where there is none: a click off the drawing gives no
    /// point, and a point no face is near picks nothing. The boundary is made
    /// on another thread; `picking` is set until the latest click's lands,
    /// and a boundary that lands after a later click or clear is dropped.
    func pick(at point: P3<WorldSpace>?) {
        pickGeneration &+= 1
        guard let scene, let navigator, let point else {
            clearPick("picked nothing")
            return
        }
        let geometry = scene.geometry
        let faces = faces(of: scene)
        let tolerance = geometry.pickTolerance(at: point, unitsPerPixel: navigator.framing.unitsPerPixel)
        guard let pick = geometry.face(at: point, within: tolerance, faces: faces) else {
            clearPick("picked nothing at \(point.v)")
            return
        }
        picked = pick
        drawBoundary(of: pick, on: scene) { boundary in
            String(format: "picked %@ at (%.4f, %.4f, %.4f), %d curves round it",
                   "\(pick)", point.x, point.y, point.z, boundary)
        }
    }

    /// The scene's faces, made once while its geometry is kept.
    func faces(of scene: Scene) -> [Face] {
        if pickFaces?.content != scene.content { pickFaces = (scene.content, scene.geometry.faces) }
        return pickFaces?.faces ?? []
    }

    /// Make the boundary of `pick` on another thread and draw it over
    /// `scene` when it lands, unless a later click or clear came first.
    private func drawBoundary(of pick: FacePick, on scene: Scene, status: @escaping (Int) -> String) {
        let generation = pickGeneration
        let geometry = scene.geometry
        let faces = faces(of: scene)
        picking = true
        let content = scene.content
        Task {
            let boundary = await Task.detached(priority: .userInitiated) {
                geometry.boundary(of: pick, faces: faces)
            }.value
            guard generation == pickGeneration else { return }
            picking = false
            guard let scene = self.scene, scene.content == content else { return }
            let widest = scene.layers.map(\.spec.width).max() ?? 1
            let spec = LayerSpec(name: "selection", role: .scaffold, source: .file(vertices: "", offsets: ""),
                                 width: widest, heightPolicy: .surface, color: Self.accentColor)
            let paths = geometry.parametricSurface != nil
                ? PolylineSet(paths: boundary.paths, coords: boundary.coords)
                : PolylineSet(paths: boundary.paths)
            let layer = Layer(spec: spec, paths: paths)
            selection = Selection(pick: pick, layer: layer, scene: scene.drawing(scene.layers + [layer]), base: scene.curves)
            pickStatus = status(boundary.paths.count)
        }
    }

    /// No pick: what a click off the drawing leaves.
    private func clearPick(_ status: String) {
        selection = nil
        picked = nil
        picking = false
        pickStatus = status
    }

    /// Carry the pick across a new scene: its boundary drawn over the new
    /// curves where the geometry is the same. Where the geometry is new --
    /// a surface edited, a landscape restyled -- the face is kept if the
    /// new geometry has it, and its boundary made again; if not, the pick
    /// is cleared.
    func keepSelection() {
        guard let picked else { return }
        guard let scene else {
            clearPick("picked nothing")
            return
        }
        if let selection, scene.content == selection.scene.content {
            if scene.curves != selection.base {
                self.selection = Selection(pick: selection.pick, layer: selection.layer,
                                           scene: scene.drawing(scene.layers + [selection.layer]), base: scene.curves)
            }
            return
        }
        selection = nil
        guard faces(of: scene).contains(where: { $0.id == picked.face && picked.copy < $0.copies.count }) else {
            pickGeneration &+= 1
            clearPick("the picked face is gone")
            return
        }
        pickGeneration &+= 1
        drawBoundary(of: picked, on: scene) { "picked \(picked), \($0) curves round it" }
    }

    /// The system's accent colour as the stroke's `#rrggbb`.
    static var accentColor: String {
        let c = NSColor.controlAccentColor.usingColorSpace(.sRGB) ?? .systemBlue
        func byte(_ x: CGFloat) -> Int { Int((min(max(x, 0), 1) * 255).rounded()) }
        return String(format: "#%02x%02x%02x", byte(c.redComponent), byte(c.greenComponent), byte(c.blueComponent))
    }
}

private extension SceneGeometry {
    var parametricSurface: ParametricSurface? {
        if case .parametric(let s) = self { return s }
        return nil
    }
}
