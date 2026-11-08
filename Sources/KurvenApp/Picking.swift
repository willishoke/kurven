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
    /// on another thread; `picking` is set until it lands.
    func pick(at point: P3<WorldSpace>?) {
        guard let scene, let navigator, let point else {
            selection = nil
            pickStatus = "picked nothing"
            return
        }
        let geometry = scene.geometry
        if pickFaces?.content != scene.content { pickFaces = (scene.content, geometry.faces) }
        let faces = pickFaces?.faces ?? []
        let tolerance = geometry.pickTolerance(unitsPerPixel: navigator.framing.unitsPerPixel)
        guard let pick = geometry.face(at: point, within: tolerance, faces: faces) else {
            selection = nil
            pickStatus = "picked nothing at \(point.v)"
            return
        }
        picking = true
        let content = scene.content
        Task {
            let boundary = await Task.detached(priority: .userInitiated) {
                geometry.boundary(of: pick, faces: faces)
            }.value
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
            pickStatus = String(format: "picked %@ at (%.4f, %.4f, %.4f), %d curves round it",
                                "\(pick)", point.x, point.y, point.z, boundary.paths.count)
        }
    }

    /// Carry the pick across a new scene: its boundary drawn over the new
    /// curves where the geometry is the same, and dropped where it is not.
    func keepSelection() {
        guard let selection else { return }
        guard let scene, scene.content == selection.scene.content else {
            self.selection = nil
            return
        }
        if scene.curves != selection.base {
            self.selection = Selection(pick: selection.pick, layer: selection.layer,
                                       scene: scene.drawing(scene.layers + [selection.layer]), base: scene.curves)
        }
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
