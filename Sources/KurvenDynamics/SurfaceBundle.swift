import Foundation
import KurvenCore

/// A parametric surface's `.kurven` bundle: its manifest, and the drawing its
/// request builds.
///
/// The bundle is the manifest alone (`SurfaceManifest`): the request, which
/// `SurfaceDrawing.build` turns into the same drawing `kurven-cli surface`
/// draws, and the layers as they were kept, which restyle it
/// (`SurfaceDrawing.restyled(_:)`). A landscape's bundle carries its grids
/// because f was sampled once; a parametric surface's map is code, and costs
/// less to run again than to store.
public struct SurfaceBundle: Sendable {
    public let url: URL
    public let manifest: SurfaceManifest
    public let drawing: SurfaceDrawing

    /// The bundle that keeps `drawing`: its request, its layers as drawn, and
    /// the camera presets it opens at.
    public init(url: URL, drawing: SurfaceDrawing, presets: [CameraPreset] = [SurfaceRequest.preset],
                gitSha: String = "native") {
        self.url = url
        self.drawing = drawing
        manifest = SurfaceManifest(
            surface: drawing.request, layers: drawing.layers.map(\.spec), presets: presets,
            provenance: Provenance(function: Self.function(drawing.request), example: "surface", params: [:],
                                   cpuCount: 1, gitSha: gitSha))
    }

    private init(url: URL, manifest: SurfaceManifest, drawing: SurfaceDrawing) {
        self.url = url; self.manifest = manifest; self.drawing = drawing
    }

    /// The same bundle drawing `layers` in place of its own: its request,
    /// presets and provenance kept, the surface and its identity too, so
    /// only the curves are new. The curves the request draws itself are the
    /// request's, as `layers` lists them or not (`SurfaceDrawing.keeping(_:from:)`):
    /// layers asked for while the request was being edited name its curves
    /// as they were before the edit.
    public func restyled(_ layers: [LayerSpec]) throws -> SurfaceBundle {
        let drawing = try self.drawing.keeping(layers)
        var manifest = self.manifest
        manifest.layers = drawing.layers.map(\.spec)
        return SurfaceBundle(url: url, manifest: manifest, drawing: drawing)
    }

    /// The same bundle recording `gitSha` as the build that made it: what an
    /// edit undone, or a drag let go where it began, puts back.
    public func recording(gitSha: String) -> SurfaceBundle {
        var manifest = self.manifest
        manifest.provenance.gitSha = gitSha
        return SurfaceBundle(url: url, manifest: manifest, drawing: drawing)
    }

    /// The mathematics a request draws, for the provenance.
    static func function(_ request: SurfaceRequest) -> String {
        switch request.shape {
        case .torus: "torus"
        case .sn(let m, _, _, _), .relief(let m, _, _, _, _, _, _, _, _): "sn(z, \(m))"
        case .periodic(let expression, _, _, _, _): expression
        case .forced(let system, _, _, _, _, _, _): system
        }
    }

    /// Whether the bundle at `url` holds a parametric surface rather than a
    /// landscape: its manifest has a `surface` (`BundleManifest`). False
    /// where it has no manifest, for the landscape's reader to say so.
    public static func holdsSurface(at url: URL) throws -> Bool {
        let file = url.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: file.path) else { return false }
        return try JSONValue.parse(Data(contentsOf: file)).object("Manifest")["surface"] != nil
    }

    /// Read a bundle and build its drawing: the request built, then drawn
    /// with the manifest's layers.
    public static func read(at url: URL) throws -> SurfaceBundle {
        let file = url.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: file.path) else {
            throw BundleError.missingFile("manifest.json", in: url.lastPathComponent)
        }
        let manifest = try SurfaceManifest(json: JSONValue.parse(Data(contentsOf: file)))
        guard !manifest.presets.isEmpty else {
            throw ManifestError.unsupported("a parametric surface with no camera preset: it has no view to open at "
                                            + "or to bake from")
        }
        let drawing = try SurfaceDrawing.build(manifest.surface).restyled(manifest.layers)
        return SurfaceBundle(url: url, manifest: manifest, drawing: drawing)
    }

    /// Write the bundle: a directory holding its manifest, with `layers` in
    /// place of its own when given -- the layers as they are drawn.
    public func write(to url: URL, layers: [LayerSpec]? = nil) throws {
        var manifest = self.manifest
        if let layers { manifest.layers = layers }
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        try manifest.canonicalJSON.write(to: url.appendingPathComponent("manifest.json"),
                                         atomically: true, encoding: .utf8)
    }

    /// The scene the bundle describes under one of its presets.
    public func scene(preset: CameraPreset) -> Scene {
        Scene(surface: drawing.surface, layers: drawing.layers, camera: .oblique(preset.projection),
              depthTolerance: preset.depthTolerance)
    }
}
