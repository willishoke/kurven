import Foundation

/// A polyhedron's `.kurven` bundle: its manifest, and the layers drawn on its
/// solid.
///
/// The bundle is the manifest alone (`PolyhedronManifest`): the vertices and
/// loops, from which the solid is built as `kurven-cli surface solid` builds
/// it, and the layers, its edges and the hatching laid on its faces. Reading
/// it draws what the CLI draws from the same solid, byte for byte.
public struct PolyhedronBundle: Sendable {
    public let url: URL
    public let manifest: PolyhedronManifest
    /// The layers as drawn on the solid, before a camera: every edge, and
    /// every hatch line.
    public let layers: [Layer]

    public var solid: Solid { manifest.solid }

    /// The bundle that keeps `solid` with `layers` drawn on it, and the
    /// camera presets it opens at. The layers are those a polyhedron's
    /// manifest takes (`PolyhedronManifest.unsupported(_:on:)`).
    public init(url: URL, solid: Solid, layers: [LayerSpec], presets: [CameraPreset], function: String,
                gitSha: String = "native") {
        let manifest = PolyhedronManifest(
            solid: solid, layers: layers, presets: presets,
            provenance: Provenance(function: function, example: "polyhedron", params: [:], cpuCount: 1,
                                   gitSha: gitSha))
        self.init(url: url, manifest: manifest, layers: layers.map { PolyhedronBundle.drawn($0, on: solid) })
    }

    private init(url: URL, manifest: PolyhedronManifest, layers: [Layer]) {
        self.url = url; self.manifest = manifest; self.layers = layers
    }

    /// One layer drawn on the solid: its edges, or its hatch lines, as the
    /// CLI draws them. The scene tests both for its camera
    /// (`Scene.visible(_:depthTolerance:image:)`).
    static func drawn(_ spec: LayerSpec, on solid: Solid) -> Layer {
        switch spec.source {
        case .hatching(let hatching, .solid): Layer(spec: spec, paths: solid.hatchPaths(hatching))
        default: Layer(spec: spec, paths: solid.edgePaths)
        }
    }

    /// The same solid with `layers` drawn on it, each kept as it is drawn
    /// here when its source is unchanged, so a restyle of one hatching lays
    /// that hatching and nothing else; or why the solid cannot take them.
    /// The layers keep their places and presets: a restyle, not a new
    /// polyhedron.
    public func restyled(_ layers: [LayerSpec]) throws -> PolyhedronBundle {
        if let why = PolyhedronManifest.unsupported(layers, on: solid) { throw ManifestError.invalid(why) }
        var manifest = self.manifest
        manifest.layers = layers
        return PolyhedronBundle(url: url, manifest: manifest, layers: layers.map { spec in
            if let kept = self.layers.first(where: { $0.spec.source == spec.source }) {
                return Layer(spec: spec, paths: kept.paths)
            }
            return PolyhedronBundle.drawn(spec, on: solid)
        })
    }

    /// Whether the bundle at `url` holds a polyhedron: its manifest has a
    /// `polyhedron` (`BundleManifest`). False where it has no manifest, for
    /// the landscape's reader to say so.
    public static func holdsPolyhedron(at url: URL) throws -> Bool {
        let file = url.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: file.path) else { return false }
        return try JSONValue.parse(Data(contentsOf: file)).object("Manifest")["polyhedron"] != nil
    }

    /// Read a bundle: the solid built from its vertices and loops, checked,
    /// and its layers drawn on it.
    public static func read(at url: URL) throws -> PolyhedronBundle {
        let file = url.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: file.path) else {
            throw BundleError.missingFile("manifest.json", in: url.lastPathComponent)
        }
        let manifest = try PolyhedronManifest(json: JSONValue.parse(Data(contentsOf: file)))
        guard !manifest.presets.isEmpty else {
            throw ManifestError.unsupported("a polyhedron with no camera preset: it has no view to open at or to "
                                            + "bake from")
        }
        return PolyhedronBundle(url: url, manifest: manifest,
                                layers: manifest.layers.map { drawn($0, on: manifest.solid) })
    }

    /// Write the bundle: a directory holding its manifest.
    public func write(to url: URL) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        try manifest.canonicalJSON.write(to: url.appendingPathComponent("manifest.json"),
                                         atomically: true, encoding: .utf8)
    }

    /// The scene the bundle describes under one of its presets.
    public func scene(preset: CameraPreset) -> Scene {
        Scene(solid: solid, layers: layers, camera: .oblique(preset.projection),
              depthTolerance: preset.depthTolerance)
    }
}
