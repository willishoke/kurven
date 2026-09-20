import Foundation
import simd

/// Reading a `.kurven` bundle into values.
public enum BundleError: Error, CustomStringConvertible {
    case missingFile(String, in: String)
    case shapeMismatch(String, expected: [Int], found: [Int])
    case badOffsets(String, last: Int, vertices: Int)
    case missingWalls(String)

    public var description: String {
        switch self {
        case .missingFile(let f, let b): "bundle \(b): no file \(f)"
        case .shapeMismatch(let f, let e, let g):
            "bundle: \(f) has shape \(g), manifest says \(e)"
        case .badOffsets(let n, let l, let v):
            "bundle: layer \(n) offsets end at \(l) but there are \(v) vertices"
        case .missingWalls(let f): "bundle: manifest names a wall mesh at \(f), which is absent"
        }
    }
}

/// A decoded bundle: the manifest, the grids, and the ink.
///
/// Arrays are eager. A bundle is a value the whole point of which is that it is
/// already computed; lazily faulting parts of it in would trade the one property
/// it has for nothing.
public struct KurvenBundle: Sendable {
    public let url: URL
    public let manifest: Manifest
    public let surface: Surface
    public let layers: [Layer]
    /// Present when the manifest carries `Walls.mesh`; nil when the walls are
    /// derived from a perimeter instead.
    public let wallMesh: Mesh<WorldSpace>?

    public init(url: URL, manifest: Manifest, surface: Surface, layers: [Layer],
                wallMesh: Mesh<WorldSpace>?) {
        self.url = url; self.manifest = manifest; self.surface = surface
        self.layers = layers; self.wallMesh = wallMesh
    }

    /// A bundle whose every layer is described: the ink is derived from the
    /// grids, as `read` derives it for a `--derived` bundle on disk.
    public init(url: URL, manifest: Manifest, surface: Surface) {
        self.init(url: url, manifest: manifest, surface: surface,
                  layers: manifest.layers.map {
                      Layer(spec: $0, paths: surface.ink($0, occluder: manifest.occluder))
                  },
                  wallMesh: nil)
    }

    public func layer(_ name: String) throws -> Layer {
        guard let l = layers.first(where: { $0.spec.name == name }) else {
            throw ManifestError.noSuchLayer(name, have: layers.map(\.spec.name))
        }
        return l
    }

    /// The occluding walls, whether dumped or described.
    public func walls() -> Mesh<WorldSpace> {
        switch manifest.occluder.walls {
        case .none: return .empty
        case .mesh: return wallMesh ?? .empty
        case .perimeter(let p, let base):
            return Mesh.walls(of: p, surface: surface, base: base,
                              tiles: manifest.occluder.tiles)
        }
    }

    public static func read(at url: URL) throws -> KurvenBundle {
        let name = url.lastPathComponent
        func file(_ rel: String) throws -> URL {
            let u = url.appendingPathComponent(rel)
            guard FileManager.default.fileExists(atPath: u.path) else {
                throw BundleError.missingFile(rel, in: name)
            }
            return u
        }

        let manifest = try Manifest.read(contentsOf: file("manifest.json"))

        func grid(_ ref: GridRef) throws -> Grid2D<Float> {
            let a = try NPY.read(contentsOf: file(ref.file))
            guard a.shape == [ref.shape.ny, ref.shape.nx] else {
                throw BundleError.shapeMismatch(ref.file,
                                                expected: [ref.shape.ny, ref.shape.nx],
                                                found: a.shape)
            }
            return Grid2D(width: ref.shape.nx, height: ref.shape.ny,
                          domain: manifest.domain, values: try a.floats())
        }

        let height = try grid(manifest.height)
        let phase = try manifest.phase.map(grid)

        // A bundle written from a cached grid has no evaluator behind it, so its
        // heights were looked up by nearest pixel. `provenance.function` naming
        // a cache is the only signal of that in the file; treat a bundle whose
        // params mention a cache as cached.
        let cached = manifest.provenance.params["cache"] != nil

        let surface = Surface(height: height, phase: phase,
                              caps: manifest.caps, cached: cached)

        var layers: [Layer] = []
        for spec in manifest.layers {
            guard let files = spec.files else {
                // Described, not dumped: derive the ink from the grids the
                // bundle already carries. This is what a `--derived` bundle
                // trades its layer files for, and what makes the levels, the
                // hatch spacing and the cap editable.
                layers.append(Layer(spec: spec,
                                    paths: surface.ink(spec, occluder: manifest.occluder)))
                continue
            }
            let v = try NPY.read(contentsOf: file(files.vertices))
            let o = try NPY.read(contentsOf: file(files.offsets))
            let verts = try v.rows3().map { P3<WorldSpace>($0) }
            let offsets = try o.ints()
            guard offsets.last ?? 0 == verts.count else {
                throw BundleError.badOffsets(spec.name, last: offsets.last ?? 0,
                                             vertices: verts.count)
            }
            layers.append(Layer(spec: spec,
                                paths: PolylineSet(vertices: verts, offsets: offsets)))
        }

        var wallMesh: Mesh<WorldSpace>?
        if case .mesh(let vf, let tf) = manifest.occluder.walls {
            let v = try NPY.read(contentsOf: file(vf))
            let t = try NPY.read(contentsOf: file(tf))
            wallMesh = Mesh(vertices: try v.rows3().map { P3<WorldSpace>($0) },
                            triangles: try t.rows3i())
        }

        return KurvenBundle(url: url, manifest: manifest, surface: surface,
                            layers: layers, wallMesh: wallMesh)
    }
}

public extension KurvenBundle {
    /// Re-derive one described layer at different levels.
    ///
    /// Only meaningful for `LayerSource.contour`: a dumped layer is an answer
    /// with no question behind it, and this returns it unchanged. That
    /// asymmetry is the point of describing layers at all -- a bundle exported
    /// with `--derived` has editable level sets and one exported without does
    /// not, and the type says which you have.
    func layer(_ spec: LayerSpec, levels: [Double]) -> Layer {
        guard case .contour(let field, _, let keep, let tiled) = spec.source else {
            return redrawn(spec)
        }
        var edited = spec
        edited.source = .contour(field: field, levels: levels, keep: keep, tiled: tiled)
        return redrawn(edited)
    }

    /// One layer, drawn from whatever its spec now says.
    ///
    /// A described layer is derived again; a dumped one keeps the ink it was
    /// read with, because the bundle carries no question that could produce it
    /// a second time.
    func redrawn(_ spec: LayerSpec) -> Layer {
        guard spec.files == nil else {
            let existing = layers.first { $0.spec.name == spec.name }
            return Layer(spec: spec, paths: existing?.paths ?? .empty)
        }
        return Layer(spec: spec, paths: surface.ink(spec, occluder: manifest.occluder))
    }

    /// The same grids under a different manifest.
    ///
    /// The landscape does not change -- the samples are the samples -- but the
    /// cap, the levels, the hatch spacing and the walls all can, and all of them
    /// are derived from the manifest. This is the edit path for every one of
    /// them, and the reason a cap slider costs milliseconds instead of a round
    /// trip to Python: only the *ink* and the wall curtains are rebuilt, from
    /// grids that never left memory.
    ///
    /// It is a restyling, not a resampling: passing a manifest whose grids,
    /// domain or shape differ from this bundle's would describe a landscape
    /// these arrays are not, so those fields are taken from this bundle and the
    /// rest from the argument.
    func restyled(_ manifest: Manifest) -> KurvenBundle {
        var honest = manifest
        honest.domain = self.manifest.domain
        honest.height = self.manifest.height
        honest.phase = self.manifest.phase
        let restyledSurface = Surface(height: surface.height, phase: surface.phase,
                                      caps: honest.caps, cached: surface.cached)
        let bundle = KurvenBundle(url: url, manifest: honest, surface: restyledSurface,
                                  layers: [], wallMesh: wallMesh)
        return KurvenBundle(url: url, manifest: honest, surface: restyledSurface,
                            layers: honest.layers.map { spec in
                                spec.files == nil ? bundle.redrawn(spec)
                                                  : redrawn(spec)
                            },
                            wallMesh: wallMesh)
    }

    /// The levels a described layer was exported with, or nil when it is dumped.
    static func levels(of spec: LayerSpec) -> [Double]? {
        if case .contour(_, let levels, _, _) = spec.source { return levels }
        return nil
    }
}

public extension KurvenBundle {
    /// Write a described bundle: the manifest and the two grids.
    ///
    /// Every layer of a landscape is a description, so this is the whole of
    /// it. A bundle carrying dumped ink would need its layer files written too,
    /// and a manifest naming files that are not there is worse than refusing.
    func write(to url: URL) throws {
        guard manifest.layers.allSatisfy({ $0.files == nil }) else {
            throw BundleError.missingFile("layers/*.npy (this bundle has dumped layers)",
                                          in: url.lastPathComponent)
        }
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        let height = surface.height
        try NPY.write(height.values, shape: [height.height, height.width],
                      to: url.appendingPathComponent("height.npy"))
        if let phase = surface.phase {
            try NPY.write(phase.values, shape: [phase.height, phase.width],
                          to: url.appendingPathComponent("phase.npy"))
        }
        var written = manifest
        written.height = GridRef(file: "height.npy",
                                 shape: (ny: height.height, nx: height.width), dtype: .float32)
        if let phase = surface.phase {
            written.phase = GridRef(file: "phase.npy",
                                    shape: (ny: phase.height, nx: phase.width), dtype: .float32)
        }
        // The manifest goes last, so a bundle whose manifest exists is a
        // bundle whose arrays are complete.
        try written.canonicalJSON.write(to: url.appendingPathComponent("manifest.json"),
                                        atomically: true, encoding: .utf8)
    }
}
