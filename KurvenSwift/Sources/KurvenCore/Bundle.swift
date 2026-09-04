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
            return Mesh.walls(of: p, surface: surface, base: base)
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

        var layers: [Layer] = []
        for spec in manifest.layers {
            let v = try NPY.read(contentsOf: file(spec.vertices))
            let o = try NPY.read(contentsOf: file(spec.offsets))
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

        return KurvenBundle(
            url: url, manifest: manifest,
            surface: Surface(height: height, phase: phase,
                             caps: manifest.caps, cached: cached),
            layers: layers, wallMesh: wallMesh)
    }
}
