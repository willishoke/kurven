import Foundation
import simd

/// An indexed triangle mesh.
///
/// Only the *walls* are ever a mesh. The heightfield is a texture plus a step,
/// rasterized implicitly (see `Occluder`), because an explicit mesh for the
/// elliptic plate is six million triangles and for gamma's bake two hundred
/// million. Wall curtains are a few thousand, so they are values.
public struct Mesh<S>: Sendable, Equatable {
    public let vertices: [P3<S>]
    public let triangles: [SIMD3<Int32>]

    public init(vertices: [P3<S>], triangles: [SIMD3<Int32>]) {
        self.vertices = vertices; self.triangles = triangles
    }

    public static var empty: Mesh<S> { Mesh(vertices: [], triangles: []) }
    public var isEmpty: Bool { triangles.isEmpty }
    public var bounds: AABB<S>? { AABB(vertices) }

    /// Concatenate, offsetting triangle indices into the combined vertex array.
    public static func concat(_ meshes: [Mesh<S>]) -> Mesh<S> {
        var verts: [P3<S>] = []
        var tris: [SIMD3<Int32>] = []
        for m in meshes {
            let offset = Int32(verts.count)
            verts.append(contentsOf: m.vertices)
            tris.append(contentsOf: m.triangles.map { $0 &+ SIMD3(repeating: offset) })
        }
        return Mesh(vertices: verts, triangles: tris)
    }

    public func mapped<T>(_ f: Transform<S, T>) -> Mesh<T> {
        Mesh<T>(vertices: vertices.map { f($0) }, triangles: triangles)
    }
}

public extension Mesh<WorldSpace> {
    /// A vertical ruled strip along a boundary polyline, from `base` up to the
    /// surface -- `occluder.wall_curtain`.
    ///
    /// Derived walls sit at the crest the *grid* says, where the Python plate's
    /// dumped walls sit at the crest the analytic function says. For a bundle
    /// that ships `Walls.mesh` this is unused; for `Walls.perimeter` it is the
    /// definition, and the difference between the two is the grid's
    /// interpolation error, which is a thing to measure rather than assume.
    static func wallCurtain(from a: P2<WorldSpace>, to b: P2<WorldSpace>,
                            samples: Int, surface: Surface, base: Double,
                            tiles: [Affine2] = [.identity]) -> Mesh<WorldSpace> {
        guard samples >= 2 else { return .empty }
        var top: [P3<WorldSpace>] = []
        var bottom: [P3<WorldSpace>] = []
        top.reserveCapacity(samples); bottom.reserveCapacity(samples)
        for i in 0..<samples {
            let t = Double(i) / Double(samples - 1)
            let p = P2<DomainSpace>(a.x * (1 - t) + b.x * t, a.y * (1 - t) + b.y * t)
            top.append(P3(p.x, p.y, surface.height(at: p, tiles: tiles)))
            bottom.append(P3(p.x, p.y, base))
        }
        let n = Int32(samples)
        var tris: [SIMD3<Int32>] = []
        tris.reserveCapacity(2 * (samples - 1))
        for i in 0..<Int32(samples - 1) {
            tris.append(SIMD3(i, i + 1, i + n))
            tris.append(SIMD3(i + 1, i + 1 + n, i + n))
        }
        return Mesh(vertices: top + bottom, triangles: tris)
    }

    /// Every edge of a boundary as a curtain.
    static func walls(of perimeter: BoundaryPerimeter, surface: Surface,
                      base: Double, tiles: [Affine2] = [.identity]) -> Mesh<WorldSpace> {
        concat(perimeter.edges.map {
            wallCurtain(from: $0.start, to: $0.end, samples: $0.density,
                        surface: surface, base: base, tiles: tiles)
        })
    }
}
