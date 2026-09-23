import Foundation
import simd

/// A set of polylines in compressed sparse row form: one flat vertex array plus
/// path offsets.
///
/// Path identity is structural. The Python pipeline tags each vertex with a
/// running integer and recovers paths with `groupby`, which forces ad-hoc offset
/// arithmetic (`BUMP = 1_000_000`, `100 * (i + 2)`) at every place two families
/// of curves are concatenated, purely to stop unrelated runs from welding.
/// Offsets cannot weld.
public struct PolylineSet<S>: Sendable, Equatable {
    /// All vertices of all paths, in order.
    public let vertices: [P3<S>]
    /// `count + 1` offsets; path `i` is `vertices[offsets[i]..<offsets[i + 1]]`.
    public let offsets: [Int]
    /// Where on a surface each vertex lies, parallel to `vertices`, for ink
    /// drawn *on* a parametric surface. `nil` for ink that only has a position.
    ///
    /// A position says where the ink is; a surface coordinate says which piece
    /// of the surface it belongs to, and that is the question hidden-line
    /// removal actually asks. Two sheets a hair apart in depth are far apart
    /// in coordinates. Kept unwrapped along periodic axes, so a path that
    /// crosses a seam does not jump a whole period between two vertices.
    public let coords: [P2<ParamSpace>]?

    public init(vertices: [P3<S>], offsets: [Int], coords: [P2<ParamSpace>]? = nil) {
        precondition(offsets.first == 0 || vertices.isEmpty,
                     "CSR offsets must start at 0")
        precondition(offsets.last ?? 0 == vertices.count,
                     "CSR offsets end at \(offsets.last ?? 0), \(vertices.count) vertices")
        precondition(coords == nil || coords!.count == vertices.count,
                     "\(coords?.count ?? 0) surface coordinates for \(vertices.count) vertices")
        self.vertices = vertices; self.offsets = offsets; self.coords = coords
    }

    public init(paths: [[P3<S>]]) {
        var verts: [P3<S>] = []
        var offs: [Int] = [0]
        for p in paths where p.count >= 2 {
            verts.append(contentsOf: p)
            offs.append(verts.count)
        }
        self.vertices = verts; self.offsets = offs; self.coords = nil
    }

    /// Paths with a surface coordinate for every vertex.
    public init(paths: [[P3<S>]], coords: [[P2<ParamSpace>]]) {
        precondition(paths.count == coords.count, "one coordinate path per path")
        var verts: [P3<S>] = []
        var cs: [P2<ParamSpace>] = []
        var offs: [Int] = [0]
        for (p, c) in zip(paths, coords) where p.count >= 2 {
            precondition(p.count == c.count, "one coordinate per vertex")
            verts.append(contentsOf: p)
            cs.append(contentsOf: c)
            offs.append(verts.count)
        }
        self.vertices = verts; self.offsets = offs; self.coords = cs
    }

    public static var empty: PolylineSet<S> { PolylineSet(vertices: [], offsets: [0]) }

    public var count: Int { max(offsets.count - 1, 0) }
    public var isEmpty: Bool { count == 0 }

    public subscript(path i: Int) -> ArraySlice<P3<S>> {
        vertices[offsets[i]..<offsets[i + 1]]
    }

    public var paths: [ArraySlice<P3<S>>] { (0..<count).map { self[path: $0] } }

    /// Map every vertex through a transform, keeping the path structure.
    public func mapped<T>(_ f: Transform<S, T>) -> PolylineSet<T> {
        PolylineSet<T>(vertices: vertices.map { f($0) }, offsets: offsets, coords: coords)
    }

    /// The surface coordinates of path `i`, when the set has them.
    public func coords(path i: Int) -> ArraySlice<P2<ParamSpace>>? {
        coords.map { $0[offsets[i]..<offsets[i + 1]] }
    }

    public var bounds: AABB<S>? { AABB(vertices) }

    /// Total 2D length, ignoring z -- the "ink length" an end-to-end
    /// equivalence check compares between two renderings.
    public var inkLength: Double {
        var total = 0.0
        for i in 0..<count {
            let p = self[path: i]
            for (a, b) in zip(p, p.dropFirst()) {
                total += simd_length(SIMD2(b.x - a.x, b.y - a.y))
            }
        }
        return total
    }
}

/// A drawn layer: geometry plus how it is drawn.
public struct Layer: Sendable, Equatable {
    public let spec: LayerSpec
    public let paths: PolylineSet<WorldSpace>

    public init(spec: LayerSpec, paths: PolylineSet<WorldSpace>) {
        self.spec = spec; self.paths = paths
    }
}
