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

    /// The truncated surface where a lattice cell straddles the cap: the
    /// pieces the heightfield pass gets wrong.
    ///
    /// The heightfield is rasterized from heights clamped *at the lattice
    /// vertices* and interpolated between them, so a cell with one corner
    /// under the cap and another over it is drawn as one slope from the cap
    /// down to the low corner. The corner where the wall meets the cap is cut
    /// off, by up to the cell's whole rise -- half a world unit on a pole's
    /// flank -- and the ink sits where the *interpolated* height reaches the
    /// cap, out over the notch: the rim, the hatch across the cap, a phase
    /// line carried up to it. Ink from behind shows through the notches, and
    /// ink on the silhouette is judged against a wall that is not there.
    ///
    /// What a cell's samples describe is min(interpolated height, cap): each
    /// of the cell's two triangles cut by the cap along the line where the
    /// interpolated excess is zero, which is the line marching squares draws
    /// the rim on. This mesh is those pieces, for the cells that need them and
    /// no others. It is drawn with the heightfield under the same MAX blend,
    /// and since the clamped-at-vertices triangle lies under the true pieces
    /// everywhere (min is concave), the union is the pieces exactly: nothing
    /// has to be skipped in the heightfield pass and nothing is drawn wrong
    /// twice.
    ///
    /// The lattice is the rasterizer's: every `step`-th sample, a cell kept
    /// only when all four corners lie in the region, once per tile. A banded
    /// cap is interpolated across a cell the way the pre-clamped grid is.
    static func capRim(of surface: Surface, step: Int, region: Region,
                       tiles: [Affine2]) -> Mesh<WorldSpace> {
        if case .none = surface.caps { return .empty }
        let g = surface.height
        let step = max(step, 1)
        let nx = (g.width + step - 1) / step, ny = (g.height + step - 1) / step
        guard nx >= 2, ny >= 2 else { return .empty }
        let xs = (0..<nx).map { g.position(x: $0 * step, y: 0).x }
        let ys = (0..<ny).map { g.position(x: 0, y: $0 * step).y }
        let colCap = xs.map { surface.caps.height(atX: $0) }

        func corner(_ i: Int, _ j: Int) -> RimCorner {
            RimCorner(x: xs[i], y: ys[j],
                      h: Double(g[min(i * step, g.width - 1), min(j * step, g.height - 1)]),
                      cap: colCap[i])
        }

        /// The pieces of one triangle that the cap plane cuts it into; none
        /// when it lies wholly on one side, where the rasterizer is right.
        func pieces(_ t: [RimCorner]) -> [[P3<WorldSpace>]] {
            let over = t.filter { $0.excess > 0 }.count
            let under = t.filter { $0.excess < 0 }.count
            guard over > 0, under > 0 else { return [] }
            // The lone corner is the one on its own side of the cap; a corner
            // exactly on the cap joins whichever side leaves one alone.
            let lone = over == 1 ? t.firstIndex { $0.excess > 0 }! : t.firstIndex { $0.excess < 0 }!
            let A = t[lone], B = t[(lone + 1) % 3], C = t[(lone + 2) % 3]
            let P = A.crossing(to: B), Q = A.crossing(to: C)
            // On the flat side a point sits on the cap; on the sloped side on
            // the interpolated height. At P and Q the two agree.
            func flat(_ k: RimCorner) -> P3<WorldSpace> { P3(k.x, k.y, k.cap) }
            func sloped(_ k: RimCorner) -> P3<WorldSpace> { P3(k.x, k.y, k.h) }
            let aSide = A.excess > 0 ? flat : sloped
            let otherSide = A.excess > 0 ? sloped : flat
            return [[aSide(A), aSide(P), aSide(Q)],
                    [otherSide(P), otherSide(B), otherSide(C)],
                    [otherSide(P), otherSide(C), otherSide(Q)]]
        }

        // Straddling cells are found once; they lie only along the rims. Each
        // keeps its corners, for the region rule below.
        var cells: [(corners: [P2<WorldSpace>], pieces: [[P3<WorldSpace>]])] = []
        for j in 0..<(ny - 1) {
            for i in 0..<(nx - 1) {
                let c00 = corner(i, j), c10 = corner(i + 1, j)
                let c01 = corner(i, j + 1), c11 = corner(i + 1, j + 1)
                let corners = [c00, c10, c01, c11]
                guard corners.contains(where: { $0.excess > 0 }),
                      corners.contains(where: { $0.excess < 0 }) else { continue }
                // The rasterizer's two triangles: (00, 10, 01) and (10, 11, 01).
                let found = pieces([c00, c10, c01]) + pieces([c10, c11, c01])
                guard !found.isEmpty else { continue }
                cells.append((corners.map { P2($0.x, $0.y) }, found))
            }
        }
        guard !cells.isEmpty else { return .empty }

        var vertices: [P3<WorldSpace>] = []
        var triangles: [SIMD3<Int32>] = []
        for tile in tiles {
            for cell in cells {
                // The rule the heightfield pass culls by: all four corners in.
                guard cell.corners.allSatisfy({ region.contains(tile(P3($0.x, $0.y, 0)).xy) })
                else { continue }
                for piece in cell.pieces {
                    let base = Int32(vertices.count)
                    vertices.append(contentsOf: piece.map { tile($0) })
                    triangles.append(SIMD3(base, base + 1, base + 2))
                }
            }
        }
        return Mesh(vertices: vertices, triangles: triangles)
    }
}

/// A lattice corner of a cell on a cap rim: where it is, how high the sample
/// is, and the cap there. Its excess over the cap is what `Mesh.capRim`
/// solves the cut on.
private struct RimCorner {
    var x: Double, y: Double, h: Double, cap: Double
    var excess: Double { h - cap }
    /// Along the edge to `other`, the point where the excess is zero.
    func crossing(to other: RimCorner) -> RimCorner {
        let t = excess / (excess - other.excess)
        return RimCorner(x: x + (other.x - x) * t, y: y + (other.y - y) * t,
                         h: h + (other.h - h) * t, cap: cap + (other.cap - cap) * t)
    }
}
