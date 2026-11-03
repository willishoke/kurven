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
    /// Derived walls sit at the top edge the *grid* says, where the Python
    /// plate's dumped walls sit at the top edge the analytic function says.
    /// For a bundle that ships `Walls.mesh` this is unused; for
    /// `Walls.perimeter` it is the definition, and the difference between the
    /// two is the grid's interpolation error, which is a thing to measure
    /// rather than assume.
    ///
    /// The curtain stands on the footprint's side of its edge: `inward` is
    /// the edge's normal into the footprint, and where the edge stands on a
    /// band boundary, or ends on one, its top is the band's on that side
    /// (`Surface.wallHeight`), not the band the boundary starts. Where the
    /// edge crosses a tile boundary between two samples and the copies' tops
    /// there differ, it has two more there, one read through the tile on
    /// each side (`Surface.tileBoundaryCrossings`), so its top steps down the
    /// riser there rather than ramping across.
    static func wallCurtain(from a: P2<WorldSpace>, to b: P2<WorldSpace>, inward: SIMD2<Double>,
                            samples: Int, surface: Surface, base: Double,
                            tiles: [Affine2] = [.identity]) -> Mesh<WorldSpace> {
        guard samples >= 2 else { return .empty }
        var top: [P3<WorldSpace>] = []
        var bottom: [P3<WorldSpace>] = []
        top.reserveCapacity(samples); bottom.reserveCapacity(samples)
        let edge = PerimeterEdge(start: a, end: b, density: samples)
        func station(_ t: Double, through back: Affine2? = nil) {
            let p = Surface.point(from: a, to: b, at: t)
            top.append(P3(p.x, p.y, surface.wallHeight(at: p, t: t, from: a, to: b, inward: inward, tiles: tiles,
                                                       through: back)))
            bottom.append(P3(p.x, p.y, base))
        }
        for i in 0..<samples {
            let t = Double(i) / Double(samples - 1)
            if i > 0 {
                let previous = Double(i - 1) / Double(samples - 1)
                for crossing in surface.tileBoundaryCrossings(of: edge, from: previous, to: t, inward: inward, tiles: tiles) {
                    let p = Surface.point(from: a, to: b, at: crossing.t)
                    func top(_ back: Affine2) -> Double {
                        surface.wallHeight(at: p, t: crossing.t, from: a, to: b, inward: inward, tiles: tiles, through: back)
                    }
                    let (near, far) = (top(crossing.near), top(crossing.far))
                    guard Surface.stands(near, over: far) || Surface.stands(far, over: near) else { continue }
                    if crossing.t != previous { station(crossing.t, through: crossing.near) }
                    if crossing.t != t { station(crossing.t, through: crossing.far) }
                }
            }
            station(t)
        }
        let samples = top.count
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
        concat(perimeter.edges.indices.map { k in
            let edge = perimeter.edges[k]
            return wallCurtain(from: edge.start, to: edge.end, inward: perimeter.inward(k),
                               samples: edge.density, surface: surface, base: base, tiles: tiles)
        })
    }

    /// The risers at the tile boundaries where two copies meet with different
    /// tops (`Surface.tileBoundaryRisers(tiles:region:refine:)`), as the
    /// depth pass draws the copies either side: each copy's top along the
    /// tile boundary the height its rasterized grid has there
    /// (`Surface.drawnHeight(at:step:)`), every `step`-th sample, and between
    /// the two a vertical strip, a pair of triangles between each two stops
    /// where the riser stands (`Surface.tileBoundaryStretches`). The
    /// heightfield's instances end at the tile boundary, each at its own top,
    /// and without the strip the drop between them is a gap the depth pass
    /// sees through.
    static func tileBoundaryRisers(of surface: Surface, step: Int, region: Region, tiles: [Affine2]) -> Mesh<WorldSpace> {
        var vertices: [P3<WorldSpace>] = []
        var triangles: [SIMD3<Int32>] = []
        for boundary in surface.tileBoundaries(tiles: tiles) {
            guard let first = tiles[boundary.first].inverse, let second = tiles[boundary.second].inverse else { continue }
            let backs = [first, second]
            func tops(_ w: P2<WorldSpace>) -> [Double] {
                backs.map { surface.drawnHeight(at: surface.ontoDomain($0(P3(w.x, w.y, 0)).xy), step: step) }
            }
            let stretches = surface.tileBoundaryStretches(boundary, backs: backs, region: region, step: step, tops: tops) { q in
                let drawn = surface.drawnSurface(at: q, step: step)
                return drawn.interpolated - drawn.truncationHeight
            }
            for stretch in stretches {
                for (a, c) in zip(stretch, stretch.dropFirst()) {
                    let base = Int32(vertices.count)
                    for stop in [a, c] {
                        let w = boundary.point(stop.s), z = tops(stop.read)
                        vertices.append(P3(w.x, w.y, z.min() ?? .nan))
                        vertices.append(P3(w.x, w.y, z.max() ?? .nan))
                    }
                    triangles.append(SIMD3(base, base + 2, base + 1))
                    triangles.append(SIMD3(base + 1, base + 2, base + 3))
                }
            }
        }
        return Mesh(vertices: vertices, triangles: triangles)
    }

    /// The truncated surface where a grid cell straddles the truncation: the
    /// pieces the heightfield pass gets wrong.
    ///
    /// The heightfield is rasterized from heights clamped *at the grid
    /// vertices* and interpolated between them, so a cell with one corner
    /// below the truncation and another over it is drawn as one slope from the truncation
    /// down to the low corner. The corner where the wall meets the truncation is cut
    /// off, by up to the cell's whole rise -- half a world unit on a pole's
    /// graph -- and the curves sit where the *interpolated* height reaches the
    /// truncation, out over the notch: the truncation edge, the hatch across the truncation, a phase
    /// line carried up to it. Curves from behind show through the notches, and
    /// curves on the outline are tested against a wall that is not there.
    ///
    /// What a cell's samples describe is min(interpolated height, truncation): each
    /// of the cell's two triangles cut by the truncation along the line where the
    /// interpolated excess is zero, which is the line marching squares draws
    /// the truncation edge on. This mesh is those pieces, for the cells that need them and
    /// no others. It is drawn with the heightfield under the same MAX blend,
    /// and since the clamped-at-vertices triangle lies under the true pieces
    /// everywhere (min is concave), the union is the pieces exactly: nothing
    /// has to be skipped in the heightfield pass and nothing is drawn wrong
    /// twice.
    ///
    /// The grid is the rasterizer's: every `step`-th sample, a cell kept
    /// only when all four corners lie in the region, once per tile. A banded
    /// truncation is interpolated across a cell the way the pre-clamped grid is.
    static func truncationEdgeMesh(of surface: Surface, step: Int, region: Region,
                                   tiles: [Affine2]) -> Mesh<WorldSpace> {
        if case .none = surface.truncation { return .empty }
        let g = surface.height
        let step = max(step, 1)
        let nx = (g.width + step - 1) / step, ny = (g.height + step - 1) / step
        guard nx >= 2, ny >= 2 else { return .empty }
        let xs = (0..<nx).map { g.position(x: $0 * step, y: 0).x }
        let ys = (0..<ny).map { g.position(x: 0, y: $0 * step).y }
        let colTruncation = xs.map { surface.truncationHeight(atX: $0) }

        func corner(_ i: Int, _ j: Int) -> TruncationCorner {
            TruncationCorner(x: xs[i], y: ys[j],
                             h: Double(g[min(i * step, g.width - 1), min(j * step, g.height - 1)]),
                             truncationHeight: colTruncation[i])
        }

        /// The pieces of one triangle that the truncation plane cuts it into; none
        /// when it lies wholly on one side, where the rasterizer is right.
        func pieces(_ t: [TruncationCorner]) -> [[P3<WorldSpace>]] {
            let over = t.filter { $0.excess > 0 }.count
            let under = t.filter { $0.excess < 0 }.count
            guard over > 0, under > 0 else { return [] }
            // The lone corner is the one on its own side of the truncation; a corner
            // exactly on the plateau joins whichever side leaves one alone.
            let lone = over == 1 ? t.firstIndex { $0.excess > 0 }! : t.firstIndex { $0.excess < 0 }!
            let A = t[lone], B = t[(lone + 1) % 3], C = t[(lone + 2) % 3]
            let P = A.crossing(to: B), Q = A.crossing(to: C)
            // On the flat side a point sits on the plateau; on the sloped side on
            // the interpolated height. At P and Q the two agree.
            func flat(_ k: TruncationCorner) -> P3<WorldSpace> { P3(k.x, k.y, k.truncationHeight) }
            func sloped(_ k: TruncationCorner) -> P3<WorldSpace> { P3(k.x, k.y, k.h) }
            let aSide = A.excess > 0 ? flat : sloped
            let otherSide = A.excess > 0 ? sloped : flat
            return [[aSide(A), aSide(P), aSide(Q)],
                    [otherSide(P), otherSide(B), otherSide(C)],
                    [otherSide(P), otherSide(C), otherSide(Q)]]
        }

        // Straddling cells are found once; they lie only along the truncation edges. Each
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

/// A grid corner of a cell on a truncation edge: where it is, how high the sample
/// is, and the truncation there. Its excess over the truncation is what `Mesh.truncationEdgeMesh`
/// solves the cut on.
private struct TruncationCorner {
    var x: Double, y: Double, h: Double, truncationHeight: Double
    var excess: Double { h - truncationHeight }
    /// Along the edge to `other`, the point where the excess is zero.
    func crossing(to other: TruncationCorner) -> TruncationCorner {
        let t = excess / (excess - other.excess)
        return TruncationCorner(x: x + (other.x - x) * t, y: y + (other.y - y) * t,
                                h: h + (other.h - h) * t,
                                truncationHeight: truncationHeight + (other.truncationHeight - truncationHeight) * t)
    }
}
