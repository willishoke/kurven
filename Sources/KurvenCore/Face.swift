import Foundation
import simd
import Synchronization

/// Which part of a solid's boundary a face is, in the order a landscape
/// lists its faces (`Surface.faces(tiles:region:walls:refine:)`).
public enum FaceID: Sendable, Hashable, Comparable {
    /// A landscape's graph, where z = |f|.
    case graph
    /// The plateau of one band of the truncation, numbered as
    /// `Truncation.band(atX:)` numbers them; a uniform truncation's one
    /// plateau is band 0.
    case plateau(band: Int)
    /// The riser where a banded truncation steps, at the end of band `band`
    /// toward greater Re z.
    case riser(band: Int)
    /// The wall along one edge of the walls' perimeter.
    case wall(edge: Int)
    /// A parametric surface's one face.
    case surface
    /// A polyhedron's face, as `Solid.loops` lists them.
    case solid(face: Int)
}

/// A face in the boundary-representation sense (Mäntylä 1988): a region of
/// one smooth surface, flat or curved, bounded by loops. A solid's faces
/// together are its manifold, the surface it is bounded by, in boundary
/// representation its shell. That is closed for a polyhedron, a torus, or a
/// landscape standing on walls given as a perimeter; a landscape's faces
/// leave its sides open where it has no walls, or walls given as a mesh,
/// which describe no face.
///
/// A face is a map from a parameter rectangle into space, trimmed: the part
/// of the parameter plane it covers is its *trimmed domain*, bounded by its
/// trim loops (`trimLoops`), and each copy's `contains` is the domain's point
/// membership classification (Requicha 1980). A landscape's faces are its
/// graph, a plateau per band of the truncation, a riser where the truncation
/// steps between two bands, and a wall along each edge of the walls' perimeter
/// (`Surface.faces(tiles:region:walls:refine:)`); a parametric surface is
/// one face (`ParametricSurface.face`); a polyhedron's faces are its loops,
/// flat (`Solid.face(_:)`). `SceneGeometry.faces` lists a scene's.
///
/// A face is a view onto the grids and the solids it is read from, and
/// costs nothing to make: it holds closures, not samples.
public struct Face: Sendable {
    public let id: FaceID
    /// The parameter rectangle the map is defined on, and where a field on
    /// the face is sampled to be contoured. It covers the trimmed domain,
    /// except up a wall or a riser with no finite truncation over it: there
    /// it stops at the grid's highest sample, and f can rise over its
    /// samples.
    public let domain: (u: ParamAxis, v: ParamAxis)
    /// The point of space at a point of the parameter plane, with its two
    /// partial derivatives: the face where it is first, before a copy moves
    /// it (`Copy.place`).
    public let map: @Sendable (P2<ParamSpace>) -> SurfaceJet
    /// Where the face is in space. One copy, on the identity, but for a tiled
    /// landscape's graph, plateaus and risers, which have one per tile, as
    /// the curves on them do (`Occluder.tiles`): contoured once, then
    /// replicated.
    public let copies: [Copy]
    private let loops: Computed<[[P2<ParamSpace>]]>?

    /// The closed curves in the parameter plane that bound the face's
    /// trimmed domain even-odd, before a copy is cut to the footprint, each
    /// ending where it begins: where a contour on the face is cut, and which
    /// side of each is kept (`contours(of:_:on:)`). A polyhedron face's are
    /// its loops. The graph's are the domain's edge and the plateaus' loops,
    /// the truncation edge among them (`Surface.graphTrimLoops(tiles:refine:)`),
    /// placed on f with a refiner, the first time they are asked, and shared
    /// by every copy.
    ///
    /// The graph's bound it only even-odd: a stretch along the domain's edge
    /// or a band boundary can be passed twice, bounding nothing, and the
    /// crease between a riser and the higher plateau lies on them though the
    /// graph does not reach it. They are not one oriented loop per component
    /// of the boundary, as a B-rep's are; and a copy's set
    /// (`Copy.trimLoops`) bounds nothing on its own, for it also needs the
    /// footprint, which `contains` tests.
    ///
    /// Nil where the face has none to give: a parametric surface, whose
    /// domain is its whole rectangle, and, not built yet, a plateau, a riser
    /// and a wall.
    public var trimLoops: [[P2<ParamSpace>]]? { loops?.value }

    /// The contours of a field on one copy of the face, clipped to its
    /// trimmed domain: cut where they cross the face's trim loops or the
    /// footprint's perimeter where the copy stands, kept where the loops
    /// hold them even-odd and the footprint holds them
    /// (`Contour.levels(of:_:over:trimLoops:contains:cuts:)`). On a face with
    /// no trim loops they are kept where the copy's `contains` holds, a
    /// segment at a time.
    public func contours(of field: ScalarField, _ levels: [Double], on copy: Copy)
        -> [(level: Double, paths: [[P2<ParamSpace>]])] {
        guard let loops = trimLoops else {
            return Contour.levels(of: field, levels, over: domain, contains: copy.contains)
        }
        return Contour.levels(of: field, levels, over: domain, trimLoops: loops,
                              contains: copy.footprint?.contains, cuts: copy.footprint?.perimeter ?? [])
    }

    /// One copy of a face: where it is, which way it faces, and how much of
    /// it is there.
    public struct Copy: Sendable {
        /// The map from the face's own place to this copy's.
        public let tile: Affine2
        /// `+1` when `du × dv`, carried into place (`place`), points out of
        /// the solid the face bounds, `-1` when it points in; nil when the
        /// face bounds no solid. The face's own orientation turned by the
        /// tile: a reflected copy's is the other way.
        public let outward: Double?
        /// Whether a point of the parameter plane lies in this copy's
        /// trimmed domain. Answered anywhere in the plane, not only on the
        /// rectangle: the face's own trim is part of it (a landscape face
        /// ends at the domain's edge, a wall at its edge's ends), and so is
        /// the footprint, tested at the copy's point in space, as the mesh
        /// and visibility test it. On a periodic axis it answers the same a
        /// period along.
        public let contains: @Sendable (P2<ParamSpace>) -> Bool
        fileprivate let loops: Computed<[[P2<ParamSpace>]]>?
        /// The footprint where this copy stands: its perimeter carried back
        /// into the parameter plane, and whether it holds a point there.
        fileprivate let footprint: (perimeter: [[P2<ParamSpace>]], contains: @Sendable (P2<ParamSpace>) -> Bool)?

        /// The closed curves this copy's trimmed domain ends on: the face's
        /// trim loops and, where a footprint cuts the copy, the footprint's
        /// perimeter where the copy stands, carried back into the parameter
        /// plane. Not intersected: with a footprint they bound nothing
        /// even-odd, and which side of the perimeter is in the copy is the
        /// footprint's to say (`Face.contours(of:_:on:)`). Nil where the
        /// face's are.
        public var trimLoops: [[P2<ParamSpace>]]? { loops.map { $0.value + (footprint?.perimeter ?? []) } }

        /// A jet of the face's map carried into place: its point moved by the
        /// tile, its derivatives turned with it, heights kept.
        public func place(_ jet: SurfaceJet) -> SurfaceJet {
            func turned(_ d: SIMD3<Double>) -> SIMD3<Double> {
                SIMD3(tile.a * d.x + tile.b * d.y, tile.c * d.x + tile.d * d.y, d.z)
            }
            return SurfaceJet(position: tile(P3<WorldSpace>(jet.position)).v,
                              du: turned(jet.du), dv: turned(jet.dv))
        }
    }

    /// A face in one place, on the identity.
    public init(id: FaceID, domain: (u: ParamAxis, v: ParamAxis),
                contains: @escaping @Sendable (P2<ParamSpace>) -> Bool,
                trimLoops: (@Sendable () -> [[P2<ParamSpace>]])? = nil,
                map: @escaping @Sendable (P2<ParamSpace>) -> SurfaceJet, outward: Double?) {
        self.init(id: id, domain: domain, map: map, outward: outward, tiles: [.identity],
                  trimLoops: trimLoops) { p, _ in contains(p) }
    }

    /// A face copied by `tiles`: each copy faces as `outward` says, turned by
    /// its tile, and holds what `contains` says asked with its tile. Its
    /// trim loops are what `trimLoops` makes, once, when they are first
    /// asked, and its footprint `footprint`'s asked with its tile, nil where
    /// none cuts it.
    public init(id: FaceID, domain: (u: ParamAxis, v: ParamAxis),
                map: @escaping @Sendable (P2<ParamSpace>) -> SurfaceJet, outward: Double?,
                tiles: [Affine2], trimLoops: (@Sendable () -> [[P2<ParamSpace>]])? = nil,
                footprint: (@Sendable (Affine2) -> (perimeter: [[P2<ParamSpace>]],
                                                     contains: @Sendable (P2<ParamSpace>) -> Bool)?)? = nil,
                contains: @escaping @Sendable (P2<ParamSpace>, Affine2) -> Bool) {
        let loops = trimLoops.map(Computed.init)
        self.id = id; self.domain = domain; self.map = map; self.loops = loops
        copies = tiles.map { tile in
            let turn: Double = tile.a * tile.d - tile.b * tile.c < 0 ? -1 : 1
            return Copy(tile: tile, outward: outward.map { $0 * turn }, contains: { contains($0, tile) },
                        loops: loops, footprint: footprint?(tile))
        }
    }
}

/// A value made the first time it is asked, and kept: shared by whatever
/// holds it, as a face's copies share its trim loops.
final class Computed<Value: Sendable>: Sendable {
    private let make: @Sendable () -> Value
    private let made = Mutex<Value?>(nil)

    init(_ make: @escaping @Sendable () -> Value) { self.make = make }

    /// The lock is held while it is made, so it is made once.
    var value: Value {
        made.withLock { made in
            if let made { return made }
            let value = make()
            made = value
            return value
        }
    }
}

extension ParamAxis {
    /// Whether a coordinate lies on the axis: anywhere on a periodic one,
    /// within the range, ends included, on a bounded one.
    func covers(_ x: Double) -> Bool {
        periodic || (min(range.lo, range.hi)...max(range.lo, range.hi)).contains(x)
    }
}

public extension SceneGeometry {
    /// The faces of the solid the scene draws.
    var faces: [Face] {
        switch self {
        case .heightfield(let h): h.faces
        case .parametric(let s): [s.face]
        case .solid(let s): s.faces.indices.map(s.face)
        }
    }
}

// MARK: - a parametric surface

public extension ParametricSurface {
    /// The surface as one face: its whole parameter rectangle, glued where
    /// an axis is periodic.
    var face: Face {
        let (u, v) = (self.u, self.v)
        return Face(id: .surface, domain: (u, v), contains: { u.covers($0.x) && v.covers($0.y) },
                    map: map, outward: outward)
    }
}

// MARK: - a polyhedron

public extension Solid {
    /// Face `i` as a `Face`: flat, its parameter rectangle the bounding box
    /// of its loops in its frame (`FaceFrame`), the map the frame's, and its
    /// trimmed domain what the loops bound, even-odd, so a hole is out. Its
    /// trim loops are its loops.
    ///
    /// Membership is read off the loops, not the convex pieces: the pieces
    /// are how visibility and hatching cut the face, and the face is what
    /// they cut. A flat face's fields are linear in its frame, so the
    /// rectangle is sampled at its corners.
    func face(_ i: Int) -> Face {
        let frame = frames[i]
        let loops = self.loops[i].map { $0.map { frame.coordinates(vertices[$0].v) } }
        let corners = Array(loops.joined())
        let lo = corners.reduce(corners[0], simd_min), hi = corners.reduce(corners[0], simd_max)
        let (u, v) = (frame.u, frame.v)
        return Face(
            id: .solid(face: i),
            domain: (ParamAxis(Interval(lo: lo.x, hi: hi.x), periodic: false, samples: 2),
                     ParamAxis(Interval(lo: lo.y, hi: hi.y), periodic: false, samples: 2)),
            contains: { p in
                // A ray from p along +u crosses the loops an odd number of
                // times exactly when p is inside; a side is half open in v,
                // so a ray through a corner counts it once.
                var inside = false
                for loop in loops {
                    for k in loop.indices {
                        let a = loop[k], b = loop[(k + 1) % loop.count]
                        guard (a.y > p.y) != (b.y > p.y) else { continue }
                        if p.x < a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y) { inside.toggle() }
                    }
                }
                return inside
            },
            trimLoops: { loops.map { ($0 + [$0[0]]).map { P2($0) } } },
            map: { p in SurfaceJet(position: frame.point(p.v), du: u, dv: v) },
            outward: 1)
    }
}

// MARK: - a landscape

public extension Heightfield {
    /// The landscape's faces: `Surface.faces(tiles:region:walls:refine:)`
    /// over its tiles, footprint and walls, placed on f where it has the
    /// function.
    var faces: [Face] {
        surface.faces(tiles: tiles, region: region, walls: walls, refine: refine)
    }
}

public extension Surface {
    /// The faces of the landscape: the graph, a plateau per band of the
    /// truncation, a riser wherever the truncation steps between two bands,
    /// and a wall along each edge of the walls' perimeter, in that order.
    /// Nothing for a grid too small to span its domain.
    ///
    /// The graph, the plateaus and the risers lie over the domain and are
    /// copied by the tiles: contoured once, then replicated, as the curves on
    /// them are. Each copy is cut to the footprint where it stands in space,
    /// as the mesh and visibility cut it. The walls stand on the perimeter in
    /// space, their top edge read through whichever tile is under it. Walls
    /// given as a mesh describe no face.
    ///
    /// The faces bound the solid the drawing tests curves against. With a
    /// refiner they are f's: the graph is |f| and every membership that
    /// compares |f| with the truncation reads f. Without one the grid is the
    /// surface. Where |f| is not a number it counts as over any truncation,
    /// as the facing reads it.
    func faces(tiles: [Affine2] = [.identity], region: Region = .full, walls: Walls = .none,
               refine: ContourRefine? = nil) -> [Face] {
        guard let graph = graph(tiles: tiles, region: region, refine: refine) else { return [] }
        return [graph] + plateaus(tiles: tiles, region: region, refine: refine)
            + risers(tiles: tiles, region: region, refine: refine)
            + self.walls(walls, tiles: tiles, refine: refine)
    }

    /// The parameter rectangle of the graph and the plateaus: the domain,
    /// sampled as the height grid is. Nil when the grid has fewer than two
    /// samples along an axis, too few to span it.
    var parameterRectangle: (u: ParamAxis, v: ParamAxis)? {
        guard height.width >= 2, height.height >= 2 else { return nil }
        return (ParamAxis(domain.real, periodic: false, samples: height.width),
                ParamAxis(domain.imag, periodic: false, samples: height.height))
    }

    /// The graph, where z = |f| under the truncation: over the parameter
    /// rectangle, its trimmed domain where |f| is under the truncation
    /// height, in the footprint. Its trim loops are the domain's edge and the
    /// plateaus' loops (`graphTrimLoops(tiles:refine:)`), each copy's with
    /// the footprint's perimeter carried back through its tile. Nil when the
    /// grid is too small to span its domain. Its derivatives are the slope
    /// the graph's normal is read from (`graphNormal(at:tiles:refine:)`), so
    /// `du × dv` is the normal the facing reads: f's own, untruncated, with a
    /// refiner; without one the grid's chords, truncated.
    func graph(tiles: [Affine2] = [.identity], region: Region = .full,
               refine: ContourRefine? = nil) -> Face? {
        guard let rectangle = parameterRectangle else { return nil }
        let magnitude = faceMagnitude(refine)
        let surface = self
        return Face(
            id: .graph, domain: rectangle,
            map: { p in
                let n = surface.graphNormal(at: P2(p.x, p.y), tiles: tiles, refine: refine)
                return SurfaceJet(position: SIMD3(p.x, p.y, magnitude(P2(p.x, p.y))),
                                  du: SIMD3(1, 0, -n.x), dv: SIMD3(0, 1, -n.y))
            },
            outward: 1, tiles: tiles,
            trimLoops: {
                surface.graphTrimLoops(tiles: tiles, refine: refine).loops.map { $0.map { P2<ParamSpace>($0.v) } }
            },
            footprint: { tile in
                guard case .inside(let perimeter) = region, let back = tile.inverse else { return nil }
                let corners = perimeter.edges.map { e -> P2<ParamSpace> in
                    let q = back(P3(e.start.x, e.start.y, 0))
                    return P2(q.x, q.y)
                }
                return (corners.isEmpty ? [] : [corners + [corners[0]]],
                        { p in region.contains(tile(P3(p.x, p.y, 0)).xy) })
            },
            contains: { p, tile in
                rectangle.u.covers(p.x) && rectangle.v.covers(p.y)
                    && region.contains(tile(P3(p.x, p.y, 0)).xy)
                    && magnitude(P2(p.x, p.y)) < surface.truncationHeight(atX: p.x)
            })
    }

    /// A plateau per band of the truncation that meets the domain and has a
    /// finite height: over the domain, where |f| reaches the band's height
    /// in the band, the domain's right edge in the band before it
    /// (`band(atX:from:)`).
    func plateaus(tiles: [Affine2] = [.identity], region: Region = .full,
                  refine: ContourRefine? = nil) -> [Face] {
        guard let rectangle = parameterRectangle else { return [] }
        let magnitude = faceMagnitude(refine)
        let surface = self
        let real = domain.real
        return truncation.bands(from: min(real.lo, real.hi), to: max(real.lo, real.hi)).compactMap { span in
            let h = span.height, k = span.band
            guard h.isFinite else { return nil }
            return Face(
                id: .plateau(band: k), domain: rectangle,
                map: { p in SurfaceJet(position: SIMD3(p.x, p.y, h), du: SIMD3(1, 0, 0), dv: SIMD3(0, 1, 0)) },
                outward: 1, tiles: tiles,
                contains: { p, tile in
                    rectangle.u.covers(p.x) && rectangle.v.covers(p.y)
                        && region.contains(tile(P3(p.x, p.y, 0)).xy)
                        && surface.band(atX: p.x) == k && !(magnitude(P2(p.x, p.y)) < h)
                })
        }
    }

    /// A riser at each boundary x = b inside the domain between two bands of
    /// different heights: the vertical face there, its parameters (y, z),
    /// from the lesser height up to |f| or the greater, whichever is lower,
    /// facing the lower band. It stands where the footprint holds both
    /// sides of the boundary, read a hair either side of it. Where it holds
    /// one, a wall on the boundary reads the band on the footprint's side
    /// (`walls(_:tiles:refine:)`) and rises to the solid's top there, and
    /// there is no drop to face. A boundary on the domain's edge has none:
    /// the band beyond the far edge starts outside the domain
    /// (`truncationHeight(atX:from:)`).
    func risers(tiles: [Affine2] = [.identity], region: Region = .full,
                refine: ContourRefine? = nil) -> [Face] {
        guard let rectangle = parameterRectangle else { return [] }
        let magnitude = faceMagnitude(refine)
        let real = domain.real
        let spans = truncation.bands(from: min(real.lo, real.hi), to: max(real.lo, real.hi))
        let highest = highestSample
        let side = 1e-9 * cell
        let y = rectangle.v
        return zip(spans, spans.dropFirst()).compactMap { left, right in
            let b = left.hi
            let lesser = min(left.height, right.height), greater = max(left.height, right.height)
            guard lesser.isFinite, lesser < greater else { return nil }
            return Face(
                id: .riser(band: left.band),
                domain: (y, ParamAxis(Interval(lo: lesser, hi: greater.isFinite ? greater : max(highest, lesser)),
                                      periodic: false, samples: 2)),
                map: { p in SurfaceJet(position: SIMD3(b, p.x, p.y), du: SIMD3(0, 1, 0), dv: SIMD3(0, 0, 1)) },
                outward: left.height > right.height ? 1 : -1, tiles: tiles,
                contains: { p, tile in
                    guard y.covers(p.x), p.y >= lesser, p.y <= min(magnitude(P2(b, p.x)), greater) else { return false }
                    return region.contains(tile(P3(b - side, p.x, 0)).xy)
                        && region.contains(tile(P3(b + side, p.x, 0)).xy)
                })
        }
    }

    /// A wall along each edge of the perimeter: its parameters the distance
    /// along the edge and the height, from the base up to the top edge, the
    /// truncated surface over the edge read through the tile under it, as
    /// the wall's curtain reads it (`Mesh.wallCurtain`), here exact rather
    /// than sampled: the truncation read from the footprint's side
    /// (`Surface.footprintSide`). The point on the edge is `a + (b - a) t`,
    /// which keeps a vertical edge's x, so a wall standing on a band's
    /// boundary reads the band its footprint is in, all along it. The
    /// perimeter's turn says which side is out: counter-clockwise from
    /// above, `du × dv` is.
    func walls(_ walls: Walls, tiles: [Affine2] = [.identity],
               refine: ContourRefine? = nil) -> [Face] {
        guard case .perimeter(let perimeter, let base) = walls else { return [] }
        let magnitude = faceMagnitude(refine)
        let truncation = self.truncation
        let grid = self
        let turn = perimeter.turn
        let top = truncation.highest.isFinite ? truncation.highest : highestSample
        return perimeter.edges.enumerated().compactMap { k, edge in
            let along = edge.end.v - edge.start.v, length = simd_length(along)
            guard length > 0 else { return nil }
            let s = ParamAxis(Interval(lo: 0, hi: length), periodic: false, samples: max(edge.density, 2))
            let inward = perimeter.inward(k)
            @Sendable func foot(_ distance: Double) -> P2<DomainSpace> {
                P2(edge.start.v + along * (distance / length))
            }
            return Face(
                id: .wall(edge: k),
                domain: (s, ParamAxis(Interval(lo: base, hi: max(top, base)), periodic: false, samples: 2)),
                contains: { p in
                    guard s.covers(p.x), p.y >= base else { return false }
                    let (q, back) = grid.tiled(foot(p.x), tiles: tiles)
                    let side = Surface.footprintSide(at: p.x / length, along: along, inward: inward, back: back)
                    return p.y <= min(magnitude(q), grid.truncationHeight(atX: q.x, from: side))
                },
                map: { p in
                    let q = foot(p.x)
                    return SurfaceJet(position: SIMD3(q.x, q.y, p.y),
                                      du: SIMD3(along.x / length, along.y / length, 0), dv: SIMD3(0, 0, 1))
                },
                outward: turn < 0 ? -1 : 1)
        }
    }
}

extension Surface {
    /// |f| at a point of the domain, untruncated, as the faces read it: f's
    /// own with a refiner, the grid's without, and over any truncation
    /// where it is not a number.
    fileprivate func faceMagnitude(_ refine: ContourRefine?) -> @Sendable (P2<DomainSpace>) -> Double {
        let grid = self
        let f = refine?.magnitude
        return { p in
            let u = f?(p) ?? grid.magnitude(at: p)
            return u.isNaN ? .infinity : u
        }
    }

    /// The greatest finite sample of the height grid, or 0.
    fileprivate var highestSample: Double {
        Double(height.values.lazy.filter(\.isFinite).max() ?? 0)
    }
}
