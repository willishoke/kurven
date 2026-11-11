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
    /// The riser at the tile boundary between the copies on two tiles, by
    /// their place in the tiles: where the copy on `tile` stands over the
    /// copy on `facing`, and faces it
    /// (`Surface.tileBoundaryRisers(tiles:region:refine:)`).
    case tileBoundaryRiser(tile: Int, facing: Int)
    /// The wall along one edge of the walls' perimeter.
    case wall(edge: Int)
    /// A parametric surface's one face.
    case surface
    /// A polyhedron's face, as `Solid.loops` lists them.
    case solid(face: Int)
}

/// A kind of face, as `FaceID` names it less which one: what a layer's
/// curves lie on (`SceneGeometry.face(of:)`).
public enum FaceKind: Sendable, Hashable {
    case graph, plateau, riser, wall, surface, solid
}

public extension FaceID {
    var kind: FaceKind {
        switch self {
        case .graph: .graph
        case .plateau: .plateau
        case .riser, .tileBoundaryRiser: .riser
        case .wall: .wall
        case .surface: .surface
        case .solid: .solid
        }
    }
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
/// graph, a plateau per band of the truncation, a riser wherever the drawn
/// top drops inside the solid, where the truncation steps between two bands
/// or two copies meet at a tile boundary with different tops, and a wall
/// along each edge of the walls' perimeter
/// (`Surface.faces(tiles:region:walls:refine:)`); a parametric surface is one
/// face (`ParametricSurface.face`); a polyhedron's faces are its loops, flat
/// (`Solid.face(_:)`). `SceneGeometry.faces` lists a scene's.
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
    /// landscape's graph, plateaus and band boundaries' risers, which have
    /// one per tile, as
    /// the curves on them do (`Occluder.tiles`): contoured once, then
    /// replicated.
    public let copies: [Copy]
    private let loops: Computed<[[P2<ParamSpace>]]>?
    /// Where a point of space stands on the face, before a copy moves it: a
    /// point of the parameter plane. A flat face's is the point's orthogonal
    /// projection onto its plane; the graph's and a plateau's the point of the
    /// domain under it; a parametric surface's the nearest point, by Newton's
    /// method (`ParametricSurface.nearestParameter(to:)`). What a pick reads
    /// a face's distance off (`Face.pick(_:among:within:tie:)`).
    public let foot: @Sendable (SIMD3<Double>) -> P2<ParamSpace>
    /// A copy's boundary, given its tile: the curves of the parameter plane
    /// its trimmed domain ends on (`boundary(of:)`).
    let bounding: @Sendable (Affine2) -> [[P2<ParamSpace>]]

    /// The closed curves in the parameter plane that bound the face's trimmed
    /// domain even-odd, before a copy is cut to the footprint, each ending
    /// where it begins: where a contour on the face is cut, and which side of
    /// each is kept (`contours(of:_:on:)`). A polyhedron face's are its
    /// loops. The graph's are the domain's edge and the plateaus' loops, the
    /// truncation edge among them (`Surface.graphTrimLoops(refine:)`), placed
    /// on f with a refiner, the first time they are asked, and shared by
    /// every copy. A copy ends at a tile boundary, which is the domain's
    /// edge, and the loops are closed along it as a band's are across a
    /// riser: the top of a riser at a tile boundary lies on them, carried
    /// back by the higher copy's tile.
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
    /// On the graph of f under a truncation, |f| less the truncation
    /// height: negative on the graph, zero on the truncation edge, whose
    /// chords the trim loops are. A contour on the face is cut where it
    /// reaches 0, on the edge itself rather than on a chord of it
    /// (`contours(of:_:on:)`). Nil on every other face, on the graph where
    /// nothing is truncated, and on a grid's graph, whose truncation edge is
    /// the loops' chords themselves, solved linearly on the grid.
    public let truncationExcess: (@Sendable (P2<ParamSpace>) -> Double)?

    /// The contours of a field on one copy of the face, clipped to its
    /// trimmed domain: cut where they cross the face's trim loops or the
    /// footprint's perimeter where the copy stands, kept where the loops
    /// hold them even-odd and the footprint holds them
    /// (`Contour.levels(of:_:over:trimLoops:contains:cuts:)`). On a face with
    /// no trim loops they are kept where the copy's `contains` holds, a
    /// segment at a time. On the graph they are cut on the truncation edge
    /// exactly (`truncationExcess`), and a chord is refined where it sags
    /// off the graph as well as where it strays from the level set: the
    /// graph's parameter plane is the world's xy, so its sag is a length in
    /// the cell's own units.
    public func contours(of field: ScalarField, _ levels: [Double], on copy: Copy)
        -> [(level: Double, paths: [[P2<ParamSpace>]])] {
        guard let loops = trimLoops else {
            return Contour.levels(of: field, levels, over: domain, contains: copy.contains)
        }
        let map = self.map
        var lift: (@Sendable (P2<ParamSpace>) -> SIMD3<Double>)?
        if id.kind == .graph { lift = { @Sendable p in map(p).position } }
        return Contour.levels(of: field, levels, over: domain, trimLoops: loops,
                              contains: copy.footprint?.contains, cuts: copy.footprint?.perimeter ?? [],
                              excess: truncationExcess, lift: lift)
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
                map: @escaping @Sendable (P2<ParamSpace>) -> SurfaceJet, outward: Double?,
                foot: @escaping @Sendable (SIMD3<Double>) -> P2<ParamSpace>,
                bounding: @escaping @Sendable (Affine2) -> [[P2<ParamSpace>]]) {
        self.init(id: id, domain: domain, map: map, outward: outward, tiles: [.identity],
                  trimLoops: trimLoops, foot: foot, bounding: bounding) { p, _ in contains(p) }
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
                truncationExcess: (@Sendable (P2<ParamSpace>) -> Double)? = nil,
                foot: @escaping @Sendable (SIMD3<Double>) -> P2<ParamSpace>,
                bounding: @escaping @Sendable (Affine2) -> [[P2<ParamSpace>]],
                contains: @escaping @Sendable (P2<ParamSpace>, Affine2) -> Bool) {
        let loops = trimLoops.map(Computed.init)
        self.id = id; self.domain = domain; self.map = map; self.loops = loops
        self.truncationExcess = truncationExcess
        self.foot = foot; self.bounding = bounding
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

    /// The kind of face a layer's curves lie on, which is what their
    /// visibility is read from: the facing of that face under each point,
    /// then the scene's occlusion test (`Scene.visible(_:depthTolerance:image:)`).
    /// Nil for curves on no one face, tested by occlusion alone: a fold,
    /// edge-on by construction; a crease between two faces, as an outline, a
    /// truncation edge and a polyhedron's edges are; anything else stored
    /// with nothing said of where it lies.
    ///
    /// On a landscape a contour or a hatching lies on the faces it selects
    /// (`FaceSelector`): the contours of |f| and arg f on the graph, and
    /// over the truncation, lifted onto it, on a plateau, whose facing the
    /// graph's test reads. A wall's edges lie on the walls, and stored curves
    /// given the roles of |f| and arg f on the graph. The truncation edge,
    /// with the risers' tops and bottoms, is a crease. On a parametric
    /// surface every curve but an outline lies on its one face. On a
    /// polyhedron its hatching lies on its faces.
    func face(of spec: LayerSpec) -> FaceKind? {
        if case .foldCurves = spec.source { return nil }
        switch self {
        case .heightfield:
            switch spec.source {
            case .contour(_, _, _, let faces, _), .hatching(_, let faces): return faces.kind
            case .wallEdges: return .wall
            default: return spec.role == .magnitude || spec.role == .phase ? .graph : nil
            }
        case .parametric:
            return spec.role == .outline ? nil : .surface
        case .solid:
            if case .hatching(_, .solid) = spec.source { return .solid }
            return nil
        }
    }
}

// MARK: - a parametric surface

public extension ParametricSurface {
    /// The surface as one face: its whole parameter rectangle, glued where
    /// an axis is periodic.
    var face: Face {
        let (u, v) = (self.u, self.v)
        let surface = self
        return Face(id: .surface, domain: (u, v), contains: { u.covers($0.x) && v.covers($0.y) },
                    map: map, outward: outward, foot: { surface.nearestParameter(to: $0) },
                    bounding: { _ in surface.seams })
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
            outward: 1,
            foot: { P2(frame.coordinates($0)) },
            bounding: { _ in loops.map { ($0 + [$0[0]]).map { P2($0) } } })
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
    /// truncation, a riser wherever the truncation steps between two bands, a
    /// riser either way up at each tile boundary where two copies meet, and a
    /// wall along each edge of the walls' perimeter, in that order. Nothing
    /// for a grid too small to span its domain.
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
            + tileBoundaryRisers(tiles: tiles, region: region, refine: refine)
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
    /// plateaus' loops (`graphTrimLoops(refine:)`), each copy's with
    /// the footprint's perimeter carried back through its tile. Nil when the
    /// grid is too small to span its domain. Its derivatives are the slope
    /// the graph's normal is read from (`graphNormal(at:tiles:refine:)`), on
    /// the face's own domain, before a copy moves it, so `du × dv` is the
    /// normal the facing reads: f's own, untruncated, with a refiner;
    /// without one the grid's chords, truncated.
    func graph(tiles: [Affine2] = [.identity], region: Region = .full,
               refine: ContourRefine? = nil) -> Face? {
        guard let rectangle = parameterRectangle else { return nil }
        let magnitude = faceMagnitude(refine)
        let surface = self
        let loops = graphLoops(refine: refine)
        var excess: (@Sendable (P2<ParamSpace>) -> Double)?
        if truncation != .none, refine?.magnitude != nil {
            excess = { @Sendable p in
                let h = surface.truncationHeight(atX: p.x)
                return h.isFinite ? magnitude(P2(p.x, p.y)) - h : -.infinity
            }
        }
        return Face(
            id: .graph, domain: rectangle,
            map: { p in
                let n = surface.graphNormal(at: P2(p.x, p.y), tiles: [.identity], refine: refine)
                return SurfaceJet(position: SIMD3(p.x, p.y, magnitude(P2(p.x, p.y))),
                                  du: SIMD3(1, 0, -n.x), dv: SIMD3(0, 1, -n.y))
            },
            outward: 1, tiles: tiles,
            trimLoops: { loops.value },
            footprint: { tile in Surface.footprint(region, on: tile) },
            truncationExcess: excess,
            foot: { P2($0.x, $0.y) },
            bounding: { tile in surface.topBoundary(.graph, loops: loops.value, region: region, tile: tile) },
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
        let loops = graphLoops(refine: refine)
        return truncation.bands(from: min(real.lo, real.hi), to: max(real.lo, real.hi)).compactMap { span in
            let h = span.height, k = span.band
            guard h.isFinite else { return nil }
            return Face(
                id: .plateau(band: k), domain: rectangle,
                map: { p in SurfaceJet(position: SIMD3(p.x, p.y, h), du: SIMD3(1, 0, 0), dv: SIMD3(0, 1, 0)) },
                outward: 1, tiles: tiles,
                foot: { P2($0.x, $0.y) },
                bounding: { tile in surface.topBoundary(.plateau(band: k), loops: loops.value, region: region, tile: tile) },
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
        let tolerance = profileTolerance(refine)
        return zip(spans, spans.dropFirst()).compactMap { left, right in
            let b = left.hi
            let lesser = min(left.height, right.height), greater = max(left.height, right.height)
            guard lesser.isFinite, lesser < greater else { return nil }
            let z = ParamAxis(Interval(lo: lesser, hi: greater.isFinite ? greater : max(highest, lesser)),
                              periodic: false, samples: 2)
            let inRiser: @Sendable (P2<ParamSpace>, Affine2) -> Bool = { p, tile in
                guard y.covers(p.x), p.y >= lesser, p.y <= min(magnitude(P2(b, p.x)), greater) else { return false }
                return region.contains(tile(P3(b - side, p.x, 0)).xy)
                    && region.contains(tile(P3(b + side, p.x, 0)).xy)
            }
            return Face(
                id: .riser(band: left.band),
                domain: (y, z),
                map: { p in SurfaceJet(position: SIMD3(b, p.x, p.y), du: SIMD3(0, 1, 0), dv: SIMD3(0, 0, 1)) },
                outward: left.height > right.height ? 1 : -1, tiles: tiles,
                foot: { P2($0.y, $0.z) },
                // Where it stands along its boundary, the rows and where the
                // footprint's perimeter crosses the boundary seeding the search,
                // from the lesser height up to its top.
                bounding: { tile in
                    let stops = (Surface.footprint(region, on: tile)?.perimeter ?? []).flatMap {
                        Surface.crossings(of: $0, atX: b)
                    }
                    let stations = ((0..<y.samples).map(y.coordinate) + stops).filter(y.covers).sorted()
                    return Surface.columnLoops(stations, eps: side, tolerance: tolerance,
                                               stands: { inRiser(P2($0, lesser), tile) },
                                               bottom: { _ in lesser },
                                               top: { min(magnitude(P2(b, $0)), greater, z.range.hi) })
                },
                contains: inRiser)
        }
    }

    /// A riser at each tile boundary where two copies meet
    /// (`tileBoundaries(tiles:)`), one each way up: the vertical face there,
    /// its parameters the distance along the tile boundary and the height,
    /// from the lower copy's top up to the higher's, facing the lower copy.
    /// Each copy's top is read on the tile boundary through its own tile, at
    /// the point of the domain's edge the tile boundary point is the copy of
    /// (`tileBoundaryTop`), and the riser stands where one stands over the
    /// other (`stands(_:over:)`): a turned, translated or glide-reflected
    /// neighbour's top can, one reflected across the tile boundary never
    /// does, reading the same points of the domain. As at a band boundary,
    /// it stands where the footprint holds both sides of the tile boundary,
    /// read a hair either side of it; where it holds one, a wall on the tile
    /// boundary bounds the copy its footprint is in (`wallTile`) and rises
    /// to that copy's top.
    /// The parameter rectangle runs from 0, under any |f|, to the highest
    /// truncation, or the grid's highest sample without one.
    func tileBoundaryRisers(tiles: [Affine2] = [.identity], region: Region = .full,
                    refine: ContourRefine? = nil) -> [Face] {
        let magnitude = faceMagnitude(refine)
        let surface = self
        let side = 1e-9 * cell
        let top = truncation.highest.isFinite ? truncation.highest : highestSample
        return tileBoundaries(tiles: tiles).flatMap { boundary -> [Face] in
            guard let firstBack = tiles[boundary.first].inverse, let secondBack = tiles[boundary.second].inverse else { return [] }
            let length = boundary.length, n = boundary.normal
            let s = ParamAxis(Interval(lo: 0, hi: length), periodic: false,
                              samples: max(Int((length / surface.cell).rounded(.up)) + 1, 2))
            let along = (boundary.end.v - boundary.start.v) / length
            // Which side of the tile boundary the second copy is on: its tile's
            // image of the domain's centre.
            let d = domain
            let centre = tiles[boundary.second](P3(0.5 * (d.real.lo + d.real.hi), 0.5 * (d.imag.lo + d.imag.hi), 0)).xy.v
            let secondLeft = simd_dot(centre - boundary.start.v, n) > 0
            let tolerance = surface.profileTolerance(refine)
            // The stations a riser's boundary is found from along the tile
            // boundary: the grid's spacing, and where the footprint's
            // perimeter crosses it.
            let start = boundary.start.v
            let stops = (Surface.footprint(region, on: .identity)?.perimeter ?? []).flatMap { loop in
                zip(loop, loop.dropFirst()).compactMap { a, c -> Double? in
                    let e = c.v - a.v, r = a.v - start
                    let det = along.x * e.y - along.y * e.x
                    guard det != 0 else { return nil }
                    let t = (r.x * e.y - r.y * e.x) / det, u = (r.x * along.y - r.y * along.x) / det
                    return u >= 0 && u <= 1 && t > 0 && t < length ? t : nil
                }
            }
            let stations = ((0..<s.samples).map(s.coordinate) + stops).sorted()
            return [(boundary.first, firstBack, boundary.second, secondBack, !secondLeft),
                    (boundary.second, secondBack, boundary.first, firstBack, secondLeft)].map { higher, up, lower, down, lowerRight in
                @Sendable func high(_ s: Double) -> Double { surface.tileBoundaryTop(boundary.point(s), back: up, magnitude: magnitude) }
                @Sendable func low(_ s: Double) -> Double { surface.tileBoundaryTop(boundary.point(s), back: down, magnitude: magnitude) }
                let inRiser: @Sendable (P2<ParamSpace>) -> Bool = { p in
                    guard s.covers(p.x) else { return false }
                    let w = boundary.point(p.x)
                    guard region.contains(P2(w.v - side * n)), region.contains(P2(w.v + side * n)) else { return false }
                    let (high, low) = (high(p.x), low(p.x))
                    return Surface.stands(high, over: low) && p.y >= low && p.y <= high
                }
                return Face(
                    id: .tileBoundaryRiser(tile: higher, facing: lower),
                    domain: (s, ParamAxis(Interval(lo: 0, hi: max(top, 0)), periodic: false, samples: 2)),
                    contains: inRiser,
                    map: { p in
                        let w = boundary.point(p.x)
                        return SurfaceJet(position: SIMD3(w.x, w.y, p.y), du: SIMD3(along.x, along.y, 0), dv: SIMD3(0, 0, 1))
                    },
                    // du × dv is the right of the tile boundary's direction.
                    outward: lowerRight ? 1 : -1,
                    foot: { q in P2(simd_dot(SIMD2(q.x, q.y) - boundary.start.v, along), q.z) },
                    // Where it stands along the tile boundary, from the lower
                    // copy's top up to the higher's.
                    bounding: { _ in
                        Surface.columnLoops(stations, eps: side, tolerance: tolerance,
                                            stands: { inRiser(P2($0, low($0))) }, bottom: low, top: high)
                    })
            }
        }.sorted { $0.id < $1.id }
    }

    /// A wall along each edge of the perimeter: its parameters the distance
    /// along the edge and the height, from the base up to the top edge, the
    /// truncated surface over the edge read through the tile under it, as
    /// the wall's curtain reads it (`Mesh.wallCurtain`), here exact rather
    /// than sampled: the truncation read from the footprint's side
    /// (`Surface.footprintSide`). The point on the edge is `a + (b - a) t`,
    /// which keeps a vertical edge's x, so a wall standing on a band's
    /// boundary reads the band its footprint is in, all along it. The sign
    /// of the perimeter's signed area says which side is out:
    /// counter-clockwise from above, `du × dv` is.
    func walls(_ walls: Walls, tiles: [Affine2] = [.identity],
               refine: ContourRefine? = nil) -> [Face] {
        guard case .perimeter(let perimeter, let base) = walls else { return [] }
        let magnitude = faceMagnitude(refine)
        let truncation = self.truncation
        let grid = self
        let area = perimeter.signedArea
        let top = truncation.highest.isFinite ? truncation.highest : highestSample
        return perimeter.edges.enumerated().compactMap { k, edge in
            let along = edge.end.v - edge.start.v, length = simd_length(along)
            guard length > 0 else { return nil }
            let s = ParamAxis(Interval(lo: 0, hi: length), periodic: false, samples: max(edge.density, 2))
            let inward = perimeter.inward(k)
            @Sendable func foot(_ distance: Double) -> P2<DomainSpace> {
                P2(edge.start.v + along * (distance / length))
            }
            let unit = along / length
            return Face(
                id: .wall(edge: k),
                domain: (s, ParamAxis(Interval(lo: base, hi: max(top, base)), periodic: false, samples: 2)),
                contains: { p in
                    guard s.covers(p.x), p.y >= base else { return false }
                    let top = grid.wallFoot(at: foot(p.x), t: p.x / length, along: along, inward: inward, tiles: tiles)
                    return p.y <= min(magnitude(top.point), top.truncationHeight)
                },
                map: { p in
                    let q = foot(p.x)
                    return SurfaceJet(position: SIMD3(q.x, q.y, p.y),
                                      du: SIMD3(along.x / length, along.y / length, 0), dv: SIMD3(0, 0, 1))
                },
                outward: area < 0 ? -1 : 1,
                foot: { q in P2(simd_dot(SIMD2(q.x, q.y) - edge.start.v, unit), q.z) },
                // Its bottom, its far end, its top edge as its layer draws it,
                // back along, and its near end.
                bounding: { _ in
                    let topEdge = grid.topEdge(of: edge, inward: inward, tiles: tiles, refine: refine)
                    return [[P2(0, base), P2(length, base)]
                            + topEdge.reversed().map { P2(simd_dot($0.xy.v - edge.start.v, unit), $0.z) }
                            + [P2(0, base)]]
                })
        }
    }
}

extension Surface {
    /// The footprint where a copy on `tile` stands, in the parameter plane:
    /// its perimeter carried back through the tile, closed, and whether it
    /// holds the point of space a point of the plane lands on. Nil where
    /// there is no footprint. A tile with no inverse carries no perimeter
    /// back, and the footprint is asked a point at a time.
    static func footprint(_ region: Region, on tile: Affine2)
        -> (perimeter: [[P2<ParamSpace>]], contains: @Sendable (P2<ParamSpace>) -> Bool)? {
        guard case .inside(let perimeter) = region else { return nil }
        let corners = tile.inverse.map { back in
            perimeter.edges.map { e -> P2<ParamSpace> in
                let q = back(P3(e.start.x, e.start.y, 0))
                return P2(q.x, q.y)
            }
        } ?? []
        return (corners.isEmpty ? [] : [corners + [corners[0]]],
                { p in region.contains(tile(P3(p.x, p.y, 0)).xy) })
    }

    /// |f| at a point of the domain, untruncated, as the faces read it: f's
    /// own with a refiner, the grid's without, and over any truncation
    /// where it is not a number.
    func faceMagnitude(_ refine: ContourRefine?) -> @Sendable (P2<DomainSpace>) -> Double {
        let grid = self
        let f = refine?.magnitude
        return { p in
            let u = f?(p) ?? grid.magnitude(at: p)
            return u.isNaN ? .infinity : u
        }
    }

    /// The greatest finite sample of the height grid, or 0.
    var highestSample: Double {
        Double(height.values.lazy.filter(\.isFinite).max() ?? 0)
    }

    /// The graph's trim loops (`graphTrimLoops(refine:)`), in the parameter
    /// plane, made the first time they are asked.
    func graphLoops(refine: ContourRefine?) -> Computed<[[P2<ParamSpace>]]> {
        let surface = self
        return Computed { surface.graphTrimLoops(refine: refine).loops.map { $0.map { P2<ParamSpace>($0.v) } } }
    }

    /// The boundary of a copy of the graph or of a plateau on `tile`, in the
    /// parameter plane, as the graph's trim loops `loops` bound it: the
    /// trimmed domain read even-odd off them, as a contour on the face is
    /// clipped (`Face.contours(of:_:on:)`). The graph's is the region the
    /// loops bound, a plateau's the region its band's plateau loops bound in
    /// its band, each cut to the footprint where the copy stands. Its pieces
    /// are those of the loops, of each band boundary inside the domain,
    /// across it, and of the footprint's perimeter, each cut where another
    /// crosses it (`Face.pieces(of:eps:)`), and kept where the face lies a
    /// ten-millionth of a cell on one side and not the other: a stretch two
    /// loops pass, bounding nothing, is left out, and so is a band boundary
    /// or a stretch of the perimeter the face does not reach. Ends within a
    /// millionth of a cell are one (`Face.snapped(_:eps:)`).
    func topBoundary(_ id: FaceID, loops: [[P2<ParamSpace>]], region: Region, tile: Affine2) -> [[P2<ParamSpace>]] {
        let index = TrimIndex(id == .graph ? loops : Array(loops.dropFirst()), periods: (nil, nil))
        let real = domain.real, imag = domain.imag
        let (y0, y1) = (min(imag.lo, imag.hi), max(imag.lo, imag.hi))
        let spans = truncation.bands(from: min(real.lo, real.hi), to: max(real.lo, real.hi))
        let boundaries = zip(spans, spans.dropFirst()).map { left, _ in [SIMD2(left.hi, y0), SIMD2(left.hi, y1)] }
        let perimeter = (Surface.footprint(region, on: tile)?.perimeter ?? []).map { $0.map(\.v) }
        func holds(_ p: SIMD2<Double>) -> Bool {
            guard index.encloses(p), region.contains(tile(P3(p.x, p.y, 0)).xy) else { return false }
            if case .plateau(let k) = id { return band(atX: p.x) == k }
            return true
        }
        let h = 1e-7 * cell
        let kept = Face.pieces(of: loops.map { $0.map(\.v) } + boundaries + perimeter, eps: 1e-9 * cell).filter { a, b in
            let d = b - a, length = simd_length(d)
            guard length > 0 else { return false }
            let m = 0.5 * (a + b), n = SIMD2(-d.y, d.x) / length
            return holds(m + h * n) != holds(m - h * n)
        }
        return Face.chained(Face.snapped(kept, eps: 1e-6 * cell)).map { $0.map { P2<ParamSpace>($0) } }
    }

    /// The boundary of a vertical face whose trimmed domain is, at each
    /// point along it, an interval from `bottom` up to `top`: a closed loop
    /// round each stretch where it `stands`. The stretches are found at
    /// `stations`, each end where the verdict turns between two placed by
    /// bisection to within `eps`; a loop runs along the bottom, up the far
    /// end, back along the top and down the near end, the bottom and the
    /// top through the stations between and subdivided to `tolerance`
    /// (`profile(_:tolerance:_:)`).
    static func columnLoops(_ stations: [Double], eps: Double, tolerance: Double,
                            stands: (Double) -> Bool, bottom: (Double) -> Double,
                            top: (Double) -> Double) -> [[P2<ParamSpace>]] {
        guard let first = stations.first else { return [] }
        var stretches: [(lo: Double, hi: Double)] = []
        var previous = first, standing = stands(first)
        var start: Double? = standing ? first : nil
        for s in stations.dropFirst() where s > previous {
            let verdict = stands(s)
            if verdict != standing {
                var (a, c) = (previous, s)
                while c - a > eps {
                    let m = 0.5 * (a + c)
                    guard m > a, m < c else { break }
                    if stands(m) == standing { a = m } else { c = m }
                }
                if let lo = start, standing { stretches.append((lo, a)); start = nil } else { start = c }
            }
            previous = s; standing = verdict
        }
        if let lo = start, standing { stretches.append((lo, previous)) }
        return stretches.compactMap { lo, hi in
            guard hi > lo else { return nil }
            let along = [lo] + stations.filter { $0 > lo && $0 < hi } + [hi]
            let under = profile(along, tolerance: tolerance, bottom)
            return under + profile(along, tolerance: tolerance, top).reversed() + [under[0]]
        }
    }

    /// How far a profile along a riser's line may depart from its chord: the
    /// refiner's tolerance on f, and on the grid, whose heights are linear
    /// between its rows, a millionth of a cell.
    func profileTolerance(_ refine: ContourRefine?) -> Double {
        (refine?.magnitude != nil ? refine?.tolerance ?? 1e-6 : 1e-6) * cell
    }

    /// The heights where a closed polyline of the domain crosses the line
    /// Re z = `x`.
    static func crossings(of loop: [P2<ParamSpace>], atX x: Double) -> [Double] {
        zip(loop, loop.dropFirst()).compactMap { a, b in
            guard a.x != b.x, (a.x - x) * (b.x - x) <= 0 else { return nil }
            return a.y + (x - a.x) / (b.x - a.x) * (b.y - a.y)
        }
    }

    /// The curve `(s, top(s))` through `stations`, each chord halved until
    /// its midpoint lies within `tolerance` of the curve, ten deep.
    static func profile(_ stations: [Double], tolerance: Double,
                        _ top: (Double) -> Double) -> [P2<ParamSpace>] {
        var out: [P2<ParamSpace>] = []
        func subdivide(_ s0: Double, _ z0: Double, _ s1: Double, _ z1: Double, depth: Int) {
            let sm = 0.5 * (s0 + s1), zm = top(sm)
            guard depth < 10, sm > s0, sm < s1, abs(zm - 0.5 * (z0 + z1)) > tolerance else { return }
            subdivide(s0, z0, sm, zm, depth: depth + 1)
            out.append(P2(sm, zm))
            subdivide(sm, zm, s1, z1, depth: depth + 1)
        }
        for (k, s) in stations.enumerated() {
            let z = top(s)
            if k > 0, let last = out.last { subdivide(last.x, last.y, s, z, depth: 0) }
            out.append(P2(s, z))
        }
        return out
    }
}
