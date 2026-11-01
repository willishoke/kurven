import Foundation
import simd

/// Which part of a solid's boundary a face is.
public enum FaceID: Sendable, Hashable {
    /// A landscape's graph, where z = |f|.
    case graph
    /// The plateau of one band of the truncation, numbered as
    /// `Truncation.band(atX:)` numbers them; a uniform truncation's one
    /// plateau is band 0.
    case plateau(band: Int)
    /// The wall along one edge of the walls' perimeter.
    case wall(edge: Int)
    /// The riser where a banded truncation steps, at the end of band `band`
    /// toward greater Re z.
    case riser(band: Int)
    /// A parametric surface's one face.
    case surface
    /// A polyhedron's face, as `Solid.loops` lists them.
    case solid(face: Int)
}

/// A face in the boundary-representation sense (Mäntylä 1988): a region of
/// one smooth surface, flat or curved, bounded by loops. The faces of a
/// solid together are its manifold, the closed surface it is bounded by;
/// in boundary representation, its shell.
///
/// A face is a map from a parameter rectangle into space, trimmed: the part
/// of the rectangle it covers is its *trimmed domain*, bounded by its trim
/// loops, and `contains` is the domain's point membership classification
/// (Requicha 1980). A landscape's faces are its graph, a plateau per band
/// of the truncation, a riser where the truncation steps between two bands,
/// and a wall along each edge of the walls' perimeter
/// (`Surface.faces(tiles:region:walls:refine:)`); a parametric surface is
/// one face (`ParametricSurface.face`); a polyhedron's faces are its loops,
/// flat (`Solid.face(_:)`). `SceneGeometry.faces` lists a scene's.
///
/// A face is a view onto the grids and the solids it is read from, and
/// costs nothing to make: it holds closures, not samples.
public struct Face: Sendable {
    public let id: FaceID
    /// The parameter rectangle the map is defined on, and where a field on
    /// the face is sampled to be contoured.
    public let domain: (u: ParamAxis, v: ParamAxis)
    /// Whether a point of the parameter rectangle lies in the trimmed
    /// domain. On a periodic axis it answers the same a period along.
    public let contains: @Sendable (P2<ParamSpace>) -> Bool
    /// The point of space at a point of the parameter rectangle, with its
    /// two partial derivatives.
    public let map: @Sendable (P2<ParamSpace>) -> SurfaceJet
    /// `+1` when `du × dv` points out of the solid the face bounds, `-1`
    /// when it points in; nil when it bounds no solid.
    public let outward: Double?
    /// The copies of the face in space: the face itself is the identity's,
    /// and a landscape's graph, plateaus and risers are copied by its tiles,
    /// as its curves are (`Occluder.tiles`).
    public let instances: [Affine2]

    public init(id: FaceID, domain: (u: ParamAxis, v: ParamAxis),
                contains: @escaping @Sendable (P2<ParamSpace>) -> Bool,
                map: @escaping @Sendable (P2<ParamSpace>) -> SurfaceJet,
                outward: Double?, instances: [Affine2] = [.identity]) {
        self.id = id; self.domain = domain; self.contains = contains; self.map = map
        self.outward = outward; self.instances = instances
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
    /// trimmed domain what the loops bound, even-odd, so a hole is out.
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
    /// The graph, the plateaus and the risers lie over the domain, are
    /// trimmed to the footprint, and are copied by the tiles: contoured
    /// once, then replicated, as the curves on them are. The walls stand on
    /// the perimeter in world space, their top edge read through whichever
    /// tile is under it. Walls given as a mesh describe no face.
    ///
    /// With a refiner the faces are f's: the graph is |f| and every
    /// membership that compares |f| with the truncation reads f. Without one
    /// the grid is the surface. Where |f| is not a number it counts as
    /// over any truncation, as the facing reads it.
    func faces(tiles: [Affine2] = [.identity], region: Region = .full, walls: Walls = .none,
               refine: ContourRefine? = nil) -> [Face] {
        guard let graph = graph(tiles: tiles, region: region, refine: refine) else { return [] }
        return [graph] + plateaus(tiles: tiles, region: region, refine: refine)
            + risers(tiles: tiles, region: region, refine: refine)
            + self.walls(walls, tiles: tiles, refine: refine)
    }

    /// The graph, where z = |f| under the truncation: its parameter
    /// rectangle the domain, sampled as the height grid is, and its trimmed
    /// domain the footprint where |f| is under the truncation height. Nil
    /// when the grid has fewer than two samples along an axis, too few to
    /// span it. The derivatives are the untruncated slope: f's, by
    /// differences a step small against the cell, as the fold curves read it
    /// (`Heightfield.graphNormal(at:)`), or the grid's a cell wide.
    func graph(tiles: [Affine2] = [.identity], region: Region = .full,
               refine: ContourRefine? = nil) -> Face? {
        guard let rectangle = rectangle else { return nil }
        let magnitude = faceMagnitude(refine)
        let truncation = self.truncation
        let g = height
        let dx = abs(g.domain.real.length) / Double(max(g.width - 1, 1))
        let dy = abs(g.domain.imag.length) / Double(max(g.height - 1, 1))
        let grid = self
        let slope: @Sendable (P2<DomainSpace>) -> SIMD2<Double> = { p in
            // f's own slope by differences a step small against the cell;
            // the grid's chords a cell wide where there is no f, or its
            // slope is not finite.
            if refine?.magnitude != nil {
                let step = 1e-4 * max(dx, dy)
                let hx = (magnitude(P2(p.x + step, p.y)) - magnitude(P2(p.x - step, p.y))) / (2 * step)
                let hy = (magnitude(P2(p.x, p.y + step)) - magnitude(P2(p.x, p.y - step))) / (2 * step)
                if hx.isFinite, hy.isFinite { return SIMD2(hx, hy) }
            }
            let hx = (grid.magnitude(at: P2(p.x + dx, p.y)) - grid.magnitude(at: P2(p.x - dx, p.y))) / (2 * dx)
            let hy = (grid.magnitude(at: P2(p.x, p.y + dy)) - grid.magnitude(at: P2(p.x, p.y - dy))) / (2 * dy)
            return SIMD2(hx, hy)
        }
        return Face(
            id: .graph, domain: rectangle,
            contains: { p in
                rectangle.u.covers(p.x) && rectangle.v.covers(p.y) && region.contains(P2(p.x, p.y))
                    && magnitude(P2(p.x, p.y)) < truncation.height(atX: p.x)
            },
            map: { p in
                let q = P2<DomainSpace>(p.v), s = slope(q)
                return SurfaceJet(position: SIMD3(p.x, p.y, magnitude(q)),
                                  du: SIMD3(1, 0, s.x), dv: SIMD3(0, 1, s.y))
            },
            outward: 1, instances: tiles)
    }

    /// A plateau per band of the truncation that meets the domain and has a
    /// finite height: over the domain, where |f| reaches the band's height
    /// in the band.
    func plateaus(tiles: [Affine2] = [.identity], region: Region = .full,
                  refine: ContourRefine? = nil) -> [Face] {
        guard let rectangle = rectangle else { return [] }
        let magnitude = faceMagnitude(refine)
        let truncation = self.truncation
        let real = domain.real
        return truncation.bands(from: min(real.lo, real.hi), to: max(real.lo, real.hi)).compactMap { band in
            let h = band.height, k = band.band
            guard h.isFinite else { return nil }
            return Face(
                id: .plateau(band: k), domain: rectangle,
                contains: { p in
                    rectangle.u.covers(p.x) && rectangle.v.covers(p.y) && region.contains(P2(p.x, p.y))
                        && truncation.band(atX: p.x) == k && !(magnitude(P2(p.x, p.y)) < h)
                },
                map: { p in SurfaceJet(position: SIMD3(p.x, p.y, h), du: SIMD3(1, 0, 0), dv: SIMD3(0, 1, 0)) },
                outward: 1, instances: tiles)
        }
    }

    /// A riser at each boundary inside the domain between two bands of
    /// different heights: the vertical face x = b, its parameters (y, z),
    /// from the lesser height up to |f| or the greater height, whichever is
    /// lower, where |f| is over the lesser. It faces the lower band.
    func risers(tiles: [Affine2] = [.identity], region: Region = .full,
                refine: ContourRefine? = nil) -> [Face] {
        guard let rectangle = rectangle else { return [] }
        let magnitude = faceMagnitude(refine)
        let real = domain.real
        let bands = truncation.bands(from: min(real.lo, real.hi), to: max(real.lo, real.hi))
        let highest = highestSample
        return zip(bands, bands.dropFirst()).compactMap { left, right in
            let b = left.hi
            let lesser = min(left.height, right.height), greater = max(left.height, right.height)
            guard lesser.isFinite, lesser < greater else { return nil }
            let y = rectangle.v
            return Face(
                id: .riser(band: left.band),
                domain: (y, ParamAxis(Interval(lo: lesser, hi: greater.isFinite ? greater : max(highest, lesser)),
                                      periodic: false, samples: 2)),
                contains: { p in
                    y.covers(p.x) && region.contains(P2(b, p.x))
                        && p.y >= lesser && p.y <= min(magnitude(P2(b, p.x)), greater)
                },
                map: { p in SurfaceJet(position: SIMD3(b, p.x, p.y), du: SIMD3(0, 1, 0), dv: SIMD3(0, 0, 1)) },
                outward: left.height > right.height ? 1 : -1, instances: tiles)
        }
    }

    /// A wall along each edge of the perimeter: its parameters the distance
    /// along the edge and the height, from the base up to the top edge, the
    /// truncated surface over the edge (`topEdge(of:tiles:refine:)`, here
    /// exact rather than sampled). The perimeter's turn says which side is
    /// out: counter-clockwise from above, `du × dv` is.
    ///
    /// The rectangle's height runs up to the highest truncation, or without
    /// one to the grid's highest sample: a bound to sample a field on, not
    /// part of membership, since f can rise over its samples.
    func walls(_ walls: Walls, tiles: [Affine2] = [.identity],
               refine: ContourRefine? = nil) -> [Face] {
        guard case .perimeter(let perimeter, let base) = walls else { return [] }
        let magnitude = faceMagnitude(refine)
        let truncation = self.truncation
        let grid = self
        let corners = perimeter.edges.map(\.start.v)
        let turn = corners.indices.reduce(0.0) { a, k in
            let p = corners[k], q = corners[(k + 1) % corners.count]
            return a + p.x * q.y - q.x * p.y
        }
        let top = truncation.highest.isFinite ? truncation.highest : highestSample
        return perimeter.edges.enumerated().compactMap { k, edge in
            let along = edge.end.v - edge.start.v, length = simd_length(along)
            guard length > 0 else { return nil }
            let s = ParamAxis(Interval(lo: 0, hi: length), periodic: false, samples: max(edge.density, 2))
            @Sendable func foot(_ distance: Double) -> P2<DomainSpace> {
                let q = Surface.point(on: edge, at: distance / length)
                return P2(q.x, q.y)
            }
            return Face(
                id: .wall(edge: k),
                domain: (s, ParamAxis(Interval(lo: base, hi: max(top, base)), periodic: false, samples: 2)),
                contains: { p in
                    guard s.covers(p.x), p.y >= base else { return false }
                    let q = foot(p.x)
                    return p.y <= min(magnitude(grid.inTile(q, tiles: tiles)), truncation.height(atX: q.x))
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
    /// The parameter rectangle of the graph and the plateaus: the domain,
    /// sampled as the height grid is. Nil when the grid has fewer than two
    /// samples along an axis, too few to span it.
    fileprivate var rectangle: (u: ParamAxis, v: ParamAxis)? {
        guard height.width >= 2, height.height >= 2 else { return nil }
        return (ParamAxis(domain.real, periodic: false, samples: height.width),
                ParamAxis(domain.imag, periodic: false, samples: height.height))
    }

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
