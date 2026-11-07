import Foundation
import simd

/// Contours of a face's coordinates and height on a landscape's faces.
///
/// |f| and arg f have the refiner and their own rules (`Surface.derive`);
/// u, v and the height are smooth on every face, with no singular points to
/// seed, and are contoured by the plain routine: marching squares on the
/// face's parameter rectangle, each vertex bisected onto the level set
/// (`Contour.levels(of:_:over:trimLoops:contains:cuts:)`), then cut where it
/// leaves the face.
public extension Surface {
    /// The contours of `field` at `levels` on `faces`, in world space: on
    /// the graph cut at its trim loops, the truncation edge among them, and
    /// lifted to |f|; on the plateaus cut where each ends
    /// (`plateauContours(of:_:refine:)`); on a wall or a riser cut at its top
    /// and bottom edges as drawn, a riser's a hair onto its lower side, as its
    /// hatching stands. Copied to every tile where the faces are `tiled`.
    /// Kept where `restriction` holds at each vertex, a run cut where it does
    /// not, read as membership (`holds(_:at:onGraph:)`) at the point of the
    /// domain the vertex came from, before a copy moves it, as `derive` reads
    /// a band. Nothing for |f| and arg f, which `derive` draws, nor on a
    /// parametric surface's or a solid's faces, which a landscape has none of.
    func contours(of field: ContourField, _ levels: [Double], on faces: FaceSelector,
                  restriction: Restriction, in context: HatchContext,
                  refine: ContourRefine? = nil) -> [[P3<WorldSpace>]] {
        guard !field.isFieldOfF, !levels.isEmpty else { return [] }
        switch faces {
        case .graph(let tiled):
            return graphContours(of: field, levels, tiles: tiled ? context.tiles : [.identity],
                                 region: context.region, restriction: restriction, refine: refine)
        case .plateaus(let tiled):
            let placed = Surface.restricted(plateauContours(of: field, levels, refine: refine), restriction,
                                            onGraph: false) { $0.xy.v }
            return Surface.clip(tiled ? Surface.replicate(placed, context.tiles) : placed, to: context.region)
        case .walls(let edges, _, _, _):
            guard let perimeter = context.perimeter else { return [] }
            return wallContours(of: field, levels, perimeter, edges: edges, base: context.base,
                                tiles: context.tiles, restriction: restriction, refine: refine)
        case .risers(let tiled, _):
            return riserContours(of: field, levels, tiles: context.tiles, tiled: tiled,
                                 region: context.region, restriction: restriction, refine: refine)
        case .surface, .solid:
            return []
        }
    }

    /// Whether a contour on a face holds a point under `restriction`, read
    /// at `p`, the point of the domain it came from, as membership (D6). A
    /// face is cut to the footprint where it stands already, so `.footprint`
    /// holds all of it; `.belowTruncation` is membership in the graph, so it
    /// holds all of the graph and nothing of another face; a band holds what
    /// it bounds, open.
    static func holds(_ restriction: Restriction, at p: SIMD2<Double>, onGraph: Bool) -> Bool {
        switch restriction {
        case .all, .footprint: true
        case .belowTruncation: onGraph
        case .band(let axis, let lo, let hi): axis == .real ? p.x > lo && p.x < hi : p.y > lo && p.y < hi
        case .intersection(let all): all.allSatisfy { holds($0, at: p, onGraph: onGraph) }
        }
    }

    /// The runs of `paths` at whose every vertex `restriction` holds, read at
    /// the point of the domain `domain` carries it to.
    static func restricted(_ paths: [[P3<WorldSpace>]], _ restriction: Restriction, onGraph: Bool,
                           domain: (P3<WorldSpace>) -> SIMD2<Double>) -> [[P3<WorldSpace>]] {
        guard restriction != .all else { return paths }
        var out: [[P3<WorldSpace>]] = []
        for path in paths {
            var run: [P3<WorldSpace>] = []
            for v in path {
                if holds(restriction, at: domain(v), onGraph: onGraph) {
                    run.append(v)
                } else {
                    if run.count >= 2 { out.append(run) }
                    run = []
                }
            }
            if run.count >= 2 { out.append(run) }
        }
        return out
    }

    /// The contours of u = Re z, v = Im z or the height |f| on the graph,
    /// each copy's clipped to its trimmed domain (`Face.contours(of:_:on:)`):
    /// cut where they cross the truncation edge, the domain's edge or the
    /// footprint where the copy stands. Lifted to |f|, f's own with a
    /// refiner, restricted where they lie in the domain, and carried to the
    /// copy's tile.
    func graphContours(of field: ContourField, _ levels: [Double], tiles: [Affine2], region: Region,
                       restriction: Restriction = .all, refine: ContourRefine?) -> [[P3<WorldSpace>]] {
        guard let face = graph(tiles: tiles, region: region, refine: refine) else { return [] }
        let magnitude = faceMagnitude(refine)
        let scalar = ScalarField { p in
            switch field {
            case .u: p.x
            case .v: p.y
            case .height, .magnitude, .phase: magnitude(P2(p.x, p.y))
            }
        }
        return face.copies.flatMap { copy in
            let lifted = face.contours(of: scalar, levels, on: copy).flatMap(\.paths).map { path in
                path.map { p in P3<WorldSpace>(p.x, p.y, magnitude(P2(p.x, p.y))) }
            }
            return Surface.restricted(lifted, restriction, onGraph: true) { $0.xy.v }.map { $0.map { copy.tile($0) } }
        }
    }

    /// The contours of u, the distance along the edge, and of v or the
    /// height, both z, on the walls along `edges`: over each wall's parameter
    /// rectangle, from `base` up, cut at its trim loop, its bottom edge at
    /// `base`, its ends, and its top edge as `wallEdges` draws it
    /// (`topEdge(of:inward:tiles:refine:)`). A line of u is vertical, from
    /// the base to the top edge; one of the height horizontal, where the top
    /// edge stands over it.
    /// A restriction is read at the foot's point of the domain, through the
    /// tile under it.
    func wallContours(of field: ContourField, _ levels: [Double], _ perimeter: BoundaryPerimeter,
                      edges: [Int], base: Double, tiles: [Affine2], restriction: Restriction = .all,
                      refine: ContourRefine?) -> [[P3<WorldSpace>]] {
        let ceiling = truncation.highest.isFinite ? truncation.highest : highestSample
        var out: [[P3<WorldSpace>]] = []
        for index in edges where index >= 0 && index < perimeter.edges.count {
            let edge = perimeter.edges[index]
            let along = edge.end.v - edge.start.v, length = simd_length(along)
            guard length > 0 else { continue }
            let inward = perimeter.inward(index)
            let top = topEdge(of: edge, inward: inward, tiles: tiles, refine: refine).map {
                P2<ParamSpace>(simd_dot($0.xy.v - edge.start.v, along) / length, $0.z.isFinite ? min($0.z, ceiling) : ceiling)
            }
            let placed = verticalFaceContours(of: field, levels, spacing: length / Double(max(edge.density, 2) - 1),
                                        bottom: [P2(0, base), P2(length, base)], top: top) { s, z in
                let p = edge.start.v + along * (s / length)
                return P3(p.x, p.y, z)
            }
            out += Surface.restricted(placed, restriction, onGraph: false) { v in
                let t = simd_dot(v.xy.v - edge.start.v, along) / (length * length)
                return wallTile(at: P2(v.x, v.y), t: t, along: along, inward: inward, tiles: tiles).point.v
            }
        }
        return out
    }

    /// The contours of u, the distance along the line a riser stands on,
    /// and of v or the height, both z, on the risers where they stand
    /// (`riserStretches(tiles:tiled:region:refine:)`): over each stretch, cut
    /// at its trim loop, its top and bottom edges as the truncation edge's
    /// layer draws them, and its ends; each vertex a hair onto the lower side,
    /// as the riser's hatching and bottom edge stand. A restriction is read
    /// at the point of the domain under it, through the tile of the copy the
    /// riser faces there.
    func riserContours(of field: ContourField, _ levels: [Double], tiles: [Affine2], tiled: Bool,
                       region: Region, restriction: Restriction = .all,
                       refine: ContourRefine?) -> [[P3<WorldSpace>]] {
        let side = 1e-9 * cell
        var out: [[P3<WorldSpace>]] = []
        for riser in riserStretches(tiles: tiles, tiled: tiled, region: region, refine: refine) {
            let start = riser.point(0).v, d = riser.point(1).v - start
            let dd = simd_dot(d, d)
            guard dd > 0 else { continue }
            func onLine(_ v: P3<WorldSpace>) -> P2<ParamSpace> {
                P2(simd_dot(v.xy.v - start, d) / dd * riser.length, v.z)
            }
            let placed = verticalFaceContours(of: field, levels, spacing: cell,
                                              bottom: riser.bottom.map(onLine), top: riser.top.map(onLine)) { s, z in
                let w = riser.point(s / riser.length)
                let o = side * riser.lower(w)
                return P3(w.x + o.x, w.y + o.y, z)
            }
            out += Surface.restricted(placed, restriction, onGraph: false) { v in
                riser.back(P2(v.x, v.y))(P3(v.x, v.y, 0)).xy.v
            }
        }
        return out
    }

    /// The contours of u = s or v = height = z on a vertical face between
    /// two curves of (s, z), `bottom` and `top`, each in order along s: over
    /// the rectangle the two span, sampled about every `spacing` along s, cut
    /// where they cross the loop the two close, and placed in space by
    /// `point`.
    func verticalFaceContours(of field: ContourField, _ levels: [Double], spacing: Double,
                              bottom: [P2<ParamSpace>], top: [P2<ParamSpace>],
                              point: (Double, Double) -> P3<WorldSpace>) -> [[P3<WorldSpace>]] {
        guard let b0 = bottom.first, bottom.count >= 2, top.count >= 2 else { return [] }
        let loop = bottom + top.reversed() + [b0]
        let s = loop.map(\.x), z = loop.map(\.y)
        guard let s0 = s.min(), let s1 = s.max(), let z0 = z.min(), let z1 = z.max(),
              s1 > s0, z1 > z0, z0.isFinite, z1.isFinite else { return [] }
        let rectangle = (ParamAxis(Interval(lo: s0, hi: s1), periodic: false,
                                   samples: max(Int(((s1 - s0) / spacing).rounded(.up)) + 1, 2)),
                         ParamAxis(Interval(lo: z0, hi: z1), periodic: false, samples: 2))
        let scalar = ScalarField { field == .u ? $0.x : $0.y }
        return Contour.levels(of: scalar, levels, over: rectangle, trimLoops: [loop]).flatMap(\.paths).map {
            $0.map { point($0.x, $0.y) }
        }
    }
}
