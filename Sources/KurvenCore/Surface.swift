import Foundation
import simd

/// The landscape: |f| on a grid, plus how it is truncated.
///
/// `height` is unclamped, and the truncation is a separate value. That separation is
/// what lets a consumer change the truncation without resampling -- the truncation is a
/// drawing decision about how much of a pole to show, and the samples do not
/// depend on it.
public struct Surface: Sendable {
    public let height: Grid2D<Float>
    public let phase: Grid2D<Float>?
    public let truncation: Truncation
    /// True when the samples came from a cache rather than an evaluator, in
    /// which case the Python side looks heights up by nearest pixel and this
    /// side must too, or the walls sit at a different top edge than the drawing's.
    public let cached: Bool

    public init(height: Grid2D<Float>, phase: Grid2D<Float>?, truncation: Truncation,
                cached: Bool = false) {
        self.height = height; self.phase = phase; self.truncation = truncation; self.cached = cached
    }

    public var domain: Domain { height.domain }

    /// The clamped heightfield -- the surface the occluder is meshed from.
    public func clamped() -> Grid2D<Float> {
        switch truncation {
        case .none:
            return height
        case .uniform(let z):
            let truncationHeight = Float(z)
            return Grid2D(width: height.width, height: height.height,
                          domain: height.domain, values: height.values.map { min($0, truncationHeight) })
        case .realBands:
            var out = height.values
            for x in 0..<height.width {
                let truncationHeight = Float(self.truncationHeight(atX: height.position(x: x, y: 0).x))
                for y in 0..<height.height { out[y * height.width + x] = min(out[y * height.width + x], truncationHeight) }
            }
            return Grid2D(width: height.width, height: height.height,
                          domain: height.domain, values: out)
        }
    }

    /// |f| at an arbitrary domain point, unclamped.
    public func magnitude(at p: P2<DomainSpace>) -> Double {
        cached ? height.nearest(p) : height.sample(p)
    }

    /// The height a wall rises to: `magnitude`, truncated. `Surface.height_at`.
    public func height(at p: P2<DomainSpace>) -> Double {
        min(magnitude(at: p), truncationHeight(atX: p.x))
    }

    /// The truncation height over Re z = x on the landscape: the band x lies
    /// in (`Truncation.height(atX:)`), but on the domain's right edge the
    /// one-sided limit from the left, from inside the domain. A band that
    /// starts on the right edge lies beyond the domain, and nothing of the
    /// landscape stands in it: the graph, the plateaus, the depth pass and
    /// the right wall end with the band before it, and there is no riser
    /// there.
    ///
    /// Elsewhere the limit is taken from `side`, the right unless a wall
    /// reads it: a wall on a band boundary reads the band on its
    /// footprint's side (`Surface.footprintSide`).
    ///
    /// The right edge is read at the domain's own end, not at the grid's
    /// last column, which its spacing can put a rounding past it.
    public func truncationHeight(atX x: Double, from side: Truncation.Side = .right) -> Double {
        guard case .realBands = truncation else { return truncation.height(atX: x) }
        return isOnRightEdge(x) ? truncation.height(atX: rightEdge, from: .left) : truncation.height(atX: x, from: side)
    }

    /// The band over Re z = x on the landscape, as `truncationHeight(atX:from:)`
    /// reads it: on the domain's right edge, the band before it.
    public func band(atX x: Double, from side: Truncation.Side = .right) -> Int? {
        guard case .realBands = truncation else { return truncation.band(atX: x) }
        return isOnRightEdge(x) ? truncation.band(atX: rightEdge, from: .left) : truncation.band(atX: x, from: side)
    }

    /// Re z of the domain's right edge, its greater end.
    var rightEdge: Double { max(domain.real.lo, domain.real.hi) }

    /// Whether Re z = x is the domain's right edge: its greater end, or the
    /// grid's last column there, which its spacing can put a rounding off it.
    func isOnRightEdge(_ x: Double) -> Bool {
        let real = domain.real
        guard real.lo <= real.hi else { return x == real.lo }
        return x == real.hi || x == height.position(x: height.width - 1, y: 0).x
    }

    /// Every `step`-th sample, truncated and lifted, without materializing a grid.
    ///
    /// The bounds pass folds over three million points on elliptic and six
    /// million on zeta; it wants none of them kept. `clamped().decimated()`
    /// would allocate both intermediates to hand them over one at a time.
    public func forEachSample(step: Int, _ body: (P3<WorldSpace>) -> Void) {
        let g = height
        let columns = Array(stride(from: 0, to: g.width, by: max(step, 1)))
        // The truncation depends only on x, so it is one lookup per sampled column
        // rather than one per sample -- which is the whole difference between
        // `Truncation.realBands` costing nothing and costing a branch per point.
        let columnTruncation = columns.map { truncationHeight(atX: g.position(x: $0, y: 0).x) }
        for y in stride(from: 0, to: g.height, by: max(step, 1)) {
            for (i, x) in columns.enumerated() {
                let p = g.position(x: x, y: y)
                body(P3(p.x, p.y, min(Double(g[x, y]), columnTruncation[i])))
            }
        }
    }

    /// The height of a *tiled* landscape at a world point.
    ///
    /// The grid covers one fundamental tile; the drawing covers its images under
    /// the occluder's affine maps. A point out on the drawing is therefore not in
    /// the grid, and asking the grid about it returns whatever is nearest the
    /// edge -- which for elliptic, whose cutout runs across all nineteen tiles
    /// and whose tile edge is a pole, is a spire-high wall standing along the
    /// whole boundary.
    ///
    /// The point is mapped back through the tile it belongs to, chosen as the
    /// *nearest* rather than the containing one. Containment is too strict to be
    /// useful here: elliptic samples its tile on `[-K + eps, -eps]` to keep off
    /// the poles, so the cutout perimeter -- drawn at the exact quarter-periods
    /// -- lies a hair outside every tile image, and an exact test matches
    /// nothing at all.
    func height(at p: P2<DomainSpace>, tiles: [Affine2]) -> Double {
        height(at: inTile(p, tiles: tiles))
    }

    /// |f| at a world point of a tiled landscape, untruncated: `height(at:tiles:)`
    /// before the truncation.
    func magnitude(at p: P2<DomainSpace>, tiles: [Affine2]) -> Double {
        magnitude(at: inTile(p, tiles: tiles))
    }

    /// The outward normal of the truncated surface over a world point of a
    /// tiled landscape: `(-∂h/∂x, -∂h/∂y, 1)`, not normalized, by central
    /// differences a grid spacing wide, the height read through whichever
    /// tile the point is in. The differences run along the grid's own axes
    /// as the tile under the point carries them into place, a cell of each
    /// axis's own length, and the slope they give is carried back into
    /// world coordinates: so a copy's normal is the copy of the normal, a
    /// reflected copy's turned with it, and a turned copy's read off the
    /// grid's chords, not a cell of one axis's length along the other. On a
    /// plateau the differences vanish and the normal is up.
    ///
    /// With the function (`refine`), the slope is f's own, read by
    /// differences a step small against the cell and large against
    /// rounding, as the refiner reads it: the curves on the surface are placed
    /// on f to the last bit, and a facing read off the grid a cell wide
    /// tests it with a ruler a cell long -- a ring ends a stencil's width
    /// short of the fold on a convex graph and shows a stencil's width past
    /// it on a hump, whatever the grid. Without the function, the grid
    /// is the surface and its chords are the slope.
    func normal(at p: P2<WorldSpace>, tiles: [Affine2], refine: ContourRefine?) -> SIMD3<Double> {
        let g = height
        let dx = abs(g.domain.real.length) / Double(max(g.width - 1, 1))
        let dy = abs(g.domain.imag.length) / Double(max(g.height - 1, 1))
        if let magnitude = refine?.magnitude {
            let step = 1e-4 * max(dx, dy)
            func h(_ x: Double, _ y: Double) -> Double {
                let q = inTile(P2<DomainSpace>(x, y), tiles: tiles)
                let u = magnitude(q)
                return min(u.isNaN ? .infinity : u, truncationHeight(atX: q.x))
            }
            let hx = (h(p.x + step, p.y) - h(p.x - step, p.y)) / (2 * step)
            let hy = (h(p.x, p.y + step) - h(p.x, p.y - step)) / (2 * step)
            if hx.isFinite, hy.isFinite { return SIMD3(-hx, -hy, 1) }
        }
        func h(_ q: SIMD2<Double>) -> Double {
            height(at: P2<DomainSpace>(q.x, q.y), tiles: tiles)
        }
        // The grid's axes in place, and the slope along each.
        let back = tile(under: P2(p.x, p.y), tiles: tiles).back
        let forward = back.inverse ?? .identity
        let u = SIMD2(forward.a, forward.c) * dx, v = SIMD2(forward.b, forward.d) * dy
        let hu = (h(p.v + u) - h(p.v - u)) / (2 * dx)
        let hv = (h(p.v + v) - h(p.v - v)) / (2 * dy)
        return SIMD3(-(back.a * hu + back.c * hv), -(back.b * hu + back.d * hv), 1)
    }

    /// The normal of the graph over `p`: f's own slope, untruncated, read as
    /// `normal(at:tiles:refine:)` reads it, so a fold is traced on the one
    /// surface the graph is and continues below the truncation, where the
    /// truncation edge cuts it. The truncated normal without the function,
    /// or where f's slope is not finite.
    func graphNormal(at p: P2<WorldSpace>, tiles: [Affine2], refine: ContourRefine?) -> SIMD3<Double> {
        guard let magnitude = refine?.magnitude else { return normal(at: p, tiles: tiles, refine: refine) }
        let g = height
        let dx = abs(g.domain.real.length) / Double(max(g.width - 1, 1))
        let dy = abs(g.domain.imag.length) / Double(max(g.height - 1, 1))
        let step = 1e-4 * max(dx, dy)
        func h(_ x: Double, _ y: Double) -> Double {
            let u = magnitude(inTile(P2<DomainSpace>(x, y), tiles: tiles))
            return u.isNaN ? .infinity : u
        }
        let hx = (h(p.x + step, p.y) - h(p.x - step, p.y)) / (2 * step)
        let hy = (h(p.x, p.y + step) - h(p.x, p.y - step)) / (2 * step)
        guard hx.isFinite, hy.isFinite else { return normal(at: p, tiles: tiles, refine: refine) }
        return SIMD3(-hx, -hy, 1)
    }

    /// The point of the fundamental tile a world point is the image of.
    func inTile(_ p: P2<DomainSpace>, tiles: [Affine2]) -> P2<DomainSpace> {
        tile(under: p, tiles: tiles).point
    }

    /// The tile under a world point: the point of the fundamental tile it is
    /// the image of (`inTile(_:tiles:)`), and the inverse of the tile, which
    /// carries it back. The identity for one tile, or where none is
    /// invertible.
    func tile(under p: P2<DomainSpace>, tiles: [Affine2]) -> (point: P2<DomainSpace>, back: Affine2) {
        guard tiles.count > 1 else { return (p, .identity) }
        let d = domain
        let (xlo, xhi) = (min(d.real.lo, d.real.hi), max(d.real.lo, d.real.hi))
        let (ylo, yhi) = (min(d.imag.lo, d.imag.hi), max(d.imag.lo, d.imag.hi))

        var best: (point: P2<DomainSpace>, back: Affine2)?
        var bestDistance = Double.infinity
        for tile in tiles {
            guard let back = tile.inverse else { continue }
            let q = back(P3<WorldSpace>(p.x, p.y, 0))
            // How far outside the domain box this tile puts the point; zero
            // when it is inside.
            let dx = max(xlo - q.x, 0) + max(q.x - xhi, 0)
            let dy = max(ylo - q.y, 0) + max(q.y - yhi, 0)
            let distance = dx + dy
            if distance < bestDistance {
                bestDistance = distance
                best = (P2<DomainSpace>(q.x, q.y), back)
                if distance == 0 { break }
            }
        }
        return best ?? (p, .identity)
    }

    /// A point of the domain's rectangle, or a rounding off it: taken into
    /// it, and onto an edge within `edgeTolerance` of one. A point of a tile
    /// boundary carried back through either tile lands a rounding either side
    /// of the domain's edge, and the far edge reads the band before it only
    /// on the edge itself.
    func ontoDomain(_ q: P2<WorldSpace>) -> P2<DomainSpace> {
        let d = domain, tolerance = edgeTolerance
        func onto(_ v: Double, _ a: Double, _ b: Double) -> Double {
            let (lo, hi) = (min(a, b), max(a, b))
            let c = min(max(v, lo), hi)
            return abs(c - lo) <= tolerance ? lo : abs(c - hi) <= tolerance ? hi : c
        }
        return P2(onto(q.x, d.real.lo, d.real.hi), onto(q.y, d.imag.lo, d.imag.hi))
    }

    /// Where two tiles' copies of the domain meet edge to edge: the stretch
    /// of space the two images share along a side of each, with the copies
    /// on either side of it. Each pair of invertible tiles meets along one
    /// at most, their images being convex and apart; tiles whose images
    /// overlap, or touch at a corner, meet along none.
    struct TileBoundary: Sendable {
        /// The two tiles, by their place in the list, the lesser first.
        let first: Int, second: Int
        /// Where the stretch begins and ends: from its end with the lesser x,
        /// or y where they agree, so the tile boundary is the same whichever
        /// tile is listed first.
        let start: P2<WorldSpace>, end: P2<WorldSpace>

        var length: Double { simd_length(end.v - start.v) }
        /// The point `s` along the tile boundary from its start, keeping a
        /// tile boundary's x or y where it runs along an axis.
        func point(_ s: Double) -> P2<WorldSpace> {
            let t = s / length
            return P2(start.x == end.x ? start.x : start.x + (end.x - start.x) * t,
                      start.y == end.y ? start.y : start.y + (end.y - start.y) * t)
        }
        /// The unit normal to the left of the tile boundary's direction.
        var normal: SIMD2<Double> {
            let d = (end.v - start.v) / length
            return SIMD2(-d.y, d.x)
        }
    }

    /// The tile boundaries between `tiles`' copies of the domain, in the
    /// order of their pairs of tiles. A copy's side and another's lie along
    /// one line within `edgeTolerance`, overlapping by more than that, with
    /// the copies' centres either side.
    func tileBoundaries(tiles: [Affine2]) -> [TileBoundary] {
        guard tiles.count > 1 else { return [] }
        let d = domain
        let (xlo, xhi) = (min(d.real.lo, d.real.hi), max(d.real.lo, d.real.hi))
        let (ylo, yhi) = (min(d.imag.lo, d.imag.hi), max(d.imag.lo, d.imag.hi))
        let corners = [SIMD2(xlo, ylo), SIMD2(xhi, ylo), SIMD2(xhi, yhi), SIMD2(xlo, yhi)]
        let centre = 0.5 * SIMD2(xlo + xhi, ylo + yhi)
        func image(_ tile: Affine2, _ p: SIMD2<Double>) -> SIMD2<Double> { tile(P3<WorldSpace>(p.x, p.y, 0)).xy.v }
        let tolerance = edgeTolerance
        var out: [TileBoundary] = []
        for i in tiles.indices where tiles[i].inverse != nil {
            for j in tiles.indices where j > i && tiles[j].inverse != nil {
                let ci = image(tiles[i], centre), cj = image(tiles[j], centre)
                sides: for e in 0..<4 {
                    let a = image(tiles[i], corners[e]), b = image(tiles[i], corners[(e + 1) % 4])
                    let length = simd_length(b - a)
                    guard length > tolerance else { continue }
                    let u = (b - a) / length, n = SIMD2(-u.y, u.x)
                    guard simd_dot(ci - a, n) * simd_dot(cj - a, n) < 0 else { continue }
                    for f in 0..<4 {
                        let p = image(tiles[j], corners[f]), q = image(tiles[j], corners[(f + 1) % 4])
                        guard abs(simd_dot(p - a, n)) <= tolerance, abs(simd_dot(q - a, n)) <= tolerance else { continue }
                        let sp = simd_dot(p - a, u), sq = simd_dot(q - a, u)
                        let lo = max(0, min(sp, sq)), hi = min(length, max(sp, sq))
                        guard hi - lo > tolerance else { continue }
                        // From the end with the lesser x, or y, whichever
                        // tile's side it is read along.
                        let ends = [lo == 0 ? a : a + u * lo, hi == length ? b : a + u * hi]
                            .sorted { ($0.x, $0.y) < ($1.x, $1.y) }
                        out.append(TileBoundary(first: i, second: j, start: P2(ends[0]), end: P2(ends[1])))
                        break sides
                    }
                }
            }
        }
        return out
    }

    /// The top of the copy on a tile over a point of a tile boundary, `back`
    /// the tile's inverse: min(|f|, h(x)) at the point of the domain's edge
    /// the tile boundary point is the copy of, `magnitude` reading |f| there.
    func tileBoundaryTop(_ w: P2<WorldSpace>, back: Affine2, magnitude: (P2<DomainSpace>) -> Double) -> Double {
        let q = ontoDomain(back(P3(w.x, w.y, 0)).xy)
        return min(magnitude(q), truncationHeight(atX: q.x))
    }

    /// Whether a copy's top over a tile boundary stands over another's: by
    /// more than a billionth of the lower, a rounding of the point two tiles
    /// carry back, as copies reflected across the tile boundary do.
    static func stands(_ higher: Double, over lower: Double) -> Bool {
        higher > lower + 1e-9 * max(1, abs(lower))
    }

    /// The height a wall rises to at its foot `p`, `t` of the way along its
    /// edge from `a` to `b`, `inward` its normal into the footprint: the
    /// truncated surface over the foot, read through the tile under it, as
    /// `height(at:tiles:)` reads it, but with the truncation's one-sided
    /// limit from the footprint's side (`Surface.footprintSide`).
    func wallHeight(at p: P2<DomainSpace>, t: Double, from a: P2<WorldSpace>, to b: P2<WorldSpace>,
                    inward: SIMD2<Double>, tiles: [Affine2], through: Affine2? = nil) -> Double {
        let foot = wallFoot(at: p, t: t, along: b.v - a.v, inward: inward, tiles: tiles, through: through)
        return min(magnitude(at: foot.point), foot.truncationHeight)
    }

    /// A wall's foot `p`, `t` of the way along its edge, `along` the edge's
    /// direction and `inward` its normal into the footprint, as the domain
    /// reads it: the point of the domain the foot is the image of, carried
    /// back through the tile its footprint is in (`wallTile`), and the
    /// truncation over it, read at its x there (`wallFootX`) from the
    /// footprint's side (`Surface.footprintSide`). Every reader of a wall's
    /// top reads it so: the curtain, the top edge, the corners and the wall
    /// face.
    ///
    /// A foot where the wall crosses a tile boundary stands on two copies,
    /// and the top edge reads it through each (`through`): carried back by
    /// that tile, onto the domain's edge.
    func wallFoot(at p: P2<DomainSpace>, t: Double, along: SIMD2<Double>, inward: SIMD2<Double>,
                  tiles: [Affine2], through: Affine2? = nil) -> (point: P2<DomainSpace>, truncationHeight: Double) {
        let (q, back) = through.map { back in (ontoDomain(back(P3(p.x, p.y, 0)).xy), back) }
            ?? wallTile(at: p, t: t, along: along, inward: inward, tiles: tiles)
        let side = Surface.footprintSide(at: t, along: along, inward: inward, back: back)
        return (q, truncationHeight(atX: wallFootX(q.x, at: t, along: along, back: back), from: side))
    }

    /// The tile a wall's foot `p` is read through, and the foot carried back
    /// through it: the tile under the footprint beside the foot, a billionth
    /// of a cell in along `inward`, and at an end of the wall along the wall
    /// as well. A wall on a tile boundary stands on both tiles; it bounds the copy its footprint is in, and the tile under the foot
    /// itself would be whichever comes first.
    func wallTile(at p: P2<DomainSpace>, t: Double, along: SIMD2<Double>, inward: SIMD2<Double>,
                  tiles: [Affine2]) -> (point: P2<DomainSpace>, back: Affine2) {
        guard tiles.count > 1 else { return (p, .identity) }
        let length = simd_length(along)
        let into = length > 0 ? along / length * (t == 0 ? 1 : t == 1 ? -1 : 0) : .zero
        let back = tile(under: P2(p.v + edgeTolerance * (inward + into)), tiles: tiles).back
        let q = back(P3(p.x, p.y, 0))
        return (P2(q.x, q.y), back)
    }

    /// Re z of a wall's foot carried back into the domain by `back`, taken
    /// onto a band boundary within `edgeTolerance` of it where the wall
    /// runs parallel to the imaginary axis, as one standing on a boundary
    /// does, or ends: carried back through a tile, a foot
    /// on the boundary's image lands a rounding either side of it (the wall
    /// at 4.6 under x ↦ 5 - x comes back at 0.40000000000000036), and would
    /// read the band on whichever side it fell before its footprint's side
    /// could count. A foot between the ends of a wall that crosses a
    /// boundary keeps its x, so the wall steps where it crosses.
    func wallFootX(_ x: Double, at t: Double, along: SIMD2<Double>, back: Affine2) -> Double {
        guard case .realBands(let bands, _) = truncation,
              t == 0 || t == 1 || Surface.isParallelToImaginaryAxis(along, back: back) else { return x }
        let tolerance = edgeTolerance
        return bands.lazy.map(\.below).first { abs(x - $0) <= tolerance } ?? x
    }

    /// Whether an edge running along `along` is parallel to the imaginary
    /// axis once carried back into the domain by `back`: the carried
    /// direction's part along Re z is a rounding of its length. Not exactly
    /// zero: a tile written from cos and sin turns a quarter with cos(π/2) =
    /// 6.1e-17, and a wall on a boundary's image there carries back
    /// crossing it at that slope.
    static func isParallelToImaginaryAxis(_ along: SIMD2<Double>, back: Affine2) -> Bool {
        let carried = SIMD2(back.a * along.x + back.b * along.y, back.c * along.x + back.d * along.y)
        return abs(carried.x) <= 1e-12 * simd_length(carried)
    }

    /// The side of Re z a wall's foot reads the truncation from, at `t` of
    /// the way along its edge, `along` the edge's direction and `inward` its
    /// normal into the footprint, both carried back into the domain by
    /// `back`: the side the footprint is on. Along a wall parallel to the
    /// imaginary axis, as one on a band boundary is, that is the way `inward`
    /// points; at an end of a wall that is not, the way the wall goes, so its top runs on to its end in its own band. Elsewhere, on a
    /// wall that crosses a boundary, the right, as `Truncation.height(atX:)`
    /// reads it, so the wall steps there. The sides differ only on a
    /// boundary.
    static func footprintSide(at t: Double, along: SIMD2<Double>, inward: SIMD2<Double>,
                              back: Affine2) -> Truncation.Side {
        func carried(_ d: SIMD2<Double>) -> Double { back.a * d.x + back.b * d.y }
        let across = carried(along)
        let x = isParallelToImaginaryAxis(along, back: back) ? carried(inward) : t == 0 ? across : t == 1 ? -across : 0
        return x < 0 ? .left : .right
    }

    /// The height as the depth pass draws it over a domain point: the grid
    /// decimated by `step`, each cell as the two triangles the rasterizer
    /// splits it into -- from corner (1, 0) to corner (0, 1) -- then truncated.
    /// `height(at:)` interpolates bilinearly, which is what the Python plate
    /// placed its curves by and the fixtures pin; this is what the curves are drawn
    /// against, and the two differ within a cell by its twist.
    func drawnHeight(at p: P2<DomainSpace>, step: Int) -> Double {
        let drawn = drawnSurface(at: p, step: step)
        return min(drawn.interpolated, drawn.truncationHeight)
    }

    /// `drawnHeight` as the two numbers it is the lesser of -- the grid's
    /// interpolated height and the truncation over the point -- with the cosine of
    /// the interpolating triangle's tilt, `1 / sqrt(1 + |∇h|²)`, which turns a
    /// vertical gap to that triangle into a perpendicular distance.
    func drawnSurface(at p: P2<DomainSpace>, step: Int)
        -> (interpolated: Double, truncationHeight: Double, cosine: Double)
    {
        let g = height
        let step = max(step, 1)
        let nx = (g.width + step - 1) / step, ny = (g.height + step - 1) / step
        let truncationHeight = self.truncationHeight(atX: p.x)
        guard nx >= 2, ny >= 2 else { return (Double(g[0, 0]), truncationHeight, 1) }
        let dx = abs(g.domain.real.length) / Double(max(g.width - 1, 1)) * Double(step)
        let dy = abs(g.domain.imag.length) / Double(max(g.height - 1, 1)) * Double(step)
        let f = g.index(of: p) / Double(step)
        let fx = min(max(f.x, 0), Double(nx - 1)), fy = min(max(f.y, 0), Double(ny - 1))
        let i = min(Int(fx), nx - 2), j = min(Int(fy), ny - 2)
        let u = fx - Double(i), v = fy - Double(j)
        let h00 = Double(g[i * step, j * step]), h10 = Double(g[(i + 1) * step, j * step])
        let h01 = Double(g[i * step, (j + 1) * step]), h11 = Double(g[(i + 1) * step, (j + 1) * step])
        let lower = u + v <= 1
        let h = lower
            ? h00 + u * (h10 - h00) + v * (h01 - h00)
            : h11 + (1 - u) * (h01 - h11) + (1 - v) * (h10 - h11)
        let hx = (lower ? h10 - h00 : h11 - h01) / dx
        let hy = (lower ? h01 - h00 : h11 - h10) / dy
        let cosine = 1 / (1 + hx * hx + hy * hy).squareRoot()
        return (h.isNaN ? .infinity : h, truncationHeight, cosine.isNaN ? 0 : cosine)
    }

    /// Lift a domain point onto the (truncated) surface.
    public func lift(_ p: P2<DomainSpace>) -> P3<WorldSpace> {
        P3(p.x, p.y, height(at: p))
    }
}

// MARK: - deriving curves from the grids

/// What a consumer that has the function supplies, so the curves can be placed by
/// f rather than by the grid. Core has the grids and not the function;
/// `KurvenLandscape`'s `ContourRefiner` builds one of these.
public struct ContourRefine: Sendable {
    /// Moves the paths of one level of one field onto the function's own
    /// level set, given the field, the level, the marching-squares paths in
    /// domain space, and the ceiling: the height above which the curves are cut
    /// away (`Restriction.belowTruncation`), so a stretch of a phase line wholly above it
    /// need not be placed. `.infinity` places all of it. It is how |f| and
    /// arg f place their contours as scalar fields: `Surface.field(of:refine:ceiling:)`
    /// builds each a `place` that calls this with the tag and the ceiling
    /// closed over. It is also how the truncation edge is placed.
    public typealias Contours = @Sendable (ContourField, Double, [[P2<DomainSpace>]], Double)
        -> [[P2<DomainSpace>]]
    /// |f| at a domain point, untruncated, as the grid would hold it.
    public typealias Magnitude = @Sendable (P2<DomainSpace>) -> Double
    /// A zero of f within a few cells of a domain point, or nil.
    public typealias Zero = @Sendable (P2<DomainSpace>) -> P2<DomainSpace>?

    public let contours: Contours
    /// With it, the top edge of a cut face gains a vertex at each minimum of |f|
    /// between two grid nodes (`Surface.topEdge`), and a contour that ends on a
    /// zero of f ends at the zero's height (`Surface.derive`). Nil for a
    /// consumer that can place contours and not heights.
    public let magnitude: Magnitude?
    /// With it, a fold curve that ends beside a zero of f is carried to the
    /// zero (`Heightfield.foldCurves`), as the refiner carries the phase lines.
    public let zero: Zero?
    /// How far a chord may depart from the curve it stands for, in cells:
    /// the refiner's own tolerance, which the top edge of a cut face and the
    /// fold curves are subdivided to as well, so every curve of the drawing
    /// is a curve to the same tolerance.
    public let tolerance: Double
    /// The same refiner with nothing shared: f compiled again, into closures
    /// of its own. Each thread of a parallel visibility test takes one
    /// (`Heightfield.parallelMap(_:chunk:worker:_:)`): a closure's context is
    /// a reference-counted object, and with every core calling the same one
    /// the counts were the cores writing one cache line in turn -- at
    /// eighteen threads a traversal cost twice its single-threaded time, and
    /// forked, a third more. Nil, a refiner that cannot be forked, is shared.
    public let fork: (@Sendable () -> ContourRefine)?

    public init(contours: @escaping Contours, magnitude: Magnitude? = nil,
                zero: Zero? = nil, tolerance: Double = 0.02,
                fork: (@Sendable () -> ContourRefine)? = nil) {
        self.tolerance = tolerance
        self.contours = contours; self.magnitude = magnitude; self.zero = zero
        self.fork = fork
    }

    /// A copy for one thread: `fork`'s, or this one.
    public func forked() -> ContourRefine { fork?() ?? self }
}

public extension Surface {
    /// Lift a domain-space path onto the surface, by policy.
    ///
    /// `Surface.lift_contours`' three height policies, and the reason a bundle
    /// records which one a layer used: a magnitude isocontour sits at exactly
    /// its own level, a phase contour sits wherever the surface is, and an
    /// unclamped lift is a third thing again. A consumer regenerating a layer
    /// cannot guess which was meant.
    func lift(_ path: some Sequence<P2<DomainSpace>>, policy: HeightPolicy,
              level: Double) -> [P3<WorldSpace>] {
        path.map { p in
            switch policy {
            case .surface: P3(p.x, p.y, height(at: p))
            case .level: P3(p.x, p.y, level)
            case .magnitude: P3(p.x, p.y, magnitude(at: p))
            }
        }
    }

    /// Whether a lifted vertex survives a `Restriction`.
    func admits(_ p: P3<WorldSpace>, _ restriction: Restriction, region: Region) -> Bool {
        switch restriction {
        case .all: true
        case .footprint: region.contains(p.xy)
        case .belowTruncation: p.z <= truncationHeight(atX: p.x)
        case .band(let axis, let lo, let hi):
            switch axis {
            case .real: p.x > lo && p.x < hi
            case .imag: p.y > lo && p.y < hi
            }
        case .intersection(let all): all.allSatisfy { admits(p, $0, region: region) }
        }
    }

    /// Where the segment from `a`, below the truncation, to `b`, over it, reaches the
    /// truncation -- on the truncation edge, at exactly the truncation height. Nil when it does not,
    /// or when there is no truncation.
    ///
    /// Solved on the lifted heights, linearly: that is the same interpolation
    /// the truncation edge is contoured by, so the crossing lands on the drawn truncation edge to
    /// within a cell's bend.
    func truncationCrossing(from a: P3<WorldSpace>, to b: P3<WorldSpace>) -> P3<WorldSpace>? {
        let ea = a.z - truncationHeight(atX: a.x), eb = b.z - truncationHeight(atX: b.x)
        guard ea.isFinite, eb.isFinite, ea <= 0, eb > 0 else { return nil }
        let t = ea / (ea - eb)
        let x = a.x + (b.x - a.x) * t
        return P3(x, a.y + (b.y - a.y) * t, truncationHeight(atX: x))
    }

    /// |f| or arg f, as `tag` names it, as a scalar field on the graph,
    /// contoured on its grid and placed on f by `refine`
    /// (`ContourRefine.Contours`), which is handed the tag and `ceiling`, the
    /// height above which a layer's curves are cut away. Without a refiner
    /// the grid's paths are the contours. Nil for arg f on a landscape with
    /// no phase grid.
    ///
    /// arg f is circle-valued and has no period here: the refiner owns its
    /// branch cut, the level ±π. Beside a zero of f a grid edge can span
    /// more than π of phase, which the refiner keeps by testing the edge's
    /// midpoint on f, and the rule a circle-valued field is contoured by --
    /// no crossing on an edge whose samples differ by more than half a
    /// period -- would drop it. So the refiner is handed the grid's own
    /// paths, the spurious crossings along the wrap included, and at the
    /// branch cut the zero level of arg(-f) (`Contour.cut`), which it places
    /// there. Without a refiner the branch cut has no contour: nothing would
    /// drop the wrap's crossings.
    func field(of tag: ContourField, refine: ContourRefine?, ceiling: Double = .infinity)
        -> ScalarField? {
        let samples: Grid2D<Float>
        switch tag {
        case .magnitude: samples = height
        case .phase:
            guard let phase else { return nil }
            samples = phase
        }
        return ScalarField(samples: samples) { level, paths in
            let isCut = tag == .phase && Contour.isCut(level)
            guard let refine else { return isCut ? [] : paths }
            let lines = isCut ? Contour.cut(of: samples) : paths.map { $0.map { P2<DomainSpace>($0.v) } }
            return refine.contours(tag, level, lines, ceiling).map { $0.map { P2<ParamSpace>($0.v) } }
        }
    }

    /// Every level of a described layer, contoured, lifted, filtered and tiled.
    ///
    /// A path that leaves the kept region and comes back is split where it left,
    /// not closed across the gap. Filtering vertices and keeping the survivors
    /// as one polyline welds the far side of a contour to the near side; on the
    /// zeta plate that drew straight chords up to half the width of the domain
    /// across ground the cutout had deliberately removed.
    ///
    /// Where a path leaves by going *over the truncation*, the split is at the truncation
    /// itself: the run gains a vertex on the truncation edge (`truncationCrossing`) rather than
    /// ending at its last vertex under it, which on a pole's graph is up to
    /// half a world unit short. Only with a refiner: without one the curves are
    /// exactly what Python derives from the same grids, which ends at the
    /// last sample, and the fixtures hold it to that.
    ///
    /// The contours are the field's (`field(of:refine:ceiling:)`), asked for
    /// through `Contour.levels(of:_:over:trimLoops:contains:cuts:)` as any
    /// field's are, over the graph's parameter rectangle
    /// (`parameterRectangle`).
    /// Not over its trimmed domain: the restriction says which part of the
    /// domain a layer keeps, and a run is carried to the truncation edge by
    /// `truncationCrossing`, as below. A layer kept to the footprint is cut
    /// to it exactly, a tiled layer's copy on each tile where it lands. A phase level of ±π is the branch cut,
    /// where arg f jumps, and the grid has no contour at it. With a refiner
    /// it is drawn once, after the other levels, as one line of curves the
    /// way the Jahnke-Emde plates draw it.
    func derive(_ source: LayerSource, policy: HeightPolicy, region: Region,
                tiles: [Affine2], refine: ContourRefine? = nil) -> PolylineSet<WorldSpace> {
        guard case .contour(let tag, let levels, let restriction, let tiled) = source,
              let rectangle = parameterRectangle else {
            return .empty
        }
        let carry = refine != nil && restriction.requiresBelowTruncation
        // Curves lifted to |f| and kept below the truncation are cut away wherever |f|
        // is above it, so the refiner need not place it there: on zeta and
        // 1/Γ that was nineteen phase vertices of every twenty. A tenth over
        // the truncation is room for a chord whose ends lie above it and whose middle
        // dips under. Curves lifted to the truncated surface are drawn on the plateau's
        // top, and placed all over.
        let ceiling = restriction.requiresBelowTruncation && policy == .magnitude ? 1.1 * truncation.highest : .infinity
        guard let scalar = field(of: tag, refine: refine, ceiling: ceiling) else { return .empty }

        // A refiner has the function the grid was sampled from and moves
        // the vertices onto its true level set, in domain space, before
        // the lift: the grid decides which contours exist, f decides
        // exactly where they run. Every level at once, as `Contour.levels`
        // hands them on: a level is a few closed rings, each followed along
        // its own length, and refined one level at a time a layer kept two
        // cores of ten busy.
        //
        // Marching squares runs at the branch cut too, and its paths are
        // dropped for `Contour.cut`'s: about 0.3-0.6 ms at res 1000, under
        // 1% of the refined work. Skipping it would want the branch cut as a
        // field of its own and a second `Contour.levels` call.
        let contoured = Contour.levels(of: scalar, tag.levels(levels), over: rectangle)

        /// The runs of the contours a restriction keeps, lifted: a path cut
        /// where a vertex leaves the kept region, and carried to the
        /// truncation edge where it leaves over the truncation.
        func kept(_ contoured: [(level: Double, paths: [[P2<ParamSpace>]])], region: Region) -> [[P3<WorldSpace>]] {
            var paths: [[P3<WorldSpace>]] = []
            for (level, placed) in contoured {
                for line in placed.map({ $0.map { P2<DomainSpace>($0.v) } }) {
                    // Lifted onto f where the function is at hand: a phase line
                    // lifted to the grid's height stands a chord's error off
                    // the surface the rings, the folds and the top edge are drawn
                    // on, and the test hides it there or shows it floating.
                    var lifted: [P3<WorldSpace>]
                    if policy != .level, let magnitude = refine?.magnitude {
                        lifted = line.map { p in
                            let u = magnitude(p)
                            guard u.isFinite else { return lift([p], policy: policy, level: level)[0] }
                            return P3(p.x, p.y, policy == .surface ? min(u, truncationHeight(atX: p.x)) : u)
                        }
                    } else {
                        lifted = lift(line, policy: policy, level: level)
                    }
                    // A run the refiner carried to a zero of f ends at the zero's
                    // own height. The grid has no sample there: its chord over a
                    // pit's floor bottoms out at the nearest sample's height, a
                    // cell's rise above the zero, and lifting the end to the
                    // chord would hang the fan of phase lines that far above the
                    // top edge that f draws under them. Only an end that *is* a zero,
                    // to rounding against the grid's own height, so a vertex the
                    // grid places a hair under f is left on the drawn surface.
                    if policy != .level, let magnitude = refine?.magnitude, !lifted.isEmpty {
                        for i in Set([0, lifted.count - 1]) {
                            let u = magnitude(line[i])
                            if u.isFinite, u <= 1e-6 * lifted[i].z {
                                lifted[i] = P3(lifted[i].x, lifted[i].y, u)
                            }
                        }
                    }
                    var run: [P3<WorldSpace>] = []
                    var previous: (vertex: P3<WorldSpace>, admitted: Bool)?
                    for v in lifted {
                        let admitted = admits(v, restriction, region: region)
                        if admitted {
                            if carry, let p = previous, !p.admitted,
                               let truncationEdge = truncationCrossing(from: v, to: p.vertex),
                               admits(truncationEdge, restriction, region: region) {
                                run.append(truncationEdge)
                            }
                            run.append(v)
                        } else {
                            if carry, let p = previous, p.admitted,
                               let truncationEdge = truncationCrossing(from: p.vertex, to: v),
                               admits(truncationEdge, restriction, region: region) {
                                run.append(truncationEdge)
                            }
                            if run.count >= 2 { paths.append(run) }
                            run = []
                        }
                        previous = (v, admitted)
                    }
                    if run.count >= 2 { paths.append(run) }
                }
            }
            return paths
        }

        // The footprint is where a copy stands in space, so each copy, on
        // each tile or in place, is clipped to it there, exactly, as a
        // face's copy is (`Face.contours(of:_:on:)`): cut where the
        // footprint's perimeter, carried back through the tile, crosses a
        // contour, and kept where the footprint holds the copy's point. What
        // else the restriction asks is read on the domain, where the copy
        // came from: a band of Re z or Im z is the function's, and the
        // truncation is read at the point's own x through the tile under it,
        // which is the domain's.
        if restriction.requiresFootprint, case .inside = region {
            return PolylineSet(paths: (tiled ? tiles : [.identity]).flatMap { tile -> [[P3<WorldSpace>]] in
                guard let footprint = Surface.footprint(region, on: tile) else { return [] }
                let clipped = Contour.clipped(contoured, over: rectangle, contains: footprint.contains,
                                              cuts: footprint.perimeter)
                let paths = kept(clipped, region: .full)
                return tiled ? paths.map { $0.map { tile($0) } } : paths
            })
        }
        let paths = kept(contoured, region: region)
        guard tiled else { return PolylineSet(paths: paths) }
        // Replicated by the same maps the occluder instances the heightfield
        // with, so a tiled contour cannot drift from the tile it lies on.
        return PolylineSet(paths: tiles.flatMap { tile in
            paths.map { $0.map { tile($0) } }
        })
    }
}

public extension ContourField {
    /// The levels a layer's contours are asked for: as listed, but for arg f
    /// a level of ±π is the branch cut, asked once, as π, after the rest.
    func levels(_ listed: [Double]) -> [Double] {
        guard self == .phase else { return listed }
        let plain = listed.filter { !Contour.isCut($0) }
        return plain.count < listed.count ? plain + [.pi] : listed
    }
}
