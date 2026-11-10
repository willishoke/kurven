import Foundation
import simd

/// A landscape's solid on its grid, as another geometry's curves meet it in
/// a composite: the slab under the drawn surface, down to its base, as
/// convex prisms, and the spans of a segment whose lines of sight pass
/// through one, in closed form.
///
/// Over each of the grid's triangles, in each copy, the solid is the points
/// over the triangle, above the base, under the triangle's plane and under
/// the truncation: an intersection of half-spaces, so convex, split where a
/// band boundary crosses the triangle and, under a footprint, by the
/// footprint's convex pieces. The lines of sight from a segment `a -> b`
/// sweep the strip a + s (b − a) + t d, 0 ≤ s ≤ 1, t ≥ 0, and each prism's
/// half-spaces are half-planes of (s, t): clipped by them, the strip leaves a
/// convex polygon, and where it has area the lines through its span of s pass
/// through the prism's interior. A line that only grazes a prism leaves a
/// polygon of no area and hides nothing, and a point on the slab's boundary
/// -- a wall, the floor, the top -- is hidden only where its line goes in:
/// the polygon starts a billionth of a cell along the line. The prisms a
/// strip can meet are found by walking its footprint over the grid, row by
/// row.
///
/// The prisms are the surface the traversal reads on the grid
/// (`Surface.drawnSurface(at:step:)`): each cell split along the diagonal
/// from its (1, 0) corner to its (0, 1) corner, the raw samples interpolated
/// and cut at the truncation, over the copies' boxes and the footprint.
struct Slab: Sendable {
    /// One copy of the grid: world (x, y) to the decimated grid's index
    /// coordinates (u, v), and the footprint's convex pieces there.
    struct Copy: Sendable {
        let linear: simd_double2x2
        let offset: SIMD2<Double>
        let pieces: [(halfPlanes: [(n: SIMD3<Double>, c: Double)], lo: SIMD2<Double>, hi: SIMD2<Double>)]
        /// The copy's box corners in world space, for how far a line can run
        /// in it.
        let corners: [SIMD3<Double>]
    }

    let copies: [Copy]
    let nx: Int, ny: Int
    /// The decimated grid's samples, untruncated, row by row.
    let heights: [Double]
    let base: Double, top: Double
    /// The truncation's bands as stretches of u, with their heights.
    let bands: [(lo: Double, hi: Double, height: Double)]
    /// Where a line starts to count: a billionth of a cell along it.
    let start: Double

    init(_ h: Heightfield, name: String) throws {
        let s = h.surface, g = s.height
        let step = max(h.step, 1)
        nx = (g.width + step - 1) / step; ny = (g.height + step - 1) / step
        var heights = [Double](repeating: 0, count: nx * ny)
        for j in 0..<ny { for i in 0..<nx { heights[j * nx + i] = Double(g[i * step, j * step]) } }
        self.heights = heights
        var top = -Double.infinity, floor = Double.infinity
        h.forEachSample { top = max(top, $0.z); floor = min(floor, $0.z) }
        for v in h.occluder.vertices { floor = min(floor, v.z) }
        var base = floor
        if case .perimeter(_, let b) = h.walls { base = b }
        self.base = base; self.top = top
        let d = g.domain
        let su = Double(g.width - 1) / (d.real.length * Double(step))
        let sv = Double(g.height - 1) / (d.imag.length * Double(step))
        let cell = max(abs(1 / su), abs(1 / sv))
        start = 1e-9 * cell
        func u(_ x: Double) -> Double { (x - d.real.lo) * su }
        let spans = s.truncation.bands(from: min(d.real.lo, d.real.hi), to: max(d.real.lo, d.real.hi))
        if spans.isEmpty {
            bands = [(-.infinity, .infinity, .infinity)]
        } else {
            // In order along u, the first and last open to either side.
            var b = spans.map { span in (lo: min(u(span.lo), u(span.hi)), hi: max(u(span.lo), u(span.hi)), height: span.height) }
                .sorted { $0.lo < $1.lo }
            b[0].lo = -.infinity; b[b.count - 1].hi = .infinity
            bands = b
        }
        // The footprint's convex pieces in world (x, y).
        var footprint: [[SIMD2<Double>]]?
        if case .inside(let perimeter) = h.region {
            var corners = perimeter.edges.map { $0.start.v }
            var area = 0.0
            for k in corners.indices {
                let p = corners[k], q = corners[(k + 1) % corners.count]
                area += p.x * q.y - q.x * p.y
            }
            if area < 0 { corners.reverse() }
            do {
                footprint = try ConvexPartition.pieces([Array(corners.indices)], point: { corners[$0] },
                                                       tolerance: 1e-12 * abs(area), face: 0)
                    .map { $0.map { corners[$0] } }
            } catch {
                throw CompositeError("\(name): its footprint is not one simple polygon (\(error))")
            }
        }
        let domainCorners = [SIMD2(d.real.lo, d.imag.lo), SIMD2(d.real.hi, d.imag.lo),
                             SIMD2(d.real.hi, d.imag.hi), SIMD2(d.real.lo, d.imag.hi)]
        copies = try h.tiles.map { tile in
            guard let back = tile.inverse else { throw CompositeError("\(name): a tile has no inverse") }
            let scale = simd_double2x2(rows: [SIMD2(su, 0), SIMD2(0, sv)])
            let linear = scale * simd_double2x2(rows: [SIMD2(back.a, back.b), SIMD2(back.c, back.d)])
            let offset = scale * (SIMD2(back.tx, back.ty) - SIMD2(d.real.lo, d.imag.lo))
            // A world half-plane n · w ≤ c in (u, v): w = L⁻¹ (uv − o).
            let inverse = linear.inverse
            let pieces = (footprint ?? []).map { polygon -> (halfPlanes: [(n: SIMD3<Double>, c: Double)], lo: SIMD2<Double>, hi: SIMD2<Double>) in
                var planes: [(n: SIMD3<Double>, c: Double)] = []
                for k in polygon.indices {
                    let p = polygon[k], q = polygon[(k + 1) % polygon.count]
                    let n = SIMD2(q.y - p.y, p.x - q.x)   // outward of a counter-clockwise piece
                    let m = inverse.transpose * n
                    planes.append((SIMD3(m.x, m.y, 0), simd_dot(n, p) + simd_dot(m, offset)))
                }
                let mapped = polygon.map { linear * $0 + offset }
                return (planes, mapped.reduce(mapped[0], simd_min), mapped.reduce(mapped[0], simd_max))
            }
            let corners = domainCorners.flatMap { c -> [SIMD3<Double>] in
                let w = tile(P3<WorldSpace>(c.x, c.y, 0))
                return [SIMD3(w.x, w.y, base), SIMD3(w.x, w.y, top)]
            }
            return Copy(linear: linear, offset: offset, pieces: pieces, corners: corners)
        }
    }

    /// The spans of the segment `a -> b`, in the landscape's own
    /// coordinates, whose lines of sight along the unit `d` pass through
    /// the slab's interior.
    func hidden(from a: SIMD3<Double>, to b: SIMD3<Double>, toward d: SIMD3<Double>) -> [(Double, Double)] {
        var out: [(Double, Double)] = []
        let e = b - a
        for copy in copies {
            func flat(_ p: SIMD3<Double>, moved: Bool) -> SIMD3<Double> {
                let q = copy.linear * SIMD2(p.x, p.y) + (moved ? copy.offset : .zero)
                return SIMD3(q.x, q.y, p.z)
            }
            let A = flat(a, moved: true), E = flat(e, moved: false), D = flat(d, moved: false)
            let reach = copy.corners.map { simd_length($0 - a) }.max()! + simd_length(e)
            let (t0, t1) = (start, 1.01 * reach + 1)
            let box: [(n: SIMD3<Double>, c: Double)] = [
                (SIMD3(-1, 0, 0), 0), (SIMD3(1, 0, 0), Double(nx - 1)), (SIMD3(0, -1, 0), 0),
                (SIMD3(0, 1, 0), Double(ny - 1)), (SIMD3(0, 0, -1), -base), (SIMD3(0, 0, 1), top),
            ]
            let swept = Slab.clip(box, a: A, e: E, d: D, t0: t0, t1: t1)
            guard !swept.isEmpty else { continue }
            if let span = floorSpan(a: A, e: E, d: D, copy: copy) { out.append(span) }
            // The strip's footprint on the grid, walked row by row.
            let foot = swept.map { SIMD2(A.x + $0.x * E.x + $0.y * D.x, A.y + $0.x * E.y + $0.y * D.y) }
            let v = foot.map(\.y)
            let j0 = max(Int(v.min()!.rounded(.down)), 0)
            let j1 = min(max(Int(v.max()!.rounded(.up)) - 1, j0), ny - 2)
            guard j0 <= j1 else { continue }
            for j in j0...j1 {
                let row = Slab.clip2D(foot, axis: 1, lo: Double(j), hi: Double(j + 1))
                guard !row.isEmpty else { continue }
                let u = row.map(\.x)
                let i0 = max(Int(u.min()!.rounded(.down)), 0)
                let i1 = min(max(Int(u.max()!.rounded(.up)) - 1, i0), nx - 2)
                guard i0 <= i1 else { continue }
                for i in i0...i1 {
                    for upper in [false, true] {
                        for halfSpaces in prisms(i, j, upper: upper, copy: copy) {
                            let polygon = Slab.clip(halfSpaces, a: A, e: E, d: D, t0: t0, t1: t1)
                            if let span = Slab.span(of: polygon) { out.append(span) }
                        }
                    }
                }
            }
        }
        return out
    }

    /// The span of a segment whose lines of sight cross the floor, the
    /// slab's face at the base, over the copy's box and footprint: a sheet
    /// of the boundary, opaque where the slab over it is as thin as a
    /// rounding -- where the surface comes down to the base -- as much as
    /// where it is thick. The line from a + s e crosses the base at a point
    /// affine in s, so the span is where that point is in each of the
    /// floor's convex pieces, and the crossing is past the line's start.
    func floorSpan(a: SIMD3<Double>, e: SIMD3<Double>, d: SIMD3<Double>, copy: Copy) -> (Double, Double)? {
        guard abs(d.z) > 1e-12 else { return nil }
        // t(s) = t0 + s t1 where the line meets the base; the point there
        // is p0 + s p1 in (u, v).
        let tb0 = (base - a.z) / d.z, tb1 = -e.z / d.z
        let p0 = SIMD2(a.x + tb0 * d.x, a.y + tb0 * d.y), p1 = SIMD2(e.x + tb1 * d.x, e.y + tb1 * d.y)
        let box: [(n: SIMD3<Double>, c: Double)] = [(SIMD3(-1, 0, 0), 0), (SIMD3(1, 0, 0), Double(nx - 1)),
                                                    (SIMD3(0, -1, 0), 0), (SIMD3(0, 1, 0), Double(ny - 1))]
        var best: (Double, Double)?
        for planes in copy.pieces.isEmpty ? [box] : copy.pieces.map({ box + $0.halfPlanes }) {
            // Each half-plane n · p ≤ c, and t(s) ≥ start, as k0 + s k1 ≤ 0.
            var lo = 0.0, hi = 1.0
            for (k0, k1) in planes.map({ (simd_dot(SIMD2($0.n.x, $0.n.y), p0) - $0.c, simd_dot(SIMD2($0.n.x, $0.n.y), p1)) })
                + [(start - tb0, -tb1)] {
                if abs(k1) < 1e-300 {
                    if k0 > 0 { lo = 1; hi = 0 }
                    continue
                }
                let root = -k0 / k1
                if k1 > 0 { hi = min(hi, root) } else { lo = max(lo, root) }
            }
            if hi > lo { best = best.map { (min($0.0, lo), max($0.1, hi)) } ?? (lo, hi) }
        }
        return best
    }

    /// The convex prisms over one triangle of cell (i, j), in index
    /// coordinates: the lower triangle has the cell's (0, 0) corner, the
    /// upper its (1, 1) corner.
    func prisms(_ i: Int, _ j: Int, upper: Bool, copy: Copy) -> [[(n: SIMD3<Double>, c: Double)]] {
        let h00 = heights[j * nx + i], h10 = heights[j * nx + i + 1]
        let h01 = heights[(j + 1) * nx + i], h11 = heights[(j + 1) * nx + i + 1]
        let (fi, fj) = (Double(i), Double(j))
        var common: [(n: SIMD3<Double>, c: Double)] = [(SIMD3(0, 0, -1), -base)]
        if upper {
            common += [(SIMD3(1, 0, 0), fi + 1), (SIMD3(0, 1, 0), fj + 1), (SIMD3(-1, -1, 0), -(fi + fj + 1))]
            // z ≤ h11 + (i + 1 − u)(h01 − h11) + (j + 1 − v)(h10 − h11)
            let gu = -(h01 - h11), gv = -(h10 - h11)
            if [h11, h01, h10].allSatisfy(\.isFinite) {
                common.append((SIMD3(-gu, -gv, 1), h11 + (fi + 1) * (h01 - h11) + (fj + 1) * (h10 - h11)))
            }
        } else {
            common += [(SIMD3(-1, 0, 0), -fi), (SIMD3(0, -1, 0), -fj), (SIMD3(1, 1, 0), fi + fj + 1)]
            // z ≤ h00 + (u − i)(h10 − h00) + (v − j)(h01 − h00)
            let gu = h10 - h00, gv = h01 - h00
            if [h00, h10, h01].allSatisfy(\.isFinite) {
                common.append((SIMD3(-gu, -gv, 1), h00 - fi * gu - fj * gv))
            }
        }
        var out: [[(n: SIMD3<Double>, c: Double)]] = []
        for band in bands where band.lo < fi + 1 && band.hi > fi {
            var planes = common
            if band.lo.isFinite { planes.append((SIMD3(-1, 0, 0), -band.lo)) }
            if band.hi.isFinite { planes.append((SIMD3(1, 0, 0), band.hi)) }
            if band.height.isFinite { planes.append((SIMD3(0, 0, 1), band.height)) }
            if copy.pieces.isEmpty {
                out.append(planes)
            } else {
                for piece in copy.pieces where piece.lo.x <= fi + 1 && piece.hi.x >= fi
                    && piece.lo.y <= fj + 1 && piece.hi.y >= fj {
                    out.append(planes + piece.halfPlanes)
                }
            }
        }
        return out
    }

    /// The part of the strip a + s e + t d, 0 ≤ s ≤ 1, t0 ≤ t ≤ t1, inside
    /// every half-space n · x ≤ c: a convex polygon in (s, t), empty where
    /// none is left.
    static func clip(_ halfSpaces: [(n: SIMD3<Double>, c: Double)], a: SIMD3<Double>, e: SIMD3<Double>,
                     d: SIMD3<Double>, t0: Double, t1: Double) -> [SIMD2<Double>] {
        var polygon = [SIMD2(0, t0), SIMD2(1, t0), SIMD2(1, t1), SIMD2(0, t1)]
        for (n, c) in halfSpaces {
            let k = c - simd_dot(n, a), ks = simd_dot(n, e), kt = simd_dot(n, d)
            polygon = clip2D(polygon) { ks * $0.x + kt * $0.y - k }
            if polygon.count < 3 { return [] }
        }
        return polygon
    }

    /// A polygon clipped to where the affine `f` is not positive.
    static func clip2D(_ polygon: [SIMD2<Double>], _ f: (SIMD2<Double>) -> Double) -> [SIMD2<Double>] {
        var out: [SIMD2<Double>] = []
        out.reserveCapacity(polygon.count + 1)
        for k in polygon.indices {
            let p = polygon[k], q = polygon[(k + 1) % polygon.count]
            let fp = f(p), fq = f(q)
            if fp <= 0 { out.append(p) }
            if (fp <= 0) != (fq <= 0) { out.append(p + (q - p) * (fp / (fp - fq))) }
        }
        return out
    }

    /// A polygon clipped to `lo ≤ x[axis] ≤ hi`.
    static func clip2D(_ polygon: [SIMD2<Double>], axis: Int, lo: Double, hi: Double) -> [SIMD2<Double>] {
        clip2D(clip2D(polygon) { lo - $0[axis] }) { $0[axis] - hi }
    }

    /// The span of s a clipped strip covers, where it has area: a polygon
    /// whose area is a rounding of its box's is a line that grazes.
    static func span(of polygon: [SIMD2<Double>]) -> (Double, Double)? {
        guard polygon.count >= 3 else { return nil }
        var lo = polygon[0], hi = polygon[0], area = 0.0
        for k in polygon.indices {
            let p = polygon[k], q = polygon[(k + 1) % polygon.count]
            lo = simd_min(lo, p); hi = simd_max(hi, p)
            area += p.x * q.y - q.x * p.y
        }
        let extent = hi - lo
        guard extent.x > 0, extent.y > 0, abs(area) > 2e-10 * extent.x * extent.y else { return nil }
        return (max(lo.x, 0), min(hi.x, 1))
    }
}
