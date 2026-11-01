import Foundation
import simd

/// A scalar field on a face's parameter rectangle: what a level set is taken
/// of. A coordinate, a height, |f| or arg f, the facing.
///
/// A field on a periodic axis takes the same value a period along it. A
/// *circle-valued* field, such as the coordinate along a periodic axis, is
/// defined up to whole multiples of its own period, and a level is too.
public struct ScalarField: Sendable {
    /// Moves the marching-squares paths of one level onto the field's own
    /// level set: the hook through which a field with a routine of its own,
    /// as |f| and arg f have the refiner, places its contours. Whatever else
    /// the routine needs, such as which of f's fields it places or the
    /// ceiling above which a layer's curves are cut away, it closes over.
    public typealias Place = @Sendable (_ level: Double, _ paths: [[P2<ParamSpace>]])
        -> [[P2<ParamSpace>]]

    /// How a field's level sets are found.
    public enum Contouring: Sendable {
        /// Sampled on the parameter rectangle's grid, and each crossing
        /// bisected on the value: a field with no singular points. `period`
        /// is the field's period when it is circle-valued: 2π for an angle.
        case bisection(value: @Sendable (P2<ParamSpace>) -> Double, period: Double?)
        /// Contoured on `samples`, the grid the routine was built against,
        /// and the paths handed to `place`: a field with a routine of its
        /// own, which owns its branch cut if it has one (arg f's, as
        /// `Surface.field(of:refine:ceiling:)` says).
        case placed(samples: Grid2D<Float>, place: Place)
    }

    public let contouring: Contouring

    /// A smooth field, contoured by bisection on its value.
    public init(period: Double? = nil, value: @escaping @Sendable (P2<ParamSpace>) -> Double) {
        self.contouring = .bisection(value: value, period: period)
    }

    /// A field whose contours are placed by a routine of its own, from
    /// marching squares on `samples`.
    public init(samples: Grid2D<Float>, place: @escaping Place) {
        self.contouring = .placed(samples: samples, place: place)
    }
}

public extension Contour {
    /// The level sets of a field over a parameter rectangle, in level order,
    /// each vertex its surface coordinate.
    ///
    /// A field contoured by bisection is sampled on the rectangle's grid and
    /// contoured on its difference from the level, every crossing then
    /// bisected to the last bit along its grid edge: the grid decides which
    /// contours exist, the field where they run. A periodic axis is sampled
    /// one cell past its end -- the first sample again, as `ParamAxis.sample`
    /// reads it -- so a contour that crosses the seam is one path, and its
    /// coordinates are unwrapped along it: a closed one ends a period past
    /// its first vertex, as a coordinate curve does. A circle-valued field's
    /// difference is taken the short way round, and it jumps by a period at
    /// the level opposite; an edge whose two samples differ by more than
    /// half a period is that jump, and has no crossing.
    ///
    /// A field with a routine of its own is contoured on its samples, by
    /// marching squares at each level, and the paths handed to the routine,
    /// a level at a time, the levels in parallel. The samples are their own
    /// grid: nothing wraps or is unwrapped there.
    ///
    /// `contains` says which points of the rectangle lie in the face's
    /// trimmed domain, and `trimLoops` are the closed curves its boundary
    /// lies on (`Face.trimLoops`). A path is clipped to the domain exactly,
    /// as hatching is clipped to a polyhedron face's convex pieces: cut
    /// wherever a segment of it crosses a trim loop, each piece kept where
    /// `contains` holds at its middle, and the pieces kept on either side of
    /// a cut joined. So a run ends on a trim loop wherever the loop crosses
    /// it, between two vertices or through one, and a hole narrower than a
    /// step still cuts the path. With no loop across it a path is kept or
    /// dropped a segment at a time. On a periodic axis `contains` is asked
    /// at unwrapped coordinates, and must answer the same a period along;
    /// the loops are met a period along too.
    static func levels(of field: ScalarField, _ levels: [Double],
                       over rectangle: (u: ParamAxis, v: ParamAxis),
                       contains: (@Sendable (P2<ParamSpace>) -> Bool)? = nil,
                       trimLoops: [[P2<ParamSpace>]] = [])
        -> [(level: Double, paths: [[P2<ParamSpace>]])] {
        let contoured: [(level: Double, paths: [[P2<ParamSpace>]])]
        switch field.contouring {
        case .bisection(let value, let period):
            contoured = plainLevels(of: value, period: period, levels, over: rectangle)
        case .placed(let samples, let place):
            let grid = Contour.levels(of: samples, levels)
            let paths = Heightfield.parallelMap(Array(grid.indices), chunk: 1) { k in
                place(grid[k].level, grid[k].paths.map { $0.map { P2<ParamSpace>($0.v) } })
            }
            contoured = Array(zip(levels, paths))
        }
        guard let contains else { return contoured }
        let (u, v) = rectangle
        let loops = TrimIndex(trimLoops, periods: (u.period, v.period))
        return contoured.map { contour in
            (contour.level, contour.paths.flatMap { path -> [[P2<ParamSpace>]] in
                guard let first = path.first, let last = path.last, path.count >= 2 else { return [] }
                // A closed path ends where it began, or a period along where
                // it was unwrapped.
                let closed = abs(u.difference(last.x, first.x)) <= 1e-12 * u.range.length
                    && abs(v.difference(last.y, first.y)) <= 1e-12 * v.range.length
                return trimmed(path, closed: closed, contains: contains, loops: loops)
            })
        }
    }

    private static func plainLevels(of value: @escaping @Sendable (P2<ParamSpace>) -> Double,
                                    period: Double?, _ levels: [Double],
                                    over rectangle: (u: ParamAxis, v: ParamAxis))
        -> [(level: Double, paths: [[P2<ParamSpace>]])] {
        let (u, v) = rectangle
        // Row by row in parallel.
        let samples = Heightfield.parallelMap(Array(0..<(u.samples * v.samples)), chunk: u.samples) { k in
            value(P2(u.coordinate(k % u.samples), v.coordinate(k / u.samples)))
        }
        let w = u.periodic ? u.samples + 1 : u.samples, h = v.periodic ? v.samples + 1 : v.samples
        let domain = Domain(real: u.range, imag: v.range)

        // Each level is a grid of its own, the field less the level, so a
        // sample's sign is the field's own however near the level it lies.
        @Sendable func lines(at level: Double) -> [[P2<ParamSpace>]] {
            @Sendable func excess(_ z: Double) -> Double {
                period.map { remainder(z - level, $0) } ?? z - level
            }
            var g = [Double](repeating: 0, count: w * h)
            for j in 0..<h {
                for i in 0..<w { g[j * w + i] = excess(samples[v.sample(j) * u.samples + u.sample(i)]) }
            }
            let grid = Grid2D(width: w, height: h, domain: domain, values: g.map { Float($0) })
            let paths: [[P2<ParamSpace>]] = Contour.lines(
                of: grid, level: 0, index: nil, wraps: (u.periodic, v.periodic)
            ) { a, b in
                let ga = g[a.y * w + a.x], gb = g[b.y * w + b.x]
                if let period, abs(ga - gb) > period / 2 { return nil }
                let pa = SIMD2(u.coordinate(a.x), v.coordinate(a.y))
                let pb = SIMD2(u.coordinate(b.x), v.coordinate(b.y))
                // From the end the grid has at or under the level to the end
                // over it, so the bisection keeps the grid's own bracket.
                let (under, over) = grid[a.x, a.y] > 0 ? (pb, pa) : (pa, pb)
                let bracket = bisect(under, over) { !(excess(value(P2($0))) > 0) }
                return P2(0.5 * (bracket.holds + bracket.fails))
            }
            return paths.map { unwrapped($0, u.period, v.period) }
        }

        return zip(levels, Heightfield.parallelMap(levels, chunk: 1, lines)).map { ($0, $1) }
    }

    /// A path whose every step is the short way round a periodic axis.
    private static func unwrapped(_ path: [P2<ParamSpace>], _ pu: Double?, _ pv: Double?)
        -> [P2<ParamSpace>] {
        var out = path
        for k in out.indices.dropFirst() {
            var c = out[k].v
            let previous = out[k - 1].v
            if let pu { c.x -= pu * ((c.x - previous.x) / pu).rounded() }
            if let pv { c.y -= pv * ((c.y - previous.y) / pv).rounded() }
            out[k] = P2(c)
        }
        return out
    }

    /// The runs of a path inside the trimmed domain: the path cut where it
    /// crosses a trim loop, segment by segment, each piece kept where
    /// `contains` holds at its middle, and the pieces kept either side of a
    /// cut joined into one run, with no vertex at the cut.
    ///
    /// A closed path's last vertex is its first, or its first a period
    /// along. Where it is kept through both, the run through its last vertex
    /// and the run through its first are one run, so a loop is not split
    /// where it happens to begin.
    private static func trimmed(_ path: [P2<ParamSpace>], closed: Bool,
                                contains: (P2<ParamSpace>) -> Bool, loops: TrimIndex) -> [[P2<ParamSpace>]] {
        var runs: [[P2<ParamSpace>]] = [], run: [P2<ParamSpace>] = []
        // Whether the run under way ends at a cut, and whether the path is
        // kept from its first vertex.
        var endsAtCut = false, keptFromStart = false
        for k in 0..<(path.count - 1) {
            let a = path[k].v, b = path[k + 1].v
            let cuts = loops.crossings(a, b)
            var from = a
            for (i, to) in (cuts + [b]).enumerated() where to != from {
                defer { from = to }
                guard contains(P2(0.5 * (from + to))) else {
                    if run.count >= 2 { runs.append(run) }
                    run = []
                    continue
                }
                if run.isEmpty {
                    run = [P2(from)]
                    if k == 0, from == a { keptFromStart = true }
                } else if endsAtCut, run.count >= 2 {
                    run.removeLast()
                }
                run.append(P2(to))
                endsAtCut = i < cuts.count
            }
        }
        if closed, keptFromStart, !run.isEmpty, let first = runs.first {
            let shift = path[path.count - 1].v - path[0].v
            runs.removeFirst()
            run += first.dropFirst().map { P2($0.v + shift) }
        }
        if run.count >= 2 { runs.append(run) }
        return runs
    }
}

/// The segments of a face's trim loops, bucketed on a grid over their
/// bounding box, so a segment of a contour is met only with those near it.
/// On a periodic axis a segment is met with the loops a period along too.
struct TrimIndex {
    private let starts: [SIMD2<Double>], ends: [SIMD2<Double>]
    private let lo: SIMD2<Double>, hi: SIMD2<Double>, cell: SIMD2<Double>
    private let columns: Int, rows: Int
    private let buckets: [[Int32]]
    private let periods: (u: Double?, v: Double?)

    /// Each loop ends where it begins, or a period along; its segments are
    /// what is met, so a chain left open is met only where it runs.
    init(_ loops: [[P2<ParamSpace>]], periods: (u: Double?, v: Double?)) {
        var starts: [SIMD2<Double>] = [], ends: [SIMD2<Double>] = []
        for loop in loops {
            for (a, b) in zip(loop, loop.dropFirst()) where a != b { starts.append(a.v); ends.append(b.v) }
        }
        self.starts = starts; self.ends = ends; self.periods = periods
        var lo = SIMD2(Double.infinity, .infinity), hi = -lo
        for (a, b) in zip(starts, ends) { lo = simd_min(lo, simd_min(a, b)); hi = simd_max(hi, simd_max(a, b)) }
        let n = max(1, min(1024, Int(Double(starts.count).squareRoot())))
        columns = n; rows = n
        self.lo = lo; self.hi = hi
        let extent = hi - lo
        cell = SIMD2(extent.x > 0 ? extent.x / Double(n) : 1, extent.y > 0 ? extent.y / Double(n) : 1)
        var buckets = [[Int32]](repeating: [], count: starts.isEmpty ? 0 : n * n)
        for s in starts.indices {
            let (i0, j0, i1, j1) = Self.cells(simd_min(starts[s], ends[s]), simd_max(starts[s], ends[s]),
                                              lo: lo, cell: cell, columns: n, rows: n)
            for j in j0...j1 { for i in i0...i1 { buckets[j * n + i].append(Int32(s)) } }
        }
        self.buckets = buckets
    }

    private static func cells(_ a: SIMD2<Double>, _ b: SIMD2<Double>, lo: SIMD2<Double>, cell: SIMD2<Double>,
                              columns: Int, rows: Int) -> (Int, Int, Int, Int) {
        func index(_ x: Double, _ lo: Double, _ size: Double, _ count: Int) -> Int {
            let f = ((x - lo) / size).rounded(.down)
            return f < 0 ? 0 : f >= Double(count) ? count - 1 : Int(f)
        }
        return (index(a.x, lo.x, cell.x, columns), index(a.y, lo.y, cell.y, rows),
                index(b.x, lo.x, cell.x, columns), index(b.y, lo.y, cell.y, rows))
    }

    /// Where the segment from `p` to `q` crosses a trim loop, strictly
    /// between its ends, in order from `p`: each a point of the loop. Two
    /// crossings a rounding apart along the segment are one; a loop that runs
    /// along the segment crosses it at its own vertices.
    func crossings(_ p: SIMD2<Double>, _ q: SIMD2<Double>) -> [SIMD2<Double>] {
        guard !starts.isEmpty else { return [] }
        let qlo = simd_min(p, q), qhi = simd_max(p, q)
        // The whole periods by which the loops are moved to meet the segment.
        func shifts(_ period: Double?, _ qlo: Double, _ qhi: Double, _ lo: Double, _ hi: Double) -> [Int] {
            guard let period, period > 0 else { return [0] }
            let a = Int(((qlo - hi) / period).rounded(.up)), b = Int(((qhi - lo) / period).rounded(.down))
            return a <= b ? Array(a...b) : []
        }
        let us = shifts(periods.u, qlo.x, qhi.x, lo.x, hi.x), vs = shifts(periods.v, qlo.y, qhi.y, lo.y, hi.y)
        let d = q - p, dd = simd_dot(d, d)
        guard dd > 0 else { return [] }
        func cross(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double { a.x * b.y - a.y * b.x }
        var found: [(t: Double, point: SIMD2<Double>)] = []
        for ku in us {
            for kv in vs {
                let shift = SIMD2(Double(ku) * (periods.u ?? 0), Double(kv) * (periods.v ?? 0))
                let blo = qlo - shift, bhi = qhi - shift
                guard blo.x <= hi.x, bhi.x >= lo.x, blo.y <= hi.y, bhi.y >= lo.y else { continue }
                let (i0, j0, i1, j1) = Self.cells(blo, bhi, lo: lo, cell: cell, columns: columns, rows: rows)
                var near: [Int32] = []
                for j in j0...j1 { for i in i0...i1 { near += buckets[j * columns + i] } }
                if i0 != i1 || j0 != j1 { near.sort() }
                for (n, s) in near.enumerated() where n == 0 || near[n - 1] != s {
                    let a = starts[Int(s)] + shift, b = ends[Int(s)] + shift
                    guard simd_min(a, b).x <= qhi.x, simd_max(a, b).x >= qlo.x,
                          simd_min(a, b).y <= qhi.y, simd_max(a, b).y >= qlo.y else { continue }
                    let e = b - a, w = a - p
                    let den = cross(d, e), length = dd.squareRoot()
                    if abs(den) <= 1e-14 * length * simd_length(e) {
                        // Parallel: crossed only where it runs along the
                        // segment, at the loop's own vertices.
                        guard abs(cross(w, d)) <= 1e-14 * length * max(simd_length(w), length) else { continue }
                        for x in [a, b] {
                            let t = simd_dot(x - p, d) / dd
                            if t > 0, t < 1 { found.append((t, x)) }
                        }
                        continue
                    }
                    let t = cross(w, e) / den, u = cross(w, d) / den
                    guard t > 0, t < 1, u >= -1e-12, u <= 1 + 1e-12 else { continue }
                    found.append((t, a + min(max(u, 0), 1) * e))
                }
            }
        }
        found.sort { ($0.t, $0.point.x, $0.point.y) < ($1.t, $1.point.x, $1.point.y) }
        var out: [SIMD2<Double>] = []
        var last = -Double.infinity
        for (t, point) in found where t > 1e-12 && t < 1 - 1e-12 && t - last > 1e-12 {
            out.append(point); last = t
        }
        return out
    }
}
