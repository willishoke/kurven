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
    /// trimmed domain; on a periodic axis it is asked at unwrapped
    /// coordinates, and must answer the same a period along. It is asked at
    /// a path's vertices: a run is cut where a vertex leaves the trimmed
    /// domain, and ended on its boundary by bisection along the chord it
    /// leaves by. Where the boundary passes between two vertices and back,
    /// as a hole narrower than a step does, the path is not cut: a
    /// membership test cannot say where to look, and clipping exactly needs
    /// the boundary itself.
    static func levels(of field: ScalarField, _ levels: [Double],
                       over rectangle: (u: ParamAxis, v: ParamAxis),
                       contains: (@Sendable (P2<ParamSpace>) -> Bool)? = nil)
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
        return contoured.map { contour in
            (contour.level, contour.paths.flatMap { path -> [[P2<ParamSpace>]] in
                guard let first = path.first, let last = path.last, path.count >= 2 else { return [] }
                // A closed path ends where it began, or a period along where
                // it was unwrapped.
                let closed = abs(u.difference(last.x, first.x)) <= 1e-12 * u.range.length
                    && abs(v.difference(last.y, first.y)) <= 1e-12 * v.range.length
                return trimmed(path, closed: closed, contains: contains)
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

    /// The runs of a path inside the trimmed domain, each end where a vertex
    /// leaves it bisected onto its boundary along the chord.
    ///
    /// A closed path's last vertex is its first, or its first a period
    /// along, and is taken to lie where the first does, so rounding at the
    /// seam cannot cut a loop at its closing vertex. It is walked from a
    /// vertex outside, so the run through its first vertex is not split in
    /// two there.
    private static func trimmed(_ path: [P2<ParamSpace>], closed: Bool,
                                contains: (P2<ParamSpace>) -> Bool) -> [[P2<ParamSpace>]] {
        var points = path, inside = path.map(contains)
        if closed {
            inside[inside.count - 1] = inside[0]
            guard let s = inside.firstIndex(of: false) else { return [path] }
            if s > 0 {
                let shift = path[path.count - 1].v - path[0].v
                points = Array(path[s...]) + path[1...s].map { P2($0.v + shift) }
                inside = Array(inside[s...]) + Array(inside[1...s])
            }
        }
        var runs: [[P2<ParamSpace>]] = [], run: [P2<ParamSpace>] = []
        for k in points.indices {
            let c = points[k]
            if k > 0, inside[k] != inside[k - 1] {
                let p = points[k - 1]
                let (kept, _) = bisect(inside[k] ? c.v : p.v, inside[k] ? p.v : c.v) {
                    contains(P2($0))
                }
                // Where the boundary is a vertex, the vertex is the end.
                if inside[k] {
                    run = kept == c.v ? [] : [P2(kept)]
                } else {
                    if kept != p.v { run.append(P2(kept)) }
                    if run.count >= 2 { runs.append(run) }
                    run = []
                }
            }
            if inside[k] { run.append(c) }
        }
        if run.count >= 2 { runs.append(run) }
        return runs
    }
}
