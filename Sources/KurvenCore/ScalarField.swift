import Foundation
import Dispatch
import simd

/// A scalar field on a face's parameter rectangle: what a level set is taken
/// of. A coordinate, a height, |f| or arg f, the facing.
///
/// A field on a periodic axis takes the same value a period along it. A
/// *circle-valued* field, such as arg f or the coordinate along a periodic
/// axis, is defined up to whole multiples of its own `period`, and a level
/// is too.
public struct ScalarField: Sendable {
    /// Moves the marching-squares paths of one level onto the field's own
    /// level set: the hook through which a field with a routine of its own,
    /// as |f| and arg f have the refiner, places its contours.
    public typealias Place = @Sendable (_ level: Double, _ paths: [[P2<ParamSpace>]])
        -> [[P2<ParamSpace>]]

    public let value: @Sendable (P2<ParamSpace>) -> Double
    /// The grid `place` is handed the contours of: the samples it was built
    /// against. Nil for a field without one, which is sampled on the
    /// parameter rectangle's own grid.
    public let samples: Grid2D<Float>?
    /// The field's period when it is circle-valued: 2π for arg f.
    public let period: Double?
    /// Nil for a field with no singular points, whose level sets are placed
    /// by bisection on `value`.
    public let place: Place?

    /// A smooth field, contoured by bisection on its value.
    public init(period: Double? = nil, value: @escaping @Sendable (P2<ParamSpace>) -> Double) {
        self.value = value; self.period = period
        self.samples = nil; self.place = nil
    }

    /// A field whose contours are placed by a routine of its own, from
    /// marching squares on `samples`.
    public init(samples: Grid2D<Float>, period: Double? = nil,
                value: @escaping @Sendable (P2<ParamSpace>) -> Double,
                place: @escaping Place) {
        self.value = value; self.period = period
        self.samples = samples; self.place = place
    }
}

public extension Contour {
    /// The level sets of a field over a parameter rectangle, in level order,
    /// each vertex its surface coordinate.
    ///
    /// A field with a `place` routine is contoured on its samples and handed
    /// to it. Any other is sampled on the rectangle's grid and contoured on
    /// its difference from the level, every crossing then bisected to the
    /// last bit along its grid edge: the grid decides which contours exist,
    /// the field where they run. A periodic axis is sampled one cell past its
    /// end -- the first sample again, as `ParamAxis.sample` reads it -- so a
    /// contour that crosses the seam is one path, and its coordinates are
    /// unwrapped along it: a closed one ends a period past its first vertex,
    /// as a coordinate curve does. A circle-valued field's difference is
    /// taken the short way round, and it jumps by a period at the level
    /// opposite; an edge whose two samples differ by more than half a
    /// period is that jump, and has no crossing.
    ///
    /// `contains`, the trim, says which points of the rectangle the face
    /// covers; on a periodic axis it is asked at unwrapped coordinates, and
    /// answers the same a period along. A path is cut where it leaves, each
    /// run ended on the trim by bisection along the chord it leaves by.
    static func levels(of field: ScalarField, _ levels: [Double],
                       over rectangle: (u: ParamAxis, v: ParamAxis),
                       contains: (@Sendable (P2<ParamSpace>) -> Bool)? = nil)
        -> [(level: Double, paths: [[P2<ParamSpace>]])] {
        let contoured: [(level: Double, paths: [[P2<ParamSpace>]])]
        if let place = field.place, let samples = field.samples {
            contoured = Contour.levels(of: samples, levels).map { contour in
                (contour.level, place(contour.level, contour.paths.map { $0.map { P2<ParamSpace>($0.v) } }))
            }
        } else {
            contoured = plainLevels(of: field, levels, over: rectangle)
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

    private static func plainLevels(of field: ScalarField, _ levels: [Double],
                                    over rectangle: (u: ParamAxis, v: ParamAxis))
        -> [(level: Double, paths: [[P2<ParamSpace>]])] {
        let (u, v) = rectangle
        // Row by row in parallel, each writing only its own row.
        var stored = [Double](repeating: 0, count: u.samples * v.samples)
        stored.withUnsafeMutableBufferPointer { out in
            nonisolated(unsafe) let base = out.baseAddress!
            DispatchQueue.concurrentPerform(iterations: v.samples) { j in
                for i in 0..<u.samples {
                    base[j * u.samples + i] = field.value(P2(u.coordinate(i), v.coordinate(j)))
                }
            }
        }
        let samples = stored
        let w = u.periodic ? u.samples + 1 : u.samples, h = v.periodic ? v.samples + 1 : v.samples
        let domain = Domain(real: u.range, imag: v.range)

        // Each level is a grid of its own, the field less the level, so a
        // sample's sign is the field's own however near the level it lies.
        @Sendable func lines(at level: Double) -> [[P2<ParamSpace>]] {
            @Sendable func excess(_ z: Double) -> Double {
                field.period.map { remainder(z - level, $0) } ?? z - level
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
                if let period = field.period, abs(ga - gb) > period / 2 { return nil }
                let pa = SIMD2(u.coordinate(a.x), v.coordinate(a.y))
                let pb = SIMD2(u.coordinate(b.x), v.coordinate(b.y))
                // From the end the grid has at or under the level to the end
                // over it, so the bisection keeps the grid's own bracket.
                var (under, over) = grid[a.x, a.y] > 0 ? (pb, pa) : (pa, pb)
                for _ in 0..<60 {
                    let m = 0.5 * (under + over)
                    if excess(field.value(P2(m))) > 0 { over = m } else { under = m }
                }
                return P2(0.5 * (under + over))
            }
            return paths.map { unwrapped($0, u.period, v.period) }
        }

        guard levels.count > 1 else { return levels.map { ($0, lines(at: $0)) } }
        var results = [[[P2<ParamSpace>]]](repeating: [], count: levels.count)
        results.withUnsafeMutableBufferPointer { out in
            nonisolated(unsafe) let base = out.baseAddress!
            DispatchQueue.concurrentPerform(iterations: levels.count) { k in
                base[k] = lines(at: levels[k])
            }
        }
        return zip(levels, results).map { ($0, $1) }
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

    /// The runs of a path where `contains` holds, each end that leaves the
    /// trim bisected onto it along its chord.
    private static func trimmed(_ path: [P2<ParamSpace>], closed: Bool,
                                contains: (P2<ParamSpace>) -> Bool) -> [[P2<ParamSpace>]] {
        var runs: [[P2<ParamSpace>]] = [], run: [P2<ParamSpace>] = []
        var previous: (point: P2<ParamSpace>, inside: Bool)?
        for c in path {
            let inside = contains(c)
            if let p = previous, p.inside != inside {
                var (kept, cut) = inside ? (c.v, p.point.v) : (p.point.v, c.v)
                for _ in 0..<60 {
                    let m = 0.5 * (kept + cut)
                    if contains(P2(m)) { kept = m } else { cut = m }
                }
                if inside {
                    run = [P2(kept)]
                } else {
                    run.append(P2(kept))
                    if run.count >= 2 { runs.append(run) }
                    run = []
                }
            }
            if inside { run.append(c) }
            previous = (c, inside)
        }
        if run.count >= 2 { runs.append(run) }
        // A closed path cut somewhere and inside where it begins: one curve,
        // so its first run carries on from its last.
        guard closed, runs.count >= 2, contains(path[0]) else { return runs }
        let shift = path[path.count - 1].v - path[0].v
        let last = runs.removeLast()
        runs[0] = last + runs[0].dropFirst().map { P2($0.v + shift) }
        return runs
    }
}
