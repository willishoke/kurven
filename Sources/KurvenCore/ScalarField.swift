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
    /// half a period is that jump, and has no crossing. A level the field
    /// takes all along a bounded side of the rectangle draws nothing along it:
    /// the side is the face's edge.
    ///
    /// A field with a routine of its own is contoured on its samples, by
    /// marching squares at each level, and the paths handed to the routine,
    /// a level at a time, the levels in parallel. The samples are their own
    /// grid: nothing wraps or is unwrapped there.
    ///
    /// A path is clipped to a trimmed domain exactly, as hatching is clipped
    /// to a polyhedron face's convex pieces: cut wherever a segment of it
    /// crosses one of `trimLoops` or `cuts`, and each piece kept where the
    /// domain holds its middle. The domain is what `trimLoops` bound
    /// even-odd (`Face.trimLoops`), read by a ray along +u, and of that what
    /// `contains` holds, for whatever the loops do not carry -- a copy's
    /// footprint -- with `cuts` the curves where `contains` changes. The
    /// pieces kept either side of a cut are joined. So a run ends on a loop
    /// wherever the loop crosses it, between two vertices or through one,
    /// and a hole narrower than a step still cuts the path. Where neither
    /// is given nothing is trimmed; where a domain's boundary is left out of
    /// the loops and the cuts, a path is kept or dropped a segment at a
    /// time there. Loops that bound even-odd lie in the plane: no periodic
    /// axis runs across them. On a periodic axis `contains` is asked at
    /// unwrapped coordinates, and must answer the same a period along; the
    /// cuts are met a period along too.
    ///
    /// Where the domain's edge is a level set, the loops' chords only
    /// approximate it, and `excess` gives it exactly: negative inside the
    /// domain, zero on that edge, as |f| less the truncation height is on
    /// the graph. A run is then kept where `excess` is negative, cut where it
    /// changes sign, bisected along the chord to the last bit, and a run
    /// cut short of that edge by a chord of the loops is carried on to it
    /// (`trimmed(_:closed:holds:contains:loops:excess:)`).
    ///
    /// A field contoured by bisection has its chords refined
    /// (`refined(_:excess:cell:period:lift:)`); `lift` is the face's point in
    /// space, given where the parameter plane is the world's xy, as on the
    /// graph, so a chord's sag off the face is a length in the cell's units.
    static func levels(of field: ScalarField, _ levels: [Double],
                       over rectangle: (u: ParamAxis, v: ParamAxis),
                       trimLoops: [[P2<ParamSpace>]]? = nil,
                       contains: (@Sendable (P2<ParamSpace>) -> Bool)? = nil,
                       cuts: [[P2<ParamSpace>]] = [],
                       excess: (@Sendable (P2<ParamSpace>) -> Double)? = nil,
                       lift: (@Sendable (P2<ParamSpace>) -> SIMD3<Double>)? = nil)
        -> [(level: Double, paths: [[P2<ParamSpace>]])] {
        let contoured: [(level: Double, paths: [[P2<ParamSpace>]])]
        switch field.contouring {
        case .bisection(let value, let period):
            contoured = plainLevels(of: value, period: period, levels, over: rectangle, lift: lift)
        case .placed(let samples, let place):
            let grid = Contour.levels(of: samples, levels)
            let paths = Heightfield.parallelMap(Array(grid.indices), chunk: 1) { k in
                place(grid[k].level, grid[k].paths.map { $0.map { P2<ParamSpace>($0.v) } })
            }
            contoured = Array(zip(levels, paths))
        }
        return clipped(contoured, over: rectangle, trimLoops: trimLoops, contains: contains, cuts: cuts,
                       excess: excess)
    }

    /// Level sets already found, clipped to a trimmed domain as
    /// `levels(of:_:over:trimLoops:contains:cuts:)` clips them: so a family
    /// contoured once can be clipped to each copy of a face, a copy's
    /// footprint being where the copy stands.
    static func clipped(_ contoured: [(level: Double, paths: [[P2<ParamSpace>]])],
                        over rectangle: (u: ParamAxis, v: ParamAxis),
                        trimLoops: [[P2<ParamSpace>]]? = nil,
                        contains: (@Sendable (P2<ParamSpace>) -> Bool)? = nil,
                        cuts: [[P2<ParamSpace>]] = [],
                        excess: (@Sendable (P2<ParamSpace>) -> Double)? = nil)
        -> [(level: Double, paths: [[P2<ParamSpace>]])] {
        guard trimLoops != nil || contains != nil else { return contoured }
        let (u, v) = rectangle
        let crossed = TrimIndex((trimLoops ?? []) + cuts, periods: (u.period, v.period))
        let bounding = trimLoops.map { TrimIndex($0, periods: (nil, nil)) }
        let holds: (P2<ParamSpace>) -> Bool = { p in
            (bounding?.encloses(p.v) ?? true) && (contains?(p) ?? true)
        }
        return contoured.map { contour in
            (contour.level, contour.paths.flatMap { path -> [[P2<ParamSpace>]] in
                guard let first = path.first, let last = path.last, path.count >= 2 else { return [] }
                // A closed path ends where it began, or a period along where
                // it was unwrapped.
                let closed = abs(u.difference(last.x, first.x)) <= 1e-12 * u.range.length
                    && abs(v.difference(last.y, first.y)) <= 1e-12 * v.range.length
                return trimmed(path, closed: closed, holds: holds, contains: contains, loops: crossed,
                               excess: excess)
            })
        }
    }

    private static func plainLevels(of value: @escaping @Sendable (P2<ParamSpace>) -> Double,
                                    period: Double?, _ levels: [Double],
                                    over rectangle: (u: ParamAxis, v: ParamAxis),
                                    lift: (@Sendable (P2<ParamSpace>) -> SIMD3<Double>)?)
        -> [(level: Double, paths: [[P2<ParamSpace>]])] {
        let (u, v) = rectangle
        let cell = max(abs(u.spacing), abs(v.spacing))
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
                // over it, so the bisection keeps the grid's own bracket. An
                // end on the level to the bit is the crossing, not half a
                // rounding past it: a coordinate's contour is on its level.
                let (under, over) = grid[a.x, a.y] > 0 ? (pb, pa) : (pa, pb)
                let bracket = bisect(under, over) { !(excess(value(P2($0))) > 0) }
                if excess(value(P2(bracket.holds))) == 0 { return P2(bracket.holds) }
                return P2(0.5 * (bracket.holds + bracket.fails))
            }
            return paths.flatMap { offEdges(unwrapped($0, u.period, v.period), rectangle) }.map {
                refined($0, excess: { excess(value($0)) }, cell: cell, period: period, lift: lift)
            }
        }

        return zip(levels, Heightfield.parallelMap(levels, chunk: 1, lines)).map { ($0, $1) }
    }

    /// How far a chord of the plain routine may stray, in cells of the
    /// parameter rectangle: the refiner's tolerance for |f| and arg f
    /// (`Refiner.tolerance`), so the families drawn together on a face are
    /// held to one bound. On the 64-sample landscape it is 1.3e-3, a third
    /// of a pixel in a 1600-pixel drawing.
    static let chordTolerance = 0.02
    /// How many times a chord may be halved: at most 31 vertices are added
    /// between two of the grid's, as the refiner's `maxDepth`.
    static let chordDepth = 5

    /// A path with each chord split where it strays from the level set: its
    /// midpoint's distance from it, estimated as the excess over the level
    /// there over the excess's gradient, more than `chordTolerance` of a
    /// cell; or, given `lift`, its midpoint in space more than that from the
    /// face's point at the chord's midpoint, the chord's sag. The vertices
    /// marching squares placed lie on the level set; between two, the curve
    /// can turn within a cell, and on a steep graph its chord can sag far
    /// off the face. A chord is split at the point of the level set on its
    /// normal through its midpoint, sought toward Newton's step and doubled
    /// out to the chord's length until the excess changes sign, then
    /// bisected to the last bit, as a crossing is. A chord split for its
    /// sag alone is split at its midpoint where none is found there, the
    /// midpoint being within the tolerance of the level set. Each half is
    /// measured again, down to `chordDepth` halvings. A chord is kept as it
    /// is where its midpoint has no value or no gradient, where it strays
    /// and no point of the level set is found along its normal, and past
    /// the last halving.
    private static func refined(_ path: [P2<ParamSpace>], excess: (P2<ParamSpace>) -> Double,
                                cell: Double, period: Double?,
                                lift: ((P2<ParamSpace>) -> SIMD3<Double>)?) -> [P2<ParamSpace>] {
        guard path.count >= 2 else { return path }
        let tolerance = chordTolerance * cell, step = 1e-4 * cell
        // A circle-valued excess jumps by a period at the level opposite:
        // past a quarter of one, a sign change is that jump.
        func near(_ r: Double) -> Bool { r.isFinite && period.map { abs(r) < $0 / 4 } ?? true }
        func f(_ p: SIMD2<Double>) -> Double { excess(P2(p)) }
        var out = [path[0]]
        func split(_ a: SIMD2<Double>, _ la: SIMD3<Double>?, _ b: SIMD2<Double>, _ lb: SIMD3<Double>?,
                   depth: Int) {
            let chord = simd_distance(a, b)
            guard depth < chordDepth, chord > 1e-9 * cell else { return }
            let m = 0.5 * (a + b), r = f(m)
            guard near(r) else { return }
            let normal = SIMD2(a.y - b.y, b.x - a.x) / chord
            var error = 0.0, gn = 0.0, slope = 0.0
            if r != 0 {
                let g = SIMD2(f(m + SIMD2(step, 0)) - f(m - SIMD2(step, 0)),
                              f(m + SIMD2(0, step)) - f(m - SIMD2(0, step))) / (2 * step)
                slope = simd_length(g)
                guard slope > 0, slope.isFinite else { return }
                error = abs(r) / slope
                gn = simd_dot(g, normal)
            }
            var sag = 0.0
            if let lift, let la, let lb {
                let d = simd_distance(lift(P2(m)), 0.5 * (la + lb))
                if d.isFinite { sag = d }
            }
            guard error > tolerance || sag > tolerance else { return }
            /// The point of the level set on the chord's normal through its
            /// midpoint; nil for a chord across the level set rather than
            /// along it, or with none within a chord's length.
            func onLevel() -> SIMD2<Double>? {
                guard abs(gn) > 1e-3 * slope else { return nil }
                let toward: Double = r / gn > 0 ? -1 : 1
                var reach = abs(r / gn)
                while true {
                    let t = toward * min(reach, chord), rt = f(m + t * normal)
                    guard near(rt) else { return nil }
                    let far = m + t * normal
                    if rt == 0 { return far }
                    if (rt > 0) != (r > 0) {
                        // From the end at or under the level to the end over
                        // it, as a crossing is placed.
                        let (under, over) = r > 0 ? (far, m) : (m, far)
                        let bracket = bisect(under, over) { !(f($0) > 0) }
                        return f(bracket.holds) == 0 ? bracket.holds : 0.5 * (bracket.holds + bracket.fails)
                    }
                    guard reach < chord else { return nil }
                    reach *= 2
                }
            }
            var point = m
            if r != 0 {
                if let q = onLevel() { point = q } else if error > tolerance { return }
            }
            let lp = lift?(P2(point))
            split(a, la, point, lp, depth: depth + 1)
            out.append(P2(point))
            split(point, lp, b, lb, depth: depth + 1)
        }
        var la = lift?(path[0])
        for b in path.dropFirst() {
            let lb = lift?(b)
            split(out[out.count - 1].v, la, b.v, lb, depth: 0)
            out.append(b)
            la = lb
        }
        return out
    }

    /// A path less its segments along a bounded side of the rectangle, cut
    /// there. The rectangle is open: a level its field takes all along a side
    /// draws nothing there, at the lower end of an axis as at the upper, since
    /// the side is the face's edge, as a polyhedron's hatching leaves a line
    /// along a side of its face to the edge (`Solid.hatchLines(_:onFace:)`).
    /// Marching squares counts a sample on the level as under it, so a level
    /// at an axis's lower end drew the side, and one at its upper end nothing.
    private static func offEdges(_ path: [P2<ParamSpace>], _ rectangle: (u: ParamAxis, v: ParamAxis))
        -> [[P2<ParamSpace>]] {
        func along(_ axis: ParamAxis, _ a: Double, _ b: Double) -> Bool {
            guard !axis.periodic, a == b else { return false }
            return a == axis.range.lo || a == axis.range.hi || a == axis.coordinate(0) || a == axis.coordinate(axis.cells)
        }
        var runs: [[P2<ParamSpace>]] = [], run: [P2<ParamSpace>] = []
        for p in path {
            if let q = run.last, along(rectangle.u, q.x, p.x) || along(rectangle.v, q.y, p.y) {
                if run.count >= 2 { runs.append(run) }
                run = []
            }
            run.append(p)
        }
        if run.count >= 2 { runs.append(run) }
        return runs
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
    /// crosses a loop, segment by segment, each piece kept where `holds`
    /// holds at its middle, and the pieces kept either side of a cut joined
    /// into one run, with no vertex at the cut.
    ///
    /// A closed path's last vertex is its first, or its first a period
    /// along. Where it is kept through both, the run through its last vertex
    /// and the run through its first are one run, so a loop is not split
    /// where it happens to begin.
    ///
    /// Given `excess`, whose zero level the loops follow as chords, the
    /// loops' cuts are moved onto that level. Two runs the loops cut apart
    /// where the path passes between a chord and the level are joined. An
    /// end cut on a loop where `excess` is still negative stops short of the
    /// level its chord stands for: it is carried on along the path to where
    /// `excess` reaches 0, bisected on the segment where it does, so long as
    /// the path crosses no loop and stays where `contains` holds on the way;
    /// otherwise the cut was another edge's, and stands. Then each run is
    /// kept where `excess` is negative and cut where it changes sign,
    /// bisected along the segment to the last bit and ended on its negative
    /// side: a run past the loops' chord, or one the loops missed, is cut
    /// back to the level, and no vertex lies over it.
    private static func trimmed(_ path: [P2<ParamSpace>], closed: Bool,
                                holds: (P2<ParamSpace>) -> Bool, contains: ((P2<ParamSpace>) -> Bool)?,
                                loops: TrimIndex, excess: ((P2<ParamSpace>) -> Double)?) -> [[P2<ParamSpace>]] {
        // Where an end of a run was cut on a loop: the segment of the path,
        // counted on round a closed path past its last, so a run carried
        // round where the path begins ends on segment n + k for its k.
        let n = path.count - 1, shift = path[n].v - path[0].v
        func vertex(_ j: Int) -> P2<ParamSpace> { j <= n ? path[j] : P2(path[j - n].v + shift) }
        var runs: [(points: [P2<ParamSpace>], start: Int?, end: Int?)] = []
        var run: [P2<ParamSpace>] = [], start: Int?
        // Whether the run under way ends at a cut, and whether the path is
        // kept from its first vertex.
        var endsAtCut = false, keptFromStart = false
        for k in 0..<n {
            let a = path[k].v, b = path[k + 1].v
            let cuts = loops.crossings(a, b)
            var from = a
            for (i, to) in (cuts + [b]).enumerated() where to != from {
                defer { from = to }
                guard holds(P2(0.5 * (from + to))) else {
                    if run.count >= 2 { runs.append((run, start, endsAtCut ? k : nil)) }
                    run = []
                    continue
                }
                if run.isEmpty {
                    run = [P2(from)]
                    start = i > 0 ? k : nil
                    if k == 0, from == a { keptFromStart = true }
                } else if endsAtCut, run.count >= 2 {
                    run.removeLast()
                }
                run.append(P2(to))
                endsAtCut = i < cuts.count
            }
        }
        var end: Int?
        if closed, keptFromStart, !run.isEmpty, let first = runs.first {
            runs.removeFirst()
            run += first.points.dropFirst().map { P2($0.v + shift) }
            end = first.end.map { $0 + n }
        }
        if run.count >= 2 { runs.append((run, start, end)) }
        guard let excess else { return runs.map(\.points) }

        /// The vertices that carry a run on from an end at `x`, out along
        /// `outward`, to where `excess` reaches 0; nil where it does not
        /// before the path crosses a loop, leaves `contains` or ends.
        func carried(from x: SIMD2<Double>, _ outward: [P2<ParamSpace>]) -> [P2<ParamSpace>]? {
            guard excess(P2(x)) < 0 else { return nil }
            var previous = x, added: [P2<ParamSpace>] = []
            for w in outward.map(\.v) where w != previous {
                guard loops.crossings(previous, w).isEmpty,
                      contains?(P2(0.5 * (previous + w))) ?? true else { return nil }
                if !(excess(P2(w)) < 0) {
                    let edge = bisect(previous, w) { excess(P2($0)) < 0 }.holds
                    return added.last?.v == edge ? added : added + [P2(edge)]
                }
                added.append(P2(w))
                previous = w
            }
            return nil
        }
        // Two runs cut apart where the path leaves the loops and comes back
        // between their chords and the level, through a sliver of the
        // domain, are one: the cuts and the path between them, its vertices
        // and the midpoints of its steps, are all where `excess` is negative
        // and `contains` holds. A path across a region outside the domain
        // that the loops caught has its cuts on that region's chords, over
        // the level. So are the last run
        // and the first of a closed path, across where it begins.
        func inside(_ between: [P2<ParamSpace>]) -> Bool {
            zip(between, between.dropFirst()).allSatisfy { a, b in
                let m = P2<ParamSpace>(0.5 * (a.v + b.v))
                return excess(a) < 0 && excess(m) < 0 && contains?(m) ?? true
            } && between.last.map { excess($0) < 0 } ?? false
        }
        var joined: [(points: [P2<ParamSpace>], start: Int?, end: Int?)] = []
        for next in runs {
            if let previous = joined.last, let cut = previous.end, let back = next.start, cut <= back,
               let x = previous.points.last, let y = next.points.first {
                let interior = ((cut + 1)..<(back + 1)).map(vertex)
                if inside([x] + interior + [y]) {
                    joined[joined.count - 1] = (previous.points.dropLast() + interior + next.points.dropFirst(),
                                                previous.start, next.end)
                    continue
                }
            }
            joined.append(next)
        }
        if closed, let last = joined.last, let cut = last.end, let first = joined.first, let back = first.start,
           cut < back + n, let x = last.points.last, let y = first.points.first {
            let interior = ((cut + 1)..<(back + n + 1)).map(vertex)
            if inside([x] + interior + [P2(y.v + shift)]) {
                if joined.count == 1 {
                    joined[0] = (last.points.dropLast() + interior + [P2(y.v + shift)], nil, nil)
                } else {
                    joined[joined.count - 1] = (last.points.dropLast() + interior
                                                    + first.points.dropFirst().map { P2($0.v + shift) },
                                                last.start, first.end.map { $0 + n })
                    joined.removeFirst()
                }
            }
        }
        var out: [[P2<ParamSpace>]] = []
        for (points, start, end) in joined {
            var points = points
            if let end, let last = points.last,
               let more = carried(from: last.v, ((end + 1)...(end < n ? n : min(2 * n, end + n))).map(vertex)) {
                points.removeLast()
                points += more
            }
            if let start, let first = points.first,
               let more = carried(from: first.v, (0...start).reversed().map(vertex)) {
                points.removeFirst()
                points = more.reversed() + points
            }
            // Kept where `excess` is negative, cut where it changes sign. A
            // chord whose midpoint lies on the other side from both its ends
            // passes through a hole in the domain, or a region outside it,
            // between them, and is read at its midpoint too.
            var read: [P2<ParamSpace>] = [], inside: [Bool] = []
            for p in points {
                let isInside = excess(p) < 0
                if let q = read.last, isInside == inside[inside.count - 1] {
                    let m = P2<ParamSpace>(0.5 * (q.v + p.v))
                    if (excess(m) < 0) != isInside { read.append(m); inside.append(!isInside) }
                }
                read.append(p); inside.append(isInside)
            }
            points = read
            var piece: [P2<ParamSpace>] = []
            func add(_ p: SIMD2<Double>) { if piece.last?.v != p { piece.append(P2(p)) } }
            for (k, p) in points.enumerated() {
                if k > 0, inside[k] != inside[k - 1] {
                    let (i, o) = inside[k] ? (p.v, points[k - 1].v) : (points[k - 1].v, p.v)
                    add(bisect(i, o) { excess(P2($0)) < 0 }.holds)
                    if !inside[k] {
                        if piece.count >= 2 { out.append(piece) }
                        piece = []
                    }
                }
                if inside[k] { add(p.v) }
            }
            if piece.count >= 2 { out.append(piece) }
        }
        return out
    }
}

/// The segments of a face's trim loops, bucketed on a grid over their
/// bounding box, so a segment of a contour, or a ray along +u, is met only
/// with those near it. On a periodic axis a segment is met with the loops a
/// period along too.
struct TrimIndex {
    private let starts: [SIMD2<Double>], ends: [SIMD2<Double>]
    private let lo: SIMD2<Double>, hi: SIMD2<Double>, cell: SIMD2<Double>
    private let columns: Int, rows: Int
    private let buckets: [[Int32]]
    /// By rows alone, for the ray: each segment once in each row it spans.
    private let strips: [[Int32]]
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
        var strips = [[Int32]](repeating: [], count: starts.isEmpty ? 0 : n)
        for s in starts.indices {
            let (i0, j0, i1, j1) = Self.cells(simd_min(starts[s], ends[s]), simd_max(starts[s], ends[s]),
                                              lo: lo, cell: cell, columns: n, rows: n)
            for j in j0...j1 {
                strips[j].append(Int32(s))
                for i in i0...i1 { buckets[j * n + i].append(Int32(s)) }
            }
        }
        self.buckets = buckets; self.strips = strips
    }

    /// Whether the loops wind round `p` an odd number of times: a ray from
    /// it along +u crosses an odd number of their segments, each half open
    /// in v, so a ray through a vertex counts it once. The loops' own
    /// plane: no period is applied.
    func encloses(_ p: SIMD2<Double>) -> Bool {
        guard !starts.isEmpty, p.y >= lo.y, p.y <= hi.y else { return false }
        let (_, j, _, _) = Self.cells(p, p, lo: lo, cell: cell, columns: columns, rows: rows)
        var inside = false
        for s in strips[j] {
            let a = starts[Int(s)], b = ends[Int(s)]
            guard (a.y > p.y) != (b.y > p.y) else { continue }
            if p.x < a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y) { inside.toggle() }
        }
        return inside
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
        // A rounding of the coordinates: how far off the segment a loop may
        // lie and still be met, so a loop along it is met however the two
        // were rounded.
        let pad = 1e-12 * max(simd_reduce_max(simd_abs(p)), simd_reduce_max(simd_abs(q)), simd_length(q - p))
        let qlo = simd_min(p, q) - pad, qhi = simd_max(p, q) + pad
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
                        guard abs(cross(w, d)) <= length * pad else { continue }
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
