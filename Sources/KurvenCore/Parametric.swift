import Foundation
import Dispatch
import simd

/// One direction of a parameter rectangle.
///
/// A periodic axis closes on itself: `samples` points at `lo + i * length /
/// samples`, and the last cell runs from the last sample back to the first. A
/// bounded axis has `samples` points spanning `[lo, hi]` inclusive, as a
/// `Grid2D` does. The difference is one cell, and getting it wrong leaves a
/// torus with a slit down one side or a sphere with a doubled seam.
public struct ParamAxis: Sendable, Equatable {
    public var range: Interval
    public var periodic: Bool
    public var samples: Int

    public init(_ range: Interval, periodic: Bool, samples: Int) {
        precondition(samples >= 2, "an axis needs at least two samples")
        self.range = range; self.periodic = periodic; self.samples = samples
    }

    /// A whole turn, `[0, 2π)`.
    public static func angle(samples: Int) -> ParamAxis {
        ParamAxis(Interval(lo: 0, hi: 2 * .pi), periodic: true, samples: samples)
    }

    public var cells: Int { periodic ? samples : samples - 1 }
    public var spacing: Double { range.length / Double(cells) }
    public var period: Double? { periodic ? range.length : nil }

    /// The coordinate of grid index `i`, *unwrapped*: on a periodic axis
    /// `i == samples` is one period past `i == 0`, which is what lets a cell
    /// that closes the seam interpolate across it rather than back through
    /// the whole range.
    public func coordinate(_ i: Int) -> Double { range.lo + Double(i) * spacing }

    /// The stored sample an unwrapped grid index refers to.
    public func sample(_ i: Int) -> Int {
        periodic ? ((i % samples) + samples) % samples : i
    }

    /// The signed difference `a - b` taken the short way round a periodic
    /// axis; plain subtraction on a bounded one.
    public func difference(_ a: Double, _ b: Double) -> Double {
        guard let period else { return a - b }
        let d = a - b
        return d - period * (d / period).rounded()
    }
}

/// A point of a surface with its two partial derivatives.
public struct SurfaceJet: Sendable, Equatable {
    public var position: SIMD3<Double>
    public var du: SIMD3<Double>
    public var dv: SIMD3<Double>

    public init(position: SIMD3<Double>, du: SIMD3<Double>, dv: SIMD3<Double>) {
        self.position = position; self.du = du; self.dv = dv
    }

    /// `du × dv`: the normal of the parametrization's own orientation, not
    /// normalized.
    public var normal: SIMD3<Double> { simd_cross(du, dv) }
}

/// The solid a surface bounds, for sphere tracing along a line of sight.
///
/// A point of space is inside the solid exactly when its signed distance
/// from the base surface, along the base's normal line through its foot, is
/// under the surface's normal offset from the base at that foot. The round torus
/// has its foot in closed form, and a surface displaced along the torus's
/// normal keeps that foot and adds the displacement as its normal offset, so
/// the test is a point test with no raster in it. The gap, distance less
/// normal offset, is positive outside, negative inside and zero on the surface,
/// and over the points outside the solid and within the shell of the base
/// it changes by at most `lipschitz` per unit of travel: sphere tracing, which
/// steps by the gap over that constant cannot step through the solid, and
/// a line of sight is clear exactly when every step along it is.
public struct TubularNeighbourhood: Sendable {
    /// The surface coordinate whose normal line passes through a point,
    /// and the point's signed distance along it, positive outward.
    public let foot: @Sendable (SIMD3<Double>) -> (coordinate: P2<ParamSpace>, distance: Double)
    /// How far the surface stands off the base along the normal at a
    /// coordinate, positive outward.
    public let normalOffset: @Sendable (P2<ParamSpace>) -> Double
    /// The normal offset's bounds: it is never more than `above` nor less than
    /// `-below`. Beyond `above` a point is outside, under `-below` inside.
    public let above: Double, below: Double
    /// The most the gap changes per unit of travel, over the points outside
    /// the solid and within the shell.
    public let lipschitz: Double
    /// How fast the foot's coordinate moves per unit of travel, per axis,
    /// over the points between `inside` under the base and `outside` over
    /// it: what a normal offset's gradient is multiplied by.
    public let spread: @Sendable (_ outside: Double, _ inside: Double) -> SIMD2<Double>
    /// A ball containing the solid.
    public let centre: SIMD3<Double>, radius: Double

    public init(foot: @escaping @Sendable (SIMD3<Double>) -> (coordinate: P2<ParamSpace>, distance: Double),
                normalOffset: @escaping @Sendable (P2<ParamSpace>) -> Double,
                above: Double, below: Double, lipschitz: Double,
                spread: @escaping @Sendable (Double, Double) -> SIMD2<Double>,
                centre: SIMD3<Double>, radius: Double) {
        self.foot = foot; self.normalOffset = normalOffset
        self.above = above; self.below = below; self.lipschitz = lipschitz
        self.spread = spread; self.centre = centre; self.radius = radius
    }
}

/// A surface given as a map from a parameter rectangle into space.
///
/// The map is exact and carries its derivatives, because the questions asked
/// of it -- which way does it face here, and exactly where along this segment
/// does it turn edge-on -- are asked at the ink's own coordinates, and a normal
/// read off the nearest grid point flips sign between neighbours precisely
/// where the answer matters, at a fold. The grid is what gets rasterized;
/// the map is what gets asked.
///
/// A heightfield is the case `(u, v) -> (u, v, h(u, v))` with neither axis
/// periodic; a torus has both periodic.
public struct ParametricSurface: Sendable {
    public let u: ParamAxis
    public let v: ParamAxis
    public let map: @Sendable (P2<ParamSpace>) -> SurfaceJet
    /// The grid, `u` fastest: `positions[j * u.samples + i]`.
    public let positions: [P3<WorldSpace>]
    /// `+1` when `du × dv` points out of the solid the surface bounds, `-1`
    /// when it points in; `nil` when the surface bounds no solid.
    ///
    /// Only a closed, embedded, orientable surface bounds a solid, and only
    /// for such a surface is "faces away from the viewer" the same as
    /// "hidden": a line of sight that reaches the back of it must first have
    /// entered the solid through the front. That is a fact the caller knows
    /// about its geometry -- it cannot be read off samples -- so it is
    /// declared, and the orientation is then computed rather than guessed.
    public let outward: Double?
    /// The solid the surface bounds as a point test, for surfaces that have
    /// one in closed form; nil otherwise, and the ink is then tested by the
    /// coordinate image.
    public let neighbourhood: TubularNeighbourhood?

    /// - Parameter encloses: the surface is closed, embedded and orientable.
    ///   Declaring it for a surface that passes through itself, or has a
    ///   boundary, culls ink that is in plain view.
    public init(u: ParamAxis, v: ParamAxis, encloses: Bool, neighbourhood: TubularNeighbourhood? = nil,
                map: @escaping @Sendable (P2<ParamSpace>) -> SurfaceJet) {
        self.u = u; self.v = v; self.map = map
        self.neighbourhood = encloses ? neighbourhood : nil
        var positions: [P3<WorldSpace>] = []
        positions.reserveCapacity(u.samples * v.samples)
        // Six times the enclosed volume, by the divergence theorem over the
        // grid: the sum of p · (du × dv) du dv. Only its sign is used, and
        // the sign is robust at any resolution that resolves the surface.
        var volume = 0.0
        for j in 0..<v.samples {
            for i in 0..<u.samples {
                let jet = map(P2(u.coordinate(i), v.coordinate(j)))
                positions.append(P3(jet.position))
                volume += simd_dot(jet.position, jet.normal)
            }
        }
        self.positions = positions
        if encloses {
            precondition(volume.isFinite && abs(volume) > 0,
                         "a surface declared to enclose a solid has no volume")
            outward = volume > 0 ? 1 : -1
        } else {
            outward = nil
        }
    }

    /// A surface whose grid is already computed -- because evaluating the
    /// map at every grid point is the expensive part and the caller did it
    /// once already, in parallel or for its own checks. The map still answers
    /// every question asked at an arbitrary point.
    ///
    /// The orientation comes from the grid's own central-difference
    /// normals rather than the map's: only its sign is used, and a grid
    /// fine enough to draw is fine enough for the sign of its volume.
    public init(u: ParamAxis, v: ParamAxis, encloses: Bool, neighbourhood: TubularNeighbourhood? = nil,
                positions: [P3<WorldSpace>],
                map: @escaping @Sendable (P2<ParamSpace>) -> SurfaceJet) {
        precondition(positions.count == u.samples * v.samples,
                     "\(positions.count) positions for a \(u.samples)x\(v.samples) grid")
        self.u = u; self.v = v; self.map = map; self.positions = positions
        self.neighbourhood = encloses ? neighbourhood : nil
        guard encloses else { outward = nil; return }
        func at(_ i: Int, _ j: Int) -> SIMD3<Double> {
            let ii = u.periodic ? u.sample(i) : min(max(i, 0), u.samples - 1)
            let jj = v.periodic ? v.sample(j) : min(max(j, 0), v.samples - 1)
            return positions[jj * u.samples + ii].v
        }
        var volume = 0.0
        for j in 0..<v.samples {
            for i in 0..<u.samples {
                let du = at(i + 1, j) - at(i - 1, j), dv = at(i, j + 1) - at(i, j - 1)
                volume += simd_dot(at(i, j), simd_cross(du, dv))
            }
        }
        precondition(volume.isFinite && abs(volume) > 0,
                     "a surface declared to enclose a solid has no volume")
        outward = volume > 0 ? 1 : -1
    }

    public func position(_ i: Int, _ j: Int) -> P3<WorldSpace> {
        positions[v.sample(j) * u.samples + u.sample(i)]
    }

    /// How squarely the surface faces the viewer at `c`: the cosine between
    /// the outward normal and the direction back toward the eye. Positive
    /// facing, negative facing away, zero on a fold. `nil` for a surface that
    /// has no outside, where the question has no answer.
    public func facing(_ c: P2<ParamSpace>, sight: SIMD3<Double>) -> Double? {
        guard let outward else { return nil }
        let n = map(c).normal
        let len = simd_length(n)
        guard len > 0 else { return 0 }
        return -outward * simd_dot(n, sight) / len
    }
}

public extension ParametricSurface {
    /// Lines of constant `u` -- `u` of them, evenly spaced -- and of constant
    /// `v`, each drawn through `resolution` steps of the other coordinate
    /// with its surface coordinate beside every vertex.
    ///
    /// On a periodic axis the lines sit at `lo + k * length / count` and a
    /// line running along it closes, its last vertex one period past its
    /// first so the coordinates stay continuous; on a bounded axis they
    /// include both ends.
    func parameterLines(counts lines: (u: Int, v: Int),
                        resolution: Int) -> PolylineSet<WorldSpace> {
        var coords: [[P2<ParamSpace>]] = []
        func place(_ axis: ParamAxis, _ k: Int, of count: Int) -> Double {
            axis.range.lo + axis.range.length * Double(k)
                / Double(axis.periodic ? count : max(count - 1, 1))
        }
        func run(_ axis: ParamAxis) -> [Double] {
            (0...resolution).map { axis.range.lo + axis.range.length * Double($0) / Double(resolution) }
        }
        for k in 0..<lines.u {
            let fixed = place(u, k, of: lines.u)
            coords.append(run(v).map { P2<ParamSpace>(fixed, $0) })
        }
        for k in 0..<lines.v {
            let fixed = place(v, k, of: lines.v)
            coords.append(run(u).map { P2<ParamSpace>($0, fixed) })
        }
        // A line at a time, in parallel: the map may be expensive, as a
        // fitted torus's Fourier series is. The lines are settled before
        // the threads start, so each reads an immutable copy.
        let lines = coords
        var paths = [[P3<WorldSpace>]](repeating: [], count: lines.count)
        paths.withUnsafeMutableBufferPointer { out in
            nonisolated(unsafe) let base = out.baseAddress!
            DispatchQueue.concurrentPerform(iterations: lines.count) { k in
                base[k] = lines[k].map { P3(map($0).position) }
            }
        }
        return PolylineSet(paths: paths, coords: lines)
    }

    /// The fold lines for lines of sight along `sight`: where the surface turns
    /// edge-on, `n · sight = 0`, which is where its outline and every inner
    /// silhouette lie.
    ///
    /// Traced by marching squares on that product over the parameter
    /// rectangle -- sampled one step past a periodic axis's end, so a fold
    /// crosses the seam instead of stopping at it -- and every vertex then
    /// moved onto the exact fold by bisection along the product's gradient.
    /// The grid decides which folds exist; the map decides where they run.
    /// Where the sign changes across a crease rather than through a zero
    /// -- a cap's rim, which the rim's own ink draws -- the vertex is
    /// dropped and the line cut there. From a cut the fold is traced on in
    /// steps a sixteenth of a cell, each set onto the fold by the map, to
    /// where it meets the crease or another cut: a fold down a steep wall
    /// can run its whole visible length inside one cell, where the grid
    /// has no vertex for it. Where the grid has none of a fold at all,
    /// `creases` -- the rims, in surface coordinates -- seed it: a fold
    /// meets a crease where the facing a hair to one side of it changes
    /// sign along it, and is traced from there both ways.
    /// They depend on the camera, so they are derived for one, never stored.
    func foldLines(sight: SIMD3<Double>, grid: (u: Int, v: Int)? = nil,
                   creases: [[P2<ParamSpace>]] = []) -> PolylineSet<WorldSpace> {
        let nu = grid?.u ?? u.cells, nv = grid?.v ?? v.cells
        @Sendable func edgeOn(_ c: P2<ParamSpace>) -> Double {
            let n = map(c).normal
            let len = simd_length(n)
            return len > 0 ? simd_dot(n, sight) / len : 0
        }
        let du = u.range.length / Double(nu), dv = v.range.length / Double(nv)
        // Row by row in parallel: each writes only its own row. The map is
        // the expensive part, and this is asked again whenever the camera
        // moves.
        var values = [Float](repeating: 0, count: (nu + 1) * (nv + 1))
        values.withUnsafeMutableBufferPointer { out in
            nonisolated(unsafe) let base = out.baseAddress!
            DispatchQueue.concurrentPerform(iterations: nv + 1) { j in
                for i in 0...nu {
                    base[j * (nu + 1) + i] = Float(edgeOn(P2(u.range.lo + Double(i) * du,
                                                             v.range.lo + Double(j) * dv)))
                }
            }
        }
        let field = Grid2D(width: nu + 1, height: nv + 1,
                           domain: Domain(real: u.range, imag: v.range), values: values)

        // Onto the fold: bracket a sign change along the gradient within a
        // cell either way, then bisect it to the last bit. A fold is a
        // zero the product passes through, changing in proportion to the
        // span across it; at a crease it changes all at once, even where a
        // differenced normal smooths the jump over a hair, so a span a
        // hundredth of the size holds most of it. That is no fold: nil.
        // Nor is a vertex the map finds no zero beside, within a cell
        // either way: the grid saw a sign change the map does not have,
        // as across a crease, where a grid cell holds the cap and the
        // wall.
        let h = 0.25 * min(du, dv)
        @Sendable func gradient(_ c: SIMD2<Double>, _ h: Double) -> SIMD2<Double> {
            SIMD2(edgeOn(P2(c.x + h, c.y)) - edgeOn(P2(c.x - h, c.y)),
                  edgeOn(P2(c.x, c.y + h)) - edgeOn(P2(c.x, c.y - h)))
        }
        // The zero within `r` either way along the unit `e`, or nil.
        @Sendable func onto(_ c: SIMD2<Double>, along e: SIMD2<Double>, within r: Double) -> SIMD2<Double>? {
            let d = e * r
            var a = c - d, b = c + d
            var fa = edgeOn(P2(a))
            guard (fa > 0) != (edgeOn(P2(b)) > 0) else { return nil }
            for _ in 0..<60 {
                let m = 0.5 * (a + b)
                let fm = edgeOn(P2(m))
                if (fm > 0) == (fa > 0) { a = m; fa = fm } else { b = m }
            }
            let m = 0.5 * (a + b)
            func across(_ t: Double) -> Double { abs(edgeOn(P2(m + t * d)) - edgeOn(P2(m - t * d))) }
            return across(1e-3) > 0.5 * across(1e-1) ? nil : m
        }
        func refine(_ c: P2<ParamSpace>) -> P2<ParamSpace>? {
            let g = gradient(c.v, h)
            let len = simd_length(g)
            guard len > 0 else { return c }
            return onto(c.v, along: g / len, within: max(du, dv)).map { P2($0) }
        }

        // The runs, and which of their ends a dropped vertex cut.
        var runs: [(c: [P2<ParamSpace>], cutStart: Bool, cutEnd: Bool)] = []
        for line in Contour.lines(of: field, level: 0) {
            var run: [P2<ParamSpace>] = [], cut = false
            for v in line {
                if let c = refine(P2<ParamSpace>(v.x, v.y)) { run.append(c); continue }
                if run.count >= 2 { runs.append((run, cut, true)) }
                run = []
                cut = true
            }
            if run.count >= 2 { runs.append((run, cut, false)) }
        }

        // On from each cut, a step at a time: along the fold's tangent, then
        // onto the fold along the gradient within a step either way. Where
        // the step no longer finds the fold it has met a crease, and the
        // last step is cut back by bisection to just short of it. It ends
        // at another cut it comes within a step and a half of, and where it
        // comes back onto fold the grid already has.
        let step = min(du, dv) / 16
        let lo = SIMD2(u.range.lo - du, v.range.lo - dv), hi = SIMD2(u.range.hi + du, v.range.hi + dv)
        let scale = SIMD2(du, dv)
        var cuts: [(run: Int, atEnd: Bool)] = []
        for (i, r) in runs.enumerated() {
            if r.cutStart { cuts.append((i, false)) }
            if r.cutEnd { cuts.append((i, true)) }
        }
        let ends = cuts, lines = runs.map { $0.c }
        let endPoint = ends.map { e in lines[e.run][e.atEnd ? lines[e.run].count - 1 : 0].v }
        @Sendable func cell(_ c: SIMD2<Double>) -> SIMD2<Int> {
            SIMD2(Int((c.x / du).rounded(.down)), Int((c.y / dv).rounded(.down)))
        }
        var byCell: [SIMD2<Int>: [(run: Int, at: SIMD2<Double>)]] = [:]
        for (i, r) in lines.enumerated() { for c in r { byCell[cell(c.v), default: []].append((i, c.v)) } }
        let grid = byCell
        // Fold the grid has, within a quarter cell: any run's but this
        // one's own about where the trace set out.
        @Sendable func covered(_ z: SIMD2<Double>, from run: Int, _ p0: SIMD2<Double>) -> Bool {
            let o = cell(z)
            for a in -1...1 { for b in -1...1 {
                for q in grid[o &+ SIMD2(a, b)] ?? [] where simd_length((q.at - z) / scale) < 0.25 {
                    if q.run != run || simd_length((q.at - p0) / scale) > 0.5 { return true }
                }
            } }
            return false
        }
        // From `p0` on the fold, setting out along `heading`; `run` is the
        // run it leaves, -1 for none, and `k` the cut it leaves, -1 for none.
        @Sendable func trace(from p0: SIMD2<Double>, heading h0: SIMD2<Double>,
                             run: Int, cut k: Int) -> (points: [SIMD2<Double>], meets: Int?) {
            var heading = h0
            var p = p0, out: [SIMD2<Double>] = []
            func next(_ t: SIMD2<Double>, _ tau: Double) -> SIMD2<Double>? {
                let q = p + tau * step * t
                guard all(q .> lo), all(q .< hi) else { return nil }
                let g = gradient(q, 0.25 * step)
                let len = simd_length(g)
                return len > 0 ? onto(q, along: g / len, within: step) : nil
            }
            for _ in 0..<128 {
                let g = gradient(p, 0.25 * step)
                let len = simd_length(g)
                guard len > 0 else { break }
                var t = SIMD2(-g.y, g.x) / len
                if simd_dot(t, heading) < 0 { t = -t }
                guard let z = next(t, 1) else {
                    var a = 0.0, b = 1.0, last: SIMD2<Double>?
                    for _ in 0..<30 {
                        let mid = 0.5 * (a + b)
                        if let z = next(t, mid) { a = mid; last = z } else { b = mid }
                    }
                    if let last { out.append(last) }
                    break
                }
                if let j = endPoint.indices.first(where: { $0 != k && simd_length(endPoint[$0] - z) < 1.5 * step }) {
                    return (out + [z, endPoint[j]], j)
                }
                if covered(z, from: run, p0) { break }
                out.append(z)
                heading = z - p
                p = z
            }
            return (out, nil)
        }
        var traced = [(points: [SIMD2<Double>], meets: Int?)](repeating: ([], nil), count: ends.count)
        traced.withUnsafeMutableBufferPointer { out in
            nonisolated(unsafe) let base = out.baseAddress!
            DispatchQueue.concurrentPerform(iterations: ends.count) { k in
                // The heading from a vertex half a cell back: two grid
                // vertices in one cell can come out of refinement reversed.
                let r = lines[ends[k].run], p0 = endPoint[k]
                let back = ends[k].atEnd ? Array(r.reversed().dropFirst()) : Array(r.dropFirst())
                let behind = back.first { simd_length((p0 - $0.v) / scale) > 0.5 } ?? back[0]
                base[k] = trace(from: p0, heading: p0 - behind.v, run: ends[k].run, cut: k)
            }
        }
        // Two cuts traced to each other are one piece of fold: keep one.
        for k in ends.indices where !traced[k].points.isEmpty {
            if let j = traced[k].meets, traced[j].meets == k, j < k { continue }
            let pts = traced[k].points.map { P2<ParamSpace>($0) }
            let i = ends[k].run
            runs[i].c = ends[k].atEnd ? runs[i].c + pts : pts.reversed() + runs[i].c
        }
        var paths: [[P3<WorldSpace>]] = [], coords: [[P2<ParamSpace>]] = []
        for r in runs { coords.append(r.c) }

        // From the creases: wherever the facing a fiftieth of a cell to one
        // side of a crease changes sign between two of its vertices, a fold
        // meets it there. One already ended within a tenth of a cell is
        // drawn; any other is found on that line beside the crease and
        // traced both ways, to the crease and away from it.
        var drawn: [SIMD2<Int>: [SIMD2<Double>]] = [:]
        for r in coords { for c in r { drawn[cell(c.v), default: []].append(c.v) } }
        func reached(_ j: SIMD2<Double>) -> Bool {
            let o = cell(j)
            for a in -1...1 { for b in -1...1 {
                if (drawn[o &+ SIMD2(a, b)] ?? []).contains(where: { simd_length(($0 - j) / scale) < 0.1 }) { return true }
            } }
            return false
        }
        var seeds: [(a: SIMD2<Double>, b: SIMD2<Double>, off: SIMD2<Double>)] = []
        for line in creases where line.count >= 2 {
            for i in 0..<(line.count - 1) {
                let a = line[i].v, b = line[i + 1].v
                let along = (b - a) / scale
                guard simd_length(along) > 0 else { continue }
                let off = 0.02 * SIMD2(-along.y, along.x) / simd_length(along) * scale
                for side in [1.0, -1.0] {
                    let fa = edgeOn(P2(a + side * off)), fb = edgeOn(P2(b + side * off))
                    guard (fa > 0) != (fb > 0) else { continue }
                    if !reached(a + fa / (fa - fb) * (b - a)) { seeds.append((a + side * off, b + side * off, side * off)) }
                }
            }
        }
        let starts = seeds
        var seeded = [[P2<ParamSpace>]](repeating: [], count: starts.count)
        seeded.withUnsafeMutableBufferPointer { out in
            nonisolated(unsafe) let base = out.baseAddress!
            DispatchQueue.concurrentPerform(iterations: starts.count) { k in
                // Onto the fold along the line beside the crease, which
                // brackets it and never crosses the crease.
                let (a, b) = (starts[k].a, starts[k].b)
                guard let p0 = onto(0.5 * (a + b), along: simd_normalize(b - a), within: 0.5 * simd_length(b - a))
                else { return }
                let toward = trace(from: p0, heading: -starts[k].off, run: -1, cut: -1).points
                let away = trace(from: p0, heading: starts[k].off, run: -1, cut: -1).points
                base[k] = (toward.reversed() + [p0] + away).map { P2($0) }
            }
        }
        for c in seeded where c.count >= 2 { coords.append(c) }

        for c in coords { paths.append(c.map { P3(map($0).position) }) }
        return PolylineSet(paths: paths, coords: coords)
    }

    /// The grid's unit normals, by central differences on the grid --
    /// one-sided at a bounded axis's ends -- `u` fastest, as `positions`.
    ///
    /// They do not depend on the camera, so they are computed once per
    /// surface and handed to `gridFoldLines` for every camera after.
    /// Where the grid is degenerate the normal is zero.
    func gridNormals() -> [SIMD3<Float>] {
        let nu = u.samples, nv = v.samples
        @Sendable func at(_ i: Int, _ j: Int) -> SIMD3<Double> {
            let ii = u.periodic ? u.sample(i) : min(max(i, 0), nu - 1)
            let jj = v.periodic ? v.sample(j) : min(max(j, 0), nv - 1)
            return positions[jj * nu + ii].v
        }
        var normals = [SIMD3<Float>](repeating: .zero, count: nu * nv)
        normals.withUnsafeMutableBufferPointer { out in
            nonisolated(unsafe) let base = out.baseAddress!
            DispatchQueue.concurrentPerform(iterations: nv) { j in
                for i in 0..<nu {
                    let n = simd_cross(at(i + 1, j) - at(i - 1, j), at(i, j + 1) - at(i, j - 1))
                    let len = simd_length(n)
                    base[j * nu + i] = len > 0 ? SIMD3<Float>(n / len) : .zero
                }
            }
        }
        return normals
    }

    /// The fold lines as the grid sees them: `foldLines` without asking
    /// the map anything.
    ///
    /// Marching squares on `n · sight` at the grid points, from normals
    /// computed once (`gridNormals`), and each vertex placed by bilinear
    /// interpolation of the grid -- no refinement onto the exact fold. So
    /// a fold is as accurate as the grid is fine, which at a drawing
    /// grid is well under a pixel, and costs a dot product per grid
    /// point: fast enough to follow the camera. The preview draws these; the
    /// bake keeps `foldLines`, whose vertices are exact.
    func gridFoldLines(normals: [SIMD3<Float>],
                          sight: SIMD3<Double>) -> PolylineSet<WorldSpace> {
        precondition(normals.count == positions.count,
                     "\(normals.count) normals for \(positions.count) grid points")
        // One column and row past a periodic axis's end, as `foldLines`
        // samples: the seam's cells close on the first samples.
        let nu = u.cells, nv = v.cells, stride = u.samples
        let s = SIMD3<Float>(sight)
        var values = [Float](repeating: 0, count: (nu + 1) * (nv + 1))
        values.withUnsafeMutableBufferPointer { out in
            nonisolated(unsafe) let base = out.baseAddress!
            DispatchQueue.concurrentPerform(iterations: nv + 1) { j in
                let row = v.sample(j) * stride
                for i in 0...nu {
                    base[j * (nu + 1) + i] = simd_dot(normals[row + u.sample(i)], s)
                }
            }
        }
        let field = Grid2D(width: nu + 1, height: nv + 1,
                           domain: Domain(real: u.range, imag: v.range), values: values)

        var paths: [[P3<WorldSpace>]] = [], coords: [[P2<ParamSpace>]] = []
        for line in Contour.lines(of: field, level: 0) {
            let c = line.map { P2<ParamSpace>($0.x, $0.y) }
            coords.append(c)
            paths.append(c.map(interpolate))
        }
        return PolylineSet(paths: paths, coords: coords)
    }

    /// The grid's bilinear patch at a coordinate: the surface as it is
    /// drawn, rather than as the map has it.
    ///
    /// The grid is what gets rasterized, so ink placed this way lies on
    /// the drawn surface to the last bit, and it costs four grid reads
    /// where the map may cost a Fourier series. A coordinate any number of
    /// periods past a periodic axis's range wraps; on a bounded axis it is
    /// clamped to the end cell.
    func interpolate(_ c: P2<ParamSpace>) -> P3<WorldSpace> {
        let (i, j, a, b) = cell(c)
        let p00 = position(i, j).v, p10 = position(i + 1, j).v
        let p01 = position(i, j + 1).v, p11 = position(i + 1, j + 1).v
        return P3((1 - b) * ((1 - a) * p00 + a * p10) + b * ((1 - a) * p01 + a * p11))
    }

    /// The grid's normals (`gridNormals`) at a coordinate, by the same
    /// bilinear patch as `interpolate`, unit length; zero where the patch's
    /// corners cancel.
    ///
    /// Within a grid cell of the exact normal, and so, like a grid
    /// fold, within a pixel of where the facing changes at any drawing
    /// grid: enough for the preview, which asks a normal of every ink
    /// vertex and asks it again whenever the ink moves. The bake keeps the
    /// map's.
    func interpolateNormal(_ c: P2<ParamSpace>, normals: [SIMD3<Float>]) -> SIMD3<Float> {
        precondition(normals.count == positions.count,
                     "\(normals.count) normals for \(positions.count) grid points")
        let (i, j, a, b) = cell(c)
        func at(_ i: Int, _ j: Int) -> SIMD3<Float> {
            normals[v.sample(j) * u.samples + u.sample(i)]
        }
        let (fa, fb) = (Float(a), Float(b))
        let n = (1 - fb) * ((1 - fa) * at(i, j) + fa * at(i + 1, j))
            + fb * ((1 - fa) * at(i, j + 1) + fa * at(i + 1, j + 1))
        let len = simd_length(n)
        return len > 0 ? n / len : .zero
    }

    /// The grid cell a coordinate falls in, and where in it: the lower
    /// corner's unwrapped indices and the fractions along each axis. A
    /// periodic axis wraps any number of periods away; a bounded one clamps
    /// to its end cell.
    private func cell(_ c: P2<ParamSpace>) -> (i: Int, j: Int, a: Double, b: Double) {
        func split(_ x: Double, _ axis: ParamAxis) -> (Int, Double) {
            let f = (x - axis.range.lo) / axis.spacing
            var k = Int(f.rounded(.down))
            if !axis.periodic { k = min(max(k, 0), axis.cells - 1) }
            return (k, f - Double(k))
        }
        let (i, a) = split(c.x, u), (j, b) = split(c.y, v)
        return (i, j, a, b)
    }

    /// Windings: straight lines through the parameter rectangle, `slope`
    /// turns of `v` per turn of `u`, glued into curves on the torus.
    ///
    /// A winding is what a torus's coordinates make of a straight line, and
    /// the same line on every torus with these coordinates: it lives on the
    /// flat torus, and the map -- any map -- is what puts it in space. The
    /// parameter lines are its two ends, slope zero and slope infinite; the
    /// trajectory of a forced system, in the coordinates of its fitted torus,
    /// is the one with the system's own rotation number.
    ///
    /// `count` strands start at `start` -- the rectangle's origin unless
    /// given -- and evenly spaced from it along `v`, and each runs `turns`
    /// turns of `u` -- or, when `slope` is a fraction `p/q` with `q ≤
    /// turns`, the `q` turns after which it closes, so a rational winding is
    /// one closed (p, q) curve and an irrational one is a stretch of a curve
    /// that never closes. A vertex per grid cell along the steeper axis,
    /// placed by `interpolate`, so the ink lies on the drawn surface; each
    /// vertex carries its coordinate, continuous along a path and never more
    /// than eight turns from zero, so the preview's float32 still resolves a
    /// pixel's worth of it: a path is ended and the next begun from the same
    /// vertex every eight turns.
    func winding(slope: Double, turns: Int, count: Int = 1,
                 start: P2<ParamSpace>? = nil) -> PolylineSet<WorldSpace> {
        precondition(u.periodic && v.periodic, "a winding is a curve on a torus")
        precondition(turns >= 1 && count >= 1 && slope.isFinite,
                     "a winding wants a finite slope and at least one turn of one strand")
        let drawn = Self.windingCloses(slope: slope, within: turns) ?? turns
        let steps = max(u.cells, Int((abs(slope) * Double(v.cells)).rounded(.up)))
        let du = u.range.length / Double(steps), dv = slope * v.range.length / Double(steps)
        let span = 8 * u.range.length
        let origin = start ?? P2(u.range.lo, v.range.lo)
        var paths: [[P3<WorldSpace>]] = [], coords: [[P2<ParamSpace>]] = []
        for strand in 0..<count {
            let v0 = origin.y + v.range.length * Double(strand) / Double(count)
            var path: [P3<WorldSpace>] = [], coord: [P2<ParamSpace>] = []
            var offset = SIMD2(0.0, 0.0)
            for m in 0...(drawn * steps) {
                let c = P2<ParamSpace>(origin.x + Double(m) * du, v0 + Double(m) * dv)
                if !path.isEmpty, c.x - origin.x - offset.x > span {
                    // The seam of a path: the same vertex ends one and
                    // begins the next, its coordinates brought back near
                    // the rectangle's origin by whole periods.
                    paths.append(path); coords.append(coord)
                    let last = coord[coord.count - 1].v + offset
                    offset = SIMD2(
                        u.range.length * ((c.x - u.range.lo) / u.range.length).rounded(.down),
                        v.range.length * ((c.y - v.range.lo) / v.range.length).rounded(.down))
                    path = [path[path.count - 1]]
                    coord = [P2(last - offset)]
                }
                path.append(interpolate(c))
                coord.append(P2(c.v - offset))
            }
            paths.append(path); coords.append(coord)
        }
        return PolylineSet(paths: paths, coords: coords)
    }

    /// The turns after which a winding of `slope` closes -- the denominator
    /// of a fraction, to within rounding -- when that is within `turns`;
    /// nil for a winding that runs out its turns open.
    static func windingCloses(slope: Double, within turns: Int) -> Int? {
        guard turns >= 1 else { return nil }
        return (1...turns).first { k in
            let w = slope * Double(k)
            return abs(w - w.rounded()) <= 1e-9 * max(1, abs(w))
        }
    }

    /// The torus of revolution about the z axis: `u` around the axis, `v`
    /// around the tube.
    ///
    /// It bounds a solid exactly when the tube does not reach the axis,
    /// `major > minor`; the horn and spindle tori pass through themselves.
    static func torus(major R: Double, minor r: Double,
                      samples: (u: Int, v: Int)) -> ParametricSurface {
        torus(major: R, minor: r, u: .angle(samples: samples.u), v: .angle(samples: samples.v))
    }

    /// The same torus over any two periodic axes: each axis's range is one
    /// turn, so a rectangle of periods -- a doubly periodic function's
    /// fundamental domain -- glues into it edge to edge.
    static func torus(major R: Double, minor r: Double,
                      u: ParamAxis, v: ParamAxis) -> ParametricSurface {
        precondition(u.periodic && v.periodic, "a torus is periodic both ways")
        let (su0, sv0) = (u.range.lo, v.range.lo)
        let (ku, kv) = (2 * Double.pi / u.range.length, 2 * Double.pi / v.range.length)
        // The solid: a point's foot is the nearest point of the core circle
        // turned toward it, its distance how far it stands from that circle
        // less the tube's radius. The ring angle moves fastest where the
        // ring is narrowest, the tube angle where the tube is.
        let neighbourhood = TubularNeighbourhood(
            foot: { q in
                var a = atan2(q.y, q.x), rho = (q.x * q.x + q.y * q.y).squareRoot()
                var b = atan2(q.z, rho - R)
                if a < 0 { a += 2 * .pi }
                if b < 0 { b += 2 * .pi }
                let tube = ((rho - R) * (rho - R) + q.z * q.z).squareRoot()
                return (P2(su0 + a / ku, sv0 + b / kv), tube - r)
            },
            normalOffset: { _ in 0 }, above: 0, below: 0, lipschitz: 1,
            spread: { outside, inside in
                SIMD2(1 / (ku * max(R - r - outside, 1e-12)), 1 / (kv * max(r - inside, 1e-12)))
            },
            centre: .zero, radius: R + r)
        return ParametricSurface(u: u, v: v, encloses: R > r, neighbourhood: neighbourhood) { c in
            let a = ku * (c.x - su0), b = kv * (c.y - sv0)
            let (ca, sa, cb, sb) = (cos(a), sin(a), cos(b), sin(b))
            let ring = R + r * cb
            return SurfaceJet(position: SIMD3(ring * ca, ring * sa, r * sb),
                              du: ku * SIMD3(-ring * sa, ring * ca, 0),
                              dv: kv * SIMD3(-r * sb * ca, -r * sb * sa, r * cb))
        }
    }
}

public extension Transform where A == WorldSpace, B == ViewSpace {
    /// The direction a line of sight travels through the world, unit length.
    ///
    /// Under an affine camera the eye looks along the one direction that
    /// changes neither view x nor view y -- the kernel of the first two rows
    /// -- signed so that travelling it takes view z down, away from the eye.
    /// That is *not* the gradient of view z unless the camera is orthonormal,
    /// and the plate cameras are not: they shear. Asking the gradient instead
    /// tilts every fold by the shear.
    var projectionDirection: SIMD3<Double> {
        let c = m.columns
        let row0 = SIMD3(c.0.x, c.1.x, c.2.x)
        let row1 = SIMD3(c.0.y, c.1.y, c.2.y)
        let row2 = SIMD3(c.0.z, c.1.z, c.2.z)
        var d = simd_normalize(simd_cross(row0, row1))
        if simd_dot(row2, d) > 0 { d = -d }
        return d
    }
}

public extension ParametricSurface {
    /// This surface displaced along its normal: p ↦ p + amplitude · height(p) · n(p).
    ///
    /// A landscape is the plane displaced along its normal by |f|; the same
    /// sentence over any base surface puts the landscape on it. The result
    /// is the image of a section of the base's normal bundle under its
    /// normal exponential map, which is an embedding as long as the
    /// displacement stays under the base's reach -- for a round torus,
    /// min(r, R − r) -- and `encloses` is the caller's statement that it
    /// does. The jet is the chain rule, d(p + h n) = dp + dh ⊗ n + h dn,
    /// with dh and dn, the shape operator of the base, by central
    /// differences on the base's map.
    /// `ceiling` bounds the height, for the displaced solid's shell: the
    /// displacement is never more than `amplitude` times it.
    func displaced(by height: @escaping @Sendable (P2<ParamSpace>) -> Double,
                   amplitude: Double, ceiling: Double, encloses: Bool) -> ParametricSurface {
        let base = map, sign = outward ?? 1
        let eu = 1e-5 * u.spacing, ev = 1e-5 * v.spacing
        @Sendable func unitNormal(_ c: P2<ParamSpace>) -> SIMD3<Double> {
            let n = base(c).normal
            return sign * n / max(simd_length(n), .leastNormalMagnitude)
        }
        @Sendable func h(_ c: P2<ParamSpace>) -> Double { amplitude * height(c) }
        let displaced = ParametricSurface(u: u, v: v, encloses: encloses) { c in
            let jet = base(c), n = unitNormal(c), hc = h(c)
            let cu1 = P2<ParamSpace>(c.x + eu, c.y), cu0 = P2<ParamSpace>(c.x - eu, c.y)
            let cv1 = P2<ParamSpace>(c.x, c.y + ev), cv0 = P2<ParamSpace>(c.x, c.y - ev)
            let dnu = (unitNormal(cu1) - unitNormal(cu0)) / (2 * eu)
            let dnv = (unitNormal(cv1) - unitNormal(cv0)) / (2 * ev)
            let dhu = (h(cu1) - h(cu0)) / (2 * eu), dhv = (h(cv1) - h(cv0)) / (2 * ev)
            return SurfaceJet(position: jet.position + hc * n,
                              du: jet.du + dhu * n + hc * dnu,
                              dv: jet.dv + dhv * n + hc * dnv)
        }
        // The displaced solid, when the base has one: the same foot, the
        // displacement added to the normal offset. The gap's Lipschitz constant
        // needs the displacement's steepest slope, read off the grid
        // just built -- the displacement at a grid point is what was
        // added to the base there -- in parameter units, times how fast
        // the foot's coordinate can move; with half as much again, for the
        // slope between grid points.
        guard encloses, let e = neighbourhood else { return displaced }
        let nu = u.samples, nv = v.samples
        var normalOffsets = [Double](repeating: 0, count: nu * nv)
        normalOffsets.withUnsafeMutableBufferPointer { out in
            nonisolated(unsafe) let dst = out.baseAddress!
            DispatchQueue.concurrentPerform(iterations: nv) { j in
                for i in 0..<nu {
                    let c = P2<ParamSpace>(u.coordinate(i), v.coordinate(j))
                    dst[j * nu + i] = simd_dot(displaced.positions[j * nu + i].v - base(c).position,
                                               unitNormal(c))
                }
            }
        }
        var su = 0.0, sv = 0.0
        for j in 0..<nv {
            for i in 0..<nu {
                let t = normalOffsets[j * nu + i]
                if i + 1 < nu || u.periodic {
                    su = max(su, abs(normalOffsets[j * nu + u.sample(i + 1)] - t) / u.spacing)
                }
                if j + 1 < nv || v.periodic {
                    sv = max(sv, abs(normalOffsets[v.sample(j + 1) * nu + i] - t) / v.spacing)
                }
            }
        }
        let above = e.above + max(amplitude, 0) * ceiling
        let below = e.below + max(-amplitude, 0) * ceiling
        let spread = e.spread(above, below)
        let normalOffset = e.normalOffset
        let lifted = TubularNeighbourhood(
            foot: e.foot, normalOffset: { c in normalOffset(c) + h(c) },
            above: above, below: below,
            lipschitz: e.lipschitz + 1.5 * (su * spread.x + sv * spread.y),
            spread: e.spread, centre: e.centre, radius: e.radius + above)
        return ParametricSurface(u: u, v: v, encloses: true, neighbourhood: lifted,
                                 positions: displaced.positions, map: displaced.map)
    }
}
