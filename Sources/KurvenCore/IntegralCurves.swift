import Foundation
import simd

/// Integral curves of a vector field on a face (do Carmo §3-4): the curves
/// whose tangent at every point is the field there. The slope lines of a
/// scalar field are the integral curves of its gradient (Cayley 1859, "On
/// contour and slope lines"); a linear curve on a torus is one of a constant
/// field, and so is a forced system's trajectory on its fitted torus.
///
/// A field is a vector field, signed (D9). Where its integral curves are
/// known in closed form, as a constant field's straight lines are, they are
/// read off it (`VectorField.flow`); otherwise they are integrated on the
/// face's parameter plane with RK4 at unit speed, from a seed both ways
/// (`Face.integralCurve(of:through:on:limits:)`).
///
/// The integrator keeps the heading continuous: a step along which the
/// field turns a right angle or more from the heading it began with is
/// refused and halved, so no step crosses a singular point of the field,
/// where its direction is undefined, as at a zero or a pole of f for the
/// slope of |f|. Each step is RK4 taken once whole and once in two halves,
/// and is refused where the two disagree by more than `accuracy`, where the
/// chord to it sags off the curve by more than half the refiner's
/// tolerance, or, on the graph, where the lifted chord sags off the graph
/// by as much: so every chord of the polyline is within the tolerance of
/// the curve it stands for, on the face and in space. A curve ends:
///
/// - at a singular point, where the steps refused there have shrunk below
///   `leastStep`, carried to the point itself when the field can name it
///   (`VectorField.singularPoint`: a zero of f);
/// - where it leaves the copy's trimmed domain, at the domain's edge, the
///   footprint's perimeter or the truncation edge, bisected on membership
///   (`Face.Copy.contains`) along the step to the last bit, so on the
///   truncation edge it ends where |f| is the truncation height;
/// - back at its seed, a closed curve, which needs nothing traced backward;
/// - at `maxSteps` steps each way, the step limit.
public struct VectorField: Sendable {
    /// The field at a point of the face's parameter plane, in its
    /// coordinates; nil where it is undefined or zero: at a singular point.
    public let value: @Sendable (P2<ParamSpace>) -> SIMD2<Double>?
    /// The point a time `t` along the integral curve through `p`, where the
    /// curves are known in closed form; nil where they are integrated.
    public let flow: (@Sendable (P2<ParamSpace>, Double) -> P2<ParamSpace>)?
    /// The singular point an integral curve that ends beside `p` runs into,
    /// where the field knows its singular points: a zero of f, for the slope
    /// of |f|. Nil, or nil from it, where it does not.
    public let singularPoint: (@Sendable (P2<ParamSpace>) -> P2<ParamSpace>?)?

    public init(value: @escaping @Sendable (P2<ParamSpace>) -> SIMD2<Double>?,
                flow: (@Sendable (P2<ParamSpace>, Double) -> P2<ParamSpace>)? = nil,
                singularPoint: (@Sendable (P2<ParamSpace>) -> P2<ParamSpace>?)? = nil) {
        self.value = value; self.flow = flow; self.singularPoint = singularPoint
    }

    /// The constant field `w`: its integral curves are the straight lines
    /// `p + t w`, in closed form, so a curve sampled at whole times is the
    /// line stepped by `w`, each coordinate `p + t · w` to the bit.
    public static func constant(_ w: SIMD2<Double>) -> VectorField {
        VectorField(value: { _ in w }, flow: { p, t in P2(p.x + t * w.x, p.y + t * w.y) })
    }
}

/// How an integral curve is traced, and why it ends where it does.
public enum IntegralCurve {
    /// Why a traced integral curve ends.
    public enum End: Sendable, Equatable {
        /// At a singular point of the field: the steps refused there have
        /// shrunk below `Limits.leastStep`.
        case singular
        /// Where it leaves the copy's trimmed domain.
        case boundary
        /// Back at its seed.
        case closed
        /// At `Limits.maxSteps`.
        case stepLimit
    }

    /// The tolerances a curve is traced to, lengths in cells of the face's
    /// parameter rectangle (`Face.cell`).
    public struct Limits: Sendable, Equatable {
        /// How far a chord may depart from the curve it stands for: the
        /// refiner's tolerance (`ContourRefine.tolerance`). A step's chord is
        /// held to half of it, on the face and lifted, and a curve closes
        /// where it passes within half of it of its seed.
        public var tolerance: Double
        /// The longest step.
        public var largestStep: Double
        /// The shortest step: one refused below it ends the curve at a
        /// singular point.
        public var leastStep: Double
        /// How far a whole RK4 step may land from the same step in two
        /// halves.
        public var accuracy: Double
        /// The most steps each way from the seed.
        public var maxSteps: Int

        public init(tolerance: Double = 0.02, largestStep: Double = 0.5, leastStep: Double = 1e-4,
                    accuracy: Double = 1e-7, maxSteps: Int = 20_000) {
            self.tolerance = tolerance; self.largestStep = largestStep; self.leastStep = leastStep
            self.accuracy = accuracy; self.maxSteps = maxSteps
        }
    }

    /// An integral curve traced from a seed both ways: its polyline in the
    /// parameter plane, from its backward end to its forward end, and why
    /// each end is where it is. A closed curve begins and ends at its seed,
    /// both ends `.closed`.
    public struct Traced: Sendable {
        public var path: [P2<ParamSpace>]
        public var backward: End
        public var forward: End
    }
}

public extension Face {
    /// The length the face's tolerances are measured in: the larger spacing
    /// of its parameter rectangle's grid. On the graph, a cell of the grid.
    var cell: Double { max(abs(domain.u.spacing), abs(domain.v.spacing)) }

    /// The integral curve of `field` through `seed` on `copy`, traced both
    /// ways from it (`VectorField`), or nil where the seed is off the copy's
    /// trimmed domain or the field is undefined there. Coordinates run on
    /// along a periodic axis, unwrapped, so the polyline never jumps a
    /// period.
    func integralCurve(of field: VectorField, through seed: P2<ParamSpace>, on copy: Copy,
                       limits: IntegralCurve.Limits = IntegralCurve.Limits()) -> IntegralCurve.Traced? {
        guard copy.contains(seed), let w = field.value(seed), simd_length(w) > 0, simd_length(w).isFinite else {
            return nil
        }
        let forward = trace(field, from: seed.v, sign: 1, on: copy, limits: limits)
        if forward.end == .closed {
            return IntegralCurve.Traced(path: forward.path.map { P2($0) }, backward: .closed, forward: .closed)
        }
        let backward = trace(field, from: seed.v, sign: -1, on: copy, limits: limits)
        let path = backward.path.reversed() + forward.path.dropFirst()
        return IntegralCurve.Traced(path: path.map { P2($0) }, backward: backward.end, forward: forward.end)
    }

    /// One way of an integral curve from `seed`, along the field's own sign
    /// or against it: the polyline from the seed, and why it ends.
    private func trace(_ field: VectorField, from seed: SIMD2<Double>, sign: Double, on copy: Copy,
                       limits: IntegralCurve.Limits) -> (path: [SIMD2<Double>], end: IntegralCurve.End) {
        let cell = self.cell
        let largest = limits.largestStep * cell, least = limits.leastStep * cell
        let accuracy = limits.accuracy * cell, sag = 0.5 * limits.tolerance * cell
        let closeTo = 0.5 * limits.tolerance * cell
        let periods = SIMD2(domain.u.period ?? 0, domain.v.period ?? 0)
        let lift = self.lift
        func inside(_ p: SIMD2<Double>) -> Bool { copy.contains(P2(p)) }
        /// The unit direction of travel at `p`, nil at a singular point.
        func direction(_ p: SIMD2<Double>) -> SIMD2<Double>? {
            guard let w = field.value(P2(p)) else { return nil }
            let n = simd_length(w)
            guard n > 0, n.isFinite else { return nil }
            return sign * w / n
        }
        /// One RK4 step of length `h` from `p`, whose direction is `k1`;
        /// nil where a stage is undefined or turns a right angle or more
        /// from `heading`.
        func step(_ p: SIMD2<Double>, _ h: Double, _ k1: SIMD2<Double>, heading: SIMD2<Double>) -> SIMD2<Double>? {
            guard let k2 = direction(p + 0.5 * h * k1), simd_dot(k2, heading) > 0,
                  let k3 = direction(p + 0.5 * h * k2), simd_dot(k3, heading) > 0,
                  let k4 = direction(p + h * k3), simd_dot(k4, heading) > 0 else { return nil }
            let middle: SIMD2<Double> = k2 + k3
            let sum: SIMD2<Double> = k1 + k4 + 2 * middle
            return p + (h / 6) * sum
        }
        /// The singular point the curve ends at, beside `p`, where the field
        /// names one on the copy.
        func ended(_ path: [SIMD2<Double>]) -> [SIMD2<Double>] {
            guard let p = path.last, let s = field.singularPoint?(P2(p)), inside(s.v),
                  simd_distance(s.v, p) <= limits.tolerance * cell, s.v != p else { return path }
            return path + [s.v]
        }

        var path = [seed], p = seed
        var h = largest, steps = 0, attempts = 0
        var left = false
        let start = direction(seed)
        var heading = start
        while true {
            guard steps < limits.maxSteps, attempts < 10 * limits.maxSteps else { return (path, .stepLimit) }
            attempts += 1
            guard let k1 = heading else { return (ended(path), .singular) }
            // The step whole, and in two halves; the halves are kept.
            guard let whole = step(p, h, k1, heading: k1),
                  let mid = step(p, 0.5 * h, k1, heading: k1),
                  let k1Mid = direction(mid), simd_dot(k1Mid, k1) > 0,
                  let q = step(mid, 0.5 * h, k1Mid, heading: k1) else {
                h *= 0.5
                if h < least { return (ended(path), .singular) }
                continue
            }
            let error = simd_distance(q, whole)
            var sagged = simd_distance(mid, 0.5 * (p + q)) > sag
            if !sagged, let lift {
                sagged = simd_distance(lift(P2(mid)), 0.5 * (lift(P2(p)) + lift(P2(q)))) > sag
            }
            if !(error <= accuracy) || sagged {
                h *= 0.5
                if h < least { return (ended(path), .singular) }
                continue
            }
            // Out of the trimmed domain within the step: the half that
            // leaves it bisected on membership, a step of that half's own
            // from its start, to the last bit, and the last point inside
            // kept. At the half's whole length the step is the half itself,
            // outside, and at none its start, inside.
            if !inside(mid) || !inside(q) {
                let (from, k) = inside(mid) ? (mid, k1Mid) : (p, k1)
                if from == mid { path.append(mid) }
                var lo = 0.0, hi = 0.5 * h
                for _ in 0..<64 where hi - lo > 1e-15 * h {
                    let t = 0.5 * (lo + hi)
                    if let r = step(from, t, k, heading: k1), inside(r) { lo = t } else { hi = t }
                }
                if lo > 0, let end = step(from, lo, k, heading: k1) { path.append(end) }
                return (path, .boundary)
            }
            // Back at the seed, heading the way it set out: closed, on the
            // seed's own image a whole number of periods along.
            if sign > 0, left, let start, simd_dot(k1, start) > 0 {
                var image = seed
                for axis in 0..<2 where periods[axis] > 0 {
                    image[axis] += periods[axis] * ((q[axis] - seed[axis]) / periods[axis]).rounded()
                }
                let d = q - p, t = simd_dot(d, d) > 0 ? min(max(simd_dot(image - p, d) / simd_dot(d, d), 0), 1) : 0
                if simd_distance(p + t * d, image) <= closeTo {
                    path.append(image)
                    return (path, .closed)
                }
            }
            path.append(q)
            p = q
            heading = direction(q)
            steps += 1
            if !left, simd_distance(q, seed) > 4 * closeTo { left = true }
            let grow = error > 0 ? min(2, 0.9 * Foundation.pow(accuracy / error, 0.2)) : 2
            h = min(largest, h * max(grow, 1))
        }
    }

    /// The slope of a scalar field on the face: its gradient in the metric
    /// the face's map induces, `g⁻¹ dφ`, so its integral curves cross the
    /// field's level sets at right angles on the face, not merely in the
    /// parameter plane. `dφ` by central differences of `value` a
    /// ten-thousandth of a cell wide, as the graph's normal is read
    /// (`Surface.graphNormal`), each taken the short way round a field of
    /// `period`. Undefined where `value` is not a number, the gradient
    /// vanishes or the map's derivatives are degenerate.
    func slope(of value: @escaping @Sendable (P2<ParamSpace>) -> Double, period: Double? = nil,
               singularPoint: (@Sendable (P2<ParamSpace>) -> P2<ParamSpace>?)? = nil) -> VectorField {
        let h = 1e-4 * cell, map = self.map
        @Sendable func difference(_ a: Double, _ b: Double) -> Double {
            guard let period else { return a - b }
            let d = a - b
            return d - period * (d / period).rounded()
        }
        return VectorField(value: { p in
            let du = difference(value(P2(p.x + h, p.y)), value(P2(p.x - h, p.y))) / (2 * h)
            let dv = difference(value(P2(p.x, p.y + h)), value(P2(p.x, p.y - h))) / (2 * h)
            return Face.raised(SIMD2(du, dv), map(p))
        }, singularPoint: singularPoint)
    }

    /// The slope of the height, the map's z, read off the map's own jet:
    /// `dφ` is the z of its two derivatives. On the graph the height is |f|,
    /// and its slope lines are the lines of steepest ascent and descent.
    func heightSlope(singularPoint: (@Sendable (P2<ParamSpace>) -> P2<ParamSpace>?)? = nil) -> VectorField {
        let map = self.map
        return VectorField(value: { p in
            let jet = map(p)
            return Face.raised(SIMD2(jet.du.z, jet.dv.z), jet)
        }, singularPoint: singularPoint)
    }

    /// A differential raised to a vector by the first fundamental form of
    /// `jet`: `g⁻¹ dφ`. Nil where it is not finite, is zero, or `g` is
    /// singular.
    static func raised(_ d: SIMD2<Double>, _ jet: SurfaceJet) -> SIMD2<Double>? {
        let e = simd_dot(jet.du, jet.du), f = simd_dot(jet.du, jet.dv), g = simd_dot(jet.dv, jet.dv)
        let det = e * g - f * f
        guard det > 0, det.isFinite, d.x.isFinite, d.y.isFinite else { return nil }
        let w = SIMD2(g * d.x - f * d.y, e * d.y - f * d.x) / det
        guard w != .zero, w.x.isFinite, w.y.isFinite else { return nil }
        return w
    }
}

public extension Seeds {
    /// The seeds along `paths`, polylines of the level set in the parameter
    /// plane, in order along each. At listed values of a second field,
    /// `value` read along each chord, taken the short way round a field of
    /// `period`: each crossing of a value bisected along its chord to the last
    /// bit, a chord across the jump half a period from the value skipped, and
    /// a crossing at a vertex counted once. Nothing without `value`. By arc
    /// length, measured on the curve in space, the polyline's vertices
    /// carried there by `position`.
    func points(along paths: [[P2<ParamSpace>]], value: ((P2<ParamSpace>) -> Double)?, period: Double? = nil,
                position: (P2<ParamSpace>) -> SIMD3<Double>) -> [P2<ParamSpace>] {
        var seeds: [P2<ParamSpace>] = []
        switch spacing {
        case .values(_, let values):
            guard let value else { return [] }
            func residual(_ x: Double, _ c: Double) -> Double {
                guard let period else { return x - c }
                let d = x - c
                return d - period * (d / period).rounded()
            }
            for path in paths where path.count >= 2 {
                let read = path.map(value)
                for k in 1..<path.count {
                    let a = path[k - 1].v, b = path[k].v
                    for c in values {
                        let ra = residual(read[k - 1], c), rb = residual(read[k], c)
                        guard ra.isFinite, rb.isFinite, (ra >= 0) != (rb >= 0) else { continue }
                        if let period, abs(ra - rb) > period / 2 { continue }
                        var lo = 0.0, hi = 1.0
                        for _ in 0..<64 {
                            let t = 0.5 * (lo + hi)
                            guard t > lo, t < hi else { break }
                            let r = residual(value(P2(a + t * (b - a))), c)
                            guard r.isFinite else { break }
                            if (r >= 0) == (ra >= 0) { lo = t } else { hi = t }
                        }
                        seeds.append(P2(a + 0.5 * (lo + hi) * (b - a)))
                    }
                }
            }
        case .arcLength(let spacing):
            for path in paths where path.count >= 2 {
                let placed = path.map(position)
                var lengths = [0.0]
                for k in 1..<path.count { lengths.append(lengths[k - 1] + simd_distance(placed[k], placed[k - 1])) }
                let total = lengths[lengths.count - 1]
                guard total > 0, total.isFinite else { continue }
                let n = max(1, Int((total / spacing).rounded()))
                var k = 1
                for i in 0..<n {
                    let s = (Double(i) + 0.5) * total / Double(n)
                    while k < path.count - 1, lengths[k] < s { k += 1 }
                    let chord = lengths[k] - lengths[k - 1]
                    let t = chord > 0 ? (s - lengths[k - 1]) / chord : 0
                    seeds.append(P2(path[k - 1].v + t * (path[k].v - path[k - 1].v)))
                }
            }
        }
        return seeds
    }
}

public extension Surface {
    /// The integral curves of `field` through `seeds` on `faces`, in world
    /// space: on the graph, each copy's
    /// (`graphIntegralCurves(of:seeds:tiles:region:refine:limits:)`), lifted
    /// to |f|, f's own with a refiner, and carried to the copy's tile, on
    /// every tile where `tiled`. Nothing on another face, nor where the
    /// layer could not be decoded (`LayerSource.unsupported(_:_:on:)`).
    func integralCurves(of field: DirectionField, seeds: Seeds, on faces: FaceSelector, in context: HatchContext,
                        refine: ContourRefine? = nil,
                        limits: IntegralCurve.Limits = IntegralCurve.Limits()) -> [[P3<WorldSpace>]] {
        guard case .graph(let tiled) = faces else { return [] }
        let magnitude = faceMagnitude(refine)
        return graphIntegralCurves(of: field, seeds: seeds, tiles: tiled ? context.tiles : [.identity],
                                   region: context.region, refine: refine, limits: limits).flatMap { tile, curves in
            curves.map { curve in curve.path.map { p in tile(P3<WorldSpace>(p.x, p.y, magnitude(P2(p.x, p.y)))) } }
        }
    }

    /// The integral curves of `field` on each copy of the graph, in its
    /// parameter plane, with why each ends, beside the copy's tile: one
    /// through each seed on the copy, the seeds placed along the copy's own
    /// level set of the seeds' field, clipped to its trimmed domain
    /// (`Face.contours(of:_:on:)`; `Seeds.points(along:value:period:position:)`).
    /// Copies that differ only in their tile, with no footprint to cut them,
    /// are traced once. The slope of |f| or of the height is read off the
    /// graph's own jet (`Face.heightSlope(singularPoint:)`), so its normal and
    /// its slope lines are read off the same derivatives; of arg f, u or v by
    /// central differences (`Face.slope(of:period:singularPoint:)`). Both of
    /// f's fields end their slope lines on the zeros of f where the refiner
    /// finds one (`ContourRefine.zero`). arg f is read off f, so without a
    /// refiner that has it there are no seeds placed by it, and no slope of
    /// it. Nothing where the layer could not be decoded
    /// (`LayerSource.unsupported(_:_:on:)`).
    func graphIntegralCurves(of field: DirectionField, seeds: Seeds, tiles: [Affine2], region: Region,
                             refine: ContourRefine?, limits: IntegralCurve.Limits = IntegralCurve.Limits())
        -> [(tile: Affine2, curves: [IntegralCurve.Traced])] {
        guard LayerSource.unsupported(field, seeds, on: .graph(tiled: false)) == nil,
              let face = graph(tiles: tiles, region: region, refine: refine) else { return [] }
        let magnitude = faceMagnitude(refine)
        let zero: (@Sendable (P2<ParamSpace>) -> P2<ParamSpace>?)? = refine?.zero.map { zero in
            { @Sendable p in zero(P2(p.v)).map { P2($0.v) } }
        }
        let phase: (@Sendable (P2<ParamSpace>) -> Double)? = refine?.phase.map { phase in
            { @Sendable p in phase(P2(p.v)) }
        }
        let vectors: VectorField
        switch field {
        case .slope(of: .magnitude), .slope(of: .height):
            vectors = face.heightSlope(singularPoint: zero)
        case .slope(of: .phase):
            guard let phase else { return [] }
            vectors = face.slope(of: phase, period: 2 * .pi, singularPoint: zero)
        case .slope(of: .u):
            vectors = face.slope(of: { $0.x })
        case .slope(of: .v):
            vectors = face.slope(of: { $0.y })
        case .slope(of: .isophote):
            return []
        }
        /// A field on the graph by name: its scalar field, to contour, and
        /// its value and period, to read along a curve.
        func named(_ name: ContourField) -> (scalar: ScalarField?, value: ((P2<ParamSpace>) -> Double)?, period: Double?) {
            switch name {
            case .u: (ScalarField { $0.x }, { $0.x }, nil)
            case .v: (ScalarField { $0.y }, { $0.y }, nil)
            case .height: (ScalarField { magnitude(P2($0.x, $0.y)) }, { magnitude(P2($0.x, $0.y)) }, nil)
            case .magnitude: (self.field(of: .magnitude, refine: refine), { magnitude(P2($0.x, $0.y)) }, nil)
            case .phase: (self.field(of: .phase, refine: refine), phase, 2 * .pi)
            case .isophote: (nil, nil, nil)
            }
        }
        guard let scalar = named(seeds.field).scalar else { return [] }
        let at: (value: ((P2<ParamSpace>) -> Double)?, period: Double?)
        if case .values(let of, _) = seeds.spacing {
            let n = named(of)
            at = (n.value, n.period)
        } else {
            at = (nil, nil)
        }
        let map = face.map
        let traced: [Face.Copy] = if case .inside = region { face.copies } else { Array(face.copies.prefix(1)) }
        let curves = traced.map { copy -> [IntegralCurve.Traced] in
            let level = face.contours(of: scalar, [seeds.level], on: copy).flatMap(\.paths)
            let points = seeds.points(along: level, value: at.value, period: at.period) { map($0).position }
            return Heightfield.parallelMap(points, chunk: 1) { seed in
                face.integralCurve(of: vectors, through: seed, on: copy, limits: limits)
            }.compactMap { $0 }.filter { $0.path.count >= 2 }
        }
        return face.copies.indices.map { k in (face.copies[k].tile, curves[min(k, curves.count - 1)]) }
    }
}
