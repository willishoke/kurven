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
/// the curve it stands for, on the face, and in space but where the graph
/// is steeper than that resolves. A step is halved for its sag in space at
/// most five times below the longest step (`Limits.liftedHalvings`), as the
/// plain contour routine halves a grid chord at most five times, so a
/// steep but smooth graph costs a bounded number of vertices a cell and is
/// followed to its end. A curve ends:
///
/// - at a singular point, where the field is undefined or every step turns
///   against the heading, so the steps refused there have shrunk below
///   `leastStep`: a zero of f, which the curve runs down into, carried to
///   the point itself when the field can name it
///   (`VectorField.singularPoint`), or a pole, which it runs up into;
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
        /// How many times a step may be halved below the longest for its
        /// chord's sag in space: past it the chord is kept.
        public var liftedHalvings: Int

        public init(tolerance: Double = 0.02, largestStep: Double = 0.5, leastStep: Double = 1e-4,
                    accuracy: Double = 1e-7, maxSteps: Int = 20_000, liftedHalvings: Int = 5) {
            self.tolerance = tolerance; self.largestStep = largestStep; self.leastStep = leastStep
            self.accuracy = accuracy; self.maxSteps = maxSteps; self.liftedHalvings = liftedHalvings
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

        // The shortest step whose chord is still held to the face in space.
        let liftedLeast = largest * Foundation.pow(0.5, Double(limits.liftedHalvings))
        var path = [seed], p = seed
        var lifted: SIMD3<Double>?
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
            if !sagged, let lift, h > liftedLeast {
                let from = lifted ?? lift(P2(p))
                lifted = from
                sagged = simd_distance(lift(P2(mid)), 0.5 * (from + lift(P2(q)))) > sag
            }
            if !(error <= accuracy) || sagged {
                h *= 0.5
                if h < least { return (ended(path), .singular) }
                continue
            }
            // Out of the trimmed domain within the step: the half that
            // leaves it bisected on membership, a step of that half's own
            // from its start, and the last point inside kept. At the half's
            // whole length the step is the half itself, outside, and at none
            // its start, inside.
            if !inside(mid) || !inside(q) {
                let (from, k) = inside(mid) ? (mid, k1Mid) : (p, k1)
                if from == mid { path.append(mid) }
                let t = bisect(0, 0.5 * h) { t in step(from, t, k, heading: k1).map(inside) ?? false }.holds
                if t > 0, let end = step(from, t, k, heading: k1) { path.append(end) }
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
            lifted = nil
            // The direction at q is the next step's first stage.
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
        let h = 1e-4 * cell, map = self.map, gradient = heightGradient
        @Sendable func difference(_ a: Double, _ b: Double) -> Double {
            period.map { shortWay(a - b, period: $0) } ?? a - b
        }
        return VectorField(value: { p in
            let du = difference(value(P2(p.x + h, p.y)), value(P2(p.x - h, p.y))) / (2 * h)
            let dv = difference(value(P2(p.x, p.y + h)), value(P2(p.x, p.y - h))) / (2 * h)
            if let gradient { return Face.raised(SIMD2(du, dv), overGraphOf: gradient(p)) }
            return Face.raised(SIMD2(du, dv), map(p))
        }, singularPoint: singularPoint)
    }

    /// The slope of the height, the map's z, read off the map's own jet:
    /// `dφ` is the z of its two derivatives. On the graph the height is |f|,
    /// and its slope lines are the lines of steepest ascent and descent; its
    /// jet is the height's derivatives alone there (`heightGradient`), the
    /// same numbers without its point.
    func heightSlope(singularPoint: (@Sendable (P2<ParamSpace>) -> P2<ParamSpace>?)? = nil) -> VectorField {
        let map = self.map
        if let gradient = heightGradient {
            return VectorField(value: { p in
                let a = gradient(p)
                return Face.raised(a, overGraphOf: a)
            }, singularPoint: singularPoint)
        }
        return VectorField(value: { p in
            let jet = map(p)
            return Face.raised(SIMD2(jet.du.z, jet.dv.z), jet)
        }, singularPoint: singularPoint)
    }

    /// A differential raised to a vector over the graph of a height whose
    /// gradient is `a`, where the parameter plane is the world's xy: there
    /// g = I + a aᵀ, and g⁻¹ = (I + a⊥ a⊥ᵀ) / (1 + |a|²) for a⊥ = (−a_y, a_x),
    /// since |a|² I = a aᵀ + a⊥ a⊥ᵀ. Read so, nothing cancels however steep
    /// the graph: the adjugate's terms in |a|³ cancel to |a|, and on a pole's
    /// spire, |a| some 10⁶, turned the direction by a thousandth. The slope
    /// of the height itself, dφ = a, is a / (1 + |a|²) to the bit.
    static func raised(_ d: SIMD2<Double>, overGraphOf a: SIMD2<Double>) -> SIMD2<Double>? {
        let perpendicular = SIMD2(-a.y, a.x)
        let w = (d + perpendicular * simd_dot(perpendicular, d)) / (1 + simd_length_squared(a))
        guard w != .zero, w.x.isFinite, w.y.isFinite else { return nil }
        return w
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
    /// plane, in order along each.
    ///
    /// At listed values of a second field, `value`, read the short way round
    /// a field of `period`, and the values the same modulo it taken once.
    /// Where a value falls strictly between a chord's two readings it is
    /// bisected along the chord to the last bit, a chord across the jump half
    /// a period from the value skipped; where a vertex reads the value
    /// exactly, the seed is the vertex, once, whether the field crosses the
    /// value there or touches it, and a closed path's last vertex, its first
    /// again, is not read twice. Nothing without `value`.
    ///
    /// By arc length, measured on the curve in space, the polyline's vertices
    /// carried there by `position`: a path of length L takes round(L /
    /// spacing) seeds at the middles of as many equal stretches, so a piece
    /// of the level set shorter than half the spacing, a fragment the trimmed
    /// domain cuts off or a small ring, takes none, and no two seeds on one
    /// piece are closer than three quarters of the spacing. A path whose ratio is
    /// not a number or past ten million takes none: a layer's spacing is held
    /// to a cell at least (`Seeds.unsupported(on:cell:)`).
    func points(along paths: [[P2<ParamSpace>]], value: ((P2<ParamSpace>) -> Double)?, period: Double? = nil,
                position: (P2<ParamSpace>) -> SIMD3<Double>) -> [P2<ParamSpace>] {
        var seeds: [P2<ParamSpace>] = []
        switch spacing {
        case .values(_, let listed):
            guard let value else { return [] }
            func residual(_ x: Double, _ c: Double) -> Double {
                period.map { shortWay(x - c, period: $0) } ?? x - c
            }
            var values: [Double] = []
            for c in listed where !values.contains(where: { abs(residual($0, c)) <= 1e-12 * max(1, abs(c)) }) {
                values.append(c)
            }
            for path in paths where path.count >= 2 {
                let read = path.map(value)
                let closed = path.first == path.last
                for k in path.indices {
                    if k > 0 {
                        let a = path[k - 1].v, b = path[k].v
                        for c in values {
                            let ra = residual(read[k - 1], c), rb = residual(read[k], c)
                            guard ra.isFinite, rb.isFinite, ra != 0, rb != 0, (ra > 0) != (rb > 0) else { continue }
                            if let period, abs(ra - rb) > period / 2 { continue }
                            let t = bisect(0, 1) { t in
                                let r = residual(value(P2(a + t * (b - a))), c)
                                return r.isFinite && (r > 0) == (ra > 0)
                            }
                            seeds.append(P2(a + 0.5 * (t.holds + t.fails) * (b - a)))
                        }
                    }
                    if closed, k == path.count - 1 { continue }
                    for c in values where residual(read[k], c) == 0 { seeds.append(path[k]) }
                }
            }
        case .arcLength(let spacing):
            for path in paths where path.count >= 2 {
                let placed = path.map(position)
                var lengths = [0.0]
                for k in 1..<path.count { lengths.append(lengths[k - 1] + simd_distance(placed[k], placed[k - 1])) }
                let total = lengths[lengths.count - 1], ratio = total / spacing
                guard total > 0, ratio.isFinite, ratio < 1e7 else { continue }
                let n = Int(ratio.rounded())
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
    /// layer could not be decoded (`LayerSource.unsupported(_:_:on:cell:)`).
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
    /// it. Nothing where the layer could not be decoded or its seeds lie
    /// closer than a cell of the grid
    /// (`LayerSource.unsupported(_:_:on:cell:)`). The seeds are traced in
    /// parallel, each thread with a refiner of its own (`ContourRefine.fork`).
    func graphIntegralCurves(of field: DirectionField, seeds: Seeds, tiles: [Affine2], region: Region,
                             refine: ContourRefine?, limits: IntegralCurve.Limits = IntegralCurve.Limits())
        -> [(tile: Affine2, curves: [IntegralCurve.Traced])] {
        guard let face = graph(tiles: tiles, region: region, refine: refine),
              LayerSource.unsupported(field, seeds, on: .graph(tiled: false), cell: face.cell) == nil else { return [] }
        let magnitude = faceMagnitude(refine)
        let phase: (@Sendable (P2<ParamSpace>) -> Double)? = refine?.phase.map { phase in
            { @Sendable p in phase(P2(p.v)) }
        }
        /// The field on a graph read with `refine`, a thread's own.
        @Sendable func vectors(on face: Face, _ refine: ContourRefine?) -> VectorField? {
            let zero: (@Sendable (P2<ParamSpace>) -> P2<ParamSpace>?)? = refine?.zero.map { zero in
                { @Sendable p in zero(P2(p.v)).map { P2($0.v) } }
            }
            switch field {
            case .slope(of: .magnitude), .slope(of: .height):
                return face.heightSlope(singularPoint: zero)
            case .slope(of: .phase):
                guard let phase = refine?.phase else { return nil }
                return face.slope(of: { phase(P2($0.v)) }, period: 2 * .pi, singularPoint: zero)
            case .slope(of: .u):
                return face.slope(of: { $0.x })
            case .slope(of: .v):
                return face.slope(of: { $0.y })
            case .slope(of: .isophote):
                return nil
            }
        }
        guard vectors(on: face, refine) != nil else { return [] }
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
        guard let scalar = named(seeds.field).scalar, let lift = face.lift else { return [] }
        let at: (value: ((P2<ParamSpace>) -> Double)?, period: Double?)
        if case .values(let of, _) = seeds.spacing {
            let n = named(of)
            at = (n.value, n.period)
        } else {
            at = (nil, nil)
        }
        let surface = self
        let traced = if case .inside = region { Array(face.copies.indices) } else { [0] }
        let curves = traced.map { k -> [IntegralCurve.Traced] in
            let level = face.contours(of: scalar, [seeds.level], on: face.copies[k]).flatMap(\.paths)
            let points = seeds.points(along: level, value: at.value, period: at.period, position: lift)
            // Each thread its own refiner, f compiled again into closures of
            // its own (`ContourRefine.fork`), and the graph and field on it.
            return Heightfield.parallelMap(points, chunk: 1, worker: { () -> (face: Face, vectors: VectorField)? in
                let own = refine?.forked()
                guard let face = surface.graph(tiles: tiles, region: region, refine: own),
                      let vectors = vectors(on: face, own) else { return nil }
                return (face, vectors)
            }) { worker, seed in
                worker.flatMap { $0.face.integralCurve(of: $0.vectors, through: seed, on: $0.face.copies[k], limits: limits) }
            }.compactMap { $0 }.filter { $0.path.count >= 2 }
        }
        return face.copies.indices.map { k in (face.copies[k].tile, curves[min(k, curves.count - 1)]) }
    }
}
