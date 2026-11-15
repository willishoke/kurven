import Foundation
import simd

/// Lines of curvature (do Carmo §3-2): the integral curves of a surface's
/// principal directions, the directions in which its normal curvature is
/// greatest and least. The two families cross at right angles in space. On a
/// surface of revolution they are the meridians and the parallels, so on the
/// round torus they are its coordinate curves.
///
/// The principal directions are a field of line elements, with no sign, so
/// they are a vector field whose integral curves are traced unsigned
/// (`VectorField.signed`, D9). They are undefined at an umbilic, where the
/// two principal curvatures are equal and every direction is principal, so
/// a line of curvature that runs into one ends there, at a singular point of
/// the field (`IntegralCurve.End.singular`).
public extension Face {
    /// How near the two principal curvatures may come before a point is
    /// taken for an umbilic, where the principal directions are undefined:
    /// k₁ − k₂ under this fraction of |k₁| + |k₂|, well above the rounding
    /// of second derivatives in closed form, under which the directions
    /// would be noise. A planar point, k₁ = k₂ = 0, is an umbilic too.
    ///
    /// A line of curvature heading into an umbilic does not reach it. Round
    /// a generic umbilic the principal directions turn half a turn, one way
    /// or the other, for each turn round it (index ±½), so a step that would
    /// pass it samples a direction a right angle from the heading, and is
    /// refused at 45° and halved until it falls under the least step: the
    /// line ends at a singular point (`IntegralCurve.End.singular`), short of
    /// the umbilic by about the least step.
    static let umbilicTolerance = 1e-6

    /// The principal curvatures at a point, the greater first, measured
    /// against the normal into the solid the copy bounds, so a convex
    /// surface's are positive; where it bounds none, against −(du × dv), the
    /// parametrization's own. With them, each one's principal direction in
    /// the parameter plane, unit length in space, and whether the point is
    /// an umbilic (`umbilicTolerance`). Nil where the face's map has no
    /// second derivatives (`secondOrderJet`) or the point is singular.
    ///
    /// Read from the first and second fundamental forms, the second off the
    /// map's second derivatives as they are given, in closed form: in an
    /// orthonormal frame of the tangent plane, e₁ along du, the second
    /// fundamental form is a symmetric matrix [[a, b], [b, c]], whose
    /// eigenvalues (a + c)/2 ± √(((a − c)/2)² + b²) are the curvatures and
    /// whose eigenvectors, at ½ atan2(b, (a − c)/2) to e₁ and a right angle
    /// from that, the directions. Their difference is read without
    /// cancellation, so an umbilic is told by it to the rounding of the
    /// derivatives.
    func principalCurvatures(at p: P2<ParamSpace>, on copy: Copy)
        -> (greater: Double, lesser: Double, directions: (greater: SIMD2<Double>, lesser: SIMD2<Double>),
            umbilic: Bool)? {
        guard let second = secondOrderJet else { return nil }
        let j = second(p)
        return Face.principal(j.jet, j.second, inward: -(copy.outward ?? 1))
    }

    /// The principal directions of `curvature` on `copy`, as an unsigned
    /// vector field (`VectorField.signed`): its integral curves are that
    /// family's lines of curvature. Undefined at an umbilic. Nil where the
    /// face's map has no second derivatives.
    func principalDirections(of curvature: PrincipalCurvature, on copy: Copy) -> VectorField? {
        guard let second = secondOrderJet else { return nil }
        let inward = -(copy.outward ?? 1)
        // One evaluation of the map's jet of order two a sample.
        return VectorField(value: { p in
            let j = second(p)
            guard let k = Face.principal(j.jet, j.second, inward: inward), !k.umbilic else { return nil }
            return curvature == .greater ? k.directions.greater : k.directions.lesser
        }, signed: false)
    }

    /// The principal curvatures and directions of a jet and its second
    /// derivatives, against the unit normal `inward · (du × dv)/|du × dv|`.
    static func principal(_ jet: SurfaceJet, _ second: SecondDerivatives, inward: Double)
        -> (greater: Double, lesser: Double, directions: (greater: SIMD2<Double>, lesser: SIMD2<Double>),
            umbilic: Bool)? {
        let n = jet.normal, length = simd_length(n), du = simd_length(jet.du)
        guard length > 0, length.isFinite, du > 0 else { return nil }
        let normal = inward * n / length
        // An orthonormal frame of the tangent plane, e₁ along du and e₂ the
        // part of dv across it, as vectors of the parameter plane.
        let e1 = jet.du / du
        let across = jet.dv - simd_dot(jet.dv, e1) * e1, w = simd_length(across)
        guard w > 0 else { return nil }
        let p1 = SIMD2(1 / du, 0.0), p2 = SIMD2(-simd_dot(jet.dv, e1) / du, 1) / w
        // The second fundamental form, L du² + 2M du dv + N dv², in it.
        let l = simd_dot(second.uu, normal), m = simd_dot(second.uv, normal), nn = simd_dot(second.vv, normal)
        func form(_ x: SIMD2<Double>, _ y: SIMD2<Double>) -> Double {
            l * x.x * y.x + m * (x.x * y.y + x.y * y.x) + nn * x.y * y.y
        }
        let a = form(p1, p1), b = form(p1, p2), c = form(p2, p2)
        let mean = 0.5 * (a + c), half = 0.5 * (a - c), radius = (half * half + b * b).squareRoot()
        guard mean.isFinite, radius.isFinite else { return nil }
        let angle = 0.5 * atan2(b, half)
        let (ca, sa) = (cos(angle), sin(angle))
        let greater = mean + radius, lesser = mean - radius
        return (greater, lesser, (ca * p1 + sa * p2, -sa * p1 + ca * p2),
                2 * radius <= umbilicTolerance * (abs(greater) + abs(lesser)))
    }
}

public extension ParametricSurface {
    /// The most seeds one layer of integral curves takes: each curve runs up
    /// to `maxSteps` steps each way, and a layer holds
    /// `SurfaceRequest.maxVertices` vertices. A bound on the seeds, not a
    /// count of vertices, so approximate: a curve also takes the point its
    /// end is bisected to and the middle of its last step, and a vertex again
    /// at each split every eight turns, which at the bound is a few percent
    /// more.
    static func maxIntegralCurves(_ limits: IntegralCurve.Limits = IntegralCurve.Limits()) -> Int {
        SurfaceRequest.maxVertices / (2 * limits.maxSteps + 1)
    }

    /// A field on the surface by name, and its period where it is
    /// circle-valued: a coordinate, or the height, the map's z. Nil for f's
    /// fields and an isophote, which the map does not carry. What a contour
    /// of a coordinate or the height is taken of, and what seeds are placed
    /// by.
    func field(_ name: ContourField) -> (value: @Sendable (P2<ParamSpace>) -> Double, period: Double?)? {
        let map = self.map
        switch name {
        case .u: return ({ $0.x }, u.period)
        case .v: return ({ $0.y }, v.period)
        case .height: return ({ map($0).position.z }, nil)
        case .magnitude, .phase, .isophote: return nil
        }
    }

    /// Where `seeds` lie on the surface's one face: along its level set of a
    /// coordinate or the height (`Face.contours(of:_:on:)`), at listed values
    /// of another or by arc length in space
    /// (`Seeds.points(along:value:period:periods:position:)`). A coordinate
    /// along a periodic axis is read the short way round, so values the same
    /// modulo its period are one, and a level set that closes across a seam
    /// is read once round. Nothing where the seeds are placed by a field the
    /// surface does not carry.
    func seedPoints(_ seeds: Seeds) -> [P2<ParamSpace>] { placed(seeds, most: nil).points }

    /// The seeds' points, or none and how many there would be where they are
    /// placed by arc length and would be more than `most`: counted off the
    /// level set's length before any is placed.
    private func placed(_ seeds: Seeds, most: Int?) -> (points: [P2<ParamSpace>], over: Int?) {
        let face = self.face, copy = face.copies[0], map = self.map
        guard let on = field(seeds.field) else { return ([], nil) }
        let level = face.contours(of: ScalarField(period: on.period, value: on.value), [seeds.level], on: copy)
            .flatMap(\.paths)
        var at: (value: @Sendable (P2<ParamSpace>) -> Double, period: Double?)?
        switch seeds.spacing {
        case .values(let of, _):
            guard let named = field(of) else { return ([], nil) }
            at = named
        case .arcLength(let spacing):
            // As `Seeds.points` counts them, round(L / spacing) a piece,
            // and past ten million, where it places none, too many as well.
            if let most {
                var count = 0.0
                for path in level where path.count >= 2 {
                    var length = 0.0
                    for (a, b) in zip(path, path.dropFirst()) { length += simd_distance(map(a).position, map(b).position) }
                    let ratio = length / spacing
                    count += ratio.isFinite ? ratio.rounded() : .infinity
                }
                if count > Double(most) { return ([], count.isFinite ? Int(count) : Int.max) }
            }
        }
        return (seeds.points(along: level, value: at?.value, period: at?.period,
                             periods: SIMD2(u.period ?? 0, v.period ?? 0), position: { map($0).position }), nil)
    }

    /// The level set `seeds` lie on as a section their curves return to
    /// (`IntegralCurve.Section`), or nil where the surface does not carry
    /// its field.
    func section(_ seeds: Seeds) -> IntegralCurve.Section? {
        field(seeds.field).map { IntegralCurve.Section(value: $0.value, level: seeds.level, period: $0.period) }
    }

    /// The integral curves of `field` through `points` on the surface's one
    /// face, in its parameter plane, with why each ends
    /// (`Face.integralCurve(of:through:on:limits:returningTo:)`), to where
    /// it first returns to `section` where one is given, in parallel.
    /// Nothing where the face has no such field: the principal directions
    /// where the map has no second derivatives, and a slope, which lies on
    /// the graph.
    func tracedIntegralCurves(of field: DirectionField, through points: [P2<ParamSpace>],
                              returningTo section: IntegralCurve.Section? = nil,
                              limits: IntegralCurve.Limits = IntegralCurve.Limits()) -> [IntegralCurve.Traced] {
        let face = self.face, copy = face.copies[0]
        guard case .principalDirection(let curvature) = field,
              let vectors = face.principalDirections(of: curvature, on: copy) else { return [] }
        return Heightfield.parallelMap(points, chunk: 1) { seed in
            face.integralCurve(of: vectors, through: seed, on: copy, limits: limits, returningTo: section)
        }.compactMap { $0 }.filter { $0.path.count >= 2 }
    }

    /// The integral curves of `field` through `seeds`, in space, each vertex
    /// on the map with its surface coordinate beside it, so they are tested
    /// as the coordinate curves are. A curve runs on across the seams, its
    /// coordinates unwrapped, and closes on its seed's image a whole number
    /// of periods along; one that does not close ends where it first comes
    /// back across the level set its seeds lie on, once round
    /// (`section(_:)`, `IntegralCurve.Section`). As a linear curve's, its
    /// coordinates are kept within eight turns of the rectangle, for the
    /// preview's float32: a path is ended and the next begun from the same
    /// vertex, brought back by whole periods, wherever it would run
    /// further.
    ///
    /// Refused where the layer could not be decoded on the surface
    /// (`LayerSource.unsupported(_:_:on:cell:)`), and where its seeds are
    /// more than `maxIntegralCurves`, before any is traced; by arc length,
    /// before any is placed.
    func integralCurves(of field: DirectionField, seeds: Seeds,
                        limits: IntegralCurve.Limits = IntegralCurve.Limits()) throws -> PolylineSet<WorldSpace> {
        if let why = LayerSource.unsupported(field, seeds, on: .surface) { throw ManifestError.unsupported(why) }
        let most = Self.maxIntegralCurves(limits)
        let (points, over) = placed(seeds, most: most)
        if let count = over ?? (points.count > most ? points.count : nil) {
            throw ManifestError.unsupported("\(count) seeds of integral curves, more than the \(most) a layer holds at "
                                            + "\(limits.maxSteps) steps each way")
        }
        let traced = tracedIntegralCurves(of: field, through: points, returningTo: section(seeds), limits: limits)
        let lo = SIMD2(u.range.lo, v.range.lo)
        let periods = SIMD2(u.period ?? 0, v.period ?? 0)
        /// The whole periods to take from `c` to bring it into the rectangle.
        func offset(_ c: SIMD2<Double>) -> SIMD2<Double> {
            var o = SIMD2<Double>(0, 0)
            for axis in 0..<2 where periods[axis] > 0 {
                o[axis] = periods[axis] * ((c[axis] - lo[axis]) / periods[axis]).rounded(.down)
            }
            return o
        }
        var coords: [[P2<ParamSpace>]] = []
        for curve in traced {
            var path: [P2<ParamSpace>] = []
            var o = offset(curve.path[0].v)
            for c in curve.path {
                let local = c.v - o - lo
                let far = (0..<2).contains { periods[$0] > 0 && abs(local[$0]) > 8 * periods[$0] }
                if far, let last = path.last {
                    coords.append(path)
                    let next = offset(c.v)
                    path = [P2(last.v + o - next)]
                    o = next
                }
                path.append(P2(c.v - o))
            }
            coords.append(path)
        }
        let lines = coords, map = self.map
        let paths = Heightfield.parallelMap(lines, chunk: 1) { line in line.map { P3<WorldSpace>(map($0).position) } }
        return PolylineSet(paths: paths, coords: lines)
    }
}

public extension LayerSpec {
    /// One family of lines of curvature on a surface over `rectangle`, as
    /// the window adds it and `surface --lines-of-curvature` draws it, named
    /// `lines_of_curvature_greater` or `_lesser`: the greater family seeded
    /// on the coordinate curve v = its range's start, at `count` values of u
    /// evenly round, the lesser on u = its start at `count` values of v. On
    /// a torus of revolution the greater family is the meridians and the
    /// lesser the parallels, so each seed draws one of them, crossing the
    /// curve it is seeded on.
    static func linesOfCurvature(_ curvature: PrincipalCurvature, over rectangle: (u: ParamAxis, v: ParamAxis),
                                 count: Int, width: Double) -> LayerSpec {
        let (along, across) = curvature == .greater ? (rectangle.u, rectangle.v) : (rectangle.v, rectangle.u)
        let values = (0..<max(count, 1)).map { along.range.lo + along.range.length * Double($0) / Double(max(count, 1)) }
        let seeds = curvature == .greater ? Seeds(on: .v, at: across.range.lo, .values(of: .u, values))
            : Seeds(on: .u, at: across.range.lo, .values(of: .v, values))
        return LayerSpec(name: "lines_of_curvature_\(curvature.rawValue)", role: .scaffold,
                         source: .integralCurves(field: .principalDirection(of: curvature), seeds: seeds, faces: .surface),
                         width: width, heightPolicy: .surface)
    }
}
