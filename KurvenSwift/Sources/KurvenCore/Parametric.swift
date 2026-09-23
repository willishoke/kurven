import Foundation
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

    /// The coordinate of lattice index `i`, *unwrapped*: on a periodic axis
    /// `i == samples` is one period past `i == 0`, which is what lets a cell
    /// that closes the seam interpolate across it rather than back through
    /// the whole range.
    public func coordinate(_ i: Int) -> Double { range.lo + Double(i) * spacing }

    /// The stored sample an unwrapped lattice index refers to.
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

/// A surface given as a map from a parameter rectangle into space.
///
/// The map is exact and carries its derivatives, because the questions asked
/// of it -- which way does it face here, and exactly where along this segment
/// does it turn edge-on -- are asked at the ink's own coordinates, and a normal
/// read off the nearest lattice point flips sign between neighbours precisely
/// where the answer matters, at a fold. The lattice is what gets rasterized;
/// the map is what gets asked.
///
/// A heightfield is the case `(u, v) -> (u, v, h(u, v))` with neither axis
/// periodic; a torus has both periodic.
public struct ParametricSurface: Sendable {
    public let u: ParamAxis
    public let v: ParamAxis
    public let map: @Sendable (P2<ParamSpace>) -> SurfaceJet
    /// The lattice, `u` fastest: `positions[j * u.samples + i]`.
    public let positions: [P3<WorldSpace>]
    /// `+1` when `du × dv` points out of the solid the surface bounds, `-1`
    /// when it points in; `nil` when the surface bounds no solid.
    ///
    /// Only a closed, embedded, orientable surface bounds a solid, and only
    /// for such a surface is "faces away from the viewer" the same as
    /// "hidden": a sight line that reaches the back of it must first have
    /// entered the solid through the front. That is a fact the caller knows
    /// about its geometry -- it cannot be read off samples -- so it is
    /// declared, and the orientation is then computed rather than guessed.
    public let outward: Double?

    /// - Parameter encloses: the surface is closed, embedded and orientable.
    ///   Declaring it for a surface that passes through itself, or has a
    ///   boundary, culls ink that is in plain view.
    public init(u: ParamAxis, v: ParamAxis, encloses: Bool,
                map: @escaping @Sendable (P2<ParamSpace>) -> SurfaceJet) {
        self.u = u; self.v = v; self.map = map
        var positions: [P3<WorldSpace>] = []
        positions.reserveCapacity(u.samples * v.samples)
        // Six times the enclosed volume, by the divergence theorem over the
        // lattice: the sum of p · (du × dv) du dv. Only its sign is used, and
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
        var paths: [[P3<WorldSpace>]] = [], coords: [[P2<ParamSpace>]] = []
        func place(_ axis: ParamAxis, _ k: Int, of count: Int) -> Double {
            axis.range.lo + axis.range.length * Double(k)
                / Double(axis.periodic ? count : max(count - 1, 1))
        }
        func run(_ axis: ParamAxis) -> [Double] {
            (0...resolution).map { axis.range.lo + axis.range.length * Double($0) / Double(resolution) }
        }
        for k in 0..<lines.u {
            let fixed = place(u, k, of: lines.u)
            let c = run(v).map { P2<ParamSpace>(fixed, $0) }
            coords.append(c); paths.append(c.map { P3(map($0).position) })
        }
        for k in 0..<lines.v {
            let fixed = place(v, k, of: lines.v)
            let c = run(u).map { P2<ParamSpace>($0, fixed) }
            coords.append(c); paths.append(c.map { P3(map($0).position) })
        }
        return PolylineSet(paths: paths, coords: coords)
    }

    /// The fold lines for sight lines along `sight`: where the surface turns
    /// edge-on, `n · sight = 0`, which is where its outline and every inner
    /// silhouette lie.
    ///
    /// Traced by marching squares on that product over the parameter
    /// rectangle -- sampled one step past a periodic axis's end, so a fold
    /// crosses the seam instead of stopping at it -- and every vertex then
    /// moved onto the exact fold by bisection along the product's gradient.
    /// The grid decides which folds exist; the map decides where they run.
    /// They depend on the camera, so they are derived for one, never stored.
    func foldLines(sight: SIMD3<Double>, grid: (u: Int, v: Int)? = nil) -> PolylineSet<WorldSpace> {
        let nu = grid?.u ?? u.cells, nv = grid?.v ?? v.cells
        func edgeOn(_ c: P2<ParamSpace>) -> Double {
            let n = map(c).normal
            let len = simd_length(n)
            return len > 0 ? simd_dot(n, sight) / len : 0
        }
        let du = u.range.length / Double(nu), dv = v.range.length / Double(nv)
        var values: [Float] = []
        values.reserveCapacity((nu + 1) * (nv + 1))
        for j in 0...nv {
            for i in 0...nu {
                values.append(Float(edgeOn(P2(u.range.lo + Double(i) * du,
                                              v.range.lo + Double(j) * dv))))
            }
        }
        let field = Grid2D(width: nu + 1, height: nv + 1,
                           domain: Domain(real: u.range, imag: v.range), values: values)

        // Onto the fold: bracket a sign change along the gradient within a
        // cell either way, then bisect it to the last bit.
        let h = 0.25 * min(du, dv)
        func refine(_ c: P2<ParamSpace>) -> P2<ParamSpace> {
            let g = SIMD2(edgeOn(P2(c.x + h, c.y)) - edgeOn(P2(c.x - h, c.y)),
                          edgeOn(P2(c.x, c.y + h)) - edgeOn(P2(c.x, c.y - h)))
            let len = simd_length(g)
            guard len > 0 else { return c }
            let d = g / len * max(du, dv)
            var a = c.v - d, b = c.v + d
            var fa = edgeOn(P2(a))
            guard (fa > 0) != (edgeOn(P2(b)) > 0) else { return c }
            for _ in 0..<60 {
                let m = 0.5 * (a + b)
                let fm = edgeOn(P2(m))
                if (fm > 0) == (fa > 0) { a = m; fa = fm } else { b = m }
            }
            return P2(0.5 * (a + b))
        }

        var paths: [[P3<WorldSpace>]] = [], coords: [[P2<ParamSpace>]] = []
        for line in Contour.lines(of: field, level: 0) {
            let c = line.map { refine(P2<ParamSpace>($0.x, $0.y)) }
            coords.append(c)
            paths.append(c.map { P3(map($0).position) })
        }
        return PolylineSet(paths: paths, coords: coords)
    }

    /// The torus of revolution about the z axis: `u` around the axis, `v`
    /// around the tube.
    ///
    /// It bounds a solid exactly when the tube does not reach the axis,
    /// `major > minor`; the horn and spindle tori pass through themselves.
    static func torus(major R: Double, minor r: Double,
                      samples: (u: Int, v: Int)) -> ParametricSurface {
        ParametricSurface(u: .angle(samples: samples.u), v: .angle(samples: samples.v),
                          encloses: R > r) { c in
            let (cu, su, cv, sv) = (cos(c.x), sin(c.x), cos(c.y), sin(c.y))
            let ring = R + r * cv
            return SurfaceJet(position: SIMD3(ring * cu, ring * su, r * sv),
                              du: SIMD3(-ring * su, ring * cu, 0),
                              dv: SIMD3(-r * sv * cu, -r * sv * su, r * cv))
        }
    }
}

public extension Transform where A == WorldSpace, B == ViewSpace {
    /// The direction a sight line travels through the world, unit length.
    ///
    /// Under an affine camera the eye looks along the one direction that
    /// changes neither view x nor view y -- the kernel of the first two rows
    /// -- signed so that travelling it takes view z down, away from the eye.
    /// That is *not* the gradient of view z unless the camera is orthonormal,
    /// and the plate cameras are not: they shear. Asking the gradient instead
    /// tilts every fold by the shear.
    var sightLine: SIMD3<Double> {
        let c = m.columns
        let row0 = SIMD3(c.0.x, c.1.x, c.2.x)
        let row1 = SIMD3(c.0.y, c.1.y, c.2.y)
        let row2 = SIMD3(c.0.z, c.1.z, c.2.z)
        var d = simd_normalize(simd_cross(row0, row1))
        if simd_dot(row2, d) > 0 { d = -d }
        return d
    }
}
