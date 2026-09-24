import Foundation
import Dispatch
import simd
import KurvenCore
import KurvenMath

/// A periodically forced system in three dimensions: `dx/dt = f(t, x)` with
/// `f` periodic in `t` at angular frequency `forcing`.
///
/// The forcing phase `θ = ωt` is one angle of any invariant torus the system
/// has, known exactly; the other is the system's own. That is what makes the
/// torus recoverable from one trajectory without reconstructing a surface
/// from points: every sample already knows where on the torus it is.
public struct ForcedSystem: Sendable {
    public let name: String
    public let forcing: Double
    public let start: SIMD3<Double>
    /// How long to run before the trajectory is on its attractor.
    public let transient: Double
    public let tolerance: ODE.Tolerance
    public let field: @Sendable (Double, SIMD3<Double>) -> SIMD3<Double>

    public init(name: String, forcing: Double, start: SIMD3<Double>, transient: Double,
                tolerance: ODE.Tolerance = .init(relative: 1e-11, absolute: 1e-12),
                field: @escaping @Sendable (Double, SIMD3<Double>) -> SIMD3<Double>) {
        self.name = name; self.forcing = forcing; self.start = start
        self.transient = transient; self.tolerance = tolerance; self.field = field
    }
}

public extension ForcedSystem {
    /// `toroidal_manifolds.ipynb`, "Higher Dimensional Torus": the jerk
    /// oscillator `x''' = -α x' x'' - (ω₀² + 3βx²) x' + A sin ωt`, from
    /// *Mem-elements for Neuromorphic Circuits with Artificial Intelligence
    /// Applications*, with the parameters and starting point the notebook
    /// drew.
    static let jerkTorus = ForcedSystem(
        name: "jerk", forcing: 0.94,
        start: SIMD3(-1.02909432, -1.70628519, -0.18950172), transient: 400
    ) { t, u in
        let alpha = 0.6, beta = 0.5975, amplitude = 0.6, w0squared = 0.5896
        return SIMD3(u.y, u.z,
                     -alpha * u.y * u.z - (w0squared + 3 * beta * u.x * u.x) * u.y
                         + amplitude * sin(0.94 * t))
    }

    static let catalog: [String: ForcedSystem] = ["jerk": jerkTorus]
}

public enum DynamicsError: Error, CustomStringConvertible {
    case noGenerator(String)
    case notATorus(residual: Double, extent: Double)
    case unknownSystem(String, known: [String])

    public var description: String {
        switch self {
        case .noGenerator(let why):
            "dynamics: no second frequency in the spectrum (\(why))"
        case .unknownSystem(let name, let known):
            "dynamics: unknown system '\(name)'; the catalog has: \(known.joined(separator: ", "))"
        case .notATorus(let r, let e):
            String(format: "dynamics: the fitted torus misses the trajectory by %.3g, %.2g of its size; "
                   + "the orbit is not on a torus at this order -- chaotic, locked to a "
                   + "closed orbit, or wanting more harmonics", r, r / e)
        }
    }
}

/// An invariant torus of a forced system, found in its spectrum and fitted
/// as a Fourier series in the forcing phase and the system's own.
public struct InvariantTorus: Sendable {
    public let system: ForcedSystem
    public let fit: FourierTorus<SIMD3<Double>>
    /// The worst distance from the trajectory to the fitted torus at its own
    /// phases, and the trajectory's size, for scale.
    public let residual: Double
    public let extent: Double
    public let lines: [Frequency.Line]
    /// The fitted samples: the trajectory after its transient.
    public let t0: Double
    public let dt: Double
    public let samples: [SIMD3<Double>]

    public var forcing: Double { fit.frequencies.0 }
    public var internalFrequency: Double { fit.frequencies.1 }

    /// Integrate past the transient, find the second frequency among the
    /// spectral lines of the first component, and fit. Fails -- rather than
    /// fitting something -- when the spectrum has no second generator, or
    /// when the fit misses its own trajectory by more than `acceptance` of
    /// its size: the self-check that tells a torus from a chaotic or locked
    /// orbit.
    public static func find(_ system: ForcedSystem, duration: Double = 8000, dt: Double = 0.05,
                            harmonics: Int = 24, acceptance: Double = 1e-3) throws -> InvariantTorus {
        let skip = Int((system.transient / dt).rounded())
        let count = skip + Int((duration / dt).rounded())
        let run = try ODE.sample(system.field, from: 0, system.start, every: dt, count: count,
                                 tolerance: system.tolerance)
        let samples = Array(run.states[skip...])
        let t0 = Double(skip) * dt
        let lines = Frequency.lines(samples.map(\.x), t0: t0, dt: dt, count: 12)
        guard let g = Frequency.generator(lines, forcing: system.forcing) else {
            throw DynamicsError.noGenerator("every line is a harmonic of the forcing")
        }
        guard g.explained >= 2 else {
            throw DynamicsError.noGenerator("no candidate explains another line")
        }
        let fit = FourierTorus.fit(samples, t0: t0, dt: dt,
                                   frequencies: (system.forcing, g.frequency), harmonics: harmonics)
        let residual = fit.residual(samples, t0: t0, dt: dt, lanes: 3)
        var lo = SIMD3<Double>(repeating: .infinity), hi = -lo
        for s in samples { lo = simd_min(lo, s); hi = simd_max(hi, s) }
        let extent = (hi - lo).max()
        guard residual <= acceptance * extent else {
            throw DynamicsError.notATorus(residual: residual, extent: extent)
        }
        return InvariantTorus(system: system, fit: fit, residual: residual, extent: extent,
                              lines: lines, t0: t0, dt: dt, samples: samples)
    }
}

/// The torus of a forced system drawn by revolution: the forcing phase `θ`
/// is a real angle about the vertical, and each θ-slice -- the invariant
/// circle the stroboscopic map leaves at that phase -- is drawn in its own
/// half-plane, two state components as radius and height.
///
/// Projected straight to three state coordinates the torus can collapse into
/// a ribbon that passes through itself, and nothing drawn on that can be
/// told front from back. By revolution each slice has a half-plane to
/// itself, so the torus is embedded exactly when every slice is a simple
/// closed curve on the positive side of the axis -- which is checked, not
/// assumed.
public struct Revolution: Sendable, Equatable {
    /// The state components drawn as radius and as height.
    public var radial: Int
    public var axial: Int
    /// The state point drawn at the ring's centre line, and state units per
    /// world unit.
    public var centre: SIMD2<Double>
    public var scale: Double
    /// The ring's radius, in world units.
    public var radius: Double

    public init(radial: Int = 0, axial: Int = 1, centre: SIMD2<Double>, scale: Double,
                radius: Double) {
        self.radial = radial; self.axial = axial; self.centre = centre
        self.scale = scale; self.radius = radius
    }

    /// Centred on the trajectory's range in the two components, scaled so
    /// the larger spans two world units, on a ring of `radius`.
    public static func fitting(_ torus: InvariantTorus, radial: Int = 0, axial: Int = 1,
                               radius: Double = 2.5) -> Revolution {
        var lo = SIMD2<Double>(repeating: .infinity), hi = -lo
        for s in torus.samples {
            let p = SIMD2(s[radial], s[axial])
            lo = simd_min(lo, p); hi = simd_max(hi, p)
        }
        return Revolution(radial: radial, axial: axial, centre: 0.5 * (lo + hi),
                          scale: 2 / max((hi - lo).max(), .leastNormalMagnitude), radius: radius)
    }

    /// A state at forcing phase `θ`, placed.
    public func place(_ x: SIMD3<Double>, _ theta: Double) -> SIMD3<Double> {
        let r = radius + scale * (x[radial] - centre.x)
        return SIMD3(r * cos(theta), r * sin(theta), scale * (x[axial] - centre.y))
    }

    /// The fitted torus through this embedding, with both partials.
    public func jet(_ torus: FourierTorus<SIMD3<Double>>, _ theta: Double,
                    _ phi: Double) -> SurfaceJet {
        let k = torus.jet(theta, phi)
        let r = radius + scale * (k.value[radial] - centre.x)
        let (c, s) = (cos(theta), sin(theta))
        let drTheta = scale * k.dTheta[radial], drPhi = scale * k.dPhi[radial]
        return SurfaceJet(position: SIMD3(r * c, r * s, scale * (k.value[axial] - centre.y)),
                          du: SIMD3(drTheta * c - r * s, drTheta * s + r * c,
                                    scale * k.dTheta[axial]),
                          dv: SIMD3(drPhi * c, drPhi * s, scale * k.dPhi[axial]))
    }
}

public extension InvariantTorus {
    /// The torus as a parametric surface: `u` the forcing phase, `v` the
    /// system's own, both whole turns. The lattice is evaluated once, in
    /// parallel, and checked slice by slice: the surface is declared to
    /// bound a solid -- and so to hide whatever faces away -- only when it
    /// is embedded.
    func surface(_ e: Revolution, lattice: (u: Int, v: Int) = (1024, 512))
        -> (surface: ParametricSurface, embedded: Bool)
    {
        surface(e, lattice: lattice, values: values(lattice: lattice))
    }

    /// The fitted torus at every lattice point, in state space, before any
    /// embedding: the expensive half of `surface`, a whole Fourier series per
    /// point. Kept, it lets a new revolution be placed without evaluating the
    /// series again.
    func values(lattice: (u: Int, v: Int)) -> [SIMD3<Double>] {
        let u = ParamAxis.angle(samples: lattice.u), v = ParamAxis.angle(samples: lattice.v)
        let fit = self.fit
        var values = [SIMD3<Double>](repeating: .zero, count: lattice.u * lattice.v)
        values.withUnsafeMutableBufferPointer { out in
            nonisolated(unsafe) let base = out.baseAddress!
            DispatchQueue.concurrentPerform(iterations: lattice.v) { j in
                let phi = v.coordinate(j)
                for i in 0..<lattice.u {
                    base[j * lattice.u + i] = fit.value(u.coordinate(i), phi)
                }
            }
        }
        return values
    }

    /// The torus placed by `e` from its lattice `values` (`values(lattice:)`):
    /// the cheap half of `surface`.
    func surface(_ e: Revolution, lattice: (u: Int, v: Int), values: [SIMD3<Double>])
        -> (surface: ParametricSurface, embedded: Bool)
    {
        precondition(values.count == lattice.u * lattice.v,
                     "\(values.count) values for a \(lattice.u)x\(lattice.v) lattice")
        let u = ParamAxis.angle(samples: lattice.u), v = ParamAxis.angle(samples: lattice.v)
        let fit = self.fit
        var positions = [P3<WorldSpace>](repeating: P3(0, 0, 0), count: lattice.u * lattice.v)
        positions.withUnsafeMutableBufferPointer { out in
            nonisolated(unsafe) let base = out.baseAddress!
            DispatchQueue.concurrentPerform(iterations: lattice.v) { j in
                for i in 0..<lattice.u {
                    base[j * lattice.u + i] = P3(e.place(values[j * lattice.u + i],
                                                         u.coordinate(i)))
                }
            }
        }
        let embedded = Revolution.slicesAreSimple(positions, lattice)
        let surface = ParametricSurface(u: u, v: v, encloses: embedded, positions: positions) {
            e.jet(fit, $0.x, $0.y)
        }
        return (surface, embedded)
    }

    /// The trajectory itself as ink, from `start` for `duration`, every `dt`:
    /// each vertex the integrated state placed at its own forcing phase, and
    /// its surface coordinate `(ωt, Ωt)`.
    ///
    /// Coordinates are kept continuous along a path, so a segment never jumps
    /// a period, and small, so the preview's float32 can still resolve a
    /// pixel's worth of them: a path is ended, and the next begun from the
    /// same vertex, every eight turns of the forcing.
    func trajectory(_ e: Revolution, from start: Double? = nil, duration: Double,
                    every dt: Double = 0.01) throws -> PolylineSet<WorldSpace> {
        trajectory(e, try run(from: start, duration: duration, every: dt))
    }

    /// The integrated states a trajectory is drawn from, in state space: the
    /// expensive half of `trajectory`, and the half no embedding changes.
    func run(from start: Double? = nil, duration: Double,
             every dt: Double = 0.01) throws -> Run {
        let begin = start ?? t0
        var lead = system.start
        if begin > 0 {
            try ODE.integrate(system.field, from: 0, system.start, to: begin,
                              tolerance: system.tolerance) { lead = $0.state(at: $0.t1) }
        }
        let count = Int((duration / dt).rounded()) + 1
        let states = try ODE.sample(system.field, from: begin, lead, every: dt, count: count,
                                    tolerance: system.tolerance).states
        return Run(begin: begin, dt: dt, states: states)
    }

    /// A run of the system: its states at `begin + k dt`.
    struct Run: Sendable {
        public let begin: Double
        public let dt: Double
        public let states: [SIMD3<Double>]

        /// The first `count` states: the same run, shorter. Exact but for the
        /// last few samples, which a run that ends there reads off a step cut
        /// short at its end -- close to the integrator's tolerance, and so
        /// good for a draft.
        public func prefix(_ count: Int) -> Run {
            Run(begin: begin, dt: dt, states: Array(states.prefix(max(count, 2))))
        }
    }

    /// A run placed by `e`: the cheap half of `trajectory`.
    func trajectory(_ e: Revolution, _ run: Run) -> PolylineSet<WorldSpace> {
        let begin = run.begin, dt = run.dt, states = run.states
        let turn = 2 * Double.pi, span = 8 * turn
        func phases(_ k: Int) -> (Double, Double) {
            let t = begin + Double(k) * dt
            return (forcing * t, internalFrequency * t)
        }
        var paths: [[P3<WorldSpace>]] = [], coords: [[P2<ParamSpace>]] = []
        var path: [P3<WorldSpace>] = [], coord: [P2<ParamSpace>] = []
        var offset = (0.0, 0.0)
        for (k, x) in states.enumerated() {
            let (theta, phi) = phases(k)
            if path.isEmpty || theta - offset.0 > span {
                // A new path, its coordinates brought back near zero; after
                // the first it begins from the previous path's last vertex,
                // so the ink does not break.
                offset = (turn * (theta / turn).rounded(.down), turn * (phi / turn).rounded(.down))
                if let last = path.last {
                    paths.append(path); coords.append(coord)
                    let (t0, p0) = phases(k - 1)
                    path = [last]
                    coord = [P2(t0 - offset.0, p0 - offset.1)]
                }
            }
            path.append(P3(e.place(x, theta)))
            coord.append(P2(theta - offset.0, phi - offset.1))
        }
        if path.count >= 2 { paths.append(path); coords.append(coord) }
        return PolylineSet(paths: paths, coords: coords)
    }
}

extension Revolution {
    /// Whether every θ-slice of a lattice is a simple closed curve on the
    /// positive side of the axis: the condition for the revolved torus to be
    /// embedded. Each slice is a closed polygon in its own (radius, height)
    /// half-plane; two of its edges that are not neighbours must not meet.
    static func slicesAreSimple(_ positions: [P3<WorldSpace>], _ lattice: (u: Int, v: Int)) -> Bool {
        var ok = [Bool](repeating: true, count: lattice.u)
        ok.withUnsafeMutableBufferPointer { out in
            nonisolated(unsafe) let base = out.baseAddress!
            DispatchQueue.concurrentPerform(iterations: lattice.u) { i in
                // The radius along the slice's own half-plane, signed: a
                // point past the axis is at a negative radius, not at the
                // same distance on the far side, which `hypot` would say and
                // which would never fail the test below.
                let theta = ParamAxis.angle(samples: lattice.u).coordinate(i)
                let (c, s) = (cos(theta), sin(theta))
                var slice: [SIMD2<Double>] = []
                slice.reserveCapacity(lattice.v)
                for j in 0..<lattice.v {
                    let p = positions[j * lattice.u + i]
                    slice.append(SIMD2(p.x * c + p.y * s, p.z))
                }
                base[i] = slice.allSatisfy { $0.x > 0 } && isSimpleClosed(slice)
            }
        }
        return ok.allSatisfy { $0 }
    }

    /// A closed polygon with no two non-adjacent edges touching.
    public static func isSimpleClosed(_ p: [SIMD2<Double>]) -> Bool {
        let n = p.count
        guard n >= 3 else { return false }
        func cross(_ o: SIMD2<Double>, _ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double {
            (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
        }
        func within(_ a: SIMD2<Double>, _ b: SIMD2<Double>, _ q: SIMD2<Double>) -> Bool {
            q.x >= min(a.x, b.x) && q.x <= max(a.x, b.x) && q.y >= min(a.y, b.y) && q.y <= max(a.y, b.y)
        }
        // Segments ab and cd meet: they straddle each other, or an end of
        // one lies exactly on the other.
        func meet(_ a: SIMD2<Double>, _ b: SIMD2<Double>, _ c: SIMD2<Double>, _ d: SIMD2<Double>) -> Bool {
            guard max(a.x, b.x) >= min(c.x, d.x), max(c.x, d.x) >= min(a.x, b.x),
                  max(a.y, b.y) >= min(c.y, d.y), max(c.y, d.y) >= min(a.y, b.y) else { return false }
            let d1 = cross(c, d, a), d2 = cross(c, d, b), d3 = cross(a, b, c), d4 = cross(a, b, d)
            if ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0)) {
                return true
            }
            return (d1 == 0 && within(c, d, a)) || (d2 == 0 && within(c, d, b))
                || (d3 == 0 && within(a, b, c)) || (d4 == 0 && within(a, b, d))
        }
        for i in 0..<(n - 2) {
            let a = p[i], b = p[(i + 1) % n]
            for j in (i + 2)..<n where !(i == 0 && j == n - 1) {
                if meet(a, b, p[j], p[(j + 1) % n]) { return false }
            }
        }
        return true
    }
}
