import Foundation

/// Ordinary differential equations: Dormand–Prince 5(4) with dense output.
///
/// The pair every general-purpose integrator is built on (MATLAB's `ode45`,
/// scipy's `RK45`, Hairer's `DOPRI5`): seven stages, the seventh reused as the
/// next step's first, a fifth-order solution propagated and a fourth-order
/// one for the error estimate, and Shampine's fourth-order interpolant
/// between steps. The interpolant is what makes sampling at fixed times free
/// and exact to the method's order, so the integrator chooses its own steps
/// and the caller still gets a uniform time series.
///
/// Generic over SIMD vectors, so a three-dimensional system is a
/// `SIMD3<Double>` and nothing is allocated per step. A system of dimension
/// that is not a SIMD width pads with lanes whose derivative is zero; the
/// error norm is a maximum over lanes, which a zero lane cannot affect.
public enum ODE {
    /// Step-size control.
    public struct Tolerance: Sendable, Equatable {
        public var relative: Double
        public var absolute: Double
        /// A cap on the step, or nil for none. A system with a discontinuous
        /// right-hand side wants one, so a step cannot stride over the
        /// discontinuity and accept an error estimate that never saw it.
        public var maxStep: Double?

        public init(relative: Double = 1e-10, absolute: Double = 1e-12, maxStep: Double? = nil) {
            self.relative = relative; self.absolute = absolute; self.maxStep = maxStep
        }
    }

    public struct Statistics: Sendable, Equatable {
        public var accepted = 0
        public var rejected = 0
        public var evaluations = 0
        @inlinable public init() {}
    }

    public enum Failure: Error, CustomStringConvertible, Equatable {
        /// The step fell below what the time's own precision can resolve.
        case stepUnderflow(at: Double)
        case nonFinite(at: Double)

        public var description: String {
            switch self {
            case .stepUnderflow(let t): "ode: the step size underflowed at t = \(t)"
            case .nonFinite(let t): "ode: the solution stopped being finite at t = \(t)"
            }
        }
    }

    // The Dormand–Prince tableau, exactly as published (J. R. Dormand and
    // P. J. Prince, J. Comp. Appl. Math. 6, 1980), and Hairer's coefficients
    // for the dense output (`contd5` in DOPRI5).
    @usableFromInline static let c2 = 1.0 / 5, c3 = 3.0 / 10, c4 = 4.0 / 5, c5 = 8.0 / 9
    @usableFromInline static let a21 = 1.0 / 5
    @usableFromInline static let a31 = 3.0 / 40, a32 = 9.0 / 40
    @usableFromInline static let a41 = 44.0 / 45, a42 = -56.0 / 15, a43 = 32.0 / 9
    @usableFromInline static let a51 = 19372.0 / 6561, a52 = -25360.0 / 2187, a53 = 64448.0 / 6561,
               a54 = -212.0 / 729
    @usableFromInline static let a61 = 9017.0 / 3168, a62 = -355.0 / 33, a63 = 46732.0 / 5247,
               a64 = 49.0 / 176, a65 = -5103.0 / 18656
    @usableFromInline static let a71 = 35.0 / 384, a73 = 500.0 / 1113, a74 = 125.0 / 192,
               a75 = -2187.0 / 6784, a76 = 11.0 / 84
    // Fifth order minus fourth: the error estimate's weights.
    @usableFromInline static let e1 = 71.0 / 57600, e3 = -71.0 / 16695, e4 = 71.0 / 1920,
               e5 = -17253.0 / 339200, e6 = 22.0 / 525, e7 = -1.0 / 40
    @usableFromInline static let d1 = -12715105075.0 / 11282082432, d3 = 87487479700.0 / 32700410799,
               d4 = -10690763975.0 / 1880347072, d5 = 701980252875.0 / 199316789632,
               d6 = -1453857185.0 / 822651844, d7 = 69997945.0 / 29380423

    /// One accepted step, as the interpolant needs it: the state at `t` is
    /// `r1 + θ(r2 + (1-θ)(r3 + θ(r4 + (1-θ) r5)))` with `θ = (t - t0) / h`.
    public struct Step<V: SIMD> where V.Scalar == Double {
        public let t0: Double
        public let h: Double
        @usableFromInline let r1, r2, r3, r4, r5: V

        @usableFromInline
        init(t0: Double, h: Double, r1: V, r2: V, r3: V, r4: V, r5: V) {
            self.t0 = t0; self.h = h; self.r1 = r1; self.r2 = r2; self.r3 = r3
            self.r4 = r4; self.r5 = r5
        }

        @inlinable
        public func state(at t: Double) -> V {
            let s = (t - t0) / h, s1 = 1 - s
            return r1 + s * (r2 + s1 * (r3 + s * (r4 + s1 * r5)))
        }
        @inlinable public var t1: Double { t0 + h }
    }

    /// One Dormand–Prince step of length `h` from `(t, y)`, given `k1 =
    /// f(t, y)`: the step with its interpolant, the last stage (the next
    /// step's first), and the local error estimate, fifth order minus fourth.
    @inlinable
    static func attempt<V: SIMD>(_ f: (Double, V) -> V, _ t: Double, _ y: V, _ k1: V,
                                 _ h: Double) -> (step: Step<V>, k7: V, error: V)
        where V.Scalar == Double
    {
        let k2 = f(t + c2 * h, y + h * (a21 * k1))
        let k3 = f(t + c3 * h, y + h * (a31 * k1 + a32 * k2))
        let k4 = f(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3))
        let k5 = f(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
        let k6 = f(t + h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
        let yNew = y + h * (a71 * k1 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6)
        let k7 = f(t + h, yNew)
        let dy = yNew - y
        let r3 = h * k1 - dy
        let step = Step(t0: t, h: h, r1: y, r2: dy, r3: r3, r4: dy - h * k7 - r3,
                        r5: h * (d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6 + d7 * k7))
        return (step, k7, h * (e1 * k1 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6 + e7 * k7))
    }

    /// `steps` equal steps with no error control: the method's own order,
    /// laid bare, which is what a test of the tableau measures.
    @inlinable
    public static func fixed<V: SIMD>(_ f: (Double, V) -> V, from t0: Double, _ y0: V,
                                      to t1: Double, steps: Int)
        -> (state: V, lastError: V) where V.Scalar == Double
    {
        let h = (t1 - t0) / Double(steps)
        var y = y0, k1 = f(t0, y0), error = V()
        for i in 0..<steps {
            let trial = attempt(f, t0 + Double(i) * h, y, k1, h)
            y = trial.step.r1 + trial.step.r2; k1 = trial.k7; error = trial.error
        }
        return (y, error)
    }

    /// Integrate `dy/dt = f(t, y)` from `(t0, y0)` to `t1`, handing every
    /// accepted step to `body` in order.
    ///
    /// Step control is Hairer's: the error is the largest over components of
    /// `|err| / (atol + rtol * max(|y0|, |y1|))`, a step is accepted when that
    /// is at most one, and the next step is scaled by `0.9 err^(-1/5)`, held
    /// between a fifth and ten times, and not allowed to grow straight after
    /// a rejection. The first step is Hairer's `hinit` estimate.
    @inlinable @discardableResult
    public static func integrate<V: SIMD>(
        _ f: (Double, V) -> V, from t0: Double, _ y0: V, to t1: Double,
        tolerance tol: Tolerance = Tolerance(),
        _ body: (Step<V>) throws -> Void
    ) throws -> Statistics where V.Scalar == Double {
        var stats = Statistics()
        guard t1 > t0 else { return stats }
        func norm(_ v: V, _ scale: V) -> Double { (absolute(v) / scale).max() }
        func scale(_ a: V, _ b: V) -> V {
            tol.absolute + tol.relative * pointwiseMax(absolute(a), absolute(b))
        }
        let hmax = min(tol.maxStep ?? .infinity, t1 - t0)

        var t = t0, y = y0
        var k1 = f(t, y); stats.evaluations += 1

        // hinit: a step whose first-order Euler move is about 1% of the
        // state, then revised from the second derivative it implies.
        var h: Double = {
            let sc = scale(y, y)
            let d0 = norm(y, sc), d1 = norm(k1, sc)
            var h0 = (d0 < 1e-5 || d1 < 1e-5) ? 1e-6 : 0.01 * d0 / d1
            h0 = min(h0, hmax)
            let k = f(t + h0, y + h0 * k1); stats.evaluations += 1
            let d2 = norm(k - k1, sc) / h0
            let h1 = max(d1, d2) <= 1e-15 ? max(1e-6, h0 * 1e-3)
                : pow(0.01 / max(d1, d2), 1.0 / 5)
            return min(100 * h0, h1, hmax)
        }()

        var rejectedLast = false
        while t < t1 {
            if t + h >= t1 { h = t1 - t }
            guard h > 16 * t.ulp.magnitude else { throw Failure.stepUnderflow(at: t) }

            let trial = attempt(f, t, y, k1, h)
            let yNew = trial.step.r1 + trial.step.r2, k7 = trial.k7
            stats.evaluations += 6

            let err = norm(trial.error, scale(y, yNew))
            guard err.isFinite else {
                // An overflow in a trial stage is a step too long, not a
                // failure; a non-finite state after a tiny step is.
                if h < 1e-12 * max(1, abs(t)) { throw Failure.nonFinite(at: t) }
                h *= 0.1; rejectedLast = true; stats.rejected += 1
                continue
            }
            var factor = 0.9 * pow(max(err, 1e-10), -1.0 / 5)
            factor = min(10, max(0.2, factor))
            if err <= 1 {
                try body(trial.step)
                stats.accepted += 1
                t += h; y = yNew; k1 = k7
                if rejectedLast { factor = min(factor, 1) }
                rejectedLast = false
            } else {
                stats.rejected += 1
                factor = min(factor, 1)
                rejectedLast = true
            }
            h = min(h * factor, hmax)
        }
        return stats
    }

    /// The solution at `t0 + k dt` for `k = 0..<count`, read off the dense
    /// output: the steps are the integrator's, the samples are the caller's.
    @inlinable
    public static func sample<V: SIMD>(
        _ f: (Double, V) -> V, from t0: Double, _ y0: V, every dt: Double, count: Int,
        tolerance: Tolerance = Tolerance()
    ) throws -> (states: [V], statistics: Statistics) where V.Scalar == Double {
        precondition(dt > 0 && count >= 1)
        var out: [V] = [y0]
        out.reserveCapacity(count)
        let end = t0 + dt * Double(count - 1)
        var last: Step<V>?
        let stats = try integrate(f, from: t0, y0, to: end, tolerance: tolerance) { step in
            // Every sample time this step covers.
            while out.count < count {
                let t = t0 + dt * Double(out.count)
                guard t <= step.t1 else { break }
                out.append(step.state(at: t))
            }
            last = step
        }
        // The final step ends on `end` to rounding; a last sample that lands a
        // hair past it is read from that step all the same.
        while out.count < count, let step = last {
            out.append(step.state(at: t0 + dt * Double(out.count)))
        }
        return (out, stats)
    }
}

@inlinable @inline(__always)
func pointwiseMax<V: SIMD>(_ a: V, _ b: V) -> V where V.Scalar == Double {
    a.replacing(with: b, where: a .< b)
}

@inlinable @inline(__always)
func absolute<V: SIMD>(_ v: V) -> V where V.Scalar == Double {
    v.replacing(with: -v, where: v .< 0)
}
