import Foundation
import simd
import KurvenMath

// MARK: - ordinary differential equations
//
// Every check here is against something exact: a closed-form solution, a
// conserved quantity, or the order the tableau is supposed to have. The
// order test is the one a mistyped coefficient cannot pass -- the method
// still converges with a wrong digit in it, just more slowly.

func odeTests() {
    Check.suite("ode: Dormand–Prince is fifth order, and its error estimate too") {
        // x'' = -36x, forced at 3t, over four time units: fast enough that the
        // error stays far above rounding while the step shrinks into the
        // asymptotic range, and time-dependent so no stage is right by
        // symmetry.
        let w = 6.0, p = 3.0
        let f = { (t: Double, y: SIMD2<Double>) -> SIMD2<Double> in
            SIMD2(y.y, -w * w * y.x + sin(p * t))
        }
        // From rest: x = (sin pt − (p/w) sin wt) / (w² − p²).
        func exact(_ t: Double) -> Double { (sin(p * t) - p / w * sin(w * t)) / (w * w - p * p) }
        var errors: [Double] = [], estimates: [Double] = []
        for n in [100, 200, 400, 800, 1600] {
            let r = ODE.fixed(f, from: 0, SIMD2(0, 0), to: 4, steps: n)
            errors.append(abs(r.state.x - exact(4)))
            // The last step's estimate is a local error, O(h⁵) for a 5(4) pair.
            estimates.append(simd_length(r.lastError))
        }
        // Dormand and Prince made the fifth-order error constant small on
        // purpose, so at coarse steps the sixth-order term shows through and
        // the observed order comes down to five from above. A mistyped
        // coefficient pulls it below five and keeps it there.
        let orders = zip(errors, errors.dropFirst()).map { log2($0 / $1) }
        let local = zip(estimates, estimates.dropFirst()).map { log2($0 / $1) }
        let falling = zip(orders, orders.dropFirst()).allSatisfy { $0 >= $1 - 0.05 }
        Check.expect(errors.last! > 1e-14 && orders.allSatisfy { $0 > 4.9 }
                     && falling && abs(orders.last! - 5) < 0.2,
                     "halving the step divides the global error by 2⁵, in the limit",
                     orders.map { String(format: "%.2f", $0) }.joined(separator: ", ")
                     + String(format: "; finest error %.1e", errors.last!))
        Check.expect(local.allSatisfy { abs($0 - 5) < 0.2 },
                     "and the error estimate is O(h⁵) per step",
                     local.map { String(format: "%.2f", $0) }.joined(separator: ", "))
    }

    Check.suite("ode: closed-form solutions, adaptively") {
        // The harmonic oscillator, a hundred time units: sixteen turns.
        let spring = { (_: Double, y: SIMD2<Double>) in SIMD2(y.y, -y.x) }
        var worst = 0.0, samples = 0
        let stats = try ODE.integrate(spring, from: 0, SIMD2(1, 0), to: 100) { step in
            // The interpolant, at three points inside every step.
            for s in [0.25, 0.5, 0.75] {
                let t = step.t0 + s * step.h
                worst = max(worst, simd_length(step.state(at: t) - SIMD2(cos(t), -sin(t))))
                samples += 1
            }
        }
        Check.expect(worst < 1e-8, "the oscillator, read anywhere between steps",
                     "max error \(worst) over \(samples) interior points, "
                     + "\(stats.accepted) steps, \(stats.rejected) rejected")

        // Forced: x'' + x = sin 2t from rest is (2/3) sin t − (1/3) sin 2t.
        // The time argument is live, so a stage evaluated at the wrong c_i
        // shows here and nowhere in an autonomous test.
        let forced = { (t: Double, y: SIMD2<Double>) in SIMD2(y.y, -y.x + sin(2 * t)) }
        let run = try ODE.sample(forced, from: 0, SIMD2(0, 0), every: 0.05, count: 2001)
        var drift = 0.0
        for (k, y) in run.states.enumerated() {
            let t = 0.05 * Double(k)
            drift = max(drift, abs(y.x - (2.0 / 3 * sin(t) - 1.0 / 3 * sin(2 * t))))
        }
        Check.expect(run.states.count == 2001 && drift < 1e-8,
                     "a forced oscillator, sampled every 0.05 to t = 100",
                     "max error \(drift)")
    }

    Check.suite("ode: a Kepler orbit keeps its energy and angular momentum") {
        // Eccentricity 0.6, ten periods: the close pass is where the step
        // control earns its keep.
        let kepler = { (_: Double, s: SIMD4<Double>) -> SIMD4<Double> in
            let r = SIMD2(s.x, s.y), v = SIMD2(s.z, s.w)
            let a = -r / pow(simd_length_squared(r), 1.5)
            return SIMD4(v.x, v.y, a.x, a.y)
        }
        let e = 0.6
        let start = SIMD4<Double>(1 - e, 0, 0, ((1 + e) / (1 - e)).squareRoot())
        func energy(_ s: SIMD4<Double>) -> Double {
            0.5 * (s.z * s.z + s.w * s.w) - 1 / (s.x * s.x + s.y * s.y).squareRoot()
        }
        func momentum(_ s: SIMD4<Double>) -> Double { s.x * s.w - s.y * s.z }
        var last = start
        let stats = try ODE.integrate(kepler, from: 0, start, to: 20 * .pi,
                                      tolerance: .init(relative: 1e-11, absolute: 1e-13)) {
            last = $0.state(at: $0.t1)
        }
        let dE = abs(energy(last) - energy(start)) / abs(energy(start))
        let dL = abs(momentum(last) - momentum(start)) / abs(momentum(start))
        // Back where it started, too: the period is 2π for a = 1.
        let home = simd_length(last - start)
        Check.expect(dE < 1e-8 && dL < 1e-8 && home < 1e-6,
                     "after ten periods",
                     "energy \(dE), momentum \(dL), distance from the start \(home); "
                     + "\(stats.accepted) steps")
    }

    Check.suite("ode: a discontinuous right-hand side, with a step cap") {
        // y' = sign(sin t): a triangle wave, y(t) = 1 − cos-like zigzag. Its
        // corners are where an unbounded step would stride over the change
        // and accept an estimate that never saw it.
        let zigzag = { (t: Double, y: SIMD2<Double>) -> SIMD2<Double> in
            SIMD2(sin(t) >= 0 ? 1 : -1, 0)
        }
        func exact(_ t: Double) -> Double {
            let p = t.truncatingRemainder(dividingBy: 2 * .pi)
            return p <= .pi ? p : 2 * .pi - p
        }
        let run = try ODE.sample(zigzag, from: 0, SIMD2(0, 0), every: 0.1, count: 301,
                                 tolerance: .init(maxStep: 0.05))
        let worst = run.states.enumerated().map { abs($1.x - exact(0.1 * Double($0))) }.max()!
        Check.expect(worst < 1e-6, "tracks the zigzag through every corner", "max error \(worst)")
    }
}

// MARK: - frequency analysis and the Fourier torus

func frequencyTests() {
    Check.suite("frequency: the FFT is the DFT") {
        let rng = SplitMix(seed: 3)
        let n = 256
        let xr = (0..<n).map { _ in rng.next(-1, 1) }, xi = (0..<n).map { _ in rng.next(-1, 1) }
        var re = xr, im = xi
        Frequency.fft(&re, &im)
        var worst = 0.0
        for k in 0..<n {
            var sr = 0.0, si = 0.0
            for j in 0..<n {
                let a = -2 * Double.pi * Double(j * k % n) / Double(n)
                sr += xr[j] * cos(a) - xi[j] * sin(a)
                si += xr[j] * sin(a) + xi[j] * cos(a)
            }
            worst = max(worst, hypot(sr - re[k], si - im[k]))
        }
        Check.expect(worst < 1e-11, "every bin, to rounding", "max \(worst)")
    }

    Check.suite("frequency: lines and the generator of a torus's spectrum") {
        // Lines at ω, Ω, ω + 2Ω and 2ω − Ω, with the strongest non-forcing
        // line not the generator itself -- the case that makes "take the
        // strongest line" wrong.
        let omega = 0.94, Omega = 0.396_221_328_3
        let dt = 0.05, n = 80_000
        let x = (0..<n).map { k -> Double in
            let t = 400 + Double(k) * dt
            return cos(omega * t) + 0.3 * cos(Omega * t + 0.3)
                + 0.8 * cos((omega + 2 * Omega) * t + 1) + 0.1 * cos((2 * omega - Omega) * t)
        }
        let lines = Frequency.lines(x, t0: 400, dt: dt, count: 4)
        let truth = [omega, Omega, omega + 2 * Omega, 2 * omega - Omega]
        let worst = truth.map { f in lines.map { abs($0.frequency - f) }.min() ?? .infinity }.max()!
        Check.expect(lines.count == 4 && worst < 1e-9,
                     "each line is found to its exact frequency", "worst \(worst)")
        let g = Frequency.generator(lines, forcing: omega)
        Check.expect(g != nil && abs(g!.frequency - Omega) < 1e-9 && g!.explained == 3,
                     "and the generator is Ω though ω + 2Ω is the stronger line",
                     g.map { "Ω = \($0.frequency), explains \($0.explained)" } ?? "none")
    }

    Check.suite("frequency: a torus that is a Fourier series is fitted exactly") {
        // K(θ, φ) of degree three in each angle, sampled along the line
        // (ω₁t, ω₂t): the fit at M = 4 has every true coefficient and should
        // reproduce K everywhere on the torus, not only along the samples.
        func K(_ a: Double, _ b: Double) -> SIMD3<Double> {
            SIMD3(2 * cos(a) + 0.5 * cos(a + b) - 0.2 * sin(3 * b),
                  2 * sin(a) + 0.3 * sin(2 * a - b) + 0.1,
                  0.7 * sin(b) + 0.2 * cos(a - 3 * b))
        }
        let w1 = 0.94, w2 = 0.396_221_328_3, dt = 0.05
        // What the averages cannot remove is a finite run's leakage between
        // lines, and the Hann² window makes that fall steeply with length:
        // measured at two lengths, it must.
        func errors(_ n: Int) -> (value: Double, slope: Double) {
            let samples = (0..<n).map { k -> SIMD3<Double> in
                let t = Double(k) * dt
                return K(w1 * t, w2 * t)
            }
            let fit = FourierTorus.fit(samples, t0: 0, dt: dt, frequencies: (w1, w2), harmonics: 4)
            let rng = SplitMix(seed: 5)
            var worst = 0.0, slope = 0.0
            for _ in 0..<500 {
                let a = rng.next(0, 2 * .pi), b = rng.next(0, 2 * .pi)
                let j = fit.jet(a, b)
                worst = max(worst, simd_length(j.value - K(a, b)))
                let h = 1e-6
                let da = (K(a + h, b) - K(a - h, b)) / (2 * h)
                let db = (K(a, b + h) - K(a, b - h)) / (2 * h)
                slope = max(slope, simd_length(j.dTheta - da), simd_length(j.dPhi - db))
            }
            return (worst, slope)
        }
        let short = errors(80_000), long = errors(160_000)
        Check.expect(long.value < 1e-8 && short.value / long.value > 8,
                     "everywhere on the torus, not only on the line, and better with length",
                     "max \(short.value) at T = 4000, \(long.value) at T = 8000")
        Check.expect(long.slope < 1e-7, "and so are its derivatives", "max \(long.slope)")
    }
}
