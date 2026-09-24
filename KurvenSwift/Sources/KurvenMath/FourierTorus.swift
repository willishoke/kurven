import Foundation
import Dispatch

/// An invariant torus as a two-dimensional Fourier series,
/// `K(θ, φ) = Σ c_{mn} e^{i(mθ + nφ)}` over `|m|, |n| ≤ M`, fitted to a
/// quasi-periodic trajectory with basic frequencies `(ω₁, ω₂)`.
///
/// On the torus the trajectory is the straight line `(θ, φ) = (ω₁t, ω₂t)`,
/// and with incommensurate frequencies that line covers it evenly -- so a
/// coefficient is a time average, `c_{mn} = ⟨w(t) x(t) e^{-i(mω₁ + nω₂)t}⟩`,
/// weighted by the Hann² window so the finite run does not leak each line
/// into its neighbours. No linear system is solved; on the systems measured
/// this matches least squares at the same order, for a cost of one pass over
/// the samples per coefficient block.
///
/// A real signal's coefficients come in conjugate pairs, `c_{-m,-n} =
/// conj(c_{mn})`, so only `m ≥ 0` is stored: `K = Re Σ_{m≥0} μ_m Σ_n c_{mn}
/// e^{i(mθ+nφ)}` with `μ_0 = 1` and `μ_m = 2` otherwise.
public struct FourierTorus<V: SIMD>: Sendable where V.Scalar == Double, V: Sendable {
    public let frequencies: (Double, Double)
    public let harmonics: Int
    /// `c[(m * (2M + 1) + n + M)]`, real and imaginary parts, one lane per
    /// state component.
    @usableFromInline let re: [V]
    @usableFromInline let im: [V]

    @usableFromInline var width: Int { 2 * harmonics + 1 }

    @usableFromInline
    init(frequencies: (Double, Double), harmonics: Int, re: [V], im: [V]) {
        self.frequencies = frequencies; self.harmonics = harmonics; self.re = re; self.im = im
    }

    /// The fit, over samples at `t0 + k dt`.
    @inlinable
    public static func fit(_ samples: [V], t0: Double, dt: Double,
                           frequencies: (Double, Double), harmonics M: Int) -> FourierTorus {
        let n = samples.count
        let w = Frequency.window(n)
        let width = 2 * M + 1, blocks = (M + 1) * width
        // Split over chunks of samples, each accumulating its own sums.
        let chunks = max(1, min(ProcessInfo.processInfo.activeProcessorCount, n / 4096))
        var partial = [[V]](repeating: [], count: 2 * chunks)
        partial.withUnsafeMutableBufferPointer { out in
            nonisolated(unsafe) let base = out.baseAddress!
            DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
                var sr = [V](repeating: V(), count: blocks), si = sr
                let a = UnsafeMutablePointer<Complex>.allocate(capacity: M + 1 + width)
                defer { a.deallocate() }
                let b = a + (M + 1)
                for k in stride(from: chunk * n / chunks, to: (chunk + 1) * n / chunks, by: 1) {
                    let t = t0 + Double(k) * dt
                    phasors(-frequencies.0 * t, -frequencies.1 * t, M, a, b)
                    let x = w[k] * samples[k]
                    for m in 0...M {
                        for j in 0..<width {
                            let e = a[m] * b[j]
                            sr[m * width + j] += e.re * x
                            si[m * width + j] += e.im * x
                        }
                    }
                }
                base[2 * chunk] = sr; base[2 * chunk + 1] = si
            }
        }
        var re = [V](repeating: V(), count: blocks), im = re
        for chunk in 0..<chunks {
            for i in 0..<blocks {
                re[i] += partial[2 * chunk][i]; im[i] += partial[2 * chunk + 1][i]
            }
        }
        for i in 0..<blocks { re[i] /= Double(n); im[i] /= Double(n) }
        return FourierTorus(frequencies: frequencies, harmonics: M, re: re, im: im)
    }

    /// `e^{imα}` for `0 ≤ m ≤ M` into `a`, and `e^{inβ}` for `-M ≤ n ≤ M`
    /// into `b` (at `n + M`), by multiplication from one sine and cosine each.
    @inlinable @inline(__always)
    static func phasors(_ alpha: Double, _ beta: Double, _ M: Int,
                        _ a: UnsafeMutablePointer<Complex>, _ b: UnsafeMutablePointer<Complex>) {
        let ea = Complex(cos(alpha), sin(alpha)), eb = Complex(cos(beta), sin(beta))
        a[0] = Complex(1)
        for m in stride(from: 1, through: M, by: 1) { a[m] = a[m - 1] * ea }
        b[M] = Complex(1)
        let conj = Complex(eb.re, -eb.im)
        for n in stride(from: 1, through: M, by: 1) {
            b[M + n] = b[M + n - 1] * eb
            b[M - n] = b[M - n + 1] * conj
        }
    }

    /// The torus at `(θ, φ)`, with its two partial derivatives. Allocates
    /// nothing: the phasors live on the stack for the call.
    @inlinable
    public func jet(_ theta: Double, _ phi: Double) -> (value: V, dTheta: V, dPhi: V) {
        let M = harmonics, width = self.width
        return withUnsafeTemporaryAllocation(of: Complex.self, capacity: M + 1 + width) { buf in
            let a = buf.baseAddress!, b = a + (M + 1)
            Self.phasors(theta, phi, M, a, b)
            var value = V(), dTheta = V(), dPhi = V()
            for m in 0...M {
                let mu = m == 0 ? 1.0 : 2.0
                for j in 0..<width {
                    let e = a[m] * b[j]
                    let i = m * width + j
                    // Re(c e), and Re(i k c e) = -k Im(c e), lane by lane.
                    let real = e.re * re[i] - e.im * im[i]
                    let imag = e.re * im[i] + e.im * re[i]
                    value += mu * real
                    dTheta -= (mu * Double(m)) * imag
                    dPhi -= (mu * Double(j - M)) * imag
                }
            }
            return (value, dTheta, dPhi)
        }
    }

    @inlinable
    public func value(_ theta: Double, _ phi: Double) -> V { jet(theta, phi).value }

    /// The worst distance, lane-wise Euclidean over the first `lanes` lanes,
    /// between the samples and the torus at their own phases: how well the
    /// fit holds the trajectory it came from.
    @inlinable
    public func residual(_ samples: [V], t0: Double, dt: Double, lanes: Int) -> Double {
        let n = samples.count
        let chunks = max(1, min(ProcessInfo.processInfo.activeProcessorCount, n / 1024))
        var worst = [Double](repeating: 0, count: chunks)
        worst.withUnsafeMutableBufferPointer { out in
            nonisolated(unsafe) let base = out.baseAddress!
            DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
                var w = 0.0
                for k in stride(from: chunk * n / chunks, to: (chunk + 1) * n / chunks, by: 1) {
                    let t = t0 + Double(k) * dt
                    let d = value(frequencies.0 * t, frequencies.1 * t) - samples[k]
                    var s = 0.0
                    for l in 0..<lanes { s += d[l] * d[l] }
                    w = max(w, s.squareRoot())
                }
                base[chunk] = w
            }
        }
        return worst.max() ?? 0
    }
}
