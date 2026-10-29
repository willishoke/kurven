import Foundation

/// Frequency analysis of a quasi-periodic signal.
///
/// A trajectory winding round an invariant torus is a sum of sinusoids at
/// the integer combinations `m ω₁ + n ω₂` of two basic frequencies. The FFT
/// finds each peak to within a bin; refining it -- maximizing the windowed
/// amplitude `|⟨w x e^{-iνt}⟩|` over ν, Laskar's NAFF -- finds it to many
/// orders of magnitude better, because the window makes the amplitude a
/// smooth function of ν with its maximum exactly on the line. That precision
/// is not a nicety: an error δ in a frequency drifts the phase by δ·T across
/// the run, and the torus fit reads coordinates from those phases.
public enum Frequency {
    /// The Hann window squared, normalized to mean one. Its sidelobes fall
    /// fast enough that neighbouring lines do not pull a peak off its true
    /// frequency, and its squared form is what the torus fit averages with.
    public static func window(_ count: Int) -> [Double] {
        guard count > 1 else { return [1] }
        var w = (0..<count).map { k -> Double in
            let s = Double(k) / Double(count - 1)
            let h = 1 - cos(2 * .pi * s)
            return h * h
        }
        let mean = w.reduce(0, +) / Double(count)
        for i in w.indices { w[i] /= mean }
        return w
    }

    /// `⟨w x e^{-iνt}⟩` over samples at `t0 + k dt`: the windowed amplitude
    /// at angular frequency ν. The phasor is advanced by multiplication and
    /// renormalized every thousand samples, so its drift over a long series
    /// stays at rounding.
    public static func amplitude(_ x: [Double], weights w: [Double], t0: Double, dt: Double,
                                 at nu: Double) -> Complex {
        let step = Complex(cos(-nu * dt), sin(-nu * dt))
        var phase = Complex(cos(-nu * t0), sin(-nu * t0))
        var re = 0.0, im = 0.0
        for k in x.indices {
            let a = w[k] * x[k]
            re += a * phase.re; im += a * phase.im
            phase = phase * step
            if k % 1024 == 1023 {
                let m = phase.magnitude
                phase = Complex(phase.re / m, phase.im / m)
            }
        }
        let n = Double(x.count)
        return Complex(re / n, im / n)
    }

    /// In-place radix-2 FFT, forward (`e^{-i...}`); the length must be a
    /// power of two.
    public static func fft(_ re: inout [Double], _ im: inout [Double]) {
        let n = re.count
        precondition(n == im.count && n > 0 && n & (n - 1) == 0, "FFT length must be 2^k")
        var j = 0
        for i in 1..<max(n, 2) where i < n {
            var bit = n >> 1
            while j & bit != 0 { j ^= bit; bit >>= 1 }
            j |= bit
            if i < j { re.swapAt(i, j); im.swapAt(i, j) }
        }
        var len = 2
        while len <= n {
            let angle = -2 * Double.pi / Double(len)
            for start in stride(from: 0, to: n, by: len) {
                for k in 0..<(len / 2) {
                    let c = cos(angle * Double(k)), s = sin(angle * Double(k))
                    let a = start + k, b = a + len / 2
                    let tr = re[b] * c - im[b] * s, ti = re[b] * s + im[b] * c
                    re[b] = re[a] - tr; im[b] = im[a] - ti
                    re[a] += tr; im[a] += ti
                }
            }
            len <<= 1
        }
    }

    public struct Line: Sendable, Equatable {
        /// Angular frequency.
        public var frequency: Double
        public var amplitude: Double
    }

    /// The strongest `count` spectral lines of `x`, each refined to its
    /// exact frequency.
    ///
    /// The windowed, mean-removed series is transformed at twice its length
    /// or more, the local maxima of the spectrum taken strongest first --
    /// no two within three bins of the series' own resolution, `2π/T` -- and
    /// each refined by golden-section search on the windowed amplitude over
    /// the bin either side.
    public static func lines(_ x: [Double], t0: Double, dt: Double, count: Int) -> [Line] {
        let n = x.count
        guard n >= 8 else { return [] }
        let w = window(n)
        let mean = zip(w, x).reduce(0) { $0 + $1.0 * $1.1 } / Double(n)
        let centred = x.map { $0 - mean }

        var size = 1
        while size < 2 * n { size <<= 1 }
        var re = [Double](repeating: 0, count: size), im = re
        for k in 0..<n { re[k] = w[k] * centred[k] }
        fft(&re, &im)
        let binWidth = 2 * Double.pi / (dt * Double(size))
        let magnitude = (0...(size / 2)).map { (re[$0] * re[$0] + im[$0] * im[$0]).squareRoot() }
        let maxima = (1..<(size / 2)).filter {
            magnitude[$0] > magnitude[$0 - 1] && magnitude[$0] >= magnitude[$0 + 1]
        }.sorted { magnitude[$0] > magnitude[$1] }

        let resolution = 2 * Double.pi / (dt * Double(n))
        var found: [Line] = []
        for bin in maxima {
            let guess = Double(bin) * binWidth
            guard found.allSatisfy({ abs($0.frequency - guess) > 3 * resolution }) else { continue }
            let amp = { (nu: Double) in amplitude(centred, weights: w, t0: t0, dt: dt, at: nu).magnitude }
            var a = guess - resolution, b = guess + resolution
            let g = (5.0.squareRoot() - 1) / 2
            var c = b - g * (b - a), d = a + g * (b - a)
            var fc = amp(c), fd = amp(d)
            while b - a > 1e-13 * max(1, guess) {
                if fc > fd { b = d; d = c; fd = fc; c = b - g * (b - a); fc = amp(c) }
                else { a = c; c = d; fc = fd; d = a + g * (b - a); fd = amp(d) }
            }
            let nu = 0.5 * (a + b)
            found.append(Line(frequency: nu, amplitude: amp(nu)))
            if found.count == count { break }
        }
        return found.sorted { $0.amplitude > $1.amplitude }
    }

    /// The second basic frequency of a torus forced at ω, from its lines.
    ///
    /// Every line is `m ω + n Ω`. Reduced mod ω the harmonics of the forcing
    /// vanish and the rest are `n Ω mod ω`, so the generator is the reduced
    /// frequency of which the others are small multiples -- not simply the
    /// strongest line, which on a torus can as well be `2Ω` and would then
    /// describe half of it. Each candidate, from the strongest lines, is
    /// scored by how many lines it explains with `|n| ≤ maxMultiple`, and
    /// ties go to the stronger. Returned in `(0, ω/2]`: `Ω` and `ω − Ω`
    /// generate the same torus with the second angle reversed.
    public static func generator(_ lines: [Line], forcing omega: Double,
                                 maxMultiple: Int = 4, tolerance: Double = 1e-7)
        -> (frequency: Double, explained: Int)?
    {
        func reduce(_ nu: Double) -> Double {
            let r = nu.truncatingRemainder(dividingBy: omega)
            return r < 0 ? r + omega : r
        }
        let tol = tolerance * omega
        let reduced = lines.map { reduce($0.frequency) }
            .filter { $0 > tol && omega - $0 > tol }     // not a harmonic of ω
        var best: (frequency: Double, explained: Int)?
        for c in reduced {
            let explained = reduced.filter { r in
                (1...maxMultiple).contains { n in
                    let k = reduce(Double(n) * c)
                    return abs(k - r) < Double(n) * tol || abs(omega - k - r) < Double(n) * tol
                }
            }.count
            if best == nil || explained > best!.explained {
                best = (min(c, omega - c), explained)
            }
        }
        return best
    }
}
