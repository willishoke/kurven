import Foundation

/// The principal branch of the Lambert W function: scipy's algorithm, with a
/// tighter stopping tolerance than its ufunc default.
public enum LambertW {
    static let expMinusOne = Foundation.exp(-1.0)

    public static func w0(_ z: Complex, tolerance: Double = 1e-12) -> Complex {
        if z.re == 0 && z.im == 0 { return .zero }
        if z.re == -expMinusOne && z.im == 0 { return Complex(-1, 0) }
        var w: Complex
        if (z + expMinusOne).magnitude < 0.3 {
            // Series about the branch point.
            let p = Complex.sqrt(2 * (M_E * z + 1))
            w = -1 + p - p * p / 3
        } else if -1.0 < z.re && z.re < 1.5 && abs(z.im) < 1.0
                    && -2.5 * abs(z.im) - 0.2 < z.re {
            // Padé approximant about the origin.
            let num = [12.85106382978723404255, 12.34042553191489361902, 1.0]
            let den = [32.53191489361702127660, 14.34042553191489361702, 1.0]
            w = z * Gamma.evalPoly(num, z) / Gamma.evalPoly(den, z)
        } else {
            w = Complex.log(z)
            w = w - Complex.log(w)
        }
        // Halley's iteration, in the form that keeps its exponential bounded.
        if w.re >= 0 {
            for _ in 0..<100 {
                let ew = Complex.exp(-w)
                let wewz = w - z * ew
                let wn = w - wewz / (w + 1 - (w + 2) * wewz / (2 * w + 2))
                if (wn - w).magnitude <= tolerance * wn.magnitude { return wn }
                w = wn
            }
        } else {
            for _ in 0..<100 {
                let ew = Complex.exp(w)
                let wew = w * ew
                let wewz = wew - z
                let wn = w - wewz / (wew + ew - (w + 2) * wewz / (2 * w + 2))
                if (wn - w).magnitude <= tolerance * wn.magnitude { return wn }
                w = wn
            }
        }
        return w
    }
}
