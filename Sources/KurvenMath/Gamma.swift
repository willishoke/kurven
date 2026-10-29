import Foundation

/// The gamma family: log Γ, Γ, 1/Γ and ψ, for complex arguments.
///
/// `logGamma` is scipy's algorithm (Hare, "Computing the principal branch of
/// log-Gamma", 1997): Stirling's series where it converges, a Taylor series
/// around 1 and 2, the reflection formula on the left, and otherwise a
/// recurrence that counts how many times the shifted product crosses the
/// negative real axis so the branch stays the principal one. Γ and 1/Γ are its
/// exponentials, which is also how scipy computes them, so the two agree to
/// the last few bits rather than to a tolerance.
public enum Gamma {
    static let smallX = 7.0
    static let smallY = 7.0
    static let taylorRadius = 0.2
    static let logPi = Foundation.log(Double.pi)
    static let halfLog2Pi = 0.5 * Foundation.log(2 * Double.pi)

    /// B_{2n} / (2n (2n - 1)), highest power of 1/z^2 first.
    static let stirling: [Double] = [
        -2.955065359477124183e-2, 6.4102564102564102564e-3,
        -1.9175269175269175269e-3, 8.4175084175084175084e-4,
        -5.952380952380952381e-4, 7.9365079365079365079e-4,
        -2.7777777777777777778e-3, 8.3333333333333333333e-2,
    ]

    /// log Γ(1 + x) = -γ x + Σ (-1)^k ζ(k)/k x^k, k = 2...23, highest first.
    static let taylor: [Double] = [
        -0.04347826605304026, 0.04545455629320467, -0.04761907033014223,
        0.05000004769810169, -0.05263167937961666, 0.055555767627403614,
        -0.05882397865868458, 0.06250095514121304, -0.06666870588242046,
        0.07143294629536134, -0.0769325164113522, 0.083353840546109,
        -0.09095401714582904, 0.10009945751278179, -0.11133426586956469,
        0.12550966952474304, -0.14404989676884614, 0.16955717699740822,
        -0.207385551028674, 0.27058080842778454, -0.40068563438653143,
        0.8224670334241132, -0.5772156649015329,
    ]

    public static let euler = 0.5772156649015329

    @inline(__always)
    static func evalPoly(_ coefficients: [Double], _ z: Complex) -> Complex {
        var r = Complex.zero
        for c in coefficients { r = r * z + c }
        return r
    }

    static func stirlingSeries(_ z: Complex) -> Complex {
        let rz = 1 / z
        let rzz = rz / z
        return (z - 0.5) * Complex.log(z) - z + halfLog2Pi + rz * evalPoly(stirling, rzz)
    }

    static func taylorSeries(_ z: Complex) -> Complex {
        let w = z - 1
        return w * evalPoly(taylor, w)
    }

    /// Requires Im z >= 0 (not a negative zero).
    static func recurrence(_ z: Complex) -> Complex {
        var signflips = 0
        var sb = false
        var shiftprod = z
        var w = Complex(z.re + 1, z.im)
        while w.re <= smallX {
            shiftprod = shiftprod * w
            let nsb = shiftprod.im.sign == .minus
            if nsb && !sb { signflips += 1 }
            sb = nsb
            w = Complex(w.re + 1, w.im)
        }
        return stirlingSeries(w) - Complex.log(shiftprod)
            - Complex(0, Double(signflips) * 2 * .pi)
    }

    /// True at 0, -1, -2, ...: where Γ has its poles.
    public static func isPole(_ z: Complex) -> Bool {
        z.im == 0 && z.re <= 0 && z.re == z.re.rounded(.down)
    }

    /// The principal branch of log Γ. NaN at the poles.
    public static func logGamma(_ z: Complex) -> Complex {
        if isPole(z) { return Complex(.nan, .nan) }
        if z.re > smallX || abs(z.im) > smallY { return stirlingSeries(z) }
        if (z - 1).magnitude <= taylorRadius { return taylorSeries(z) }
        if (z - 2).magnitude <= taylorRadius {
            return Complex.log(z - 1) + taylorSeries(z - 1)
        }
        if z.re < 0.1 {
            let tmp = copysign(2 * .pi, z.im) * (0.5 * z.re + 0.25).rounded(.down)
            return Complex(logPi, tmp) - Complex.log(Trig.sinPi(z)) - logGamma(1 - z)
        }
        if z.im.sign == .plus { return recurrence(z) }
        return recurrence(z.conjugate).conjugate
    }

    /// Γ(z); infinite at the poles rather than NaN, because a landscape is a
    /// picture of where they are.
    public static func gamma(_ z: Complex) -> Complex {
        if isPole(z) { return Complex(.infinity, 0) }
        return Complex.exp(logGamma(z))
    }

    /// 1/Γ(z), entire: exactly zero at the poles of Γ.
    public static func rgamma(_ z: Complex) -> Complex {
        if isPole(z) { return .zero }
        return Complex.exp(-logGamma(z))
    }

    /// B_{2k} / (2k), k = 1...8.
    static let digammaAsymptotic: [Double] = [
        1.0 / 12, -1.0 / 120, 1.0 / 252, -1.0 / 240,
        1.0 / 132, -691.0 / 32760, 1.0 / 12, -3617.0 / 8160,
    ]

    /// ψ(z) = Γ'(z)/Γ(z): reflection onto Re z >= 1/2, recurrence out to
    /// |z| >= 16, then the asymptotic series.
    public static func digamma(_ z: Complex) -> Complex {
        if isPole(z) { return Complex(.infinity, 0) }
        var res = Complex.zero
        var w = z
        if w.re < 0.5 {
            res -= .pi * Trig.cosPi(w) / Trig.sinPi(w)
            w = 1 - w
        }
        while w.magnitude < 16 {
            res -= 1 / w
            w += .one
        }
        let rz = 1 / w
        let rzz = rz * rz
        var tail = Complex.zero
        var p = rzz
        for c in digammaAsymptotic {
            tail += c * p
            p *= rzz
        }
        return res + Complex.log(w) - 0.5 * rz - tail
    }
}

/// sin(πz) and cos(πz) with exact zeros at the integers and half-integers.
///
/// The real reductions are scipy's, including the detail that `cosPi` is a
/// *positive* zero at every half-integer: that zero, multiplied by sinh(πy),
/// is the imaginary part of sin(πz) on the lines Re z = n + 1/2, and its sign
/// picks the branch of the logarithm in `logGamma`'s reflection formula.
public enum Trig {
    public static func sinPi(_ x: Double) -> Double {
        var s = 1.0
        var r = fmod(x, 2.0)
        if r < 0 { r += 2 }
        if r > 1 { r -= 1; s = -1 }
        if r > 0.5 { r = 1 - r }
        if r == 0.5 { return s }
        return s * Foundation.sin(.pi * r)
    }

    public static func cosPi(_ x: Double) -> Double {
        var r = fmod(abs(x), 2.0)
        var s = 1.0
        if r > 1 { r -= 1; s = -1 }
        if r == 0.5 { return 0 }
        return s * Foundation.sin(.pi * (0.5 - r))
    }

    public static func sinPi(_ z: Complex) -> Complex {
        let piy = Double.pi * z.im
        return Complex(sinPi(z.re) * Foundation.cosh(piy), cosPi(z.re) * Foundation.sinh(piy))
    }

    public static func cosPi(_ z: Complex) -> Complex {
        let piy = Double.pi * z.im
        return Complex(cosPi(z.re) * Foundation.cosh(piy), -sinPi(z.re) * Foundation.sinh(piy))
    }
}
