import Foundation

/// The Riemann zeta function of a complex argument: `kurven.expr`'s algorithm,
/// point by point.
///
/// Borwein's alternating series on Re s >= 1/2, with the order chosen from
/// |Im s| so a disc around the origin does not pay the ζ plate's price; the
/// functional equation below the critical line; and Euler-Maclaurin at the
/// points where Borwein's own factor `1 - 2^(1-s)` vanishes (σ = 1,
/// t = 2πk/ln 2), where the quotient would otherwise be 0/0.
public enum Zeta {
    static let logGrowth = Foundation.log(3 + Foundation.sqrt(8.0))

    /// B_{2k} for k = 0...12.
    static let bernoulliEven: [Double] = [
        1, 1.0 / 6, -1.0 / 30, 1.0 / 42, -1.0 / 30, 5.0 / 66, -691.0 / 2730,
        7.0 / 6, -3617.0 / 510, 43867.0 / 798, -174611.0 / 330,
        854513.0 / 138, -236364091.0 / 2730,
    ]

    /// `d_k`, k = 0...n, by the ratio between consecutive terms: the closed
    /// form is a sum of factorials that overflows long before the n this needs.
    static func borweinCoefficients(_ n: Int) -> [Double] {
        var d = [Double](repeating: 0, count: n + 1)
        var term = 1.0 / Double(n)
        var total = 0.0
        for j in 0...n {
            total += term
            d[j] = Double(n) * total
            term *= 4.0 * Double(n + j) * Double(n - j) / Double((2 * j + 1) * (2 * j + 2))
        }
        return d
    }

    /// The orders are few (they depend on |t| through a ceiling), so the
    /// coefficient tables are memoized behind a lock.
    private static let cache = Cache()
    private final class Cache: @unchecked Sendable {
        private var tables: [Int: [Double]] = [:]
        private let lock = NSLock()
        func table(_ n: Int) -> [Double] {
            lock.lock(); defer { lock.unlock() }
            if let t = tables[n] { return t }
            let t = borweinCoefficients(n)
            tables[n] = t
            return t
        }
    }

    /// How many terms the error bound needs at this height:
    /// `|γ_n(s)| ≲ 3(1 + 2|t|)e^{π|t|/2}(3+√8)^{-n}`.
    static func order(forHeight t: Double) -> Int {
        let need = (.pi * t / 2 + Foundation.log(3.0 * (1 + 2 * t))
                    + 15 * Foundation.log(10.0)) / logGrowth
        return Int(min(max(need.rounded(.up), 16), 200))
    }

    static func eulerMaclaurin(_ s: Complex, terms: Int = 12) -> Complex {
        let cut = max(10, Int(abs(s.im)) + 10)
        var total = Complex.zero
        for k in 1..<cut {
            total += Complex.exp(-s * Foundation.log(Double(k)))
        }
        let c = Double(cut)
        total += Complex.pow(Complex(c), 1 - s) / (s - 1) + 0.5 * Complex.pow(Complex(c), -s)
        var product = s
        var factorial = 1.0
        for j in 1...terms {
            factorial *= Double((2 * j - 1) * (2 * j))
            total += bernoulliEven[j] / factorial * product
                * Complex.pow(Complex(c), -s - Double(2 * j - 1))
            product = product * (s + Double(2 * j - 1)) * (s + Double(2 * j))
        }
        return total
    }

    /// ζ on Re s >= 1/2.
    static func halfPlane(_ s: Complex) -> Complex {
        let n = order(forHeight: abs(s.im))
        let d = cache.table(n)
        var total = Complex.zero
        for k in 0..<n {
            var w = (d[k] - d[n]) / d[n]
            if k % 2 == 1 { w = -w }
            total += w * Complex.exp(-s * Foundation.log(Double(k + 1)))
        }
        let eta = 1 - Complex.pow(Complex(2), 1 - s)
        if eta.magnitude < 1e-5 { return eulerMaclaurin(s) }
        return -total / eta
    }

    public static func zeta(_ s: Complex) -> Complex {
        if s.re == 1 && s.im == 0 { return Complex(.infinity, 0) }
        if s.re == 0 && s.im == 0 { return Complex(-0.5, 0) }
        if s.re >= 0.5 { return halfPlane(s) }
        // The functional equation: 2^s π^(s-1) sin(πs/2) Γ(1-s) ζ(1-s).
        return Complex.pow(Complex(2), s) * Complex.pow(Complex(.pi), s - 1)
            * Complex.sin(.pi * s / 2) * Gamma.gamma(1 - s) * halfPlane(1 - s)
    }
}
