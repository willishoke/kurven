import Foundation

/// The Jacobi elliptic functions sn, cn, dn of a complex argument.
///
/// The real ones are cephes' `ellpj` (scipy's `ellipj`): the arithmetic-
/// geometric mean, with the two ends of the parameter range handled by their
/// own expansions. The imaginary direction comes from the addition formulas
/// with the complementary parameter, exactly as `kurven.expr._jacobi` writes
/// them, so the two sides agree to rounding rather than to a tolerance.
public enum Jacobi {
    public struct Real: Sendable { public var sn, cn, dn, ph: Double }

    /// `ellpj(u, m)` for 0 <= m <= 1.
    public static func real(_ u: Double, _ m: Double) -> Real {
        if m < 0 || m > 1 || m.isNaN {
            return Real(sn: .nan, cn: .nan, dn: .nan, ph: .nan)
        }
        if m < 1e-9 {
            let t = sin(u), b = cos(u)
            let ai = 0.25 * m * (u - t * b)
            return Real(sn: t - ai * b, cn: b + ai * t, dn: 1 - 0.5 * m * t * t, ph: u - ai)
        }
        if m >= 0.9999999999 {
            var ai = 0.25 * (1 - m)
            let b = cosh(u), t = tanh(u), phi = 1 / b
            let twon = b * sinh(u)
            let sn = t + ai * (twon - u) / (b * b)
            let ph = 2 * atan(exp(u)) - .pi / 2 + ai * (twon - u) / b
            ai *= t * phi
            return Real(sn: sn, cn: phi - ai * (twon - u), dn: phi + ai * (twon + u), ph: ph)
        }
        var a = [Double](repeating: 1, count: 9)
        var c = [Double](repeating: 0, count: 9)
        c[0] = m.squareRoot()
        var b = (1 - m).squareRoot()
        var twon = 1.0
        var i = 0
        while abs(c[i] / a[i]) > 2.220446049250313e-16 {
            if i > 7 { break }
            let ai = a[i]
            i += 1
            c[i] = (ai - b) / 2
            let t = (ai * b).squareRoot()
            a[i] = (ai + b) / 2
            b = t
            twon *= 2
        }
        var phi = twon * a[i] * u
        var bb = 0.0
        while i > 0 {
            let t = c[i] * sin(phi) / a[i]
            bb = phi
            phi = (asin(t) + phi) / 2
            i -= 1
        }
        let sn = sin(phi), t = cos(phi)
        let dnfac = cos(phi - bb)
        let dn = abs(dnfac) < 0.1 ? (1 - m * sn * sn).squareRoot() : t / dnfac
        return Real(sn: sn, cn: t, dn: dn, ph: phi)
    }

    /// (sn, cn, dn)(z, m) for complex z.
    public static func complex(_ z: Complex, _ m: Double) -> (sn: Complex, cn: Complex, dn: Complex) {
        let u = real(z.re, m)
        let v = real(z.im, 1 - m)
        let (s, c, d) = (u.sn, u.cn, u.dn)
        let (s1, c1, d1) = (v.sn, v.cn, v.dn)
        let den = c1 * c1 + m * s * s * s1 * s1
        let sn = Complex(s * d1, c * d * s1 * c1) / den
        let cn = Complex(c * c1, -s * d * s1 * d1) / den
        let dn = Complex(d * c1 * d1, -m * s * c * s1) / den
        return (sn, cn, dn)
    }
}
