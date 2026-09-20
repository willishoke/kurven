import Foundation

/// The exponential integral E₁(z), and Ei as `kurven.expr` continues it.
public enum ExpInt {
    /// E₁(z), cut along the negative real axis.
    ///
    /// The Maclaurin series where it does not cancel -- it alternates only
    /// through the imaginary direction, so its loss is e^(|z| - |Re z|) on
    /// the left, which is harmless near the negative real axis, exactly where
    /// the continued fraction converges slowest -- and the continued fraction
    /// `e^{-z}/(z + 1/(1 + 1/(z + 2/(1 + 2/(z + ...)))))` by modified Lentz
    /// everywhere else.
    public static func e1(_ z: Complex) -> Complex {
        let a0 = z.magnitude
        if a0 == 0 { return Complex(.infinity, 0) }
        if a0 <= 10 || (z.re < 0 && abs(z.im) <= 12 && a0 <= 60) {
            var total = Complex.one
            var cr = Complex.one
            for k in 1..<400 {
                cr = -cr * Double(k) * z / Double((k + 1) * (k + 1))
                total += cr
                if cr.magnitude <= total.magnitude * 1e-16 { break }
            }
            return -Gamma.euler - Complex.log(z) + z * total
        }
        let tiny = 1e-300
        var f = z
        var c = f
        var d = Complex.zero
        var k = 1
        for i in 1..<2000 {
            let a = Double(k)
            let b = i % 2 == 1 ? Complex.one : z
            d = b + a * d
            if d.re == 0 && d.im == 0 { d = Complex(tiny) }
            c = b + a / c
            if c.re == 0 && c.im == 0 { c = Complex(tiny) }
            d = 1 / d
            let delta = c * d
            f *= delta
            if i % 2 == 0 { k += 1 }
            if (delta - 1).magnitude < 1e-16 { break }
        }
        var result = Complex.exp(-z) / f
        if z.re <= 0 && z.im == 0 {
            result -= Complex(0, z.im.sign == .plus ? .pi : -.pi)
        }
        return result
    }

    /// `expi(z) = -E₁(-z)`, the language's continuation of Ei -- including
    /// the ∓iπ it carries on the positive real axis, which is what the Python
    /// side evaluates and what the fixtures pin.
    public static func expi(_ z: Complex) -> Complex {
        -e1(-z)
    }
}
