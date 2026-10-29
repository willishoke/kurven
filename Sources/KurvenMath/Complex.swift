import Foundation

/// A complex number in double precision, with the elementary functions on the
/// principal branches numpy uses.
///
/// The frontend evaluates the expression language itself, so every function
/// the language offers has to exist here, and has to agree with numpy and
/// scipy at every point of the plane -- including *on* the branch cuts, where
/// the sign of a zero imaginary part decides which side of the cut a value
/// comes from. The formulas below are written component by component where
/// that matters (`1 - z*z` is not `Complex(1 - (x*x - y*y), -2*x*y)` under
/// IEEE rules), and `tests/fixtures/expr` holds them to numpy's answers on a
/// grid that runs along the real axis at exactly `im = 0`.
public struct Complex: Hashable, Sendable, CustomStringConvertible {
    public var re: Double
    public var im: Double

    @inlinable public init(_ re: Double, _ im: Double = 0) { self.re = re; self.im = im }

    public static let zero = Complex(0, 0)
    public static let one = Complex(1, 0)
    public static let i = Complex(0, 1)

    public var description: String { "(\(re)\(im.sign == .minus ? "-" : "+")\(abs(im))j)" }

    @inlinable public var isFinite: Bool { re.isFinite && im.isFinite }
    @inlinable public var isNaN: Bool { re.isNaN || im.isNaN }
    @inlinable public var conjugate: Complex { Complex(re, -im) }
    /// |z|, without intermediate overflow.
    @inlinable public var magnitude: Double { hypot(re, im) }
    /// arg z in (-pi, pi], numpy's `angle`.
    @inlinable public var argument: Double { atan2(im, re) }
    @inlinable public var squaredMagnitude: Double { re * re + im * im }

    // MARK: arithmetic

    @inlinable public static prefix func - (z: Complex) -> Complex { Complex(-z.re, -z.im) }
    @inlinable public static func + (a: Complex, b: Complex) -> Complex { Complex(a.re + b.re, a.im + b.im) }
    @inlinable public static func - (a: Complex, b: Complex) -> Complex { Complex(a.re - b.re, a.im - b.im) }
    @inlinable public static func * (a: Complex, b: Complex) -> Complex {
        Complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
    }
    /// Smith's algorithm, as numpy divides.
    @inlinable public static func / (a: Complex, b: Complex) -> Complex {
        let absBr = abs(b.re), absBi = abs(b.im)
        if absBr >= absBi {
            if absBr == 0 && absBi == 0 {
                // x/0: numpy gives inf/nan components; the sampler pushes any
                // non-finite value to its ceiling, so the exact pattern is moot.
                return Complex(a.re / absBr, a.im / absBi)
            }
            let r = b.im / b.re
            let d = b.re + b.im * r
            return Complex((a.re + a.im * r) / d, (a.im - a.re * r) / d)
        }
        let r = b.re / b.im
        let d = b.re * r + b.im
        return Complex((a.re * r + a.im) / d, (a.im * r - a.re) / d)
    }

    @inlinable public static func + (a: Complex, b: Double) -> Complex { Complex(a.re + b, a.im) }
    @inlinable public static func + (a: Double, b: Complex) -> Complex { Complex(a + b.re, b.im) }
    @inlinable public static func - (a: Complex, b: Double) -> Complex { Complex(a.re - b, a.im) }
    @inlinable public static func - (a: Double, b: Complex) -> Complex { Complex(a - b.re, -b.im) }
    @inlinable public static func * (a: Complex, b: Double) -> Complex { Complex(a.re * b, a.im * b) }
    @inlinable public static func * (a: Double, b: Complex) -> Complex { Complex(a * b.re, a * b.im) }
    @inlinable public static func / (a: Complex, b: Double) -> Complex { Complex(a.re / b, a.im / b) }
    @inlinable public static func / (a: Double, b: Complex) -> Complex { Complex(a, 0) / b }

    @inlinable public static func += (a: inout Complex, b: Complex) { a = a + b }
    @inlinable public static func -= (a: inout Complex, b: Complex) { a = a - b }
    @inlinable public static func *= (a: inout Complex, b: Complex) { a = a * b }
    @inlinable public static func /= (a: inout Complex, b: Complex) { a = a / b }
    @inlinable public static func *= (a: inout Complex, b: Double) { a = a * b }
    @inlinable public static func /= (a: inout Complex, b: Double) { a = a / b }

    /// `i * z`, with the signed zeros `-imag(z) + i real(z)` carries -- the
    /// form the C library uses to reduce atan to atanh, and what keeps the two
    /// on the same side of their cuts.
    @inlinable public var timesI: Complex { Complex(-im, re) }
    @inlinable public var timesMinusI: Complex { Complex(im, -re) }

    // MARK: elementary functions

    public static func exp(_ z: Complex) -> Complex {
        let e = Foundation.exp(z.re)
        if z.im == 0 { return Complex(e, z.im) }
        return Complex(e * Foundation.cos(z.im), e * Foundation.sin(z.im))
    }

    public static func log(_ z: Complex) -> Complex {
        Complex(Foundation.log(z.magnitude), z.argument)
    }

    /// C99 `csqrt`: the principal branch, cut along the negative real axis,
    /// with the side decided by the sign of the zero.
    public static func sqrt(_ z: Complex) -> Complex {
        if z.re == 0 && z.im == 0 { return Complex(0, z.im) }
        let t = Foundation.sqrt((abs(z.re) + z.magnitude) / 2)
        if z.re >= 0 { return Complex(t, z.im / (2 * t)) }
        return Complex(abs(z.im) / (2 * t), copysign(t, z.im))
    }

    public static func sin(_ z: Complex) -> Complex {
        Complex(Foundation.sin(z.re) * Foundation.cosh(z.im),
                Foundation.cos(z.re) * Foundation.sinh(z.im))
    }

    public static func cos(_ z: Complex) -> Complex {
        Complex(Foundation.cos(z.re) * Foundation.cosh(z.im),
                -Foundation.sin(z.re) * Foundation.sinh(z.im))
    }

    public static func sinh(_ z: Complex) -> Complex {
        Complex(Foundation.sinh(z.re) * Foundation.cos(z.im),
                Foundation.cosh(z.re) * Foundation.sin(z.im))
    }

    public static func cosh(_ z: Complex) -> Complex {
        Complex(Foundation.cosh(z.re) * Foundation.cos(z.im),
                Foundation.sinh(z.re) * Foundation.sin(z.im))
    }

    /// `(sinh 2x + i sin 2y) / (cosh 2x + cos 2y)`, saturating far from the
    /// imaginary axis where cosh would overflow.
    public static func tanh(_ z: Complex) -> Complex {
        if abs(z.re) > 350 {
            return Complex(copysign(1, z.re),
                           copysign(0, Foundation.sin(2 * z.im)))
        }
        let d = Foundation.cosh(2 * z.re) + Foundation.cos(2 * z.im)
        return Complex(Foundation.sinh(2 * z.re) / d, Foundation.sin(2 * z.im) / d)
    }

    public static func tan(_ z: Complex) -> Complex {
        tanh(z.timesI).timesMinusI
    }

    /// `-i log(iz + sqrt(1 - z^2))`.
    public static func asin(_ z: Complex) -> Complex {
        let oneMinusSquare = Complex(1 - (z.re * z.re - z.im * z.im), -2 * z.re * z.im)
        return log(z.timesI + sqrt(oneMinusSquare)).timesMinusI
    }

    public static func acos(_ z: Complex) -> Complex {
        let a = asin(z)
        return Complex(.pi / 2 - a.re, -a.im)
    }

    public static func atanh(_ z: Complex) -> Complex {
        let plus = Complex(1 + z.re, z.im)
        let minus = Complex(1 - z.re, -z.im)
        let d = log(plus) - log(minus)
        return Complex(d.re / 2, d.im / 2)
    }

    public static func atan(_ z: Complex) -> Complex {
        atanh(z.timesI).timesMinusI
    }

    /// `log(z + sqrt(z^2 + 1))`.
    public static func asinh(_ z: Complex) -> Complex {
        let squarePlusOne = Complex(z.re * z.re - z.im * z.im + 1, 2 * z.re * z.im)
        return log(z + sqrt(squarePlusOne))
    }

    /// `log(z + sqrt(z + 1) sqrt(z - 1))` -- the product of two roots rather
    /// than the root of a product, which is what puts the cut on (-inf, 1).
    public static func acosh(_ z: Complex) -> Complex {
        log(z + sqrt(Complex(z.re + 1, z.im)) * sqrt(Complex(z.re - 1, z.im)))
    }

    /// `a^b`. A small integer power is repeated multiplication, as numpy does
    /// it, so `z^2` at a real z is exactly real rather than `exp(2 log z)`.
    public static func pow(_ a: Complex, _ b: Complex) -> Complex {
        if b.re == 0 && b.im == 0 { return .one }
        if a.re == 0 && a.im == 0 {
            if b.im == 0 && b.re > 0 { return .zero }
            return Complex(.infinity, 0)
        }
        if b.im == 0, b.re == b.re.rounded(), abs(b.re) < 100 {
            return integerPower(a, Int(b.re))
        }
        return exp(b * log(a))
    }

    static func integerPower(_ a: Complex, _ n: Int) -> Complex {
        if n < 0 { return .one / integerPower(a, -n) }
        var result = Complex.one
        var base = a
        var k = n
        while k > 0 {
            if k & 1 == 1 { result = result * base }
            k >>= 1
            if k > 0 { base = base * base }
        }
        return result
    }
}
