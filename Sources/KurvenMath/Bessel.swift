import Foundation

/// Bessel functions of real order and complex argument: J, Y, I, K, and the
/// Airy functions through them.
///
/// Two engines, each used only where it is accurate, and each holding scipy's
/// AMOS-based answers to about 1e-11 across the windows the landscapes use
/// (`tests/fixtures/expr` pins the numbers):
///
/// - **J and Y** (`cylinder`): the power series under |z| = 12 or when the
///   order exceeds the argument -- its cancellation is e^(|z| - |Im z|),
///   tempered by 1/Γ(ν + k + 1) for large orders -- and otherwise the Hankel
///   expansions at the fractional part of the order, carried up by the
///   three-term recurrence in the one direction each solution is stable in.
///   Everything is reduced to the first quadrant first: conjugation for the
///   lower half-plane, the continuation formulas for the left.
/// - **I and K** (`modified`): Temme's series under |z| = 2 and Steed's
///   continued fraction beyond, on Re z > 0 (Thompson and Barnett 1987, the
///   complex form of Numerical Recipes' `bessik`), with the continuation
///   formulas for the left half-plane and the J/Y engine on the imaginary
///   axis itself, where the fraction cannot converge.
public enum Bessel {
    static let eps = 2.220446049250313e-16
    static let tiny = 1e-300

    public struct Cylinder: Sendable {
        public var j, y, h1, h2: Complex
    }

    // MARK: J and Y

    /// Σ (-1)^k (z/2)^{2k+ν} / (k! Γ(ν+k+1)); ν not a negative integer.
    static func jSeries(_ nu: Double, _ z: Complex) -> Complex {
        let half = z / 2
        var term = Complex.pow(half, Complex(nu)) * Gamma.rgamma(Complex(nu + 1))
        var total = term
        let q = -(half * half)
        var k = 0
        while true {
            k += 1
            term *= q / (Double(k) * (nu + Double(k)))
            total += term
            if term.magnitude <= eps * total.magnitude || k > 500 { break }
        }
        return total
    }

    static func harmonic(_ n: Int) -> Double {
        var h = 0.0
        if n > 0 { for k in 1...n { h += 1.0 / Double(k) } }
        return h
    }

    static func factorial(_ n: Int) -> Double {
        var f = 1.0
        if n > 1 { for k in 2...n { f *= Double(k) } }
        return f
    }

    /// Y_n for integer n >= 0, DLMF 10.8.1.
    static func ySeriesInteger(_ n: Int, _ z: Complex) -> Complex {
        let half = z / 2
        let h2 = half * half
        var total = Complex.zero
        if n > 0 {
            var p = Complex.pow(half, Complex(Double(-n)))
            for k in 0..<n {
                total -= factorial(n - k - 1) / factorial(k) * p
                p *= h2
            }
        }
        total /= Double.pi
        let jn = jSeries(Double(n), z)
        total += (2 / Double.pi) * Complex.log(half) * jn
        var p = Complex.pow(half, Complex(Double(n))) / factorial(n)
        var k = 0
        var s = Complex.zero
        while true {
            let psi = (harmonic(k) - Gamma.euler) + (harmonic(n + k) - Gamma.euler)
            let d = psi * p
            s += d
            if (d.magnitude <= eps * s.magnitude && k > 2) || k > 500 { break }
            k += 1
            p *= -h2 / (Double(k) * Double(n + k))
        }
        return total - s / Double.pi
    }

    /// The Hankel expansions, summed to their smallest term.
    static func hankelAsymptotic(_ nu: Double, _ z: Complex) -> (h1: Complex, h2: Complex) {
        let mu = 4 * nu * nu
        var s1 = Complex.one
        var s2 = Complex.one
        var term = Complex.one
        var ik = Complex.one
        var smallest = 1.0
        var k = 0
        while k < 200 {
            k += 1
            let odd = Double(2 * k - 1)
            term *= (mu - odd * odd) / (8 * Double(k) * z)
            ik = ik.timesI
            let mag = term.magnitude
            if mag > smallest { break }
            smallest = mag
            s1 += ik * term
            s2 += ik.conjugate * term
            if mag < 1e-15 { break }
        }
        let omega = z - nu * .pi / 2 - .pi / 4
        let pref = Complex.sqrt(2 / (.pi * z))
        let e = Complex.exp(omega.timesI)
        return (pref * e * s1, pref * s2 / e)
    }

    /// J, Y, H1, H2 for ν >= 0 in the closed first quadrant, z != 0.
    static func cylinderFirstQuadrant(_ nu: Double, _ z: Complex) -> Cylinder {
        let a = z.magnitude
        if a >= 12 && nu <= a {
            let nl = Int(nu)
            let mu = nu - Double(nl)
            var (h1a, h2a) = hankelAsymptotic(mu, z)
            if nl == 0 {
                return Cylinder(j: (h1a + h2a) / 2, y: ((h1a - h2a) / 2).timesMinusI,
                                h1: h1a, h2: h2a)
            }
            var (h1b, _) = hankelAsymptotic(mu + 1, z)
            let jmu = (h1a + h2a) / 2
            // H1 forward: the dominant solution below |z| in the upper
            // half-plane, and dominant above it everywhere.
            for k in 1...nl {
                (h1a, h1b) = (h1b, (2 * (mu + Double(k)) / z) * h1b - h1a)
            }
            // J by Miller's backward recurrence, scaled to the accurate J_mu.
            let n = max(nl, Int(a)) + 40
            var jp = Complex.zero
            var jc = Complex(tiny)
            var jnu = Complex.zero
            for k in stride(from: n, through: 1, by: -1) {
                let jm = (2 * (mu + Double(k)) / z) * jc - jp
                (jp, jc) = (jc, jm)
                if k - 1 == nl { jnu = jc }
                if jc.magnitude > 1e200 {
                    jp /= 1e200; jc /= 1e200; jnu /= 1e200
                }
            }
            jnu *= jmu / jc
            let y = (h1a - jnu).timesMinusI
            return Cylinder(j: jnu, y: y, h1: h1a, h2: 2 * jnu - h1a)
        }
        let j = jSeries(nu, z)
        let n = nu.rounded()
        let y: Complex
        if nu == n {
            y = ySeriesInteger(Int(n), z)
        } else {
            let jm = jSeries(-nu, z)
            y = (j * Trig.cosPi(nu) - jm) / Trig.sinPi(nu)
        }
        return Cylinder(j: j, y: y, h1: j + y.timesI, h2: j - y.timesI)
    }

    /// J_ν, Y_ν and the two Hankel functions anywhere on the principal sheet.
    public static func cylinder(_ nu: Double, _ z: Complex) -> Cylinder {
        if z.re == 0 && z.im == 0 {
            let j: Complex
            if nu == 0 { j = .one }
            else if nu > 0 || nu == nu.rounded() { j = .zero }
            else { j = Complex(.infinity, 0) }
            let inf = Complex(-.infinity, 0)
            return Cylinder(j: j, y: inf, h1: inf, h2: Complex(.infinity, 0))
        }
        if nu < 0 {
            // J_{-ν} = cos(νπ) J_ν - sin(νπ) Y_ν ; Y_{-ν} = sin(νπ) J_ν + cos(νπ) Y_ν.
            let c = cylinder(-nu, z)
            let co = Trig.cosPi(-nu), si = Trig.sinPi(-nu)
            let j = co * c.j - si * c.y
            let y = si * c.j + co * c.y
            return Cylinder(j: j, y: y, h1: j + y.timesI, h2: j - y.timesI)
        }
        if z.re < 0 {
            // z = z' e^{±iπ}: DLMF 10.11.
            let zr = -z
            let s: Double = z.im.sign == .minus ? -1 : 1
            let c = cylinder(nu, zr)
            let co = Trig.cosPi(nu)
            let rot = Complex.exp(Complex(0, s * nu * .pi))
            let j = rot * c.j
            let y = c.y / rot + (2 * s * co) * c.j.timesI
            let h1: Complex, h2: Complex
            if s > 0 {
                h1 = -(c.h2 / rot)
                h2 = rot * c.h1 + (2 * co) * c.h2
            } else {
                h2 = -(c.h1 / rot)
                h1 = rot * c.h2 + (2 * co) * c.h1
            }
            return Cylinder(j: j, y: y, h1: h1, h2: h2)
        }
        if z.im.sign == .minus {
            let c = cylinderFirstQuadrant(nu, z.conjugate)
            return Cylinder(j: c.j.conjugate, y: c.y.conjugate,
                            h1: c.h2.conjugate, h2: c.h1.conjugate)
        }
        return cylinderFirstQuadrant(nu, z)
    }

    public static func j(_ nu: Double, _ z: Complex) -> Complex { cylinder(nu, z).j }
    public static func y(_ nu: Double, _ z: Complex) -> Complex { cylinder(nu, z).y }

    // MARK: I and K

    /// `(1/Γ(1-μ) - 1/Γ(1+μ))/(2μ)` and `(1/Γ(1-μ) + 1/Γ(1+μ))/2`, plus the
    /// two reciprocals; the first by its Taylor series near μ = 0, where the
    /// quotient would cancel.
    static func gammaPair(_ mu: Double) -> (g1: Double, g2: Double, gp: Double, gm: Double) {
        let gp = Gamma.rgamma(Complex(1 + mu)).re
        let gm = Gamma.rgamma(Complex(1 - mu)).re
        let g1: Double
        if abs(mu) < 1e-3 {
            let c4 = -0.0420026350340952
            g1 = -Gamma.euler - c4 * mu * mu
        } else {
            g1 = (gm - gp) / (2 * mu)
        }
        return (g1, (gm + gp) / 2, gp, gm)
    }

    /// I_ν, K_ν for ν >= 0 and Re z > 0.
    static func modifiedRightHalf(_ nu: Double, _ z: Complex) -> (i: Complex, k: Complex) {
        let maxit = 20000
        let nl = Int(nu + 0.5)
        let mu = nu - Double(nl)
        let mu2 = mu * mu
        let xi = 1 / z
        let xi2 = 2 * xi

        // CF1 for I'_ν / I_ν by modified Lentz.
        var b = xi2 * nu
        var h = nu * xi
        if h.magnitude < tiny { h = Complex(tiny) }
        var d = Complex.zero
        var c = h
        for _ in 1...maxit {
            b += xi2
            let bd = b + d
            d = (bd.re == 0 && bd.im == 0) ? Complex(1 / tiny) : 1 / bd
            c = b + 1 / c
            let delta = c * d
            h *= delta
            if (delta - 1).magnitude < eps { break }
        }
        var ril = Complex(tiny)
        var ripl = h * ril
        let ril1 = ril
        var fact = nu * xi
        if nl > 0 {
            for _ in 1...nl {
                let ritemp = fact * ril + ripl
                fact -= xi
                ripl = fact * ritemp + ril
                ril = ritemp
            }
        }
        let f = ripl / ril

        var rkmu: Complex
        var rk1: Complex
        if z.magnitude < 2 {
            // Temme's series.
            let x2 = z / 2
            let pimu = Double.pi * mu
            let fact1 = abs(pimu) < eps ? 1 : pimu / sin(pimu)
            let dlog = -Complex.log(x2)
            var e = mu * dlog
            let fact2 = e.magnitude < eps ? Complex.one : Complex.sinh(e) / e
            let g = gammaPair(mu)
            var ff = fact1 * (g.g1 * Complex.cosh(e) + g.g2 * fact2 * dlog)
            var total = ff
            e = Complex.exp(e)
            var p = 0.5 * e / g.gp
            var q = 0.5 / (e * g.gm)
            var cc = Complex.one
            let dd = x2 * x2
            var sum1 = p
            var i = 0
            while true {
                i += 1
                let di = Double(i)
                ff = (di * ff + p + q) / (di * di - mu2)
                cc *= dd / di
                p /= (di - mu)
                q /= (di + mu)
                let delta = cc * ff
                total += delta
                sum1 += cc * (p - di * ff)
                if delta.magnitude < total.magnitude * eps || i > maxit { break }
            }
            rkmu = total
            rk1 = sum1 * xi2
        } else {
            // Steed's CF2.
            var bb = 2 * (1 + z)
            var dd = 1 / bb
            var hh = dd
            var delh = dd
            var q1 = Complex.zero
            var q2 = Complex.one
            let a1 = 0.25 - mu2
            var q = Complex(a1)
            var cc = Complex(a1)
            var a = -a1
            var s = 1 + q * delh
            for i in 2...maxit {
                a -= Double(2 * (i - 1))
                cc = -a * cc / Double(i)
                let qnew = (q1 - bb * q2) / a
                q1 = q2
                q2 = qnew
                q += cc * qnew
                bb += Complex(2)
                dd = 1 / (bb + a * dd)
                delh = (bb * dd - 1) * delh
                hh += delh
                let dels = q * delh
                s += dels
                if dels.magnitude < s.magnitude * eps { break }
            }
            hh = a1 * hh
            rkmu = Complex.sqrt(.pi / (2 * z)) * Complex.exp(-z) / s
            rk1 = rkmu * (mu + z + 0.5 - hh) * xi
        }
        // Wronskian: I_μ = (1/z) / (K_{μ+1} + K_μ (f - μ/z)).
        let rimu = xi / (rk1 + rkmu * (f - mu * xi))
        let inu = rimu * (ril1 / ril)
        var rkmup = rkmu
        var rkp = rk1
        if nl > 0 {
            for i in 1...nl {
                let rktemp = (mu + Double(i)) * xi2 * rkp + rkmup
                rkmup = rkp
                rkp = rktemp
            }
        }
        return (inu, rkmup)
    }

    /// I and K on and near the imaginary axis, through J and Y:
    /// DLMF 10.27.6 and 10.27.8.
    static func modifiedViaCylinder(_ nu: Double, _ z: Complex) -> (i: Complex, k: Complex) {
        if !(z.re < 0 && z.im > 0) {
            let c = cylinder(nu, z.timesI)
            let rot = Complex.exp(Complex(0, -nu * .pi / 2))
            return (rot * c.j, (Complex(0, .pi / 2) / rot) * c.h1)
        }
        let c = cylinder(nu, z.timesMinusI)
        let rot = Complex.exp(Complex(0, nu * .pi / 2))
        return (rot * c.j, -(Complex(0, .pi / 2) / rot) * c.h2)
    }

    /// I_ν and K_ν anywhere on the principal sheet.
    public static func modified(_ nu: Double, _ z: Complex) -> (i: Complex, k: Complex) {
        if z.re == 0 && z.im == 0 {
            let i: Complex
            if nu == 0 { i = .one }
            else if nu > 0 || nu == nu.rounded() { i = .zero }
            else { i = Complex(.infinity, 0) }
            return (i, Complex(.infinity, 0))
        }
        if nu < 0 {
            // I_{-ν} = I_ν + (2/π) sin(νπ) K_ν ; K_{-ν} = K_ν.
            let m = modified(-nu, z)
            return (m.i + (2 / Double.pi) * Trig.sinPi(-nu) * m.k, m.k)
        }
        if z.re == 0 { return modifiedViaCylinder(nu, z) }
        if z.re < 0 {
            // I(z' e^{±iπ}) = e^{±iνπ} I(z') ; K(z' e^{±iπ}) = e^{∓iνπ} K(z') ∓ iπ I(z').
            let zr = -z
            let s: Double = z.im.sign == .minus ? -1 : 1
            let m = modified(nu, zr)
            let rot = Complex.exp(Complex(0, s * nu * .pi))
            return (rot * m.i, m.k / rot - Complex(0, s * .pi) * m.i)
        }
        if abs(z.re) < 0.25 * z.magnitude && z.magnitude >= 2 {
            return modifiedViaCylinder(nu, z)
        }
        return modifiedRightHalf(nu, z)
    }

    public static func i(_ nu: Double, _ z: Complex) -> Complex { modified(nu, z).i }
    public static func k(_ nu: Double, _ z: Complex) -> Complex { modified(nu, z).k }
}

/// Ai and Bi: the Maclaurin series inside |z| = 2, the Bessel forms
/// (DLMF 9.6) for |arg z| < 2π/3, and the rotation identities (9.2.10,
/// 9.2.12) in the remaining sector, where both rotated points fall back into
/// the Bessel one.
public enum Airy {
    static let ai0 = 0.3550280538878172
    static let aip0 = -0.2588194037928068
    static let bi0 = 0.6149266274460007
    static let bip0 = 0.4482883573538264
    static let omega = Complex.exp(Complex(0, 2 * .pi / 3))

    static func series(_ z: Complex) -> (ai: Complex, bi: Complex) {
        let z3 = z * z * z
        var f = Complex.one
        var g = z
        var tf = Complex.one
        var tg = z
        var k = 0
        while true {
            k += 1
            tf *= z3 / Double((3 * k - 1) * (3 * k))
            tg *= z3 / Double((3 * k) * (3 * k + 1))
            f += tf
            g += tg
            if tf.magnitude + tg.magnitude <= 2.220446049250313e-16 * (f.magnitude + g.magnitude)
                || k > 200 { break }
        }
        return (ai0 * f + aip0 * g, bi0 * f + bip0 * g)
    }

    static func bessel(_ z: Complex) -> (ai: Complex, bi: Complex) {
        let zeta = (2.0 / 3.0) * Complex.pow(z, Complex(1.5))
        let root = Complex.sqrt(z / 3)
        let k = Bessel.k(1.0 / 3.0, zeta)
        let ip = Bessel.i(1.0 / 3.0, zeta)
        let im = Bessel.i(-1.0 / 3.0, zeta)
        return (root * k / Double.pi, root * (im + ip))
    }

    public static func airy(_ z: Complex) -> (ai: Complex, bi: Complex) {
        if z.magnitude <= 2 { return series(z) }
        if abs(z.argument) < 2 * .pi / 3 - 1e-9 { return bessel(z) }
        let a1 = bessel(omega * z).ai
        let a2 = bessel(z / omega).ai
        let ai = -(omega * a1 + a2 / omega)
        let bi = Complex.exp(Complex(0, .pi / 6)) * a1 + Complex.exp(Complex(0, -.pi / 6)) * a2
        return (ai, bi)
    }

    public static func ai(_ z: Complex) -> Complex { airy(z).ai }
    public static func bi(_ z: Complex) -> Complex { airy(z).bi }
}
