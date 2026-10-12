import Foundation

/// Zeros of a real function of one variable.
public enum Root {
    /// A zero of `f` between `a` and `b`, whose values `fa` and `fb` there
    /// differ in sign, by Brent's method (R. P. Brent, *Algorithms for
    /// Minimization without Derivatives*, 1973, ch. 4).
    ///
    /// Each step tries inverse quadratic interpolation through the last three
    /// points, or a secant step through the last two, and keeps it only when
    /// it lands inside the bracket and shrinks it fast enough; otherwise it
    /// bisects. So it converges wherever bisection would, and superlinearly
    /// where f is smooth. This is the iteration of scipy's `brentq`.
    ///
    /// Stops when the bracket is narrower than `xtol` plus a few ulps of the
    /// zero, or when |f| ≤ `ftol`. Returns nil when the ends do not bracket a
    /// zero, or when f is not finite at a point the iteration asks for; after
    /// `maxIterations` steps, the best point so far.
    public static func brent(_ f: (Double) -> Double,
                             _ a: Double, _ b: Double, fa: Double, fb: Double,
                             xtol: Double, ftol: Double = 0,
                             maxIterations: Int = 100) -> Double? {
        let rtol = 4 * Double.ulpOfOne
        var xpre = a, xcur = b, fpre = fa, fcur = fb
        guard fpre.isFinite, fcur.isFinite else { return nil }
        if fpre == 0 { return xpre }
        if fcur == 0 { return xcur }
        guard fpre.sign != fcur.sign else { return nil }
        // The other end of the bracket, and the last two step lengths.
        var xblk = 0.0, fblk = 0.0, spre = 0.0, scur = 0.0
        for _ in 0..<maxIterations {
            if fpre != 0, fcur != 0, fpre.sign != fcur.sign {
                xblk = xpre; fblk = fpre
                spre = xcur - xpre; scur = spre
            }
            // Keep xcur the better of the two ends.
            if abs(fblk) < abs(fcur) {
                xpre = xcur; xcur = xblk; xblk = xpre
                fpre = fcur; fcur = fblk; fblk = fpre
            }
            let delta = (xtol + rtol * abs(xcur)) / 2
            let sbis = (xblk - xcur) / 2
            if fcur == 0 || abs(sbis) < delta || abs(fcur) <= ftol { return xcur }
            if abs(spre) > delta, abs(fcur) < abs(fpre) {
                let stry: Double
                if xpre == xblk {
                    stry = -fcur * (xcur - xpre) / (fcur - fpre)
                } else {
                    let dpre = (fpre - fcur) / (xpre - xcur)
                    let dblk = (fblk - fcur) / (xblk - xcur)
                    stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
                }
                if 2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - delta) {
                    spre = scur; scur = stry
                } else {
                    spre = sbis; scur = sbis
                }
            } else {
                spre = sbis; scur = sbis
            }
            xpre = xcur; fpre = fcur
            xcur += abs(scur) > delta ? scur : (sbis > 0 ? delta : -delta)
            fcur = f(xcur)
            guard fcur.isFinite else { return nil }
        }
        return xcur
    }
}
