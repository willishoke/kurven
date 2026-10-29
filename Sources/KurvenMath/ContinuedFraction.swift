import Foundation

/// The continued fraction of a number, and the fractions it is nearly.
///
/// A winding at an irrational slope never closes, but it nearly closes at
/// every convergent `p/q` of the slope's continued fraction: after `q` turns
/// it is back within `1/q` of a turn of where it began. Those denominators
/// are where a trajectory drawn for `q` turns cuts its torus most evenly
/// (the three-gap theorem), and those fractions are the windings a
/// quasi-periodic orbit would lock to. Both are what a slider should rest on.
public enum ContinuedFraction {
    /// A fraction `p/q` in lowest terms.
    public struct Convergent: Equatable, Sendable {
        public let p: Int
        public let q: Int
        public var value: Double { Double(p) / Double(q) }
        public init(_ p: Int, _ q: Int) { self.p = p; self.q = q }
    }

    /// The partial quotients `[a₀; a₁, a₂, …]` of `x`, at most `count` of
    /// them, stopping when the remainder is zero to within `tolerance` -- a
    /// fraction's expansion ends -- or the next quotient would be
    /// meaningless at double precision.
    public static func quotients(of x: Double, count: Int = 12,
                                 tolerance: Double = 1e-12) -> [Int] {
        precondition(x.isFinite, "a continued fraction wants a finite number")
        var out: [Int] = []
        var r = x
        for _ in 0..<count {
            let a = r.rounded(.down)
            guard abs(a) < 1e15 else { break }
            out.append(Int(a))
            let frac = r - a
            guard frac > tolerance * max(1, abs(x)) else { break }
            r = 1 / frac
        }
        return out
    }

    /// The convergents of `x` with denominators up to `maxDenominator`,
    /// smallest first: `p/q = [a₀; a₁, …, aₖ]` for each `k`.
    public static func convergents(of x: Double, maxDenominator: Int = 1_000_000,
                                   tolerance: Double = 1e-12) -> [Convergent] {
        var out: [Convergent] = []
        var (p0, q0, p1, q1) = (1, 0, 0, 1)   // h₋₁/k₋₁ and h₋₂/k₋₂
        for a in quotients(of: x, count: 40, tolerance: tolerance) {
            let (p, q) = (a * p0 + p1, a * q0 + q1)
            guard q <= maxDenominator else { break }
            out.append(Convergent(p, q))
            (p1, q1, p0, q0) = (p0, q0, p, q)
        }
        return out
    }

    /// `[a₀; a₁, a₂, …]`, as it is written.
    public static func describe(_ quotients: [Int], ellipsis: Bool) -> String {
        guard let first = quotients.first else { return "[]" }
        let rest = quotients.dropFirst().map(String.init).joined(separator: ", ")
        return "[\(first)" + (rest.isEmpty ? "" : "; " + rest) + (ellipsis ? ", …" : "") + "]"
    }
}
