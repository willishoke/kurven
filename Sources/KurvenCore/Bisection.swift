import simd

/// Bisection to the last bit: the bracket from `a`, where `holds` is true,
/// to `b`, where it is not, halved sixty times at its midpoint
/// 0.5 * (a + b), each time keeping the half whose ends still disagree.
/// Sixty halvings take a bracket a grid cell wide past the last bit of its
/// ends. The caller checks that the ends disagree; the answer is the final
/// bracket, `holds` true at its first end and false at its second, and most
/// callers take its midpoint.
///
/// A sign change is a `holds` that compares with the sign at `a`:
/// `bisect(a, b) { (f($0) > 0) == (f(a) > 0) }`, with f(a) computed once.
@inlinable
public func bisect(_ a: Double, _ b: Double,
                   holds: (Double) -> Bool) -> (holds: Double, fails: Double) {
    bisect(a, b, midpoint: { 0.5 * ($0 + $1) }, holds: holds)
}

/// `bisect(_:_:holds:)` along the segment between two points.
@inlinable
public func bisect(_ a: SIMD2<Double>, _ b: SIMD2<Double>,
                   holds: (SIMD2<Double>) -> Bool) -> (holds: SIMD2<Double>, fails: SIMD2<Double>) {
    bisect(a, b, midpoint: { 0.5 * ($0 + $1) }, holds: holds)
}

@inlinable
func bisect<V>(_ a: V, _ b: V, midpoint: (V, V) -> V,
               holds: (V) -> Bool) -> (holds: V, fails: V) {
    var (a, b) = (a, b)
    for _ in 0..<60 {
        let m = midpoint(a, b)
        if holds(m) { a = m } else { b = m }
    }
    return (a, b)
}
