import simd

/// Coordinate spaces, as uninhabited types.
///
/// The dominant historical bug class in kurven is the `(imag, real)` column
/// order: every Python module documents it and several commits exist only to
/// fix having forgotten it. Here the convention is a type. A point in the
/// complex plane and a point on the plate are different types, a transform
/// between them is a `Transform<A, B>`, and composing them in the wrong order
/// does not compile. The docstring is replaced by the compiler.
public enum DomainSpace: Sendable {}   // (real, imag) in the complex plane
public enum WorldSpace: Sendable {}    // (x = real, y = imag, z = |f|)
public enum ViewSpace: Sendable {}     // after the camera; z increases toward the viewer
public enum PlateSpace: Sendable {}    // 2D output, SVG units

/// A point in `S`.
public struct P2<S>: Sendable, Equatable, Hashable {
    public var v: SIMD2<Double>
    public init(_ v: SIMD2<Double>) { self.v = v }
    public init(_ x: Double, _ y: Double) { self.v = SIMD2(x, y) }
    public var x: Double { v.x }
    public var y: Double { v.y }
}

public struct P3<S>: Sendable, Equatable, Hashable {
    public var v: SIMD3<Double>
    public init(_ v: SIMD3<Double>) { self.v = v }
    public init(_ x: Double, _ y: Double, _ z: Double) { self.v = SIMD3(x, y, z) }
    public var x: Double { v.x }
    public var y: Double { v.y }
    public var z: Double { v.z }
    public var xy: P2<S> { P2(v.x, v.y) }
}

/// A linear-plus-translation map from `A` to `B`.
///
/// Stored column-major as `simd_double4x4` (the simd convention), applied as
/// `m * (p, 1)`. `>>>` reads left to right: `world >>> view` is "do the world
/// transform, then the view one", and the phantom parameters make the only
/// legal composition the one that type-checks.
public struct Transform<A, B>: Sendable, Equatable {
    public var m: simd_double4x4
    public init(_ m: simd_double4x4) { self.m = m }

    public static var identity: Transform<A, B> { Transform(matrix_identity_double4x4) }

    /// Build from a 3×3 linear part in row-major order, plus a translation.
    public init(rows: (SIMD3<Double>, SIMD3<Double>, SIMD3<Double>),
                translation t: SIMD3<Double> = .zero) {
        let (r0, r1, r2) = rows
        self.m = simd_double4x4(
            SIMD4(r0.x, r1.x, r2.x, 0),
            SIMD4(r0.y, r1.y, r2.y, 0),
            SIMD4(r0.z, r1.z, r2.z, 0),
            SIMD4(t.x, t.y, t.z, 1))
    }

    public func callAsFunction(_ p: P3<A>) -> P3<B> {
        let q = m * SIMD4(p.v.x, p.v.y, p.v.z, 1)
        return P3<B>(SIMD3(q.x, q.y, q.z))
    }

    public func apply(_ ps: [P3<A>]) -> [P3<B>] { ps.map { self($0) } }

    public var inverse: Transform<B, A> { Transform<B, A>(m.inverse) }

    public var float4x4: simd_float4x4 {
        simd_float4x4(SIMD4<Float>(m.columns.0), SIMD4<Float>(m.columns.1),
                      SIMD4<Float>(m.columns.2), SIMD4<Float>(m.columns.3))
    }
}

infix operator >>>: MultiplicationPrecedence

/// `a >>> b`: apply `a`, then `b`.
public func >>> <A, B, C>(a: Transform<A, B>, b: Transform<B, C>) -> Transform<A, C> {
    Transform<A, C>(b.m * a.m)
}

/// An axis-aligned bounding box in `S`. `nil`-free: an empty box is `nil`.
public struct AABB<S>: Sendable, Equatable {
    public var lo: SIMD3<Double>
    public var hi: SIMD3<Double>

    public init(lo: SIMD3<Double>, hi: SIMD3<Double>) { self.lo = lo; self.hi = hi }

    public init?(_ points: some Sequence<P3<S>>) {
        var lo = SIMD3<Double>(repeating: .infinity)
        var hi = SIMD3<Double>(repeating: -.infinity)
        var any = false
        for p in points {
            lo = simd_min(lo, p.v); hi = simd_max(hi, p.v); any = true
        }
        guard any else { return nil }
        self.lo = lo; self.hi = hi
    }

    public func union(_ other: AABB<S>) -> AABB<S> {
        AABB(lo: simd_min(lo, other.lo), hi: simd_max(hi, other.hi))
    }

    public var size: SIMD3<Double> { hi - lo }
    public var center: SIMD3<Double> { (lo + hi) / 2 }
}

/// Degrees, kept distinct from radians because the plate parameters are stated
/// in degrees and every conversion is a chance to lose a factor of pi.
public struct Angle: Sendable, Equatable, Hashable {
    public var radians: Double
    public init(radians: Double) { self.radians = radians }
    public init(degrees: Double) { self.radians = degrees * .pi / 180 }
    public var degrees: Double { radians * 180 / .pi }
}

public extension P3 {
    /// A point displaced by a world-space vector. Spelled out rather than
    /// operator-overloaded: a point plus a point is not a thing, and the
    /// phantom space is there to keep that from being expressible.
    init(target: P3<S>, plus delta: SIMD3<Double>) {
        self.init(target.v + delta)
    }
}
