import simd

/// What hides a point of a curve beyond its own face's facing: anything of
/// the solid on its line of sight toward the eye.
///
/// Visibility asks two questions of a point (`docs/architecture.md`): does
/// the face under it face the eye, and is nothing in front of it. This is the
/// second, one test for each kind of solid, each exact, with no resolution
/// in it: grid traversal through a landscape (`GridTraversal`), sphere
/// tracing against the tubular neighbourhood of the round and the displaced
/// torus (`SphereTracing`), and closed-form occlusion by a polyhedron's flat
/// pieces (`Solid.Occlusion`, whose segment form stays the polyhedron's own
/// way of testing its edges and hatching). The coordinate image is not one:
/// it says whether a point is the front-most of its own surface, nothing
/// about any other solid, and only to its resolution
/// (`SurfaceVisibility.isFrontMost`).
///
/// A landscape's is public as `HeightfieldVisibility`, which asks the
/// traversal. Each scene's visibility asks its test as its own type
/// (`Scene.visible(_:depthTolerance:image:)`); the protocol is what a
/// composite asks each of its geometries through about another's curves
/// (`Composite`), and what the `occlusion:` suite holds each of them to a ray
/// cast by.
public protocol OcclusionTest: Sendable {
    /// Whether nothing of this solid stands on the open ray from `p` toward the eye.
    func isClear(_ p: P3<WorldSpace>) -> Bool
    /// The same of a point of another solid, which this one forgives
    /// nothing: `isClear` may forgive a point of its own surface for lying
    /// a rounding or a depth tolerance inside the solid, and a point of
    /// another solid inside this one is hidden by it.
    func isClear(ofAnotherSolid p: P3<WorldSpace>) -> Bool
}

extension GridTraversal: OcclusionTest {
    func isClear(_ p: P3<WorldSpace>) -> Bool { unoccluded(p) }
    func isClear(ofAnotherSolid p: P3<WorldSpace>) -> Bool { unoccluded(p, forgiven: false, elsewhere: true) }
}

/// Sphere tracing (Hart 1996): the occlusion test of a surface with a
/// tubular neighbourhood in closed form, the round torus and a torus
/// displaced along its normal.
public struct SphereTracing: OcclusionTest {
    public let neighbourhood: TubularNeighbourhood
    /// The direction lines of sight travel, in world space.
    public let sight: SIMD3<Double>

    /// The surface's sphere tracing along `sight`; nil where it has no
    /// tubular neighbourhood.
    public init?(_ surface: ParametricSurface, sight: SIMD3<Double>) {
        guard let neighbourhood = surface.neighbourhood else { return nil }
        self.neighbourhood = neighbourhood; self.sight = sight
    }

    /// Up the line of sight from `p`, stepping by the gap over the solid's
    /// Lipschitz constant, which cannot reach the solid, until the line has
    /// left the ball that holds it. Above the shell the gap is at least the
    /// height over it, and that is the step, with no normal offset asked. The
    /// first step is a small fraction of the solid's size: the curves lie on
    /// the surface to rounding, where the gap is zero and says nothing, and
    /// from a point facing the eye the line is outside from there on. On a
    /// fold the line is tangent, and the gap grows or shrinks with the
    /// surface's curvature along it: a convex fold is seen, a concave one
    /// lies behind the surface it folds into, and the first step decides
    /// which. A step shorter than that fraction is never taken, so an
    /// occluder thinner than it along the line can be stepped over; it is a
    /// hundred-thousandth of the solid, under any pixel.
    public func isClear(_ p: P3<WorldSpace>) -> Bool {
        let e = neighbourhood
        let q = p.v
        let d = -sight
        let o = q - e.centre
        let b = simd_dot(o, d), c = simd_dot(o, o) - e.radius * e.radius
        let disc = b * b - c
        guard disc > 0 else { return true }
        let exit = -b + disc.squareRoot()
        let eps = 1e-9 * e.radius, least = 1e-5 * e.radius
        var t = least
        while t < exit {
            let (coord, distance) = e.foot(q + t * d)
            let over = distance - e.above
            if over > least { t += over; continue }
            if distance < -e.below - eps { return false }
            let gap = distance - e.normalOffset(coord)
            if gap < -eps { return false }
            t += max(gap / e.lipschitz, least)
        }
        return true
    }

    /// Whether a point of another solid is clear: its margin is not
    /// negative (`margin(ofAnotherSolid:)`).
    public func isClear(ofAnotherSolid p: P3<WorldSpace>) -> Bool { margin(ofAnotherSolid: p) >= 0 }

    /// How far a point of another solid can move and keep its verdict, with
    /// the verdict in the sign: positive where its line of sight is clear,
    /// and every point within the margin of it has a clear line too;
    /// negative where the line passes through the solid, and every point
    /// within the margin's size of it has a line through the solid too.
    ///
    /// The line is traced from the point itself, not a step past it, in
    /// steps of half the gap over the Lipschitz constant: a step of half the
    /// gap leaves every point it passes at least half the gap from the
    /// solid, so the least half-gap is a distance no line within it can
    /// close. Past a ball a hundredth larger than the one that holds the
    /// solid, every point is that hundredth away. Where the line enters the
    /// solid the margin is the deepest the line goes, by the gap, which on
    /// the round torus is the distance to the tube and so exact inside too.
    /// A step is never under sphere tracing's least step, a hundred-thousandth
    /// of the ball, and a margin under it is a verdict to that resolution.
    public func margin(ofAnotherSolid p: P3<WorldSpace>) -> Double {
        let e = neighbourhood
        let q = p.v, d = -sight
        let lipschitz = max(e.lipschitz, 1), least = 1e-5 * e.radius
        func gap(_ x: SIMD3<Double>) -> Double {
            let (coord, distance) = e.foot(x)
            let over = distance - e.above
            return over > 0 ? over : (distance - e.normalOffset(coord)) / lipschitz
        }
        let o = q - e.centre, outer = 1.01 * e.radius
        let b = simd_dot(o, d), c = simd_dot(o, o) - outer * outer
        let disc = b * b - c
        guard disc > 0, -b + disc.squareRoot() > 0 else {
            return simd_length(o + max(-b, 0) * d) - e.radius
        }
        let exit = -b + disc.squareRoot()
        var t = max(-b - disc.squareRoot(), 0)
        var margin = outer - e.radius
        while t <= exit {
            let g = gap(q + t * d)
            if g < 0 {
                var deepest = -g, u = t
                while u <= exit {
                    let h = gap(q + u * d)
                    if h >= 0 { break }
                    deepest = max(deepest, -h)
                    u += max(-h, least)
                }
                return -deepest
            }
            margin = min(margin, 0.5 * g)
            t += max(0.5 * g, least)
        }
        return margin
    }
}

extension Solid.Occlusion: OcclusionTest {
    /// The point form: the segment form asked of the segment from `p` to
    /// itself, which a piece hides whole or not at all. A piece through `p`
    /// stands in front of it by nothing, and a piece seen edge-on hides
    /// nothing, as on a segment. Nothing draws with it: a polyhedron's
    /// edges and hatching are tested by the segment form, and so are the
    /// segments of another geometry's curves in a composite (`Composite`).
    /// The `occlusion:` suite holds it to a ray cast.
    public func isClear(_ p: P3<WorldSpace>) -> Bool {
        let q = view(p)
        return hidden(from: q, to: q, excluding: { _ in false }).isEmpty
    }

    /// The same test, which excludes no piece: a point inside the solid has
    /// one in front of it.
    public func isClear(ofAnotherSolid p: P3<WorldSpace>) -> Bool { isClear(p) }
}
