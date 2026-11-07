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
/// (`Scene.visible(_:depthTolerance:image:)`); the protocol is what a scene
/// of several solids will ask them through, and what the `occlusion:`
/// suite holds each of them to a ray cast by.
public protocol OcclusionTest: Sendable {
    /// Whether nothing of this solid stands on the open ray from `p` toward the eye.
    func isClear(_ p: P3<WorldSpace>) -> Bool
}

extension GridTraversal: OcclusionTest {
    func isClear(_ p: P3<WorldSpace>) -> Bool { unoccluded(p) }
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
}

extension Solid.Occlusion: OcclusionTest {
    /// The point form: the segment form asked of the segment from `p` to
    /// itself, which a piece hides whole or not at all. A piece through `p`
    /// stands in front of it by nothing, and a piece seen edge-on hides
    /// nothing, as on a segment. Nothing draws with it yet: a polyhedron's
    /// edges and hatching are tested by the segment form, and the point form
    /// is for a scene of several solids, held to a ray cast meanwhile by the
    /// `occlusion:` suite.
    public func isClear(_ p: P3<WorldSpace>) -> Bool {
        let q = view(p)
        return hidden(from: q, to: q, excluding: { _ in false }).isEmpty
    }
}
