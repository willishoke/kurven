//! What hides a point of a curve beyond its own face's facing: the port of
//! `Sources/KurvenCore/Occlusion.swift`.
//!
//! The landscape's test, the grid traversal, is whole. Sphere tracing is
//! ported whole too, ahead of Phase 4 (it reads only a
//! `TubularNeighbourhood`, whose type is here); nothing in Phase 3 builds a
//! surface with one, so it is held to the ray cast only when the torus comes.
//! A polyhedron's point form waits for Phase 5 with the rest of
//! `Solid.Occlusion`.

use kurven_math::swift;

use crate::parametric::{ParametricSurface, PortRefusal, TubularNeighbourhood};
use crate::simd::Vec3;
use crate::solid::Solid;
use crate::spaces::{P3, Transform, ViewSpace, WorldSpace};
use crate::traversal::GridTraversal;

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
/// (`SurfaceVisibility::is_front_most`).
///
/// A landscape's is public as `HeightfieldVisibility`, which asks the
/// traversal. Each scene's visibility asks its test as its own type
/// (`Scene::visible`); the trait is what a composite asks each of its
/// geometries through about another's curves (`Composite`), and what the
/// `occlusion:` suite holds each of them to a ray cast by.
pub trait OcclusionTest {
    /// Whether nothing of this solid stands on the open ray from `p` toward the eye.
    fn is_clear(&self, p: P3<WorldSpace>) -> bool;
    /// The same of a point of another solid, which this one forgives
    /// nothing: `is_clear` may forgive a point of its own surface for lying
    /// a rounding or a depth tolerance inside the solid, and a point of
    /// another solid inside this one is hidden by it.
    fn is_clear_of_another_solid(&self, p: P3<WorldSpace>) -> bool;
}

impl OcclusionTest for GridTraversal {
    fn is_clear(&self, p: P3<WorldSpace>) -> bool {
        self.unoccluded(p, true, false, None)
    }
    fn is_clear_of_another_solid(&self, p: P3<WorldSpace>) -> bool {
        self.unoccluded(p, false, true, None)
    }
}

/// Sphere tracing (Hart 1996): the occlusion test of a surface with a
/// tubular neighbourhood in closed form, the round torus and a torus
/// displaced along its normal.
#[derive(Debug, Clone)]
pub struct SphereTracing {
    pub neighbourhood: TubularNeighbourhood,
    /// The direction lines of sight travel, in world space.
    pub sight: Vec3,
}

impl SphereTracing {
    /// The surface's sphere tracing along `sight`; None where it has no
    /// tubular neighbourhood.
    pub fn new(surface: &ParametricSurface, sight: Vec3) -> Option<SphereTracing> {
        let neighbourhood = surface.neighbourhood.clone()?;
        Some(SphereTracing { neighbourhood, sight })
    }

    /// How far a point of another solid can move and keep its verdict, with
    /// the verdict in the sign: positive where its line of sight is clear,
    /// and every point within the clearance of it has a clear line too;
    /// negative where the line passes through the solid, and every point
    /// within the clearance's size of it has a line through the solid too.
    ///
    /// The line is traced from the point itself, not a step past it, in
    /// steps of half the gap over the Lipschitz constant: a step of half the
    /// gap leaves every point it passes at least half the gap from the
    /// solid, so the least half-gap is a distance no line within it can
    /// close. Past a ball a hundredth larger than the one that holds the
    /// solid, every point is that hundredth away. Where the line enters the
    /// solid the clearance is the deepest the line goes, by the gap, which on
    /// the round torus is the distance to the tube and so exact inside too.
    /// A step is never under sphere tracing's least step, a hundred-thousandth
    /// of the ball, and a clearance under it is a verdict to that resolution.
    pub fn clearance_of_another_solid(&self, p: P3<WorldSpace>) -> f64 {
        let e = &self.neighbourhood;
        let q = p.v;
        let d = -self.sight;
        let lipschitz = swift::max(e.lipschitz, 1.0);
        let least = 1e-5 * e.radius;
        let gap = |x: Vec3| -> f64 {
            let (coord, distance) = (e.foot)(x);
            let over = distance - e.above;
            if over > 0.0 { over } else { (distance - (e.normal_offset)(coord)) / lipschitz }
        };
        let o = q - e.centre;
        let outer = 1.01 * e.radius;
        let b = o.dot(d);
        let c = o.dot(o) - outer * outer;
        let disc = b * b - c;
        if !(disc > 0.0 && -b + disc.sqrt() > 0.0) {
            return (o + swift::max(-b, 0.0) * d).length() - e.radius;
        }
        let exit = -b + disc.sqrt();
        let mut t = swift::max(-b - disc.sqrt(), 0.0);
        let mut clearance = outer - e.radius;
        while t <= exit {
            let g = gap(q + t * d);
            if g < 0.0 {
                let mut deepest = -g;
                let mut u = t;
                while u <= exit {
                    let h = gap(q + u * d);
                    if h >= 0.0 {
                        break;
                    }
                    deepest = swift::max(deepest, -h);
                    u += swift::max(-h, least);
                }
                return -deepest;
            }
            clearance = swift::min(clearance, 0.5 * g);
            t += swift::max(0.5 * g, least);
        }
        clearance
    }
}

impl OcclusionTest for SphereTracing {
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
    fn is_clear(&self, p: P3<WorldSpace>) -> bool {
        let e = &self.neighbourhood;
        let q = p.v;
        let d = -self.sight;
        let o = q - e.centre;
        let b = o.dot(d);
        let c = o.dot(o) - e.radius * e.radius;
        let disc = b * b - c;
        if !(disc > 0.0) {
            return true;
        }
        let exit = -b + disc.sqrt();
        let eps = 1e-9 * e.radius;
        let least = 1e-5 * e.radius;
        let mut t = least;
        while t < exit {
            let (coord, distance) = (e.foot)(q + t * d);
            let over = distance - e.above;
            if over > least {
                t += over;
                continue;
            }
            if distance < -e.below - eps {
                return false;
            }
            let gap = distance - (e.normal_offset)(coord);
            if gap < -eps {
                return false;
            }
            t += swift::max(gap / e.lipschitz, least);
        }
        true
    }

    /// Whether a point of another solid is clear: its clearance is not
    /// negative (`clearance_of_another_solid`).
    fn is_clear_of_another_solid(&self, p: P3<WorldSpace>) -> bool {
        self.clearance_of_another_solid(p) >= 0.0
    }
}

/// Closed-form occlusion by a polyhedron's flat pieces: Swift's
/// `Solid.Occlusion`, declared with its fields for the types' sake.
///
/// Under an orthographic camera a segment and a piece both project to
/// the picture plane, their overlap along the segment is an interval,
/// and over it the difference of their view depths is linear: where that
/// difference says the piece is nearer, the segment is hidden. The hidden
/// intervals are subtracted from the segment, exactly.
#[derive(Debug, Clone)]
pub struct SolidOcclusion {
    #[allow(dead_code)] // read by the segment form, Phase 5's
    solid: Solid,
    pub view: Transform<WorldSpace, ViewSpace>,
    #[allow(dead_code)]
    projected: Vec<P3<ViewSpace>>,
    /// Twice each piece's signed area in projection; zero when seen
    /// edge-on, where it hides nothing.
    #[allow(dead_code)]
    area2: Vec<f64>,
    /// The solid's size in view space: the farthest any vertex lies from
    /// the centre of its bounds. The tolerances are fractions of it, so
    /// they are the same wherever the solid stands.
    #[allow(dead_code)]
    scale: f64,
    #[allow(dead_code)]
    eps: f64,
    /// The direction back toward the eye, in world space.
    pub to_eye: Vec3,
}

impl Solid {
    /// The solid's closed-form occlusion along `view` (`occlusion(view:)`).
    pub fn occlusion(&self, _view: &Transform<WorldSpace, ViewSpace>) -> Result<SolidOcclusion, PortRefusal> {
        // PORT: phase 5 -- Solid.Occlusion's init (the vertices projected, each piece's doubled signed area from the bounds' centre, the scale and eps), its segment form `hidden(from:to:excluding:)` and `visible`, and its conformance to OcclusionTest: the point form, the segment from p to itself (`hidden(from: q, to: q, excluding: { _ in false }).isEmpty`), the same for a point of another solid.
        Err(PortRefusal { phase: 5, what: "Solid.Occlusion" })
    }
}
