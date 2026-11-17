//! What hides a point of a curve beyond its own face's facing: the port of
//! `Sources/KurvenCore/Occlusion.swift`.
//!
//! Whole: the landscape's test, the grid traversal; sphere tracing (it reads
//! only a `TubularNeighbourhood`); and a polyhedron's closed-form occlusion (`Solid.Occlusion`, here
//! `SolidOcclusion`, which Swift nests in Solid.swift).

use std::sync::Arc;

use kurven_math::swift;

use crate::parametric::{ParametricSurface, TubularNeighbourhood};
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

/// Closed-form occlusion by a polyhedron's flat pieces (`Solid.Occlusion`):
/// the pieces projected to the picture plane, ready to test against, built
/// once per camera, then asked about any flat segment.
///
/// Under an orthographic camera a segment and a piece both project to
/// the picture plane, their overlap along the segment is an interval,
/// and over it the difference of their view depths is linear: where that
/// difference says the piece is nearer, the segment is hidden. The hidden
/// intervals are subtracted from the segment, exactly.
#[derive(Debug, Clone)]
pub struct SolidOcclusion {
    /// The solid's convex pieces, shared with it.
    pieces: Arc<Vec<Vec<usize>>>,
    pub view: Transform<WorldSpace, ViewSpace>,
    pub projected: Vec<P3<ViewSpace>>,
    /// Twice each piece's signed area in projection; zero when seen
    /// edge-on, where it hides nothing.
    area2: Vec<f64>,
    /// The solid's size in view space: the farthest any vertex lies from
    /// the centre of its bounds. The tolerances are fractions of it, so
    /// they are the same wherever the solid stands.
    pub scale: f64,
    pub eps: f64,
    /// The direction back toward the eye, in world space.
    pub to_eye: Vec3,
}

impl SolidOcclusion {
    pub fn new(solid: &Solid, view: Transform<WorldSpace, ViewSpace>) -> SolidOcclusion {
        let to_eye = -view.projection_direction();
        let projected: Vec<P3<ViewSpace>> = solid.vertices.iter().map(|&p| view.apply(p)).collect();
        let points: Vec<Vec3> = projected.iter().map(|p| p.v).collect();
        let lo = points.iter().fold(points[0], |a, &p| a.min(p));
        let hi = points.iter().fold(points[0], |a, &p| a.max(p));
        let centre = 0.5 * (lo + hi);
        let scale = points.iter().map(|&p| (p - centre).length()).reduce(swift::max).expect("a vertex");
        let eps = 1e-9 * scale;
        // From the centre, so a solid far from the origin is not measured
        // in the rounding of its distance from it.
        let area2 = solid
            .pieces
            .iter()
            .map(|piece| {
                let mut a = 0.0;
                for k in 0..piece.len() {
                    let p = points[piece[k]] - centre;
                    let q = points[piece[(k + 1) % piece.len()]] - centre;
                    a += p.x * q.y - q.x * p.y;
                }
                a
            })
            .collect();
        SolidOcclusion { pieces: solid.pieces.clone(), view, projected, area2, scale, eps, to_eye }
    }

    /// Whether piece `f` is seen edge-on: it projects to a line, of no
    /// area, and hides nothing and shows nothing.
    pub fn is_edge_on(&self, f: usize) -> bool {
        self.area2[f].abs() <= 1e-9 * self.scale * self.scale
    }

    /// The sub-intervals of `[0, 1]` along the segment `pa -> pb` (in
    /// view space) over which a piece not `excluding` stands in front:
    /// sorted and disjoint.
    pub fn hidden(&self, pa: P3<ViewSpace>, pb: P3<ViewSpace>, excluding: &dyn Fn(usize) -> bool) -> Vec<(f64, f64)> {
        let projected = &self.projected;
        let eps = self.eps;
        let mut hidden: Vec<(f64, f64)> = Vec::new();
        for (f, piece) in self.pieces.iter().enumerate() {
            if excluding(f) || self.is_edge_on(f) {
                continue;
            }
            let sign = if self.area2[f] > 0.0 { 1.0 } else { -1.0 };
            // The segment's overlap with the piece's projection: clipped
            // against each side's half-plane.
            let (mut lo, mut hi) = (0.0, 1.0);
            for k in 0..piece.len() {
                let p = projected[piece[k]];
                let q = projected[piece[(k + 1) % piece.len()]];
                let (ex, ey) = (q.x() - p.x(), q.y() - p.y());
                // Inside is to the left of p -> q when the polygon winds
                // positively, to the right otherwise.
                let f0 = sign * (ex * (pa.y() - p.y()) - ey * (pa.x() - p.x()));
                let f1 = sign * (ex * (pb.y() - p.y()) - ey * (pb.x() - p.x()));
                // f(t) = f0 + t (f1 - f0) must be >= 0.
                if f0 < 0.0 && f1 < 0.0 {
                    lo = 1.0;
                    hi = 0.0;
                    break;
                }
                if f0 < 0.0 {
                    lo = swift::max(lo, f0 / (f0 - f1));
                }
                if f1 < 0.0 {
                    hi = swift::min(hi, f0 / (f0 - f1));
                }
            }
            if !(lo < hi) {
                continue;
            }
            // The piece's view depth over the picture plane is affine:
            // z = z0 + (gx, gy) · (x - x0, y - y0), from the plane's
            // normal in view space. A piece can have straight corners,
            // so the normal is taken from its first corner and the first
            // two after it that are not in line with it.
            let o = projected[piece[0]].v;
            let mut n = Vec3::ZERO;
            for k in 1..piece.len() - 1 {
                let a = projected[piece[k]].v - o;
                let b = projected[piece[k + 1]].v - o;
                n = a.cross(b);
                if n.length() > 1e-9 * a.length() * b.length() {
                    break;
                }
            }
            if !(n.z.abs() > 1e-9 * self.scale * self.scale) {
                continue;
            }
            let gap = |t: f64| {
                let p = pa.v + t * (pb.v - pa.v);
                let z = o.z - (n.x * (p.x - o.x) + n.y * (p.y - o.y)) / n.z;
                z - p.z
            };
            // gap > eps means the piece is in front, and gap is linear.
            let (g0, g1) = (gap(lo), gap(hi));
            let (mut a, mut b) = (lo, hi);
            if g0 <= eps && g1 <= eps {
                continue;
            }
            if g0 <= eps {
                a = lo + (hi - lo) * (eps - g0) / (g1 - g0);
            }
            if g1 <= eps {
                b = lo + (hi - lo) * (eps - g0) / (g1 - g0);
            }
            if a < b {
                hidden.push((a, b));
            }
        }
        sort_by_start(&mut hidden);
        let mut merged: Vec<(f64, f64)> = Vec::new();
        for (a, b) in hidden {
            match merged.last_mut() {
                Some(last) if a <= last.1 => last.1 = swift::max(last.1, b),
                _ => merged.push((a, b)),
            }
        }
        merged
    }

    /// The complement: the spans of the segment that can be seen, as
    /// fractions of the way along it.
    pub fn visible(&self, pa: P3<ViewSpace>, pb: P3<ViewSpace>, excluding: &dyn Fn(usize) -> bool) -> Vec<(f64, f64)> {
        let mut t = 0.0;
        let mut spans = Vec::new();
        for (a, b) in self.hidden(pa, pb, excluding) {
            if a > t {
                spans.push((t, a));
            }
            t = swift::max(t, b);
        }
        if t < 1.0 {
            spans.push((t, 1.0));
        }
        spans
    }
}

/// Intervals sorted by their start, stably, as Swift's `sort { $0.0 < $1.0 }`
/// sorts them (ties keep their order).
pub fn sort_by_start(spans: &mut [(f64, f64)]) {
    spans.sort_by(|x, y| swift::ascending(x.0, y.0));
}

impl OcclusionTest for SolidOcclusion {
    /// The point form: the segment form asked of the segment from `p` to
    /// itself, which a piece hides whole or not at all. A piece through `p`
    /// stands in front of it by nothing, and a piece seen edge-on hides
    /// nothing, as on a segment. Nothing draws with it: a polyhedron's
    /// edges and hatching are tested by the segment form, and so are the
    /// segments of another geometry's curves in a composite (`Composite`).
    /// The `occlusion:` suite holds it to a ray cast.
    fn is_clear(&self, p: P3<WorldSpace>) -> bool {
        let q = self.view.apply(p);
        self.hidden(q, q, &|_| false).is_empty()
    }

    /// The same test, which excludes no piece: a point inside the solid has
    /// one in front of it.
    fn is_clear_of_another_solid(&self, p: P3<WorldSpace>) -> bool {
        self.is_clear(p)
    }
}
