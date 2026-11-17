//! What the surface suites share: Surfaces.swift's analytic answers for the
//! round torus (`torusSees`, `realRoots`, `torusSeesFromOutside`), its test
//! cameras and view bounds, and `compareWithRayCast`.

#![allow(dead_code)]

use kurven_core::camera::{rotation_x, rotation_z};
use kurven_core::depth::DepthImage;
use kurven_core::parametric::ParametricSurface;
use kurven_core::simd::{Mat3, Vec2, Vec3};
use kurven_core::spaces::{AABB, Angle, P2, P3, ParamSpace, Transform, ViewSpace, WorldSpace};
use kurven_core::splitmix::SplitMix;
use kurven_core::visibility::SurfaceVisibility;
use kurven_math::{real, swift};

/// Whether a point of the torus of revolution `(R, r)` about z is seen from
/// direction `eye` (unit, pointing from the point toward the viewer).
///
/// Along `p + s e` the torus is `F = (|x|² + R² − r²)² − 4R²(x² + y²) = 0`,
/// a monic quartic in `s` whose constant term vanishes because `p` is on the
/// torus. Dividing by `s` leaves a cubic `G`; the point is hidden exactly when
/// `G` has a root at some `s > 0` inside the torus's bounding sphere.
pub fn torus_sees(p: Vec3, big_r: f64, r: f64, e: Vec3) -> bool {
    let b = 2.0 * p.dot(e);
    let c = p.length_squared() + big_r * big_r - r * r;
    let a2 = e.x * e.x + e.y * e.y;
    let b2 = 2.0 * (p.x * e.x + p.y * e.y);
    let k2 = b * b + 2.0 * c - 4.0 * big_r * big_r * a2;
    let k1 = 2.0 * b * c - 4.0 * big_r * big_r * b2;
    let g = |s: f64| ((s + 2.0 * b) * s + k2) * s + k1;
    let eps = 1e-9 * (big_r + r);
    let far = 2.5 * (big_r + r);
    // G' = 3s² + 4Bs + k2: its real roots split [eps, far] into monotone
    // pieces, and a monotone piece holds a root exactly when its ends differ
    // in sign.
    let mut cuts = vec![eps];
    let disc = 16.0 * b * b - 12.0 * k2;
    if disc > 0.0 {
        for s in [(-4.0 * b - real::sqrt(disc)) / 6.0, (-4.0 * b + real::sqrt(disc)) / 6.0] {
            if s > eps && s < far {
                cuts.push(s);
            }
        }
    }
    cuts.push(far);
    cuts.sort_by(|a, b| real::total_order(*a, *b));
    for w in cuts.windows(2) {
        let (ga, gb) = (g(w[0]), g(w[1]));
        if ga == 0.0 || gb == 0.0 || (ga > 0.0) != (gb > 0.0) {
            return false;
        }
    }
    true
}

/// The real roots of a polynomial in `[a, b]`, coefficients highest first.
pub fn real_roots(c: &[f64], a: f64, b: f64) -> Vec<f64> {
    let n = c.len() - 1;
    let p = |x: f64| c.iter().fold(0.0, |acc, &k| acc * x + k);
    if n < 1 {
        return Vec::new();
    }
    if n == 1 {
        let x = -c[1] / c[0];
        return if x >= a && x <= b { vec![x] } else { Vec::new() };
    }
    let derivative: Vec<f64> = (0..n).map(|i| c[i] * (n - i) as f64).collect();
    let mut cuts = vec![a];
    cuts.extend(real_roots(&derivative, a, b));
    cuts.push(b);
    let mut roots = Vec::new();
    for w in cuts.windows(2) {
        let (mut l, mut h) = (w[0], w[1]);
        let (pl, ph) = (p(l), p(h));
        if pl == 0.0 {
            roots.push(l);
            continue;
        }
        if !((pl > 0.0) != (ph > 0.0) || ph == 0.0) {
            continue;
        }
        for _ in 0..200 {
            if !(h - l > 0.0) {
                break;
            }
            let m = 0.5 * (l + h);
            if (p(m) > 0.0) == (pl > 0.0) {
                l = m;
            } else {
                h = m;
            }
        }
        roots.push(0.5 * (l + h));
    }
    roots
}

/// Whether a point *off* the torus, just outside it, is seen from `eye`: the
/// full quartic along the line of sight, which has no root at the point itself.
pub fn torus_sees_from_outside(p: Vec3, big_r: f64, r: f64, e: Vec3) -> bool {
    let b = 2.0 * p.dot(e);
    let c = p.length_squared() + big_r * big_r - r * r;
    let a2 = e.x * e.x + e.y * e.y;
    let b2 = 2.0 * (p.x * e.x + p.y * e.y);
    let c2 = p.x * p.x + p.y * p.y;
    let quartic = [
        1.0,
        2.0 * b,
        b * b + 2.0 * c - 4.0 * big_r * big_r * a2,
        2.0 * b * c - 4.0 * big_r * big_r * b2,
        c * c - 4.0 * big_r * big_r * c2,
    ];
    assert!(quartic[4] > 0.0, "the point is not outside the torus");
    real_roots(&quartic, 0.0, 2.5 * (big_r + r) + p.length()).is_empty()
}

/// Elevation and azimuth as the oblique camera have them, with an optional
/// oblique shear in front, so the view is affine but not orthonormal.
pub fn test_camera(elevation: f64, azimuth: f64, shear: f64) -> Transform<WorldSpace, ViewSpace> {
    let s = Mat3::from_rows(Vec3::new(1.0, shear, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0));
    let m = rotation_x(Angle::from_degrees(elevation)) * rotation_z(Angle::from_degrees(azimuth)) * s;
    Transform::from_rows((m.row(0), m.row(1), m.row(2)), Vec3::ZERO)
}

/// The view-space bounds of a surface's grid.
pub fn view_bounds(s: &ParametricSurface, view: &Transform<WorldSpace, ViewSpace>) -> AABB<ViewSpace> {
    let points: Vec<P3<ViewSpace>> = s.positions.iter().map(|&p| view.apply(p)).collect();
    AABB::of(&points).expect("a surface with a grid")
}

/// The depth under a view point (`DepthImage.depth(under:)`).
pub fn depth_under(depth: &DepthImage, p: P2<ViewSpace>) -> f64 {
    depth.depth(p)
}

/// Hold a visibility test to the exact ray-cast at 20000 seeded points of the
/// torus `(R, r)`: (wrong, unexplained, depth test wrong, an example).
///
/// A disagreement is excused only within two pixels of a place where the
/// truth itself changes -- a fold, or the edge of something in front -- found
/// by asking the ray-cast at the eight neighbours two pixels away in each
/// parameter direction. The depth test at a 1% depth tolerance is scored on the same
/// points, for comparison.
pub fn compare_with_ray_cast(
    vis: &SurfaceVisibility,
    big_r: f64,
    r: f64,
    view: &Transform<WorldSpace, ViewSpace>,
) -> (usize, usize, usize, String) {
    let surface = &vis.surface;
    let image = vis.image.as_ref().expect("an image");
    let frame = image.frame();
    let eye = -vis.sight;
    let unit = frame.units_per_pixel();
    let px = swift::max(unit.x, unit.y);
    let mut rng = SplitMix::new(0x5EED);
    let (mut wrong, mut unexplained, mut depth_wrong) = (0, 0, 0);
    let mut example = String::new();
    let two_pi = 2.0 * core::f64::consts::PI;
    for _ in 0..20_000 {
        let c = P2::<ParamSpace>::new(rng.next(0.0, two_pi), rng.next(0.0, two_pi));
        let jet = (surface.map)(c);
        let q = view.apply(P3::from_vec(jet.position));
        let truth = torus_sees(jet.position, big_r, r, eye);
        if (q.z() + 0.01 * (big_r + r) > image.depth.depth(q.xy())) != truth {
            depth_wrong += 1;
        }
        if vis.is_visible(q, c).expect("an image") == truth {
            continue;
        }
        wrong += 1;
        let speed = |d: Vec3| {
            let a = view.apply(P3::from_vec(jet.position));
            let b = view.apply(P3::from_vec(jet.position + d));
            Vec2::new(b.x() - a.x(), b.y() - a.y()).length()
        };
        let hu = swift::min(2.0 * px / swift::max(speed(jet.du), 1e-12), 0.3);
        let hv = swift::min(2.0 * px / swift::max(speed(jet.dv), 1e-12), 0.3);
        let mut near = false;
        for (su, sv) in
            [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
        {
            let p = (surface.map)(P2::new(c.x() + su * hu, c.y() + sv * hv)).position;
            if torus_sees(p, big_r, r, eye) != truth {
                near = true;
                break;
            }
        }
        if !near {
            unexplained += 1;
            if example.is_empty() {
                example = format!("e.g. ({:.4}, {:.4}) truth {}", c.x(), c.y(), if truth { "seen" } else { "hidden" });
            }
        }
    }
    (wrong, unexplained, depth_wrong, example)
}
