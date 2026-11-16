//! The visibility suites of `Sources/kurven-test` that need the scene and
//! the refiner but not Hatch.swift's strokes: the landscape part of
//! Occlusion.swift's suite, Facing.swift's facing and fold suites and its
//! navigation check, and Faces.swift's tile-boundary visibility and fold
//! suites. Swift's tolerances throughout.

// Swift's `guard x > 0 else` refuses a NaN, and so does `if !(x > 0.0)`.
#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::manual_range_contains)]

mod common;

use common::{build, drawn_triangles, first_hit, random_heightfield, stress_cameras};
use kurven_core::camera::Camera;
use kurven_core::depth::{DepthFrame, DepthImage};
use kurven_core::grid::Grid2D;
use kurven_core::hatching::FaceKind;
use kurven_core::hidden_line::{HeightfieldVisibility, HiddenLine};
use kurven_core::manifest::{
    Affine2, Domain, Interval, LayerSource, ObliqueProjection, RealBand, Region, Side, Truncation, Walls,
};
use kurven_core::mesh::Mesh;
use kurven_core::navigation::Orbit;
use kurven_core::occlusion::OcclusionTest;
use kurven_core::polyline::PolylineSet;
use kurven_core::scene::{Heightfield, PreviewMode, Scene};
use kurven_core::simd::{Vec2, Vec3};
use kurven_core::spaces::{Angle, DomainSpace, P2, P3, Transform, ViewSpace, WorldSpace};
use kurven_core::splitmix::SplitMix;
use kurven_core::surface::{ContourRefine, Surface};
use kurven_landscape::{Catalog, LandscapeRequest, NativeLandscape};
use kurven_math::{real, swift};

/// Two directions across the line of sight, for the nudges.
fn across(d: Vec3) -> [Vec3; 4] {
    let a = d.cross(if d.z.abs() < 0.9 { Vec3::new(0.0, 0.0, 1.0) } else { Vec3::new(1.0, 0.0, 0.0) }).normalized();
    let b = d.cross(a);
    [a, -a, b, -b]
}

/// The landscape part of "occlusion: each kind of solid's test is a cast of
/// the line of sight": points over its drawn surface, with no depth
/// tolerance, which outside the solid is the plain question of whether the
/// line of sight meets the triangles.
#[test]
fn a_landscapes_grid_traversal_is_the_cast_against_its_drawn_triangles() {
    let field = random_heightfield(4, 0xC0FFEE, Truncation::Uniform(2.2), 3.0);
    let mesh = drawn_triangles(&field, -1.0);
    let d = field.surface.domain();
    let mut rng = SplitMix::new(0x0CC2);
    let mut points: Vec<Vec3> = Vec::new();
    while points.len() < 400 {
        let x = rng.next(d.real.lo, d.real.hi);
        let y = rng.next(d.imag.lo, d.imag.hi);
        let p = P2::<WorldSpace>::new(x, y);
        let floor = field.drawn_height(p) + 0.01;
        points.push(Vec3::new(p.x(), p.y(), rng.next(floor, swift::max(floor, 3.5))));
    }
    let (mut compared, mut wrong, mut undecided, mut seen) = (0, 0, 0, 0);
    for (_, view) in stress_cameras() {
        let test: &dyn OcclusionTest = &HeightfieldVisibility::new(field.clone(), &view, 0.0);
        let toward = -view.projection_direction();
        for &p in &points {
            let cast = |q: Vec3| first_hit(&mesh, q, toward, 1e-10).is_none();
            let truth = cast(p);
            if !across(toward).iter().all(|&a| cast(p + 1e-7 * a) == truth) {
                undecided += 1;
                continue;
            }
            compared += 1;
            if truth {
                seen += 1;
            }
            if test.is_clear(P3::from_vec(p)) != truth {
                wrong += 1;
            }
        }
    }
    assert!(
        wrong == 0 && compared > 10_000 && seen > compared / 10 && seen < compared,
        "a landscape's grid traversal is the cast against its drawn triangles: {wrong} of {compared} disagree, \
         {seen} clear, {undecided} grazing"
    );
}

/// The polyhedra of the occlusion suite: the cube, the frame and Császár's
/// torus, each held to a cast against its pieces.
#[test]
#[ignore = "PORT: phase 5 -- Solid.Occlusion's point form (Solid::occlusion refuses until then), and the named solids (Solid.platonic, Solid.frame, Solid.csaszar)"]
fn the_point_form_of_occlusion_by_a_polyhedrons_pieces_is_the_cast() {}

/// The round torus of the occlusion suite, and a parametric surface with
/// no tubular neighbourhood having no sphere tracing.
#[test]
#[ignore = "PORT: phase 4 -- ParametricSurface::torus refuses until then; SphereTracing itself is ported and waits on it"]
fn the_round_toruss_sphere_tracing_is_the_walk_up_the_line_of_sight() {}

/// The ridge of Facing.swift: a Gaussian along y, 801 by 21.
fn ridge() -> (Surface, Heightfield, impl Fn(f64) -> f64, impl Fn(f64) -> f64) {
    let (amplitude, width) = (3.0, 0.5);
    let height = move |x: f64| amplitude * real::exp(-x * x / (width * width));
    let slope = move |x: f64| -2.0 * x / (width * width) * height(x);
    let (nx, ny) = (801usize, 21usize);
    let domain = Domain::new(Interval::new(-2.0, 2.0), Interval::new(0.0, 1.0));
    let mut values = vec![0f32; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            values[j * nx + i] = height(-2.0 + 4.0 * i as f64 / (nx - 1) as f64) as f32;
        }
    }
    let surface = Surface::new(Grid2D::new(nx, ny, domain, values), None, Truncation::None, false);
    let field = Heightfield::new(
        surface.clone(),
        Mesh::new(Vec::new(), Vec::new()),
        vec![Affine2::IDENTITY],
        Region::Full,
        1,
        None,
        Walls::None,
    );
    (surface, field, height, slope)
}

/// A camera whose line of sight crosses the ridge: of the two drawing
/// turns, the one that looks more along x.
fn across_the_ridge() -> Camera {
    let cameras = [0.0, 90.0].map(|z| {
        Camera::oblique(&ObliqueProjection { shear: 0.0, x_angle: -55.0, z_angle: z, flip_x: false, y_scale: None })
    });
    // Swift's `max(by:)`: the first of the greatest.
    if cameras[0].view.projection_direction().x.abs() < cameras[1].view.projection_direction().x.abs() {
        cameras[1]
    } else {
        cameras[0]
    }
}

/// Where the surface turns edge-on: h' = sight.z / sight.x, bisected.
fn analytic_folds(slope: &impl Fn(f64) -> f64, critical: f64) -> Vec<f64> {
    let mut folds = Vec::new();
    let mut previous = -2.0;
    for i in 0..=399 {
        let x = -1.99 + i as f64 * 0.01;
        if (slope(previous) - critical > 0.0) != (slope(x) - critical > 0.0) {
            let (mut lo, mut hi) = (previous, x);
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                if (slope(lo) - critical > 0.0) == (slope(mid) - critical > 0.0) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            folds.push(0.5 * (lo + hi));
        }
        previous = x;
    }
    folds
}

/// "facing: curves on a heightfield is hidden where the surface turns away".
///
/// A Gaussian ridge along y, seen from an oblique camera whose line of sight has a
/// component across it: the near graph faces the eye, the steep part of the
/// far graph faces away, and the far graph's lower slope faces the eye again. Curves run
/// straight over the ridge on the surface must be cut at the two folds, found
/// on the exact gradient; with nothing in front of it, the depth test alone
/// keeps every vertex, which is the leak this test exists to close.
#[test]
fn curves_on_a_heightfield_are_hidden_where_the_surface_turns_away() {
    let (surface, field, height, slope) = ridge();
    let camera = across_the_ridge();
    let sight = camera.view.projection_direction();
    assert!(sight.x.abs() > 0.3, "the camera looks across the ridge: sight {sight:?}");

    // The normal by differences agrees with the analytic gradient.
    let mut worst = 0.0f64;
    for i in 0..=60 {
        let x = -1.5 + i as f64 * 0.05;
        let n = field.normal(P2::new(x, 0.5));
        worst = swift::max(worst, (n.x + slope(x)).abs() / swift::max(1.0, slope(x).abs()));
    }
    assert!(
        worst < 1e-3,
        "the surface normal is the gradient, read off the grid: worst relative slope error {worst:.1e}"
    );

    // Nothing in front of anything: the depth test keeps every vertex.
    let path: Vec<P3<WorldSpace>> =
        (0..=400).map(|i| -2.0 + i as f64 * 0.01).map(|x| P3::new(x, 0.5, height(x))).collect();
    let curves = PolylineSet::from_paths(&[path]);
    let projected = curves.mapped(&camera.view);
    let scene = Scene::heightfield_scene(
        surface,
        Mesh::new(Vec::new(), Vec::new()),
        vec![Affine2::IDENTITY],
        Region::Full,
        1,
        Vec::new(),
        camera,
        PreviewMode::Drawing,
        0.02,
        None,
        Walls::None,
    );
    let bounds = scene.view_bounds().unwrap().expect("the ridge has bounds");
    let frame = DepthFrame::covering(&bounds, 100);
    let empty = DepthImage::new(frame, vec![f32::NEG_INFINITY; (frame.rows * frame.cols) as usize], f32::NEG_INFINITY);
    let by_depth = HiddenLine::clip(&projected, &empty, 0.02);
    // An infinite depth tolerance disarms the traversal: the facing alone decides.
    let cut =
        field.visible_curves(&curves, Some(FaceKind::Graph), &camera.view, f64::INFINITY, 0.0).mapped(&camera.view);
    assert!(
        by_depth.count() == 1 && by_depth.vertices.len() == curves.vertices.len(),
        "tested by depth alone, the whole path is kept"
    );

    // Where the surface turns edge-on: facing = -(n · sight)/|n| = 0,
    // with n = (-h', 0, 1), so h' = sight.z / sight.x.
    let folds = analytic_folds(&slope, sight.z / sight.x);
    assert_eq!(folds.len(), 2, "the far graph has two folds: steep enough to turn away, then not: {folds:?}");
    assert_eq!(cut.count(), 2, "the curves are cut into two runs, one each side of the hidden stretch");
    let ends = [cut.vertices[cut.offsets[1] - 1], cut.vertices[cut.offsets[1]]];
    let expected: Vec<P3<ViewSpace>> = folds.iter().map(|&x| camera.view.apply(P3::new(x, 0.5, height(x)))).collect();
    let miss = ends.iter().zip(&expected).map(|(e, x)| (e.v - x.v).length()).fold(f64::NEG_INFINITY, swift::max);
    assert!(miss < 2e-3, "the curves are cut into two runs, ending and resuming on the exact folds: within {miss:.1e}");
    let kept = (cut.offsets[1] - cut.offsets[0]) + (cut.offsets[2] - cut.offsets[1]);
    let n = curves.vertices.len();
    assert!(kept < n && kept > n / 2, "and the stretch between them, facing away, is gone: {} of {n} hidden", n - kept);
    // With the traversal armed, the far lower slope -- facing the eye again, but
    // behind the ridge from where the eye is -- goes too: the line of sight
    // from it passes under the top edge.
    let visible = field.visible_curves(&curves, Some(FaceKind::Graph), &camera.view, 0.02, 0.0).mapped(&camera.view);
    let last_kept = visible.vertices.last().map_or(f64::INFINITY, |v| (v.v - expected[0].v).length());
    assert!(
        visible.count() == 1 && last_kept < 2e-3,
        "tested by the solid as well, only the near graph remains, up to the first fold: {} runs, last vertex \
         {last_kept:.1e} from the fold",
        visible.count()
    );
}

/// "folds: a heightfield's fold curves are where it turns edge-on".
///
/// The same ridge: its folds are the two lines x = const where h'(x) is the
/// line of sight's slope, straight along y. `Heightfield::fold_curves` must find
/// both, nothing else, and put every vertex on them to rounding.
#[test]
fn a_heightfields_fold_curves_are_where_it_turns_edge_on() {
    let (_, field, _, slope) = ridge();
    let camera = across_the_ridge();
    let sight = camera.view.projection_direction();
    let folds = analytic_folds(&slope, sight.z / sight.x);
    let nearest = |x: f64| folds.iter().map(|f| (x - f).abs()).fold(f64::INFINITY, swift::min);
    let traced = field.fold_curves(&camera.view, true);
    assert_eq!(traced.count(), 2, "two fold curves, one per analytic fold at {folds:?}");
    let worst = traced.vertices.iter().map(|v| nearest(v.x())).fold(0.0, swift::max);
    assert!(worst < 1e-3, "every vertex lies on one of them: worst {worst:.1e} in x");
    for path in traced.paths() {
        let ys: Vec<f64> = path.iter().map(|v| v.y()).collect();
        let (lo, hi) = (ys.iter().copied().fold(1.0, swift::min), ys.iter().copied().fold(0.0, swift::max));
        assert!(lo < 0.05 && hi > 0.95, "and each runs the whole width of the ridge: y from {lo:.2} to {hi:.2}");
    }
    let grid = field.fold_curves(&camera.view, false);
    let grid_worst = grid.vertices.iter().map(|v| nearest(v.x())).fold(0.0, swift::max);
    assert!(
        grid.count() == traced.count() && grid_worst < 4.0 / 800.0,
        "read off the grid alone, the same lines within a cell: worst {grid_worst:.1e} in x"
    );
}

/// "folds: a landscape's pits have their far edges". Swift's last check
/// bakes the drawing and asks that the fold layer draws; here the fold
/// layer of `Scene::visible_layers`, which is what the bake draws.
#[test]
fn a_landscapes_pits_have_their_far_edges() {
    let preset = Catalog::native().preset("rgamma").unwrap();
    let bundle = build(&LandscapeRequest::from_preset(preset, 240), true);
    let scene = Scene::from_bundle(&bundle, &bundle.manifest.presets[0]);
    let h = scene.heightfield().unwrap();
    assert!(
        scene.layers().iter().any(|l| matches!(l.spec.source, LayerSource::FoldCurves)),
        "a landscape describes its fold curves"
    );
    let view = scene.camera.view;
    let folds = h.fold_curves(&view, true);
    let curves: f64 = folds.paths().map(|p| p.windows(2).map(|w| (w[1].v - w[0].v).length()).sum::<f64>()).sum();
    assert!(folds.count() > 0 && curves > 1.0, "it has fold curves: {} lines, {curves:.2} units", folds.count());
    assert!(scene.view_bounds().unwrap().is_some(), "the drawing has bounds");
    let visibility = HeightfieldVisibility::new(h.clone(), &view, 0.02);
    // Every vertex but a run's end carried to a zero (on the floor) or
    // to the truncation edge (on the crease, where the facing jumps).
    let truncation = &h.surface.truncation;
    let off_fold: Vec<(P3<WorldSpace>, f64)> = folds
        .vertices
        .iter()
        .filter(|v| v.z() > 1e-9 && v.z() < truncation.height_at(v.x()) - 1e-9)
        .map(|&v| (v, visibility.facing(P2::new(v.x(), v.y())).abs()))
        .collect();
    let worst = off_fold.iter().fold(None, |m: Option<(P3<WorldSpace>, f64)>, &(v, f)| match m {
        Some((_, g)) if !(g < f) => m,
        _ => Some((v, f)),
    });
    let edge_on = worst.map_or(f64::INFINITY, |w| w.1);
    let off_count = off_fold.iter().filter(|w| w.1 > 1e-6).count();
    assert!(
        edge_on < 1e-6,
        "every vertex is where the surface faces exactly edge-on: worst |facing| {edge_on:.1e} at {:?}, \
         {off_count} of {} off",
        worst.map(|w| w.0),
        off_fold.len()
    );
    let at_zeros = folds.vertices.iter().filter(|v| v.z() <= 1e-9).count();
    eprintln!(
        "rgamma's folds: {} lines, {curves:.2} units, {} vertices off the floor and the crease, worst |facing| \
         {edge_on:.1e}, {at_zeros} on the floor",
        folds.count(),
        off_fold.len()
    );
    assert!(
        at_zeros >= 2,
        "and the pits' outlines run down to their zeros on the front edge: {at_zeros} fold ends on the floor"
    );
    // Nothing on a plateau but the one vertex where a fold up a graph meets
    // the truncation edge, which ends its run.
    let (mut interior_on_plateau, mut ends_on_truncation_edge) = (0, 0);
    for p in folds.paths() {
        for (k, v) in p.iter().enumerate() {
            if v.z() >= truncation.height_at(v.x()) - 1e-9 {
                if k == 0 || k == p.len() - 1 {
                    ends_on_truncation_edge += 1;
                } else {
                    interior_on_plateau += 1;
                }
            }
        }
    }
    assert!(
        interior_on_plateau == 0,
        "and none is on a plateau but a run's end on the truncation edge: {interior_on_plateau} interior vertices on \
         a plateau, {ends_on_truncation_edge} ends on a truncation edge"
    );
    // The pit at -3: its far edge is a fold running down to the floor.
    let lowest = folds
        .vertices
        .iter()
        .filter(|v| (v.x() + 3.0).abs() < 0.4 && v.y() < 0.6)
        .map(|v| v.z())
        .fold(f64::INFINITY, swift::min);
    assert!(lowest < 0.5, "the pit at -3 has a fold reaching down its wall: lowest fold vertex at z = {lowest:.3}");
    // The drawing draws them.
    let index = scene.layers().iter().position(|l| matches!(l.spec.source, LayerSource::FoldCurves)).unwrap();
    let drawn = scene.visible(&scene.layers()[index], None, None).unwrap();
    assert!(!drawn[0].1.vertices.is_empty(), "the drawing draws the fold layer: {} strokes", drawn[0].1.count());
}

/// "walls: every camera navigation reaches sees from above the floor's plane".
#[test]
fn every_camera_navigation_reaches_sees_from_above_the_floors_plane() {
    let mut lowest = f64::INFINITY;
    let limit = Orbit::ELEVATION_LIMIT_DEGREES;
    let steps = (2.0 * limit / 0.5).round() as i64;
    for e in 0..=steps {
        let elevation = -limit + e as f64 * 0.5;
        for a in 0..24 {
            let azimuth = a as f64 * 15.0;
            let mut orbit = Orbit::matching(&NativeLandscape::PROJECTION, P3::new(0.0, 0.0, 0.0));
            orbit.elevation = Angle::from_degrees(elevation);
            orbit.azimuth = Angle::from_degrees(azimuth);
            lowest = swift::min(lowest, -orbit.camera().view.projection_direction().z);
        }
    }
    assert!(lowest > 0.0, "a line of sight rises toward the eye: least rise {lowest}");
}

/// "visibility: a line of sight that passes a tile boundary under the higher
/// copy's top is hidden, whichever tile comes first".
#[test]
fn a_line_of_sight_that_passes_a_tile_boundary_under_the_higher_copys_top_is_hidden() {
    // |f| = max(1/5, 2 - 20(1 - Re z)) on the unit square, steep up to 2
    // at its right edge, the second tile turned, (x, y) ↦ (1 + y, 1 - x),
    // whose bottom edge meets the first's right edge at 1/5: a riser 1.8
    // high faces the second copy. A point on the second copy a hair from
    // the tile boundary, its line of sight toward -Re z rising 80/3 a
    // unit, passes the tile boundary at 1 and is over the first copy
    // again a column on: it is hidden, the riser in its way, with either
    // tile listed first. A point 1.3 higher passes over the riser and is
    // seen. On the grid.
    let unit = Domain::new(Interval::new(0.0, 1.0), Interval::new(0.0, 1.0));
    let values: Vec<f32> =
        (0..441).map(|k| swift::max(0.2, 2.0 - 20.0 * (1.0 - (k % 21) as f64 / 20.0)) as f32).collect();
    let steep = Surface::new(Grid2D::new(21, 21, unit, values), None, Truncation::Uniform(10.0), false);
    let turned = Affine2 { a: 0.0, b: 1.0, tx: 1.0, c: -1.0, d: 0.0, ty: 1.0 };
    let toward = Vec3::new(-1.0, 0.0, 80.0 / 3.0).normalized();
    let r0 = Vec3::new(0.0, 1.0, 0.0);
    let r1 = toward.cross(r0).normalized();
    let view = Transform::<WorldSpace, ViewSpace>::from_rows((r0, r1, toward), Vec3::ZERO);
    for (name, tiles) in [
        ("the identity first", vec![Affine2::IDENTITY, turned]),
        ("the turned tile first", vec![turned, Affine2::IDENTITY]),
    ] {
        let h = Heightfield::new(steep.clone(), Mesh::empty(), tiles, Region::Full, 1, None, Walls::None);
        let behind = h.visible_curves(
            &PolylineSet::from_paths(&[[P3::new(1.03, 0.5, 0.2), P3::new(1.03, 0.45, 0.2)]]),
            None,
            &view,
            1e-6,
            0.0,
        );
        let over = h.visible_curves(
            &PolylineSet::from_paths(&[[P3::new(1.03, 0.5, 1.5), P3::new(1.03, 0.45, 1.5)]]),
            None,
            &view,
            1e-6,
            0.0,
        );
        assert!(
            behind.vertices.is_empty() && over.vertices.len() == 2,
            "{name}: a point behind the riser is hidden, and one over it seen: {} of 2 behind seen, {} of 2 over seen",
            behind.vertices.len(),
            over.vertices.len()
        );
    }
}

/// The landscape a Faces.swift suite draws: a surface, its tiles and the
/// function behind it (Swift's `Landscape`, less its faces).
struct Landscape {
    name: String,
    surface: Surface,
    tiles: Vec<Affine2>,
    region: Region,
    refine: Option<ContourRefine>,
}

impl Landscape {
    fn heightfield(&self) -> Heightfield {
        Heightfield::new(
            self.surface.clone(),
            Mesh::empty(),
            self.tiles.clone(),
            self.region.clone(),
            1,
            self.refine.clone(),
            Walls::None,
        )
    }

    /// A world point carried back to the domain through the nearest tile,
    /// and that tile's inverse (Swift's `carriedBack`).
    fn carried_back(&self, w: Vec2) -> (Vec2, Affine2) {
        if self.tiles.len() <= 1 {
            return (w, Affine2::IDENTITY);
        }
        let (real, imag) = (self.surface.domain().real, self.surface.domain().imag);
        let mut best = (w, Affine2::IDENTITY);
        let mut nearest = f64::INFINITY;
        for tile in &self.tiles {
            let Some(back) = tile.inverse() else { continue };
            let q = back.apply(P3::new(w.x, w.y, 0.0)).xy().v;
            let d = swift::max(swift::min(real.lo, real.hi) - q.x, 0.0)
                + swift::max(q.x - swift::max(real.lo, real.hi), 0.0)
                + swift::max(swift::min(imag.lo, imag.hi) - q.y, 0.0)
                + swift::max(q.y - swift::max(imag.lo, imag.hi), 0.0);
            if d < nearest {
                nearest = d;
                best = (q, back);
            }
            if d == 0.0 {
                break;
            }
        }
        best
    }
}

/// The distance from `p` to the nearest segment of `paths`, or infinity
/// where none lies within `r` (Swift's `Segments.distance(_:within:)`, by
/// brute force).
fn distance_within(paths: &[Vec<Vec2>], p: Vec2, r: f64) -> f64 {
    let mut best = f64::INFINITY;
    for path in paths {
        for w in path.windows(2) {
            let (a, b) = (w[0], w[1]);
            if a == b {
                continue;
            }
            let d = b - a;
            let t = swift::max(0.0, swift::min(1.0, (p - a).dot(d) / d.dot(d)));
            best = swift::min(best, (p - (a + t * d)).length());
        }
    }
    if best <= r { best } else { f64::INFINITY }
}

/// The direction lines of sight travel: the kernel of the view's first
/// two rows, away from the eye.
fn sight_of(camera: &Camera) -> Vec3 {
    let m = camera.view.m.columns;
    let rows =
        (Vec3::new(m[0].x, m[1].x, m[2].x), Vec3::new(m[0].y, m[1].y, m[2].y), Vec3::new(m[0].z, m[1].z, m[2].z));
    let mut sight = rows.0.cross(rows.1).normalized();
    if rows.2.dot(sight) > 0.0 {
        sight = -sight;
    }
    sight
}

fn oblique(z_angle: f64) -> Camera {
    Camera::oblique(&ObliqueProjection { shear: 0.0, x_angle: -55.0, z_angle, flip_x: false, y_scale: None })
}

/// "folds: on tiles under a banded truncation, a fold runs on the graph
/// wherever the facing turns, read through the tile".
#[test]
fn on_tiles_under_a_banded_truncation_a_fold_runs_on_the_graph_wherever_the_facing_turns() {
    // The truncation under a fold is read at the point's own x through
    // the tile under it, as the surface is: every vertex of a fold lies
    // on the graph, |f| under the truncation there, or on its
    // truncation edge; and wherever the facing changes sign between two
    // points of the graph in the footprint, well under the truncation,
    // a fold passes between them.
    // A tower, 1/(z - 1/2), banded on either side of its pole, 4 | 2 and
    // 2 | 4, in two tiles, the second reflected across the right edge or
    // turned a quarter, (x, y) ↦ (5/2 - y, x): the truncation under the
    // second differs from the one at the world x by a band about the
    // pole's image. On f and on the grid.
    let domain = Domain::new(Interval::new(-0.5, 1.5), Interval::new(-1.0, 1.0));
    let reflected = Affine2 { a: -1.0, b: 0.0, tx: 3.0, c: 0.0, d: 1.0, ty: 0.0 };
    let turned = Affine2 { a: 0.0, b: -1.0, tx: 2.5, c: 1.0, d: 0.0, ty: 0.0 };
    let mut landscapes = Vec::new();
    for (left, right, tile, name, on_f) in [
        (4.0, 2.0, reflected, "reflected", true),
        (2.0, 4.0, turned, "turned", true),
        (2.0, 4.0, reflected, "reflected", false),
        (4.0, 2.0, turned, "turned", false),
    ] {
        let mut request = LandscapeRequest::new("1/(z - 0.5)", domain, 48);
        request.truncation =
            Some(Truncation::RealBands { bands: vec![RealBand { below: 0.5, height: left }], beyond: right });
        let tower = build(&request, on_f);
        landscapes.push(Landscape {
            name: format!(
                "a tower banded at {} | {} in two tiles, the second {name}, {}",
                left as i64,
                right as i64,
                if on_f { "on f" } else { "on the grid" }
            ),
            surface: tower.surface.clone(),
            tiles: vec![Affine2::IDENTITY, tile],
            region: Region::Full,
            refine: tower.refine.clone(),
        });
    }
    let camera = oblique(30.0);
    let sight = sight_of(&camera);
    for l in &landscapes {
        let h = l.heightfield();
        let surface = &l.surface;
        let (real, imag) = (surface.domain().real, surface.domain().imag);
        let cell = swift::max(
            real.length().abs() / (surface.height.width - 1) as f64,
            imag.length().abs() / (surface.height.height - 1) as f64,
        );
        let Truncation::RealBands { bands, .. } = &surface.truncation else { continue };
        let boundaries: Vec<f64> = bands.iter().map(|b| b.below).collect();
        let magnitude = |q: Vec2| -> f64 {
            let u = match l.refine.as_ref().and_then(|r| r.magnitude.as_ref()) {
                Some(m) => m(P2::new(q.x, q.y)),
                None => surface.magnitude_at(P2::new(q.x, q.y)),
            };
            if u.is_nan() { f64::INFINITY } else { u }
        };
        let folds = h.fold_curves(&camera.view, true);
        let (mut vertices, mut off) = (0, Vec::new());
        for v in &folds.vertices {
            let q = l.carried_back(v.xy().v).0;
            if boundaries.iter().any(|b| (q.x - b).abs() <= 1e-9 * cell) {
                continue;
            }
            vertices += 1;
            let top = surface.truncation_height_at(q.x);
            if !(magnitude(q) <= top * (1.0 + 1e-6)) {
                off.push(format!("{:?}, |f| {} over {top}", v, magnitude(q)));
            }
        }
        // The facing on a grid of each tile's image, and the folds near
        // each sign change.
        let traced: Vec<Vec<Vec2>> = folds.paths().map(|p| p.iter().map(|v| v.xy().v).collect()).collect();
        let on_f = l.refine.as_ref().and_then(|r| r.magnitude.as_ref()).is_some();
        let facing = |w: Vec2| -> f64 {
            let n = if on_f { h.graph_normal(P2::from_vec(w)) } else { h.normal(P2::from_vec(w)) };
            -n.dot(sight) / n.length()
        };
        let (mut turns, mut missing) = (0, Vec::new());
        for tile in &l.tiles {
            let (nx, ny) = (161, 97);
            let point = |i: usize, j: usize| -> (Vec2, Vec2) {
                let p = Vec2::new(
                    real.lo + real.length() * (i as f64 + 0.37) / nx as f64,
                    imag.lo + imag.length() * (j as f64 + 0.61) / ny as f64,
                );
                (p, tile.apply(P3::new(p.x, p.y, 0.0)).xy().v)
            };
            // On the graph well under the truncation, away from a band
            // boundary, the domain's edge and the footprint's, and held
            // by this tile alone.
            let clear = |p: Vec2, w: Vec2| -> bool {
                let reach = 2.0 * cell;
                if !(p.x - reach > swift::min(real.lo, real.hi)
                    && p.x + reach < swift::max(real.lo, real.hi)
                    && p.y - reach > swift::min(imag.lo, imag.hi)
                    && p.y + reach < swift::max(imag.lo, imag.hi)
                    && !boundaries.iter().any(|b| (p.x - b).abs() <= reach)
                    && [
                        Vec2::new(reach, 0.0),
                        Vec2::new(-reach, 0.0),
                        Vec2::new(0.0, reach),
                        Vec2::new(0.0, -reach),
                        Vec2::ZERO,
                    ]
                    .iter()
                    .all(|&o| l.region.contains(P2::from_vec(w + o)))
                    && Some(l.carried_back(w).1) == tile.inverse())
                {
                    return false;
                }
                magnitude(p) < 0.98 * surface.truncation_height_at(p.x)
            };
            for j in 0..ny {
                let mut previous: Option<(Vec2, Vec2, f64)> = None;
                for i in 0..nx {
                    let (p, w) = point(i, j);
                    let here = (p, w, facing(w));
                    if let Some(a) = previous
                        && (a.2 > 0.0) != (here.2 > 0.0)
                        && clear(a.0, a.1)
                        && clear(p, w)
                        && clear(0.5 * (a.0 + p), 0.5 * (a.1 + w))
                    {
                        turns += 1;
                        let mid = 0.5 * (a.1 + w);
                        let reach = (w - a.1).length() + cell;
                        if distance_within(&traced, mid, reach).is_infinite() {
                            missing.push(format!("{mid:?}"));
                        }
                    }
                    previous = Some(here);
                }
            }
        }
        assert!(
            vertices > 50 && off.is_empty() && turns > 20 && missing.is_empty(),
            "{}: every fold vertex on the graph, and a fold wherever the facing turns on it: {} of {vertices} \
             vertices on a plateau, {} of {turns} turns with no fold: {:?} {:?}",
            l.name,
            off.len(),
            missing.len(),
            off.iter().take(2).collect::<Vec<_>>(),
            missing.iter().take(2).collect::<Vec<_>>()
        );
    }
}

/// "folds: a fold that runs into a zero of f ends on it, on every tile".
#[test]
fn a_fold_that_runs_into_a_zero_of_f_ends_on_it_on_every_tile() {
    // A cone, 10(z - 0.3), over Im z in [0, 1], its apex on the domain's
    // edge, in three tiles, the second reflected across Re z = 1 and the
    // third turned a quarter, (x, y) ↦ (4 - y, x): its outline runs down
    // a generator into the apex in each copy, and each copy's fold is
    // carried onto the apex's image there, at its height, the floor,
    // where placed alone it stops a hair short.
    let mut request =
        LandscapeRequest::new("10*(z - 0.3)", Domain::new(Interval::new(-1.0, 1.0), Interval::new(0.0, 1.0)), 30);
    request.truncation = Some(Truncation::Uniform(30.0));
    let cone = build(&request, true);
    let tiles = vec![
        Affine2::IDENTITY,
        Affine2 { a: -1.0, b: 0.0, tx: 2.0, c: 0.0, d: 1.0, ty: 0.0 },
        Affine2 { a: 0.0, b: -1.0, tx: 4.0, c: 1.0, d: 0.0, ty: 0.0 },
    ];
    let h = Heightfield::new(
        cone.surface.clone(),
        Mesh::empty(),
        tiles.clone(),
        Region::Full,
        1,
        cone.refine.clone(),
        Walls::None,
    );
    // Under cameras turned every quarter, so each copy shows the apex's
    // outline under some: every end near an apex is on it, and every
    // tile has one.
    let mut report = Vec::new();
    let mut carried = vec![0; tiles.len()];
    for z_angle in [30.0, 120.0, -150.0, -60.0] {
        let camera = oblique(z_angle);
        let folds = h.fold_curves(&camera.view, true);
        let ends: Vec<P3<WorldSpace>> = folds.paths().flat_map(|r| [r[0], r[r.len() - 1]]).collect();
        for (k, tile) in tiles.iter().enumerate() {
            let apex = tile.apply(P3::new(0.3, 0.0, 0.0)).xy().v;
            let near: Vec<&P3<WorldSpace>> = ends.iter().filter(|e| (e.xy().v - apex).length() < 0.2).collect();
            let on_it = near.iter().filter(|e| (e.xy().v - apex).length() <= 1e-6 && e.z().abs() <= 1e-6).count();
            carried[k] += on_it;
            if on_it != near.len() {
                let nearest = near.iter().map(|e| (e.xy().v - apex).length()).fold(f64::INFINITY, swift::min);
                report.push(format!(
                    "tile {k} at {}°: {on_it} of {} ends near the apex on it, nearest {nearest}",
                    z_angle as i64,
                    near.len()
                ));
            }
        }
    }
    assert!(
        report.is_empty() && carried.iter().all(|&c| c > 0),
        "the cone's folds end on the apex on every tile, at its height: ends on an apex by tile {carried:?}: {:?}",
        report.iter().take(3).collect::<Vec<_>>()
    );
}

/// "folds: every chord of a fold lies within the refiner's tolerance of the
/// fold, on every tile".
#[test]
fn every_chord_of_a_fold_lies_within_the_refiners_tolerance_of_the_fold_on_every_tile() {
    // The midpoint of each chord, on the graph under the truncation and
    // off the domain's edge, is within the refiner's tolerance of the
    // fold: the nearest sign change of the facing along the chord's
    // normal, bisected, read through the tile under each point as the
    // facing is. On f, in two tiles, the second reflected, turned a
    // quarter or beside the first; a fold's chords on a second tile are
    // as fine as on the first.
    let domain = Domain::new(Interval::new(-0.5, 1.5), Interval::new(-1.0, 1.0));
    let camera = oblique(30.0);
    let sight = sight_of(&camera);
    for (expression, resolution) in [("2*z*z*z - z", 20), ("1/(z - 0.5) + 0.3*z*z", 24), ("1/(z - 0.5)", 16)] {
        let mut request = LandscapeRequest::new(expression, domain, resolution);
        request.truncation = Some(Truncation::Uniform(5.0));
        let bundle = build(&request, true);
        let Some(refine) = bundle.refine.clone() else { continue };
        let Some(magnitude) = refine.magnitude.clone() else { continue };
        let surface = bundle.surface.clone();
        let cell = swift::max(
            domain.real.length().abs() / (surface.height.width - 1) as f64,
            domain.imag.length().abs() / (surface.height.height - 1) as f64,
        );
        for (name, tile) in [
            ("reflected", Affine2 { a: -1.0, b: 0.0, tx: 3.0, c: 0.0, d: 1.0, ty: 0.0 }),
            ("turned", Affine2 { a: 0.0, b: -1.0, tx: 2.5, c: 1.0, d: 0.0, ty: 0.0 }),
            ("beside", Affine2 { a: 1.0, b: 0.0, tx: 2.0, c: 0.0, d: 1.0, ty: 0.0 }),
        ] {
            let l = Landscape {
                name: format!("{expression} at res {resolution}, a second tile {name}"),
                surface: surface.clone(),
                tiles: vec![Affine2::IDENTITY, tile],
                region: Region::Full,
                refine: Some(refine.clone()),
            };
            let h = l.heightfield();
            let facing = |w: Vec2| -> f64 {
                let n = h.graph_normal(P2::from_vec(w));
                -n.dot(sight) / n.length()
            };
            let folds = h.fold_curves(&camera.view, true);
            let (mut chords, mut worst, mut off) = ([0usize, 0], [0.0f64, 0.0], Vec::new());
            for run in folds.paths() {
                for w in run.windows(2) {
                    let (a, c) = (w[0], w[1]);
                    let mid = 0.5 * (a.xy().v + c.xy().v);
                    let length = (c.xy().v - a.xy().v).length();
                    if !(length > 1e-9 * cell) {
                        continue;
                    }
                    let (q, back) = l.carried_back(mid);
                    if !(q.x > domain.real.lo + cell
                        && q.x < domain.real.hi - cell
                        && q.y > domain.imag.lo + cell
                        && q.y < domain.imag.hi - cell
                        && magnitude(P2::<DomainSpace>::new(q.x, q.y)) < 0.9 * surface.truncation_height_at(q.x))
                    {
                        continue;
                    }
                    let n = Vec2::new(-(c.y() - a.y()), c.x() - a.x()) / length;
                    let f0 = facing(mid);
                    let mut distance = f64::INFINITY;
                    for sign in [1.0, -1.0] {
                        let (mut lo, mut hi) = (0.0, 0.5 * cell * sign);
                        if (facing(mid + hi * n) > 0.0) == (f0 > 0.0) {
                            continue;
                        }
                        for _ in 0..60 {
                            let t = 0.5 * (lo + hi);
                            if (facing(mid + t * n) > 0.0) == (f0 > 0.0) {
                                lo = t;
                            } else {
                                hi = t;
                            }
                        }
                        distance = swift::min(distance, (0.5 * (lo + hi)).abs());
                    }
                    let t = if back == Affine2::IDENTITY { 0 } else { 1 };
                    chords[t] += 1;
                    worst[t] = swift::max(worst[t], distance / cell);
                    if !(distance <= refine.tolerance * cell) {
                        off.push(format!("tile {t} at ({:.4}, {:.4}), {:.3} cells", mid.x, mid.y, distance / cell));
                    }
                }
            }
            assert!(
                chords.iter().all(|&c| c > 30) && off.is_empty(),
                "{}, on f: every chord of a fold within {} cells of the fold, on both tiles: {} of {chords:?} chords \
                 off, worst {worst:?} cells: {:?}",
                l.name,
                refine.tolerance,
                off.len(),
                off.iter().take(3).collect::<Vec<_>>()
            );
        }
    }
}

/// The folds' half of "faces: a contour or a fold carried to the truncation
/// edge across a band boundary ends on the drawn top": every point of the
/// folds under four cameras, vertices and seven points along each chord,
/// lies on min(|f|, h(x)) -- on a band boundary the higher side's -- within
/// the refiner's tolerance, in height or across the plane. The contour
/// layers' half waits for Hatch.swift: a truncated graph's contours are cut
/// at its trim loops (`graphTrimLoops`), which the stub has only to the
/// domain's edge.
#[test]
fn a_fold_carried_to_the_truncation_edge_across_a_band_boundary_ends_on_the_drawn_top() {
    let domain = Domain::new(Interval::new(-0.5, 1.5), Interval::new(-1.0, 1.0));
    let cases = [
        ("a tower", "1/(z-0.45)", domain, 200, vec![RealBand { below: 0.5, height: 2.0 }], 4.0),
        (
            "a tower, its boundary on a column",
            "1/(z-0.45)",
            domain,
            201,
            vec![RealBand { below: 0.5, height: 2.0 }],
            4.0,
        ),
        (
            "gamma under three bands",
            "gamma(z)",
            Domain::new(Interval::new(-4.5, 4.5), Interval::new(-2.5, 2.5)),
            600,
            vec![
                RealBand { below: -2.5, height: 0.5 },
                RealBand { below: -1.5, height: 1.0 },
                RealBand { below: 0.5, height: 1.5 },
            ],
            2.5,
        ),
    ];
    for (name, expression, domain, resolution, bands, beyond) in cases {
        let mut request = LandscapeRequest::new(expression, domain, resolution);
        request.truncation = Some(Truncation::RealBands { bands: bands.clone(), beyond });
        let built = build(&request, true);
        let refine = built.refine.clone().expect("a refiner");
        let magnitude = refine.magnitude.clone().expect("its magnitude");
        let surface = built.surface.clone();
        let g = &*surface.height;
        let cell = swift::max(
            g.domain.real.length().abs() / (g.width - 1) as f64,
            g.domain.imag.length().abs() / (g.height - 1) as f64,
        );
        // How far a point lies off the drawn top: in height, and in
        // cells across the plane where the top is the graph.
        let off = |v: P3<WorldSpace>| -> (f64, f64) {
            let u = magnitude(P2::new(v.x(), v.y()));
            let f = if u.is_nan() { f64::INFINITY } else { u };
            let on_boundary = bands.iter().any(|b| (v.x() - b.below).abs() <= 1e-9 * cell);
            let sides: &[Side] = if on_boundary { &[Side::Left, Side::Right] } else { &[Side::Right] };
            let heights: Vec<f64> = sides.iter().map(|&s| surface.truncation_height_at_from(v.x(), s)).collect();
            let h = heights[1..].iter().fold(heights[0], |m, &x| if m < x { x } else { m });
            let d = v.z() - swift::min(f, h);
            if !(f < h) {
                return (d, f64::INFINITY);
            }
            let e = 1e-4 * cell;
            let gx = (magnitude(P2::new(v.x() + e, v.y())) - magnitude(P2::new(v.x() - e, v.y()))) / (2.0 * e);
            let gy = (magnitude(P2::new(v.x(), v.y() + e)) - magnitude(P2::new(v.x(), v.y() - e))) / (2.0 * e);
            (d, d.abs() / (gx * gx + gy * gy).sqrt() / cell)
        };
        let h = Scene::from_bundle(&built, &built.manifest.presets[0]).heightfield().unwrap().clone();
        for z_angle in [-90.0, 90.0, -20.0, -160.0] {
            let camera = Camera::oblique(&ObliqueProjection {
                shear: 0.5,
                x_angle: -55.0,
                z_angle,
                flip_x: true,
                y_scale: None,
            });
            let paths = h.fold_curves(&camera.view, true);
            let (mut points, mut above, mut below, mut wrong) = (0, 0.0f64, 0.0f64, Vec::new());
            for run in paths.paths() {
                for w in run.windows(2) {
                    let (a, b) = (w[0], w[1]);
                    for s in 0..=8 {
                        let t = s as f64 / 8.0;
                        let v = P3::<WorldSpace>::new(
                            a.x() + t * (b.x() - a.x()),
                            a.y() + t * (b.y() - a.y()),
                            a.z() + t * (b.z() - a.z()),
                        );
                        let (d, cells) = off(v);
                        if !d.is_finite() {
                            continue;
                        }
                        points += 1;
                        above = swift::max(above, d);
                        below = swift::min(below, d);
                        if !(d.abs() <= refine.tolerance * cell || cells <= refine.tolerance) {
                            wrong.push(format!("({:.4}, {:.4}, {:.4}) {:+.4}", v.x(), v.y(), v.z(), d));
                        }
                    }
                }
            }
            assert!(
                points > 100 && wrong.is_empty(),
                "{name}: every point of the folds seen from {}° on the drawn top, within {} cells: {} of {points} points \
                 off, as far as {above:+.4} and {below:+.4}: {:?}",
                z_angle as i64,
                refine.tolerance,
                wrong.len(),
                wrong.iter().take(3).collect::<Vec<_>>()
            );
        }
    }
}
