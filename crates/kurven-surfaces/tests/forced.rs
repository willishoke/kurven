//! Forced.swift's suites: a forced oscillator's invariant torus, found,
//! fitted and drawn (`forcedTorusTests`), the preview's folds read off the
//! grid (`gridFoldTests`), and a surface drawing described and built
//! (`surfaceDrawingTests`), with Swift's tolerances throughout. What times
//! or compares the GPU preview needs a GPU, and is the browser lane's, or
//! left out where that lane cannot run it (each says which).
//!
//! The system is `toroidal_manifolds.ipynb`'s. Its second frequency was found
//! independently in a scipy prototype (DOP853 at rtol 1e-11, NAFF by
//! `minimize_scalar`) as 1.3362213283 -- that is, ω + 0.3962213283 -- and the
//! generator here is taken in (0, ω/2], so the two must agree to that
//! prototype's precision.

mod common;

use common::{alone, par_map, sharing};
use std::f64::consts::PI;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use kurven_bake::{BakeOptions, bake};
use kurven_core::camera::Camera;
use kurven_core::manifest::{HeightPolicy, LayerRole, LayerSource, LayerSpec, ObliqueProjection};
use kurven_core::navigation::{Framing, Gesture, Navigator, Orbit, Viewport};
use kurven_core::parametric::ParametricSurface;
use kurven_core::polyline::{Layer, PolylineSet};
use kurven_core::scene::{PreviewMode, Scene};
use kurven_core::simd::Vec2;
use kurven_core::spaces::{P2, P3, ParamSpace, ViewSpace};
use kurven_core::surface_request::{Field, LinearCurves, Shape, Style, SurfaceRequest, surface_key};
use kurven_landscape::torus::{PeriodicOptions, PeriodicTorus};
use kurven_math::{ContinuedFraction, real, swift};
use kurven_surfaces::{ForcedSystem, InvariantTorus, Revolution, SurfaceDrawing};

/// The notebook's oscillator's torus, found once for the file: an 8000-unit
/// run, fitted at 24 harmonics (`InvariantTorus.find(.jerkTorus)`).
fn found() -> &'static InvariantTorus {
    static FOUND: OnceLock<InvariantTorus> = OnceLock::new();
    FOUND.get_or_init(|| {
        InvariantTorus::find_default(&ForcedSystem::jerk_torus(), 8000.0, 24)
            .unwrap_or_else(|e| panic!("the notebook's oscillator has a torus: {e}"))
    })
}

/// `Revolution.fitting(found)` at Swift's defaults: radial 0, axial 1, radius 2.5.
fn revolution(found: &InvariantTorus) -> Revolution {
    Revolution::fitting(found, 0, 1, 2.5)
}

/// Swift's `Check.expect`s of one suite: each failure kept, with its detail,
/// and the suite failed once at its end with all of them.
#[derive(Default)]
struct Checks(Vec<String>);

impl Checks {
    fn expect(&mut self, ok: bool, what: impl Into<String>) {
        if !ok {
            self.0.push(what.into());
        }
    }

    fn done(self) {
        assert!(self.0.is_empty(), "{}", self.0.join("\n"));
    }
}

fn milliseconds(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// "dynamics: the notebook's forced oscillator winds round a torus"
#[test]
fn the_notebooks_forced_oscillator_winds_round_a_torus() {
    let found = found();
    let mut checks = Checks::default();
    checks.expect(
        (found.internal_frequency() - 0.396_221_328_3).abs() < 1e-8,
        format!("its second frequency is the prototype's: Ω = {:.10}", found.internal_frequency()),
    );
    checks.expect(
        found.residual < 1e-4 * found.extent,
        format!(
            "and the fitted torus holds the trajectory it came from: worst miss {:.2e}, {:.1e} of the attractor's size",
            found.residual,
            found.residual / found.extent
        ),
    );
    checks.done();
}

/// "dynamics: drawn by revolution, it is embedded and the trajectory is on it"
#[test]
fn drawn_by_revolution_it_is_embedded_and_the_trajectory_is_on_it() {
    let found = found();
    let mut checks = Checks::default();
    let square = [Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(1.0, 1.0), Vec2::new(0.0, 1.0)];
    let bow = [Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0), Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)];
    checks.expect(
        Revolution::is_simple_closed(&square) && !Revolution::is_simple_closed(&bow),
        "a square is a simple closed curve and a bow tie is not",
    );

    let e = revolution(found);
    let (surface, embedded) = found.surface(&e, (512, 256)).expect("the revolved torus");
    checks.expect(
        embedded && surface.outward.is_some(),
        "every slice is a simple closed curve clear of the axis, so it bounds a solid",
    );

    let curves = found.trajectory_integrated(&e, None, 500.0, 0.02).expect("the integrated trajectory");
    let coords = curves.coords.as_ref().expect("the trajectory carries coordinates");
    // At a fixed forcing phase the embedding is an isometry of (radius,
    // height), scaled: the curves are at most `scale` times the fit's own
    // miss from the surface.
    let mut worst = 0.0;
    for (k, &c) in coords.iter().enumerate() {
        worst = swift::max(worst, (curves.vertices[k].v - (surface.map)(c).position).length());
    }
    let bound = e.scale * found.residual;
    checks.expect(
        curves.count() > 1 && worst <= bound * (1.0 + 1e-9) + 1e-12,
        format!(
            "the trajectory lies on the drawn torus, within the fit's own miss: worst {worst:.2e}, bound {bound:.2e}, \
             over {} vertices in {} paths",
            curves.vertices.len(),
            curves.count()
        ),
    );

    // What the plates draw is not this curve but the linear curve at the
    // rotation number from the samples' start, placed from the grid.
    // Held to the integrated trajectory: every integrated vertex's
    // coordinate lies on that leaf -- at its wrapped θ, the leaf passes
    // at φ = ρ(θ + 2πk) for the k turns it has made -- and the grid at
    // that coordinate is the integrated point, to the fit's miss plus a
    // cell's sag.
    let turn = 2.0 * PI;
    let (start, rho) = (found.start_phase(), found.rotation_number());
    let (mut off_leaf, mut off_grid) = (0.0, 0.0);
    for (k, &c) in coords.iter().enumerate() {
        let nearest = (0..=200)
            .map(|k| {
                let d = (c.y() - rho * (c.x() + turn * k as f64)) / turn;
                (d - d.round()).abs()
            })
            .fold(f64::INFINITY, swift::min);
        off_leaf = swift::max(off_leaf, nearest);
        off_grid = swift::max(off_grid, (curves.vertices[k].v - surface.interpolate(c).v).length());
    }
    let drawn = found.trajectory(&surface, 12);
    let first = drawn.coords.as_ref().expect("the drawn trajectory carries coordinates")[0];
    let on_leaf = ((start.y() - rho * start.x()) / turn).abs() % 1.0;
    checks.expect(
        off_leaf < 1e-9 && first == start && swift::min(on_leaf, 1.0 - on_leaf) < 1e-9,
        format!("the integrated trajectory lies on the leaf φ = ρθ the drawn one starts on: {off_leaf:.1e} of a turn off the leaf"),
    );
    checks.expect(
        off_grid < bound + 2e-3,
        format!(
            "and the grid at its coordinates is the integrated point: worst {off_grid:.2e}; the fit's miss {bound:.2e}"
        ),
    );
    let convergents: Vec<String> =
        ContinuedFraction::convergents(rho, 400, 1e-12).iter().map(|c| format!("{}/{}", c.p, c.q)).collect();
    checks.expect(
        (rho - 0.421_512_051).abs() < 1e-8
            && convergents == ["0/1", "1/2", "2/5", "3/7", "8/19", "43/102", "51/121", "94/223", "145/344"],
        format!(
            "the rotation number is nearly 8/19, 43/102, 145/344: where a drawing fills evenly: Ω/ω = {rho:.9}, \
             convergents {convergents:?}"
        ),
    );
    checks.done();
}

/// "dynamics: the drawing bakes, and its front is the torus's outside"
#[test]
fn the_drawing_bakes_and_its_front_is_the_torus_outside() {
    let found = found();
    let mut checks = Checks::default();
    let e = revolution(found);
    let (surface, _) = found.surface(&e, (512, 256)).expect("the revolved torus");
    let trajectory = Layer {
        spec: LayerSpec::new("trajectory", LayerRole::Scaffold, LayerSource::Trajectory, 0.08, HeightPolicy::Surface),
        paths: found.trajectory_integrated(&e, None, 500.0, 0.02).expect("the integrated trajectory"),
    };
    let folds = Layer {
        spec: LayerSpec::new("folds", LayerRole::Outline, LayerSource::FoldCurves, 0.5, HeightPolicy::Surface),
        paths: PolylineSet::empty(),
    };
    let camera =
        Camera::oblique(&ObliqueProjection { shear: 0.0, x_angle: -58.0, z_angle: 35.0, flip_x: false, y_scale: None });
    let scene = Scene::parametric_scene(surface.clone(), vec![trajectory, folds], camera, PreviewMode::Drawing, 0.0);
    let baked = bake(&scene, &BakeOptions::new(2000)).expect("a bake");
    // On an embedded surface with its outside computed rightly, what is
    // front-most at any pixel faces the eye; only pixels along the
    // outline, where the surface is edge-on, may not.
    let image = baked.surface.as_ref().expect("a parametric bake carries its coordinate image");
    let sight = camera.view.projection_direction();
    let (mut covered, mut away) = (0usize, 0usize);
    for c in image.coords.iter().filter(|c| !c[0].is_nan()) {
        covered += 1;
        let c = P2::<ParamSpace>::new(f64::from(c[0]), f64::from(c[1]));
        if surface.facing(c, sight).unwrap_or(1.0) <= 0.0 {
            away += 1;
        }
    }
    checks.expect(
        covered > 100_000 && (away as f64) < 0.002 * covered as f64,
        format!("every front-most pixel faces the eye, but along the edge: {away} of {covered} face away"),
    );
    let layers = &baked.strokes.layers;
    checks.expect(
        layers.len() == 2 && layers[0].1.total_length() > 0.0 && layers[1].1.total_length() > 0.0,
        format!(
            "and it draws both the trajectory and the folds: {} trajectory runs, {} fold runs",
            layers.first().map_or(0, |l| l.1.count()),
            layers.get(1).map_or(0, |l| l.1.count())
        ),
    );
    checks.done();
}

// MARK: - the preview's folds, read off the grid
//
// The preview re-derives the folds whenever the camera moves, so they have to
// cost a dot product per grid point and not a map evaluation per grid
// point: the forced torus's map is a 1,225-term Fourier series. Held to two
// things: they are where the exact folds are, to a pixel of the preview, and
// an orbit frame of the forced torus is fast enough to follow the pointer.

/// The worst distance from any vertex of either set to the other set's
/// segments, in view units: the symmetric Hausdorff distance of the two as
/// drawn, to the vertices' own spacing.
fn hausdorff(a: &PolylineSet<ViewSpace>, b: &PolylineSet<ViewSpace>) -> f64 {
    fn segments(s: &PolylineSet<ViewSpace>) -> Vec<(Vec2, Vec2)> {
        let mut out = Vec::new();
        for p in 0..s.count() {
            for i in s.offsets[p]..(s.offsets[p + 1] - 1) {
                out.push((s.vertices[i].v.xy(), s.vertices[i + 1].v.xy()));
            }
        }
        out
    }
    fn directed(from: &PolylineSet<ViewSpace>, to: &[(Vec2, Vec2)]) -> f64 {
        let mut worst = 0.0;
        for v in &from.vertices {
            let q = v.v.xy();
            let mut best = f64::INFINITY;
            for &(p0, p1) in to {
                let d = p1 - p0;
                let l2 = d.length_squared();
                let t = if l2 > 0.0 { swift::min(swift::max((q - p0).dot(d) / l2, 0.0), 1.0) } else { 0.0 };
                best = swift::min(best, (q - (p0 + t * d)).length_squared());
            }
            worst = swift::max(worst, best);
        }
        real::sqrt(worst)
    }
    swift::max(directed(a, &segments(b)), directed(b, &segments(a)))
}

/// The viewport the preview suites frame in.
const VIEWPORT: Viewport = Viewport { width: 1600, height: 1200 };

/// The forced torus at the preview's grid, 1024 x 512, built once for the
/// file, as `found` is: two suites read it.
fn forced_preview_surface() -> &'static ParametricSurface {
    static SURFACE: OnceLock<ParametricSurface> = OnceLock::new();
    SURFACE.get_or_init(|| {
        let found = found();
        found.surface(&revolution(found), (1024, 512)).expect("the revolved torus").0
    })
}

/// "surfaces: the preview's grid folds are the exact folds, to a pixel"
#[test]
fn the_previews_grid_folds_are_the_exact_folds_to_a_pixel() {
    let forced = forced_preview_surface();
    let sn = PeriodicTorus::jacobi_sn(0.64, 0.0, &PeriodicOptions::default()).expect("the sn torus builds").surface;
    let mut checks = Checks::default();
    // Each surface under each projection, independent of the others: side by
    // side, checked in turn.
    let projections = [
        ObliqueProjection { shear: 0.0, x_angle: -55.0, z_angle: 30.0, flip_x: false, y_scale: None },
        ObliqueProjection { shear: 0.5, x_angle: -30.0, z_angle: 110.0, flip_x: false, y_scale: None },
    ];
    let normals = [forced.grid_normals(), sn.grid_normals()];
    let cases: Vec<(usize, &str, &ParametricSurface, ObliqueProjection)> = [("forced", forced), ("sn", &sn)]
        .into_iter()
        .enumerate()
        .flat_map(|(k, (name, surface))| projections.iter().map(move |&projection| (k, name, surface, projection)))
        .collect();
    let _beside = sharing();
    let outcomes = par_map(&cases, |&(k, name, surface, projection)| {
        let camera = Camera::oblique(&projection);
        let sight = camera.view.projection_direction();
        let exact = surface.fold_curves(sight, None, &[]);
        let grid = surface.grid_fold_curves(&normals[k], sight).expect("a normal per grid point");
        let scene = Scene::parametric_scene(surface.clone(), Vec::new(), camera, PreviewMode::Drawing, 0.0);
        let Some(bounds) = scene.quick_bounds(20_000) else {
            return (false, format!("{name} has bounds"));
        };
        let px = Framing::fitting(&bounds, VIEWPORT, Framing::FIT_MARGIN).units_per_pixel;
        let d = hausdorff(&exact.mapped(&camera.view), &grid.mapped(&camera.view)) / px;
        let label = format!(
            "{name}, x {}° z {}°: within a pixel at {}x{}: Hausdorff {d:.3} px; {} exact runs, {} grid runs",
            projection.x_angle as i64,
            projection.z_angle as i64,
            VIEWPORT.width,
            VIEWPORT.height,
            exact.count(),
            grid.count()
        );
        (exact.count() > 0 && grid.count() > 0 && d < 1.0, label)
    });
    for (ok, label) in outcomes {
        println!("{label}");
        checks.expect(ok, label);
    }
    checks.done();
}

/// "surfaces: an orbit frame of the forced torus, folds and all, keeps up",
/// its CPU half: the grid folds of each frame of the orbit.
///
/// Swift times a whole preview frame (`MetalRenderer.renderPreview`), the
/// median of seven under 25 ms, which is the GPU preview's and waits for it
/// (`an_orbit_frame_of_the_forced_torus_keeps_up_on_the_gpu`). What the
/// frame computes on the CPU at each new camera is the grid folds, from
/// normals laid out by the first frame: held here to the exact folds, to a
/// pixel of the navigator's framing, at each of the seven cameras, and to
/// Swift's 25 ms a frame, which the folds alone must keep within for the
/// frame to. The scene is the surface alone: its bounds are read off the
/// grid, and the trajectory Swift's scene carries is drawn by the GPU.
#[test]
fn an_orbit_frame_of_the_forced_torus_folds_and_all_keeps_up() {
    let forced = forced_preview_surface();
    let mut checks = Checks::default();
    let orbit = Orbit::matching(
        &ObliqueProjection { shear: 0.0, x_angle: -55.0, z_angle: 30.0, flip_x: false, y_scale: None },
        P3::new(0.0, 0.0, 0.0),
    );
    let scene = Scene::parametric_scene(forced.clone(), Vec::new(), orbit.camera(), PreviewMode::Drawing, 0.0);
    let bounds = scene.quick_bounds(20_000).expect("the forced torus has bounds");
    let mut navigator = Navigator::new(orbit, Framing::fitting(&bounds, VIEWPORT, Framing::FIT_MARGIN));
    // The first frame lays out the curves and the normals; the orbit after
    // it pays only for what the camera changes.
    let normals = forced.grid_normals();
    let mut frames: Vec<f64> = Vec::new();
    // The frames in turn, each frame's grid folds timed alone; then each
    // frame's exact folds, which nothing times and which are independent
    // of one another, side by side, and checked in turn.
    // The timed stretch runs alone: no other test of the file meanwhile.
    let timed = alone();
    let mut shown = Vec::new();
    for _ in 0..7 {
        navigator = navigator.applying(&Gesture::Orbit(Vec2::new(12.0, 3.0)), VIEWPORT);
        let camera = navigator.camera();
        let sight = camera.view.projection_direction();
        let t = Instant::now();
        let grid = forced.grid_fold_curves(&normals, sight).expect("a normal per grid point");
        frames.push(milliseconds(t.elapsed()));
        shown.push((camera, navigator.framing.units_per_pixel, grid));
    }
    drop(timed);
    let _beside = sharing();
    let outcomes = par_map(&shown, |(camera, units_per_pixel, grid)| {
        let sight = camera.view.projection_direction();
        let exact = forced.fold_curves(sight, None, &[]);
        let d = hausdorff(&exact.mapped(&camera.view), &grid.mapped(&camera.view)) / units_per_pixel;
        (exact.count(), grid.count(), d)
    });
    for (frame, (exact, grid, d)) in outcomes.into_iter().enumerate() {
        checks.expect(
            exact > 0 && grid > 0 && d < 1.0,
            format!(
                "frame {frame}: the grid folds are the exact folds, to a pixel: Hausdorff {d:.3} px; {exact} exact runs, \
                 {grid} grid runs"
            ),
        );
    }
    let mut sorted = frames.clone();
    sorted.sort_by(|a, b| real::total_order(*a, *b));
    let median = sorted[sorted.len() / 2];
    let listed = frames.iter().map(|f| format!("{f:.1}")).collect::<Vec<_>>().join(", ");
    println!("grid folds: median {median:.1} ms of {listed}");
    checks.expect(
        median < 25.0,
        format!("a frame's folds at a new camera take under 25 ms: median {median:.1} ms of {listed}"),
    );
    checks.done();
}

/// "surfaces: an orbit frame of the forced torus, folds and all, keeps up":
/// Swift's bound, a whole preview frame (`MetalRenderer.renderPreview`) at a
/// new camera, median under 25 ms. Not ported: a whole frame is the page's
/// now, the engine's draft in the Worker and the GPU's passes, and the
/// forced torus's fit (80 s natively) is past what the browser lane waits
/// for; the grid folds, the CPU half, are held to 25 ms above
/// (crates/FINDINGS.md, "Swift's GPU suites in the browser lane").
#[test]
#[ignore = "needs a GPU, and the forced torus's fit is past the browser lane's time"]
fn an_orbit_frame_of_the_forced_torus_keeps_up_on_the_gpu() {}

// MARK: - a surface drawing, described and built

/// A forced request at grid 256, as the suite writes `forcedRequest`.
fn forced_request(radius: f64, turns: i64, lines: Option<(i64, i64)>) -> SurfaceRequest {
    SurfaceRequest {
        grid: 256,
        lines,
        ..SurfaceRequest::new(Shape::Forced {
            system: "jerk".to_string(),
            fit: 8000.0,
            harmonics: 24,
            radial: 0,
            axial: 1,
            radius,
            turns,
        })
    }
}

fn sn_request(m: f64, big_r: f64, r: f64, grid: i64) -> SurfaceRequest {
    SurfaceRequest {
        grid: 256,
        ..SurfaceRequest::new(Shape::Sn { modulus: m, major: big_r, minor: r, grid, conformal: false })
    }
}

/// Conformal sn, its minor radius the one its modulus pins.
fn pinned(m: f64, big_r: f64, grid: i64) -> SurfaceRequest {
    SurfaceRequest {
        grid: 256,
        ..SurfaceRequest::new(Shape::Sn {
            modulus: m,
            major: big_r,
            minor: Shape::conformal_minor(m, big_r),
            grid,
            conformal: true,
        })
    }
}

fn build(request: &SurfaceRequest, previous: Option<&SurfaceDrawing>) -> SurfaceDrawing {
    SurfaceDrawing::build(request, previous, false).unwrap_or_else(|e| panic!("{request:?} builds: {e}"))
}

fn names(d: &SurfaceDrawing) -> Vec<&str> {
    d.layers.iter().map(|l| l.spec.name.as_str()).collect()
}

/// "drawings: a surface request builds its drawing, and an edit keeps what it can"
///
/// Left out, for Phase 7: Swift's last check, that a preview drawn after
/// each edit by the renderer that drew the one before is the preview a
/// fresh renderer draws (`MetalRenderer.renderPreview`, compared as PNG
/// bytes) -- what an edit can leave stale is the renderer's textures,
/// normals and curves.
#[test]
fn a_surface_request_builds_its_drawing_and_an_edit_keeps_what_it_can() {
    let mut checks = Checks::default();
    // The fit and the re-place are timed against each other, alone: no
    // other test of the file runs meanwhile.
    let timed = alone();
    let t = Instant::now();
    let first = build(&forced_request(2.5, 30, None), None);
    let fitted = t.elapsed();
    let t = Instant::now();
    let second = build(&forced_request(3.0, 30, None), Some(&first));
    let kept = t.elapsed();
    drop(timed);
    checks.expect(
        first.torus().is_some() && first.embedded && names(&first) == ["trajectory", "folds"],
        format!("a forced drawing is its trajectory and its folds, on an embedded torus: {:?}", names(&first)),
    );
    let tight = build(&forced_request(0.6, 30, None), Some(&first));
    checks.expect(
        !tight.embedded && tight.surface.outward.is_none(),
        "a ring too tight for its slices passes through its axis, and says so",
    );
    checks.expect(
        second.revolution.as_ref().map(|r| r.radius) == Some(3.0) && kept < fitted / 4,
        format!(
            "moving the revolution re-places the torus without refitting it: {:.1} ms to fit, {:.1} ms to re-place",
            milliseconds(fitted),
            milliseconds(kept)
        ),
    );
    println!("{:.1} ms to fit, {:.1} ms to re-place", milliseconds(fitted), milliseconds(kept));

    let torus = build(
        &SurfaceRequest {
            grid: 128,
            lines: Some((12, 6)),
            style: Style { folds: 0.0, ..Style::default() },
            ..SurfaceRequest::new(Shape::Torus { major: 2.0, minor: 1.0 })
        },
        None,
    );
    checks.expect(
        names(&torus) == ["lines"] && torus.surface.outward.is_some(),
        "a fold width of zero leaves the folds out",
    );
    let spindle =
        build(&SurfaceRequest { grid: 128, ..SurfaceRequest::new(Shape::Torus { major: 1.0, minor: 1.2 }) }, None);
    checks.expect(spindle.surface.outward.is_none(), "a torus with R ≤ r bounds nothing, so hides nothing");

    // An edit, built from the drawing before it, is exactly the drawing built
    // from nothing -- so the window's drawing depends on its request alone
    // -- and keeps the surface's identity exactly when the surface is
    // unchanged.
    fn same(a: &SurfaceDrawing, b: &SurfaceDrawing) -> bool {
        a.surface.positions == b.surface.positions
            && a.embedded == b.embedded
            && a.layers.iter().map(|l| &l.spec).eq(b.layers.iter().map(|l| &l.spec))
            && a.layers.iter().map(|l| &l.paths).eq(b.layers.iter().map(|l| &l.paths))
    }
    // Each edit is three builds, independent of every other edit: listed in
    // turn, built side by side, and checked in turn.
    let mut edits: Vec<(&str, SurfaceRequest, SurfaceRequest, bool)> = Vec::new();
    let mut edit = |label: &'static str, from: &SurfaceRequest, to: &SurfaceRequest, keeps_surface: bool| {
        edits.push((label, from.clone(), to.clone(), keeps_surface));
    };
    let ring = SurfaceRequest {
        grid: 128,
        lines: Some((12, 6)),
        ..SurfaceRequest::new(Shape::Torus { major: 2.0, minor: 1.0 })
    };
    let mut thicker = ring.clone();
    thicker.style.lines = 0.5;
    let mut more = ring.clone();
    more.lines = Some((24, 12));
    edit("torus, line width", &ring, &thicker, true);
    edit("torus, line counts", &ring, &more, true);
    edit(
        "torus, radii",
        &ring,
        &SurfaceRequest {
            grid: 128,
            lines: Some((12, 6)),
            ..SurfaceRequest::new(Shape::Torus { major: 3.0, minor: 0.5 })
        },
        false,
    );
    let sn = sn_request(0.64, 2.0, 1.0, 200);
    edit("sn, radii", &sn, &sn_request(0.64, 2.5, 0.8, 200), false);
    edit("sn, grid", &sn, &sn_request(0.64, 2.0, 1.0, 150), true);
    edit("sn, modulus", &sn, &sn_request(0.5, 2.0, 1.0, 200), false);
    // Conformal, sn's contours are re-placed on the torus its modulus
    // pins, parametrized isothermally round the tube; off again, on the
    // same torus in its angles.
    let conformal_on = sn.setting(Field::Conformal, 1.0, &surface_key).expect("sn turns conformal");
    edit("sn, conformal on", &sn, &conformal_on, false);
    let conformal_off =
        pinned(0.64, 2.0, 200).setting(Field::Conformal, 0.0, &surface_key).expect("conformal sn turns plain");
    edit("sn, conformal off", &pinned(0.64, 2.0, 200), &conformal_off, false);
    edit("conformal sn, modulus", &pinned(0.64, 2.0, 200), &pinned(0.5, 2.0, 200), false);
    edit("conformal sn, major radius", &pinned(0.64, 2.0, 200), &pinned(0.64, 2.5, 200), false);
    edit("conformal sn, grid", &pinned(0.64, 2.0, 200), &pinned(0.64, 2.0, 150), true);
    let forced = forced_request(2.5, 30, None);
    edit("forced, turns", &forced, &forced_request(2.5, 19, None), true);
    edit("forced, lines", &forced, &forced_request(2.5, 30, Some((12, 6))), true);
    edit("forced, radius", &forced, &forced_request(3.0, 30, None), false);

    // A linear curve is curves alone, on any torus: turning it on, moving its
    // slope and adding strands all keep the surface. Its slope moves
    // under a slider, so this is the edit that must cost only its own
    // vertices.
    let mut wound = ring.clone();
    wound.linear_curves = Some(LinearCurves { slope: 0.4, turns: 24, count: 1 });
    let mut steeper = wound.clone();
    if let Some(l) = &mut steeper.linear_curves {
        l.slope = 0.618;
    }
    let mut braided = wound.clone();
    if let Some(l) = &mut braided.linear_curves {
        l.count = 3;
    }
    edit("torus, linear curve on", &ring, &wound, true);
    edit("torus, linear curve slope", &wound, &steeper, true);
    edit("torus, linear curve strands", &wound, &braided, true);
    let mut forced_wound = forced.clone();
    forced_wound.linear_curves = Some(LinearCurves { slope: 1.0 / 3.0, turns: 24, count: 2 });
    let mut forced_steeper = forced_wound.clone();
    if let Some(l) = &mut forced_steeper.linear_curves {
        l.slope = 0.5;
    }
    edit("forced, linear curve on", &forced, &forced_wound, true);
    edit("forced, linear curve slope", &forced_wound, &forced_steeper, true);
    let beside = sharing();
    let outcomes = par_map(&edits, |(_, from, to, keeps_surface)| {
        let before = build(from, None);
        let edited = build(to, Some(&before));
        let fresh = build(to, None);
        same(&edited, &fresh) && (edited.content == before.content) == *keeps_surface
    });
    drop(beside);
    for ((label, _, _, keeps_surface), ok) in edits.iter().zip(outcomes) {
        checks.expect(
            ok,
            format!(
                "{label}: the edit is the drawing built afresh, and {}",
                if *keeps_surface { "keeps the surface" } else { "has a new surface" }
            ),
        );
    }
    let wound_drawing = build(&forced_wound, None);
    checks.expect(
        names(&wound_drawing) == ["trajectory", "linearCurves", "folds"],
        format!(
            "a forced drawing's linear curve is drawn between its trajectory and its folds: {:?}",
            names(&wound_drawing)
        ),
    );
    // The linear curve's fields edit the linear curve and nothing else, and are
    // absent until there is one.
    let mut by_field = forced.clone();
    by_field.set(Field::Slope, 0.7);
    checks.expect(
        by_field.get(Field::Slope).is_none() && by_field == forced,
        "without a linear curve, its fields are nil and setting them changes nothing",
    );
    let mut by_field = forced_wound.clone();
    by_field.set(Field::Slope, 0.7);
    by_field.set(Field::Turns, 12.0);
    by_field.set(Field::Strands, 4.0);
    checks.expect(
        by_field.linear_curves == Some(LinearCurves { slope: 0.7, turns: 12, count: 4 })
            && by_field.shape == forced_wound.shape,
        "and with one, they are its slope, turns and strands",
    );

    // The trajectory is a linear curve: its turns are its own, and the whole
    // of it -- turns, coordinates, slope -- comes from the fit, with
    // nothing integrated.
    let thirty = build(&forced_request(2.5, 30, None), None);
    let trajectory = &thirty.layers[0].paths;
    let c = trajectory.coords.as_ref().expect("the trajectory carries coordinates");
    let spans: Vec<f64> = (0..trajectory.count())
        .map(|k| (c[trajectory.offsets[k + 1] - 1].x() - c[trajectory.offsets[k]].x()) / (2.0 * PI))
        .collect();
    let total: f64 = spans.iter().sum();
    checks.expect(
        (total - 30.0).abs() < 1e-9 && spans.len() == 4 && spans.iter().all(|&s| s <= 8.0 + 1e-9),
        format!(
            "thirty turns of trajectory are thirty turns of the linear curve, in four paths: {total:.6} turns in {} paths",
            spans.len()
        ),
    );

    let lorenz = SurfaceDrawing::build(
        &SurfaceRequest::new(Shape::Forced {
            system: "lorenz".to_string(),
            fit: 100.0,
            harmonics: 4,
            radial: 0,
            axial: 1,
            radius: 2.5,
            turns: 1,
        }),
        None,
        false,
    );
    checks.expect(lorenz.is_err(), "an unknown system is refused by name");
    checks.done();
}
