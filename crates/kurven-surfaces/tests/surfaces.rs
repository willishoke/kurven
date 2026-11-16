//! Surfaces.swift's suites on parametric surfaces and visibility by surface
//! coordinate (`surfaceTests`), its bakes of parametric scenes
//! (`surfaceBakeTests`) and its fold curves (`foldTests`), with Swift's
//! tolerances throughout. The GPU suite (`surfaceGPUTests`) waits for the
//! GPU preview.
//!
//! Nothing here has a Python oracle, and nothing needs one: the torus of
//! revolution has exact visibility. A point is hidden when the ray from it
//! toward the eye meets the torus again, and along that ray the torus's
//! implicit equation is a quartic with a known root at the point itself --
//! so a cubic, whose real roots are bracketed by its critical points.

// Swift's `guard x > 0 else` refuses a NaN, and so does `if !(x > 0.0)`.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

mod common;

use common::{compare_with_ray_cast, test_camera, torus_sees, torus_sees_from_outside, view_bounds};
use kurven_bake::{BakeOptions, bake, bake_with_image};
use kurven_core::camera::Camera;
use kurven_core::depth::DepthFrame;
use kurven_core::hidden_line::HiddenLine;
use kurven_core::manifest::{HeightPolicy, LayerRole, LayerSource, LayerSpec, ObliqueProjection};
use kurven_core::parametric::{ParamAxis, ParamScalar, ParametricSurface};
use kurven_core::polyline::{Layer, PolylineSet};
use kurven_core::scene::{PreviewMode, Scene};
use kurven_core::simd::{Vec2, Vec3};
use kurven_core::spaces::{P2, P3, ParamSpace, Transform, ViewSpace, WorldSpace};
use kurven_core::splitmix::SplitMix;
use kurven_core::surface_image::SurfaceImage;
use kurven_core::surface_request::{Shape, Style, SurfaceRequest};
use kurven_core::svg;
use kurven_core::visibility::SurfaceVisibility;
use kurven_math::{real, swift};
use kurven_surfaces::SurfaceDrawing;
use std::f64::consts::PI;
use std::sync::Arc;

/// The planar length of a view-space segment.
fn planar(a: P3<ViewSpace>, b: P3<ViewSpace>) -> f64 {
    Vec2::new(b.x() - a.x(), b.y() - a.y()).length()
}

/// The torus the suites share, at 384 x 192.
fn shared_torus() -> ParametricSurface {
    ParametricSurface::torus(2.0, 1.0, (384, 192)).expect("a round torus")
}

/// `HiddenLine.clip(_:on:)`, Swift's `onFolds` and `onCrease` false.
fn clip(paths: &PolylineSet<ViewSpace>, vis: &SurfaceVisibility) -> PolylineSet<kurven_core::spaces::PageSpace> {
    HiddenLine::clip_on(paths, vis, false, false).expect("curves with coordinates")
}

/// The visible length of curves on the torus `(R, r)`, by the exact ray-cast:
/// every segment divided `k` ways, each piece kept when its midpoint is seen.
fn ray_cast_length(
    curves: &PolylineSet<WorldSpace>,
    surface: &ParametricSurface,
    big_r: f64,
    r: f64,
    view: &Transform<WorldSpace, ViewSpace>,
    k: usize,
) -> f64 {
    let eye = -view.projection_direction();
    let coords = curves.coords.as_ref().expect("curves with coordinates");
    let mut total = 0.0;
    for path in 0..curves.count() {
        for i in curves.offsets[path]..(curves.offsets[path + 1] - 1) {
            let (a, b) = (coords[i].v, coords[i + 1].v);
            let at = |t: f64| (surface.map)(P2::from_vec(a + (b - a) * t)).position;
            for j in 0..k {
                let (t0, t1) = (j as f64 / k as f64, (j + 1) as f64 / k as f64);
                if !torus_sees(at(0.5 * (t0 + t1)), big_r, r, eye) {
                    continue;
                }
                let p = view.apply(P3::from_vec(at(t0)));
                let q = view.apply(P3::from_vec(at(t1)));
                total += planar(p, q);
            }
        }
    }
    total
}

/// "surfaces: curves can carry where on a surface it lies"
#[test]
fn curves_can_carry_where_on_a_surface_it_lies() {
    let a: Vec<P3<WorldSpace>> = vec![P3::new(0.0, 0.0, 0.0), P3::new(1.0, 0.0, 0.0), P3::new(2.0, 0.0, 0.0)];
    let ca: Vec<P2<ParamSpace>> = vec![P2::new(0.0, 0.0), P2::new(0.5, 0.0), P2::new(1.0, 0.0)];
    let set =
        PolylineSet::from_paths_with_coords(&[a.clone(), vec![P3::new(9.0, 9.0, 9.0)]], &[ca, vec![P2::new(9.0, 9.0)]]);
    assert!(
        set.count() == 1 && set.coords.as_ref().map(|c| c.len()) == Some(3),
        "a singleton path drops its coordinates with it"
    );
    let moved = set.mapped(&Transform::<WorldSpace, ViewSpace>::identity());
    assert!(moved.coords == set.coords, "a camera moves the curves and not its coordinates");
    assert!(
        PolylineSet::from_paths(&[a]).coords.is_none(),
        "curves without coordinates says so rather than inventing them"
    );
}

/// "surfaces: which side is outside is computed, not guessed"
#[test]
fn which_side_is_outside_is_computed_not_guessed() {
    let t = ParametricSurface::torus(2.0, 1.0, (64, 32)).expect("a round torus");
    let out = (t.map)(P2::new(0.0, 0.0)).normal() * t.outward.unwrap_or(0.0);
    assert!(
        t.outward.is_some() && out.x > 0.0,
        "the torus's outward normal at its outer equator points away from the axis: outward {:?}",
        t.outward
    );
    // The same torus with the tube angle run backwards: the parametrization's
    // normal flips, and the outward one must not.
    let map = t.map.clone();
    let flipped = ParametricSurface::new(
        ParamAxis::angle(64),
        ParamAxis::angle(32),
        true,
        None,
        None,
        Arc::new(move |c: P2<ParamSpace>| {
            let mut j = map(P2::new(c.x(), -c.y()));
            j.dv = -j.dv;
            j
        }),
    )
    .expect("the flipped torus has a volume");
    let out_flipped = (flipped.map)(P2::new(0.0, 0.0)).normal() * flipped.outward.unwrap_or(0.0);
    assert!(
        flipped.outward == Some(-t.outward.unwrap_or(0.0)) && out_flipped.x > 0.0,
        "reversing the parametrization reverses the sign and keeps the outside"
    );
    let horn = ParametricSurface::torus(1.0, 1.0, (64, 32)).expect("a horn torus");
    assert!(horn.outward.is_none(), "a horn torus passes through itself, so it bounds nothing and culls nothing");
}

/// "surfaces: the line of sight is the camera's kernel, not the gradient of depth"
#[test]
fn the_line_of_sight_is_the_cameras_kernel_not_the_gradient_of_depth() {
    let view = test_camera(-55.0, 30.0, 0.5);
    let d = view.projection_direction();
    let p = P3::<WorldSpace>::new(0.3, -0.7, 0.2);
    let a = view.apply(p);
    let b = view.apply(P3::plus(p, d));
    let dxy = planar(a, b);
    assert!(
        dxy < 1e-12 && b.z() < a.z(),
        "travelling it leaves view x and y alone and moves away from the eye: Δxy {dxy}, Δz {}",
        b.z() - a.z()
    );
    let c = view.m.columns;
    let gradient = -Vec3::new(c[0].z, c[1].z, c[2].z).normalized();
    let angle = real::acos(swift::min(1.0, gradient.dot(d))) * 180.0 / PI;
    assert!(angle > 1.0, "under a shear the two differ, which is why it matters: {angle:.1}°");
}

/// "surfaces: the rasterized coordinates land where their depth and pixel say"
#[test]
fn the_rasterized_coordinates_land_where_their_depth_and_pixel_say() {
    let torus = shared_torus();
    let view = test_camera(-60.0, 30.0, 0.0);
    let frame = DepthFrame::covering(&view_bounds(&torus, &view), 512);
    let image = SurfaceImage::rasterize(&torus, &view, frame);
    let (mut dz, mut dxy, mut covered) = (0.0, 0.0, 0i64);
    for r in 0..frame.rows {
        for c in 0..frame.cols {
            let Some(q) = image.coordinate(r, c) else { continue };
            covered += 1;
            let p = view.apply(P3::from_vec((torus.map)(q).position));
            let s = frame.coordinate(r, c);
            dz = swift::max(dz, (p.z() - image.depth.at(r, c)).abs());
            dxy = swift::max(dxy, Vec2::new(p.x() - s.x(), p.y() - s.y()).length());
        }
    }
    // The grid is a chord approximation of the torus: its sagitta at
    // 384 x 192 is (R + r)(1 − cos(π/384)) ≈ 1e-4 around the axis.
    assert!(covered > frame.rows * frame.cols / 5, "the torus covers the frame: {covered} pixels");
    assert!(dz < 1e-3, "a pixel's coordinate maps back to its depth: max {dz}");
    assert!(dxy < 1e-3, "and to its own sample point: max {dxy}");
}

/// "surfaces: visibility by coordinate agrees with the exact ray-cast"
#[test]
fn visibility_by_coordinate_agrees_with_the_exact_ray_cast() {
    // Two tori -- one with room in the hole, one fat enough that most of
    // it is hidden behind itself -- under cameras from nearly overhead to
    // nearly edge-on, one of them sheared as the plates are.
    let cameras = [
        ("elevation 60°", test_camera(-60.0, 30.0, 0.0)),
        ("nearly overhead", test_camera(-85.0, 10.0, 0.0)),
        ("nearly edge-on", test_camera(-10.0, 40.0, 0.0)),
        ("sheared", test_camera(-35.0, -20.0, 0.5)),
    ];
    let mut failures: Vec<String> = Vec::new();
    for (big_r, r) in [(2.0, 1.0), (1.25, 1.0)] {
        let surface = ParametricSurface::torus(big_r, r, (384, 192)).expect("a round torus");
        for (name, view) in &cameras {
            let frame = DepthFrame::covering(&view_bounds(&surface, view), 800);
            let image = SurfaceImage::rasterize(&surface, view, frame);
            let vis = SurfaceVisibility::new(surface.clone(), image, *view);
            let (wrong, unexplained, depth_wrong, example) = compare_with_ray_cast(&vis, big_r, r, view);
            let n = 20_000;
            let label = format!("R/r = {}, {name}", big_r / r);
            println!("{label}: {wrong} of {n} disagree, {unexplained} unexplained (depth test: {depth_wrong})");
            if unexplained != 0 {
                failures.push(format!(
                    "{label}: every disagreement is within two pixels of a visibility edge: {wrong} of {n} \
                     disagree, {unexplained} unexplained {example} (depth test: {depth_wrong})"
                ));
            }
            // The round torus bounds a solid sphere tracing can ask directly,
            // and then there is no pixel to be within.
            if !(vis.traces_spheres() && wrong == 0) {
                failures.push(format!("{label}: by sphere tracing, no disagreement at all: {wrong} of {n} disagree"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// "surfaces: sphere tracing on a displaced torus agrees with a fine walk up the line of sight"
#[test]
fn sphere_tracing_on_a_displaced_torus_agrees_with_a_fine_walk_up_the_line_of_sight() {
    // The torus in relief under a smooth bump field, truncated at 1: the
    // solid sphere tracing is held to is the tubular neighbourhood's own point test,
    // walked up the line of sight in steps of a ten-thousandth of the
    // solid, which is as fine as the oracle gets without being the
    // sphere tracing. A disagreement is excused where the walk's least gap is
    // within rounding of zero: a line grazing the surface, which no
    // test with a tolerance decides the same way twice.
    let big_r = 2.0;
    let r = 1.0;
    let base = ParametricSurface::torus(big_r, r, (256, 128)).expect("a round torus");
    let height: ParamScalar = Arc::new(|c: P2<ParamSpace>| {
        swift::min(0.5 * (1.0 + real::cos(3.0 * c.x()) * real::cos(2.0 * c.y())) + 0.3, 1.0)
    });
    let relief = base.displaced(height, 0.4, 1.0, true).expect("the displaced torus");
    assert!(relief.neighbourhood.is_some(), "the displaced torus keeps the torus's foot");
    let mut failures: Vec<String> = Vec::new();
    for (name, view) in [
        ("elevation 60°", test_camera(-60.0, 30.0, 0.0)),
        ("low, over the far wall", test_camera(-80.0, 40.0, 0.0)),
        ("sheared", test_camera(-35.0, -20.0, 0.5)),
    ] {
        let frame = DepthFrame::covering(&view_bounds(&relief, &view), 64);
        let vis = SurfaceVisibility::new(relief.clone(), SurfaceImage::rasterize(&relief, &view, frame), view);
        let e = relief.neighbourhood.as_ref().expect("a neighbourhood");
        let d = -view.projection_direction();
        let mut rng = SplitMix::new(0xC0FFEE);
        let (mut wrong, mut unexplained, mut tested, mut seen) = (0, 0, 0, 0);
        let mut example = String::new();
        while tested < 1500 {
            let c = P2::<ParamSpace>::new(rng.next(0.0, 2.0 * PI), rng.next(0.0, 2.0 * PI));
            let Some(f) = vis.facing(c) else { continue };
            if !(f > 0.0) {
                continue;
            }
            tested += 1;
            let q = (relief.map)(c).position;
            let traversal = vis.is_clear(q);
            if traversal {
                seen += 1;
            }
            // The walk: every sample from just off the surface to past
            // the ball, and the least gap it saw.
            let step = 1e-4 * e.radius;
            let mut t = 1e-5 * e.radius;
            let mut least = f64::INFINITY;
            let mut clear = true;
            while t < 2.0 * e.radius + q.length() {
                let (coord, distance) = (e.foot)(q + t * d);
                let gap = distance - (e.normal_offset)(coord);
                least = swift::min(least, gap);
                if gap < -1e-9 * e.radius {
                    clear = false;
                    break;
                }
                t += step;
            }
            if traversal == clear {
                continue;
            }
            wrong += 1;
            if least.abs() > 1e-6 * e.radius {
                unexplained += 1;
                if example.is_empty() {
                    example = format!(
                        "e.g. ({:.4}, {:.4}) sphere tracing {}, walk {}, least gap {:.2e}",
                        c.x(),
                        c.y(),
                        if traversal { "clear" } else { "hidden" },
                        if clear { "clear" } else { "hidden" },
                        least
                    );
                }
            }
        }
        println!("{name}: {seen} of {tested} seen, {wrong} disagree, {unexplained} unexplained");
        if !(seen > tested / 4 && seen < tested) {
            failures.push(format!("{name}: the sample has both seen and hidden points: {seen} of {tested} seen"));
        }
        if unexplained != 0 {
            failures.push(format!(
                "{name}: sphere tracing and the walk agree except where the line grazes: {wrong} of {tested} \
                 disagree, {unexplained} unexplained {example}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// "surfaces: a segment across a fold is cut on the fold"
#[test]
fn a_segment_across_a_fold_is_cut_on_the_fold() {
    let torus = shared_torus();
    let view = test_camera(-35.0, -20.0, 0.5);
    let frame = DepthFrame::covering(&view_bounds(&torus, &view), 256);
    let vis = SurfaceVisibility::new(torus.clone(), SurfaceImage::rasterize(&torus, &view, frame), view);
    let mut rng = SplitMix::new(7);
    let (mut found, mut worst, mut straddles) = (0, 0.0, true);
    while found < 200 {
        let a = P2::<ParamSpace>::new(rng.next(0.0, 2.0 * PI), rng.next(0.0, 2.0 * PI));
        let b = P2::<ParamSpace>::new(a.x() + rng.next(-0.2, 0.2), a.y() + rng.next(-0.2, 0.2));
        let Some(t) = vis.fold_crossing(a, b) else { continue };
        found += 1;
        let at = |s: f64| P2::<ParamSpace>::from_vec(a.v + (b.v - a.v) * s);
        worst = swift::max(worst, vis.facing(at(t)).unwrap_or(1.0).abs());
        let before = vis.facing(at(swift::max(t - 1e-9, 0.0))).expect("a torus faces");
        let after = vis.facing(at(swift::min(t + 1e-9, 1.0))).expect("a torus faces");
        if (before > 0.0) == (after > 0.0) {
            straddles = false;
        }
    }
    assert!(worst < 1e-10, "the cut is where the surface is edge-on: max |cos| {worst} over {found} segments");
    assert!(straddles, "and it faces the eye on exactly one side of the cut");
}

/// "surfaces: clipped circles on a torus have the ray-cast's visible length"
#[test]
fn clipped_circles_on_a_torus_have_the_ray_casts_visible_length() {
    let torus = shared_torus();
    let mut failures: Vec<String> = Vec::new();
    for (name, view) in [("elevation 60°", test_camera(-60.0, 30.0, 0.0)), ("sheared", test_camera(-35.0, -20.0, 0.5))]
    {
        let frame = DepthFrame::covering(&view_bounds(&torus, &view), 1000);
        let vis = SurfaceVisibility::new(torus.clone(), SurfaceImage::rasterize(&torus, &view, frame), view);
        // Twenty-four circles each way round, 1500 vertices apiece.
        let mut paths: Vec<Vec<P3<ViewSpace>>> = Vec::new();
        let mut coords: Vec<Vec<P2<ParamSpace>>> = Vec::new();
        let k = 1500;
        for line in 0..24 {
            let fixed = 2.0 * PI * (line as f64 + 0.5) / 24.0;
            for around in [true, false] {
                let mut p: Vec<P3<ViewSpace>> = Vec::new();
                let mut c: Vec<P2<ParamSpace>> = Vec::new();
                for i in 0..=k {
                    let s = 2.0 * PI * i as f64 / k as f64;
                    let q = if around { P2::new(s, fixed) } else { P2::new(fixed, s) };
                    c.push(q);
                    p.push(view.apply(P3::from_vec((torus.map)(q).position)));
                }
                paths.push(p);
                coords.push(c);
            }
        }
        let clipped = clip(&PolylineSet::from_paths_with_coords(&paths, &coords), &vis);
        // The truth, by the midpoint of every segment of a far finer
        // division of the same circles.
        let mut truth = 0.0;
        let fine = 20_000;
        for line in 0..24 {
            let fixed = 2.0 * PI * (line as f64 + 0.5) / 24.0;
            for around in [true, false] {
                let at = |s: f64| -> P2<ParamSpace> { if around { P2::new(s, fixed) } else { P2::new(fixed, s) } };
                for i in 0..fine {
                    let s0 = 2.0 * PI * i as f64 / fine as f64;
                    let s1 = 2.0 * PI * (i + 1) as f64 / fine as f64;
                    let mid = (torus.map)(at(0.5 * (s0 + s1))).position;
                    if !torus_sees(mid, 2.0, 1.0, -vis.sight) {
                        continue;
                    }
                    let a = view.apply(P3::from_vec((torus.map)(at(s0)).position));
                    let b = view.apply(P3::from_vec((torus.map)(at(s1)).position));
                    truth += planar(a, b);
                }
            }
        }
        let length = clipped.total_length();
        let error = (length - truth).abs() / truth;
        let detail = format!("{length:.4} vs {truth:.4}, {:.3}% off", 100.0 * error);
        println!("{name}: {detail}");
        if !(error < 0.002) {
            failures.push(format!("{name}: visible length matches the ray-cast: {detail}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// "metal: the GPU coordinate pass is the CPU rasterizer's" (`surfaceGPUTests`):
/// Metal's coordinate pass held to the CPU rasterizer.
#[test]
#[ignore = "PORT: phase 7 -- the GPU preview"]
fn the_gpu_coordinate_pass_is_the_cpu_rasterizers() {}

/// "bake: `surface sn` bakes the same SVG at resolutions 1000 and 3000"
#[test]
fn surface_sn_bakes_the_same_svg_at_resolutions_1000_and_3000() {
    // The drawing `kurven-cli surface sn` bakes, with its defaults: every
    // curve on the round torus is tested by its facing and sphere
    // tracing, and no pixel decides any of it.
    let mut request =
        SurfaceRequest::new(Shape::Sn { modulus: 0.64, major: 2.0, minor: 1.0, grid: 600, conformal: false });
    request.grid = 1024;
    request.name = "sn".to_string();
    request.style = Style { lines: 0.3, trajectory: 0.08, folds: 0.6, linear_curves: 0.3 };
    let scene = SurfaceDrawing::build(&request, None, false)
        .expect("the sn drawing")
        .scene(Camera::oblique(&SurfaceRequest::PROJECTION));
    let svgs: Vec<String> = [1000, 3000]
        .iter()
        .map(|&r| {
            let baked = bake(&scene, &BakeOptions::new(r)).expect("a bake");
            svg::render(&baked.strokes, svg::FIGURE_WIDTH, svg::MARGIN, svg::DECIMALS)
        })
        .collect();
    let strokes = svgs[0].matches("<polyline").count();
    assert!(
        strokes > 100 && svgs[0] == svgs[1],
        "the two SVGs are one, byte for byte: {strokes} strokes, {} and {} bytes",
        svgs[0].len(),
        svgs[1].len()
    );
}

/// "bake: a parametric scene bakes its curves by where it lies on the surface"
#[test]
fn a_parametric_scene_bakes_its_curves_by_where_it_lies_on_the_surface() {
    let torus = ParametricSurface::torus(2.0, 1.0, (512, 256)).expect("a round torus");
    let curves = torus.coordinate_curves((24, 12), 2048);
    let spec = LayerSpec::new(
        "lines",
        LayerRole::Scaffold,
        LayerSource::CoordinateCurves { u: 24, v: 12 },
        0.3,
        HeightPolicy::Surface,
    );
    // An oblique camera, shear and all, as a real drawing would have it.
    let camera =
        Camera::oblique(&ObliqueProjection { shear: 0.5, x_angle: -50.0, z_angle: 25.0, flip_x: false, y_scale: None });
    let scene = Scene::parametric_scene(
        torus.clone(),
        vec![Layer { spec: spec.clone(), paths: curves.clone() }],
        camera,
        PreviewMode::Drawing,
        0.03,
    );
    let single = BakeOptions { tiles: Some(1), ..BakeOptions::new(2400) };
    // Rendered with its image, as Swift's bake always is: the torus is
    // sphere traced, so `bake` alone renders none, and the stitching is
    // what is asked of it below.
    let whole = bake_with_image(&scene, &single).expect("a bake");
    let truth = ray_cast_length(&curves, &torus, 2.0, 1.0, &camera.view, 8);
    let length = whole.strokes.total_length();
    let error = (length - truth).abs() / truth;
    println!("by coordinate: {length:.4} vs {truth:.4}, {:.3}% off", 100.0 * error);
    assert!(
        whole.surface.is_some() && error < 0.002,
        "the baked drawing carries the ray-cast's visible length: {length:.4} vs {truth:.4}, {:.3}% off",
        100.0 * error
    );

    // Tiling splits the render and never the clip, and depth and
    // coordinates are stitched by the same code over the same sub-frames.
    let split = bake_with_image(&scene, &BakeOptions { tiles: Some(3), ..BakeOptions::new(2400) }).expect("a bake");
    let same = whole.strokes.layers.len() == split.strokes.layers.len()
        && whole.strokes.layers.iter().zip(&split.strokes.layers).all(|(a, b)| a.1 == b.1);
    let (whole_coords, split_coords) =
        (&whole.surface.as_ref().expect("an image").coords, &split.surface.as_ref().expect("an image").coords);
    let moved = whole_coords
        .iter()
        .zip(split_coords)
        .filter(|(a, b)| {
            let (dx, dy) = (f64::from(a[0] - b[0]), f64::from(a[1] - b[1]));
            !(a[0].is_nan() && b[0].is_nan())
                && (a[0].is_nan() != b[0].is_nan() || real::sqrt(dx * dx + dy * dy) > 1e-3)
        })
        .count();
    assert!(
        same,
        "a tiled bake draws what a single-pass bake draws: {moved} of {} pixel coordinates differ",
        whole_coords.len()
    );

    // A linear curve is tested the same way, by the same coordinate image:
    // curves the surface passes never saw, placed from the grid rather
    // than the map, still carries the ray-cast's visible length.
    let linear_curves = torus.linear_curves(2.0 / 5.0, 5, 3, None);
    let wound = bake(
        &scene.drawing(vec![Layer {
            spec: LayerSpec::new(
                "linearCurves",
                LayerRole::Scaffold,
                LayerSource::LinearCurves { slope: 0.4, turns: 5, count: 3 },
                0.3,
                HeightPolicy::Surface,
            ),
            paths: linear_curves.clone(),
        }]),
        &single,
    )
    .expect("a bake");
    let linear_curve_truth = ray_cast_length(&linear_curves, &torus, 2.0, 1.0, &camera.view, 8);
    let wound_length = wound.strokes.total_length();
    let linear_curve_error = (wound_length - linear_curve_truth).abs() / linear_curve_truth;
    assert!(
        linear_curve_error < 0.002,
        "a linear curve bakes to the ray-cast's visible length too: {wound_length:.4} vs {linear_curve_truth:.4}, \
         {:.3}% off",
        100.0 * linear_curve_error
    );

    // The same lines without coordinates are tested by depth, at the
    // scene's depth tolerance -- still a drawing, and a measurably worse one.
    let bare = Layer { spec, paths: PolylineSet::new(curves.vertices.clone(), curves.offsets.clone(), None) };
    let by_depth = bake(&scene.drawing(vec![bare]), &single).expect("a bake");
    let depth_error = (by_depth.strokes.total_length() - truth).abs() / truth;
    assert!(
        by_depth.strokes.path_count() > 0 && depth_error > error,
        "curves without coordinates falls back to the depth test: {:.3}% off by depth, {:.3}% by coordinate",
        100.0 * depth_error,
        100.0 * error
    );

    // With no tubular neighbourhood there is no exact test, and the
    // coordinate image the bake renders stands in for one: the bake
    // draws what clipping by that image draws.
    let unbounded =
        ParametricSurface::new(torus.u, torus.v, true, None, None, torus.map.clone()).expect("the torus has a volume");
    let by_image = bake(
        &Scene::parametric_scene(unbounded.clone(), scene.layers().to_vec(), camera, PreviewMode::Drawing, 0.03),
        &single,
    )
    .expect("a bake");
    let clipped = clip(
        &curves.mapped(&camera.view),
        &SurfaceVisibility::new(unbounded, by_image.surface.clone().expect("an image"), camera.view),
    );
    let image_error = (by_image.strokes.total_length() - truth).abs() / truth;
    assert!(
        by_image.strokes.layers.first().map(|l| &l.1) == Some(&clipped) && image_error < 0.01,
        "a surface with no tubular neighbourhood bakes by its coordinate image: {} strokes, {:.3}% off the ray-cast",
        by_image.strokes.path_count(),
        100.0 * image_error
    );
}

/// "surfaces: fold curves lie on the folds, and are seen where the ray-cast sees them"
#[test]
fn fold_curves_lie_on_the_folds_and_are_seen_where_the_ray_cast_sees_them() {
    let mut failures: Vec<String> = Vec::new();
    for (big_r, r) in [(2.0, 1.0), (1.25, 1.0)] {
        let torus = ParametricSurface::torus(big_r, r, (512, 256)).expect("a round torus");
        for (name, x_angle, shear) in [("elevation 50°", -50.0, 0.0), ("sheared", -30.0, 0.5)] {
            let camera =
                Camera::oblique(&ObliqueProjection { shear, x_angle, z_angle: 25.0, flip_x: false, y_scale: None });
            let sight = camera.view.projection_direction();
            let folds = torus.fold_curves(sight, None, &[]);
            let coords = folds.coords.as_ref().expect("fold curves carry coordinates");
            let mut worst = 0.0;
            for &c in coords {
                let n = (torus.map)(c).normal();
                worst = swift::max(worst, n.dot(sight).abs() / n.length());
            }
            let label = format!("R/r = {}, {name}", big_r / r);
            if !(folds.count() > 0 && worst < 1e-9) {
                failures.push(format!("{label}: every vertex is edge-on: max |cos| {worst}"));
            }

            let spec = LayerSpec::new("folds", LayerRole::Outline, LayerSource::FoldCurves, 0.6, HeightPolicy::Surface);
            let scene = Scene::parametric_scene(
                torus.clone(),
                vec![Layer { spec, paths: PolylineSet::empty() }],
                camera,
                PreviewMode::Drawing,
                0.0,
            );
            let baked = bake(&scene, &BakeOptions::new(2400)).expect("a bake");
            // On a fold the line of sight is tangent to the torus, so the
            // ray-cast is asked a hair outside it, where it has an answer.
            let eye = -sight;
            let mut truth = 0.0;
            for path in 0..folds.count() {
                for i in folds.offsets[path]..(folds.offsets[path + 1] - 1) {
                    let mid = (torus.map)(P2::from_vec(0.5 * (coords[i].v + coords[i + 1].v)));
                    let out = mid.normal().normalized() * torus.outward.unwrap_or(1.0);
                    if !torus_sees_from_outside(mid.position + 1e-7 * (big_r + r) * out, big_r, r, eye) {
                        continue;
                    }
                    let a = camera.view.apply(folds.vertices[i]);
                    let b = camera.view.apply(folds.vertices[i + 1]);
                    truth += planar(a, b);
                }
            }
            // What the front-most test cannot see is a fold running on
            // behind its own sheet past a cusp: there the sheet in front
            // is the fold's own neighbourhood for the first couple of
            // pixels. So the bound is a few pixels per cusp, not zero.
            // (Swift reads the pixel off the bake's depth buffer; the port's
            // bake frames the coordinate image the same way.)
            let unit = baked.frame.expect("a frame").units_per_pixel();
            let px = swift::max(unit.x, unit.y);
            let length = baked.strokes.total_length();
            let excess = length - truth;
            let detail = format!("{length:.4} vs {truth:.4}, {:+.1} px", excess / px);
            println!("{label}: max |cos| {worst}, {detail}");
            if !(excess.abs() < 16.0 * px) {
                failures.push(format!("{label}: the visible fold length is the ray-cast's, to a few pixels: {detail}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
