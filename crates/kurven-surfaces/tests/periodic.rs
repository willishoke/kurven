//! Surfaces.swift's suites on the truncation edge as a crease (`creaseTests`),
//! on doubly periodic functions drawn on their tori (`periodicTorusTests`)
//! and on linear curves (`linearCurveTests`), with Swift's tolerances
//! throughout. The two GPU preview suites (`surfacePreviewTests`,
//! `edgeOnPreviewTests`) wait for the GPU preview.
//!
//! `periodicTorusTests` opens with "math: the quarter periods are the
//! complete elliptic integral", which is kurven-math's
//! `the_quarter_periods_are_the_complete_elliptic_integral`
//! (crates/kurven-math/tests/dynamics.rs) and is not repeated here.

// Swift's `guard x > 0 else` refuses a NaN, and so does `if !(x > 0.0)`.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

mod common;

use common::torus_sees;
use kurven_bake::{BakeOptions, bake};
use kurven_core::camera::Camera;
use kurven_core::hidden_line::HiddenLine;
use kurven_core::manifest::{ContourField, LayerSource, ObliqueProjection};
use kurven_core::parametric::{ParamScalar, ParametricSurface};
use kurven_core::polyline::{Layer, PolylineSet};
use kurven_core::scene::{PreviewMode, Scene};
use kurven_core::simd::{Vec2, Vec3};
use kurven_core::spaces::{P2, P3, PageSpace, ParamSpace, Transform, ViewSpace, WorldSpace};
use kurven_core::surface_request::SurfaceRequest;
use kurven_core::visibility::SurfaceVisibility;
use kurven_landscape::{Drawing, PeriodicOptions, PeriodicTorus};
use kurven_math::{Complex, Jacobi, real, swift};
use std::f64::consts::PI;
use std::sync::Arc;

/// Swift's `Heightfield.parallelMap`: the same map, spread over threads,
/// each item computed exactly as it would be in a plain loop.
fn par_map<T: Sync, R: Send>(items: &[T], body: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = items.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> =
            items.chunks(chunk).map(|c| scope.spawn(|| c.iter().map(&body).collect::<Vec<R>>())).collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

/// Record a failed expectation, and print its detail either way, as
/// `Check.expect` reports every measure and goes on with the suite.
fn expect(failures: &mut Vec<String>, ok: bool, what: &str, detail: &str) {
    println!("{} {what}: {detail}", if ok { "ok  " } else { "FAIL" });
    if !ok {
        failures.push(format!("{what}: {detail}"));
    }
}

/// The planar length of a view-space segment.
fn planar(a: P3<ViewSpace>, b: P3<ViewSpace>) -> f64 {
    Vec2::new(b.x() - a.x(), b.y() - a.y()).length()
}

/// The visible length of curves on the torus `(R, r)`, by the exact ray-cast:
/// every segment divided `k` ways, each piece kept when its midpoint is seen
/// (Surfaces.swift's `rayCastLength`).
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

/// The sheets of a parametric surface on a line of sight, found by brute
/// force, with no image in it.
///
/// Every grid cell whose box in view, widened by half, holds the line of
/// sight is solved onto it by Newton's method from the cell's centre, and
/// gives the sheet over that cell where the solve settles within a cell and
/// a half of the centre. The cells are binned by their boxes, so a line of
/// sight asks only the cells about it.
struct SheetSearch {
    surface: ParametricSurface,
    view: Transform<WorldSpace, ViewSpace>,
    boxes: Vec<(Vec2, Vec2)>,
    bins: Vec<Vec<i32>>,
    lo: Vec2,
    size: Vec2,
}

impl SheetSearch {
    const COUNT: usize = 400;

    fn new(surface: &ParametricSurface, view: Transform<WorldSpace, ViewSpace>) -> SheetSearch {
        let nu = surface.u.samples;
        let n = SheetSearch::COUNT;
        let grid: Vec<P3<ViewSpace>> = surface.positions.iter().map(|&p| view.apply(p)).collect();
        let mut boxes: Vec<(Vec2, Vec2)> = Vec::with_capacity(surface.u.cells() * surface.v.cells());
        for j in 0..surface.v.cells() {
            for i in 0..surface.u.cells() {
                let mut lo = Vec2::splat(f64::INFINITY);
                let mut hi = -lo;
                for (a, b) in [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)] {
                    let q = grid[surface.v.sample(b as i64) * nu + surface.u.sample(a as i64)];
                    lo = lo.min(Vec2::new(q.x(), q.y()));
                    hi = hi.max(Vec2::new(q.x(), q.y()));
                }
                boxes.push((lo - 0.5 * (hi - lo), hi + 0.5 * (hi - lo)));
            }
        }
        let lo = boxes.iter().fold(Vec2::splat(f64::INFINITY), |m, b| m.min(b.0));
        let hi = boxes.iter().fold(-Vec2::splat(f64::INFINITY), |m, b| m.max(b.1));
        let size = (hi - lo) / n as f64;
        let bin = |p: Vec2| -> (usize, usize) {
            let top = n as i64 - 1;
            (
                (((p.x - lo.x) / size.x) as i64).clamp(0, top) as usize,
                (((p.y - lo.y) / size.y) as i64).clamp(0, top) as usize,
            )
        };
        let mut bins: Vec<Vec<i32>> = vec![Vec::new(); n * n];
        for (k, b) in boxes.iter().enumerate() {
            let (b0, b1) = (bin(b.0), bin(b.1));
            for by in b0.1..=b1.1 {
                for bx in b0.0..=b1.0 {
                    bins[by * n + bx].push(k as i32);
                }
            }
        }
        SheetSearch { surface: surface.clone(), view, boxes, bins, lo, size }
    }

    /// The view depth of the nearest sheet on the line of sight through `p`
    /// other than the neighbourhood of coordinate `own`, the sheets within
    /// two grid cells of it; minus infinity where there is none.
    fn nearest(&self, p: P2<ViewSpace>, own: P2<ParamSpace>) -> f64 {
        let n = SheetSearch::COUNT;
        let cu = self.surface.u.cells();
        let (du, dv) = (self.surface.u.spacing(), self.surface.v.spacing());
        let scale = swift::max(swift::max(p.x().abs(), p.y().abs()), 1.0);
        let b = Vec2::new((p.x() - self.lo.x) / self.size.x, (p.y() - self.lo.y) / self.size.y);
        if !(b.x >= 0.0 && b.y >= 0.0 && b.x < n as f64 && b.y < n as f64) {
            return f64::NEG_INFINITY;
        }
        let mut nearest = f64::NEG_INFINITY;
        for &k in &self.bins[(b.y as usize) * n + b.x as usize] {
            let (blo, bhi) = self.boxes[k as usize];
            let inside = p.x() >= blo.x && p.x() <= bhi.x && p.y() >= blo.y && p.y() <= bhi.y;
            if !inside {
                continue;
            }
            let k = k as usize;
            let centre = Vec2::new(
                self.surface.u.coordinate((k % cu) as i64) + 0.5 * du,
                self.surface.v.coordinate((k / cu) as i64) + 0.5 * dv,
            );
            let mut c = centre;
            let mut at = P3::<ViewSpace>::new(p.x(), p.y(), 0.0);
            let mut settled = false;
            for _ in 0..40 {
                let jet = (self.surface.map)(P2::from_vec(c));
                at = self.view.apply(P3::from_vec(jet.position));
                let r = Vec2::new(at.x() - p.x(), at.y() - p.y());
                if r.length() < 1e-12 * scale {
                    settled = true;
                    break;
                }
                let a = self.view.apply(P3::from_vec(jet.position + jet.du)).v - at.v;
                let e = self.view.apply(P3::from_vec(jet.position + jet.dv)).v - at.v;
                let det = a.x * e.y - a.y * e.x;
                if det == 0.0 || !det.is_finite() {
                    break;
                }
                c -= Vec2::new(r.x * e.y - r.y * e.x, a.x * r.y - a.y * r.x) / det;
                if (c.x - centre.x).abs() > 3.0 * du || (c.y - centre.y).abs() > 3.0 * dv {
                    break;
                }
            }
            let other = self.surface.u.difference(c.x, own.x()).abs() > 2.0 * du
                || self.surface.v.difference(c.y, own.y()).abs() > 2.0 * dv;
            let keep = settled && (c.x - centre.x).abs() <= 1.5 * du && (c.y - centre.y).abs() <= 1.5 * dv && other;
            if !keep {
                continue;
            }
            nearest = swift::max(nearest, at.z());
        }
        nearest
    }
}

/// The four points a hair either way along each axis.
fn beside(c: P2<ParamSpace>, t: f64) -> [P2<ParamSpace>; 4] {
    [P2::new(c.x() + t, c.y()), P2::new(c.x() - t, c.y()), P2::new(c.x(), c.y() + t), P2::new(c.x(), c.y() - t)]
}

/// The first layer of a drawing named `name`.
fn layer_named(layers: &[Layer], name: &str) -> Layer {
    layers.iter().find(|l| l.spec.name == name).unwrap_or_else(|| panic!("a layer {name}")).clone()
}

/// "surfaces: a truncation edge is a crease, drawn once and seen where either face is"
#[test]
fn a_truncation_edge_is_a_crease_drawn_once_and_seen_where_either_face_is() {
    // sn in relief, truncated and floored, so that each pole is a chimney
    // whose flat top meets its wall in a crease: the drawing the gaps
    // were found on, two periods each way.
    let base = PeriodicTorus::jacobi_sn(
        0.64,
        0.125,
        &PeriodicOptions { resolution: 300, grid: 512, truncated: true, ..PeriodicOptions::default() },
    )
    .expect("sn, truncated")
    .repeated(2, 2, 1024)
    .expect("two periods each way");
    let height = base.height.clone().expect("a height");
    let floor = 1.5;
    let floored: ParamScalar = Arc::new(move |c: P2<ParamSpace>| swift::max(height(c) - floor, 0.0));
    let drawing = Drawing {
        surface: base.surface.clone(),
        layers: base.layers.clone(),
        height: Some(floored.clone()),
        ceiling: base.ceiling - floor,
        radii: base.radii,
        isothermal: false,
        contours: None,
    }
    .relief(0.45, 1.0)
    .expect("the relief");
    let surface = drawing.surface.clone();
    let ceiling = drawing.ceiling;
    let camera =
        Camera::oblique(&ObliqueProjection { shear: 0.0, x_angle: -55.0, z_angle: 30.0, flip_x: false, y_scale: None });
    let sight = camera.view.projection_direction();
    let h = |c: P2<ParamSpace>| floored(c);
    let cell = swift::min(surface.u.spacing(), surface.v.spacing());
    let mut failures: Vec<String> = Vec::new();

    // The facing changes sign across the truncation edge as well as through a fold,
    // and only the second is a fold: no piece of fold runs along the
    // plateau's edge, truncated one side and not the other at both its
    // ends. A fold that meets the truncation edge ends on it, below.
    let truncation_edge_layer = layer_named(&drawing.layers, "truncation edge");
    let creases = Scene::parametric_scene(
        surface.clone(),
        vec![truncation_edge_layer.clone()],
        camera,
        PreviewMode::Drawing,
        0.0,
    )
    .creases();
    let folds = surface.fold_curves(sight, None, &creases);
    let fc = folds.coords.clone().expect("fold curves carry coordinates");
    let on_truncation_edge = |c: P2<ParamSpace>| {
        let hs = beside(c, 1e-4 * cell).map(h);
        hs.iter().any(|&x| x >= ceiling) && hs.iter().any(|&x| x < ceiling)
    };
    let mut along = 0;
    for path in 0..folds.count() {
        for i in folds.offsets[path]..(folds.offsets[path + 1] - 1) {
            if on_truncation_edge(fc[i]) && on_truncation_edge(fc[i + 1]) {
                along += 1;
            }
        }
    }
    expect(
        &mut failures,
        folds.count() > 0 && along == 0,
        "no fold is drawn along a crease",
        &format!("{along} of {} fold segments along a truncation edge", fc.len() - folds.count()),
    );
    // Nor beside one: where the grid saw the sign change across a
    // crease and the map finds no zero within a cell, there is no fold
    // to move the vertex onto, and it is not drawn where it was.
    let worst = fc.iter().map(|&c| surface.facing(c, sight).unwrap_or(0.0).abs()).fold(0.0, swift::max);
    expect(
        &mut failures,
        worst < 1e-6,
        "every fold vertex on the relief is edge-on",
        &format!("max |cos| {worst:.1e}"),
    );

    // Where a fold meets a truncation edge, it ends on the truncation edge. The wall or the truncation
    // turns edge-on between two truncation edge vertices where its facing a hair
    // off the truncation edge changes sign, and the fold starts there; on a steep
    // wall its whole visible length can lie within a grid cell.
    let truncation_edge_coords = &truncation_edge_layer.paths;
    let side = |c: P2<ParamSpace>, truncation_height: bool| -> f64 {
        let e = 1e-3 * cell;
        let g = Vec2::new(
            h(P2::new(c.x() + e, c.y())) - h(P2::new(c.x() - e, c.y())),
            h(P2::new(c.x(), c.y() + e)) - h(P2::new(c.x(), c.y() - e)),
        );
        let n = if g.length() > 0.0 { g / g.length() } else { Vec2::new(1.0, 0.0) };
        let shift = if truncation_height { 0.02 } else { -0.02 };
        surface.facing(P2::from_vec(c.v + shift * cell * n), sight).unwrap_or(0.0)
    };
    let (mut farthest, mut junctions) = (0.0, 0);
    let rc = truncation_edge_coords.coords.as_ref().expect("the truncation edge carries coordinates");
    for path in 0..truncation_edge_coords.count() {
        for i in truncation_edge_coords.offsets[path]..(truncation_edge_coords.offsets[path + 1] - 1) {
            for truncation_height in [false, true] {
                let (fa, fb) = (side(rc[i], truncation_height), side(rc[i + 1], truncation_height));
                if (fa > 0.0) == (fb > 0.0) {
                    continue;
                }
                let t = fa / (fa - fb);
                let j = rc[i].v + t * (rc[i + 1].v - rc[i].v);
                let nearest = fc.iter().map(|q| (q.v - j).length() / cell).fold(f64::INFINITY, swift::min);
                farthest = swift::max(farthest, nearest);
                junctions += 1;
            }
        }
    }
    expect(
        &mut failures,
        junctions > 0 && farthest < 0.1,
        "a fold that meets a truncation edge ends on it",
        &format!("{junctions} junctions, farthest fold end {farthest:.3} cells off"),
    );

    // And where it meets it in sight, the truncation edge's seen piece ends where the
    // fold's does: a cut in the truncation edge is bisected on the truncation edge, not on the
    // chord between two of its vertices, which crosses the truncation.
    let sphere_tracer =
        SurfaceVisibility::sphere_tracing(surface.clone(), camera.view).expect("the relief bounds a solid");
    let seen_truncation_edge =
        HiddenLine::clip_on(&truncation_edge_layer.paths.mapped(&camera.view), &sphere_tracer, true, true)
            .expect("a clip");
    let seen_folds = HiddenLine::clip_on(&folds.mapped(&camera.view), &sphere_tracer, true, false).expect("a clip");
    let ends = |p: &PolylineSet<PageSpace>| -> Vec<Vec2> {
        (0..p.count())
            .flat_map(|i| [p.vertices[p.offsets[i]], p.vertices[p.offsets[i + 1] - 1]].map(|q| Vec2::new(q.x(), q.y())))
            .collect()
    };
    let fold_ends = ends(&seen_folds);
    let meetings: Vec<f64> = ends(&seen_truncation_edge)
        .iter()
        .filter_map(|&e| {
            let m = fold_ends.iter().map(|&f| (f - e).length()).fold(f64::INFINITY, swift::min);
            if m < 0.01 { Some(m) } else { None }
        })
        .collect();
    let widest = meetings.iter().copied().reduce(swift::max).unwrap_or(f64::NAN);
    expect(
        &mut failures,
        !meetings.is_empty() && widest < 0.001,
        "where a fold meets the truncation edge in sight, the two seen pieces meet",
        &format!("{} meetings, widest {widest:.5}", meetings.len()),
    );

    // The truncation edge is a convex edge: seen where the truncation or the wall beside
    // it faces the eye and nothing stands in front. The normal on the
    // crease itself is half of each, and says nothing.
    let truncation_edge = truncation_edge_layer.clone();
    // Asked at the truncation edge's vertices, which are on the crease: a chord's
    // midpoint is inside the plateau, where only the truncation is beside it.
    let coords = truncation_edge.paths.coords.as_ref().expect("coordinates");
    let seen: Vec<bool> = coords
        .iter()
        .map(|&c| {
            beside(c, 1e-3 * cell).iter().any(|&q| surface.facing(q, sight).unwrap_or(0.0) > 0.0)
                && sphere_tracer.is_clear((surface.map)(c).position)
        })
        .collect();
    // A chord whose ends disagree counts half, and is the reference's
    // own uncertainty: the bake finds where along it the verdict turns.
    let (mut truth, mut slack) = (0.0, 0.0);
    let paths = &truncation_edge.paths;
    for path in 0..paths.count() {
        for i in paths.offsets[path]..(paths.offsets[path + 1] - 1) {
            if !(seen[i] || seen[i + 1]) {
                continue;
            }
            let length = planar(camera.view.apply(paths.vertices[i]), camera.view.apply(paths.vertices[i + 1]));
            if seen[i] && seen[i + 1] {
                truth += length;
            } else {
                truth += 0.5 * length;
                slack += 0.5 * length;
            }
        }
    }
    let scene =
        Scene::parametric_scene(surface.clone(), vec![truncation_edge.clone()], camera, PreviewMode::Drawing, 0.0);
    let baked = bake(&scene, &BakeOptions::new(2000)).expect("a bake");
    // (Swift reads the pixel off the bake's depth buffer; the port's bake
    // renders the coordinate image over the same frame.)
    let unit = baked.surface.as_ref().expect("an image").frame().units_per_pixel();
    let px = swift::max(unit.x, unit.y);
    let length = baked.strokes.total_length();
    let excess = length - truth;
    expect(
        &mut failures,
        truth > 0.0 && excess.abs() < slack + 2.0 * px,
        "the baked truncation edge is the convex edge's visible length, to its turning chords",
        &format!("{length:.4} vs {truth:.4}, {:+.1} px of {:.1}", excess / px, slack / px + 2.0),
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// "surfaces: in relief past its reach, the truncation edge is hidden only by a sheet in front of it"
#[test]
fn in_relief_past_its_reach_the_truncation_edge_is_hidden_only_by_a_sheet_in_front_of_it() {
    // sn in relief at amplitude 0.4 stands 0.4 × 3 = 1.2 off the torus
    // at the truncation, past its reach of 1: the surface passes through
    // itself, bounds no solid, and the coordinate image decides what is
    // in front. Along the truncation edge the image's chords fall a few
    // pixels short of the crease, and the pixels round a vertex there can
    // hold only the sheet behind it: at resolution 3000 that hid twenty
    // vertices with nothing in front of them.
    let drawing = PeriodicTorus::jacobi_sn(
        0.64,
        0.0,
        &PeriodicOptions { resolution: 600, grid: 1024, truncated: true, ..PeriodicOptions::default() },
    )
    .expect("sn, truncated")
    .relief(0.4, 1.0)
    .expect("the relief");
    let surface = drawing.surface.clone();
    let truncation_edge = layer_named(&drawing.layers, "truncation edge");
    let coords = truncation_edge.paths.coords.clone().expect("coordinates");
    let camera = Camera::oblique(&SurfaceRequest::PROJECTION);
    let points: Vec<P3<ViewSpace>> =
        coords.iter().map(|&c| camera.view.apply(P3::from_vec((surface.map)(c).position))).collect();
    let indices: Vec<usize> = (0..coords.len()).collect();
    // Seen, by brute force: no other sheet nearer the eye by more than a millionth.
    let search = SheetSearch::new(&surface, camera.view);
    let front = par_map(&indices, |&k| search.nearest(points[k].xy(), coords[k]));
    let seen: Vec<bool> = points.iter().zip(&front).map(|(p, &f)| p.z() + 1e-6 >= f).collect();
    let scene = Scene::parametric_scene(surface.clone(), vec![truncation_edge], camera, PreviewMode::Drawing, 0.0);
    let mut failures: Vec<String> = Vec::new();
    let example = |list: &[usize]| {
        list.first().map_or(String::new(), |&k| format!(", e.g. ({:.5}, {:.5})", coords[k].x(), coords[k].y()))
    };
    for resolution in [1500, 3000, 6000] {
        let baked = bake(&scene, &BakeOptions::new(resolution)).expect("a bake");
        let vis = SurfaceVisibility::new(surface.clone(), baked.surface.clone().expect("an image"), camera.view);
        let unit = vis.image.as_ref().expect("an image").frame().units_per_pixel();
        let px = swift::max(unit.x, unit.y);
        let kept = par_map(&indices, |&k| vis.is_visible(points[k], coords[k]).expect("an image"));
        let hidden: Vec<usize> = indices.iter().copied().filter(|&k| seen[k] && !kept[k]).collect();
        let label = format!("resolution {resolution}");
        expect(
            &mut failures,
            surface.outward.is_none() && !vis.traces_spheres() && seen.contains(&true) && hidden.is_empty(),
            &format!("{label}: no truncation edge vertex is hidden with nothing in front of it"),
            &format!(
                "{} of {} seen by brute force{}",
                hidden.len(),
                seen.iter().filter(|&&s| s).count(),
                example(&hidden)
            ),
        );
        // Nor kept with a sheet in front of it, except near that sheet's
        // outline: within two pixels, or within a grid cell, measured in
        // view at the vertex, since the image draws each sheet as the
        // grid's chords. Found by asking the brute force at the eight
        // neighbours that far away, at the vertex's depth. This is the
        // image's own resolution, and the fix must not swing past it.
        let wrongly_kept: Vec<usize> = indices.iter().copied().filter(|&k| !seen[k] && kept[k]).collect();
        let unexplained: Vec<usize> = wrongly_kept
            .iter()
            .copied()
            .filter(|&k| {
                let jet = (surface.map)(coords[k]);
                let cell = [jet.du * surface.u.spacing(), jet.dv * surface.v.spacing()]
                    .map(|d| {
                        let q = camera.view.apply(P3::from_vec(jet.position + d));
                        Vec2::new(q.x() - points[k].x(), q.y() - points[k].y()).length()
                    })
                    .into_iter()
                    .reduce(swift::max)
                    .expect("two cells");
                let reach = swift::max(2.0 * px, cell);
                let around = [
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (0.0, 1.0),
                    (0.0, -1.0),
                    (1.0, 1.0),
                    (1.0, -1.0),
                    (-1.0, 1.0),
                    (-1.0, -1.0),
                ];
                around.iter().all(|&(dx, dy)| {
                    let q = P2::<ViewSpace>::new(points[k].x() + reach * dx, points[k].y() + reach * dy);
                    points[k].z() + 1e-6 < search.nearest(q, coords[k])
                })
            })
            .collect();
        expect(
            &mut failures,
            unexplained.is_empty(),
            &format!("{label}: none is kept with a sheet in front of it, but beside that sheet's outline"),
            &format!(
                "{} kept that brute force hides, {} farther than two pixels and a grid cell{}",
                wrongly_kept.len(),
                unexplained.len(),
                example(&unexplained)
            ),
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// "surfaces: a doubly periodic function drawn on its torus"
#[test]
fn a_doubly_periodic_function_drawn_on_its_torus() {
    let m = 0.64;
    let drawing = PeriodicTorus::jacobi_sn(m, 0.0, &PeriodicOptions::default()).expect("sn on its torus");
    let (mut worst, mut off, mut vertices, mut singular) = (0.0, 0.0, 0, 0);
    let mut worst_where = String::new();
    for layer in &drawing.layers {
        let LayerSource::Contour { field, levels, .. } = &layer.spec.source else { continue };
        let coords = layer.paths.coords.as_ref().expect("contours carry coordinates");
        for (k, &c) in coords.iter().enumerate() {
            vertices += 1;
            let f = Jacobi::complex(Complex::new(c.x(), c.y()), m).sn;
            // Every phase contour ends at a zero or a pole, where arg f
            // has no value; its last vertex there is on every level.
            if *field == ContourField::Phase && (f.magnitude() < 1e-4 || f.magnitude() > 1e4) {
                singular += 1;
                continue;
            }
            let value = if *field == ContourField::Magnitude { f.magnitude() } else { f.argument() };
            let level = levels
                .iter()
                .copied()
                .min_by(|a, b| real::total_order((a - value).abs(), (b - value).abs()))
                .expect("a level");
            let scale = if *field == ContourField::Magnitude { swift::max(level, 1.0) } else { 1.0 };
            if (value - level).abs() / scale > worst {
                worst_where = format!(
                    "{} at ({:.5}, {:.5}) |f| {:.3e} value {value:.4} level {level:.4}",
                    layer.spec.name,
                    c.x(),
                    c.y(),
                    f.magnitude()
                );
            }
            worst = swift::max(worst, (value - level).abs() / scale);
            off = swift::max(off, (layer.paths.vertices[k].v - (drawing.surface.map)(c).position).length());
        }
    }
    let mut failures: Vec<String> = Vec::new();
    expect(
        &mut failures,
        vertices > 10_000 && worst < 1e-6,
        "every contour vertex is on its level of sn",
        &format!("{vertices} vertices, worst {worst} {worst_where}; {singular} phase ends at a zero or pole not asked"),
    );
    expect(&mut failures, off < 1e-12, "and sits on the torus where its coordinate says", &format!("{off}"));

    let camera =
        Camera::oblique(&ObliqueProjection { shear: 0.5, x_angle: -50.0, z_angle: 25.0, flip_x: false, y_scale: None });
    let contours: Vec<Layer> =
        drawing.layers.iter().filter(|l| matches!(l.spec.source, LayerSource::Contour { .. })).cloned().collect();
    let baked = bake(
        &Scene::parametric_scene(drawing.surface.clone(), contours.clone(), camera, PreviewMode::Drawing, 0.0),
        &BakeOptions::new(2400),
    )
    .expect("a bake");
    let truth: f64 =
        contours.iter().map(|l| ray_cast_length(&l.paths, &drawing.surface, 2.0, 1.0, &camera.view, 4)).sum();
    let length = baked.strokes.total_length();
    let error = (length - truth).abs() / truth;
    expect(
        &mut failures,
        error < 0.003,
        "and it is seen where the ray-cast sees it",
        &format!("{length:.4} vs {truth:.4}, {:.3}% off", 100.0 * error),
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Whether `c` lies on the edge of the rectangle `u` x `v`, to `on_edge`.
fn on_rectangle_edge(c: P2<ParamSpace>, u: (f64, f64), v: (f64, f64), on_edge: f64) -> bool {
    (c.x() - u.0).abs() < on_edge
        || (c.x() - u.1).abs() < on_edge
        || (c.y() - v.0).abs() < on_edge
        || (c.y() - v.1).abs() < on_edge
}

/// The ends on the seam with no other end at the same point of the torus, to 1e-8.
fn unmatched(seam: &[(String, Vec3)]) -> Vec<String> {
    seam.iter()
        .enumerate()
        .filter(|&(k, end)| !seam.iter().enumerate().any(|(j, other)| j != k && (other.1 - end.1).length() < 1e-8))
        .map(|(_, end)| end.0.clone())
        .collect()
}

/// "surfaces: every line on the sn torus ends on a zero, a pole, or its continuation across the seam"
#[test]
fn every_line_on_the_sn_torus_ends_on_a_zero_a_pole_or_its_continuation_across_the_seam() {
    // The torus has no seam; the rectangle it is drawn from has one, and sn
    // puts both its zeros and one pole on it. Every line is computed on the
    // plane and cut to the rectangle, so a line may end in only three places:
    // on a zero, on a pole, or on the rectangle's edge, where its
    // continuation begins at the same point of the torus.
    let m = 0.64;
    let drawing = PeriodicTorus::jacobi_sn(m, 0.0, &PeriodicOptions::default()).expect("sn on its torus");
    let u = (drawing.surface.u.range.lo, drawing.surface.u.range.hi);
    let v = (drawing.surface.v.range.lo, drawing.surface.v.range.hi);
    let on_edge = 1e-7;
    let mut loose: Vec<String> = Vec::new();
    let mut seam: Vec<(String, Vec3)> = Vec::new();
    let (mut zeros, mut poles) = (0, 0);
    for layer in &drawing.layers {
        let LayerSource::Contour { .. } = layer.spec.source else { continue };
        let Some(coords) = &layer.paths.coords else { continue };
        let offsets = &layer.paths.offsets;
        for i in 0..(offsets.len() - 1) {
            let run = &coords[offsets[i]..offsets[i + 1]];
            let (Some(&a), Some(&b)) = (run.first(), run.last()) else { continue };
            if a == b {
                continue;
            }
            for c in [a, b] {
                let f = Jacobi::complex(Complex::new(c.x(), c.y()), m).sn;
                if f.magnitude() < 1e-6 {
                    zeros += 1;
                    continue;
                }
                if !f.is_finite() || f.magnitude() > 1e6 {
                    poles += 1;
                    continue;
                }
                let text = format!("{} ({:.5}, {:.5}) |f| {:.3}", layer.spec.name, c.x(), c.y(), f.magnitude());
                if on_rectangle_edge(c, u, v, on_edge) {
                    seam.push((text, (drawing.surface.map)(c).position));
                } else {
                    loose.push(text);
                }
            }
        }
    }
    let unmatched = unmatched(&seam);
    let mut failures: Vec<String> = Vec::new();
    expect(
        &mut failures,
        zeros > 0 && poles > 0,
        "phase lines end on the zeros and on the poles",
        &format!("{zeros} ends on zeros, {poles} on poles"),
    );
    expect(
        &mut failures,
        loose.is_empty(),
        "no line ends in the open",
        &format!("{} do: {}", loose.len(), loose.iter().take(6).cloned().collect::<Vec<_>>().join("; ")),
    );
    expect(
        &mut failures,
        !seam.is_empty() && unmatched.is_empty(),
        "every end on the seam meets its continuation on the torus, to 1e-8",
        &format!(
            "{} ends on the seam, {} unmatched{}",
            seam.len(),
            unmatched.len(),
            if unmatched.is_empty() {
                String::new()
            } else {
                format!(": {}", unmatched.iter().take(6).cloned().collect::<Vec<_>>().join("; "))
            }
        ),
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// "surfaces: a double zero and a double pole each hold two rays of every level"
#[test]
fn a_double_zero_and_a_double_pole_each_hold_two_rays_of_every_level() {
    // sn² has periods 2K and 2iK', double zeros and double poles. At a
    // double point every level has two rays, which the grid cannot
    // resolve inside the point's cell: the fan is read off the normal form,
    // f(z) ≈ a (z − z₀)^±2, and its rays are what the grid's lines are
    // spliced onto. The rectangle is shifted so that one zero and one pole
    // are interior, where every ray of the fan ends on the point itself.
    let m = 0.64;
    let (k, kp) = (Jacobi::quarter_period(m), Jacobi::quarter_period(1.0 - m));
    let drawing = PeriodicTorus::drawing(
        &format!("sn(z, {m})^2"),
        (2.0 * k, 2.0 * kp),
        &PeriodicOptions { origin: Complex::new(-k / 2.0, -kp / 2.0), ..PeriodicOptions::default() },
    )
    .expect("sn² on its torus");
    let u = (drawing.surface.u.range.lo, drawing.surface.u.range.hi);
    let v = (drawing.surface.v.range.lo, drawing.surface.v.range.hi);
    let (mut at_zero, mut at_pole, mut levels) = (0, 0, 0);
    let mut loose: Vec<String> = Vec::new();
    let mut seam: Vec<(String, Vec3)> = Vec::new();
    for layer in &drawing.layers {
        let LayerSource::Contour { field, levels: list, .. } = &layer.spec.source else { continue };
        if *field != ContourField::Phase {
            continue;
        }
        let Some(coords) = &layer.paths.coords else { continue };
        // The cut, ±π, is drawn once.
        levels += list.iter().filter(|&&x| x > -PI + 1e-9).count();
        let offsets = &layer.paths.offsets;
        for i in 0..(offsets.len() - 1) {
            let run = &coords[offsets[i]..offsets[i + 1]];
            let (Some(&a), Some(&b)) = (run.first(), run.last()) else { continue };
            if a == b {
                continue;
            }
            for c in [a, b] {
                if real::hypot(c.x(), c.y()) < 1e-6 {
                    at_zero += 1;
                    continue;
                }
                if real::hypot(c.x(), c.y() - kp) < 1e-6 {
                    at_pole += 1;
                    continue;
                }
                let f = Jacobi::complex(Complex::new(c.x(), c.y()), m).sn;
                let f2 = f.magnitude() * f.magnitude();
                if f2 < 1e-6 || !f.is_finite() || f2 > 1e6 {
                    continue;
                }
                let text = format!("{} ({:.5}, {:.5}) |f| {f2:.3}", layer.spec.name, c.x(), c.y());
                if on_rectangle_edge(c, u, v, 1e-7) {
                    seam.push((text, (drawing.surface.map)(c).position));
                } else {
                    loose.push(text);
                }
            }
        }
    }
    let unmatched = unmatched(&seam);
    let mut failures: Vec<String> = Vec::new();
    expect(
        &mut failures,
        at_zero == 2 * levels && at_pole == 2 * levels,
        &format!("two rays of each of the {levels} levels end on the double zero and on the double pole"),
        &format!("{at_zero} ends on the zero, {at_pole} on the pole"),
    );
    expect(
        &mut failures,
        loose.is_empty(),
        "and no line ends in the open",
        &format!("{} do: {}", loose.len(), loose.iter().take(6).cloned().collect::<Vec<_>>().join("; ")),
    );
    expect(
        &mut failures,
        !seam.is_empty() && unmatched.is_empty(),
        "and every end on the seam meets its continuation, to 1e-8",
        &format!("{} ends on the seam, {} unmatched", seam.len(), unmatched.len()),
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// "preview: a parametric surface draws what its bake draws" (`surfacePreviewTests`):
/// the preview's curves, at rest and in a draft, against the bake's strokes
/// rasterized into the preview's pixels.
#[test]
#[ignore = "PORT: phase 7 -- the GPU preview"]
fn a_parametric_surface_preview_draws_what_its_bake_draws() {}

/// "surfaces: a linear curve is a straight line on the flat torus, placed on the grid"
#[test]
fn a_linear_curve_is_a_straight_line_on_the_flat_torus_placed_on_the_grid() {
    let torus = ParametricSurface::torus(2.0, 1.0, (512, 256)).expect("a round torus");
    let turn = 2.0 * PI;
    let mut failures: Vec<String> = Vec::new();

    // The interpolator returns the grid at its own points, and is
    // within a grid cell's sag of the map between them.
    let (mut exact, mut sag) = (true, 0.0);
    for j in (0..256).step_by(17) {
        for i in (0..512).step_by(13) {
            let c = P2::<ParamSpace>::new(torus.u.coordinate(i), torus.v.coordinate(j));
            exact = exact && (torus.interpolate(c).v - torus.position(i, j).v).length() < 1e-12;
            let mid = P2::<ParamSpace>::new(
                c.x() + 0.5 * torus.u.spacing() + 7.0 * turn,
                c.y() + 0.5 * torus.v.spacing() - 3.0 * turn,
            );
            sag = swift::max(sag, (torus.interpolate(mid).v - (torus.map)(mid).position).length());
        }
    }
    expect(&mut failures, exact, "the grid interpolates to itself at its own points", "");
    expect(
        &mut failures,
        sag < 1e-3 && sag > 0.0,
        "and to the map between them, whole periods away",
        &format!("{sag:.2e} off"),
    );

    // Its normals interpolate to within a cell's turn of the map's:
    // on this grid the tube turns 360°/256 a cell, so under a degree.
    let normals = torus.grid_normals();
    let mut tilt = 0.0;
    for j in (0..256).step_by(5) {
        for i in (0..512).step_by(7) {
            let c = P2::<ParamSpace>::new(
                torus.u.coordinate(i) + 0.3 * torus.u.spacing() - 2.0 * turn,
                torus.v.coordinate(j) + 0.7 * torus.v.spacing() + 5.0 * turn,
            );
            let f = torus.interpolate_normal(c, &normals).expect("a normal");
            let n = Vec3::new(f64::from(f[0]), f64::from(f[1]), f64::from(f[2]));
            let m = (torus.map)(c).normal().normalized();
            tilt = swift::max(tilt, real::acos(swift::min(1.0, n.dot(m))) * 180.0 / PI);
        }
    }
    expect(
        &mut failures,
        tilt < 1.0,
        "and its normals to within a degree of the map's, whole periods away",
        &format!("worst tilt {tilt:.3}°"),
    );

    // A rational slope closes after its denominator's turns, and stops
    // there: one closed (p, q) curve, straight in coordinates.
    let closed = torus.linear_curves(2.0 / 5.0, 24, 1, None);
    let steps = 512;
    expect(
        &mut failures,
        closed.count() == 1 && closed.vertices.len() == 5 * steps + 1,
        "slope 2/5 closes after five turns, a vertex per grid cell",
        &format!("{} paths, {} vertices", closed.count(), closed.vertices.len()),
    );
    let ends = (closed.vertices[0].v - closed.vertices[closed.vertices.len() - 1].v).length();
    expect(&mut failures, ends < 1e-9, "and its last vertex is its first", &format!("{ends:.1e} apart"));
    let coords = closed.coords.as_ref().expect("linear curves carry coordinates");
    let straight = coords.iter().all(|c| (c.y() - 0.4 * c.x()).abs() < 1e-9);
    let rises = coords[coords.len() - 1].x() - coords[0].x();
    expect(
        &mut failures,
        straight && (rises - 5.0 * turn).abs() < 1e-9,
        "its coordinates run straight, v = 2/5 u, over five turns of u",
        &format!("rises {rises}"),
    );
    let placed = closed
        .vertices
        .iter()
        .zip(coords)
        .map(|(p, &c)| (p.v - (torus.map)(c).position).length())
        .reduce(swift::max)
        .expect("vertices");
    expect(
        &mut failures,
        placed < 1e-3,
        "every vertex lies on the drawn surface at its coordinate",
        &format!("{placed:.2e} from the map"),
    );

    // An irrational slope never closes: all the turns asked for, broken
    // into paths of at most eight turns that share their seam vertex,
    // coordinates continuous across it by whole periods.
    let open = torus.linear_curves(real::sqrt(2.0) - 1.0, 20, 1, None);
    let oc = open.coords.as_ref().expect("linear curves carry coordinates");
    let mut seams = true;
    for k in 1..open.count() {
        let (end, start) = (open.offsets[k] - 1, open.offsets[k]);
        let du = (oc[end].x() - oc[start].x()) / turn;
        let dv = (oc[end].y() - oc[start].y()) / turn;
        seams = seams
            && open.vertices[end] == open.vertices[start]
            && (du - du.round()).abs() < 1e-9
            && (dv - dv.round()).abs() < 1e-9;
    }
    let spans: Vec<f64> =
        (0..open.count()).map(|k| (oc[open.offsets[k + 1] - 1].x() - oc[open.offsets[k]].x()) / turn).collect();
    let far = (open.vertices[0].v - open.vertices[open.vertices.len() - 1].v).length();
    expect(
        &mut failures,
        open.count() == 3 && spans.iter().all(|&s| s <= 8.0 + 1e-9) && (spans.iter().sum::<f64>() - 20.0).abs() < 1e-9,
        "√2 − 1 runs its twenty turns in three paths of at most eight",
        &format!("{} paths spanning {:?}", open.count(), spans.iter().map(|s| format!("{s:.2}")).collect::<Vec<_>>()),
    );
    expect(
        &mut failures,
        seams && far > 0.1,
        "which share their seam vertex and never close",
        &format!("ends {far:.3} apart"),
    );

    // Strands start evenly spaced along v; slope zero is the v-lines.
    let strands = torus.linear_curves(0.0, 3, 4, None);
    let sc = strands.coords.as_ref().expect("linear curves carry coordinates");
    let starts: Vec<f64> = (0..strands.count()).map(|k| sc[strands.offsets[k]].y()).collect();
    let want: Vec<f64> = [0.0, 0.25, 0.5, 0.75].iter().map(|x| x * turn).collect();
    expect(
        &mut failures,
        strands.count() == 4 && starts == want && sc.iter().all(|c| c.x() <= turn + 1e-9),
        "four strands of slope zero are four lines of constant v, one turn each",
        &format!("{} strands starting at {starts:?}", strands.count()),
    );

    // The steeper axis sets the vertex count: slope 4 walks the tube
    // four times per turn and gets a vertex per v-cell of that.
    let steep = torus.linear_curves(4.0, 1, 1, None);
    expect(
        &mut failures,
        steep.vertices.len() == 4 * 256 + 1,
        "slope four is sampled once per v-cell",
        &format!("{} vertices", steep.vertices.len()),
    );

    let source = LayerSource::LinearCurves { slope: 0.4, turns: 24, count: 3 };
    expect(
        &mut failures,
        LayerSource::from_json(&source.json()).ok() == Some(source.clone()),
        "a linear curve layer source round-trips through the manifest",
        "",
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// "preview: a torus nearly edge-on keeps its outline whole at rest" (`edgeOnPreviewTests`):
/// sn seen from half a degree off its equatorial plane, the window drawing
/// at rest what sphere tracing keeps, and the draft held to showing the
/// difference.
#[test]
#[ignore = "PORT: phase 7 -- the GPU preview"]
fn a_torus_nearly_edge_on_keeps_its_outline_whole_at_rest() {}
