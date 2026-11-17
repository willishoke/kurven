//! `Sources/kurven-test/Stress.swift`: visibility under stress, every suite,
//! with Swift's tolerances.
//!
//! The principle these suites hold the hidden-line rules to: visibility is a
//! function of the geometry alone, never of the output resolution. A rule
//! that consults a pixel fails this by construction -- a pixel's depth span
//! on a steep graph or an edge-on wall exceeds any depth tolerance, and the span
//! changes with the resolution -- and the rules here are built so that no
//! pixel is consulted: the facing of the surface the curves lie on, and the
//! traversal up the line of sight through the drawn solid (`GridTraversal`), which is
//! exact because the drawn solid is piecewise linear. So the suites ask the
//! strongest questions such a rule can be asked: the same bake at every
//! depth resolution, the same drawing scaled when the drawing is scaled, and
//! on a grid too coarse for any approximation to hide in, agreement with
//! a brute-force ray cast against the triangles the depth pass draws.
//!
//! The oracles are brute force, as in Swift; where Swift spreads them over
//! its cores with `parallelMap`, the tests here spread them over threads
//! with `std::thread::scope` (the port's own code is sequential).

// Swift's `guard x > 0 else` refuses a NaN, and so does `if !(x > 0.0)`;
// the oracles keep Swift's guards as they are written.
#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::manual_range_contains)]

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{
    MeshIndex, alone, build, densely_hidden, drawn_triangles, exactly_hidden, first_hit, par_map, per_unit,
    random_heightfield, sharing, stress_cameras, test_camera,
};
use kurven_core::grid::Grid2D;
use kurven_core::hatching::FaceKind;
use kurven_core::hidden_line::HeightfieldVisibility;
use kurven_core::manifest::{
    Affine2, BoundaryPerimeter, Domain, Interval, PerimeterEdge, RealBand, Region, Truncation, Walls,
};
use kurven_core::mesh::Mesh;
use kurven_core::occlusion::OcclusionTest;
use kurven_core::polyline::{Layer, PolylineSet};
use kurven_core::scene::{Heightfield, Scene};
use kurven_core::simd::{Vec2, Vec3};
use kurven_core::spaces::{DomainSpace, P2, P3, WorldSpace};
use kurven_core::splitmix::SplitMix;
use kurven_core::surface::{ContourRefine, Magnitude, Surface};
use kurven_landscape::{Catalog, LandscapeRequest};
use kurven_math::{real, swift};

fn rgamma() -> kurven_core::bundle::KurvenBundle {
    let preset = Catalog::native().preset("rgamma").unwrap();
    build(&LandscapeRequest::from_preset(preset, 240), true)
}

/// "stress: a heightfield bake is the same drawing at every depth resolution".
/// Swift bakes at 64, 256, 1024 and 4096 pixels and compares the strokes.
/// Here nothing of the visibility reads a depth buffer at all -- every layer
/// is tested by its face and the traversal (`Scene::visible_layers`) -- so
/// the drawing has no resolution to depend on, and the suite holds what is
/// left to hold: the drawing draws, it is the same drawing every time it is
/// asked for, and on the steep case the solid hides what lies behind a
/// nearer graph and not the drawing.
#[test]
fn a_heightfield_bake_is_the_same_drawing_at_every_depth_resolution() {
    // rgamma's two drawings are timed, alone: no other test of the file
    // runs meanwhile.
    let timed = alone();
    let bundle = rgamma();
    let scene = Scene::from_bundle(&bundle, &bundle.manifest.presets[0]);
    let start = Instant::now();
    let reference = scene.visible_layers(None, None).unwrap();
    let again = scene.visible_layers(None, None).unwrap();
    let elapsed = start.elapsed();
    drop(timed);
    let _beside = sharing();
    let paths: usize = reference.iter().map(|(_, p)| p.count()).sum();
    let length: f64 = reference.iter().map(|(_, p)| p.total_length()).sum();
    eprintln!("rgamma: {paths} paths, {length:.1} units, {elapsed:?} for two drawings");
    assert!(paths > 200 && length > 100.0, "the drawing draws: {paths} paths, {length:.1} units of curves");
    let differing: Vec<&str> = reference
        .iter()
        .zip(&again)
        .filter(|((_, a), (_, b))| a != b)
        .map(|((l, _), _)| l.spec.name.as_str())
        .collect();
    assert!(
        reference.len() == again.len() && differing.is_empty(),
        "every layer is what it was, vertex for vertex; differs: {differing:?}"
    );
    assert!(elapsed < Duration::from_secs(60), "two drawings ran in reasonable time: {elapsed:?}");

    let (same, kept, facing_only, buried, behind) = steep_drawings();
    eprintln!("steep: {kept} of {facing_only} units kept; {behind} behind, {buried} buried");
    assert!(same, "the steep drawing too is the same strokes every time");
    assert!(
        kept > 0.5 * facing_only,
        "of the facing contour curves, the solid hides what lies behind a nearer graph: {kept:.1} of \
         {facing_only:.1} units kept; {behind} hidden vertices above their own triangle, {buried} under it"
    );
}

/// The steep case of `a_heightfield_bake_is_the_same_drawing_at_every_depth_resolution`:
/// erf under a truncation of 14 over a four-unit window, where the depth
/// test lost a third of the contour curves to pixel span. The curves the
/// test removes are removed by the solid, not by a pixel. Whether its two
/// drawings are the same strokes; the length of the facing contour curves
/// kept, and of all of them; and of the hidden facing vertices, those
/// buried under their own drawn surface and those behind a nearer graph.
fn steep_drawings() -> (bool, f64, f64, usize, usize) {
    let mut steep = LandscapeRequest::from_preset(Catalog::native().preset("erf").unwrap(), 240);
    steep.domain = Domain::new(Interval::new(-2.0, 2.0), Interval::new(-2.0, 2.0));
    steep.truncation = Some(Truncation::Uniform(14.0));
    let steep_bundle = build(&steep, true);
    let steep_scene = Scene::from_bundle(&steep_bundle, &steep_bundle.manifest.presets[0]);
    // The same scene drawn twice, one after the other: the second drawing,
    // made after the first, is its strokes.
    let first = steep_scene.visible_layers(None, None).unwrap();
    let second = steep_scene.visible_layers(None, None).unwrap();
    let same = first == second;
    // What the solid hides of the facing contour curves, and why: curves
    // behind a nearer graph, or curves under its own drawn surface by more
    // than the depth tolerance. Reported, since neither number is a target;
    // the floor only says the traversal is not erasing the drawing. Each
    // layer on its own, summed in order.
    let h = steep_scene.heightfield().unwrap();
    let view = steep_scene.camera.view;
    let vis = HeightfieldVisibility::new(h.clone(), &view, steep_scene.depth_tolerance);
    let layers: Vec<&Layer> = steep_scene
        .layers()
        .iter()
        .filter(|layer| layer.spec.clipped && steep_scene.geometry().face_of(&layer.spec) == Some(FaceKind::Graph))
        .collect();
    let per_layer = par_map(&layers, |layer| {
        let kept = h
            .visible_curves(&layer.paths, Some(FaceKind::Graph), &view, steep_scene.depth_tolerance, 0.0)
            .total_length();
        let facing_only =
            h.visible_curves(&layer.paths, Some(FaceKind::Graph), &view, f64::INFINITY, 0.0).total_length();
        let (mut buried, mut behind) = (0, 0);
        for &v in &layer.paths.vertices {
            if vis.facing(v.xy()) > 0.0 && !vis.is_clear(v) {
                if v.z() < h.drawn_height(v.xy()) {
                    buried += 1;
                } else {
                    behind += 1;
                }
            }
        }
        (kept, facing_only, buried, behind)
    });
    let (mut kept, mut facing_only, mut buried, mut behind) = (0.0, 0.0, 0, 0);
    for (k, f, u, b) in per_layer {
        kept += k;
        facing_only += f;
        buried += u;
        behind += b;
    }
    (same, kept, facing_only, buried, behind)
}

/// "stress: on a real drawing the traversal is the brute-force ray cast".
/// Swift also counts what a 1000-pixel depth buffer hid, and asks that in
/// the pit at -4 both the traversal and the cast hide more than it; there
/// is no rasterizer in the port's core to make that buffer (the bake's is
/// Phase 3's bake wave), so the depth buffer's half of the pit's check is
/// left out and the traversal and cast are still held to agree there.
#[test]
fn on_a_real_drawing_the_traversal_is_the_brute_force_ray_cast() {
    let _beside = sharing();
    let bundle = rgamma();
    let scene = Scene::from_bundle(&bundle, &bundle.manifest.presets[0]);
    // The grid test: the drawing without its function, whose solid is
    // the drawn triangles the cast is against. With the function the
    // test's solid is f (below).
    let with_f = scene.heightfield().unwrap().clone();
    let h = with_f.on_grid();
    let view = scene.camera.view;
    let mesh = drawn_triangles(&h, 0.0);
    let vis = HeightfieldVisibility::new(h.clone(), &view, scene.depth_tolerance);
    let toward = -view.projection_direction();
    let per_unit = per_unit(&view, toward);
    // The brute force over the drawn triangles, through an index of those
    // each point's lines can meet (`MeshIndex`): the same answers.
    let index = MeshIndex::new(&mesh, toward);
    let d = h.surface.domain();
    let (blo, bhi) = (
        Vec2::new(d.real.lo.min(d.real.hi), d.imag.lo.min(d.imag.hi)),
        Vec2::new(d.real.lo.max(d.real.hi), d.imag.lo.max(d.imag.hi)),
    );
    let g = &*h.surface.height;
    let cell =
        (g.domain.real.length().abs() / (g.width - 1) as f64).max(g.domain.imag.length().abs() / (g.height - 1) as f64);
    let graph_layers: Vec<&Layer> = scene
        .layers()
        .iter()
        .filter(|l| l.spec.clipped && scene.geometry().face_of(&l.spec) == Some(FaceKind::Graph))
        .collect();
    let (mut total, mut traversal_hidden, mut cast_hidden, mut disagree) = (0, 0, 0, 0);
    let (mut pit_total, mut pit_traversal, mut pit_cast) = (0, 0, 0);
    let start = Instant::now();
    for layer in &graph_layers {
        let vertices = &layer.paths.vertices;
        let facing: Vec<bool> = vertices.iter().map(|v| vis.facing(v.xy()) > 0.0).collect();
        let traversal: Vec<bool> = par_map(vertices, |&v| vis.is_clear(v));
        let cast: Vec<bool> = par_map(vertices, |&v| {
            let strictly_inside =
                v.x() > blo.x + 1e-9 && v.x() < bhi.x - 1e-9 && v.y() > blo.y + 1e-9 && v.y() < bhi.y - 1e-9;
            !exactly_hidden(&index, v.v, toward, scene.depth_tolerance, per_unit, strictly_inside, cell)
        });
        for (k, v) in vertices.iter().enumerate() {
            if !facing[k] {
                continue;
            }
            total += 1;
            traversal_hidden += usize::from(!traversal[k]);
            cast_hidden += usize::from(!cast[k]);
            disagree += usize::from(traversal[k] != cast[k]);
            if v.x() > -4.3 && v.x() < -3.7 && v.y() < 0.6 {
                pit_total += 1;
                pit_traversal += usize::from(!traversal[k]);
                pit_cast += usize::from(!cast[k]);
            }
        }
    }
    let elapsed = start.elapsed();
    eprintln!(
        "grid: {total} vertices, traversal {traversal_hidden}, cast {cast_hidden}, pit {pit_total}/{pit_traversal}/{pit_cast}, {elapsed:?}"
    );
    assert!(
        disagree == 0 && total > 5000,
        "the traversal and the cast agree on every facing contour vertex: {total} vertices; traversal hides \
         {traversal_hidden}, cast hides {cast_hidden}"
    );
    assert!(
        pit_traversal == pit_cast,
        "in the pit at -4 both hide the same: {pit_total} facing vertices: traversal {pit_traversal}, cast {pit_cast}"
    );
    assert!(elapsed < Duration::from_secs(60), "in reasonable time: {elapsed:?}");

    // With the function, the test's solid is f: the traversal samples the
    // line of sight at the grid's crossings and reads f there, and is
    // held to a cast that walks the line in steps of a sixty-fourth of
    // a cell and bisects every crossing of f. What the traversal can miss
    // is a dip of f within one cell along the line; the count is the
    // measure of that, on the real drawing.
    let vis_f = HeightfieldVisibility::new(with_f.clone(), &view, scene.depth_tolerance);
    let magnitude = with_f.refine.as_ref().unwrap().magnitude.clone().unwrap();
    let (mut total_f, mut traversal_f, mut cast_f, mut disagree_f) = (0, 0, 0, 0);
    let mut examples = String::new();
    let start = Instant::now();
    for layer in &graph_layers {
        let vertices = &layer.paths.vertices;
        let facing: Vec<bool> = vertices.iter().map(|v| vis_f.facing(v.xy()) > 0.0).collect();
        let traversal: Vec<bool> = par_map(vertices, |&v| vis_f.is_clear(v));
        let cast: Vec<Option<f64>> = par_map(vertices, |&v| {
            densely_hidden(&with_f, &magnitude, v.v, toward, vis_f.depth_tolerance, per_unit, cell)
        });
        for k in 0..vertices.len() {
            if !facing[k] {
                continue;
            }
            total_f += 1;
            traversal_f += usize::from(!traversal[k]);
            cast_f += usize::from(cast[k].is_some());
            if traversal[k] != cast[k].is_none() {
                disagree_f += 1;
                if disagree_f <= 6 {
                    let v = vertices[k];
                    examples += &format!(
                        " [({:.3}, {:.3}, {:.3}) truncation {:.2} |f| {:.4}: traversal {}, cast {}]",
                        v.x(),
                        v.y(),
                        v.z(),
                        with_f.surface.truncation.height_at(v.x()),
                        magnitude(P2::new(v.x(), v.y())),
                        if traversal[k] { "clear" } else { "hidden" },
                        cast[k].map_or("clear".to_string(), |t| format!("hides at t = {:.4} cells", t / cell))
                    );
                }
            }
        }
    }
    let elapsed = start.elapsed();
    eprintln!("f: {total_f} vertices, traversal {traversal_f}, cast {cast_f}, {elapsed:?}");
    assert!(
        disagree_f == 0 && total_f > 5000,
        "with the function, the traversal against f is the dense cast against f: {total_f} vertices; traversal \
         hides {traversal_f}, cast hides {cast_f}, {disagree_f} disagree{examples}"
    );
    assert!(elapsed < Duration::from_secs(120), "in reasonable time: {elapsed:?}");
}

/// "stress: scaling the drawing scales the drawing".
#[test]
fn scaling_the_drawing_scales_the_drawing() {
    // The whole drawing -- heights, domain, truncations, walls, curves and depth tolerance --
    // scaled by a power of two, so every float scales exactly; the tested
    // curves must be the tested curves scaled, vertex for vertex. A rule with a
    // pixel or an absolute tolerance in it would not pass this; one that
    // is geometry alone cannot fail it.
    let bundle = rgamma();
    let scene = Scene::from_bundle(&bundle, &bundle.manifest.presets[0]);
    let h = scene.heightfield().unwrap();
    let visible = scene.visible_layers(None, None).unwrap();
    for s in [8.0, 0.125] {
        let g = &*h.surface.height;
        let domain = Domain::new(
            Interval::new(g.domain.real.lo * s, g.domain.real.hi * s),
            Interval::new(g.domain.imag.lo * s, g.domain.imag.hi * s),
        );
        let truncation = match &h.surface.truncation {
            Truncation::None => Truncation::None,
            Truncation::Uniform(z) => Truncation::Uniform(z * s),
            Truncation::RealBands { bands, beyond } => Truncation::RealBands {
                bands: bands.iter().map(|b| RealBand { below: b.below * s, height: b.height * s }).collect(),
                beyond: beyond * s,
            },
        };
        let surface = Surface::new(
            Grid2D::new(g.width, g.height, domain, g.values.iter().map(|v| v * s as f32).collect()),
            h.surface.phase.as_deref().cloned(),
            truncation,
            false,
        );
        let occluder = Mesh::new(
            h.occluder.vertices.iter().map(|v| P3::from_vec(v.v * s)).collect(),
            h.occluder.triangles.clone(),
        );
        let layers: Vec<Layer> = scene
            .layers()
            .iter()
            .map(|layer| Layer {
                spec: layer.spec.clone(),
                paths: PolylineSet::new(
                    layer.paths.vertices.iter().map(|v| P3::from_vec(v.v * s)).collect(),
                    layer.paths.offsets.clone(),
                    None,
                ),
            })
            .collect();
        // The scaled drawing is the scaled function: the refiner's |f| and
        // its zeros, read at the point scaled back, scaled up again. The
        // contours are not re-derived under the test, so they pass.
        let refine = h.refine.as_ref().map(|r| {
            let mut out = ContourRefine::new(Arc::new(|_, _, lines, _, _| lines.to_vec()), None);
            out.magnitude = r
                .magnitude
                .clone()
                .map(|m| -> Magnitude { Arc::new(move |p: P2<DomainSpace>| s * m(P2::new(p.x() / s, p.y() / s))) });
            out.zero = r.zero.clone().map(|z| -> kurven_core::surface::Zero {
                Arc::new(move |p: P2<DomainSpace>| {
                    z(P2::new(p.x() / s, p.y() / s)).map(|q| P2::new(q.x() * s, q.y() * s))
                })
            });
            out
        });
        // The walls the wall strokes are tested by, on the perimeter scaled.
        let walls = match &h.walls {
            Walls::Perimeter { perimeter, base } => Walls::Perimeter {
                perimeter: BoundaryPerimeter::new(
                    perimeter
                        .edges()
                        .iter()
                        .map(|e| PerimeterEdge {
                            start: P2::from_vec(e.start.v * s),
                            end: P2::from_vec(e.end.v * s),
                            density: e.density,
                        })
                        .collect(),
                ),
                base: base * s,
            },
            other => other.clone(),
        };
        let scaled = Scene::heightfield_scene(
            surface,
            occluder,
            h.tiles.clone(),
            h.region.clone(),
            h.step,
            layers,
            scene.camera,
            kurven_core::scene::PreviewMode::Drawing,
            scene.depth_tolerance * s,
            refine,
            walls,
        );
        let visible_scaled = scaled.visible_layers(None, None).unwrap();
        let (mut worst, mut structure) = (0.0f64, true);
        for ((layer, a), (_, b)) in visible.iter().zip(&visible_scaled) {
            if a.offsets != b.offsets {
                structure = false;
                eprintln!(
                    "x{s}: {} has other runs: {} vs {} paths, {} vs {} vertices",
                    layer.spec.name,
                    a.count(),
                    b.count(),
                    a.vertices.len(),
                    b.vertices.len()
                );
                continue;
            }
            for (p, q) in a.vertices.iter().zip(&b.vertices) {
                worst = worst.max((p.v * s - q.v).length() / s);
            }
        }
        eprintln!("x{s}: worst {worst:.1e}");
        assert!(
            structure && worst < 1e-6,
            "x{s}: every layer is the drawing scaled, vertex for vertex: worst |Δ| / scale {worst:.1e} over {} layers",
            visible.len()
        );
    }
}

/// "stress: on a reflected or a turned tile the traversal on f is the walk
/// up the line of sight".
#[test]
fn on_a_reflected_or_a_turned_tile_the_traversal_on_f_is_the_walk_up_the_line_of_sight() {
    let _beside = sharing();
    // Two poles of a smooth bump on a grid of eight, so that the line of
    // sight dips under f between crossings, where only the search of
    // each interval, bracketed by the rate the gap changes, finds it.
    // On a second tile that rate is f's slope along the line carried
    // back into the domain, the frame the slope is read in. The walk:
    // f's own top under the line, a five-thousandth of a unit at a time,
    // and the least gap it saw; a point is compared where that is clear
    // of zero by a ten-thousandth.
    let domain = Domain::new(Interval::new(-1.0, 1.0), Interval::new(-1.0, 1.0));
    let mut request = LandscapeRequest::new("0.3/((z - 0.4)^2 + 0.01) + exp(z)", domain, 8);
    request.truncation = Some(Truncation::Uniform(4.0));
    let bundle = build(&request, true);
    let base = Scene::from_bundle(&bundle, &bundle.manifest.presets[0]).heightfield().unwrap().clone();
    for (name, second) in [
        ("reflected", Affine2 { a: -1.0, b: 0.0, tx: 2.0, c: 0.0, d: 1.0, ty: 0.0 }),
        ("turned", Affine2 { a: 0.0, b: -1.0, tx: 2.0, c: 1.0, d: 0.0, ty: 0.0 }),
    ] {
        let h = Heightfield::new(
            base.surface.clone(),
            Mesh::empty(),
            vec![Affine2::IDENTITY, second],
            Region::Full,
            1,
            base.refine.clone(),
            Walls::None,
        );
        let mut rng = SplitMix::new(5);
        let mut points: Vec<P3<WorldSpace>> = Vec::new();
        while points.len() < 120 {
            let x = rng.next(-1.0, 3.0);
            let y = rng.next(-1.0, 1.0);
            let z = h.surface_height(P2::new(x, y));
            if !z.is_finite() {
                continue;
            }
            points.push(P3::new(x, y, z + rng.next(0.001, 0.3)));
        }
        let cameras = stress_cameras();
        let results: Vec<(usize, usize, usize, String)> = par_map(&cameras, |(camera, view)| {
            let vis = HeightfieldVisibility::new(h.clone(), view, 0.0);
            let d = -view.projection_direction();
            let (mut compared, mut wrong, mut seen) = (0, 0, 0);
            let mut example = String::new();
            for &p in &points {
                let (mut t, mut least) = (0.0, f64::INFINITY);
                while t < 20.0 {
                    let q = p.v + t * d;
                    if q.z > 4.01 {
                        break;
                    }
                    if q.x >= -1.0 && q.x <= 3.0 && q.y >= -1.0 && q.y <= 1.0 {
                        least = swift::min(least, q.z - h.surface_height(P2::new(q.x, q.y)));
                        // The least gap only falls: under f by more than the
                        // ten-thousandth the comparison asks, the walk's
                        // verdict is made.
                        if least < -1e-4 {
                            break;
                        }
                    }
                    t += 2e-4;
                }
                if !(least.abs() > 1e-4) {
                    continue;
                }
                compared += 1;
                if least > 0.0 {
                    seen += 1;
                }
                if vis.is_clear(p) != (least > 0.0) {
                    wrong += 1;
                    if example.is_empty() {
                        example = format!(
                            " e.g. {camera} at ({:.4}, {:.4}, {:.4}), least gap {least:.1e}",
                            p.x(),
                            p.y(),
                            p.z()
                        );
                    }
                }
            }
            (compared, wrong, seen, example)
        });
        let compared: usize = results.iter().map(|r| r.0).sum();
        let wrong: usize = results.iter().map(|r| r.1).sum();
        let seen: usize = results.iter().map(|r| r.2).sum();
        let example = results.iter().map(|r| r.3.as_str()).find(|e| !e.is_empty()).unwrap_or("");
        eprintln!("{name}: {wrong} of {compared} disagree, {seen} clear");
        assert!(
            wrong == 0 && compared > 2000 && seen > compared / 10 && seen < compared,
            "{name}: the traversal and the walk agree: {wrong} of {compared} disagree, {seen} clear{example}"
        );
    }
}

/// "stress: on a 4x4 grid the traversal is the brute-force ray cast, exactly".
#[test]
fn on_a_4x4_grid_the_traversal_is_the_brute_force_ray_cast_exactly() {
    let _beside = sharing();
    // Heights drawn at random on a grid of four cells a side: every
    // cell is a different slope, every crossing is a crease, and no
    // approximation has anywhere to hide. The traversal is held to a ray
    // cast against the triangles the depth pass draws -- the grid's
    // cells as it splits them, the truncation edge pieces, the walls -- at
    // points on the drawn surface, on its grid lines, and anywhere in
    // the box; under every camera; at three depth tolerances. The oracle is
    // `exactly_hidden`: brute force over the triangles, with the depth tolerance
    // meaning what the traversal means by it, and at a depth tolerance of zero also
    // the plain question of whether the ray crosses a triangle at all.
    // The two must agree wherever the oracle's own answer is not within
    // rounding of changing.
    for (label, truncation) in [("uncut", Truncation::None), ("truncated at 2.2", Truncation::Uniform(2.2))] {
        let field = random_heightfield(4, 0xC0FFEE, truncation, 3.0);
        let base = -1.0;
        let mesh = drawn_triangles(&field, base);
        let mut rng = SplitMix::new(42);
        let d = field.surface.domain();
        let (blo, bhi) = (
            Vec2::new(d.real.lo.min(d.real.hi), d.imag.lo.min(d.imag.hi)),
            Vec2::new(d.real.lo.max(d.real.hi), d.imag.lo.max(d.imag.hi)),
        );
        let mut points: Vec<P3<WorldSpace>> = Vec::new();
        for _ in 0..300 {
            let x = rng.next(d.real.lo, d.real.hi);
            let y = rng.next(d.imag.lo, d.imag.hi);
            let p = P2::<WorldSpace>::new(x, y);
            points.push(P3::new(p.x(), p.y(), field.drawn_height(p)));
        }
        for _ in 0..100 {
            let x = rng.next(d.real.lo, d.real.hi);
            let y = rng.next(d.imag.lo, d.imag.hi);
            let z = rng.next(base, 4.0);
            points.push(P3::new(x, y, z));
        }
        // On the grid lines and diagonals exactly, where two triangles meet.
        for _ in 0..100 {
            let i = (rng.next(0.0, 4.999) as i64) as f64;
            let t = rng.next(0.0, 4.0);
            let on_x = rng.next(0.0, 1.0) < 0.5;
            let (u, v) = if on_x { (i, t) } else { (t, i) };
            let p = P2::<WorldSpace>::new(d.real.lo + u * d.real.length() / 4.0, d.imag.lo + v * d.imag.length() / 4.0);
            points.push(P3::new(p.x(), p.y(), field.drawn_height(p)));
        }
        let start = Instant::now();
        let cameras = stress_cameras();
        let results: Vec<(usize, usize, usize, String)> = par_map(&cameras, |(name, view)| {
            let toward = -view.projection_direction();
            let per_unit = per_unit(view, toward);
            let (mut compared, mut disagree, mut undecidable) = (0, 0, 0);
            let mut example = String::new();
            for depth_tolerance in [0.0, 1e-6, 0.05] {
                let vis = HeightfieldVisibility::new(field.clone(), view, depth_tolerance);
                if (vis.tolerance_along_projection() - depth_tolerance / per_unit).abs()
                    > 1e-12 * (1.0 + depth_tolerance)
                {
                    disagree += 1;
                }
                for &p in &points {
                    let strictly_inside =
                        p.x() > blo.x + 1e-9 && p.x() < bhi.x - 1e-9 && p.y() > blo.y + 1e-9 && p.y() < bhi.y - 1e-9;
                    let hidden = |m: f64| exactly_hidden(&mesh, p.v, toward, m, per_unit, strictly_inside, 0.5);
                    let truth = hidden(depth_tolerance);
                    // The two brute forces must agree with each other too,
                    // where neither is on a fence.
                    if depth_tolerance == 0.0
                        && first_hit(&mesh, p.v, toward, 1e-10).is_some() != truth
                        && hidden(-1e-9) == hidden(1e-9)
                    {
                        disagree += 1;
                    }
                    // The oracle a hair either side of the depth tolerance: where
                    // it changes its mind, rounding decides and no
                    // answer is wrong.
                    if !(hidden(depth_tolerance - 1e-9) == truth && hidden(depth_tolerance + 1e-9) == truth) {
                        undecidable += 1;
                        continue;
                    }
                    compared += 1;
                    if vis.is_clear(p) == truth {
                        disagree += 1;
                        if example.is_empty() {
                            example = format!(
                                "e.g. {name} m={depth_tolerance} at ({:.3}, {:.3}, {:.3}): oracle {}",
                                p.x(),
                                p.y(),
                                p.z(),
                                if truth { "hidden" } else { "seen" }
                            );
                        }
                    }
                }
            }
            (compared, disagree, undecidable, example)
        });
        let elapsed = start.elapsed();
        let compared: usize = results.iter().map(|r| r.0).sum();
        let disagree: usize = results.iter().map(|r| r.1).sum();
        let undecidable: usize = results.iter().map(|r| r.2).sum();
        let example = results.iter().map(|r| r.3.as_str()).find(|e| !e.is_empty()).unwrap_or("");
        assert!(
            disagree == 0 && compared > 40_000,
            "{label}: the traversal agrees with the ray cast at every decidable point: {disagree} of {compared} \
             disagree, {undecidable} within rounding of the depth tolerance {example}"
        );
        assert!(elapsed < Duration::from_secs(60), "{label}: in reasonable time: {elapsed:?}");
    }
}

/// "stress: the facing read off the grid converges on the drawn triangles".
#[test]
fn the_facing_read_off_the_grid_converges_on_the_drawn_triangles() {
    // The one approximation left in the predicate, measured rather than
    // hidden. Curves on the surface are cut by the central-difference normal
    // -- the normal the fold curves are traced from, so a contour ends on
    // the outline as drawn -- while the solid it is traversed through is
    // the triangulated grid, whose triangles turn at grid lines. The
    // two can disagree about which way the surface faces only within a
    // cell of a fold, and that band shrinks with the cell: the
    // disagreement, as a fraction of random surface points, must fall
    // as the grid is refined and be small at a drawing grid. On a
    // 4x4 grid it is large, and that is a fact about central
    // differences on four cells, not a tolerance chosen to pass.
    let height = |x: f64, y: f64| -> f64 {
        2.5 * real::exp(-(x * x + 0.6 * y * y) / 0.3) + 0.4 * real::sin(3.0 * x) * real::cos(2.0 * y)
    };
    let view = test_camera(-40.0, 30.0, 0.0);
    let sight = view.projection_direction();
    let mut rates: Vec<(usize, f64)> = Vec::new();
    for cells in [4usize, 8, 16, 32, 64, 128] {
        let n = cells + 1;
        let domain = Domain::new(Interval::new(-1.5, 1.5), Interval::new(-1.5, 1.5));
        let mut values = vec![0f32; n * n];
        for j in 0..n {
            for i in 0..n {
                values[j * n + i] =
                    height(-1.5 + 3.0 * i as f64 / cells as f64, -1.5 + 3.0 * j as f64 / cells as f64) as f32;
            }
        }
        let field = Heightfield::new(
            Surface::new(Grid2D::new(n, n, domain, values), None, Truncation::None, false),
            Mesh::empty(),
            vec![Affine2::IDENTITY],
            Region::Full,
            1,
            None,
            Walls::None,
        );
        let vis = HeightfieldVisibility::new(field.clone(), &view, 0.0);
        let mut rng = SplitMix::new(cells as u64);
        let cell = 3.0 / cells as f64;
        let mut differ = 0;
        let total = 20_000;
        for _ in 0..total {
            let x = rng.next(-1.4, 1.4);
            let y = rng.next(-1.4, 1.4);
            let p = P2::<WorldSpace>::new(x, y);
            // The triangle's own normal: the drawn surface's one-sided slopes
            // a hair into the triangle the point is in.
            let e = 1e-6 * cell;
            let h0 = field.drawn_height(p);
            let hx = (field.drawn_height(P2::new(p.x() + e, p.y())) - h0) / e;
            let hy = (field.drawn_height(P2::new(p.x(), p.y() + e)) - h0) / e;
            let triangle = -Vec3::new(-hx, -hy, 1.0).dot(sight);
            if (triangle > 0.0) != (vis.facing(p) > 0.0) {
                differ += 1;
            }
        }
        rates.push((cells, differ as f64 / total as f64));
    }
    let text: Vec<String> = rates.iter().map(|(c, r)| format!("{c}: {:.2}%", 100.0 * r)).collect();
    let text = text.join("  ");
    let falls = rates.windows(2).all(|w| !(w[1].1 > 0.75 * w[0].1 && w[0].1 > 0.002));
    assert!(falls, "the disagreement falls as the grid is refined: {text}");
    assert!(rates.last().unwrap().1 < 0.01, "and is under 1% of surface points at 128 cells: {text}");
}
