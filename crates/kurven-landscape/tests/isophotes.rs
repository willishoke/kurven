//! Isophotes.swift: isophotes and half-Lambert tone on curved faces (phase
//! 11a), held to closed forms -- the graph of exp(z), where
//! n · l = 1 / √(1 + e^{2x}); flat faces, which have none -- and the
//! visibility of both kinds of layer, held to the dense cast against f. The
//! round torus's suites build it through `SurfaceDrawing.build` (Phase 4)
//! and are stubbed below, ignored; "isophotes and tone: seen exactly where
//! the torus's own cast and the grid landscape's cast say" holds its grid
//! landscape here, by the composite gate (`gate::composite_verdict`), and
//! its torus in an ignored stub. Swift's tolerances.

#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::type_complexity)]

mod common;
mod gate;

use std::collections::HashSet;

use common::{build, densely_hidden, per_unit};
use kurven_core::camera::Camera;
use kurven_core::face::{Face, FaceID};
use kurven_core::fixtures;
use kurven_core::hatch::HatchContext;
use kurven_core::hatching::{FaceKind, FaceSelector, Hatching, Tone};
use kurven_core::hidden_line::HeightfieldVisibility;
use kurven_core::json::{self, JsonObject};
use kurven_core::manifest::{
    Affine2, ContourField, Domain, HeightPolicy, Interval, LayerRole, LayerSource, LayerSpec, Provenance, RealBand,
    Region, Restriction, SurfaceManifest, Truncation,
};
use kurven_core::parametric::ParametricSurface;
use kurven_core::polyhedron_bundle::PolyhedronBundle;
use kurven_core::polyline::PolylineSet;
use kurven_core::scene::{PreviewMode, Scene};
use kurven_core::simd::{Vec2, Vec3};
use kurven_core::spaces::{AABB, P2, P3, ViewSpace, WorldSpace};
use kurven_core::surface_request::{Shape, SurfaceRequest};
use kurven_landscape::{Catalog, LandscapeRequest};
use kurven_math::real;
use kurven_math::swift;
use kurven_surfaces::{SurfaceDrawing, SurfaceRequestExt};

/// exp(z) over [−1, 1]² at 48 samples, untruncated and with no layers, as
/// the graph suites build it with the refiner.
fn exp_landscape() -> kurven_core::bundle::KurvenBundle {
    let mut request =
        LandscapeRequest::new("exp(z)", Domain::new(Interval::new(-1.0, 1.0), Interval::new(-1.0, 1.0)), 48);
    request.truncation = Some(Truncation::None);
    request.layers = Some(Vec::new());
    build(&request, true)
}

/// Swift's `min(by:)` on the distance to `x`: the first nearest of `levels`.
fn nearest(levels: &[f64], x: f64) -> f64 {
    let mut best = levels[0];
    for &c in &levels[1..] {
        if (c - x).abs() < (best - x).abs() {
            best = c;
        }
    }
    best
}

/// The largest of `xs`, or `None` when there are none (Swift's `max()`).
fn largest(xs: impl Iterator<Item = f64>) -> Option<f64> {
    xs.fold(None, |m, x| Some(m.map_or(x, |m: f64| if x > m { x } else { m })))
}

/// "isophotes: a flat face has none, at any level, its own constant n · l
/// included": a polyhedron's faces, and a banded tower's plateaus, riser and
/// walls, have no isophote field and draw no isophote, while the graph,
/// curved, has one.
#[test]
fn a_flat_face_has_no_isophotes_at_any_level() {
    let light = Vec3::new(-0.4, -0.6, 0.7);
    let l = light.normalized();
    let (mut faces, mut drawn, mut fields) = (0, 0, 0);
    let mut kinds = HashSet::new();
    for name in ["frame", "cube"] {
        let bundle = PolyhedronBundle::read(fixtures::shared(&format!("contract/polyhedron_{name}.kurven")))
            .unwrap_or_else(|e| panic!("polyhedron_{name}: {e:?}"));
        let solid = bundle.solid();
        for i in 0..solid.faces.len() {
            let f = solid.face(i);
            faces += 1;
            kinds.insert(f.id.kind());
            let c = solid.frames[i].normal.dot(l);
            if f.isophote(light, &f.copies[0]).is_some() {
                fields += 1;
            }
            drawn += f.isophotes(light, &[c, 0.0, 0.5, -0.5], &f.copies[0]).iter().map(|(_, p)| p.len()).sum::<usize>();
        }
    }
    // A tower under two bands with walls: plateaus, a riser, walls.
    let mut request =
        LandscapeRequest::new("1/(z - 0.45)", Domain::new(Interval::new(-0.5, 1.5), Interval::new(-1.0, 1.0)), 32);
    request.truncation = Some(Truncation::RealBands { bands: vec![RealBand { below: 0.5, height: 2.0 }], beyond: 4.0 });
    let built = build(&request, true);
    let refine = built.refine.as_ref();
    let context = HatchContext::new(&built.manifest.occluder);
    let landscape_faces =
        built.surface.faces(&[Affine2::IDENTITY], &Region::Full, &built.manifest.occluder.walls, refine);
    for f in landscape_faces.iter().filter(|f| f.is_flat()) {
        faces += 1;
        kinds.insert(f.id.kind());
        let p = P2::new(
            f.domain.u.range.lo + 0.5 * f.domain.u.range.length(),
            f.domain.v.range.lo + 0.5 * f.domain.v.range.length(),
        );
        let c = f.outward_normal(p, &f.copies[0]).unwrap_or(Vec3::new(0.0, 0.0, 1.0)).dot(l);
        if f.isophote(light, &f.copies[0]).is_some() {
            fields += 1;
        }
        drawn += f.isophotes(light, &[c, 0.0], &f.copies[0]).iter().map(|(_, p)| p.len()).sum::<usize>();
    }
    for selector in [
        FaceSelector::Plateaus { tiled: false },
        FaceSelector::Walls { edges: vec![0, 1, 2, 3], end_lines: false, base: 0.0, top_offset: 0.0 },
        FaceSelector::Risers { tiled: false, end_lines: false },
    ] {
        drawn += built
            .surface
            .contours(
                ContourField::Isophote,
                &[0.0, 0.7, l.z],
                &selector,
                &Restriction::All,
                &context,
                refine,
                Some(light),
            )
            .len();
    }
    let wanted: HashSet<FaceKind> = [FaceKind::Plateau, FaceKind::Wall, FaceKind::Riser, FaceKind::Solid].into();
    assert!(
        faces > 20 && kinds == wanted && fields == 0 && drawn == 0,
        "no isophote field and no curve on a polyhedron's faces, plateaus, walls or a riser: {faces} faces of kinds \
         {kinds:?}, {fields} fields, {drawn} curves"
    );
    let graph = landscape_faces.iter().find(|f| f.id == FaceID::Graph).expect("the landscape has a graph");
    assert!(!graph.is_flat() && graph.isophote(light, &graph.copies[0]).is_some(), "the graph, curved, has one");
}

/// "isophotes: on the graph of exp(z) lit along z, n · l = 1/√(1 + e^{2x}),
/// and the isophote at c is Re z = ½ log(1/c² − 1)": each isophote a line
/// across the domain on its level, under a light at a slant still on
/// n · l = c by the jet and by the closed form, and the layer's curves lifted
/// onto |f|.
#[test]
fn on_the_graph_of_exp_z_lit_along_z_the_isophote_is_a_line_of_re_z() {
    let built = exp_landscape();
    let (surface, refine) = (&built.surface, built.refine.as_ref());
    let graph = surface.graph(&[Affine2::IDENTITY], &Region::Full, refine).expect("exp(z) has a graph");
    let g = &graph.copies[0];
    let levels = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let z = Vec3::new(0.0, 0.0, 1.0);
    let (mut residual, mut closed, mut spans, mut counts) = (0.0f64, 0.0f64, true, Vec::new());
    for (c, paths) in graph.isophotes(z, &levels, g) {
        let x = 0.5 * real::log(1.0 / (c * c) - 1.0);
        counts.push(paths.len());
        for path in &paths {
            for &p in path {
                residual = swift::max(residual, (graph.outward_normal(p, g).unwrap().dot(z) - c).abs());
                closed = swift::max(closed, (p.x() - x).abs());
            }
            let lo = path.iter().map(|p| p.y()).fold(f64::INFINITY, swift::min);
            let hi = path.iter().map(|p| p.y()).fold(f64::NEG_INFINITY, swift::max);
            if (lo + 1.0).abs() > 1e-12 || (hi - 1.0).abs() > 1e-12 {
                spans = false;
            }
        }
    }
    // The graph's slope is f's, read by central differences a
    // ten-thousandth of a cell wide (`Surface.graphNormal`), whose rounding
    // is about 1e-11 here: so n · l, read off the graph's own jet, is c to
    // that, and so is the closed form.
    let mut failures = Vec::new();
    if !(residual <= 1e-10) {
        failures.push(format!(
            "every vertex is on n · l = c, the normal read off the graph's own jet: worst {residual:.1e}"
        ));
    }
    if !(closed <= 1e-10 && counts.iter().all(|&n| n == 1) && spans) {
        failures.push(format!(
            "the isophote at c is the line Re z = ½ log(1/c² − 1), across the whole domain: worst {closed:.1e} off \
             the line; paths per level {counts:?}, spanning {spans}"
        ));
    }
    // Under a light at a slant the isophotes curve; still on n · l = c. Held
    // both to the graph's own jet and to its normal in closed form,
    // (−e^x, 0, 1)/√(1 + e^{2x}), which no code under test computes.
    let slant = Vec3::new(1.0, 0.5, 2.0);
    let l = slant.normalized();
    let (mut slanted, mut slanted_closed, mut vertices) = (0.0f64, 0.0f64, 0);
    for (c, paths) in graph.isophotes(slant, &[0.5, 0.7, 0.85, 0.95], g) {
        for &p in paths.iter().flatten() {
            vertices += 1;
            slanted = swift::max(slanted, (graph.outward_normal(p, g).unwrap().dot(l) - c).abs());
            let e = real::exp(p.x());
            let n = Vec3::new(-e, 0.0, 1.0) / real::sqrt(1.0 + e * e);
            slanted_closed = swift::max(slanted_closed, (n.dot(l) - c).abs());
        }
    }
    if !(slanted <= 1e-10 && slanted_closed <= 1e-10 && vertices > 40) {
        failures.push(format!(
            "under a light at a slant, every vertex is on n · l = c, n the jet's and the closed form's: worst \
             {slanted:.1e} by the jet, {slanted_closed:.1e} by the closed form, over {vertices} vertices"
        ));
    }
    // Through the layer's own path: lifted onto |f|.
    let layer = surface.contours(
        ContourField::Isophote,
        &levels,
        &FaceSelector::Graph { tiled: false },
        &Restriction::All,
        &HatchContext::new(&built.manifest.occluder),
        refine,
        Some(z),
    );
    let lifted = largest(layer.iter().flatten().map(|p| (p.z() - real::exp(p.x())).abs() / real::exp(p.x())))
        .unwrap_or(f64::INFINITY);
    if !(layer.len() == levels.len() && lifted <= 1e-12) {
        failures
            .push(format!("the layer's curves lie on the graph, at |f|: {} paths, worst {lifted:.1e}", layer.len()));
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// "isophotes and tone: on a tiled graph, each copy under its own normal,
/// reflected and turned": exp(z) over [−1, 1]², copied by a reflection
/// across Re z = 2 and a quarter turn moved to Re z = 8, each copy's
/// isophotes on n · l = c and its tone's width step the copy's darkness's,
/// the copy's normal in closed form.
#[test]
fn on_a_tiled_graph_each_copy_under_its_own_normal() {
    let built = exp_landscape();
    let tiles = vec![
        Affine2::IDENTITY,
        Affine2 { a: -1.0, b: 0.0, tx: 4.0, c: 0.0, d: 1.0, ty: 0.0 },
        Affine2 { a: 0.0, b: -1.0, tx: 8.0, c: 1.0, d: 0.0, ty: 0.0 },
    ];
    let context = HatchContext { perimeter: None, region: Region::Full, tiles: tiles.clone(), base: 0.0 };
    let light = Vec3::new(0.6, -0.8, 1.2);
    let l = light.normalized();
    // The tile a world point is a copy of, the point of the domain it came
    // from, and the copy's outward normal there in closed form.
    let origin = |w: P3<WorldSpace>| -> Option<(usize, Vec2, Vec3)> {
        for (k, tile) in tiles.iter().enumerate() {
            let q = tile.inverse().unwrap().apply(w);
            if !(q.x().abs() <= 1.0 + 1e-9 && q.y().abs() <= 1.0 + 1e-9) {
                continue;
            }
            let e = real::exp(q.x());
            let xy = Vec2::new(tile.a * -e + tile.b * 0.0, tile.c * -e + tile.d * 0.0);
            return Some((k, Vec2::new(q.x(), q.y()), Vec3::new(xy.x, xy.y, 1.0) / real::sqrt(1.0 + e * e)));
        }
        None
    };
    let levels = [0.3, 0.5, 0.7, 0.85];
    let isophotes = built.surface.contours(
        ContourField::Isophote,
        &levels,
        &FaceSelector::Graph { tiled: true },
        &Restriction::All,
        &context,
        built.refine.as_ref(),
        Some(light),
    );
    let (mut off, mut off_f, mut lost, mut per_tile) = (0.0f64, 0.0f64, 0, [0; 3]);
    for path in &isophotes {
        let Some((tile, _, n)) = origin(path[0]) else {
            lost += 1;
            continue;
        };
        per_tile[tile] += 1;
        // The path's level: the nearest to its first vertex's n · l.
        let c = nearest(&levels, n.dot(l));
        for &w in path {
            let Some((_, p, n)) = origin(w) else {
                lost += 1;
                continue;
            };
            off = swift::max(off, (n.dot(l) - c).abs());
            off_f = swift::max(off_f, (w.z() - real::exp(p.x)).abs() / real::exp(p.x));
        }
    }
    let mut failures = Vec::new();
    if !(off <= 1e-10 && off_f <= 1e-12 && lost == 0 && per_tile.iter().all(|&n| n > 0)) {
        failures.push(format!(
            "every isophote vertex of every copy is on n · l = c, the copy's normal in closed form: worst {off:.1e}, \
             off |f| by {off_f:.1e}; paths per tile {per_tile:?}, {lost} lost"
        ));
    }
    let hatching = Hatching::new(0.0, 0.05, None, Tone::HalfLambert { light }, 6);
    let (mut wrong, mut samples, mut changes) = (0, 0, 0);
    let mut tiles_hatched = HashSet::new();
    for (step, paths) in built.surface.graph_hatching(&hatching, true, &context, built.refine.as_ref()) {
        for path in &paths {
            for w in path.windows(2) {
                let (a, b) = (w[0], w[1]);
                for t in [0.0, 0.5, 1.0] {
                    let Some((tile, _, n)) = origin(P3::from_vec(a.v + (b.v - a.v) * t)) else {
                        wrong += 1;
                        continue;
                    };
                    tiles_hatched.insert(tile);
                    let nd = 6.0 * (1.0 - n.dot(l)) / 2.0;
                    if !((nd - nd.round()).abs() > 1e-9) {
                        continue;
                    }
                    samples += 1;
                    if nd.ceil() as i64 != step {
                        wrong += 1;
                    }
                }
            }
            for end in [path[0], path[path.len() - 1]] {
                let Some((_, p, n)) = origin(end) else { continue };
                if !(p.x.abs() < 1.0 - 1e-12) {
                    continue;
                }
                let nd = 6.0 * (1.0 - n.dot(l)) / 2.0;
                if (nd - nd.round()).abs() <= 1e-10 {
                    changes += 1;
                } else {
                    wrong += 1;
                }
            }
        }
    }
    let mut hatched: Vec<usize> = tiles_hatched.into_iter().collect();
    hatched.sort();
    if !(wrong == 0 && samples > 3000 && changes > 30 && hatched == [0, 1, 2]) {
        failures.push(format!(
            "the tone on every copy has the width step of the copy's darkness at every sample, changing on its \
             isophotes: {wrong} wrong of {samples} samples, {changes} changes, tiles {hatched:?}"
        ));
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// "tone: on the graph, along a coordinate at the plateaus' levels, cut
/// where the width step changes": half-Lambert hatching along Re z on the
/// graph of exp(z), every piece on its line Im z = c and on |f|, its width
/// step the darkness's at every sample, read off the graph's jet, changing
/// only on an isophote.
#[test]
fn tone_on_the_graph_runs_along_a_coordinate_cut_where_the_width_step_changes() {
    let built = exp_landscape();
    let light = Vec3::new(1.0, 0.5, 2.0);
    let l = light.normalized();
    let hatching = Hatching::new(0.0, 0.05, None, Tone::HalfLambert { light }, 6);
    let refine = built.refine.as_ref();
    let graph = built.surface.graph(&[Affine2::IDENTITY], &Region::Full, refine).expect("exp(z) has a graph");
    let copy = &graph.copies[0];
    let steps = built.surface.graph_hatching(&hatching, false, &HatchContext::new(&built.manifest.occluder), refine);
    let levels = Face::hatch_levels(&graph.domain.v, 0.05);
    let (mut off_line, mut wrong, mut samples, mut changes, mut on_f) = (0.0f64, 0, 0, 0, 0.0f64);
    let darkness_steps = |p: Vec3| -> f64 {
        let nl = graph.outward_normal(P2::new(p.x, p.y), copy).unwrap().dot(l);
        6.0 * (1.0 - nl) / 2.0
    };
    for (step, paths) in &steps {
        for path in paths {
            let y = nearest(&levels, path[0].y());
            off_line = swift::max(off_line, largest(path.iter().map(|p| (p.y() - y).abs())).unwrap());
            on_f = swift::max(
                on_f,
                largest(path.iter().map(|p| (p.z() - real::exp(p.x())).abs() / real::exp(p.x()))).unwrap(),
            );
            for w in path.windows(2) {
                let (a, b) = (w[0], w[1]);
                for t in [0.0, 0.5, 1.0] {
                    let nd = darkness_steps(a.v + (b.v - a.v) * t);
                    if !((nd - nd.round()).abs() > 1e-9) {
                        continue;
                    }
                    samples += 1;
                    if nd.ceil() as i64 != *step {
                        wrong += 1;
                    }
                }
            }
            for end in [path[0], path[path.len() - 1]] {
                if !(end.x().abs() < 1.0 - 1e-12) {
                    continue;
                }
                let nd = darkness_steps(end.v);
                if (nd - nd.round()).abs() <= 1e-10 {
                    changes += 1;
                } else {
                    wrong += 1;
                }
            }
        }
    }
    let mut failures = Vec::new();
    if !(off_line <= 1e-15 && on_f <= 1e-12 && levels.len() == 39) {
        failures.push(format!(
            "every piece lies on its line Im z = c, on the graph at |f|: {} lines, off by {off_line:.1e}, off |f| by \
             {on_f:.1e}",
            levels.len()
        ));
    }
    if !(wrong == 0 && samples > 1000 && changes > 20) {
        failures.push(format!(
            "its width step is the darkness's at every sample, read off the graph's jet, and it changes only on an \
             isophote, to the jet's rounding: {wrong} wrong of {samples} samples, {changes} changes"
        ));
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Swift's `SegmentIndex` (Composite.swift): a drawing's segments in view
/// space, bucketed on a 400 by 400 grid over `size` from `lo`, with the
/// ends of its runs. (As integral_curves.rs ports it.)
struct SegmentIndex {
    lo: Vec2,
    cell: f64,
    n: usize,
    buckets: Vec<Vec<(Vec3, Vec3)>>,
    ends: Vec<Vec<Vec3>>,
}

impl SegmentIndex {
    fn new(paths: &PolylineSet<ViewSpace>, lo: Vec2, size: f64) -> SegmentIndex {
        let n = 400;
        let mut index = SegmentIndex {
            lo,
            cell: size / n as f64,
            n,
            buckets: vec![Vec::new(); n * n],
            ends: vec![Vec::new(); n * n],
        };
        for path in paths.paths() {
            for w in path.windows(2) {
                let (a, b) = (w[0], w[1]);
                let (i0, j0) = index.key(Vec2::new(swift::min(a.x(), b.x()), swift::min(a.y(), b.y())));
                let (i1, j1) = index.key(Vec2::new(swift::max(a.x(), b.x()), swift::max(a.y(), b.y())));
                for i in i0..=i1 {
                    for j in j0..=j1 {
                        index.buckets[i * n + j].push((a.v, b.v));
                    }
                }
            }
            for e in [path[0], path[path.len() - 1]] {
                let (i, j) = index.key(Vec2::new(e.x(), e.y()));
                index.ends[i * n + j].push(e.v);
            }
        }
        index
    }

    fn key(&self, p: Vec2) -> (usize, usize) {
        let k = (p - self.lo) / self.cell;
        let clamp = |x: f64| -> usize {
            let f = x.floor();
            if !(f > 0.0) {
                0
            } else if f >= (self.n - 1) as f64 {
                self.n - 1
            } else {
                f as usize
            }
        };
        (clamp(k.x), clamp(k.y))
    }

    /// Whether a segment passes within `tolerance` of `p`, in space.
    fn holds(&self, p: Vec3, tolerance: f64) -> bool {
        let (i, j) = self.key(Vec2::new(p.x, p.y));
        self.buckets[i * self.n + j].iter().any(|&(a, b)| {
            let ab = b - a;
            let l2 = ab.dot(ab);
            let t = if l2 > 0.0 { swift::min(swift::max((p - a).dot(ab) / l2, 0.0), 1.0) } else { 0.0 };
            (p - (a + t * ab)).length() <= tolerance
        })
    }

    /// The nearest run end to `p`, within the cells about it.
    fn nearest_end(&self, p: Vec3) -> f64 {
        let (i, j) = self.key(Vec2::new(p.x, p.y));
        let mut best = f64::INFINITY;
        for di in -1i64..=1 {
            for dj in -1i64..=1 {
                let (a, b) = (i as i64 + di, j as i64 + dj);
                if a < 0 || a >= self.n as i64 || b < 0 || b >= self.n as i64 {
                    continue;
                }
                for &e in &self.ends[a as usize * self.n + b as usize] {
                    best = swift::min(best, (e - p).length());
                }
            }
        }
        best
    }
}

/// "isophotes and tone: on f, seen exactly where the dense cast against f
/// says": the cubic at 64 with isophotes and four widths of half-Lambert
/// tone on its graph, each vertex drawn by `Scene::visible_layers` exactly
/// where the dense cast against f (Stress.swift's `denselyHidden`) sees it.
#[test]
fn on_f_seen_exactly_where_the_dense_cast_against_f_says() {
    let built = build(&LandscapeRequest::from_preset(Catalog::native().preset("cubic").unwrap(), 64), true);
    let light = Vec3::new(-0.4, -0.6, 0.7);
    let mut m = built.manifest.clone();
    m.layers = vec![
        LayerSpec::new(
            "isophotes",
            LayerRole::Scaffold,
            LayerSource::Contour {
                field: ContourField::Isophote,
                levels: vec![-0.6, -0.3, 0.0, 0.3, 0.6, 0.9],
                restriction: Restriction::All,
                faces: FaceSelector::Graph { tiled: false },
                light: Some(light),
            },
            0.3,
            HeightPolicy::Surface,
        ),
        LayerSpec::new(
            "tone",
            LayerRole::Scaffold,
            LayerSource::Hatching {
                hatching: Hatching::new(0.0, 0.06, None, Tone::HalfLambert { light }, 4),
                faces: FaceSelector::Graph { tiled: false },
            },
            1.0,
            HeightPolicy::Surface,
        ),
    ];
    let bundle = built.restyled(&m);
    let scene = Scene::from_bundle(&bundle, &bundle.manifest.presets[0]);
    let h = scene.heightfield().unwrap();
    let view = scene.camera.view;
    let magnitude = h.refine.as_ref().unwrap().magnitude.clone().unwrap();
    let vis = HeightfieldVisibility::new(h.clone(), &view, scene.depth_tolerance);
    let toward = -view.projection_direction();
    let per_unit = per_unit(&view, toward);
    let grid = &*h.surface.height;
    let cell = swift::max(
        grid.domain.real.length().abs() / (grid.width - 1) as f64,
        grid.domain.imag.length().abs() / (grid.height - 1) as f64,
    );
    let (mut compared, mut wrong, mut layers, mut example) = (0, 0, 0, String::new());
    for (layer, seen) in scene.visible_layers(None, None).unwrap() {
        layers += 1;
        let mapped = layer.paths.mapped(&view);
        let Some(bounds) = AABB::of(seen.vertices.iter().chain(&mapped.vertices)) else { continue };
        let size = swift::max(bounds.hi.x - bounds.lo.x, bounds.hi.y - bounds.lo.y);
        let index =
            SegmentIndex::new(&seen, Vec2::new(bounds.lo.x, bounds.lo.y) - Vec2::splat(1e-6 * size), 1.00001 * size);
        let vertices = &layer.paths.vertices;
        let truths: Vec<Option<bool>> = common::par_map(vertices, |&v| {
            let f = vis.facing(v.xy());
            if !(f.abs() > 1e-9) {
                return None;
            }
            if !(f > 0.0) {
                return Some(false);
            }
            Some(densely_hidden(h, &magnitude, v.v, toward, vis.depth_tolerance, per_unit, cell).is_none())
        });
        for (v, truth) in vertices.iter().zip(truths) {
            let p = view.apply(*v).v;
            let Some(truth) = truth else { continue };
            if !(index.nearest_end(p) >= 1e-6 * size) {
                continue;
            }
            compared += 1;
            if truth != index.holds(p, 1e-9 * size) {
                wrong += 1;
                if example.is_empty() {
                    example = format!(
                        " e.g. {} at ({:.4}, {:.4}, {:.4}): the cast says {}",
                        layer.spec.name,
                        v.x(),
                        v.y(),
                        v.z(),
                        if truth { "seen" } else { "hidden" }
                    );
                }
            }
        }
    }
    assert!(
        wrong == 0 && compared > 3000 && layers == 5,
        "every vertex of the isophotes and of each width of the tone is drawn exactly where the dense cast sees it: \
         {layers} layers, {compared} compared, {wrong} wrong{example}"
    );
}

/// A layer source read as `geometry`'s manifest reads it: a landscape's
/// directly, a parametric surface's as the one layer of a round torus's
/// manifest (Isophotes.swift's `source(_:on:)`). The error as text.
fn source(text: &str, geometry: &str) -> Result<LayerSource, String> {
    match geometry {
        "surface" => {
            let provenance = Provenance {
                function: String::new(),
                example: String::new(),
                params: JsonObject::new(),
                cpu_count: 1,
                git_sha: String::new(),
            };
            let empty = SurfaceManifest::new(
                SurfaceRequest::new(Shape::Torus { major: 2.0, minor: 1.0 }),
                Vec::new(),
                Vec::new(),
                provenance,
            );
            let layer = format!(
                r#"{{"name": "x", "role": "scaffold", "width": 0.3, "heightPolicy": "surface", "source": {text}}}"#
            );
            let manifest = empty.canonical_json().replace(r#""layers":[]"#, &format!(r#""layers":[{layer}]"#));
            let json = json::parse(&manifest).map_err(|e| e.to_string())?;
            let m = SurfaceManifest::from_json(&json).map_err(|e| e.to_string())?;
            Ok(m.layers[0].source.clone())
        }
        _ => {
            let json = json::parse(text).map_err(|e| e.to_string())?;
            LayerSource::from_json(&json).map_err(|e| e.to_string())
        }
    }
}

/// Whether reading `text` as `geometry`'s is refused, saying `words`.
fn refused(text: &str, geometry: &str, words: &str) -> bool {
    match source(text, geometry) {
        Ok(_) => false,
        Err(e) => e.contains(words),
    }
}

/// sn in relief at `amplitude` over `floor`, as the manifest suite asks.
fn relief(amplitude: f64, floor: Option<f64>) -> SurfaceRequest {
    SurfaceRequest::new(Shape::Relief {
        modulus: 0.64,
        major: 2.0,
        minor: 1.0,
        amplitude,
        repeats: (1, 1),
        grid: 120,
        turn: 0.0,
        floor,
        soft: false,
    })
}

/// "isophotes and tone: the manifest", every part but those that build a
/// surface (`SurfaceDrawing.build`, Phase 4) or ask `SurfaceRequest.unsupported`
/// (KurvenDynamics' SurfaceDrawing.swift, Phase 4) its hatching-vertex rule:
/// an isophote's and a curved face's hatching's sources decoded, round
/// tripped and refused; whether a relief bounds a solid; and the refusals of
/// `SurfaceManifest.unsupported`, which is the first clause of
/// `SurfaceRequest.unsupported` and the one these sources reach.
#[test]
fn the_manifest() {
    let iso = r#"{"kind": "contour", "field": "isophote", "levels": [0.5], "restriction": {"kind": "all"}, "light": [0, 0, 2], "faces": "#;
    let read = |text: String, geometry: &str| source(&text, geometry).unwrap_or_else(|e| panic!("{text}: {e}"));
    let graph_iso = read(iso.to_string() + r#"{"kind": "graph", "tiled": false}}"#, "landscape");
    let surface_iso = read(iso.to_string() + r#"{"kind": "surface"}}"#, "surface");
    let tone = r#"{"kind": "hatching", "angle": 0, "spacing": 0.1, "widthSteps": 4, "tone": {"kind": "halfLambert", "light": [1, 0, 1]}, "faces": "#;
    let surface_tone = read(tone.to_string() + r#"{"kind": "surface"}}"#, "surface");
    let graph_tone = read(tone.to_string() + r#"{"kind": "graph", "tiled": true}}"#, "landscape");
    let mut round_trips = true;
    for s in [&graph_iso, &graph_tone] {
        if source(&s.json().canonical(), "landscape").as_ref() != Ok(s) {
            round_trips = false;
        }
    }
    for s in [&surface_iso, &surface_tone] {
        if source(&s.json().canonical(), "surface").as_ref() != Ok(s) {
            round_trips = false;
        }
    }
    let mut failures = Vec::new();
    if !(graph_iso.light() == Some(Vec3::new(0.0, 0.0, 2.0))
        && surface_tone.light() == Some(Vec3::new(1.0, 0.0, 1.0))
        && round_trips
        && graph_iso.json().canonical().contains(r#""light":[0.0,0.0,2.0]"#)
        && !surface_tone.json().canonical().contains(r#""step""#))
    {
        failures.push(format!(
            "an isophote keeps its light as written, a curved face's hatching writes no step, and both round-trip: \
             lights {:?} and {:?}, round trips {round_trips}, {} and {}",
            graph_iso.light(),
            surface_tone.light(),
            graph_iso.json().canonical(),
            surface_tone.json().canonical()
        ));
    }
    let plain = LayerSource::Contour {
        field: ContourField::U,
        levels: vec![1.0],
        restriction: Restriction::All,
        faces: FaceSelector::Graph { tiled: false },
        light: None,
    };
    if plain.json().canonical().contains("light") {
        failures.push(format!("a contour of anything else writes no light: {}", plain.json().canonical()));
    }
    let graph = r#"{"kind": "graph"}}"#;
    let iso_refusals = [
        (iso.replace(r#""light": [0, 0, 2], "#, "") + graph, "landscape", "light"),
        (iso.replace("[0, 0, 2]", "[0, 0, 0]") + graph, "landscape", "not all zero"),
        (iso.replace("isophote", "u") + graph, "landscape", "only an isophote has one"),
        (iso.to_string() + r#"{"kind": "plateaus"}}"#, "landscape", "constant on a flat face"),
        (iso.to_string() + r#"{"kind": "walls", "edges": [0]}}"#, "landscape", "constant on a flat face"),
        (iso.to_string() + r#"{"kind": "risers"}}"#, "landscape", "constant on a flat face"),
    ];
    let accepted: Vec<String> = iso_refusals
        .iter()
        .filter(|(text, geometry, words)| !refused(text, geometry, words))
        .map(|(text, geometry, words)| {
            format!("{text} on a {geometry}, wanted '{words}': {:?}", source(text, geometry))
        })
        .collect();
    if !accepted.is_empty() {
        failures.push(format!(
            "an isophote with no light, a light with no direction, a light on another field, and isophotes on a flat \
             face are refused: not as asked {accepted:#?}"
        ));
    }
    let tone_refusals = [
        (tone.replace(r#""angle": 0"#, r#""angle": 0.5"#) + graph, "landscape", "along a coordinate"),
        (tone.replace(r#""widthSteps": 4"#, r#""widthSteps": 17"#) + r#"{"kind": "surface"}}"#, "surface", "17 widths"),
        (
            r#"{"kind": "hatching", "angle": 0, "spacing": 0.1, "widthSteps": 1, "tone": {"kind": "turing", "scale": 1, "seed": 1}, "faces": {"kind": "graph"}}"#
                .to_string(),
            "landscape",
            "Turing",
        ),
        (
            tone.replace(r#""angle": 0"#, r#""angle": 0, "step": 0.1"#) + r#"{"kind": "surface"}}"#,
            "surface",
            "a step on a curved face",
        ),
        (tone.to_string() + r#"{"kind": "solid"}}"#, "surface", "one face"),
    ];
    let accepted: Vec<String> = tone_refusals
        .iter()
        .filter(|(text, geometry, words)| !refused(text, geometry, words))
        .map(|(text, geometry, words)| {
            format!("{text} on a {geometry}, wanted '{words}': {:?}", source(text, geometry))
        })
        .collect();
    if !accepted.is_empty() {
        failures.push(format!(
            "hatching on a curved face at a slant, in too many widths, under Turing's tone, with a step, or on \
             another face is refused: not as asked {accepted:#?}"
        ));
    }
    // Read off the request before anything is built, as the window asks
    // before an edit is taken: the outward side.
    let lit = LayerSource::Contour {
        field: ContourField::Isophote,
        levels: vec![0.5],
        restriction: Restriction::All,
        faces: FaceSelector::Surface,
        light: Some(Vec3::new(0.0, 0.0, 1.0)),
    };
    let tone_source = LayerSource::Hatching {
        hatching: Hatching::new(0.0, 0.1, None, Tone::HalfLambert { light: Vec3::new(0.0, 0.0, 1.0) }, 1),
        faces: FaceSelector::Surface,
    };
    let unsupported = |request: &SurfaceRequest, s: &LayerSource| SurfaceManifest::unsupported(s, request);
    let says = |why: Option<String>, words: &str| why.is_some_and(|w| w.contains(words));
    let bounds = [
        relief(0.3, None).bounds_solid(),
        relief(1.0 / 3.0, None).bounds_solid(),
        relief(0.4, None).bounds_solid(),
        relief(0.4, Some(1.0)).bounds_solid(),
        SurfaceRequest::new(Shape::Torus { major: 1.0, minor: 1.0 }).bounds_solid(),
    ];
    if !(bounds == [Some(true), Some(false), Some(false), Some(true), Some(false)]
        && unsupported(&relief(0.3, None), &lit).is_none()
        && unsupported(&relief(0.3, None), &tone_source).is_none()
        && says(unsupported(&relief(0.4, None), &lit), "bounds no solid")
        && says(unsupported(&relief(0.4, None), &tone_source), "bounds no solid")
        && unsupported(&relief(0.4, None), &LayerSource::CoordinateCurves { u: 4, v: 4 }).is_none())
    {
        failures.push(format!(
            "a relief bounds a solid while |A| times the height it displaces stays under the tube's reach, and \
             isophotes and half-Lambert tone are refused on one that does not: bounds {bounds:?}; at 0.3 {:?} and \
             {:?}; at 0.4 {:?} and {:?}",
            unsupported(&relief(0.3, None), &lit),
            unsupported(&relief(0.3, None), &tone_source),
            unsupported(&relief(0.4, None), &lit),
            unsupported(&relief(0.4, None), &tone_source)
        ));
    }
    let unlit = LayerSource::Contour {
        field: ContourField::Isophote,
        levels: vec![0.5],
        restriction: Restriction::All,
        faces: FaceSelector::Surface,
        light: None,
    };
    let torus = SurfaceRequest::new(Shape::Torus { major: 2.0, minor: 1.0 });
    if !says(unsupported(&torus, &unlit), "no light") {
        failures.push(format!(
            "an isophote with no light is refused before anything is built: {:?}",
            unsupported(&torus, &unlit)
        ));
    }
    let negative = r#"{"name": "x", "role": "scaffold", "width": -1, "heightPolicy": "surface", "source": {"kind": "contour", "field": "u", "levels": [0.5], "restriction": {"kind": "all"}, "faces": {"kind": "graph"}}}"#;
    if LayerSpec::from_json(&json::parse(negative).unwrap()).is_ok() {
        failures.push("a layer of negative width is refused".to_string());
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// "isophotes and tone: the manifest", the parts that need a parametric
/// surface built (`SurfaceDrawing.build`) or `SurfaceRequest.unsupported`
/// (KurvenDynamics' SurfaceDrawing.swift), both Phase 4's: a relief at
/// amplitudes 0.3, 1/3 and 0.4, with no floor and over 2.5, built at grid
/// 64, has an outward side exactly where `bounds_solid` says (six of six);
/// hatching on the torus (2, 1) along v every 1e-6 is refused as taking
/// "more than the" vertices a layer holds; and an isophote with no light,
/// built in code into a torus drawing's `restyled`, is refused saying "no
/// light", not drawn as the height.
#[test]
fn the_manifest_against_the_surface_built() {
    let mut failures: Vec<String> = Vec::new();
    let relief = |amplitude: f64, floor: Option<f64>| {
        SurfaceRequest::new(Shape::Relief {
            modulus: 0.64,
            major: 2.0,
            minor: 1.0,
            amplitude,
            repeats: (1, 1),
            grid: 120,
            turn: 0.0,
            floor,
            soft: false,
        })
    };
    let (mut agree, mut disagree) = (0, Vec::new());
    for amplitude in [0.3, 1.0 / 3.0, 0.4] {
        for floor in [None, Some(2.5)] {
            let mut request = relief(amplitude, floor);
            request.grid = 64;
            let built =
                SurfaceDrawing::build(&request, None, false).expect("the relief builds").surface.outward.is_some();
            if Some(built) == request.bounds_solid() {
                agree += 1;
            } else {
                disagree.push(format!("{amplitude}, {}", floor.unwrap_or(0.0)));
            }
        }
    }
    if !(disagree.is_empty() && agree == 6) {
        failures
            .push(format!("the relief built bounds a solid exactly where its request says: disagree at {disagree:?}"));
    }
    let too_fine = LayerSource::Hatching {
        hatching: Hatching::new(core::f64::consts::FRAC_PI_2, 1e-6, None, Tone::Uniform, 1),
        faces: FaceSelector::Surface,
    };
    let unlit = LayerSource::Contour {
        field: ContourField::Isophote,
        levels: vec![0.5],
        restriction: Restriction::All,
        faces: FaceSelector::Surface,
        light: None,
    };
    let torus = SurfaceRequest::new(Shape::Torus { major: 2.0, minor: 1.0 });
    if !(SurfaceRequestExt::unsupported(&torus, &too_fine).is_some_and(|w| w.contains("more than the"))
        && SurfaceRequestExt::unsupported(&torus, &unlit).is_some_and(|w| w.contains("no light")))
    {
        failures.push(
            "hatching past a layer's vertices, and an isophote with no light, are refused before anything is built"
                .to_string(),
        );
    }
    let small = SurfaceRequest { grid: 64, ..torus };
    let refused = SurfaceDrawing::build(&small, None, false)
        .expect("the torus builds")
        .restyled(&[LayerSpec::new("x", LayerRole::Scaffold, unlit, 0.3, HeightPolicy::Surface)])
        .err()
        .is_some_and(|e| e.to_string().contains("no light"));
    if !refused {
        failures.push("an isophote with no light, built in code, is refused, not drawn as the height".to_string());
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// "isophotes: on the round torus lit along z, n · l = sin v, and the
/// isophote at c is v = asin c or π − asin c": the torus (2, 1) at grid 512,
/// u and v over [0, 2π) with du × dv outward, its normal the closed form's to
/// 1e-15 at 64 × 64 points; under lights (0, 0, 1) and (0, 0, 2.5), every
/// isophote vertex at the twelve levels on v = asin c or π − asin c to
/// 1e-12, each level two closed circles a whole turn round; and nothing at
/// c = ±1 or past it.
#[test]
fn on_the_round_torus_lit_along_z_the_isophote_is_v_asin_c() {
    let torus = round_torus();
    let face = torus.face();
    let copy = &face.copies[0];
    let two_pi = 2.0 * core::f64::consts::PI;
    let mut failures: Vec<String> = Vec::new();
    // The convention first: u round the axis from 0, v round the tube
    // from the outer equator, both over [0, 2π), the outward normal the
    // closed form's, so n · z = sin v.
    if !(torus.u.range.lo == 0.0
        && torus.v.range.lo == 0.0
        && (torus.u.range.length() - two_pi).abs() < 1e-15
        && (torus.v.range.length() - two_pi).abs() < 1e-15
        && torus.outward == Some(1.0))
    {
        failures.push("u and v run over [0, 2π), du × dv outward".to_string());
    }
    let mut off: f64 = 0.0;
    for i in 0..64 {
        for j in 0..64 {
            let p = P2::new(two_pi * (i as f64 + 0.37) / 64.0, two_pi * (j as f64 + 0.61) / 64.0);
            let n = face.outward_normal(p, copy).expect("a normal");
            off = swift::max(
                swift::max(off, (n - torus_normal(p.x(), p.y())).length()),
                (n.dot(Vec3::new(0.0, 0.0, 1.0)) - real::sin(p.y())).abs(),
            );
        }
    }
    if !(off <= 1e-15) {
        failures.push(format!("the outward normal is (cos v cos u, cos v sin u, sin v): off by {off:.1e}"));
    }
    // Lit along z, and along z at another length: the field normalizes
    // the light.
    let levels = [-0.999999, -0.9999, -0.99, -0.9, -0.5, -0.1, 0.0, 0.3, 0.7, 0.99, 0.9999, 0.999999];
    for light in [Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 2.5)] {
        let (mut worst, mut closed, mut counts, mut example) = (0.0f64, true, Vec::new(), String::new());
        for (&c, (level, paths)) in levels.iter().zip(face.isophotes(light, &levels, copy)) {
            assert_eq!(level, c);
            counts.push(paths.len());
            let a = real::asin(c);
            let targets = [a, core::f64::consts::PI - a];
            for path in &paths {
                for p in path {
                    let d = swift::min(apart(p.y(), targets[0]), apart(p.y(), targets[1]));
                    if d > worst {
                        worst = d;
                        example = format!("c = {c} at v = {}", p.y());
                    }
                }
                // A whole turn round the axis, ending where it began.
                let (first, last) = (path[0], path[path.len() - 1]);
                if ((last.x() - first.x()).abs() - two_pi).abs() > 1e-12 || (last.y() - first.y()).abs() > 1e-12 {
                    closed = false;
                }
            }
        }
        if !(worst <= 1e-12) {
            failures.push(format!(
                "light {light:?}: every vertex of the isophote at c lies on v = asin c or π − asin c: worst {worst:.1e} rad, {example}"
            ));
        }
        if !(counts.iter().all(|&n| n == 2) && closed) {
            failures.push(format!(
                "light {light:?}: each isophote is the two circles, each a whole turn round the axis: paths per level {counts:?}"
            ));
        }
    }
    // At ±1 n · l touches its extreme on a circle, v = ±π/2; past it,
    // nothing. A level there draws nothing, by rule.
    let extremes = face.isophotes(Vec3::new(0.0, 0.0, 1.0), &[-1.5, -1.0, 1.0, 1.5], copy);
    if !extremes.is_empty() {
        failures.push(format!("at c = ±1 and past it, the isophote draws nothing: {} levels drawn", extremes.len()));
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The round torus (2, 1) as `SurfaceDrawing::build` builds a torus
/// request at grid 512, as Isophotes.swift's suites have it.
fn round_torus() -> ParametricSurface {
    let request = SurfaceRequest { grid: 512, ..SurfaceRequest::new(Shape::Torus { major: 2.0, minor: 1.0 }) };
    SurfaceDrawing::build(&request, None, false).expect("the torus builds").surface
}

/// How far apart two angles are, round the circle.
fn apart(a: f64, b: f64) -> f64 {
    let two_pi = 2.0 * core::f64::consts::PI;
    let d = real::fmod(a - b, two_pi);
    swift::min(d.abs(), two_pi - d.abs())
}

/// The round torus's outward unit normal at (u, v), in closed form.
fn torus_normal(u: f64, v: f64) -> Vec3 {
    Vec3::new(real::cos(v) * real::cos(u), real::cos(v) * real::sin(u), real::sin(v))
}

/// "tone: on the torus, half-Lambert hatching runs along a coordinate, its
/// width the darkness's step at every sample, changing on the isophotes":
/// along u (24 lines, 8 widths) and along v (36 lines, 5 widths), every piece
/// on its line, the pieces covering each line a whole turn, the width step
/// ⌈N (1 − n · l)/2⌉ at every sample (over 10 000), the width changing only
/// on an isophote within 1e-12 (over 100 ends); and the bake, a layer per
/// width step at k/8 of the layer's width with that step's pieces.
#[test]
fn tone_on_the_torus_runs_along_a_coordinate() {
    let torus = round_torus();
    let two_pi = 2.0 * core::f64::consts::PI;
    let mut failures: Vec<String> = Vec::new();
    for (hatching, along_u) in [
        (Hatching::new(0.0, two_pi / 24.0, None, Tone::HalfLambert { light: Vec3::new(1.0, -1.0, 1.5) }, 8), true),
        (
            Hatching::new(
                core::f64::consts::FRAC_PI_2,
                two_pi / 36.0,
                None,
                Tone::HalfLambert { light: Vec3::new(-0.4, -0.6, 0.7) },
                5,
            ),
            false,
        ),
    ] {
        let v = tone_verdict(&torus, &hatching, along_u);
        let what = if along_u { "along u" } else { "along v" };
        if !(v.off_line <= 1e-12 && v.gaps <= 1e-9) {
            failures.push(format!(
                "{what}: every piece lies on its line of the other coordinate, and the pieces cover each line, a whole \
                 turn: {} pieces, off by {:.1e}, cover a turn to {:.1e}",
                v.pieces, v.off_line, v.gaps
            ));
        }
        if !(v.wrong_step == 0 && v.samples > 10_000) {
            failures.push(format!(
                "{what}: at every sample a piece's width step is ⌈{} (1 − n · l)/2⌉: {} of {} samples wrong",
                hatching.width_steps, v.wrong_step, v.samples
            ));
        }
        if !(v.off_isophote == 0.0 && v.changes > 100) {
            failures.push(format!(
                "{what}: the width changes only on an isophote n · l = 1 − 2k/{}, within 1e-12: {} piece ends on one",
                hatching.width_steps, v.changes
            ));
        }
    }
    // The bake: a layer per width step at k/N of the layer's width,
    // each holding that step's pieces.
    let hatching = Hatching::new(0.0, two_pi / 24.0, None, Tone::HalfLambert { light: Vec3::new(1.0, -1.0, 1.5) }, 8);
    let spec = LayerSpec::new(
        "tone",
        LayerRole::Scaffold,
        LayerSource::Hatching { hatching, faces: FaceSelector::Surface },
        1.5,
        HeightPolicy::Surface,
    );
    let scene = Scene::parametric_scene(
        torus.clone(),
        vec![kurven_core::polyline::Layer { spec, paths: PolylineSet::empty() }],
        Camera::oblique(&SurfaceRequest::PROJECTION),
        PreviewMode::Drawing,
        0.0,
    );
    let steps = torus.coordinate_hatching(&hatching);
    let layers = scene.visible_layers(None, None).expect("the torus's tone draws");
    let widths = layers.iter().zip(&steps).all(|((drawn, _), (w, paths))| {
        drawn.spec.width == 1.5 * *w as f64 / 8.0 && drawn.spec.name == format!("tone {w}/8") && drawn.paths == *paths
    });
    if !(layers.len() == steps.len() && steps.len() >= 6 && widths) {
        failures.push(format!(
            "baked as a layer per width step, at k/8 of the layer's width, each with that step's pieces: {:?}",
            layers.iter().map(|(l, _)| format!("{} {}", l.spec.name, l.spec.width)).collect::<Vec<_>>()
        ));
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// What `tone_verdict` measures of a torus's half-Lambert hatching.
struct ToneVerdict {
    pieces: usize,
    off_line: f64,
    wrong_step: usize,
    samples: usize,
    off_isophote: f64,
    changes: usize,
    gaps: f64,
}

/// Isophotes.swift's `toneVerdict`: every piece of the hatching on its line
/// of the other coordinate, its width step the darkness's at every sample
/// of it, and where the width changes, on an isophote.
fn tone_verdict(torus: &ParametricSurface, hatching: &Hatching, along_u: bool) -> ToneVerdict {
    let Tone::HalfLambert { light } = hatching.tone else { panic!("a half-Lambert tone") };
    let two_pi = 2.0 * core::f64::consts::PI;
    let l = light.normalized();
    let n = hatching.width_steps as f64;
    let darkness = |p: Vec2| (1.0 - torus_normal(p.x, p.y).dot(l)) / 2.0;
    let levels = Face::hatch_levels(if along_u { &torus.v } else { &torus.u }, hatching.spacing);
    let mut v =
        ToneVerdict { pieces: 0, off_line: 0.0, wrong_step: 0, samples: 0, off_isophote: 0.0, changes: 0, gaps: 0.0 };
    let mut extent: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
    let mut ends: std::collections::BTreeMap<usize, Vec<f64>> = std::collections::BTreeMap::new();
    let across = |p: Vec2| if along_u { p.y } else { p.x };
    let along = |p: Vec2| if along_u { p.x } else { p.y };
    for (step, set) in torus.coordinate_hatching(hatching) {
        for k in 0..set.count() {
            let coords: Vec<Vec2> = set.coords_of_path(k).expect("coordinates").iter().map(|c| c.v).collect();
            v.pieces += 1;
            // Swift's `min` over the indices: the first of the nearest.
            let mut line = 0;
            for i in 1..levels.len() {
                if apart(across(coords[0]), levels[i]) < apart(across(coords[0]), levels[line]) {
                    line = i;
                }
            }
            for c in &coords {
                v.off_line = swift::max(v.off_line, apart(across(*c), levels[line]));
            }
            *extent.entry(line).or_insert(0.0) += (along(coords[coords.len() - 1]) - along(coords[0])).abs();
            for w in coords.windows(2) {
                for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
                    let d = darkness(w[0] + (w[1] - w[0]) * t);
                    // A sample on a step's boundary is the boundary's.
                    if !((n * d - (n * d).round()).abs() > 1e-9) {
                        continue;
                    }
                    v.samples += 1;
                    if (n * d).ceil() as i64 != step {
                        v.wrong_step += 1;
                    }
                }
            }
            for end in [coords[0], coords[coords.len() - 1]] {
                let nd = n * darkness(end);
                if (nd - nd.round()).abs() <= 1e-12 * n {
                    v.changes += 1;
                } else {
                    ends.entry(line).or_default().push(along(end));
                }
            }
        }
    }
    // Off an isophote, a piece ends only where its line does: a closed
    // line's two ends, a turn apart.
    for e in ends.values() {
        if e.len() > 2 || (e.len() == 2 && apart(e[0], e[1]) > 1e-12) {
            v.off_isophote = f64::INFINITY;
        }
    }
    v.gaps = (0..levels.len())
        .map(|i| (extent.get(&i).copied().unwrap_or(0.0) - two_pi).abs())
        .fold(f64::NEG_INFINITY, swift::max);
    if levels.is_empty() {
        v.gaps = f64::INFINITY;
    }
    v
}

/// "isophotes and tone: seen exactly where the torus's own cast and the
/// grid landscape's cast say", its grid landscape: the cubic
/// (expr/landscape_cubic.kurven, on its grid, its own normals the grid's)
/// restyled with isophotes at -0.6, -0.3, 0, 0.3, 0.6 and 0.9 and four
/// widths of half-Lambert tone every 0.04 on its graph, lit from
/// (-0.4, -0.6, 0.7), alone in a composite under its own camera
/// (`gate::composite_verdict`, no other part's cast): at every vertex of
/// those layers facing the eye, drawn exactly where its cast
/// (Stress.swift's `exactlyHidden` against the drawn triangles) sees it,
/// over 5 000 compared.
#[test]
fn isophotes_and_tone_on_the_grid_landscape_are_seen_exactly_where_its_cast_says() {
    let light = Vec3::new(-0.4, -0.6, 0.7);
    let read = gate::cubic_bundle();
    let mut m = read.manifest.clone();
    m.layers = vec![
        LayerSpec::new(
            "isophotes",
            LayerRole::Scaffold,
            LayerSource::Contour {
                field: ContourField::Isophote,
                levels: vec![-0.6, -0.3, 0.0, 0.3, 0.6, 0.9],
                restriction: Restriction::All,
                faces: FaceSelector::Graph { tiled: false },
                light: Some(light),
            },
            0.3,
            HeightPolicy::Surface,
        ),
        LayerSpec::new(
            "tone",
            LayerRole::Scaffold,
            LayerSource::Hatching {
                hatching: Hatching::new(0.0, 0.04, None, Tone::HalfLambert { light }, 4),
                faces: FaceSelector::Graph { tiled: false },
            },
            1.0,
            HeightPolicy::Surface,
        ),
    ];
    let bundle = read.restyled(&m);
    let landscape = Scene::from_bundle(&bundle, &bundle.manifest.presets[0]);
    let camera = landscape.camera;
    let start = std::time::Instant::now();
    let composite = kurven_core::composite::Composite::new(
        vec![("cubic".into(), landscape, kurven_core::composite::Placement::IDENTITY)],
        camera,
    )
    .unwrap();
    let clear: gate::Cast = std::sync::Arc::new(|_, _| Some(true));
    let g = gate::composite_verdict(&composite, &[clear]).unwrap();
    eprintln!("grid: {} compared, {} wrong, {:?}", g.own_compared, g.own_wrong, start.elapsed());
    assert!(
        g.own_wrong == 0 && g.own_compared > 5000,
        "the grid landscape's isophotes and tone, at every vertex, are drawn exactly where its cast sees them: {} \
         compared, {} wrong{}",
        g.own_compared,
        g.own_wrong,
        g.own_example
    );
}

/// "isophotes and tone: seen exactly where the torus's own cast and the
/// grid landscape's cast say", its torus: the round torus (2, 1) at grid
/// 512 (`SurfaceDrawing.build`, Phase 4) restyled with isophotes at -0.75
/// to 0.75 by 0.25 and four widths of half-Lambert tone every 2π/48 along
/// u, lit from (-0.4, -0.6, 0.7), alone in a composite under
/// `SurfaceRequest.projection`'s oblique camera, every other cast clear: at
/// every vertex, drawn exactly where its own cast (`gate::torus_own_cast`)
/// sees it, over 20 000 compared and over 5 000 of them seen.
#[test]
fn isophotes_and_tone_on_the_torus_are_seen_exactly_where_its_own_cast_says() {
    let light = Vec3::new(-0.4, -0.6, 0.7);
    let isophotes = LayerSpec::new(
        "isophotes",
        LayerRole::Scaffold,
        LayerSource::Contour {
            field: ContourField::Isophote,
            levels: vec![-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75],
            restriction: Restriction::All,
            faces: FaceSelector::Surface,
            light: Some(light),
        },
        0.3,
        HeightPolicy::Surface,
    );
    let tone = LayerSpec::new(
        "tone",
        LayerRole::Scaffold,
        LayerSource::Hatching {
            hatching: Hatching::new(0.0, 2.0 * core::f64::consts::PI / 48.0, None, Tone::HalfLambert { light }, 4),
            faces: FaceSelector::Surface,
        },
        1.0,
        HeightPolicy::Surface,
    );
    let request = SurfaceRequest { grid: 512, ..SurfaceRequest::new(Shape::Torus { major: 2.0, minor: 1.0 }) };
    let drawing = SurfaceDrawing::build(&request, None, false)
        .expect("the torus builds")
        .restyled(&[isophotes, tone])
        .expect("its isophotes and tone draw");
    let camera = Camera::oblique(&SurfaceRequest::PROJECTION);
    let composite = kurven_core::composite::Composite::new(
        vec![("torus".into(), drawing.scene(camera), kurven_core::composite::Placement::IDENTITY)],
        camera,
    )
    .unwrap();
    let clear: gate::Cast = std::sync::Arc::new(|_, _| Some(true));
    let t = gate::composite_verdict(&composite, &[clear]).unwrap();
    assert!(
        t.own_wrong == 0 && t.own_compared > 20_000 && t.torus_seen > 5000,
        "the torus's isophotes and tone, at every vertex, are drawn exactly where its own cast sees them: {} \
         compared, {} seen, {} wrong{}",
        t.own_compared,
        t.torus_seen,
        t.own_wrong,
        t.own_example
    );
}
