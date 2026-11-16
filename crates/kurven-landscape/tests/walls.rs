//! Facing.swift's suites on the strokes Hatch.swift draws: the walls' hatch
//! and edges tested by the wall, on a notch and at a tile boundary, and
//! against a walk up the line of sight; the risers' hatching and bottom
//! edges where they stand, which layers hatch them, and against the walk;
//! no stroke drawn through a riser; the plateaus' hatching tested by the
//! plateau; and a vertical stroke its two ends. Swift's tolerances. (Its
//! facing, fold and navigation suites are in `visibility.rs`.)
//!
//! The walks are brute force, as in Swift; where Swift spreads them over its
//! cores with `parallelMap`, these spread them over threads with
//! `std::thread::scope` (`common::par_map`), each camera's verdicts
//! independent of the others'.

// Swift's `guard x > 0 else` refuses a NaN, and so does `if !(x > 0.0)`.
#![allow(
    clippy::neg_cmp_op_on_partial_ord,
    clippy::manual_range_contains,
    clippy::type_complexity,
    clippy::too_many_arguments
)]

mod common;
mod support;

use std::collections::{HashMap, HashSet};

use common::{build, par_map, random_heightfield, stress_cameras, test_camera};
use kurven_core::bundle::KurvenBundle;
use kurven_core::hatch::HatchContext;
use kurven_core::hatching::{FaceKind, FaceSelector};
use kurven_core::hidden_line::{HeightfieldVisibility, HiddenLine};
use kurven_core::manifest::{
    Affine2, BoundaryPerimeter, Domain, Interval, LayerRole, LayerSource, Manifest, PerimeterEdge, RealBand, Region,
    Truncation, Walls,
};
use kurven_core::mesh::Mesh;
use kurven_core::polyline::{Layer, PolylineSet};
use kurven_core::scene::{Heightfield, Scene};
use kurven_core::simd::{Vec2, Vec3};
use kurven_core::spaces::{P2, P3, WorldSpace};
use kurven_core::surface::{Magnitude, Surface};
use kurven_landscape::{Catalog, LandscapeRequest};
use kurven_math::swift;

fn domain(r0: f64, r1: f64, i0: f64, i1: f64) -> Domain {
    Domain::new(Interval::new(r0, r1), Interval::new(i0, i1))
}

fn request(expression: &str, domain: Domain, resolution: i64, truncation: Truncation) -> LandscapeRequest {
    let mut r = LandscapeRequest::new(expression, domain, resolution);
    r.truncation = Some(truncation);
    r
}

fn banded(bands: &[(f64, f64)], beyond: f64) -> Truncation {
    Truncation::RealBands { bands: bands.iter().map(|&(below, height)| RealBand { below, height }).collect(), beyond }
}

fn affine(a: f64, b: f64, tx: f64, c: f64, d: f64, ty: f64) -> Affine2 {
    Affine2 { a, b, tx, c, d, ty }
}

/// A perimeter through the corners, in order, nine samples an edge.
fn perimeter(corners: &[(f64, f64)]) -> BoundaryPerimeter {
    BoundaryPerimeter::new(
        (0..corners.len())
            .map(|k| {
                let (a, b) = (corners[k], corners[(k + 1) % corners.len()]);
                PerimeterEdge { start: P2::new(a.0, a.1), end: P2::new(b.0, b.1), density: 9 }
            })
            .collect(),
    )
}

fn all_edges(p: &BoundaryPerimeter) -> Vec<i64> {
    (0..p.edges().len() as i64).collect()
}

fn paths(set: &PolylineSet<WorldSpace>) -> Vec<Vec<P3<WorldSpace>>> {
    set.paths().map(|p| p.to_vec()).collect()
}

fn length(p: &[P3<WorldSpace>]) -> f64 {
    p.windows(2).map(|w| (w[1].v - w[0].v).length()).sum()
}

fn set_length(s: &PolylineSet<WorldSpace>) -> f64 {
    s.paths().map(length).sum()
}

/// A double's bits, -0 read as 0, as Swift's `Hashable` has it.
fn bits(x: f64) -> u64 {
    if x == 0.0 { 0 } else { x.to_bits() }
}

fn key2(x: f64, y: f64) -> (u64, u64) {
    (bits(x), bits(y))
}

fn key3(v: Vec3) -> (u64, u64, u64) {
    (bits(v.x), bits(v.y), bits(v.z))
}

/// Each kept run, by the stroke it lies on: its bottom and top.
fn runs(set: &PolylineSet<WorldSpace>) -> HashMap<(u64, u64), Vec<(f64, f64)>> {
    let mut out: HashMap<(u64, u64), Vec<(f64, f64)>> = HashMap::new();
    for p in set.paths() {
        out.entry(key2(p[0].x(), p[0].y())).or_default().push((p[0].z(), p[p.len() - 1].z()));
    }
    out
}

/// Whether a stroke from `base` to `top` hidden below `below` is kept as one
/// run from there to its top, or not at all where the whole is hidden.
fn agrees(kept: &[(f64, f64)], below: f64, top: f64) -> bool {
    let runs: Vec<&(f64, f64)> = kept.iter().filter(|r| r.0 < top - 1e-3).collect();
    if below < top - 1e-3 {
        runs.len() == 1 && (runs[0].0 - below).abs() < 1e-3 && (runs[0].1 - top).abs() < 1e-3
    } else {
        runs.is_empty()
    }
}

fn rgamma() -> KurvenBundle {
    build(&LandscapeRequest::from_preset(Catalog::native().preset("rgamma").unwrap(), 240), true)
}

// MARK: - the hatching and edges of a cut face are tested by the wall

/// "walls: the hatching and edges of a cut face are tested by the wall".
/// Swift's last check bakes the scene at 400 and asks that the wall hatch
/// drawn is the tested one, stroke for stroke; a landscape's bake strokes
/// are `Scene::visible_layers` through `HiddenLine::pass`, which is what is
/// held to the tested curves here.
#[test]
fn the_hatching_and_edges_of_a_cut_face_are_tested_by_the_wall() {
    let bundle = rgamma();
    let scene = Scene::from_bundle(&bundle, &bundle.manifest.presets[0]);
    let h = scene.heightfield().unwrap();
    let view = scene.camera.view;
    let sight = view.projection_direction();
    let d = h.surface.domain();
    let (lo, hi) = (
        Vec2::new(swift::min(d.real.lo, d.real.hi), swift::min(d.imag.lo, d.imag.hi)),
        Vec2::new(swift::max(d.real.lo, d.real.hi), swift::max(d.imag.lo, d.imag.hi)),
    );
    // Which walls face the eye, by their outward normals: left, right,
    // front, back.
    let faces: Vec<bool> =
        [Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0)]
            .iter()
            .map(|n| -n.dot(sight) > 0.0)
            .collect();
    assert!(
        faces.contains(&true) && faces.contains(&false),
        "the oblique camera sees some walls and not others: left {} right {} front {} back {}",
        faces[0],
        faces[1],
        faces[2],
        faces[3]
    );
    let walls_of = |p: P3<WorldSpace>| -> Vec<usize> {
        let slack = 1e-6;
        let mut on = Vec::new();
        if (p.x() - lo.x).abs() <= slack {
            on.push(0);
        }
        if (p.x() - hi.x).abs() <= slack {
            on.push(1);
        }
        if (p.y() - lo.y).abs() <= slack {
            on.push(2);
        }
        if (p.y() - hi.y).abs() <= slack {
            on.push(3);
        }
        on
    };
    // The hatch lies in the face: whole strokes on facing walls, nothing
    // on the others.
    let hatch = scene.layers().iter().find(|l| l.spec.name == "wall_hatch").unwrap();
    let kept_hatch = h.visible_wall_curves(&hatch.paths, &view, scene.depth_tolerance, 0.0);
    let mut wanted = 0.0;
    for p in hatch.paths.paths() {
        if p.iter().all(|&v| walls_of(v).iter().any(|&w| faces[w])) {
            wanted += length(p);
        }
    }
    assert!(
        (set_length(&kept_hatch) - wanted).abs() < 1e-9 * swift::max(wanted, 1.0),
        "wall_hatch: what is kept is every stroke on a wall that faces the eye: {:.3} of {:.3} units, {} of {} paths",
        set_length(&kept_hatch),
        set_length(&hatch.paths),
        kept_hatch.count(),
        hatch.paths.count()
    );
    let strays: Vec<&P3<WorldSpace>> =
        kept_hatch.vertices.iter().filter(|&&v| !walls_of(v).iter().any(|&w| faces[w])).collect();
    let note: Vec<String> = strays
        .iter()
        .take(3)
        .map(|v| {
            format!(
                "({:.3}, {:.3}, {:.4}) f {:.4} walls {:?}",
                v.x(),
                v.y(),
                v.z(),
                h.surface_height(P2::new(v.x(), v.y())),
                walls_of(**v)
            )
        })
        .collect();
    if let Some(&&stray) = strays.first() {
        let visibility = HeightfieldVisibility::new(h.clone(), &view, scene.depth_tolerance);
        eprintln!("    {}", visibility.explain(stray).replace('\n', "\n    "));
    }
    assert!(
        strays.is_empty(),
        "wall_hatch: and nothing on a wall that faces away: {} stray vertices {}",
        strays.len(),
        note.join("; ")
    );

    // The edges: a top edge is the surface's edge and shows over any wall
    // where the surface does; a bottom edge shows only on a facing wall.
    let wall_edges = scene.layers().iter().find(|l| l.spec.name == "wall_edges").unwrap();
    let kept_edges = h.visible_wall_curves(&wall_edges.paths, &view, scene.depth_tolerance, 0.0);
    let is_top_edge = |p: &[P3<WorldSpace>]| -> bool {
        p.iter().all(|v| (v.z() - h.surface_height(P2::new(v.x(), v.y()))).abs() < 1e-9 * swift::max(1.0, v.z().abs()))
            && !p.iter().all(|v| v.z() == 0.0)
    };
    let wall_of = |p: &[P3<WorldSpace>]| -> Option<usize> {
        walls_of(p[0]).into_iter().find(|&w| p.iter().all(|&v| walls_of(v).contains(&w)))
    };
    let (mut top_edge_wanted, mut top_edge_got) = ([0.0f64; 4], [0.0f64; 4]);
    let mut bottom_edge_hidden = true;
    for p in wall_edges.paths.paths() {
        if let Some(wall) = wall_of(p)
            && is_top_edge(p)
        {
            top_edge_wanted[wall] += length(p);
        }
    }
    for p in kept_edges.paths() {
        let Some(wall) = wall_of(p) else { continue };
        if is_top_edge(p) {
            top_edge_got[wall] += length(p);
        } else if p.iter().all(|v| v.z() == 0.0) && !faces[wall] {
            bottom_edge_hidden = false;
        }
    }
    assert!(
        top_edge_got.iter().zip(&top_edge_wanted).all(|(g, w)| *g >= 0.95 * w),
        "wall_edges: every wall's top edge is kept, the far edge included: kept {:?}",
        top_edge_got.iter().zip(&top_edge_wanted).map(|(g, w)| format!("{g:.1}/{w:.1}")).collect::<Vec<_>>()
    );
    assert!(bottom_edge_hidden, "wall_edges: and no bottom edge of a wall that faces away");

    // Where the truncation edge of a plateau ends at the front edge -- the notch of a
    // pit -- the top edge has a vertex on it, so the two meet.
    let truncation_edge = &scene.layers().iter().find(|l| l.spec.name == "truncation_edge").unwrap().paths;
    let mut truncation_edge_ends: Vec<P3<WorldSpace>> = Vec::new();
    for p in truncation_edge.paths() {
        for end in [p[0], p[p.len() - 1]] {
            if (end.y() - lo.y).abs() <= 1e-9 {
                truncation_edge_ends.push(end);
            }
        }
    }
    let top_edge_vertices: Vec<&P3<WorldSpace>> =
        wall_edges.paths.vertices.iter().filter(|v| (v.y() - lo.y).abs() <= 1e-9).collect();
    let mut missed = 0.0f64;
    for end in &truncation_edge_ends {
        let nearest = top_edge_vertices
            .iter()
            .map(|v| (v.v - end.v).length())
            .fold(f64::INFINITY, |m, x| if x < m { x } else { m });
        missed = swift::max(missed, nearest);
    }
    assert!(
        truncation_edge_ends.len() >= 2 && missed < 1e-9,
        "the top edge meets the truncation edge where the truncation edge ends, at every notch: {} truncation edge \
         ends on the front edge, worst gap {missed:.1e}",
        truncation_edge_ends.len()
    );
    // The bake draws the tested curves as it is: the hatch on the edge-on
    // right wall is whole, not spotty.
    let hatch_index = scene.layers().iter().position(|l| l.spec.name == "wall_hatch").unwrap();
    let baked = scene.visible_layers(None, None).unwrap();
    let drawn = HiddenLine::pass(&baked[hatch_index].1);
    let visible = h.visible_wall_curves(&scene.layers()[hatch_index].paths, &view, scene.depth_tolerance, 0.0);
    assert!(
        drawn.count() == visible.count() && drawn.vertices.len() == visible.vertices.len(),
        "the bake draws the wall hatch exactly as tested, stroke for stroke: {} strokes drawn, {} tested",
        drawn.count(),
        visible.count()
    );
}

// MARK: - a wall is its face, wherever it stands

/// Whether `v` lies on perimeter edge `k`, between its ends.
fn on_edge(walls: &BoundaryPerimeter, k: usize, v: Vec2) -> bool {
    let e = &walls.edges()[k];
    let along = e.end.v - e.start.v;
    let r = v - e.start.v;
    let t = r.dot(along) / along.dot(along);
    t > -1e-9 && t < 1.0 + 1e-9 && (r - t * along).length() < 1e-9
}

/// Each edge's outward normal in the plane, from the perimeter's turn.
fn outward_normals(walls: &BoundaryPerimeter) -> Vec<Vec2> {
    let turn = if walls.signed_area() < 0.0 { -1.0 } else { 1.0 };
    walls
        .edges()
        .iter()
        .map(|e| {
            let d = (e.end.v - e.start.v).normalized();
            turn * Vec2::new(d.y, -d.x)
        })
        .collect()
}

/// "walls: a stroke on a wall is tested by the wall it stands on, on a
/// notch and at a tile boundary".
#[test]
fn a_stroke_on_a_wall_is_tested_by_the_wall_it_stands_on() {
    let fixture = support::hatch_fixture();
    let surface = Surface::new(fixture.grid.clone(), None, Truncation::Uniform(2.0), false);
    let boxed = perimeter(&[(-2.0, -1.0), (2.5, -1.0), (2.5, 1.5), (-2.0, 1.5)]);
    let notched = perimeter(&[
        (-2.0, -1.0),
        (0.5, -1.0),
        (0.5, 0.0),
        (1.5, 0.0),
        (1.5, -1.0),
        (2.5, -1.0),
        (2.5, 1.5),
        (-2.0, 1.5),
    ]);
    let clockwise = perimeter(&[
        (-2.0, 1.5),
        (2.5, 1.5),
        (2.5, -1.0),
        (1.5, -1.0),
        (1.5, 0.0),
        (0.5, 0.0),
        (0.5, -1.0),
        (-2.0, -1.0),
    ]);
    let layouts: Vec<(&str, BoundaryPerimeter, Region, Vec<Affine2>)> = vec![
        ("a notch cut from the front", notched.clone(), Region::Inside(notched), vec![Affine2::IDENTITY]),
        (
            "the notched footprint walked clockwise",
            clockwise.clone(),
            Region::Inside(clockwise),
            vec![Affine2::IDENTITY],
        ),
        // The first tile's walls, and a second tile beside it: the box
        // edge they meet on is the first tile's right wall alone.
        (
            "a second tile beside the first",
            boxed,
            Region::Full,
            vec![Affine2::IDENTITY, affine(1.0, 0.0, 4.5, 0.0, 1.0, 0.0)],
        ),
    ];
    for (name, walls, region, tiles) in layouts {
        let h = Heightfield::new(
            surface.clone(),
            Mesh::empty(),
            tiles.clone(),
            region.clone(),
            1,
            None,
            Walls::Perimeter { perimeter: walls.clone(), base: 0.0 },
        );
        let context =
            HatchContext { perimeter: Some(walls.clone()), region: region.clone(), tiles: tiles.clone(), base: 0.0 };
        // The walls' strokes, from the floor: where the second tile's top
        // stands over the first's at the box edge they meet on, a riser
        // stands over the wall there, and its hatching, a hair onto the
        // lower side, rises from its bottom (the risers' suites).
        let both = surface
            .hatch(&LayerSource::wall_hatching(all_edges(&walls), 0.1, false, 0.0, 0.0), &context, None)
            .unwrap();
        let hatch = PolylineSet::from_paths(&paths(&both).into_iter().filter(|p| p[0].z() == 0.0).collect::<Vec<_>>());
        let normals = outward_normals(&walls);
        let count = walls.edges().len();
        for (seen_from, view) in
            [("from the left", test_camera(-55.0, -60.0, 0.0)), ("from the right", test_camera(-55.0, 150.0, 0.0))]
        {
            let sight = view.projection_direction();
            let facing = |k: usize| -> bool { -Vec3::new(normals[k].x, normals[k].y, 0.0).dot(sight) > 0.0 };
            let kept = h.on_grid().visible_wall_curves(&hatch, &view, f64::INFINITY, 0.0);
            let (mut wanted, mut facing_away) = (0, 0);
            for p in hatch.paths() {
                let on: Vec<usize> = (0..count).filter(|&k| p.iter().all(|v| on_edge(&walls, k, v.xy().v))).collect();
                if on.iter().any(|&k| facing(k)) {
                    wanted += 1;
                } else {
                    facing_away += 1;
                }
            }
            let strays =
                kept.vertices.iter().filter(|v| !(0..count).any(|k| on_edge(&walls, k, v.xy().v) && facing(k))).count();
            assert!(
                kept.count() == wanted && strays == 0 && facing_away > 0,
                "{name}, seen {seen_from}: every stroke on a wall that faces the eye, and none on one that faces away: \
                 {} of {} strokes kept, {wanted} wanted, {facing_away} facing away, {strays} stray vertices",
                kept.count(),
                hatch.count()
            );
        }
    }
}

/// The highest a grid's samples stand and the steepest it rises between two.
fn highest_and_steepest(h: &Heightfield) -> (f64, f64) {
    let g = &*h.surface.height;
    let dom = h.surface.domain();
    let (dx, dy) = (dom.real.length() / (g.width - 1) as f64, dom.imag.length() / (g.height - 1) as f64);
    let (mut highest, mut steepest) = (0.0f64, 0.0f64);
    for j in 0..g.height {
        for i in 0..g.width {
            let z = f64::from(g.at(i, j));
            highest = swift::max(highest, z);
            if i + 1 < g.width {
                steepest = swift::max(steepest, (f64::from(g.at(i + 1, j)) - z).abs() / dx);
            }
            if j + 1 < g.height {
                steepest = swift::max(steepest, (f64::from(g.at(i, j + 1)) - z).abs() / dy);
            }
        }
    }
    (highest, steepest)
}

/// 257 vertices along each two-vertex stroke, from its first to its last.
fn densified(strokes: &[Vec<P3<WorldSpace>>]) -> PolylineSet<WorldSpace> {
    PolylineSet::from_paths(
        &strokes
            .iter()
            .map(|p| {
                let (a, b) = (p[0], p[p.len() - 1]);
                (0..=256)
                    .map(|j| if j == 256 { b } else { P3::from_vec(a.v + (b.v - a.v) * (j as f64 / 256.0)) })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
    )
}

// Where a wall's line of sight meets terrain -- the next copy, from a
// wall on a tile boundary, or the far side of a notch -- the terrain hides
// the wall's hatch from its bottom up. What it hides of the stroke at a
// point of the wall is everything below the greatest of h(q(t)) - t d_z
// along the line of sight, q(t) its footprint and h the drawn height
// there: the highest the terrain stands over the line, carried back down
// it. The walk reads that a five-hundredth of a unit at a time, then a
// hundred-thousandth around every sample within reach of the greatest;
// h is linear over each triangle, so the reach is its steepest slope
// over the coarse step. Each stroke on a wall that faces the eye, tested
// from its two ends and at 256 vertices, keeps one run from there to its
// top, under every camera of the stress orbit. A run that starts within a
// thousandth of the top is a stub where the top edge, which is on the
// surface, is forgiven the depth tolerance's reach, and counts as nothing
// kept.
//
// Across a notch, one disagreement is allowed, and counted: the
// traversal keeps more than the walk where the terrain stands highest
// over the line at the footprint's edge, since it does not visit the
// line's crossings of the footprint's perimeter, and a crossing it visits
// on the perimeter is inside or out by the even-odd rule's rounding.

/// "walls: a stroke is hidden up to where the walk up the line of sight
/// meets terrain, on abutting copies and across a notch".
#[test]
fn a_stroke_is_hidden_up_to_where_the_walk_up_the_line_of_sight_meets_terrain() {
    let p2 = |x: f64, y: f64| (x, y);
    let mut layouts: Vec<(String, Heightfield, BoundaryPerimeter)> = Vec::new();
    for seed in [3u64, 17, 41] {
        // A tile, the next copy to its right and the next above it: two of
        // its walls stand on tile boundaries.
        let small = random_heightfield(8, seed, Truncation::None, 3.0).surface;
        let d = small.domain();
        let boxed = perimeter(&[
            p2(d.real.lo, d.imag.lo),
            p2(d.real.hi, d.imag.lo),
            p2(d.real.hi, d.imag.hi),
            p2(d.real.lo, d.imag.hi),
        ]);
        let tiles = vec![
            Affine2::IDENTITY,
            affine(1.0, 0.0, d.real.length(), 0.0, 1.0, 0.0),
            affine(1.0, 0.0, 0.0, 0.0, 1.0, d.imag.length()),
        ];
        layouts.push((
            format!("seed {seed}, abutting copies"),
            Heightfield::new(
                small,
                Mesh::empty(),
                tiles,
                Region::Full,
                1,
                None,
                Walls::Perimeter { perimeter: boxed.clone(), base: 0.0 },
            ),
            boxed,
        ));
        // A footprint with a notch cut into its far side, walled all round,
        // the notch along grid lines: inside a cell a wall's top edge
        // follows the surface's interpolated height, not the drawn
        // triangles, and the traversal hides it at its start.
        let fine = random_heightfield(16, seed, Truncation::None, 3.0).surface;
        let notched = perimeter(&[
            (-1.0, -1.5),
            (1.0, -1.5),
            (1.0, 0.5),
            (0.25, 0.5),
            (0.25, -0.5),
            (-0.25, -0.5),
            (-0.25, 0.5),
            (-1.0, 0.5),
        ]);
        layouts.push((
            format!("seed {seed}, a notch"),
            Heightfield::new(
                fine,
                Mesh::empty(),
                vec![Affine2::IDENTITY],
                Region::Inside(notched.clone()),
                1,
                None,
                Walls::Perimeter { perimeter: notched.clone(), base: 0.0 },
            ),
            notched,
        ));
    }
    let cameras = stress_cameras();
    for (name, h, walls) in &layouts {
        let dom = h.surface.domain();
        let context = HatchContext {
            perimeter: Some(walls.clone()),
            region: h.region.clone(),
            tiles: h.tiles.clone(),
            base: 0.0,
        };
        let hatch = h
            .surface
            .hatch(&LayerSource::wall_hatching(all_edges(walls), 0.1, false, 0.0, 0.0), &context, None)
            .unwrap();
        let strokes = paths(&hatch);
        let dense = densified(&strokes);
        let normals = outward_normals(walls);
        let wall = |at: Vec2| -> Option<usize> { (0..walls.edges().len()).find(|&k| on_edge(walls, k, at)) };
        // The drawn height over a point of the terrain, in the footprint
        // and on a copy of the domain; None elsewhere.
        let terrain = |q: Vec2| -> Option<f64> {
            let w = P2::<WorldSpace>::new(q.x, q.y);
            let on_copy = h.tiles.iter().any(|tile| {
                let Some(back) = tile.inverse() else { return false };
                let u = back.apply(P3::new(q.x, q.y, 0.0));
                u.x() >= dom.real.lo && u.x() <= dom.real.hi && u.y() >= dom.imag.lo && u.y() <= dom.imag.hi
            });
            (h.region.contains(w) && on_copy).then(|| h.drawn_height(w))
        };
        let (highest, steepest) = highest_and_steepest(h);
        let tallies: Vec<(usize, usize, usize, usize, String)> = par_map(&cameras, |(camera, view)| {
            let d = -view.projection_direction();
            let dxy = Vec2::new(d.x, d.y);
            // The height below which terrain hides the stroke at `at`, or its
            // base; and where on the line's footprint the terrain stands
            // highest over it.
            let hidden_below = |at: Vec2, base: f64| -> (f64, Option<Vec2>) {
                let (coarse, fine) = (2e-3, 1e-5);
                let reach = 2.0 * (2.0f64.sqrt() * steepest * dxy.length() + d.z.abs()) * coarse;
                let value = |t: f64| terrain(at + t * dxy).map(|z| z - t * d.z);
                let mut samples: Vec<(f64, f64)> = Vec::new();
                let (mut best, mut t_best, mut t) = (base, None, fine);
                while highest - t * d.z > best - reach && t * dxy.length() < 10.0 {
                    if let Some(v) = value(t) {
                        samples.push((t, v));
                        if v > best {
                            best = v;
                            t_best = Some(t);
                        }
                    }
                    t += coarse;
                }
                for &(st, sv) in &samples {
                    if !(sv > best - reach) {
                        continue;
                    }
                    let mut u = swift::max(st - coarse, fine);
                    while u < st + coarse {
                        if let Some(v) = value(u)
                            && v > best
                        {
                            best = v;
                            t_best = Some(u);
                        }
                        u += fine;
                    }
                }
                (best, t_best.map(|t| at + t * dxy))
            };
            let two = runs(&h.visible_wall_curves(&hatch, view, 0.02, 0.0));
            let many = runs(&h.visible_wall_curves(&dense, view, 0.02, 0.0));
            let (mut compared, mut partly, mut wrong, mut explained) = (0, 0, 0, 0);
            let mut example = String::new();
            for p in &strokes {
                let at = Vec2::new(p[0].x(), p[0].y());
                let (base, top) = (p[0].z(), p[p.len() - 1].z());
                // A wall seen edge-on is left out: its line of sight runs along
                // the domain's edge, in and out of the terrain by rounding.
                let Some(k) = wall(at) else { continue };
                if !(normals[k].dot(dxy) > 1e-6) {
                    continue;
                }
                let (below, highest_at) = hidden_below(at, base);
                compared += 1;
                let at_edge = walls.edges().iter().any(|e| {
                    let Some(q) = highest_at else { return false };
                    if matches!(h.region, Region::Full) {
                        return false;
                    }
                    let along = e.end.v - e.start.v;
                    let t = swift::min(swift::max((q - e.start.v).dot(along) / along.dot(along), 0.0), 1.0);
                    (q - e.start.v - t * along).length() < 1e-4
                });
                if below > base + 1e-3 && below < top - 1e-3 {
                    partly += 1;
                }
                let key = key2(at.x, at.y);
                for (kind, kept) in [("two ends", two.get(&key)), ("256 vertices", many.get(&key))] {
                    let kept: &[(f64, f64)] = kept.map_or(&[], |v| v.as_slice());
                    if agrees(kept, below, top) {
                        continue;
                    }
                    let first_lo = kept.iter().find(|r| r.0 < top - 1e-3).map_or(top, |r| r.0);
                    if at_edge && first_lo < below {
                        explained += 1;
                        continue;
                    }
                    wrong += 1;
                    if example.is_empty() {
                        example = format!(
                            "; e.g. {camera}, {kind}, at ({:.3}, {:.3}) hidden below {below:.4}, kept {:?}",
                            at.x,
                            at.y,
                            kept.iter().map(|r| format!("{:.4} to {:.4}", r.0, r.1)).collect::<Vec<_>>()
                        );
                    }
                }
            }
            (compared, partly, wrong, explained, example)
        });
        let compared: usize = tallies.iter().map(|t| t.0).sum();
        let partly: usize = tallies.iter().map(|t| t.1).sum();
        let wrong: usize = tallies.iter().map(|t| t.2).sum();
        let explained: usize = tallies.iter().map(|t| t.3).sum();
        let example = tallies.iter().map(|t| t.4.as_str()).find(|e| !e.is_empty()).unwrap_or("");
        assert!(
            wrong == 0 && partly > 10,
            "{name}: a stroke on a wall that faces the eye keeps what the walk sees: {wrong} disagree, of {compared} \
             strokes each tested twice; {explained} at the footprint's edge; {partly} partly hidden{example}"
        );
    }
}

// MARK: - curves on a riser are tested by the riser
//
// A riser is a vertical face of the solid, as a wall is: a hatching that
// selects the risers hatches it, and the truncation edge's layer draws its
// bottom edge with its top. Both stand a hair onto the lower side, and a point on them is tested
// by the riser's facing and then by the traversal, forgiven nothing.

/// The larger of a grid cell's two sides.
fn cell_of(surface: &Surface) -> f64 {
    let g = &*surface.height;
    swift::max(
        g.domain.real.length().abs() / (g.width - 1) as f64,
        g.domain.imag.length().abs() / (g.height - 1) as f64,
    )
}

/// "risers: a riser's hatching and bottom edge lie on it, from its bottom
/// to its top, where it stands".
#[test]
fn a_risers_hatching_and_bottom_edge_lie_on_it_from_its_bottom_to_its_top() {
    // 1/(z - 1/2) over Re [-1/2, 3/2], Im [-1, 1], at 2 left of Re z = 1/2
    // and 4 beyond: the riser stands where |f| = 1/|Im z| reaches 2 on the
    // boundary, |Im z| ≤ 1/2, from 2 up to min(1/|Im z|, 4). Hatched at
    // 0.07, the boundary is 29 whole gaps across the domain, as the
    // plateaus divide it, so a line stands at each Im z = -1 + 2k/29 the
    // riser holds, none at its ends. On a footprint with a notch cut
    // from Im z = 0.2 back it stands to 0.2, and on one that ends at the
    // boundary nowhere: the wall there rises to the top. And 1/(z -
    // 0.45) cut at 4, in two tiles side by side, the second translated
    // by 2: at Re z = 3/2 the second's top, 1/|0.95 - i Im z|, stands
    // over the first's, 1/|1.05 + i Im z|, all along the tile boundary,
    // and the riser between them faces the first.
    let window = domain(-0.5, 1.5, -1.0, 1.0);
    let tower = build(&request("1/(z - 0.5)", window, 48, banded(&[(0.5, 2.0)], 4.0)), true);
    let pair = build(&request("1/(z - 0.45)", window, 48, Truncation::Uniform(4.0)), true);
    let boxed = perimeter(&[(-0.5, -1.0), (1.5, -1.0), (1.5, 1.0), (-0.5, 1.0)]);
    let notched = perimeter(&[
        (-0.5, -1.0),
        (1.5, -1.0),
        (1.5, 1.0),
        (0.7, 1.0),
        (0.7, 0.2),
        (0.3, 0.2),
        (0.3, 1.0),
        (-0.5, 1.0),
    ]);
    let half = perimeter(&[(-0.5, -1.0), (0.5, -1.0), (0.5, 1.0), (-0.5, 1.0)]);
    let side = affine(1.0, 0.0, 2.0, 0.0, 1.0, 0.0);
    let tower_top = |y: f64| swift::min(1.0 / y.abs(), 4.0);
    let two = |_: f64| 2.0;
    let pair_bottom = |y: f64| 1.0 / kurven_math::real::hypot(1.05, y);
    let pair_top = |y: f64| 1.0 / kurven_math::real::hypot(0.95, y);
    let cases: Vec<(
        &str,
        &KurvenBundle,
        BoundaryPerimeter,
        Region,
        Vec<Affine2>,
        f64,
        Option<(f64, f64)>,
        &dyn Fn(f64) -> f64,
        &dyn Fn(f64) -> f64,
    )> = vec![
        (
            "a tower at 2 | 4",
            &tower,
            boxed.clone(),
            Region::Full,
            vec![Affine2::IDENTITY],
            0.5,
            Some((-0.5, 0.5)),
            &two,
            &tower_top,
        ),
        (
            "a tower at 2 | 4, a notch cut from Im z = 0.2 back",
            &tower,
            notched.clone(),
            Region::Inside(notched),
            vec![Affine2::IDENTITY],
            0.5,
            Some((-0.5, 0.2)),
            &two,
            &tower_top,
        ),
        (
            "a tower at 2 | 4, the footprint ending at the boundary",
            &tower,
            half.clone(),
            Region::Inside(half),
            vec![Affine2::IDENTITY],
            0.5,
            None,
            &two,
            &tower_top,
        ),
        (
            "two tiles side by side meeting with different tops",
            &pair,
            boxed,
            Region::Full,
            vec![Affine2::IDENTITY, side],
            1.5,
            Some((-1.0, 1.0)),
            &pair_bottom,
            &pair_top,
        ),
    ];
    for (name, built, walls, region, tiles, at, stands, bottom, top) in cases {
        let surface = &built.surface;
        let refine = built.refine.as_ref().expect("a refiner");
        let cell = cell_of(surface);
        let tolerance = refine.tolerance * cell;
        let tiled = tiles.len() > 1;
        let context =
            HatchContext { perimeter: Some(walls.clone()), region: region.clone(), tiles: tiles.clone(), base: 0.0 };
        let sources = [
            LayerSource::wall_hatching(all_edges(&walls), 0.07, false, 0.0, 0.0),
            LayerSource::riser_hatching(0.07, tiled, false),
        ];
        let strokes: Vec<Vec<P3<WorldSpace>>> =
            sources.iter().flat_map(|s| paths(&surface.hatch(s, &context, Some(refine)).unwrap())).collect();
        let edges = surface.hatch(&LayerSource::TruncationEdge { tiled }, &context, Some(refine)).unwrap();
        // A billionth of a cell onto the lower side of the riser's line,
        // Re z = `at`, which is to its left in all four.
        let on_lower_side = |v: &P3<WorldSpace>| v.x() < at && at - v.x() < 1e-9 * cell * 2.0;
        // Exactly on a wall: on the line of a perimeter edge, all of
        // whose edges run along an axis.
        let on_wall = |v: &P3<WorldSpace>| {
            walls.edges().iter().any(|e| {
                if e.start.x() == e.end.x() {
                    v.x() == e.start.x()
                        && v.y() >= swift::min(e.start.y(), e.end.y())
                        && v.y() <= swift::max(e.start.y(), e.end.y())
                } else {
                    v.y() == e.start.y()
                        && v.x() >= swift::min(e.start.x(), e.end.x())
                        && v.x() <= swift::max(e.start.x(), e.end.x())
                }
            })
        };
        let risers: Vec<&Vec<P3<WorldSpace>>> = strokes.iter().filter(|p| p.iter().all(on_lower_side)).collect();
        let strays = strokes.iter().filter(|p| !p.iter().all(on_wall) && !p.iter().all(on_lower_side)).count();
        let n = 30;
        let gaps: Vec<f64> = (1..(n - 1)).map(|k| -1.0 + 2.0 * k as f64 / (n - 1) as f64).collect();
        let wanted: Vec<f64> =
            stands.map_or(Vec::new(), |s| gaps.iter().copied().filter(|&g| g > s.0 && g < s.1).collect());
        let mut off: Vec<String> = Vec::new();
        for p in &risers {
            if !(p.len() == 2
                && p[0].y() == p[1].y()
                && (p[0].z() - bottom(p[0].y())).abs() <= tolerance
                && (p[1].z() - top(p[0].y())).abs() <= tolerance)
            {
                off.push(format!("Im z = {:.4} from {:.6} to {:.6}", p[0].y(), p[0].z(), p[p.len() - 1].z()));
            }
        }
        let mut ats: Vec<f64> = risers.iter().map(|p| p[0].y()).collect();
        ats.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let placed = ats.len() == wanted.len() && ats.iter().zip(&wanted).all(|(a, w)| (a - w).abs() <= 1e-12);
        assert!(
            placed && off.is_empty() && strays == 0,
            "{name}: a hatch line at each whole gap the riser holds, from its bottom to its top, a hair onto the lower \
             side: {} lines, {} wanted, {} not from bottom to top, {strays} on neither a wall nor the riser{}",
            risers.len(),
            wanted.len(),
            off.len(),
            if off.is_empty() {
                String::new()
            } else {
                format!(": {}", off.iter().take(3).cloned().collect::<Vec<_>>().join(", "))
            }
        );
        let bottoms: Vec<Vec<P3<WorldSpace>>> =
            paths(&edges).into_iter().filter(|p| p.iter().all(on_lower_side)).collect();
        let bottom_off = bottoms
            .iter()
            .flatten()
            .map(|v| (v.z() - bottom(v.y())).abs())
            .fold(0.0f64, |m, x| if m < x { x } else { m });
        let ends: Vec<(f64, f64)> = bottoms.iter().map(|b| (b[0].y(), b[b.len() - 1].y())).collect();
        let spans = match stands {
            Some(s) => ends.len() == 1 && (ends[0].0 - s.0).abs() <= 1e-9 && (ends[0].1 - s.1).abs() <= 1e-9,
            None => ends.is_empty(),
        };
        assert!(
            spans && bottom_off <= tolerance,
            "{name}: its bottom edge runs where it stands, at the lower top, a hair onto the lower side: {} bottoms, \
             from {:?}, off the lower top by {bottom_off:.2e}",
            bottoms.len(),
            ends.iter().map(|e| format!("{:.6} to {:.6}", e.0, e.1)).collect::<Vec<_>>()
        );
    }
}

/// "risers: only a layer that selects the risers hatches them, in place or
/// on every tile".
#[test]
fn only_a_layer_that_selects_the_risers_hatches_them() {
    // The tower at 2 | 4 as a bundle draws it: a wall hatching on all
    // four walls, the risers' hatching in place, and the truncation edge
    // in place. A riser's line is counted over every hatching layer by
    // where it stands: in place, a hair left of Re z = 1/2; on a second
    // tile beside the first, of 5/2; at the tile boundary between, of 3/2.
    let window = domain(-0.5, 1.5, -1.0, 1.0);
    let built = build(&request("1/(z - 0.45)", window, 48, banded(&[(0.5, 2.0)], 4.0)), true);
    let g = &*built.surface.height;
    let cell = swift::max(window.real.length() / (g.width - 1) as f64, window.imag.length() / (g.height - 1) as f64);
    // The hatching's lines on each riser, under `manifest`.
    let risers = |manifest: &Manifest| -> (usize, usize, usize) {
        let bundle =
            KurvenBundle::described(built.url.clone(), manifest.clone(), built.surface.clone(), built.refine.clone());
        let feet: Vec<P3<WorldSpace>> = bundle
            .layers
            .iter()
            .filter(|l| matches!(l.spec.source, LayerSource::Hatching { .. }))
            .flat_map(|l| l.paths.paths().map(|p| p[0]).collect::<Vec<_>>())
            .collect();
        let on = |x: f64| feet.iter().filter(|f| x - f.x() > 0.5e-9 * cell && x - f.x() < 2e-9 * cell).count();
        (on(0.5), on(2.5), on(1.5))
    };
    let with = |edit: &dyn Fn(&mut Manifest)| -> Manifest {
        let mut m = built.manifest.clone();
        edit(&mut m);
        m
    };
    let walls = |edges: Vec<i64>| {
        move |m: &mut Manifest| {
            for spec in &mut m.layers {
                if let LayerSource::Hatching { faces: FaceSelector::Walls { edges: e, .. }, .. } = &mut spec.source {
                    *e = edges.clone();
                }
            }
        }
    };
    let risers_tiled = |tiled: Option<bool>| {
        move |m: &mut Manifest| {
            m.layers.retain_mut(|spec| {
                let LayerSource::Hatching { faces: FaceSelector::Risers { tiled: t, .. }, .. } = &mut spec.source
                else {
                    return true;
                };
                match tiled {
                    None => false,
                    Some(tiled) => {
                        *t = tiled;
                        true
                    }
                }
            });
        }
    };
    let beside = |m: &mut Manifest| m.occluder.tiles = vec![Affine2::IDENTITY, affine(1.0, 0.0, 2.0, 0.0, 1.0, 0.0)];
    let all = risers(&built.manifest);
    assert!(
        all.0 > 30 && all.1 == 0 && all.2 == 0,
        "every wall and the risers hatched, in place: the riser, in place: {all:?}"
    );
    let none = risers(&with(&risers_tiled(None)));
    assert!(none == (0, 0, 0), "every wall hatched and no layer on the risers: no riser: {none:?}");
    for edges in [vec![0, 2], vec![1], vec![]] {
        let some = risers(&with(&walls(edges.clone())));
        assert!(some.0 == all.0, "walls {edges:?} hatched: the same riser lines: {some:?}");
    }
    let edgeless = risers(&with(&|m: &mut Manifest| {
        m.layers.retain(|spec| !matches!(spec.source, LayerSource::TruncationEdge { .. }))
    }));
    assert!(edgeless.0 == all.0, "no truncation edge drawn: the same riser lines: {edgeless:?}");
    let untiled = risers(&with(&beside));
    assert!(
        untiled.0 == all.0 && untiled.1 == 0 && untiled.2 == 0,
        "two tiles, the risers hatched in place: the riser in place, none on the second tile or between them: \
         {untiled:?}"
    );
    let tiled = risers(&with(&|m: &mut Manifest| {
        beside(m);
        risers_tiled(Some(true))(m)
    }));
    assert!(
        tiled.0 == all.0 && tiled.1 == all.0 && tiled.2 > 30,
        "two tiles, the risers hatched on every tile: the riser on both, and the one between them: {tiled:?}"
    );
}

// Where a riser faces away, the solid on its higher side stands between
// its hatching and bottom edge and the eye, a hair along the line of
// sight; where it faces the eye, what stands in front hides them from the
// bottom up. What the solid hides of a vertical line at a point is
// everything below the greatest of h(q(t)) - t d_z along the line of
// sight, q(t) its footprint and h the solid's top there: min(|f|, h(x))
// on the copy under it, the greater where two meet. A point is seen when
// that is not above it. The walk reads that a five-hundredth of a unit at
// a time, then a hundred-thousandth around each greatest within reach of
// the best, as the walls' walk does. A hatch line, tested from its two
// ends and at 256 vertices, keeps one run from there to its top, or none;
// each vertex of a bottom edge is kept where the walk sees it and dropped
// where it is under the solid by a thousandth, under every camera of the
// stress orbit. A riser seen edge-on is left out, as a wall is: its line
// of sight runs along it, a hair from its plane. On a tower, and mirrored,
// on Γ with three risers, and on the tower in two tiles, the second beside
// the first and then turned, both copies' risers and the one at the tile
// boundary between; and on the grid, on random terrain in abutting copies,
// where the traversal's skin is the drawing's depth tolerance and a riser
// can be lower than it.

/// "risers: a riser's hatching and bottom edge are hidden up to where the
/// walk up the line of sight meets the solid".
#[test]
fn a_risers_hatching_and_bottom_edge_are_hidden_up_to_where_the_walk_meets_the_solid() {
    let window = domain(-0.5, 1.5, -1.0, 1.0);
    let tower = request("1/(z - 0.45)", window, 100, banded(&[(0.5, 2.0)], 4.0));
    // The tower mirrored, the higher band on the left: the traversal's
    // sample on a band boundary reads the band beyond it, here the
    // lower, and where |f| falls under a bottom's height within a column
    // of the boundary, no sample finds the higher side's solid in front
    // of the bottom; the riser's facing hides it.
    let mirrored = request("1/(z - 0.55)", window, 100, banded(&[(0.5, 4.0)], 2.0));
    let gamma =
        request("gamma(z)", domain(-4.5, 4.5, -2.5, 2.5), 180, banded(&[(-2.5, 0.5), (-1.5, 1.0), (0.5, 1.5)], 2.5));
    let side = affine(1.0, 0.0, 2.0, 0.0, 1.0, 0.0);
    let turned = affine(0.0, -1.0, 2.5, 1.0, 0.0, -0.5);
    type Line = (Vec2, Vec2);
    let mut landscapes: Vec<(
        String,
        Surface,
        Option<kurven_core::surface::ContourRefine>,
        Vec<Affine2>,
        Vec<Line>,
        f64,
    )> = Vec::new();
    let on_f: Vec<(&str, &LandscapeRequest, Vec<Affine2>, Vec<Line>, f64)> = vec![
        ("a tower at 2 | 4", &tower, vec![Affine2::IDENTITY], vec![], 0.025),
        ("a tower at 4 | 2, the higher band on the left", &mirrored, vec![Affine2::IDENTITY], vec![], 0.025),
        ("Γ at 0.5 | 1 | 1.5 | 2.5", &gamma, vec![Affine2::IDENTITY], vec![], 0.1),
        (
            "a tower at 2 | 4 in two tiles side by side",
            &tower,
            vec![Affine2::IDENTITY, side],
            vec![(Vec2::new(1.5, 0.0), Vec2::new(1.0, 0.0))],
            0.05,
        ),
        (
            "a tower at 2 | 4 in two tiles, the second turned",
            &tower,
            vec![Affine2::IDENTITY, turned],
            vec![(Vec2::new(1.5, 0.0), Vec2::new(1.0, 0.0))],
            0.05,
        ),
    ];
    for (name, r, tiles, lines, spacing) in on_f {
        let built = build(r, true);
        landscapes.push((name.to_string(), built.surface, built.refine, tiles, lines, spacing));
    }
    // On the grid the traversal forgives a point on the surface the
    // drawing's depth tolerance, a skin a riser's own curves are not
    // forgiven: random terrain on a tile and the next copies to its right
    // and above it, as the walls' walk has it, with risers all along the
    // two tile boundaries, some lower than the skin.
    for seed in [3u64, 17] {
        let terrain = random_heightfield(8, seed, Truncation::None, 3.0).surface;
        let d = terrain.domain();
        landscapes.push((
            format!("seed {seed}, abutting copies, on the grid"),
            terrain,
            None,
            vec![
                Affine2::IDENTITY,
                affine(1.0, 0.0, d.real.length(), 0.0, 1.0, 0.0),
                affine(1.0, 0.0, 0.0, 0.0, 1.0, d.imag.length()),
            ],
            vec![(Vec2::new(d.real.hi, 0.0), Vec2::new(1.0, 0.0)), (Vec2::new(0.0, d.imag.hi), Vec2::new(0.0, 1.0))],
            0.1,
        ));
    }
    let cameras = stress_cameras();
    let mut partly = 0;
    for (name, surface, refine, tiles, tile_boundaries, spacing) in &landscapes {
        let magnitude: Option<Magnitude> = refine.as_ref().and_then(|r| r.magnitude.clone());
        let dom = surface.domain();
        let g = &*surface.height;
        let (lo, hi) = (swift::min(dom.real.lo, dom.real.hi), swift::max(dom.real.lo, dom.real.hi));
        let (ylo, yhi) = (swift::min(dom.imag.lo, dom.imag.hi), swift::max(dom.imag.lo, dom.imag.hi));
        let boxed = perimeter(&[(lo, ylo), (hi, ylo), (hi, yhi), (lo, yhi)]);
        let h = Heightfield::new(
            surface.clone(),
            Mesh::empty(),
            tiles.clone(),
            Region::Full,
            1,
            refine.clone(),
            Walls::Perimeter { perimeter: boxed.clone(), base: 0.0 },
        );
        let context = HatchContext { perimeter: Some(boxed), region: Region::Full, tiles: tiles.clone(), base: 0.0 };
        let cell = cell_of(surface);
        // The lines the risers stand on, in place: the band boundaries
        // the truncation steps at inside the domain, on every copy, and
        // the tile boundaries.
        let mut lines: Vec<Line> = tile_boundaries.clone();
        if let Truncation::RealBands { bands, .. } = &surface.truncation {
            for b in bands.iter().map(|b| b.below) {
                if !(b > lo && b < hi) {
                    continue;
                }
                // Square to the copy's boundary: the gradient of the
                // domain's x, carried back.
                for tile in tiles {
                    if let Some(back) = tile.inverse() {
                        lines.push((tile.apply(P3::new(b, 0.0, 0.0)).xy().v, Vec2::new(back.a, back.b).normalized()));
                    }
                }
            }
        }
        // The riser a point stands a hair onto a side of, a billionth of a
        // cell, as its curves stand, not a rounding: its line's normal.
        let riser = |v: P3<WorldSpace>| -> Option<Vec2> {
            lines
                .iter()
                .find(|line| {
                    let d = (Vec2::new(v.x(), v.y()) - line.0).dot(line.1).abs();
                    d > 0.5e-9 * cell && d < 2e-9 * cell
                })
                .map(|line| line.1)
        };
        let on_riser = |v: &P3<WorldSpace>| riser(*v).is_some();
        let hatch =
            surface.hatch(&LayerSource::riser_hatching(*spacing, true, false), &context, refine.as_ref()).unwrap();
        let edges = surface.hatch(&LayerSource::TruncationEdge { tiled: true }, &context, refine.as_ref()).unwrap();
        let strokes: Vec<Vec<P3<WorldSpace>>> = paths(&hatch).into_iter().filter(|p| p.iter().all(on_riser)).collect();
        let bottoms: Vec<Vec<P3<WorldSpace>>> = paths(&edges).into_iter().filter(|p| p.iter().all(on_riser)).collect();
        let ends = PolylineSet::from_paths(&strokes);
        let dense = densified(&strokes);
        let bottom_set = PolylineSet::from_paths(&bottoms);
        // The solid's top over a point of the plane: on the copy under it,
        // the greater where two meet; None off every copy.
        let top = |q: Vec2| -> Option<f64> {
            let mut best: Option<f64> = None;
            for tile in tiles {
                let Some(back) = tile.inverse() else { continue };
                let p = back.apply(P3::new(q.x, q.y, 0.0));
                if !(p.x() >= lo && p.x() <= hi && p.y() >= ylo && p.y() <= yhi) {
                    continue;
                }
                // On the grid, the drawn triangles, through the tile under
                // the point.
                let Some(magnitude) = &magnitude else { return Some(h.drawn_height(P2::new(q.x, q.y))) };
                let u = magnitude(P2::new(p.x(), p.y()));
                best = Some(swift::max(
                    best.unwrap_or(f64::NEG_INFINITY),
                    swift::min(if u.is_nan() { f64::INFINITY } else { u }, surface.truncation_height_at(p.x())),
                ));
            }
            best
        };
        // The steepest the truncated top rises between two samples in one
        // band, and the highest it stands.
        let (mut highest, mut steepest) = (0.0f64, 0.0f64);
        let (dx, dy) = (dom.real.length() / (g.width - 1) as f64, dom.imag.length() / (g.height - 1) as f64);
        let truncated = |i: usize, j: usize| -> f64 {
            swift::min(f64::from(g.at(i, j)), surface.truncation_height_at(g.position(i, 0).x()))
        };
        for j in 0..g.height {
            for i in 0..g.width {
                let (z, x) = (truncated(i, j), g.position(i, 0).x());
                if !z.is_finite() {
                    continue;
                }
                highest = swift::max(highest, z);
                if i + 1 < g.width
                    && surface.truncation_height_at(g.position(i + 1, 0).x()) == surface.truncation_height_at(x)
                {
                    steepest = swift::max(steepest, (truncated(i + 1, j) - z).abs() / dx.abs());
                }
                if j + 1 < g.height {
                    steepest = swift::max(steepest, (truncated(i, j + 1) - z).abs() / dy.abs());
                }
            }
        }
        let extent = tiles.len() as f64 * swift::max(dom.real.length().abs(), dom.imag.length().abs()) * 2.0;
        let tallies: Vec<[usize; 9]> = par_map(&cameras, |(camera, view)| {
            let d = -view.projection_direction();
            let dxy = Vec2::new(d.x, d.y);
            // The height below which the solid hides a vertical line at
            // `at`, or `base`.
            let hidden_below = |at: Vec2, base: f64| -> f64 {
                let (coarse, fine) = (2e-3, 1e-5);
                let reach = 2.0 * (2.0 * 2.0f64.sqrt() * steepest * dxy.length() + d.z.abs()) * coarse;
                let value = |t: f64| top(at + t * dxy).map(|z| z - t * d.z);
                let mut samples: Vec<(f64, f64)> = Vec::new();
                let (mut best, mut t) = (base, fine);
                while highest - t * d.z > best - reach && t * dxy.length() < extent {
                    if let Some(v) = value(t) {
                        samples.push((t, v));
                        best = swift::max(best, v);
                    }
                    t += coarse;
                }
                // Around each greatest of the samples within reach of the
                // best, finely: a riser's step lies between two samples.
                for k in 0..samples.len() {
                    let (st, sv) = samples[k];
                    if !(sv > best - reach
                        && (k == 0 || sv >= samples[k - 1].1)
                        && (k + 1 == samples.len() || sv >= samples[k + 1].1))
                    {
                        continue;
                    }
                    let mut u = swift::max(st - coarse, fine);
                    while u < st + coarse {
                        if let Some(v) = value(u)
                            && v > best
                        {
                            best = v;
                        }
                        u += fine;
                    }
                }
                best
            };
            // As the risers' hatching layer and the truncation edge's test
            // them (`SceneGeometry::face_of`).
            let two = runs(&h.visible_curves(&ends, Some(FaceKind::Riser), view, 0.02, 0.0));
            let many = runs(&h.visible_curves(&dense, Some(FaceKind::Riser), view, 0.02, 0.0));
            let edge_on = |v: P3<WorldSpace>| riser(v).is_none_or(|n| n.dot(dxy).abs() <= 1e-6);
            let mut t = [0usize; 9];
            let [compared, whole, kept, wrong, vertices, seen, hidden, wrong_vertices, partly] = &mut t;
            let mut example = String::new();
            for p in &strokes {
                if edge_on(p[0]) {
                    continue;
                }
                let at = Vec2::new(p[0].x(), p[0].y());
                let (base, top_z) = (p[0].z(), p[1].z());
                let below = hidden_below(at, base);
                *compared += 1;
                if below > base + 1e-3 && below < top_z - 1e-3 {
                    *partly += 1;
                }
                if below >= top_z - 1e-3 {
                    *whole += 1;
                } else {
                    *kept += 1;
                }
                let key = key2(at.x, at.y);
                for (kind, got) in [("two ends", two.get(&key)), ("256 vertices", many.get(&key))] {
                    let got: &[(f64, f64)] = got.map_or(&[], |v| v.as_slice());
                    if !agrees(got, below, top_z) {
                        *wrong += 1;
                        if example.is_empty() {
                            example = format!(
                                "; e.g. {camera}, {kind}, a line at ({:.4}, {:.4}) hidden below {below:.4}, kept {:?}",
                                at.x,
                                at.y,
                                got.iter().map(|r| format!("{:.4} to {:.4}", r.0, r.1)).collect::<Vec<_>>()
                            );
                        }
                    }
                }
            }
            let kept_bottoms: HashSet<(u64, u64, u64)> =
                h.visible_curves(&bottom_set, None, view, 0.02, 0.0).vertices.iter().map(|v| key3(v.v)).collect();
            for &v in bottoms.iter().flatten() {
                if edge_on(v) {
                    continue;
                }
                // Where the riser has no height its bottom is its top, and a
                // vertex there seen alone is no run to draw; nor where it
                // ends on the window's edge, from which a line of sight
                // leaves the solid at once, and its neighbours' do not.
                let across = riser(v).map_or(v.z(), |n| {
                    let sides: Vec<f64> =
                        [-4.0, 4.0].iter().filter_map(|s| top(Vec2::new(v.x(), v.y()) + s * 1e-9 * cell * n)).collect();
                    sides
                        .into_iter()
                        .fold(None, |m: Option<f64>, x| Some(m.map_or(x, |b| if b < x { x } else { b })))
                        .unwrap_or(v.z())
                });
                let on_window_edge = tiles.iter().filter_map(|t| t.inverse()).any(|back| {
                    let q = back.apply(v);
                    (q.y() - ylo).abs() <= 1e-12
                        || (q.y() - yhi).abs() <= 1e-12
                        || (q.x() - lo).abs() <= 1e-12
                        || (q.x() - hi).abs() <= 1e-12
                });
                if !(across > v.z() + 1e-3) || on_window_edge {
                    continue;
                }
                let over = hidden_below(Vec2::new(v.x(), v.y()), v.z()) - v.z();
                *vertices += 1;
                if over > 1e-3 {
                    *hidden += 1;
                    if kept_bottoms.contains(&key3(v.v)) {
                        *wrong_vertices += 1;
                        if example.is_empty() {
                            example = format!(
                                "; e.g. {camera}, a bottom's vertex at ({:.4}, {:.4}, {:.4}) under the solid by {over:.4}, kept",
                                v.x(),
                                v.y(),
                                v.z()
                            );
                        }
                    }
                } else if over <= 1e-7 {
                    *seen += 1;
                    if !kept_bottoms.contains(&key3(v.v)) {
                        *wrong_vertices += 1;
                        if example.is_empty() {
                            example = format!(
                                "; e.g. {camera}, a bottom's vertex at ({:.4}, {:.4}, {:.4}) in view, dropped",
                                v.x(),
                                v.y(),
                                v.z()
                            );
                        }
                    }
                }
            }
            if !example.is_empty() {
                eprintln!("{name}{example}");
            }
            t
        });
        let sum = |k: usize| tallies.iter().map(|t| t[k]).sum::<usize>();
        let (compared, whole, kept, wrong, vertices, seen, hidden, wrong_vertices) =
            (sum(0), sum(1), sum(2), sum(3), sum(4), sum(5), sum(6), sum(7));
        partly += sum(8);
        assert!(
            wrong == 0 && wrong_vertices == 0 && whole > 100 && kept > 100 && seen > 100 && hidden > 100,
            "{name}: a riser's hatching and bottom edge keep what the walk up the line of sight sees: {wrong} of \
             {compared} lines disagree, each tested twice ({whole} hidden whole, {kept} kept); {wrong_vertices} of \
             {vertices} bottom vertices ({seen} seen, {hidden} hidden)"
        );
    }
    assert!(partly > 10, "and some lines are hidden in part, from the bottom up: {partly} partly hidden");
}

/// The walk up a line of sight over a landscape on its tiles, for the riser
/// suites: the solid's top, closed, h(q) = min(|f|, h(x)) on the copy under q,
/// the greater where two meet and the greater band's height on a band
/// boundary, and how far under it a point lies along its line of sight.
struct SolidWalk {
    surface: Surface,
    magnitude: Magnitude,
    backs: Vec<Affine2>,
    x: (f64, f64),
    y: (f64, f64),
    /// The band boundaries the truncation steps at, each with the greater
    /// height either side.
    steps: Vec<(f64, f64)>,
    /// The copies' bounding box in the plane.
    lo: Vec2,
    hi: Vec2,
    /// The highest the truncated top stands, and the steepest it rises
    /// between two samples in one band.
    highest: f64,
    steepest: f64,
}

impl SolidWalk {
    fn new(surface: Surface, magnitude: Magnitude, tiles: &[Affine2]) -> SolidWalk {
        let backs: Vec<Affine2> = tiles.iter().filter_map(|t| t.inverse()).collect();
        let dom = surface.domain();
        let g = &*surface.height;
        let x = (swift::min(dom.real.lo, dom.real.hi), swift::max(dom.real.lo, dom.real.hi));
        let y = (swift::min(dom.imag.lo, dom.imag.hi), swift::max(dom.imag.lo, dom.imag.hi));
        let spans = surface.truncation.bands(x.0, x.1);
        let steps: Vec<(f64, f64)> = spans
            .windows(2)
            .filter(|w| w[0].height != w[1].height)
            .map(|w| (w[0].hi, swift::max(w[0].height, w[1].height)))
            .collect();
        let (mut lo, mut hi) = (Vec2::splat(f64::INFINITY), Vec2::splat(f64::NEG_INFINITY));
        for tile in tiles {
            for (cx, cy) in [(x.0, y.0), (x.1, y.0), (x.0, y.1), (x.1, y.1)] {
                let w = tile.apply(P3::new(cx, cy, 0.0)).xy().v;
                lo = lo.min(w);
                hi = hi.max(w);
            }
        }
        let (mut highest, mut steepest) = (0.0f64, 0.0f64);
        let (dx, dy) = (dom.real.length() / (g.width - 1) as f64, dom.imag.length() / (g.height - 1) as f64);
        let truncated =
            |i: usize, j: usize| swift::min(f64::from(g.at(i, j)), surface.truncation_height_at(g.position(i, 0).x()));
        for j in 0..g.height {
            for i in 0..g.width {
                let (z, xi) = (truncated(i, j), g.position(i, 0).x());
                if !z.is_finite() {
                    continue;
                }
                highest = swift::max(highest, z);
                if i + 1 < g.width
                    && surface.truncation_height_at(g.position(i + 1, 0).x()) == surface.truncation_height_at(xi)
                {
                    steepest = swift::max(steepest, (truncated(i + 1, j) - z).abs() / dx.abs());
                }
                if j + 1 < g.height {
                    steepest = swift::max(steepest, (truncated(i, j + 1) - z).abs() / dy.abs());
                }
            }
        }
        SolidWalk { surface, magnitude, backs, x, y, steps, lo, hi, highest, steepest }
    }

    /// Whether a point of the domain lies in it, to a rounding.
    fn in_domain(&self, p: P3<WorldSpace>, e: f64) -> bool {
        p.x() >= self.x.0 - e && p.x() <= self.x.1 + e && p.y() >= self.y.0 - e && p.y() <= self.y.1 + e
    }

    /// The solid's top over a point of the plane, closed; None off every copy.
    fn top(&self, q: Vec2) -> Option<f64> {
        let mut best: Option<f64> = None;
        for back in &self.backs {
            let p = back.apply(P3::new(q.x, q.y, 0.0));
            if !self.in_domain(p, 1e-12) {
                continue;
            }
            let px = swift::min(swift::max(p.x(), self.x.0), self.x.1);
            let py = swift::min(swift::max(p.y(), self.y.0), self.y.1);
            let u = (self.magnitude)(P2::new(px, py));
            let h = self
                .steps
                .iter()
                .find(|s| (s.0 - px).abs() <= 1e-12)
                .map_or_else(|| self.surface.truncation_height_at(px), |s| s.1);
            best = Some(swift::max(
                best.unwrap_or(f64::NEG_INFINITY),
                swift::min(if u.is_nan() { f64::INFINITY } else { u }, h),
            ));
        }
        best
    }

    /// How squarely the graph faces along `d` at a point of it, through the
    /// copy it lies on: the normal by differences either side, the lesser of
    /// the two, so that a crease faces neither way; None off every copy or on
    /// a plateau.
    fn graph_facing(&self, p: P3<WorldSpace>, d: Vec3) -> Option<f64> {
        let back = *self.backs.iter().find(|b| self.in_domain(b.apply(p), 1e-9))?;
        let e = 1e-6;
        let height = |q: Vec2| -> Option<f64> {
            let w = back.apply(P3::new(q.x, q.y, 0.0));
            if !self.in_domain(w, 1e-9) {
                return None;
            }
            let u = (self.magnitude)(P2::new(w.x(), w.y()));
            (u < self.surface.truncation_height_at(w.x()) - 1e-6).then_some(u)
        };
        let at = Vec2::new(p.x(), p.y());
        let u = height(at)?;
        let mut least = f64::INFINITY;
        for s in [-1.0, 1.0] {
            let ux = height(at + Vec2::new(s * e, 0.0))?;
            let uy = height(at + Vec2::new(0.0, s * e))?;
            let n = Vec3::new(-(ux - u) / (s * e), -(uy - u) / (s * e), 1.0);
            least = swift::min(least, n.dot(d) / n.length());
        }
        Some(least)
    }

    /// Whether `p` lies on the domain's edge on a copy, a wall, and its line
    /// of sight runs along it.
    fn edge_on(&self, p: P3<WorldSpace>, d: Vec3) -> bool {
        self.backs.iter().any(|back| {
            let o = back.apply(p);
            let r = Vec2::new(back.a * d.x + back.b * d.y, back.c * d.x + back.d * d.y);
            let e = 1e-9;
            self.in_domain(o, e)
                && (((o.x() - self.x.0).abs() <= e || (o.x() - self.x.1).abs() <= e) && r.x.abs() <= 1e-6
                    || ((o.y() - self.y.0).abs() <= e || (o.y() - self.y.1).abs() <= e) && r.y.abs() <= 1e-6)
        })
    }

    /// Whether `p` stands at a corner of the domain on a copy, where its top
    /// edge meets a vertical edge and the walls' facing has the say.
    fn at_corner(&self, p: P3<WorldSpace>) -> bool {
        self.backs.iter().any(|back| {
            let (o, e) = (back.apply(p), 1e-9);
            ((o.x() - self.x.0).abs() <= e || (o.x() - self.x.1).abs() <= e)
                && ((o.y() - self.y.0).abs() <= e || (o.y() - self.y.1).abs() <= e)
        })
    }

    /// Whether `p` lies on or a hair from a riser's line, a millionth: a
    /// band boundary on a copy, or a copy's edge where another meets it. The
    /// traversal and the walk each read such a point from their own side.
    fn beside_riser(&self, p: P3<WorldSpace>) -> bool {
        let e = 1e-6;
        let on: Vec<&Affine2> = self.backs.iter().filter(|b| self.in_domain(b.apply(p), e)).collect();
        on.iter().any(|back| {
            let o = back.apply(p);
            self.steps.iter().any(|s| (o.x() - s.0).abs() <= e)
                || on.len() > 1
                    && ((o.x() - self.x.0).abs() <= e
                        || (o.x() - self.x.1).abs() <= e
                        || (o.y() - self.y.0).abs() <= e
                        || (o.y() - self.y.1).abs() <= e)
        })
    }

    /// Where the line of sight from `p` along `d` crosses a riser's line
    /// below the highest top: a band boundary on a copy, or the edge of a
    /// copy where another meets it.
    fn crossings(&self, p: P3<WorldSpace>, d: Vec3) -> Vec<f64> {
        let mut out = Vec::new();
        for back in &self.backs {
            let o = back.apply(P3::new(p.x(), p.y(), 0.0));
            let r = Vec2::new(back.a * d.x + back.b * d.y, back.c * d.x + back.d * d.y);
            let mut lines: Vec<(usize, f64, bool)> = self.steps.iter().map(|s| (0, s.0, false)).collect();
            lines.extend([(0, self.x.0, true), (0, self.x.1, true), (1, self.y.0, true), (1, self.y.1, true)]);
            for (axis, level, edge) in lines {
                let ra = if axis == 0 { r.x } else { r.y };
                if !(ra.abs() > 1e-15) {
                    continue;
                }
                let t = (level - if axis == 0 { o.x() } else { o.y() }) / ra;
                if !(t > 0.0 && p.z() + t * d.z < self.highest) {
                    continue;
                }
                let w = P3::<WorldSpace>::new(o.x() + t * r.x, o.y() + t * r.y, 0.0);
                if !self.in_domain(w, 1e-9) {
                    continue;
                }
                if edge {
                    let q = P3::<WorldSpace>::new(p.x() + t * d.x, p.y() + t * d.y, 0.0);
                    if !(self.backs.iter().filter(|b| self.in_domain(b.apply(q), 1e-9)).count() > 1) {
                        continue;
                    }
                }
                out.push(t);
            }
        }
        out
    }

    /// How far under the solid `p` lies along its line of sight `d`: the
    /// greatest of h(q(t)) - z(t), less p.z, read a five-hundredth of a unit
    /// at a time from a hundred-thousandth on, then a hundred-thousandth at
    /// a time around each greatest within reach of the best, and at each of
    /// `crossings`, where the closed top is the higher side's. A riser's step
    /// lies between two samples, and a line through its top corner is in the
    /// solid for less than any step.
    fn depth(&self, p: P3<WorldSpace>, d: Vec3, crossings: &[f64]) -> f64 {
        let (at, dxy) = (Vec2::new(p.x(), p.y()), Vec2::new(d.x, d.y));
        let (coarse, fine) = (2e-3, 1e-5);
        let reach = 2.0 * (2.0 * 2.0f64.sqrt() * self.steepest * dxy.length() + d.z.abs()) * coarse;
        // Where the line's footprint leaves the copies' bounding box.
        let mut end = f64::INFINITY;
        for (a, o, l, h) in [(dxy.x, at.x, self.lo.x, self.hi.x), (dxy.y, at.y, self.lo.y, self.hi.y)] {
            if a.abs() > 1e-15 {
                end = swift::min(end, ((if a > 0.0 { h } else { l }) - o) / a);
            }
        }
        let value = |t: f64| self.top(at + t * dxy).map(|z| z - t * d.z);
        let mut samples: Vec<(f64, f64)> = Vec::new();
        let (mut best, mut t) = (p.z(), fine);
        while self.highest - t * d.z > best - reach && t <= end + coarse {
            if let Some(v) = value(t) {
                samples.push((t, v));
                best = swift::max(best, v);
            }
            t += coarse;
        }
        for k in 0..samples.len() {
            let (st, sv) = samples[k];
            if !(sv > best - reach
                && (k == 0 || sv >= samples[k - 1].1)
                && (k + 1 == samples.len() || sv >= samples[k + 1].1))
            {
                continue;
            }
            let mut u = swift::max(st - coarse, fine);
            while u < st + coarse {
                if let Some(v) = value(u)
                    && v > best
                {
                    best = v;
                }
                u += fine;
            }
        }
        for &t in crossings {
            if let Some(v) = value(t)
                && v > best
            {
                best = v;
            }
        }
        best - p.z()
    }
}

// A line of sight that crosses a riser's line between the two tops
// there, over the lower and under the higher, passes through the solid,
// however soon it comes out on the higher side. The walk up the line of
// sight (`SolidWalk`) reads the greatest of h(q(t)) - z(t), as above, and
// the solid closed at each riser's line it crosses: at the higher top.
// Under every camera of the stress orbit, every vertex of the plateau
// hatching, the contours and the wall edges that is kept, and whose line
// of sight crosses a riser's line below the highest top, is under the
// solid by no more than the depth tolerance; and none dropped is in
// plain view of the walk, under the solid by no more than a millionth,
// on a face turned to the eye: a plateau seen from above, the graph
// facing the eye by a margin, a wall's top edge. On the dip, two
// poles at 0.49 ± 0.1i whose truncation steps from 0.8 to 4 at Re z =
// 1/2, where a line of sight passes into the higher band a hair beyond
// the boundary and under its top, between two grid columns; on the tower
// in two tiles, the second beside the first and then turned; and on Γ
// with three risers.

/// "risers: no stroke is drawn through a riser, and none is hidden that the
/// walk up the line of sight sees".
#[test]
fn no_stroke_is_drawn_through_a_riser_and_none_is_hidden_that_the_walk_sees() {
    let window = domain(-0.5, 1.5, -1.0, 1.0);
    let dip = request(
        "1/(z - 0.49 - 0.1*i) + 1/(z - 0.49 + 0.1*i)",
        domain(-0.5, 1.5, -1.0, 1.3),
        60,
        banded(&[(0.5, 0.8)], 4.0),
    );
    let tower = request("1/(z - 0.45)", window, 60, banded(&[(0.5, 2.0)], 4.0));
    let gamma =
        request("gamma(z)", domain(-4.5, 4.5, -2.5, 2.5), 120, banded(&[(-2.5, 0.5), (-1.5, 1.0), (0.5, 1.5)], 2.5));
    let side = affine(1.0, 0.0, 2.0, 0.0, 1.0, 0.0);
    let turned = affine(0.0, -1.0, 2.5, 1.0, 0.0, -0.5);
    let cases: Vec<(&str, &LandscapeRequest, Vec<Affine2>)> = vec![
        ("the dip at 0.8 | 4", &dip, vec![Affine2::IDENTITY]),
        ("a tower at 2 | 4 in two tiles side by side", &tower, vec![Affine2::IDENTITY, side]),
        ("a tower at 2 | 4 in two tiles, the second turned", &tower, vec![Affine2::IDENTITY, turned]),
        ("Γ at 0.5 | 1 | 1.5 | 2.5", &gamma, vec![Affine2::IDENTITY]),
    ];
    let depth_tolerance = 0.02;
    let cameras = stress_cameras();
    for (name, r, tiles) in cases {
        let built = build(r, true);
        let magnitude = built.refine.as_ref().and_then(|r| r.magnitude.clone()).expect("a refiner");
        // On two tiles the walls stand round both copies, and every layer
        // is drawn on both.
        let mut manifest = built.manifest.clone();
        manifest.occluder.tiles = tiles.clone();
        if tiles.len() > 1 {
            manifest.occluder.walls = Walls::Perimeter {
                perimeter: perimeter(&[(-0.5, -1.0), (3.5, -1.0), (3.5, 1.0), (-0.5, 1.0)]),
                base: 0.0,
            };
            for spec in &mut manifest.layers {
                match &mut spec.source {
                    LayerSource::Contour { faces, .. } => *faces = FaceSelector::Graph { tiled: true },
                    LayerSource::Hatching { faces: faces @ FaceSelector::Plateaus { .. }, .. } => {
                        *faces = FaceSelector::Plateaus { tiled: true }
                    }
                    LayerSource::Hatching { faces: FaceSelector::Risers { tiled, .. }, .. } => *tiled = true,
                    LayerSource::TruncationEdge { tiled } => *tiled = true,
                    _ => {}
                }
            }
        }
        let bundle =
            KurvenBundle::described(built.url.clone(), manifest.clone(), built.surface.clone(), built.refine.clone());
        let h = Heightfield::new(
            bundle.surface.clone(),
            bundle.occluder(),
            tiles.clone(),
            manifest.occluder.region.clone(),
            manifest.occluder.step as usize,
            bundle.refine.clone(),
            manifest.occluder.walls.clone(),
        );
        let walk = SolidWalk::new(bundle.surface.clone(), magnitude, &tiles);
        let mut layers: Vec<(&Layer, FaceKind)> = vec![
            (bundle.layer("plateau_hatch").unwrap(), FaceKind::Plateau),
            (bundle.layer("wall_edges").unwrap(), FaceKind::Wall),
        ];
        layers.extend(
            bundle
                .layers
                .iter()
                .filter(|l| l.spec.role == LayerRole::Magnitude || l.spec.role == LayerRole::Phase)
                .map(|l| (l, FaceKind::Graph)),
        );
        let tallies: Vec<(usize, usize, usize, usize, f64, String, String)> = par_map(&cameras, |(camera, view)| {
            let d = -view.projection_direction();
            let (mut kept, mut under, mut dropped, mut seen, mut worst) = (0, 0, 0, 0, 0.0f64);
            let (mut under_example, mut seen_example) = (String::new(), String::new());
            for (layer, face) in &layers {
                let visible = h.visible_curves(&layer.paths, Some(*face), view, depth_tolerance, 0.0);
                // The curves' own vertices, not the ends of runs bisected
                // to where the verdict turns: a line through a riser's
                // corner at the domain's edge is in the solid on one side
                // of it and clear on the other.
                let kept_set: HashSet<(u64, u64, u64)> = visible.vertices.iter().map(|v| key3(v.v)).collect();
                let mut own: Vec<P3<WorldSpace>> = Vec::new();
                let mut own_keys: HashSet<(u64, u64, u64)> = HashSet::new();
                for &v in &layer.paths.vertices {
                    if own_keys.insert(key3(v.v)) {
                        own.push(v);
                    }
                }
                for &p in own.iter().filter(|v| kept_set.contains(&key3(v.v))) {
                    // A wall seen edge-on is left out: its line of sight
                    // runs along it, in and out of the solid by rounding.
                    let ts: Vec<f64> = walk.crossings(p, d).into_iter().filter(|&t| t >= 1e-7).collect();
                    if ts.is_empty() || walk.edge_on(p, d) {
                        continue;
                    }
                    let depth = walk.depth(p, d, &ts);
                    kept += 1;
                    if depth > worst {
                        worst = depth;
                    }
                    if depth > depth_tolerance {
                        under += 1;
                        if under_example.is_empty() {
                            under_example = format!(
                                "; e.g. {camera}, {}, ({:.4}, {:.4}, {:.4}) under the solid by {depth:.4}, kept",
                                layer.spec.name,
                                p.x(),
                                p.y(),
                                p.z()
                            );
                        }
                    }
                }
                // What is dropped and faces the eye: a plateau seen from
                // above, the graph facing the eye by a margin, off the
                // truncation edge, and a wall's top edge off the domain's
                // corners. Not a vertex on or a hair from a riser's line,
                // which the traversal and the walk each read from their
                // own side.
                for &p in own.iter().filter(|v| !kept_set.contains(&key3(v.v))) {
                    let all = walk.crossings(p, d);
                    if all.is_empty() || walk.beside_riser(p) {
                        continue;
                    }
                    let facing = match face {
                        FaceKind::Plateau => d.z > 1e-3,
                        FaceKind::Graph => walk.graph_facing(p, d).is_some_and(|f| f > 0.05),
                        _ => walk.top(Vec2::new(p.x(), p.y())).is_some_and(|top| {
                            (p.z() - top).abs() <= 1e-9 * swift::max(1.0, top.abs()) && !walk.at_corner(p)
                        }),
                    };
                    if !facing {
                        continue;
                    }
                    let depth = walk.depth(p, d, &all);
                    dropped += 1;
                    if depth <= 1e-6 {
                        seen += 1;
                        if seen_example.is_empty() {
                            seen_example = format!(
                                "; e.g. {camera}, {}, ({:.4}, {:.4}, {:.4}) in plain view, dropped",
                                layer.spec.name,
                                p.x(),
                                p.y(),
                                p.z()
                            );
                        }
                    }
                }
            }
            (kept, under, dropped, seen, worst, under_example, seen_example)
        });
        let kept: usize = tallies.iter().map(|t| t.0).sum();
        let under: usize = tallies.iter().map(|t| t.1).sum();
        let dropped: usize = tallies.iter().map(|t| t.2).sum();
        let seen: usize = tallies.iter().map(|t| t.3).sum();
        let worst = tallies.iter().map(|t| t.4).fold(0.0f64, |m, x| if x > m { x } else { m });
        let under_example = tallies.iter().map(|t| t.5.as_str()).find(|e| !e.is_empty()).unwrap_or("");
        let seen_example = tallies.iter().map(|t| t.6.as_str()).find(|e| !e.is_empty()).unwrap_or("");
        assert!(
            under == 0 && seen == 0 && kept > 1000 && dropped > 100,
            "{name}: what is drawn and what is dropped where a line of sight crosses a riser agree with the walk: \
             {under} of {kept} kept vertices under the solid by more than the depth tolerance, worst \
             {worst:.4}{under_example}; {seen} of {dropped} dropped on a face turned to the eye in plain \
             view{seen_example}"
        );
    }
}

// MARK: - curves on a plateau are tested by the plateau

/// "plateaus: the hatching on a plateau is tested by the plateau".
#[test]
fn the_hatching_on_a_plateau_is_tested_by_the_plateau() {
    let bundle = rgamma();
    let scene = Scene::from_bundle(&bundle, &bundle.manifest.presets[0]);
    let h = scene.heightfield().unwrap();
    let hatch = scene.layers().iter().find(|l| l.spec.name == "plateau_hatch").unwrap();
    assert!(
        scene.geometry().face_of(&hatch.spec) == Some(FaceKind::Plateau) && hatch.paths.count() > 0,
        "the plateau hatching lies on the plateaus: {} lines",
        hatch.paths.count()
    );
    let above = scene.camera.view;
    let on_plateaus = h.visible_curves(&hatch.paths, Some(FaceKind::Plateau), &above, scene.depth_tolerance, 0.0);
    let alone = h.visible_curves(&hatch.paths, None, &above, scene.depth_tolerance, 0.0);
    assert!(
        on_plateaus.count() > 0 && on_plateaus.vertices == alone.vertices && on_plateaus.offsets == alone.offsets,
        "seen from above, what the traversal alone keeps: {} runs",
        on_plateaus.count()
    );
    let below = test_camera(120.0, 30.0, 0.0);
    assert!(below.projection_direction().z > 0.0, "a camera under the landscape looks up");
    // With the traversal disarmed, on the grid, so that the facing alone
    // decides.
    let grid = h.on_grid();
    let from_below = grid.visible_curves(&hatch.paths, Some(FaceKind::Plateau), &below, f64::INFINITY, 0.0);
    let disarmed = grid.visible_curves(&hatch.paths, None, &below, f64::INFINITY, 0.0);
    assert!(
        from_below.count() == 0 && disarmed.count() > 0,
        "seen from below, nothing, by the facing alone: {} runs; the disarmed traversal alone keeps {}",
        from_below.count(),
        disarmed.count()
    );
}

// MARK: - a vertical stroke on a wall is its two ends
//
// A wall's hatch lines and its corners' vertical edges had a vertex every step
// of height, a vertex a cell up to |f|; untruncated, exp(10z) on a grid of 16
// put fourteen million on the walls, and exp(z²) and sin(30z) asked for more
// memory than there is. No vertex between the ends is needed: the solid is
// everything under the surface and every line of sight rises toward the eye,
// so what is hidden of a vertical stroke is one piece from its bottom up, and
// the bake bisects to where it ends.

/// "walls: a vertical stroke is its two ends, however high |f| stands".
#[test]
fn a_vertical_stroke_is_its_two_ends_however_high_the_modulus_stands() {
    // Untruncated, at sizes the subdivision every step survives, so that
    // this cannot itself run out of memory: hundreds of thousands of
    // vertices on the walls there, where two a stroke is dozens.
    for (expression, half, spacing) in [("exp(10*z)", 1.0, 0.5), ("exp(z^2)", 3.0, 1.5), ("sin(30*z)", 0.3, 0.15)] {
        let d = domain(-half, half, -half, half);
        let mut r = request(expression, d, 8, Truncation::None);
        r.spacing = Some(spacing);
        let bundle = build(&r, true);
        let g = &*bundle.surface.height;
        let cell = swift::max(d.real.length() / (g.width - 1) as f64, d.imag.length() / (g.height - 1) as f64);
        let highest = g.values.iter().map(|&v| f64::from(v)).fold(None, |m: Option<f64>, x| match m {
            Some(b) if !(b < x) => Some(b),
            _ => Some(x),
        });
        let highest = highest.unwrap_or(0.0);
        assert!(
            highest > 1000.0 * cell,
            "{expression}: the walls rise a thousand cells and more: max |f| {highest:.4e}, a cell {cell:.3}"
        );
        // The budget reads the number of strokes and nothing of |f|: two
        // vertices a hatch line, and two a corner's vertical edge.
        let hatch = &bundle.layers.iter().find(|l| l.spec.name == "wall_hatch").unwrap().paths;
        assert!(
            hatch.count() > 0 && hatch.vertices.len() <= 2 * hatch.count(),
            "{expression}: the wall hatch has two vertices a line: {} vertices on {} lines",
            hatch.vertices.len(),
            hatch.count()
        );
        let edges = &bundle.layers.iter().find(|l| l.spec.name == "wall_edges").unwrap().paths;
        let verticals: Vec<&[P3<WorldSpace>]> =
            edges.paths().filter(|p| p.iter().all(|v| v.x() == p[0].x() && v.y() == p[0].y())).collect();
        let corners: usize = verticals.iter().map(|p| p.len()).sum();
        assert!(
            verticals.len() == 4 && corners <= 2 * verticals.len(),
            "{expression}: and each corner's vertical edge two: {corners} vertices on {} vertical edges",
            verticals.len()
        );
    }
}
