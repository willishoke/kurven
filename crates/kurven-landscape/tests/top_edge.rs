//! `Sources/kurven-test/TopEdge.swift`: a wall's top edge between the
//! grid's nodes (the grid's chord without a refiner, f's own profile with
//! one, down to the zero the phase lines fan to), the folds run to the zero
//! and to the truncation edge, the banded truncation edge and the risers'
//! tops, the window's resolution, and the pit of 1/Γ drawn from f alone.
//! Every suite, with Swift's tolerances. None bakes: what a bake would draw
//! of a landscape is `Scene::visible_layers`, which these suites ask through
//! the heightfield's own tests, as Swift's do.

// Swift's `guard x > 0 else` refuses a NaN, and so does `if !(x > 0.0)`.
#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::manual_range_contains, clippy::type_complexity)]

mod common;

use common::build;
use kurven_core::bundle::KurvenBundle;
use kurven_core::hatching::FaceKind;
use kurven_core::hidden_line::HeightfieldVisibility;
use kurven_core::manifest::{Domain, Interval, RealBand, Truncation};
use kurven_core::navigation::{Framing, Viewport};
use kurven_core::occlusion::OcclusionTest;
use kurven_core::polyline::{Layer, PolylineSet};
use kurven_core::scene::{Heightfield, Scene};
use kurven_core::simd::{Vec2, Vec3};
use kurven_core::spaces::{P2, P3, WorldSpace};
use kurven_landscape::{Catalog, GridShape, LandscapeRequest, NativeLandscape};
use kurven_math::{Complex, real, swift};

const PI: f64 = core::f64::consts::PI;

fn request(expression: &str, domain: Domain, resolution: i64, truncation: Truncation) -> LandscapeRequest {
    let mut r = LandscapeRequest::new(expression, domain, resolution);
    r.truncation = Some(truncation);
    r
}

fn domain(r0: f64, r1: f64, i0: f64, i1: f64) -> Domain {
    Domain::new(Interval::new(r0, r1), Interval::new(i0, i1))
}

fn banded(bands: &[(f64, f64)], beyond: f64) -> Truncation {
    Truncation::RealBands { bands: bands.iter().map(|&(below, height)| RealBand { below, height }).collect(), beyond }
}

/// A scene's heightfield, as `Scene(bundle:preset:)` with the bundle's first
/// preset makes it.
fn scene(bundle: &KurvenBundle) -> Scene {
    Scene::from_bundle(bundle, &bundle.manifest.presets[0])
}

fn paths(set: &PolylineSet<WorldSpace>) -> Vec<Vec<P3<WorldSpace>>> {
    set.paths().map(|p| p.to_vec()).collect()
}

/// Swift's `Sequence.max()` on doubles: the running maximum replaced only
/// when it is less than the next.
fn seq_max(xs: impl IntoIterator<Item = f64>) -> Option<f64> {
    xs.into_iter().fold(None, |m, x| match m {
        Some(b) if !(b < x) => Some(b),
        _ => Some(x),
    })
}

fn seq_min(xs: impl IntoIterator<Item = f64>) -> Option<f64> {
    xs.into_iter().fold(None, |m, x| match m {
        Some(b) if !(x < b) => Some(b),
        _ => Some(x),
    })
}

/// Swift's `min(by: { $0.z < $1.z })`: the first of the lowest.
fn lowest(p: &[P3<WorldSpace>]) -> P3<WorldSpace> {
    let mut best = p[0];
    for &v in &p[1..] {
        if v.z() < best.z() {
            best = v;
        }
    }
    best
}

/// Whether a path's heights are not all one (Swift's `Set(p.map { $0.z }).count > 1`).
fn varies(p: &[P3<WorldSpace>]) -> bool {
    p.iter().any(|v| v.z() != p[0].z())
}

/// The longest stroke of a bundle's wall edges along Im z = `y` whose height
/// varies: that wall's top edge.
fn top_edge_at(bundle: &KurvenBundle, y: f64) -> Vec<P3<WorldSpace>> {
    let wall_edges = &bundle.layer("wall_edges").unwrap().paths;
    let mut best: Vec<P3<WorldSpace>> = Vec::new();
    for p in wall_edges.paths() {
        if p.iter().all(|v| (v.y() - y).abs() < 1e-9) && p.len() >= 2 && varies(p) && p.len() > best.len() {
            best = p.to_vec();
        }
    }
    best
}

/// The top edge of the front edge (y at the domain's low imaginary bound) of
/// a bundle's wall edges: the one stroke there whose height varies.
fn front_top_edge(bundle: &KurvenBundle) -> Vec<P3<WorldSpace>> {
    top_edge_at(bundle, bundle.manifest.domain.imag.lo)
}

fn nx(bundle: &KurvenBundle) -> usize {
    bundle.manifest.height.shape.1 as usize
}

fn ny(bundle: &KurvenBundle) -> usize {
    bundle.manifest.height.shape.0 as usize
}

/// "top edge to the zero: the refined top edge gains the minimum of |f|
/// between two nodes".
#[test]
fn the_refined_top_edge_gains_the_minimum_of_the_modulus_between_two_nodes() {
    // f(z) = z over Re [-1, 1], Im [0, 1] on an 8 x 8 grid: the zero at 0
    // lies on the front edge between the nodes at -1/7 and 1/7, where the
    // grid's chord bottoms out at 1/7 and |z| itself reaches the floor.
    let r = request("z", domain(-1.0, 1.0, 0.0, 1.0), 8, Truncation::Uniform(10.0));
    let plain = build(&r, false);
    let refined = build(&r, true);
    assert!(nx(&refined) == 8 && ny(&refined) == 8, "the grid is 8 x 8: {} x {}", nx(&refined), ny(&refined));

    let chord = front_top_edge(&plain);
    let top_edge = front_top_edge(&refined);
    let (chord_low, top_edge_low) = (lowest(&chord), lowest(&top_edge));
    assert!(
        (chord_low.z() - 1.0 / 7.0).abs() < 1e-6 && (chord_low.x().abs() - 1.0 / 7.0).abs() < 1e-9,
        "from the grid alone the top edge bottoms out at the node beside the zero: lowest ({:.4}, {:.4})",
        chord_low.x(),
        chord_low.z()
    );
    assert!(
        top_edge_low.z().abs() < 1e-9 && top_edge_low.x().abs() < 1e-9,
        "with a refiner it reaches the zero, at the zero: lowest ({:.2e}, {:.2e})",
        top_edge_low.x(),
        top_edge_low.z()
    );
    assert!(
        top_edge.len() > chord.len(),
        "vertices were added and none removed: {} -> {}",
        chord.len(),
        top_edge.len()
    );
    // The nodes are where they were, at f's height, which is the grid's
    // to the grid's rounding; |z| is straight either side of the zero, so
    // the zero is the one new vertex; and the vertices run along the
    // edge in order.
    let nodes_kept =
        chord.iter().all(|v| top_edge.iter().any(|t| (t.x() - v.x()).abs() < 1e-12 && (t.z() - v.z()).abs() < 1e-6));
    assert!(nodes_kept, "the grid's nodes are still vertices, at f's height there");
    assert!(
        top_edge.len() == chord.len() + 1,
        "and the zero is the one vertex added: |z| is straight either side: {} -> {}",
        chord.len(),
        top_edge.len()
    );
    let ordered = top_edge.windows(2).all(|w| w[0].x() < w[1].x()) || top_edge.windows(2).all(|w| w[0].x() > w[1].x());
    assert!(
        ordered,
        "the vertices run along the edge in order: {:?}",
        top_edge.iter().map(|v| format!("{:.3}", v.x())).collect::<Vec<_>>()
    );

    // The phase lines end at the zero, at the zero's height: the fan meets
    // the top edge on the floor rather than a cell's rise above it.
    let ends = |b: &KurvenBundle, name: &str| -> Vec<P3<WorldSpace>> {
        b.layer(name)
            .unwrap()
            .paths
            .paths()
            .flat_map(|p| [p[0], p[p.len() - 1]])
            .filter(|v| v.x().abs() < 1e-6 && v.y().abs() < 1e-6)
            .collect()
    };
    let mut fan = ends(&refined, "ang_major");
    fan.extend(ends(&refined, "ang_minor"));
    assert!(fan.len() >= 4, "the refined phase lines end on the zero: {} ends", fan.len());
    assert!(
        fan.iter().all(|v| v.z().abs() < 1e-9),
        "at height zero: highest {:.2e}",
        seq_max(fan.iter().map(|v| v.z())).unwrap_or(-1.0)
    );
    // The grid's own phase lines reach x = 0 too, by interpolating the
    // phase across the zero, and are lifted to the chord over it.
    let mut plain_fan = ends(&plain, "ang_major");
    plain_fan.extend(ends(&plain, "ang_minor"));
    assert!(
        !plain_fan.is_empty() && plain_fan.iter().all(|v| (v.z() - 1.0 / 7.0).abs() < 1e-6),
        "where the grid's own phase lines hang a cell's rise above the floor: {} ends at z = {:.4}",
        plain_fan.len(),
        plain_fan.first().map_or(-1.0, |v| v.z())
    );

    // The hatch of the front wall stands under the top edge as drawn: on the
    // V down to the zero, not on the chord over it.
    let tops = |b: &KurvenBundle| -> Vec<(f64, f64)> {
        b.layer("wall_hatch")
            .unwrap()
            .paths
            .paths()
            .filter(|p| p.iter().all(|v| v.y().abs() < 1e-9))
            .map(|p| (p[0].x(), seq_max(p.iter().map(|v| v.z())).unwrap()))
            .collect()
    };
    let on_top_edge = |tops: &[(f64, f64)], top_edge: &[P3<WorldSpace>]| -> f64 {
        seq_max(tops.iter().map(|&(tx, tz)| {
            let w = top_edge
                .windows(2)
                .find(|w| swift::min(w[0].x(), w[1].x()) - 1e-12 <= tx && tx <= swift::max(w[0].x(), w[1].x()) + 1e-12)
                .expect("a stroke stands under the top edge");
            let (a, b) = (w[0], w[1]);
            let u = if b.x() == a.x() { 0.0 } else { (tx - a.x()) / (b.x() - a.x()) };
            (tz - (a.z() + (b.z() - a.z()) * u)).abs()
        }))
        .unwrap_or(1.0)
    };
    let (refined_tops, plain_tops) = (tops(&refined), tops(&plain));
    let worst = on_top_edge(&refined_tops, &top_edge);
    assert!(
        !refined_tops.is_empty() && worst < 1e-9,
        "the front wall's hatch strokes rise to the refined top edge: {} strokes, worst {worst:.1e}",
        refined_tops.len()
    );
    let worst = on_top_edge(&plain_tops, &chord);
    assert!(worst < 1e-6, "and to the grid's without a refiner: worst {worst:.1e}");
    let in_the_v: Vec<&(f64, f64)> = refined_tops.iter().filter(|t| t.0.abs() < 1.0 / 7.0).collect();
    assert!(
        !in_the_v.is_empty() && in_the_v.iter().all(|t| t.1 < 1.0 / 7.0 - 1e-9),
        "so the strokes in the pit stop under the V, not at the chord: {} strokes",
        in_the_v.len()
    );
}

/// "top edge to the zero: the fold curves run down to the zero as well".
#[test]
fn the_fold_curves_run_down_to_the_zero_as_well() {
    // A steeper cone, 10|z|, so that its far graph turns away from the
    // oblique camera: its outlines are two generators running into the
    // apex, which the grid loses a cell short of the floor.
    let r = request("10*z", domain(-1.0, 1.0, 0.0, 1.0), 8, Truncation::Uniform(20.0));
    let plain = build(&r, false);
    let refined = build(&r, true);
    let ends = |b: &KurvenBundle| -> Vec<P3<WorldSpace>> {
        let s = scene(b);
        let folds = s.heightfield().unwrap().fold_curves(&s.camera.view, true);
        folds.paths().flat_map(|p| [p[0], p[p.len() - 1]]).collect()
    };
    let carried: Vec<P3<WorldSpace>> =
        ends(&refined).into_iter().filter(|v| v.x().abs() < 1e-9 && v.y().abs() < 1e-9).collect();
    assert!(
        !carried.is_empty() && carried.iter().all(|v| v.z().abs() < 1e-9),
        "with a refiner a fold ends at the zero, on the floor: {} ends there",
        carried.len()
    );
    let lowest = seq_min(ends(&plain).iter().map(|v| v.z())).unwrap_or(-1.0);
    assert!(lowest > 0.05, "which the grid's own folds stop short of: lowest end at z = {lowest:.3}");
}

/// "top edge to the zero: the truncation crossings stay the grid's, so the
/// top edge meets the truncation edge".
#[test]
fn the_truncation_crossings_stay_the_grids_so_the_top_edge_meets_the_truncation_edge() {
    // 1/z over Re [-1, 1], Im [0.1, 1.1] under a truncation of 2: the front edge
    // crosses the truncation twice, at x = ±sqrt(0.24). The truncation edge is contoured on
    // the grid, so the top edge's crossing is solved on the grid too, and the
    // two coincide; f's own crossing is a few hundredths to one side.
    let refined = build(&request("1/z", domain(-1.0, 1.0, 0.1, 1.1), 8, Truncation::Uniform(2.0)), true);
    let top_edge = front_top_edge(&refined);
    let truncation_edge = &refined.layer("truncation_edge").unwrap().paths;
    let mut truncation_edge_ends: Vec<P3<WorldSpace>> = Vec::new();
    for p in truncation_edge.paths() {
        for end in [p[0], p[p.len() - 1]] {
            if (end.y() - 0.1).abs() < 1e-9 {
                truncation_edge_ends.push(end);
            }
        }
    }
    assert!(
        truncation_edge_ends.len() == 2,
        "the truncation edge ends twice on the front edge: {}",
        truncation_edge_ends.len()
    );
    let mut missed = 0.0f64;
    for end in &truncation_edge_ends {
        missed = swift::max(missed, seq_min(top_edge.iter().map(|v| (v.v - end.v).length())).unwrap_or(f64::INFINITY));
    }
    assert!(missed < 1e-9, "and the top edge has a vertex at each end: worst gap {missed:.1e}");
    let true_crossing = 0.24f64.sqrt();
    let off_f = seq_max(truncation_edge_ends.iter().map(|v| (v.x().abs() - true_crossing).abs())).unwrap();
    assert!(off_f < 1e-9, "which is f's crossing, |1/z| = 2 at x = ±√0.24, to rounding: {off_f:.1e} from f's");
    // 1/|z| is curved along the edge, so between the nodes and the
    // crossings the top edge is subdivided: every vertex lies on f's
    // profile, and no chord's midpoint departs from it by more than the
    // refiner's tolerance of a cell.
    let nodes = nx(&refined);
    assert!(
        top_edge.len() > nodes + 2,
        "the top edge is the nodes, the two crossings, and f's profile between: {} vertices",
        top_edge.len()
    );
    let profile = |x: f64| swift::min(1.0 / (x * x + 0.01).sqrt(), 2.0);
    let off_profile = seq_max(top_edge.iter().map(|v| (v.z() - profile(v.x())).abs())).unwrap();
    assert!(off_profile < 1e-6, "every vertex is on f's profile, truncated: worst {off_profile:.1e}");
    let cell = 2.0 / (nodes - 1) as f64;
    let sag = seq_max(
        top_edge.windows(2).map(|w| (0.5 * (w[0].z() + w[1].z()) - profile(0.5 * (w[0].x() + w[1].x()))).abs()),
    )
    .unwrap();
    assert!(
        sag <= 0.02 * cell + 1e-12,
        "and no chord departs from it by more than 0.02 of a cell: worst sag {sag:.2e}, cell {cell:.3}"
    );
    assert!(top_edge.iter().all(|v| v.z() <= 2.0 + 1e-12), "and none rises above the truncation");
}

/// "rings the grid cannot see: the pit carries the same levels at 24, 48
/// and 96 samples".
#[test]
fn rings_the_grid_cannot_see_the_pit_carries_the_same_levels_at_every_grid() {
    // A ring at level L round the zero at -3 is a loop of radius L/6.
    // Marching squares draws a loop only if a node lies inside it, and
    // no node is at the zero, so every level below the nearest node's
    // value went missing, and which levels those were depended on the
    // grid. Seeded from the zero, every requested level is drawn at
    // every grid, each ring on its level, each cut to the window on
    // the front edge where the zero sits.
    let d = domain(-3.8, -2.2, 0.0, 0.8);
    let mut levels_seen: Vec<Vec<i64>> = Vec::new();
    for res in [24i64, 48, 96] {
        let bundle = build(&request("1/gamma(z)", d, res, Truncation::Uniform(2.0)), true);
        let refiner =
            NativeLandscape::refiner("1/gamma(z)", d, GridShape { n_real: res as usize, n_imag: res as usize })
                .unwrap();
        let mut rings: Vec<(f64, Vec<P3<WorldSpace>>)> = Vec::new();
        for name in ["mag_major", "mag_minor"] {
            for p in bundle.layer(name).unwrap().paths.paths() {
                let xs = p.iter().map(|v| v.x());
                if !(seq_min(xs.clone()).unwrap() < -3.0
                    && seq_max(xs).unwrap() > -3.0
                    && p.iter().all(|v| ((v.x() + 3.0) * (v.x() + 3.0) + v.y() * v.y()).sqrt() < 0.3))
                {
                    continue;
                }
                rings.push((p[0].z(), p.to_vec()));
            }
        }
        // Swift's stable sort by `<`.
        rings.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let (mut off_level, mut off_edge) = (0.0f64, 0.0f64);
        for (level, path) in &rings {
            for v in path {
                off_level = swift::max(off_level, (refiner.magnitude(Complex::new(v.x(), v.y())) - level).abs());
            }
            off_edge = swift::max(off_edge, swift::max(path[0].y().abs(), path[path.len() - 1].y().abs()));
        }
        assert!(
            off_level < 1e-6,
            "{res}: every vertex of every ring in the pit is on its level: {} rings, worst {off_level:.1e}",
            rings.len()
        );
        assert!(
            off_edge < 1e-9,
            "{res}: and each ring ends on the front edge, where the zero is: worst {off_edge:.1e}"
        );
        let cents: Vec<i64> = rings.iter().map(|r| (r.0 * 100.0).round() as i64).collect();
        let distinct: std::collections::HashSet<i64> = cents.iter().copied().collect();
        assert!(
            distinct.len() == rings.len(),
            "{res}: and no level is drawn twice: {} rings, {} levels",
            rings.len(),
            distinct.len()
        );
        levels_seen.push(cents);
    }
    let (l24, l48, l96) = (&levels_seen[0], &levels_seen[1], &levels_seen[2]);
    assert!(
        l24 == l48 && l48 == l96 && l24.first() == Some(&4),
        "the same levels at every grid, down to the lowest asked for: 24: {l24:?}  48: {l48:?}  96: {l96:?}"
    );
}

/// "folds to the truncation edge: a tower's outline runs up to the truncation".
#[test]
fn folds_to_the_truncation_edge_a_towers_outline_runs_up_to_the_truncation() {
    // 1/z truncated at 2: a tower round the pole, its truncation edge the circle
    // |z| = 1/2. The fold either side of the tower is its outline,
    // and it meets the truncation edge where the truncation edge's tangent lies along the sight
    // line. Traced on the grid the fold's last vertex sat on the
    // crease at the truncation height and was dropped as truncation curves, so the
    // outline stopped a cell short of the truncation edge.
    let bundle = build(&request("1/z", domain(-1.0, 1.0, -1.0, 1.0), 24, Truncation::Uniform(2.0)), true);
    let s = scene(&bundle);
    let h = s.heightfield().unwrap();
    let folds = h.fold_curves(&s.camera.view, true);
    let (mut at_truncation_edge, mut on_plateau, mut worst_off_truncation_edge) = (0, 0, 0.0f64);
    for p in folds.paths() {
        on_plateau += p.iter().filter(|v| v.z() >= 2.0 - 1e-9).count();
        for end in [p[0], p[p.len() - 1]] {
            if end.z() >= 2.0 - 1e-9 {
                at_truncation_edge += 1;
                worst_off_truncation_edge =
                    swift::max(worst_off_truncation_edge, ((end.x() * end.x() + end.y() * end.y()).sqrt() - 0.5).abs());
            }
        }
    }
    assert!(
        at_truncation_edge >= 2,
        "the outline reaches the truncation edge at both sides of the tower: {at_truncation_edge} fold ends at the \
         truncation height"
    );
    // To the step the facing is read at, a ten-thousandth of a cell:
    // across the crease the facing jumps, and the bisection lands where
    // the differenced facing crosses zero, within that step of it.
    assert!(
        worst_off_truncation_edge < 1e-4 * (2.0 / 23.0),
        "and ends on the truncation edge, |z| = 1/2, to the facing's step: worst {worst_off_truncation_edge:.1e} off"
    );
    assert!(
        on_plateau == at_truncation_edge,
        "and nothing of a fold lies on the plateau but those ends: {on_plateau} vertices at the truncation height, \
         {at_truncation_edge} of them ends"
    );
    let seen = h.visible_folds(&s.camera.view, s.depth_tolerance, true, 0.0);
    let seen_at_truncation_edge =
        seen.paths().filter(|p| p[0].z() >= 2.0 - 1e-9 || p[p.len() - 1].z() >= 2.0 - 1e-9).count();
    assert!(
        seen_at_truncation_edge >= 2,
        "and both are seen up to the truncation edge: {seen_at_truncation_edge} seen runs end on the truncation edge"
    );
}

/// "folds arrive along the fold: a tower's outline meets the truncation edge
/// where the graph turns edge-on".
#[test]
fn folds_arrive_along_the_fold_where_the_graph_turns_edge_on() {
    // (z - 1)/(z + 1) over Re [-2.1, 2.1], Im [0, 1.4], truncated at 3, on a
    // grid of 24: the pole at -1 sits on the front edge, and the tower's
    // right graph carries an outline up to the truncation edge. Traced on the
    // truncated surface the fold's last vertex was where a grid edge
    // crossed the crease, a cell from where the fold meets the truncation edge, and
    // the straight chord joining them cut across the truncation's projection.
    // Traced on the graph the fold runs on below the truncation and is ended
    // at its own crossing of |f| = h(x), so it arrives along itself.
    let bundle = build(&request("(z-1)/(z+1)", domain(-2.1, 2.1, 0.0, 1.4), 24, Truncation::Uniform(3.0)), true);
    let s = scene(&bundle);
    let h = s.heightfield().unwrap();
    let cell = 4.2 / 23.0;
    let f = |x: f64, y: f64| ((x - 1.0) * (x - 1.0) + y * y).sqrt() / ((x + 1.0) * (x + 1.0) + y * y).sqrt();
    // The line of sight, as the kernel of the oblique projection.
    let o = s.camera.page_coordinates(P3::new(0.0, 0.0, 0.0));
    let cols = [P3::<WorldSpace>::new(1.0, 0.0, 0.0), P3::new(0.0, 1.0, 0.0), P3::new(0.0, 0.0, 1.0)]
        .map(|p| s.camera.page_coordinates(p));
    let row0 = Vec3::new(cols[0].x() - o.x(), cols[1].x() - o.x(), cols[2].x() - o.x());
    let row1 = Vec3::new(cols[0].y() - o.y(), cols[1].y() - o.y(), cols[2].y() - o.y());
    let sight = row0.cross(row1).normalized();
    // The graph's facing: f's own slope, untruncated.
    let graph_facing = |x: f64, y: f64| -> f64 {
        let e = 1e-6;
        let n = Vec3::new(-(f(x + e, y) - f(x - e, y)) / (2.0 * e), -(f(x, y + e) - f(x, y - e)) / (2.0 * e), 1.0);
        -n.dot(sight) / n.length()
    };
    let folds = h.fold_curves(&s.camera.view, true);
    let (mut truncation_edge_ends, mut worst_facing, mut worst_off_truncation_edge) = (0, 0.0f64, 0.0f64);
    let (mut under_truncation, mut longest_last_chord, mut off_fold) = (0, 0.0f64, 0.0f64);
    for p in folds.paths() {
        for v in p {
            if v.z() >= 3.0 - 1e-9 {
                truncation_edge_ends += 1;
                worst_facing = swift::max(worst_facing, graph_facing(v.x(), v.y()).abs());
                worst_off_truncation_edge = swift::max(worst_off_truncation_edge, (f(v.x(), v.y()) - 3.0).abs());
            } else {
                if f(v.x(), v.y()) > 3.0 {
                    under_truncation += 1;
                }
                off_fold = swift::max(off_fold, graph_facing(v.x(), v.y()).abs());
            }
        }
        for w in p.windows(2) {
            if w[0].z() >= 3.0 - 1e-9 || w[1].z() >= 3.0 - 1e-9 {
                longest_last_chord = swift::max(longest_last_chord, (w[0].v - w[1].v).length());
            }
        }
    }
    assert!(
        truncation_edge_ends >= 1,
        "a fold ends on the truncation edge: {truncation_edge_ends} truncation edge ends"
    );
    assert!(worst_off_truncation_edge < 1e-9, "on |f| = 3 to the last bit: worst {worst_off_truncation_edge:.1e} off");
    assert!(
        worst_facing < 1e-6,
        "where the graph turns edge-on, not where a grid edge crosses the crease: graph facing {worst_facing:.1e} at a \
         truncation edge end"
    );
    assert!(off_fold < 1e-6, "and every graph vertex is on the fold: facing {off_fold:.1e}");
    assert!(under_truncation == 0, "no fold vertex lies below the truncation: {under_truncation} do");
    assert!(
        longest_last_chord < 0.25 * cell,
        "the last chord to the truncation edge is a subdivided piece of the fold, not a cell-long jump: {:.3} cells",
        longest_last_chord / cell
    );
    let seen = h.visible_folds(&s.camera.view, s.depth_tolerance, true, 0.0);
    let reaches = seen.paths().any(|p| {
        let (first, last) = (p[0].z(), p[p.len() - 1].z());
        swift::max(first, last) >= 3.0 - 1e-9 && swift::min(first, last) < 2.2
    });
    assert!(reaches, "and the outline is seen in one piece from the graph up to the truncation edge");
    // The truncation's hatch is ruled on f: no stroke reaches past the truncation edge.
    let hatch = &bundle.layer("plateau_hatch").unwrap().paths;
    let (mut past, mut ends_off, mut strokes) = (0, 0.0f64, 0);
    for p in hatch.paths() {
        strokes += 1;
        past += p.iter().filter(|v| f(v.x(), v.y()) < 3.0 - 1e-9).count();
        for end in [p[0], p[p.len() - 1]] {
            if end.x().abs() < 2.1 - 1e-9 && end.y() > 1e-9 && end.y() < 1.4 - 1e-9 {
                ends_off = swift::max(ends_off, (f(end.x(), end.y()) - 3.0).abs());
            }
        }
    }
    assert!(
        strokes > 0 && past == 0,
        "no plateau hatching vertex lies outside the truncation edge drawn on f: {past} of {strokes} strokes' vertices"
    );
    assert!(ends_off < 1e-9, "and every stroke ends on |f| = 3: worst {ends_off:.1e} off");
}

/// "a top edge comes out from behind a tower where the traversal turns, not
/// at the next node".
#[test]
fn a_top_edge_comes_out_from_behind_a_tower_where_the_traversal_turns() {
    // tan(z) over Re [-0.7, 2.3], Im [0, 1], truncated at 2.5, on a grid of
    // 24: the tower at pi/2 hides a stretch of the back edge's top edge.
    // The top edge is concave there, so the chord between two of its
    // vertices lies under it, in the back wall, which faces away: a
    // run's end bisected along the chord found nothing admitted and the
    // top edge began at the next node, half a cell past where it comes
    // into view beside the outline's cusp.
    let bundle = build(&request("tan(z)", domain(-0.7, 2.3, 0.0, 1.0), 24, Truncation::Uniform(2.5)), true);
    let s = scene(&bundle);
    let h = s.heightfield().unwrap();
    let view = s.camera.view;
    let cell = 3.0 / 23.0;
    let top_edge = &bundle.layer("wall_edges").unwrap().paths;
    let drawn = h.visible_wall_curves(top_edge, &view, s.depth_tolerance, 0.0);
    // the run of the back top edge that begins right of the tower
    let mut start: Option<P3<WorldSpace>> = None;
    for p in drawn.paths() {
        if !(p.iter().all(|v| (v.y() - 1.0).abs() < 1e-9) && p[0].z() > 0.01) {
            continue;
        }
        let mut left = p[0];
        for &v in &p[1..] {
            if v.x() < left.x() {
                left = v;
            }
        }
        if left.x() > 1.5 && left.x() < 2.0 {
            start = Some(left);
        }
    }
    let s0 = start.expect("the back top edge's run right of the tower is drawn");
    let vis = HeightfieldVisibility::new(h.clone(), &view, s.depth_tolerance);
    let at = |x: f64| P3::<WorldSpace>::new(x, 1.0, h.surface_height(P2::new(x, 1.0)));
    let node = (s0.x() + 0.7) / cell;
    assert!((node - node.round()).abs() > 1e-3, "and begins between nodes: at x = {:.4}, node {node:.3}", s0.x());
    let (before, after) = (vis.is_clear(at(s0.x() - 1e-4 * cell)), vis.is_clear(at(s0.x() + 1e-4 * cell)));
    assert!(
        !before && after,
        "where the top edge comes into view, to a ten-thousandth of a cell: x = {:.5}: hidden before {}, clear after {}",
        s0.x(),
        if before { "no" } else { "yes" },
        if after { "yes" } else { "no" }
    );
    let under = h.surface_height(P2::new(s0.x(), s0.y())) - s0.z();
    assert!(under.abs() < 1e-9, "on the top edge: {under:.2e} under f");
}

/// "a chord the refiner cannot place is not drawn".
#[test]
fn a_chord_the_refiner_cannot_place_is_not_drawn() {
    // exp(1/z) over Re [-1.1, 0.4], Im [0, 0.75], truncated at 3, on a grid
    // of 24: an essential singularity at the origin, on the front edge.
    // The circles of one phase level crowd into it, so marching squares
    // joins vertices of different circles, and the phase line along the
    // real axis jumps from the floor to the truncation at the origin. Those
    // chords used to be drawn as they were: hooks and zigzags beside the
    // origin, and a line up the cliff.
    let bundle = build(&request("exp(1/z)", domain(-1.1, 0.4, 0.0, 0.75), 24, Truncation::Uniform(3.0)), true);
    let cell = 1.5 / 23.0;
    let f = |x: f64, y: f64| Complex::exp(1.0 / Complex::new(x, y));
    let ph = |x: f64, y: f64| f(x, y).argument();
    let wrap = |mut r: f64| {
        while r > PI {
            r -= 2.0 * PI;
        }
        while r < -PI {
            r += 2.0 * PI;
        }
        r
    };
    let (mut chords, mut strayed, mut jumps, mut worst_stray) = (0, 0, 0, 0.0f64);
    let mut examples: Vec<String> = Vec::new();
    let layers: Vec<&Layer> = bundle.manifest.layers.iter().zip(bundle.layers()).map(|(_, l)| l).collect();
    let named = |prefix: &str| -> Vec<&Layer> {
        bundle
            .manifest
            .layers
            .iter()
            .zip(&layers)
            .filter(|(spec, _)| spec.name.starts_with(prefix))
            .map(|(_, l)| *l)
            .collect()
    };
    for layer in named("ang") {
        for p in layer.paths.paths() {
            for w in p.windows(2) {
                let (a, b) = (w[0], w[1]);
                chords += 1;
                // the chord's departure from the level set at its quarter
                // points: the phase residual against the level its far end
                // sits on, over the gradient of the phase there, in cells
                let level = f(b.x(), b.y()).argument();
                for t in [0.25, 0.5, 0.75] {
                    let q = (a.x() + t * (b.x() - a.x()), a.y() + t * (b.y() - a.y()));
                    let r = wrap(f(q.0, q.1).argument() - level);
                    let e = 1e-5 * cell;
                    let dx = wrap(ph(q.0 + e, q.1) - ph(q.0 - e, q.1));
                    let dy = wrap(ph(q.0, q.1 + e) - ph(q.0, q.1 - e));
                    let slope = (dx * dx + dy * dy).sqrt() / (2.0 * e);
                    let off = if slope > 0.0 && slope.is_finite() { r.abs() / slope / cell } else { r.abs() };
                    if off > 0.021 {
                        strayed += 1;
                        if examples.len() < 4 {
                            examples.push(format!(
                                "({:.5}, {:.5}, {:.3})->({:.5}, {:.5}, {:.3}) t {t:.2} off {off:.3} cells r {r:.2}",
                                a.x(),
                                a.y(),
                                a.z(),
                                b.x(),
                                b.y(),
                                b.z()
                            ));
                        }
                    }
                    worst_stray = swift::max(worst_stray, off);
                }
                // no chord climbs the cliff: a jump in height of more than a cell's worth of slope
                if (a.z() - b.z()).abs() > 1.5 {
                    jumps += 1;
                    examples.push(format!(
                        "jump ({:.5}, {:.5}, {:.3})->({:.5}, {:.5}, {:.3})",
                        a.x(),
                        a.y(),
                        a.z(),
                        b.x(),
                        b.y(),
                        b.z()
                    ));
                }
            }
        }
    }
    assert!(chords > 100, "phase chords are drawn beside the singularity: {chords} chords");
    assert!(
        strayed == 0,
        "and every one of them lies on the level set to the tolerance at its quarter points: {strayed} quarter points \
         off, worst {worst_stray:.3} cells  {}",
        examples.join("; ")
    );
    assert!(
        jumps == 0,
        "and none climbs the cliff at the origin: {jumps} chords rise by more than 1.5  {}",
        examples.join("; ")
    );
    // the rings all pass through the origin and are placed: every ring vertex is on its level
    let (mut ring_vertices, mut off) = (0, 0);
    for layer in named("mag") {
        for v in &layer.paths.vertices {
            if v.x() * v.x() + v.y() * v.y() > 1e-12 {
                ring_vertices += 1;
                if (f(v.x(), v.y()).magnitude() - v.z()).abs() > 1e-6 * swift::max(1.0, v.z()) {
                    off += 1;
                }
            }
        }
    }
    assert!(
        ring_vertices > 100 && off == 0,
        "every ring vertex off the origin is on its level: {off} of {ring_vertices} off"
    );
    // The phase lines are followed on f to the origin: marching squares
    // ends them a cell short, where it can no longer bracket a phase
    // that turns many times within a cell; from there each is traced
    // along its circle until f's phase runs out, which is where exp(1/z)
    // underflows a hair from the origin.
    let (mut interior_ends, mut far_ends, mut worst_end) = (0, 0, 0.0f64);
    for layer in named("ang") {
        for p in layer.paths.paths() {
            for e in [p[0], p[p.len() - 1]] {
                let on_edge = (e.x() + 1.1).abs() < 1e-6
                    || (e.x() - 0.4).abs() < 1e-6
                    || e.y().abs() < 1e-6
                    || (e.y() - 0.75).abs() < 1e-6;
                let on_truncation_edge = f(e.x(), e.y()).magnitude() >= 3.0 - 1e-3;
                if on_edge || on_truncation_edge || !(e.x() < 0.1) || !(e.y() < 0.2) {
                    continue;
                }
                interior_ends += 1;
                let d = (e.x() * e.x() + e.y() * e.y()).sqrt();
                worst_end = swift::max(worst_end, d);
                if d > 0.1 * cell {
                    far_ends += 1;
                }
            }
        }
    }
    assert!(
        interior_ends >= 10 && far_ends == 0,
        "and the phase lines beside the origin run to within a tenth of a cell of it: {interior_ends} ends beside \
         it, {far_ends} further off, worst {:.4} cells",
        worst_end / cell
    );
}

/// "banded truncation edge: a tower cut at two heights, its truncation edge
/// on f band by band, with the riser's top".
#[test]
fn banded_truncation_edge_a_tower_cut_at_two_heights() {
    // 1/(z - 1/2) over Re [-1/2, 3/2], Im [-1, 1]: a tower at 1/2, truncated
    // at 2 for Re z < 1/2 and at 4 beyond. The truncation edge is the half circle
    // |z - 1/2| = 1/2 on the left and |z - 1/2| = 1/4 on the right, and
    // between them, along Re z = 1/2, the top of the riser: 2 at |Im z|
    // = 1/2, rising with |f| = 1/|Im z| to 4 at |Im z| = 1/4, flat at 4
    // between.
    let bundle = build(&request("1/(z - 0.5)", domain(-0.5, 1.5, -1.0, 1.0), 32, banded(&[(0.5, 2.0)], 4.0)), true);
    let truncation_edge = paths(&bundle.layer("truncation_edge").unwrap().paths);
    let f = |v: P3<WorldSpace>| 1.0 / ((v.x() - 0.5) * (v.x() - 0.5) + v.y() * v.y()).sqrt();
    let (mut left_off, mut right_off, mut riser_off) = (0.0f64, 0.0f64, 0.0f64);
    let (mut left, mut right, mut risers) = (0, 0, 0);
    let (mut left_ends, mut right_ends, mut riser_ends): (Vec<f64>, Vec<f64>, Vec<f64>) = (vec![], vec![], vec![]);
    let mut bottoms: Vec<&Vec<P3<WorldSpace>>> = Vec::new();
    for p in &truncation_edge {
        let on_riser = p.iter().all(|v| (v.x() - 0.5).abs() < 1e-12);
        // The riser's bottom stands a hair onto the lower side, a
        // billionth of a cell.
        if p.iter().all(|v| v.x() < 0.5 && 0.5 - v.x() < 1e-9) {
            bottoms.push(p);
        } else if on_riser {
            risers += 1;
            for &v in p {
                riser_off = swift::max(riser_off, (v.z() - swift::min(f(v), 4.0)).abs());
            }
            riser_ends.extend([p[0].y(), p[p.len() - 1].y()]);
            // (Swift measures the chords' sag below; its loop here adds
            // nothing, a product with zero.)
        } else if p.iter().all(|v| v.x() <= 0.5 + 1e-12) {
            left += 1;
            for &v in p {
                left_off = swift::max(left_off, swift::max((f(v) - 2.0).abs(), (v.z() - 2.0).abs()));
            }
            for e in [p[0], p[p.len() - 1]] {
                if (e.x() - 0.5).abs() < 1e-12 {
                    left_ends.push(e.y());
                }
            }
        } else {
            right += 1;
            for &v in p {
                right_off = swift::max(right_off, swift::max((f(v) - 4.0).abs(), (v.z() - 4.0).abs()));
            }
            for e in [p[0], p[p.len() - 1]] {
                if (e.x() - 0.5).abs() < 1e-12 {
                    right_ends.push(e.y());
                }
            }
        }
    }
    assert!(
        left >= 1 && left_off < 1e-9,
        "left of the boundary the truncation edge is |f| = 2 at height 2: {left} pieces, worst {left_off:.1e}"
    );
    assert!(
        right >= 1 && right_off < 1e-9,
        "beyond it the truncation edge is |f| = 4 at height 4: {right} pieces, worst {right_off:.1e}"
    );
    let le = seq_max(left_ends.iter().map(|y| (y.abs() - 0.5).abs())).unwrap_or(f64::INFINITY);
    let re = seq_max(right_ends.iter().map(|y| (y.abs() - 0.25).abs())).unwrap_or(f64::INFINITY);
    assert!(
        left_ends.len() == 2 && le < 1e-9,
        "the left truncation edge ends on the boundary at Im z = ±1/2: {} ends, worst {le:.1e}",
        left_ends.len()
    );
    assert!(
        right_ends.len() == 2 && re < 1e-9,
        "the right truncation edge ends on the boundary at Im z = ±1/4: {} ends, worst {re:.1e}",
        right_ends.len()
    );
    let se = seq_max(riser_ends.iter().map(|y| (y.abs() - 0.5).abs())).unwrap_or(f64::INFINITY);
    assert!(
        risers == 1 && se < 1e-9 && riser_off < 1e-9,
        "and the riser's top runs along the boundary from one left end to the other, at min(|f|, 4): {risers} \
         pieces, ends off by {se:.1e}, heights off by {riser_off:.1e}"
    );
    let bottom_ends =
        seq_max(bottoms.iter().flat_map(|b| [b[0].y(), b[b.len() - 1].y()]).map(|y| (y.abs() - 0.5).abs()))
            .unwrap_or(f64::INFINITY);
    assert!(
        bottoms.len() == 1 && bottom_ends < 1e-9 && bottoms.iter().all(|b| b.iter().all(|v| v.z() == 2.0)),
        "and its bottom under it at 2, on the lower band's side, from one end to the other: {} pieces, ends off by \
         {bottom_ends:.1e}",
        bottoms.len()
    );
    let mut sag = 0.0f64;
    for p in &truncation_edge {
        if !p.iter().all(|v| (v.x() - 0.5).abs() < 1e-12) {
            continue;
        }
        for w in p.windows(2) {
            let ym = 0.5 * (w[0].y() + w[1].y());
            sag = swift::max(sag, (swift::min(1.0 / ym.abs(), 4.0) - 0.5 * (w[0].z() + w[1].z())).abs());
        }
    }
    let cell = 2.0 / 31.0;
    assert!(
        sag <= 0.02 * cell + 1e-12,
        "and is a curve to the refiner's tolerance: worst sag {sag:.2e} against {:.2e}",
        0.02 * cell
    );
    // The top edge of the front edge (Im z = -1) meets the riser at Re z = 1/2: two
    // vertices there, at each band's height, |f| = 1 being under both.
    let top_edge = top_edge_at(&bundle, -1.0);
    let at_riser: Vec<&P3<WorldSpace>> = top_edge.iter().filter(|v| (v.x() - 0.5).abs() < 1e-9).collect();
    assert!(
        at_riser.len() == 2 && at_riser.iter().all(|v| (v.z() - 1.0).abs() < 1e-9),
        "the front top edge has two vertices on the riser at Re z = 1/2, both at |f| = 1 there: {} vertices, heights \
         {:?}",
        at_riser.len(),
        at_riser.iter().map(|v| format!("{:.4}", v.z())).collect::<Vec<_>>()
    );
    assert!(
        top_edge.iter().all(|&v| v.z() <= swift::min(f(v), if v.x() < 0.5 { 2.0 } else { 4.0 }) + 1e-9),
        "and none rises above its band's truncation"
    );
}

/// "banded truncation edge: a wall's top edge steps at a riser between the
/// band heights, and no higher".
#[test]
fn banded_truncation_edge_a_walls_top_edge_steps_at_a_riser_and_no_higher() {
    // 1/(z - 0.45) over Re [-1/2, 3/2], Im [-1, 1], truncated at 1/2 for
    // Re z < 1/2 and at 4 beyond. Where the front and back walls meet the
    // boundary |f| = 1/|0.05 ± i| ≈ 0.999 lies between the two, so the
    // top edge steps there from 1/2 to |f|: the truncation, not |f|,
    // changes across the step, and the top edge reaches neither band's
    // truncation on it.
    let bundle = build(&request("1/(z - 0.45)", domain(-0.5, 1.5, -1.0, 1.0), 32, banded(&[(0.5, 0.5)], 4.0)), true);
    let f = |v: P3<WorldSpace>| 1.0 / ((v.x() - 0.45) * (v.x() - 0.45) + v.y() * v.y()).sqrt();
    let top = 1.0 / (0.05f64 * 0.05 + 1.0).sqrt();
    for y in [-1.0, 1.0] {
        let top_edge = top_edge_at(&bundle, y);
        let at_riser: Vec<&P3<WorldSpace>> = top_edge.iter().filter(|v| (v.x() - 0.5).abs() < 1e-9).collect();
        assert!(
            at_riser.len() == 2
                && at_riser.iter().any(|v| v.x() < 0.5)
                && at_riser.iter().any(|v| v.x() >= 0.5)
                && at_riser
                    .iter()
                    .all(|&&v| (v.z() - swift::min(f(v), if v.x() < 0.5 { 0.5 } else { 4.0 })).abs() < 1e-9),
            "at Im z = {y} the top edge has two vertices on the riser, at 1/2 and at |f| there: {} vertices, heights \
             {:?}",
            at_riser.len(),
            at_riser.iter().map(|v| format!("{:.4}", v.z())).collect::<Vec<_>>()
        );
        let highest = seq_max(top_edge.iter().map(|v| v.z())).unwrap_or(f64::INFINITY);
        assert!(
            highest <= top + 1e-9,
            "and none above the higher side's top there, |f| = {top:.4}: highest {highest:.4}"
        );
    }
}

/// "banded truncation edge: a riser's top that reaches the domain's edge is
/// drawn to it".
#[test]
fn banded_truncation_edge_a_risers_top_that_reaches_the_domains_edge_is_drawn_to_it() {
    // The tower 1/(z - 1/2) at 2 | 4 again, over Im [-0.3, 1]: the band
    // at 2 holds the boundary Re z = 1/2 from Im z = -1/2, off the
    // domain, to 1/2, so the riser's top runs from the domain's edge at
    // Im z = -0.3, where its truncation edge has no end, to Im z = 1/2.
    let bundle = build(&request("1/(z - 0.5)", domain(-0.5, 1.5, -0.3, 1.0), 32, banded(&[(0.5, 2.0)], 4.0)), true);
    let tops: Vec<Vec<P3<WorldSpace>>> = paths(&bundle.layer("truncation_edge").unwrap().paths)
        .into_iter()
        .filter(|p| p.iter().all(|v| v.x() == 0.5))
        .collect();
    let off = seq_max(tops.iter().flatten().map(|v| (v.z() - swift::min(1.0 / v.y().abs(), 4.0)).abs()))
        .unwrap_or(f64::INFINITY);
    assert!(
        tops.len() == 1
            && tops[0].first().map(|v| v.y()) == Some(-0.3)
            && (tops[0].last().map_or(0.0, |v| v.y()) - 0.5).abs() < 1e-9
            && off < 1e-9,
        "the riser's top runs from Im z = -0.3 to 1/2 along Re z = 1/2, at min(|f|, 4): {} tops, {:?}, heights off by \
         {off}",
        tops.len(),
        tops.iter().map(|t| format!("from {} to {}", t[0].y(), t[t.len() - 1].y())).collect::<Vec<_>>()
    );
}

/// "banded truncation edge: six risers, one on each boundary, topped in
/// order along Re z".
#[test]
fn banded_truncation_edge_six_risers_topped_in_order_along_re_z() {
    // 1/sin(πz) over Re [-1/2, 13/2], its poles at the integers, banded at
    // 2 and 4 by turns with a boundary through each pole from 1 to 6. At
    // each the plateau of the band at 2 holds the boundary where
    // 1/sinh(π|Im z|) ≥ 2, so each riser's top runs along it between
    // Im z = ±asinh(1/2)/π, at min(|f|, 4); and they come in order along
    // Re z, whatever order the process hashes the boundaries in.
    let bands: Vec<(f64, f64)> = (1..=6).map(|k| (k as f64, if k % 2 == 1 { 2.0 } else { 4.0 })).collect();
    let bundle = build(&request("1/sin(pi*z)", domain(-0.5, 6.5, -1.0, 1.0), 224, banded(&bands, 2.0)), true);
    let reach = real::asinh(0.5) / PI;
    let (mut tops, mut off): (Vec<(f64, f64)>, f64) = (Vec::new(), 0.0);
    for p in bundle.layer("truncation_edge").unwrap().paths.paths() {
        let x = p[0].x();
        if !(x == x.round() && x >= 1.0 && x <= 6.0 && p.iter().all(|v| v.x() == x)) {
            continue;
        }
        tops.push((x, p[0].y()));
        off = swift::max(off, swift::max((p[0].y() + reach).abs(), (p[p.len() - 1].y() - reach).abs()));
        for v in p {
            let u = 1.0
                / real::hypot(
                    real::sin(PI * v.x()) * real::cosh(PI * v.y()),
                    real::cos(PI * v.x()) * real::sinh(PI * v.y()),
                );
            off = swift::max(off, (v.z() - swift::min(u, 4.0)).abs());
        }
    }
    let xs: Vec<f64> = tops.iter().map(|t| t.0).collect();
    assert!(
        xs == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] && off < 1e-9,
        "one riser's top on each boundary, from Im z = -asinh(1/2)/π to asinh(1/2)/π at min(|f|, 4), in order along \
         Re z: tops at Re z = {xs:?}, off by {off}"
    );
}

/// "a window's curves are tested to its pixel: every end within the
/// resolution asked for".
#[test]
fn a_windows_curves_are_tested_to_its_pixel() {
    // The preview places the ends of tested curves to a quarter of its pixel
    // (`Scene.resolution`), and a bake to the last bit. The two drawings
    // differ only where a run ends, and there by less than what was
    // asked for, measured in the view plane: a point of either lies
    // within it of the other. A run that is itself shorter than that --
    // a vertex a hair in front of where the fold passes behind, drawn to
    // the last bit as a speck of no length -- is not drawn at all.
    let bundle = build(&LandscapeRequest::from_preset(Catalog::native().preset("cn").unwrap(), 240), true);
    let s = scene(&bundle);
    let h = s.heightfield().unwrap();
    let view = s.camera.view;
    let bounds = s.quick_bounds(20_000).unwrap();
    let pixel = Framing::fitting(&bounds, Viewport::new(1600, 1000), Framing::FIT_MARGIN).units_per_pixel;
    let resolution = 0.25 * pixel;
    let on_screen = |set: &PolylineSet<WorldSpace>| -> Vec<Vec<Vec2>> {
        set.paths()
            .map(|p| {
                p.iter()
                    .map(|&v| {
                        let q = view.apply(v);
                        Vec2::new(q.x(), q.y())
                    })
                    .collect()
            })
            .collect()
    };
    let distance = |p: Vec2, runs: &[Vec<Vec2>]| -> f64 {
        let mut best = f64::INFINITY;
        for run in runs {
            for w in run.windows(2) {
                let (a, b) = (w[0], w[1]);
                let ab = b - a;
                let l2 = ab.dot(ab);
                let t = if l2 > 0.0 { swift::max(0.0, swift::min(1.0, (p - a).dot(ab) / l2)) } else { 0.0 };
                best = swift::min(best, (a + t * ab - p).length());
            }
        }
        best
    };
    let length = |run: &[Vec2]| -> f64 { run.windows(2).map(|w| (w[1] - w[0]).length()).sum() };
    let (mut worst, mut longest_dropped, mut fewer) = (0.0f64, 0.0f64, 0i64);
    let mut pairs = vec![(
        h.visible_folds(&view, s.depth_tolerance, true, 0.0),
        h.visible_folds(&view, s.depth_tolerance, true, resolution),
    )];
    for layer in s.layers() {
        if s.geometry().face_of(&layer.spec) == Some(FaceKind::Wall) && layer.spec.clipped {
            pairs.push((
                h.visible_wall_curves(&layer.paths, &view, s.depth_tolerance, 0.0),
                h.visible_wall_curves(&layer.paths, &view, s.depth_tolerance, resolution),
            ));
        }
    }
    for (exact, coarse) in &pairs {
        let (a, b) = (on_screen(exact), on_screen(coarse));
        for run in &a {
            let far = seq_max(run.iter().map(|&p| distance(p, &b))).unwrap_or(0.0);
            if far > resolution && length(run) <= resolution {
                longest_dropped = swift::max(longest_dropped, length(run));
            } else {
                worst = swift::max(worst, far);
            }
        }
        for run in &b {
            for &p in run {
                worst = swift::max(worst, distance(p, &a));
            }
        }
        fewer += exact.vertices.len() as i64 - coarse.vertices.len() as i64;
    }
    assert!(
        worst <= resolution * (1.0 + 1e-9),
        "each drawing lies within a quarter pixel of the other: worst {:.3} of a pixel",
        worst / pixel
    );
    assert!(fewer > 0, "and the ends a pixel cannot show are not placed: {fewer} fewer vertices");
    assert!(
        longest_dropped <= resolution,
        "and what is not drawn is a speck: longest {:.4} of a pixel",
        longest_dropped / pixel
    );
}

/// The distance from `v` to the nearest segment of `set`, in space.
fn to_segments(set: &PolylineSet<WorldSpace>, v: P3<WorldSpace>) -> f64 {
    let mut best = f64::INFINITY;
    for p in set.paths() {
        for w in p.windows(2) {
            let (ab, ap) = (w[1].v - w[0].v, v.v - w[0].v);
            let l = ab.dot(ab);
            let t = if l > 0.0 { swift::min(swift::max(ap.dot(ab) / l, 0.0), 1.0) } else { 0.0 };
            best = swift::min(best, (v.v - (w[0].v + t * ab)).length());
        }
    }
    best
}

/// The length of a set's runs in space (Swift's `curves(_:)`).
fn length_3d(set: &PolylineSet<WorldSpace>) -> f64 {
    set.paths().map(|p| p.windows(2).map(|w| (w[1].v - w[0].v).length()).sum::<f64>()).sum()
}

fn indent(text: &str) -> String {
    format!("      {}", text.replace('\n', "\n      "))
}

/// "one surface: on a grid of 24 the pit of 1/Γ at -3 is drawn from f alone".
#[test]
fn one_surface_on_a_grid_of_24_the_pit_of_reciprocal_gamma_is_drawn_from_f_alone() {
    // The figure's own window: 1/Γ over Re [-3.8, -2.2], Im [0, 0.8],
    // truncated at 2, on 24 samples across, so that a cell is a visible
    // fraction of the object and no approximation has anywhere to hide.
    // The rings are placed on f, the folds are traced on f, the top edge is
    // f's profile and the curves are tested against f, so: every ring that
    // ends on a fold ends on the drawn fold to the refiner's tolerance;
    // the top edge and the truncation edge meet to rounding; and the fold down the
    // pit's far graph ends at the zero, where the top edge bottoms out.
    let bundle = build(&request("1/gamma(z)", domain(-3.8, -2.2, 0.0, 0.8), 24, Truncation::Uniform(2.0)), true);
    let s = scene(&bundle);
    let h: &Heightfield = s.heightfield().unwrap();
    let cell = 1.6 / 23.0;
    let view = s.camera.view;
    let vis = HeightfieldVisibility::new(h.clone(), &view, s.depth_tolerance);
    // The tested curves, in world space: the folds that show, and each
    // surface layer less what cannot be seen.
    let folds = h.visible_folds(&view, s.depth_tolerance, true, 0.0);
    let visible: Vec<(&Layer, PolylineSet<WorldSpace>)> = s
        .layers()
        .iter()
        .map(|l| (l, h.visible_curves(&l.paths, Some(FaceKind::Graph), &view, s.depth_tolerance, 0.0)))
        .collect();
    // The pit is a bowl: the graph is convex there, so the solid under
    // it is hollow, and a line of sight tangent to the graph -- which is
    // what a fold is -- runs under the graph, through the graph's own
    // material, until it leaves through the cut. No fold in a pit can
    // be seen; what bounds the rings there is the cut's top edge. The
    // traced fold runs to the zero (below), and the test hides it:
    // what little shows lies on the cut itself.
    let traced = h.fold_curves(&view, true);
    // The only fold vertices that can be seen are on the cut, whose
    // lines of sight leave the solid at once: a fold is tested with no
    // depth tolerance, since its vertices lie on f exactly.
    let off_cut = seq_max(folds.vertices.iter().map(|v| v.y())).unwrap_or(0.0);
    assert!(
        traced.count() >= 1 && length_3d(&traced) > 0.5 && off_cut < 1e-3 * cell,
        "no fold is seen in the pit: a line of sight tangent to a bowl's graph runs through the graph: {:.3} units \
         traced, {:.3} seen, furthest {off_cut:.1e} from the cut",
        length_3d(&traced),
        length_3d(&folds)
    );
    // The top edge of the front wall, as drawn, projected onto the drawing:
    // what a ring that passes behind it must end on.
    let top_edge = top_edge_at(&bundle, 0.0);
    let top_edge_on_page: Vec<_> = top_edge.iter().map(|&v| view.apply(v)).collect();
    let to_top_edge_on_page = |v: P3<WorldSpace>| -> f64 {
        let q = view.apply(v);
        let mut best = f64::INFINITY;
        for w in top_edge_on_page.windows(2) {
            let (a, b) = (w[0], w[1]);
            let ab = Vec2::new(b.x() - a.x(), b.y() - a.y());
            let aq = Vec2::new(q.x() - a.x(), q.y() - a.y());
            let l = ab.dot(ab);
            let t = if l > 0.0 { swift::min(swift::max(aq.dot(ab) / l, 0.0), 1.0) } else { 0.0 };
            best = swift::min(best, (aq - t * ab).length());
        }
        best
    };
    // Every end of every tested ring, sorted by what it ends on: the
    // domain's edge, the truncation, a zero, a fold (facing zero there), or
    // the traversal (hidden behind the top edge). The last two are the ones
    // the grid used to leave short.
    let (mut worst, mut ends, mut worst_top_edge, mut by_traversal) = (0.0f64, 0, 0.0f64, 0);
    let (mut on_edge_count, mut truncation_height, mut zero) = (0, 0, 0);
    for (layer, paths) in &visible {
        if !(layer.spec.clipped && s.geometry().face_of(&layer.spec) == Some(FaceKind::Graph)) {
            continue;
        }
        for p in paths.paths() {
            for end in [p[0], p[p.len() - 1]] {
                let on_edge = (end.x() + 3.8).abs() < 1e-6
                    || (end.x() + 2.2).abs() < 1e-6
                    || end.y().abs() < 1e-6
                    || (end.y() - 0.8).abs() < 1e-6;
                if on_edge {
                    on_edge_count += 1;
                    continue;
                }
                if end.z() > 2.0 - 1e-6 {
                    truncation_height += 1;
                    continue;
                }
                if end.z() < 1e-6 {
                    zero += 1;
                    continue;
                }
                if vis.facing(end.xy()).abs() < 1e-6 {
                    ends += 1;
                    let g = to_segments(&folds, end);
                    worst = swift::max(worst, g);
                    if g > 0.02 * cell {
                        // On failure: the traversal's own account of the end
                        // and of the nearest vertex of the traced fold.
                        eprintln!(
                            "    fold-cut end ({:.4}, {:.4}, {:.4}) {}: {g:.2e} from folds, run of {}",
                            end.x(),
                            end.y(),
                            end.z(),
                            layer.spec.name,
                            p.len()
                        );
                        eprintln!("{}", indent(&vis.explain(end)));
                    }
                } else {
                    by_traversal += 1;
                    let g = to_top_edge_on_page(end);
                    worst_top_edge = swift::max(worst_top_edge, g);
                    if g > 2.0 * 0.02 * cell {
                        // On failure: the traversal's account of the end.
                        eprintln!(
                            "    top edge-cut end ({:.4}, {:.4}, {:.4}) {}: {g:.2e} on the page, facing {:.3}, run of {}",
                            end.x(),
                            end.y(),
                            end.z(),
                            layer.spec.name,
                            vis.facing(end.xy()),
                            p.len()
                        );
                        eprintln!("{}", indent(&vis.explain(end)));
                    }
                }
            }
        }
    }
    assert!(
        ends == 0 || worst <= 0.02 * cell + 1e-9,
        "a ring cut on a fold ends on the drawn fold, within the refiner's tolerance: {ends} ends, worst {worst:.2e} \
         against {:.2e}",
        0.02 * cell
    );
    // Within the top edge's own tolerance, and the depth tolerance the test
    // forgives, which is the tolerance again.
    assert!(
        by_traversal >= 10 && worst_top_edge <= 2.0 * 0.02 * cell + 1e-9,
        "every ring that passes behind the top edge ends on the top edge as drawn, on the drawing: {by_traversal} \
         ends, worst {worst_top_edge:.2e} against {:.2e}; {on_edge_count} on the edge, {truncation_height} on the \
         plateau, {zero} at zeros",
        2.0 * 0.02 * cell
    );
    // The top edge and the truncation edge.
    let mut truncation_edge_ends: Vec<P3<WorldSpace>> = Vec::new();
    for p in bundle.layer("truncation_edge").unwrap().paths.paths() {
        for end in [p[0], p[p.len() - 1]] {
            if end.y().abs() < 1e-9 {
                truncation_edge_ends.push(end);
            }
        }
    }
    let mut missed = 0.0f64;
    for end in &truncation_edge_ends {
        missed = swift::max(missed, seq_min(top_edge.iter().map(|v| (v.v - end.v).length())).unwrap_or(f64::INFINITY));
    }
    assert!(
        !truncation_edge_ends.is_empty() && missed < 1e-9,
        "the truncation edge's ends on the front edge are vertices of the top edge: {} ends, worst gap {missed:.1e}",
        truncation_edge_ends.len()
    );
    let top_edge_low = lowest(&top_edge);
    assert!(
        (top_edge_low.x() + 3.0).abs() < 1e-6 && top_edge_low.z() < 1e-9,
        "the top edge bottoms out at the zero: ({:.6}, {:.1e})",
        top_edge_low.x(),
        top_edge_low.z()
    );
    let fold_at_zero = traced.vertices.iter().any(|v| (v.x() + 3.0).abs() < 1e-6 && v.y().abs() < 1e-6 && v.z() < 1e-9);
    assert!(fold_at_zero, "and the traced fold ends there");
    // The top edge and the folds are curves to the tolerance: no fold chord
    // departs from the fold by more than it, measured by the facing at
    // the chord's midpoint against the facing's slope across it.
    let mut fold_sag = 0.0f64;
    for p in traced.paths() {
        for w in p.windows(2) {
            let (a, b) = (w[0], w[1]);
            if !(a.z() > 1e-9 && b.z() > 1e-9) {
                continue;
            }
            let mid = P2::<WorldSpace>::new(0.5 * (a.x() + b.x()), 0.5 * (a.y() + b.y()));
            let chord = ((b.x() - a.x()) * (b.x() - a.x()) + (b.y() - a.y()) * (b.y() - a.y())).sqrt();
            if !(chord > 1e-12) {
                continue;
            }
            let n = Vec2::new(-(b.y() - a.y()), b.x() - a.x()) / chord;
            let e = 1e-3 * cell;
            let slope = (vis.facing(P2::new(mid.x() + e * n.x, mid.y() + e * n.y))
                - vis.facing(P2::new(mid.x() - e * n.x, mid.y() - e * n.y)))
                / (2.0 * e);
            if !(slope.abs() > 1e-9) {
                continue;
            }
            fold_sag = swift::max(fold_sag, (vis.facing(mid) / slope).abs());
        }
    }
    assert!(
        fold_sag <= 0.02 * cell * 1.5,
        "no chord of the traced fold departs from it by more than the tolerance: worst {fold_sag:.2e} against {:.2e}",
        0.02 * cell
    );
}
