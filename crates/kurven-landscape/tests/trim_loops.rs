//! The graph's trim loops and the banded truncation edge on f, from
//! `Sources/kurven-test/Faces.swift` (its run from "the graph's trim loops
//! close a plateau that turns back a column short of a band boundary" to "a
//! pole beside a band boundary, at every resolution", and "the graph's trim
//! loops and its truncation edge are the same in whatever order the tiles
//! are listed"), with Swift's tolerances. Swift builds each landscape with
//! `NativeLandscape.build(_:refine: true)`; here its grids and refiner are
//! built as that does (`common::build`).

// Swift's `!(d <= tolerance)` keeps a distance that is not a number outside.
#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::type_complexity)]

mod common;

use common::{Built, Segments, bands, build, domain};
use kurven_core::hatch::HatchContext;
use kurven_core::manifest::{Affine2, LayerSource, Region, Truncation, Walls};
use kurven_core::simd::Vec2;
use kurven_core::spaces::{P2, P3, WorldSpace};
use kurven_math::real;

/// Two poles either side of Re z = 1/2 and a zero between them on it.
const HOLE_BESIDE_A_BOUNDARY: &str = "1/(z - (0.4804 - 0.0550i)) + 1/(z - (0.5196 + 0.0550i))";
/// Two poles above and below a zero just left of Re z = 1/2.
const RING_BESIDE_A_BOUNDARY: &str = "1/(z - (0.482 + 0.0853333i)) + 1/(z - (0.482 - 0.2146667i))";

fn window() -> kurven_core::manifest::Domain {
    domain(-0.5, 1.5, -1.0, 1.0)
}

/// The truncation edge's placed pieces, less the risers' tops that end it.
fn placed(trim: &kurven_core::hatch::GraphTrimLoops) -> &[Vec<P3<WorldSpace>>] {
    &trim.truncation_edge[..trim.truncation_edge.len() - trim.riser_tops.len()]
}

fn loop_segments(trim: &kurven_core::hatch::GraphTrimLoops, cell: f64) -> Segments {
    Segments::new(&trim.loops.iter().map(|l| l.iter().map(|p| p.v).collect()).collect::<Vec<_>>(), cell)
}

fn open_chains(trim: &kurven_core::hatch::GraphTrimLoops) -> Vec<String> {
    trim.loops.iter().filter(|l| l.first() != l.last()).map(|l| format!("{:?} to {:?}", l[0], l[l.len() - 1])).collect()
}

/// Whether the loops hold exactly |f| < h on a lattice, away from them.
fn holds_the_hole(
    built: &Built,
    loops: &Segments,
    cell: f64,
    points: impl Iterator<Item = Vec2>,
    h: f64,
) -> (usize, Vec<String>) {
    let magnitude = built.magnitude();
    let (mut asked, mut wrong) = (0, Vec::new());
    for p in points {
        if loops.distance(p, 0.06 * cell).is_finite() {
            continue;
        }
        asked += 1;
        if loops.inside(p) != (magnitude(P2::from_vec(p)) < h) {
            wrong.push(format!("{p:?}"));
        }
    }
    (asked, wrong)
}

#[test]
fn the_graphs_trim_loops_close_a_plateau_that_turns_back_a_column_short_of_a_band_boundary() {
    // A pole's plateau that comes within a column of a band boundary without
    // crossing it, |f| under the truncation all along the boundary: in the
    // cell the boundary cuts, contoured over the plateau's own band, its line
    // turns back short of the boundary, so the plateau's loop closes, in the
    // drawing as in the loops. On a coarse grid, and on a fine one with the
    // plateau nearer the boundary.
    for (expression, resolution) in [("1/(z + 0.03)", 32), ("1/(z + 0.01)", 200)] {
        let pole = build(expression, window(), resolution, bands(&[(0.5, 2.0)], 4.0));
        let trim = pole.surface.graph_trim_loops(Some(&pole.refine));
        let open = open_chains(&trim);
        assert!(
            trim.open == 0 && open.is_empty(),
            "{expression} at res {resolution}: its plateau's loop, a column short of the boundary at Re z = 1/2, is closed: {} counted, {open:?}",
            trim.open
        );
    }
}

#[test]
fn a_hole_within_a_column_of_a_band_boundary_is_drawn_in_each_band_at_its_own_height_once() {
    // Round the zero at 1/2 on the boundary Re z = 1/2, at 1.5 | 1.8, the
    // plateau has a hole a sixteenth of a column across: a half disc at 1.5
    // left of the boundary and one at 1.8 right of it. On windows whose row
    // through the zero sees it on the boundary, and on ones whose grid sees
    // nothing there, where it is the refiner's ring round the zero, seeded
    // once a band and cut on the boundary. And with the zero a thousandth off
    // the boundary, the poles moved with it.
    let off_the_boundary = "1/(z - (0.5206 + 0.0650i)) + 1/(z - (0.4814 - 0.0450i))";
    let mut failures = Vec::new();
    for (expression, re, im) in [
        (HOLE_BESIDE_A_BOUNDARY, (-0.3, 1.5), (-0.5, 0.5)),
        (HOLE_BESIDE_A_BOUNDARY, (0.3, 1.0), (-1.0, 0.5)),
        (HOLE_BESIDE_A_BOUNDARY, (0.0, 1.0), (-0.5, 0.5)),
        (HOLE_BESIDE_A_BOUNDARY, (-0.5, 1.5), (-1.0, 1.0)),
        (off_the_boundary, (0.0, 1.0), (-0.5, 0.5)),
    ] {
        let built = build(expression, domain(re.0, re.1, im.0, im.1), 24, bands(&[(0.5, 1.5)], 1.8));
        let magnitude = built.magnitude();
        let trim = built.surface.graph_trim_loops(Some(&built.refine));
        let hole: Vec<&Vec<P3<WorldSpace>>> = placed(&trim)
            .iter()
            .filter(|p| p.iter().all(|v| (v.x() - 0.5).abs() < 0.05 && v.y().abs() < 0.05))
            .collect();
        let left = hole.iter().filter(|p| p.iter().all(|v| v.x() <= 0.5 && v.z() == 1.5)).count();
        let right = hole.iter().filter(|p| p.iter().all(|v| v.x() >= 0.5 && v.z() == 1.8)).count();
        let on_boundary = hole.iter().all(|p| p.first() != p.last() && p[0].x() == 0.5 && p[p.len() - 1].x() == 0.5);
        let on_level = hole
            .iter()
            .flat_map(|p| p.iter())
            .all(|v| (magnitude(P2::new(v.x(), v.y())) - v.z()).abs() <= 1e-6 * v.z());
        if !(hole.len() == 2 && left == 1 && right == 1 && on_boundary && on_level && trim.open == 0) {
            failures.push(format!(
                "{expression} Re {re:?}, Im {im:?}: {} pieces, {left} left, {right} right, on the boundary {on_boundary}, on the level {on_level}, {} open",
                hole.len(),
                trim.open
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn a_zeros_ring_is_drawn_once_where_a_cell_a_band_boundary_cuts_drew_an_arc_of_it() {
    let built = build(RING_BESIDE_A_BOUNDARY, window(), 16, bands(&[(0.5, 1.5)], 1.8));
    let trim = built.surface.graph_trim_loops(Some(&built.refine));
    let arcs: Vec<&Vec<P3<WorldSpace>>> = placed(&trim)
        .iter()
        .filter(|p| {
            p.iter()
                .all(|v| v.x() >= 0.5 && v.z() == 1.8 && (v.x() - 0.482).abs() < 0.05 && (v.y() + 0.0647).abs() < 0.05)
        })
        .collect();
    assert!(
        trim.open == 0 && arcs.len() == 1 && arcs.iter().all(|a| a[0].x() == 0.5 && a[a.len() - 1].x() == 0.5),
        "the arc right of the boundary is drawn once, from the boundary to the boundary, and the loops close: {} arcs, {} open",
        arcs.len(),
        trim.open
    );
}

#[test]
fn a_zeros_ring_beside_a_saddle_is_drawn_though_the_edge_round_another_zero_passes_near_it() {
    let built = build(
        "353.319490401*(z - (0.650093325017 + 0.014383779377i))*(z - (0.769125500952 - 0.211731000630i))^2/(z - (0.536334390878 - 0.621521184752i))",
        window(),
        24,
        bands(&[(0.721440736242, 1.5)], 4.0),
    );
    let trim = built.surface.graph_trim_loops(Some(&built.refine));
    let cell = 2.0 / 23.0;
    let loops = loop_segments(&trim, cell);
    let points = (0..10).flat_map(|i| {
        (0..9).map(move |j| Vec2::new(0.6 + (i as f64 + 0.37) * cell / 7.0, -0.045 + (j as f64 + 0.61) * cell / 7.0))
    });
    let (asked, wrong) = holds_the_hole(&built, &loops, cell, points, 1.5);
    assert!(
        asked > 50 && wrong.is_empty() && trim.open == 0,
        "round the simple zero the loops hold its hole, and only it: {} of {asked} wrong: {:?}, {} open",
        wrong.len(),
        &wrong[..wrong.len().min(3)],
        trim.open
    );
}

#[test]
fn one_ring_is_drawn_round_two_zeros_in_one_hole() {
    let built = build(
        "4119.1074405*(z - (0.307498198308-0.144345961027i))*(z - (0.348393843985-0.145720969539i))/(z - (0.764249524783+0.504053378065i))",
        window(),
        24,
        bands(&[(0.114094111388, 2.0)], 4.0),
    );
    let trim = built.surface.graph_trim_loops(Some(&built.refine));
    let cell = 2.0 / 23.0;
    let loops = loop_segments(&trim, cell);
    let points = (0..9).flat_map(|i| {
        (0..7).map(move |j| Vec2::new(0.28 + (i as f64 + 0.37) * cell / 7.0, -0.18 + (j as f64 + 0.61) * cell / 7.0))
    });
    let (asked, wrong) = holds_the_hole(&built, &loops, cell, points, 4.0);
    let rings = trim
        .truncation_edge
        .iter()
        .filter(|p| p.iter().all(|v| (v.x() - 0.33).abs() < 0.05 && (v.y() + 0.145).abs() < 0.05))
        .count();
    assert!(
        asked > 40 && wrong.is_empty() && rings == 1 && trim.open == 0,
        "the loops hold the hole round both zeros, drawn once: {rings} rings, {} of {asked} wrong: {:?}, {} open",
        wrong.len(),
        &wrong[..wrong.len().min(3)],
        trim.open
    );
}

#[test]
fn a_rings_part_in_a_band_is_left_out_where_the_bands_edge_already_joins_its_ends() {
    let built = build(
        "595.849573804*(z - (0.724529786873+0.145657568132i))*(z - (0.825622995430+0.075733727046i))/(z - (0.294022667147+0.587508313898i))",
        window(),
        24,
        bands(&[(0.760571784330, 3.0)], 4.0),
    );
    let trim = built.surface.graph_trim_loops(Some(&built.refine));
    let open = open_chains(&trim);
    assert!(trim.open == 0 && open.is_empty(), "the band's loops close: {} counted, {open:?}", trim.open);
}

#[test]
fn a_chord_is_taken_past_a_zero_only_off_its_own_run_and_within_its_band() {
    let cell = 2.0 / 23.0;
    let mut failures = Vec::new();
    for (name, expression, band, beyond, x, y) in [
        (
            "the lower zero's hole",
            "948.115004635*(z - (0.563659351434+0.277678080119i))*(z - (0.550062724850+0.205189440392i))/(z - (0.856142777298+0.772831571274i))",
            (0.411407161624, 2.0),
            4.0,
            (0.52, 0.585),
            (0.183, 0.212),
        ),
        (
            "the hole of a four-vertex loop",
            "416.937178124*(z - (0.840040450901-0.068301508238i))*(z - (0.716014763340-0.121761607551i))/(z - (0.113910517434+0.575219223905i))",
            (0.532950907488, 2.0),
            2.5,
            (0.81, 0.87),
            (-0.11, -0.035),
        ),
    ] {
        let built = build(expression, window(), 24, bands(&[band], beyond));
        let trim = built.surface.graph_trim_loops(Some(&built.refine));
        let loops = loop_segments(&trim, cell);
        let h = built.surface.truncation.height_at(x.0);
        let (ni, nj) = (((x.1 - x.0) / cell * 14.0) as usize, ((y.1 - y.0) / cell * 14.0) as usize);
        let points = (0..=ni)
            .flat_map(|i| (0..=nj).map(move |j| Vec2::new(x.0 + i as f64 * cell / 14.0, y.0 + j as f64 * cell / 14.0)));
        let (asked, wrong) = holds_the_hole(&built, &loops, cell, points, h);
        if !(asked > 20 && wrong.is_empty()) {
            failures.push(format!("{name}: {} of {asked} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(3)]));
        }
    }
    let b = 0.711993911036;
    let built = build(
        "1703.90688736*(z - (0.732858379215+0.011300545905i))*(z - (0.811553422942-0.149386716774i))^2/(z - (0.183946050532+0.373413325442i))",
        window(),
        24,
        bands(&[(b, 2.0)], 2.5),
    );
    let trim = built.surface.graph_trim_loops(Some(&built.refine));
    let outside: Vec<String> = placed(&trim)
        .iter()
        .flatten()
        .filter(|v| if v.z() == 2.5 { v.x() < b } else { v.x() > b })
        .map(|v| format!("{v:?}"))
        .collect();
    if !(trim.open == 0 && outside.is_empty()) {
        failures.push(format!(
            "each band's truncation edge keeps to its band: {} outside: {:?}, {} open",
            outside.len(),
            &outside[..outside.len().min(3)],
            trim.open
        ));
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn the_truncation_edge_in_a_cell_a_band_boundary_cuts_keeps_to_the_level_within_the_refiners_tolerance() {
    let mixed = "(z - (1.1934990849887146 - 0.20536860263116311i))*(z - (1.2234990849887146 - 0.2253686026311631i))/((z - (1.2524804507563325 - 0.078069596745760084i))*(z - (1.399997924755525 + 0.40148084601005929i))^2)";
    let w = window();
    let cases: Vec<(&str, &str, kurven_core::manifest::Domain, i64, Vec<(f64, f64)>, f64)> = vec![
        (
            "a zero on the boundary, its hole wider than the part",
            "1/(z - (0.59724288035304796 - 0.13390328401912382i)) + 1/(z - (0.44728054417818569 - 0.37915093303657482i))",
            w,
            24,
            vec![(0.5222617122656168, 1.5)],
            4.0,
        ),
        ("a pole within a chord", mixed, w, 16, vec![(0.8333333333333333, 3.0), (1.3001429547193402, 4.0)], 1.0),
        (
            "a plateau leaving its band between two rows",
            "1/(z - (-0.095946946980314418 - 0.0047827152763062486i))",
            w,
            16,
            vec![(-0.42375125887220094, 1.2)],
            3.0,
        ),
        (
            "a zero on a column between two poles",
            "1/(z - (0.59076180115334842-0.093833748934804229i)) + 1/(z - (0.5145013567413883-0.13573974784644616i))",
            w,
            20,
            vec![(0.5526315789473684, 3.0)],
            1.8,
        ),
        (
            "a zero in a pole's cell",
            "1/(z - (0.42240022182367271-0.40227584782117776i)) + 1/(z - (0.37140972304490555-0.48709752678493168i))",
            w,
            20,
            vec![(0.3969049724342891, 3.0)],
            4.0,
        ),
        (
            "a chord beside a zero",
            "(z - (0.40691152265074942-0.15224404800258995i))/((z - (0.62359753483716163-0.13770458918473533i))*(z - (0.57445433013745983-0.14044013384631204i)))",
            w,
            16,
            vec![(0.45369534601865147, 1.5)],
            4.0,
        ),
        (
            "the hole round a zero on the boundary",
            HOLE_BESIDE_A_BOUNDARY,
            domain(-0.3, 1.5, -0.5, 0.5),
            24,
            vec![(0.5, 1.5)],
            1.8,
        ),
        ("a zero's ring and a cut cell's arc of it", RING_BESIDE_A_BOUNDARY, w, 16, vec![(0.5, 1.5)], 1.8),
        ("the reviewer's tower", "1/(z - 1.4)", w, 32, vec![(1.45, 0.5)], 1.0),
    ];
    let mut failures = Vec::new();
    for (name, expression, d, resolution, band_list, beyond) in cases {
        let built = build(expression, d, resolution, bands(&band_list, beyond));
        let magnitude = built.magnitude();
        let surface = &built.surface;
        let g = &*surface.height;
        let dx = g.domain.real.length().abs() / (g.width - 1) as f64;
        let cell = dx.max(g.domain.imag.length().abs() / (g.height - 1) as f64);
        let trim = surface.graph_trim_loops(Some(&built.refine));
        let (mut chords, mut worst) = (0, 0.0f64);
        for piece in placed(&trim) {
            for w in piece.windows(2) {
                let (a, b) = (w[0], w[1]);
                let m = P2::new(0.5 * (a.x() + b.x()), 0.5 * (a.y() + b.y()));
                if !band_list.iter().any(|&(below, _)| (m.x() - below).abs() <= dx) {
                    continue;
                }
                let e = 1e-4 * cell;
                let gx = (magnitude(P2::new(m.x() + e, m.y())) - magnitude(P2::new(m.x() - e, m.y()))) / (2.0 * e);
                let gy = (magnitude(P2::new(m.x(), m.y() + e)) - magnitude(P2::new(m.x(), m.y() - e))) / (2.0 * e);
                chords += 1;
                worst = worst.max((magnitude(m) - a.z()).abs() / (gx * gx + gy * gy).sqrt() / cell);
            }
        }
        if !(chords > 1 && worst <= built.refine.tolerance) {
            failures.push(format!("{name}: {chords} chords, worst {worst:.4} cells off"));
        }
        // And the loops hold the graph there, |f| under the truncation, read
        // on a lattice a seventh of a cell across, away from them.
        let loops = loop_segments(&trim, cell);
        let (real, imag) = (surface.domain().real, surface.domain().imag);
        let (mut asked, mut wrong) = (0, Vec::new());
        for &(below, _) in &band_list {
            for i in 0..14 {
                for j in 0..=((imag.length().abs() / cell * 7.0) as usize) {
                    let p = Vec2::new(
                        below - dx + (i as f64 + 0.37) * dx / 7.0,
                        imag.lo.min(imag.hi) + (j as f64 + 0.61) * cell / 7.0,
                    );
                    if !(p.x > real.lo.min(real.hi)
                        && p.x < real.lo.max(real.hi)
                        && p.y < imag.lo.max(imag.hi)
                        && loops.distance(p, 0.06 * cell).is_infinite())
                    {
                        continue;
                    }
                    asked += 1;
                    if loops.inside(p) != (magnitude(P2::from_vec(p)) < surface.truncation.height_at(p.x)) {
                        wrong.push(format!("{p:?}"));
                    }
                }
            }
        }
        if !(asked > 200 && wrong.is_empty()) {
            failures.push(format!(
                "{name}: within a column of a boundary the loops hold |f| < h(x): {} of {asked} wrong: {:?}",
                wrong.len(),
                &wrong[..wrong.len().min(3)]
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn a_risers_top_follows_f_through_a_dip_narrower_than_its_stretch() {
    let dip = "1/(z-0.49-0.1*i)+1/(z-0.49+0.1*i)";
    let tall = domain(-0.5, 1.5, -1.0, 1.3);
    let mut cases: Vec<(String, &str, kurven_core::manifest::Domain, i64, (f64, f64), f64)> = Vec::new();
    for resolution in [55, 60, 70, 80, 100] {
        cases.push((format!("two poles with a dip between, res {resolution}"), dip, tall, resolution, (0.5, 0.8), 4.0));
    }
    for shift in [0.003, 0.007, 0.011, 0.017] {
        cases.push((
            format!("two poles with a dip between, res 60, Im z moved {shift}"),
            dip,
            domain(-0.5, 1.5, -1.0 + shift, 1.3 + shift),
            60,
            (0.5, 0.8),
            4.0,
        ));
    }
    for (apart, expression) in
        [(0.06, "1/(z-0.495-0.03*i)+1/(z-0.495+0.03*i)"), (0.04, "1/(z-0.497-0.02*i)+1/(z-0.497+0.02*i)")]
    {
        cases.push((format!("two poles {apart} apart, res 60"), expression, tall, 60, (0.5, 0.8), 40.0));
    }
    cases.push(("a tower".to_string(), "1/(z-0.45)", window(), 200, (0.5, 2.0), 4.0));
    let mut failures = Vec::new();
    for (name, expression, d, resolution, band, beyond) in cases {
        let built = build(expression, d, resolution, bands(&[band], beyond));
        let magnitude = built.magnitude();
        let g = &*built.surface.height;
        let cell = (g.domain.real.length().abs() / (g.width - 1) as f64)
            .max(g.domain.imag.length().abs() / (g.height - 1) as f64);
        let greater = band.1.max(beyond);
        let trim = built.surface.graph_trim_loops(Some(&built.refine));
        let (mut points, mut worst, mut off) = (0, 0.0f64, Vec::new());
        for (b, top) in &trim.riser_tops {
            for w in top.windows(2) {
                let (p, q) = (w[0], w[1]);
                for s in 0..=8 {
                    let t = s as f64 / 8.0;
                    let (y, z) = (p.y() + t * (q.y() - p.y()), p.z() + t * (q.z() - p.z()));
                    let u = magnitude(P2::new(*b, y));
                    let d = z - (if u.is_nan() { f64::INFINITY } else { u }).min(greater);
                    if !d.is_finite() {
                        continue;
                    }
                    points += 1;
                    if d.abs() > worst.abs() {
                        worst = d;
                    }
                    if !(d.abs() <= 0.75 * built.refine.tolerance * cell) {
                        off.push(format!("({b:.4}, {y:.4}, {z:.4}) {d:+.6}"));
                    }
                }
            }
        }
        if !(points > 50 && off.is_empty()) {
            failures.push(format!(
                "{name}: {} of {points} points off, worst {:.2} of the tolerance: {:?}",
                off.len(),
                worst.abs() / (built.refine.tolerance * cell),
                &off[..off.len().min(3)]
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn a_band_narrower_than_a_column_between_a_boundary_and_the_far_edge_has_its_plateaus_loop() {
    let tower = build("1/(z - 1.4)", window(), 32, bands(&[(1.45, 0.5)], 1.0));
    let trim = tower.surface.graph_trim_loops(Some(&tower.refine));
    let own: Vec<&Vec<P2<_>>> =
        trim.loops.iter().filter(|l| l.first() == l.last() && l.iter().all(|p| p.x() >= 1.45)).collect();
    let span = own.first().map_or(0.0, |l| {
        l.iter().map(|p| p.y()).fold(f64::NEG_INFINITY, f64::max)
            - l.iter().map(|p| p.y()).fold(f64::INFINITY, f64::min)
    });
    assert!(
        trim.open == 0 && own.len() == 1 && span > 1.99,
        "the narrow band's plateau is one closed loop, from Im z = -0.9987 to 0.9987: {} loops in the band, {} open",
        own.len(),
        trim.open
    );
    let faces = tower.surface.faces(&[Affine2::IDENTITY], &Region::Full, &Walls::None, Some(&tower.refine));
    let graph = faces.first().expect("the graph");
    assert_eq!(graph.id, kurven_core::face::FaceID::Graph);
    let loops = graph.trim_loops().expect("the graph has trim loops");
    let cell = 2.0 / 31.0;
    let segments = Segments::new(&loops.iter().map(|l| l.iter().map(|p| p.v).collect()).collect::<Vec<_>>(), cell);
    let (mut asked, mut wrong, mut example) = (0, 0, String::new());
    for i in 0..5 {
        for j in 0..160 {
            let p = Vec2::new(1.45 + 0.05 * (i as f64 + 0.5) / 5.0, -1.0 + 2.0 * (j as f64 + 0.5) / 160.0);
            if segments.distance(p, 0.05 * cell).is_finite() {
                continue;
            }
            asked += 1;
            if segments.inside(p) != (graph.copies[0].contains)(P2::from_vec(p)) {
                wrong += 1;
                if example.is_empty() {
                    example = format!("{p:?}: the loops say {}", segments.inside(p));
                }
            }
        }
    }
    assert!(
        asked > 600 && wrong == 0,
        "between the boundary and the far edge, a point is in the graph where its loops hold it: {wrong} of {asked} disagree {example}"
    );
}

#[test]
fn a_pole_beside_a_band_boundary_at_every_resolution_has_its_truncation_edge_on_its_own_bands_level() {
    let (mut cases, mut wrong, mut off) = (0, 0, 0);
    let (mut examples, mut open) = (Vec::new(), Vec::new());
    for (left, right) in [(2.0, 4.0), (4.0, 2.0)] {
        for resolution in [24, 32, 48, 64, 100, 200] {
            for k in 0..41 {
                let c = -0.2 + 1.4 * k as f64 / 40.0;
                let cy = 0.13 * real::sin(k as f64);
                // Swift interpolates the two doubles with `description`.
                let expression = format!("1/(z - ({} + {}i))", swift_description(c), swift_description(cy));
                let pole = build(&expression, window(), resolution, bands(&[(0.5, left)], right));
                let magnitude = pole.magnitude();
                let trim = pole.surface.graph_trim_loops(Some(&pole.refine));
                let name = format!("{left}|{right} res {resolution} c {c:.2}");
                cases += 1;
                if trim.open > 0 {
                    open.push(format!("{name}: {}", trim.open));
                }
                for v in placed(&trim).iter().flatten().filter(|v| v.x() != 0.5) {
                    let h = if v.x() < 0.5 { left } else { right };
                    if v.z() != h {
                        wrong += 1;
                        if examples.len() < 3 {
                            examples.push(format!("{name}: ({}, {}) at {}", v.x(), v.y(), v.z()));
                        }
                    }
                    if !((magnitude(P2::new(v.x(), v.y())) - v.z()).abs() <= 1e-6 * v.z()) {
                        off += 1;
                        if examples.len() < 3 {
                            examples.push(format!(
                                "{name}: ({}, {}) |f| {}",
                                v.x(),
                                v.y(),
                                magnitude(P2::new(v.x(), v.y()))
                            ));
                        }
                    }
                }
            }
        }
    }
    assert!(
        cases == 492 && wrong == 0 && off == 0,
        "every vertex of 492 poles' truncation edges lies in the band at its x, on |f| = that band's height: {wrong} in the wrong band, {off} off the level: {examples:?}"
    );
    assert!(open.is_empty(), "and none of their loops is open: {open:?}");
}

/// Swift's `Double.description` for the numbers this suite writes into an
/// expression: Rust's shortest round trip, with Swift's spelling of an
/// exponent.
fn swift_description(x: f64) -> String {
    kurven_core::json::number(x)
}

#[test]
fn the_graphs_trim_loops_and_its_truncation_edge_are_the_same_in_whatever_order_the_tiles_are_listed() {
    let pole = build("1/(z - 0.95)", domain(0.0, 1.0, -0.5, 0.5), 40, Truncation::Uniform(3.0));
    let tower = build("1/(z - 0.5)", window(), 32, bands(&[(0.5, 2.0)], 4.0));
    let affine = |a: f64, tx: f64| Affine2 { a, b: 0.0, tx, c: 0.0, d: 1.0, ty: 0.0 };
    let mut failures = Vec::new();
    for (name, built, tile) in [
        ("1/(z - 0.95) cut at 3, a tile beside it", &pole, affine(1.0, 1.0)),
        ("a tower banded at 2 | 4, a tile beside it", &tower, affine(1.0, 2.0)),
        ("a tower banded at 2 | 4, a tile reflected across its right edge", &tower, affine(-1.0, 3.0)),
    ] {
        let drawn = |tiles: &[Affine2]| -> (Vec<Vec<Vec2>>, Vec<[f64; 3]>) {
            let faces = built.surface.faces(tiles, &Region::Full, &Walls::None, Some(&built.refine));
            let loops = faces
                .first()
                .and_then(|g| g.trim_loops())
                .map(|l| l.iter().map(|lp| lp.iter().map(|p| p.v).collect()).collect())
                .unwrap_or_default();
            let context = HatchContext { perimeter: None, region: Region::Full, tiles: tiles.to_vec(), base: 0.0 };
            let mut edge: Vec<[f64; 3]> = built
                .surface
                .hatch(&LayerSource::TruncationEdge { tiled: true }, &context, Some(&built.refine))
                .map(|set| set.vertices.iter().map(|v| [v.x(), v.y(), v.z()]).collect())
                .unwrap_or_default();
            edge.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (loops, edge)
        };
        let (first, second) = (drawn(&[Affine2::IDENTITY, tile]), drawn(&[tile, Affine2::IDENTITY]));
        let open = second.0.iter().chain(&first.0).filter(|l| l.first() != l.last()).count();
        if !(!first.0.is_empty() && first.0 == second.0 && open == 0 && !first.1.is_empty() && first.1 == second.1) {
            failures.push(format!(
                "{name}: {open} loops open, loops {}, {} and {} vertices drawn, {}",
                if first.0 == second.0 { "the same" } else { "differ" },
                first.1.len(),
                second.1.len(),
                if first.1 == second.1 { "the same" } else { "different" }
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
