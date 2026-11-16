//! Faces.swift's suites that hold the faces to the drawing Hatch.swift
//! draws: "faces: the graph's trim loops are its truncation edge, closed,
//! and a contour on it is cut where it crosses them", over the face suites'
//! landscapes (`support::face_landscapes`) and the landscapes on f the suite
//! adds. Swift's tolerances.

// Swift's `!(d <= tolerance)` keeps a distance that is not a number outside.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

mod support;

use kurven_core::face::FaceID;
use kurven_core::hatch::HatchContext;
use kurven_core::manifest::{Affine2, ContourField, LayerSource, Region, Truncation, Walls};
use kurven_core::scalar_field::{self, ScalarField, Trim};
use kurven_core::simd::Vec2;
use kurven_core::spaces::{P2, P3, WorldSpace};
use support::{Landscape, Segments, bands, build, domain, face_landscapes, footprint};

/// What the trim-loop suite counts on one landscape's graph.
#[derive(Default)]
struct TrimTally {
    open: usize,
    drawn: usize,
    drawn_off: usize,
    worst_drawn: f64,
    asked: usize,
    misplaced: usize,
    worst_misplaced: f64,
    runs: usize,
    ends_off: usize,
    worst_end: f64,
    samples: usize,
    held: usize,
    misclipped: usize,
    riser_asked: usize,
    riser_missing: usize,
    examples: std::collections::BTreeMap<String, Vec<String>>,
}

impl TrimTally {
    fn note(&mut self, kind: &str, what: String) {
        let list = self.examples.entry(kind.to_string()).or_default();
        if list.len() < 3 {
            list.push(what);
        }
    }

    fn examples(&self, kind: &str) -> String {
        self.examples.get(kind).map(|l| l.join("; ")).unwrap_or_default()
    }
}

fn segments(paths: &[Vec<P2<kurven_core::spaces::ParamSpace>>], cell: f64) -> Segments {
    Segments::new(&paths.iter().map(|l| l.iter().map(|p| p.v).collect()).collect::<Vec<_>>(), cell)
}

/// The graph's trim loops held to the drawing, to membership, and as the
/// cut of a contour, on every copy. Membership by the loops is held to
/// `contains` except within `margin` cells of them; nor, without a refiner,
/// in a cell whose two columns read different truncation heights. A
/// contour's clip is held to the loops themselves, which decide it, to a
/// billionth of a cell.
fn trim_loops(l: &Landscape, margin: f64) -> Option<TrimTally> {
    let graph = l.faces.first()?;
    if graph.id != FaceID::Graph {
        return None;
    }
    let loops = graph.trim_loops()?;
    let surface = &l.surface;
    let (real, imag) = (surface.domain().real, surface.domain().imag);
    let cell = (real.length().abs() / (surface.height.width - 1) as f64)
        .max(imag.length().abs() / (surface.height.height - 1) as f64);
    let tolerance = 1e-9 * cell;
    let own = segments(loops, cell);
    let grid = &*surface.height;
    let columns: Vec<f64> = (0..grid.width).map(|x| grid.position(x, 0).x()).collect();
    let straddled: Vec<usize> = (0..grid.width - 1)
        .filter(|&i| surface.truncation.height_at(columns[i]) != surface.truncation.height_at(columns[i + 1]))
        .collect();
    let on_f = l.refine.as_ref().is_some_and(|r| r.magnitude.is_some());
    let straddles = |p: Vec2| -> bool {
        if on_f {
            return false;
        }
        straddled.iter().any(|&i| {
            let (a, b) = (columns[i].min(columns[i + 1]), columns[i].max(columns[i + 1]));
            p.x >= a && p.x <= b
        })
    };
    let mut t = TrimTally::default();
    // The open chains as `graph_trim_loops` counts them, and the face's
    // loops its, with that many open.
    let reported = surface.graph_trim_loops(l.refine.as_ref());
    t.open = reported.open;
    let reported_loops: Vec<Vec<Vec2>> = reported.loops.iter().map(|lp| lp.iter().map(|p| p.v).collect()).collect();
    let face_loops: Vec<Vec<Vec2>> = loops.iter().map(|lp| lp.iter().map(|p| p.v).collect()).collect();
    if reported_loops != face_loops || loops.iter().filter(|lp| lp.first() != lp.last()).count() != reported.open {
        t.open += 1;
        t.note("drawn", "the face's loops are not graph_trim_loops', or not as many open".into());
    }

    // On f, each riser's top drawn wherever the lower band's plateau holds
    // its boundary.
    if let (Some(magnitude), Truncation::RealBands { bands, .. }) =
        (l.refine.as_ref().and_then(|r| r.magnitude.as_ref()), &surface.truncation)
    {
        let (lo, hi) = (real.lo.min(real.hi), real.lo.max(real.hi));
        let tops: Vec<&Vec<P3<WorldSpace>>> =
            reported.truncation_edge.iter().filter(|p| p.len() >= 2 && p.iter().all(|v| v.x() == p[0].x())).collect();
        let mut boundaries: Vec<f64> = bands.iter().map(|b| b.below).filter(|&b| b > lo && b < hi).collect();
        boundaries.sort_by(|a, b| a.partial_cmp(b).unwrap());
        boundaries.dedup();
        for b in boundaries {
            let left = surface.truncation.height_at(b.next_down());
            let right = surface.truncation.height_at(b);
            let lesser = left.min(right);
            if !(lesser.is_finite() && lesser < left.max(right)) {
                continue;
            }
            for j in 0..97 {
                let y = imag.lo + imag.length() * (j as f64 + 0.5) / 97.0;
                let u = magnitude(P2::new(b, y));
                if !((u - lesser).abs() > 1e-3 * lesser) {
                    continue;
                }
                t.riser_asked += 1;
                let drawn = tops.iter().any(|p| {
                    let ys = p.iter().map(|v| v.y());
                    p[0].x() == b
                        && y >= ys.clone().fold(f64::INFINITY, f64::min)
                        && y <= ys.fold(f64::NEG_INFINITY, f64::max)
                });
                if drawn != (u > lesser) {
                    t.riser_missing += 1;
                    t.note(
                        "riser",
                        format!(
                            "({b}, {y}), |f| = {u} against {lesser}, {}",
                            if drawn { "a top drawn" } else { "no top" }
                        ),
                    );
                }
            }
        }
    }

    // The truncation edge as the layer draws it, replicated and cut to the
    // footprint: each vertex on the loops, carried back by some tile. Less
    // the risers' bottoms it draws with it, which lie a hair onto the lower
    // side, each vertex within two hairs of the loops, and on them at most
    // where another loop meets the riser.
    let context = HatchContext { perimeter: None, region: l.region.clone(), tiles: l.tiles.clone(), base: 0.0 };
    if let Some(drawn) = surface.hatch(&LayerSource::TruncationEdge { tiled: true }, &context, l.refine.as_ref()) {
        let distance = |v: P3<WorldSpace>| -> f64 {
            l.tiles
                .iter()
                .filter_map(|t| t.inverse())
                .map(|back| own.distance(back.apply(v).xy().v, cell))
                .fold(f64::INFINITY, f64::min)
        };
        let paths: Vec<&[P3<WorldSpace>]> = drawn
            .paths()
            .filter(|path| {
                let d: Vec<f64> = path.iter().map(|&v| distance(v)).collect();
                !(d.iter().all(|&x| x < 2.0 * tolerance) && d.iter().any(|&x| x > 0.5 * tolerance))
            })
            .collect();
        for &v in paths.iter().flat_map(|p| p.iter()) {
            t.drawn += 1;
            let d = distance(v);
            if !(d <= tolerance) {
                t.drawn_off += 1;
                t.worst_drawn = t.worst_drawn.max(d);
                t.note("drawn", format!("drawn vertex {v:?} {} cell(s) off the loops", d / cell));
            }
        }
    }

    for copy in &graph.copies {
        // Membership: even-odd over the face's loops, and the footprint where
        // the copy stands.
        for j in 0..47 {
            for i in 0..83 {
                let p = Vec2::new(
                    real.lo + real.length() * (i as f64 + 0.37) / 83.0,
                    imag.lo + imag.length() * (j as f64 + 0.61) / 47.0,
                );
                if own.distance(p, margin * cell).is_finite() || straddles(p) {
                    continue;
                }
                t.asked += 1;
                let by_loops = own.inside(p) && l.region.contains(copy.tile.apply(P3::new(p.x, p.y, 0.0)).xy());
                if by_loops != (copy.contains)(P2::from_vec(p)) {
                    t.misplaced += 1;
                    t.worst_misplaced = t.worst_misplaced.max(own.distance(p, 10.0 * cell) / cell);
                    t.note(
                        "placed",
                        format!("{p:?} on tile {:?}: the loops say {by_loops}, contains {}", copy.tile, !by_loops),
                    );
                }
            }
        }

        // A contour on the copy: lines of x and of y across the rectangle,
        // and level sets of |f| about the truncation, which curve and run
        // along the truncation edge.
        let Some(copy_loops) = copy.trim_loops() else { continue };
        let cut = segments(&copy_loops, cell);
        let excess = graph.truncation_excess.clone();
        let held = |q: Vec2| -> bool {
            excess.as_ref().map_or_else(|| own.inside(q), |e| e(P2::from_vec(q)) < 0.0)
                && l.region.contains(copy.tile.apply(P3::new(q.x, q.y, 0.0)).xy())
        };
        let mut families: Vec<(ScalarField, Vec<f64>)> = vec![
            (
                ScalarField::new(None, |p| p.x()),
                (1..=13).map(|k| real.lo + real.length() * (k as f64 - 0.29) / 13.0).collect(),
            ),
            (
                ScalarField::new(None, |p| p.y()),
                (1..=9).map(|k| imag.lo + imag.length() * (k as f64 - 0.43) / 9.0).collect(),
            ),
        ];
        let mut heights: Vec<f64> = surface
            .truncation
            .bands(real.lo.min(real.hi), real.lo.max(real.hi))
            .iter()
            .map(|s| s.height)
            .filter(|h| h.is_finite())
            .collect();
        heights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        heights.dedup();
        if let Some(magnitude) = surface.field(ContourField::Magnitude, l.refine.as_ref(), f64::INFINITY)
            && !heights.is_empty()
        {
            families.push((magnitude, heights.iter().flat_map(|h| [0.6, 0.93, 1.07, 1.4].map(|k| k * h)).collect()));
        }
        for (field, levels) in &families {
            let whole = scalar_field::levels(field, levels, graph.domain, &Trim::default());
            let clipped = graph.contours(field, levels, copy);
            let paths: Vec<&Vec<P2<_>>> = whole.iter().flat_map(|(_, p)| p).collect();
            let runs: Vec<&Vec<P2<_>>> = clipped.iter().flat_map(|(_, p)| p).collect();
            let path_ends: Vec<P2<_>> = paths.iter().flat_map(|p| [p[0], p[p.len() - 1]]).collect();
            let kept = Segments::new(&runs.iter().map(|r| r.iter().map(|p| p.v).collect()).collect::<Vec<_>>(), cell);
            t.runs += runs.len();
            for run in &runs {
                for e in [run[0], run[run.len() - 1]] {
                    if path_ends.contains(&e) {
                        continue;
                    }
                    let d = cut.distance(e.v, cell);
                    if !(d <= tolerance) && !excess.as_ref().is_some_and(|x| x(e).abs() <= 1e-9) {
                        t.ends_off += 1;
                        t.worst_end = t.worst_end.max(d / cell);
                        t.note("clipped", format!("a run ends at {:?}, {} cell(s) off the loops", e.v, d / cell));
                    }
                }
            }
            for path in &paths {
                for w in path.windows(2) {
                    let (a, b) = (w[0], w[1]);
                    for k in 0..16 {
                        let q = a.v + (b.v - a.v) * ((k as f64 + 0.5) / 16.0);
                        if cut.distance(q, tolerance).is_finite() {
                            continue;
                        }
                        t.samples += 1;
                        let h = held(q);
                        if h {
                            t.held += 1;
                        }
                        let on_run = kept.distance(q, cell) <= tolerance;
                        if on_run != h {
                            t.misclipped += 1;
                            t.note(
                                "clipped",
                                format!(
                                    "{q:?} on tile {:?} {}, the loops say otherwise",
                                    copy.tile,
                                    if on_run { "kept" } else { "cut away" }
                                ),
                            );
                        }
                    }
                }
            }
        }
    }
    Some(t)
}

#[test]
fn the_graphs_trim_loops_are_its_truncation_edge_closed_and_a_contour_on_it_is_cut_where_it_crosses_them() {
    let mut landscapes: Vec<Landscape> = face_landscapes().into_iter().filter(|l| l.has_graph()).collect();
    let w = domain(-0.5, 1.5, -1.0, 1.0);
    let on_f = |name: &str, expression: &str, d, resolution, truncation| {
        let built = build(expression, d, resolution, truncation);
        Landscape::new(name, built.surface, vec![Affine2::IDENTITY], Region::Full, Walls::None, Some(built.refine))
    };
    landscapes.push(on_f("a tower banded at 2 | 4, on f", "1/(z - 0.5)", w, 32, bands(&[(0.5, 2.0)], 4.0)));
    landscapes.push(on_f(
        "a tower banded at 2 | 4 over Im [-0.3, 1], on f",
        "1/(z - 0.5)",
        domain(-0.5, 1.5, -0.3, 1.0),
        32,
        bands(&[(0.5, 2.0)], 4.0),
    ));
    let alternating: Vec<(f64, f64)> = (1..=6).map(|k| (k as f64, if k % 2 == 1 { 2.0 } else { 4.0 })).collect();
    landscapes.push(on_f(
        "1/sin(πz) banded at 2 | 4 by turns through its poles, on f",
        "1/sin(pi*z)",
        domain(-0.5, 6.5, -1.0, 1.0),
        224,
        bands(&alternating, 2.0),
    ));
    landscapes.push(on_f(
        "1/(z - 0.4) banded at 2 | 20, its plateau across the boundary, on f",
        "1/(z - 0.4)",
        w,
        32,
        bands(&[(0.5, 2.0)], 20.0),
    ));
    for (expression, resolution) in [("1/(z + 0.03)", 32), ("1/(z + 0.01)", 200)] {
        landscapes.push(on_f(
            &format!("{expression} at res {resolution} banded at 2 | 4, a column short of the boundary, on f"),
            expression,
            w,
            resolution,
            bands(&[(0.5, 2.0)], 4.0),
        ));
    }
    landscapes.push(on_f("the reviewer's tower at 0.5 | 1, on f", "1/(z - 1.4)", w, 32, bands(&[(1.45, 0.5)], 1.0)));
    let hole = "1/(z - (0.4804 - 0.0550i)) + 1/(z - (0.5196 + 0.0550i))";
    for (re, im) in [((-0.3, 1.5), (-0.5, 0.5)), ((0.3, 1.0), (-1.0, 0.5))] {
        landscapes.push(on_f(
            &format!("a hole round a zero on a band boundary at 1.5 | 1.8 over Re {re:?}, Im {im:?}, on f"),
            hole,
            domain(re.0, re.1, im.0, im.1),
            24,
            bands(&[(0.5, 1.5)], 1.8),
        ));
    }
    landscapes.push(on_f(
        "a zero's ring over a band boundary from the band beside, at 1.5 | 1.8, on f",
        "1/(z - (0.482 + 0.0853333i)) + 1/(z - (0.482 - 0.2146667i))",
        w,
        16,
        bands(&[(0.5, 1.5)], 1.8),
    ));
    landscapes.push(on_f(
        "1/(z³ - 1) truncated at 5, on f",
        "1/(z^3 - 1)",
        domain(-1.3, 1.6, -1.2, 1.5),
        120,
        Truncation::Uniform(5.0),
    ));
    if let Some(hatch) =
        landscapes.iter().find(|l| l.name == "the hatch fixture, low bands, in four tiles, on a footprint across them")
    {
        let besides = vec![Affine2::IDENTITY, Affine2 { a: 1.0, b: 0.0, tx: 4.5, c: 0.0, d: 1.0, ty: 0.0 }];
        let across = footprint(&[(-1.5, -0.6), (6.4, -0.9), (6.1, 1.3), (-1.7, 1.4)], 9);
        let l = Landscape::new(
            "the hatch fixture, low bands, in two tiles side by side, on a footprint across them",
            hatch.surface.clone(),
            besides,
            Region::Inside(across),
            Walls::None,
            None,
        );
        landscapes.push(l);
    }
    let mut failures = Vec::new();
    for l in &landscapes {
        // How far the loops and `contains` may part: measured, not bounded
        // (0.015 cells on Γ, 0.043 on the tower, 0.16 on the hatch fixture).
        let margin = if l.refine.is_none() { 0.25 } else { 0.05 };
        let Some(t) = trim_loops(l, margin) else {
            if !l.faces.is_empty() {
                failures.push(format!("{}: the graph has trim loops", l.name));
            }
            continue;
        };
        if t.riser_asked > 0 && t.riser_missing != 0 {
            failures.push(format!(
                "{}: each riser's top is drawn where the lower band's plateau holds its boundary: {} of {} wrong: {}",
                l.name,
                t.riser_missing,
                t.riser_asked,
                t.examples("riser")
            ));
        }
        if !(t.open == 0 && t.drawn_off == 0) {
            failures.push(format!(
                "{}: the graph's trim loops are closed, and the truncation edge drawn lies on them: {} open, {} of {} drawn vertices off, worst {}: {}",
                l.name,
                t.open,
                t.drawn_off,
                t.drawn,
                t.worst_drawn,
                t.examples("drawn")
            ));
        }
        if !(t.asked > 1000 && t.misplaced == 0) {
            failures.push(format!(
                "{}: a point the loops and the footprint hold is in the graph: {} of {} disagree, worst {} cells: {}",
                l.name,
                t.misplaced,
                t.asked,
                t.worst_misplaced,
                t.examples("placed")
            ));
        }
        // A footprint that holds none of the graph keeps no run of it.
        if !((t.runs > 0 || t.held == 0) && t.ends_off == 0 && t.samples > 200 && t.misclipped == 0) {
            failures.push(format!(
                "{}: contours on each copy are cut where they cross its loops: {} of {} runs end off (worst {} cells), {} of {} samples misclipped: {}",
                l.name,
                t.ends_off,
                t.runs,
                t.worst_end,
                t.misclipped,
                t.samples,
                t.examples("clipped")
            ));
        }
    }
    assert!(failures.is_empty(), "{} landscapes: {failures:#?}", landscapes.len());
}
