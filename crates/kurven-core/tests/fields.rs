//! Contours of a `ScalarField` on a parameter rectangle, held to closed
//! forms: the port of `Sources/kurven-test/Fields.swift`, with Swift's
//! tolerances.

use std::sync::{Arc, Mutex};

use kurven_core::grid::Grid2D;
use kurven_core::hatching::FaceSelector;
use kurven_core::manifest::{
    ContourField, Domain, HeightPolicy, Interval, LayerSource, Region, Restriction, Truncation,
};
use kurven_core::parametric::{ParamAxis, ParamRectangle, ParametricSurface};
use kurven_core::scalar_field::{self, ParamPaths, ScalarField, Trim};
use kurven_core::simd::Vec2;
use kurven_core::spaces::{DomainSpace, P2, P3, ParamSpace, WorldSpace};
use kurven_core::surface::{ContourRefine, Surface};
use kurven_core::{contour, manifest::Affine2};
use kurven_math::real;

const PI: f64 = core::f64::consts::PI;

/// The Hausdorff distance between two polylines in parameter coordinates,
/// taken on the flat torus where an axis has a period: each vertex of one to
/// the nearest point of the other, both ways.
fn hausdorff(a: &[P2<ParamSpace>], b: &[P2<ParamSpace>], periods: (Option<f64>, Option<f64>)) -> f64 {
    let directed = |from: &[P2<ParamSpace>], to: &[P2<ParamSpace>]| -> f64 {
        let mut worst: f64 = 0.0;
        for p in from {
            let mut best = f64::INFINITY;
            for w in to.windows(2) {
                let (s, t) = (w[0].v, w[1].v);
                let mut q = p.v;
                if let Some(pu) = periods.0 {
                    q.x -= pu * ((q.x - s.x) / pu).round();
                }
                if let Some(pv) = periods.1 {
                    q.y -= pv * ((q.y - s.y) / pv).round();
                }
                let d = t - s;
                let k =
                    if d.length_squared() > 0.0 { ((q - s).dot(d) / d.length_squared()).clamp(0.0, 1.0) } else { 0.0 };
                best = best.min((q - (s + k * d)).length());
            }
            worst = worst.max(best);
        }
        worst
    };
    directed(a, b).max(directed(b, a))
}

fn square(lo: f64, hi: f64, samples: usize) -> ParamRectangle {
    ParamRectangle::new(
        ParamAxis::new(Interval::new(lo, hi), false, samples),
        ParamAxis::new(Interval::new(lo, hi), false, samples),
    )
}

fn torus(u: usize, v: usize) -> ParamRectangle {
    ParamRectangle::new(ParamAxis::angle(u), ParamAxis::angle(v))
}

fn levels(field: &ScalarField, at: &[f64], over: ParamRectangle) -> Vec<(f64, ParamPaths)> {
    scalar_field::levels(field, at, over, &Trim::default())
}

/// "fields: the contours of u and v on the torus are its coordinate curves"
#[test]
#[ignore = "PORT: phase 4 (ParametricSurface.torus and its coordinate curves)"]
fn the_contours_of_u_and_v_on_the_torus_are_its_coordinate_curves() {
    let torus = ParametricSurface::torus(3.0, 1.0, (100, 50)).expect("the torus");
    let (u, v) = (torus.u, torus.v);
    let curves = torus.coordinate_curves((36, 18), 720);
    let u_levels: Vec<f64> = (0..36).map(|k| u.range.lo + u.range.length() * k as f64 / 36.0).collect();
    let v_levels: Vec<f64> = (0..18).map(|k| v.range.lo + v.range.length() * k as f64 / 18.0).collect();
    let rect = ParamRectangle::new(u, v);
    let us = levels(&ScalarField::new(Some(u.range.length()), |p| p.x()), &u_levels, rect);
    let vs = levels(&ScalarField::new(Some(v.range.length()), |p| p.y()), &v_levels, rect);
    let contours: Vec<ParamPaths> = us.into_iter().chain(vs).map(|(_, p)| p).collect();
    assert!(contours.iter().all(|c| c.len() == 1) && contours.len() == curves.count(), "one closed path per level");
    let mut worst: f64 = 0.0;
    for (k, paths) in contours.iter().enumerate() {
        worst = worst.max(hausdorff(&paths[0], curves.coords_of_path(k).unwrap(), (u.period(), v.period())));
    }
    assert!(worst < 1e-12 * 2.0 * PI, "each lies on its coordinate curve, to 1e-12 of the period: {worst}");
}

/// "fields: the contours of u and v on the torus are its coordinate curves",
/// its second half: a smooth field with no period, x² + y² on a square,
/// whose level 1/4 is the circle of radius 1/2.
#[test]
fn a_contour_of_x2_plus_y2_lies_on_its_circle() {
    let circle =
        levels(&ScalarField::new(None, |p| p.x() * p.x() + p.y() * p.y()), &[0.25], square(-1.0, 1.0, 41)).remove(0).1;
    let off = circle.iter().flatten().map(|p| (p.v.length() - 0.5).abs()).fold(0.0, f64::max);
    assert!(
        circle.len() == 1 && circle[0].first() == circle[0].last() && off < 1e-12,
        "a contour of x² + y² lies on its circle, to 1e-12: {} paths, {off} off",
        circle.len()
    );
}

/// "fields: a contour closes across the seam"
#[test]
fn a_contour_closes_across_the_seam() {
    // cos u + cos v = 1.2 is a loop round the rectangle's corner, which
    // the two seams cut into four pieces of the rectangle.
    let field = ScalarField::new(None, |p| real::cos(p.x()) + real::cos(p.y()));
    let paths = levels(&field, &[1.2], torus(64, 48)).remove(0).1;
    assert_eq!(paths.len(), 1, "one path");
    let lp = &paths[0];
    let off = lp.iter().map(|p| (real::cos(p.x()) + real::cos(p.y()) - 1.2).abs()).fold(0.0, f64::max);
    let step = lp.windows(2).map(|w| (w[1].v - w[0].v).length()).fold(0.0, f64::max);
    let cell = 2.0 * PI / 48.0;
    assert!(
        lp.first() == lp.last() && step < 2.0 * cell,
        "closed, its coordinates continuous across both seams: longest step {} cells",
        step / cell
    );
    assert!(lp.iter().any(|p| p.x() < 0.0) || lp.iter().any(|p| p.x() > PI), "and running round the corner");
    assert!(off < 1e-12, "on the level set: {off}");
}

/// "fields: a level through grid vertices is drawn once"
#[test]
fn a_level_through_grid_vertices_is_drawn_once() {
    // Seventy-two samples round u: every level of thirty-six is a column
    // of the grid, to rounding, and the first is the seam.
    let at: Vec<f64> = (0..36).map(|k| 2.0 * PI * k as f64 / 36.0).collect();
    let contours = levels(&ScalarField::new(Some(2.0 * PI), |p| p.x()), &at, torus(72, 30));
    let counts: Vec<usize> = contours.iter().map(|c| c.1.len()).collect();
    assert!(counts.iter().all(|&c| c == 1), "one path at each level: {counts:?}");
    let mut off: f64 = 0.0;
    for (level, paths) in &contours {
        for p in paths.iter().flatten() {
            off = off.max(real::remainder(p.x() - level, 2.0 * PI).abs());
        }
    }
    assert!(off < 1e-14, "on its level: {off}");

    // x + y = 1 on a grid of tenths runs corner to corner through the
    // vertices of the anti-diagonal.
    let line = levels(&ScalarField::new(None, |p| p.x() + p.y()), &[1.0], square(0.0, 1.0, 11)).remove(0).1;
    let ends = line.first().map(|l| {
        [l[0], l[l.len() - 1]]
            .map(|e| (e.v - Vec2::new(0.0, 1.0)).length().min((e.v - Vec2::new(1.0, 0.0)).length()))
            .into_iter()
            .fold(0.0, f64::max)
    });
    assert!(
        line.len() == 1 && ends.unwrap_or(1.0) < 1e-15 && line[0].iter().all(|p| (p.x() + p.y() - 1.0).abs() < 1e-15),
        "a diagonal through vertices is one path from corner to corner: {} paths",
        line.len()
    );
}

/// "fields: a level that touches nothing draws nothing"
#[test]
fn a_level_that_touches_nothing_draws_nothing() {
    let bowl = ScalarField::new(None, |p| p.x() * p.x() + p.y() * p.y());
    let outside = levels(&bowl, &[-1.0, 2.5], square(-1.0, 1.0, 21));
    assert!(outside.iter().all(|c| c.1.is_empty()), "a level beyond the field's range");

    // A circle-valued field over [-0.3, 0.3] has nothing at π, where the
    // difference from the level jumps by a period wherever the field is 0.
    let small = ScalarField::new(Some(2.0 * PI), |p| 0.3 * real::sin(p.x()));
    let opposite = levels(&small, &[PI], torus(40, 20)).remove(0).1;
    assert!(opposite.is_empty(), "a circle-valued level opposite every value: {} paths", opposite.len());
}

fn samples_of(n: usize, f: impl Fn(f64, f64) -> f64) -> Grid2D<f32> {
    let domain = Domain::new(Interval::new(-1.0, 1.0), Interval::new(-1.0, 1.0));
    let mut values = vec![0.0f32; n * n];
    for j in 0..n {
        for i in 0..n {
            let x = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
            let y = -1.0 + 2.0 * j as f64 / (n - 1) as f64;
            values[j * n + i] = f(x, y) as f32;
        }
    }
    Grid2D::new(n, n, domain, values)
}

fn in_parameters(paths: &[Vec<P2<DomainSpace>>]) -> ParamPaths {
    paths.iter().map(|p| p.iter().map(|q| P2::from_vec(q.v)).collect()).collect()
}

/// "fields: a field with a routine of its own is handed the grid's contours"
#[test]
fn a_field_with_a_routine_of_its_own_is_handed_the_grids_contours() {
    // Samples of x² + y² on a field whose value is something else: the
    // routine is handed the samples' contours, and its answer returned.
    let n = 41;
    let samples = Arc::new(samples_of(n, |x, y| x * x + y * y));
    let field = ScalarField::placed(
        samples.clone(),
        Arc::new(|level: f64, paths: &[Vec<P2<ParamSpace>>]| -> ParamPaths {
            paths.iter().map(|p| p.iter().map(|q| P2::new(q.x() + level, q.y())).collect()).collect()
        }),
    );
    let got = levels(&field, &[0.25], square(-1.0, 1.0, n)).remove(0).1;
    let want: ParamPaths = contour::lines(&samples, 0.25, None)
        .iter()
        .map(|p| p.iter().map(|q| P2::new(q.x() + 0.25, q.y())).collect())
        .collect();
    assert!(got == want, "the grid's contours, as the routine places them: {} paths vs {}", got.len(), want.len());
}

/// "fields: arg f hands the refiner the grid's own contours, and the branch cut at ±π"
#[test]
fn arg_f_hands_the_refiner_the_grids_own_contours_and_the_branch_cut() {
    // |z| and arg z on a grid that keeps off the origin. The refiner here
    // moves nothing, and says what it was asked: the field's routine is
    // handed what the grid has, so what comes back is what it was handed.
    let m = 40;
    let surface = Surface::new(
        samples_of(m, |x, y| real::sqrt(x * x + y * y)),
        Some(samples_of(m, |x, y| real::atan2(y, x))),
        Truncation::Uniform(0.8),
        false,
    );
    let asked: Arc<Mutex<Vec<(ContourField, f64, f64)>>> = Arc::new(Mutex::new(Vec::new()));
    let record = asked.clone();
    let unmoved = ContourRefine::new(
        Arc::new(move |field, level, paths: &[Vec<P2<DomainSpace>>], ceiling, _| {
            record.lock().unwrap().push((field, level, ceiling));
            paths.to_vec()
        }),
        None,
    );
    let phase = surface.phase.clone().unwrap();
    let at = ContourField::Phase.levels(&[-PI, 3.0, PI]);
    assert_eq!(at, vec![3.0, PI], "the branch cut is asked once, after the rest");

    // No period: at a level beside the branch cut the refiner is handed the
    // grid's contours whole, crossings along the wrap included, which
    // only it can tell from a level set beside a zero.
    let rectangle = surface.parameter_rectangle().unwrap();
    let field = surface.field(ContourField::Phase, Some(&unmoved), f64::INFINITY).unwrap();
    let placed = levels(&field, &at, rectangle);
    let grid = in_parameters(&contour::levels(&phase, &[3.0])[0].1);
    let along_wrap =
        placed[0].1.iter().flatten().filter(|p| p.x() < -0.2 && p.y().abs() < 1.0 / (m - 1) as f64).count();
    assert!(placed[0].1 == grid && along_wrap > 0, "the level 3 is the grid's, with its crossings along the wrap");
    let cut = in_parameters(&contour::cut(&phase));
    assert!(placed[1].1 == cut && !cut.is_empty(), "the level π is the branch cut, `contour::cut`");
    let bare = levels(&surface.field(ContourField::Phase, None, f64::INFINITY).unwrap(), &at, rectangle);
    assert!(bare[0].1 == grid && bare[1].1.is_empty(), "without a refiner the grid's contours, and no branch cut");

    // A landscape layer is its field's contours at the levels it asks
    // for, lifted; the refiner is handed the field's tag and the
    // layer's ceiling, a tenth over the truncation for curves lifted to
    // |f| and kept below it.
    asked.lock().unwrap().clear();
    let source = |field, levels: Vec<f64>, restriction| LayerSource::Contour {
        field,
        levels,
        restriction,
        faces: FaceSelector::Graph { tiled: false },
        light: None,
    };
    let layer = surface.derive(
        &source(ContourField::Phase, vec![-PI, 3.0, PI], Restriction::All),
        HeightPolicy::Level,
        &Region::Full,
        &[Affine2::IDENTITY],
        Some(&unmoved),
    );
    let lifted: Vec<Vec<P3<WorldSpace>>> = placed
        .iter()
        .flat_map(|(level, paths)| {
            paths
                .iter()
                .filter(|p| p.len() >= 2)
                .map(move |p| p.iter().map(|q| P3::new(q.x(), q.y(), *level)).collect())
        })
        .collect();
    let got: Vec<Vec<P3<WorldSpace>>> = (0..layer.count()).map(|k| layer.path(k).to_vec()).collect();
    assert!(got == lifted, "a phase layer is the field's contours, lifted at their levels");
    let _ = surface.derive(
        &source(ContourField::Magnitude, vec![0.5], Restriction::BelowTruncation),
        HeightPolicy::Magnitude,
        &Region::Full,
        &[Affine2::IDENTITY],
        Some(&unmoved),
    );
    let mut calls = asked.lock().unwrap().clone();
    calls.sort_by(|a, b| (a.0.raw(), a.1).partial_cmp(&(b.0.raw(), b.1)).unwrap());
    assert_eq!(
        calls,
        vec![
            (ContourField::Magnitude, 0.5, 1.1 * 0.8),
            (ContourField::Phase, 3.0, f64::INFINITY),
            (ContourField::Phase, PI, f64::INFINITY)
        ],
        "placed under the field's tag and the layer's ceiling"
    );
}

/// A regular polygon, as a trim loop.
fn polygon(c: Vec2, r: f64, n: usize) -> Vec<P2<ParamSpace>> {
    let mut corners: Vec<P2<ParamSpace>> = (0..n)
        .map(|k| {
            let a = 2.0 * PI * k as f64 / n as f64;
            P2::from_vec(c + r * Vec2::new(real::cos(a), real::sin(a)))
        })
        .collect();
    corners.push(corners[0]);
    corners
}

fn distance(p: P2<ParamSpace>, lp: &[P2<ParamSpace>]) -> f64 {
    lp.windows(2)
        .map(|w| {
            let (a, b) = (w[0].v, w[1].v);
            let d = b - a;
            let t = ((p.v - a).dot(d) / d.length_squared()).clamp(0.0, 1.0);
            (p.v - (a + t * d)).length()
        })
        .fold(f64::INFINITY, f64::min)
}

fn path_length(runs: &ParamPaths) -> f64 {
    runs.iter().map(|r| r.windows(2).map(|w| (w[1].v - w[0].v).length()).sum::<f64>()).sum()
}

fn trimmed_by(loops: Vec<Vec<P2<ParamSpace>>>) -> Trim {
    Trim { trim_loops: Some(loops), ..Trim::default() }
}

/// "fields: a contour is cut where it crosses a trim loop, and kept where the loops hold it even-odd"
#[test]
fn a_contour_is_cut_where_it_crosses_a_trim_loop() {
    let sq = square(-1.0, 1.0, 41);
    let x_field = ScalarField::new(None, |p| p.x());

    let disc = polygon(Vec2::ZERO, 1.0, 720);
    let chord = scalar_field::levels(&x_field, &[0.5], sq, &trimmed_by(vec![disc.clone()])).remove(0).1;
    let ends = chord.first().map(|c| distance(c[0], &disc).max(distance(c[c.len() - 1], &disc)));
    assert!(chord.len() == 1 && ends.unwrap_or(1.0) < 1e-15, "a chord of the disc ends on its loop: {ends:?}");

    // A hole narrower than a step, between two of the line's vertices:
    // x = 0.31 crosses rows a twentieth apart, at Im 0 and 0.05, and the
    // hole spans 0.005 to 0.045 between them. The square's edge is the
    // outer loop.
    let edge =
        vec![P2::new(-1.0, -1.0), P2::new(1.0, -1.0), P2::new(1.0, 1.0), P2::new(-1.0, 1.0), P2::new(-1.0, -1.0)];
    let hole = polygon(Vec2::new(0.31, 0.025), 0.02, 64);
    let split = scalar_field::levels(&x_field, &[0.31], sq, &trimmed_by(vec![edge.clone(), hole.clone()])).remove(0).1;
    let hole_ends: Vec<P2<ParamSpace>> =
        split.iter().flat_map(|r| [r[0], r[r.len() - 1]]).filter(|p| p.y().abs() < 0.5).collect();
    let across: Vec<f64> = hole
        .windows(2)
        .filter_map(|w| {
            let (a, b) = (w[0], w[1]);
            if !((a.x() - 0.31) * (b.x() - 0.31) <= 0.0 && a.x() != b.x()) {
                return None;
            }
            Some(a.y() + (0.31 - a.x()) * (b.y() - a.y()) / (b.x() - a.x()))
        })
        .collect();
    let gap =
        across.iter().copied().fold(f64::NEG_INFINITY, f64::max) - across.iter().copied().fold(f64::INFINITY, f64::min);
    let length = path_length(&split);
    assert!(
        split.len() == 2
            && hole_ends.len() == 2
            && hole_ends.iter().all(|&p| distance(p, &hole) < 1e-15)
            && (length - (2.0 - gap)).abs() < 1e-12,
        "x = 0.31 with a hole of radius 0.02 at (0.31, 0.025) is two runs, ended on the hole: {} runs, {length} long against {}",
        split.len(),
        2.0 - gap
    );

    // Where the loops are chords of a circle, the loops decide, not the
    // circle: every run of 400 lines across a 24-gon hole ends on it.
    let gon = polygon(Vec2::new(0.0, 0.013), 0.3, 24);
    let (mut gon_runs, mut gon_off) = (0, 0.0f64);
    for k in 0..400 {
        let level = -0.3 + 0.6 * (k as f64 + 0.5) / 400.0;
        let runs =
            scalar_field::levels(&x_field, &[level], sq, &trimmed_by(vec![edge.clone(), gon.clone()])).remove(0).1;
        for r in runs {
            gon_runs += 1;
            for e in [r[0], r[r.len() - 1]] {
                if e.y().abs() < 0.99 {
                    gon_off = gon_off.max(distance(e, &gon));
                }
            }
        }
    }
    assert!(
        gon_runs >= 800 && gon_off < 1e-15,
        "lines across a polygonal hole end on it: {gon_runs} runs, {gon_off} off"
    );

    // A boundary that runs along the line, given as a cut with the
    // membership it bounds: the line is cut where the stretch begins and
    // ends, not at the vertices either side.
    let along: ParamPaths = vec![vec![P2::new(0.31, 0.012), P2::new(0.31, 0.088)]];
    let aside = scalar_field::levels(
        &x_field,
        &[0.31],
        sq,
        &Trim { contains: Some(Arc::new(|p| !(p.y() > 0.012 && p.y() < 0.088))), cuts: along, ..Trim::default() },
    )
    .remove(0)
    .1;
    let mut aside_ends: Vec<f64> =
        aside.iter().flat_map(|r| [r[0].y(), r[r.len() - 1].y()]).filter(|y| y.abs() < 0.5).collect();
    aside_ends.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(
        aside.len() == 2 && aside_ends == [0.012, 0.088],
        "a boundary along the line cuts it where it begins and ends"
    );

    // A closed loop cut by a half-plane, bounded by a rectangle on the
    // cut, is one arc, on either side, whether the loop begins on the
    // cut, inside or outside: the ring begins at its lowest point,
    // (0, -1/2).
    let ring = ScalarField::new(None, |p| p.x() * p.x() + p.y() * p.y());
    let cell = 2.0 / 40.0;
    for cut in [0.0, 0.2, -0.2] {
        for (name, side) in [("right", 1.0), ("left", -1.0)] {
            let half = vec![
                P2::new(cut, -2.0),
                P2::new(cut, 2.0),
                P2::new(cut + 3.0 * side, 2.0),
                P2::new(cut + 3.0 * side, -2.0),
                P2::new(cut, -2.0),
            ];
            let arcs = scalar_field::levels(&ring, &[0.25], sq, &trimmed_by(vec![half])).remove(0).1;
            let ends = arcs.first().map(|a| (a[0].x() - cut).abs().max((a[a.len() - 1].x() - cut).abs()));
            // Within a rounding of the cut an end may lie either side.
            let kept = arcs.iter().all(|a| a[1..a.len() - 1].iter().all(|p| side * (p.x() - cut) >= 0.0));
            let step = arcs
                .iter()
                .map(|a| a.windows(2).map(|w| (w[1].v - w[0].v).length()).fold(0.0, f64::max))
                .fold(0.0, f64::max);
            assert!(
                arcs.len() == 1 && ends.unwrap_or(1.0) < 1e-15 && kept && step < 2.0 * cell,
                "the {name} of x = {cut} on a ring is one arc, ended on the cut: {} arcs, ends {ends:?} off, longest step {} cells",
                arcs.len(),
                step / cell
            );
        }
    }

    // A closed contour across the seam whose closing vertex lies a period
    // past its first, cut where its membership changes, through its
    // vertices: v = 1 on the torus, from u = 0 to 2π, where sin 2u is 0
    // and -4.9e-16. Kept where sin 2u >= 0 it is two quarters, not one
    // run with a chord across the quarter between; so too where the loop
    // begins inside the domain, or outside it. On a torus no loop bounds
    // anything even-odd, so the domain is `contains`, and the cuts are
    // lines of u, closed a period along v and met a period along u too.
    let t = torus(40, 30);
    type G = fn(f64) -> f64;
    let quarters: [(&str, G, [f64; 4]); 3] = [
        ("sin 2u >= 0, the loop beginning on its boundary", |u| real::sin(2.0 * u), [0.0, 0.5, 1.0, 1.5]),
        ("cos 2u >= 0, beginning inside", |u| real::cos(2.0 * u), [0.25, 0.75, 1.25, 1.75]),
        ("-cos 2u >= 0, beginning outside", |u| -real::cos(2.0 * u), [0.25, 0.75, 1.25, 1.75]),
    ];
    let v_field = ScalarField::new(Some(2.0 * PI), |p| p.y());
    for (name, g, zeros) in quarters {
        let lines: ParamPaths = zeros.iter().map(|&z| vec![P2::new(z * PI, 0.0), P2::new(z * PI, 2.0 * PI)]).collect();
        let runs = scalar_field::levels(
            &v_field,
            &[1.0],
            t,
            &Trim { contains: Some(Arc::new(move |p| g(p.x()) >= 0.0)), cuts: lines, ..Trim::default() },
        )
        .remove(0)
        .1;
        let length = path_length(&runs);
        let ends = runs.iter().flat_map(|r| [r[0], r[r.len() - 1]]).map(|p| g(p.x()).abs()).fold(0.0, f64::max);
        // A boundary that falls on a vertex ends the run there, once.
        let repeated = runs.iter().any(|r| r.windows(2).any(|w| w[0] == w[1]));
        assert!(
            runs.len() == 2 && (length - PI).abs() < 1e-9 && ends < 1e-14 && !repeated,
            "v = 1 kept where {name} is two quarters, ended on their boundaries: {} runs, {} π long, ends {ends} off",
            runs.len(),
            length / PI
        );
    }

    // The same lines a quarter cell off the vertices, so each cuts a
    // segment: u = π/2 + π/80 and so on, met a period along at the seam.
    let shifted: Vec<f64> = [0.5, 1.0, 1.5, 2.0].iter().map(|k| k * PI + PI / 80.0).collect();
    let off_lines: ParamPaths = shifted.iter().map(|&x| vec![P2::new(x, 0.0), P2::new(x, 2.0 * PI)]).collect();
    let off_runs = scalar_field::levels(
        &v_field,
        &[1.0],
        t,
        &Trim {
            contains: Some(Arc::new(|p| real::sin(2.0 * (p.x() - PI / 80.0)) >= 0.0)),
            cuts: off_lines,
            ..Trim::default()
        },
    )
    .remove(0)
    .1;
    let off_line =
        |x: f64| shifted.iter().map(|s| real::remainder(x - s, 2.0 * PI).abs()).fold(f64::INFINITY, f64::min);
    let off_ends = off_runs.iter().flat_map(|r| [r[0].x(), r[r.len() - 1].x()]).map(off_line).fold(0.0, f64::max);
    assert!(
        off_runs.len() == 2 && off_ends < 1e-14,
        "cut between vertices, a quarter cell off them, it is two quarters ended on the lines: {} runs, ends {off_ends} off",
        off_runs.len()
    );
}
