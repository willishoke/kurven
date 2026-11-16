//! Chords.swift's suites on a landscape built with the refiner: "chords: a
//! contour on a truncated graph ends on the truncation edge, and no vertex
//! stands over it", its families of u, v and the height, and apart its
//! isophotes and the graph's hatchings, plain and half-Lambert; "chords: the
//! plain routine's chords stay within its tolerance of the level set, and of
//! the graph", its cone and bowl, and apart the bowl's isophotes; and
//! "chords: on the graph of sin z, every isophote cut on the truncation
//! edge, every chord near its level, no run split". Swift's tolerances.

#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::type_complexity)]

mod support;

use std::collections::HashSet;
use std::f64::consts::{FRAC_PI_2, PI};

use kurven_core::bundle::KurvenBundle;
use kurven_core::hatch::HatchContext;
use kurven_core::hatching::{FaceSelector, Hatching, Tone};
use kurven_core::manifest::{Affine2, ContourField, Region, Restriction, Truncation};
use kurven_core::scalar_field::CHORD_TOLERANCE;
use kurven_core::simd::{Vec2, Vec3};
use kurven_core::spaces::{P2, P3, ParamSpace, WorldSpace};
use kurven_landscape::{LandscapeRequest, NativeLandscape};
use kurven_math::{real, swift};
use support::{Built, bands, build, domain};

fn on_domain_edge(p: P3<WorldSpace>) -> bool {
    (p.x().abs() - 2.0).abs() <= 1e-12 || (p.y().abs() - 2.0).abs() <= 1e-12
}

/// 1/(z² − 1) cut at 2/3 at 32 samples, with the occluder
/// `NativeLandscape.build` writes: walls on the domain's rectangle, one
/// tile, the whole region.
fn peanut() -> (Built, HatchContext) {
    let square = domain(-2.0, 2.0, -2.0, 2.0);
    let built = build("1/(z^2 - 1)", square, 32, Truncation::Uniform(2.0 / 3.0));
    let density = NativeLandscape::wall_density(square, built.shape).unwrap();
    let context = HatchContext {
        perimeter: Some(NativeLandscape::rectangle(square, &density)),
        region: Region::Full,
        tiles: vec![Affine2::IDENTITY],
        base: 0.0,
    };
    (built, context)
}

const PEANUT_LINES: [f64; 8] = [-1.5, -1.2, -0.5, -0.05, 0.03, 0.1, 0.4, 1.3];

/// The suite's verdict on each family: no vertex over the truncation, and,
/// where `wanted` is given, at least that many ends inside the domain, each
/// on |f| = 2/3; a line of u on its own line. A failure's text for each
/// family that fails.
fn peanut_verdict(families: Vec<(&str, Vec<Vec<P3<WorldSpace>>>, Option<usize>)>) -> Vec<String> {
    let h = 2.0 / 3.0;
    let f = |p: P3<WorldSpace>| 1.0 / real::hypot(p.x() * p.x() - p.y() * p.y() - 1.0, 2.0 * p.x() * p.y());
    let mut failures = Vec::new();
    for (name, paths, wanted) in families {
        let (mut over, mut off_edge, mut off_line, mut ends, mut vertices) = (0.0f64, 0.0f64, 0.0f64, 0, 0);
        for path in &paths {
            for &p in path {
                vertices += 1;
                over = over.max(f(p).max(p.z()) - h);
            }
            // A line of u is cut on its own line: its ends are where Re z = c
            // meets |z² − 1| = 3/2.
            if name == "u" {
                let worst = path
                    .iter()
                    .map(|p| PEANUT_LINES.iter().map(|c| (p.x() - c).abs()).fold(f64::INFINITY, f64::min))
                    .fold(f64::NEG_INFINITY, f64::max);
                off_line = off_line.max(worst);
            }
            if wanted.is_none() || path[0] == path[path.len() - 1] {
                continue;
            }
            for end in [path[0], path[path.len() - 1]] {
                if !on_domain_edge(end) {
                    ends += 1;
                    off_edge = off_edge.max((f(end) - h).abs());
                }
            }
        }
        if !(over <= 1e-12 * h && off_edge <= 1e-12 * h && off_line <= 1e-15 && ends >= wanted.unwrap_or(0)) {
            let what = if wanted.is_none() { "" } else { ", every end inside the domain on |f| = 2/3" };
            failures.push(format!(
                "{name}: no vertex over the truncation{what}: {vertices} vertices, worst {over:.1e} over; {ends} ends, \
                 worst {off_edge:.1e} off, {off_line:.1e} off its line"
            ));
        }
    }
    failures
}

/// 1/(z² − 1) cut at 2/3: the plateau is the Cassini oval |z² − 1| < 3/2, a
/// peanut, convex at its ends and pinched at its waist, so the truncation
/// edge's chords lie over the plateau at the ends and over the graph at the
/// waist. A cut on a chord stood over the plateau at the ends and short of
/// the edge at the waist.
#[test]
fn a_contour_on_a_truncated_graph_ends_on_the_truncation_edge_and_no_vertex_stands_over_it() {
    let (built, context) = peanut();
    let (surface, refine) = (&built.surface, Some(&built.refine));
    let graph = FaceSelector::Graph { tiled: false };
    let failures = peanut_verdict(vec![
        (
            "u",
            surface.contours(ContourField::U, &PEANUT_LINES, &graph, &Restriction::All, &context, refine, None),
            Some(16),
        ),
        (
            "v",
            surface.contours(ContourField::V, &[-0.6, 0.2, 0.5], &graph, &Restriction::All, &context, refine, None),
            Some(6),
        ),
        (
            "the height",
            surface.contours(ContourField::Height, &[0.4, 0.6], &graph, &Restriction::All, &context, refine, None),
            Some(0),
        ),
    ]);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The same suite's isophotes (`Face.isophotes`) and the graph's hatchings
/// (`Surface.graphHatching`, `Face.coordinateHatching`), plain and
/// half-Lambert: each family's vertices under the truncation, and the
/// isophotes' and plain hatching's ends inside the domain on |f| = 2/3; a
/// half-Lambert piece also ends where its width changes, so its ends are
/// not counted.
#[test]
fn its_isophotes_and_graph_hatchings_end_on_the_truncation_edge() {
    let (built, context) = peanut();
    let (surface, refine) = (&built.surface, Some(&built.refine));
    let light = Vec3::new(-0.4, -0.6, 0.7);
    let flat = |steps: Vec<(i64, Vec<Vec<P3<WorldSpace>>>)>| steps.into_iter().flat_map(|(_, p)| p).collect();
    let failures = peanut_verdict(vec![
        (
            "isophotes",
            surface.contours(
                ContourField::Isophote,
                &[-0.5, -0.25, 0.0, 0.25, 0.5, 0.75],
                &FaceSelector::Graph { tiled: false },
                &Restriction::All,
                &context,
                refine,
                Some(light),
            ),
            Some(10),
        ),
        ("hatching", flat(surface.graph_hatching(&Hatching::plain(0.0, 0.1), false, &context, refine)), Some(30)),
        (
            "half-Lambert hatching",
            flat(surface.graph_hatching(
                &Hatching::new(FRAC_PI_2, 0.1, None, Tone::HalfLambert { light }, 4),
                false,
                &context,
                refine,
            )),
            None,
        ),
    ]);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Chords.swift's `landscape(_:resolution:truncation:)`: f over [-2, 2]² as
/// `NativeLandscape.build` makes it with the refiner and no layers.
fn landscape(expression: &str, resolution: i64, truncation: Truncation) -> KurvenBundle {
    let mut request = LandscapeRequest::new(expression, domain(-2.0, 2.0, -2.0, 2.0), resolution);
    request.truncation = Some(truncation);
    request.layers = Some(Vec::new());
    NativeLandscape::build(&request, "native", true).unwrap()
}

/// "chords: the plain routine's chords stay within its tolerance of the
/// level set, and of the graph": the cone and the bowl.
#[test]
fn the_plain_routines_chords_stay_within_its_tolerance() {
    let tolerance = CHORD_TOLERANCE;
    // The height on the graph of z, a cone: its level sets are circles
    // |z| = r, and the gradient of |z| has unit length, so a midpoint's
    // distance from its circle is ||m| − r|, which the routine estimates
    // exactly.
    let cone = landscape("z", 16, Truncation::None);
    let cell = 4.0 / 15.0;
    let radii = [0.4, 0.7, 1.3, 1.9];
    let (mut vertex_off, mut chord_off, mut chords) = (0.0f64, 0.0f64, 0);
    let graph = FaceSelector::Graph { tiled: false };
    let context = HatchContext::new(&cone.manifest.occluder);
    for &r in &radii {
        for path in cone.surface.contours(
            ContourField::Height,
            &[r],
            &graph,
            &Restriction::All,
            &context,
            cone.refine.as_ref(),
            None,
        ) {
            vertex_off = vertex_off.max(path.iter().map(|p| (real::hypot(p.x(), p.y()) - r).abs()).fold(0.0, f64::max));
            for w in path.windows(2) {
                chords += 1;
                let (a, b) = (w[0], w[1]);
                chord_off = chord_off.max((real::hypot(0.5 * (a.x() + b.x()), 0.5 * (a.y() + b.y())) - r).abs());
            }
        }
    }
    assert!(
        vertex_off <= 1e-12 && chord_off <= tolerance * cell && chords > 40,
        "the height on the graph of z: every chord's midpoint within {tolerance} of a cell of its circle: \
         {chords} chords, worst {:.4} of a cell; vertices off by {vertex_off:.1e}",
        chord_off / cell
    );

    // The graph of z², |f| = x² + y²: a line of u = c lies on the parabola
    // z = c² + y², and a chord from y₁ to y₂ sags (y₁ − y₂)²/4 off it,
    // a fifteenth of a cell across a whole cell.
    let bowl = landscape("z^2", 16, Truncation::None);
    let context = HatchContext::new(&bowl.manifest.occluder);
    let (mut sag, mut closed_sag) = (0.0f64, 0.0f64);
    chords = 0;
    for path in bowl.surface.contours(
        ContourField::U,
        &[-1.3, -0.5, 0.2, 0.9],
        &graph,
        &Restriction::All,
        &context,
        bowl.refine.as_ref(),
        None,
    ) {
        for w in path.windows(2) {
            chords += 1;
            let m = 0.5 * (w[0].v + w[1].v);
            sag = sag.max((m.z - (m.x * m.x + m.y * m.y)).abs());
            closed_sag = closed_sag.max((w[0].y() - w[1].y()) * (w[0].y() - w[1].y()) / 4.0);
        }
    }
    assert!(
        sag <= tolerance * cell && (sag - closed_sag).abs() <= 1e-12 && chords > 40,
        "a line of u on the graph of z²: every chord's midpoint within {tolerance} of a cell of the graph: \
         {chords} chords, worst sag {:.4} of a cell",
        sag / cell
    );
}

/// The same suite's last part: the graph of z², |f| = x² + y², lit along z,
/// where n · l = 1/√(1 + 4r²), so the isophote at c is the circle
/// r = √(1/c² − 1)/2: each chord's midpoint within the tolerance of a cell
/// of its circle.
#[test]
fn the_isophotes_of_the_graph_of_z2_stay_within_the_chord_tolerance() {
    let tolerance = CHORD_TOLERANCE;
    let cell = 4.0 / 15.0;
    let bowl = landscape("z^2", 16, Truncation::None);
    let graph = bowl.surface.graph(&[Affine2::IDENTITY], &Region::Full, bowl.refine.as_ref()).expect("z² has a graph");
    let levels = [0.3, 0.5, 0.7, 0.9];
    let (mut isophote_off, mut chords) = (0.0f64, 0);
    let contoured = graph.isophotes(Vec3::new(0.0, 0.0, 1.0), &levels, &graph.copies[0]);
    for (k, (_, paths)) in contoured.iter().enumerate() {
        let r = real::sqrt(1.0 / (levels[k] * levels[k]) - 1.0) / 2.0;
        for path in paths {
            for w in path.windows(2) {
                chords += 1;
                isophote_off = swift::max(isophote_off, ((0.5 * (w[0].v + w[1].v)).length() - r).abs());
            }
        }
    }
    assert!(
        isophote_off <= tolerance * cell && chords > 40,
        "the isophotes of the graph of z² lit along z: every chord's midpoint within {tolerance} of a cell of its \
         circle: {chords} chords, worst {:.4} of a cell; levels drawn {:?}",
        isophote_off / cell,
        contoured.iter().map(|(c, _)| *c).collect::<Vec<_>>()
    );
}

/// Swift's `stride(from:through:by:)` on doubles: `from + k · by` for each
/// k while it is not past `through`.
fn stride_through(from: f64, through: f64, by: f64) -> Vec<f64> {
    let mut out = Vec::new();
    let mut k = 0.0;
    loop {
        let x = from + k * by;
        if !(x <= through) {
            break;
        }
        out.push(x);
        k += 1.0;
    }
    out
}

/// Pairs of runs where one ends where the next begins: one curve split.
fn abutting(runs: &[Vec<P2<ParamSpace>>]) -> usize {
    // Swift's `Set<[Double]>`: −0 and 0 are one key.
    let key = |p: P2<ParamSpace>| ((p.x() + 0.0).to_bits(), (p.y() + 0.0).to_bits());
    let starts: HashSet<(u64, u64)> = runs.iter().map(|r| key(r[0])).collect();
    runs.iter().filter(|r| r[0] != r[r.len() - 1] && starts.contains(&key(r[r.len() - 1]))).count()
}

/// "chords: on the graph of sin z, every isophote cut on the truncation
/// edge, every chord near its level, no run split": isophotes of sin z cut
/// at 3/2, at 40 samples cut on the truncation edge and never split; at 32
/// under a low light each vertex on its level and each chord's midpoint near
/// it, held to n · l in closed form; and Γ in bands, its closed isophote
/// round a pole cut by the truncation alone one run.
#[test]
fn on_the_graph_of_sin_z_every_isophote_is_cut_on_the_truncation_edge() {
    // |sin z|, and the graph's unit normal in closed form: for analytic f,
    // ∇|f| = (Re f̄f′, −Im f̄f′)/|f|.
    let f = |p: Vec2| (real::sin(p.x) * real::cosh(p.y), real::cos(p.x) * real::sinh(p.y));
    let magnitude = |p: Vec2| {
        let w = f(p);
        real::hypot(w.0, w.1)
    };
    let lit = |p: Vec2, l: Vec3| {
        let w = f(p);
        let d = (real::cos(p.x) * real::cosh(p.y), -real::sin(p.x) * real::sinh(p.y));
        let m = real::hypot(w.0, w.1);
        let g = Vec2::new(w.0 * d.0 + w.1 * d.1, -(w.0 * d.1 - w.1 * d.0)) / m;
        (Vec3::new(-g.x, -g.y, 1.0) / real::sqrt(1.0 + g.length_squared())).dot(l.normalized())
    };
    let window = domain(-5.0, 5.0, -1.5, 1.5);
    let on_edge = |p: P2<ParamSpace>| (p.x().abs() - 5.0).abs() <= 1e-12 || (p.y().abs() - 1.5).abs() <= 1e-12;
    let h = 1.5;
    let sine = |resolution: i64| {
        let mut request = LandscapeRequest::new("sin(z)", window, resolution);
        request.truncation = Some(Truncation::Uniform(h));
        request.layers = Some(Vec::new());
        NativeLandscape::build(&request, "native", true).unwrap()
    };
    let mut failures = Vec::new();
    // At 40 samples a cut lies within 1e-4 of the next vertex, where its own
    // crossing rounds past the loops' filter.
    let coarse = sine(40);
    let graph = coarse.surface.graph(&[Affine2::IDENTITY], &Region::Full, coarse.refine.as_ref()).unwrap();
    let light = Vec3::new(real::cos(PI / 8.0) * real::cos(1.1), real::sin(PI / 8.0) * real::cos(1.1), real::sin(1.1));
    let (mut over, mut off_edge, mut ends, mut split) = (0.0f64, 0.0f64, 0, 0);
    for (_, paths) in graph.isophotes(light, &stride_through(-0.95, 0.95, 0.05), &graph.copies[0]) {
        split += abutting(&paths);
        for path in &paths {
            for p in path {
                over = swift::max(over, magnitude(p.v) - h);
            }
            if path[0] == path[path.len() - 1] {
                continue;
            }
            for end in [path[0], path[path.len() - 1]] {
                if !on_edge(end) {
                    ends += 1;
                    off_edge = swift::max(off_edge, (magnitude(end.v) - h).abs());
                }
            }
        }
    }
    if !(over <= 1e-12 * h && off_edge <= 1e-12 * h && ends > 50 && split == 0) {
        failures.push(format!(
            "every isophote of sin z at 40 samples is under h, each cut inside the domain on |sin z| = 3/2, none \
             split: {ends} ends, worst {off_edge:.1e} off; worst {over:.1e} over; {split} split"
        ));
    }
    // At 32 samples under a low light, a chord beside an extremum of n · l,
    // where Newton's step points away from the level set across it.
    let fine = sine(32);
    let low_graph = fine.surface.graph(&[Affine2::IDENTITY], &Region::Full, fine.refine.as_ref()).unwrap();
    let low = Vec3::new(0.8, 0.1, 0.5);
    let cell = 10.0 / 31.0;
    // A vertex is held to the field it was placed on, n · l through the
    // graph's own jet, whose normal is f's by central differences; a chord's
    // midpoint to the closed form, the distance estimated as the residual
    // over the gradient, as the routine estimates it, so to within a few
    // percent of the tolerance. A cut on the truncation edge lies on its
    // chord, not on the level. At a zero of f the graph is a cone, whose
    // isophotes all meet at its apex, where n · l jumps: a chord within a
    // quarter of a cell of one is not measured.
    let low_copy = &low_graph.copies[0];
    let (mut chord_off, mut vertex_off, mut chords, mut worst_at) = (0.0f64, 0.0f64, 0, Vec2::new(0.0, 0.0));
    for (c, paths) in low_graph.isophotes(low, &[0.3, 0.6, 0.75, 0.9], low_copy) {
        for path in &paths {
            for &v in path {
                if !((magnitude(v.v) - h).abs() > 1e-9) {
                    continue;
                }
                let n = low_graph.outward_normal(v, low_copy).unwrap();
                vertex_off = swift::max(vertex_off, (n.dot(low.normalized()) - c).abs());
            }
            for w in path.windows(2) {
                chords += 1;
                let (m, step) = (0.5 * (w[0].v + w[1].v), 1e-6);
                if !(real::hypot(m.x - (m.x / PI).round() * PI, m.y) > 0.25 * cell) {
                    continue;
                }
                let g = Vec2::new(
                    lit(m + Vec2::new(step, 0.0), low) - lit(m - Vec2::new(step, 0.0), low),
                    lit(m + Vec2::new(0.0, step), low) - lit(m - Vec2::new(0.0, step), low),
                ) / (2.0 * step);
                let off = (lit(m, low) - c).abs() / g.length();
                if off > chord_off {
                    chord_off = off;
                    worst_at = m;
                }
            }
        }
    }
    if !(chord_off <= CHORD_TOLERANCE * cell * 1.05 && vertex_off <= 1e-9 && chords > 200) {
        failures.push(format!(
            "every isophote of sin z at 32 samples under a low light: each vertex on its level, each chord's \
             midpoint near it: {chords} chords, worst {:.4} of a cell at ({:.4}, {:.4}); vertices off by \
             {vertex_off:.1e}",
            chord_off / cell,
            worst_at.x,
            worst_at.y
        ));
    }
    // Γ in bands, its closed isophote round a pole cut by the truncation
    // alone: one run, not two meeting where the path began.
    let mut request = LandscapeRequest::new("gamma(z)", domain(-4.5, 4.5, -2.5, 2.5), 120);
    request.truncation = Some(bands(&[(-3.5, 2.5), (-2.5, 3.0), (-1.5, 4.0), (0.5, 5.5)], 4.0));
    request.layers = Some(Vec::new());
    let gamma = NativeLandscape::build(&request, "native", true).unwrap();
    let gamma_graph = gamma.surface.graph(&[Affine2::IDENTITY], &Region::Full, gamma.refine.as_ref()).unwrap();
    let gamma_split: Vec<usize> = gamma_graph
        .isophotes(Vec3::new(-0.4, -0.6, 0.7), &[-0.6, -0.3, 0.1, 0.5, 0.85], &gamma_graph.copies[0])
        .iter()
        .map(|(_, paths)| abutting(paths))
        .collect();
    if !gamma_split.iter().all(|&n| n == 0) {
        failures.push(format!(
            "no isophote of Γ in bands is split where its path began: runs abutting per level {gamma_split:?}"
        ));
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
