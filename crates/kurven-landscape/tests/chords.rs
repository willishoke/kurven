//! Chords.swift's suites on a landscape built with the refiner: "chords: a
//! contour on a truncated graph ends on the truncation edge, and no vertex
//! stands over it", its families of u, v and the height, and "chords: the
//! plain routine's chords stay within its tolerance of the level set, and of
//! the graph", its cone and bowl. The isophotes and the graph's hatchings,
//! plain and half-Lambert, are Phase 5's (Isophotes.swift) and are split off
//! below, ignored, as is "chords: on the graph of sin z, every isophote cut
//! on the truncation edge, every chord near its level, no run split", which
//! is isophotes throughout. Swift's tolerances.

#![allow(clippy::type_complexity)]

mod support;

use kurven_core::bundle::KurvenBundle;
use kurven_core::hatch::HatchContext;
use kurven_core::hatching::FaceSelector;
use kurven_core::manifest::{Affine2, ContourField, Region, Restriction, Truncation};
use kurven_core::scalar_field::CHORD_TOLERANCE;
use kurven_core::spaces::{P3, WorldSpace};
use kurven_landscape::{LandscapeRequest, NativeLandscape};
use kurven_math::real;
use support::{build, domain};

fn on_domain_edge(p: P3<WorldSpace>) -> bool {
    (p.x().abs() - 2.0).abs() <= 1e-12 || (p.y().abs() - 2.0).abs() <= 1e-12
}

/// 1/(z² − 1) cut at 2/3: the plateau is the Cassini oval |z² − 1| < 3/2, a
/// peanut, convex at its ends and pinched at its waist, so the truncation
/// edge's chords lie over the plateau at the ends and over the graph at the
/// waist. A cut on a chord stood over the plateau at the ends and short of
/// the edge at the waist.
#[test]
fn a_contour_on_a_truncated_graph_ends_on_the_truncation_edge_and_no_vertex_stands_over_it() {
    let h = 2.0 / 3.0;
    let square = domain(-2.0, 2.0, -2.0, 2.0);
    let built = build("1/(z^2 - 1)", square, 32, Truncation::Uniform(h));
    let (surface, refine) = (&built.surface, Some(&built.refine));
    // The occluder `NativeLandscape.build` writes: walls on the domain's
    // rectangle, one tile, the whole region.
    let density = NativeLandscape::wall_density(square, built.shape).unwrap();
    let context = HatchContext {
        perimeter: Some(NativeLandscape::rectangle(square, &density)),
        region: Region::Full,
        tiles: vec![Affine2::IDENTITY],
        base: 0.0,
    };
    let f = |p: P3<WorldSpace>| 1.0 / real::hypot(p.x() * p.x() - p.y() * p.y() - 1.0, 2.0 * p.x() * p.y());
    let lines = [-1.5, -1.2, -0.5, -0.05, 0.03, 0.1, 0.4, 1.3];
    let graph = FaceSelector::Graph { tiled: false };
    let families: Vec<(&str, Vec<Vec<P3<WorldSpace>>>, usize)> = vec![
        ("u", surface.contours(ContourField::U, &lines, &graph, &Restriction::All, &context, refine, None), 16),
        (
            "v",
            surface.contours(ContourField::V, &[-0.6, 0.2, 0.5], &graph, &Restriction::All, &context, refine, None),
            6,
        ),
        (
            "the height",
            surface.contours(ContourField::Height, &[0.4, 0.6], &graph, &Restriction::All, &context, refine, None),
            0,
        ),
    ];
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
                    .map(|p| lines.iter().map(|c| (p.x() - c).abs()).fold(f64::INFINITY, f64::min))
                    .fold(f64::NEG_INFINITY, f64::max);
                off_line = off_line.max(worst);
            }
            if path[0] == path[path.len() - 1] {
                continue;
            }
            for end in [path[0], path[path.len() - 1]] {
                if !on_domain_edge(end) {
                    ends += 1;
                    off_edge = off_edge.max((f(end) - h).abs());
                }
            }
        }
        if !(over <= 1e-12 * h && off_edge <= 1e-12 * h && off_line <= 1e-15 && ends >= wanted) {
            failures.push(format!(
                "{name}: {vertices} vertices, worst {over:.1e} over; {ends} ends, worst {off_edge:.1e} off, {off_line:.1e} off its line"
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The same suite's isophotes (`Face.isophotes`) and the graph's hatchings
/// (`Surface.graphHatching`, `Face.coordinateHatching`), plain and
/// half-Lambert: Isophotes.swift, Phase 5's.
#[test]
#[ignore = "PORT: phase 5 (Isophotes.swift): isophotes and graph hatching draw nothing until it is ported"]
fn its_isophotes_and_graph_hatchings_end_on_the_truncation_edge() {
    panic!("Phase 5: Face.isophotes and Surface.graphHatching are not ported");
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

/// The same suite's last part: the isophotes of the graph of z² lit along
/// z, each chord's midpoint within the tolerance of its circle
/// r = √(1/c² − 1)/2.
#[test]
#[ignore = "PORT: phase 5 (Face.isophotes, Isophotes.swift)"]
fn the_isophotes_of_the_graph_of_z2_stay_within_the_chord_tolerance() {
    panic!("Phase 5: Face.isophotes is not ported");
}

/// "chords: on the graph of sin z, every isophote cut on the truncation
/// edge, every chord near its level, no run split": isophotes of sin z at 40
/// and 32 samples and of Γ in bands, nothing in it but isophotes.
#[test]
#[ignore = "PORT: phase 5 (Face.isophotes, Isophotes.swift): every part of the suite is isophotes"]
fn on_the_graph_of_sin_z_every_isophote_is_cut_on_the_truncation_edge() {
    panic!("Phase 5: Face.isophotes is not ported");
}
