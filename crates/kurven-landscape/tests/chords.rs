//! Chords.swift's "chords: a contour on a truncated graph ends on the
//! truncation edge, and no vertex stands over it", its families of u, v and
//! the height. The isophotes and the graph's hatchings, plain and
//! half-Lambert, are Phase 5's (Isophotes.swift) and are split off below,
//! ignored. Swift's tolerances.

#![allow(clippy::type_complexity)]

mod support;

use kurven_core::hatch::HatchContext;
use kurven_core::hatching::FaceSelector;
use kurven_core::manifest::{Affine2, ContourField, Region, Restriction, Truncation};
use kurven_core::spaces::{P3, WorldSpace};
use kurven_landscape::NativeLandscape;
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
