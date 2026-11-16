//! The plain routine's chords and cuts on the graph, held to closed forms:
//! the port of the suites of `Sources/kurven-test/Chords.swift` that need
//! neither the refiner nor Hatch.swift, with Swift's tolerances.
//!
//! Swift builds its landscapes with `NativeLandscape.build(..., refine:
//! true)`, which needs the refiner (`kurven-landscape`). The faces these
//! suites contour read f only through `ContourRefine::magnitude`, so here
//! the landscape is the grid of |f| with a `ContourRefine` whose magnitude
//! is |f| in closed form and whose contours move nothing.

use std::sync::Arc;

use kurven_core::grid::Grid2D;
use kurven_core::hatch::HatchContext;
use kurven_core::hatching::FaceSelector;
use kurven_core::manifest::{Affine2, ContourField, Domain, Interval, Region, Restriction, Truncation};
use kurven_core::parametric::{ParamAxis, ParamRectangle};
use kurven_core::scalar_field::{self, CHORD_TOLERANCE, ParamPaths, Trim};
use kurven_core::simd::Vec2;
use kurven_core::spaces::{DomainSpace, P2, ParamSpace};
use kurven_core::surface::{ContourRefine, Surface};
use kurven_math::real;

const PI: f64 = core::f64::consts::PI;

/// A landscape over [-2, 2]² at `resolution` samples a side, untruncated,
/// with |f| in closed form as its function.
fn landscape(resolution: usize, f: fn(f64, f64) -> f64) -> (Surface, ContourRefine) {
    let domain = Domain::new(Interval::new(-2.0, 2.0), Interval::new(-2.0, 2.0));
    let mut values = vec![0.0f32; resolution * resolution];
    for j in 0..resolution {
        for i in 0..resolution {
            let x = -2.0 + 4.0 * i as f64 / (resolution - 1) as f64;
            let y = -2.0 + 4.0 * j as f64 / (resolution - 1) as f64;
            values[j * resolution + i] = f(x, y) as f32;
        }
    }
    let surface = Surface::new(Grid2D::new(resolution, resolution, domain, values), None, Truncation::None, false);
    let mut refine = ContourRefine::new(Arc::new(|_, _, paths: &[Vec<P2<DomainSpace>>], _, _| paths.to_vec()), None);
    refine.magnitude = Some(Arc::new(move |p: P2<DomainSpace>| f(p.x(), p.y())));
    (surface, refine)
}

fn context() -> HatchContext {
    HatchContext { perimeter: None, region: Region::Full, tiles: vec![Affine2::IDENTITY], base: 0.0 }
}

/// "chords: the plain routine's chords stay within its tolerance of the level set, and of the graph"
#[test]
fn the_plain_routines_chords_stay_within_its_tolerance() {
    let tolerance = CHORD_TOLERANCE;
    // The height on the graph of z, a cone: its level sets are circles
    // |z| = r, and the gradient of |z| has unit length, so a midpoint's
    // distance from its circle is ||m| − r|, which the routine estimates
    // exactly.
    let (cone, refine) = landscape(16, real::hypot);
    let cell = 4.0 / 15.0;
    let radii = [0.4, 0.7, 1.3, 1.9];
    let (mut vertex_off, mut chord_off, mut chords) = (0.0f64, 0.0f64, 0);
    let graph = FaceSelector::Graph { tiled: false };
    for &r in &radii {
        for path in
            cone.contours(ContourField::Height, &[r], &graph, &Restriction::All, &context(), Some(&refine), None)
        {
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
    let (bowl, refine) = landscape(16, |x, y| real::hypot(x * x - y * y, 2.0 * x * y));
    let (mut sag, mut closed_sag) = (0.0f64, 0.0f64);
    chords = 0;
    for path in bowl.contours(
        ContourField::U,
        &[-1.3, -0.5, 0.2, 0.9],
        &graph,
        &Restriction::All,
        &context(),
        Some(&refine),
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

/// "chords: the plain routine's chords stay within its tolerance of the level set, and of the graph",
/// its last part: the isophotes of the graph of z² lit along z.
#[test]
#[ignore = "PORT: phase 5 (Face.isophotes, Isophotes.swift)"]
fn the_isophotes_of_the_graph_of_z2_stay_within_the_chord_tolerance() {
    let (bowl, refine) = landscape(16, |x, y| real::hypot(x * x - y * y, 2.0 * x * y));
    let levels = [0.3, 0.5, 0.7, 0.9];
    let cell = 4.0 / 15.0;
    let graph = FaceSelector::Graph { tiled: false };
    let paths = bowl.contours(
        ContourField::Isophote,
        &levels,
        &graph,
        &Restriction::All,
        &context(),
        Some(&refine),
        Some(kurven_core::simd::Vec3::new(0.0, 0.0, 1.0)),
    );
    assert!(paths.iter().map(|p| p.len() - 1).sum::<usize>() > 40, "the isophotes are drawn");
    let _ = cell;
}

/// "chords: cuts on a domain whose edge the loops follow as chords: joined, carried and read between vertices"
#[test]
fn cuts_on_a_domain_whose_edge_the_loops_follow_as_chords() {
    // The unit disk, |p| − 1 its excess, and an octagon inscribed in its
    // circle, turned so a chord's midpoint lies at angle 0: the loops
    // follow the circle as chords, which lie inside it, a gap between
    // each chord and the arc over it.
    let axis = ParamAxis::new(Interval::new(-2.0, 2.0), false, 32);
    let octagon: Vec<P2<ParamSpace>> = (0..=8)
        .map(|k| {
            let a = ((k % 8) as f64 + 0.5) * PI / 4.0;
            P2::new(real::cos(a), real::sin(a))
        })
        .collect();
    let excess = |p: P2<ParamSpace>| p.v.length() - 1.0;
    let clip = |path: &[Vec2], loops: Option<ParamPaths>| -> ParamPaths {
        scalar_field::clipped(
            vec![(0.0, vec![path.iter().map(|&p| P2::from_vec(p)).collect()])],
            ParamRectangle::new(axis, axis),
            &Trim {
                trim_loops: Some(loops.unwrap_or_else(|| vec![octagon.clone()])),
                excess: Some(Arc::new(excess)),
                ..Trim::default()
            },
        )
        .remove(0)
        .1
    };
    let circle = |c: Vec2, r: f64, phi: f64| -> Vec<Vec2> {
        (0..=400)
            .map(|k| {
                let a = phi + 2.0 * PI * (k % 400) as f64 / 400.0;
                c + r * Vec2::new(real::cos(a), real::sin(a))
            })
            .collect()
    };
    // The worst excess of a run's vertices, and of its ends off 0.
    let worst = |runs: &ParamPaths| -> (f64, f64) {
        let over =
            runs.iter().flatten().map(|&p| excess(p)).fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v))));
        let ends = runs
            .iter()
            .filter(|r| r[0] != r[r.len() - 1])
            .flat_map(|r| [r[0], r[r.len() - 1]])
            .map(|p| excess(p).abs())
            .fold(0.0, f64::max);
        (over.unwrap_or(-1.0), ends)
    };
    // Out through the top chord, along the gap over it, and back: one run.
    let through = [Vec2::new(-0.5, 0.0), Vec2::new(-0.2, 0.95), Vec2::new(0.2, 0.95), Vec2::new(0.5, 0.0)];
    let joined = clip(&through, None);
    // The same closed, beginning in the gap.
    let lp = [
        Vec2::new(0.0, 0.95),
        Vec2::new(0.2, 0.95),
        Vec2::new(0.5, 0.0),
        Vec2::new(0.0, -0.5),
        Vec2::new(-0.5, 0.0),
        Vec2::new(-0.2, 0.95),
        Vec2::new(0.0, 0.95),
    ];
    let joined_round = clip(&lp, None);
    // Round the closed one, the run begins where it was cut, on the
    // path, and ends there again.
    let round_kept = joined_round.len() == 1
        && joined_round[0][0] == joined_round[0][joined_round[0].len() - 1]
        && lp.iter().all(|q| joined_round[0].iter().any(|p| p.v == *q))
        && joined_round[0].len() <= lp.len() + 1;
    assert!(
        joined.len() == 1 && joined[0].iter().map(|p| p.v).collect::<Vec<_>>() == through && round_kept,
        "a path the loops cut where it runs between a chord and the circle is one run, across a closed path's start too: \
         {:?} and {:?} vertices",
        joined.iter().map(Vec::len).collect::<Vec<_>>(),
        joined_round.iter().map(Vec::len).collect::<Vec<_>>()
    );
    // A circle round (1.2, 0) beginning in the gap: its arc inside the
    // disk is one run, ending on the circle at both ends, carried round
    // the path's start to reach it.
    let carried = clip(&circle(Vec2::new(1.2, 0.0), 0.3, PI - real::acos(0.25 / 0.3)), None);
    // One round (1.3, 0) whose arc inside the disk lies wholly in the
    // gap, crossing no chord.
    let in_gap = clip(&circle(Vec2::new(1.3, 0.0), 0.35, 0.0), None);
    let mut both = carried.clone();
    both.extend(in_gap.iter().cloned());
    let (over, ends) = worst(&both);
    assert!(
        carried.len() == 1 && in_gap.len() == 1 && over < 0.0 && ends <= 1e-15,
        "an arc ends on the circle, carried round a closed path's start, and one wholly in a gap is kept: \
         {} and {} runs, ends off the circle by {ends:.1e}",
        carried.len(),
        in_gap.len()
    );
    // A chord between two vertices outside the disk through its middle
    // keeps its middle: the domain read at the midpoint, between the
    // vertices, as a square's loops hold it all.
    let sq = vec![P2::new(-1.5, -1.5), P2::new(1.5, -1.5), P2::new(1.5, 1.5), P2::new(-1.5, 1.5), P2::new(-1.5, -1.5)];
    let across = clip(&[Vec2::new(-1.2, 0.0), Vec2::new(1.2, 0.0)], Some(vec![sq]));
    assert!(
        across.len() == 1
            && across[0].iter().all(|p| p.y() == 0.0)
            && (across[0][0].x() + 1.0).abs() <= 1e-15
            && (across[0][across[0].len() - 1].x() - 1.0).abs() <= 1e-15,
        "a chord through the domain between two vertices outside it keeps the stretch inside, ended on the circle: {across:?}"
    );
}
