//! Cuts on a domain whose edge the loops follow, held to closed forms: the
//! suite of `Sources/kurven-test/Chords.swift` that builds no landscape,
//! with Swift's tolerances. Its suites on a landscape built with the
//! refiner are in `kurven-landscape/tests/chords.rs`.

use std::sync::Arc;

use kurven_core::manifest::Interval;
use kurven_core::parametric::{ParamAxis, ParamRectangle};
use kurven_core::scalar_field::{self, ParamPaths, Trim};
use kurven_core::simd::Vec2;
use kurven_core::spaces::{P2, ParamSpace};
use kurven_math::real;

const PI: f64 = core::f64::consts::PI;

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
