//! `Expressions.swift`'s refiner suites that build a bundle
//! (`NativeLandscape.build`), restyle it, or ask `refiner(for: bundle)`, at
//! Swift's tolerances. The ones that need no bundle are `refine.rs`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use kurven_core::bundle::KurvenBundle;
use kurven_core::manifest::{ContourField, Domain, Interval, LayerRole, Truncation};
use kurven_core::spaces::{P3, WorldSpace};
use kurven_core::surface::ContourRefine;
use kurven_landscape::{Catalog, LandscapeRequest, LandscapeStyle, NativeLandscape};
use kurven_math::Complex;

fn preset(name: &str, resolution: i64) -> LandscapeRequest {
    LandscapeRequest::from_preset(Catalog::native().preset(name).unwrap(), resolution)
}

/// "refine: contours placed by the function".
#[test]
fn contours_placed_by_the_function() {
    let request = preset("gamma", 300);
    let (_bundle, reports) = NativeLandscape::benchmark_refinement(&request, 0.02, 5).unwrap();
    assert!(!reports.is_empty());
    for r in &reports {
        let what = format!("{} ({}, {} levels)", r.layer, r.field.raw(), r.levels);
        match r.field {
            ContourField::Magnitude => {
                // At rounding, but for a vertex solved at a saddle of |f|,
                // which is accepted within a hundredth of the tolerance.
                assert!(
                    r.refined_vertex.max <= 1e-2 * 0.02 + 1e-9,
                    "{what}: every vertex is on the level set: max {:.1e} cells, was {:.3}",
                    r.refined_vertex.max,
                    r.grid_vertex.max
                );
                assert!(
                    r.refined_chord.p95 <= 0.02 + 1e-9 && r.refined_chord.max <= 0.03,
                    "{what}: the chords stay within tolerance: p95 {:.4} max {:.4} cells, was max {:.3}",
                    r.refined_chord.p95,
                    r.refined_chord.max,
                    r.grid_chord.max
                );
            }
            ContourField::Phase => {
                assert!(
                    r.wrap_vertices > 0,
                    "{what}: the grid had vertices on the wrap: {} of {}",
                    r.wrap_vertices,
                    r.grid_vertices
                );
                // Within the float32 grid's own precision: an edge the grid
                // saw a crossing on but f does not is snapped to its nearer
                // end, which is where the level set passes within rounding.
                assert!(
                    r.refined_vertex.max < 1e-3,
                    "{what}: and the refined vertices are all on the level set: max {:.1e} cells, was {:.1}",
                    r.refined_vertex.max,
                    r.grid_vertex.max
                );
            }
            _ => panic!("{what}: only f's fields are refined"),
        }
        assert!(
            r.refine_seconds < Duration::from_secs(1),
            "{what}: in reasonable time: {:?} for {} vertices",
            r.refine_seconds,
            r.refined_vertices
        );
    }
    // A restyle re-derives through the refiner rather than losing it.
    let refined = NativeLandscape::build(&request, "native", true).unwrap();
    assert!(refined.refine.is_some(), "a native landscape carries its refiner");
    let mut manifest = refined.manifest.clone();
    manifest.truncation = Truncation::Uniform(3.0);
    let restyled = refined.restyled(&manifest);
    assert!(restyled.refine.is_some(), "and a restyled one still does");
    let plain = NativeLandscape::build(&request, "native", false).unwrap();
    let a = &restyled.layer("mag_major").unwrap().paths;
    let plain_restyled = plain.restyled(&manifest);
    let b = &plain_restyled.layer("mag_major").unwrap().paths;
    assert!(
        a.vertices.len() != b.vertices.len() || a.total_length() != b.total_length(),
        "so its curves after the restyle is the refined curves, not the grid's: {} vs {} paths",
        a.count(),
        b.count()
    );
}

/// "refine: phase contours end on the zeros of f".
#[test]
fn phase_contours_end_on_the_zeros_of_f() {
    // 1/Γ vanishes at the non-positive integers, which the rgamma window
    // holds from 0 down to -5 along its real edge. Every phase meets at
    // each, so each is where a fan of phase contours should end -- and the
    // grid's end at the last cell edge before it, up to a cell short.
    let bundle = NativeLandscape::build(&preset("rgamma", 240), "native", true).unwrap();
    let refiner = NativeLandscape::refiner_for_bundle(&bundle).unwrap();
    let cell = refiner.cell();
    let zeros = [0.0, -1.0, -2.0, -3.0, -4.0, -5.0];
    let mut on_zero = [0; 6];
    let mut short_of_zero = [0; 6];
    let mut worst = 0.0f64;
    for layer in bundle.layers.iter().filter(|l| l.spec.role == LayerRole::Phase) {
        for path in layer.paths.paths() {
            let (Some(first), Some(last)) = (path.first(), path.last()) else { continue };
            for (k, &z0) in zeros.iter().enumerate() {
                let ends: Vec<f64> =
                    [first, last].iter().map(|v| ((v.x() - z0) * (v.x() - z0) + v.y() * v.y()).sqrt()).collect();
                if ends.iter().any(|&e| e <= 1e-9 * cell) {
                    on_zero[k] += 1;
                } else {
                    // A path that comes this close and ends on neither end is
                    // the grid's, stopped short.
                    let near = ends.iter().copied().fold(f64::INFINITY, f64::min);
                    if near <= 2.0 * cell {
                        short_of_zero[k] += 1;
                        worst = worst.max(near / cell);
                    }
                }
            }
        }
    }
    // Twelve phase directions, 30 degrees apart; the window is the upper
    // half-plane, so five rays leave each zero into it, and the two along
    // the real axis lie on the window's edge.
    assert!(on_zero.iter().all(|&n| n >= 5), "a fan of phase contours ends on each zero, to rounding: {on_zero:?}");
    assert!(
        short_of_zero.iter().all(|&n| n == 0),
        "and none stops within two cells of a zero without reaching it: short {short_of_zero:?}, the worst by \
         {worst:.2} cells"
    );
    assert!(
        bundle.layer("ang_major").unwrap().paths.vertices.iter().all(|v| v.z() >= 0.0),
        "the zero's own vertex lifts to the floor and nothing below it"
    );
}

/// "refine: the branch cut is drawn once, as one line".
#[test]
fn the_branch_cut_is_drawn_once_as_one_line() {
    // 1/Γ is real and negative on (-1, 0), (-3, -2) and (-5, -4) along the
    // window's edge, and the cut curves leave the real axis from there into
    // the upper half-plane. The grid has no contour at ±π; the wrap bundles
    // the plain contours carry along the cut are spurious; the refined
    // drawing draws the cut itself, once.
    let request = preset("rgamma", 240);
    let bundle = NativeLandscape::build(&request, "native", true).unwrap();
    let refiner = NativeLandscape::refiner_for_bundle(&bundle).unwrap();
    let cell = refiner.cell();
    let cut_paths = |b: &KurvenBundle| -> Vec<Vec<P3<WorldSpace>>> {
        let mut out = Vec::new();
        for layer in b.layers.iter().filter(|l| l.spec.role == LayerRole::Phase) {
            for path in layer.paths.paths() {
                // Every vertex on Im f = 0 with Re f < 0, but for the two
                // ends: a zero, where the phase is anyone's, or a truncation
                // edge vertex, placed by interpolation within a cell's bend.
                let on_cut = path
                    .iter()
                    .filter(|v| {
                        let f = refiner.value(Complex::new(v.x(), v.y()));
                        f.re < 0.0 && f.im.abs() <= 1e-6 * f.magnitude()
                    })
                    .count();
                if path.len() >= 3 && on_cut + 2 >= path.len() {
                    out.push(path.to_vec());
                }
            }
        }
        out
    };
    let apart = |a: &P3<WorldSpace>, b: &P3<WorldSpace>| {
        ((a.x() - b.x()) * (a.x() - b.x()) + (a.y() - b.y()) * (a.y() - b.y())).sqrt()
    };
    let length = |p: &[P3<WorldSpace>]| p.windows(2).map(|w| apart(&w[0], &w[1])).sum::<f64>();
    let cuts: Vec<Vec<P3<WorldSpace>>> = cut_paths(&bundle).into_iter().filter(|p| length(p) > 10.0 * cell).collect();
    assert!(
        !cuts.is_empty(),
        "the refined drawing has the cut as curves: {} paths, lengths {:?}",
        cuts.len(),
        cuts.iter().map(|p| length(p)).collect::<Vec<_>>()
    );
    // Once: no two cut paths run together.
    let mut doubled = 0;
    for (i, a) in cuts.iter().enumerate() {
        for b in &cuts[i + 1..] {
            let near = a.iter().filter(|v| b.iter().any(|w| apart(w, v) < 0.1 * cell)).count();
            if near > a.len() / 2 {
                doubled += 1;
            }
        }
    }
    assert_eq!(doubled, 0, "and no two of them run together");
    let zeros = [0.0, -1.0, -2.0, -3.0, -4.0, -5.0];
    let from_zero = cuts
        .iter()
        .filter(|p| {
            [p[0], p[p.len() - 1]]
                .iter()
                .any(|e| zeros.iter().any(|&z| (e.x() - z).abs() <= 1e-9 * cell && e.y().abs() <= 1e-9 * cell))
        })
        .count();
    assert!(from_zero > 0, "a cut leaves a zero, as the phase does: {from_zero} of {}", cuts.len());
    let plain = NativeLandscape::build(&request, "native", false).unwrap();
    assert!(
        cut_paths(&plain).into_iter().all(|p| length(&p) <= 10.0 * cell),
        "without a refiner the grid draws no cut, only its wrap bundles"
    );
}

/// "refine: a restyle places only the curves it has not placed before".
#[test]
fn a_restyle_places_only_the_curves_it_has_not_placed_before() {
    // The truncation slider's path in the app: the same grids under a new
    // truncation. Before the refiner remembered, every phase line of the
    // landscape was solved again on each step -- 4.5 s on zeta at 600², for
    // curves that differ from the step before only in where they are clipped.
    let built = NativeLandscape::build(&preset("rgamma", 240), "native", true).unwrap();
    let inner = built.refine.clone().unwrap();
    let solved: Arc<Mutex<HashMap<ContourField, usize>>> = Arc::new(Mutex::new(HashMap::new()));
    let counted = {
        let (solved, inner_contours) = (solved.clone(), inner.contours.clone());
        ContourRefine {
            contours: Arc::new(move |field, level, paths, ceiling, span| {
                *solved.lock().unwrap().entry(field).or_default() += 1;
                inner_contours(field, level, paths, ceiling, span)
            }),
            rings: inner.rings.clone(),
            magnitude: inner.magnitude.clone(),
            phase: None,
            zero: inner.zero.clone(),
            tolerance: inner.tolerance,
            fork: None,
        }
        .remembering()
    };
    let bundle = KurvenBundle::new(
        built.url.clone(),
        built.manifest.clone(),
        built.surface.clone(),
        built.layers.clone(),
        built.wall_mesh.clone(),
        Some(counted),
    );
    let truncation_height = LandscapeStyle::ceiling(&built.manifest.truncation).unwrap();
    let (mut first, mut second) = (bundle.manifest.clone(), bundle.manifest.clone());
    // Two truncations on one rung of the ceiling's ladder
    // (`ContourRefine::rung`), as a slider's neighbouring steps are.
    let near = [0.97, 0.95, 0.9]
        .into_iter()
        .find(|&f| {
            ContourRefine::rung(1.1 * truncation_height * 0.9 * f) == ContourRefine::rung(1.1 * truncation_height * 0.9)
        })
        .unwrap();
    first.truncation = Truncation::Uniform(truncation_height * 0.9);
    second.truncation = Truncation::Uniform(truncation_height * 0.9 * near);
    let once = bundle.restyled(&first);
    let asked = solved.lock().unwrap().clone();
    let twice = once.restyled(&second);
    let again = solved.lock().unwrap().clone();
    let phase = |m: &HashMap<ContourField, usize>| m.get(&ContourField::Phase).copied().unwrap_or(0);
    assert!(phase(&asked) > 0, "the first restyle places the phase lines: {asked:?}");
    assert_eq!(phase(&again), phase(&asked), "and the next truncation places none of them again");
    // The curves are the refiner's answer, remembered or not, to the bit.
    let fresh = NativeLandscape::refiner_for_bundle(&built).unwrap().refine_hook();
    let cold = KurvenBundle::new(
        built.url.clone(),
        built.manifest.clone(),
        built.surface.clone(),
        built.layers.clone(),
        built.wall_mesh.clone(),
        Some(fresh),
    )
    .restyled(&second);
    assert!(
        twice.layers == cold.layers,
        "and the curves are the curves a cold refiner draws under that truncation: {:?} vs {:?}",
        twice.layers.iter().map(|l| l.paths.vertices.len()).collect::<Vec<_>>(),
        cold.layers.iter().map(|l| l.paths.vertices.len()).collect::<Vec<_>>()
    );
}

/// "refine: a chord on the level set, split for its sag, is placed".
#[test]
fn a_chord_on_the_level_set_split_for_its_sag_is_placed() {
    // A phase line through a pole is a ray, the phase constant along it and
    // |f| not, so a chord of it can be split for its sag alone with its
    // midpoint already on the level set. The search along its normal then
    // began from a step of zero, to rounding, and doubled it for ever: none
    // of these, at a pole of order two, one and three, finished building.
    // Each is built on a thread of its own against a deadline, so that the
    // hang, come back, fails the run and does not stall it.
    let square = Domain::new(Interval::new(-1.0, 1.0), Interval::new(-1.0, 1.0));
    let expressions = ["0.3/(z - 0.2 - 0.1*i)^2", "1/(z - (-0.206 + 0.304*i))", "0.1/(z - (-0.575 + 0.147*i))^3"];
    let (done, finished) = mpsc::channel();
    for expression in expressions {
        let done = done.clone();
        std::thread::spawn(move || {
            let request = LandscapeRequest::new(expression, square, 24);
            let _ = done.send(NativeLandscape::build(&request, "native", true).is_ok());
        });
    }
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let mut built = 0;
    let mut count = 0;
    for _ in expressions {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        match finished.recv_timeout(left) {
            Ok(ok) => {
                count += 1;
                if ok {
                    built += 1;
                }
            }
            Err(_) => break,
        }
    }
    assert!(
        count == expressions.len() && built == expressions.len(),
        "a landscape with such a chord builds in under ten seconds: {count} of {} finished",
        expressions.len()
    );
}
