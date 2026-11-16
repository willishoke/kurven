//! `Sources/kurven-test/Composite.swift`: several geometries in one scene,
//! every suite, with Swift's tolerances.
//!
//! A composite's every curve, held point by point to the brute-force casts
//! the stress, solids and occlusion suites run, each against its own
//! geometry where it stands (`gate`, which holds lines 1-760 of the Swift
//! file). The round torus is Phase 4's: the compositions and placements
//! that stand it in a composite, and the surfaces refused at the door (the
//! forced torus, the displaced one), are held in ignored tests below, each
//! saying what it would hold.

#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::type_complexity)]

mod common;
mod gate;

use std::time::{Duration, Instant};

use gate::{
    Cast, CompositeVerdict, Parts, WITH_TORUS, composite_cast, composite_verdict, cubic_bundle, cubic_landscape,
    gate_compositions, polyhedron, surface,
};
use kurven_core::camera::Camera;
use kurven_core::composite::{Composite, Placement};
use kurven_core::hidden_line::HeightfieldVisibility;
use kurven_core::occlusion::OcclusionTest;
use kurven_core::scene::Scene;
use kurven_core::simd::Vec3;
use kurven_core::spaces::P3;
use kurven_landscape::NativeLandscape;

/// The compositions whose part meets the terrain: their contact line is
/// drawn, and partly hidden by another part. The others draw none.
const TOUCHING: [&str; 8] = [
    "a cube standing on the landscape behind a tower",
    "the round torus sunk into the foot of a tower",
    "a frame sunk into the slope of a tower",
    "all three, moved",
    "a cube flush against the landscape's front wall",
    "a cube over the towers, where a tower's corner hid a hatch line's span",
    "a cube behind the spike's apex",
    "two landscapes side by side, a cube across the gap",
];

/// One composition through the gate under every camera, its verdicts
/// summed, and the suite's three checks of it: the failures, as text.
fn gate(name: &str, parts: &Parts, cameras: &[(String, Camera)]) -> Vec<String> {
    let mut total = CompositeVerdict::default();
    let start = Instant::now();
    for (_, camera) in cameras {
        let composite = Composite::new(parts.clone(), *camera).unwrap_or_else(|e| panic!("{name}: {e}"));
        let casts: Vec<Cast> = composite.parts.iter().map(composite_cast).collect();
        total.add(composite_verdict(&composite, &casts).unwrap_or_else(|e| panic!("{name}: {e}")));
    }
    let elapsed = start.elapsed();
    let mut failures = Vec::new();
    let summary = format!(
        "{} of {} disagree, {} hidden by another, {} at a boundary, {} vertices off the drawing alone, {} on a \
         contact line, {} of them hidden by another, {} of them on no landscape's top, {elapsed:?}{}",
        total.wrong,
        total.compared,
        total.hidden_by_other,
        total.undecided,
        total.outside_alone,
        total.contact_compared,
        total.contact_hidden,
        total.off_terrain,
        total.example
    );
    eprintln!(
        "{name}: {summary}; own: {} of {} disagree, {} seen on the torus",
        total.own_wrong, total.own_compared, total.torus_seen
    );
    if !(total.wrong == 0
        && total.outside_alone == 0
        && total.off_terrain == 0
        && total.compared > 20_000
        && total.hidden_by_other > 0)
    {
        failures
            .push(format!("{name}: drawn exactly where no other geometry's cast meets the line of sight: {summary}"));
    }
    if TOUCHING.contains(&name) {
        if !(total.contact_compared > 1_000
            && total.contact_hidden > 0
            && total.contact_hidden < total.contact_compared)
        {
            failures.push(format!(
                "{name}: its contact line is drawn, and partly hidden by another part: {} compared, {} hidden",
                total.contact_compared, total.contact_hidden
            ));
        }
    } else if total.contact_compared != 0 {
        failures.push(format!(
            "{name}: no part meets the terrain, and no contact line is drawn: {} compared",
            total.contact_compared
        ));
    }
    let has_torus = parts.iter().any(|p| p.1.parametric().is_some());
    if !(total.own_wrong == 0 && total.own_compared > 2_000 && (!has_torus || total.torus_seen > 1_000)) {
        failures.push(format!(
            "{name}: each geometry alone is its own cast: {} of {} disagree, {} seen on the torus{}",
            total.own_wrong, total.own_compared, total.torus_seen, total.own_example
        ));
    }
    failures
}

/// "composite: every curve is seen exactly where no geometry's brute-force
/// cast meets its line of sight", the eight compositions with no torus in
/// them, under the landscape's camera, el 30° az 37° and el 55° sheared:
/// over the three cameras every composition compares more than 20 000
/// samples, every one drawn exactly where no other part's cast meets its
/// line of sight, some hidden by another; no composite vertex off the runs
/// drawn alone; a touching composition's contact line compared at over
/// 1 000 samples, some but not all hidden by another part, and no sample
/// of it on no landscape's top; a floating one's none; and each geometry
/// alone its own cast at over 2 000 points. The three compositions with
/// the round torus are `every_curve_with_the_round_torus` (ignored).
#[test]
fn every_curve_is_seen_exactly_where_no_geometrys_brute_force_cast_meets_its_line_of_sight() {
    let (cameras, compositions) = gate_compositions();
    let start = Instant::now();
    let runs: Vec<(&String, &Parts)> = compositions
        .iter()
        .filter(|(name, _)| !WITH_TORUS.contains(&name.as_str()))
        .map(|(name, parts)| (name, parts.as_ref().unwrap_or_else(|e| panic!("{name}: {e}"))))
        .collect();
    let ran = runs.len();
    // The compositions are independent, and each composite's own drawing
    // runs on one thread: spread over threads, the failures kept in order.
    let failures: Vec<String> =
        common::par_map(&runs, |&(name, parts)| gate(name, parts, &cameras)).into_iter().flatten().collect();
    eprintln!("{ran} compositions in {:?}", start.elapsed());
    assert!(ran == 8, "the eight compositions with no torus ran: {ran}");
    assert!(failures.is_empty(), "{failures:#?}");
}

/// "composite: every curve is seen exactly where no geometry's brute-force
/// cast meets its line of sight", the three compositions that stand the
/// round torus in a composite (contract/surface_torus.kurven,
/// `SurfaceBundle`, Phase 4): the torus ringing a tower at scale 0.35 at
/// (1, 0, 1.6), which meets nothing and draws no contact line; the torus
/// sunk into the foot of a tower at scale 0.3 at (0.4, 1.3, 0.3); and all
/// three moved -- the landscape by (0.4, 0.2, -0.1), the cube at scale 0.35
/// at (-0.8, 0.3, 0.2), the torus at 0.35 at (1.4, 0.2, 1.5). Each held to
/// the same three checks, the torus's own curves seen at over 1 000 of its
/// own cast's points (`torus_own_cast`), and the torus's contact line drawn
/// by `Composite::visible_contact` (`clipOnChords`, also Phase 4).
#[test]
fn every_curve_with_the_round_torus() {
    let (cameras, compositions) = gate_compositions();
    let mut failures = Vec::new();
    for (name, parts) in &compositions {
        if !WITH_TORUS.contains(&name.as_str()) {
            continue;
        }
        let parts = parts.as_ref().unwrap_or_else(|e| panic!("{name}: {e}"));
        failures.extend(gate(name, parts, &cameras));
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// "composite: a landscape's test of another solid's point has a floor and
/// a boundary": under the landscape's own camera, whose lines of sight run
/// toward the front wall (−y) and up, a point a hair inside the front wall,
/// on it and a hair outside, each under the wall's top, is seen; one under
/// the surface inside is not; one under the base whose line leaves the
/// footprint before it reaches the base is seen, and one whose line comes
/// up through the floor is not.
#[test]
fn a_landscapes_test_of_another_solids_point_has_a_floor_and_a_boundary() {
    let bundle = cubic_bundle();
    let mut failures = Vec::new();
    for (name, b) in [("on the grid", &bundle)] {
        let scene = Scene::from_bundle(b, &b.manifest.presets[0]);
        let test =
            HeightfieldVisibility::new(scene.heightfield().unwrap().clone(), &scene.camera.view, scene.depth_tolerance);
        for (what, p, clear) in [
            ("a hair inside the front wall", P3::new(0.0, -2.0 + 1e-12, 0.05), true),
            ("on the front wall", P3::new(0.0, -2.0, 0.05), true),
            ("a hair outside the front wall", P3::new(0.0, -2.0 - 1e-12, 0.05), true),
            ("under the surface inside", P3::new(0.0, -1.9, 0.05), false),
            ("under the base, its line leaving through the wall", P3::new(0.0, -1.95, -0.05), true),
            ("under the base, its line coming up through the floor", P3::new(0.0, -1.5, -0.05), false),
        ] {
            let got = test.is_clear_of_another_solid(p);
            if got != clear {
                failures.push(format!(
                    "{name}: a point {what} is {}: it was {}",
                    if clear { "seen" } else { "hidden" },
                    if got { "seen" } else { "hidden" }
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// "composite: a surface with no exact occlusion test is refused", the
/// parts that need no parametric surface: a landscape scaled is refused; a
/// landscape on its function is refused, saying "on its function"; a cube
/// at scale 1e3 and at 1e-3 on the landscape is drawn in under 20 s each;
/// and a cube placed at scales 1e-3 and 1e3 and offset (1e4, -1e4, 1e4) is
/// taken, at 1e200, 1e-20, 0.999e-3, offset 1e300 and 1.001e4 refused. The
/// forced and displaced tori, and the torus at either end of the scales,
/// are `a_surface_with_no_exact_occlusion_test_is_refused_the_tori`
/// (ignored).
#[test]
fn a_surface_with_no_exact_occlusion_test_is_refused() {
    let bundle = cubic_bundle();
    let landscape = cubic_landscape();
    let mut failures = Vec::new();
    let refused = |parts: Parts, camera: Camera| Composite::new(parts, camera).err().map(|e| e.0);
    let cube = polyhedron("cube");
    if refused(
        vec![
            ("landscape".into(), landscape.clone(), Placement::new(2.0, Vec3::ZERO)),
            ("cube".into(), cube.clone(), Placement::IDENTITY),
        ],
        landscape.camera,
    )
    .is_none()
    {
        failures.push("a landscape scaled is refused: it was taken".to_string());
    }
    // A landscape on its function: nothing bounds f's slope, so it
    // cannot cut another's curves exactly, and it is refused.
    let refiner = NativeLandscape::refiner_for_bundle(&bundle).expect("the cubic records its expression");
    let refined = bundle.refined(Some(refiner.refine_hook()));
    let on_f = Scene::from_bundle(&refined, &refined.manifest.presets[0]);
    match refused(
        vec![
            ("landscape".into(), on_f.clone(), Placement::IDENTITY),
            ("cube".into(), cube.clone(), Placement::IDENTITY),
        ],
        on_f.camera,
    ) {
        None => failures.push("a landscape on its function is refused in a composite: it was taken".to_string()),
        Some(why) if !why.contains("on its function") => {
            failures.push(format!("a landscape on its function is refused in a composite, saying why: {why}"))
        }
        Some(_) => {}
    }
    // At either end of the range of scales, drawn in reasonable time:
    // each part works only over the ball or box that holds it.
    for (what, placement) in [
        ("a cube at scale 1e3", Placement::new(1e3, Vec3::ZERO)),
        ("a cube at scale 1e-3", Placement::new(1e-3, Vec3::new(1.0, 1.0, 1.0))),
    ] {
        let start = Instant::now();
        let drawn = Composite::new(
            vec![
                ("landscape".into(), landscape.clone(), Placement::IDENTITY),
                ("part".into(), cube.clone(), placement),
            ],
            landscape.camera,
        )
        .and_then(|c| c.visible_layers())
        .map(|layers| !layers.is_empty());
        let elapsed = start.elapsed();
        eprintln!("{what}: {drawn:?} in {elapsed:?}");
        if !(drawn == Ok(true) && elapsed < Duration::from_secs(20)) {
            failures.push(format!("{what} on the landscape is drawn in reasonable time: {drawn:?}, {elapsed:?}"));
        }
    }
    // Placements at either end of the range, and past it.
    for (what, placement, taken) in [
        ("scale 1e-3", Placement::new(1e-3, Vec3::ZERO), true),
        ("scale 1e3", Placement::new(1e3, Vec3::ZERO), true),
        ("offset 1e4", Placement::new(1.0, Vec3::new(1e4, -1e4, 1e4)), true),
        ("scale 1e200", Placement::new(1e200, Vec3::ZERO), false),
        ("scale 1e-20", Placement::new(1e-20, Vec3::ZERO), false),
        ("scale 0.999e-3", Placement::new(0.999e-3, Vec3::ZERO), false),
        ("offset 1e300", Placement::new(1.0, Vec3::new(0.0, 1e300, 0.0)), false),
        ("offset 1.001e4", Placement::new(1.0, Vec3::new(0.0, 0.0, -1.001e4)), false),
    ] {
        let made = refused(
            vec![
                ("landscape".into(), landscape.clone(), Placement::IDENTITY),
                ("cube".into(), cube.clone(), placement),
            ],
            landscape.camera,
        )
        .is_none();
        if made != taken {
            failures.push(format!("a cube placed at {what} is {}", if taken { "taken" } else { "refused" }));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// "composite: a surface with no exact occlusion test is refused", the
/// parts that need a parametric surface (contract/surface_*.kurven,
/// `SurfaceBundle`, Phase 4): the forced torus (no tubular neighbourhood)
/// beside the landscape is refused, the error saying "no exact occlusion
/// test"; the displaced torus (surface_relief) is refused, saying
/// "displaced"; and the round torus at scale 1e-3 and at 1e3, offset
/// (0, 0, 1), on the landscape is drawn in under 20 s each.
#[test]
fn a_surface_with_no_exact_occlusion_test_is_refused_the_tori() {
    let landscape = cubic_landscape();
    let mut failures = Vec::new();
    for (name, words) in [("forced", "no exact occlusion test"), ("relief", "displaced")] {
        let part = surface(name).unwrap_or_else(|e| panic!("{name}: {e}"));
        match Composite::new(
            vec![
                ("landscape".into(), landscape.clone(), Placement::IDENTITY),
                (name.into(), part, Placement::IDENTITY),
            ],
            landscape.camera,
        ) {
            Ok(_) => failures.push(format!("the {name} torus is refused in a composite: it was taken")),
            Err(e) if !e.0.contains(words) => failures.push(format!("the {name} torus is refused, saying why: {e}")),
            Err(_) => {}
        }
    }
    for (what, scale) in [("the round torus at scale 1e-3", 1e-3), ("the round torus at scale 1e3", 1e3)] {
        let torus = surface("torus").unwrap_or_else(|e| panic!("torus: {e}"));
        let start = Instant::now();
        let drawn = Composite::new(
            vec![
                ("landscape".into(), landscape.clone(), Placement::IDENTITY),
                ("part".into(), torus, Placement::new(scale, Vec3::new(0.0, 0.0, 1.0))),
            ],
            landscape.camera,
        )
        .and_then(|c| c.visible_layers())
        .map(|layers| !layers.is_empty());
        let elapsed = start.elapsed();
        if !(drawn == Ok(true) && elapsed < Duration::from_secs(20)) {
            failures.push(format!("{what} on the landscape is drawn in reasonable time: {drawn:?}, {elapsed:?}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
