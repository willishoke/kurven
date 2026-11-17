//! LinesOfCurvature.swift's suites (phase 11c): the integral curves of a
//! surface's principal directions, an unsigned field, held to closed forms.
//! On the round torus, a surface of revolution, they are the meridians and the
//! parallels, its coordinate curves, closed after one turn across the seams.
//! On a triaxial ellipsoid they run into its four umbilics and end there, and
//! a seed on one draws nothing. On the forced torus, which has none, the two
//! families cross at right angles in space, and each line, which does not
//! close, ends where it first returns to the coordinate curve it was seeded
//! on. Then the manifest, and visibility as the coordinate curves have it.
//! Swift's tolerances throughout.

// Swift's `guard x > 0 else` refuses a NaN, and so does `if !(x > 0.0)`.
#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::manual_range_contains)]

mod common;

use common::par_map;
use kurven_core::camera::Camera;
use kurven_core::hatching::FaceSelector;
use kurven_core::integral_curves::{End, Limits, Section, Traced, VectorField};
use kurven_core::json;
use kurven_core::manifest::{
    ContourField, DirectionField, Interval, LayerSource, LayerSpec, PrincipalCurvature, Provenance, Seeds, Spacing,
    SurfaceManifest,
};
use kurven_core::parametric::{ParamAxis, ParametricSurface, SecondDerivatives, SecondOrderJet, SurfaceJet};
use kurven_core::polyline::PolylineSet;
use kurven_core::scalar_field::short_way;
use kurven_core::simd::{Vec2, Vec3};
use kurven_core::spaces::{P2, ParamSpace, ViewSpace};
use kurven_core::surface_request::{Shape, SurfaceRequest};
use kurven_math::{real, swift};
use kurven_surfaces::SurfaceDrawing;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

const TWO_PI: f64 = 2.0 * PI;

/// The round torus as the CLI draws it.
fn round_torus() -> ParametricSurface {
    ParametricSurface::torus(2.0, 1.0, (1024, 512)).expect("a round torus")
}

/// The forced torus as `surface forced` builds it, fitted once for the
/// suites that ask.
fn forced() -> &'static ParametricSurface {
    static FORCED: OnceLock<ParametricSurface> = OnceLock::new();
    FORCED.get_or_init(|| {
        let request = SurfaceRequest::new(Shape::Forced {
            system: "jerk".to_string(),
            fit: 8000.0,
            harmonics: 24,
            radial: 0,
            axial: 1,
            radius: 2.5,
            turns: 1,
        });
        SurfaceDrawing::build(&request, None, false).expect("the forced torus").surface
    })
}

/// Swift's variadic `max(x, y, z, rest...)`: a left fold of `max`.
fn max_of(xs: &[f64]) -> f64 {
    xs[1..].iter().fold(xs[0], |m, &x| swift::max(m, x))
}

/// Swift's `[Double].max()`, `empty` on none.
fn max_or(xs: impl IntoIterator<Item = f64>, empty: f64) -> f64 {
    let mut it = xs.into_iter();
    let Some(first) = it.next() else { return empty };
    it.fold(first, swift::max)
}

/// Swift's `[Double].min()`, `empty` on none.
fn min_or(xs: impl IntoIterator<Item = f64>, empty: f64) -> f64 {
    let mut it = xs.into_iter();
    let Some(first) = it.next() else { return empty };
    it.fold(first, swift::min)
}

/// Swift's `"\(end)"`.
fn end_name(e: End) -> &'static str {
    match e {
        End::Singular => "singular",
        End::Boundary => "boundary",
        End::Closed => "closed",
        End::Returned => "returned",
        End::StepLimit => "stepLimit",
    }
}

/// Swift's `Check.expect`: recorded, printed, and asserted together at the end.
fn expect(failures: &mut Vec<String>, ok: bool, what: &str, detail: String) {
    println!("{} {what}: {detail}", if ok { "ok" } else { "FAILED" });
    if !ok {
        failures.push(format!("{what}: {detail}"));
    }
}

/// "lines of curvature: on the round torus, the principal curvatures and directions are the closed form's"
#[test]
fn on_the_round_torus_the_principal_curvatures_and_directions_are_the_closed_forms() {
    let torus = round_torus();
    let face = torus.face();
    let copy = &face.copies[0];
    let (mut curvature_off, mut direction_off, mut count) = (0.0, 0.0, 0);
    for j in 0..64 {
        for i in 0..64 {
            let p = P2::<ParamSpace>::new(TWO_PI * (i as f64 + 0.37) / 64.0, TWO_PI * (j as f64 + 0.61) / 64.0);
            let Some(k) = face.principal_curvatures(p, copy) else { continue };
            count += 1;
            // The tube's 1/r, and the parallel's cos v / (R + r cos v),
            // both against the inward normal.
            let cv = real::cos(p.y());
            curvature_off = max_of(&[curvature_off, (k.greater - 1.0).abs(), (k.lesser - cv / (2.0 + cv)).abs()]);
            let (g, l) = (k.directions.0.normalized(), k.directions.1.normalized());
            direction_off = max_of(&[direction_off, g.x.abs(), l.y.abs()]);
        }
    }
    assert!(
        count == 4096 && curvature_off <= 1e-14 && direction_off <= 1e-14,
        "the greater is the tube's 1/r along the meridian, the lesser cos v / (R + r cos v) along the parallel: \
         {count} points; curvatures off by {curvature_off:.1e}, directions off their axes by {direction_off:.1e}"
    );
}

/// "lines of curvature: on the round torus they are its coordinate curves, closed after one turn across the seams"
#[test]
fn on_the_round_torus_they_are_its_coordinate_curves_closed_after_one_turn_across_the_seams() {
    let torus = round_torus();
    let cell = torus.face().cell();
    let mut failures = Vec::new();
    // Each family from seeds on the coordinate curve across it, at values
    // the same modulo a turn once: 2π and −2π are 0 again. Seeds next to
    // the seams, and on them, so every line crosses one. (6.2831 is a seed
    // just short of the seam, not 2π approximated.)
    #[allow(clippy::approx_constant)]
    let values = vec![0.0, 1.3, 3.0, TWO_PI - 0.01, -0.5, TWO_PI, -TWO_PI, 6.2831];
    for (curvature, along, on, levels) in [
        (PrincipalCurvature::Greater, ContourField::U, ContourField::V, [0.0, 2.5, 6.25]),
        (PrincipalCurvature::Lesser, ContourField::V, ContourField::U, [0.0, 4.0, 6.27]),
    ] {
        let (mut curves, mut closed, mut drift, mut turn_off, mut world) = (0, 0, 0.0, 0.0, 0.0);
        let mut seeds = 0;
        for level in levels {
            let s = Seeds { field: on, level, spacing: Spacing::Values(along, values.clone()) };
            let points = torus.seed_points(&s);
            seeds += points.len();
            let traced = torus.traced_integral_curves(
                &DirectionField::PrincipalDirection(curvature),
                &points,
                torus.section(&s).as_ref(),
                &Limits::default(),
            );
            for (seed, curve) in points.iter().zip(&traced) {
                curves += 1;
                if curve.backward == End::Closed && curve.forward == End::Closed {
                    closed += 1;
                }
                // The fixed coordinate stays the seed's; the other runs
                // one turn, to the seed's image a period along.
                let fixed = if curvature == PrincipalCurvature::Greater { 0 } else { 1 };
                for p in &curve.path {
                    drift = swift::max(drift, (p.v[fixed] - seed.v[fixed]).abs());
                    // How far in space from the coordinate curve through
                    // the seed: the point with that one coordinate set.
                    let mut q = p.v;
                    q[fixed] = seed.v[fixed];
                    world =
                        swift::max(world, ((torus.map)(P2::from_vec(q)).position - (torus.map)(*p).position).length());
                }
                let run = curve.path[curve.path.len() - 1].v - curve.path[0].v;
                turn_off = max_of(&[turn_off, (run[1 - fixed].abs() - TWO_PI).abs(), run[fixed].abs()]);
            }
        }
        // Six distinct values a level, three levels.
        expect(
            &mut failures,
            seeds == 18 && curves == 18 && closed == 18 && drift <= 1e-12 && turn_off <= 1e-12,
            &format!(
                "the {} family: each line is the coordinate curve {} runs along, closed on its seed's image one turn \
                 along",
                curvature.raw(),
                along.raw()
            ),
            format!(
                "{seeds} seeds, {curves} curves, {closed} closed; worst {drift:.1e} rad ({:.1e} cell) off the \
                 coordinate curve, {world:.1e} in space; a turn off by {turn_off:.1e}",
                drift / cell
            ),
        );
    }
    // In space, as a layer: every vertex on the map, with its coordinate.
    let layer = LayerSpec::lines_of_curvature(PrincipalCurvature::Lesser, (torus.u, torus.v), 6, 0.3);
    let LayerSource::IntegralCurves { field, seeds, .. } = &layer.source else {
        panic!("lines of curvature are integral curves")
    };
    let drawn = torus.integral_curves(field, seeds, &Limits::default()).expect("the parallels");
    let mut on_map = true;
    for k in 0..drawn.count() {
        for (v, c) in drawn.path(k).iter().zip(drawn.coords_of_path(k).expect("coordinates")) {
            if v.v != (torus.map)(*c).position {
                on_map = false;
            }
        }
    }
    expect(
        &mut failures,
        drawn.count() == 6 && on_map,
        "drawn as a layer, each parallel is one path whose vertices are the map's at their coordinates",
        format!("{} paths", drawn.count()),
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// x²/a² + y²/b² + z²/c² = 1, a > b > c.
const A: f64 = 3.0;
const B: f64 = 2.0;
const C: f64 = 1.0;

fn ellipsoid_jet(p: P2<ParamSpace>) -> SurfaceJet {
    let (cu, su, cv, sv) = (real::cos(p.x()), real::sin(p.x()), real::cos(p.y()), real::sin(p.y()));
    SurfaceJet::new(
        Vec3::new(A * cv * cu, B * sv, C * cv * su),
        Vec3::new(-A * cv * su, 0.0, C * cv * cu),
        Vec3::new(-A * sv * cu, B * cv, -C * sv * su),
    )
}

fn ellipsoid_second(p: P2<ParamSpace>) -> SecondOrderJet {
    let (cu, su, cv, sv) = (real::cos(p.x()), real::sin(p.x()), real::cos(p.y()), real::sin(p.y()));
    SecondOrderJet {
        jet: ellipsoid_jet(p),
        second: SecondDerivatives {
            uu: Vec3::new(-A * cv * cu, 0.0, -C * cv * su),
            uv: Vec3::new(A * sv * su, 0.0, -C * sv * cu),
            vv: Vec3::new(-A * cv * cu, -B * sv, -C * cv * su),
        },
    }
}

/// "lines of curvature: a triaxial ellipsoid's run into its umbilics and end there; a seed on one draws nothing"
#[test]
fn a_triaxial_ellipsoids_run_into_its_umbilics_and_end_there_a_seed_on_one_draws_nothing() {
    // x²/a² + y²/b² + z²/c² = 1, a > b > c, its poles on the middle axis
    // so the umbilics, in the plane of the longest and the shortest, lie
    // on the equator v = 0, where cos u = √((a² − b²)/(a² − c²)) up to
    // sign and sin u = √((b² − c²)/(a² − c²)) likewise.
    let (a, b, c) = (A, B, C);
    let ellipsoid = ParametricSurface::new(
        ParamAxis::angle(256),
        ParamAxis::new(Interval::new(-PI / 2.0, PI / 2.0), false, 129),
        true,
        None,
        Some(Arc::new(ellipsoid_second)),
        Arc::new(ellipsoid_jet),
    )
    .expect("the ellipsoid");
    let face = ellipsoid.face();
    let copy = &face.copies[0];
    let mut failures = Vec::new();
    let u0 = real::atan2(real::sqrt((b * b - c * c) / (a * a - c * c)), real::sqrt((a * a - b * b) / (a * a - c * c)));
    let umbilics: Vec<P2<ParamSpace>> = [u0, PI - u0, PI + u0, TWO_PI - u0].iter().map(|&u| P2::new(u, 0.0)).collect();
    let read: Vec<_> = umbilics.iter().filter_map(|&p| face.principal_curvatures(p, copy)).collect();
    let gap = max_or(read.iter().map(|k| (k.greater - k.lesser).abs() / k.greater.abs()), f64::INFINITY);
    // At (a, 0, 0): the curvature across, along u, is a/c², along v a/b²,
    // both positive against the inward normal.
    let vertex = face.principal_curvatures(P2::new(0.0, 0.0), copy);
    expect(
        &mut failures,
        read.len() == 4
            && read.iter().all(|k| k.umbilic)
            && gap <= 1e-14
            && vertex.map(|k| {
                (k.greater - a / (c * c)).abs() <= 1e-14 && (k.lesser - a / (b * b)).abs() <= 1e-14 && !k.umbilic
            }) == Some(true),
        "the four umbilics read as umbilics, and the vertex (a, 0, 0) as a/c² and a/b², convex",
        format!("curvatures equal at them to {gap:.1e}"),
    );
    // From (a, 0, 0) the greater family runs along the equator both
    // ways into the umbilics either side; from (0, 0, c) the lesser does,
    // into the same two from the other side.
    let (mut ends, mut worst, mut in_space) = (Vec::<End>::new(), 0.0, 0.0);
    for (curvature, seed, targets) in [
        (PrincipalCurvature::Greater, P2::<ParamSpace>::new(0.0, 0.0), [TWO_PI - u0, u0]),
        (PrincipalCurvature::Lesser, P2::<ParamSpace>::new(PI / 2.0, 0.0), [u0, PI - u0]),
    ] {
        let curve = face
            .principal_directions(curvature, copy)
            .and_then(|field| face.integral_curve(&field, seed, copy, &Limits::default(), None));
        let Some(curve) = curve else {
            ends.push(End::Boundary);
            continue;
        };
        ends.extend([curve.backward, curve.forward]);
        for end in [curve.path[0], curve.path[curve.path.len() - 1]] {
            // Swift's `min(by:)`: the first that none after it beats.
            let away = |t: f64| short_way(t - end.x(), TWO_PI).abs();
            let mut target = targets[0];
            for &t in &targets[1..] {
                if away(t) < away(target) {
                    target = t;
                }
            }
            let target = P2::<ParamSpace>::new(target, 0.0);
            worst = swift::max(worst, real::hypot(short_way(end.x() - target.x(), TWO_PI), end.y() - target.y()));
            in_space =
                swift::max(in_space, ((ellipsoid.map)(end).position - (ellipsoid.map)(target).position).length());
        }
    }
    // Within the least step of it: 1e-4 cell, 2.5e-6 rad here.
    expect(
        &mut failures,
        ends == [End::Singular; 4] && worst <= 1e-5 && in_space <= 2e-5,
        "a line of either family heading into an umbilic ends at it, a singular point of the field, about the least \
         step short",
        format!(
            "ends {:?}; worst {worst:.1e} rad, {in_space:.1e} in space, from the umbilic",
            ends.iter().map(|&e| end_name(e)).collect::<Vec<_>>()
        ),
    );
    let field = face.principal_directions(PrincipalCurvature::Greater, copy).expect("a field");
    let on_one: Vec<Traced> =
        umbilics.iter().filter_map(|&p| face.integral_curve(&field, p, copy, &Limits::default(), None)).collect();
    let seeded = ellipsoid.traced_integral_curves(
        &DirectionField::PrincipalDirection(PrincipalCurvature::Lesser),
        &ellipsoid.seed_points(&Seeds {
            field: ContourField::V,
            level: 0.0,
            spacing: Spacing::Values(ContourField::U, vec![u0, 1.0]),
        }),
        None,
        &Limits::default(),
    );
    expect(
        &mut failures,
        on_one.is_empty() && seeded.len() == 1,
        "a seed on an umbilic draws nothing, and its neighbour on the same level set draws its line",
        format!("{} from the umbilics, {} of two seeds", on_one.len(), seeded.len()),
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// "lines of curvature: the jet of order two is the map's, its second derivatives converging as h² on both tori"
#[test]
fn the_jet_of_order_two_is_the_maps_its_second_derivatives_converging_as_h_squared_on_both_tori() {
    // Central differences of the map's own first derivatives, at three
    // widths: off the closed form by O(h²), so a hundredth of it each
    // tenth of h, down to rounding, about 1e-10 at the narrowest; a wrong
    // term stays put.
    let torus = round_torus();
    let mut failures = Vec::new();
    for (name, surface) in [("round torus", &torus), ("forced torus", forced())] {
        let Some(second) = surface.second_order_jet.as_ref() else {
            expect(&mut failures, false, &format!("the {name} has a jet of order two"), String::new());
            continue;
        };
        let (mut off, mut jet_off) = (Vec::<f64>::new(), 0.0);
        for h in [1e-3, 1e-4, 1e-5] {
            let mut worst = 0.0;
            for k in 0..500usize {
                let p = Vec2::new(
                    TWO_PI * ((k * 37) % 500) as f64 / 500.0 + 0.013,
                    TWO_PI * ((k * 91) % 500) as f64 / 500.0 + 0.029,
                );
                let j = second(P2::from_vec(p));
                let up = (surface.map)(P2::new(p.x + h, p.y));
                let um = (surface.map)(P2::new(p.x - h, p.y));
                let vp = (surface.map)(P2::new(p.x, p.y + h));
                let vm = (surface.map)(P2::new(p.x, p.y - h));
                let uu = (up.du - um.du) / (2.0 * h);
                let vv = (vp.dv - vm.dv) / (2.0 * h);
                let uv = (vp.du - vm.du) / (2.0 * h);
                let vu = (up.dv - um.dv) / (2.0 * h);
                worst = max_of(&[
                    worst,
                    (uu - j.second.uu).length(),
                    (vv - j.second.vv).length(),
                    (uv - j.second.uv).length(),
                    (vu - j.second.uv).length(),
                ]);
                if h == 1e-3 {
                    let m = (surface.map)(P2::from_vec(p));
                    jet_off = max_of(&[
                        jet_off,
                        (m.position - j.jet.position).length(),
                        (m.du - j.jet.du).length(),
                        (m.dv - j.jet.dv).length(),
                    ]);
                }
            }
            off.push(worst);
        }
        expect(
            &mut failures,
            jet_off <= 1e-12 && off[1] <= off[0] / 50.0 && off[2] <= swift::max(off[1] / 50.0, 1e-9) && off[2] <= 1e-7,
            &format!("the {name}'s second derivatives are its first derivatives' differences in the limit, as h²"),
            format!(
                "off {:.1e} at h = 1e-3, {:.1e} at 1e-4, {:.1e} at 1e-5; its first-order jet off the map's by \
                 {jet_off:.1e}",
                off[0], off[1], off[2]
            ),
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// "lines of curvature: a closed curve that returns away from its seed closes; one that turns back returns the other way"
#[test]
fn a_closed_curve_that_returns_away_from_its_seed_closes_one_that_turns_back_returns_the_other_way() {
    let torus = round_torus();
    let face = torus.face();
    let copy = &face.copies[0];
    let mut failures = Vec::new();
    let section = Section::new(Arc::new(|p: P2<ParamSpace>| p.x()), 0.0, Some(TWO_PI));
    // A (p, q) torus curve, a constant field, crosses u = 0 p times the
    // same way: followed past its first return, it closes on its seed
    // after p turns of u for p up to four, and past that is cut at its
    // first return.
    let mut whole: Vec<String> = Vec::new();
    for p in 1..=5 {
        let line = face
            .integral_curve(
                &VectorField::constant(Vec2::new(p as f64, 1.0)),
                P2::new(0.0, 0.3),
                copy,
                &Limits::default(),
                Some(&section),
            )
            .expect("a line");
        let run = (line.path[line.path.len() - 1].v - line.path[0].v) / TWO_PI;
        whole.push(format!("{p}: {} {:.0}×{:.0}", end_name(line.forward), run.x, run.y));
    }
    expect(
        &mut failures,
        whole == ["1: closed 1×1", "2: closed 2×1", "3: closed 3×1", "4: closed 4×1", "5: returned 1×0"],
        "a (p, 1) curve closes whole for p up to four, crossing u = 0 p times; at five it is cut at its first return",
        format!("{whole:?}"),
    );
    // Leaving u = 0 forward, it turns back across it and winds round the
    // other way: it returns once round that way, and is not traced
    // backward, where the old rule ran it to the step limit both ways.
    // Its drift, 0.37 turns of u a turn of v, closes it after a hundred,
    // far past the returns it is followed for; as 0.45 does below.
    let back = VectorField::new(Arc::new(|p: P2<ParamSpace>| Some(Vec2::new(real::cos(p.y()) - 0.37, 1.0))));
    let line = face.integral_curve(&back, P2::new(0.0, 0.0), copy, &Limits::default(), Some(&section)).expect("a line");
    let u: Vec<f64> = line.path.iter().map(|p| p.x()).collect();
    let (u_last, u_max) = (u[u.len() - 1], max_or(u.iter().copied(), f64::NAN));
    expect(
        &mut failures,
        line.backward == End::Returned
            && line.forward == End::Returned
            && (u_last + TWO_PI).abs() <= 1e-12
            && u_max > 0.1,
        "a line that turns back across its section and winds the other way returns once round that way",
        format!(
            "ends {}/{}, from u = 0 out to {u_max:.3} and back round to {u_last:.15}, {} vertices",
            end_name(line.backward),
            end_name(line.forward),
            u.len()
        ),
    );
    // One that only wanders back across the section and on again has
    // not returned: it runs on to the next turn.
    let wander = VectorField::new(Arc::new(|p: P2<ParamSpace>| Some(Vec2::new(3.0 * real::cos(p.y()) + 0.45, 1.0))));
    let on = face.integral_curve(&wander, P2::new(0.0, 0.0), copy, &Limits::default(), Some(&section)).expect("a line");
    let reached: Vec<f64> = on.path.iter().map(|p| p.x()).collect();
    let r_last = reached[reached.len() - 1];
    let r_min = min_or(reached.iter().copied(), f64::NAN);
    expect(
        &mut failures,
        on.forward == End::Returned && (r_last - TWO_PI).abs() <= 1e-12 && r_min < -0.1,
        "a line that wanders back across its section and on again returns only once round",
        format!("back to u = {r_min:.3}, on to {r_last:.15}"),
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Each chord against its family's principal direction at its middle,
/// read independently of the jet of order two: the shape operator I⁻¹ II
/// from central differences of the map's positions alone, a width of 3e-4.
/// The direction, and du and dv as differenced.
fn independent(
    surface: &ParametricSurface,
    outward: Option<f64>,
    p: Vec2,
    curvature: PrincipalCurvature,
) -> (Vec3, Vec3, Vec3) {
    let h = 3e-4;
    let x = |du: f64, dv: f64| (surface.map)(P2::new(p.x + du * h, p.y + dv * h)).position;
    let c = x(0.0, 0.0);
    let xu = (x(1.0, 0.0) - x(-1.0, 0.0)) / (2.0 * h);
    let xv = (x(0.0, 1.0) - x(0.0, -1.0)) / (2.0 * h);
    let xuu = (x(1.0, 0.0) - 2.0 * c + x(-1.0, 0.0)) / (h * h);
    let xvv = (x(0.0, 1.0) - 2.0 * c + x(0.0, -1.0)) / (h * h);
    let xuv = (x(1.0, 1.0) - x(1.0, -1.0) - x(-1.0, 1.0) + x(-1.0, -1.0)) / (4.0 * h * h);
    let n = -outward.unwrap_or(1.0) * xu.cross(xv).normalized();
    let (e, f, g) = (xu.dot(xu), xu.dot(xv), xv.dot(xv));
    let (l, m, nn) = (xuu.dot(n), xuv.dot(n), xvv.dot(n));
    let det = e * g - f * f;
    let s11 = (g * l - f * m) / det;
    let s12 = (g * m - f * nn) / det;
    let s21 = (e * m - f * l) / det;
    let s22 = (e * nn - f * m) / det;
    let mean = 0.5 * (s11 + s22);
    let gap = real::sqrt(swift::max(0.0, mean * mean - (s11 * s22 - s12 * s21)));
    let k = if curvature == PrincipalCurvature::Greater { mean + gap } else { mean - gap };
    let a = Vec2::new(s12, k - s11);
    let b = Vec2::new(k - s22, s21);
    let w = if a.length() > b.length() { a } else { b };
    ((w.x * xu + w.y * xv).normalized(), xu, xv)
}

/// A point carried into the period rectangle by whole turns.
fn wrap(p: Vec2) -> Vec2 {
    p - TWO_PI * Vec2::new((p.x / TWO_PI).floor(), (p.y / TWO_PI).floor())
}

const BIN: f64 = 0.05;

/// A chord of the lesser family in a bin: its path, its index along it, and
/// its ends carried into the period rectangle.
type Chord = (usize, usize, Vec2, Vec2);

/// The bins of the parameter plane a chord's box covers.
fn bins(a: Vec2, b: Vec2) -> Vec<(i64, i64)> {
    let lo = a.min(b) / BIN;
    let hi = a.max(b) / BIN;
    let mut out = Vec::new();
    for x in (lo.x.floor() as i64)..=(hi.x.floor() as i64) {
        for y in (lo.y.floor() as i64)..=(hi.y.floor() as i64) {
            out.push((x, y));
        }
    }
    out
}

/// The tangent at fraction `s` along chord `i` of `path`, from the chords'
/// directions at their middles.
fn tangent(path: &[Vec2], i: usize, s: f64) -> Vec2 {
    let chord = |k: usize| path[k + 1] - path[k];
    let j = if s < 0.5 { i as i64 - 1 } else { i as i64 + 1 };
    if j < 0 || j >= path.len() as i64 - 1 {
        return chord(i);
    }
    let j = j as usize;
    let here = chord(i).length();
    let there = chord(j).length();
    // From the middle of chord i to the middle of chord j, in length.
    let w = (s - 0.5).abs() * here / (0.5 * (here + there));
    (1.0 - w) * chord(i) / here + w * chord(j) / there
}

/// "lines of curvature: on the forced torus the two families cross at right angles in space, each once round"
#[test]
fn on_the_forced_torus_the_two_families_cross_at_right_angles_in_space_each_once_round() {
    let surface = forced();
    let face = surface.face();
    let copy = &face.copies[0];
    let mut failures = Vec::new();
    // No umbilic: the principal curvatures stay apart everywhere.
    let mut least = f64::INFINITY;
    for j in 0..128 {
        for i in 0..256 {
            let p = P2::<ParamSpace>::new(TWO_PI * i as f64 / 256.0, TWO_PI * j as f64 / 128.0);
            if let Some(k) = face.principal_curvatures(p, copy) {
                least = swift::min(least, k.greater - k.lesser);
            }
        }
    }
    let mut families: Vec<Vec<Vec<Vec2>>> = Vec::new();
    let (mut returned, mut curves, mut off_section) = (0, 0, 0.0);
    let mut took = [Duration::ZERO, Duration::ZERO];
    // Each family traced twice, timed with the returns a curve is followed
    // for past its first, and without. A seed's curve is traced on its own
    // (`traced_integral_curves` maps each seed in turn), so each tracing is
    // spread over threads a few seeds at a time, its curves joined in the
    // seeds' order and its time the sum of its parts'.
    let curvatures = [PrincipalCurvature::Greater, PrincipalCurvature::Lesser];
    let seeded: Vec<(DirectionField, Vec<P2<ParamSpace>>, Option<Section>)> = curvatures
        .iter()
        .map(|&curvature| {
            let layer = LayerSpec::lines_of_curvature(curvature, (surface.u, surface.v), 24, 0.3);
            let LayerSource::IntegralCurves { field, seeds, .. } = layer.source else {
                panic!("lines of curvature are integral curves")
            };
            let points = surface.seed_points(&seeds);
            let section = surface.section(&seeds);
            (field, points, section)
        })
        .collect();
    let mut tracings: Vec<(usize, bool, usize)> = Vec::new();
    for (family, (_, points, _)) in seeded.iter().enumerate() {
        for once_round in [false, true] {
            for start in (0..points.len()).step_by(4) {
                tracings.push((family, once_round, start));
            }
        }
    }
    let parts = par_map(&tracings, |&(family, once_round, start)| {
        let (field, points, section) = &seeded[family];
        let limits = if once_round { Limits { closing_returns: 0, ..Limits::default() } } else { Limits::default() };
        let seeds = &points[start..(start + 4).min(points.len())];
        let clock = Instant::now();
        let traced = surface.traced_integral_curves(field, seeds, section.as_ref(), &limits);
        (traced, clock.elapsed())
    });
    let mut traced_all = curvatures.iter().enumerate().flat_map(|(family, _)| {
        [false, true].map(|once_round| {
            let mut curves: Vec<Traced> = Vec::new();
            let mut time = Duration::ZERO;
            for (&(f, o, _), (traced, elapsed)) in tracings.iter().zip(&parts) {
                if f == family && o == once_round {
                    curves.extend(traced.iter().cloned());
                    time += *elapsed;
                }
            }
            (curves, time)
        })
    });
    for curvature in curvatures {
        let (traced, full) = traced_all.next().expect("a tracing");
        let (_, once_round) = traced_all.next().expect("a tracing");
        took[0] += full;
        took[1] += once_round;
        let fixed = if curvature == PrincipalCurvature::Greater { 1 } else { 0 };
        for curve in &traced {
            curves += 1;
            if curve.backward == End::Returned && curve.forward == End::Returned {
                returned += 1;
            }
            for end in [curve.path[0], curve.path[curve.path.len() - 1]] {
                off_section = swift::max(off_section, short_way(end.v[fixed], TWO_PI).abs());
            }
        }
        families.push(traced.iter().map(|t| t.path.iter().map(|p| p.v).collect()).collect());
    }
    // Each chord against its family's principal direction at its middle,
    // read independently of the jet of order two: the shape operator
    // I⁻¹ II from central differences of the map's positions alone, a
    // width of 3e-4. So a wrong second derivative, which the right angle
    // between the families cannot show, turns the lines off it.
    let (mut chord_off, mut chords) = (0.0, 0usize);
    for (family, curvature) in families.iter().zip([PrincipalCurvature::Greater, PrincipalCurvature::Lesser]) {
        let worst = par_map(family, |path: &Vec<Vec2>| {
            let mut w = 0.0;
            for pair in path.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                let (d, xu, xv) = independent(surface, copy.outward, 0.5 * (a + b), curvature);
                let chord = ((b - a).x * xu + (b - a).y * xv).normalized();
                w = swift::max(w, real::acos(swift::min(1.0, chord.dot(d).abs())));
            }
            (w, path.len() - 1)
        });
        chord_off = swift::max(chord_off, max_or(worst.iter().map(|w| w.0), f64::INFINITY));
        chords += worst.iter().map(|w| w.1).sum::<usize>();
    }
    // Where a chord of one family crosses a chord of the other, each
    // carried into the period rectangle by whole turns, the angle between
    // them in space, each carried there by the map's derivatives: between
    // the chords themselves, which the sag rule lets turn from the curve
    // by up to 4s/L, 4.6° at the longest step; between the curves'
    // tangents there, each the chords' directions either side
    // interpolated to the crossing, as a chord's direction is its curve's
    // at its middle to second order; and between the field's directions,
    // orthogonal by construction.
    let mut index: HashMap<(i64, i64), Vec<Chord>> = HashMap::new();
    for (n, path) in families[1].iter().enumerate() {
        for i in 0..(path.len() - 1) {
            let a0 = wrap(path[i]);
            let b0 = a0 + (path[i + 1] - path[i]);
            for k in bins(a0, b0) {
                index.entry(k).or_default().push((n, i, a0, b0));
            }
        }
    }
    let greater_field = face.principal_directions(PrincipalCurvature::Greater, copy).expect("a field");
    let lesser_field = face.principal_directions(PrincipalCurvature::Lesser, copy).expect("a field");
    let (mut crossings, mut chord_worst, mut tangent_worst, mut field_worst) = (0, 0.0, 0.0, 0.0);
    for path in &families[0] {
        for i in 0..(path.len() - 1) {
            let p = wrap(path[i]);
            let r = path[i + 1] - path[i];
            let mut met: HashSet<(usize, usize)> = HashSet::new();
            for k in bins(p, p + r) {
                let Some(others) = index.get(&k) else { continue };
                for &(other_path, other_chord, oa, ob) in others {
                    let (c, e) = (oa, ob - oa);
                    let den = r.x * e.y - r.y * e.x;
                    if den == 0.0 {
                        continue;
                    }
                    let s = ((c.x - p.x) * e.y - (c.y - p.y) * e.x) / den;
                    let t = ((c.x - p.x) * r.y - (c.y - p.y) * r.x) / den;
                    if !(s >= 0.0 && s < 1.0 && t >= 0.0 && t < 1.0 && met.insert((other_path, other_chord))) {
                        continue;
                    }
                    crossings += 1;
                    let x = P2::<ParamSpace>::from_vec(p + s * r);
                    let jet = (surface.map)(x);
                    let space = |w: Vec2| (w.x * jet.du + w.y * jet.dv).normalized();
                    chord_worst = swift::max(chord_worst, real::asin(swift::min(1.0, space(r).dot(space(e)).abs())));
                    let along = tangent(path, i, s);
                    let across = tangent(&families[1][other_path], other_chord, t);
                    tangent_worst =
                        swift::max(tangent_worst, real::asin(swift::min(1.0, space(along).dot(space(across)).abs())));
                    if let (Some(g), Some(l)) = ((greater_field.value)(x), (lesser_field.value)(x)) {
                        field_worst =
                            swift::max(field_worst, real::asin(swift::min(1.0, space(g).dot(space(l)).abs())));
                    }
                }
            }
        }
    }
    let degrees = 180.0 / PI;
    expect(
        &mut failures,
        chords > 50_000 && chord_off * degrees <= 0.25,
        "each chord runs along its family's principal direction, read from the map's positions alone",
        format!("{chords} chords, off it by at most {:.3}°", chord_off * degrees),
    );
    expect(
        &mut failures,
        least > 0.3 && curves == 48 && returned == 48 && off_section <= 1e-12,
        "the forced torus has no umbilic, and each of 24 lines of either family runs once round, from the coordinate \
         curve it was seeded on back to it",
        format!(
            "k₁ − k₂ ≥ {least:.3}; {curves} curves, {returned} returned, ends off the section by {off_section:.1e}; \
             traced in {:?}, {:?} once round alone",
            took[0], took[1]
        ),
    );
    // Each of 24 lines of one family crosses each of the other's about once.
    expect(
        &mut failures,
        crossings > 500 && tangent_worst * degrees <= 0.3 && chord_worst * degrees <= 4.0 && field_worst <= 1e-12,
        "where a line of one family crosses one of the other, they meet at right angles in space",
        format!(
            "{crossings} crossings; off a right angle by at most {:.3}° between the curves' tangents, {:.3}° between \
             their chords, {field_worst:.1e} rad between the field's directions",
            tangent_worst * degrees,
            chord_worst * degrees
        ),
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The torus request the manifest suite reads layers on.
fn torus_request() -> SurfaceRequest {
    SurfaceRequest::new(Shape::Torus { major: 2.0, minor: 1.0 })
}

/// A parametric surface's manifest of `request` holding the layer `text`, read.
fn read_manifest(text: &str, request: &SurfaceRequest) -> Result<SurfaceManifest, String> {
    let provenance = Provenance {
        function: String::new(),
        example: String::new(),
        params: Default::default(),
        cpu_count: 1,
        git_sha: String::new(),
    };
    let empty = SurfaceManifest::new(request.clone(), Vec::new(), Vec::new(), provenance);
    let layer =
        format!(r#"{{"name": "x", "role": "scaffold", "width": 0.3, "heightPolicy": "surface", "source": {text}}}"#);
    let whole = empty.canonical_json().replace(r#""layers":[]"#, &format!(r#""layers":[{layer}]"#));
    let parsed = json::parse(&whole).map_err(|e| e.to_string())?;
    SurfaceManifest::from_json(&parsed).map_err(|e| e.to_string())
}

/// "lines of curvature: the manifest"
#[test]
fn the_manifest() {
    let torus = round_torus();
    let mut failures = Vec::new();
    let head = r#"{"kind": "integralCurves", "field": {"kind": "principalDirection", "of": "greater"}, "#;
    let seeds =
        r#""seeds": {"on": {"field": "v", "level": 0}, "at": {"kind": "values", "field": "u", "values": [0, 1, 2]}}, "#;
    let source = format!(r#"{head}{seeds}"faces": {{"kind": "surface"}}}}"#);
    let refused = |text: &str, request: &SurfaceRequest, words: &str| match read_manifest(text, request) {
        Ok(_) => false,
        Err(e) => e.contains(words),
    };
    // On a surface the height takes any sign: the torus's equators are
    // its level set z = 0, which a meridian crosses twice, the inner one
    // the other way, so it runs on to its seed and closes.
    let equators = read_manifest(
        &source.replace(r#""on": {"field": "v", "level": 0}"#, r#""on": {"field": "height", "level": 0}"#),
        &torus_request(),
    )
    .expect("seeds on the height");
    let crossing = match equators.layers.first().map(|l| &l.source) {
        Some(LayerSource::IntegralCurves { field, seeds, .. }) => torus
            .traced_integral_curves(field, &torus.seed_points(seeds), torus.section(seeds).as_ref(), &Limits::default())
            .iter()
            .filter(|t| t.forward == End::Closed)
            .count(),
        _ => 0,
    };
    // Each meridian crosses both, so each is drawn twice: seeds are not
    // told apart by the curves through them.
    expect(
        &mut failures,
        crossing == 6,
        "seeds on the surface's height at 0, its two equators, at three values of u: three meridians, each from both \
         equators, closed",
        format!("{crossing} closed"),
    );
    let m = read_manifest(&source, &torus_request()).expect("lines of curvature on the torus");
    let again = SurfaceManifest::from_json(&json::parse(&m.canonical_json()).expect("json")).expect("a manifest");
    let expected = LayerSource::IntegralCurves {
        field: DirectionField::PrincipalDirection(PrincipalCurvature::Greater),
        seeds: Seeds {
            field: ContourField::V,
            level: 0.0,
            spacing: Spacing::Values(ContourField::U, vec![0.0, 1.0, 2.0]),
        },
        faces: FaceSelector::Surface,
    };
    expect(
        &mut failures,
        m.layers.first().map(|l| &l.source) == Some(&expected) && again == m && expected.kind() == "lines of curvature",
        "lines of curvature on a parametric surface decode as written and round-trip",
        String::new(),
    );
    let relief = SurfaceRequest::new(Shape::Relief {
        modulus: 0.64,
        major: 2.0,
        minor: 1.0,
        amplitude: 0.4,
        repeats: (1, 1),
        grid: 64,
        turn: 0.0,
        floor: None,
        soft: false,
    });
    let refusals: Vec<(String, Option<&SurfaceRequest>, &str)> = vec![
        (source.replace(r#""of": "greater""#, r#""of": "middle""#), None, "not one of"),
        (
            source.replace(r#""kind": "principalDirection", "of": "greater""#, r#""kind": "slope", "of": "height""#),
            None,
            "slope lines on the surface: they lie on the graph so far",
        ),
        (
            source.replace(r#""field": "u""#, r#""field": "magnitude""#),
            None,
            "seeds placed by magnitude on a parametric surface",
        ),
        (
            source.replace(r#""on": {"field": "v""#, r#""on": {"field": "phase""#),
            None,
            "seeds placed by phase on a parametric surface",
        ),
        (source.replace(r#""field": "u""#, r#""field": "v""#), None, "they are one function"),
        (
            source.replace(
                r#"{"kind": "values", "field": "u", "values": [0, 1, 2]}"#,
                r#"{"kind": "arcLength", "spacing": 0}"#,
            ),
            None,
            "a spacing is positive",
        ),
        (
            source.replace(r#"{"kind": "surface"}"#, r#"{"kind": "plateaus"}"#),
            None,
            "integral curves on the plateaus of a parametric surface",
        ),
        (source.clone(), Some(&relief), "lines of curvature on sn in relief"),
    ];
    let missed: Vec<&str> = refusals
        .iter()
        .filter(|(text, request, words)| !refused(text, request.unwrap_or(&torus_request()), words))
        .map(|r| r.2)
        .collect();
    expect(
        &mut failures,
        missed.is_empty(),
        "a curvature not named, a slope, seeds placed by f's fields or by the field they lie on, a spacing not \
         positive, faces not the surface's, and sn in relief are refused, saying why",
        format!("not refused: {missed:?}"),
    );
    // A landscape's manifest takes none, on the graph or any face.
    let landscape = |text: &str, words: &str| match json::parse(text) {
        Err(e) => e.to_string().contains(words),
        Ok(parsed) => match LayerSource::from_json(&parsed) {
            Ok(_) => false,
            Err(e) => e.to_string().contains(words),
        },
    };
    expect(
        &mut failures,
        landscape(
            &source.replace(r#"{"kind": "surface"}"#, r#"{"kind": "graph", "tiled": false}"#),
            "lines of curvature on the graph: they lie on a parametric surface",
        ) && landscape(&source, "this manifest holds a landscape"),
        "a landscape's manifest refuses lines of curvature, on its graph or a surface",
        String::new(),
    );
    // More seeds than a layer's vertices could hold are refused before
    // any is traced.
    let most = ParametricSurface::max_integral_curves(&Limits::default());
    let words = format!("more than the {most} a layer holds");
    let many = Seeds {
        field: ContourField::V,
        level: 0.0,
        spacing: Spacing::Values(
            ContourField::U,
            (0..(most + 1)).map(|k| TWO_PI * k as f64 / (most + 1) as f64).collect(),
        ),
    };
    let too_many = match torus.integral_curves(
        &DirectionField::PrincipalDirection(PrincipalCurvature::Greater),
        &many,
        &Limits::default(),
    ) {
        Ok(_) => false,
        Err(e) => e.to_string().contains(&words),
    };
    expect(
        &mut failures,
        most == 419 && too_many,
        &format!("more than {most} seeds, {} vertices each at most, are refused", 2 * 20_000 + 1),
        String::new(),
    );
    // By arc length they are counted off the level set's length before
    // any is placed: a hundred-thousandth apart round the outer equator
    // would be some two million.
    let clock = Instant::now();
    let tiny = match torus.integral_curves(
        &DirectionField::PrincipalDirection(PrincipalCurvature::Greater),
        &Seeds { field: ContourField::V, level: 0.0, spacing: Spacing::ArcLength(1e-5) },
        &Limits::default(),
    ) {
        Ok(_) => false,
        Err(e) => e.to_string().contains(&words),
    };
    let took = clock.elapsed();
    expect(
        &mut failures,
        tiny && took < Duration::from_secs(2),
        "seeds a hundred-thousandth apart by arc length are refused before any is placed",
        format!("in {took:?}"),
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The first and last points of every path, in the view's plane.
fn ends(set: &PolylineSet<ViewSpace>) -> Vec<Vec2> {
    (0..set.count())
        .flat_map(|k| {
            let path = set.path(k);
            [path[0], path[path.len() - 1]].map(|p| Vec2::new(p.x(), p.y()))
        })
        .collect()
}

/// "lines of curvature: on the round torus, seen exactly where its coordinate curves are"
#[test]
fn on_the_round_torus_seen_exactly_where_its_coordinate_curves_are() {
    let mut request = torus_request();
    request.lines = Some((12, 0));
    let drawing = SurfaceDrawing::build(&request, None, false).expect("the torus drawing");
    let s = &drawing.surface;
    let layer = LayerSpec::lines_of_curvature(PrincipalCurvature::Greater, (s.u, s.v), 12, 0.3);
    let mut specs: Vec<LayerSpec> = drawing.layers.iter().map(|l| l.spec.clone()).collect();
    specs.push(layer.clone());
    let scene = drawing
        .restyled(&specs)
        .expect("restyled with the meridians")
        .scene(Camera::oblique(&SurfaceRequest::PROJECTION));
    let seen = scene.visible_layers(None, None).expect("the layers seen");
    let lines = seen.iter().find(|l| l.0.spec.name == "lines").map(|l| &l.1);
    let meridians = seen.iter().find(|l| l.0.spec.name == layer.name).map(|l| &l.1);
    let (Some(lines), Some(meridians)) = (lines, meridians) else { panic!("both layers are drawn") };
    let Some(bounds) = lines.bounds() else { panic!("both layers are drawn") };
    let size = swift::max(bounds.size().x, bounds.size().y);
    let (a, b) = (ends(meridians), ends(lines));
    let off = max_or(a.iter().map(|&p| min_or(b.iter().map(|&q| (q - p).length()), f64::INFINITY)), f64::INFINITY);
    let lengths = (meridians.total_length(), lines.total_length());
    assert!(
        meridians.count() == lines.count()
            && meridians.count() > 12
            && off <= 1e-5 * size
            && (lengths.0 - lengths.1).abs() <= 1e-4 * lengths.1,
        "the greater family's lines, seen and hidden by the same test, are the meridians as the coordinate curves \
         draw them: {} and {} pieces seen; ends within {:.1e} of the drawing's size; lengths {:.6} and {:.6}",
        meridians.count(),
        lines.count(),
        off / size,
        lengths.0,
        lengths.1
    );
}
