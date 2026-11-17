//! Conformal.swift's suites: sn on its conformal torus (phase 11e), with
//! Swift's tolerances throughout.
//!
//! Isothermal coordinates round the tube, on the one round torus whose flat
//! rectangle has the aspect of sn's, so the modulus pins the minor radius.
//! A conformal map keeps angles, and the level sets of |f| and arg f of an
//! analytic f cross at right angles in the plane, so on this torus they
//! cross at right angles in space. Held two ways that do not share a
//! derivative: at the drawn contours' own crossings, their tangents pushed
//! through the map's jet, as the drawing computes it; and the map's
//! conformality from differences of its positions alone, with the tube
//! angle against a quadrature of ∫ r dθ / (R + r cos θ). Then the drawn
//! chords, which the turning of the contours bounds, and the request.

use kurven_core::fixtures;
use kurven_core::json::{self, JsonObject, JsonValue};
use kurven_core::parametric::ParametricSurface;
use kurven_core::simd::{Vec2, Vec3};
use kurven_core::spaces::P2;
use kurven_core::surface_request::{Field, Shape, SurfaceRequest, surface_key};
use kurven_core::text;
use kurven_math::{Complex, Jacobi, real, swift};
use kurven_surfaces::SurfaceDrawing;
use std::collections::HashSet;
use std::f64::consts::PI;

const M: f64 = 0.64;
const BIG_R: f64 = 2.0;

/// K and K′ at the suites' modulus.
fn periods() -> (f64, f64) {
    (Jacobi::quarter_period(M), Jacobi::quarter_period(1.0 - M))
}

/// The minor radius the modulus pins.
fn minor() -> f64 {
    Shape::conformal_minor(M, BIG_R)
}

/// sn drawn conformally.
fn conformal_request() -> SurfaceRequest {
    SurfaceRequest::new(Shape::Sn { modulus: M, major: BIG_R, minor: minor(), grid: 600, conformal: true })
}

/// The catalog's sn, on the same ring: the negative control.
fn plain_request() -> SurfaceRequest {
    SurfaceRequest::new(Shape::Sn { modulus: M, major: BIG_R, minor: 1.0, grid: 600, conformal: false })
}

fn build(request: &SurfaceRequest) -> SurfaceDrawing {
    SurfaceDrawing::build(request, None, false).expect("the sn drawing")
}

/// Record a failed expectation, and print its detail either way, as
/// `Check.expect` reports every measure and goes on with the suite.
fn expect(failures: &mut Vec<String>, ok: bool, what: &str, detail: &str) {
    println!("{} {what}: {detail}", if ok { "ok  " } else { "FAIL" });
    if !ok {
        failures.push(format!("{what}: {detail}"));
    }
}

/// Swift's `min(max(x, -1), 1)`.
fn unit_clamp(x: f64) -> f64 {
    swift::min(swift::max(x, -1.0), 1.0)
}

/// "conformal sn: the modulus pins r/R, with the real period round the ring"
#[test]
fn the_modulus_pins_r_over_r_with_the_real_period_round_the_ring() {
    let (k, kp) = periods();
    let r = minor();
    let mut failures: Vec<String> = Vec::new();
    // K and K′ as Jacobi computes them, through the pin's own copy of
    // the mean, to the bit, over the whole range of the modulus.
    let mut unequal: Vec<f64> = Vec::new();
    for m in [1e-15, 1e-9, 1e-4, 0.01, 0.1, 0.25, 0.5, 0.64, 0.75, 0.9, 0.99, 0.999999, 1.0 - 1e-12] {
        let a = Jacobi::quarter_period(1.0 - m) / (2.0 * Jacobi::quarter_period(m));
        if Shape::conformal_minor(m, 3.0) != 3.0 * a / real::sqrt(1.0 + a * a) {
            unequal.push(m);
        }
    }
    expect(
        &mut failures,
        unequal.is_empty(),
        "the pin reads K and K′ as Jacobi.quarterPeriod does, bit for bit",
        &format!("unequal at {unequal:?}"),
    );
    let drawing = build(&conformal_request());
    let s = &drawing.surface;
    // Round the ring along u: z and the distance from the axis stay put
    // and the point turns a quarter about z in a quarter of 4K. Round the
    // tube along v: half of 2K′ takes the outer equator to the inner.
    let p0 = (s.map)(P2::new(0.0, 0.0)).position;
    let pu = (s.map)(P2::new(k, 0.0)).position;
    let pv = (s.map)(P2::new(0.0, kp)).position;
    let along = s.u.range.length() == 4.0 * k
        && s.v.range.length() == 2.0 * kp
        && (pu.z - p0.z).abs() < 1e-15
        && (pu.length() - p0.length()).abs() < 1e-14
        && pu.normalized().dot(p0.normalized()).abs() < 1e-15
        && (p0.x - (BIG_R + r)).abs() < 1e-15
        && (pv.x - (BIG_R - r)).abs() < 1e-14
        && pv.z.abs() < 1e-14;
    expect(
        &mut failures,
        along,
        "4K runs round the ring and 2K′ round the tube, from the outer equator to the inner and back",
        &format!("u {} v {}; p0 {p0:?} pu {pu:?} pv {pv:?}", s.u.range.length(), s.v.range.length()),
    );
    // The flat torus's aspect, tube over ring, is the rectangle's.
    let aspect = r / real::sqrt((BIG_R - r) * (BIG_R + r));
    let want = kp / (2.0 * k);
    expect(
        &mut failures,
        (aspect / want - 1.0).abs() < 1e-15 && (r / BIG_R - want / real::sqrt(1.0 + want * want)).abs() < 1e-15,
        "r / √(R² − r²) = K′ / (2K): r/R = a / √(1 + a²), a = K′ / (2K)",
        &format!("r/R = {:.15} at m = {M}; aspect off by {:.1e}", r / BIG_R, (aspect / want - 1.0).abs()),
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// "conformal sn: the tube angle is the closed form's, by quadrature, across the seam"
#[test]
fn the_tube_angle_is_the_closed_forms_by_quadrature_across_the_seam() {
    // w(θ) = ∫₀^θ r dθ′ / (R + r cos θ′) by Simpson's rule, against v
    // laid evenly in w: w = (v / 2K′) · 2πr / √(R² − r²). θ is read off the
    // position, not the closed form; unwrapped coordinates run a turn
    // past the seam.
    let (_, kp) = periods();
    let r = minor();
    let s = build(&conformal_request()).surface;
    let w = |theta: f64| -> f64 {
        let n = 4000;
        let h = theta / n as f64;
        let g = |x: f64| r / (BIG_R + r * real::cos(x));
        let mut sum = g(0.0) + g(theta);
        for i in 1..n {
            sum += (if i % 2 == 1 { 4.0 } else { 2.0 }) * g(i as f64 * h);
        }
        sum * h / 3.0
    };
    let turn = 2.0 * PI * r / real::sqrt((BIG_R - r) * (BIG_R + r));
    let mut worst = 0.0;
    let mut vs: Vec<f64> = (0..=64).map(|j| 2.0 * kp * j as f64 / 64.0).collect();
    vs.extend([1e-12, 2.0 * kp - 1e-12, kp - 1e-9, kp + 1e-9, 2.0 * kp + 0.3, 4.0 * kp - 1e-12, -0.4]);
    for &v in &vs {
        let p = (s.map)(P2::new(0.7, v)).position;
        let mut theta = real::atan2(p.z, real::sqrt(p.x * p.x + p.y * p.y) - BIG_R);
        if theta < 0.0 {
            theta += 2.0 * PI;
        }
        // Whole turns, from the coordinate, then the angle within one,
        // which by a seam can read a turn either side.
        let turns = (v / (2.0 * kp)).floor();
        let part = v / (2.0 * kp) - turns;
        if theta > PI && part < 0.25 {
            theta -= 2.0 * PI;
        }
        if theta < PI && part > 0.75 {
            theta += 2.0 * PI;
        }
        worst = swift::max(worst, (w(theta) + turns * turn - v / (2.0 * kp) * turn).abs());
    }
    let detail = format!("{} coordinates, seams and unwrapped included; worst {worst:.1e}", vs.len());
    println!("{detail}");
    assert!(worst < 1e-10, "w of the tube angle at v is v's share of w's turn, 2πr / √(R² − r²): {detail}");
}

/// The crossings of a drawing's |f| contours with its arg f contours: each
/// the point of the parameter rectangle where a chord of one meets a chord
/// of the other, with the chords in space and how far each stroke turns at
/// the chord's ends.
struct Crossing {
    c: Vec2,
    chords: (Vec3, Vec3),
    turning: f64,
}

struct Chord {
    a: Vec2,
    b: Vec2,
    world: Vec3,
    turning: f64,
}

fn chords(drawing: &SurfaceDrawing, prefix: &str) -> Vec<Chord> {
    let mut out: Vec<Chord> = Vec::new();
    for layer in drawing.layers.iter().filter(|l| l.spec.name.starts_with(prefix)) {
        let Some(coords) = &layer.paths.coords else { continue };
        let (o, x) = (&layer.paths.offsets, &layer.paths.vertices);
        for k in 0..(o.len() - 1) {
            if o[k + 1] - o[k] < 2 {
                continue;
            }
            let turn = |i: usize| -> f64 {
                if !(i > o[k] && i < o[k + 1] - 1) {
                    return 0.0;
                }
                let d0 = (x[i].v - x[i - 1].v).normalized();
                let d1 = (x[i + 1].v - x[i].v).normalized();
                real::acos(unit_clamp(d0.dot(d1)))
            };
            for i in o[k]..(o[k + 1] - 1) {
                out.push(Chord {
                    a: coords[i].v,
                    b: coords[i + 1].v,
                    world: x[i + 1].v - x[i].v,
                    turning: swift::max(turn(i), turn(i + 1)),
                });
            }
        }
    }
    out
}

fn crossings(drawing: &SurfaceDrawing) -> Vec<Crossing> {
    let magnitude = chords(drawing, "mag_");
    let phase = chords(drawing, "ang_");
    // Phase chords in buckets of the rectangle, each in every bucket its
    // box touches.
    let size = Vec2::new(drawing.surface.u.range.length(), drawing.surface.v.range.length());
    let lo = Vec2::new(drawing.surface.u.range.lo, drawing.surface.v.range.lo);
    let (nx, ny) = (256i64, 128i64);
    let bucket = |p: Vec2| -> (usize, usize) {
        let q = Vec2::new((p.x - lo.x) / size.x * nx as f64, (p.y - lo.y) / size.y * ny as f64);
        ((q.x.floor() as i64).clamp(0, nx - 1) as usize, (q.y.floor() as i64).clamp(0, ny - 1) as usize)
    };
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); (nx * ny) as usize];
    for (i, chord) in phase.iter().enumerate() {
        let (b0, b1) = (bucket(chord.a.min(chord.b)), bucket(chord.a.max(chord.b)));
        for y in b0.1..=b1.1 {
            for x in b0.0..=b1.0 {
                buckets[y * nx as usize + x].push(i);
            }
        }
    }
    let cross = |a: Vec2, b: Vec2| a.x * b.y - a.y * b.x;
    let mut out: Vec<Crossing> = Vec::new();
    for chord in &magnitude {
        let (b0, b1) = (bucket(chord.a.min(chord.b)), bucket(chord.a.max(chord.b)));
        let mut seen: HashSet<usize> = HashSet::new();
        for y in b0.1..=b1.1 {
            for x in b0.0..=b1.0 {
                for &j in &buckets[y * nx as usize + x] {
                    if !seen.insert(j) {
                        continue;
                    }
                    let other = &phase[j];
                    let (d, e) = (chord.b - chord.a, other.b - other.a);
                    let den = cross(d, e);
                    if den == 0.0 {
                        continue;
                    }
                    let t = cross(other.a - chord.a, e) / den;
                    let s = cross(other.a - chord.a, d) / den;
                    if !((0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&s)) {
                        continue;
                    }
                    out.push(Crossing {
                        c: chord.a + t * d,
                        chords: (chord.world, other.world),
                        turning: chord.turning + other.turning,
                    });
                }
            }
        }
    }
    out
}

/// How far from a right angle, in radians, the level sets of |sn| and arg sn
/// through `c` cross in space on `surface`: their tangents in the parameter
/// plane, each square to its field's gradient there, pushed through the
/// map's jet. With g = sn′ / sn = cn dn / sn, the gradient of log |f| is
/// (Re g, −Im g) and of arg f (Im g, Re g). None where f is zero, a pole, or
/// critical.
fn off_square(c: Vec2, surface: &ParametricSurface) -> Option<f64> {
    let f = Jacobi::complex(Complex::new(c.x, c.y), M);
    let g = f.cn * f.dn / f.sn;
    if !(g.is_finite() && g.magnitude() > 1e-8 && g.magnitude() < 1e8) {
        return None;
    }
    let magnitude = Vec2::new(g.re, -g.im);
    let phase = Vec2::new(g.im, g.re);
    let tm = Vec2::new(-magnitude.y, magnitude.x);
    let tp = Vec2::new(-phase.y, phase.x);
    let jet = (surface.map)(P2::from_vec(c));
    let a = tm.x * jet.du + tm.y * jet.dv;
    let b = tp.x * jet.du + tp.y * jet.dv;
    Some(real::asin(unit_clamp(a.dot(b) / (a.length() * b.length()))).abs())
}

/// "conformal sn: wherever an |f| contour crosses an arg f contour, the angle in space is π/2"
#[test]
fn wherever_an_f_contour_crosses_an_arg_f_contour_the_angle_in_space_is_a_right_angle() {
    let (k, kp) = periods();
    let r = minor();
    let drawing = build(&conformal_request());
    let control = build(&plain_request());
    let found = crossings(&drawing);
    let mut failures: Vec<String> = Vec::new();
    let (mut worst, mut counted, mut seams, mut inside, mut outside) = (0.0, 0, 0, 0, 0);
    for x in &found {
        let Some(off) = off_square(x.c, &drawing.surface) else { continue };
        counted += 1;
        worst = swift::max(worst, off);
        let (u, v) = (x.c.x / (4.0 * k), x.c.y / (2.0 * kp));
        if swift::min(swift::min(swift::min(u, 1.0 - u), v), 1.0 - v) < 0.02 {
            seams += 1;
        }
        let p = (drawing.surface.map)(P2::from_vec(x.c)).position;
        let cosine = (real::sqrt(p.x * p.x + p.y * p.y) - BIG_R) / r;
        if cosine < -0.9 {
            inside += 1;
        }
        if cosine > 0.9 {
            outside += 1;
        }
    }
    expect(
        &mut failures,
        counted > 700 && seams > 50 && inside > 100 && outside > 50 && worst <= 1e-6,
        "at every crossing of the drawn contours, over the torus, the tangents meet at π/2 in space",
        &format!(
            "{counted} crossings ({seams} by a seam, {inside} on the tube's inside, {outside} outside); worst {worst:.1e} rad"
        ),
    );
    // And through every point of a grid laid on the seams themselves,
    // on the equators and between.
    let (mut dense, mut points) = (0.0, 0);
    for j in 0..=96 {
        for i in 0..=192 {
            let c =
                Vec2::new(4.0 * k * i as f64 / 192.0, 2.0 * kp * j as f64 / 96.0 + if j == 96 { -1e-12 } else { 0.0 });
            let Some(off) = off_square(c, &drawing.surface) else { continue };
            points += 1;
            dense = swift::max(dense, off);
        }
    }
    expect(
        &mut failures,
        points > 18000 && dense <= 1e-6,
        "and through every point of a grid that lies on the seams",
        &format!("{points} points; worst {dense:.1e} rad"),
    );
    // The negative control: the catalog's sn on the same ring, its tube
    // in its angle.
    let control_found = crossings(&control);
    let (mut missed, mut controls) = (0.0, 0);
    for x in &control_found {
        let Some(off) = off_square(x.c, &control.surface) else { continue };
        controls += 1;
        missed = swift::max(missed, off);
    }
    expect(
        &mut failures,
        controls > 700 && missed > 0.3,
        "on the catalog's torus, r = 1, the same contours cross far from square",
        &format!("{controls} crossings; worst {missed:.3} rad ({:.1}°)", missed * 180.0 / PI),
    );
    // The chords themselves. A chord's direction is the mean of its
    // curve's tangent over the arc it spans, so it lies within the arc's
    // turning of the tangent at any point of it; the turning is bounded
    // here by the most the stroke turns at either end of the chord.
    let chords_off = |x: &Crossing| -> f64 {
        let (a, b) = x.chords;
        real::asin(unit_clamp(a.dot(b) / (a.length() * b.length()))).abs()
    };
    // sn's critical points, sn′ = cn dn = 0, where |f| = 1 and arg f
    // = 0 or π both cross themselves and their strokes turn corners.
    let critical = [Vec2::new(k, 0.0), Vec2::new(3.0 * k, 0.0), Vec2::new(k, kp), Vec2::new(3.0 * k, kp)];
    let from_critical = |c: Vec2| -> f64 {
        critical
            .iter()
            .map(|&p| {
                let d = c - p;
                Vec2::new(d.x - (d.x / (4.0 * k)).round() * 4.0 * k, d.y - (d.y / (2.0 * kp)).round() * 2.0 * kp)
                    .length()
            })
            .reduce(swift::min)
            .expect("four critical points")
    };
    let (mut over, mut chord_worst, mut worst_from, mut bound, mut resolved, mut resolved_worst) =
        (0, 0.0, 0.0, 0.0, 0, 0.0);
    let mut away = 0.0;
    for x in &found {
        let off = chords_off(x);
        if off > chord_worst {
            chord_worst = off;
            worst_from = from_critical(x.c);
        }
        if from_critical(x.c) > 0.05 * k {
            away = swift::max(away, off);
        }
        bound = swift::max(bound, x.turning);
        if off > x.turning + 1e-9 {
            over += 1;
        }
        if x.turning <= 2.0 * PI / 180.0 {
            resolved += 1;
            resolved_worst = swift::max(resolved_worst, off);
        }
    }
    let missed_by_chords = control_found.iter().filter(|x| chords_off(x) > x.turning + 0.05).count();
    expect(
        &mut failures,
        over == 0 && resolved > 200 && resolved_worst < 2.0 * PI / 180.0 && missed_by_chords > 200,
        "the drawn chords cross square to within the strokes' turning there; the control's do not",
        &format!(
            "{} crossings, worst {:.3}° ({worst_from:.1e} from a critical point of sn), the strokes turning up to \
             {:.1}°; {:.3}° farther than K/20 from one; {resolved} where the strokes turn under 2°, worst {:.3}°; \
             {missed_by_chords} control crossings off by 3° more than their turning ({over} over their turning)",
            found.len(),
            chord_worst * 180.0 / PI,
            bound * 180.0 / PI,
            away * 180.0 / PI,
            resolved_worst * 180.0 / PI
        ),
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// "conformal sn: the map is conformal by differences of its positions alone"
#[test]
fn the_map_is_conformal_by_differences_of_its_positions_alone() {
    let (k, kp) = periods();
    let r = minor();
    let s = build(&conformal_request()).surface;
    let control = build(&plain_request()).surface;
    let (hu, hv) = (1e-5 * 4.0 * k, 1e-5 * 2.0 * kp);
    let differences = |surface: &ParametricSurface, c: Vec2| -> (Vec3, Vec3) {
        let p = |du: f64, dv: f64| (surface.map)(P2::new(c.x + du, c.y + dv)).position;
        ((p(hu, 0.0) - p(-hu, 0.0)) / (2.0 * hu), (p(0.0, hv) - p(0.0, -hv)) / (2.0 * hv))
    };
    let off_conformal = |xu: Vec3, xv: Vec3| -> f64 {
        swift::max(xu.dot(xv).abs() / (xu.length() * xv.length()), (xu.length() / xv.length() - 1.0).abs())
    };
    let (mut worst, mut jet_off, mut control_worst, mut second_off, mut foot_off) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for j in 0..=48 {
        for i in 0..=48 {
            let c = Vec2::new(4.0 * k * i as f64 / 48.0, 2.0 * kp * j as f64 / 48.0);
            let (xu, xv) = differences(&s, c);
            worst = swift::max(worst, off_conformal(xu, xv));
            let jet = (s.map)(P2::from_vec(c));
            jet_off = swift::max(
                swift::max(jet_off, (jet.du - xu).length() / xu.length()),
                (jet.dv - xv).length() / xv.length(),
            );
            let (yu, yv) = differences(&control, c);
            control_worst = swift::max(control_worst, off_conformal(yu, yv));
            // The second derivatives, against central differences of
            // the first, as 11c held the round torus's.
            if let Some(second) = &s.second_order_jet {
                let e = second(P2::from_vec(c)).second;
                let at = |du: f64, dv: f64| (s.map)(P2::new(c.x + du, c.y + dv));
                let uu = (at(hu, 0.0).du - at(-hu, 0.0).du) / (2.0 * hu);
                let uv = (at(0.0, hv).du - at(0.0, -hv).du) / (2.0 * hv);
                let vv = (at(0.0, hv).dv - at(0.0, -hv).dv) / (2.0 * hv);
                let scale = xu.length() * xu.length() / (BIG_R + r);
                second_off = swift::max(
                    swift::max(swift::max(second_off, (e.uu - uu).length() / scale), (e.uv - uv).length() / scale),
                    (e.vv - vv).length() / scale,
                );
            } else {
                second_off = f64::INFINITY;
            }
            // The solid's foot reads each point back to its coordinate.
            if let Some(neighbourhood) = &s.neighbourhood {
                let (coordinate, distance) = (neighbourhood.foot)(jet.position);
                let d = coordinate.v - c;
                let wrapped =
                    Vec2::new(d.x - (d.x / (4.0 * k)).round() * 4.0 * k, d.y - (d.y / (2.0 * kp)).round() * 2.0 * kp);
                foot_off = swift::max(swift::max(foot_off, wrapped.length()), distance.abs());
            } else {
                foot_off = f64::INFINITY;
            }
        }
    }
    let mut failures: Vec<String> = Vec::new();
    expect(
        &mut failures,
        worst < 1e-8 && control_worst > 0.3,
        "x_u · x_v = 0 and |x_u| = |x_v| by central differences of positions; not on the catalog's torus",
        &format!("worst {worst:.1e} over 2401 points, seams included; the control off by {control_worst:.2}"),
    );
    expect(
        &mut failures,
        jet_off < 1e-8 && second_off < 1e-7,
        "the jet the drawing pushes tangents through is the positions' differences, and so are its second derivatives",
        &format!("first off by {jet_off:.1e}, second by {second_off:.1e} of their size"),
    );
    expect(
        &mut failures,
        foot_off < 1e-12,
        "the solid's foot reads each point back to its own coordinate, across the seams",
        &format!("worst {foot_off:.1e}"),
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// "conformal sn: the request pins the minor radius, saves only when on, and refuses the rest"
#[test]
fn the_request_pins_the_minor_radius_saves_only_when_on_and_refuses_the_rest() {
    let r = minor();
    let conformal = conformal_request();
    let plain = plain_request();
    let refused = |request: &SurfaceRequest, field: Field, value: f64| -> Option<String> {
        request.setting(field, value, &surface_key).err().map(|e| e.to_string())
    };
    let set = |request: &SurfaceRequest, field: Field, value: f64| -> SurfaceRequest {
        request
            .setting(field, value, &surface_key)
            .unwrap_or_else(|e| panic!("setting {} to {value}: {e}", field.raw()))
    };
    let on = set(&plain, Field::Conformal, 1.0);
    let wider = set(&conformal, Field::Modulus, 0.3);
    let larger = set(&conformal, Field::Major, 3.0);
    let off = set(&conformal, Field::Conformal, 0.0);
    let mut failures: Vec<String> = Vec::new();
    expect(
        &mut failures,
        on == conformal
            && on.get(Field::Conformal) == Some(1.0)
            && plain.get(Field::Conformal) == Some(0.0)
            && wider.get(Field::Minor) == Some(Shape::conformal_minor(0.3, BIG_R))
            && larger.get(Field::Minor) == Some(Shape::conformal_minor(M, 3.0))
            && off
                == SurfaceRequest::new(Shape::Sn { modulus: M, major: BIG_R, minor: r, grid: 600, conformal: false })
            && conformal.drafted().conformal()
            && SurfaceRequest::new(Shape::Torus { major: 2.0, minor: 1.0 }).get(Field::Conformal).is_none(),
        "switched on, the modulus pins the minor radius, and a modulus or a major radius moves it",
        "",
    );
    expect(
        &mut failures,
        conformal.fields() == vec![Field::Modulus, Field::Major, Field::Grid]
            && plain.fields() == vec![Field::Modulus, Field::Major, Field::Minor, Field::Grid],
        "a conformal sn lists no minor radius among its sliders",
        &format!("{:?} and {:?}", conformal.fields(), plain.fields()),
    );
    let pinned = format!(
        "surface.shape.minor is pinned by the modulus on a conformal sn, to {} round a ring of 2.0, not 1.0",
        text::double(r)
    );
    let refusals = [
        (refused(&conformal, Field::Minor, 1.0), pinned),
        (
            refused(&plain, Field::Conformal, 0.5),
            "surface.shape.conformal must be 1 (on) or 0 (off), not 0.5".to_string(),
        ),
        (refused(&conformal, Field::Modulus, 1.5), "modulus must lie strictly between 0 and 1".to_string()),
        (refused(&conformal, Field::Modulus, 1e-300), "too near 0 for sn to be drawn conformally".to_string()),
    ];
    expect(
        &mut failures,
        refusals.iter().all(|(got, want)| got.as_ref().is_some_and(|g| g.contains(want.as_str()))),
        "a minor radius set outright, a switch at neither 0 nor 1, and a modulus it cannot pin by are refused",
        &refusals.iter().map(|(got, _)| format!("{got:?}")).collect::<Vec<_>>().join(" | "),
    );
    // Saved: the key only when on, and a bundle saved before it reads
    // as off. Read back, the same request.
    let text = conformal.json().canonical();
    let back = SurfaceRequest::from_json(&json::parse(&text).expect("the request's JSON")).expect("read back");
    let old = fixtures::shared_json("contract/surface.kurven/manifest.json");
    let saved = SurfaceRequest::from_json(old.get("surface").expect("a surface")).expect("the contract's request");
    expect(
        &mut failures,
        back == conformal
            && back.json().canonical() == text
            && text.contains("\"conformal\":true")
            && !plain.json().canonical().contains("conformal")
            && !saved.conformal(),
        "the switch is written only when on, read back exactly, and a bundle without it is off",
        &text,
    );
    let shape = |change: &dyn Fn(&mut JsonObject), request: &SurfaceRequest| -> JsonValue {
        let mut o = request.json().as_object().cloned().unwrap_or_default();
        let mut s = o.get("shape").and_then(|v| v.as_object()).cloned().unwrap_or_default();
        change(&mut s);
        o.insert("shape".to_string(), JsonValue::Object(s));
        JsonValue::Object(o)
    };
    let relief = SurfaceRequest::new(Shape::Relief {
        modulus: M,
        major: BIG_R,
        minor: 1.0,
        amplitude: 0.3,
        repeats: (1, 1),
        grid: 120,
        turn: 0.0,
        floor: None,
        soft: false,
    });
    let message: Vec<String> = [
        shape(
            &|s| {
                s.insert("conformal".to_string(), JsonValue::Bool(true));
            },
            &relief,
        ),
        shape(
            &|s| {
                s.insert("conformal".to_string(), JsonValue::Bool(false));
            },
            &SurfaceRequest::new(Shape::Torus { major: 2.0, minor: 1.0 }),
        ),
        shape(
            &|s| {
                s.insert("minor".to_string(), JsonValue::Double(0.8));
            },
            &conformal,
        ),
        shape(
            &|s| {
                s.insert("conformal".to_string(), JsonValue::Int(1));
            },
            &plain,
        ),
    ]
    .iter()
    .map(|json| match SurfaceRequest::from_json(json) {
        Ok(_) => "read".to_string(),
        Err(e) => e.to_string(),
    })
    .collect();
    expect(
        &mut failures,
        message[0].contains("conformal on a relief shape: only plain sn")
            && message[1].contains("conformal on a torus shape")
            && message[2].contains("is pinned by the modulus")
            && message[3].contains("should be a boolean"),
        "read, conformal is refused on any other shape, with a minor radius it does not pin, or not a boolean",
        &message.join(" | "),
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
