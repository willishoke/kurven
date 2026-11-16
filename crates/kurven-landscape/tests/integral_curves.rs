//! `Sources/kurven-test/IntegralCurves.swift`'s suites on landscapes built
//! from f, with Swift's tolerances (the ones on polylines and the manifest
//! are `kurven-core/tests/integral_curves.rs`), and the slope-lines golden
//! of `tests/fixtures/fields`, its curves before visibility.
//!
//! Integral curves and slope lines, held to closed forms. Where f is
//! analytic and not zero, log f is analytic too, so the slope lines of |f|
//! are the level sets of arg f: seeded on a contour of |f| at listed angles
//! of arg f, each stays on the phase contour at its angle, to the refiner's
//! tolerance, on sn's landscape, a rational function's and exp(z)'s, whose
//! phase contours are the lines Im z = θ, tiled too. Each stopping rule on
//! its own: a zero of f, a pole, the domain's edge, the footprint, the
//! truncation edge, a closed curve and the step limit.

use std::collections::BTreeMap;
use std::f64::consts::PI;

use kurven_core::bundle::KurvenBundle;
use kurven_core::face::Face;
use kurven_core::fixtures;
use kurven_core::hatch::HatchContext;
use kurven_core::hatching::FaceSelector;
use kurven_core::integral_curves::{End, Limits, Traced};
use kurven_core::manifest::{
    Affine2, BoundaryPerimeter, ContourField, DirectionField, Domain, Interval, LayerSource, PerimeterEdge, Region,
    Seeds, Spacing, Truncation,
};
use kurven_core::scalar_field::{self, ParamPaths, Trim};
use kurven_core::simd::Vec2;
use kurven_core::spaces::{P2, P3, ParamSpace, WorldSpace};
use kurven_core::surface::{ContourRefine, Surface};
use kurven_landscape::{LandscapeRequest, NativeLandscape};
use kurven_math::real;

type P = P2<ParamSpace>;
const TWO_PI: f64 = 2.0 * PI;

fn wrapped(a: f64) -> f64 {
    a - TWO_PI * (a / TWO_PI).round()
}

fn landscape(
    expression: &str,
    re: (f64, f64),
    im: (f64, f64),
    resolution: i64,
    truncation: Truncation,
) -> KurvenBundle {
    let request = LandscapeRequest {
        expression: expression.to_string(),
        domain: Domain::new(Interval::new(re.0, re.1), Interval::new(im.0, im.1)),
        resolution,
        truncation: Some(truncation),
        spacing: None,
        layers: Some(Vec::new()),
        name: String::new(),
    };
    NativeLandscape::build(&request, "native", true).unwrap()
}

/// The cell of a landscape's grid.
fn cell(surface: &Surface) -> f64 {
    let g = &surface.height;
    (g.domain.real.length().abs() / (g.width - 1) as f64).max(g.domain.imag.length().abs() / (g.height - 1) as f64)
}

fn on_magnitude_at_phase(level: f64, angles: &[f64]) -> Seeds {
    Seeds { field: ContourField::Magnitude, level, spacing: Spacing::Values(ContourField::Phase, angles.to_vec()) }
}

const SLOPE: DirectionField = DirectionField::Slope(ContourField::Magnitude);

fn context(region: Region, tiles: Vec<Affine2>) -> HatchContext {
    HatchContext { perimeter: None, region, tiles, base: 0.0 }
}

fn graph_curves(
    bundle: &KurvenBundle,
    field: DirectionField,
    seeds: &Seeds,
    region: Region,
    limits: &Limits,
) -> Vec<Traced> {
    bundle.surface.graph_integral_curves(&field, seeds, &[Affine2::IDENTITY], &region, bundle.refine.as_ref(), limits)
        [0]
    .1
    .clone()
}

fn distance_to(paths: &ParamPaths, p: Vec2) -> f64 {
    let mut best = f64::INFINITY;
    for path in paths {
        for w in path.windows(2) {
            let (a, b) = (w[0].v, w[1].v);
            let d = b - a;
            let dd = d.dot(d);
            let t = if dd > 0.0 { ((p - a).dot(d) / dd).clamp(0.0, 1.0) } else { 0.0 };
            best = best.min((a + t * d - p).length());
        }
    }
    best
}

struct Verdict {
    curves: usize,
    vertices: usize,
    true_worst: f64,
    vertex_worst: f64,
    drawn_worst: f64,
    ends: BTreeMap<End, usize>,
    example: String,
}

/// Slope lines of |f| seeded on |f| = `level` at `angles` of arg f, held
/// to the phase contours: the worst distance of a vertex or a chord's
/// midpoint from the true level set arg f = θ, |arg f − θ| / |∇ arg f|
/// (|∇ arg f| = |∇ log |f|| by Cauchy–Riemann, read by central
/// differences), and of a vertex from the refiner's drawn contour at θ,
/// both in cells; with how many curves and vertices, and why they end.
fn phase_verdict(bundle: &KurvenBundle, level: f64, angles: &[f64], crossing: &[f64], tiles: &[Affine2]) -> Verdict {
    let surface = &bundle.surface;
    let refine = bundle.refine.as_ref().unwrap();
    let magnitude = refine.magnitude.clone().unwrap();
    let phase = refine.phase.clone().unwrap();
    let c = cell(surface);
    let seeds = on_magnitude_at_phase(level, angles);
    let traced = surface.graph_integral_curves(&SLOPE, &seeds, tiles, &Region::Full, Some(refine), &Limits::default());
    // The refiner's phase contours at each angle, as the drawing has them.
    let rectangle = surface.parameter_rectangle().unwrap();
    let drawn = scalar_field::levels(
        &surface.field(ContourField::Phase, Some(refine), f64::INFINITY).unwrap(),
        angles,
        rectangle,
        &Trim::default(),
    );
    let m = |p: Vec2| magnitude(P2::from_vec(p));
    let ph = |p: Vec2| phase(P2::from_vec(p));
    let grad_log = |p: Vec2| {
        let h = 1e-6 * c;
        let l = |q: Vec2| real::log(m(q));
        Vec2::new(
            l(p + Vec2::new(h, 0.0)) - l(p - Vec2::new(h, 0.0)),
            l(p + Vec2::new(0.0, h)) - l(p - Vec2::new(0.0, h)),
        )
        .length()
            / (2.0 * h)
    };
    let mut v = Verdict {
        curves: 0,
        vertices: 0,
        true_worst: 0.0,
        vertex_worst: 0.0,
        drawn_worst: 0.0,
        ends: BTreeMap::new(),
        example: String::new(),
    };
    let mut drawn_example = String::new();
    for (_, set) in traced.iter().take(1) {
        for curve in set {
            v.curves += 1;
            *v.ends.entry(curve.backward).or_default() += 1;
            *v.ends.entry(curve.forward).or_default() += 1;
            let middle = curve.path[curve.path.len() / 2].v;
            // Swift's `min(by:)`: the first of the least.
            let mut k = 0;
            for j in 1..angles.len() {
                if wrapped(ph(middle) - angles[j]).abs() < wrapped(ph(middle) - angles[k]).abs() {
                    k = j;
                }
            }
            let theta = angles[k];
            let contour = &drawn[k].1;
            let mut samples: Vec<Vec2> = curve.path.iter().map(|p| p.v).collect();
            samples.extend(curve.path.windows(2).map(|w| 0.5 * (w[0].v + w[1].v)));
            // An end on a zero of f, where arg f is undefined: the samples
            // within a thousandth of a cell of it are not read.
            let zeros: Vec<Vec2> =
                [curve.path[0].v, curve.path[curve.path.len() - 1].v].into_iter().filter(|&z| m(z) < 1e-9).collect();
            for (i, &p) in samples.iter().enumerate() {
                if zeros.iter().any(|&z| (z - p).length() < 1e-3 * c) {
                    continue;
                }
                let (g, mp) = (grad_log(p), m(p));
                if !(g.is_finite() && g > 0.0 && mp > 0.0) {
                    continue;
                }
                let off = wrapped(ph(p) - theta).abs() / g / c;
                if i < curve.path.len() {
                    v.vertex_worst = v.vertex_worst.max(off);
                }
                if off > v.true_worst {
                    v.true_worst = off;
                    v.example = format!("θ = {theta:.3} at ({:.5}, {:.5})", p.x, p.y);
                }
                if i < curve.path.len() && !crossing.contains(&theta) {
                    v.vertices += 1;
                    let d = distance_to(contour, p) / c;
                    if d > v.drawn_worst {
                        v.drawn_worst = d;
                        drawn_example = format!(" drawn: θ = {theta:.3} at ({:.5}, {:.5}), |f| {mp:.3}", p.x, p.y);
                    }
                }
            }
        }
    }
    v.example += &drawn_example;
    v
}

/// "integral curves: slope lines of |f| are the phase contours, to the
/// refiner's tolerance, on sn, a rational function and exp(z)".
#[test]
fn slope_lines_of_the_modulus_are_the_phase_contours() {
    // arg sn = 0 runs through sn's critical points K + 2mK, where its
    // contours cross and the drawn ones leave the crossing open; a slope
    // line there ends at the critical point, on the true contour, and is
    // held to it alone.
    let angles = [-2.6, -2.0, -1.2, -0.5, 0.0, 0.4, 1.1, 1.9, 2.7];
    let cases: Vec<(&str, KurvenBundle, f64, Vec<f64>, usize)> = vec![
        (
            "sn(z, 0.64)",
            landscape("sn(z, 0.64)", (-4.0, 4.0), (-3.5, 3.5), 200, Truncation::Uniform(3.0)),
            0.6,
            vec![0.0],
            34,
        ),
        (
            "(z^2 - 1)/(z^2 + 1)",
            landscape("(z^2 - 1)/(z^2 + 1)", (-2.5, 2.5), (-2.0, 2.0), 160, Truncation::Uniform(4.0)),
            0.5,
            vec![],
            18,
        ),
        ("exp(z)", landscape("exp(z)", (-1.0, 1.0), (-1.0, 1.0), 48, Truncation::None), 1.0, vec![], 3),
    ];
    for (name, bundle, level, crossing, count) in cases {
        let v = phase_verdict(&bundle, level, &angles, &crossing, &[Affine2::IDENTITY]);
        assert!(
            v.curves == count && v.true_worst <= 0.02 && v.drawn_worst <= 0.02,
            "{name}: every slope line seeded on |f| = {level} at an angle of arg f stays on its phase contour: {} \
             curves, {} vertices; worst {:.1e} cell off arg f = θ at a vertex, {:.1e} at a chord's midpoint, {:.1e} \
             off the drawn contour; ends {:?} {}",
            v.curves,
            v.vertices,
            v.vertex_worst,
            v.true_worst,
            v.drawn_worst,
            v.ends,
            v.example
        );
    }
}

/// "integral curves: each stopping rule, on its own".
#[test]
fn each_stopping_rule_on_its_own() {
    // (z² − 1)/(z² + 1): zeros at ±1, poles at ±i.
    let rational = "(z^2 - 1)/(z^2 + 1)";
    let truncated = landscape(rational, (-2.5, 2.5), (-2.0, 2.0), 160, Truncation::Uniform(4.0));
    let h = &truncated.surface;
    let refine = truncated.refine.as_ref().unwrap();
    let magnitude = refine.magnitude.clone().unwrap();
    let m = |p: Vec2| magnitude(P2::from_vec(p));
    let c = cell(h);
    let seeds = on_magnitude_at_phase(0.5, &[-2.6, -1.2, 0.4, 1.9]);
    let curves = graph_curves(&truncated, SLOPE, &seeds, Region::Full, &Limits::default());
    // Downhill into a zero of f, carried to the zero itself; uphill to the
    // truncation edge, where |f| is the truncation height.
    let (mut at_zero, mut on_edge, mut wrong) = (0, 0, 0);
    let (mut zero_off, mut edge_off) = (0.0f64, 0.0f64);
    for curve in &curves {
        for (end, p) in [(curve.backward, curve.path[0]), (curve.forward, curve.path[curve.path.len() - 1])] {
            match end {
                End::Singular => {
                    at_zero += 1;
                    zero_off =
                        zero_off.max((p.v - Vec2::new(1.0, 0.0)).length().min((p.v - Vec2::new(-1.0, 0.0)).length()));
                }
                End::Boundary => {
                    on_edge += 1;
                    edge_off = edge_off.max((m(p.v) - 4.0).abs() / 4.0);
                }
                _ => wrong += 1,
            }
        }
    }
    assert!(
        curves.len() == 8 && at_zero == 8 && on_edge == 8 && wrong == 0 && zero_off <= 1e-12 && edge_off <= 1e-12,
        "on (z² − 1)/(z² + 1) truncated at 4, each slope line runs down into a zero of f and up to the truncation \
         edge, exactly: {} curves, {at_zero} at a zero (off by {zero_off:.1e}), {on_edge} on the edge (|f| off 4 by \
         {edge_off:.1e}), {wrong} otherwise",
        curves.len()
    );
    // Untruncated, uphill into a pole: it ends within a thousandth of a
    // cell of it, where the step it would take shrinks below the least.
    let open = landscape(rational, (-2.5, 2.5), (-2.0, 2.0), 160, Truncation::None);
    let pole_curves = graph_curves(&open, SLOPE, &seeds, Region::Full, &Limits::default());
    let pole_off = pole_curves
        .iter()
        .map(|t| {
            let last = t.path[t.path.len() - 1].v;
            (last - Vec2::new(0.0, 1.0)).length().min((last - Vec2::new(0.0, -1.0)).length()) / c
        })
        .fold(None, |a: Option<f64>, b| Some(a.map_or(b, |a| a.max(b))))
        .unwrap_or(f64::INFINITY);
    // Into a pole the field turns against any step that would cross it, so
    // the steps shrink to the least, the curve at the pole: the lifted
    // chords' sag is refined only so far, and a steep spire does not stop it.
    assert!(
        pole_curves.len() == 8 && pole_curves.iter().all(|t| t.forward == End::Singular) && pole_off <= 1e-3,
        "untruncated, each runs up into a pole of f and ends at it, under a thousandth of a cell short: {} curves, \
         worst {pole_off:.1e} cell short; ends {:?}",
        pole_curves.len(),
        pole_curves.iter().map(|t| t.forward).collect::<Vec<_>>()
    );
    // exp(z): the slope lines are the lines Im z = θ, out to the domain's
    // edge both ways; and to a footprint's edge, a diamond, where one cuts
    // the copy.
    let exp = landscape("exp(z)", (-1.0, 1.0), (-1.0, 1.0), 48, Truncation::None);
    let lines =
        graph_curves(&exp, SLOPE, &on_magnitude_at_phase(1.0, &[-0.7, 0.2, 0.9]), Region::Full, &Limits::default());
    let edge_ends = lines
        .iter()
        .map(|t| (t.path[0].x() + 1.0).abs().max((t.path[t.path.len() - 1].x() - 1.0).abs()))
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        lines.len() == 3
            && lines.iter().all(|t| t.backward == End::Boundary && t.forward == End::Boundary)
            && edge_ends <= 1e-12,
        "on exp(z) each runs from the domain's edge at Re z = −1 to its edge at Re z = 1, to the last bit: {} curves, \
         ends off the edge by {edge_ends:.1e}",
        lines.len()
    );
    let corners = [(0.8, 0.0), (0.0, 0.8), (-0.8, 0.0), (0.0, -0.8)];
    let diamond = BoundaryPerimeter::new(
        (0..4)
            .map(|k| {
                let (a, b) = (corners[k], corners[(k + 1) % 4]);
                PerimeterEdge { start: P2::new(a.0, a.1), end: P2::new(b.0, b.1), density: 8 }
            })
            .collect(),
    );
    let cut = graph_curves(
        &exp,
        SLOPE,
        &on_magnitude_at_phase(1.0, &[-0.5, 0.2, 0.6]),
        Region::Inside(diamond),
        &Limits::default(),
    );
    let footprint_off = cut
        .iter()
        .flat_map(|t| [t.path[0].v, t.path[t.path.len() - 1].v])
        .map(|p| (p.x.abs() + p.y.abs() - 0.8).abs())
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        cut.len() == 3
            && cut.iter().all(|t| t.backward == End::Boundary && t.forward == End::Boundary)
            && footprint_off <= 1e-12,
        "cut to a footprint, each ends on the footprint's perimeter where the copy stands, to the last bit: {} curves, \
         ends off the diamond by {footprint_off:.1e}",
        cut.len()
    );
    // The slope of arg f: its integral curves are the contours of |f|,
    // closed round the zeros of f. Seeded on arg f = 0.3 at |f| = 0.2 and
    // 0.4, each closes on its seed, on |f| = c all round.
    let loops = graph_curves(
        &truncated,
        DirectionField::Slope(ContourField::Phase),
        &Seeds {
            field: ContourField::Phase,
            level: 0.3,
            spacing: Spacing::Values(ContourField::Magnitude, vec![0.2, 0.4]),
        },
        Region::Full,
        &Limits::default(),
    );
    let mut loop_off = 0.0f64;
    for lp in &loops {
        let first = m(lp.path[0].v);
        let level = if (first - 0.4).abs() < (first - 0.2).abs() { 0.4 } else { 0.2 };
        for p in &lp.path {
            let e = 1e-6 * c;
            let grad = Vec2::new(
                m(p.v + Vec2::new(e, 0.0)) - m(p.v - Vec2::new(e, 0.0)),
                m(p.v + Vec2::new(0.0, e)) - m(p.v - Vec2::new(0.0, e)),
            )
            .length()
                / (2.0 * e);
            loop_off = loop_off.max((m(p.v) - level).abs() / grad / c);
        }
    }
    assert!(
        loops.len() == 4
            && loops.iter().all(|t| t.backward == End::Closed
                && t.forward == End::Closed
                && t.path[0] == t.path[t.path.len() - 1])
            && loop_off <= 0.02,
        "the slope lines of arg f close round the zeros on their seeds, on the contours of |f|: {} loops, worst \
         {loop_off:.1e} cell off |f| = c; ends {:?}",
        loops.len(),
        loops.iter().map(|t| (t.backward, t.forward)).collect::<Vec<_>>()
    );
    // The step limit: each way stops after so many steps.
    let capped = Limits { max_steps: 5, ..Limits::default() };
    let short = graph_curves(&exp, SLOPE, &on_magnitude_at_phase(1.0, &[0.2]), Region::Full, &capped);
    assert!(
        short.len() == 1
            && short[0].backward == End::StepLimit
            && short[0].forward == End::StepLimit
            && short[0].path.len() == 11,
        "at the step limit each way ends after that many steps: {:?}",
        short.iter().map(|t| (t.backward, t.forward, t.path.len())).collect::<Vec<_>>()
    );
    // A seed off the trimmed domain, over the truncation, starts nothing.
    let graph = h.graph(&[Affine2::IDENTITY], &Region::Full, Some(refine)).unwrap();
    let field = graph.height_slope(None);
    let curve = |x: f64, y: f64| graph.integral_curve(&field, P::new(x, y), &graph.copies[0], &Limits::default(), None);
    assert!(
        curve(0.0, 1.02).is_none() && curve(3.0, 0.0).is_none() && curve(0.5, 0.5).is_some(),
        "a seed over the truncation, or off the domain, starts no curve"
    );
}

/// "integral curves: the slope of a coordinate is raised by the graph's
/// metric, and a steep graph is followed to its end".
#[test]
fn the_slope_of_a_coordinate_is_raised_by_the_graphs_metric() {
    // The graph of |exp(z²/2)| = h = e^{(x² − y²)/2}: h_x = x h, h_y = −y h.
    // Its first fundamental form is g = I + ∇h ∇hᵀ, so the slope of u is
    // g⁻¹e₁ ∝ (1 + h_y², −h_x h_y) and of v g⁻¹e₂ ∝ (−h_x h_y, 1 + h_x²):
    // not along the axes where h_x h_y ≠ 0, as the plane's gradient is.
    let b = landscape("exp(z^2/2)", (-1.0, 1.0), (-1.0, 1.0), 48, Truncation::None);
    let graph = b.surface.graph(&[Affine2::IDENTITY], &Region::Full, b.refine.as_ref()).unwrap();
    let su = graph.slope(std::sync::Arc::new(|p: P| p.x()), None, None);
    let sv = graph.slope(std::sync::Arc::new(|p: P| p.y()), None, None);
    let (mut worst, mut turned, mut sampled) = (0.0f64, 0.0f64, 0);
    for i in 0..9 {
        for j in 0..9 {
            let (x, y) = (-0.9 + 0.225 * i as f64, -0.9 + 0.225 * j as f64);
            let h = real::exp((x * x - y * y) / 2.0);
            let (hx, hy) = (x * h, -y * h);
            for (field, closed) in
                [(&su, Vec2::new(1.0 + hy * hy, -hx * hy)), (&sv, Vec2::new(-hx * hy, 1.0 + hx * hx))]
            {
                let Some(w) = (field.value)(P::new(x, y)) else {
                    worst = f64::INFINITY;
                    continue;
                };
                sampled += 1;
                worst = worst.max((w.normalized() - closed.normalized()).length());
                let n = closed.normalized();
                turned =
                    turned.max(1.0 - if n.x == 0.0 { 1.0 } else { n.dot(Vec2::new(closed.x, 0.0).normalized()) }.abs());
            }
        }
    }
    // The general rule, the adjugate of g from the jet, agrees where the
    // graph is gentle.
    let mut general = 0.0f64;
    for p in [P::new(0.3, -0.4), P::new(-0.7, 0.2), P::new(0.5, 0.5)] {
        let jet = (graph.map)(p);
        for d in [Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(0.3, -2.0)] {
            let x = Face::raised_over_graph(d, Vec2::new(jet.du.z, jet.dv.z)).unwrap();
            let y = Face::raised(d, jet).unwrap();
            general = general.max((x - y).length() / x.length());
        }
    }
    assert!(
        sampled == 162 && worst <= 1e-8 && turned > 0.05 && general <= 1e-12,
        "on |exp(z²/2)|'s graph the slope of u is ∝ (1 + h_y², −h_x h_y) and of v ∝ (−h_x h_y, 1 + h_x²), g⁻¹ raised: \
         {sampled} samples, worst {worst:.1e} off the closed form's direction, which leaves its axis by up to \
         {turned:.2}; the jet's adjugate agrees to {general:.1e}"
    );
    // exp(20z): its slope lines are the lines Im z = θ, straight in the
    // plane, on a graph that rises to e^20. The chord's sag in space is
    // refined at most five halvings below the longest step, so the line
    // reaches the domain's edge at Re z = 1.
    let steep = landscape("exp(20*z)", (-1.0, 1.0), (-0.3, 0.3), 48, Truncation::None);
    let line = graph_curves(&steep, SLOPE, &on_magnitude_at_phase(1.0, &[0.0]), Region::Full, &Limits::default());
    let c = cell(&steep.surface);
    let off_axis = line.first().map_or(f64::INFINITY, |t| t.path.iter().map(|p| p.y().abs()).fold(0.0, f64::max));
    let l = &line[0];
    assert!(
        line.len() == 1
            && l.backward == End::Boundary
            && l.forward == End::Boundary
            && (l.path[0].x() + 1.0).abs() <= 1e-12
            && (l.path[l.path.len() - 1].x() - 1.0).abs() <= 1e-12
            && l.path.len() <= (2.0 / c * 64.0) as usize
            && off_axis <= 1e-9,
        "on exp(20z) the slope line runs from Re z = −1 to Re z = 1, to the domain's edge both ways, its vertices \
         bounded: {:?}/{:?} from {:.6} to {:.6}, {} vertices ({:.0} a cell), off Im z = 0 by {off_axis:.1e}",
        l.backward,
        l.forward,
        l.path[0].x(),
        l.path[l.path.len() - 1].x(),
        l.path.len(),
        l.path.len() as f64 / (2.0 / c)
    );
}

/// "integral curves: seeds once each, and no closer than the grid
/// resolves", its half on a landscape (the half on polylines is
/// kurven-core's).
#[test]
fn seeds_once_each_and_no_closer_than_the_grid_resolves() {
    let rational = landscape("(z^2 - 1)/(z^2 + 1)", (-2.5, 2.5), (-2.0, 2.0), 160, Truncation::Uniform(4.0));
    let count = |values: &[f64]| {
        graph_curves(&rational, SLOPE, &on_magnitude_at_phase(0.5, values), Region::Full, &Limits::default()).len()
    };
    let (one, cut) = (count(&[1.0]), count(&[PI]));
    let twice = [count(&[1.0, 1.0 + 2.0 * PI]), count(&[PI, -PI]), count(&[1.0, 1.0])];
    assert!(
        one == 2 && cut == 2 && twice == [2, 2, 2],
        "values the same modulo 2π seed once: |f| = 1/2 round each zero meets arg f = 1 once, and π once: arg f = 1: \
         {one}, π: {cut}; 1 and 1 + 2π, π and −π, 1 twice: {twice:?}"
    );
    // Every path refuses seeds closer than a cell: the layer's own check,
    // given the cell, and the derivation, which draws nothing.
    let c = cell(&rational.surface);
    let close = Seeds { field: ContourField::Magnitude, level: 0.5, spacing: Spacing::ArcLength(0.5 * c) };
    let refused =
        LayerSource::unsupported_integral_curves(&SLOPE, &close, &FaceSelector::Graph { tiled: false }, Some(c));
    let full = context(Region::Full, vec![Affine2::IDENTITY]);
    let drawn = rational.surface.integral_curves(
        &SLOPE,
        &close,
        &FaceSelector::Graph { tiled: false },
        &full,
        rational.refine.as_ref(),
        &Limits::default(),
    );
    let far = rational.surface.integral_curves(
        &SLOPE,
        &Seeds { field: ContourField::Magnitude, level: 0.5, spacing: Spacing::ArcLength(2.0 * c) },
        &FaceSelector::Graph { tiled: false },
        &full,
        rational.refine.as_ref(),
        &Limits::default(),
    );
    assert!(
        refused.as_deref().is_some_and(|r| r.contains("closer than a cell of the grid"))
            && drawn.is_empty()
            && !far.is_empty(),
        "seeds closer than a cell are refused by the layer's check and draw nothing; two cells apart they draw: {:?}; \
         {} and {} curves",
        refused,
        drawn.len(),
        far.len()
    );
}

/// "integral curves: seeds by arc length, and on a tiled graph each copy's
/// own".
#[test]
fn seeds_by_arc_length_and_on_a_tiled_graph_each_copys_own() {
    // exp(z) over [−1, 1]²: |f| = 1 is the segment Re z = 0, length 2 in
    // space, so seeds a quarter apart are the middles of its eight quarters,
    // and the slope lines through them Im z = −7/8 ... 7/8.
    let exp_landscape = landscape("exp(z)", (-1.0, 1.0), (-1.0, 1.0), 48, Truncation::None);
    let even = graph_curves(
        &exp_landscape,
        SLOPE,
        &Seeds { field: ContourField::Magnitude, level: 1.0, spacing: Spacing::ArcLength(0.25) },
        Region::Full,
        &Limits::default(),
    );
    let mut ys: Vec<f64> = even.iter().map(|t| t.path[0].y()).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let expected: Vec<f64> = (0..8).map(|k| -1.0 + 0.25 * (k as f64 + 0.5)).collect();
    let flat = even
        .iter()
        .flat_map(|t| t.path.iter().map(move |p| (p.y() - t.path[0].y()).abs()))
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        ys.len() == 8 && ys.iter().zip(&expected).all(|(a, b)| (a - b).abs() <= 1e-9) && flat <= 1e-9,
        "on exp(z), seeds a quarter apart along |f| = 1 are the middles of its quarters, each slope line level in \
         Im z: {} seeds at Im z = {ys:?}; Im z varies by {flat:.1e} along a line",
        ys.len()
    );
    // Tiled by a reflection across Re z = 2 and a quarter turn moved to
    // Re z = 8: every copy's slope lines are its image of Im z = θ, at
    // |f| = e^x of the point of the domain it came from.
    let tiles = vec![
        Affine2::IDENTITY,
        Affine2 { a: -1.0, b: 0.0, tx: 4.0, c: 0.0, d: 1.0, ty: 0.0 },
        Affine2 { a: 0.0, b: -1.0, tx: 8.0, c: 1.0, d: 0.0, ty: 0.0 },
    ];
    let angles = [-0.7, 0.2, 0.9];
    let paths = exp_landscape.surface.integral_curves(
        &SLOPE,
        &on_magnitude_at_phase(1.0, &angles),
        &FaceSelector::Graph { tiled: true },
        &context(Region::Full, tiles.clone()),
        exp_landscape.refine.as_ref(),
        &Limits::default(),
    );
    let (mut off, mut off_z, mut lost, mut per_tile) = (0.0f64, 0.0f64, 0, [0, 0, 0]);
    let back = |k: usize, w: P3<WorldSpace>| tiles[k].inverse().unwrap().apply(w);
    for path in &paths {
        let mut tile: Option<usize> = None;
        for &w in path {
            let Some(k) = (0..tiles.len()).find(|&k| {
                let q = back(k, w);
                q.x().abs() <= 1.0 + 1e-9 && q.y().abs() <= 1.0 + 1e-9
            }) else {
                lost += 1;
                continue;
            };
            if tile.is_none() {
                tile = Some(k);
                per_tile[k] += 1;
            }
            let q = back(k, w);
            off = off.max(angles.iter().map(|a| (q.y() - a).abs()).fold(f64::INFINITY, f64::min));
            off_z = off_z.max((w.z() - real::exp(q.x())).abs() / real::exp(q.x()));
        }
    }
    assert!(
        paths.len() == 9 && per_tile == [3, 3, 3] && lost == 0 && off <= 1e-9 && off_z <= 1e-12,
        "on a tiled graph each copy's slope lines are its image of Im z = θ, lifted to |f|: {} paths, worst {off:.1e} \
         off Im z = θ, {off_z:.1e} off |f|; per tile {per_tile:?}",
        paths.len()
    );
    // sn on two tiles, its phase contours held as on one.
    let sn = landscape("sn(z, 0.64)", (-4.0, 4.0), (-3.5, 3.5), 200, Truncation::Uniform(3.0));
    let two_tiles = vec![Affine2::IDENTITY, Affine2 { a: 1.0, b: 0.0, tx: 8.0, c: 0.0, d: 1.0, ty: 0.0 }];
    let tiled = sn.surface.integral_curves(
        &SLOPE,
        &on_magnitude_at_phase(0.6, &[-2.0, 1.1]),
        &FaceSelector::Graph { tiled: true },
        &context(Region::Full, two_tiles.clone()),
        sn.refine.as_ref(),
        &Limits::default(),
    );
    let refine = sn.refine.as_ref().unwrap();
    let (phase, sn_magnitude) = (refine.phase.clone().unwrap(), refine.magnitude.clone().unwrap());
    let sn_cell = cell(&sn.surface);
    let (mut sn_off, mut per_copy) = (0.0f64, [0, 0]);
    for path in &tiled {
        // The copy a path lies on, read at its middle, off the tile boundary
        // its ends can stand on.
        let k = if path[path.len() / 2].x() > 4.0 { 1 } else { 0 };
        per_copy[k] += 1;
        for &w in path {
            let q = two_tiles[k].inverse().unwrap().apply(w);
            let (p, e) = (Vec2::new(q.x(), q.y()), 1e-6 * sn_cell);
            let l = |r: Vec2| real::log(sn_magnitude(P2::from_vec(r)));
            let g = Vec2::new(
                l(p + Vec2::new(e, 0.0)) - l(p - Vec2::new(e, 0.0)),
                l(p + Vec2::new(0.0, e)) - l(p - Vec2::new(0.0, e)),
            )
            .length()
                / (2.0 * e);
            if !(sn_magnitude(P2::from_vec(p)) > 1e-9 && g.is_finite() && g > 0.0 && 1.0 / g > 1e-3 * sn_cell) {
                continue;
            }
            let ph = phase(P2::from_vec(p));
            let near = [-2.0, 1.1].iter().map(|t| wrapped(ph - t).abs()).fold(f64::INFINITY, f64::min);
            sn_off = sn_off.max(near / g / sn_cell);
        }
    }
    assert!(
        per_copy[0] == per_copy[1] && per_copy[0] >= 4 && sn_off <= 1e-5,
        "sn's slope lines on two tiles, each carried back by its tile, lie on their phase contours on both: {} paths, \
         worst {sn_off:.1e} cell off arg f = θ; per copy {per_copy:?}",
        tiled.len()
    );
}

#[test]
#[ignore = "needs Scene visibility (visibleLayers, HeightfieldVisibility, the dense cast), ported in parallel"]
fn on_f_seen_exactly_where_the_dense_cast_against_f_says() {}

/// The slope-lines golden of `tests/fixtures/fields` (PR #60), read and
/// refined as the bake reads it (`kurven-cli`'s `loadScene` attaches
/// `NativeLandscape.refiner(for:)`), its slope lines before visibility:
/// twelve seeds on |f| = 1/2 round each of the two zeros of
/// (z² − 1)/(z² + 1), each line down into its zero and up to the
/// truncation edge round a pole or out to the domain's edge, on its phase
/// contour to the refiner's tolerance. But arg f = ±π, which runs along the
/// real axis from each zero into f's saddle at 0 (f' = 4z/(z² + 1)²): those
/// two end there, singular, within a thousandth of a cell, and are held to
/// the true contour alone, as sn's lines through its critical points are.
/// The drawing itself is compared with the bake.
#[test]
fn the_slope_lines_golden_reads_and_draws_its_slope_lines() {
    let read = KurvenBundle::read(fixtures::shared("fields/landscape_slope_lines.kurven")).unwrap();
    let refiner = NativeLandscape::refiner_for_bundle(&read).expect("the golden records its expression");
    let bundle = read.refined(Some(refiner.refine_hook()));
    let spec = bundle.manifest.layer("slope_lines").unwrap().clone();
    let LayerSource::IntegralCurves { field, seeds, faces } = &spec.source else { panic!("slope_lines is a slope") };
    let refine: &ContourRefine = bundle.refine.as_ref().unwrap();
    let occluder = &bundle.manifest.occluder;
    let traced = bundle.surface.graph_integral_curves(
        field,
        seeds,
        &occluder.tiles,
        &occluder.region,
        Some(refine),
        &Limits::default(),
    );
    let curves = &traced[0].1;
    let world = bundle.surface.integral_curves(
        field,
        seeds,
        faces,
        &HatchContext::new(occluder),
        Some(refine),
        &Limits::default(),
    );
    let magnitude = refine.magnitude.clone().unwrap();
    let m = |p: Vec2| magnitude(P2::from_vec(p));
    let truncation = bundle.manifest.truncation.highest();
    let d = bundle.manifest.domain;
    let mut wrong = Vec::new();
    for t in curves {
        let (first, last) = (t.path[0].v, t.path[t.path.len() - 1].v);
        let at_zero = |p: Vec2| (p - Vec2::new(1.0, 0.0)).length().min((p - Vec2::new(-1.0, 0.0)).length()) <= 1e-12;
        let on_edge = |p: Vec2| {
            (m(p) - truncation).abs() <= 1e-12 * truncation
                || (p.x - d.real.lo).abs() <= 1e-12
                || (p.x - d.real.hi).abs() <= 1e-12
                || (p.y - d.imag.lo).abs() <= 1e-12
                || (p.y - d.imag.hi).abs() <= 1e-12
        };
        let at_saddle = |p: Vec2| p.length() <= 1e-3 * cell(&bundle.surface);
        let forward_ok =
            (t.forward == End::Boundary && on_edge(last)) || (t.forward == End::Singular && at_saddle(last));
        if !(t.backward == End::Singular && at_zero(first) && forward_ok) {
            wrong.push(format!("{:?} at {first:?} / {:?} at {last:?}", t.backward, t.forward));
        }
    }
    let angles = match &seeds.spacing {
        Spacing::Values(_, values) => values.clone(),
        Spacing::ArcLength(_) => panic!("the golden seeds at values"),
    };
    let saddle: Vec<f64> = angles.iter().copied().filter(|a| a.abs() == PI).collect();
    let v = phase_verdict(&bundle, seeds.level, &angles, &saddle, &occluder.tiles);
    assert!(
        curves.len() == 24 && world.len() == 24 && wrong.is_empty() && v.true_worst <= 0.02 && v.drawn_worst <= 0.02,
        "the golden's slope lines: {} curves ({} in space), each from a zero of f to the truncation or the domain's \
         edge (otherwise: {wrong:?}); worst {:.1e} cell off arg f = θ, {:.1e} off the drawn contour",
        curves.len(),
        world.len(),
        v.true_worst,
        v.drawn_worst
    );
    // In space each is lifted to |f|, f's own.
    let lifted = world.iter().flatten().map(|w| (w.z() - m(Vec2::new(w.x(), w.y()))).abs()).fold(0.0, f64::max);
    assert!(lifted == 0.0, "each vertex lifted to |f| exactly: {lifted:e}");
}
