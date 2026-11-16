//! The refiner, held to Swift's suites that need no bundle, and to the
//! mathematics directly: level sets whose shape is known in closed form.

// The points of the Swift suites are written as Swift writes them, with 17
// digits, so the literals are the same in both.
#![allow(clippy::excessive_precision)]

use std::f64::consts::PI;

use kurven_core::contour;
use kurven_core::grid::Grid2D;
use kurven_core::manifest::{ContourField, Domain, Interval};
use kurven_core::spaces::{DomainSpace, P2};
use kurven_core::surface::ContourRefine;
use kurven_landscape::{Catalog, ContourRefiner, LandscapeRequest, NativeLandscape, SingularPoint};
use kurven_math::{Complex, Expression};

type P = P2<DomainSpace>;
const EVERYWHERE: std::ops::RangeInclusive<f64> = ContourRefine::EVERYWHERE;
/// A point the search should find: where, and of what order.
type Expected = (f64, f64, i64);

fn domain(r0: f64, r1: f64, i0: f64, i1: f64) -> Domain {
    Domain::new(Interval::new(r0, r1), Interval::new(i0, i1))
}

/// A landscape's refiner and its float32 grids, as `NativeLandscape.build`
/// makes them (`Surface`'s height and phase grids are these).
struct Landscape {
    refiner: ContourRefiner,
    height: Grid2D<f32>,
    phase: Grid2D<f32>,
    magnitudes: Vec<f64>,
}

fn landscape(expression: &str, domain: Domain, resolution: i64) -> Landscape {
    let compiled = Expression::compile(expression).unwrap();
    let samples = NativeLandscape::sample(&compiled, domain, resolution).unwrap();
    let s = samples.shape;
    Landscape {
        refiner: NativeLandscape::refiner(expression, domain, s).unwrap(),
        height: Grid2D::new(s.n_real, s.n_imag, domain, NativeLandscape::quantize(&samples.magnitude)),
        phase: Grid2D::new(s.n_real, s.n_imag, domain, NativeLandscape::quantize(&samples.phase)),
        magnitudes: samples.magnitude,
    }
}

fn norm(p: P) -> f64 {
    (p.x() * p.x() + p.y() * p.y()).sqrt()
}

fn vertices(runs: &[Vec<P>]) -> usize {
    runs.iter().map(Vec::len).sum()
}

/// The distance from v to the nearest segment of `runs`, as the Swift suite
/// measures it.
fn distance_to(v: P, runs: &[Vec<P>]) -> f64 {
    let mut best = f64::INFINITY;
    for run in runs {
        for w in run.windows(2) {
            let (p, q) = (w[0], w[1]);
            let (dx, dy) = (q.x() - p.x(), q.y() - p.y());
            let l2 = dx * dx + dy * dy;
            let t = if l2 > 0.0 { (((v.x() - p.x()) * dx + (v.y() - p.y()) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
            let (ex, ey) = (p.x() + t * dx - v.x(), p.y() + t * dy - v.y());
            best = best.min((ex * ex + ey * ey).sqrt());
        }
    }
    best
}

fn found_as_expected(found: &[SingularPoint], expected: &[(f64, f64, i64)], cell: f64) -> bool {
    expected.iter().all(|&(x, y, order)| {
        found.iter().any(|s| {
            s.order == order && (s.point.x() - x).abs() <= 1e-9 * cell && (s.point.y() - y).abs() <= 1e-9 * cell
        })
    }) && found.len() == expected.len()
}

fn describe(found: &[SingularPoint]) -> String {
    found
        .iter()
        .map(|s| format!("order {} at {:.12}{:+.12}i", s.order, s.point.x(), s.point.y()))
        .collect::<Vec<_>>()
        .join(", ")
}

// MARK: Swift's suites

/// `Expressions.swift`'s "refine: a zero or pole of order more than one is
/// found". Swift builds the bundle and asks `refiner(for: bundle)`, which is
/// `refiner(for: expression, domain:, shape:)` at the bundle's grid.
#[test]
fn a_zero_or_pole_of_order_more_than_one_is_found() {
    let square = domain(-1.0, 1.0, -1.0, 1.0);
    let cases: Vec<(&str, Vec<Expected>)> = vec![
        ("0.3/(z - 0.2 - 0.13*i)^2", vec![(0.2, 0.13, -2)]),
        ("0.1/(z - 0.2 - 0.13*i)^3", vec![(0.2, 0.13, -3)]),
        ("0.03/(z - 0.2 - 0.13*i)^4", vec![(0.2, 0.13, -4)]),
        ("(z - 0.2 - 0.13*i)^3", vec![(0.2, 0.13, 3)]),
        // And the zero between them: 0.2 w² + 0.3 w - 0.12 = 0, w = z - c.
        (
            "0.3/(z - 0.2 - 0.13*i)^2 + 0.2/(z - 0.6 - 0.13*i)",
            vec![(0.2, 0.13, -2), (0.6, 0.13, -1), (0.2 + (0.186f64.sqrt() - 0.3) / 0.4, 0.13, 1)],
        ),
    ];
    for (expression, expected) in cases {
        let shape = NativeLandscape::grid_shape(square, 24).unwrap();
        let refiner = NativeLandscape::refiner(expression, square, shape).unwrap();
        let found = refiner.singular_points();
        assert!(found_as_expected(&found, &expected, refiner.cell()), "{expression}: {}", describe(&found));
    }
}

/// `Expressions.swift`'s "refine: a zero or pole the grid's nodes do not see
/// is found, by the winding number of its cell".
#[test]
fn a_zero_or_pole_the_nodes_do_not_see_is_found_by_the_winding_number_of_its_cell() {
    let window = domain(-0.5, 1.5, -1.0, 1.0);
    let pair =
        |p: (f64, f64), q: (f64, f64)| vec![(p.0, p.1, -1), (q.0, q.1, -1), ((p.0 + q.0) / 2.0, (p.1 + q.1) / 2.0, 1)];
    let cases: Vec<(String, i64, Vec<Expected>)> = vec![
        ("1/(z - (0.4804 - 0.0550i)) + 1/(z - (0.5196 + 0.0550i))".into(), 24, pair((0.4804, -0.055), (0.5196, 0.055))),
        (
            "1/(z - (0.59076180115334842-0.093833748934804229i)) + 1/(z - (0.5145013567413883-0.13573974784644616i))"
                .into(),
            20,
            pair((0.59076180115334842, -0.093833748934804229), (0.5145013567413883, -0.13573974784644616)),
        ),
        (
            "1/(z - (0.42240022182367271-0.40227584782117776i)) + 1/(z - (0.37140972304490555-0.48709752678493168i))"
                .into(),
            20,
            pair((0.42240022182367271, -0.40227584782117776), (0.37140972304490555, -0.48709752678493168)),
        ),
        (
            "(z - (0.40691152265074942-0.15224404800258995i))".to_string()
                + "/((z - (0.62359753483716163-0.13770458918473533i))*(z - (0.57445433013745983-0.14044013384631204i)))",
            16,
            vec![
                (0.40691152265074942, -0.15224404800258995, 1),
                (0.62359753483716163, -0.13770458918473533, -1),
                (0.57445433013745983, -0.14044013384631204, -1),
            ],
        ),
    ];
    for (expression, resolution, expected) in cases {
        let shape = NativeLandscape::grid_shape(window, resolution).unwrap();
        let refiner = NativeLandscape::refiner(&expression, window, shape).unwrap();
        let found = refiner.singular_points();
        assert!(
            found_as_expected(&found, &expected, refiner.cell()),
            "{expression} at res {resolution}: {}",
            describe(&found)
        );
    }
}

/// `Expressions.swift`'s "refine: by an accumulation of zeros or poles, each
/// order is read round the point alone".
#[test]
fn by_an_accumulation_each_order_is_read_round_the_point_alone() {
    let window = domain(-0.5, 1.5, -1.0, 1.0);
    for (expression, sign) in [("sin(1/z)", 1), ("1/sin(1/z)", -1)] {
        let shape = NativeLandscape::grid_shape(window, 400).unwrap();
        let refiner = NativeLandscape::refiner(expression, window, shape).unwrap();
        let found = refiner.singular_points();
        let other: Vec<&SingularPoint> = found.iter().filter(|s| s.order != sign).collect();
        assert!(
            found.len() > 50 && other.is_empty(),
            "{expression}: every point found is of order {sign}: {} found, {} not",
            found.len(),
            other.len()
        );
    }
}

/// `Expressions.swift`'s "refine: a phase line is placed only where it can be
/// drawn", on the grids `NativeLandscape.build` would give the bundle.
#[test]
fn a_phase_line_is_placed_only_where_it_can_be_drawn() {
    let request = LandscapeRequest::from_preset(Catalog::native().preset("tan").unwrap(), 240);
    let plain = landscape(&request.expression, request.domain, request.resolution);
    let truncation =
        request.truncation.clone().unwrap_or_else(|| NativeLandscape::default_truncation(&plain.magnitudes));
    let truncation_height = truncation.highest();
    let refiner = &plain.refiner;
    let cell = refiner.cell();
    let drawn = |p: P| refiner.magnitude(Complex::new(p.x(), p.y())) <= truncation_height;
    let (mut everywhere, mut under, mut lost, mut added) = (0, 0, 0, 0);
    for (level, paths) in contour::levels(&plain.phase, &[-2.618, -1.047, 0.524, 2.094]) {
        let all = refiner.refine(ContourField::Phase, level, &paths, &[], f64::INFINITY, EVERYWHERE);
        let truncated = refiner.refine(ContourField::Phase, level, &paths, &[], 1.1 * truncation_height, EVERYWHERE);
        everywhere += vertices(&all);
        under += vertices(&truncated);
        for v in all.iter().flatten() {
            if drawn(*v) && distance_to(*v, &truncated) > 0.05 * cell {
                lost += 1;
            }
        }
        for v in truncated.iter().flatten() {
            if drawn(*v) && distance_to(*v, &all) > 0.05 * cell {
                added += 1;
            }
        }
    }
    assert!(
        lost == 0 && added == 0,
        "below the truncation, the lines placed under a ceiling are the lines placed everywhere: {lost} lost, {added} added"
    );
    assert!(
        under * 3 < everywhere * 2,
        "and a good part of the placing was over the truncation: {under} of {everywhere}"
    );
}

// MARK: the mathematics

/// Level sets of |z| are circles about 0. Every refined vertex lies on its
/// circle to the root finder's tolerance, and every chord's midpoint lies
/// within the stated 0.02 cells of it: the sagitta, in closed form, and the
/// refiner's own estimate (`chord_errors`) agree on that.
#[test]
fn circles_of_the_modulus_of_z_are_placed_to_the_tolerance() {
    let window = domain(-1.0, 1.0, -1.0, 1.0);
    let l = landscape("z", window, 41);
    let cell = l.refiner.cell();
    for r in [0.13, 0.37, 0.8] {
        let paths = contour::lines(&l.height, r, None);
        let runs = l.refiner.refine(ContourField::Magnitude, r, &paths, &[], f64::INFINITY, EVERYWHERE);
        assert_eq!(runs.len(), 1, "|z| = {r} is one curve");
        let run = &runs[0];
        assert_eq!(run[0], run[run.len() - 1], "and closed");
        let grid_vertices = vertices(&paths);
        // The grid's chords sag 0.05 cells on the smallest circle (c² / 8r for
        // chords about a cell long), and 0.008 on the largest, which needs none.
        assert!(
            run.len() >= grid_vertices && (r > 0.5 || run.len() > grid_vertices),
            "{} vertices from {grid_vertices}",
            run.len()
        );
        // A vertex the float32 grid put within rounding of a node is left
        // there when it is within a hundredth of the tolerance of the level
        // (`snap`): 1e-5 cells here. Every other vertex is the root finder's.
        for &p in run {
            assert!((norm(p) - r).abs() <= 1e-2 * 0.02 * cell, "a vertex off |z| = {r} by {:e}", norm(p) - r);
        }
        let snapped = run.iter().filter(|p| (norm(**p) - r).abs() <= 1e-9).count();
        assert!(snapped * 10 >= run.len() * 9, "most vertices on the level to 1e-9: {snapped} of {}", run.len());
        for w in run.windows(2) {
            let m = P::new((w[0].x() + w[1].x()) / 2.0, (w[0].y() + w[1].y()) / 2.0);
            let sagitta = r - norm(m);
            assert!(
                sagitta >= -1e-2 * 0.02 * cell && sagitta <= 0.02 * cell,
                "a chord of |z| = {r} {:.4} cells off",
                sagitta / cell
            );
        }
        let worst = l.refiner.chord_errors(run, ContourField::Magnitude, r).into_iter().fold(0.0, f64::max);
        assert!(worst <= 0.02, "the refiner's own estimate: {worst}");
        let worst = l.refiner.position_errors(run, ContourField::Magnitude, r).into_iter().fold(0.0, f64::max);
        assert!(worst <= 1e-2 * 0.02, "vertices on the level: {worst}");
    }
}

/// The modulus of a polynomial, |z³ − 1| = c: three ovals round the cube
/// roots of unity below c = 1, one curve above. Placed on the level, chords
/// within 0.02 cells, the grid's few vertices at least doubled.
#[test]
fn level_sets_of_a_polynomial_are_placed_to_the_tolerance() {
    let window = domain(-1.6, 1.6, -1.6, 1.6);
    let l = landscape("z^3 - 1", window, 33);
    for (c, curves) in [(0.6, 3), (1.6, 1)] {
        let paths = contour::lines(&l.height, c, None);
        let runs = l.refiner.refine(ContourField::Magnitude, c, &paths, &[], f64::INFINITY, EVERYWHERE);
        assert_eq!(runs.len(), curves, "|z³ − 1| = {c}");
        for run in &runs {
            assert_eq!(run[0], run[run.len() - 1], "each closed");
            for &p in run {
                let z = Complex::new(p.x(), p.y());
                let v = (z * z * z - Complex::ONE).magnitude();
                assert!((v - c).abs() <= 1e-9, "a vertex at |f| = {v}");
            }
            let worst = l.refiner.chord_errors(run, ContourField::Magnitude, c).into_iter().fold(0.0, f64::max);
            assert!(worst <= 0.02, "chords within 0.02 cells: {worst}");
        }
    }
}

/// A ring the grid cannot see: a level of |z| below |f| at every corner of
/// the cell the zero lies in has no grid curve, and the hook's rings draw
/// it, a circle of that radius about the zero.
#[test]
fn a_ring_the_grid_cannot_see_is_seeded_from_the_zero() {
    // Nodes at -1.05 + 0.1 k: the zero at 0 is the centre of a cell, the
    // corners |f| = 0.0707 from it.
    let window = domain(-1.05, 0.95, -1.05, 0.95);
    let l = landscape("z", window, 21);
    let cell = l.refiner.cell();
    let points = l.refiner.singular_points();
    assert_eq!(points.len(), 1);
    assert_eq!(points[0].order, 1);
    assert!(norm(points[0].point) <= 1e-12, "the zero, at {:?}", points[0].point);
    let level = 0.03;
    assert!(contour::lines(&l.height, level, None).is_empty(), "the grid sees nothing at {level}");
    let hook = l.refiner.hook(points.clone());
    let rings = (hook.rings.as_ref().expect("a zero to seed from"))(level, &[]);
    assert_eq!(rings.len(), 1, "one ring");
    let ring = &rings[0];
    assert_eq!(ring[0], ring[ring.len() - 1], "closed");
    assert!(ring.len() >= 25);
    for &p in ring {
        assert!((norm(p) - level).abs() <= 1e-9, "on |z| = {level}: {:e}", norm(p) - level);
    }
    for w in ring.windows(2) {
        let m = P::new((w[0].x() + w[1].x()) / 2.0, (w[0].y() + w[1].y()) / 2.0);
        assert!(level - norm(m) <= 0.02 * cell, "chords within the tolerance");
    }
    // A level the grid did see is left to the grid.
    let seen = contour::lines(&l.height, 0.3, None);
    assert!(!seen.is_empty());
    assert!((hook.rings.as_ref().unwrap())(0.3, &seen).is_empty(), "a level the grid sees has no seeded ring");
}

/// The phase lines of z³ are rays from its triple zero: at a level θ, the
/// three rays at angles (θ + 2πk)/3. Each refined line starts on the zero,
/// where the normal form seeds it, lies on its ray, and ends on the
/// window's edge.
#[test]
fn phase_lines_of_z_cubed_are_rays_from_the_zero() {
    let window = domain(-1.0125, 0.9875, -1.0125, 0.9875);
    let l = landscape("z^3", window, 41);
    let points = l.refiner.singular_points();
    assert_eq!(points.len(), 1, "one point: {points:?}");
    assert_eq!(points[0].order, 3, "of order three");
    assert!((points[0].coefficient - Complex::ONE).magnitude() < 1e-6, "with coefficient 1: {}", points[0].coefficient);
    for theta in [0.5, -2.0, 2.9] {
        let paths: Vec<Vec<P>> = contour::levels(&l.phase, &[theta]).into_iter().flat_map(|(_, paths)| paths).collect();
        let runs = l.refiner.refine(ContourField::Phase, theta, &paths, &points, f64::INFINITY, EVERYWHERE);
        assert_eq!(runs.len(), 3, "θ = {theta}: three rays");
        let mut angles = Vec::new();
        for run in &runs {
            let (first, last) = (run[0], run[run.len() - 1]);
            let (origin, edge) = if norm(first) < norm(last) { (first, last) } else { (last, first) };
            assert!(norm(origin) <= 1e-9, "θ = {theta}: a ray from the zero, not {origin:?}");
            assert!(!window_interior(edge), "to the edge, not {edge:?}");
            for &p in run.iter().filter(|p| norm(**p) > 1e-6) {
                let off = ContourRefiner::wrapped(3.0 * kurven_math::real::atan2(p.y(), p.x()) - theta);
                assert!(off.abs() * norm(p) <= 1e-9, "θ = {theta}: a vertex {off:e} off its ray");
            }
            angles.push(kurven_math::real::atan2(edge.y(), edge.x()));
        }
        angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for k in 0..3 {
            let gap = ContourRefiner::wrapped(angles[(k + 1) % 3] - angles[k]);
            assert!((gap.abs() - 2.0 * PI / 3.0).abs() < 1e-9, "the rays a third of a turn apart: {angles:?}");
        }
    }
}

/// The branch cut, where arg f passes from π to −π, is the zero level of
/// arg(−f). Marching squares on the wrapped grid emits a bundle of
/// crossings along it; refined, it is drawn once, as one line: for z the
/// negative real axis from the zero to the edge, for z³ three rays.
#[test]
fn the_branch_cut_is_drawn_once() {
    let window = domain(-1.0125, 0.9875, -1.0125, 0.9875);
    for (expression, rays) in [("z", 1), ("z^3", 3)] {
        let l = landscape(expression, window, 41);
        let points = l.refiner.singular_points();
        let paths = contour::cut(&l.phase);
        assert!(!paths.is_empty());
        let runs = l.refiner.refine(ContourField::Phase, PI, &paths, &points, f64::INFINITY, EVERYWHERE);
        assert_eq!(runs.len(), rays, "{expression}: the cut as {rays} lines, not {}", runs.len());
        for run in &runs {
            for &p in run.iter().filter(|p| norm(**p) > 1e-6) {
                let k = rays as f64;
                let off = ContourRefiner::wrapped(k * kurven_math::real::atan2(p.y(), p.x()) - PI);
                assert!(off.abs() * norm(p) <= 1e-9, "{expression}: a vertex {off:e} off the cut");
            }
            let (a, b) = (run[0], run[run.len() - 1]);
            assert!(norm(a).min(norm(b)) <= 1e-9, "{expression}: from the zero");
        }
        // Drawn once: no vertex of one line lies on another away from the zero.
        for (i, run) in runs.iter().enumerate() {
            let others: Vec<Vec<P>> =
                runs.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, r)| r.clone()).collect();
            for &p in run.iter().filter(|p| norm(**p) > 0.1) {
                assert!(others.is_empty() || distance_to(p, &others) > 0.1, "{expression}: a stretch drawn twice");
            }
        }
        if expression == "z" {
            assert!(runs[0].iter().all(|p| p.y().abs() <= 1e-12 && p.x() <= 1e-12), "along the negative real axis");
        }
    }
}

/// Newton on f and on 1/f, Schröder past forty steps, and the winding
/// number, on z²(z − 0.5)/(z + 0.3i)³.
#[test]
fn zeros_poles_and_winding_numbers() {
    let window = domain(-1.0, 1.0, -1.0, 1.0);
    let l = landscape("z^2*(z - 0.5)/(z + 0.3*i)^3", window, 40);
    let r = &l.refiner;
    let near = |p: Option<P>, x: f64, y: f64| p.is_some_and(|p| (p.x() - x).abs() <= 1e-9 && (p.y() - y).abs() <= 1e-9);
    assert!(near(r.zero_near(P::new(0.03, -0.02)), 0.0, 0.0), "the double zero, by Schröder's step");
    assert!(near(r.zero_near(P::new(0.47, 0.03)), 0.5, 0.0), "the simple zero");
    assert!(near(r.pole_near(P::new(0.02, -0.27)), 0.0, -0.3), "the triple pole, on 1/f");
    assert_eq!(r.winding_number(P::new(0.0, 0.0), 0.1), Some(2));
    assert_eq!(r.winding_number(P::new(0.5, 0.0), 0.1), Some(1));
    assert_eq!(r.winding_number(P::new(0.0, -0.3), 0.1), Some(-3));
    assert_eq!(r.winding_number(P::new(0.0, 0.0), 0.9), Some(0), "two, one, less three");
    assert_eq!(r.winding_number(P::new(0.25, 0.0), 0.25), None, "through a zero, unreadable");
    let found = r.singular_points();
    assert!(
        found_as_expected(&found, &[(0.0, 0.0, 2), (0.5, 0.0, 1), (0.0, -0.3, -3)], r.cell()),
        "{}",
        describe(&found)
    );
    // The leading coefficient at the simple zero: f'(0.5) = 0.25 / (0.5 + 0.3i)³.
    let s = found.iter().find(|s| s.order == 1).unwrap();
    let w = Complex::new(0.5, 0.3);
    let want = Complex::real(0.25) / (w * w * w);
    assert!((s.coefficient - want).magnitude() < 1e-3 * want.magnitude(), "{} against {want}", s.coefficient);
}

/// The hook builds the `ContourRefine` the surface reads: its contours are
/// the refiner's, remembered to the bit, and its fork places the same.
#[test]
fn the_hook_is_the_refiner_remembered() {
    let window = domain(-2.0, 2.0, -2.0, 2.0);
    let l = landscape("1/(z^3 - 1)", window, 48);
    let hook = l.refiner.refine_hook();
    assert_eq!(hook.tolerance, 0.02);
    let paths: Vec<Vec<P>> = contour::levels(&l.phase, &[1.0]).into_iter().flat_map(|(_, p)| p).collect();
    let points = l.refiner.singular_points();
    assert_eq!(points.len(), 3, "three simple poles");
    let direct = l.refiner.refine(ContourField::Phase, 1.0, &paths, &points, f64::INFINITY, EVERYWHERE);
    let first = (hook.contours)(ContourField::Phase, 1.0, &paths, f64::INFINITY, EVERYWHERE);
    let again = (hook.contours)(ContourField::Phase, 1.0, &paths, f64::INFINITY, EVERYWHERE);
    let forked = (hook.forked().contours)(ContourField::Phase, 1.0, &paths, f64::INFINITY, EVERYWHERE);
    assert_eq!(first, direct, "the hook places what the refiner places");
    assert_eq!(again, first, "and remembers it");
    assert_eq!(forked, first, "and so does a fork");
    // The phase and magnitude it hands on are the refiner's.
    let p = P::new(0.3, 0.7);
    assert_eq!((hook.magnitude.as_ref().unwrap())(p), l.refiner.magnitude(Complex::new(0.3, 0.7)));
    assert_eq!((hook.phase.as_ref().unwrap())(p).to_bits(), l.refiner.phase(Complex::new(0.3, 0.7)).to_bits());
    assert!(hook.rings.is_none(), "no zeros to seed rings from");
}

#[test]
fn a_run_lying_along_another_is_drawn_once() {
    let l = landscape("z", domain(-1.0, 1.0, -1.0, 1.0), 21);
    let long: Vec<P> = (0..=10).map(|k| P::new(-0.5 + 0.1 * k as f64, 0.2)).collect();
    let stretch: Vec<P> = (2..=5).map(|k| P::new(-0.5 + 0.1 * k as f64, 0.2 + 0.001)).collect();
    let apart: Vec<P> = (0..=3).map(|k| P::new(-0.5 + 0.1 * k as f64, 0.5)).collect();
    let kept = l.refiner.drawn_once(vec![stretch.clone(), long.clone(), apart.clone()], |r| r);
    assert_eq!(kept, vec![long.clone(), apart], "the copy of a stretch goes");
    let kept = l.refiner.drawn_once(vec![long.clone(), long.clone()], |r| r);
    assert_eq!(kept, vec![long], "of two alike, the first stays");
}

/// Whether p is inside the shifted window of the ray tests, off its edge.
fn window_interior(p: P) -> bool {
    let (lo, hi, slack) = (-1.0125, 0.9875, 1e-9);
    p.x() > lo + slack && p.x() < hi - slack && p.y() > lo + slack && p.y() < hi - slack
}

// MARK: the phase lines the gate once found Swift and the port drawing otherwise

/// Where f is real and negative along the window's edge, the branch cut lies
/// on the edge and arg f there is π or −π by the sign of Im f's zero: z − 1
/// is x − 1 + 0i on the real axis and −(1 − z) is x − 1 − 0i. Quantized,
/// both are Swift's `Float.pi` (`quantize` clamps into the samples' range),
/// which `contour::cut` turns to exactly zero, so the edge's nodes lie on the
/// level whichever the sign, and the grid's tie decides: a node on the level
/// is not over it, and the edge is drawn where the window beside it is over
/// it, arg(−f) = arg(1 − z) > 0, and not where it is under. So the cut along
/// the upper edge of the lower half-plane is drawn, along the lower edge of
/// the upper half-plane it is not, and neither depends on the zero's sign.
/// Γ's and cn's cuts lie so on the lower edge of landscape-gamma and
/// gallery-cn, and the grid draws them along it on one side of each saddle
/// of f there. Turned by π rounded to nearest, the nodes were ±2.4e-7 by the
/// sign, and the edge was drawn or not as it fell (crates/FINDINGS.md,
/// 2026-11-16).
#[test]
fn the_cut_along_the_windows_edge_does_not_hang_on_the_sign_of_a_zero() {
    for (window, edge, drawn) in
        [(domain(-0.5125, 0.4875, 0.0, 1.0), 0.0, false), (domain(-0.5125, 0.4875, -1.0, 0.0), 0.0, true)]
    {
        let mut answers = Vec::new();
        for (expression, sign) in [("z - 1", 1.0f32), ("-(1 - z)", -1.0f32)] {
            let l = landscape(expression, window, 41);
            let row = if window.imag.lo == edge { 0 } else { l.phase.height - 1 };
            let on_edge: Vec<f32> = (0..l.phase.width).map(|i| l.phase.at(i, row)).collect();
            assert!(
                on_edge.iter().all(|&v| v == sign * kurven_core::contour::SWIFT_FLOAT_PI),
                "{expression}: arg f on the edge is {sign}π, quantized to Float.pi: {on_edge:?}"
            );
            let points = l.refiner.singular_points();
            assert!(points.is_empty(), "{expression}: its zero at 1 is outside: {points:?}");
            answers.push(l.refiner.refine(
                ContourField::Phase,
                PI,
                &contour::cut(&l.phase),
                &points,
                f64::INFINITY,
                EVERYWHERE,
            ));
        }
        assert_eq!(answers[0], answers[1], "the same cut whichever the sign of Im f's zero");
        let runs = &answers[0];
        if drawn {
            assert_eq!(runs.len(), 1, "the edge as one line: {runs:?}");
            let run = &runs[0];
            assert!(run.iter().all(|p| p.y() == edge), "along the edge");
            let (lo, hi) = (run[0].x().min(run[run.len() - 1].x()), run[0].x().max(run[run.len() - 1].x()));
            let side = |x: f64, to: f64| (x - to).abs() <= 1e-12;
            assert!(side(lo, window.real.lo) && side(hi, window.real.hi), "from side to side: {lo} {hi}");
        } else {
            assert!(runs.is_empty(), "the window lies under the level beside the edge: {runs:?}");
        }
    }
}

/// The cut of 1/(z³ − 1) is the zero level of arg(−f), and −f = 1 + z³ +
/// O(z⁶) at the saddle z = 0, where f' and f'' vanish: three lines cross
/// there, six branches leaving at multiples of 60°: at 60°, 180° and 300° to
/// the window's edge, at 0°, 120° and 240° to the poles at the cube roots of
/// unity. On the gallery corners' window and grid every branch is drawn,
/// and two are drawn twice, as Swift draws them: the
/// grid's path from the 60° branch to the 0° one cannot be placed across the
/// saddle (its last chord, between the 120° and 60° branches, is 0.026 of a
/// cell off at the depth limit, over the 0.02 tolerance, so no rounding
/// decides it), and the two pieces, followed on f, cross the saddle onto the
/// 300° and 240° branches another path already draws. `drawn_once` drops
/// only a run wholly along another, so both stay: Swift's defect, held here
/// so that a change to it is seen (crates/FINDINGS.md, 2026-11-16).
#[test]
fn the_cut_through_a_saddle_draws_every_branch_and_two_of_them_twice_as_swift() {
    let l = landscape("1/(z^3 - 1)", domain(-1.3, 1.6, -1.2, 1.5), 300);
    assert_eq!((l.phase.width, l.phase.height), (300, 279), "the gallery corners' grid");
    let cell = l.refiner.cell();
    let hook = l.refiner.refine_hook();
    // As the corners' ang_major is asked: under the truncation at 5.
    let runs = (hook.contours)(ContourField::Phase, PI, &contour::cut(&l.phase), 1.1 * 5.0, EVERYWHERE);
    for run in &runs {
        for &p in run {
            let f = l.refiner.value(Complex::new(p.x(), p.y()));
            assert!(
                !f.is_finite() || (f.re < 0.0 && f.im.abs() <= 1e-6 * f.magnitude()),
                "every vertex on the cut: f({p:?}) = {f}"
            );
        }
    }
    // How many runs pass within a twentieth of a cell of the branch at angle
    // k · 60°, half a unit out from the saddle.
    let times: Vec<usize> = (0..6)
        .map(|k| {
            let a = PI / 3.0 * k as f64;
            let p = P::new(0.5 * kurven_math::real::cos(a), 0.5 * kurven_math::real::sin(a));
            runs.iter().filter(|r| distance_to(p, std::slice::from_ref(*r)) <= 0.05 * cell).count()
        })
        .collect();
    assert_eq!(
        times,
        vec![1, 1, 1, 1, 2, 2],
        "branches at 0°, 60°, ..., 300°, each drawn so many times: Swift's, with the 240° and 300° \
         branches drawn twice. Once `drawnOnce` drops a stretch lying along another, in Swift and \
         the port together (crates/FINDINGS.md, 2026-11-16), the answer is [1, 1, 1, 1, 1, 1]"
    );
}
