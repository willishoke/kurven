//! `Sources/kurven-test/IntegralCurves.swift`'s suites that need no
//! landscape built from f: the manifest, the seeds placed along a polyline,
//! and the constant field's closed form. The suites on a landscape need the
//! refiner, which kurven-core cannot depend on: they are
//! `kurven-landscape/tests/integral_curves.rs`.

use kurven_core::fixtures;
use kurven_core::hatching::FaceSelector;
use kurven_core::integral_curves::VectorField;
use kurven_core::json;
use kurven_core::manifest::{
    ContourField, DirectionField, HeightPolicy, LayerRole, LayerSource, LayerSpec, Manifest, Provenance, Seeds,
    Spacing, SurfaceManifest,
};
use kurven_core::parametric::ParametricSurface;
use kurven_core::simd::{Vec2, Vec3};
use kurven_core::spaces::{P2, ParamSpace};
use kurven_core::surface_request::{Shape, SurfaceRequest};

type P = P2<ParamSpace>;

fn plane(p: P) -> Vec3 {
    Vec3::new(p.x(), p.y(), 0.0)
}

/// The seeds suite's half on polylines given outright ("integral curves:
/// seeds once each, and no closer than the grid resolves"); its half on a
/// landscape is in kurven-landscape.
#[test]
fn a_value_read_at_a_vertex_seeds_it_once_and_arc_length_seeds_round_a_ratio() {
    // A value read exactly at a vertex: once, whether the field crosses it
    // there (−, 0, +) or touches it (−, 0, −), and a closed path's first
    // vertex once.
    let line = vec![P::new(0.0, 0.0), P::new(1.0, 0.0), P::new(2.0, 0.0)];
    let seeds_at =
        Seeds { field: ContourField::Magnitude, level: 1.0, spacing: Spacing::Values(ContourField::U, vec![0.0]) };
    let touch_value = |p: P| -(p.x() - 1.0) * (p.x() - 1.0);
    let touch = seeds_at.points(std::slice::from_ref(&line), Some(&touch_value), None, Vec2::ZERO, &plane);
    let cross_value = |p: P| p.x() - 1.0;
    let cross = seeds_at.points(std::slice::from_ref(&line), Some(&cross_value), None, Vec2::ZERO, &plane);
    let square = vec![P::new(0.0, 0.0), P::new(1.0, 0.0), P::new(1.0, 1.0), P::new(0.0, 1.0), P::new(0.0, 0.0)];
    let corner_value = |p: P| p.x() + p.y();
    let corner = seeds_at.points(&[square], Some(&corner_value), None, Vec2::ZERO, &plane);
    assert!(
        touch == [P::new(1.0, 0.0)] && cross == [P::new(1.0, 0.0)] && corner == [P::new(0.0, 0.0)],
        "a value read at a vertex seeds the vertex once, touched or crossed, and a closed path's first vertex once: \
         touched {touch:?}, crossed {cross:?}, closed {corner:?}"
    );
    // By arc length a piece of the level set takes round(L / spacing)
    // seeds: none under half the spacing; and a ratio no grid resolves
    // seeds nothing rather than trapping.
    let unit = vec![P::new(0.0, 0.0), P::new(1.0, 0.0)];
    let spaced = |s: f64| {
        Seeds { field: ContourField::Magnitude, level: 1.0, spacing: Spacing::ArcLength(s) }
            .points(std::slice::from_ref(&unit), None, None, Vec2::ZERO, &plane)
            .len()
    };
    let counts: Vec<usize> = [3.0, 1.5, 0.25, 1e-300, 5e-324].iter().map(|&s| spaced(s)).collect();
    assert_eq!(
        counts,
        [0, 1, 4, 0, 0],
        "a unit stretch spaced 3 apart takes no seed, 1.5 one, a quarter four, and 1e-300 none, without trapping"
    );
}

/// The first half of "integral curves: a linear curve is the constant
/// field's, in closed form, to the bit". The other half draws the torus's
/// linear curves: Phase 4's.
#[test]
fn the_constant_fields_flow_is_p_plus_t_w_to_the_bit() {
    let w = Vec2::new(0.1, 0.37);
    let field = VectorField::constant(w);
    let p = P::new(0.25, -1.5);
    let flow = field.flow.as_ref().unwrap();
    assert!(
        (0..=50).all(|m| flow(p, m as f64) == P::new(p.x() + m as f64 * w.x, p.y() + m as f64 * w.y))
            && (field.value)(P::new(3.0, 4.0)) == Some(w),
        "the constant field's flow is p + t w, each coordinate to the bit"
    );
}

/// Its second half. Swift builds the torus as `SurfaceDrawing.build` of a
/// torus request at grid 64, which is `ParametricSurface.torus(major: 2,
/// minor: 1, samples: (64, 32))`; kurven-core cannot reach the drawing, so
/// the torus is built here as the drawing builds it.
#[test]
fn the_torus_linear_curve_is_the_constant_fields_integral_curve() {
    let torus = ParametricSurface::torus(2.0, 1.0, (64, 32)).expect("the torus");
    let (slope, turns, count) = (0.4, 5, 2);
    let curves = torus.linear_curves(slope, turns, count, None);
    let steps = (torus.u.cells() as f64).max((slope * torus.v.cells() as f64).ceil());
    let du = torus.u.range.length() / steps;
    let dv = slope * torus.v.range.length() / steps;
    // The first strand's first path, before any seam: its vertex m is
    // the start moved by m steps.
    let strand = curves.coords_of_path(0).unwrap();
    let exact = strand
        .iter()
        .enumerate()
        .all(|(m, &c)| c == P::new(torus.u.range.lo + m as f64 * du, torus.v.range.lo + m as f64 * dv));
    assert!(
        exact && strand.len() > 10 && curves.count() > 0,
        "the torus's linear curve is the constant field's integral curve, its vertices the closed form's to the bit: {} vertices",
        strand.len()
    );
}

/// "integral curves: the manifest".
#[test]
fn the_manifest() {
    let source =
        |text: &str| LayerSource::from_json(&json::parse(text).map_err(|e| e.to_string())?).map_err(|e| e.to_string());
    let refused = |text: &str, words: &str| match source(text) {
        Ok(_) => false,
        Err(e) => e.contains(words),
    };
    let head = r#"{"kind": "integralCurves", "field": {"kind": "slope", "of": "magnitude"}, "#;
    let graph = r#""faces": {"kind": "graph", "tiled": true}}"#;
    let by_values = format!(
        r#"{head}"seeds": {{"on": {{"field": "magnitude", "level": 0.6}}, "at": {{"kind": "values", "field": "phase", "values": [-1, 0.5]}}}}, {graph}"#
    );
    let by_length = format!(
        r#"{head}"seeds": {{"on": {{"field": "u", "level": 0.25}}, "at": {{"kind": "arcLength", "spacing": 0.3}}}}, {graph}"#
    );
    let a = source(&by_values).unwrap();
    let b = source(&by_length).unwrap();
    let a_again = source(&a.json().canonical()).unwrap();
    let b_again = source(&b.json().canonical()).unwrap();
    assert!(
        a == LayerSource::IntegralCurves {
            field: DirectionField::Slope(ContourField::Magnitude),
            seeds: Seeds {
                field: ContourField::Magnitude,
                level: 0.6,
                spacing: Spacing::Values(ContourField::Phase, vec![-1.0, 0.5])
            },
            faces: FaceSelector::Graph { tiled: true },
        } && b
            == LayerSource::IntegralCurves {
                field: DirectionField::Slope(ContourField::Magnitude),
                seeds: Seeds { field: ContourField::U, level: 0.25, spacing: Spacing::ArcLength(0.3) },
                faces: FaceSelector::Graph { tiled: true },
            }
            && a_again == a
            && b_again == b
            && a.kind() == "slope lines",
        "integral curves seeded at values or by arc length decode as written and round-trip"
    );
    let r = |text: &str, from: &str, to: &str| text.replace(from, to);
    let refusals: Vec<(String, &str)> = vec![
        (r(&by_values, r#""level": 0.6"#, r#""level": 0"#), "f's zeros or nothing"),
        (r(&by_values, r#""level": 0.6"#, r#""level": -1"#), "f's zeros or nothing"),
        (r(&by_values, "[-1, 0.5]", "[]"), "no values"),
        (r(&by_values, r#""field": "phase""#, r#""field": "magnitude""#), "they are one function"),
        (
            r(&by_values, r#""field": "phase", "values": [-1, 0.5]"#, r#""field": "height", "values": [0.5]"#),
            "they are one function",
        ),
        (
            r(
                &r(&by_values, r#""on": {"field": "magnitude""#, r#""on": {"field": "height""#),
                r#""field": "phase", "values": [-1, 0.5]"#,
                r#""field": "magnitude", "values": [0.5]"#,
            ),
            "they are one function",
        ),
        (
            r(
                &r(&by_values, r#""on": {"field": "magnitude""#, r#""on": {"field": "phase""#),
                r#""field": "phase", "values": [-1, 0.5]"#,
                r#""field": "height", "values": [0, 1]"#,
            ),
            "it is positive",
        ),
        (r(&by_values, r#""field": "phase""#, r#""field": "isophote""#), "isophote"),
        (r(&by_values, r#""of": "magnitude""#, r#""of": "isophote""#), "no light"),
        (r(&by_values, r#""of": "magnitude""#, r#""of": "curl""#), "not one of"),
        (r(&by_values, r#""kind": "slope""#, r#""kind": "principal""#), "not one of"),
        (r(&by_length, r#""spacing": 0.3"#, r#""spacing": 0"#), "a spacing is positive"),
        (r(&by_length, r#""spacing": 0.3"#, r#""spacing": -2"#), "a spacing is positive"),
        (r(&by_length, r#""kind": "arcLength""#, r#""kind": "evenly""#), "not one of"),
        (
            r(&by_values, r#"{"kind": "graph", "tiled": true}"#, r#"{"kind": "plateaus"}"#),
            "they lie on the graph so far",
        ),
        (
            r(&by_values, r#"{"kind": "graph", "tiled": true}"#, r#"{"kind": "walls", "edges": [0]}"#),
            "they lie on the graph so far",
        ),
        (
            r(&by_values, r#"{"kind": "graph", "tiled": true}"#, r#"{"kind": "surface"}"#),
            "this manifest holds a landscape",
        ),
    ];
    let missed: Vec<&str> = refusals.iter().filter(|(text, words)| !refused(text, words)).map(|(_, w)| *w).collect();
    assert!(
        missed.is_empty(),
        "seeds on |f| or the height at 0 or below, at no values, at values of the function they lie on, by any name, \
         or a height not positive, by an isophote, the slope of an isophote, a field or kind unknown, a spacing not \
         positive, and integral curves off the graph are refused, saying why; not refused: {missed:?}"
    );
    let nan = r(&by_values, r#""level": 0.6"#, r#""level": 1e999"#);
    assert!(source(&nan).is_err(), "a level that is not a number is refused");

    // A parametric surface's manifest takes none on the graph, which it has not.
    let empty = SurfaceManifest::new(
        SurfaceRequest::new(Shape::Torus { major: 2.0, minor: 1.0 }),
        Vec::new(),
        Vec::new(),
        Provenance {
            function: String::new(),
            example: String::new(),
            params: Default::default(),
            cpu_count: 1,
            git_sha: String::new(),
        },
    );
    let layer = format!(
        r#"{{"name": "x", "role": "scaffold", "width": 0.3, "heightPolicy": "surface", "source": {by_values}}}"#
    );
    let text = empty.canonical_json().replace(r#""layers":[]"#, &format!(r#""layers":[{layer}]"#));
    let on_surface = match SurfaceManifest::from_json(&json::parse(&text).unwrap()) {
        Ok(_) => false,
        Err(e) => e.to_string().contains("integral curves on the graph of a parametric surface"),
    };
    assert!(on_surface, "a parametric surface's manifest refuses integral curves on the graph, saying it is one face");

    // A landscape's manifest: seeds closer than a cell of its grid.
    let cubic = std::fs::read_to_string(fixtures::shared("expr/landscape_cubic.kurven/manifest.json")).unwrap();
    let m = Manifest::from_json(&json::parse(&cubic).unwrap()).unwrap();
    let mut with_layer = m.clone();
    with_layer.layers.push(LayerSpec::new(
        "slope",
        LayerRole::Scaffold,
        LayerSource::IntegralCurves {
            field: DirectionField::Slope(ContourField::Magnitude),
            seeds: Seeds { field: ContourField::Magnitude, level: 1.0, spacing: Spacing::ArcLength(1e-4) },
            faces: FaceSelector::Graph { tiled: false },
        },
        0.3,
        HeightPolicy::Surface,
    ));
    let too_close = match Manifest::from_json(&json::parse(&with_layer.canonical_json()).unwrap()) {
        Ok(_) => false,
        Err(e) => e.to_string().contains("closer than a cell of the grid"),
    };
    let last = with_layer.layers.len() - 1;
    with_layer.layers[last].source = LayerSource::IntegralCurves {
        field: DirectionField::Slope(ContourField::Magnitude),
        seeds: Seeds { field: ContourField::Magnitude, level: 1.0, spacing: Spacing::ArcLength(0.5) },
        faces: FaceSelector::Graph { tiled: false },
    };
    let round_trip = Manifest::from_json(&json::parse(&with_layer.canonical_json()).unwrap()).unwrap();
    assert!(
        too_close && round_trip == with_layer,
        "a landscape's manifest refuses seeds closer than a cell of its grid, and keeps them a cell apart or more"
    );
}
