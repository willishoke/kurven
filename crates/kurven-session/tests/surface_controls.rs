//! A surface's controls (Phase 8, step 5; SurfaceSection.swift): the rows the
//! state lists with the engine's reach, a typed value clamped to it, a
//! slider rested at its step or a detent, the switches and the component
//! picker, each one undo step named for the control.

use kurven_core::json::{self, JsonValue};
use kurven_session::protocol::Session;
use kurven_session::surface_controls::{control, quantize};

fn call(session: &mut Session, request: &str) -> JsonValue {
    json::parse(&session.call(request).json).unwrap()
}

fn ok(session: &mut Session, request: &str) -> JsonValue {
    let r = call(session, request);
    assert_eq!(r.get("ok"), Some(&JsonValue::Bool(true)), "{request}: {r:?}");
    r.get("state").unwrap().clone()
}

fn controls(state: &JsonValue) -> &JsonValue {
    state.get("surfaceControls").unwrap()
}

fn row<'a>(state: &'a JsonValue, field: &str) -> &'a JsonValue {
    let rows = controls(state).get("rows").and_then(JsonValue::as_array).unwrap();
    let linear = controls(state).get("linearCurve").and_then(|l| l.get("rows")).and_then(JsonValue::as_array);
    rows.iter()
        .chain(linear.into_iter().flatten())
        .find(|r| r.get("field").and_then(JsonValue::as_str) == Some(field))
        .unwrap_or_else(|| panic!("no row {field}"))
}

fn value(state: &JsonValue, field: &str) -> f64 {
    row(state, field).get("value").and_then(JsonValue::as_f64).unwrap()
}

fn undo_name(state: &JsonValue) -> Option<&str> {
    state.get("undo").and_then(|u| u.get("name")).and_then(JsonValue::as_str)
}

#[test]
fn a_torus_lists_its_radii_with_the_engines_reach() {
    let mut s = Session::new();
    let state = ok(&mut s, r#"{"kind":"createSurface","preset":"torus"}"#);
    let c = controls(&state);
    assert_eq!(c.get("kind").and_then(JsonValue::as_str), Some("torus"));
    let fields: Vec<&str> = c
        .get("rows")
        .and_then(JsonValue::as_array)
        .unwrap()
        .iter()
        .map(|r| r.get("field").unwrap().as_str().unwrap())
        .collect();
    assert_eq!(fields, ["major", "minor"]);
    let major = row(&state, "major");
    assert_eq!(major.get("label").and_then(JsonValue::as_str), Some("Major radius"));
    assert_eq!(major.get("min").and_then(JsonValue::as_f64), Some(0.5));
    assert_eq!(major.get("max").and_then(JsonValue::as_f64), Some(5.0));
    // A plain torus is its lines: no switch for them, its counts shown.
    assert_eq!(c.get("linesSwitch"), Some(&JsonValue::Bool(false)));
    assert_eq!(c.get("lines"), Some(&json::ints(&[36, 18])));
    // A value typed past the reach is clamped to it, one step named for the control.
    let state = ok(&mut s, r#"{"kind":"surfaceField","field":"major","value":7}"#);
    assert_eq!(value(&state, "major"), 5.0);
    assert_eq!(undo_name(&state), Some("Major radius"));
    // A field the torus has not is refused, touching nothing.
    let r = call(&mut s, r#"{"kind":"surfaceField","field":"modulus","value":0.5}"#);
    assert_eq!(r.get("touched"), Some(&JsonValue::Bool(false)));
    let r = call(&mut s, r#"{"kind":"surfaceLines","on":false}"#);
    assert_eq!(r.get("touched"), Some(&JsonValue::Bool(false)));
    // The counts, clamped to 1...200.
    let state = ok(&mut s, r#"{"kind":"surfaceLineCount","axis":"around","count":500}"#);
    assert_eq!(controls(&state).get("lines"), Some(&json::ints(&[200, 18])));
    assert_eq!(undo_name(&state), Some("Lines Around"));
    let state = ok(&mut s, r#"{"kind":"undo"}"#);
    assert_eq!(controls(&state).get("lines"), Some(&json::ints(&[36, 18])));
}

#[test]
fn the_linear_curve_starts_at_two_fifths_and_its_slider_rests_at_fractions() {
    let mut s = Session::new();
    ok(&mut s, r#"{"kind":"createSurface","preset":"torus"}"#);
    let state = ok(&mut s, r#"{"kind":"linearCurve","on":true}"#);
    assert_eq!(undo_name(&state), Some("Linear curve"));
    assert_eq!(value(&state, "slope"), 0.4);
    assert_eq!(value(&state, "turns"), 24.0);
    assert_eq!(value(&state, "strands"), 1.0);
    let description = controls(&state).get("linearCurve").unwrap().get("description").unwrap();
    assert_eq!(description.as_str(), Some("One strand at 2/5: closes after 5 turns."));
    // A drag of the slope: each move rested at a fraction within reach, or
    // at the step; one undo step for the drag.
    let drag = |s: &mut Session, phase: &str, v: Option<f64>| {
        let value = v.map_or(String::new(), |v| format!(r#","value":{v},"rest":true"#));
        ok(
            s,
            &format!(
                r#"{{"kind":"drag","control":{{"slider":"surfaceField","field":"slope","label":"Slope"}},"phase":"{phase}"{value}}}"#
            ),
        )
    };
    drag(&mut s, "begin", None);
    let state = drag(&mut s, "move", Some(0.6612));
    assert_eq!(value(&state, "slope"), 2.0 / 3.0, "within reach of 2/3");
    let state = drag(&mut s, "move", Some(0.6013));
    assert_eq!(value(&state, "slope"), 0.6, "3/5");
    let state = drag(&mut s, "move", Some(0.5555));
    assert_eq!(value(&state, "slope"), (0.5555f64 / 0.001).round() * 0.001, "no fraction within reach: the step");
    let state = drag(&mut s, "end", None);
    assert_eq!(undo_name(&state), Some("Slope"));
    let state = ok(&mut s, r#"{"kind":"linearCurve","on":false}"#);
    assert!(controls(&state).get("linearCurve") == Some(&JsonValue::Null));
}

#[test]
fn quantize_takes_the_first_nearest_detent_within_reach() {
    assert_eq!(quantize(0.401, 0.001, &[0.4, 0.5], 0.015), 0.4);
    assert_eq!(quantize(0.45, 0.001, &[0.4, 0.5], 0.015), 0.45);
    // Two equally near: the first, as Swift's `min(by:)` keeps it.
    assert_eq!(quantize(0.45, 0.001, &[0.44, 0.46], 0.015), 0.44);
    assert_eq!(quantize(117.0, 1.0, &[115.0, 344.0], 4.0), 115.0);
    assert!(!control(kurven_core::surface_request::Field::Harmonics).slides);
}

#[test]
fn sn_switches_its_conformal_drawing_and_its_coordinate_curves() {
    let mut s = Session::new();
    let state = ok(&mut s, r#"{"kind":"createSurface","preset":"sn"}"#);
    let c = controls(&state);
    assert_eq!(c.get("conformal"), Some(&JsonValue::Bool(false)));
    assert_eq!(c.get("linesSwitch"), Some(&JsonValue::Bool(true)));
    assert_eq!(c.get("lines"), Some(&JsonValue::Null));
    assert!(c.get("notes").and_then(JsonValue::as_str).unwrap().starts_with("Jacobi's sn"));
    let state = ok(&mut s, r#"{"kind":"surfaceField","field":"conformal","value":1}"#);
    assert_eq!(controls(&state).get("conformal"), Some(&JsonValue::Bool(true)));
    assert_eq!(undo_name(&state), Some("Conformal"));
    // Conformal, the minor radius is pinned and not listed.
    let rows = controls(&state).get("rows").and_then(JsonValue::as_array).unwrap();
    assert!(rows.iter().all(|r| r.get("field").and_then(JsonValue::as_str) != Some("minor")));
    let state = ok(&mut s, r#"{"kind":"surfaceLines","on":true}"#);
    assert_eq!(controls(&state).get("lines"), Some(&json::ints(&[36, 18])));
    assert_eq!(undo_name(&state), Some("Coordinate Curves"));
}

#[test]
fn the_forced_torus_swaps_its_components_and_rests_its_slope_at_its_own() {
    let mut s = Session::new();
    let state = ok(&mut s, r#"{"kind":"createSurface","preset":"forced"}"#);
    let c = controls(&state);
    assert_eq!(c.get("components"), Some(&json::object([("radial", JsonValue::Int(0)), ("axial", JsonValue::Int(1))])));
    assert_eq!(row(&state, "harmonics").get("slides"), Some(&JsonValue::Bool(false)));
    // The turns slider rests at the convergents' denominators.
    let detents = row(&state, "trajectoryTurns").get("detents").and_then(JsonValue::as_array).unwrap();
    assert!(detents.iter().any(|d| d.as_f64() == Some(344.0)), "{detents:?}");
    // Choosing the other's component swaps them.
    let state = ok(&mut s, r#"{"kind":"component","field":"radial","value":1}"#);
    assert_eq!(
        controls(&state).get("components"),
        Some(&json::object([("radial", JsonValue::Int(1)), ("axial", JsonValue::Int(0))]))
    );
    assert_eq!(undo_name(&state), Some("Radius from"));
    // The linear curve starts at the trajectory's own Ω/ω.
    let state = ok(&mut s, r#"{"kind":"linearCurve","on":true}"#);
    let description = controls(&state).get("linearCurve").unwrap().get("description").unwrap().as_str().unwrap();
    assert!(description.starts_with("One strand at the trajectory's own Ω/ω, "), "{description}");
}
