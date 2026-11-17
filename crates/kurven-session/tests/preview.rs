//! The preview's engine half (`kurven_session::preview`): what a frame
//! draws, as the page's WebGPU renderer takes it. The pictures are the
//! browser lane's (scripts/check-browser.sh); this holds the values and the
//! keys, which decide what a camera change uploads.

use std::collections::BTreeSet;

use kurven_core::json::{self, JsonValue};
use kurven_session::preview::{Pipeline, color};
use kurven_session::protocol::Session;

fn reply(session: &mut Session, request: &str) -> (JsonValue, Vec<String>) {
    let r = session.call(request);
    let names = r.arrays.iter().map(|(n, _)| n.clone()).collect();
    (json::parse(&r.json).unwrap(), names)
}

fn open(session: &mut Session, name: &str) {
    let dir = format!("{}/../fixtures/swift/session/inputs/{name}.kurven", env!("CARGO_MANIFEST_DIR"));
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let file = path.file_name().unwrap().to_string_lossy().into_owned();
        session.stage(&dir, &file, std::fs::read(&path).unwrap());
    }
    let (r, _) = reply(session, &format!(r#"{{"kind":"open","name":{}}}"#, json::quote(&dir, false)));
    assert_eq!(r.get("ok"), Some(&JsonValue::Bool(true)), "{r:?}");
}

fn keys(frame: &JsonValue) -> Vec<String> {
    frame
        .get("layers")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l.get("key").unwrap().as_str().unwrap().to_string())
        .collect()
}

/// The `have` a page holding everything `frame` sent would send.
fn holding(frame: &JsonValue) -> String {
    let content = frame.get("content").unwrap().as_str().unwrap();
    let layers: Vec<JsonValue> = keys(frame).iter().map(|k| json::string(k)).collect();
    json::object([("content", json::string(content)), ("layers", JsonValue::Array(layers))]).canonical()
}

fn layer_field<'a>(frame: &'a JsonValue, name: &str, field: &str) -> &'a JsonValue {
    let layers = frame.get("layers").unwrap().as_array().unwrap();
    let layer = layers.iter().find(|l| l.get("name").unwrap().as_str() == Some(name)).unwrap();
    layer.get(field).unwrap()
}

#[test]
fn a_camera_change_sends_only_what_the_camera_moved() {
    let mut s = Session::new();
    open(&mut s, "gamma");
    let (first, names) = reply(&mut s, r#"{"kind":"preview","draft":true}"#);
    assert_eq!(first.get("ok"), Some(&JsonValue::Bool(true)), "{first:?}");
    assert!(first.get("resources").unwrap().get("kind").unwrap().as_str() == Some("heightfield"));
    assert!(names.contains(&"resource.heights".to_string()), "{names:?}");
    // Asked again holding all of it: nothing but the JSON comes back.
    let have = holding(&first);
    let (again, names) = reply(&mut s, &format!(r#"{{"kind":"preview","draft":true,"have":{have}}}"#));
    assert!(names.is_empty(), "held, and sent again: {names:?}");
    assert_eq!(again.get("resources"), Some(&JsonValue::Null));
    assert_eq!(keys(&again), keys(&first));
    // Orbit: the folds are the camera's, traced again for it; the contours
    // and the walls of a draft are not.
    reply(&mut s, r#"{"kind":"gesture","gesture":{"kind":"orbit","delta":[20,5]}}"#);
    let (moved, names) = reply(&mut s, &format!(r#"{{"kind":"preview","draft":true,"have":{have}}}"#));
    let changed: BTreeSet<String> =
        keys(&moved).into_iter().zip(keys(&first)).filter(|(a, b)| a != b).map(|(a, _)| a).collect();
    let folds = layer_field(&moved, "folds", "key").as_str().unwrap().to_string();
    assert_eq!(changed, BTreeSet::from([folds]), "only the folds move with the camera");
    assert!(names.iter().all(|n| n.starts_with("layer.")) && !names.is_empty(), "{names:?}");
    // The uniforms are the camera's.
    assert_ne!(moved.get("uniforms").unwrap().get("view"), first.get("uniforms").unwrap().get("view"));
}

#[test]
fn at_rest_the_tested_strokes_come_as_they_are_drawn() {
    let mut s = Session::new();
    open(&mut s, "gamma");
    let (draft, _) = reply(&mut s, r#"{"kind":"preview","draft":true}"#);
    let (rest, _) = reply(&mut s, r#"{"kind":"preview","draft":false}"#);
    let tolerance = |f: &JsonValue, name: &str| layer_field(f, name, "depthTolerance").clone();
    // A draft leaves the folds to the depth test; at rest they are tested by
    // the solid, and drawn as they come.
    assert!(tolerance(&draft, "folds").as_f64().is_some());
    assert_eq!(tolerance(&rest, "folds"), JsonValue::Null);
    // A contour is the depth test's either way.
    let contour = rest.get("layers").unwrap().as_array().unwrap()[0].get("name").unwrap().as_str().unwrap().to_string();
    assert!(tolerance(&rest, &contour).as_f64().is_some(), "{contour}");
}

#[test]
fn a_solid_at_rest_is_cut_in_closed_form_and_a_surface_by_sphere_tracing() {
    let mut s = Session::new();
    open(&mut s, "frame");
    let (draft, _) = reply(&mut s, r#"{"kind":"preview","draft":true}"#);
    let (rest, _) = reply(&mut s, r#"{"kind":"preview","draft":false}"#);
    assert_eq!(rest.get("resources").unwrap().get("kind").unwrap().as_str(), Some("solid"));
    assert!(layer_field(&draft, "edges", "depthTolerance").as_f64().is_some());
    assert_eq!(layer_field(&rest, "edges", "depthTolerance"), &JsonValue::Null);

    let mut s = Session::new();
    open(&mut s, "sn");
    let (draft, _) = reply(&mut s, r#"{"kind":"preview","draft":true}"#);
    let (rest, names) = reply(&mut s, r#"{"kind":"preview","draft":false}"#);
    assert_eq!(rest.get("resources").unwrap().get("kind").unwrap().as_str(), Some("parametric"));
    assert!(names.contains(&"resource.positions".to_string()));
    // A draft draws the folds from the grid, by the surface pipeline; at rest
    // they are sphere traced, and the stroke pipeline draws them as they come.
    assert_eq!(layer_field(&draft, "folds", "pipeline").as_str(), Some("surface"));
    assert_eq!(layer_field(&draft, "folds", "onFolds"), &JsonValue::Bool(true));
    assert_eq!(layer_field(&rest, "folds", "pipeline").as_str(), Some("stroke"));
    assert_eq!(layer_field(&rest, "folds", "depthTolerance"), &JsonValue::Null);
    assert!(rest.get("uniforms").unwrap().get("surfaceDepthRange").unwrap().as_array().is_some());
}

#[test]
fn colors_are_swifts() {
    assert_eq!(color("#000000"), [0.0, 0.0, 0.0, 1.0]);
    assert_eq!(color("#ff8000"), [1.0, 128.0 / 255.0, 0.0, 1.0]);
    assert_eq!(color("#f80"), [1.0, 136.0 / 255.0, 0.0, 1.0]);
    assert_eq!(color("red"), [0.0, 0.0, 0.0, 1.0]);
    assert_eq!(Pipeline::Surface { on_folds: true }, Pipeline::Surface { on_folds: true });
}

/// An engine restarted counts its `ContentID`s from one again, so the same
/// bundle opened by two engines can be the same number; each engine's name
/// (`seed`, a Worker's at load) starts its keys, and a page holding the
/// first's keys is sent everything by the second.
#[test]
fn keys_held_from_another_engine_name_nothing() {
    let mut a = Session::new();
    let mut b = Session::new();
    reply(&mut a, r#"{"kind":"seed","nonce":"a1"}"#);
    reply(&mut b, r#"{"kind":"seed","nonce":"b2"}"#);
    open(&mut a, "gamma");
    open(&mut b, "gamma");
    let (first, _) = reply(&mut a, r#"{"kind":"preview","draft":true}"#);
    let have = holding(&first);
    let (second, names) = reply(&mut b, &format!(r#"{{"kind":"preview","draft":true,"have":{have}}}"#));
    assert_ne!(first.get("content"), second.get("content"));
    assert!(keys(&first).iter().all(|k| !keys(&second).contains(k)), "no key of one engine is the other's");
    assert!(names.contains(&"resource.heights".to_string()), "everything sent again: {names:?}");
    let (r, _) = reply(&mut a, r#"{"kind":"seed","nonce":"no-dashes"}"#);
    assert_eq!(r.get("ok"), Some(&JsonValue::Bool(false)));
}

/// A zoom moves the framing, not the camera: a sphere-traced surface's
/// layout at rest and a draft's are kept (Swift's resolutions 0 and
/// infinite serve every zoom); a landscape's tested layout at rest is kept
/// zooming out and tested again zooming in (`VisibleLines.serves`).
#[test]
fn a_zoom_keeps_what_serves_it() {
    let zoom =
        |factor: f64| format!(r#"{{"kind":"gesture","gesture":{{"kind":"zoom","factor":{factor},"at":[600,400]}}}}"#);
    let mut s = Session::new();
    open(&mut s, "sn");
    for draft in [true, false] {
        let request = format!(r#"{{"kind":"preview","draft":{draft}}}"#);
        let (before, _) = reply(&mut s, &request);
        reply(&mut s, &zoom(1.5));
        let (after, _) = reply(&mut s, &request);
        assert_eq!(keys(&before), keys(&after), "sn, draft {draft}: a zoom keeps every layout");
    }

    let mut s = Session::new();
    open(&mut s, "gamma");
    let rest = r#"{"kind":"preview","draft":false}"#;
    let (at, _) = reply(&mut s, rest);
    let folds = |f: &JsonValue| layer_field(f, "folds", "key").as_str().unwrap().to_string();
    // Zoomed out, the held layout, placed more finely, serves.
    reply(&mut s, &zoom(0.5));
    let (out, _) = reply(&mut s, &format!(r#"{{"kind":"preview","draft":false,"have":{}}}"#, holding(&at)));
    assert_eq!(folds(&out), folds(&at), "zooming out keeps the tested folds");
    // Zoomed in past it, they are tested again.
    reply(&mut s, &zoom(4.0));
    let (inside, _) = reply(&mut s, &format!(r#"{{"kind":"preview","draft":false,"have":{}}}"#, holding(&at)));
    assert_ne!(folds(&inside), folds(&at), "zooming in tests the folds again");
}
