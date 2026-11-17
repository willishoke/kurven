//! The protocol's own rules: what a refusal says it touched, indices
//! refused rather than truncated, the camera set to the bit.

use kurven_core::json::{self, JsonValue};
use kurven_session::protocol::Session;

fn reply(session: &mut Session, request: &str) -> JsonValue {
    json::parse(&session.call(request).json).unwrap()
}

fn input(name: &str) -> String {
    format!("{}/../fixtures/swift/session/inputs/{name}.kurven", env!("CARGO_MANIFEST_DIR"))
}

fn open(session: &mut Session, name: &str) {
    let dir = input(name);
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let file = path.file_name().unwrap().to_string_lossy().into_owned();
        // Staged under a path written as `./file`, which reads as `file`.
        session.stage(&dir, &format!("./{file}"), std::fs::read(&path).unwrap());
    }
    let r = reply(session, &format!(r#"{{"kind":"open","name":{}}}"#, json::quote(&dir, false)));
    assert_eq!(r.get("ok"), Some(&JsonValue::Bool(true)), "{r:?}");
}

#[test]
fn a_refusal_says_whether_it_touched_the_document() {
    let mut s = Session::new();
    // Nothing staged under the name: refused before the document.
    let r = reply(&mut s, r#"{"kind":"open","name":"nowhere.kurven"}"#);
    assert_eq!(r.get("ok"), Some(&JsonValue::Bool(false)));
    assert_eq!(r.get("touched"), Some(&JsonValue::Bool(false)));
    // A bundle that cannot be read: the document is a failure now.
    s.stage("broken.kurven", "manifest.json", b"nope".to_vec());
    let r = reply(&mut s, r#"{"kind":"open","name":"broken.kurven"}"#);
    assert_eq!(r.get("touched"), Some(&JsonValue::Bool(true)));
    assert_eq!(r.get("state").unwrap().get("kind").unwrap().as_str(), Some("failed"));
}

#[test]
fn an_index_that_is_not_one_is_refused() {
    let mut s = Session::new();
    open(&mut s, "gamma");
    for bad in ["-1", "1.5", "1e300", "\"2\""] {
        let r = reply(&mut s, &format!(r#"{{"kind":"hide","layer":{bad}}}"#));
        assert_eq!(r.get("ok"), Some(&JsonValue::Bool(false)), "hide {bad}");
        let r = reply(&mut s, &format!(r#"{{"kind":"preset","index":{bad}}}"#));
        assert_eq!(r.get("ok"), Some(&JsonValue::Bool(false)), "preset {bad}");
    }
    let r = reply(&mut s, r#"{"kind":"viewport","width":0,"height":800}"#);
    assert_eq!(r.get("ok"), Some(&JsonValue::Bool(false)));
    let r = reply(&mut s, r#"{"kind":"hide","layer":2}"#);
    assert_eq!(r.get("ok"), Some(&JsonValue::Bool(true)));
    let r = reply(&mut s, r#"{"kind":"hide","layer":99}"#);
    assert_eq!(r.get("ok"), Some(&JsonValue::Bool(false)));
}

#[test]
fn a_run_of_gestures_is_one_camera_request_to_the_bit() {
    let gestures = [
        r#"{"kind":"gesture","gesture":{"kind":"orbit","delta":[31.7,-12.25]}}"#,
        r#"{"kind":"gesture","gesture":{"kind":"zoom","factor":1.37,"at":[411.5,280]}}"#,
        r#"{"kind":"gesture","gesture":{"kind":"pan","delta":[-17,9.5]}}"#,
        r#"{"kind":"gesture","gesture":{"kind":"roll","radians":0.1}}"#,
        r#"{"kind":"gesture","gesture":{"kind":"retarget","point":[0.5,0.25,1.5]}}"#,
    ];
    let mut a = Session::new();
    open(&mut a, "gamma");
    let mut last = JsonValue::Null;
    for g in gestures {
        last = reply(&mut a, g);
    }
    let exact = last.get("state").unwrap().get("camera").unwrap().get("exact").unwrap().clone();
    let mut b = Session::new();
    open(&mut b, "gamma");
    let JsonValue::Object(mut request) = exact else { panic!() };
    request.insert("kind".to_string(), json::string("camera"));
    let r = reply(&mut b, &JsonValue::Object(request).canonical());
    assert_eq!(r.get("state"), last.get("state"));
    let (fa, fb) = (a.call(r#"{"kind":"frame","draft":false}"#), b.call(r#"{"kind":"frame","draft":false}"#));
    assert_eq!(fa, fb, "the same frame, array for array");
}

#[test]
fn a_control_naming_a_layer_without_it_is_refused_touching_nothing() {
    let mut s = Session::new();
    open(&mut s, "gamma");
    for request in [
        r#"{"kind":"levels","layer":"nope","count":5}"#,
        r#"{"kind":"resetLevels","layer":"nope"}"#,
        r#"{"kind":"strokes","layer":"mag_major","count":5}"#,
        r#"{"kind":"light","layer":"wall_hatch","direction":[1,0,1]}"#,
        r#"{"kind":"drag","control":{"slider":"levels","name":"wall_hatch"},"phase":"begin"}"#,
    ] {
        let r = reply(&mut s, request);
        assert_eq!(r.get("ok"), Some(&JsonValue::Bool(false)), "{request}");
        assert_eq!(r.get("touched"), Some(&JsonValue::Bool(false)), "{request}");
    }
    let r = reply(&mut s, r#"{"kind":"levels","layer":"mag_major","count":5}"#);
    assert_eq!(r.get("ok"), Some(&JsonValue::Bool(true)));
}

#[test]
fn a_save_that_cannot_be_made_says_why() {
    let mut s = Session::new();
    let r = reply(&mut s, r#"{"kind":"save","name":"x.kurven"}"#);
    assert_eq!(r.get("error").and_then(JsonValue::as_str), Some("nothing is open to save"));
}
