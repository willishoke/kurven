//! The protocol's own rules: what a refusal says it touched, indices
//! refused rather than truncated, the camera set to the bit.

use kurven_core::json::{self, JsonObject, JsonValue};
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

#[test]
fn an_expression_is_refused_where_it_went_wrong_touching_nothing() {
    let mut s = Session::new();
    reply(&mut s, r#"{"kind":"viewport","width":200,"height":150}"#);
    reply(&mut s, r#"{"kind":"create","preset":"gamma"}"#);
    let r = reply(&mut s, r#"{"kind":"expression","expression":"gamma(z, 2)"}"#);
    assert_eq!(r.get("ok"), Some(&JsonValue::Bool(false)));
    assert_eq!(r.get("touched"), Some(&JsonValue::Bool(false)));
    let at = r.get("at").unwrap();
    assert_eq!(at.get("position").and_then(JsonValue::as_i64), Some(0));
    assert_eq!(at.get("length").and_then(JsonValue::as_i64), Some(11));
    assert_eq!(at.get("message").and_then(JsonValue::as_str), Some("gamma takes 1 argument, not 2"));
    // Counted in characters, as the language counts them: Γ is one.
    let r = reply(&mut s, r#"{"kind":"expression","expression":"Γ(z) + )"}"#);
    assert_eq!(r.get("at").unwrap().get("position").and_then(JsonValue::as_i64), Some(7));
    // Empty input, at its start, as language.json freezes it.
    let r = reply(&mut s, r#"{"kind":"expression","expression":""}"#);
    assert_eq!(r.get("at").unwrap().get("position").and_then(JsonValue::as_i64), Some(0));
}

#[test]
fn the_catalog_lists_its_presets_with_their_windows() {
    let mut s = Session::new();
    let r = reply(&mut s, r#"{"kind":"catalog"}"#);
    let presets = r.get("presets").and_then(JsonValue::as_array).unwrap();
    let names: Vec<&str> = presets.iter().filter_map(|p| p.get("name").and_then(JsonValue::as_str)).collect();
    assert!(names.contains(&"gamma") && names.contains(&"cn"), "{names:?}");
    assert!(presets.iter().all(|p| p.get("window").is_some() && p.get("label").is_some()));
}

/// The document's edited state, its camera and what the bake writes.
fn snapshot(s: &mut Session) -> (JsonValue, JsonValue, Vec<u8>) {
    let edited = reply(s, r#"{"kind":"edited"}"#).get("edited").cloned().unwrap();
    let camera =
        reply(s, r#"{"kind":"state"}"#).get("state").unwrap().get("camera").unwrap().get("exact").cloned().unwrap();
    let bake = s.call(r#"{"kind":"bake","svg":true}"#);
    let svg = bake.arrays.into_iter().find(|(n, _)| n == "svg").unwrap().1.into_bytes();
    (edited, camera, svg)
}

#[test]
fn a_restored_landscape_is_the_one_edited_and_bakes_the_same_bytes() {
    let mut a = Session::new();
    for r in [
        r#"{"kind":"viewport","width":400,"height":300}"#,
        r#"{"kind":"create","preset":"gamma"}"#,
        r#"{"kind":"set","key":"res","value":140}"#,
        r#"{"kind":"expression","expression":"gamma(z) + 1/z"}"#,
        r#"{"kind":"set","key":"rMax","value":3.25}"#,
        r#"{"kind":"truncation","truncation":{"kind":"uniform","height":3.5}}"#,
        r#"{"kind":"levels","layer":"mag_major","count":5}"#,
        r#"{"kind":"gesture","gesture":{"kind":"orbit","delta":[30,-12]}}"#,
        r#"{"kind":"gesture","gesture":{"kind":"zoom","factor":1.3,"at":[120,80]}}"#,
        r#"{"kind":"depthTolerance","value":0.03}"#,
    ] {
        let r = reply(&mut a, r);
        assert_eq!(r.get("ok"), Some(&JsonValue::Bool(true)), "{r:?}");
    }
    let (edited, camera, svg) = snapshot(&mut a);
    assert_eq!(edited.get("levelCounts").unwrap().get("mag_major").and_then(JsonValue::as_i64), Some(5));

    // Another viewport: the camera, set outright, carries the framing.
    let mut b = Session::new();
    reply(&mut b, r#"{"kind":"viewport","width":640,"height":640}"#);
    let mut restore = edited.as_object().unwrap().clone();
    restore.insert("kind".to_string(), json::string("restore"));
    let r = reply(&mut b, &JsonValue::Object(restore).canonical());
    assert_eq!(r.get("ok"), Some(&JsonValue::Bool(true)), "{r:?}");
    let state = r.get("state").unwrap();
    assert_eq!(state.get("undo").unwrap().get("can"), Some(&JsonValue::Bool(false)));
    let mut cam = camera.as_object().unwrap().clone();
    cam.insert("kind".to_string(), json::string("camera"));
    reply(&mut b, &JsonValue::Object(cam).canonical());
    reply(&mut b, r#"{"kind":"depthTolerance","value":0.03}"#);
    let (edited_b, camera_b, svg_b) = snapshot(&mut b);
    assert_eq!(edited_b, edited);
    assert_eq!(camera_b, camera);
    assert!(svg == svg_b, "the restored landscape bakes other bytes");
}

#[test]
fn every_catalog_landscape_restores_from_what_is_edited() {
    let catalog = reply(&mut Session::new(), r#"{"kind":"catalog"}"#);
    for preset in catalog.get("presets").and_then(JsonValue::as_array).unwrap() {
        let name = preset.get("name").and_then(JsonValue::as_str).unwrap();
        let mut a = Session::new();
        reply(&mut a, r#"{"kind":"viewport","width":200,"height":150}"#);
        reply(&mut a, &format!(r#"{{"kind":"create","preset":"{name}"}}"#));
        let edited = reply(&mut a, r#"{"kind":"edited"}"#).get("edited").cloned().unwrap();
        let mut restore = edited.as_object().unwrap().clone();
        restore.insert("kind".to_string(), json::string("restore"));
        let mut b = Session::new();
        reply(&mut b, r#"{"kind":"viewport","width":200,"height":150}"#);
        let r = reply(&mut b, &JsonValue::Object(restore).canonical());
        assert_eq!(r.get("ok"), Some(&JsonValue::Bool(true)), "{name}: {r:?}");
        assert_eq!(reply(&mut b, r#"{"kind":"edited"}"#).get("edited"), Some(&edited), "{name}");
    }
}

#[test]
fn a_link_asking_for_more_than_the_controls_reach_is_refused_touching_nothing() {
    let mut a = Session::new();
    reply(&mut a, r#"{"kind":"viewport","width":200,"height":150}"#);
    reply(&mut a, r#"{"kind":"create","preset":"gamma"}"#);
    let edited = reply(&mut a, r#"{"kind":"edited"}"#).get("edited").cloned().unwrap();
    let base = edited.as_object().unwrap().clone();
    let landscape = |key: &str, value: JsonValue| {
        let mut o = base.clone();
        let mut l = o["landscape"].as_object().unwrap().clone();
        l.insert(key.to_string(), value);
        o.insert("landscape".to_string(), JsonValue::Object(l));
        o
    };
    let layer = |name: &str, change: &dyn Fn(&mut JsonObject)| {
        let mut o = base.clone();
        let layers: Vec<JsonValue> = o["layers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| {
                let mut l = l.as_object().unwrap().clone();
                if l["name"].as_str() == Some(name) {
                    let mut source = l["source"].as_object().unwrap().clone();
                    change(&mut source);
                    l.insert("source".to_string(), JsonValue::Object(source));
                }
                JsonValue::Object(l)
            })
            .collect();
        o.insert("layers".to_string(), JsonValue::Array(layers));
        o
    };
    let mut counts = base.clone();
    counts.insert("levelCounts".to_string(), json::object([("mag_major", JsonValue::Int(1_000_000_000))]));
    let mut window = base.clone();
    let mut l = window["landscape"].as_object().unwrap().clone();
    l.insert("domain".to_string(), json::parse(r#"{"real":{"lo":1,"hi":1},"imag":{"lo":0,"hi":2}}"#).unwrap());
    window.insert("landscape".to_string(), JsonValue::Object(l));
    let cases = [
        ("resolution", landscape("resolution", JsonValue::Int(1_000_000))),
        ("levels", counts),
        ("window", window),
        (
            "hatching",
            layer("wall_hatch", &|s| {
                s.insert("spacing".to_string(), JsonValue::Double(1e-9));
            }),
        ),
        (
            "contour levels",
            layer("mag_minor", &|s| {
                s.insert(
                    "levels".to_string(),
                    json::doubles(&(0..5000).map(|k| f64::from(k) * 1e-3).collect::<Vec<_>>()),
                );
            }),
        ),
    ];
    for (what, mut link) in cases {
        link.insert("kind".to_string(), json::string("restore"));
        let mut b = Session::new();
        let r = reply(&mut b, &JsonValue::Object(link).canonical());
        assert_eq!(r.get("ok"), Some(&JsonValue::Bool(false)), "{what}");
        assert_eq!(r.get("touched"), Some(&JsonValue::Bool(false)), "{what}");
        assert_eq!(r.get("state").unwrap().get("kind").and_then(JsonValue::as_str), Some("empty"), "{what}");
        println!("{what}: {}", r.get("error").and_then(JsonValue::as_str).unwrap());
    }
    for bad in [r#"{"kind":"bakeResolution","value":0}"#, r#"{"kind":"bakeResolution","value":1000000000}"#] {
        assert_eq!(reply(&mut a, bad).get("touched"), Some(&JsonValue::Bool(false)), "{bad}");
    }
    let mut camera = reply(&mut a, r#"{"kind":"state"}"#)
        .get("state")
        .unwrap()
        .get("camera")
        .unwrap()
        .get("exact")
        .unwrap()
        .as_object()
        .unwrap()
        .clone();
    camera.insert("kind".to_string(), json::string("camera"));
    camera.insert("unitsPerPixel".to_string(), JsonValue::Double(0.0));
    assert_eq!(reply(&mut a, &JsonValue::Object(camera).canonical()).get("touched"), Some(&JsonValue::Bool(false)));
}

#[test]
fn gestures_in_one_request_leave_the_camera_each_alone_leaves() {
    let gestures = [
        r#"{"kind":"orbit","delta":[13.5,-4]}"#,
        r#"{"kind":"zoom","factor":1.37,"at":[40,20]}"#,
        r#"{"kind":"orbit","delta":[2,900]}"#,
        r#"{"kind":"pan","delta":[-7,3.25]}"#,
        r#"{"kind":"zoom","factor":0.61,"at":[150,90]}"#,
        r#"{"kind":"orbit","delta":[-31,-2.5]}"#,
    ];
    let open = |s: &mut Session| {
        reply(s, r#"{"kind":"viewport","width":200,"height":150}"#);
        reply(s, r#"{"kind":"create","preset":"sin"}"#);
    };
    let (mut one, mut all) = (Session::new(), Session::new());
    open(&mut one);
    open(&mut all);
    for g in gestures {
        reply(&mut one, &format!(r#"{{"kind":"gesture","gesture":{g}}}"#));
    }
    reply(&mut all, &format!(r#"{{"kind":"gestures","gestures":[{}]}}"#, gestures.join(",")));
    let camera = |s: &mut Session| {
        reply(s, r#"{"kind":"state"}"#).get("state").unwrap().get("camera").unwrap().get("exact").cloned().unwrap()
    };
    assert_eq!(camera(&mut one), camera(&mut all));
    // A malformed gesture applies none of them.
    let before = camera(&mut all);
    let r = reply(&mut all, r#"{"kind":"gestures","gestures":[{"kind":"orbit","delta":[1,1]},{"kind":"spin"}]}"#);
    assert_eq!(r.get("touched"), Some(&JsonValue::Bool(false)));
    assert_eq!(camera(&mut all), before);
}
