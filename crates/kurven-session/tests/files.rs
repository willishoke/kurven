//! A `.kurven` bundle opened from a file (Phase 8, step 7): every bundle the
//! session's fixtures hold opens; one that is malformed, or asks for more
//! than the controls reach, is refused before anything is built, and the
//! open document and its undo history are as they were.

use kurven_core::json::{self, JsonValue};
use kurven_session::protocol::Session;

fn call(session: &mut Session, request: &str) -> JsonValue {
    json::parse(&session.call(request).json).unwrap()
}

fn dir(name: &str) -> String {
    format!("{}/../fixtures/swift/session/inputs/{name}.kurven", env!("CARGO_MANIFEST_DIR"))
}

fn files(name: &str) -> Vec<(String, Vec<u8>)> {
    std::fs::read_dir(dir(name))
        .unwrap()
        .map(|e| {
            let path = e.unwrap().path();
            (path.file_name().unwrap().to_string_lossy().into_owned(), std::fs::read(&path).unwrap())
        })
        .collect()
}

fn stage(session: &mut Session, name: &str, files: &[(String, Vec<u8>)]) {
    for (path, bytes) in files {
        session.stage(name, path, bytes.clone());
    }
}

fn open(session: &mut Session, name: &str) -> JsonValue {
    call(session, &format!(r#"{{"kind":"open","name":"{name}","untrusted":true}}"#))
}

/// The fixture's manifest with `change` made to it.
fn changed(name: &str, change: impl FnOnce(&mut JsonValue)) -> Vec<(String, Vec<u8>)> {
    let mut change = Some(change);
    files(name)
        .into_iter()
        .map(|(path, bytes)| {
            if path != "manifest.json" {
                return (path, bytes);
            }
            let mut m = json::parse(std::str::from_utf8(&bytes).unwrap()).unwrap();
            (change.take().unwrap())(&mut m);
            (path, m.canonical().into_bytes())
        })
        .collect()
}

fn set(v: &mut JsonValue, path: &[&str], value: JsonValue) {
    let mut at = v;
    for key in &path[..path.len() - 1] {
        at = match at {
            JsonValue::Object(o) => o.get_mut(*key).unwrap(),
            JsonValue::Array(a) => &mut a[key.parse::<usize>().unwrap_or_else(|_| panic!("{key} of {path:?}"))],
            _ => panic!("no {key}"),
        };
    }
    match at {
        JsonValue::Object(o) => {
            o.insert(path[path.len() - 1].to_string(), value);
        }
        JsonValue::Array(a) => a[path[path.len() - 1].parse::<usize>().unwrap()] = value,
        _ => panic!("not a container"),
    }
}

#[test]
fn every_fixture_bundle_opens_from_a_file() {
    for name in ["gamma", "banded", "rational", "sn", "relief", "torus", "frame", "cube"] {
        let mut s = Session::new();
        stage(&mut s, name, &files(name));
        let r = open(&mut s, name);
        assert_eq!(r.get("ok"), Some(&JsonValue::Bool(true)), "{name}: {r:?}");
        let kind = r.get("state").unwrap().get("kind").unwrap().as_str().unwrap();
        assert!(["landscape", "surface", "polyhedron"].contains(&kind), "{name}: {kind}");
    }
}

#[test]
fn a_bundle_refused_leaves_the_document_and_its_undo_alone() {
    let mut s = Session::new();
    call(&mut s, r#"{"kind":"create","preset":"sin"}"#);
    call(&mut s, r#"{"kind":"truncation","truncation":{"kind":"uniform","height":3}}"#);
    let before = call(&mut s, r#"{"kind":"state"}"#).get("state").unwrap().clone();
    assert_eq!(before.get("undo").unwrap().get("can"), Some(&JsonValue::Bool(true)));
    type Case<'a> = (&'a str, Vec<(String, Vec<u8>)>, &'a str);
    let cases: Vec<Case> = vec![
        ("nothing", vec![("height.npy".to_string(), vec![0; 16])], "has none"),
        ("garbage", vec![("manifest.json".to_string(), b"{not json".to_vec())], ""),
        ("schema", vec![("manifest.json".to_string(), br#"{"schema":1}"#.to_vec())], ""),
        (
            "levels",
            changed("gamma", |m| {
                let levels: Vec<f64> = (0..2000).map(|k| k as f64 * 0.01).collect();
                set(m, &["layers", "0", "source", "levels"], json::doubles(&levels));
            }),
            "2000 levels",
        ),
        (
            "hatching",
            changed("frame", |m| {
                let layers = m.get("layers").and_then(JsonValue::as_array).unwrap().to_vec();
                let k = layers
                    .iter()
                    .position(|l| {
                        l.get("source").and_then(|s| s.get("kind")).and_then(JsonValue::as_str) == Some("hatching")
                    })
                    .unwrap();
                set(m, &["layers", &k.to_string(), "source", "spacing"], JsonValue::Double(0.005));
            }),
            "strokes across the drawing",
        ),
        ("ring", changed("torus", |m| set(m, &["surface", "grid"], JsonValue::Int(60000))), "grid"),
        (
            "function",
            changed("sn", |m| set(m, &["surface", "shape", "grid"], JsonValue::Int(3000))),
            "function's samples",
        ),
        ("lines", changed("torus", |m| set(m, &["surface", "lines", "u"], JsonValue::Int(5000))), "coordinate curves"),
        (
            // An isophote under a light with no direction: refused as the
            // schema reads it (`Tone::has_direction`), touching nothing.
            "light",
            changed("torus", |m| {
                let mut layers = m.get("layers").and_then(JsonValue::as_array).unwrap().to_vec();
                layers.push(
                    json::parse(
                        r##"{"clipped":true,"color":"#000000","heightPolicy":"surface","name":"surface_isophote",
                        "role":"scaffold","width":0.2,"source":{"faces":{"kind":"surface"},"field":"isophote",
                        "kind":"contour","levels":[0.0],"light":[0.0,0.0,0.0],"restriction":{"kind":"all"}}}"##,
                    )
                    .unwrap(),
                );
                set(m, &["layers"], JsonValue::Array(layers));
            }),
            "not all zero",
        ),
        (
            // A window read the wrong way round: a negative extent would let
            // any spacing through.
            "window",
            changed("gamma", |m| {
                let lo = m.get("domain").unwrap().get("real").unwrap().get("lo").unwrap().clone();
                let hi = m.get("domain").unwrap().get("real").unwrap().get("hi").unwrap().clone();
                set(m, &["domain", "real", "lo"], hi);
                set(m, &["domain", "real", "hi"], lo);
            }),
            "no interval",
        ),
        (
            "bands",
            changed("banded", |m| {
                let bands: Vec<JsonValue> = (0..100)
                    .map(|k| json::object([("below", JsonValue::Double(k as f64)), ("height", JsonValue::Double(1.0))]))
                    .collect();
                set(m, &["truncation", "bands"], JsonValue::Array(bands));
            }),
            "bands",
        ),
    ];
    for (name, bundle, says) in cases {
        stage(&mut s, name, &bundle);
        let r = open(&mut s, name);
        assert_eq!(r.get("ok"), Some(&JsonValue::Bool(false)), "{name} was opened");
        assert_eq!(r.get("touched"), Some(&JsonValue::Bool(false)), "{name}");
        let error = r.get("error").and_then(JsonValue::as_str).unwrap();
        assert!(error.contains(says), "{name}: {error}");
        assert_eq!(r.get("state"), Some(&before), "{name}: the document is as it was");
    }
    // And the undo still undoes the edit before.
    let r = call(&mut s, r#"{"kind":"undo"}"#);
    assert_ne!(r.get("state").unwrap().get("truncation"), before.get("truncation"));
}

#[test]
fn a_light_with_no_direction_is_refused_by_request_and_by_link() {
    let mut s = Session::new();
    call(&mut s, r#"{"kind":"createSurface","preset":"torus"}"#);
    call(&mut s, r#"{"kind":"pick","point":[2,0,1]}"#);
    let r = call(&mut s, r#"{"kind":"add","name":"surface_isophote"}"#);
    assert_eq!(r.get("ok"), Some(&JsonValue::Bool(true)), "{r:?}");
    let before = call(&mut s, r#"{"kind":"state"}"#).get("state").unwrap().clone();
    for direction in ["[0,0,0]", "[1e-200,0,0]", "[1e200,1e200,1e200]"] {
        let r = call(&mut s, &format!(r#"{{"kind":"light","layer":"surface_isophote","direction":{direction}}}"#));
        assert_eq!(r.get("ok"), Some(&JsonValue::Bool(false)), "{direction}");
        assert_eq!(r.get("touched"), Some(&JsonValue::Bool(false)), "{direction}");
        assert_eq!(r.get("state"), Some(&before), "{direction}");
    }
    // The isophote's sliders read its angles.
    let layers = before.get("layers").and_then(JsonValue::as_array).unwrap();
    let iso = layers.iter().find(|l| l.get("name").and_then(JsonValue::as_str) == Some("surface_isophote")).unwrap();
    assert_eq!(iso.get("lightAngles").and_then(JsonValue::as_array).map(|a| a.len()), Some(2));
}

/// Review of PR #77, finding 3, rejected: a wall's hatching is vertical
/// strokes spaced along the wall (`wall_hatch_lines`: `stroke_count(edge
/// length, spacing)`), each its two ends, so the truncation's height moves
/// where each stroke ends and not how many there are. The window's extent
/// bounds their count, which `openable` holds to 400 across.
#[test]
fn a_walls_hatching_counts_strokes_along_the_wall_however_tall() {
    let paths = |height: f64| {
        let mut s = Session::new();
        let bundle = changed("gamma", |m| {
            set(m, &["truncation"], json::parse(&format!(r#"{{"kind":"uniform","height":{height}}}"#)).unwrap())
        });
        stage(&mut s, "gamma", &bundle);
        let r = open(&mut s, "gamma");
        assert_eq!(r.get("ok"), Some(&JsonValue::Bool(true)), "{height}: {r:?}");
        let layers = r.get("state").unwrap().get("layers").and_then(JsonValue::as_array).unwrap().to_vec();
        let wall = layers.iter().find(|l| l.get("name").and_then(JsonValue::as_str) == Some("wall_hatch")).unwrap();
        wall.get("paths").and_then(JsonValue::as_i64).unwrap()
    };
    let low = paths(5.0);
    assert!(low > 10, "{low}");
    assert_eq!(paths(1e7), low);
}
