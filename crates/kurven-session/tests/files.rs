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
