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

/// The fixture's manifest with `change` made to its text: for numbers no
/// JSON value can hold.
fn text(name: &str, change: impl Fn(&str) -> String) -> Vec<(String, Vec<u8>)> {
    files(name)
        .into_iter()
        .map(|(path, bytes)| {
            if path != "manifest.json" {
                return (path, bytes);
            }
            let before = std::str::from_utf8(&bytes).unwrap();
            let after = change(before);
            assert_ne!(before, after, "{name}: the change found its text");
            (path, after.into_bytes())
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
            // gamma's walls hatched at a ten-thousandth of its window: past
            // a file's 4000 strokes across.
            changed("gamma", |m| {
                let layers = m.get("layers").and_then(JsonValue::as_array).unwrap().to_vec();
                let k = layers
                    .iter()
                    .position(|l| l.get("name").and_then(JsonValue::as_str) == Some("wall_hatch"))
                    .unwrap();
                set(m, &["layers", &k.to_string(), "source", "spacing"], JsonValue::Double(1e-4));
            }),
            "strokes across the drawing",
        ),
        ("ring", changed("torus", |m| set(m, &["surface", "grid"], JsonValue::Int(60000))), "grid"),
        (
            "function",
            // Past Swift's own bound on a function's samples.
            changed("sn", |m| set(m, &["surface", "shape", "grid"], JsonValue::Int(5000))),
            "4096",
        ),
        (
            // A contour kept to a band of Re past a double's range: read as
            // infinite, which the state cannot write back (a trap in wasm).
            "restriction",
            text("gamma", |t| {
                t.replacen(r#"{"kind":"belowTruncation"}"#, r#"{"kind":"band","axis":"real","lo":0,"hi":1e400}"#, 1)
            }),
            "past a double's range",
        ),
        (
            // A camera preset's angle past a double's range.
            "preset",
            text("torus", |t| t.replacen(r#""xAngle":"#, r#""xAngle":1e400,"was":"#, 1)),
            "past a double's range",
        ),
        (
            // A solid's hatching read every 0.0008 along its lines: past the
            // 2^20 tone readings Swift's schema allows a polyhedron's
            // hatchings (the one face whose tone is read every `step`).
            "step",
            changed("frame", |m| set(m, &["layers", "1", "source", "step"], JsonValue::Double(0.0008))),
            "read their tones",
        ),
        (
            // A restriction of twenty terms.
            "terms",
            changed("gamma", |m| {
                let band = r#"{"kind":"band","axis":"real","lo":0,"hi":1}"#;
                let of = vec![band; 20].join(",");
                set(
                    m,
                    &["layers", "0", "source", "restriction"],
                    json::parse(&format!(r#"{{"kind":"intersection","of":[{of}]}}"#)).unwrap(),
                )
            }),
            "terms",
        ),
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

/// Review of PR #77 (second), finding 1, rejected: a hatching's `step` is
/// read only on a solid's faces, where Swift's schema bounds the readings
/// (the case "step" above); a landscape's hatching, which is all a link
/// holds, may carry no step at all: the schema refuses one ("a line is its
/// two ends on a wall or a riser ... so schema 3 writes none"). A link with
/// its walls' step at its floor, a millionth of the spacing, is refused,
/// touching nothing.
#[test]
fn a_links_hatching_may_carry_no_step() {
    let mut a = Session::new();
    call(&mut a, r#"{"kind":"create","preset":"sin"}"#);
    let mut edited = call(&mut a, r#"{"kind":"edited"}"#).get("edited").unwrap().clone();
    let layers = edited.get("layers").and_then(JsonValue::as_array).unwrap().to_vec();
    let k = layers.iter().position(|l| l.get("name").and_then(JsonValue::as_str) == Some("wall_hatch")).unwrap();
    let spacing = layers[k].get("source").unwrap().get("spacing").and_then(JsonValue::as_f64).unwrap();
    set(&mut edited, &["layers", &k.to_string(), "source", "step"], JsonValue::Double(spacing * 1e-6));
    let mut restore = edited.as_object().unwrap().clone();
    restore.insert("kind".to_string(), json::string("restore"));
    let before = call(&mut a, r#"{"kind":"state"}"#).get("state").unwrap().clone();
    let r = call(&mut a, &JsonValue::Object(restore).canonical());
    assert_eq!(r.get("ok"), Some(&JsonValue::Bool(false)), "{r:?}");
    assert_eq!(r.get("touched"), Some(&JsonValue::Bool(false)));
    assert!(r.get("error").and_then(JsonValue::as_str).unwrap().contains("a step on a landscape's hatching"));
    assert_eq!(r.get("state"), Some(&before));
}

/// Review of PR #77 (second), finding 3: a bundle opened from a file is the
/// document the same bundle opened by name gives, whatever the session had
/// set before -- the depth tolerance, the bake's resolution, the mode, the
/// refinement, the draft resolution, the viewport, a pick's status, an edit
/// to undo. (Opened either way, the depth tolerance and the bake's
/// resolution are the bundle's first preset's, as Swift's adoption sets
/// them.) And a link's landscape is made in a successor that keeps what
/// `create` keeps.
#[test]
fn a_file_opens_as_the_same_bundle_by_name_and_a_link_keeps_the_settings() {
    let settings = [
        r#"{"kind":"create","preset":"sin"}"#,
        r#"{"kind":"truncation","truncation":{"kind":"uniform","height":2}}"#,
        r#"{"kind":"depthTolerance","value":0.07}"#,
        r#"{"kind":"bakeResolution","value":3000}"#,
        r#"{"kind":"mode","mode":"depth"}"#,
        r#"{"kind":"refine","on":false}"#,
        r#"{"kind":"draftResolution","value":200}"#,
        r#"{"kind":"viewport","width":700,"height":500}"#,
        r#"{"kind":"pick","pixel":[2,2]}"#,
    ];
    for name in ["gamma", "sn", "frame"] {
        let opened = |untrusted: bool| {
            let mut s = Session::new();
            for r in settings {
                call(&mut s, r);
            }
            stage(&mut s, name, &files(name));
            let r = call(&mut s, &format!(r#"{{"kind":"open","name":"{name}","untrusted":{untrusted}}}"#));
            assert_eq!(r.get("ok"), Some(&JsonValue::Bool(true)), "{name}: {r:?}");
            r.get("state").unwrap().clone()
        };
        assert_eq!(opened(true), opened(false), "{name}: opened from a file, as by name");
    }
    // A link, restored after the settings: what `create` keeps, kept.
    let mut a = Session::new();
    call(&mut a, r#"{"kind":"create","preset":"sin"}"#);
    let mut restore = call(&mut a, r#"{"kind":"edited"}"#).get("edited").unwrap().as_object().unwrap().clone();
    restore.insert("kind".to_string(), json::string("restore"));
    let restore = JsonValue::Object(restore).canonical();
    let made = |last: &str| {
        let mut s = Session::new();
        for r in settings {
            call(&mut s, r);
        }
        let r = call(&mut s, last);
        assert_eq!(r.get("ok"), Some(&JsonValue::Bool(true)), "{r:?}");
        r.get("state").unwrap().clone()
    };
    let (restored, created) = (made(&restore), made(r#"{"kind":"create","preset":"sin"}"#));
    for key in ["depthTolerance", "bakeResolution", "mode", "refine", "draftResolution", "viewport", "status"] {
        assert_eq!(restored.get(key), created.get(key), "{key}: restored as created");
    }
    assert_eq!(restored.get("mode").and_then(JsonValue::as_str), Some("depth"));
    assert_eq!(restored.get("draftResolution").and_then(JsonValue::as_i64), Some(200));
}

/// Review of PR #77 (second), finding 5: a file is held to Swift's own
/// bounds on a surface, not to the sliders', so every bundle our tools write
/// opens: sn sampled at 2048 (`kurven-cli surface sn --grid 2048`, past the
/// Samples slider's 1200, within Swift's 4096), a torus with 300 by 150
/// coordinate curves (`--coordinate-curves`, past the counts' 200).
#[test]
fn bundles_our_tools_write_past_the_sliders_open() {
    for (name, bundle) in [
        ("sn-2048", changed("sn", |m| set(m, &["surface", "shape", "grid"], JsonValue::Int(2048)))),
        (
            "torus-300",
            changed("torus", |m| {
                set(m, &["surface", "lines", "u"], JsonValue::Int(300));
                set(m, &["surface", "lines", "v"], JsonValue::Int(150));
            }),
        ),
    ] {
        let mut s = Session::new();
        stage(&mut s, name, &bundle);
        let r = open(&mut s, name);
        assert_eq!(r.get("ok"), Some(&JsonValue::Bool(true)), "{name}: {r:?}");
    }
}

/// Review of PR #77 (second), finding 4, for links: a contour's restriction
/// in a link is held as a file's is -- a band past a double's range (read as
/// infinite), or a restriction of twenty terms, is refused, touching nothing,
/// whether in the link's layers or in the layers its request carries.
#[test]
fn a_links_restriction_is_held_as_a_files() {
    let mut a = Session::new();
    call(&mut a, r#"{"kind":"create","preset":"sin"}"#);
    let mut restore = call(&mut a, r#"{"kind":"edited"}"#).get("edited").unwrap().as_object().unwrap().clone();
    restore.insert("kind".to_string(), json::string("restore"));
    let text = JsonValue::Object(restore).canonical();
    let before = call(&mut a, r#"{"kind":"state"}"#).get("state").unwrap().clone();
    let band = r#"{"kind":"band","axis":"real","lo":0,"hi":1}"#;
    let terms = format!(r#"{{"kind":"intersection","of":[{}]}}"#, vec![band; 20].join(","));
    for (what, link, says) in [
        (
            "infinite",
            text.replacen(r#"{"kind":"belowTruncation"}"#, r#"{"kind":"band","axis":"real","lo":0,"hi":1e400}"#, 1),
            "past a double's range",
        ),
        ("terms in the request", text.replacen(r#"{"kind":"belowTruncation"}"#, &terms, 1), "terms"),
        (
            "terms in the layers",
            {
                // The link's own layers: the last "layers" of its canonical
                // text, its keys sorted, after the request's.
                let (head, tail) = text.split_at(text.rfind(r#""layers":["#).unwrap());
                format!("{head}{}", tail.replacen(r#"{"kind":"belowTruncation"}"#, &terms, 1))
            },
            "terms",
        ),
    ] {
        assert_ne!(link, text, "{what}: the link holds a restriction to change");
        let r = call(&mut a, &link);
        assert_eq!(r.get("ok"), Some(&JsonValue::Bool(false)), "{what}: {r:?}");
        assert_eq!(r.get("touched"), Some(&JsonValue::Bool(false)), "{what}");
        assert!(r.get("error").and_then(JsonValue::as_str).unwrap().contains(says), "{what}: {r:?}");
        assert_eq!(r.get("state"), Some(&before), "{what}");
    }
}
