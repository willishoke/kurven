//! What the page's step 6 reads and sends (Phase 8): the picked face's
//! caption, the periods "Function…" offers, a banded truncation's height
//! dragged as one undo step, a polyhedron described, and a layer's paths,
//! light and hatching axis in the state.

use kurven_core::json::{self, JsonValue};
use kurven_session::protocol::Session;

fn call(session: &mut Session, request: &str) -> JsonValue {
    json::parse(&session.call(request).json).unwrap()
}

fn ok(session: &mut Session, request: &str) -> JsonValue {
    let r = call(session, request);
    assert_eq!(r.get("ok"), Some(&JsonValue::Bool(true)), "{request}: {r:?}");
    r
}

fn open(session: &mut Session, name: &str) -> JsonValue {
    let dir = format!("{}/../fixtures/swift/session/inputs/{name}.kurven", env!("CARGO_MANIFEST_DIR"));
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let file = path.file_name().unwrap().to_string_lossy().into_owned();
        session.stage(name, &file, std::fs::read(&path).unwrap());
    }
    ok(session, &format!(r#"{{"kind":"open","name":"{name}"}}"#))
}

fn state(r: &JsonValue) -> &JsonValue {
    r.get("state").unwrap()
}

fn layer<'a>(r: &'a JsonValue, name: &str) -> &'a JsonValue {
    state(r)
        .get("layers")
        .and_then(JsonValue::as_array)
        .unwrap()
        .iter()
        .find(|l| l.get("name").and_then(JsonValue::as_str) == Some(name))
        .unwrap_or_else(|| panic!("no layer {name}"))
}

#[test]
fn a_pick_is_captioned_and_a_torus_offers_a_function() {
    let mut s = Session::new();
    ok(&mut s, r#"{"kind":"createSurface","preset":"torus"}"#);
    // The top of the tube, on the ring of radius 2.
    let r = ok(&mut s, r#"{"kind":"pick","point":[2,0,1]}"#);
    let picked = state(&r).get("picked").unwrap();
    assert_eq!(picked.get("face").and_then(JsonValue::as_str), Some("surface"), "{picked:?}");
    assert_eq!(picked.get("caption").and_then(JsonValue::as_str), Some("Picked the surface, which is one face."));
    assert_eq!(picked.get("takesFunction"), Some(&JsonValue::Bool(true)));
    assert_eq!(picked.get("expression").and_then(JsonValue::as_str), Some(""));
    // The periods: the catalog's for sn, the surface's own for a function it
    // does not know.
    let r = ok(&mut s, r#"{"kind":"periods","expression":"sn(z, 0.64)"}"#);
    let known = r.get("known").and_then(JsonValue::as_array).unwrap();
    assert_eq!(known.len(), 2);
    assert_eq!(r.get("offered"), r.get("known"));
    let r = ok(&mut s, r#"{"kind":"periods","expression":"z^2"}"#);
    assert_eq!(r.get("known"), Some(&JsonValue::Null));
    assert_eq!(r.get("offered").and_then(JsonValue::as_array).map(|a| a.len()), Some(2));
    // Painted: the pick is kept, and the painted function is offered back.
    ok(&mut s, r#"{"kind":"function","expression":"sin(z)"}"#);
    let r = ok(&mut s, r#"{"kind":"state"}"#);
    let picked = state(&r).get("picked").unwrap();
    assert_eq!(picked.get("expression").and_then(JsonValue::as_str), Some("sin(z)"));
    assert_eq!(state(&r).get("undo").unwrap().get("name").and_then(JsonValue::as_str), Some("Function sin(z)"));
    // Through the hole, off the drawing: nothing picked.
    let r = ok(&mut s, r#"{"kind":"pick","pixel":[600,400]}"#);
    assert_eq!(state(&r).get("picked"), Some(&JsonValue::Null));
}

#[test]
fn a_bands_height_dragged_is_one_step() {
    let mut s = Session::new();
    let r = open(&mut s, "banded");
    let t = state(&r).get("truncation").unwrap().clone();
    assert_eq!(t.get("kind").and_then(JsonValue::as_str), Some("realBands"));
    let undo_before = state(&r).get("undo").unwrap().clone();
    // The first band's height dragged to 2, then 2.5, the whole truncation each move.
    let mut moved = t.clone();
    let drag = r#"{"kind":"drag","control":{"slider":"truncation"},"phase":"#;
    ok(&mut s, &format!(r#"{drag}"begin"}}"#));
    for height in [2.0, 2.5] {
        if let JsonValue::Object(o) = &mut moved
            && let Some(JsonValue::Array(bands)) = o.get_mut("bands")
            && let Some(JsonValue::Object(b)) = bands.first_mut()
        {
            b.insert("height".to_string(), JsonValue::Double(height));
        }
        ok(&mut s, &format!(r#"{drag}"move","truncation":{}}}"#, moved.canonical()));
    }
    let r = ok(&mut s, &format!(r#"{drag}"end"}}"#));
    let now = state(&r).get("truncation").unwrap();
    assert_eq!(now.canonical(), moved.canonical());
    assert_eq!(state(&r).get("undo").unwrap().get("name").and_then(JsonValue::as_str), Some("Truncation"));
    // One undo puts the whole drag back.
    let r = ok(&mut s, r#"{"kind":"undo"}"#);
    assert_eq!(state(&r).get("truncation").unwrap().canonical(), t.canonical());
    assert_eq!(state(&r).get("undo"), Some(&undo_before));
}

#[test]
fn a_polyhedron_is_described_and_its_hatching_counted() {
    let mut s = Session::new();
    let r = open(&mut s, "frame");
    let p = state(&r).get("polyhedron").unwrap();
    println!("{p:?}");
    assert!(p.get("boundary").and_then(JsonValue::as_str).unwrap().contains(" vertices, "));
    assert_eq!(p.get("shapeLabel").and_then(JsonValue::as_str), Some("Genus"));
    assert_eq!(p.get("shape").and_then(JsonValue::as_str), Some("1, not convex"));
    // Every layer says how many paths it draws; a hatching its strokes.
    let layers = state(&r).get("layers").and_then(JsonValue::as_array).unwrap();
    assert!(layers.iter().all(|l| l.get("paths").and_then(JsonValue::as_i64).is_some()));
    assert!(layers.iter().any(|l| l.get("strokes").is_some()), "{layers:?}");
    // A landscape's plateau hatching names its axis.
    let r = open(&mut s, "gamma");
    assert_eq!(state(&r).get("polyhedron"), Some(&JsonValue::Null));
    assert_eq!(layer(&r, "plateau_hatch").get("axis").and_then(JsonValue::as_str), Some("real"));
}
