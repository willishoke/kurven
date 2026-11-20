//! The picker's pictures (Phase 8, step 4): every catalog entry is drawn in
//! a scene of its own, the open document untouched, and keyed by what would
//! change its picture.

use std::collections::BTreeSet;
use std::time::Instant;

use kurven_core::json::{self, JsonValue};
use kurven_session::protocol::Session;
use kurven_session::thumbnails::{self, Entry};

fn reply(session: &mut Session, request: &str) -> (JsonValue, Vec<String>) {
    let r = session.call(request);
    (json::parse(&r.json).unwrap(), r.arrays.into_iter().map(|(n, _)| n).collect())
}

#[test]
fn the_catalog_lists_every_entry_with_a_distinct_key() {
    let mut s = Session::new();
    let (r, _) = reply(&mut s, r#"{"kind":"catalog"}"#);
    let entries = r.get("entries").and_then(JsonValue::as_array).unwrap();
    assert_eq!(entries.len(), Entry::all().len());
    let ids: Vec<&str> = entries.iter().map(|e| e.get("id").unwrap().as_str().unwrap()).collect();
    assert!(ids.contains(&"gamma") && ids.contains(&"surface:sn") && ids.contains(&"surface:torus"));
    // The functions first, then the surfaces, as the picker groups them.
    let first_surface = ids.iter().position(|i| i.starts_with("surface:")).unwrap();
    assert!(ids[first_surface..].iter().all(|i| i.starts_with("surface:")));
    let keys: BTreeSet<&str> = entries.iter().map(|e| e.get("thumbnail").unwrap().as_str().unwrap()).collect();
    assert_eq!(keys.len(), entries.len(), "every picture its own key");
    // Stable across calls: not a process-seeded hash.
    let again = Entry::named("gamma").unwrap();
    assert_eq!(thumbnails::key(&again), thumbnails::key(&Entry::named("gamma").unwrap()));
    assert!(thumbnails::key(&again).starts_with("gamma-"));
}

#[test]
fn a_thumbnail_leaves_the_document_alone() {
    let mut s = Session::new();
    reply(&mut s, r#"{"kind":"create","preset":"sin"}"#);
    let (before, _) = reply(&mut s, r#"{"kind":"state"}"#);
    let (r, arrays) = reply(&mut s, r#"{"kind":"thumbnail","entry":"gamma"}"#);
    assert_eq!(r.get("ok"), Some(&JsonValue::Bool(true)), "{r:?}");
    assert_eq!(r.get("state"), before.get("state"), "the open document is not touched");
    // The frame the window's preview draws at rest, every layer shown, the
    // resources with it, at 440 by 300.
    let frame = r.get("uniforms").unwrap().get("frame").unwrap();
    assert_eq!(frame.get("cols").and_then(JsonValue::as_i64), Some(440));
    assert_eq!(frame.get("rows").and_then(JsonValue::as_i64), Some(300));
    assert!(r.get("resources").unwrap().get("kind").is_some());
    let layers = r.get("layers").and_then(JsonValue::as_array).unwrap();
    assert!(!layers.is_empty() && layers.iter().all(|l| l.get("visible") == Some(&JsonValue::Bool(true))));
    assert!(arrays.iter().any(|a| a == "resource.heights"));
    // Judged: strokes and the frame they map into.
    let (r, arrays) = reply(&mut s, r#"{"kind":"thumbnail","entry":"surface:torus","judged":true}"#);
    assert_eq!(r.get("ok"), Some(&JsonValue::Bool(true)), "{r:?}");
    assert!(!r.get("strokes").and_then(JsonValue::as_array).unwrap().is_empty());
    assert!(arrays.iter().any(|a| a == "stroke.0.points"));
    let (r, _) = reply(&mut s, r#"{"kind":"thumbnail","entry":"nothing"}"#);
    assert_eq!(r.get("touched"), Some(&JsonValue::Bool(false)));
}

/// Every entry draws, natively; the times it takes, for the findings.
#[test]
#[ignore = "slow: every entry, timed (cargo test --release -- --ignored)"]
fn every_entry_draws() {
    let mut s = Session::new();
    for entry in Entry::all() {
        let started = Instant::now();
        let (r, _) = reply(&mut s, &format!(r#"{{"kind":"thumbnail","entry":"{}"}}"#, entry.id()));
        let preview = started.elapsed();
        assert_eq!(r.get("ok"), Some(&JsonValue::Bool(true)), "{}: {r:?}", entry.id());
        let started = Instant::now();
        let (r, _) = reply(&mut s, &format!(r#"{{"kind":"thumbnail","entry":"{}","judged":true}}"#, entry.id()));
        assert_eq!(r.get("ok"), Some(&JsonValue::Bool(true)), "{}: {r:?}", entry.id());
        println!("{:>16}: preview {preview:?}, judged {:?}", entry.id(), started.elapsed());
    }
}
