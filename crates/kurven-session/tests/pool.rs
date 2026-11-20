//! The slow work shared among several engines holding one document
//! (`kurven_session::pool`): the judged strokes from pieces baked by other
//! sessions and put together by the first are `bake`'s, the SVG the same
//! bytes, however many sessions bake them and in whatever order; and a frame
//! at rest served with layouts another session laid out is the frame the
//! serving session lays out itself.

use kurven_bake::BakeOptions;
use kurven_bake::pieces::{assemble, bake_piece, plan};
use kurven_core::json::{self, JsonObject, JsonValue};
use kurven_session::protocol::{Reply, Session};

fn reply(session: &mut Session, request: &str) -> JsonValue {
    json::parse(&session.call(request).json).unwrap()
}

fn ok(session: &mut Session, request: &str) -> Reply {
    let r = session.call(request);
    assert!(r.json.contains(r#""ok":true"#), "{request}: {}", &r.json[..r.json.len().min(400)]);
    r
}

/// Sessions holding the same document: each replays the same requests, as a
/// pool's Workers replay the page's journal.
fn sessions(n: usize, requests: &[&str]) -> Vec<Session> {
    (0..n)
        .map(|_| {
            let mut s = Session::new();
            for r in requests {
                ok(&mut s, r);
            }
            s
        })
        .collect()
}

fn svg_of(r: Reply) -> Vec<u8> {
    r.arrays.into_iter().find(|(n, _)| n == "svg").expect("an SVG").1.into_bytes()
}

/// The pool's bake as the page's client runs it: the plan from the first
/// session, every piece baked by one of the others (handed out round the
/// pool, last piece first, so none is baked in the plan's order), and the
/// pieces given back to the first to put together.
fn pooled(pool: &mut [Session], svg: bool) -> Reply {
    let (first, helpers) = pool.split_first_mut().expect("a session");
    let plan = reply(first, r#"{"kind":"bakePlan"}"#);
    assert_eq!(plan.get("ok"), Some(&JsonValue::Bool(true)), "{plan:?}");
    let pieces = plan.get("pieces").unwrap().as_array().unwrap().to_vec();
    let mut answered: Vec<Option<Reply>> = vec![None; pieces.len()];
    for (k, p) in pieces.iter().enumerate().rev() {
        let mut piece = p.as_object().unwrap().clone();
        piece.remove("weight");
        let request = json::object([("kind", json::string("bakePiece")), ("piece", JsonValue::Object(piece))]);
        let n = helpers.len();
        answered[k] = Some(ok(&mut helpers[k % n], &request.canonical()));
    }
    let mut colors = Vec::new();
    for (k, r) in answered.into_iter().enumerate() {
        let r = r.unwrap();
        let o = json::parse(&r.json).unwrap();
        colors.push(o.get("colors").unwrap().clone());
        for (name, array) in r.arrays {
            let name = name.strip_prefix("piece.").unwrap();
            first.give(&format!("{k}.{name}"), array.into_bytes());
        }
    }
    let mut assemble = JsonObject::new();
    assemble.insert("kind".to_string(), json::string("bakeAssemble"));
    assemble.insert("svg".to_string(), JsonValue::Bool(svg));
    assemble.insert(
        "plan".to_string(),
        JsonValue::Array(
            pieces
                .iter()
                .map(|p| {
                    let mut p = p.as_object().unwrap().clone();
                    p.remove("weight");
                    JsonValue::Object(p)
                })
                .collect(),
        ),
    );
    assemble.insert("colors".to_string(), JsonValue::Array(colors));
    ok(first, &JsonValue::Object(assemble).canonical())
}

const GAMMA: &[&str] = &[
    r#"{"kind":"viewport","width":600,"height":450}"#,
    r#"{"kind":"create","preset":"gamma"}"#,
    r#"{"kind":"set","key":"res","value":200}"#,
    r#"{"kind":"truncation","truncation":{"kind":"uniform","height":3.5}}"#,
    r#"{"kind":"gesture","gesture":{"kind":"orbit","delta":[30,-12]}}"#,
];

#[test]
fn a_pooled_bake_is_the_bake_to_the_byte_at_any_pool_size() {
    let mut single = sessions(1, GAMMA).pop().unwrap();
    let alone = ok(&mut single, r#"{"kind":"bake","svg":true}"#);
    let status = reply(&mut single, r#"{"kind":"state"}"#).get("state").unwrap().get("status").unwrap().clone();
    for n in [1, 2, 3, 5] {
        let mut pool = sessions(n + 1, GAMMA);
        let together = pooled(&mut pool, true);
        assert_eq!(together.json, alone.json, "pool of {n}: the reply's JSON, state and all");
        assert_eq!(together.arrays, alone.arrays, "pool of {n}: every stroke and the SVG, to the byte");
        let said = reply(&mut pool[0], r#"{"kind":"state"}"#).get("state").unwrap().get("status").unwrap().clone();
        assert_eq!(said, status, "the bake's status, said as bake says it");
    }
    assert!(!svg_of(alone).is_empty());
}

#[test]
fn the_pieces_put_together_are_the_bake_however_the_layers_are_cut() {
    for requests in [GAMMA, &[r#"{"kind":"create","preset":"cubic"}"#, r#"{"kind":"set","key":"res","value":160}"#]] {
        let mut s = sessions(1, requests).pop().unwrap();
        let scene = s.document.framed_scene().unwrap();
        let mut options = BakeOptions::new(s.document.bake_resolution);
        options.depth_tolerance = Some(s.document.depth_tolerance);
        let whole = kurven_bake::bake(&scene, &options).unwrap().strokes;
        let mut cuts = Vec::new();
        for budget in [1, 7, 300, 2048, usize::MAX] {
            let pieces = plan(&scene, &options, budget).unwrap();
            cuts.push(pieces.len());
            let baked = pieces.iter().rev().map(|&p| bake_piece(&scene, &options, p).unwrap()).collect::<Vec<_>>();
            let baked = baked.into_iter().rev().collect();
            assert_eq!(assemble(&pieces, baked).unwrap(), whole, "cut in {} pieces", pieces.len());
        }
        assert!(cuts[0] > cuts[3] && cuts[3] >= cuts[4], "{cuts:?}");
        ok(&mut s, r#"{"kind":"state"}"#);
    }
}

#[test]
fn a_surface_or_a_solid_is_one_piece_and_bakes_as_bake_does() {
    for requests in [
        &[r#"{"kind":"viewport","width":500,"height":400}"#, r#"{"kind":"createSurface","preset":"sn"}"#][..],
        &[r#"{"kind":"viewport","width":500,"height":400}"#, r#"{"kind":"createSurface","preset":"torus"}"#][..],
    ] {
        let mut pool = sessions(3, requests);
        let plan = reply(&mut pool[0], r#"{"kind":"bakePlan"}"#);
        assert_eq!(plan.get("pieces").unwrap().as_array().unwrap().len(), 1, "{plan:?}");
        let together = pooled(&mut pool, true);
        let alone = ok(&mut pool[1], r#"{"kind":"bake","svg":true}"#);
        assert_eq!(together.arrays, alone.arrays);
    }
}

#[test]
fn a_piece_of_another_scene_is_refused_and_a_refusal_is_said() {
    let mut s = sessions(1, GAMMA).pop().unwrap();
    for piece in
        [r#"{"layer":99}"#, r#"{"layer":0,"from":5,"to":2}"#, r#"{"layer":0,"from":0,"to":100000}"#, r#"{"nope":1}"#]
    {
        let r = reply(&mut s, &format!(r#"{{"kind":"bakePiece","piece":{piece}}}"#));
        assert_eq!(r.get("ok"), Some(&JsonValue::Bool(false)), "{piece}: {r:?}");
    }
    // The folds are derived for the camera: no piece of their paths.
    let folds = s.document.framed_scene().unwrap().layers().iter().position(|l| l.spec.name == "folds").unwrap();
    let r = reply(&mut s, &format!(r#"{{"kind":"bakePiece","piece":{{"layer":{folds},"from":0,"to":1}}}}"#));
    assert_eq!(r.get("ok"), Some(&JsonValue::Bool(false)), "{r:?}");
    // A piece's refusal, given to the engine that assembles, is its status.
    let r = reply(&mut s, r#"{"kind":"bakeAssemble","failed":"bake failed: a reason"}"#);
    assert_eq!(r.get("error").and_then(JsonValue::as_str), Some("bake failed: a reason"));
    assert_eq!(
        r.get("state").unwrap().get("status").unwrap().get("bake").and_then(JsonValue::as_str),
        Some("bake failed: a reason")
    );
    // Inputs a request does not read are dropped with it.
    s.give("0.widths", vec![0; 8]);
    ok(&mut s, r#"{"kind":"state"}"#);
    let r = reply(&mut s, r##"{"kind":"bakeAssemble","plan":[{"layer":0}],"colors":[["#000000"]]}"##);
    assert!(r.get("error").and_then(JsonValue::as_str).unwrap().contains("0.widths"), "{r:?}");
}

/// A frame at rest from layouts another session laid out: the frame the
/// serving session lays out itself, key for key and segment for segment.
#[test]
fn a_frame_at_rest_served_from_layouts_laid_out_elsewhere_is_the_frame() {
    for requests in
        [GAMMA, &[r#"{"kind":"viewport","width":500,"height":400}"#, r#"{"kind":"createSurface","preset":"sn"}"#][..]]
    {
        let mut pool = sessions(3, requests);
        // Each a frame first, so the content is held and the draft keys too.
        let draft = ok(&mut pool[0], r#"{"kind":"preview","draft":true}"#);
        let held = json::parse(&draft.json).unwrap();
        let content = held.get("content").unwrap().clone();
        let keys: Vec<JsonValue> =
            held.get("layers").unwrap().as_array().unwrap().iter().map(|l| l.get("key").unwrap().clone()).collect();
        let have = json::object([("content", content), ("layers", JsonValue::Array(keys))]).canonical();

        let wanted = reply(&mut pool[0], &format!(r#"{{"kind":"restLayers","have":{have}}}"#));
        let layers = wanted.get("layers").unwrap().as_array().unwrap().to_vec();
        assert!(!layers.is_empty(), "a frame at rest has tested layers: {wanted:?}");
        let fingerprint = wanted.get("fingerprint").unwrap().clone();

        // Each layer laid out by another session.
        let mut laid = Vec::new();
        let mut given = Vec::new();
        for (k, layer) in layers.iter().enumerate() {
            let helper = &mut pool[1 + k % 2];
            let r = ok(helper, &format!(r#"{{"kind":"restLayout","layers":[{}]}}"#, layer.as_i64().unwrap()));
            let o = json::parse(&r.json).unwrap();
            assert_eq!(o.get("fingerprint"), Some(&fingerprint), "the helpers hold the same document");
            laid.extend(o.get("laid").unwrap().as_array().unwrap().iter().cloned());
            given.extend(r.arrays);
        }
        for (name, array) in given {
            pool[0].give(&name, array.into_bytes());
        }
        let request = format!(
            r#"{{"kind":"preview","draft":false,"have":{have},"laid":{},"fingerprint":{}}}"#,
            JsonValue::Array(laid.clone()).canonical(),
            fingerprint.canonical()
        );
        let served = ok(&mut pool[0], &request);
        // The same frame laid out by the serving session itself, holding the
        // same: the keys are its own (content is counted per process, so
        // another session's are not these).
        let itself = ok(&mut pool[0], &format!(r#"{{"kind":"preview","draft":false,"have":{have}}}"#));
        assert_eq!(served.json, itself.json);
        assert_eq!(served.arrays, itself.arrays);

        // Given to a session whose document moved on, refused.
        ok(&mut pool[0], r#"{"kind":"gesture","gesture":{"kind":"orbit","delta":[5,0]}}"#);
        let r = reply(
            &mut pool[0],
            &format!(r#"{{"kind":"preview","draft":false,"laid":[],"fingerprint":{}}}"#, fingerprint.canonical()),
        );
        assert_eq!(r.get("ok"), Some(&JsonValue::Bool(false)), "{r:?}");
    }
}
