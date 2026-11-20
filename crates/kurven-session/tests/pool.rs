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
    pooled_with(pool, svg, |_| {})
}

/// `pooled`, with `between` done to the first session after the pieces are
/// baked and before it puts them together.
fn pooled_with(pool: &mut [Session], svg: bool, between: impl FnOnce(&mut Session)) -> Reply {
    let (first, helpers) = pool.split_first_mut().expect("a session");
    let plan = reply(first, r#"{"kind":"bakePlan"}"#);
    assert_eq!(plan.get("ok"), Some(&JsonValue::Bool(true)), "{plan:?}");
    let pieces = plan.get("pieces").unwrap().as_array().unwrap().to_vec();
    let fingerprint = plan.get("fingerprint").unwrap().clone();
    let mut answered: Vec<Option<Reply>> = vec![None; pieces.len()];
    for (k, p) in pieces.iter().enumerate().rev() {
        let mut piece = p.as_object().unwrap().clone();
        piece.remove("weight");
        let request = json::object([
            ("kind", json::string("bakePiece")),
            ("piece", JsonValue::Object(piece)),
            ("fingerprint", fingerprint.clone()),
        ]);
        let n = helpers.len();
        answered[k] = Some(ok(&mut helpers[k % n], &request.canonical()));
    }
    between(first);
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
    assemble.insert("fingerprint".to_string(), fingerprint);
    assemble.insert("frame".to_string(), plan.get("frame").unwrap().clone());
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
        // Nothing to share: asked to, the plan's engine bakes it itself.
        let baked = ok(&mut pool[2], r#"{"kind":"bakePlan","bakeAlone":true,"svg":true}"#);
        assert!(baked.json.contains(r#""baked":true"#));
        assert_eq!(baked.arrays, alone.arrays);
    }
}

#[test]
fn a_piece_of_another_scene_is_refused_and_a_refusal_is_said() {
    let mut s = sessions(1, GAMMA).pop().unwrap();
    let f = reply(&mut s, r#"{"kind":"bakePlan"}"#).get("fingerprint").unwrap().canonical();
    for piece in
        [r#"{"layer":99}"#, r#"{"layer":0,"from":5,"to":2}"#, r#"{"layer":0,"from":0,"to":100000}"#, r#"{"nope":1}"#]
    {
        let r = reply(&mut s, &format!(r#"{{"kind":"bakePiece","piece":{piece},"fingerprint":{f}}}"#));
        assert_eq!(r.get("ok"), Some(&JsonValue::Bool(false)), "{piece}: {r:?}");
        assert_eq!(r.get("another"), None, "{piece}: refused as a piece, not as of another document");
    }
    // The folds are derived for the camera: no piece of their paths.
    let folds = s.document.framed_scene().unwrap().layers().iter().position(|l| l.spec.name == "folds").unwrap();
    let r = reply(
        &mut s,
        &format!(r#"{{"kind":"bakePiece","piece":{{"layer":{folds},"from":0,"to":1}},"fingerprint":{f}}}"#),
    );
    assert_eq!(r.get("ok"), Some(&JsonValue::Bool(false)), "{r:?}");
    // A piece asked of another document, or of none named, is refused as such.
    for fingerprint in [r#","fingerprint":"0000000000000000""#, ""] {
        let r = reply(&mut s, &format!(r#"{{"kind":"bakePiece","piece":{{"layer":0}}{fingerprint}}}"#));
        assert_eq!(r.get("another"), Some(&JsonValue::Bool(true)), "{r:?}");
    }
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
    let r = reply(&mut s, r##"{"kind":"bakeAssemble","plan":[{"layer":0}],"colors":[["#000000"]],"fingerprint":"x"}"##);
    assert!(r.get("error").and_then(JsonValue::as_str).unwrap().contains("0.widths"), "{r:?}");
    // A move makes another document of the plan's.
    ok(&mut s, r#"{"kind":"gesture","gesture":{"kind":"orbit","delta":[5,0]}}"#);
    let r = reply(&mut s, &format!(r#"{{"kind":"bakePiece","piece":{{"layer":0}},"fingerprint":{f}}}"#));
    assert_eq!(r.get("another"), Some(&JsonValue::Bool(true)), "{r:?}");
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
            let request = format!(
                r#"{{"kind":"restLayout","layers":[{}],"fingerprint":{}}}"#,
                layer.as_i64().unwrap(),
                fingerprint.canonical()
            );
            let r = ok(helper, &request);
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
        assert_eq!(r.get("another"), Some(&JsonValue::Bool(true)), "{r:?}");
        // A helper whose document moved on refuses the layout as of another.
        let r = reply(
            &mut pool[0],
            &format!(
                r#"{{"kind":"restLayout","layers":[{}],"fingerprint":{}}}"#,
                layers[0].canonical(),
                fingerprint.canonical()
            ),
        );
        assert_eq!(r.get("another"), Some(&JsonValue::Bool(true)), "{r:?}");
    }
}

/// The tested layers at rest are named by their keys alone: holding the
/// frame at rest's keys, none is wanted; holding another engine's, all are.
#[test]
fn the_layers_a_frame_at_rest_wants_are_named_by_keys_alone() {
    let mut s = sessions(1, GAMMA).pop().unwrap();
    let rest = json::parse(&ok(&mut s, r#"{"kind":"preview","draft":false}"#).json).unwrap();
    let keys: Vec<JsonValue> =
        rest.get("layers").unwrap().as_array().unwrap().iter().map(|l| l.get("key").unwrap().clone()).collect();
    let held = json::object([("layers", JsonValue::Array(keys))]).canonical();
    let none = reply(&mut s, &format!(r#"{{"kind":"restLayers","have":{held}}}"#));
    assert_eq!(none.get("layers").unwrap().as_array().unwrap().len(), 0, "{none:?}");
    let all = reply(&mut s, r#"{"kind":"restLayers","have":{"layers":[]}}"#);
    let all: Vec<i64> = all.get("layers").unwrap().as_array().unwrap().iter().filter_map(JsonValue::as_i64).collect();
    let scene = s.document.shown_scene().unwrap();
    let tested: Vec<i64> = (0..scene.layers().len())
        .filter(|&i| kurven_session::preview::is_tested_exactly(&scene.layers()[i], &scene))
        .map(|i| i as i64)
        .collect();
    assert_eq!(all, tested);
}

/// An assembly after the document moved on: the plan's document's strokes,
/// consistently (its frame, its SVG), said as a snapshot.
#[test]
fn a_bake_put_together_after_the_document_moved_on_is_the_plans_documents() {
    let mut single = sessions(1, GAMMA).pop().unwrap();
    let alone = ok(&mut single, r#"{"kind":"bake","svg":true}"#);
    let mut pool = sessions(3, GAMMA);
    let together = pooled_with(&mut pool, true, |first| {
        ok(first, r#"{"kind":"gesture","gesture":{"kind":"orbit","delta":[40,0]}}"#);
        ok(first, r#"{"kind":"truncation","truncation":{"kind":"uniform","height":2.5}}"#);
    });
    assert_eq!(together.arrays, alone.arrays, "the plan's document's strokes and SVG");
    let o = json::parse(&together.json).unwrap();
    let a = json::parse(&alone.json).unwrap();
    assert_eq!(o.get("snapshot"), Some(&JsonValue::Bool(true)));
    assert_eq!(o.get("frame"), a.get("frame"), "the plan's frame");
    let status = o.get("state").unwrap().get("status").unwrap().get("bake").and_then(JsonValue::as_str).unwrap();
    assert!(status.contains("as it was when the bake was asked"), "{status}");
}

/// A bake refused before any piece is refused as the single engine's is.
#[test]
fn a_bake_refused_before_its_pieces_is_refused_as_the_bake_is() {
    let requests = [GAMMA, &[r#"{"kind":"perspective","on":true}"#]].concat();
    let mut single = sessions(1, &requests).pop().unwrap();
    let alone = reply(&mut single, r#"{"kind":"bake"}"#);
    let mut s = sessions(1, &requests).pop().unwrap();
    let planned = reply(&mut s, r#"{"kind":"bakePlan"}"#);
    assert_eq!(planned.get("ok"), Some(&JsonValue::Bool(false)));
    assert_eq!(planned.get("error"), alone.get("error"));
    assert_eq!(planned.get("state"), alone.get("state"), "the bake's status, said as bake says it");
}

/// Each edit a slider or a switch makes changes the bake's fingerprint, so
/// a piece asked of the document before it is refused as of another: a cap
/// hatching's and a wall hatching's strokes, a level count, the refinement,
/// the truncation, the depth tolerance and the bake's resolution.
#[test]
fn a_piece_asked_before_any_edit_is_refused_after_it() {
    for edit in [
        r#"{"kind":"strokes","layer":"plateau_hatch","count":60}"#,
        r#"{"kind":"strokes","layer":"wall_hatch","count":60}"#,
        r#"{"kind":"levels","layer":"mag_major","count":4}"#,
        r#"{"kind":"refine","on":false}"#,
        r#"{"kind":"truncation","truncation":{"kind":"uniform","height":3.25}}"#,
        r#"{"kind":"depthTolerance","value":0.03}"#,
        r#"{"kind":"bakeResolution","value":3000}"#,
    ] {
        let mut s = sessions(1, GAMMA).pop().unwrap();
        let f = reply(&mut s, r#"{"kind":"bakePlan"}"#).get("fingerprint").unwrap().canonical();
        let piece = format!(r#"{{"kind":"bakePiece","piece":{{"layer":4}},"fingerprint":{f}}}"#);
        assert_eq!(reply(&mut s, &piece).get("ok"), Some(&JsonValue::Bool(true)), "{edit}: before it");
        ok(&mut s, edit);
        let r = reply(&mut s, &piece);
        assert_eq!(
            r.get("another"),
            Some(&JsonValue::Bool(true)),
            "{edit}: {}",
            r.get("error").map(|e| e.canonical()).unwrap_or_default()
        );
    }
}

/// Phase 8, steps 5 to 7, pooled: the scenes the page can now make, held by
/// several sessions as a pool's helpers hold them (each replaying the same
/// requests, a bundle opened from a file staged under its own name and let
/// go of after), bake through the pool to the bytes the session bakes alone,
/// and a frame at rest laid out elsewhere is the frame.
#[test]
fn the_page_scenes_bake_pooled_as_alone() {
    let input = |name: &str| format!("{}/../fixtures/swift/session/inputs/{name}.kurven", env!("CARGO_MANIFEST_DIR"));
    let staged = |s: &mut Session, name: &str, dir: &str| {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let file = path.file_name().unwrap().to_string_lossy().into_owned();
            s.stage(name, &file, std::fs::read(&path).unwrap());
        }
    };
    let view = r#"{"kind":"viewport","width":500,"height":400}"#;
    let scenes: Vec<(&str, Option<&str>, Vec<&str>)> = vec![
        (
            "a torus, its radius, lines and linear curve",
            None,
            vec![
                view,
                r#"{"kind":"createSurface","preset":"torus"}"#,
                r#"{"kind":"surfaceField","field":"major","value":2.6}"#,
                r#"{"kind":"surfaceLineCount","axis":"around","count":24}"#,
                r#"{"kind":"linearCurve","on":true}"#,
            ],
        ),
        (
            "sn conformal, with its coordinate curves",
            None,
            vec![
                view,
                r#"{"kind":"createSurface","preset":"sn"}"#,
                r#"{"kind":"surfaceLines","on":true}"#,
                r#"{"kind":"surfaceField","field":"conformal","value":1}"#,
            ],
        ),
        (
            "the forced torus, its components swapped",
            None,
            vec![
                view,
                r#"{"kind":"createSurface","preset":"forced"}"#,
                r#"{"kind":"component","field":"radial","value":1}"#,
            ],
        ),
        (
            "a torus painted with sn, a curve added",
            None,
            vec![
                view,
                r#"{"kind":"createSurface","preset":"torus"}"#,
                r#"{"kind":"pick","point":[2,0,1]}"#,
                r#"{"kind":"function","expression":"sn(z, 0.64)"}"#,
                r#"{"kind":"add","name":"surface_u"}"#,
            ],
        ),
        (
            "sn in relief, opened from a file",
            Some("relief"),
            vec![
                view,
                r#"{"kind":"open","name":"opened/1/relief.kurven","untrusted":true}"#,
                r#"{"kind":"unstage","name":"opened/1/relief.kurven"}"#,
                r#"{"kind":"surfaceField","field":"relief","value":0.3}"#,
            ],
        ),
        (
            "the picture frame, opened from a file, its strokes and a face hatched",
            Some("frame"),
            vec![
                view,
                r#"{"kind":"open","name":"opened/1/frame.kurven","untrusted":true}"#,
                r#"{"kind":"unstage","name":"opened/1/frame.kurven"}"#,
                r#"{"kind":"strokes","layer":"hatching","count":30}"#,
                r#"{"kind":"pick","point":[-1.5,0,1]}"#,
                r#"{"kind":"add","name":"hatching"}"#,
            ],
        ),
        (
            "gamma opened from a file, banded, its band dragged, its graph painted",
            Some("gamma"),
            vec![
                r#"{"kind":"viewport","width":600,"height":450}"#,
                r#"{"kind":"open","name":"opened/1/gamma.kurven","untrusted":true}"#,
                r#"{"kind":"unstage","name":"opened/1/gamma.kurven"}"#,
                r#"{"kind":"truncation","truncation":{"kind":"realBands","bands":[{"below":0,"height":3}],"beyond":4}}"#,
                r#"{"kind":"drag","control":{"slider":"truncation"},"phase":"begin"}"#,
                r#"{"kind":"drag","control":{"slider":"truncation"},"phase":"move","truncation":{"kind":"realBands","bands":[{"below":0,"height":3.5},{"below":1.5,"height":2}],"beyond":4}}"#,
                r#"{"kind":"drag","control":{"slider":"truncation"},"phase":"end"}"#,
                r#"{"kind":"axis","axis":"imag"}"#,
            ],
        ),
    ];
    for (what, bundle, requests) in scenes {
        let mut pool: Vec<Session> = (0..3)
            .map(|_| {
                let mut s = Session::new();
                if let Some(b) = bundle {
                    staged(&mut s, &format!("opened/1/{b}.kurven"), &input(b));
                }
                for r in &requests {
                    ok(&mut s, r);
                }
                s
            })
            .collect();
        let together = pooled(&mut pool, true);
        let alone = ok(&mut pool[1], r#"{"kind":"bake","svg":true}"#);
        assert_eq!(together.arrays, alone.arrays, "{what}: pooled, the bake alone");
    }
}

/// The edits of steps 5 and 6 each change the fingerprint a pool holds its
/// helpers to: a piece of a landscape asked before a band is dragged, its
/// hatching turned or a curve painted on it is refused after; a layout at
/// rest of a surface asked before one of its fields, its linear curve or a
/// function painted on it is refused after.
#[test]
fn the_page_edits_change_the_fingerprint() {
    let landscape: [&[&str]; 3] = [
        &[
            r#"{"kind":"drag","control":{"slider":"truncation"},"phase":"begin"}"#,
            r#"{"kind":"drag","control":{"slider":"truncation"},"phase":"move","truncation":{"kind":"realBands","bands":[{"below":0,"height":2.5}],"beyond":3}}"#,
            r#"{"kind":"drag","control":{"slider":"truncation"},"phase":"end"}"#,
        ],
        &[r#"{"kind":"axis","axis":"imag"}"#],
        &[r#"{"kind":"pick","point":[2.5,0.0,1.3293403881791355]}"#, r#"{"kind":"add","name":"graph_v"}"#],
    ];
    for edit in landscape {
        let mut s = sessions(1, GAMMA).pop().unwrap();
        let f = reply(&mut s, r#"{"kind":"bakePlan"}"#).get("fingerprint").unwrap().canonical();
        let piece = format!(r#"{{"kind":"bakePiece","piece":{{"layer":4}},"fingerprint":{f}}}"#);
        assert_eq!(reply(&mut s, &piece).get("ok"), Some(&JsonValue::Bool(true)), "{edit:?}: before it");
        for r in edit {
            ok(&mut s, r);
        }
        let r = reply(&mut s, &piece);
        assert_eq!(r.get("another"), Some(&JsonValue::Bool(true)), "{edit:?}: {r:?}");
    }
    let surface: [&[&str]; 4] = [
        &[r#"{"kind":"surfaceField","field":"major","value":2.5}"#],
        &[r#"{"kind":"surfaceLineCount","axis":"across","count":9}"#],
        &[r#"{"kind":"linearCurve","on":true}"#],
        &[r#"{"kind":"function","expression":"sn(z, 0.64)"}"#],
    ];
    for edit in surface {
        let mut s = sessions(
            1,
            &[r#"{"kind":"viewport","width":500,"height":400}"#, r#"{"kind":"createSurface","preset":"torus"}"#],
        )
        .pop()
        .unwrap();
        let wanted = reply(&mut s, r#"{"kind":"restLayers"}"#);
        let layer = wanted.get("layers").unwrap().as_array().unwrap()[0].canonical();
        let f = wanted.get("fingerprint").unwrap().canonical();
        let layout = format!(r#"{{"kind":"restLayout","layers":[{layer}],"fingerprint":{f}}}"#);
        assert_eq!(reply(&mut s, &layout).get("ok"), Some(&JsonValue::Bool(true)), "{edit:?}: before it");
        for r in edit {
            ok(&mut s, r);
        }
        let r = reply(&mut s, &layout);
        assert_eq!(r.get("another"), Some(&JsonValue::Bool(true)), "{edit:?}: {r:?}");
    }
}
