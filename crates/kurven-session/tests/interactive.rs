//! Interactive timings, native: the same requests `crates/kurven-wasm/
//! interactive.mjs` sends the wasm module, through the same protocol, for
//! cn and gamma at the app's default resolution. Ignored, since it times
//! rather than checks:
//!
//!   cargo test --release -p kurven-session --test interactive -- --ignored --nocapture

use std::time::Instant;

use kurven_session::protocol::Session;

/// The requests, in order, each named when it is timed.
pub fn script(preset: &str, lowered: f64) -> Vec<(Option<&'static str>, String)> {
    vec![
        (Some("open: sample at 600, place the contours on f"), format!(r#"{{"kind":"create","preset":"{preset}"}}"#)),
        (None, r#"{"kind":"gesture","gesture":{"kind":"orbit","delta":[30,0]}}"#.to_string()),
        (Some("camera change: draft frame (folds on the grid)"), r#"{"kind":"frame","draft":true}"#.to_string()),
        (Some("camera change: rest frame (the window's layout)"), r#"{"kind":"frame","draft":false}"#.to_string()),
        (Some("camera change: judged strokes (the bake)"), r#"{"kind":"bake"}"#.to_string()),
        (None, r#"{"kind":"drag","control":{"slider":"truncation"},"phase":"begin"}"#.to_string()),
        (
            Some("truncation drag: a step (restyle from the grids)"),
            format!(r#"{{"kind":"drag","control":{{"slider":"truncation"}},"phase":"move","value":{lowered}}}"#),
        ),
        (Some("truncation drag: draft frame"), r#"{"kind":"frame","draft":true}"#.to_string()),
        (None, r#"{"kind":"drag","control":{"slider":"truncation"},"phase":"end"}"#.to_string()),
        (Some("truncation drag: rest frame (the window's layout)"), r#"{"kind":"frame","draft":false}"#.to_string()),
        (Some("truncation drag: judged strokes (the bake)"), r#"{"kind":"bake"}"#.to_string()),
    ]
}

#[test]
#[ignore]
fn interactive_timings() {
    for (preset, lowered) in [("cn", 3.2), ("gamma", 4.0)] {
        let mut session = Session::new();
        for (name, request) in script(preset, lowered) {
            let started = Instant::now();
            let reply = session.call(&request);
            let ms = started.elapsed().as_secs_f64() * 1e3;
            assert!(reply.json.contains(r#""ok":true"#), "{request}: {}", &reply.json[..reply.json.len().min(300)]);
            if let Some(name) = name {
                println!("{preset}\t{name}\t{ms:.1} ms");
            }
        }
    }
}

/// Where the window's rest layout spends its time: the folds, then the
/// curves on the walls.
#[test]
#[ignore]
fn rest_layout_by_part() {
    use kurven_core::hatching::FaceKind;
    use kurven_core::manifest::LayerSource;
    for preset in ["cn", "gamma"] {
        let mut session = Session::new();
        session.call(&format!(r#"{{"kind":"create","preset":"{preset}"}}"#));
        session.call(r#"{"kind":"gesture","gesture":{"kind":"orbit","delta":[30,0]}}"#);
        let scene = session.document.framed_scene().unwrap();
        let h = scene.heightfield().unwrap();
        let view = scene.camera.view;
        let started = Instant::now();
        let folds = h.visible_folds(&view, scene.depth_tolerance, true, scene.resolution);
        let fold_ms = started.elapsed().as_secs_f64() * 1e3;
        let started = Instant::now();
        let mut wall_vertices = 0;
        for layer in scene.layers() {
            if layer.spec.clipped
                && !matches!(layer.spec.source, LayerSource::FoldCurves)
                && scene.geometry().face_of(&layer.spec) == Some(FaceKind::Wall)
            {
                wall_vertices += layer.paths.vertices.len();
                h.visible_wall_curves(&layer.paths, &view, scene.depth_tolerance, scene.resolution);
            }
        }
        let wall_ms = started.elapsed().as_secs_f64() * 1e3;
        println!(
            "{preset}\tfolds {fold_ms:.1} ms ({} vertices)\twalls {wall_ms:.1} ms ({wall_vertices} vertices)",
            folds.vertices.len()
        );
    }
}
