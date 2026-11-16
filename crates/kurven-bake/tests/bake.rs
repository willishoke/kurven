//! kurven-test's bake suites (`Sources/kurven-test/main.swift`, "6. the
//! bake"), as far as they reach the strokes.

use kurven_bake::{BakeError, BakeOptions, bake};
use kurven_core::bundle::KurvenBundle;
use kurven_core::fixtures;
use kurven_core::navigation::Orbit;
use kurven_core::scene::Scene;
use kurven_core::spaces::P3;

fn uniform_mesh() -> (KurvenBundle, Scene) {
    let bundle = KurvenBundle::read(fixtures::shared("contract/uniform_mesh.kurven")).expect("the fixture reads");
    let preset = bundle.manifest.preset("recip").expect("the recip preset").clone();
    let scene = Scene::from_bundle(&bundle, &preset);
    (bundle, scene)
}

/// "bake: a perspective camera is refused, with the reason".
#[test]
fn a_perspective_camera_is_refused_with_the_reason() {
    let (bundle, mut scene) = uniform_mesh();
    bake(&scene, &BakeOptions::new(64)).expect("an orthographic scene bakes");

    let mut orbit = Orbit::matching(&bundle.manifest.preset("recip").unwrap().projection, P3::new(0.0, 0.0, 0.0));
    orbit.distance = Some(10.0);
    scene.camera = orbit.camera();
    match bake(&scene, &BakeOptions::new(64)) {
        Err(error @ BakeError::Perspective) => assert!(error.to_string().contains("orthographic"), "{error}"),
        other => panic!("a perspective scene does not bake: {other:?}"),
    }
}

/// "bake: a tiled bake draws what a single-pass bake draws": the strokes'
/// half. Swift's depth half (coverage and values of the stitched buffer
/// against one pass) waits for the port's rasterizer, Phase 4; the port
/// renders no buffer, so its strokes cannot depend on the split, nor on the
/// resolution.
#[test]
fn a_tiled_bake_draws_what_a_single_pass_bake_draws() {
    let (_, scene) = uniform_mesh();
    let whole = bake(&scene, &BakeOptions { tiles: Some(1), ..BakeOptions::new(240) }).unwrap();
    assert!(whole.strokes.path_count() > 0);
    for n in [2, 3, 5] {
        let split = bake(&scene, &BakeOptions { tiles: Some(n), ..BakeOptions::new(240) }).unwrap();
        assert_eq!(split.strokes, whole.strokes, "{n}x{n} passes produce the same strokes");
    }
    let finer = bake(&scene, &BakeOptions::new(3000)).unwrap();
    assert_eq!(finer.strokes, whole.strokes, "the resolution changes no stroke");
}

#[test]
fn an_outline_is_refused_until_there_is_a_depth_buffer() {
    let (_, scene) = uniform_mesh();
    match bake(&scene, &BakeOptions { outline: Some(0.5), ..BakeOptions::new(64) }) {
        Err(BakeError::Port(refusal)) => assert_eq!(refusal.phase, 4),
        other => panic!("an outline is a typed refusal: {other:?}"),
    }
}

/// Every layer the scene draws, in order, each named by its spec and styled
/// by it.
#[test]
fn the_strokes_are_the_scenes_layers() {
    let (bundle, scene) = uniform_mesh();
    let baked = bake(&scene, &BakeOptions::new(64)).unwrap();
    assert_eq!(baked.specs.len(), baked.strokes.layers.len());
    let names: Vec<_> = baked.specs.iter().map(|s| s.name.as_str()).collect();
    let declared: Vec<_> = bundle.manifest.layers.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, declared);
    for (spec, (style, _)) in baked.specs.iter().zip(&baked.strokes.layers) {
        assert_eq!((style.color.as_str(), style.width), (spec.color.as_str(), spec.width));
    }
}
