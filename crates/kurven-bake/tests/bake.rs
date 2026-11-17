//! kurven-test's bake suites (`Sources/kurven-test/main.swift`, "6. the
//! bake"), as far as they reach the strokes.

use kurven_bake::{BakeError, BakeOptions, bake, bake_with_image};
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

/// A landscape's outline is traced from the depth pass, which the port
/// renders with the GPU preview (Phase 7); a parametric surface's comes
/// from its coordinate image's depth.
#[test]
fn a_landscapes_outline_is_refused_until_there_is_a_depth_pass() {
    let (_, scene) = uniform_mesh();
    match bake(&scene, &BakeOptions { outline: Some(0.5), ..BakeOptions::new(64) }) {
        Err(BakeError::Port(refusal)) => assert_eq!(refusal.phase, 7),
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

/// The bake asks whether a scene projects to nothing from its counts, and
/// the answer is `view_bounds`' being none: a landscape projects to
/// something, and one with no tiles, no occluder and no clipped vertex to
/// nothing, which the bake refuses as Swift's does.
#[test]
fn a_scene_that_projects_to_nothing_is_refused() {
    use kurven_core::mesh::Mesh;
    use kurven_core::scene::PreviewMode;

    let (bundle, scene) = uniform_mesh();
    assert!(!scene.projects_to_nothing().unwrap());
    assert!(scene.view_bounds().unwrap().is_some());
    let o = &bundle.manifest.occluder;
    let empty = Scene::heightfield_scene(
        bundle.surface.clone(),
        Mesh::new(Vec::new(), Vec::new()),
        Vec::new(),
        o.region.clone(),
        1,
        Vec::new(),
        scene.camera,
        PreviewMode::Drawing,
        scene.depth_tolerance,
        None,
        o.walls.clone(),
    );
    assert!(empty.projects_to_nothing().unwrap());
    assert!(empty.view_bounds().unwrap().is_none());
    let options = BakeOptions { resolution: 64, tiles: None, depth_tolerance: None, outline: None };
    assert!(matches!(bake(&empty, &options), Err(BakeError::EmptyScene)));
}

/// A torus's coordinate curves and folds, sphere traced or, with its tubular
/// neighbourhood taken away, tested by its coordinate image.
fn torus_scene(traced: bool) -> kurven_core::scene::Scene {
    use kurven_core::camera::Camera;
    use kurven_core::manifest::{HeightPolicy, LayerRole, LayerSource, LayerSpec, ObliqueProjection};
    use kurven_core::parametric::ParametricSurface;
    use kurven_core::polyline::{Layer, PolylineSet};
    use kurven_core::scene::{PreviewMode, Scene};
    let torus = ParametricSurface::torus(2.0, 1.0, (64, 32)).expect("the torus");
    let surface = if traced {
        torus
    } else {
        ParametricSurface::new(torus.u, torus.v, true, None, None, torus.map.clone()).expect("the bare torus")
    };
    let lines = surface.coordinate_curves((8, 4), 256);
    let layers = vec![
        Layer {
            spec: LayerSpec::new(
                "lines",
                LayerRole::Scaffold,
                LayerSource::CoordinateCurves { u: 8, v: 4 },
                0.3,
                HeightPolicy::Surface,
            ),
            paths: lines,
        },
        Layer {
            spec: LayerSpec::new("folds", LayerRole::Outline, LayerSource::FoldCurves, 0.6, HeightPolicy::Surface),
            paths: PolylineSet::empty(),
        },
    ];
    let camera =
        Camera::oblique(&ObliqueProjection { shear: 0.0, x_angle: -55.0, z_angle: 30.0, flip_x: false, y_scale: None });
    Scene::parametric_scene(surface, layers, camera, PreviewMode::Drawing, 0.0)
}

/// However many passes a parametric bake asks for, even more than the frame
/// has rows, the CPU rasterizer renders the frame whole, once: the image and
/// every stroke are the untiled bake's, with no empty or one-row pass.
#[test]
fn a_parametric_bake_is_rendered_whole_however_it_is_tiled() {
    for traced in [true, false] {
        let scene = torus_scene(traced);
        for (resolution, tiles) in [(2, 3), (3, 5), (5, 4), (9, 9), (240, 3)] {
            let whole = bake_with_image(&scene, &BakeOptions { tiles: Some(1), ..BakeOptions::new(resolution) })
                .expect("a bake");
            let split = bake_with_image(&scene, &BakeOptions { tiles: Some(tiles), ..BakeOptions::new(resolution) })
                .expect("a bake");
            assert_eq!(split.tiles, 1, "{resolution} pixels asked in {tiles} passes take one");
            assert_eq!(split.strokes, whole.strokes, "traced {traced}: the tiling changes no stroke");
            let (a, b) = (split.surface.expect("an image"), whole.surface.expect("an image"));
            let bits = |c: &[[f32; 2]]| c.iter().map(|p| [p[0].to_bits(), p[1].to_bits()]).collect::<Vec<_>>();
            assert!(
                a.depth == b.depth && bits(&a.coords) == bits(&b.coords),
                "traced {traced}: nor the image, to the bit"
            );
        }
    }
    // Sphere traced with no outline, nothing reads the image, and none is
    // rendered; the frame is reported all the same.
    let traced = bake(&torus_scene(true), &BakeOptions::new(300)).expect("a bake");
    assert!(traced.surface.is_none() && traced.frame.is_some_and(|f| f.rows == 300));
    let outlined =
        bake(&torus_scene(true), &BakeOptions { outline: Some(0.5), ..BakeOptions::new(300) }).expect("a bake");
    assert!(outlined.surface.is_some() && outlined.strokes.layers.len() == traced.strokes.layers.len() + 1);
}
