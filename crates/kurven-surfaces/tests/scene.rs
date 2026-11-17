//! The scene's memo of a parametric surface's fold curves: each scene's
//! own, so a scene drawing other creases from the same camera places its
//! folds from its own.

use kurven_core::camera::Camera;
use kurven_core::manifest::LayerSource;
use kurven_core::polyline::{Layer, PolylineSet};
use kurven_core::scene::{PreviewMode, Scene};
use kurven_core::spaces::ViewSpace;
use kurven_core::surface_request::{Shape, SurfaceRequest};
use kurven_surfaces::SurfaceDrawing;

fn folds(scene: &Scene) -> PolylineSet<ViewSpace> {
    let layer = scene.layers().iter().find(|l| matches!(l.spec.source, LayerSource::FoldCurves)).expect("folds");
    scene.visible(layer, None, None).expect("the folds draw").remove(0).1
}

/// sn in relief, whose truncation edge is a crease the folds are seeded
/// from: a scene's folds, then the same camera on the same surface with the
/// truncation edge taken out (`drawing`) and a clone of the first, each
/// asked after the first has placed its own. The scene without the crease
/// places what a scene built afresh with its layers places, which is not
/// the first's; the clone places the first's again.
#[test]
fn a_scene_with_other_creases_places_its_own_folds_from_the_same_camera() {
    let request = SurfaceRequest {
        grid: 256,
        ..SurfaceRequest::new(Shape::Relief {
            modulus: 0.64,
            major: 2.0,
            minor: 1.0,
            amplitude: 0.4,
            repeats: (1, 1),
            grid: 120,
            turn: 0.0,
            floor: None,
            soft: false,
        })
    };
    let drawing = SurfaceDrawing::build(&request, None, false).expect("the relief builds");
    let camera = Camera::oblique(&SurfaceRequest::PROJECTION);
    let first = drawing.scene(camera);
    assert!(!first.creases().is_empty(), "the relief has a truncation edge");
    let with = folds(&first);
    let bare: Vec<Layer> = first.layers().iter().filter(|l| l.spec.name != "truncation edge").cloned().collect();
    let without = first.drawing(bare.clone());
    assert!(without.creases().is_empty(), "and the other scene none");
    let fresh = Scene::parametric_scene(drawing.surface.clone(), bare, camera, PreviewMode::Drawing, 0.0);
    assert_eq!(folds(&without), folds(&fresh), "folds placed from the scene's own creases");
    assert_ne!(folds(&without), with, "which are not the first scene's");
    assert_eq!(folds(&first.clone()), with, "and a clone places the first's again");
}
