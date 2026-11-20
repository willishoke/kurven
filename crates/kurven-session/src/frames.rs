//! The two tiers of a frame, as the window draws them (`MetalView.swift`,
//! `LineGeometry.layOut` in `KurvenMetal/Preview.swift`): while the camera
//! or a control moves, a draft, left to the GPU's depth test; at rest, the
//! curves tested for the camera.
//!
//! The draft's engine half is a landscape's fold curves traced on the grid
//! for the camera (`Heightfield::on_grid`), every other curve as carried.
//! At rest the window's layout tests the folds and the curves on the walls
//! against the solid (`visible_folds`, `visible_wall_curves`) and leaves
//! the rest to the depth test; the judged strokes, what Export writes, are
//! the bake of the scene the window shows (`Document::bake`), every curve
//! tested, which the page draws at rest only without WebGPU. With a pool of
//! engines the tested layers of the frame at rest are laid out by other
//! engines and served by this one (`pool.rs`), as Swift's window lays them
//! out off its main thread.
//!
//! Both draw the scene the window shows (`Document::shown_scene`), the
//! picked face's boundary over it, as Swift's `MetalView` draws
//! `shownScene`; which layers are drawn is `shown_visible_layers`.

use kurven_core::manifest::LayerSource;
use kurven_core::polyline::Layer;

use crate::document::Document;

impl Document {
    /// The layers a draft frame draws, in world space: a landscape's fold
    /// curves traced on the grid for the navigator's camera, every other
    /// layer as the scene holds it. What the GPU depth-tests while the camera
    /// moves.
    pub fn draft_layers(&self) -> Vec<Layer> {
        let Some(scene) = self.shown_scene() else { return Vec::new() };
        let Some(h) = scene.heightfield() else { return scene.layers().to_vec() };
        let view = scene.camera.view;
        scene
            .layers()
            .iter()
            .map(|layer| match layer.spec.source {
                LayerSource::FoldCurves => {
                    Layer { spec: layer.spec.clone(), paths: h.on_grid().fold_curves(&view, false) }
                }
                _ => layer.clone(),
            })
            .collect()
    }

    /// The layers the window lays out at rest (`LineGeometry.layOut(draft:
    /// false)`), in world space: a landscape's fold curves on the exact fold
    /// and tested by the solid, the clipped curves on its walls tested by
    /// the wall they stand on; a sphere-traced surface's folds and clipped
    /// curves with coordinates, and a solid's clipped layers, as the bake
    /// tests them; every other layer as the scene holds it, left to the
    /// depth test (`preview::lay_out`). Swift's window does this off the main
    /// thread at each camera rest.
    pub fn rest_layout(&self) -> Vec<Layer> {
        let Some(scene) = self.shown_scene() else { return Vec::new() };
        scene
            .layers()
            .iter()
            .map(|layer| Layer { spec: layer.spec.clone(), paths: crate::preview::lay_out(layer, &scene, false).paths })
            .collect()
    }
}
