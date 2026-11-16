//! A scene to its strokes: the port of `Sources/KurvenBake/Bake.swift`,
//! strokes only.
//!
//! The realtime preview and the bake are the same computation at different
//! resolutions. In Swift the bake renders a depth buffer with Metal, for its
//! `depth` output and the outline; no stroke of a landscape or a polyhedron
//! reads it, since every layer arrives tested by its face and the scene's
//! occlusion test (`Scene::visible_layers`). So the port bakes the strokes
//! and renders no buffer: there is no PNG (the page rasterizes), no `depth`,
//! no `surface` and no `tiles` in a [`Bake`], and an outline, which is traced
//! from the buffer, is a typed refusal. The depth raster comes with Phase 4's
//! CPU rasterizer (`SurfaceImage::rasterize`), which a parametric surface
//! with no exact test reads as its coordinate image; until then a parametric
//! scene is refused too.

use std::fmt;

use kurven_core::hidden_line::HiddenLine;
use kurven_core::manifest::LayerSpec;
use kurven_core::parametric::PortRefusal;
use kurven_core::scene::{Scene, SceneGeometry};
use kurven_core::svg::{Strokes, Style};

/// Metal's largest texture side, which `BakeOptions.tileCount` splits the
/// depth pass by (`KurvenMetal/Renderer.swift`).
pub const METAL_TEXTURE_LIMIT: i64 = 16384;

/// How to bake.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BakeOptions {
    /// Depth resolution, in pixels along each axis of the whole drawing.
    /// The port renders no depth buffer, so no stroke depends on it.
    pub resolution: i64,
    /// How many tiles to split the depth *pass* into per axis. `None` picks the
    /// smallest number that fits Metal's texture limit.
    pub tiles: Option<i64>,
    /// Hidden-line depth tolerance; `None` takes the scene's.
    pub depth_tolerance: Option<f64>,
    /// Trace the outline of the drawn region and add it as a final stroke
    /// layer. Off by default: the plates are bounded by their own wall strokes, and
    /// an outline on top of that is a second edge. Traced from the depth
    /// buffer, so refused by the port until it has one.
    pub outline: Option<f64>,
}

impl BakeOptions {
    /// Swift's `init(resolution:)` with its defaults.
    pub fn new(resolution: i64) -> BakeOptions {
        BakeOptions { resolution, tiles: None, depth_tolerance: None, outline: None }
    }

    /// How many passes per axis Swift's depth pass takes at `resolution`.
    pub fn tile_count(&self, resolution: i64) -> i64 {
        if let Some(tiles) = self.tiles {
            return tiles.max(1);
        }
        1.max((resolution + METAL_TEXTURE_LIMIT - 1) / METAL_TEXTURE_LIMIT)
    }
}

/// Why a scene does not bake: Swift's two cases, with Swift's messages, and
/// what the port has not brought across yet.
#[derive(Debug, Clone, PartialEq)]
pub enum BakeError {
    EmptyScene,
    Perspective,
    Port(PortRefusal),
}

impl fmt::Display for BakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BakeError::EmptyScene => write!(f, "bake: the scene projects to nothing (no geometry, no strokes)"),
            // Swift's multi-line literal holds runs of thirteen spaces where
            // its lines were joined (crates/FINDINGS.md); kept as it is.
            BakeError::Perspective => write!(
                f,
                "bake: this camera is perspective, and only orthographic cameras bake.\nA bake clips per vertex \
                 against a depth buffer it indexes by an             affine map from view coordinates to \
                 pixels; under perspective that             map depends on depth. It could be fixed by \
                 projecting first, and             nothing has asked for it: perspective is for navigating, and \
                 the             plates are orthographic."
            ),
            BakeError::Port(refusal) => write!(f, "bake: {refusal}"),
        }
    }
}

impl std::error::Error for BakeError {}

impl From<PortRefusal> for BakeError {
    fn from(refusal: PortRefusal) -> Self {
        BakeError::Port(refusal)
    }
}

/// A baked drawing.
#[derive(Debug, Clone, PartialEq)]
pub struct Bake {
    pub strokes: Strokes,
    /// The layer each stroke layer draws, in order: a layer of the scene, or
    /// one width of a solid's hatching (`Scene::visible`). Swift's outline,
    /// the stroke layer past the last of these, is refused here.
    pub specs: Vec<LayerSpec>,
}

/// Bake a scene to strokes (Swift's `MetalRenderer.bake(_:options:)`).
///
/// Swift renders the depth buffer, stitched from `options.tile_count` passes,
/// and then takes every layer from `Scene.visibleLayers`, which reads the
/// buffer only as a parametric surface's coordinate image. The port takes the
/// layers alone: the same strokes for a landscape, at any resolution and
/// split.
pub fn bake(scene: &Scene, options: &BakeOptions) -> Result<Bake, BakeError> {
    if scene.camera.is_perspective() {
        return Err(BakeError::Perspective);
    }
    if scene.view_bounds()?.is_none() {
        return Err(BakeError::EmptyScene);
    }
    let depth_tolerance = options.depth_tolerance.unwrap_or(scene.depth_tolerance);
    if let SceneGeometry::Parametric(_) = scene.geometry() {
        // PORT: phase 4 -- render the surface's depth and coordinates
        // (`SurfaceImage::rasterize`, stitched over the tiles as the depth
        // is) and pass the image to `visible_layers`.
        return Err(BakeError::Port(PortRefusal {
            phase: 4,
            what: "a parametric surface's bake, which reads the surface's coordinate image from the depth pass",
        }));
    }
    if options.outline.is_some() {
        // PORT: phase 4 -- render the depth buffer (the CPU rasterizer that
        // comes with `SurfaceImage`) and add `DepthImage::outline` of it as
        // a last layer, `Style { color: "#000000", width }`.
        return Err(BakeError::Port(PortRefusal {
            phase: 4,
            what: "the outline, which is traced from the depth buffer the port does not render yet",
        }));
    }
    let visible = scene.visible_layers(Some(depth_tolerance), None)?;
    let layers = visible.iter().map(|(layer, seen)| (Style::of(&layer.spec), HiddenLine::pass(seen))).collect();
    Ok(Bake { strokes: Strokes { layers }, specs: visible.into_iter().map(|(layer, _)| layer.spec).collect() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_fit_the_texture_limit() {
        assert_eq!(BakeOptions::new(4000).tile_count(4000), 1);
        assert_eq!(BakeOptions::new(16384).tile_count(16384), 1);
        assert_eq!(BakeOptions::new(16385).tile_count(16385), 2);
        assert_eq!(BakeOptions::new(20000).tile_count(20000), 2);
        let split = BakeOptions { tiles: Some(0), ..BakeOptions::new(4000) };
        assert_eq!(split.tile_count(4000), 1);
        let split = BakeOptions { tiles: Some(5), ..BakeOptions::new(4000) };
        assert_eq!(split.tile_count(4000), 5);
    }

    #[test]
    fn the_messages_are_swifts() {
        assert_eq!(BakeError::EmptyScene.to_string(), "bake: the scene projects to nothing (no geometry, no strokes)");
        // Bake.swift's literal, its lines as Swift joins them.
        let gap = " ".repeat(13);
        let swift = format!(
            "bake: this camera is perspective, and only orthographic cameras bake.\nA bake clips per vertex against a \
             depth buffer it indexes by an{gap}affine map from view coordinates to pixels; under perspective \
             that{gap}map depends on depth. It could be fixed by projecting first, and{gap}nothing has asked for \
             it: perspective is for navigating, and the{gap}plates are orthographic."
        );
        assert_eq!(BakeError::Perspective.to_string(), swift);
    }
}
