//! A scene to its strokes: the port of `Sources/KurvenBake/Bake.swift`,
//! strokes only.
//!
//! The realtime preview and the bake are the same computation at different
//! resolutions. In Swift the bake renders a depth buffer with Metal, for its
//! `depth` output and the outline; no stroke of a landscape or a polyhedron
//! reads it, since every layer arrives tested by its face and the scene's
//! occlusion test (`Scene::visible_layers`). So for those the port bakes the
//! strokes and renders the pass only for the outline, which is traced from
//! its coverage, on the CPU and a strip of rows at a time
//! (`depth_pass::coverage`) -- there is no PNG (the page rasterizes). A parametric surface's coordinate image and its depth are
//! rendered by the CPU rasterizer (`SurfaceImage::rasterize`), which a
//! surface with no exact test reads, and its outline is traced from that
//! depth.

use std::fmt;
use std::sync::Arc;

pub mod pieces;

use kurven_core::depth::DepthFrame;
use kurven_core::depth_pass;
use kurven_core::hatching::FaceSelector;
use kurven_core::hidden_line::HiddenLine;
use kurven_core::hidden_line::outline_of_coverage;
use kurven_core::manifest::{LayerSource, LayerSpec};
use kurven_core::parametric::PortRefusal;
use kurven_core::polyline::PolylineSet;
use kurven_core::scene::{Scene, SceneGeometry};
use kurven_core::spaces::PageSpace;
use kurven_core::surface_image::SurfaceImage;
use kurven_core::svg::{Strokes, Style};

/// How many rows of a landscape's or a solid's depth pass are rendered at a
/// time for its outline: 512 rows of a 20000-pixel frame hold 41 MB of depth.
pub const OUTLINE_STRIP: i64 = 512;

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
    /// buffer, which the port renders on the CPU for it.
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
    /// one width of a solid's hatching (`Scene::visible`). The outline, when
    /// asked for, is the stroke layer past the last of these, traced from
    /// the depth buffer.
    pub specs: Vec<LayerSpec>,
    /// The frame a parametric scene's depth and coordinates are rendered on,
    /// `DepthFrame::covering` its view bounds at the resolution, and a
    /// landscape's or a solid's where its depth pass is rendered for the
    /// outline; None where nothing reads a depth pass.
    pub frame: Option<DepthFrame>,
    /// Whether a landscape's or a solid's depth pass was rendered: for the
    /// outline alone, the only thing its bake reads off the pass, and so
    /// kept only as the outline's coverage, never as the frame's depth.
    pub depth_rendered: bool,
    /// A parametric scene's front-most surface coordinates and depth, in one
    /// pass over the whole frame, where something reads
    /// them: a surface with no tubular neighbourhood, a clipped layer with
    /// no coordinates, the outline (`bake_with_image` renders it always, as
    /// Swift's bake does). Shared, not copied, with the visibility that reads
    /// it.
    pub surface: Option<Arc<SurfaceImage>>,
    /// How many passes the depth took, per axis: Swift's tile count for a
    /// landscape or a solid whose depth the port does not render, one where
    /// the CPU rasterizer renders it, whole.
    pub tiles: i64,
}

/// Bake a scene to strokes (Swift's `MetalRenderer.bake(_:options:)`).
///
/// Swift renders the depth buffer, stitched from `options.tile_count` passes,
/// and then takes every layer from `Scene.visibleLayers`, which reads the
/// buffer only as a parametric surface's coordinate image. The port renders
/// a parametric surface's image with the CPU rasterizer
/// (`SurfaceImage::rasterize`, Swift's own reference for its Metal pass), in
/// one pass, which Swift's stitched tiles stand for, where it is read: a surface sphere
/// traced, with every clipped layer on its coordinates and no outline asked
/// for, reads none, and the port renders none, about a hundred megabytes at
/// a 3000-pixel bake, for the same strokes. For a landscape or a solid it
/// takes the layers alone, the same strokes at any resolution and split,
/// and renders its depth pass on the CPU, whole, only for an outline.
pub fn bake(scene: &Scene, options: &BakeOptions) -> Result<Bake, BakeError> {
    bake_rendering(scene, options, false)
}

/// `bake`, rendering a parametric scene's coordinate image whether or not
/// anything reads it, as Swift's bake always does: for what holds the image
/// itself to the drawing (its tiling, its frame).
pub fn bake_with_image(scene: &Scene, options: &BakeOptions) -> Result<Bake, BakeError> {
    bake_rendering(scene, options, true)
}

fn bake_rendering(scene: &Scene, options: &BakeOptions, always: bool) -> Result<Bake, BakeError> {
    if scene.camera.is_perspective() {
        return Err(BakeError::Perspective);
    }
    let mut n = options.tile_count(options.resolution);
    let mut coverage: Option<Vec<f32>> = None;
    let depth_tolerance = options.depth_tolerance.unwrap_or(scene.depth_tolerance);
    let (frame, image) = match scene.geometry() {
        SceneGeometry::Parametric(surface) => {
            let Some(bounds) = scene.view_bounds()? else { return Err(BakeError::EmptyScene) };
            let frame = DepthFrame::covering(&bounds, options.resolution);
            // Swift splits the pass into tiles where the frame passes
            // Metal's texture limit, and stitches them. The CPU rasterizer
            // has no limit, and renders the whole frame in one pass, which
            // is what the stitched tiles stand for: the image, and so every
            // stroke, is the same however the bake asks for it to be split.
            n = 1;
            // A layer that carries no coordinates is tested by the image's
            // depth; the folds and a surface's hatching are made with theirs
            // when they are tested.
            let read = surface.neighbourhood.is_none()
                || options.outline.is_some()
                || scene.layers().iter().any(|l| {
                    l.spec.clipped
                        && l.paths.coords.is_none()
                        && !matches!(
                            l.spec.source,
                            LayerSource::FoldCurves | LayerSource::Hatching { faces: FaceSelector::Surface, .. }
                        )
                });
            let image = (always || read).then(|| Arc::new(SurfaceImage::rasterize(surface, &scene.camera.view, frame)));
            (Some(frame), image)
        }
        _ => {
            // Swift computes the drawing's view bounds here, to frame the depth
            // pass; the port renders none, so it asks only whether there is
            // anything to frame, from the counts, without projecting every sample.
            if scene.projects_to_nothing()? {
                return Err(BakeError::EmptyScene);
            }
            if options.outline.is_some() {
                // The outline is traced from the depth pass
                // (`MetalRenderer.renderDepth`), rendered here on the CPU,
                // its coverage alone kept, and a strip of rows at a time, so
                // the depth held is a strip's however large the frame: the
                // bound Swift's tiles give Metal's pass. Every sample is the
                // whole frame's, so the strips stand for one pass.
                let Some(bounds) = scene.view_bounds()? else { return Err(BakeError::EmptyScene) };
                let frame = DepthFrame::covering(&bounds, options.resolution);
                n = 1;
                coverage = Some(depth_pass::coverage(scene, &scene.camera.view, frame, OUTLINE_STRIP));
                (Some(frame), None)
            } else {
                (None, None)
            }
        }
    };
    // Every layer comes tested by its face and the scene's occlusion
    // test (`Scene::visible`). The buffer is rendered for the outline; the
    // drawing reads it, as the coordinate image, only for a parametric
    // surface with no exact test, and only there depends on the resolution.
    let visible = scene.visible_layers(Some(depth_tolerance), image.as_ref())?;
    let mut layers: Vec<(Style, PolylineSet<PageSpace>)> =
        visible.iter().map(|(layer, seen)| (Style::of(&layer.spec), HiddenLine::pass(seen))).collect();
    if let Some(width) = options.outline {
        if let Some(image) = &image {
            layers.push((Style { color: "#000000".to_string(), width }, image.depth.outline()));
        } else if let (Some(coverage), Some(frame)) = (coverage.take(), frame) {
            layers.push((Style { color: "#000000".to_string(), width }, outline_of_coverage(frame, coverage)));
        }
    }
    Ok(Bake {
        strokes: Strokes { layers },
        specs: visible.into_iter().map(|(layer, _)| layer.spec).collect(),
        frame,
        depth_rendered: options.outline.is_some() && image.is_none() && frame.is_some(),
        surface: image,
        tiles: n,
    })
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
