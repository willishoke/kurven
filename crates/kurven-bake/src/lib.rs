//! A scene to its strokes: the port of `Sources/KurvenBake/Bake.swift`,
//! strokes only.
//!
//! The realtime preview and the bake are the same computation at different
//! resolutions. In Swift the bake renders a depth buffer with Metal, for its
//! `depth` output and the outline; no stroke of a landscape or a polyhedron
//! reads it, since every layer arrives tested by its face and the scene's
//! occlusion test (`Scene::visible_layers`). So for those the port bakes the
//! strokes and renders no buffer -- there is no PNG (the page rasterizes) --
//! and an outline, which is traced from the buffer, is a typed refusal. A
//! parametric surface's coordinate image and its depth are rendered by the
//! CPU rasterizer (`SurfaceImage::rasterize`), which a surface with no exact
//! test reads, and its outline is traced from that depth.

use std::fmt;
use std::sync::Arc;

use kurven_core::depth::{DepthFrame, DepthImage};
use kurven_core::hidden_line::HiddenLine;
use kurven_core::manifest::{LayerSource, LayerSpec};
use kurven_core::parametric::{ParametricSurface, PortRefusal};
use kurven_core::polyline::PolylineSet;
use kurven_core::scene::{Scene, SceneGeometry};
use kurven_core::spaces::{PageSpace, Transform, ViewSpace, WorldSpace};
use kurven_core::surface_image::SurfaceImage;
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
    /// one width of a solid's hatching (`Scene::visible`). The outline, when
    /// asked for, is the stroke layer past the last of these, traced from
    /// the depth buffer.
    pub specs: Vec<LayerSpec>,
    /// The frame a parametric scene's depth and coordinates are rendered on,
    /// `DepthFrame::covering` its view bounds at the resolution; None for a
    /// landscape or a solid, whose depth pass the port does not render.
    pub frame: Option<DepthFrame>,
    /// A parametric scene's front-most surface coordinates and depth,
    /// stitched from however many passes it took, where something reads
    /// them: a surface with no tubular neighbourhood, a clipped layer with
    /// no coordinates, the outline (`bake_with_image` renders it always, as
    /// Swift's bake does). Shared, not copied, with the visibility that reads
    /// it.
    pub surface: Option<Arc<SurfaceImage>>,
    /// How many passes the depth took, per axis.
    pub tiles: i64,
}

/// Bake a scene to strokes (Swift's `MetalRenderer.bake(_:options:)`).
///
/// Swift renders the depth buffer, stitched from `options.tile_count` passes,
/// and then takes every layer from `Scene.visibleLayers`, which reads the
/// buffer only as a parametric surface's coordinate image. The port renders
/// a parametric surface's image with the CPU rasterizer
/// (`SurfaceImage::rasterize`, Swift's own reference for its Metal pass), in
/// the same tiles stitched the same way, where it is read: a surface sphere
/// traced, with every clipped layer on its coordinates and no outline asked
/// for, reads none, and the port renders none, about a hundred megabytes at
/// a 3000-pixel bake, for the same strokes. For a landscape or a solid it
/// takes the layers alone, the same strokes at any resolution and split,
/// and refuses an outline, which is traced from a depth pass it does not
/// render.
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
    let depth_tolerance = options.depth_tolerance.unwrap_or(scene.depth_tolerance);
    let (frame, image) = match scene.geometry() {
        SceneGeometry::Parametric(surface) => {
            let Some(bounds) = scene.view_bounds()? else { return Err(BakeError::EmptyScene) };
            let frame = DepthFrame::covering(&bounds, options.resolution);
            // Every tile at least two pixels each way. Swift's sub-frame of
            // one row has an axis of no length, which its clip divides by,
            // and one of no rows a texture Metal cannot make; the port asks
            // for no more passes than the frame has pairs of rows and of
            // columns, which changes no stroke (the passes are stitched).
            n = n.min((frame.rows.min(frame.cols) / 2).max(1));
            let read = surface.neighbourhood.is_none()
                || options.outline.is_some()
                || scene.layers().iter().any(|l| {
                    l.spec.clipped && l.paths.coords.is_none() && !matches!(l.spec.source, LayerSource::FoldCurves)
                });
            let image = (always || read).then(|| Arc::new(render(surface, &scene.camera.view, frame, n)));
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
                // PORT: phase 7 -- render the depth pass of a landscape or a solid
                // (`MetalRenderer.renderDepth`; the WebGPU depth pass) and add
                // `DepthImage::outline` of it as a last layer.
                return Err(BakeError::Port(PortRefusal {
                    phase: 7,
                    what: "the outline of a landscape or a solid, which is traced from the depth pass the port does \
                           not render",
                }));
            }
            (None, None)
        }
    };
    // Every layer comes tested by its face and the scene's occlusion
    // test (`Scene::visible`). The buffer is rendered for the outline; the
    // drawing reads it, as the coordinate image, only for a parametric
    // surface with no exact test, and only there depends on the resolution.
    let visible = scene.visible_layers(Some(depth_tolerance), image.as_ref())?;
    let mut layers: Vec<(Style, PolylineSet<PageSpace>)> =
        visible.iter().map(|(layer, seen)| (Style::of(&layer.spec), HiddenLine::pass(seen))).collect();
    if let (Some(width), Some(image)) = (options.outline, &image) {
        layers.push((Style { color: "#000000".to_string(), width }, image.depth.outline()));
    }
    Ok(Bake {
        strokes: Strokes { layers },
        specs: visible.into_iter().map(|(layer, _)| layer.spec).collect(),
        frame,
        surface: image,
        tiles: n,
    })
}

/// A parametric surface's coordinate image over `frame`, in `n x n` passes
/// stitched into one row-major image (Swift's `stitch`), each pass copied
/// into place as it is rasterized, in the order Swift visits them.
///
/// Tiles meet on pixel boundaries -- `DepthFrame::tile` splits the grid,
/// not the coordinate range -- so a pixel lands in the same place however
/// the drawing was divided, and depth and coordinates are stitched by the
/// same code over the same sub-frames, so a coordinate sits on the pixel its
/// depth does.
fn render(
    surface: &ParametricSurface,
    view: &Transform<WorldSpace, ViewSpace>,
    frame: DepthFrame,
    n: i64,
) -> SurfaceImage {
    if n == 1 {
        return SurfaceImage::rasterize(surface, view, frame);
    }
    let count = (frame.rows * frame.cols) as usize;
    let mut values = vec![f32::NEG_INFINITY; count];
    let mut coords = vec![[f32::NAN, f32::NAN]; count];
    let mut empty = f32::NEG_INFINITY;
    for ti in 0..n {
        for tj in 0..n {
            let sub = frame.tile(ti, tj, n);
            let tile = SurfaceImage::rasterize(surface, view, sub);
            empty = tile.depth.empty;
            let (row0, col0) = (frame.rows * ti / n, frame.cols * tj / n);
            let width = sub.cols as usize;
            for r in 0..sub.rows {
                let dst = ((row0 + r) * frame.cols + col0) as usize;
                let src = (r * sub.cols) as usize;
                values[dst..dst + width].copy_from_slice(&tile.depth.values[src..src + width]);
                coords[dst..dst + width].copy_from_slice(&tile.coords[src..src + width]);
            }
        }
    }
    SurfaceImage { depth: DepthImage::new(frame, values, empty), coords }
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
