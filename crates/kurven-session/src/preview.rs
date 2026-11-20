//! The preview's engine half: the CPU side of `Sources/KurvenMetal`, which
//! the page's WebGPU renderer (`web/app/src/gpu`) draws.
//!
//! Swift's `MetalRenderer` does two things in one class: it decides what a
//! frame draws -- which strokes, laid out how, tested by what, with which
//! uniforms -- and it draws it. The first half is here, in doubles where it
//! decides anything, as the engine does everything; the page gets it as
//! values and typed arrays and only draws (HANDOFF-web-port.md, rule 1:
//! state lives in the engine). So the port of each Swift piece is:
//!
//! - `SceneResources` and `SurfaceResources` (Resources.swift, Surface.swift):
//!   [`Resources`], what the depth pass draws, keyed on `Scene.content`, so a
//!   camera change uploads nothing but uniforms.
//! - `LineGeometry.layOut` and `SurfaceStrokeGeometry.layout` (Preview.swift):
//!   [`lay_out`] and the per-layer geometry of [`Document::preview`], each
//!   under a key that says what it depends on -- the curves, and the camera
//!   for the strokes tested for it -- so the page keeps what a frame did not
//!   change, as Swift's `cachedLines`, `cachedSurfaceStrokes` and
//!   `cachedFolds` keep it.
//! - `renderPreview` and `renderParametricPreview`'s choices per layer:
//!   which shader draws it, at what depth tolerance, colour and width
//!   ([`LayerDraw`]); and the uniforms (`uniforms`, `clipMatrix`, the
//!   `KVShading` and `KVStroke` values), in [`Uniforms`].
//!
//! The values the GPU takes are single precision, as Swift's `Float(...)`
//! conversions make them; everything before the conversion is the engine's
//! doubles. An infinite depth tolerance or truncation height, which Swift
//! passes to Metal as is, is `None` here: JSON cannot write an infinity, and
//! WGSL may assume none reaches a shader, so the page writes the largest
//! float in its place, which every comparison it enters treats the same way.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use kurven_core::camera::Projection;
use kurven_core::depth::DepthFrame;
use kurven_core::depth_pass::{DrawnGrid, for_each_triangle};
use kurven_core::grid::Grid2D;
use kurven_core::hatching::FaceKind;
use kurven_core::manifest::{LayerRole, LayerSource, Region};
use kurven_core::parametric::ParametricSurface;
use kurven_core::polyline::{Layer, PolylineSet};
use kurven_core::scene::{ContentID, PreviewMode, Scene, SceneGeometry};
use kurven_core::simd::Mat4;
use kurven_core::spaces::{P2, P3, ViewSpace, WorldSpace};

use crate::document::{Document, QUICK_BOUNDS};

/// Written where the depth pass drew nothing (`MetalRenderer.emptySentinel`).
/// Metal cannot clear to minus infinity, and neither can WebGPU.
pub const EMPTY_SENTINEL: f32 = -1e30;

/// What the depth pass draws, built once per scene content: Swift's
/// `SceneResources` (a heightfield, or a solid's faces through the walls'
/// pipeline) and `SurfaceResources` (a parametric surface's grid). Nothing
/// here reads the camera.
#[derive(Debug, Clone)]
pub enum Resources {
    Heightfield {
        /// The grid the shader reads heights from (`DrawnGrid`), row major:
        /// the bundle's |f|, shared, not copied, or `clamped()` under bands,
        /// whose truncation depends on x and is folded in (`truncation_baked`).
        heights: Arc<Grid2D<f32>>,
        grid_width: usize,
        grid_height: usize,
        truncation_baked: bool,
        /// The subsampling step the grid is drawn at, at least one
        /// (`DrawnGrid::step`, which says why).
        step: usize,
        /// The decimated grid, `Grid2D::decimated_extent` by the step:
        /// dimensions and domain only.
        decimated_width: usize,
        decimated_height: usize,
        domain_lo: [f64; 2],
        domain_size: [f64; 2],
        /// `KVTile`: `(a, b, c, d)` and `(tx, ty)` of each tile.
        tiles: Vec<[f64; 6]>,
        /// The region's corners, each edge's start (`inside_region`); empty
        /// for the full plane.
        region: Vec<[f64; 2]>,
        /// The occluder's triangles, three vertices each.
        walls: Vec<[f64; 3]>,
    },
    /// A solid's pieces fanned into triangles (`Solid.mesh`), drawn by the
    /// walls' pipeline; no heightfield.
    Solid { walls: Vec<[f64; 3]> },
    /// `KVSurface` and the position texture: the grid, `u` fastest.
    Parametric { positions: Vec<[f64; 3]>, samples: [usize; 2], cells: [usize; 2], lo: [f64; 2], spacing: [f64; 2] },
}

impl Resources {
    /// `SceneResources.init` / `SurfaceResources.init` for a scene.
    pub fn of(scene: &Scene) -> Resources {
        match scene.geometry() {
            SceneGeometry::Heightfield(h) => {
                let drawn = DrawnGrid::of(h);
                let (decimated_width, decimated_height, d) = drawn.grid.decimated_extent(drawn.step);
                let region = match &h.region {
                    Region::Inside(p) => p.edges().iter().map(|e| [e.start.x(), e.start.y()]).collect(),
                    Region::Full => Vec::new(),
                };
                Resources::Heightfield {
                    grid_width: drawn.grid.width,
                    grid_height: drawn.grid.height,
                    heights: drawn.grid,
                    truncation_baked: drawn.truncation_baked,
                    step: drawn.step,
                    decimated_width,
                    decimated_height,
                    domain_lo: [d.real.lo, d.imag.lo],
                    domain_size: [d.real.length(), d.imag.length()],
                    tiles: h.tiles.iter().map(|t| [t.a, t.b, t.c, t.d, t.tx, t.ty]).collect(),
                    region,
                    walls: flattened(&h.occluder.vertices, &h.occluder.triangles),
                }
            }
            SceneGeometry::Solid(solid) => {
                let mesh = solid.mesh();
                Resources::Solid { walls: flattened(&mesh.vertices, &mesh.triangles) }
            }
            SceneGeometry::Parametric(s) => Resources::Parametric {
                positions: s.positions.iter().map(|p| [p.x(), p.y(), p.z()]).collect(),
                samples: [s.u.samples, s.v.samples],
                cells: [s.u.cells(), s.v.cells()],
                lo: [s.u.range.lo, s.v.range.lo],
                spacing: [s.u.spacing(), s.v.spacing()],
            },
        }
    }
}

/// A mesh's triangles as a flat vertex list, three to a triangle
/// (`SceneResources.flattened`). A triangle naming a vertex the mesh does
/// not have is left out, where Swift's index traps.
fn flattened(vertices: &[P3<WorldSpace>], triangles: &[[i32; 3]]) -> Vec<[f64; 3]> {
    let mut out = Vec::with_capacity(3 * triangles.len());
    for_each_triangle(vertices, triangles, |a, b, c| out.extend([a, b, c].map(|p| [p.x(), p.y(), p.z()])));
    out
}

/// A frame's uniforms: `KVUniforms`, the frame-wide part of `KVShading`
/// and `KVStroke`, and what a parametric surface's passes add.
#[derive(Debug, Clone, PartialEq)]
pub struct Uniforms {
    /// World to view, column by column (`Transform.float4x4`).
    pub view: [[f64; 4]; 4],
    /// View to clip, column by column (`MetalRenderer.clipMatrix`).
    pub clip: [[f64; 4]; 4],
    /// The heightfield's truncation for the shader to apply, `None` when
    /// none is (`truncationBaked`, no truncation, not a heightfield): Swift's
    /// infinity.
    pub truncation_height: Option<f64>,
    /// The pixel grid the frame is drawn on (`Framing.frame`): what maps a
    /// view point, and so a judged stroke, to its pixel.
    pub frame: DepthFrame,
    pub depth_tolerance: f64,
    pub light_direction: [f32; 3],
    pub ambient: f32,
    /// The depth view's range, from `quickBounds`, `(0, 1)` but in the depth
    /// mode, where Swift computes it alone.
    pub depth_range: [f64; 2],
    /// A parametric surface's coordinate pass's depth range, padded.
    pub surface_depth_range: Option<[f64; 2]>,
    /// `KVStroke`: the projection direction, the eye, and the projection.
    pub sight: [f64; 3],
    pub eye: [f64; 3],
    pub perspective: bool,
}

/// Which of Swift's stroke pipelines draws a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pipeline {
    /// `kv_stroke_vertex` / `kv_stroke_fragment`: tested by depth, and by the
    /// facing of a heightfield's graph where a vertex carries its normal.
    Stroke,
    /// `kv_surface_stroke_vertex` / `kv_surface_stroke_fragment`: tested by
    /// the coordinate image, and by facing unless on the fold curves.
    Surface { on_folds: bool },
}

/// One layer of a frame: how it is drawn, and its geometry's key.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerDraw {
    pub name: String,
    /// What the geometry depends on: the same key is the same segments, so
    /// the page draws the buffer it holds under it.
    pub key: String,
    pub visible: bool,
    pub pipeline: Pipeline,
    /// `MetalRenderer.color`, `#rrggbb` as four floats.
    pub color: [f32; 4],
    /// The layer's width over the widest layer's, which the page multiplies
    /// by its stroke width in pixels.
    pub width: f64,
    /// The depth tolerance this layer is tested at; `None` is Swift's
    /// infinity, strokes drawn as they come.
    pub depth_tolerance: Option<f64>,
    /// Why the scene could not test the layer, which is then drawn as
    /// carried and depth-tested; `None` when it could.
    pub refusal: Option<String>,
    /// The segments, when the page does not hold the key and the layer is
    /// drawn.
    pub segments: Option<Segments>,
}

/// A layer's segments end to end, two vertices a segment: `KVLineVertex`
/// (position, normal) or `KVStrokeVertex` (position, normal, coordinate),
/// single precision.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Segments {
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    /// Empty but for the surface pipeline.
    pub coords: Vec<f32>,
}

impl Segments {
    pub fn count(&self) -> usize {
        self.positions.len() / 6
    }
}

/// A frame of the preview: what the page draws, and under what keys.
#[derive(Debug, Clone)]
pub struct Preview {
    /// `Scene.content`: the resources' key.
    pub content: String,
    /// The resources, when the page does not hold `content`.
    pub resources: Option<Resources>,
    pub uniforms: Uniforms,
    pub mode: PreviewMode,
    pub layers: Vec<LayerDraw>,
}

/// The engine's memo for the preview: the grid normals of the surface
/// shown, which every layout of its strokes reads (`MetalRenderer.cachedNormals`).
#[derive(Default)]
pub struct Memo {
    normals: Option<(ContentID, Arc<Vec<[f32; 3]>>)>,
    /// The document's fingerprints (`pool.rs`), the shown scene's and the
    /// bake's, each with what it was hashed for: the scene's content and
    /// curves, its camera, depth tolerance and resolution, and the bake's
    /// resolution. A pool asks for them every rest and every piece.
    pub(crate) fingerprints: std::sync::Mutex<[Option<Fingerprinted>; 2]>,
    /// The content the refusals below are of, and the layers by key that the
    /// scene could not test: kept, so a frame that sends no segments for a
    /// held key still depth-tests them.
    content: String,
    refused: BTreeMap<String, String>,
}

impl Memo {
    fn forget_unless(&mut self, content: &str) {
        if self.content != content {
            self.content = content.to_string();
            self.refused.clear();
        }
    }

    fn grid_normals(&mut self, scene: &Scene, surface: &ParametricSurface) -> Arc<Vec<[f32; 3]>> {
        if let Some((content, normals)) = &self.normals
            && *content == scene.content()
        {
            return normals.clone();
        }
        let normals = Arc::new(surface.grid_normals());
        self.normals = Some((scene.content(), normals.clone()));
        normals
    }
}

/// A fingerprint, and the words that say what it was hashed for.
pub(crate) type Fingerprinted = (Vec<u64>, String);

/// `LineGeometry.isTestedExactly`: whether a layer's line geometry is the
/// camera's -- a heightfield's fold curves, or the strokes in its walls; or,
/// on a parametric surface that carries its solid, its fold curves and every
/// clipped layer that knows where on the surface it lies; or every clipped
/// layer of a solid. Sphere tracing and a solid's occlusion sight along
/// parallel lines, so under perspective their strokes stay with the stroke
/// shader.
pub fn is_tested_exactly(layer: &Layer, scene: &Scene) -> bool {
    let folds = matches!(layer.spec.source, LayerSource::FoldCurves);
    match scene.geometry() {
        SceneGeometry::Solid(_) => layer.spec.clipped && !scene.camera.is_perspective(),
        SceneGeometry::Parametric(s) => {
            if s.neighbourhood.is_none() || scene.camera.is_perspective() {
                return false;
            }
            folds || (layer.spec.clipped && layer.paths.coords.is_some())
        }
        SceneGeometry::Heightfield(_) => {
            folds || (scene.geometry().face_of(&layer.spec) == Some(FaceKind::Wall) && layer.spec.clipped)
        }
    }
}

/// Whether any layer of the scene is tested for the camera.
pub fn scene_is_tested_exactly(scene: &Scene) -> bool {
    scene.layers().iter().any(|l| is_tested_exactly(l, scene))
}

/// `LineGeometry.layOut`'s strokes, a layer at a time: the paths the stroke
/// pipeline draws for a layer, in world space.
///
/// A heightfield's fold curves are the camera's, traced here for this one --
/// on f and refined onto the exact fold, as the bake's are -- and tested by
/// the solid here rather than per fragment: the surface is edge-on at a
/// fold, which is where a depth test is worth nothing. Wall strokes likewise:
/// tested by the wall it lies in and the solid. Both are drawn unclipped.
/// A `draft` is what a frame can afford while the camera moves: the folds
/// traced on the grid, not on f, and left to the depth test with the wall
/// strokes, which is drawn as stored. A sphere-traced surface's strokes at rest
/// are the bake's, and empty in a draft, which draws them from the surface
/// strokes; a solid's edges and hatching at rest are the bake's, cut in closed
/// form, and in a draft left, as carried, to the depth test.
pub fn lay_out(layer: &Layer, scene: &Scene, draft: bool) -> Laid {
    let tested = is_tested_exactly(layer, scene);
    let view = scene.camera.view;
    let laid = |paths| Laid { paths, refusal: None };
    match scene.geometry() {
        SceneGeometry::Parametric(_) if tested => {
            // Per fragment against the coordinate image, the folds and the
            // strokes running along them drew as dashes wherever the
            // front-most coordinate at a pixel flipped between the near
            // sheet and the one behind it, which at a fold is everywhere.
            if draft { laid(PolylineSet::new(Vec::new(), vec![0], None)) } else { world(scene, layer) }
        }
        // Every width of a hatching is drawn at the layer's.
        SceneGeometry::Solid(_) if tested && !draft => world(scene, layer),
        SceneGeometry::Heightfield(h) if matches!(layer.spec.source, LayerSource::FoldCurves) => {
            // Traced on the grid and left there, a vertex lies up to a cell
            // from the outline, which on a steep graph is far more than the
            // traversal's depth tolerance on f, so vertices fell alternately
            // in front of the graph and behind it and the fold drew as
            // dashes: at rest every vertex is moved onto the exact fold.
            laid(if draft {
                h.on_grid().fold_curves(&view, false)
            } else {
                h.visible_folds(&view, scene.depth_tolerance, true, scene.resolution)
            })
        }
        SceneGeometry::Heightfield(h)
            if tested && !draft && scene.geometry().face_of(&layer.spec) == Some(FaceKind::Wall) =>
        {
            laid(h.visible_wall_curves(&layer.paths, &view, scene.depth_tolerance, scene.resolution))
        }
        _ => laid(layer.paths.clone()),
    }
}

/// A layer laid out: its strokes, and why the scene could not test them when
/// it could not (a port refusal), in which case the strokes are the layer's
/// as carried and must be left to the depth test, not drawn as they come.
pub struct Laid {
    pub paths: PolylineSet<WorldSpace>,
    pub refusal: Option<String>,
}

/// A layer as the bake tests it (`Scene.visible`), every width of it, taken
/// back to world space for the stroke shader: `LineGeometry.solidTested` and
/// `sphereTraced`. A layer the scene cannot test (a port refusal) comes back
/// as carried, with the refusal, for the caller to depth-test.
fn world(scene: &Scene, layer: &Layer) -> Laid {
    let seen = match scene.visible(layer, None, None) {
        Ok(seen) => seen,
        Err(refusal) => return Laid { paths: layer.paths.clone(), refusal: Some(refusal.to_string()) },
    };
    let back = scene.camera.view.inverse();
    let mut vertices = Vec::new();
    let mut offsets = vec![0];
    for (_, paths) in seen {
        let base = vertices.len();
        vertices.extend(paths.vertices.iter().map(|&p: &P3<ViewSpace>| back.apply(p)));
        offsets.extend(paths.offsets.iter().skip(1).map(|&o| o + base));
    }
    Laid { paths: PolylineSet::new(vertices, offsets, None), refusal: None }
}

/// `MetalRenderer.color`: `#rrggbb` or `#rgb` to four floats, black for
/// anything else.
pub fn color(hex: &str) -> [f32; 4] {
    let v = hex.strip_prefix('#').unwrap_or(hex);
    let v: String = if v.chars().count() == 3 { v.chars().flat_map(|c| [c, c]).collect() } else { v.to_string() };
    match (v.len(), u32::from_str_radix(&v, 16)) {
        (6, Ok(n)) if v.chars().all(|c| c.is_ascii_hexdigit()) => {
            [((n >> 16) & 0xff) as f32 / 255.0, ((n >> 8) & 0xff) as f32 / 255.0, (n & 0xff) as f32 / 255.0, 1.0]
        }
        _ => [0.0, 0.0, 0.0, 1.0],
    }
}

/// FNV-1a over words: the one hash the keys use.
pub(crate) fn fnv(words: impl IntoIterator<Item = u64>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in words {
        for b in x.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// The view matrix's bits.
fn view_words(scene: &Scene) -> impl Iterator<Item = u64> + '_ {
    scene.camera.view.m.columns.into_iter().flat_map(|c| c.to_array()).map(f64::to_bits)
}

/// What a layout for the camera depends on, as Swift's `cachedLines` keys
/// it: the camera (`Camera` equality: its view and its projection) and
/// whether a draft. Not the resolution: a zoom moves the framing, not the
/// camera, and a draft (Swift's resolution infinite) and a sphere-traced or
/// a solid's layout (Swift's 0) serve every zoom. A landscape's tested
/// layout at rest, placed to the scene's resolution, adds that
/// (`resolution_key`).
fn camera_key(scene: &Scene, draft: bool) -> String {
    let projection = match scene.camera.projection {
        Projection::Orthographic { .. } => 0,
        Projection::Perspective { fov_y } => fov_y.radians.to_bits() | 1,
    };
    format!("{:016x}", fnv(view_words(scene).chain([projection, u64::from(draft)])))
}

/// The key of a landscape's tested layout at rest: `base` at the scene's
/// resolution, or a held one placed at least as finely, which serves it
/// (`VisibleLines.serves`: a layout serves a scene that asks no finer, so
/// zooming out keeps it and zooming in tests again).
fn resolution_key(base: &str, resolution: f64, have: &BTreeSet<String>) -> String {
    let prefix = format!("{base}@");
    have.range(prefix.clone()..)
        .take_while(|k| k.starts_with(&prefix))
        .find(|k| u64::from_str_radix(&k[prefix.len()..], 16).is_ok_and(|bits| f64::from_bits(bits) <= resolution))
        .cloned()
        .unwrap_or_else(|| format!("{prefix}{:016x}", resolution.to_bits()))
}

fn columns(m: &Mat4) -> [[f64; 4]; 4] {
    m.columns.map(|c| c.to_array())
}

/// Every layer's strokes for the stroke pipeline, two vertices a segment,
/// with the outward normal of a heightfield's graph under a clipped layer's
/// vertices, so the shader can hide what faces away; zero elsewhere, which
/// asks nothing (`LineGeometry.layOut`).
fn line_segments(paths: &PolylineSet<WorldSpace>, scene: &Scene, layer: &Layer) -> Segments {
    let normals: Option<Vec<[f32; 3]>> = match scene.heightfield() {
        Some(h) if layer.spec.clipped && scene.geometry().face_of(&layer.spec) == Some(FaceKind::Graph) => Some(
            paths
                .vertices
                .iter()
                .map(|p| {
                    let n = h.normal(P2::new(p.x(), p.y()));
                    [n.x as f32, n.y as f32, n.z as f32]
                })
                .collect(),
        ),
        _ => None,
    };
    let mut s = Segments::default();
    for i in 0..paths.offsets.len().saturating_sub(1) {
        let (lo, hi) = (paths.offsets[i], paths.offsets[i + 1]);
        // Separate segments, not a strip: a strip would join the end of one
        // path to the start of the next.
        for k in lo..hi.saturating_sub(1) {
            for end in [k, k + 1] {
                let p = paths.vertices[end];
                s.positions.extend([p.x() as f32, p.y() as f32, p.z() as f32]);
                s.normals.extend(normals.as_ref().map_or([0.0; 3], |n| n[end]));
            }
        }
    }
    s
}

/// `SurfaceStrokeGeometry.layout`: a layer's strokes for the surface
/// pipeline, each end with its outward normal and its coordinate; empty for
/// strokes without coordinates. Facing is not asked of fold strokes nor of
/// a surface with no outside: a zero normal says so. The normals are the
/// grid's, interpolated, within a pixel of the map's at a drawing grid.
fn surface_segments(
    paths: &PolylineSet<WorldSpace>,
    surface: &ParametricSurface,
    folds: bool,
    grid: Option<&[[f32; 3]]>,
) -> Segments {
    let Some(coords) = &paths.coords else { return Segments::default() };
    let normal = |k: usize| -> [f32; 3] {
        match (folds, surface.outward, grid) {
            (false, Some(outward), Some(grid)) => {
                let sign = outward as f32;
                let n = surface.interpolate_normal(coords[k], grid).unwrap_or([0.0; 3]);
                [n[0] * sign, n[1] * sign, n[2] * sign]
            }
            (false, Some(outward), None) => {
                let n = (surface.map)(coords[k]).normal() * outward;
                [n.x as f32, n.y as f32, n.z as f32]
            }
            _ => [0.0; 3],
        }
    };
    let mut s = Segments::default();
    for i in 0..paths.offsets.len().saturating_sub(1) {
        let (lo, hi) = (paths.offsets[i], paths.offsets[i + 1]);
        for k in lo..hi.saturating_sub(1) {
            for end in [k, k + 1] {
                let p = paths.vertices[end];
                let c = coords[end];
                s.positions.extend([p.x() as f32, p.y() as f32, p.z() as f32]);
                s.normals.extend(normal(end));
                s.coords.extend([c.x() as f32, c.y() as f32]);
            }
        }
    }
    s
}

impl Document {
    /// A frame of the preview of the scene the window shows (`shown_scene`):
    /// a draft while the camera or a control moves, the window's layout at
    /// rest. `have_content` and `have` are the keys the page holds; what it
    /// holds is not built again.
    ///
    /// `nonce` names the engine instance -- a Worker seeds it at load -- and
    /// starts every key, so a key a page held from an engine since restarted,
    /// whose counted `ContentID`s start again at one, names nothing here.
    pub fn preview(
        &mut self,
        draft: bool,
        nonce: &str,
        have_content: Option<&str>,
        have: &BTreeSet<String>,
    ) -> Option<Preview> {
        self.preview_laying(draft, nonce, have_content, have, &mut |_, layer, scene, draft| {
            lay_out(layer, scene, draft)
        })
    }

    /// `preview`, each layer the frame sends laid out by `lay`, given the
    /// layer's place, the layer, the scene and whether a draft, in place of
    /// `lay_out`: what lets a pool of engines lay out the tested layers of a
    /// frame at rest elsewhere and this one serve them under its own keys
    /// (`pool.rs`). `lay` is asked exactly where `preview` would ask
    /// `lay_out`, so with `lay_out` it is `preview`.
    pub fn preview_laying(
        &mut self,
        draft: bool,
        nonce: &str,
        have_content: Option<&str>,
        have: &BTreeSet<String>,
        lay: &mut dyn FnMut(usize, &Layer, &Scene, bool) -> Laid,
    ) -> Option<Preview> {
        let scene = self.shown_scene()?;
        let navigator = self.navigator?;
        let visible = self.shown_visible_layers();
        let frame = navigator.framing.frame(self.viewport);
        let clip = match scene.camera.projection {
            Projection::Orthographic { .. } => frame.metal_clip(),
            Projection::Perspective { fov_y } => {
                kurven_core::camera::Camera::perspective_clip(fov_y, frame.cols as f64 / frame.rows as f64)
            }
        };
        let view = scene.camera.view;
        let parametric = scene.parametric().cloned();
        let bounds =
            (matches!(self.mode, PreviewMode::Depth) || parametric.is_some()).then(|| scene.quick_bounds(QUICK_BOUNDS));
        let depth_range = match (&self.mode, bounds.flatten()) {
            (PreviewMode::Depth, Some(b)) => [b.lo.z, b.hi.z],
            _ => [0.0, 1.0],
        };
        let surface_depth_range = parametric.as_ref().map(|_| {
            let (lo, hi) = bounds.flatten().map_or((-1.0, 1.0), |b| (b.lo.z, b.hi.z));
            let pad = 0.05 * (hi - lo) + 1e-6;
            [lo - pad, hi + pad]
        });
        let truncation_height = scene.heightfield().and_then(DrawnGrid::ceiling_of);
        let lighting = match self.mode {
            PreviewMode::Shaded(l) => l,
            // `KVShading`'s light outside the shaded mode, as Swift fills it;
            // nothing reads it there.
            _ => kurven_core::scene::Lighting { direction: [0.4, -0.6, 0.7], ambient: 0.25 },
        };
        let sight = view.projection_direction();
        let eye = view.inverse().apply(P3::<ViewSpace>::new(0.0, 0.0, 0.0));
        let uniforms = Uniforms {
            view: columns(&view.m),
            clip: columns(&clip),
            truncation_height,
            frame,
            depth_tolerance: scene.depth_tolerance,
            light_direction: lighting.direction,
            ambient: lighting.ambient,
            depth_range,
            surface_depth_range,
            sight: [sight.x, sight.y, sight.z],
            eye: [eye.x(), eye.y(), eye.z()],
            perspective: scene.camera.is_perspective(),
        };

        let content = content_key(nonce, &scene);
        self.preview_memo.forget_unless(&content);
        let resources = (have_content != Some(content.as_str())).then(|| Resources::of(&scene));
        let widest = scene.layers().iter().map(|l| l.spec.width).fold(None, |m: Option<f64>, w| {
            Some(match m {
                Some(m) => kurven_math::swift::max(m, w),
                None => w,
            })
        });
        // Every key starts with the content and the curves: a new scene is new
        // geometry, whatever its layers' places.
        let curves = curves_key(&content, &scene);
        let camera = camera_key(&scene, draft);
        // The strokes of a parametric scene's layout are the camera's only
        // if some layer is tested for it (Swift's `lineGeometry(for:draft:)`
        // asks `visible && draft`).
        let lines_draft = draft && scene_is_tested_exactly(&scene);
        let grid = match &parametric {
            Some(s)
                if s.outward.is_some()
                    && scene
                        .layers()
                        .iter()
                        .any(|l| l.paths.coords.is_some() && !matches!(l.spec.source, LayerSource::FoldCurves)) =>
            {
                Some(self.preview_memo.grid_normals(&scene, s))
            }
            _ => None,
        };
        let mut layers = Vec::with_capacity(scene.layers().len());
        for (i, layer) in scene.layers().iter().enumerate() {
            let tested = is_tested_exactly(layer, &scene);
            let folds = matches!(layer.spec.source, LayerSource::FoldCurves);
            let shown = visible.contains(&i);
            let width = match widest {
                Some(w) if w > 0.0 => layer.spec.width / w,
                _ => 1.0,
            };
            let base = |key: String, pipeline: Pipeline, depth_tolerance: Option<f64>| LayerDraw {
                name: layer.spec.name.clone(),
                key,
                visible: shown,
                pipeline,
                color: color(&layer.spec.color),
                width,
                depth_tolerance,
                refusal: None,
                segments: None,
            };
            let mut draw = if let Some(s) = &parametric {
                // `renderParametricPreview`: at rest a sphere-traced surface's
                // strokes come tested by the solid, and are drawn as they
                // come; otherwise a layer with coordinates is tested per
                // fragment by the surface pipeline, and one without by depth.
                let sphere_traced = !draft && tested;
                let tolerance = (layer.spec.clipped && !sphere_traced).then_some(scene.depth_tolerance);
                let surface = layer.spec.clipped && !sphere_traced && (folds || layer.paths.coords.is_some());
                if surface {
                    let key = if folds {
                        format!("{curves}.{i}.folds.{:016x}", view_bits(&scene))
                    } else {
                        format!("{curves}.{i}.surface")
                    };
                    let mut d = base(key, Pipeline::Surface { on_folds: folds }, tolerance);
                    if shown && !have.contains(&d.key) {
                        let paths = if folds {
                            let normals = self.preview_memo.grid_normals(&scene, s);
                            s.grid_fold_curves(&normals, sight)
                                .unwrap_or_else(|_| PolylineSet::new(Vec::new(), vec![0], None))
                        } else {
                            layer.paths.clone()
                        };
                        // An outline -- a truncation edge -- lies on a crease,
                        // where the normal says nothing: laid out as fold
                        // strokes, with no facing.
                        let crease = folds || layer.spec.role == LayerRole::Outline;
                        d.segments = Some(surface_segments(&paths, s, crease, grid.as_deref().map(|g| g.as_slice())));
                    }
                    // A layer the surface pipeline has nothing of draws from
                    // the lines, as Swift's `curves.draw(i)` falls through.
                    if d.segments.as_ref().is_none_or(|g| g.count() > 0) {
                        layers.push(d);
                        continue;
                    }
                }
                let key = if tested { format!("{curves}.{i}.lines.{camera}") } else { format!("{curves}.{i}.lines") };
                base(key, Pipeline::Stroke, tolerance)
            } else {
                // `renderPreview`: unclipped strokes are drawn with an
                // unreachable depth tolerance rather than a second pipeline: a
                // cut-face hatch lies *in* the wall it hatches. A
                // heightfield's folds and wall strokes, and a solid's tested
                // strokes, were tested on the way in, unless this frame is a
                // draft, which leaves them to the test.
                let heightfield = scene.heightfield().is_some();
                let mut done = heightfield && scene.geometry().face_of(&layer.spec) == Some(FaceKind::Wall);
                if folds {
                    done = true;
                }
                if scene.solid().is_some() && tested {
                    done = true;
                }
                if draft {
                    done = false;
                }
                let tolerance = (layer.spec.clipped && !done).then_some(scene.depth_tolerance);
                // A layout is the camera's where the layer is tested for it;
                // a heightfield's draft folds are too, traced for it on the
                // grid.
                let key = if tested && !draft {
                    rest_key(&scene, &curves, i, have)
                } else if heightfield && folds {
                    format!("{curves}.{i}.lines.{camera}")
                } else {
                    format!("{curves}.{i}.lines")
                };
                base(key, Pipeline::Stroke, tolerance)
            };
            if shown && !have.contains(&draw.key) {
                let laid = lay(i, layer, &scene, if parametric.is_some() { lines_draft } else { draft });
                if let Some(refusal) = laid.refusal {
                    self.preview_memo.refused.insert(draw.key.clone(), refusal);
                }
                draw.segments = Some(line_segments(&laid.paths, &scene, layer));
            }
            // A layer the scene could not test is drawn as carried, and so is
            // left to the depth test, as a draft leaves it, with the reason.
            if let Some(refusal) = self.preview_memo.refused.get(&draw.key) {
                draw.depth_tolerance = layer.spec.clipped.then_some(scene.depth_tolerance);
                draw.refusal = Some(refusal.clone());
            }
            layers.push(draw);
        }
        Some(Preview { content, resources, uniforms, mode: self.mode, layers })
    }
}

/// The key of a scene's resources: the engine's nonce and the scene's content.
pub(crate) fn content_key(nonce: &str, scene: &Scene) -> String {
    format!("{nonce}~{}", scene.content().raw())
}

/// The key every layer's starts with: the content's and the curves'.
pub(crate) fn curves_key(content: &str, scene: &Scene) -> String {
    format!("{content}.{}", scene.curves().raw())
}

/// The key of a tested layer's layout at rest (`is_tested_exactly`, not a
/// draft): the camera's, and for a landscape the resolution its runs' ends
/// are placed to, a held one placed at least as finely serving
/// (`resolution_key`). What a frame at rest sends a layer's segments under,
/// and what `Document::rest_layers` asks the page's holdings for.
pub(crate) fn rest_key(scene: &Scene, curves: &str, i: usize, have: &BTreeSet<String>) -> String {
    let base = format!("{curves}.{i}.lines.{}", camera_key(scene, false));
    if scene.heightfield().is_some() { resolution_key(&base, scene.resolution, have) } else { base }
}

/// The view matrix's bits alone, for a parametric surface's grid folds,
/// which depend on the line of sight and nothing else of the camera
/// (`cachedFolds`, keyed by `view`).
fn view_bits(scene: &Scene) -> u64 {
    fnv(view_words(scene))
}
