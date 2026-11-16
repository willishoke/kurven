//! The scene: what the depth pass draws and the curves on it. The port of
//! the types of `Sources/KurvenCore/Scene.swift`, and of the few methods the
//! faces, the surface and the bundle reach.
//!
//! Ported here: `ContentID`, `Heightfield` (its fields, its initializer,
//! `forked`, `onGrid`, `forEachSample`, `normal`, `graphNormal`,
//! `surfaceHeight`, `drawnHeight`), `SceneGeometry`, `Scene` (its fields, its
//! four initializers, `heightfield`, `parametric`, `solid`, `looking`,
//! `drawing`, `creases`), `PreviewMode` and `Lighting`. The fold curves, the
//! visibility of curves and layers, the grid traversal, the fold placer, the
//! bounds and `parallelMap` are the next wave's.

use std::sync::atomic::{AtomicU64, Ordering};

use kurven_math::swift;

use crate::bundle::KurvenBundle;
use crate::camera::Camera;
use crate::manifest::{Affine2, CameraPreset, LayerRole, LayerSource, Region, Walls};
use crate::mesh::Mesh;
use crate::parametric::ParametricSurface;
use crate::polyline::Layer;
use crate::simd::Vec3;
use crate::solid::Solid;
use crate::spaces::{DomainSpace, P2, P3, ParamSpace, WorldSpace};
use crate::surface::{ContourRefine, Surface};

/// The identity of a piece of content.
///
/// GPU resources are a memo keyed by *what* is being drawn, and "what" cannot be
/// a hash: hashing a hundred megabytes of heightfield every frame costs more
/// than re-uploading it. So identity is assigned once, when the content is
/// built, and carried. Two scenes with the same `ContentID` are the same
/// geometry seen from possibly different places. Swift mints a UUID; the
/// port counts, which is unique within the process, all a memo needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentID {
    raw: u64,
}

impl ContentID {
    #[allow(clippy::new_without_default)] // a new identity each time, as Swift's `init()`
    pub fn new() -> ContentID {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        ContentID { raw: NEXT.fetch_add(1, Ordering::Relaxed) }
    }
}

/// A heightfield over the complex plane, as the plates draw it: the surface,
/// instanced once per tile, cut to a region, with its wall curtains.
#[derive(Debug, Clone)]
pub struct Heightfield {
    pub surface: Surface,
    /// Wall curtains, and the truncation edges where the heightfield's own triangles
    /// cut the corner (`Mesh::truncation_edge_mesh`), in world coordinates.
    pub occluder: Mesh<WorldSpace>,
    /// Heightfield instances; always at least the identity.
    pub tiles: Vec<Affine2>,
    /// The footprint each instance is clipped to.
    pub region: Region,
    /// Subsampling step for the heightfield when it is rasterized.
    pub step: usize,
    /// The function behind the grids, when a consumer attached one to the
    /// bundle: the fold curves are carried to its zeros.
    pub refine: Option<ContourRefine>,
    /// The walls as the manifest gives them: a perimeter they stand on, which
    /// is their faces (`faces`), or a mesh, already in `occluder`. Strokes on
    /// a wall are tested by the facing of the wall they stand on only where
    /// the walls are a perimeter; given none or a mesh, by the traversal
    /// alone, and a wall facing away shows its hatching.
    pub walls: Walls,
}

impl Heightfield {
    /// `walls` has no default, so that a heightfield built in code says what
    /// its wall strokes are tested by (`walls`). Swift's defaults are a full
    /// region and no function.
    pub fn new(
        surface: Surface,
        occluder: Mesh<WorldSpace>,
        tiles: Vec<Affine2>,
        region: Region,
        step: usize,
        refine: Option<ContourRefine>,
        walls: Walls,
    ) -> Heightfield {
        Heightfield { surface, occluder, tiles, region, step, refine, walls }
    }

    /// The same heightfield for one thread of a parallel visibility test: its
    /// function forked (`ContourRefine::fork`), its grids shared.
    pub fn forked(&self) -> Heightfield {
        Heightfield { refine: self.refine.as_ref().map(ContourRefine::forked), ..self.clone() }
    }

    /// The same heightfield without its function: traced and tested on the
    /// grid, whose chords are a height read each where f is an
    /// evaluation. What a preview can afford for a camera that is still
    /// moving.
    pub fn on_grid(&self) -> Heightfield {
        Heightfield { refine: None, ..self.clone() }
    }

    /// Every vertex of the decimated, truncated heightfield, once per tile.
    ///
    /// The grid is `Surface.grid_mesh`'s `self.clamped[::step, ::step]`. Not
    /// materialized: elliptic's is three million points, zeta's six, and the
    /// caller only ever folds over them.
    pub fn for_each_sample(&self, mut body: impl FnMut(P3<WorldSpace>)) {
        for tile in &self.tiles {
            self.surface.for_each_sample(self.step, |p| body(tile.apply(p)));
        }
    }

    /// The outward normal of the truncated surface over a world point,
    /// through the heightfield's tiles and its function when it has one
    /// (`Surface::normal`).
    ///
    /// The solid is what lies under the graph, so outward is up, and a
    /// line of sight that reaches a point of the surface facing away from the
    /// eye has entered the solid already: such a point is hidden, whatever
    /// the depth buffer says to within its depth tolerance. That is the test the
    /// depth tolerance cannot make -- where a steep graph turns away, the back of it
    /// lies within the depth tolerance of the front for a stretch that zoom
    /// magnifies, and back-face curves show there as ticks.
    pub fn normal(&self, p: P2<WorldSpace>) -> Vec3 {
        self.surface.normal(p, &self.tiles, self.refine.as_ref())
    }

    /// The normal of the graph over `p`, untruncated where the heightfield
    /// has its function (`Surface::graph_normal`): what the fold curves are
    /// traced by, and the graph face's.
    pub fn graph_normal(&self, p: P2<WorldSpace>) -> Vec3 {
        self.surface.graph_normal(p, &self.tiles, self.refine.as_ref())
    }

    /// The height of the surface the curves are tested against, over a world
    /// point: f's own, truncated, read through the tile the point is in, when
    /// the function is at hand; the grid's interpolated height otherwise.
    pub fn surface_height(&self, p: P2<WorldSpace>) -> f64 {
        if let Some(magnitude) = self.refine.as_ref().and_then(|r| r.magnitude.as_ref()) {
            let q = self.surface.in_tile(P2::<DomainSpace>::new(p.x(), p.y()), &self.tiles);
            let u = magnitude(q);
            if u.is_finite() {
                return swift::min(u, self.surface.truncation_height_at(q.x()));
            }
        }
        self.surface.height_at_tiles(P2::new(p.x(), p.y()), &self.tiles)
    }

    /// The height of the surface *as the depth pass draws it* over a world
    /// point: the grid decimated by `step`, each cell as the two triangles
    /// the rasterizer splits it into, truncated, through the tile the point is
    /// in. This is the solid every curve is tested against, so the
    /// test and the rasterizer cannot disagree about where the surface is.
    pub fn drawn_height(&self, p: P2<WorldSpace>) -> f64 {
        self.surface.drawn_height(self.surface.in_tile(P2::new(p.x(), p.y()), &self.tiles), self.step)
    }
}

/// What the depth pass draws.
///
/// A sum rather than a surface with optional parts: a heightfield has tiles,
/// a region, walls and a truncation, and a parametric surface has none of them --
/// what it has instead is a coordinate at every point, which is what its curves
/// is tested by. Each renderer, bound and bake says what it does with each.
#[derive(Debug, Clone)]
pub enum SceneGeometry {
    Heightfield(Heightfield),
    Parametric(ParametricSurface),
    /// A polyhedron, whose curves are its edges (`Solid`).
    Solid(Solid),
}

/// What a preview draws.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PreviewMode {
    /// The drawing: white occluding surface, black hidden-line curves.
    Drawing,
    /// A lit heightfield, for orientation.
    Shaded(Lighting),
    /// The depth attachment itself -- the substitute for a frame-capture
    /// viewer, and the reason not having Xcode costs nothing here.
    Depth,
}

/// A light for the shaded preview, in single precision as the GPU takes it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lighting {
    pub direction: [f32; 3],
    pub ambient: f32,
}

impl Lighting {
    /// The direction is normalized, as Swift's `simd_normalize` does.
    pub fn new(direction: [f32; 3], ambient: f32) -> Lighting {
        let [x, y, z] = direction;
        let length = (x * x + y * y + z * z).sqrt();
        Lighting { direction: [x / length, y / length, z / length], ambient }
    }
}

impl Default for Lighting {
    /// Swift's defaults: toward (0.4, -0.6, 0.7), ambient 0.25.
    fn default() -> Lighting {
        Lighting::new([0.4, -0.6, 0.7], 0.25)
    }
}

/// An immutable snapshot of everything a frame needs.
///
/// The renderer's signature is `renderDepth(scene:frame:)`: a frame is a
/// function of a value, not of accumulated state. Moving the camera means
/// producing a `Scene` that differs in one field, and the GPU buffers behind it
/// are a memo keyed by `content`.
///
/// The geometry is private to construction and the camera is a public
/// field, deliberately. If the surface could be reassigned in place,
/// `content` would go stale and the memo would serve the wrong texture -- so
/// the type makes the memo's premise true rather than documenting it.
/// Producing different geometry means producing a new `Scene`, which mints a
/// new `ContentID`.
#[derive(Debug, Clone)]
pub struct Scene {
    /// Identifies the *geometry*: everything the depth pass draws.
    content: ContentID,
    /// Identifies the *curves*. Separate from `content` because editing a level
    /// set changes every stroke and none of the landscape, and rebuilding a
    /// hundred-megabyte height texture to move a contour would make the slider
    /// unusable on exactly the plates where it is most interesting.
    curves: ContentID,
    geometry: SceneGeometry,
    layers: Vec<Layer>,
    pub camera: Camera,
    pub mode: PreviewMode,
    /// Hidden-line depth tolerance, in world units of depth, for curves tested by depth.
    pub depth_tolerance: f64,
    /// How finely the ends of tested curves are placed, as a length in the view
    /// plane: a window sets a fraction of its pixel, since curves placed more
    /// finely than a pixel looks the same and costs a traversal per halving.
    /// Zero, the default and a bake's, places them to the last bit.
    pub resolution: f64,
}

impl Scene {
    fn assemble(
        content: ContentID,
        curves: ContentID,
        geometry: SceneGeometry,
        layers: Vec<Layer>,
        camera: Camera,
        mode: PreviewMode,
        depth_tolerance: f64,
    ) -> Scene {
        Scene { content, curves, geometry, layers, camera, mode, depth_tolerance, resolution: 0.0 }
    }

    /// A heightfield scene. `walls` has no default: the scene's wall strokes
    /// are tested by the walls it is given (`Heightfield::walls`).
    #[allow(clippy::too_many_arguments)] // Swift's initializer, argument for argument
    pub fn heightfield_scene(
        surface: Surface,
        occluder: Mesh<WorldSpace>,
        tiles: Vec<Affine2>,
        region: Region,
        step: usize,
        layers: Vec<Layer>,
        camera: Camera,
        mode: PreviewMode,
        depth_tolerance: f64,
        refine: Option<ContourRefine>,
        walls: Walls,
    ) -> Scene {
        Scene::assemble(
            ContentID::new(),
            ContentID::new(),
            SceneGeometry::Heightfield(Heightfield::new(surface, occluder, tiles, region, step, refine, walls)),
            layers,
            camera,
            mode,
            depth_tolerance,
        )
    }

    /// A parametric surface, with curves on it.
    ///
    /// Curves that carry surface coordinates are tested by them; curves that do
    /// not falls back to depth, at `depth_tolerance`.
    pub fn parametric_scene(
        surface: ParametricSurface,
        layers: Vec<Layer>,
        camera: Camera,
        mode: PreviewMode,
        depth_tolerance: f64,
    ) -> Scene {
        Scene::assemble(
            ContentID::new(),
            ContentID::new(),
            SceneGeometry::Parametric(surface),
            layers,
            camera,
            mode,
            depth_tolerance,
        )
    }

    /// A solid, with its edges as curves. Swift's default depth tolerance is 0.
    pub fn solid_scene(
        solid: Solid,
        layers: Vec<Layer>,
        camera: Camera,
        mode: PreviewMode,
        depth_tolerance: f64,
    ) -> Scene {
        Scene::assemble(
            ContentID::new(),
            ContentID::new(),
            SceneGeometry::Solid(solid),
            layers,
            camera,
            mode,
            depth_tolerance,
        )
    }

    /// The scene a bundle describes under one of its presets.
    pub fn from_bundle(bundle: &KurvenBundle, preset: &CameraPreset) -> Scene {
        let o = &bundle.manifest.occluder;
        Scene::heightfield_scene(
            bundle.surface.clone(),
            bundle.occluder(),
            o.tiles.clone(),
            o.region.clone(),
            o.step.max(0) as usize,
            bundle.layers.clone(),
            Camera::oblique(&preset.projection),
            PreviewMode::Drawing,
            preset.depth_tolerance,
            bundle.refine.clone(),
            o.walls.clone(),
        )
    }

    pub fn content(&self) -> ContentID {
        self.content
    }

    pub fn curves(&self) -> ContentID {
        self.curves
    }

    pub fn geometry(&self) -> &SceneGeometry {
        &self.geometry
    }

    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// The heightfield, when that is what this scene draws.
    pub fn heightfield(&self) -> Option<&Heightfield> {
        match &self.geometry {
            SceneGeometry::Heightfield(h) => Some(h),
            _ => None,
        }
    }

    /// The parametric surface, when that is what this scene draws.
    pub fn parametric(&self) -> Option<&ParametricSurface> {
        match &self.geometry {
            SceneGeometry::Parametric(s) => Some(s),
            _ => None,
        }
    }

    /// The solid, when that is what this scene draws.
    pub fn solid(&self) -> Option<&Solid> {
        match &self.geometry {
            SceneGeometry::Solid(s) => Some(s),
            _ => None,
        }
    }

    /// The same content, looked at from somewhere else. Keeps both identities,
    /// so nothing at all is rebuilt -- which is the whole point of separating
    /// the camera from the content.
    pub fn looking(&self, camera: Camera) -> Scene {
        Scene { camera, ..self.clone() }
    }

    /// The same landscape, drawn with different curves. Keeps `content` and mints
    /// a new `curves`, so the heightfield stays uploaded and only the line buffer
    /// is rebuilt. (Swift's leaves `resolution` at its default, 0.)
    pub fn drawing(&self, layers: Vec<Layer>) -> Scene {
        Scene::assemble(
            self.content,
            ContentID::new(),
            self.geometry.clone(),
            layers,
            self.camera,
            self.mode,
            self.depth_tolerance,
        )
    }

    /// The creases of the surface's curves, in its coordinates: every outline
    /// that is not the folds -- a truncation edge. A fold that meets one is traced
    /// from it, where the grid may hold none of the fold.
    pub fn creases(&self) -> Vec<Vec<P2<ParamSpace>>> {
        self.layers
            .iter()
            .flat_map(|layer| -> Vec<Vec<P2<ParamSpace>>> {
                if matches!(layer.spec.source, LayerSource::FoldCurves) {
                    return Vec::new();
                }
                let (true, Some(coords)) = (layer.spec.role == LayerRole::Outline, &layer.paths.coords) else {
                    return Vec::new();
                };
                (0..layer.paths.count())
                    .map(|i| coords[layer.paths.offsets[i]..layer.paths.offsets[i + 1]].to_vec())
                    .collect()
            })
            .collect()
    }
}
