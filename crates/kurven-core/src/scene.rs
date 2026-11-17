//! The scene: what the depth pass draws and the curves on it. The port of
//! `Sources/KurvenCore/Scene.swift`, but for `GridTraversal`, which is in
//! [`crate::traversal`].
//!
//! Whole for a landscape, a parametric surface and a solid: `ContentID`,
//! `Heightfield` (its fold curves, the visibility of curves on each kind of
//! face, the facing of its walls and risers, `visibleRuns` and `turning`),
//! `FoldPlacer`, `SceneGeometry`, `Scene` (its initializers,
//! `visibleLayers`, `visible`, `viewBounds`, `quickBounds`), `PreviewMode`
//! and `Lighting`. `parallelMap` is a loop in order.

use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use kurven_math::swift;

use crate::bisection::{bisect, bisect_points};
use crate::bundle::KurvenBundle;
use crate::camera::Camera;
use crate::contour;
use crate::grid::Grid2D;
use crate::hatch::HatchContext;
use crate::hatching::{FaceKind, FaceSelector, VisibleHatching};
use crate::hidden_line::{HiddenLine, fold_crossing};
use crate::manifest::{Affine2, CameraPreset, Domain, Interval, LayerRole, LayerSource, Region, Walls};
use crate::mesh::Mesh;
use crate::parametric::{ParametricSurface, PortRefusal};
use crate::polyline::{Layer, PolylineSet};
use crate::simd::{Vec2, Vec3};
use crate::solid::Solid;
use crate::spaces::{AABB, DomainSpace, P2, P3, ParamSpace, Transform, ViewSpace, WorldSpace};
use crate::surface::{ContourRefine, Surface, TileBoundary};
use crate::surface_image::SurfaceImage;
use crate::traversal::GridTraversal;
use crate::visibility::SurfaceVisibility;

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

    /// The identity as a number: what the page keys its GPU resources by,
    /// across the Worker's boundary.
    pub fn raw(&self) -> u64 {
        self.raw
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
    /// A parametric surface's fold curves for the last line of sight they
    /// were asked for, which `view_bounds` and `visible` both read: placed
    /// once per camera, not once per ask. Each scene's own -- a clone starts
    /// empty -- so it holds only this scene's surface and creases, which
    /// cannot change under it; keyed by the line of sight, since the camera
    /// is a public field. Not in Swift, which places them twice.
    fold_memo: FoldMemo,
}

/// A parametric surface's fold curves, and the line of sight they are for.
/// Cloned empty: a clone of a scene may draw other layers, and so other
/// creases, and must not read this scene's folds.
#[derive(Debug, Default)]
struct FoldMemo(Mutex<Option<(Vec3, Arc<PolylineSet<WorldSpace>>)>>);

impl Clone for FoldMemo {
    fn clone(&self) -> FoldMemo {
        FoldMemo::default()
    }
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
        Scene {
            content,
            curves,
            geometry,
            layers,
            camera,
            mode,
            depth_tolerance,
            resolution: 0.0,
            fold_memo: FoldMemo::default(),
        }
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
            bundle.layers().to_vec(),
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

// MARK: - the visibility of curves on a heightfield

/// What a worker thread tests the points of curves by: the facing of
/// the face under a point, positive toward the eye, None where no face's
/// facing has a say; whether nothing of the solid stands between it and
/// the eye; and where the curve between two vertices runs, as a point of
/// their chord is lifted onto it, for the bisection of a run's end: for
/// most curves the chord itself.
pub struct Tests {
    pub facing: Box<dyn Fn(P3<WorldSpace>) -> Option<f64>>,
    pub clear: Box<dyn Fn(P3<WorldSpace>) -> bool>,
    pub lift: Lift,
}

impl Tests {
    /// Swift's memberwise init with `lift` at its default, the chord.
    pub fn new(
        facing: impl Fn(P3<WorldSpace>) -> Option<f64> + 'static,
        clear: impl Fn(P3<WorldSpace>) -> bool + 'static,
    ) -> Tests {
        Tests { facing: Box::new(facing), clear: Box::new(clear), lift: Box::new(|_, _, q| q) }
    }
}

/// Where the curve between two vertices runs, as a point of their chord is
/// lifted onto it (`Tests::lift`).
pub type Lift = Box<dyn Fn(P3<WorldSpace>, P3<WorldSpace>, P3<WorldSpace>) -> P3<WorldSpace>>;

/// The fraction of a segment a run's end is bisected to (`Heightfield::stop`).
pub type Stop = Box<dyn Fn(P3<WorldSpace>, P3<WorldSpace>) -> f64>;

/// A face's facing as a function of a point (`wall_facing`, `riser_facing`).
pub type FacingTest = Rc<dyn Fn(P3<WorldSpace>) -> Option<f64>>;

/// `items.map(body)`: Swift's `parallelMap(_:chunk:_:)`, a loop in order
/// here. Each item is independent, so the answer is the one the chunks give.
pub fn parallel_map<T, R>(items: &[T], _chunk: usize, body: impl FnMut(&T) -> R) -> Vec<R> {
    items.iter().map(body).collect()
}

/// `items.map(body)` with a worker's state, made by `worker` once, as a
/// thread makes it once in Swift's `parallelMap(_:chunk:worker:_:)`. A
/// forked worker answers as the one it was forked from, so the answer is
/// the one any split of the items gives.
pub fn parallel_map_with<T, R, W>(
    items: &[T],
    _chunk: usize,
    worker: impl FnOnce() -> W,
    mut body: impl FnMut(&W, &T) -> R,
) -> Vec<R> {
    if items.is_empty() {
        return Vec::new();
    }
    let state = worker();
    items.iter().map(|t| body(&state, t)).collect()
}

/// The point `t` of the way from `a` to `b`, as Swift's `a.v + (b.v - a.v) * t`.
fn along(a: P3<WorldSpace>, b: P3<WorldSpace>, t: f64) -> P3<WorldSpace> {
    P3::from_vec(a.v + (b.v - a.v) * t)
}

/// Swift's `.min()!` of a sequence of doubles: the first, then each that
/// is less.
fn sequence_min(xs: &[f64]) -> f64 {
    xs[1..].iter().fold(xs[0], |m, &x| if x < m { x } else { m })
}

/// Swift's `.max()!` of a sequence of doubles: the first, then each that
/// is greater.
fn sequence_max(xs: &[f64]) -> f64 {
    xs[1..].iter().fold(xs[0], |m, &x| if m < x { x } else { m })
}

impl Heightfield {
    /// The fold curves under a camera: where the surface turns edge-on,
    /// `n · sight = 0`, which is where its outline and every inner outline
    /// lie -- the far edge of a pit, the top edge of a graph seen from the side.
    /// The 1933 plates draw them, and the notebooks drew them from a
    /// thresholded matrix of gradients, contoured. This is that, with the
    /// line of sight as the threshold: the facing is sampled on the grid,
    /// its zero level traced by marching squares, and, when `refined`, every
    /// vertex moved onto the exact fold by bisection along the facing's
    /// gradient -- the same facing `HeightfieldVisibility` cuts curves by, so
    /// the two agree on where the surface turns. Nothing on a plateau, which is
    /// flat and faces up and has a truncation edge of its own, and nothing outside the
    /// region. They depend on the camera, so they are derived for one, never
    /// stored.
    ///
    /// A fold that ends beside a zero of f was going there: the outline
    /// of a pit runs down its graph into the zero, the apex of the cone.
    /// The grid loses it a cell or two short, where its chord across the
    /// floor flattens the facing out. With the function at hand (`refine`),
    /// an open run's end is carried to the zero the refiner finds from it,
    /// at |f|'s own height there -- the floor -- as the phase lines are.
    /// (Swift's `refined` defaults to true.)
    pub fn fold_curves(&self, view: &Transform<WorldSpace, ViewSpace>, refined: bool) -> PolylineSet<WorldSpace> {
        let g = &*self.surface.height;
        let (nx, ny) = (g.width, g.height);
        if nx < 2 || ny < 2 {
            return PolylineSet::empty();
        }
        let cell =
            swift::max(g.domain.real.length().abs() / (nx - 1) as f64, g.domain.imag.length().abs() / (ny - 1) as f64);
        let sight = view.projection_direction();
        let mut paths: Vec<Vec<P3<WorldSpace>>> = Vec::new();
        let d = g.domain;
        for tile in &self.tiles {
            // The tile's image of the domain, as a box: the tiles are
            // reflections and translations, so it is one.
            let corners =
                [(d.real.lo, d.imag.lo), (d.real.hi, d.imag.lo), (d.real.lo, d.imag.hi), (d.real.hi, d.imag.hi)]
                    .map(|(x, y)| tile.apply(P3::new(x, y, 0.0)));
            let xs = corners.map(|c| c.x());
            let ys = corners.map(|c| c.y());
            let lo = Vec2::new(sequence_min(&xs), sequence_min(&ys));
            let hi = Vec2::new(sequence_max(&xs), sequence_max(&ys));
            let sx = (hi.x - lo.x) / (nx - 1) as f64;
            let sy = (hi.y - lo.y) / (ny - 1) as f64;
            // A thread's placer: its own fork of this heightfield, on the
            // tile's grid.
            let worker = || FoldPlacer { heightfield: self.forked(), sight, cell, refined, lo, sx, sy };
            // A row at a time, each worker with its own f (`forked`): every
            // node's facing is three evaluations of f.
            let rows: Vec<usize> = (0..ny).collect();
            let rows = parallel_map_with(&rows, 1, worker, |w, &j| {
                (0..nx)
                    .map(|i| w.facing(P2::new(lo.x + i as f64 * sx, lo.y + j as f64 * sy)) as f32)
                    .collect::<Vec<f32>>()
            });
            let values: Vec<f32> = rows.into_iter().flatten().collect();
            let field = Grid2D::new(nx, ny, Domain::new(Interval::new(lo.x, hi.x), Interval::new(lo.y, hi.y)), values);
            paths.extend(worker().place(&contour::lines(&field, 0.0, None), &worker));
        }
        PolylineSet::from_paths(&paths)
    }

    /// An open fold run carried to the zero of f either end leads to, when
    /// the refiner finds one within reach that the run does not already
    /// touch, inside the region: the new vertex is the zero at |f|'s height.
    /// The zero is found in the domain, the end carried back through the
    /// tile under it and the zero carried forward, on whichever tile the run
    /// lies. `cell` is the grid spacing, which is the only length the test of
    /// "already touches" is measured against, so a scaled drawing carries the
    /// scaled fold.
    pub fn ended_at_zeros(&self, run: Vec<P3<WorldSpace>>, cell: f64) -> Vec<P3<WorldSpace>> {
        let Some(refine) = &self.refine else { return run };
        let (Some(zero), Some(magnitude)) = (&refine.zero, &refine.magnitude) else { return run };
        if run.len() < 2 || run.first() == run.last() {
            return run;
        }
        let carried = |end: P3<WorldSpace>, other: P3<WorldSpace>| -> Option<P3<WorldSpace>> {
            let (q, back) = self.surface.tile_under(P2::new(end.x(), end.y()), &self.tiles);
            let d = zero(q)?;
            let forward = back.inverse()?;
            let z = forward.apply(P3::new(d.x(), d.y(), 0.0));
            if !self.region.contains(z.xy()) {
                return None;
            }
            let to_end = ((z.x() - end.x()) * (z.x() - end.x()) + (z.y() - end.y()) * (z.y() - end.y())).sqrt();
            let to_other =
                ((z.x() - other.x()) * (z.x() - other.x()) + (z.y() - other.y()) * (z.y() - other.y())).sqrt();
            if !(to_end > 1e-9 * cell && to_end < to_other) {
                return None;
            }
            let u = magnitude(d);
            if !u.is_finite() {
                return None;
            }
            Some(P3::new(z.x(), z.y(), swift::min(u, self.surface.truncation_height_at(d.x()))))
        };
        let mut out = run;
        if let Some(z) = carried(out[0], out[out.len() - 1]) {
            out.insert(0, z);
        }
        if let Some(z) = carried(out[out.len() - 1], out[0]) {
            out.push(z);
        }
        out
    }

    /// The fold curves that can be seen: `fold_curves`, less what the surface
    /// hides. A fold cannot be tested by a depth buffer: the surface is
    /// edge-on there, so its depth changes by the whole graph within one
    /// pixel, and the pixel's value is the near side's, in front of the fold
    /// by far more than any depth tolerance. It is tested by the solid instead: a fold
    /// vertex is hidden when the line of sight from it to the eye passes under
    /// the surface (`GridTraversal`), and its facing, zero by construction,
    /// is not asked. `depth_tolerance` is the drawing's hidden-line
    /// depth tolerance, which keeps a vertex on the surface it belongs to from hiding
    /// behind that surface's own rounding.
    ///
    /// With the function at hand, curves on the surface are tested where the
    /// surface is (`on_surface`) and the depth tolerance is rounding: a fold vertex
    /// lies on f to the last bit, and forgiving the tolerance instead draws
    /// a stub of a bowl's fold beside the cut, hidden by less than the
    /// tolerance, almost on the top edge.
    ///
    /// `resolution` is how finely a fold's end behind the solid is placed,
    /// as a length in the view plane (`visible_runs`); zero, to the last bit.
    /// (Swift's defaults: `refined` true, `resolution` 0.)
    pub fn visible_folds(
        &self,
        view: &Transform<WorldSpace, ViewSpace>,
        depth_tolerance: f64,
        refined: bool,
        resolution: f64,
    ) -> PolylineSet<WorldSpace> {
        let folds = self.fold_curves(view, refined);
        if folds.vertices.is_empty() {
            return folds;
        }
        let traversal = GridTraversal::new(self.clone(), view, self.surface_depth_tolerance(depth_tolerance));
        Heightfield::visible_runs(
            &folds,
            false,
            &|| {
                let m = Rc::new(traversal.forked());
                Tests::new(|_| None, move |p| m.unoccluded_default(m.heightfield().on_surface(p)))
            },
            &Heightfield::stop(resolution, *view),
        )
    }

    /// The fraction of a segment `a -> b` to which `turning` bisects for curves
    /// placed to `resolution` in the view plane: the segment's own length on
    /// the screen, in resolutions, inverted. A preview places a run's end to
    /// a fraction of its pixel, and a traversal per halving is milliseconds on
    /// f; bisected to the last bit, as a bake is, a fold of cn cost three
    /// quarters of a second a camera for ends no pixel can show.
    pub fn stop(resolution: f64, view: Transform<WorldSpace, ViewSpace>) -> Stop {
        if !(resolution > 0.0) {
            return Box::new(|_, _| 0.0);
        }
        Box::new(move |a, b| {
            let (pa, pb) = (view.apply(a), view.apply(b));
            let length = ((pb.x() - pa.x()) * (pb.x() - pa.x()) + (pb.y() - pa.y()) * (pb.y() - pa.y())).sqrt();
            if length > 0.0 { resolution / length } else { 0.0 }
        })
    }

    /// The depth tolerance curves on the surface are tested with: the drawing's against the
    /// grid, whose chords are that far from f; rounding against f, since
    /// the curves are tested on f itself.
    pub fn surface_depth_tolerance(&self, depth_tolerance: f64) -> f64 {
        if self.refine.as_ref().and_then(|r| r.magnitude.as_ref()).is_none() {
            return depth_tolerance;
        }
        let g = &*self.surface.height;
        let cell = swift::max(
            g.domain.real.length().abs() / g.width.saturating_sub(1).max(1) as f64,
            g.domain.imag.length().abs() / g.height.saturating_sub(1).max(1) as f64,
        );
        1e-9 * cell
    }

    /// A point of curves on the surface, at the surface's own height over it:
    /// where it is tested. A ring's vertex is on f to a hundredth of the
    /// tolerance and a point of its chord to the tolerance; both are tested
    /// on f, so that a point of a ring at a fold is tested exactly as the
    /// fold is, and the ring's end and the fold's end agree.
    pub fn on_surface(&self, p: P3<WorldSpace>) -> P3<WorldSpace> {
        if self.refine.as_ref().and_then(|r| r.magnitude.as_ref()).is_none() {
            return p;
        }
        let z = self.surface_height(P2::new(p.x(), p.y()));
        if z.is_finite() { P3::new(p.x(), p.y(), z) } else { p }
    }

    /// The curves on one kind of face that can be seen, the landscape's one
    /// rule for every layer (`Scene::visible`): a point is seen when the face
    /// under it faces the eye and nothing of the solid stands between it and
    /// the eye (`GridTraversal`). `face` is what the curves lie on
    /// (`SceneGeometry::face_of`), and says which facing is read:
    ///
    /// - **The graph.** The surface bounds the solid under it, so where it
    ///   faces away from the eye the curves on it are hidden, full stop, and
    ///   no depth tolerance has a say: this is the test a depth tolerance
    ///   cannot make, since where a steep graph turns away its back lies
    ///   within the depth tolerance of its front for a stretch, and the other
    ///   family of contours would show through there as ticks along the
    ///   outline. The facing is the truncated surface's (`normal`), so a
    ///   curve lifted over the truncation reads its plateau's. A segment that
    ///   crosses a fold is cut on the fold itself, found by bisection on the
    ///   facing -- the same facing the fold curves are traced from, so a
    ///   contour ends where the outline is drawn -- and the new vertex
    ///   belongs to the run on the visible side, tested by the traversal
    ///   alone, since on the fold the facing is zero by construction. The
    ///   curves are tested on f where the function is at hand (`on_surface`),
    ///   with the depth tolerance as rounding.
    /// - **The walls.** A wall is a plane the oblique camera often sees
    ///   nearly edge-on, and the depth buffer cannot test curves on such a
    ///   face: one pixel spans the whole receding wall, keeps the near side's
    ///   depth, and erases most of the hatch, leaving spots. A point on a
    ///   wall that faces away is behind the solid the wall bounds. A vertical
    ///   edge at a corner lies in two walls and shows if either does. Which
    ///   walls a point lies in is read off the walls' faces: those whose foot
    ///   it stands on (`wall_facing`). The top edge is the
    ///   exception: it is the surface's own edge as much as the wall's top,
    ///   and the far edge of a drawing is in plain view over a wall that
    ///   faces away. A vertex at the surface's height is tested by the
    ///   traversal alone, which hides it exactly when the surface in front of
    ///   it does.
    /// - **The risers**, vertical faces too, by the same test: a point that
    ///   lies in a riser, as its hatching does, a hair onto the lower side,
    ///   is tested by the riser (`riser_facing`), as one on a
    ///   wall below its top edge is by the wall.
    /// - **The plateaus**, flat and facing up.
    /// - **No face**: a crease or anything else, tested by the traversal
    ///   alone, since such curves lie where no one face's facing is the
    ///   curves'. A riser's bottom edge, drawn with the truncation edge, is
    ///   a valley seen only where the riser faces the eye, and is tested as
    ///   a point in the riser.
    ///
    /// `resolution` is how finely a run's end is placed, as a length in the
    /// view plane; zero (Swift's default), to the last bit.
    pub fn visible_curves(
        &self,
        paths: &PolylineSet<WorldSpace>,
        face: Option<FaceKind>,
        view: &Transform<WorldSpace, ViewSpace>,
        depth_tolerance: f64,
        resolution: f64,
    ) -> PolylineSet<WorldSpace> {
        if paths.vertices.is_empty() {
            return paths.clone();
        }
        let sight = view.projection_direction();
        let stop = Heightfield::stop(resolution, *view);
        match face {
            Some(FaceKind::Graph) => {
                let traversal = GridTraversal::new(self.clone(), view, self.surface_depth_tolerance(depth_tolerance));
                Heightfield::visible_runs(
                    paths,
                    true,
                    &|| {
                        let m = Rc::new(traversal.forked());
                        let n = m.clone();
                        Tests::new(
                            move |p| Some(n.heightfield().facing(p.xy(), sight)),
                            move |p| m.unoccluded_default(m.heightfield().on_surface(p)),
                        )
                    },
                    &stop,
                )
            }
            Some(FaceKind::Wall | FaceKind::Riser) => {
                let traversal = GridTraversal::new(self.clone(), view, depth_tolerance);
                let slack = 1e-6 * traversal.cell;
                let facing = self.wall_facing(sight, slack);
                let cell = traversal.cell;
                fn on_top_edge(hf: &Heightfield, p: P3<WorldSpace>, cell: f64) -> bool {
                    let h = hf.surface_height(P2::new(p.x(), p.y()));
                    p.z() >= h - 1e-9 * swift::max(cell, h.abs())
                }
                // A top edge run ends where the traversal turns along the top
                // edge itself: the chord between two top edge vertices lies under
                // a concave top edge, and a point under the top edge is in the
                // wall, which, facing away, is never admitted -- so bisected along
                // the chord the run began at the next vertex, half a cell late,
                // and the far edge of a drawing came out from behind a tower with
                // a stub missing beside the outline's cusp. Visible on the
                // surface, the chord's points are top edge points.
                //
                // Below the top edge a point lies on the wall's plane exactly,
                // with no interpolation gap to forgive, so the traversal forgives
                // it nothing: it is on the solid's boundary, its line is outside
                // from the start, and any terrain it enters hides it. Forgiven,
                // the line from a wall on a tile boundary or across a notch passed
                // through the terrain in front within its skin, or short of the
                // depth tolerance's reach, and the verdict along a stroke turned
                // more than once. Tested where it stands, a point on a notch
                // strictly over the domain is under the surface there, and the
                // stroke was hidden whole.
                //
                // A point that lies in a riser is tested by the riser, as a
                // point on a wall below its top edge is by the wall:
                // it lies on the riser's plane, a hair onto the lower side, and
                // its bottom, at the lower top, would read as on the top edge.
                Heightfield::visible_runs(
                    paths,
                    false,
                    &|| {
                        let m = Rc::new(traversal.forked());
                        let risers = m.heightfield().riser_facing(sight, slack);
                        let (m1, m2, m3) = (m.clone(), m.clone(), m);
                        let (r1, r2, r3) = (risers.clone(), risers.clone(), risers);
                        let facing = facing.clone();
                        Tests {
                            facing: Box::new(move |p| {
                                r1(p).or_else(|| if on_top_edge(m1.heightfield(), p, cell) { None } else { facing(p) })
                            }),
                            clear: Box::new(move |p| {
                                if r2(p).is_none() && on_top_edge(m2.heightfield(), p, cell) {
                                    m2.unoccluded_default(p)
                                } else {
                                    m2.unoccluded(p, false, false, None)
                                }
                            }),
                            lift: Box::new(move |a, b, q| {
                                let h = m3.heightfield();
                                if r3(a).is_none()
                                    && r3(b).is_none()
                                    && on_top_edge(h, a, cell)
                                    && on_top_edge(h, b, cell)
                                {
                                    h.on_surface(q)
                                } else {
                                    q
                                }
                            }),
                        }
                    },
                    &stop,
                )
            }
            Some(FaceKind::Plateau) => {
                let traversal = GridTraversal::new(self.clone(), view, depth_tolerance);
                let up = -sight.z;
                Heightfield::visible_runs(
                    paths,
                    false,
                    &|| {
                        let m = traversal.forked();
                        Tests::new(move |_| Some(up), move |p| m.unoccluded_default(p))
                    },
                    &stop,
                )
            }
            Some(FaceKind::Surface | FaceKind::Solid) | None => {
                let traversal = GridTraversal::new(self.clone(), view, depth_tolerance);
                let slack = 1e-6 * traversal.cell;
                Heightfield::visible_runs(
                    paths,
                    false,
                    &|| {
                        let m = traversal.forked();
                        let risers = m.heightfield().riser_facing(sight, slack);
                        let r = risers.clone();
                        Tests {
                            facing: Box::new(move |p| risers(p)),
                            clear: Box::new(move |p| {
                                if r(p).is_none() {
                                    m.unoccluded_default(p)
                                } else {
                                    m.unoccluded(p, false, false, None)
                                }
                            }),
                            lift: Box::new(|_, _, q| q),
                        }
                    },
                    &stop,
                )
            }
        }
    }

    /// Curves that lie in a cut face -- the wall's hatch and its outline --
    /// less what cannot be seen: `visible_curves` on the walls.
    pub fn visible_wall_curves(
        &self,
        paths: &PolylineSet<WorldSpace>,
        view: &Transform<WorldSpace, ViewSpace>,
        depth_tolerance: f64,
        resolution: f64,
    ) -> PolylineSet<WorldSpace> {
        self.visible_curves(paths, Some(FaceKind::Wall), view, depth_tolerance, resolution)
    }

    /// How squarely the surface faces the viewer over `p`, for lines of
    /// sight along `sight`: positive facing, negative facing away, zero on a
    /// fold. The truncated surface's (`normal`).
    pub fn facing(&self, p: P2<WorldSpace>, sight: Vec3) -> f64 {
        let n = self.normal(p);
        -n.dot(sight) / n.length()
    }

    /// The facing of the walls a point lies in, for lines of sight along
    /// `sight`: positive where one of them faces the eye, negative where it
    /// lies in walls that all face away, None where it lies in none. Each wall
    /// is flat and faces as its face's outward normal does
    /// (`Surface.walls(_:tiles:refine:)`), read straight off the perimeter
    /// edge it stands on rather than through the face, whose height range
    /// can cost a scan of the grid; a point lies in it when it stands within
    /// `slack` of that edge, between its ends. Only walls given as a
    /// perimeter are faces: with none, or a mesh, no point lies in a wall.
    pub fn wall_facing(&self, sight: Vec3, slack: f64) -> FacingTest {
        let Walls::Perimeter { perimeter, .. } = &self.walls else { return Rc::new(|_| None) };
        let turn = if perimeter.signed_area() < 0.0 { -1.0 } else { 1.0 };
        struct Wall {
            start: Vec2,
            along: Vec2,
            length: f64,
            faces_eye: bool,
        }
        let walls: Vec<Wall> = perimeter
            .edges()
            .iter()
            .filter_map(|edge| {
                let along = edge.end.v - edge.start.v;
                let length = along.length();
                if !(length > 0.0) {
                    return None;
                }
                let u = along / length;
                // du × dv of the wall's face, du along the edge and dv up.
                let outward = turn * Vec3::new(u.y, -u.x, 0.0);
                Some(Wall { start: edge.start.v, along: u, length, faces_eye: -outward.dot(sight) > 0.0 })
            })
            .collect();
        Rc::new(move |p| {
            let (mut on_any_wall, mut faces_eye) = (false, false);
            for wall in &walls {
                let r = Vec2::new(p.x(), p.y()) - wall.start;
                let s = r.dot(wall.along);
                if !(s >= -slack
                    && s <= wall.length + slack
                    && (r.x * wall.along.y - r.y * wall.along.x).abs() <= slack)
                {
                    continue;
                }
                on_any_wall = true;
                if wall.faces_eye {
                    faces_eye = true;
                }
            }
            if faces_eye {
                Some(1.0)
            } else if on_any_wall {
                Some(-1.0)
            } else {
                None
            }
        })
    }

    /// The facing of the risers a point lies in, for lines
    /// of sight along `sight`: positive where one of them faces the eye,
    /// negative where all face away, None where it lies in none. Each riser
    /// is flat and faces the lower side (`Surface.risers(tiles:region:refine:)`,
    /// `Surface.tileBoundaryRisers(tiles:region:refine:)`): a band boundary's
    /// on every copy, turned by its tile, and a tile boundary's toward
    /// whichever copy's top is lower there.
    ///
    /// A point lies in one where the riser stands, a hair onto the lower
    /// side of its line, as its bottom edge and hatching are drawn
    /// (`Surface.RiserStretch`), within `slack` of the line, and from the
    /// lower top up to the higher, as they are drawn: to the refiner's
    /// tolerance, the bottom on a copy's graph at a tile boundary and the top
    /// a hatch line rises to. Its bottom edge is in it: a valley, where the
    /// riser meets the face below, seen only where both face the eye, and
    /// the face below is a plateau, facing up, or the lower copy's graph,
    /// which the traversal hides where it turns away. Its top is not, nor
    /// anything on the line itself: the top is a crease, seen over a riser
    /// that faces away where the face above faces the eye, and on the line
    /// stand too the truncation edge's ends at the riser, a wall's top edge
    /// stepping across it, and a wall's vertical edge at a corner on a tile
    /// boundary.
    pub fn riser_facing(&self, sight: Vec3, slack: f64) -> FacingTest {
        let s = self.surface.clone();
        let region = self.region.clone();
        let magnitude = s.face_magnitude(self.refine.as_ref());
        let cell = s.cell();
        let side = 1e-9 * s.cell();
        let drawn = self.refine.as_ref().map_or(0.0, |r| r.tolerance) * s.cell();
        // The band boundaries the truncation steps at inside the domain,
        // each with the way the riser there faces along Re z.
        let g = &*s.height;
        let corner0 = g.position(0, 0);
        let corner1 = g.position(g.width.saturating_sub(1), g.height.saturating_sub(1));
        let (y0, y1) = (swift::min(corner0.y(), corner1.y()), swift::max(corner0.y(), corner1.y()));
        let real = s.domain().real;
        let spans = s.truncation.bands(swift::min(real.lo, real.hi), swift::max(real.lo, real.hi));
        // (b, lesser, greater, toward)
        let steps: Vec<(f64, f64, f64, f64)> = spans
            .iter()
            .zip(spans.iter().skip(1))
            .filter_map(|(left, right)| {
                let lesser = swift::min(left.height, right.height);
                let greater = swift::max(left.height, right.height);
                if !(lesser.is_finite() && lesser < greater) {
                    return None;
                }
                Some((left.hi, lesser, greater, if left.height > right.height { 1.0 } else { -1.0 }))
            })
            .collect();
        let copies: Vec<(Affine2, Affine2)> =
            self.tiles.iter().filter_map(|tile| tile.inverse().map(|back| (*tile, back))).collect();
        // The tile boundaries, each with its copies' inverses and the side
        // each copy is on: its tile's image of the domain's centre.
        let d = s.domain();
        let centre = P3::<WorldSpace>::new(0.5 * (d.real.lo + d.real.hi), 0.5 * (d.imag.lo + d.imag.hi), 0.0);
        let tiles = &self.tiles;
        let boundaries: Vec<(TileBoundary, [Affine2; 2], [Vec2; 2])> = s
            .tile_boundaries(tiles)
            .into_iter()
            .filter_map(|boundary| {
                let first = tiles[boundary.first].inverse()?;
                let second = tiles[boundary.second].inverse()?;
                let n = boundary.normal();
                let sides = [boundary.first, boundary.second]
                    .map(|k| if (tiles[k].apply(centre).xy().v - boundary.start.v).dot(n) > 0.0 { n } else { -n });
                Some((boundary, [first, second], sides))
            })
            .collect();
        Rc::new(move |p| {
            let (mut in_any, mut faces_eye) = (false, false);
            let eps = 1e-9 * swift::max(cell, p.z().abs());
            for (tile, back) in &copies {
                let q = back.apply(P3::new(p.x(), p.y(), 0.0));
                for &(b, lesser, greater, toward) in &steps {
                    if !(q.y() >= y0 && q.y() <= y1) {
                        continue;
                    }
                    let off = (q.x() - b) * toward;
                    if !(off >= 0.5 * side
                        && off <= slack
                        && region.contains(tile.apply(P3::new(b - side, q.y(), 0.0)).xy())
                        && region.contains(tile.apply(P3::new(b + side, q.y(), 0.0)).xy()))
                    {
                        continue;
                    }
                    let top = swift::min(magnitude(P2::new(b, q.y())), greater);
                    if !(p.z() >= lesser - eps && p.z() <= top + swift::max(eps, drawn)) {
                        continue;
                    }
                    // Square to the copy's boundary: the gradient of the
                    // domain's x in space.
                    let outward = toward * Vec2::new(back.a, back.b).normalized();
                    in_any = true;
                    if -(outward.x * sight.x + outward.y * sight.y) > 0.0 {
                        faces_eye = true;
                    }
                }
            }
            for (boundary, backs, sides) in &boundaries {
                let length = boundary.length();
                let n = boundary.normal();
                let r = Vec2::new(p.x(), p.y()) - boundary.start.v;
                let along = r.dot((boundary.end.v - boundary.start.v) / length);
                if !(along >= 0.0 && along <= length && r.dot(n).abs() <= slack) {
                    continue;
                }
                let w = boundary.point(along);
                if !(region.contains(P2::from_vec(w.v - side * n)) && region.contains(P2::from_vec(w.v + side * n))) {
                    continue;
                }
                let z = backs.map(|back| s.tile_boundary_top(w, back, &*magnitude));
                let (low, high) = (swift::min(z[0], z[1]), swift::max(z[0], z[1]));
                let outward = if z[1] < z[0] { sides[1] } else { sides[0] };
                if !(r.dot(outward) >= 0.5 * side
                    && Surface::stands(high, low)
                    && p.z() >= low - swift::max(eps, drawn)
                    && p.z() <= high + swift::max(eps, drawn))
                {
                    continue;
                }
                in_any = true;
                if -(outward.x * sight.x + outward.y * sight.y) > 0.0 {
                    faces_eye = true;
                }
            }
            if faces_eye {
                Some(1.0)
            } else if in_any {
                Some(-1.0)
            } else {
                None
            }
        })
    }

    /// The runs of each path that can be seen: a point is seen where its
    /// face does not face away (`Tests::facing`) and it is clear
    /// (`Tests::clear`). Paths are split where a vertex is not seen; runs
    /// shorter than two vertices draw nothing and are dropped. Swift tests
    /// every vertex in parallel, since the traversal behind it is the
    /// expensive part of a bake; here in turn.
    ///
    /// On a curved face (`folds`) the facing is continuous along a curve,
    /// and a segment whose ends face opposite ways is cut on the fold
    /// between them, by bisection on the facing; the vertex there belongs to
    /// the run on the visible side and is tested by occlusion alone. On a
    /// flat face a change of facing along a segment is a change of face.
    ///
    /// Where a run ends between a seen point and one the occlusion test
    /// hides, it ends where the test turns, found by bisection along the
    /// segment to the last bit, or to `stop` of it (`stop`): a ring
    /// passing behind a top edge ends on the top edge as drawn, not at its
    /// last vertex in front of it. Not where the facing hides a point: a
    /// back-facing vertex's run already ends on the fold, and a test of where
    /// a vertex *is* -- in a wall that faces the eye, on the top edge --
    /// does not turn at a point of the segment, and bisecting it would draw
    /// the sliver of a stroke between the top edge as drawn and f. The
    /// bisection asks occlusion alone on a curved face, and on a flat one the
    /// whole test.
    ///
    /// The tests are each worker's own, made by `worker` (`parallel_map_with`):
    /// a traversal forked for the worker shares nothing of f with the others.
    pub fn visible_runs(
        paths: &PolylineSet<WorldSpace>,
        folds: bool,
        worker: &dyn Fn() -> Tests,
        stop: &dyn Fn(P3<WorldSpace>, P3<WorldSpace>) -> f64,
    ) -> PolylineSet<WorldSpace> {
        let vertices = &paths.vertices;
        // A few vertices to a thread at a time, and one crossing: a test that
        // traverses a line of sight on f is milliseconds a vertex, and at the
        // default chunk a fold of a thousand vertices was four threads' work.
        let tested: Vec<(Option<f64>, bool)> = parallel_map_with(vertices, 4, worker, |w, &v| {
            let facing = (w.facing)(v);
            (facing, facing.is_none_or(|f| f > 0.0) && (w.clear)(v))
        });
        // The segments whose ends face opposite ways, each cut on the fold.
        let mut crossing: Vec<usize> = Vec::new();
        if folds {
            for path in 0..paths.count() {
                for k in (paths.offsets[path] + 1)..paths.offsets[path + 1] {
                    if let (Some(fa), Some(fb)) = (tested[k - 1].0, tested[k].0)
                        && (fa > 0.0) != (fb > 0.0)
                    {
                        crossing.push(k);
                    }
                }
            }
        }
        let on_fold: Vec<(P3<WorldSpace>, bool)> = parallel_map_with(&crossing, 1, worker, |w, &k| {
            let (a, b) = (vertices[k - 1], vertices[k]);
            // Both facings are present: the crossing was listed for them.
            let (fa, fb) = (tested[k - 1].0.unwrap_or(0.0), tested[k].0.unwrap_or(0.0));
            let t = fold_crossing(fa, fb, |t| (w.facing)(along(a, b, t)).unwrap_or(0.0));
            let v = along(a, b, t);
            (v, (w.clear)(v))
        });
        let mut fold: Vec<Option<(P3<WorldSpace>, bool)>> = vec![None; vertices.len()];
        for (i, &k) in crossing.iter().enumerate() {
            fold[k] = Some(on_fold[i]);
        }

        // The sequence each path is tested as: its vertices with the fold
        // crossings between them, each with its verdict and whether occlusion
        // alone decided it.
        struct Item {
            v: P3<WorldSpace>,
            seen: bool,
            by_occlusion: bool,
        }
        let mut sequence: Vec<Vec<Item>> = Vec::new();
        for path in 0..paths.count() {
            let mut seq = Vec::new();
            for k in paths.offsets[path]..paths.offsets[path + 1] {
                if let Some((v, seen)) = fold[k] {
                    seq.push(Item { v, seen, by_occlusion: true });
                }
                seq.push(Item { v: vertices[k], seen: tested[k].1, by_occlusion: tested[k].0.is_none_or(|f| f > 0.0) });
            }
            sequence.push(seq);
        }
        let mut crossings: Vec<(usize, usize)> = Vec::new();
        for (i, seq) in sequence.iter().enumerate() {
            for k in 1..seq.len().max(1) {
                if seq[k - 1].seen != seq[k].seen
                    && (if seq[k - 1].seen { seq[k].by_occlusion } else { seq[k - 1].by_occlusion })
                {
                    crossings.push((i, k));
                }
            }
        }
        let turned: Vec<Option<P3<WorldSpace>>> = parallel_map_with(&crossings, 1, worker, |w, &(path, k)| {
            let (a, b) = (&sequence[path][k - 1], &sequence[path][k]);
            let holds = |p: P3<WorldSpace>| -> bool {
                if folds { (w.clear)(p) } else { (w.facing)(p).is_none_or(|f| f > 0.0) && (w.clear)(p) }
            };
            Heightfield::turning(a.v, a.seen, b.v, holds, stop(a.v, b.v), |q| (w.lift)(a.v, b.v, q))
        });
        let mut turn: Vec<Vec<Option<P3<WorldSpace>>>> = sequence.iter().map(|s| vec![None; s.len()]).collect();
        for (i, &(path, k)) in crossings.iter().enumerate() {
            turn[path][k] = turned[i];
        }

        let mut out: Vec<Vec<P3<WorldSpace>>> = Vec::new();
        for (i, seq) in sequence.iter().enumerate() {
            let mut run: Vec<P3<WorldSpace>> = Vec::new();
            for (k, item) in seq.iter().enumerate() {
                if let Some(t) = turn[i][k] {
                    run.push(t);
                }
                if item.seen {
                    run.push(item.v);
                } else {
                    if run.len() >= 2 {
                        out.push(std::mem::take(&mut run));
                    }
                    run.clear();
                }
            }
            if run.len() >= 2 {
                out.push(run);
            }
        }
        PolylineSet::from_paths(&out)
    }

    /// Where `holds` turns on the segment from `a` to `b`, given it holds at
    /// `a` iff `kept_a`: bisection to the last bit of the fraction, or to
    /// `until` of it, returning the last point at which it still holds on
    /// the kept side. (Swift's defaults: `until` 0, `lift` the identity.)
    pub fn turning(
        a: P3<WorldSpace>,
        kept_a: bool,
        b: P3<WorldSpace>,
        mut holds: impl FnMut(P3<WorldSpace>) -> bool,
        until: f64,
        lift: impl Fn(P3<WorldSpace>) -> P3<WorldSpace>,
    ) -> Option<P3<WorldSpace>> {
        let (mut lo, mut hi) = (0.0, 1.0);
        while hi - lo > swift::max(until, 1e-13) {
            let mid = 0.5 * (lo + hi);
            if holds(lift(along(a, b, mid))) == kept_a {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        // Turned within rounding of the kept end: the kept vertex is where
        // the curve ends, and a second vertex there would be a run of nothing.
        let t = if kept_a { lo } else { 1.0 - hi };
        if !(t > 1e-12) {
            return None;
        }
        Some(lift(along(a, b, if kept_a { lo } else { hi })))
    }
}

// MARK: - the fold placer

/// A heightfield's fold curves placed on f, for one worker: a fork of the
/// heightfield, and the grid of the tile the lines were traced on.
///
/// Placing a vertex on the fold is sixty bisections of four evaluations of
/// f each, the chord between two vertices as many again, and a crossing of
/// the truncation edge sixty of sixty. Each is independent of every other, so `place`
/// takes the lines in stages -- every vertex, then every truncation edge crossing, then
/// every chord, then every run's ends -- and runs each stage over its
/// pieces (in parallel in Swift, each thread with a placer of its own). The
/// pieces come back in the order they went in, so the fold is the same
/// whatever the thread count: a line placed on one core and a line placed on
/// eight are the same vertices.
#[derive(Debug, Clone)]
pub struct FoldPlacer {
    pub heightfield: Heightfield,
    /// The direction lines of sight travel, in world space.
    pub sight: Vec3,
    /// The grid spacing, the larger of the two directions.
    pub cell: f64,
    /// Whether vertices are moved onto the exact fold, or left where the
    /// grid's chords put them.
    pub refined: bool,
    /// The tile's grid: its corner and spacing.
    pub lo: Vec2,
    pub sx: f64,
    pub sy: f64,
}

/// A traced vertex placed: moved onto the fold, at the surface's
/// height, and whether it lies in the region and on the graph rather
/// than below the truncation (`FoldPlacer.Vertex`).
#[derive(Debug, Clone, Copy)]
pub struct FoldVertex {
    pub p: P2<WorldSpace>,
    pub z: f64,
    pub in_region: bool,
    pub on_graph: bool,
}

impl FoldPlacer {
    fn surface(&self) -> &Surface {
        &self.heightfield.surface
    }

    fn region(&self) -> &Region {
        &self.heightfield.region
    }

    /// With the function at hand the facing is the graph's, untruncated: the
    /// fold is a property of the one surface the graph is, and runs on
    /// below the truncation, where the truncation cuts it at the truncation edge. The truncated facing
    /// jumps at the crease, and its zero there is the crease, not the fold.
    fn on_f(&self) -> bool {
        self.heightfield.refine.as_ref().and_then(|r| r.magnitude.as_ref()).is_some()
    }

    /// How squarely the surface faces the viewer over `p`: positive facing,
    /// negative facing away, zero on the fold.
    pub fn facing(&self, p: P2<WorldSpace>) -> f64 {
        let n = if self.on_f() { self.heightfield.graph_normal(p) } else { self.heightfield.normal(p) };
        let len = n.length();
        if len > 0.0 { -n.dot(self.sight) / len } else { 0.0 }
    }

    /// The truncation over a world point, read at its x in the domain,
    /// through the tile under it, as the surface under it is.
    pub fn truncation_height(&self, p: P2<WorldSpace>) -> f64 {
        let s = self.surface();
        s.truncation_height_at(s.in_tile(P2::new(p.x(), p.y()), &self.heightfield.tiles).x())
    }

    /// |f| over a world point, untruncated, through its tile; the grid's
    /// height without the function.
    pub fn graph_height(&self, p: P2<WorldSpace>) -> f64 {
        let Some(magnitude) = self.heightfield.refine.as_ref().and_then(|r| r.magnitude.as_ref()) else {
            return self.heightfield.surface_height(p);
        };
        let u = magnitude(self.surface().in_tile(P2::new(p.x(), p.y()), &self.heightfield.tiles));
        if u.is_nan() { f64::INFINITY } else { u }
    }

    /// The surface's height over a fold vertex: f's, truncated, with the
    /// function at hand, so the fold is drawn at the height the rings that
    /// end on it are drawn at; the grid's otherwise.
    pub fn height(&self, p: P2<WorldSpace>) -> f64 {
        self.heightfield.surface_height(p)
    }

    pub fn vertex(&self, v: P2<DomainSpace>) -> FoldVertex {
        let mut p = P2::<WorldSpace>::new(v.x(), v.y());
        if self.refined {
            p = self.onto_edge(p);
        }
        let z = self.height(p);
        let truncation_height = self.truncation_height(p);
        let in_region = self.region().contains(p);
        FoldVertex { p, z, in_region, on_graph: in_region && z.is_finite() && z < truncation_height }
    }

    /// The traced lines placed on the fold, in stages. Nothing on a plateau,
    /// which faces up and has a truncation edge of its own. With the
    /// function at hand the fold is traced on the graph alone and runs on
    /// below the truncation: where a run passes under the truncation edge it ends on the truncation edge,
    /// at the fold's own crossing of |f| = h(x), so a tower's outline
    /// reaches the truncation and arrives along the fold. Each run is then
    /// subdivided to the refiner's tolerance and carried to the zero it was
    /// going to, on every tile: the facing and the height are read through
    /// the tile under each point.
    pub fn place(&self, lines: &[Vec<P2<DomainSpace>>], worker: &dyn Fn() -> FoldPlacer) -> Vec<Vec<P3<WorldSpace>>> {
        // Every vertex of every line onto the fold.
        let mut offsets = vec![0];
        let mut flat: Vec<P2<DomainSpace>> = Vec::new();
        for line in lines {
            flat.extend_from_slice(line);
            offsets.push(flat.len());
        }
        let vertices: Vec<FoldVertex> = parallel_map_with(&flat, 8, worker, |w, &v| w.vertex(v));
        // Between a vertex on the graph and the next below the truncation, the
        // fold's crossing of the truncation edge.
        let mut pairs: Vec<usize> = Vec::new();
        for i in 0..lines.len() {
            for k in (offsets[i] + 1)..offsets[i + 1] {
                let (a, b) = (vertices[k - 1], vertices[k]);
                if self.on_f()
                    && a.in_region
                    && b.in_region
                    && a.on_graph != b.on_graph
                    && self.truncation_height(b.p).is_finite()
                {
                    pairs.push(k);
                }
            }
        }
        let crossings: Vec<Option<P3<WorldSpace>>> = parallel_map_with(&pairs, 1, worker, |w, &k| {
            let (a, b) = (vertices[k - 1], vertices[k]);
            if b.on_graph { w.onto_truncation_edge(b.p, a.p) } else { w.onto_truncation_edge(a.p, b.p) }
        });
        let mut truncation_edge: Vec<Option<P3<WorldSpace>>> = vec![None; flat.len()];
        for (i, &k) in pairs.iter().enumerate() {
            truncation_edge[k] = crossings[i];
        }
        // The runs: a line cut where it leaves the graph, each piece
        // ending on the truncation edge it passed under.
        let mut runs: Vec<Vec<P3<WorldSpace>>> = Vec::new();
        for i in 0..lines.len() {
            let mut run: Vec<P3<WorldSpace>> = Vec::new();
            for k in offsets[i]..offsets[i + 1] {
                let v = vertices[k];
                if let Some(r) = truncation_edge[k] {
                    run.push(r);
                }
                if v.on_graph {
                    run.push(P3::new(v.p.x(), v.p.y(), v.z));
                } else {
                    if run.len() >= 2 {
                        runs.push(std::mem::take(&mut run));
                    }
                    run.clear();
                }
            }
            if run.len() >= 2 {
                runs.push(run);
            }
        }
        // Every chord of every run subdivided, then each run's ends carried
        // to their zeros.
        if let (true, Some(tolerance)) = (self.refined, self.heightfield.refine.as_ref().map(|r| r.tolerance)) {
            let mut chords: Vec<(usize, usize)> = Vec::new();
            for (r, run) in runs.iter().enumerate() {
                for k in 0..(run.len() - 1) {
                    chords.push((r, k));
                }
            }
            let whole = runs.clone();
            let between: Vec<Vec<P3<WorldSpace>>> = parallel_map_with(&chords, 4, worker, |w, &(r, k)| {
                let mut out = Vec::new();
                w.subdivide(whole[r][k], whole[r][k + 1], tolerance, 0, &mut out);
                out
            });
            let mut next = 0;
            for r in 0..runs.len() {
                let mut out = vec![whole[r][0]];
                for k in 0..(whole[r].len() - 1) {
                    out.extend_from_slice(&between[next]);
                    next += 1;
                    out.push(whole[r][k + 1]);
                }
                runs[r] = out;
            }
        }
        parallel_map_with(&runs, 1, worker, |w, run| w.heightfield.ended_at_zeros(run.clone(), w.cell))
    }

    /// Between two vertices on the fold, the fold itself: the chord's
    /// midpoint moved onto the fold along the chord's normal, kept when it
    /// moved further than the refiner's tolerance, and the two halves
    /// likewise -- the contour refiner's rule, so a fold is a curve to the
    /// same tolerance the rings that end on it are.
    pub fn subdivide(
        &self,
        a: P3<WorldSpace>,
        b: P3<WorldSpace>,
        tolerance: f64,
        depth: usize,
        out: &mut Vec<P3<WorldSpace>>,
    ) {
        if depth >= 5 {
            return;
        }
        let chord = ((b.x() - a.x()) * (b.x() - a.x()) + (b.y() - a.y()) * (b.y() - a.y())).sqrt();
        if !(chord > 1e-9 * self.cell) {
            return;
        }
        let mid = Vec2::new(0.5 * (a.x() + b.x()), 0.5 * (a.y() + b.y()));
        let n = Vec2::new(-(b.y() - a.y()), b.x() - a.x()) / chord;
        let lo = mid - 0.5 * chord * n;
        let hi = mid + 0.5 * chord * n;
        let flo = self.facing(P2::from_vec(lo));
        if (flo > 0.0) == (self.facing(P2::from_vec(hi)) > 0.0) {
            return;
        }
        let (holds, fails) = bisect_points(lo, hi, |q| (self.facing(P2::from_vec(q)) > 0.0) == (flo > 0.0));
        let q = 0.5 * (holds + fails);
        let p = P2::<WorldSpace>::new(q.x, q.y);
        if !self.region().contains(p) {
            return;
        }
        let z = self.height(p);
        if !(z.is_finite() && z < self.truncation_height(p) - 1e-9) {
            return;
        }
        let v = P3::<WorldSpace>::new(p.x(), p.y(), z);
        // The fold is a curve in space: its departure from the chord is
        // measured there, across and up.
        if !((v.v - 0.5 * (a.v + b.v)).length() > tolerance * self.cell) {
            return;
        }
        self.subdivide(a, v, tolerance, depth + 1, out);
        out.push(v);
        self.subdivide(v, b, tolerance, depth + 1, out);
    }

    /// Onto the fold, along the gradient: bracket a sign change within a
    /// cell either way, then bisect it to the last bit. The fallback, for a
    /// vertex that lies on no grid edge.
    pub fn onto(&self, c: P2<WorldSpace>) -> P2<WorldSpace> {
        let h = 0.25 * self.cell;
        let gx = self.facing(P2::new(c.x() + h, c.y())) - self.facing(P2::new(c.x() - h, c.y()));
        let gy = self.facing(P2::new(c.x(), c.y() + h)) - self.facing(P2::new(c.x(), c.y() - h));
        let len = (gx * gx + gy * gy).sqrt();
        if !(len > 0.0) {
            return c;
        }
        let d = Vec2::new(gx, gy) / len * self.cell;
        let (a, b) = (c.v - d, c.v + d);
        let fa = self.facing(P2::from_vec(a));
        if (fa > 0.0) == (self.facing(P2::from_vec(b)) > 0.0) {
            return c;
        }
        let (holds, fails) = bisect_points(a, b, |q| (self.facing(P2::from_vec(q)) > 0.0) == (fa > 0.0));
        P2::from_vec(0.5 * (holds + fails))
    }

    /// The grid edges a marching-squares vertex lies on: one, or two for
    /// a vertex on a node.
    pub fn grid_edges(&self, c: P2<WorldSpace>) -> Vec<(Vec2, Vec2)> {
        let g = &*self.surface().height;
        let (nx, ny) = (g.width, g.height);
        let (lo, sx, sy) = (self.lo, self.sx, self.sy);
        let fx = (c.x() - lo.x) / sx;
        let fy = (c.y() - lo.y) / sy;
        let on_column = (fx - fx.round()).abs() < 1e-6;
        let on_row = (fy - fy.round()).abs() < 1e-6;
        let mut ends = Vec::new();
        // Swift's `Int` traps on an index that is not a number; such a
        // vertex lies on no edge here.
        if on_column
            && fy < (ny as f64) - 1.0
            && let (Some(i), Some(j)) = (swift::int(fx.round()), swift::int(fy.floor()))
        {
            let (i, j) = (i as f64, j as f64);
            ends.push((Vec2::new(lo.x + i * sx, lo.y + j * sy), Vec2::new(lo.x + i * sx, lo.y + (j + 1.0) * sy)));
        }
        if on_row
            && fx < (nx as f64) - 1.0
            && let (Some(i), Some(j)) = (swift::int(fx.floor()), swift::int(fy.round()))
        {
            let (i, j) = (i as f64, j as f64);
            ends.push((Vec2::new(lo.x + i * sx, lo.y + j * sy), Vec2::new(lo.x + (i + 1.0) * sx, lo.y + j * sy)));
        }
        ends
    }

    /// Onto the fold, along the grid edge the vertex lies on: a
    /// marching-squares vertex is the chord's crossing of an edge whose two
    /// nodes face opposite ways, so the edge brackets the fold by
    /// construction, and bisecting it to the last bit lands on the fold, as
    /// the contour refiner lands on f along the same edges. Bracketing
    /// along the gradient instead can miss where the facing is sharp -- a
    /// pit one cell wide -- and leave the vertex on the chord.
    pub fn onto_edge(&self, c: P2<WorldSpace>) -> P2<WorldSpace> {
        for (a, b) in self.grid_edges(c) {
            let fa = self.facing(P2::from_vec(a));
            let fb = self.facing(P2::from_vec(b));
            if (fa > 0.0) == (fb > 0.0) {
                continue;
            }
            let (holds, fails) = bisect_points(a, b, |q| (self.facing(P2::from_vec(q)) > 0.0) == (fa > 0.0));
            return P2::from_vec(0.5 * (holds + fails));
        }
        self.onto(c)
    }

    /// Where the fold crosses the truncation edge, between a vertex on the graph and
    /// one below the truncation: the chord's points moved onto the fold along the
    /// chord's normal, as `subdivide` moves a midpoint, and the one where
    /// |f| reaches the truncation found by bisection to the last bit. The
    /// grid's crossing of the crease is where a grid edge meets the
    /// truncation edge, which is not where the fold does: joined to the graph by one
    /// straight chord a cell long, the fold cut across the truncation's projection
    /// on its way to the truncation edge.
    pub fn onto_truncation_edge(&self, a: P2<WorldSpace>, b: P2<WorldSpace>) -> Option<P3<WorldSpace>> {
        let chord = ((b.x() - a.x()) * (b.x() - a.x()) + (b.y() - a.y()) * (b.y() - a.y())).sqrt();
        if !(chord > 1e-9 * self.cell) {
            return None;
        }
        let n = Vec2::new(-(b.y() - a.y()), b.x() - a.x()) / chord;
        let on_fold = |t: f64| -> P2<WorldSpace> {
            let m = Vec2::new(a.x() + t * (b.x() - a.x()), a.y() + t * (b.y() - a.y()));
            let lo = m - 0.5 * chord * n;
            let hi = m + 0.5 * chord * n;
            let flo = self.facing(P2::from_vec(lo));
            if (flo > 0.0) == (self.facing(P2::from_vec(hi)) > 0.0) {
                return P2::from_vec(m);
            }
            let (holds, fails) = bisect_points(lo, hi, |q| (self.facing(P2::from_vec(q)) > 0.0) == (flo > 0.0));
            P2::from_vec(0.5 * (holds + fails))
        };
        let excess = |t: f64| -> f64 {
            let q = on_fold(t);
            self.graph_height(q) - self.truncation_height(q)
        };
        if !(excess(0.0) < 0.0 && excess(1.0) >= 0.0) {
            return None;
        }
        let (lo, hi) = bisect(0.0, 1.0, |t| excess(t) < 0.0);
        // Where the excess jumps rather than crosses zero, the truncation
        // steps between two bands under the fold: it leaves the graph up a
        // riser, and ends on the riser's top, at the graph's own height on
        // the graph's side, not down at the lesser band's plateau.
        let (below, above) = (on_fold(lo), on_fold(hi));
        let (near, far) = (self.truncation_height(below), self.truncation_height(above));
        if near.is_finite() && far.is_finite() && near != far {
            let z = self.graph_height(below);
            if !(self.region().contains(below) && z.is_finite() && z < near) {
                return None;
            }
            return Some(P3::new(below.x(), below.y(), z));
        }
        let q = on_fold(0.5 * (lo + hi));
        let truncation_height = self.truncation_height(q);
        if !(self.region().contains(q) && truncation_height.is_finite()) {
            return None;
        }
        Some(P3::new(q.x(), q.y(), truncation_height))
    }
}

// MARK: - the scene's visibility and bounds

impl Scene {
    /// Every layer's curves in view space, in declaration (draw) order, tested
    /// for this camera: `visible` of each.
    pub fn visible_layers(
        &self,
        depth_tolerance: Option<f64>,
        image: Option<&Arc<SurfaceImage>>,
    ) -> Result<Vec<(Layer, PolylineSet<ViewSpace>)>, PortRefusal> {
        let mut out = Vec::new();
        for layer in &self.layers {
            out.extend(self.visible(layer, depth_tolerance, image)?);
        }
        Ok(out)
    }

    /// A layer's curves in view space, tested for this camera wherever the
    /// geometry can test them: the one rule of visibility. A point of a curve
    /// is seen when the face under it faces the eye and nothing stands
    /// between it and the eye. The kind of face is the layer's
    /// (`SceneGeometry::face_of`); where its facing changes sign along a
    /// segment, the segment is cut on the fold by bisection on it; and what
    /// stands in front is decided by the scene's kind of occlusion test, one
    /// of the `OcclusionTest`s, asked here as its own type: grid traversal
    /// through a landscape, at the depth tolerance the layer's face asks for
    /// (`Heightfield::visible_curves`), sphere tracing round a surface with a
    /// tubular neighbourhood (`SurfaceVisibility`), and a polyhedron's pieces
    /// in their segment form, for its edges and hatching
    /// (`Solid.visibleEdges`, `Solid::visible_hatching`). The coordinate
    /// image, `image`, tests a parametric surface's curves only where it has
    /// no tubular neighbourhood: a point is seen when the surface in front of
    /// it is its own neighbourhood (`SurfaceVisibility`), and its fold
    /// crossings are cut on the exact facing. A parametric surface's curves
    /// that carry no coordinates fall back to the image's depth, at
    /// `depth_tolerance`. No depth buffer has any other part in it, so the
    /// drawing is a function of the geometry and the camera, never of a
    /// resolution.
    ///
    /// Curves that depend on the camera -- the fold curves of a surface, of
    /// either kind, and the edges of a solid -- are derived here, for this
    /// camera, rather than carried. A solid's hatching, and a curved face's,
    /// comes back as one layer per width step, at that fraction of the width.
    /// An unclipped layer is drawn whole, and so is a parametric surface's
    /// with neither a tubular neighbourhood nor an image. This is the bake's
    /// work, not the preview's: it is a traversal per vertex, and the preview
    /// keeps its depth test but for the strokes a camera's move changes.
    /// `depth_tolerance` is the hidden-line depth tolerance to test by; the
    /// scene's own when not given.
    pub fn visible(
        &self,
        layer: &Layer,
        depth_tolerance: Option<f64>,
        image: Option<&Arc<SurfaceImage>>,
    ) -> Result<Vec<(Layer, PolylineSet<ViewSpace>)>, PortRefusal> {
        let view = self.camera.view;
        let depth_tolerance = depth_tolerance.unwrap_or(self.depth_tolerance);
        let folds = matches!(layer.spec.source, LayerSource::FoldCurves);
        match &self.geometry {
            SceneGeometry::Parametric(s) => {
                if let LayerSource::Hatching { hatching, faces: FaceSelector::Surface } = &layer.spec.source {
                    // Along a coordinate, a layer per width step, at that
                    // fraction of the width, each tested as any curve on the
                    // surface.
                    return Ok(s
                        .coordinate_hatching(hatching)
                        .into_iter()
                        .map(|(w, paths)| {
                            let mut spec = layer.spec.clone();
                            spec.width = layer.spec.width * w as f64 / hatching.width_steps as f64;
                            if hatching.width_steps > 1 {
                                spec.name = format!("{} {w}/{}", layer.spec.name, hatching.width_steps);
                            }
                            let step = Layer { spec, paths };
                            let seen = self.visible_on_surface(s, &step, &step.paths, false, image, depth_tolerance);
                            (step, seen)
                        })
                        .collect());
                }
                let visible = if folds {
                    self.visible_on_surface(s, layer, &self.parametric_folds(s), folds, image, depth_tolerance)
                } else {
                    self.visible_on_surface(s, layer, &layer.paths, folds, image, depth_tolerance)
                };
                Ok(vec![(layer.clone(), visible)])
            }
            SceneGeometry::Heightfield(h) => {
                if let LayerSource::Hatching { hatching, faces: FaceSelector::Graph { tiled } } = &layer.spec.source {
                    let context =
                        HatchContext { perimeter: None, region: h.region.clone(), tiles: h.tiles.clone(), base: 0.0 };
                    let steps = h.surface.graph_hatching(hatching, *tiled, &context, h.refine.as_ref());
                    // A layer per width step, at that fraction of the width,
                    // each tested on the graph when clipped.
                    return Ok(steps
                        .into_iter()
                        .map(|(w, paths)| {
                            let paths = PolylineSet::from_paths(&paths);
                            let mut spec = layer.spec.clone();
                            spec.width = layer.spec.width * w as f64 / hatching.width_steps as f64;
                            if hatching.width_steps > 1 {
                                spec.name = format!("{} {w}/{}", layer.spec.name, hatching.width_steps);
                            }
                            let seen = if layer.spec.clipped {
                                h.visible_curves(&paths, Some(FaceKind::Graph), &view, depth_tolerance, 0.0)
                            } else {
                                paths.clone()
                            };
                            (Layer { spec, paths }, seen.mapped(&view))
                        })
                        .collect());
                }
                let visible = if folds {
                    h.visible_folds(&view, depth_tolerance, true, 0.0)
                } else if !layer.spec.clipped {
                    layer.paths.clone()
                } else {
                    h.visible_curves(&layer.paths, self.geometry.face_of(&layer.spec), &view, depth_tolerance, 0.0)
                };
                Ok(vec![(layer.clone(), visible.mapped(&view))])
            }
            SceneGeometry::Solid(s) => {
                if matches!(layer.spec.source, LayerSource::Edges) && layer.spec.clipped {
                    return Ok(vec![(layer.clone(), s.visible_edges(&view).mapped(&view))]);
                }
                if let LayerSource::Hatching { hatching, faces: FaceSelector::Solid } = &layer.spec.source {
                    // One layer per width step, at that fraction of the
                    // width: every hatch line when unclipped, the visible
                    // spans otherwise.
                    let steps: VisibleHatching = if layer.spec.clipped {
                        s.visible_hatching(hatching, &view)
                    } else {
                        s.hatch_lines_by_width(hatching)
                            .into_iter()
                            .map(|(w, lines)| {
                                let paths: Vec<Vec<P3<WorldSpace>>> = lines.into_iter().map(|l| l.path).collect();
                                let all = PolylineSet::from_paths(&paths);
                                (w, all.clone(), all)
                            })
                            .collect()
                    };
                    return Ok(steps
                        .into_iter()
                        .map(|(w, all, drawn)| {
                            let mut spec = layer.spec.clone();
                            spec.width = layer.spec.width * w as f64 / hatching.width_steps as f64;
                            if hatching.width_steps > 1 {
                                spec.name = format!("{} {w}/{}", layer.spec.name, hatching.width_steps);
                            }
                            (Layer { spec, paths: all }, drawn.mapped(&view))
                        })
                        .collect());
                }
                if matches!(layer.spec.source, LayerSource::File { .. }) && layer.spec.clipped {
                    let paths: Vec<Vec<P3<WorldSpace>>> = layer.paths.paths().map(|p| p.to_vec()).collect();
                    let seen = s.visible_curves_on_surface(&paths, &s.occlusion(&view));
                    return Ok(vec![(layer.clone(), PolylineSet::from_paths(&seen).mapped(&view))]);
                }
                Ok(vec![(layer.clone(), layer.paths.mapped(&view))])
            }
        }
    }

    /// A parametric surface's fold curves for this camera
    /// (`foldCurves(sight:creases:)`), placed once and remembered for the
    /// line of sight (`fold_memo`).
    fn parametric_folds(&self, s: &ParametricSurface) -> Arc<PolylineSet<WorldSpace>> {
        let sight = self.camera.view.projection_direction();
        let mut memo = self.fold_memo.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((key, folds)) = memo.as_ref()
            && *key == sight
        {
            return folds.clone();
        }
        let folds = Arc::new(s.fold_curves(sight, None, &self.creases()));
        *memo = Some((sight, folds.clone()));
        folds
    }

    /// Curves on a parametric surface, or its folds, tested by the
    /// surface's own coordinates where it has a tubular neighbourhood or
    /// an image, by the image's depth where they carry none, and drawn
    /// whole where it has neither or the layer is unclipped (Swift's
    /// `visibleOnSurface`, nested in `visible`).
    fn visible_on_surface(
        &self,
        s: &ParametricSurface,
        layer: &Layer,
        paths: &PolylineSet<WorldSpace>,
        folds: bool,
        image: Option<&Arc<SurfaceImage>>,
        depth_tolerance: f64,
    ) -> PolylineSet<ViewSpace> {
        let view = self.camera.view;
        let projected = paths.mapped(&view);
        if !layer.spec.clipped {
            return projected;
        }
        let visibility = match image {
            Some(image) => Some(SurfaceVisibility::new(s.clone(), image.clone(), view)),
            None => SurfaceVisibility::sphere_tracing(s.clone(), view).ok(),
        };
        let clipped = if let Some(visibility) = visibility.filter(|_| projected.coords.is_some()) {
            // On no face is on a fold or a crease, where the normal
            // says nothing.
            let on_folds = self.geometry.face_of(&layer.spec).is_none();
            // Neither of clip_on's preconditions can fail here: the curves
            // carry coordinates, and a visibility without an image traces
            // spheres.
            HiddenLine::clip_on(&projected, &visibility, on_folds, on_folds && !folds)
                .expect("curves with coordinates, tested by an image or by sphere tracing")
        } else if let Some(image) = image {
            HiddenLine::clip(&projected, &image.depth, depth_tolerance)
        } else {
            return projected;
        };
        PolylineSet::new(clipped.vertices.iter().map(|v| P3::from_vec(v.v)).collect(), clipped.offsets, None)
    }

    /// The view-space extent the depth buffer covers: every heightfield sample,
    /// the walls, and every *clipped* layer.
    ///
    /// This is `ZBuffer(xs.min(), xs.max(), ys.min(), ys.max(), ...)` on the
    /// Python side, and it has to be the same extent or the two rasterize onto
    /// different pixel grids and every comparison downstream is measuring the
    /// framing rather than the drawing. Two consequences of matching it exactly:
    ///
    /// - The heightfield is scanned sample by sample, not bounded by its box.
    ///   The box of a rotated landscape is looser than the hull of its samples,
    ///   and the difference is visible at bake resolution.
    /// - Masked-out samples still count. `build_occluder` emits every grid
    ///   vertex and drops only *triangles*, so `occ_rot` includes the vertices
    ///   inside zeta's cutout even though nothing references them. Excluding
    ///   them here would be more principled and would not be the same picture.
    ///
    /// Unclipped curves are excluded, as they are in Python: they are drawn but never
    /// looked up, so letting it stretch the frame would spend depth resolution
    /// on nothing and would make the buffer depend on decoration.
    /// Whether `view_bounds` is none: the scene projects to nothing, with no
    /// point of its geometry and no vertex of a clipped layer. Read off the
    /// counts, without projecting a point, which on a dense landscape is
    /// every one of its samples; a parametric surface with no positions asks
    /// `view_bounds`, since its folds are derived to be counted.
    pub fn projects_to_nothing(&self) -> Result<bool, PortRefusal> {
        let empty = match &self.geometry {
            SceneGeometry::Heightfield(h) => {
                let g = &*h.surface.height;
                h.occluder.vertices.is_empty() && (h.tiles.is_empty() || g.width == 0 || g.height == 0)
            }
            SceneGeometry::Parametric(s) => {
                if s.positions.is_empty() {
                    return Ok(self.view_bounds()?.is_none());
                }
                false
            }
            SceneGeometry::Solid(s) => s.vertices.is_empty(),
        };
        Ok(empty && self.layers.iter().filter(|l| l.spec.clipped).all(|l| l.paths.vertices.is_empty()))
    }

    pub fn view_bounds(&self) -> Result<Option<AABB<ViewSpace>>, PortRefusal> {
        let view = self.camera.view;
        let mut lo = Vec3::splat(f64::INFINITY);
        let mut hi = Vec3::splat(f64::NEG_INFINITY);
        let mut any = false;
        let mut add = |p: P3<ViewSpace>| {
            lo = lo.min(p.v);
            hi = hi.max(p.v);
            any = true;
        };
        match &self.geometry {
            SceneGeometry::Heightfield(h) => {
                for &p in &h.occluder.vertices {
                    add(view.apply(p));
                }
                h.for_each_sample(|p| add(view.apply(p)));
            }
            SceneGeometry::Parametric(s) => {
                for &p in s.positions.iter() {
                    add(view.apply(p));
                }
            }
            SceneGeometry::Solid(s) => {
                for &p in &s.vertices {
                    add(view.apply(p));
                }
            }
        }
        // The curves as carried, before any testing visibility: testing visibility is the bake's
        // expensive step and would be paid twice, and curves the test drops are
        // still curves the frame was sized to.
        for layer in self.layers.iter().filter(|l| l.spec.clipped) {
            for &v in &layer.paths.vertices {
                add(view.apply(v));
            }
        }
        // A parametric surface's folds are placed on the map, not the
        // grid, and can lie a hair outside its hull: they are derived here
        // so the frame holds them, since they are tested by the frame.
        if let Some(s) = self.parametric()
            && self.layers.iter().any(|l| matches!(l.spec.source, LayerSource::FoldCurves))
        {
            for &v in &self.parametric_folds(s).vertices {
                add(view.apply(v));
            }
        }
        Ok(any.then_some(AABB::new(lo, hi)))
    }

    /// A cheap approximate view-space bound, from a coarse subsample.
    ///
    /// `view_bounds` folds over every heightfield sample because the bake has
    /// to frame the picture exactly the way Python does. Nothing interactive can
    /// afford that -- a "fit to window" that scans six million points is not a
    /// fit, it is a stall -- and nothing interactive needs it, because a frame
    /// that is a fraction of a percent loose is a frame nobody can see is loose.
    ///
    /// The bounding *box* would be cheaper still and is much worse: the box of a
    /// rotated landscape is far bigger than the hull of its samples, so fitting
    /// to it leaves the picture small and off-centre. Subsampling keeps the
    /// shape of the hull and only loses its last few percent. (Swift's
    /// `budget` defaults to 20,000.)
    pub fn quick_bounds(&self, budget: usize) -> Option<AABB<ViewSpace>> {
        let view = self.camera.view;
        let mut lo = Vec3::splat(f64::INFINITY);
        let mut hi = Vec3::splat(f64::NEG_INFINITY);
        let mut any = false;
        let mut add = |p: P3<ViewSpace>| {
            lo = lo.min(p.v);
            hi = hi.max(p.v);
            any = true;
        };
        match &self.geometry {
            SceneGeometry::Heightfield(h) => {
                let g = &*h.surface.height;
                let per_tile = (budget / h.tiles.len().max(1)).max(16);
                let side = ((per_tile as f64).sqrt() as usize).max(4);
                let stride = h.step.max((g.width / side).max(g.height / side));
                for tile in &h.tiles {
                    h.surface.for_each_sample(stride, |p| add(view.apply(tile.apply(p))));
                }
                for &v in &h.occluder.vertices {
                    add(view.apply(v));
                }
            }
            SceneGeometry::Parametric(s) => {
                let stride = (s.positions.len() / budget.max(1)).max(1);
                for i in (0..s.positions.len()).step_by(stride) {
                    add(view.apply(s.positions[i]));
                }
            }
            SceneGeometry::Solid(s) => {
                for &p in &s.vertices {
                    add(view.apply(p));
                }
            }
        }
        any.then_some(AABB::new(lo, hi))
    }
}
