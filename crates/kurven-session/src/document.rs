//! One open bundle, a landscape, a parametric surface or a polyhedron, and
//! everything the window knows about it: the port of
//! `Sources/KurvenApp/Document.swift`, its state and its transitions, without
//! its views.
//!
//! Every mutation is a whole new value assigned into a stored property, so the
//! model holds no derived state that can go stale. In particular `scene` keeps
//! its `ContentID` across a camera change, which is what lets the renderer's
//! GPU resources survive navigation; `navigator` is the only thing a drag
//! touches.
//!
//! Swift's document runs its slow work -- a sampling, a restyle, a build, a
//! face's boundary -- on detached tasks whose results land back on the main
//! actor. Here each is a [`Job`](crate::jobs::Job) queued on the document, and
//! `settle` runs them in the order they were started, each landing as Swift's
//! continuation lands. A landing never interleaves with an edit, as Swift's
//! continuations cannot interleave with the main actor's synchronous code; so
//! an edit followed by `settle` is what Swift's headless run, which waits for
//! the document to settle after every edit, does.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;

use kurven_bake::{BakeError, BakeOptions};
use kurven_core::bundle::{BundleFiles, KurvenBundle, Source};
use kurven_core::face::Face;
use kurven_core::manifest::{CameraPreset, LayerSource, LayerSpec, Truncation};
use kurven_core::navigation::{Framing, Gesture, Navigator, Orbit, Viewport};
use kurven_core::pick::FacePick;
use kurven_core::polyhedron_bundle::PolyhedronBundle;
use kurven_core::polyline::Layer;
use kurven_core::scene::{ContentID, PreviewMode, Scene};
use kurven_core::spaces::{AABB, Angle, P2, P3, ViewSpace, WorldSpace};
use kurven_core::surface_request::SurfaceRequest;
use kurven_core::svg::Strokes;
use kurven_landscape::catalog::{Catalog, LandscapeRequest};
use kurven_landscape::landscape::NativeLandscape;
use kurven_surfaces::bundle::SurfaceBundle;
use kurven_surfaces::drawing::{SurfaceDrawing, SurfacePreset};

use crate::edits::{Edited, UndoStack};
use crate::jobs::Job;
use crate::landscape::Styling;

/// What a ready document holds: a landscape's bundle, a parametric
/// surface's, whose request the surface's controls edit, or a polyhedron's,
/// whose layers its controls restyle.
#[derive(Debug, Clone)]
pub enum Opened {
    Landscape(KurvenBundle),
    Surface(SurfaceBundle),
    Polyhedron(PolyhedronBundle),
}

/// Swift's `Document.State`. `loading` is the page's business, not the
/// engine's: a bundle is read in one call here, and a Worker's caller waits
/// for the reply.
#[derive(Debug, Clone)]
pub enum State {
    Empty,
    /// A bundle, opened, sampled or built: every kind is saved, restyled
    /// and baked.
    Ready(Opened),
    Failed(PathBuf, String),
}

/// The face a click picked, and the scene that draws its boundary.
#[derive(Debug, Clone)]
pub struct Selection {
    pub pick: FacePick,
    /// The boundary, a layer in the accent colour.
    pub layer: Layer,
    /// The scene with the boundary drawn last, built once a pick so its
    /// curves keep one identity while the camera moves.
    pub scene: Scene,
    /// The curves of the scene it was built over.
    pub base: ContentID,
}

/// The last derivation of a described layer, keyed by the levels that
/// produced it.
#[derive(Debug, Clone)]
pub struct DerivedCurves {
    pub levels: Vec<f64>,
    pub layer: Layer,
    /// Placed on f, or the grid's: a draft's.
    pub refined: bool,
}

/// One open bundle and everything the window knows about it. The page holds
/// nothing this does not.
pub struct Document {
    pub(crate) state: State,
    /// Bumped whenever `state` is written, so curves derived from one bundle
    /// are dropped if a different one is open when they land.
    pub(crate) version: u64,
    /// The landscape or the surface, and its curves. Its camera and depth
    /// tolerance are whatever they were when it was stored, and nothing reads
    /// them: `framed_scene` supplies both.
    pub(crate) scene: Option<Scene>,
    pub navigator: Option<Navigator>,
    pub viewport: Viewport,

    // The picked face (`picking.rs`).
    pub(crate) selection: Option<Selection>,
    pub(crate) picked: Option<FacePick>,
    pub(crate) pick_faces: Option<(ContentID, Arc<Vec<Face>>)>,
    pub(crate) picking: bool,
    pub(crate) pick_generation: u64,
    pub pick_status: Option<String>,

    pub mode: PreviewMode,
    /// Layer indices that are drawn. Absent means all of them.
    pub hidden_layers: BTreeSet<usize>,
    pub depth_tolerance: f64,
    /// Per-layer level-count overrides, for the layers a bundle described
    /// rather than dumped, by the layer's name. Absent means the bundle's own
    /// levels.
    pub(crate) level_counts: BTreeMap<String, i64>,

    // The landscape, when the document is one (`landscape.rs`).
    pub(crate) catalog: &'static Catalog,
    /// What the controls say. None when this document is a bundle someone
    /// made earlier rather than a landscape being chosen.
    pub(crate) landscape: Option<LandscapeRequest>,
    /// The request the open bundle actually came from.
    pub(crate) shown: Option<LandscapeRequest>,
    /// The latest edit, waiting for the request in flight to finish.
    pub(crate) wanted: Option<(LandscapeRequest, bool, bool)>,
    /// Whether the bundle on screen is a draft: its contours are the grid's.
    pub(crate) shown_draft: bool,
    pub(crate) sampling: bool,
    pub landscape_status: Option<String>,
    /// The resolution a landscape is sampled at while a control is dragged.
    pub draft_resolution: i64,
    /// The styling the controls have asked for, ahead of the bundle that will
    /// carry it. None when the bundle is up to date.
    pub(crate) styled: Option<Styling>,
    /// Curves being derived: a restyle, a change of refinement, or a layer's
    /// levels.
    pub(crate) drawing: bool,
    /// A layer's levels changed, and whether the change was a draft.
    pub(crate) curves_wanted: Option<bool>,
    pub(crate) derived_curves: BTreeMap<usize, DerivedCurves>,
    /// Whether a landscape's contours are placed by its function.
    pub(crate) refine_contours: bool,

    // The surface, when the document is one (`surfaces.rs`).
    pub(crate) surface: Option<SurfaceRequest>,
    pub(crate) surface_shown: Option<SurfaceRequest>,
    pub(crate) surface_wanted: Option<(SurfaceRequest, bool, bool)>,
    pub(crate) building: bool,
    pub surface_status: Option<String>,
    /// Bumped whenever the document becomes a different thing, so a build
    /// still in flight for the old one is dropped when it lands.
    pub(crate) generation: u64,

    // The polyhedron, when the document is one (`polyhedra.rs`).
    pub(crate) polyhedron_styled: Option<Vec<LayerSpec>>,
    pub polyhedron_status: Option<String>,

    // Painting and its undo (`paint.rs`, `edits.rs`).
    pub(crate) undo: UndoStack,
    pub(crate) surface_styled: Option<Vec<LayerSpec>>,
    pub(crate) edited_before_drag: Option<Edited>,
    pub(crate) surface_git_sha: Option<String>,
    pub(crate) landscape_git_sha: Option<String>,
    pub(crate) paint_ceiling: Option<(ContentID, f64)>,
    pub paint_status: Option<String>,

    pub bake_resolution: i64,
    pub bake_status: Option<String>,

    /// The slow work started and not yet landed, in the order it was started.
    pub(crate) jobs: VecDeque<Job>,

    /// What the preview's layouts read again every frame (`preview.rs`).
    pub(crate) preview_memo: crate::preview::Memo,
}

impl Default for Document {
    fn default() -> Self {
        Document::new()
    }
}

impl Document {
    pub fn new() -> Document {
        Document {
            state: State::Empty,
            version: 0,
            scene: None,
            navigator: None,
            viewport: Viewport::new(1200, 800),
            selection: None,
            picked: None,
            pick_faces: None,
            picking: false,
            pick_generation: 0,
            pick_status: None,
            mode: PreviewMode::Drawing,
            hidden_layers: BTreeSet::new(),
            depth_tolerance: 0.02,
            level_counts: BTreeMap::new(),
            catalog: Catalog::native(),
            landscape: None,
            shown: None,
            wanted: None,
            shown_draft: false,
            sampling: false,
            landscape_status: None,
            draft_resolution: 240,
            styled: None,
            drawing: false,
            curves_wanted: None,
            derived_curves: BTreeMap::new(),
            refine_contours: true,
            surface: None,
            surface_shown: None,
            surface_wanted: None,
            building: false,
            surface_status: None,
            generation: 0,
            polyhedron_styled: None,
            polyhedron_status: None,
            undo: UndoStack::default(),
            surface_styled: None,
            edited_before_drag: None,
            surface_git_sha: None,
            landscape_git_sha: None,
            paint_ceiling: None,
            paint_status: None,
            bake_resolution: 4000,
            bake_status: None,
            jobs: VecDeque::new(),
            preview_memo: crate::preview::Memo::default(),
        }
    }

    // MARK: - the state

    pub fn state(&self) -> &State {
        &self.state
    }

    /// Write `state`, bumping `version` as Swift's `didSet` does.
    pub(crate) fn set_state(&mut self, state: State) {
        self.state = state;
        self.version = self.version.wrapping_add(1);
    }

    pub fn scene(&self) -> Option<&Scene> {
        self.scene.as_ref()
    }

    /// Write `scene`, carrying the pick across as Swift's `didSet` does.
    pub(crate) fn set_scene(&mut self, scene: Option<Scene>) {
        self.scene = scene;
        self.keep_selection();
    }

    /// The scene as the window shows it: the stored landscape, through the
    /// navigator's camera, at the current depth tolerance.
    pub fn framed_scene(&self) -> Option<Scene> {
        self.scene.as_ref().and_then(|s| self.framed(s))
    }

    /// The scene as the window draws it: `framed_scene`, with the picked
    /// face's boundary over it. A bake draws `framed_scene`, so a pick never
    /// reaches a drawing.
    pub fn shown_scene(&self) -> Option<Scene> {
        self.selection.as_ref().map(|s| &s.scene).or(self.scene.as_ref()).and_then(|s| self.framed(s))
    }

    fn framed(&self, scene: &Scene) -> Option<Scene> {
        let navigator = self.navigator?;
        let mut scene = scene.looking(navigator.camera());
        scene.depth_tolerance = self.depth_tolerance;
        // A quarter of a drawable pixel: finer than any end of a curve can show.
        scene.resolution = 0.25 * navigator.framing.units_per_pixel;
        Some(scene)
    }

    pub fn selection(&self) -> Option<&Selection> {
        self.selection.as_ref()
    }

    pub fn picked(&self) -> Option<&FacePick> {
        self.picked.as_ref()
    }

    pub fn bundle(&self) -> Option<&KurvenBundle> {
        match &self.state {
            State::Ready(Opened::Landscape(b)) => Some(b),
            _ => None,
        }
    }

    pub fn surface_bundle(&self) -> Option<&SurfaceBundle> {
        match &self.state {
            State::Ready(Opened::Surface(b)) => Some(b),
            _ => None,
        }
    }

    pub fn polyhedron_bundle(&self) -> Option<&PolyhedronBundle> {
        match &self.state {
            State::Ready(Opened::Polyhedron(b)) => Some(b),
            _ => None,
        }
    }

    pub fn surface_drawing(&self) -> Option<&SurfaceDrawing> {
        self.surface_bundle().map(|b| &b.drawing)
    }

    /// The surface request the controls say.
    pub fn surface_request(&self) -> Option<&SurfaceRequest> {
        self.surface.as_ref()
    }

    /// The landscape request the controls say.
    pub fn landscape_request(&self) -> Option<&LandscapeRequest> {
        self.landscape.as_ref()
    }

    pub fn level_counts(&self) -> &BTreeMap<String, i64> {
        &self.level_counts
    }

    /// Whether anything is on screen: a bundle of any kind.
    pub fn is_open(&self) -> bool {
        self.bundle().is_some() || self.surface_bundle().is_some() || self.polyhedron_bundle().is_some()
    }

    pub fn url(&self) -> Option<PathBuf> {
        match &self.state {
            State::Empty => None,
            State::Ready(Opened::Landscape(b)) => Some(b.url.clone()),
            State::Ready(Opened::Surface(b)) => Some(b.url.clone()),
            State::Ready(Opened::Polyhedron(b)) => Some(b.url.clone()),
            State::Failed(u, _) => Some(u.clone()),
        }
    }

    pub fn title(&self) -> String {
        if let Some(b) = self.surface_bundle() {
            return SurfacePreset::named(&b.manifest.surface.name).map(|p| p.label).unwrap_or_else(|| stem(&b.url));
        }
        if let Some(landscape) = &self.landscape
            && self.is_landscape()
        {
            return self
                .catalog
                .preset(&landscape.name)
                .map(|p| p.label.clone())
                .unwrap_or_else(|| landscape.expression.clone());
        }
        self.url().map(|u| stem(&u)).unwrap_or_else(|| "Kurven".to_string())
    }

    /// The open bundle's layers as its manifest describes them.
    pub fn layer_specs(&self) -> Vec<LayerSpec> {
        if let Some(b) = self.bundle() {
            return b.manifest.layers.clone();
        }
        if let Some(b) = self.surface_bundle() {
            return b.manifest.layers.clone();
        }
        if let Some(b) = self.polyhedron_bundle() {
            return b.manifest.layers.clone();
        }
        Vec::new()
    }

    pub fn layers(&self) -> &[Layer] {
        self.scene.as_ref().map(|s| s.layers()).unwrap_or(&[])
    }

    /// The indices of the layers drawn, as the preview's options name them
    /// (Swift's `previewOptions`).
    pub fn visible_layers(&self) -> Vec<usize> {
        (0..self.layers().len()).filter(|i| !self.hidden_layers.contains(i)).collect()
    }

    /// `visible_layers`, with the picked face's boundary drawn: the last
    /// layer of `shown_scene` when something is picked (Swift's
    /// `shownOptions`).
    pub fn shown_visible_layers(&self) -> Vec<usize> {
        let mut visible = self.visible_layers();
        if let Some(selection) = &self.selection {
            let last = selection.scene.layers().len().saturating_sub(1);
            if !visible.contains(&last) {
                visible.push(last);
            }
        }
        visible
    }

    // MARK: - loading

    /// A bundle read, its contours placed by f when it names a function and
    /// refinement is on. A parametric surface's is built from its request,
    /// and a polyhedron's from its vertices and loops.
    pub fn read(source: &Source, refining: bool) -> Result<Opened, String> {
        if kurven_core::bundle::holds_key_in(source, "polyhedron").map_err(|e| e.to_string())? {
            return PolyhedronBundle::read_from(source).map(Opened::Polyhedron).map_err(|e| e.to_string());
        }
        if kurven_core::bundle::holds_key_in(source, "surface").map_err(|e| e.to_string())? {
            return SurfaceBundle::read_from(source).map(Opened::Surface).map_err(|e| e.to_string());
        }
        let bundle = KurvenBundle::read_from(source).map_err(|e| e.to_string())?;
        if !refining || bundle.refine.is_some() {
            return Ok(Opened::Landscape(bundle));
        }
        match NativeLandscape::refiner_for_bundle(&bundle) {
            Some(refiner) => Ok(Opened::Landscape(bundle.refined(Some(refiner.refine_hook())))),
            None => Ok(Opened::Landscape(bundle)),
        }
    }

    /// Open a bundle from its files (Swift's `load`): whatever was open is
    /// forgotten, its edits and anything queued for it with it.
    pub fn load(&mut self, name: &str, files: &BundleFiles) {
        self.load_from(&Source::Memory { name, files });
    }

    /// `load`, from wherever the bundle's files are.
    pub fn load_from(&mut self, source: &Source) {
        self.scene = None;
        self.navigator = None;
        self.generation += 1;
        // An edit queued for whatever was open is not an edit of this.
        self.wanted = None;
        self.surface_wanted = None;
        self.polyhedron_styled = None;
        self.surface_styled = None;
        self.forget_edits();
        match Document::read(source, self.refine_contours) {
            Ok(Opened::Landscape(bundle)) => self.adopt(bundle, false, false),
            Ok(Opened::Surface(bundle)) => self.adopt_surface_bundle(bundle),
            Ok(Opened::Polyhedron(bundle)) => self.adopt_polyhedron(bundle, false),
            Err(error) => self.set_state(State::Failed(source.url(), error)),
        }
    }

    /// Swap in a bundle derived from the same grids -- a restyling, not a new
    /// landscape.
    pub(crate) fn replace(&mut self, bundle: KurvenBundle) {
        self.set_state(State::Ready(Opened::Landscape(bundle)));
    }

    /// Take a freshly read bundle as the document.
    ///
    /// `keeping_camera` is for a resample: the landscape under the camera has
    /// changed, the camera has not. Re-fitting on every change would mean a
    /// window that jumps while a domain slider is dragged.
    ///
    /// Nothing here derives curves: the bundle arrives with its layers drawn,
    /// refined or not.
    pub(crate) fn adopt(&mut self, bundle: KurvenBundle, keeping_camera: bool, draft: bool) {
        self.set_state(State::Ready(Opened::Landscape(bundle.clone())));
        self.surface = None;
        self.surface_shown = None;
        self.shown_draft = draft;
        self.curves_wanted = None;
        if !keeping_camera {
            self.styled = None;
        }
        self.adopt_scene(&bundle, keeping_camera);
        // Styling asked for while this bundle was being sampled, and not in
        // the request it was sampled from, is drawn on it next (Swift's
        // `defer { redraw() }`).
        self.redraw();
    }

    fn adopt_scene(&mut self, bundle: &KurvenBundle, keeping_camera: bool) {
        let Some(preset) = bundle.manifest.presets.first().cloned() else {
            self.set_state(State::Failed(bundle.url.clone(), "the bundle declares no camera presets".to_string()));
            return;
        };
        let scene = Scene::from_bundle(bundle, &preset);
        self.derived_curves.clear();
        if keeping_camera && let Some(navigator) = self.navigator {
            self.set_scene(Some(scene.looking(navigator.camera())));
            self.adopt_landscape(bundle);
            return;
        }
        self.depth_tolerance = preset.depth_tolerance;
        self.bake_resolution = preset.buffer;
        self.hidden_layers.clear();
        self.level_counts.clear();
        let navigator = self.fitted_navigator(&scene, &preset);
        self.set_scene(Some(scene.looking(navigator.camera())));
        self.navigator = Some(navigator);
        self.adopt_landscape(bundle);
    }

    /// The navigator a document opens at: the preset's orbit, fitted to the
    /// window.
    fn fitted_navigator(&self, scene: &Scene, preset: &CameraPreset) -> Navigator {
        let mut navigator = Navigator::new(
            Orbit::matching(&preset.projection, P3::new(0.0, 0.0, 0.0)),
            Framing::new(P2::new(0.0, 0.0), 1.0),
        );
        let looked = scene.looking(navigator.camera());
        if let Some(bounds) = looked.quick_bounds(QUICK_BOUNDS) {
            navigator.framing = Framing::fitting(&bounds, self.viewport, Framing::FIT_MARGIN);
        }
        navigator
    }

    /// Take a parametric surface's bundle read from disk as the document: its
    /// request is what the controls edit, and it opens at its own first preset.
    pub(crate) fn adopt_surface_bundle(&mut self, bundle: SurfaceBundle) {
        self.surface = Some(bundle.manifest.surface.clone());
        self.surface_shown = self.surface.clone();
        self.show(bundle, false);
    }

    /// Take a built drawing as the document. A new drawing opens at the
    /// default oblique projection, fitted to the window; an edit keeps the
    /// camera, and the bundle it edits keeps its place and its presets.
    pub(crate) fn adopt_surface_drawing(&mut self, drawing: SurfaceDrawing, keeping_camera: bool, git_sha: &str) {
        let kept = if keeping_camera { self.surface_bundle().cloned() } else { None };
        let name = if drawing.request.name.is_empty() { "Surface".to_string() } else { drawing.request.name.clone() };
        let url =
            kept.as_ref().map(|b| b.url.clone()).unwrap_or_else(|| PathBuf::from(format!("/native/{name}.kurven")));
        let presets =
            kept.as_ref().map(|b| b.manifest.presets.clone()).unwrap_or_else(|| vec![SurfaceRequest::preset()]);
        self.show(SurfaceBundle::new(&url, drawing, presets, git_sha), keeping_camera);
    }

    /// Take the same surface drawn with other layers as the document, the
    /// camera kept.
    pub(crate) fn adopt_restyled(&mut self, bundle: SurfaceBundle) {
        let keeping = self.navigator.is_some() && self.scene.is_some();
        self.show(bundle, keeping);
    }

    fn show(&mut self, bundle: SurfaceBundle, keeping_camera: bool) {
        let drawing = bundle.drawing.clone();
        let before = self.surface_drawing().map(|d| d.content);
        self.set_state(State::Ready(Opened::Surface(bundle.clone())));
        self.landscape = None;
        self.shown = None;
        self.derived_curves.clear();
        if keeping_camera && let (Some(navigator), Some(scene)) = (self.navigator, self.scene.clone()) {
            // Hidden layers are hidden by name: an edit that adds the lines
            // puts a layer before the folds, and a hidden fold stays hidden.
            let hidden: BTreeSet<String> =
                self.hidden_layers.iter().filter_map(|&i| scene.layers().get(i).map(|l| l.spec.name.clone())).collect();
            self.hidden_layers =
                (0..drawing.layers.len()).filter(|&i| hidden.contains(&drawing.layers[i].spec.name)).collect();
            // The same surface keeps its identity, so the renderer keeps its
            // textures and its normals and lays out only the curves.
            let next = if before == Some(drawing.content) {
                scene.drawing(drawing.layers.clone())
            } else {
                drawing.scene(navigator.camera())
            };
            self.set_scene(Some(next));
            // A layer drawn at levels of its own is drawn at them again.
            if !self.level_counts.is_empty() {
                self.rebuild_curves(false);
            }
            return;
        }
        let preset = bundle.manifest.presets.first().cloned().unwrap_or_else(SurfaceRequest::preset);
        self.depth_tolerance = preset.depth_tolerance;
        self.bake_resolution = preset.buffer;
        self.hidden_layers.clear();
        self.level_counts.clear();
        self.curves_wanted = None;
        self.styled = None;
        let start = Navigator::new(
            Orbit::matching(&preset.projection, P3::new(0.0, 0.0, 0.0)),
            Framing::new(P2::new(0.0, 0.0), 1.0),
        );
        let scene = drawing.scene(start.camera());
        let mut navigator = start;
        if let Some(bounds) = scene.quick_bounds(QUICK_BOUNDS) {
            navigator.framing = Framing::fitting(&bounds, self.viewport, Framing::FIT_MARGIN);
        }
        self.set_scene(Some(scene));
        self.navigator = Some(navigator);
    }

    /// Take a polyhedron's bundle as the document. A bundle read from disk
    /// opens at its own first preset, fitted to the window; a restyle keeps
    /// the camera, and hides the layers that were hidden, by name.
    pub(crate) fn adopt_polyhedron(&mut self, bundle: PolyhedronBundle, keeping_camera: bool) {
        self.set_state(State::Ready(Opened::Polyhedron(bundle.clone())));
        self.landscape = None;
        self.shown = None;
        self.surface = None;
        self.surface_shown = None;
        self.derived_curves.clear();
        let layers = bundle.layers.clone();
        let preset = bundle.manifest.presets.first().cloned().unwrap_or_else(SurfaceRequest::preset);
        if keeping_camera && let (Some(navigator), Some(scene)) = (self.navigator, self.scene.clone()) {
            let hidden: BTreeSet<String> =
                self.hidden_layers.iter().filter_map(|&i| scene.layers().get(i).map(|l| l.spec.name.clone())).collect();
            self.hidden_layers = (0..layers.len()).filter(|&i| hidden.contains(&layers[i].spec.name)).collect();
            // The same solid keeps its identity, so the renderer keeps its
            // mesh and lays out only the curves.
            let next = if scene.solid() == Some(bundle.solid()) {
                scene.drawing(layers)
            } else {
                bundle.scene(&preset).looking(navigator.camera())
            };
            self.set_scene(Some(next));
            return;
        }
        self.depth_tolerance = preset.depth_tolerance;
        self.bake_resolution = preset.buffer;
        self.hidden_layers.clear();
        self.level_counts.clear();
        self.curves_wanted = None;
        self.styled = None;
        self.polyhedron_styled = None;
        self.polyhedron_status = None;
        let scene = bundle.scene(&preset);
        let navigator = self.fitted_navigator(&scene, &preset);
        self.set_scene(Some(scene.looking(navigator.camera())));
        self.navigator = Some(navigator);
    }

    // MARK: - navigation

    pub fn apply(&mut self, gesture: &Gesture) {
        if let Some(navigator) = self.navigator {
            self.navigator = Some(navigator.applying(gesture, self.viewport));
        }
    }

    /// The scene's bounds under the navigator's camera.
    fn bounds_now(&self) -> Option<AABB<ViewSpace>> {
        let navigator = self.navigator?;
        self.scene.as_ref()?.looking(navigator.camera()).quick_bounds(QUICK_BOUNDS)
    }

    pub fn fit(&mut self) {
        if let Some(bounds) = self.bounds_now() {
            self.apply(&Gesture::Fit(bounds));
        }
    }

    pub fn use_preset(&mut self, preset: &CameraPreset) {
        let (Some(scene), Some(navigator)) = (&self.scene, self.navigator) else { return };
        let camera = Orbit::matching(&preset.projection, navigator.orbit.target).camera();
        let Some(bounds) = scene.looking(camera).quick_bounds(QUICK_BOUNDS) else { return };
        self.depth_tolerance = preset.depth_tolerance;
        self.bake_resolution = preset.buffer;
        self.apply(&Gesture::Preset(preset.projection, bounds));
    }

    pub fn retarget(&mut self, world: P3<WorldSpace>) {
        self.apply(&Gesture::Retarget(world));
    }

    pub fn set_perspective(&mut self, on: bool) {
        let Some(bounds) = self.bounds_now() else { return };
        self.apply(&Gesture::Project { field_of_view: on.then(|| Angle::from_degrees(50.0)), bounds });
    }

    /// The levels a described layer is currently drawn at.
    ///
    /// An override resamples the bundle's own level range uniformly
    /// (`KurvenBundle::levels_resampled`).
    pub fn levels_for_layer(&self, index: usize) -> Option<Vec<f64>> {
        let spec = self.layer_specs().get(index)?.clone();
        let base = KurvenBundle::levels(&spec)?;
        if base.len() < 2 {
            return None;
        }
        match self.level_counts.get(&spec.name) {
            Some(&n) if n != base.len() as i64 => KurvenBundle::levels_resampled(&spec, n.max(0) as usize),
            _ => Some(base),
        }
    }

    pub fn is_derived(&self, index: usize) -> bool {
        self.layer_specs().get(index).is_some_and(|spec| KurvenBundle::levels(spec).is_some())
    }

    /// The open bundle's layers as they are drawn: each described contour at
    /// the levels its slider has set. What a save writes.
    pub fn drawn_layers(&self) -> Vec<LayerSpec> {
        self.layer_specs()
            .into_iter()
            .enumerate()
            .map(|(index, spec)| {
                let LayerSource::Contour { field, restriction, faces, light, .. } = &spec.source else { return spec };
                let Some(levels) = self.levels_for_layer(index) else { return spec };
                let mut drawn = spec.clone();
                drawn.source = LayerSource::Contour {
                    field: *field,
                    levels,
                    restriction: restriction.clone(),
                    faces: faces.clone(),
                    light: *light,
                };
                drawn
            })
            .collect()
    }

    /// Re-derive the described layers at their current levels. A `draft`,
    /// while the slider is dragged, is the grid's contours, and the release
    /// places them on f.
    pub fn rebuild_curves(&mut self, draft: bool) {
        // A full redraw asked for stays asked for, whatever drafts follow it.
        self.curves_wanted = Some(draft && self.curves_wanted.unwrap_or(true));
        self.redraw();
    }

    /// The described layers at their current levels, from `cached` where the
    /// levels are the same and the cached curves are as refined as asked.
    pub(crate) fn stroked(
        bundle: &KurvenBundle,
        levels: &[Option<Vec<f64>>],
        cached: &BTreeMap<usize, DerivedCurves>,
        draft: bool,
    ) -> (Vec<Layer>, BTreeMap<usize, DerivedCurves>) {
        // The grid's contours, for a draft: the same bundle without its refiner.
        let grid = bundle.refined(None);
        let refined = bundle.refine.is_some() && !draft;
        let mut derived = cached.clone();
        let layers = bundle
            .manifest
            .layers
            .iter()
            .enumerate()
            .map(|(index, spec)| {
                let Some(levels) = levels.get(index).cloned().flatten() else {
                    return bundle.layer(&spec.name).cloned().unwrap_or_else(|_| Layer {
                        spec: spec.clone(),
                        paths: kurven_core::polyline::PolylineSet::empty(),
                    });
                };
                // Only the layer whose slider moved is re-derived.
                if let Some(c) = cached.get(&index)
                    && c.levels == levels
                    && (c.refined || !refined)
                {
                    return c.layer.clone();
                }
                let layer = (if refined { bundle } else { &grid }).layer_at_levels(spec, &levels);
                derived.insert(index, DerivedCurves { levels, layer: layer.clone(), refined });
                layer
            })
            .collect();
        (layers, derived)
    }

    pub fn set_level_count(&mut self, n: i64, index: usize, draft: bool) {
        let Some(name) = self.layer_specs().get(index).map(|s| s.name.clone()) else { return };
        self.level_counts.insert(name, n.max(2));
        self.rebuild_curves(draft);
    }

    pub fn reset_levels(&mut self, index: usize) {
        let Some(name) = self.layer_specs().get(index).map(|s| s.name.clone()) else { return };
        self.level_counts.remove(&name);
        self.rebuild_curves(false);
    }

    pub fn toggle(&mut self, index: usize) {
        if !self.hidden_layers.remove(&index) {
            self.hidden_layers.insert(index);
        }
    }

    // MARK: - baking

    pub fn can_bake(&self) -> bool {
        !self.navigator.is_some_and(|n| n.orbit.is_perspective())
    }

    /// Bake the scene the window is showing, `framed_scene`, at
    /// `bake_resolution`: the same function `kurven-cli bake` calls, on the
    /// same value. Swift runs it on a task; a Worker's caller waits for it.
    pub fn bake(&mut self) -> Result<Strokes, String> {
        let Some(scene) = self.framed_scene() else { return Err("nothing to bake".to_string()) };
        if let Some(why) = CameraPreset::unsupported_buffer(self.bake_resolution) {
            let status = format!("the resolution {why}");
            self.bake_status = Some(status.clone());
            return Err(status);
        }
        if !self.can_bake() {
            let status = "switch off perspective to bake: the plates are orthographic, and a perspective bake has no \
                          oracle"
                .to_string();
            self.bake_status = Some(status.clone());
            return Err(status);
        }
        let mut options = BakeOptions::new(self.bake_resolution);
        options.depth_tolerance = Some(self.depth_tolerance);
        match kurven_bake::bake(&scene, &options) {
            Ok(bake) => {
                self.bake_status = Some(format!("{} paths", bake.strokes.path_count()));
                Ok(bake.strokes)
            }
            Err(error) => {
                let error: BakeError = error;
                self.bake_status = Some(format!("bake failed: {error}"));
                Err(format!("bake failed: {error}"))
            }
        }
    }
}

/// Swift's `quickBounds()` budget.
pub(crate) const QUICK_BOUNDS: usize = 20_000;

/// A bundle's name without its extension, as a title shows it.
fn stem(url: &std::path::Path) -> String {
    url.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

/// The truncation the styling says, `none` without one.
pub(crate) fn truncation_of(styling: Option<&Styling>) -> Truncation {
    styling.map(|s| s.truncation.clone()).unwrap_or(Truncation::None)
}
