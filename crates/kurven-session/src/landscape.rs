//! The half of a document that is a *choice* rather than a file: the port of
//! `Sources/KurvenApp/Landscape.swift`.
//!
//! Opening a bundle gives you one landscape to look at. This gives you the
//! function, the window and the truncation as controls, and the split between
//! what that costs is the whole design (`docs/landscapes.md`, "Two speeds of
//! edit"):
//!
//!   - **the function, the window, the resolution** need f evaluated again, so
//!     they are a new landscape (`NativeLandscape::build`) and a new bundle.
//!   - **the truncation, the contour levels, the hatch spacing** need nothing
//!     but the grids that are already here, because every layer of a
//!     landscape's bundle is a description rather than dumped curves. They
//!     are `restyle`, which re-derives from the grids.
//!
//! Both coalesce -- one job in flight, the latest edit wins -- so a slow
//! function makes the picture lag the hand and never stops the window. A
//! resample drops to a draft resolution, with the grid's contours, until the
//! drag ends.

use kurven_core::bundle::{BundleFiles, KurvenBundle};
use kurven_core::json::JsonValue;
use kurven_core::manifest::{DomainAxis, Interval, LayerSource, LayerSpec, Manifest, Truncation, Walls};
use kurven_landscape::catalog::{FunctionPreset, LandscapeRequest};
use kurven_landscape::landscape::{LandscapeStyle, NativeLandscape};

use crate::document::{Document, Opened, State};
use crate::edits::REBUILT_GIT_SHA;
use crate::jobs::{DrawJob, Job};

/// What a restyle changes: the truncation and the layers' descriptions.
/// Everything else in a manifest is the landscape's, and a restyle keeps it.
#[derive(Debug, Clone, PartialEq)]
pub struct Styling {
    pub truncation: Truncation,
    pub layers: Vec<LayerSpec>,
}

impl Styling {
    pub fn of(manifest: &Manifest) -> Styling {
        Styling { truncation: manifest.truncation.clone(), layers: manifest.layers.clone() }
    }

    pub fn apply(&self, manifest: &mut Manifest) {
        manifest.truncation = self.truncation.clone();
        manifest.layers = self.layers.clone();
    }
}

/// This request with `KurvenApp --resample`'s overrides: `rMin`, `rMax`,
/// `iMin`, `iMax` the window, as a bundle's provenance records it, `res`
/// the resolution and `truncation` a uniform truncation height, as
/// `kurven-cli landscape` spells them. Other keys are ignored.
pub fn applying_overrides(request: &LandscapeRequest, a: &[(&str, f64)]) -> LandscapeRequest {
    let get = |key: &str| a.iter().rev().find(|(k, _)| *k == key).map(|(_, v)| *v);
    let mut next = request.clone();
    next.domain = kurven_core::manifest::Domain::new(
        Interval::new(get("rMin").unwrap_or(request.domain.real.lo), get("rMax").unwrap_or(request.domain.real.hi)),
        Interval::new(get("iMin").unwrap_or(request.domain.imag.lo), get("iMax").unwrap_or(request.domain.imag.hi)),
    );
    if let Some(res) = get("res") {
        // Swift's `Int(res.rounded())`, which traps where this saturates.
        next.resolution = kurven_math::swift::int(res.round()).unwrap_or(next.resolution);
    }
    if let Some(z) = get("truncation") {
        next.truncation = Some(Truncation::Uniform(z));
    }
    next
}

impl Document {
    pub fn is_landscape(&self) -> bool {
        self.bundle().is_some_and(|b| b.manifest.provenance.example == "function")
    }

    /// Start a landscape from a catalog preset, replacing whatever is open.
    pub fn create(&mut self, preset: &FunctionPreset) {
        let mut request = LandscapeRequest::from_preset(preset, self.catalog.default_resolution);
        request.spacing = None; // let the rules derive it for this window
        self.landscape = Some(request.clone());
        self.generation += 1;
        self.surface_wanted = None;
        self.surface_styled = None;
        self.forget_edits();
        // A different function is a different landscape, not an edit of this
        // one: its styling is the new function's defaults.
        self.send(request, true, false);
    }

    /// Start a landscape from what a link names (the page's URL): its
    /// request, the styling it was drawn with and its level counts, sampled
    /// afresh and fitted as `create` starts one, with no edits to undo. Not
    /// Swift's; the page's addition (crates/FINDINGS.md, Phase 8). The
    /// request is sent with the styling in it, as `landscape_edited` sends
    /// one, and kept as the controls say it, so the document is the one that
    /// was linked: the same samples, the same styling, the same levels.
    pub fn restore_landscape(
        &mut self,
        request: LandscapeRequest,
        styling: Styling,
        level_counts: std::collections::BTreeMap<String, i64>,
    ) {
        self.landscape = Some(request.clone());
        self.generation += 1;
        self.surface_wanted = None;
        self.surface_styled = None;
        self.forget_edits();
        let mut sent = request.clone();
        sent.truncation = Some(styling.truncation);
        sent.layers = Some(styling.layers);
        self.send(sent, true, false);
        self.settle();
        if self.bundle().is_none() {
            return;
        }
        // A framed adoption reads the request back from the bundle (its
        // canonical expression, its layers); the controls say the one
        // linked, as a resample's landing keeps the one edited.
        self.landscape = Some(request);
        // ...and clears the level counts, so they are set on the landscape
        // it adopted.
        if !level_counts.is_empty() {
            self.level_counts = level_counts;
            self.rebuild_curves(false);
            self.settle();
        }
    }

    /// The controls changed. `draft` is true while one is being dragged.
    pub fn landscape_edited(&mut self, draft: bool) {
        let Some(mut request) = self.landscape.clone() else { return };
        if draft {
            request.resolution = request.resolution.min(self.draft_resolution);
        }
        // The styling travels with the request, so a change of window keeps
        // the truncation and the hatching the user chose rather than
        // re-deriving them for the new rectangle.
        if let Some(styling) = self.styling()
            && self.is_landscape()
        {
            request.truncation = Some(styling.truncation);
            request.layers = Some(styling.layers);
        }
        self.send(request, false, draft);
    }

    /// One request in flight, the latest edit queued behind it.
    fn send(&mut self, request: LandscapeRequest, framing: bool, draft: bool) {
        self.wanted = Some((request, framing, draft));
        self.pump();
    }

    pub(crate) fn pump(&mut self) {
        if self.sampling {
            return;
        }
        let Some((request, framing, draft)) = self.wanted.clone() else { return };
        if let Some(shown) = &self.shown
            && shown.samples_as(&request)
            && !framing
            && (!self.shown_draft || draft)
        {
            // The samples on screen are already the ones asked for; only the
            // styling moved, and that is not this path's business.
            self.wanted = None;
            return;
        }
        self.wanted = None;
        self.sampling = true;
        self.landscape_status = Some("sampling…".to_string());
        let refining = self.refine_contours && !draft;
        // Sampled here, so recorded as this process's, unless an undo asks
        // for the bundle it put back.
        let git_sha = self.landscape_git_sha.clone().unwrap_or_else(|| REBUILT_GIT_SHA.to_string());
        self.jobs.push_back(Job::Sample { request, framing, draft, asked: self.generation, git_sha, refining });
    }

    /// A sampling landed (the rest of Swift's `pump` task).
    pub(crate) fn sampled(
        &mut self,
        request: LandscapeRequest,
        framing: bool,
        draft: bool,
        asked: u64,
        git_sha: String,
        result: Result<KurvenBundle, String>,
    ) {
        match result {
            Ok(bundle) => {
                // A surface chosen meanwhile is the document now.
                if self.generation == asked {
                    let keeping = !framing && self.scene.is_some();
                    // What the controls ask for now: the request, and the
                    // styling, which a restyle made while this sampled may
                    // have moved past what was sent with it.
                    let (request0, styling0) = (self.landscape.clone(), self.styling());
                    let manifest = bundle.manifest.clone();
                    self.adopt(bundle, keeping, draft);
                    self.shown = Some(request.clone());
                    if keeping {
                        self.landscape = request0;
                        if let Some(styling0) = styling0
                            && Styling::of(&manifest) != styling0
                        {
                            self.styled = Some(styling0);
                            self.redraw();
                        }
                    }
                    if self.landscape_git_sha.as_deref() == Some(git_sha.as_str()) && !draft {
                        self.landscape_git_sha = None;
                    }
                }
                // Swift adds the time it took; there is no clock in a wasm
                // module, and a status is not a drawing.
                self.landscape_status = Some(format!("{}²{}", request.resolution, if draft { " (draft)" } else { "" }));
            }
            Err(error) => self.landscape_status = Some(error),
        }
        self.sampling = false;
        self.pump();
    }

    /// Note the landscape a newly opened bundle is, so its controls start
    /// where it is rather than where the catalog's defaults are. A bundle
    /// written by `landscape` records its own request in
    /// `provenance.params`. Any other bundle records no request, and opens
    /// read-only.
    pub(crate) fn adopt_landscape(&mut self, bundle: &KurvenBundle) {
        if bundle.manifest.provenance.example != "function" {
            self.landscape = None;
            self.shown = None;
            return;
        }
        let params = &bundle.manifest.provenance.params;
        let number = |key: &str| -> Option<f64> {
            match params.get(key) {
                Some(JsonValue::Double(d)) => Some(*d),
                Some(JsonValue::Int(i)) => Some(*i as f64),
                _ => None,
            }
        };
        let string = |key: &str| -> Option<String> {
            match params.get(key) {
                Some(JsonValue::String(s)) => Some(s.clone()),
                _ => None,
            }
        };
        let resolution = kurven_math::swift::int(number("resolution").unwrap_or(600.0)).unwrap_or(600);
        let mut request = LandscapeRequest::new(
            &string("expression").unwrap_or_else(|| bundle.manifest.provenance.function.clone()),
            bundle.manifest.domain,
            resolution,
        );
        request.truncation = Some(bundle.manifest.truncation.clone());
        request.layers = Some(bundle.manifest.layers.clone());
        request.name = string("name").unwrap_or_default();
        request.spacing = None;
        self.landscape = Some(request.clone());
        self.shown = Some(request);
    }

    // MARK: - restyling

    /// The styling the controls show: the latest asked for, else the bundle's.
    pub fn styling(&self) -> Option<Styling> {
        self.styled.clone().or_else(|| self.bundle().map(|b| Styling::of(&b.manifest)))
    }

    /// Edit the styling and re-derive the curves from the grids already in
    /// memory. The landscape is unchanged -- the samples are the samples -- so
    /// the camera, the framing and the window all stay exactly as they are.
    pub fn restyle(&mut self, change: impl FnOnce(&mut Manifest)) {
        let (Some(bundle), true, Some(before)) = (self.bundle(), self.navigator.is_some(), self.styling()) else {
            return;
        };
        let mut manifest = bundle.manifest.clone();
        before.apply(&mut manifest);
        change(&mut manifest);
        let after = Styling::of(&manifest);
        if after == before {
            return;
        }
        self.styled = Some(after.clone());
        // The next resample keeps the truncation given here.
        if let Some(landscape) = &mut self.landscape {
            landscape.truncation = Some(after.truncation);
        }
        self.redraw();
    }

    pub fn truncation(&self) -> Truncation {
        crate::document::truncation_of(self.styling().as_ref())
    }

    /// Switch the contour refinement on or off for the open landscape: the
    /// same grids, the curves derived again, the camera where it was.
    pub fn set_refinement(&mut self, on: bool) {
        self.refine_contours = on;
        self.redraw();
    }

    /// The one job that derives curves, coalesced like the resample: one in
    /// flight, and when it lands, whatever was asked for meanwhile. A restyle
    /// or a change of refinement comes first, since it draws every layer; a
    /// layer's levels after, on the bundle that results.
    pub fn redraw(&mut self) {
        if self.surface_bundle().is_some() {
            return self.redraw_surface();
        }
        if self.polyhedron_bundle().is_some() {
            return self.redraw_polyhedron();
        }
        if self.drawing {
            return;
        }
        let (Some(bundle), Some(target)) = (self.bundle().cloned(), self.styling()) else { return };
        let current = Styling::of(&bundle.manifest);
        if self.styled.as_ref() == Some(&current) {
            self.styled = None;
        }
        // A draft's contours are the grid's on purpose; the full build that
        // follows the drag places them.
        let refine = self.refine_contours && self.is_landscape() && !self.shown_draft;
        let refinement_moves = self.is_landscape() && !self.shown_draft && bundle.refine.is_some() != refine;
        let job = if target != current || refinement_moves {
            DrawJob::Restyle(target, if refinement_moves { Some(refine) } else { None })
        } else if let (Some(draft), true) = (self.curves_wanted, self.scene.is_some()) {
            DrawJob::Levels(draft)
        } else {
            return;
        };
        if matches!(job, DrawJob::Levels(_)) {
            self.curves_wanted = None;
        }
        self.drawing = true;
        let level_sets = (0..bundle.manifest.layers.len()).map(|i| self.levels_for_layer(i)).collect();
        self.jobs.push_back(Job::Draw {
            job,
            bundle,
            level_sets,
            cached: self.derived_curves.clone(),
            asked: self.version,
        });
    }

    /// A restyle or a layer's levels landed (the rest of Swift's `redraw`
    /// task).
    pub(crate) fn drew(&mut self, job: DrawJob, bundle: &KurvenBundle, asked: u64, drawn: Drawn) {
        self.drawing = false;
        if self.version == asked {
            self.install(drawn);
            if let DrawJob::Restyle(_, Some(true)) = job
                && bundle.refine.is_none()
                && self.bundle().is_some_and(|b| b.refine.is_none())
            {
                // Asked to refine a landscape whose function would not
                // compile again: say so on the toggle, rather than ask forever.
                self.refine_contours = false;
            }
        }
        self.redraw();
    }

    fn install(&mut self, drawn: Drawn) {
        match drawn {
            Drawn::Restyled(restyled) => {
                self.replace(restyled.clone());
                self.derived_curves.clear();
                if self.styled.as_ref() == Some(&Styling::of(&restyled.manifest)) {
                    self.styled = None;
                }
                let (Some(navigator), Some(preset)) = (self.navigator, restyled.manifest.presets.first()) else {
                    return;
                };
                let scene = kurven_core::scene::Scene::from_bundle(&restyled, preset).looking(navigator.camera());
                // Hidden layers are hidden by name: a layer added or taken off
                // moves the others.
                let old = self.scene.as_ref();
                let hidden: std::collections::BTreeSet<String> = self
                    .hidden_layers
                    .iter()
                    .filter_map(|&i| old.and_then(|s| s.layers().get(i)).map(|l| l.spec.name.clone()))
                    .collect();
                self.hidden_layers =
                    (0..scene.layers().len()).filter(|&i| hidden.contains(&scene.layers()[i].spec.name)).collect();
                self.set_scene(Some(scene));
                // A layer drawn at levels of its own is drawn at them again.
                if !self.level_counts.is_empty() {
                    self.curves_wanted = Some(self.curves_wanted.unwrap_or(false));
                }
            }
            Drawn::Stroked(layers, derived) => {
                self.derived_curves = derived;
                if let Some(scene) = &self.scene {
                    let next = scene.drawing(layers);
                    self.set_scene(Some(next));
                }
            }
        }
    }

    /// Truncate the landscape differently.
    ///
    /// Everything follows: the heightfield the occluder meshes, the top edge
    /// each wall stroke rises to, where the plateaus are and therefore every
    /// truncation stroke and truncation edge, and which contours are cut off
    /// by `Restriction::BelowTruncation`. All of it is derived, so all of it
    /// is this one assignment.
    pub fn set_truncation(&mut self, truncation: Truncation) {
        // Overrides are of the levels the truncation chose; a new truncation
        // chooses again.
        self.level_counts.clear();
        let is_landscape = self.is_landscape();
        self.restyle(|manifest| {
            // A banded truncation's risers are hatched as the walls are, and
            // under any other there are none.
            if is_banded(&manifest.truncation) != is_banded(&truncation) {
                let walls = match &manifest.occluder.walls {
                    Walls::Perimeter { perimeter, .. } => Some(perimeter.edges().len()),
                    _ => None,
                };
                manifest.layers = NativeLandscape::riser_hatching(manifest.layers.clone(), walls, &truncation);
            }
            manifest.truncation = truncation.clone();
            // The levels are absolute, so a truncation raised past them would
            // leave the top of the landscape blank. `LandscapeStyle` is the
            // rule the build used to choose them in the first place, applied
            // again to the new ceiling -- by name, because in a landscape's
            // bundle those two names are what the two magnitude families are
            // called.
            if !is_landscape || LandscapeStyle::ceiling(&truncation).is_none() {
                return;
            }
            let (major, minor) = LandscapeStyle::levels(&truncation);
            for layer in &mut manifest.layers {
                let LayerSource::Contour { field, restriction, faces, light, .. } = &layer.source else { continue };
                let chosen = match layer.name.as_str() {
                    "mag_major" => major.clone(),
                    "mag_minor" => minor.clone(),
                    _ => continue,
                };
                layer.source = LayerSource::Contour {
                    field: *field,
                    levels: chosen,
                    restriction: restriction.clone(),
                    faces: faces.clone(),
                    light: *light,
                };
            }
        });
    }

    /// The hatch spacing of one described hatching layer, in world units.
    pub fn spacing_of_layer(&self, index: usize) -> Option<f64> {
        let layers = if self.polyhedron_bundle().is_some() {
            self.polyhedron_layers()
        } else {
            self.styling().map(|s| s.layers)
        }?;
        match &layers.get(index)?.source {
            LayerSource::Hatching { hatching, .. } => Some(hatching.spacing),
            _ => None,
        }
    }

    /// How many gaps a hatching layer rules along the domain's longer side,
    /// the side `LandscapeStyle::hatch_spacing` counts its seventy strokes
    /// on; on a polyhedron, across its largest face.
    pub fn hatch_count(&self, index: usize) -> Option<i64> {
        let spacing = self.spacing_of_layer(index)?;
        let reach = self.hatch_reach()?;
        if !(spacing > 0.0) {
            return None;
        }
        // Swift's `Int((reach / spacing).rounded())`.
        Some(kurven_math::swift::int((reach / spacing).round())?.max(1))
    }

    pub fn set_hatch_count(&mut self, count: i64, index: usize) {
        let Some(reach) = self.hatch_reach() else { return };
        if count <= 0 {
            return;
        }
        self.set_spacing(reach / count as f64, index);
    }

    fn hatch_reach(&self) -> Option<f64> {
        if let Some(b) = self.polyhedron_bundle() {
            return b.solid().frames.iter().map(|f| f.extent).fold(None, |m: Option<f64>, e| {
                Some(match m {
                    Some(m) => kurven_math::swift::max(m, e),
                    None => e,
                })
            });
        }
        let domain = self.bundle()?.manifest.domain;
        let reach = kurven_math::swift::max(domain.real.length().abs(), domain.imag.length().abs());
        (reach > 0.0).then_some(reach)
    }

    pub fn set_spacing(&mut self, value: f64, index: usize) {
        if self.polyhedron_bundle().is_some() {
            return self.restyle_polyhedron(|layers| {
                let Some(layer) = layers.get_mut(index) else { return };
                let LayerSource::Hatching { hatching, faces } = &layer.source else { return };
                let (mut hatching, faces) = (*hatching, faces.clone());
                // A tone read as finely as the lines are drawn, the default,
                // stays so.
                if hatching.step == hatching.spacing {
                    hatching.step = value;
                }
                hatching.spacing = value;
                layer.source = LayerSource::Hatching { hatching, faces };
            });
        }
        self.restyle(|manifest| {
            let Some(layer) = manifest.layers.get_mut(index) else { return };
            let LayerSource::Hatching { hatching, faces } = &layer.source else { return };
            let (mut hatching, faces) = (*hatching, faces.clone());
            hatching.spacing = value;
            layer.source = LayerSource::Hatching { hatching, faces };
        });
    }

    /// Which way the strokes on the plateaus run.
    pub fn plateau_hatch_axis(&self) -> Option<DomainAxis> {
        self.styling()?.layers.iter().find_map(|spec| spec.source.plateau_hatch_axis())
    }

    pub fn set_plateau_hatch_axis(&mut self, axis: DomainAxis) {
        self.restyle(|manifest| {
            for layer in &mut manifest.layers {
                if let LayerSource::Hatching {
                    hatching,
                    faces: kurven_core::hatching::FaceSelector::Plateaus { tiled },
                } = &layer.source
                {
                    let (mut hatching, tiled) = (*hatching, *tiled);
                    hatching.angle = axis.angle();
                    layer.source = LayerSource::Hatching {
                        hatching,
                        faces: kurven_core::hatching::FaceSelector::Plateaus { tiled },
                    };
                }
            }
        });
    }

    // MARK: - keeping one

    /// What is on screen as a `.kurven` bundle's files: a landscape, a
    /// parametric surface or a polyhedron, each keeping its layers as they
    /// are drawn, at the levels their sliders set (`drawn_layers`). Swift's
    /// `saveBundle(to:)`, which writes them to a URL; `name` is the bundle's,
    /// for its status and a refusal. Refused with the reason: nothing open,
    /// or a bundle that cannot be written (dumped layers), which the status
    /// says too, as Swift's does.
    pub fn save_bundle(&mut self, name: &str) -> Result<BundleFiles, String> {
        let result = match &self.state {
            State::Ready(Opened::Landscape(bundle)) => {
                let mut manifest = bundle.manifest.clone();
                manifest.layers = self.drawn_layers();
                let saved =
                    KurvenBundle::new(bundle.url.clone(), manifest, bundle.surface.clone(), Vec::new(), None, None);
                saved.files(name).map_err(|e| e.to_string())
            }
            State::Ready(Opened::Surface(bundle)) => Ok(bundle.files(Some(&self.drawn_layers()))),
            State::Ready(Opened::Polyhedron(bundle)) => Ok(bundle.files()),
            State::Empty => return Err("nothing is open to save".to_string()),
            State::Failed(_, why) => return Err(format!("nothing to save: the bundle did not open ({why})")),
        };
        let short =
            std::path::Path::new(name).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let status = match &result {
            Ok(_) => format!("saved {short}"),
            Err(error) => format!("could not save: {error}"),
        };
        if self.bundle().is_some() {
            self.landscape_status = Some(status);
        } else if self.polyhedron_bundle().is_some() {
            self.polyhedron_status = Some(status);
        } else {
            self.surface_status = Some(status);
        }
        result
    }
}

fn is_banded(truncation: &Truncation) -> bool {
    matches!(truncation, Truncation::RealBands { .. })
}

/// What a draw job made.
pub(crate) enum Drawn {
    Restyled(KurvenBundle),
    Stroked(Vec<kurven_core::polyline::Layer>, std::collections::BTreeMap<usize, crate::document::DerivedCurves>),
}

/// A restyle of `bundle` under `styling`, refined first or not as `refine`
/// asks: the detached half of Swift's restyle job.
pub(crate) fn restyled(bundle: &KurvenBundle, styling: &Styling, refine: Option<bool>) -> KurvenBundle {
    let mut b = bundle.clone();
    if let Some(refine) = refine {
        let hook = if refine { NativeLandscape::refiner_for_bundle(&b).map(|r| r.refine_hook()) } else { None };
        b = b.refined(hook);
    }
    let mut manifest = b.manifest.clone();
    styling.apply(&mut manifest);
    b.restyled(&manifest)
}
