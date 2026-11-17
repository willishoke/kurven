//! What an edit changes and a save writes, and its undo: the port of
//! `Sources/KurvenApp/Edits.swift`.

use std::collections::BTreeMap;

use kurven_core::manifest::{LayerSpec, PolyhedronManifest, Truncation};
use kurven_core::surface_request::SurfaceRequest;
use kurven_landscape::catalog::LandscapeRequest;
use kurven_surfaces::drawing::SurfaceRequestExt;

use crate::document::{Document, Opened, State};

/// What an edit changes and a save writes: the layers, a parametric
/// surface's request, which draws some of them and carries the function, a
/// landscape's request and truncation, the level counts the sliders set, and
/// the git sha the bundle records. One value, so every edit -- painting, a
/// truncation, a surface's numbers, a landscape's window -- is one step to
/// undo, the steps unwind in order, and undoing one puts back exactly what a
/// save wrote before it.
#[derive(Debug, Clone, PartialEq)]
pub struct Edited {
    pub layers: Vec<LayerSpec>,
    pub request: Option<SurfaceRequest>,
    /// A landscape's request: its function, window and resolution. None for
    /// a landscape opened read-only, and for every other kind of bundle.
    pub landscape: Option<LandscapeRequest>,
    pub truncation: Option<Truncation>,
    pub level_counts: BTreeMap<String, i64>,
    /// The opened bundle's, until a request edit makes the drawing again in
    /// this process, which records its own.
    pub git_sha: String,
}

/// The git sha a drawing made again in this process records.
pub const REBUILT_GIT_SHA: &str = "native";

/// One step to undo or redo: running it puts `to` back, and files the
/// opposite step on the other stack, as the closure Swift's `registerUndo`
/// files does.
#[derive(Debug, Clone)]
pub(crate) struct Step {
    pub to: Edited,
    pub from: Edited,
    pub name: String,
}

/// Swift's `UndoManager`, as the document uses it: one registration per
/// event, each its own group, so a step is one registration. A new step
/// clears what could be redone.
#[derive(Debug, Clone, Default)]
pub struct UndoStack {
    pub(crate) undo: Vec<Step>,
    pub(crate) redo: Vec<Step>,
    /// Inside an undo or a redo: a registration files the opposite step.
    pub(crate) undoing: bool,
    pub(crate) redoing: bool,
}

impl UndoStack {
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    /// How many steps ⌘Z could undo.
    pub fn steps(&self) -> usize {
        self.undo.len()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// The name the Edit menu gives the step ⌘Z would undo.
    pub fn undo_name(&self) -> Option<&str> {
        self.undo.last().map(|s| s.name.as_str())
    }

    pub fn redo_name(&self) -> Option<&str> {
        self.redo.last().map(|s| s.name.as_str())
    }
}

impl Document {
    // MARK: - the state

    /// What is edited now: the layers last asked for, which a restyle in
    /// flight is still drawing, and the requests the controls say.
    pub fn edited(&self) -> Option<Edited> {
        match &self.state {
            State::Ready(Opened::Landscape(bundle)) => {
                // Sampled again when the request asks for other samples than
                // the bundle's, and recorded then as this process's.
                let resampled = self
                    .landscape
                    .as_ref()
                    .is_some_and(|l| !self.shown.as_ref().map(|s| s.samples_as(l)).unwrap_or(true));
                let styling = self.styling();
                Some(Edited {
                    layers: styling.as_ref().map(|s| s.layers.clone()).unwrap_or_default(),
                    request: None,
                    landscape: self.landscape.clone(),
                    truncation: styling.map(|s| s.truncation),
                    level_counts: self.level_counts.clone(),
                    git_sha: if resampled {
                        self.landscape_git_sha.clone().unwrap_or_else(|| REBUILT_GIT_SHA.to_string())
                    } else {
                        bundle.manifest.provenance.git_sha.clone()
                    },
                })
            }
            State::Ready(Opened::Surface(bundle)) => {
                let request = self.surface.clone().unwrap_or_else(|| bundle.manifest.surface.clone());
                let git_sha = if request == bundle.manifest.surface {
                    bundle.manifest.provenance.git_sha.clone()
                } else {
                    self.surface_git_sha.clone().unwrap_or_else(|| REBUILT_GIT_SHA.to_string())
                };
                Some(Edited {
                    layers: self.surface_styled.clone().unwrap_or_else(|| bundle.manifest.layers.clone()),
                    request: Some(request),
                    landscape: None,
                    truncation: None,
                    level_counts: self.level_counts.clone(),
                    git_sha,
                })
            }
            State::Ready(Opened::Polyhedron(bundle)) => Some(Edited {
                layers: self.polyhedron_layers().unwrap_or_default(),
                request: None,
                landscape: None,
                truncation: None,
                level_counts: self.level_counts.clone(),
                git_sha: bundle.manifest.provenance.git_sha.clone(),
            }),
            State::Empty | State::Failed(..) => None,
        }
    }

    /// Make `next` what is edited: the layers restyled from what is built, a
    /// request edit built or sampled as the controls build one, the levels
    /// drawn again.
    pub(crate) fn set_edited(&mut self, next: Edited) {
        let Some(current) = self.edited() else { return };
        if next == current {
            return;
        }
        let levels_moved = next.level_counts != current.level_counts;
        self.level_counts = next.level_counts.clone();
        match &self.state {
            State::Ready(Opened::Landscape(_)) => {
                if next.landscape != current.landscape {
                    self.landscape = next.landscape.clone();
                }
                if next.layers != current.layers || next.truncation != current.truncation {
                    let (layers, truncation) = (next.layers.clone(), next.truncation.clone());
                    self.restyle(|m| {
                        m.layers = layers;
                        if let Some(truncation) = truncation {
                            m.truncation = truncation;
                        }
                    });
                }
                if let (Some(request), Some(before)) = (&next.landscape, &current.landscape)
                    && !request.samples_as(before)
                {
                    self.landscape_git_sha = Some(next.git_sha.clone());
                    self.landscape_edited(false);
                }
            }
            State::Ready(Opened::Surface(_)) => {
                if let Some(request) = &next.request
                    && Some(request) != current.request.as_ref()
                {
                    // The layers the build keeps; asked for only when they
                    // moved, since the request's own curves in them are the
                    // old request's, which the build replaces (`keeping`).
                    if next.layers != current.layers {
                        self.surface_styled = Some(next.layers.clone());
                    }
                    self.surface_git_sha = Some(next.git_sha.clone());
                    self.surface = Some(request.clone());
                    self.surface_edited(false);
                } else if next.layers != current.layers {
                    self.restyle_surface(next.layers.clone());
                }
            }
            State::Ready(Opened::Polyhedron(_)) => {
                if next.layers != current.layers {
                    let layers = next.layers.clone();
                    self.restyle_polyhedron(|l| *l = layers);
                }
            }
            State::Empty | State::Failed(..) => return,
        }
        if levels_moved {
            self.rebuild_curves(false);
        }
    }

    /// One edit, one undo step, named in the Edit menu. An edit the document
    /// cannot take -- more hatch lines than a polyhedron holds, a layer a
    /// surface cannot draw -- is refused, saying why, and leaves no step.
    pub fn edit<E>(&mut self, name: &str, change: impl FnOnce(&mut Edited) -> Result<(), E>) -> Result<(), E> {
        let Some(before) = self.edited() else { return Ok(()) };
        let mut after = before.clone();
        change(&mut after)?;
        if after.request != before.request || !samples_alike(after.landscape.as_ref(), before.landscape.as_ref()) {
            after.git_sha = REBUILT_GIT_SHA.to_string();
        }
        if after == before {
            return Ok(());
        }
        if let Some(why) = self.refusal(&after) {
            self.paint_status = Some(why);
            return Ok(());
        }
        self.register_undo(before, after.clone(), name);
        self.set_edited(after);
        Ok(())
    }

    /// `edit`, for a change that cannot fail.
    pub fn edit_infallible(&mut self, name: &str, change: impl FnOnce(&mut Edited)) {
        let _: Result<(), std::convert::Infallible> = self.edit(name, |e| {
            change(e);
            Ok(())
        });
    }

    /// A slider taken hold of: what to undo back to when it is let go.
    pub fn begin_drag(&mut self) {
        self.edited_before_drag = self.edited();
    }

    /// A slider let go: the whole drag is one undo step, unless nothing it
    /// asked for was taken.
    ///
    /// A drag let go where it began registers nothing, and what it made
    /// again on the way records what the document was opened as: the
    /// request is the same, so the bundle saved is the same.
    pub fn end_drag(&mut self, name: &str) {
        let before = self.edited_before_drag.take();
        let (Some(before), Some(mut after)) = (before, self.edited()) else { return };
        if after.request == before.request && samples_alike(after.landscape.as_ref(), before.landscape.as_ref()) {
            after.git_sha = before.git_sha.clone();
            if before.request.is_some() {
                self.surface_git_sha = Some(before.git_sha.clone());
                self.record_surface_git_sha();
            } else if before.landscape.is_some() {
                // Sampled in full when it is let go, which records it.
                self.landscape_git_sha = Some(before.git_sha.clone());
            }
        }
        if before == after || self.refusal(&after).is_some() {
            return;
        }
        self.register_undo(before, after, name);
    }

    /// `body` as one undo step: a drag of one step, for an edit made by
    /// calls a drag also makes. Inside a drag, a step of it.
    pub fn editing(&mut self, name: &str, body: impl FnOnce(&mut Document)) {
        if self.edited_before_drag.is_some() {
            return body(self);
        }
        self.begin_drag();
        body(self);
        self.end_drag(name);
    }

    /// Why the document cannot take `next`, or None: what its manifest's
    /// reader would refuse.
    pub fn refusal(&self, next: &Edited) -> Option<String> {
        match &self.state {
            State::Ready(Opened::Polyhedron(bundle)) => PolyhedronManifest::unsupported(&next.layers, bundle.solid()),
            State::Ready(Opened::Surface(_)) => {
                let request = next.request.as_ref()?;
                if let Err(error) = request.check(&kurven_core::surface_request::surface_key) {
                    return Some(error.to_string());
                }
                let mut names = std::collections::HashSet::new();
                for layer in &next.layers {
                    if let Some(why) = request.unsupported(&layer.source) {
                        return Some(format!("layer '{}': {why}", layer.name));
                    }
                    if !names.insert(layer.name.clone()) {
                        return Some(format!("two layers named '{}'", layer.name));
                    }
                }
                None
            }
            State::Ready(Opened::Landscape(_)) | State::Empty | State::Failed(..) => None,
        }
    }

    /// ⌘Z puts `before` back, and ⌘⇧Z `after`.
    fn register_undo(&mut self, before: Edited, after: Edited, name: &str) {
        let step = Step { to: before, from: after, name: name.to_string() };
        if self.undo.undoing {
            self.undo.redo.push(step);
        } else {
            self.undo.undo.push(step);
            if !self.undo.redoing {
                self.undo.redo.clear();
            }
        }
    }

    pub fn undo_stack(&self) -> &UndoStack {
        &self.undo
    }

    /// ⌘Z: the last step put back. False when there is none.
    pub fn undo(&mut self) -> bool {
        let Some(step) = self.undo.undo.pop() else { return false };
        self.undo.undoing = true;
        self.register_undo(step.from, step.to.clone(), &step.name);
        self.set_edited(step.to);
        self.undo.undoing = false;
        true
    }

    /// ⌘⇧Z: the last step undone made again. False when there is none.
    pub fn redo(&mut self) -> bool {
        let Some(step) = self.undo.redo.pop() else { return false };
        self.undo.redoing = true;
        self.register_undo(step.from, step.to.clone(), &step.name);
        self.set_edited(step.to);
        self.undo.redoing = false;
        true
    }

    /// A different document: nothing done to the last one can be undone on
    /// it, and nothing of it is picked.
    pub fn forget_edits(&mut self) {
        self.undo = UndoStack::default();
        self.edited_before_drag = None;
        self.paint_status = None;
        self.picked = None;
        self.selection = None;
        self.surface_git_sha = None;
        self.landscape_git_sha = None;
    }

    // MARK: - the controls' edits

    /// A surface's request as its controls set it, one undo step.
    pub fn edit_surface(&mut self, next: SurfaceRequest, name: &str) {
        self.edit_infallible(name, |e| e.request = Some(next));
    }

    /// A step of a drag of a surface's control: built as a draft, or in full.
    pub fn drag_surface(&mut self, next: SurfaceRequest, draft: bool) {
        self.surface_git_sha = None;
        self.surface = Some(next);
        self.surface_edited(draft);
    }

    /// A landscape's request as its controls set it, one undo step.
    pub fn edit_landscape(&mut self, name: &str, change: impl FnOnce(&mut LandscapeRequest)) {
        self.edit_infallible(name, |edited| {
            let Some(mut request) = edited.landscape.clone() else { return };
            change(&mut request);
            edited.landscape = Some(request);
        });
    }

    /// A step of a drag of a landscape's control: sampled at the draft
    /// resolution, or in full.
    pub fn drag_landscape(&mut self, draft: bool) {
        self.landscape_git_sha = None;
        self.landscape_edited(draft);
    }

    /// A truncation typed, chosen or stepped: one undo step, with what it
    /// does to the levels and the risers' hatching (`set_truncation`).
    pub fn edit_truncation(&mut self, truncation: Truncation, name: &str) {
        self.editing(name, |d| d.set_truncation(truncation));
    }
}

pub(crate) fn samples_alike(a: Option<&LandscapeRequest>, b: Option<&LandscapeRequest>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a.samples_as(b),
        (a, b) => a.is_none() && b.is_none(),
    }
}
