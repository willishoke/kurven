//! The half of a document that is a parametric surface: the port of
//! `Sources/KurvenApp/Surfaces.swift`.
//!
//! Built as a landscape is sampled -- one build in flight, the latest edit
//! queued behind it -- through `SurfaceDrawing::build`, the function
//! `kurven-cli surface` calls, so the window and the CLI draw the same
//! drawing. The drawing on screen is a bundle (`SurfaceBundle`), the request
//! that builds it and its layers, so it is saved, restyled and baked as a
//! landscape is: a request edit here is the surface's resample, and a layer's
//! levels its restyle (`redraw_surface`).

use kurven_core::manifest::LayerSpec;
use kurven_core::surface_request::{Shape, SurfaceRequest};
use kurven_surfaces::bundle::SurfaceBundle;
use kurven_surfaces::drawing::{SurfaceDrawing, SurfacePreset};

use crate::document::Document;
use crate::edits::REBUILT_GIT_SHA;
use crate::jobs::Job;

impl Document {
    /// Start a surface from the catalog, replacing whatever is open.
    pub fn create_surface(&mut self, preset: &SurfacePreset) {
        self.generation += 1;
        self.wanted = None;
        self.surface_styled = None;
        self.forget_edits();
        self.surface = Some(preset.request.clone());
        self.send_surface(preset.request.clone(), true, false);
    }

    /// The controls changed `surface`. `draft` is true while one is being
    /// dragged: the drawing is built from `SurfaceRequest::drafted` until the
    /// drag ends, and at full resolution after.
    pub fn surface_edited(&mut self, draft: bool) {
        let Some(request) = self.surface.clone() else { return };
        self.send_surface(if draft { request.drafted() } else { request }, false, draft);
    }

    fn send_surface(&mut self, request: SurfaceRequest, framing: bool, draft: bool) {
        self.surface_wanted = Some((request, framing, draft));
        self.pump_surface();
    }

    /// Paint layers on the surface: the request's own curves follow it
    /// (`SurfaceBundle::restyled`, through `SurfaceDrawing::keeping`), and any
    /// other layer is drawn on the surface as it is (`redraw_surface`), or on
    /// the one a request edit in flight will build.
    pub fn restyle_surface(&mut self, layers: Vec<LayerSpec>) {
        self.surface_styled = Some(layers);
        self.redraw_surface();
    }

    pub(crate) fn pump_surface(&mut self) {
        if self.building {
            return;
        }
        let Some((request, framing, draft)) = self.surface_wanted.take() else { return };
        if !framing && Some(&request) == self.surface_shown.as_ref() {
            return;
        }
        // Whatever the drawing on screen can lend the next one -- a fitted
        // torus, its values on the grid, the trajectory's run, the contours
        // -- goes with the request.
        let previous = if framing { None } else { self.surface_drawing().cloned() };
        // The layers the bundle keeps -- its levels, a contour the request
        // does not draw -- survive the edit (`SurfaceDrawing::keeping`), or
        // the layers asked for since, which the edit then draws.
        let kept = if framing {
            None
        } else {
            self.surface_styled.clone().or_else(|| self.surface_bundle().map(|b| b.manifest.layers.clone()))
        };
        // Made again here, so recorded as this process's, unless an undo
        // asks for the bundle it put back.
        let git_sha = self.surface_git_sha.clone().unwrap_or_else(|| REBUILT_GIT_SHA.to_string());
        self.building = true;
        let refits = previous.as_ref().map(|p| p.refits(&request)).unwrap_or(true);
        self.surface_status = Some(
            if refits && matches!(request.shape, Shape::Forced { .. }) {
                "fitting the torus…"
            } else {
                "building…"
            }
            .to_string(),
        );
        self.jobs.push_back(Job::BuildSurface {
            request,
            framing,
            draft,
            previous,
            kept,
            git_sha,
            asked: self.generation,
        });
    }

    /// A build landed (the rest of Swift's `pumpSurface` task).
    pub(crate) fn built_surface(
        &mut self,
        request: SurfaceRequest,
        framing: bool,
        draft: bool,
        git_sha: String,
        asked: u64,
        result: Result<SurfaceDrawing, String>,
    ) {
        match result {
            Ok(built) => {
                // A landscape or another surface chosen meanwhile is the
                // document now.
                if self.generation == asked {
                    let keeping = !framing && self.scene.is_some();
                    let specs: Vec<LayerSpec> = built.layers.iter().map(|l| l.spec.clone()).collect();
                    let status = status_of(&built) + if draft { " (draft)" } else { "" };
                    self.adopt_surface_drawing(built, keeping, &git_sha);
                    if self.surface_git_sha.as_deref() == Some(git_sha.as_str()) {
                        self.surface_git_sha = None;
                    }
                    self.surface_shown = Some(request);
                    if self.surface_styled.as_ref() == Some(&specs) {
                        self.surface_styled = None;
                    }
                    self.surface_status = Some(status);
                }
            }
            Err(error) => {
                if self.generation == asked {
                    self.surface_status = Some(error);
                }
            }
        }
        self.building = false;
        self.pump_surface();
        self.record_surface_git_sha();
        // Layers or levels asked for while it built are drawn on it.
        if !self.building {
            self.redraw_surface();
        }
    }

    /// When the surface on screen is the one asked for, record the git sha
    /// asked for (`surface_git_sha`): an undo's, or a drag's let go where it
    /// began, rather than this process's.
    pub(crate) fn record_surface_git_sha(&mut self) {
        let Some(want) = self.surface_git_sha.clone() else { return };
        if self.building || self.surface_wanted.is_some() {
            return;
        }
        let Some(bundle) = self.surface_bundle().cloned() else { return };
        if Some(&bundle.manifest.surface) != self.surface.as_ref() {
            return;
        }
        if bundle.manifest.provenance.git_sha != want {
            self.adopt_restyled(bundle.recording(&want));
        }
        self.surface_git_sha = None;
    }

    /// A parametric surface's layers at the levels their sliders set, derived
    /// and coalesced as a landscape's are (`redraw`): each layer at levels of
    /// its own is drawn again on the surface (`SurfaceDrawing::restyled`), the
    /// rest kept. A draft is drawn in full: |f| and arg f are placed by the
    /// function either way.
    pub(crate) fn redraw_surface(&mut self) {
        if self.drawing || self.scene.is_none() {
            return;
        }
        let Some(bundle) = self.surface_bundle().cloned() else { return };
        if let Some(layers) = self.surface_styled.clone() {
            // A request edit in flight draws them on what it builds.
            if self.building {
                return;
            }
            if layers == bundle.manifest.layers {
                self.surface_styled = None;
                return self.redraw_surface();
            }
            self.drawing = true;
            self.jobs.push_back(Job::RestyleSurface { bundle, layers, asked: self.version });
            return;
        }
        if self.curves_wanted.is_none() {
            return;
        }
        self.curves_wanted = None;
        self.drawing = true;
        self.jobs.push_back(Job::StrokeSurface {
            drawing: bundle.drawing.clone(),
            layers: self.drawn_layers(),
            asked: self.version,
        });
    }

    /// A restyle of the surface landed.
    pub(crate) fn restyled_surface(
        &mut self,
        layers: Vec<LayerSpec>,
        asked: u64,
        result: Result<SurfaceBundle, String>,
    ) {
        self.drawing = false;
        if self.version == asked {
            match result {
                Ok(bundle) => {
                    self.adopt_restyled(bundle);
                    if self.surface_styled.as_ref() == Some(&layers) {
                        self.surface_styled = None;
                    }
                }
                Err(error) => {
                    self.surface_status = Some(error);
                    if self.surface_styled.as_ref() == Some(&layers) {
                        self.surface_styled = None;
                    }
                }
            }
        }
        self.redraw_surface();
    }

    /// A surface's layers at their levels landed.
    pub(crate) fn stroked_surface(&mut self, asked: u64, result: Result<SurfaceDrawing, String>) {
        self.drawing = false;
        if self.version == asked {
            match result {
                Ok(drawn) => {
                    if let Some(scene) = &self.scene {
                        let next = scene.drawing(drawn.layers);
                        self.set_scene(Some(next));
                    }
                }
                Err(error) => self.surface_status = Some(error),
            }
        }
        self.redraw_surface();
    }
}

/// What the section's status line says about a drawing: anything about it
/// that changes what the drawing means. Swift's also says how long it took;
/// there is no clock in a wasm module.
fn status_of(drawing: &SurfaceDrawing) -> String {
    let mut text = "built".to_string();
    if let Some(torus) = drawing.torus() {
        text += &format!("; misses its trajectory by {:.1e} of its size", torus.residual / torus.extent);
    }
    if !drawing.embedded {
        text += " — the revolution passes through itself, so back faces are kept";
    } else if drawing.surface.outward.is_none() {
        text += " — the torus reaches its axis and bounds nothing, so back faces are kept";
    }
    text
}
