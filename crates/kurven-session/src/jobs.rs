//! The document's slow work, queued and landed in order.
//!
//! Swift's document starts a detached task for each of these and lands its
//! result on the main actor, where it can never interleave with an edit.
//! Here a job holds what its task captured; `settle` runs each in the order
//! it was started, then lands it as Swift's continuation does, and a landing
//! may start more. In a Worker an edit is a message, and the reply goes once
//! the document has settled, as Swift's headless run waits for it to settle
//! after every edit (`settle()` in `App.swift`'s `headlessPaint`).
//!
//! Swift's tasks run concurrently, so two in flight at once -- a restyle and
//! a sampling started by one edit -- land in whichever order they finish.
//! Here they land in the order they were started, which is the order the
//! faster of the two finishes in every case the document meets: a restyle
//! re-derives from grids already in memory, a sampling evaluates f again.

use std::collections::BTreeMap;
use std::sync::Arc;

use kurven_core::bundle::KurvenBundle;
use kurven_core::face::Face;
use kurven_core::manifest::LayerSpec;
use kurven_core::pick::FacePick;
use kurven_core::polyhedron_bundle::PolyhedronBundle;
use kurven_core::scene::{ContentID, SceneGeometry};
use kurven_core::surface_request::SurfaceRequest;
use kurven_landscape::catalog::LandscapeRequest;
use kurven_landscape::landscape::NativeLandscape;
use kurven_surfaces::bundle::SurfaceBundle;
use kurven_surfaces::drawing::SurfaceDrawing;

use crate::document::{DerivedCurves, Document};
use crate::landscape::{Drawn, Styling};

/// A landscape's draw job: a restyle under a styling, refined first or not
/// when `refine` says, or a layer's levels, a draft's or in full.
#[derive(Debug, Clone)]
pub(crate) enum DrawJob {
    Restyle(Styling, Option<bool>),
    Levels(bool),
}

/// What a face's boundary's status says once it lands.
#[derive(Debug, Clone)]
pub(crate) enum BoundaryStatus {
    /// A click: the face, the point it was picked at.
    Picked { point: [f64; 3] },
    /// A pick carried across a new scene.
    Kept,
}

/// One piece of slow work and what it captured.
pub(crate) enum Job {
    Sample {
        request: LandscapeRequest,
        framing: bool,
        draft: bool,
        asked: u64,
        git_sha: String,
        refining: bool,
    },
    Draw {
        job: DrawJob,
        bundle: KurvenBundle,
        level_sets: Vec<Option<Vec<f64>>>,
        cached: BTreeMap<usize, DerivedCurves>,
        asked: u64,
    },
    BuildSurface {
        request: SurfaceRequest,
        framing: bool,
        draft: bool,
        previous: Option<SurfaceDrawing>,
        kept: Option<Vec<LayerSpec>>,
        git_sha: String,
        asked: u64,
    },
    RestyleSurface {
        bundle: SurfaceBundle,
        layers: Vec<LayerSpec>,
        asked: u64,
    },
    StrokeSurface {
        drawing: SurfaceDrawing,
        layers: Vec<LayerSpec>,
        asked: u64,
    },
    RestylePolyhedron {
        bundle: PolyhedronBundle,
        layers: Vec<LayerSpec>,
        asked: u64,
    },
    Boundary {
        pick: FacePick,
        geometry: SceneGeometry,
        faces: Arc<Vec<Face>>,
        generation: u64,
        content: ContentID,
        status: BoundaryStatus,
    },
}

impl Document {
    /// Whether anything is in flight: what Swift's headless `settle()` waits
    /// on, with `is_current`.
    pub fn is_busy(&self) -> bool {
        !self.jobs.is_empty() || self.building || self.drawing || self.picking || self.sampling
    }

    /// Run every job started, in order, landing each, until none is left.
    pub fn settle(&mut self) {
        while let Some(job) = self.jobs.pop_front() {
            self.run(job);
        }
    }

    fn run(&mut self, job: Job) {
        match job {
            Job::Sample { request, framing, draft, asked, git_sha, refining } => {
                let result = NativeLandscape::build(&request, &git_sha, refining).map_err(|e| e.to_string());
                self.sampled(request, framing, draft, asked, git_sha, result);
            }
            Job::Draw { job, bundle, level_sets, cached, asked } => {
                let drawn = match &job {
                    DrawJob::Restyle(styling, refine) => {
                        Drawn::Restyled(crate::landscape::restyled(&bundle, styling, *refine))
                    }
                    DrawJob::Levels(draft) => {
                        let (layers, derived) = Document::stroked(&bundle, &level_sets, &cached, *draft);
                        Drawn::Stroked(layers, derived)
                    }
                };
                self.drew(job, &bundle, asked, drawn);
            }
            Job::BuildSurface { request, framing, draft, previous, kept, git_sha, asked } => {
                let result = SurfaceDrawing::build(&request, previous.as_ref(), draft)
                    .and_then(|built| match &kept {
                        Some(kept) => built.keeping(kept, previous.as_ref()),
                        None => Ok(built),
                    })
                    .map_err(|e| e.to_string());
                self.built_surface(request, framing, draft, git_sha, asked, result);
            }
            Job::RestyleSurface { bundle, layers, asked } => {
                let result = bundle.restyled(&layers).map_err(|e| e.to_string());
                self.restyled_surface(layers, asked, result);
            }
            Job::StrokeSurface { drawing, layers, asked } => {
                let result = drawing.restyled(&layers).map_err(|e| e.to_string());
                self.stroked_surface(asked, result);
            }
            Job::RestylePolyhedron { bundle, layers, asked } => {
                let result = bundle.restyled(layers.clone()).map_err(|e| e.to_string());
                self.restyled_polyhedron(layers, asked, result);
            }
            Job::Boundary { pick, geometry, faces, generation, content, status } => {
                let boundary = geometry.boundary_of(&pick, Some(&faces));
                self.bounded(pick, &geometry, boundary, generation, content, status);
            }
        }
    }
}
