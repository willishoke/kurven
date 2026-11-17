//! The half of a document that is a polyhedron, opened from its bundle
//! (`PolyhedronBundle`): the port of `Sources/KurvenApp/Polyhedra.swift`.
//!
//! The solid is the bundle's, so what the controls edit is its layers: a
//! hatching's spacing, from the stroke slider a landscape's hatching has. A
//! restyle is coalesced as a landscape's -- one in flight, the latest edit
//! queued behind it -- and lays the hatching that changed and nothing else
//! (`PolyhedronBundle::restyled`). A restyle a polyhedron cannot take, more
//! hatch lines than it holds, is refused in the status line and the slider
//! goes back.

use kurven_core::manifest::{LayerSpec, PolyhedronManifest};
use kurven_core::polyhedron_bundle::PolyhedronBundle;

use crate::document::Document;
use crate::jobs::Job;

impl Document {
    /// The layers the controls show: the latest asked for, else the bundle's.
    pub fn polyhedron_layers(&self) -> Option<Vec<LayerSpec>> {
        self.polyhedron_styled.clone().or_else(|| self.polyhedron_bundle().map(|b| b.manifest.layers.clone()))
    }

    /// Edit the layers, and lay what changed from the solid already built.
    pub fn restyle_polyhedron(&mut self, change: impl FnOnce(&mut Vec<LayerSpec>)) {
        let Some(mut layers) = self.polyhedron_layers() else { return };
        let before = layers.clone();
        change(&mut layers);
        if layers == before {
            return;
        }
        // Refused here, so a slider that asks for more than the solid takes
        // leaves the layers as they were, and nothing to undo.
        if let Some(bundle) = self.polyhedron_bundle()
            && let Some(why) = PolyhedronManifest::unsupported(&layers, bundle.solid())
        {
            self.polyhedron_status = Some(why.clone());
            self.paint_status = Some(why);
            return;
        }
        self.polyhedron_styled = Some(layers);
        self.redraw_polyhedron();
    }

    /// The one job that lays a polyhedron's layers: the layers last asked
    /// for, when they are not the ones on screen.
    pub(crate) fn redraw_polyhedron(&mut self) {
        if self.drawing {
            return;
        }
        let (Some(bundle), Some(layers)) = (self.polyhedron_bundle().cloned(), self.polyhedron_styled.clone()) else {
            return;
        };
        if layers == bundle.manifest.layers {
            self.polyhedron_styled = None;
            return;
        }
        self.drawing = true;
        self.jobs.push_back(Job::RestylePolyhedron { bundle, layers, asked: self.version });
    }

    /// A restyle of the polyhedron landed.
    pub(crate) fn restyled_polyhedron(
        &mut self,
        layers: Vec<LayerSpec>,
        asked: u64,
        result: Result<PolyhedronBundle, String>,
    ) {
        self.drawing = false;
        if self.version == asked {
            match result {
                Ok(restyled) => {
                    self.adopt_polyhedron(restyled, true);
                    self.polyhedron_status = None;
                }
                Err(error) => {
                    self.polyhedron_status = Some(error);
                    if self.polyhedron_styled.as_ref() == Some(&layers) {
                        self.polyhedron_styled = None;
                    }
                }
            }
        }
        self.redraw_polyhedron();
    }
}
