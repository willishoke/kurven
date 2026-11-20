//! The document the page edits: the function, the window, the truncation,
//! the layers, the camera, the catalog choice, draft and full resolution,
//! and the undo stack. It takes edits and gestures as values and returns the
//! new state, as `Navigator.applying` and `Edits.swift` do in Swift, so every
//! edit is tested natively and the page holds nothing the engine does not.
//!
//! The port of `Sources/KurvenApp`'s state and edit logic, without its
//! views: `Document.swift` ([`document`]), `Edits.swift` ([`edits`]),
//! `Landscape.swift` ([`landscape`]), `Surfaces.swift` ([`surfaces`]),
//! `Polyhedra.swift` ([`polyhedra`]), `Paint.swift` ([`paint`]),
//! `Picking.swift` ([`picking`]), the sliders' callbacks ([`sliders`]), and
//! the app's headless command line ([`headless`]), which the gate replays
//! against Swift's. [`preview`] is `KurvenMetal`'s CPU half: what a preview
//! frame draws, which the page's WebGPU renderer draws. [`protocol`] is the session as messages: an edit in as
//! JSON, the state out as JSON, which `kurven-wasm` and the Worker speak;
//! [`pool`] is the slow work shared among several engines holding the same
//! document.

// `!(x > 0.0)` stands for Swift's `guard x > 0 else`, so a NaN is refused as
// Swift refuses it (kurven-core allows it for the same reason). A document's
// bundles are large values held one at a time, not arrays of them.
#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::large_enum_variant)]

pub mod document;
pub mod edits;
pub mod files;
pub mod frames;
pub mod headless;
pub(crate) mod jobs;
pub mod landscape;
pub mod paint;
pub mod picking;
pub mod polyhedra;
pub mod pool;
pub mod preview;
pub mod protocol;
pub mod sliders;
pub mod surface_controls;
pub mod surfaces;
pub mod thumbnails;

pub use document::{Document, Opened, State};
pub use edits::Edited;
