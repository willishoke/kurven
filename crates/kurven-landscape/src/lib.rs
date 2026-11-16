//! A landscape sampled here: sampling ([`landscape`]), the refiner
//! ([`refine`], `Refine.swift`), the [`catalog`]. The port of
//! `Sources/KurvenLandscape`: Phase 3's, and the torus a doubly periodic
//! function is drawn on ([`torus`], `Torus.swift`), Phase 4's.

// Swift's `guard x > 0 else` refuses a NaN, and so does `if !(x > 0.0)`;
// the partial_cmp spelling clippy asks for hides that, so the negated
// comparisons stay as the port of those guards (as in kurven-core).
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod catalog;
pub mod landscape;
pub mod refine;
pub mod torus;

pub use catalog::{Catalog, FunctionPreset, LandscapeRequest, LanguageFunction, NativeCatalog};
pub use landscape::{BuildError, GridShape, LandscapeStyle, NativeLandscape, Samples};
pub use refine::{ContourRefiner, Piece, RefinementReport, Scalar, SingularPoint, Summary};
pub use torus::{Drawing, PeriodicError, PeriodicOptions, PeriodicTorus};
