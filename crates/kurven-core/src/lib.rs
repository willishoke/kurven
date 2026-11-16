//! Pure values: spaces and transforms, the camera, navigation, contours, the
//! heightfield and its visibility, faces, hatching, solids, the bundle schema,
//! npy and SVG. The port of `Sources/KurvenCore`.
//!
//! Phase 2 holds the core values: [`spaces`] and the [`simd`] they are built
//! on, the [`camera`] and [`navigation`], the [`depth`] frame, [`grid`],
//! [`polyline`], [`bisection`] and marching squares ([`contour`]); the bundle
//! schema ([`manifest`], with the [`hatching`] values, the [`surface_request`]
//! and the [`solid`] a manifest decodes into), canonical [`json`], [`npy`] and
//! [`svg`]. [`fixtures`] finds the fixture sets on disk.

// Swift's `guard x > 0 else` refuses a NaN, and so does `if !(x > 0.0)`;
// the partial_cmp spelling clippy asks for hides that, so the negated
// comparisons stay as the port of those guards.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod bisection;
pub mod bundle;
pub mod camera;
pub mod contour;
pub mod contour_memo;
pub mod convex_partition;
pub mod cover;
pub mod depth;
pub mod face;
pub mod face_contour;
pub mod fixtures;
pub mod grid;
pub mod hatch;
pub mod hatching;
pub mod hidden_line;
pub mod integral_curves;
pub mod json;
pub mod lines_of_curvature;
pub mod manifest;
pub mod mesh;
pub mod navigation;
pub mod npy;
pub mod occlusion;
pub mod parametric;
pub mod pick;
pub mod polyline;
pub mod scalar_field;
pub mod scene;
pub mod simd;
pub mod solid;
pub mod spaces;
pub mod splitmix;
pub mod surface;
pub mod surface_image;
pub mod surface_request;
pub mod svg;
pub mod text;
pub mod traversal;
pub mod visibility;
