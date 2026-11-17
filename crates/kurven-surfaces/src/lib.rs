//! Parametric surfaces drawn: the round torus, `sn`, `periodic`, relief along
//! the normal, the forced system and its fitted torus, and their drawings and
//! bundles. The port of `Sources/KurvenDynamics`.

// Swift's `guard x > 0 else` refuses a NaN, and so does `if !(x > 0.0)`.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod bundle;
pub mod drawing;
pub mod forced;
pub mod text;

pub use bundle::{SurfaceBundle, SurfaceBundleError};
pub use drawing::{DrawingError, FunctionRefusal, SurfaceDrawing, SurfacePreset, SurfaceRequestExt, periods_of};
pub use forced::{DynamicsError, ForcedSystem, InvariantTorus, Revolution, Run};
