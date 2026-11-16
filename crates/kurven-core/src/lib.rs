//! Pure values: spaces and transforms, the camera, navigation, contours, the
//! heightfield and its visibility, faces, hatching, solids, the bundle schema,
//! npy and SVG. The port of `Sources/KurvenCore`.
//!
//! Phase 0 holds the readers the gate needs: [`npy`], [`json`], and
//! [`fixtures`], which finds the fixture sets on disk.

pub mod fixtures;
pub mod json;
pub mod npy;
