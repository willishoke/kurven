//! The headless engine's commands, as a library so the tests reach them.
//!
//! - [`svg`] reads a drawing back from the SVG kurven writes;
//! - [`compare`] holds two drawings of one scene to each other per layer.

pub mod compare;
pub mod svg;
