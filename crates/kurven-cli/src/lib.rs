//! The headless engine's commands, as a library so the tests reach them.
//!
//! - [`svg`] reads a drawing back from the SVG kurven writes;
//! - [`compare`] holds two drawings of one scene to each other per layer;
//! - [`args`] reads the command line as Swift's `Args` does;
//! - [`commands`] are `landscape` and `bake`, and the scene they read;
//! - [`references`] bakes the reference drawings as `References.swift` does;
//! - [`solid`] is `surface solid`, a polyhedron by name;
//! - [`scene`] draws several bundles as one composite.

pub mod args;
pub mod commands;
pub mod compare;
pub mod references;
pub mod scene;
pub mod solid;
pub mod svg;
