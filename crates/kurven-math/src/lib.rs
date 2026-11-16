//! Complex arithmetic, the special functions and the expression language:
//! the port of `Sources/KurvenMath`. Depends on nothing but `libm`, through
//! [`real`], the one door to the transcendental functions.
//!
//! Each module is one Swift file, by the same name in snake case, and keeps
//! its types' names: `Complex::exp(z)` is `Complex.exp(z)`, `Gamma::gamma(z)`
//! is `Gamma.gamma(z)`. Swift's enums used as namespaces are uninhabited
//! enums here, with associated functions.

pub mod complex;
pub mod format;
pub mod real;

pub use complex::Complex;
