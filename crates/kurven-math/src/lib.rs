//! Complex arithmetic, the special functions and the expression language:
//! the port of `Sources/KurvenMath`. Depends on nothing but `libm`, through
//! [`real`], the one door to the transcendental functions.
//!
//! Each module is one Swift file, by the same name in snake case, and keeps
//! its types' names: `Complex::exp(z)` is `Complex.exp(z)`, `Gamma::gamma(z)`
//! is `Gamma.gamma(z)`. Swift's enums used as namespaces are uninhabited
//! enums here, with associated functions.

pub mod bessel;
pub mod complex;
pub mod continued_fraction;
pub mod erf;
pub mod expint;
pub mod expression;
pub mod format;
pub mod fourier_torus;
pub mod frequency;
pub mod gamma;
pub mod jacobi;
pub mod lambert_w;
pub mod language;
pub mod literals;
pub mod ode;
pub mod real;
pub mod root;
pub mod simd;
pub mod swift;
pub mod tape;
pub mod zeta;

pub use bessel::{Airy, Bessel};
pub use complex::Complex;
pub use continued_fraction::ContinuedFraction;
pub use erf::Faddeeva;
pub use expint::ExpInt;
pub use expression::{Compiled, Expression, ExpressionError, Node};
pub use fourier_torus::FourierTorus;
pub use frequency::Frequency;
pub use gamma::{Gamma, Trig};
pub use jacobi::Jacobi;
pub use lambert_w::LambertW;
pub use language::Language;
pub use literals::{Literal, Role};
pub use ode::Ode;
pub use root::Root;
pub use simd::{Simd, Simd2, Simd3, Simd4};
pub use zeta::Zeta;
