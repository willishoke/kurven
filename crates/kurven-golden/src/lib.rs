//! The port's own golden sets: its answers, written by the code they pin.
//!
//! A golden file proves nothing about correctness; it proves that a change to
//! the output was made on purpose. `kurven-cli fixtures crates/fixtures/rust`
//! writes these and `scripts/check-web.sh` fails on any difference from the
//! commit, as `scripts/check.sh` does for Swift's. The same function runs in
//! `kurven-wasm` under Node, and the check requires its bytes to be native's:
//! that is what lets a native check stand for the browser.
//!
//! The sets, and what each pins:
//!
//! ```text
//! libm/table.txt   every function of kurven_math::real over a fixed grid of
//!                  arguments, as bits: the libm rule's first test. Native
//!                  and wasm must agree to the bit, and so must every
//!                  platform the check runs on.
//! math/functions.txt  every function of the expression language over a
//!                  fixed grid of the plane, through the parser and the tape.
//! math/language.txt   the canonical spellings, refusals and literals of a
//!                  fixed set of sources.
//! math/dynamics.txt   the ODE integrator, the frequency analysis, the Fourier
//!                  torus, Brent's method, continued fractions and the
//!                  quarter periods on fixed problems.
//! core/            the core values (Phase 2), on fixed inputs computed in
//!                  the generator: the plates' cameras, a navigator through
//!                  every gesture, marching squares on a ripple and a phase
//!                  grid's cut, doubles through the canonical writer,
//!                  manifests of all three kinds, an npy grid, and an SVG of
//!                  lifted contours through the recip camera.
//! landscape/       two small landscapes (Phase 3), built on their function
//!                  and baked in memory: the cubic truncated at a height,
//!                  and gamma truncated by bands of Re z. Each as its SVG
//!                  and as each layer's counts, length and vertex hash.
//! polyhedra/       the seven named solids, the frame under Turing's pattern,
//!                  the dodecahedron under half-Lambert, and a composite of
//!                  the cubic on its grid with a cube and a frame on it
//!                  (Phase 5), each baked in memory, as its SVG and summary.
//! ```
//!
//! The set grows a module at a time: once a module passes against Swift, its
//! goldens are written here.

use kurven_math::real;

mod core_values;
pub mod gallery;
mod landscapes;
mod math;
pub mod polyhedra;

/// One golden file: where it goes under the set's root, and its bytes.
pub struct Golden {
    pub path: &'static str,
    pub bytes: Vec<u8>,
}

/// Every golden file, in a fixed order.
pub fn all() -> Vec<Golden> {
    let mut all = vec![
        Golden { path: "libm/table.txt", bytes: libm_table().into_bytes() },
        Golden { path: "math/functions.txt", bytes: math::functions_table().into_bytes() },
        Golden { path: "math/language.txt", bytes: math::language_table().into_bytes() },
        Golden { path: "math/dynamics.txt", bytes: math::dynamics_table().into_bytes() },
    ];
    all.extend(core_values::all());
    all.extend(landscapes::all());
    all.extend(polyhedra::all());
    all
}

/// An f64 as text that is the same on every platform: its bits, and its
/// shortest round-trip decimal for whoever reads the diff. A NaN is `nan`
/// whatever its payload, since IEEE 754 leaves the payload of a computed NaN
/// to the platform (wasm's are nondeterministic), so no golden writes one.
pub(crate) fn bits(x: f64) -> String {
    if x.is_nan() { "nan".to_string() } else { format!("{:016x} {x:?}", x.to_bits()) }
}

/// The arguments each function of one argument is read at: every eighth
/// from −8 to 8, then values at the edges of the functions' domains and of
/// the format.
fn arguments() -> Vec<f64> {
    let mut xs: Vec<f64> = (-64..=64).map(|k| f64::from(k) / 8.0).collect();
    xs.extend([
        -0.0,
        1e-300,
        -1e-300,
        f64::from_bits(1), // the least subnormal
        f64::MIN_POSITIVE,
        1e-8,
        -1e-8,
        0.1,
        1.0 / 3.0,
        core::f64::consts::FRAC_PI_2,
        core::f64::consts::PI,
        -core::f64::consts::PI,
        core::f64::consts::E,
        12.5,
        -12.5,
        30.0,
        100.0,
        709.5,
        710.0,
        -745.5,
        1e10,
        1e22,
        1e300,
        -1e300,
        f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ]);
    xs
}

/// The pairs each function of two arguments is read at: every half from −4
/// to 4 on each axis, then pairs at the edges.
fn pairs() -> Vec<(f64, f64)> {
    let axis: Vec<f64> = (-8..=8).map(|k| f64::from(k) / 2.0).collect();
    let mut ps: Vec<(f64, f64)> = axis.iter().flat_map(|&x| axis.iter().map(move |&y| (x, y))).collect();
    ps.extend([
        (0.0, -0.0),
        (-0.0, -0.0),
        (-0.0, 1.0),
        (1e-300, 1e-300),
        (1e300, 1e300),
        (f64::MAX, 2.0),
        (2.0, 1023.5),
        (2.0, -1074.0),
        (0.1, 3.0),
        (core::f64::consts::PI, 1e-17),
        (f64::INFINITY, 1.0),
        (1.0, f64::INFINITY),
        (f64::NEG_INFINITY, f64::NEG_INFINITY),
        (f64::NAN, 1.0),
        (1.0, f64::NAN),
    ]);
    ps
}

/// The libm table: one line per call, `name(argument bits) = result bits`,
/// grouped by function in the order `kurven_math::real` declares them.
pub fn libm_table() -> String {
    type One = (&'static str, fn(f64) -> f64);
    type Two = (&'static str, fn(f64, f64) -> f64);
    let one: [One; 24] = [
        ("sin", real::sin),
        ("cos", real::cos),
        ("tan", real::tan),
        ("asin", real::asin),
        ("acos", real::acos),
        ("atan", real::atan),
        ("sinh", real::sinh),
        ("cosh", real::cosh),
        ("tanh", real::tanh),
        ("asinh", real::asinh),
        ("acosh", real::acosh),
        ("atanh", real::atanh),
        ("exp", real::exp),
        ("exp2", real::exp2),
        ("expm1", real::expm1),
        ("log", real::log),
        ("log2", real::log2),
        ("log10", real::log10),
        ("log1p", real::log1p),
        ("cbrt", real::cbrt),
        ("erf", real::erf),
        ("erfc", real::erfc),
        ("sqrt", real::sqrt),
        ("neg", |x| -x),
    ];
    let two: [Two; 6] = [
        ("atan2", real::atan2),
        ("pow", real::pow),
        ("hypot", real::hypot),
        ("fmod", real::fmod),
        ("remainder", real::remainder),
        ("copysign", real::copysign),
    ];
    let mut out = String::from(
        "# kurven_math::real over a fixed grid: name(argument bits) = result bits and shortest decimal.\n\
         # Written by kurven-cli fixtures; native and wasm must agree to the bit. A NaN is written nan.\n",
    );
    for (name, f) in one {
        for x in arguments() {
            out += &format!("{name}({}) = {}\n", bits(x), bits(f(x)));
        }
    }
    for (name, f) in two {
        for (x, y) in pairs() {
            out += &format!("{name}({}, {}) = {}\n", bits(x), bits(y), bits(f(x, y)));
        }
    }
    // a * b + c rounds twice; fused it would round once and give another
    // answer for each of these. The rule is no fused multiply-add, and a
    // target that contracted would show here.
    for (a, b, c) in [(0.1, 10.0, -1.0), (1.0 + f64::EPSILON, 1.0 - f64::EPSILON, -1.0), (1.0 / 3.0, 3.0, -1.0)] {
        out += &format!("unfused({}, {}, {}) = {}\n", bits(a), bits(b), bits(c), bits(a * b + c));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_whole_and_writes_no_nan_bits() {
        let table = libm_table();
        let calls = table.lines().filter(|l| !l.starts_with('#')).count();
        assert_eq!(calls, 24 * arguments().len() + 6 * pairs().len() + 3);
        // A NaN's bits begin 7ff8 or fff8 for the quiet ones a libm returns.
        assert!(!table.contains("7ff8000000000000") && !table.contains("fff8000000000000"));
        assert!(table.contains("sin(nan) = nan"));
    }

    /// Unfused, these come to the double nearest the rounded product plus c:
    /// 0.1 * 10 rounds to exactly 1, so the sum is 0, where fused it is the
    /// representation error of 0.1, 2⁻⁵⁴.
    #[test]
    fn multiply_and_add_round_twice() {
        let (a, b, c) = (std::hint::black_box(0.1), std::hint::black_box(10.0), std::hint::black_box(-1.0));
        assert_eq!(a * b + c, 0.0);
    }

    #[test]
    fn paths_are_relative_and_distinct() {
        let all = all();
        for g in &all {
            assert!(!g.path.starts_with('/') && !g.path.contains(".."));
        }
        let mut paths: Vec<_> = all.iter().map(|g| g.path).collect();
        paths.sort();
        paths.dedup();
        assert_eq!(paths.len(), all.len());
    }
}
