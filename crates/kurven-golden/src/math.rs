//! The golden sets of `kurven-math`: every function of the expression
//! language over a fixed grid of the plane, and the integrators and analyses
//! built on them, as bits.
//!
//! These pin the port's answers, not Swift's (the oracle suites in
//! `kurven-math/tests` hold those to numpy, scipy and mpmath to a tolerance).
//! What they prove is that native and wasm compute the same bits through the
//! whole of the math crate, special functions, parser and tape included, and
//! that a change to any of it is made on purpose.

use core::f64::consts::PI;

use kurven_math::continued_fraction::ContinuedFraction;
use kurven_math::fourier_torus::FourierTorus;
use kurven_math::frequency::Frequency;
use kurven_math::ode::{Ode, Tolerance};
use kurven_math::real;
use kurven_math::{Complex, Expression, Jacobi, Language, Root, Simd, Simd3, Simd4};

use crate::bits;

/// A complex value as two `bits`, real part first.
fn complex_bits(z: Complex) -> String {
    format!("{} | {}", bits(z.re), bits(z.im))
}

/// The points each function is read at: a coarse grid over the windows the
/// landscapes use, then the places the functions are particular about --
/// both sides of the branch cuts (a zero imaginary part of either sign), the
/// poles and branch points, far out along each axis, and the edges of the
/// format.
fn points() -> Vec<Complex> {
    let res: Vec<f64> = (-4..=4).map(|k| 1.5 * f64::from(k)).collect();
    let ims = [-4.0, -2.5, -1.0, 0.0, 1.0, 2.5, 4.0];
    let mut zs: Vec<Complex> = res.iter().flat_map(|&r| ims.iter().map(move |&i| Complex::new(r, i))).collect();
    zs.extend([
        Complex::new(-2.5, 0.0),
        Complex::new(-2.5, -0.0),
        Complex::new(-0.5, 0.0),
        Complex::new(-0.5, -0.0),
        Complex::new(0.5, -0.0),
        Complex::new(2.0, -0.0),
        Complex::new(-0.0, -0.0),
        Complex::new(0.0, 1.0),
        Complex::new(0.0, -1.0),
        Complex::new(-0.0, 2.0),
        Complex::new(1.0, 0.0),
        Complex::new(-3.0, 0.0),
        Complex::new(-1.0 / core::f64::consts::E, 0.0),
        Complex::new(0.3, 0.2),
        Complex::new(1e-8, -1e-8),
        Complex::new(30.0, 0.0),
        Complex::new(-30.0, 0.5),
        Complex::new(0.5, 30.0),
        Complex::new(12.0, -12.0),
        Complex::new(-400.0, 1.0),
        Complex::new(f64::INFINITY, 0.0),
        Complex::new(f64::NAN, 0.0),
    ]);
    zs
}

/// The language's expressions the grid is read through: each function of
/// one argument at z, and each of two at a few orders or moduli.
fn expressions() -> Vec<String> {
    let mut out = Vec::new();
    for f in Language::functions() {
        match f.parameters {
            ["argument"] => out.push(format!("{}(z)", f.name)),
            ["order", "argument"] => {
                for order in ["0", "2.7", "-2.3"] {
                    out.push(format!("{}({order}, z)", f.name));
                }
            }
            ["argument", "modulus"] => {
                for modulus in ["0.2", "0.9"] {
                    out.push(format!("{}(z, {modulus})", f.name));
                }
            }
            other => panic!("{}: no golden for parameters {other:?}", f.name),
        }
    }
    // The operators and the constants, through the same parser and tape.
    out.extend(["z^2 - 2z + 1/z".to_string(), "(z+1)!".to_string(), "2^z * pi^-z + e*i*tau".to_string()]);
    out
}

/// `math/functions.txt`: every expression of `expressions()` at every point
/// of `points()`, one line per value.
pub fn functions_table() -> String {
    let zs = points();
    let mut out = String::from(
        "# Every function of the expression language over a fixed grid, through Expression::compile and its tape:\n\
         # expression @ z = Re f bits and decimal | Im f bits and decimal. Written by kurven-cli fixtures;\n\
         # native and wasm must agree to the bit. A NaN is written nan.\n",
    );
    for text in expressions() {
        let f = Expression::compile(&text).unwrap_or_else(|e| panic!("{text}: {e}"));
        for &z in &zs {
            out += &format!("{text} @ {z} = {}\n", complex_bits(f.call(z)));
        }
    }
    out
}

/// What the canonical printer and the parser's refusals write for a fixed
/// set of sources: the language's text, which classifies characters and so
/// must also agree native and wasm.
pub fn language_table() -> String {
    let sources = [
        "-z^2+2z-1/(z^2+1)",
        "1/2z",
        "2z^2",
        "1e-3z",
        "1e16z",
        "123456789012345.6 z",
        "1 / Γ(z)",
        "ζ(z) + ψ(z)",
        "2 pi i",
        "z!^2",
        "cn(z, 0.64)",
        "ln(z) + arcsin(z)",
        "",
        "sin(z",
        "gama(z)",
        "gamma(z, 2)",
        "z..2",
        "2 PI",
        "z 2 @",
        "__import__(\"os\")",
        "Ωz",
        "z²",
    ];
    let mut out = String::from(
        "# The expression language's canonical spellings and refusals, as kurven_math writes them.\n\
         # Written by kurven-cli fixtures; native and wasm must agree byte for byte.\n",
    );
    for source in sources {
        match Expression::canonical(source) {
            Ok(text) => out += &format!("{source:?} -> {text:?}\n"),
            Err(e) => out += &format!("{source:?} refused at {:?}+{}: {}\n", e.position, e.length, e.message),
        }
    }
    for literal_text in ["-z^2 + 2^-0.5 z - 3", "sin(z)cn(z, 0.64)", "cn(2z, -0.5)^-2"] {
        for l in Expression::literals(literal_text) {
            out += &format!(
                "{literal_text:?} literal {:?} at {}+{} = {} ({})\n",
                l.text,
                l.position,
                l.length,
                bits(l.value),
                Expression::role(&l, literal_text).label()
            );
        }
    }
    out
}

/// `math/dynamics.txt`: the ODE integrator, the frequency analysis, the
/// Fourier torus, Brent's method, the continued fractions and the quarter
/// periods, each on a small fixed problem.
pub fn dynamics_table() -> String {
    let mut out = String::from(
        "# kurven_math's integrators and analyses on fixed problems, as bits.\n\
         # Written by kurven-cli fixtures; native and wasm must agree to the bit.\n",
    );
    let lanes = |v: &[f64]| v.iter().map(|&x| bits(x)).collect::<Vec<_>>().join(" | ");

    // A Kepler orbit of eccentricity 0.6, sampled every quarter period for
    // two periods.
    let kepler = |_: f64, s: Simd4| {
        let k = real::pow(s[0] * s[0] + s[1] * s[1], 1.5);
        Simd([s[2], s[3], -s[0] / k, -s[1] / k])
    };
    let start = Simd([0.4, 0.0, 0.0, real::sqrt(1.6 / 0.4)]);
    let tolerance = Tolerance { relative: 1e-11, absolute: 1e-13, max_step: None };
    let (states, stats) = Ode::sample(kepler, 0.0, start, PI / 2.0, 9, tolerance).expect("the orbit integrates");
    out += &format!(
        "kepler: {} accepted, {} rejected, {} evaluations\n",
        stats.accepted, stats.rejected, stats.evaluations
    );
    for (k, s) in states.iter().enumerate() {
        out += &format!("kepler[{k}] = {}\n", lanes(&s.0));
    }
    let (state, error) =
        Ode::fixed(|t, y| Simd([y[1], -36.0 * y[0] + real::sin(3.0 * t)]), 0.0, Simd([0.0, 0.0]), 4.0, 100);
    out += &format!("fixed = {} ; error {}\n", lanes(&state.0), lanes(&error.0));

    // Two lines and their generator, from a short series.
    let (omega, big_omega, dt) = (0.94, 0.3962213283, 0.05);
    let x: Vec<f64> = (0..4096)
        .map(|k| {
            let t = 10.0 + f64::from(k) * dt;
            real::cos(omega * t) + 0.3 * real::cos(big_omega * t + 0.3)
        })
        .collect();
    let lines = Frequency::lines(&x, 10.0, dt, 2);
    for l in &lines {
        out += &format!("line = {} | {}\n", bits(l.frequency), bits(l.amplitude));
    }
    if let Some(g) = Frequency::generator(&lines, omega, 4, 1e-7) {
        out += &format!("generator = {} explaining {}\n", bits(g.frequency), g.explained);
    }

    // A torus that is a short Fourier series, fitted from a short run.
    let k = |a: f64, b: f64| -> Simd3 {
        Simd([2.0 * real::cos(a) + 0.5 * real::cos(a + b), 2.0 * real::sin(a) + 0.1, 0.7 * real::sin(b)])
    };
    let samples: Vec<Simd3> = (0..4096)
        .map(|i| {
            let t = f64::from(i) * dt;
            k(omega * t, big_omega * t)
        })
        .collect();
    let fit = FourierTorus::fit(&samples, 0.0, dt, (omega, big_omega), 2);
    let jet = fit.second_order_jet(1.0, 2.0);
    for (name, v) in [
        ("value", jet.value),
        ("d_theta", jet.d_theta),
        ("d_phi", jet.d_phi),
        ("theta_theta", jet.theta_theta),
        ("theta_phi", jet.theta_phi),
        ("phi_phi", jet.phi_phi),
    ] {
        out += &format!("torus {name} = {}\n", lanes(&v.0));
    }
    out += &format!("torus residual = {}\n", bits(fit.residual(&samples, 0.0, dt, 3)));

    let dottie = Root::brent_default(|x| real::cos(x) - x, 0.0, 1.0, 1.0, real::cos(1.0) - 1.0, 1e-12);
    out += &format!("brent cos x = x: {}\n", dottie.map_or("none".to_string(), bits));
    let convergents = ContinuedFraction::convergents(PI, 1_000_000, 1e-12);
    out += &format!(
        "convergents of pi: {}\n",
        convergents.iter().map(|c| format!("{}/{}", c.p, c.q)).collect::<Vec<_>>().join(" ")
    );
    for m in [0.0, 0.2, 0.5, 0.64, 0.9, 0.999] {
        let r = Jacobi::real(1.3, m);
        out += &format!(
            "K({m}) = {} ; ellpj(1.3, {m}) = {}\n",
            bits(Jacobi::quarter_period(m)),
            lanes(&[r.sn, r.cn, r.dn, r.ph])
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_functions_table_is_whole_and_writes_no_nan_bits() {
        let table = functions_table();
        let values = table.lines().filter(|l| !l.starts_with('#')).count();
        assert_eq!(values, expressions().len() * points().len());
        // Every function of the table is read, each of two arguments at more
        // than one parameter.
        assert!(expressions().len() > Language::functions().len());
        assert!(!table.contains("7ff8000000000000") && !table.contains("fff8000000000000"));
    }
}
