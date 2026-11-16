//! The function table and the constants: everything the language can name.
//! The port of `Language` in `Expression.swift`.
//!
//! The table `kurven.expr.LANGUAGE` reported when there was a Python side,
//! frozen in `tests/fixtures/expr/language.json` and held to it by the tests;
//! a change here is a change there, made on purpose.

use std::collections::{BTreeMap, HashMap};
use std::sync::LazyLock;

use crate::bessel::{Airy, Bessel};
use crate::complex::Complex;
use crate::erf::Faddeeva;
use crate::expint::ExpInt;
use crate::gamma::Gamma;
use crate::jacobi::Jacobi;
use crate::lambert_w::LambertW;
use crate::zeta::Zeta;

pub enum Language {}

/// A function's body. Each captures nothing, so it is a plain function
/// pointer: calling it counts no references. (Swift keeps three closures,
/// `apply` over an array and the `unary` or `binary` body it wraps.)
#[derive(Clone, Copy)]
pub enum Body {
    Unary(fn(Complex) -> Complex),
    Binary(fn(Complex, Complex) -> Complex),
}

pub struct Function {
    pub name: &'static str,
    pub arity: usize,
    pub aliases: &'static [&'static str],
    pub help: &'static str,
    /// What each argument is, as a control would name it: `argument` for
    /// a point of the plane, `order` or `modulus` for a parameter.
    pub parameters: &'static [&'static str],
    pub body: Body,
}

/// A function by its name, for a tape's instruction or an error message.
impl core::fmt::Debug for Function {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Function({}, arity {})", self.name, self.arity)
    }
}

impl Function {
    /// The function applied to its arguments, as many as its arity. Panics on
    /// fewer, as Swift's array subscript traps.
    pub fn apply(&self, args: &[Complex]) -> Complex {
        match self.body {
            Body::Unary(f) => f(args[0]),
            Body::Binary(f) => f(args[0], args[1]),
        }
    }

    /// The body of a function of one argument, for `Node::compiled` and the
    /// tape.
    pub fn unary(&self) -> Option<fn(Complex) -> Complex> {
        match self.body {
            Body::Unary(f) => Some(f),
            Body::Binary(_) => None,
        }
    }

    pub fn binary(&self) -> Option<fn(Complex, Complex) -> Complex> {
        match self.body {
            Body::Binary(f) => Some(f),
            Body::Unary(_) => None,
        }
    }
}

pub struct Constant {
    pub name: &'static str,
    pub value: Complex,
    pub aliases: &'static [&'static str],
    pub help: &'static str,
}

/// A second argument that is a *parameter*, not a point of the plane: a
/// Bessel order or an elliptic modulus. Its imaginary part is dropped,
/// as `kurven.expr._parameter` drops it.
fn parameter(v: Complex) -> f64 {
    v.re
}

const ARGUMENT: &[&str] = &["argument"];
const ORDER_ARGUMENT: &[&str] = &["order", "argument"];
const ARGUMENT_MODULUS: &[&str] = &["argument", "modulus"];

const fn f1(
    name: &'static str,
    help: &'static str,
    aliases: &'static [&'static str],
    body: fn(Complex) -> Complex,
) -> Function {
    Function { name, arity: 1, aliases, help, parameters: ARGUMENT, body: Body::Unary(body) }
}

const fn f2(
    name: &'static str,
    help: &'static str,
    aliases: &'static [&'static str],
    parameters: &'static [&'static str],
    body: fn(Complex, Complex) -> Complex,
) -> Function {
    Function { name, arity: 2, aliases, help, parameters, body: Body::Binary(body) }
}

static FUNCTIONS: [Function; 43] = [
    f1("exp", "e raised to z", &[], Complex::exp),
    f1("log", "natural logarithm, principal branch", &["ln"], Complex::log),
    f1("sqrt", "square root, principal branch", &[], Complex::sqrt),
    f1("sin", "sine", &[], Complex::sin),
    f1("cos", "cosine", &[], Complex::cos),
    f1("tan", "tangent — poles at the odd multiples of π/2", &[], Complex::tan),
    f1("cot", "cotangent", &[], |z| 1.0 / Complex::tan(z)),
    f1("sec", "secant", &[], |z| 1.0 / Complex::cos(z)),
    f1("csc", "cosecant", &[], |z| 1.0 / Complex::sin(z)),
    f1("sinh", "hyperbolic sine", &[], Complex::sinh),
    f1("cosh", "hyperbolic cosine", &[], Complex::cosh),
    f1("tanh", "hyperbolic tangent", &[], Complex::tanh),
    f1("asin", "inverse sine", &["arcsin"], Complex::asin),
    f1("acos", "inverse cosine", &["arccos"], Complex::acos),
    f1("atan", "inverse tangent", &["arctan"], Complex::atan),
    f1("asinh", "inverse hyperbolic sine", &["arcsinh"], Complex::asinh),
    f1("acosh", "inverse hyperbolic cosine", &["arccosh"], Complex::acosh),
    f1("atanh", "inverse hyperbolic tangent", &["arctanh"], Complex::atanh),
    f1("gamma", "Γ(z) — poles at the non-positive integers", &["Γ"], Gamma::gamma),
    f1("rgamma", "1/Γ(z), entire; zeros at 0, -1, -2, ...", &[], Gamma::rgamma),
    f1("loggamma", "log Γ(z), principal branch", &["lgamma"], Gamma::log_gamma),
    f1("digamma", "ψ(z) = Γ'(z)/Γ(z)", &["psi", "ψ"], Gamma::digamma),
    f1("zeta", "ζ(z) — the pole is at z = 1", &["ζ"], Zeta::zeta),
    f1("erf", "the error function", &[], Faddeeva::erf),
    f1("erfc", "the complementary error function", &[], Faddeeva::erfc),
    f1("erfi", "the imaginary error function", &[], Faddeeva::erfi),
    f1("wofz", "the Faddeeva function w(z)", &["faddeeva"], Faddeeva::w),
    f1("expi", "the exponential integral, continued as -E₁(-z)", &["Ei"], ExpInt::expi),
    f1("lambertw", "W(z), the principal branch of the Lambert W function", &["W"], LambertW::w0),
    f1("airyai", "Ai(z)", &["Ai"], Airy::ai),
    f1("airybi", "Bi(z)", &["Bi"], Airy::bi),
    f2("besselj", "J_n(z), first argument the order", &["jv"], ORDER_ARGUMENT, |n, z| Bessel::j(parameter(n), z)),
    f2("bessely", "Y_n(z), first argument the order", &["yv"], ORDER_ARGUMENT, |n, z| Bessel::y(parameter(n), z)),
    f2("besseli", "I_n(z), first argument the order", &["iv"], ORDER_ARGUMENT, |n, z| Bessel::i(parameter(n), z)),
    f2("besselk", "K_n(z), first argument the order", &["kv"], ORDER_ARGUMENT, |n, z| Bessel::k(parameter(n), z)),
    f2("sn", "Jacobi sn(z, m), doubly periodic", &[], ARGUMENT_MODULUS, |z, m| Jacobi::complex(z, parameter(m)).sn),
    f2("cn", "Jacobi cn(z, m), doubly periodic", &[], ARGUMENT_MODULUS, |z, m| Jacobi::complex(z, parameter(m)).cn),
    f2("dn", "Jacobi dn(z, m), doubly periodic", &[], ARGUMENT_MODULUS, |z, m| Jacobi::complex(z, parameter(m)).dn),
    f1("abs", "|z|, as a real value", &[], |z| Complex::real(z.magnitude())),
    f1("re", "the real part", &[], |z| Complex::real(z.re)),
    f1("im", "the imaginary part", &[], |z| Complex::real(z.im)),
    f1("conj", "the complex conjugate", &[], |z| z.conjugate()),
    f1("arg", "the argument of z", &[], |z| Complex::real(z.argument())),
];

static CONSTANTS: [Constant; 4] = [
    Constant { name: "i", value: Complex::I, aliases: &["j"], help: "the imaginary unit" },
    Constant { name: "pi", value: Complex::real(core::f64::consts::PI), aliases: &["π"], help: "π" },
    Constant { name: "e", value: Complex::real(core::f64::consts::E), aliases: &[], help: "Euler's number" },
    Constant { name: "tau", value: Complex::real(core::f64::consts::TAU), aliases: &["τ"], help: "2π" },
];

/// The table by name, as Swift's `functionsByName` dictionary holds it.
static FUNCTIONS_BY_NAME: LazyLock<HashMap<&'static str, &'static Function>> =
    LazyLock::new(|| FUNCTIONS.iter().map(|f| (f.name, f)).collect());

/// Every spelling, mapped to the one the tree stores; ordered, so its keys
/// come sorted, as `unknown` reads them.
static CANONICAL_NAMES: LazyLock<BTreeMap<&'static str, &'static str>> = LazyLock::new(|| {
    let mut names = BTreeMap::from([("z", "z")]);
    for f in &FUNCTIONS {
        names.insert(f.name, f.name);
        for a in f.aliases {
            names.insert(a, f.name);
        }
    }
    for c in &CONSTANTS {
        names.insert(c.name, c.name);
        for a in c.aliases {
            names.insert(a, c.name);
        }
    }
    names
});

impl Language {
    pub fn functions() -> &'static [Function] {
        &FUNCTIONS
    }

    pub fn constants() -> &'static [Constant] {
        &CONSTANTS
    }

    /// Every spelling, mapped to the one the tree stores.
    pub fn canonical_names() -> &'static BTreeMap<&'static str, &'static str> {
        &CANONICAL_NAMES
    }

    pub fn canonical_name(spelling: &str) -> Option<&'static str> {
        CANONICAL_NAMES.get(spelling).copied()
    }

    /// The function of this canonical name, by hash, as Swift's dictionary
    /// finds it. The tape and `Node::compiled` look it up once, when the
    /// expression is compiled; `Node::evaluate` walks the tree and looks it
    /// up at every call, as Swift's does.
    pub fn function(name: &str) -> Option<&'static Function> {
        FUNCTIONS_BY_NAME.get(name).copied()
    }

    pub fn constant(name: &str) -> Option<&'static Constant> {
        CONSTANTS.iter().find(|c| c.name == name)
    }

    /// The message for a name the language does not have, with the nearest
    /// names it does -- `difflib.get_close_matches` at a cutoff of 0.6, by
    /// the longest common subsequence rather than its matching blocks, which
    /// agrees on every name this short.
    pub(crate) fn unknown(text: &str) -> String {
        let t: Vec<char> = text.chars().collect();
        let mut scored: Vec<(&str, f64)> = Vec::new();
        for c in CANONICAL_NAMES.keys() {
            let cc: Vec<char> = c.chars().collect();
            let ratio = 2.0 * Language::lcs(&t, &cc) as f64 / (t.len() + cc.len()) as f64;
            if ratio >= 0.6 {
                scored.push((c, ratio));
            }
        }
        scored.sort_by(|a, b| if a.1 != b.1 { crate::real::total_order(b.1, a.1) } else { a.0.cmp(b.0) });
        let close: Vec<&str> = scored.iter().take(3).map(|s| s.0).collect();
        if !close.is_empty() {
            return format!("there is no '{text}'; did you mean {}?", close.join(" or "));
        }
        format!("there is no '{text}' in this language; the variable is z, and `describe` lists what else there is")
    }

    fn lcs(a: &[char], b: &[char]) -> usize {
        if a.is_empty() || b.is_empty() {
            return 0;
        }
        let mut prev = vec![0usize; b.len() + 1];
        let mut cur = prev.clone();
        for i in 1..=a.len() {
            for j in 1..=b.len() {
                cur[j] = if a[i - 1] == b[j - 1] { prev[j - 1] + 1 } else { prev[j].max(cur[j - 1]) };
            }
            std::mem::swap(&mut prev, &mut cur);
        }
        prev[b.len()]
    }
}
