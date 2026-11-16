//! The expression language and its functions against the oracles: the port of
//! `expressionTests` in `Sources/kurven-test/Expressions.swift`.
//!
//! numpy for the elementary functions, scipy.special where it has a complex
//! form, mpmath where it does not -- at every point of every fixture grid.
//! `tests/oracles.py` names the library behind each case, and each case
//! records it.
//!
//! The comparison is `|a - b| <= tol * max(|b|, 1)`: relative where the value
//! is large, absolute where it is small, so a zero of a function is not held to
//! a relative error it cannot have, and both sides agreeing that a point is a
//! pole is agreement. The tolerance is per case, stated in the fixture, and it
//! is a statement about the algorithm: 1e-12 where this side runs the
//! oracle's own method or one as good, 1e-9 for the Bessel and Airy families,
//! whose engines are different from AMOS and agree with it to about 1e-11
//! across these windows. The tolerances are the fixture's, as Swift reads
//! them; none is changed here.

use kurven_core::fixtures::{shared_json, shared_npy};
use kurven_core::json::JsonValue;
use kurven_math::{Complex, Expression, Language};

/// `NativeLandscape.linspace`: numpy's, the last point exactly `hi`.
fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return if n == 1 { vec![lo] } else { vec![] };
    }
    let step = (hi - lo) / (n - 1) as f64;
    (0..n).map(|k| if k == n - 1 { hi } else { lo + k as f64 * step }).collect()
}

fn doubles(v: &JsonValue, key: &str) -> Vec<f64> {
    v.get(key)
        .and_then(JsonValue::as_array)
        .unwrap_or_else(|| panic!("no {key}"))
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn string<'a>(v: &'a JsonValue, key: &str) -> &'a str {
    v.get(key).and_then(JsonValue::as_str).unwrap_or_else(|| panic!("no {key}"))
}

/// Whether two values, one at least not finite, agree: both NaN or neither,
/// and each part that is infinite in either is the same infinity, sign
/// included, in both. A finite part beside an infinite one is not compared:
/// the relative error has nothing to be relative to. (Stricter than the
/// Swift suite, which takes any two values that are not finite as agreeing.)
fn non_finite_agree(a: Complex, b: Complex) -> bool {
    if a.is_nan() != b.is_nan() {
        return false;
    }
    if a.is_nan() {
        return true;
    }
    [(a.re, b.re), (a.im, b.im)].iter().all(|&(x, y)| !(x.is_infinite() || y.is_infinite()) || x == y)
}

#[test]
fn the_functions_agree_with_numpy_scipy_and_mpmath_on_the_fixture_grids() {
    let index = shared_json("expr/index.json");
    let cases = index.get("cases").and_then(JsonValue::as_array).expect("cases");
    // Every function in the table has a case, or is listed as held to
    // something else; the oracle script refuses to write a table that leaves
    // one out, and this is the same rule read from the other side.
    let named: Vec<&str> = cases.iter().map(|c| string(c, "name")).collect();
    let unheld = index.get("unheld").and_then(JsonValue::as_object);
    let uncovered: Vec<&str> = Language::functions()
        .iter()
        .map(|f| f.name)
        .filter(|n| !named.contains(&format!("table_{n}").as_str()) && unheld.is_none_or(|u| !u.contains_key(*n)))
        .collect();
    assert!(uncovered.is_empty(), "every function in the table has an oracle: {uncovered:?}");

    let mut report = Vec::new();
    let mut failed = Vec::new();
    for e in cases {
        let name = string(e, "name");
        let text = string(e, "expression");
        let real = doubles(e, "real");
        let imag = doubles(e, "imag");
        let tol = e.get("tol").and_then(JsonValue::as_f64).unwrap();
        let (nr, ni) = (real[2] as usize, imag[2] as usize);
        let rs = linspace(real[0], real[1], nr);
        let is = linspace(imag[0], imag[1], ni);
        let want = shared_npy(&format!("expr/{}", string(e, "file"))).complex_pairs().unwrap();
        assert_eq!(want.len(), 2 * nr * ni, "{name}: the table's shape");
        let f = Expression::compile(text).unwrap();
        let mut worst = 0.0f64;
        let mut worst_at = Complex::ZERO;
        let mut disagreements = 0;
        let mut non_finite = 0;
        for (i, &r) in rs.iter().enumerate() {
            for (j, &im) in is.iter().enumerate() {
                let z = Complex::new(r, im);
                let a = f.call(z);
                let k = 2 * (i * ni + j);
                let b = Complex::new(want[k], want[k + 1]);
                // A NaN from the oracle is the library declining to answer
                // (Γ at a pole it hit exactly), not a value to be held to;
                // an infinity is a value, and both sides must agree on it,
                // part by part and sign by sign.
                if b.is_nan() {
                    continue;
                }
                if !a.is_finite() || !b.is_finite() {
                    non_finite += 1;
                    if !non_finite_agree(a, b) {
                        disagreements += 1;
                        if disagreements == 1 {
                            worst_at = z;
                            worst = f64::INFINITY;
                        }
                    }
                    continue;
                }
                let err = (a - b).magnitude() / b.magnitude().max(1.0);
                if err > worst {
                    worst = err;
                    worst_at = z;
                }
                if err > tol {
                    disagreements += 1;
                }
            }
        }
        let oracle = e.get("oracle").and_then(JsonValue::as_str).unwrap_or("?");
        let line = format!(
            "{name}: {text} against {oracle}: worst {worst:.1e} at {worst_at} (tol {tol:e}), {non_finite} not finite"
        );
        if disagreements > 0 {
            failed.push(format!("{line}, {disagreements} points over"));
        }
        report.push(line);
    }
    println!("{}", report.join("\n"));
    assert!(failed.is_empty(), "{} of {} cases disagree:\n{}", failed.len(), cases.len(), failed.join("\n"));
}

/// `language.json` is frozen and edited by hand: the readings two reasonable
/// people disagree about, the character each malformed expression is refused
/// at, and the function table. It was last written by the Python
/// implementation of the language, and it is the record of what this one was
/// ported to read.
#[test]
fn the_parser_reads_the_frozen_readings() {
    let index = shared_json("expr/language.json");
    let canonical = index.get("canonical").and_then(JsonValue::as_object).expect("canonical");
    let grid = index.get("grid").expect("grid");
    let (gr, gi) = (doubles(grid, "real"), doubles(grid, "imag"));
    let points: Vec<Complex> = linspace(gr[0], gr[1], gr[2] as usize)
        .into_iter()
        .flat_map(|r| linspace(gi[0], gi[1], gi[2] as usize).into_iter().map(move |i| Complex::new(r, i)))
        .collect();
    let mut wrong = Vec::new();
    let mut unlike = Vec::new();
    for (source, want) in canonical {
        let wanted = want.as_str().unwrap_or_else(|| panic!("{source}: not a string"));
        let got = Expression::canonical(source).unwrap();
        if got != wanted {
            wrong.push(format!("{source} -> {got}, frozen {wanted}"));
        }
        // And the canonical form is a fixed point that means the same thing.
        let again = Expression::canonical(&got).unwrap();
        if again != got {
            wrong.push(format!("{got} is not a fixed point ({again})"));
        }
        let f = Expression::compile(source).unwrap();
        let g = Expression::compile(wanted).unwrap();
        let apart = points.iter().any(|&z| {
            let (a, b) = (f.call(z), g.call(z));
            if a.is_nan() && b.is_nan() {
                return false;
            }
            if !a.is_finite() || !b.is_finite() {
                return !non_finite_agree(a, b);
            }
            (a - b).magnitude() > 1e-12 * b.magnitude().max(1.0)
        });
        if apart {
            unlike.push(format!("{source} / {wanted}"));
        }
    }
    assert!(wrong.is_empty(), "{} canonical spellings: {}", canonical.len(), wrong.join("; "));
    assert!(unlike.is_empty(), "each spelling evaluates to its reading's values: {}", unlike.join("; "));
    println!("{} canonical spellings, each a fixed point with its reading's values", canonical.len());

    let errors = index.get("errors").and_then(JsonValue::as_array).expect("errors");
    let mut misplaced = Vec::new();
    for e in errors {
        let text = string(e, "text");
        let parses = e.get("parses").and_then(JsonValue::as_bool).unwrap_or(false);
        match Expression::parse(text) {
            Ok(_) => {
                if !parses {
                    misplaced.push(format!("'{text}' parsed"));
                }
            }
            Err(error) => {
                if parses {
                    misplaced.push(format!("'{text}' was refused: {error}"));
                    continue;
                }
                let position = e.get("position").and_then(JsonValue::as_i64).map(|p| p as usize);
                let length = e.get("length").and_then(JsonValue::as_i64).unwrap_or(1) as usize;
                if error.position != position || error.length != length {
                    misplaced.push(format!(
                        "'{text}' refused at {:?}+{}, frozen {position:?}+{length}",
                        error.position, error.length
                    ));
                }
                // The Swift suite holds the place only; the message is frozen
                // too, and held here as well, but for the one place Swift's
                // message is not the frozen one: of two near names equally
                // near, difflib (which wrote the fixture) lists the later in
                // the alphabet first, and Swift the earlier (crates/FINDINGS.md).
                // The port writes Swift's.
                let frozen = e.get("message").and_then(JsonValue::as_str);
                let swifts = SWIFT_MESSAGES.iter().find(|(t, _)| *t == text).map(|(_, m)| *m);
                if let Some(message) = swifts.or(frozen)
                    && error.message != message
                {
                    misplaced.push(format!("'{text}' refused with \"{}\", frozen \"{message}\"", error.message));
                }
            }
        }
    }
    assert!(
        misplaced.is_empty(),
        "{} malformed expressions refused at the same character: {}",
        errors.len(),
        misplaced.join("; ")
    );
    println!("{} malformed expressions refused at the same character, with the same message", errors.len());
    let near = Expression::parse("gama(z)").unwrap_err();
    assert!(near.message.contains("gamma"), "a near miss suggests the name it nearly is: {}", near.message);
}

/// The refusals whose message Swift writes otherwise than `language.json`
/// froze it, and what Swift writes: `Language.unknown` breaks a tie in
/// nearness by the name, ascending.
const SWIFT_MESSAGES: [(&str, &str); 1] = [("gama(z)", "there is no 'gama'; did you mean gamma or lgamma or rgamma?")];

#[test]
fn the_language_table_is_the_frozen_one() {
    let index = shared_json("expr/language.json");
    let language = index.get("language").expect("language");
    let functions = language.get("functions").and_then(JsonValue::as_array).expect("functions");
    let mine = Language::functions();
    let theirs: Vec<&str> = functions.iter().map(|f| string(f, "name")).collect();
    let ours: Vec<&str> = mine.iter().map(|f| f.name).collect();
    assert_eq!(theirs, ours, "the same {} functions, in the same order", functions.len());
    let mut differ = Vec::new();
    for (theirs, ours) in functions.iter().zip(mine) {
        // `LanguageFunction(json:)`'s defaults: arity 1, no aliases, no help.
        let arity = theirs.get("arity").and_then(JsonValue::as_i64).unwrap_or(1) as usize;
        let aliases: Vec<&str> = theirs
            .get("aliases")
            .and_then(JsonValue::as_array)
            .map(|a| a.iter().map(|x| x.as_str().unwrap()).collect())
            .unwrap_or_default();
        let help = theirs.get("help").and_then(JsonValue::as_str).unwrap_or("");
        if arity != ours.arity || aliases != ours.aliases || help != ours.help {
            differ.push(format!(
                "{}: arity {arity}/{}, aliases {aliases:?}/{:?}, help {}",
                ours.name,
                ours.arity,
                ours.aliases,
                if help == ours.help { "same" } else { "differs" }
            ));
        }
    }
    assert!(differ.is_empty(), "with the same arities, aliases and help: {}", differ.join("; "));
    let constants: Vec<&str> = language
        .get("constants")
        .and_then(JsonValue::as_array)
        .expect("constants")
        .iter()
        .map(|c| string(c, "name"))
        .collect();
    let our_constants: Vec<&str> = Language::constants().iter().map(|c| c.name).collect();
    assert_eq!(constants, our_constants, "and the same constants");
    assert_eq!(language.get("variable").and_then(JsonValue::as_str), Some("z"), "and z is the variable");
    println!("the same {} functions and {} constants", functions.len(), constants.len());
}
