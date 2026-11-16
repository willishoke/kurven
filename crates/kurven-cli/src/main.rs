//! `kurven-cli` in Rust: the port's half of the gate, and later everything
//! `Sources/kurven-cli` does.
//!
//! ```text
//! kurven-cli fixtures <dir>
//!     Write the port's golden sets under <dir> (crates/fixtures/rust).
//!     scripts/check-web.sh runs this and fails on any difference from the
//!     commit, and requires the wasm module to write the same bytes.
//!
//! kurven-cli compare A.svg B.svg [--tolerance D] [--length-tolerance R]
//!     Compare two drawings of one scene per layer: the same layers in the
//!     same order, the symmetric Hausdorff distance between each layer's
//!     strokes under D page units (1e-3 by default), and each layer's total
//!     length equal to R relative (1e-3). Stroke counts are reported, and a
//!     difference noted. Exit 0 when the two are the same within the
//!     tolerances, 1 when they differ, 2 when one cannot be read.
//!
//! kurven-cli landscape [EXPRESSION | --function NAME] [--re LO,HI] [--im LO,HI]
//!           [--res N] [--truncation H | --bands BELOW:HEIGHT,... --beyond H]
//!           [--spacing S] [--isophotes ... --hatching ...] [--slope-lines C ...] -o out.kurven
//!     Sample a function over a rectangle and write its bundle, as Swift's
//!     `landscape` does.
//!
//! kurven-cli bake <bundle> [--preset NAME] [--no-refine] [--resolution N]
//!           [--tiles N] [--depth-tolerance T] -o out.svg
//!     A landscape bundle's drawing to SVG, as Swift's `bake` writes it, its
//!     contours placed by its function unless --no-refine. Strokes only: no
//!     depth buffer is rendered, so --outline and --dump are refused, and a
//!     surface's or a polyhedron's bundle waits for Phase 4 or 5.
//!
//! kurven-cli references <fixtures> <dir> [--only NAME,...]
//!     Bake the landscape reference drawings into <dir> as Swift's
//!     `references` bakes them, with index.json.
//! ```

use std::path::Path;
use std::process::ExitCode;

use kurven_cli::args::{Args, CliError};
use kurven_cli::commands::{bake_command, landscape_command};
use kurven_cli::compare::{Tolerance, compare};
use kurven_cli::references::references_command;
use kurven_cli::svg;

const USAGE: &str = "usage: kurven-cli fixtures <dir>
       kurven-cli compare A.svg B.svg [--tolerance D] [--length-tolerance R]
       kurven-cli landscape [EXPRESSION | --function NAME] [options] -o out.kurven
       kurven-cli bake <bundle> [--preset NAME] [--no-refine] -o out.svg
       kurven-cli references <fixtures> <dir> [--only NAME,...]";

/// A command ported from Swift's kurven-cli: Swift's argument reading, and
/// its exit status 1 with `kurven-cli: <error>` on a refusal.
fn swift_command(run: fn(&Args) -> Result<(), CliError>) -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match run(&Args::new(&argv)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("kurven-cli: {error}");
            ExitCode::from(1)
        }
    }
}

fn fixtures(args: &[String]) -> Result<ExitCode, String> {
    let [dir] = args else { return Err(USAGE.into()) };
    let goldens = kurven_golden::all();
    for golden in &goldens {
        let path = Path::new(dir).join(golden.path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        std::fs::write(&path, &golden.bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    println!("wrote {} golden file(s) under {dir}", goldens.len());
    Ok(ExitCode::SUCCESS)
}

fn positive(flag: &str, value: Option<&String>) -> Result<f64, String> {
    let text = value.ok_or_else(|| format!("{flag} wants a value"))?;
    match text.parse::<f64>() {
        Ok(v) if v > 0.0 && v.is_finite() => Ok(v),
        _ => Err(format!("{flag} wants a positive number, not {text}")),
    }
}

fn compare_command(args: &[String]) -> Result<ExitCode, String> {
    let mut tolerance = Tolerance::default();
    let mut files = Vec::new();
    let mut words = args.iter();
    while let Some(word) = words.next() {
        match word.as_str() {
            "--tolerance" => tolerance.distance = positive(word, words.next())?,
            "--length-tolerance" => tolerance.length = positive(word, words.next())?,
            flag if flag.starts_with("--") => return Err(format!("compare takes no {flag}\n{USAGE}")),
            file => files.push(file),
        }
    }
    let [a, b] = files[..] else { return Err(USAGE.into()) };
    let (da, db) = (svg::read(a)?, svg::read(b)?);
    let report = compare(&da, &db, tolerance);
    println!("a: {a}\nb: {b}\n{report}");
    Ok(if report.same() { ExitCode::SUCCESS } else { ExitCode::from(1) })
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("fixtures") => fixtures(&args[1..]),
        Some("compare") => compare_command(&args[1..]),
        Some("landscape") => return swift_command(landscape_command),
        Some("bake") => return swift_command(bake_command),
        Some("references") => return swift_command(references_command),
        _ => Err(USAGE.into()),
    };
    result.unwrap_or_else(|message| {
        eprintln!("kurven-cli: {message}");
        ExitCode::from(2)
    })
}
