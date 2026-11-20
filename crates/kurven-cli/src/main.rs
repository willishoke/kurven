//! `kurven-cli` in Rust: the port's half of the gate, and later everything
//! `Sources/kurven-cli` does.
//!
//! ```text
//! kurven-cli fixtures <dir>
//!     Write the port's golden sets under <dir> (crates/fixtures/rust).
//!     scripts/check-web.sh runs this and fails on any difference from the
//!     commit, and requires the wasm module to write the same bytes.
//!
//! kurven-cli compare A.svg B.svg [--tolerance D] [--length-tolerance R] [--skip I,J,...]
//!     Compare two drawings of one scene per layer: the same layers in the
//!     same order, the symmetric Hausdorff distance between each layer's
//!     strokes under D page units (1e-3 by default), and each layer's total
//!     length equal to R relative (1e-3). Stroke counts are reported, and a
//!     difference noted. The layers --skip lists, by their place in the
//!     SVG, are measured and reported but left out of the verdict. Exit 0 when the two are the same within the
//!     tolerances, 1 when they differ, 2 when one cannot be read.
//!
//! kurven-cli landscape [EXPRESSION | --function NAME] [--re LO,HI] [--im LO,HI]
//!           [--res N] [--truncation H | --bands BELOW:HEIGHT,... --beyond H]
//!           [--spacing S] [--isophotes ... --hatching ...] [--slope-lines C ...] -o out.kurven
//!     Sample a function over a rectangle and write its bundle, as Swift's
//!     `landscape` does.
//!
//! kurven-cli surface solid [--kind NAME] [--size S] [--hole H] [--depth D]
//!           [--hatching uniform|turing|half-lambert ...] [--save B] [-o out.svg]
//!     A polyhedron by name, baked to SVG or kept as a bundle, as Swift's
//!     `surface solid` does.
//!
//! kurven-cli scene A.kurven [--at X,Y,Z[,S]] B.kurven [--at ...] ... [--preset NAME]
//!           [--no-refine] -o out.svg
//!     Several bundles drawn as one composite, each placed by the --at after
//!     it, as Swift's `scene` does.
//!
//! kurven-cli bake <bundle> [--preset NAME] [--no-refine] [--resolution N]
//!           [--tiles N] [--depth-tolerance T] -o out.svg
//!     A landscape's or a parametric surface's bundle drawn to SVG, as Swift's
//!     `bake` writes it, a landscape's contours placed by its function unless
//!     --no-refine. A surface's coordinate image is rasterized on the CPU;
//!     a landscape's or a polyhedron's depth pass is not rendered, so its
//!     --outline is refused.
//!
//! kurven-cli surface [torus | sn | periodic | forced] [options] [--save B] [-o out.svg]
//!     A parametric surface's drawing, as Swift's `surface` builds it: saved
//!     as a bundle with --save, baked to -o.
//!
//! kurven-cli references <fixtures> <dir> [--only NAME,...]
//!     Bake the landscape reference drawings into <dir> as Swift's
//!     `references` bakes them, with index.json.
//!
//! kurven-cli app BUNDLE [--set K=V] [--levels L=N] [--strokes L=N] [--relief A]
//!           [--truncation H] [--pick PX,PY] [--add NAME] [--remove LAYER]
//!           [--function EXPR [--periods P1,P2]] [--undo] [--redo] ... [--save B] [--bake S]
//!     The app's command line with no window (`KurvenApp BUNDLE ...`), its
//!     edits run in order through the session (kurven-session), what it
//!     saves and bakes written where it says.
//!
//! kurven-cli session REQUESTS [-o out.svg] [--state S.json] [--edited E.json] [--stage NAME DIR]... [--save NAME DIR]
//!     The page's requests (one JSON object a line, as the page sends them to
//!     its Worker) run in order through the session, then the judged strokes
//!     of the document they left written as the page's Export writes them.
//! ```

use std::path::Path;
use std::process::ExitCode;

use kurven_cli::args::{Args, CliError};
use kurven_cli::commands::{bake_command, landscape_command};
use kurven_cli::compare::{Tolerance, compare_skipping};
use kurven_cli::references::references_command;
use kurven_cli::scene::scene_command;
use kurven_cli::surface::surface_command;
use kurven_cli::svg;

const USAGE: &str = "usage: kurven-cli fixtures <dir>
       kurven-cli compare A.svg B.svg [--tolerance D] [--length-tolerance R] [--skip I,J,...]
       kurven-cli landscape [EXPRESSION | --function NAME] [options] -o out.kurven
       kurven-cli bake <bundle> [--preset NAME] [--no-refine] [--outline W] [--dump PREFIX] -o out.svg
       kurven-cli references <fixtures> <dir> [--only NAME,...]
       kurven-cli surface [torus | sn | periodic | forced] [options] [--save B] [-o out.svg]
       kurven-cli surface solid [--kind NAME] [options] [--save B] [-o out.svg]
       kurven-cli scene A.kurven B.kurven [--at X,Y,Z[,S]] ... [--preset NAME] [--no-refine] -o out.svg
       kurven-cli app BUNDLE [edits as the app takes them] [--save B] [--bake S]
       kurven-cli session REQUESTS [-o out.svg] [--state S.json] [--edited E.json] [--stage NAME DIR]... [--save NAME DIR]";

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
    let mut skip = Vec::new();
    let mut words = args.iter();
    while let Some(word) = words.next() {
        match word.as_str() {
            "--tolerance" => tolerance.distance = positive(word, words.next())?,
            "--length-tolerance" => tolerance.length = positive(word, words.next())?,
            "--skip" => {
                let list = words.next().ok_or("--skip wants layer indices, as in 1,3")?;
                for i in list.split(',') {
                    skip.push(
                        i.parse::<usize>().map_err(|_| format!("--skip wants layer indices, as in 1,3, not {list}"))?,
                    );
                }
            }
            flag if flag.starts_with("--") => return Err(format!("compare takes no {flag}\n{USAGE}")),
            file => files.push(file),
        }
    }
    let [a, b] = files[..] else { return Err(USAGE.into()) };
    let (da, db) = (svg::read(a)?, svg::read(b)?);
    let report = compare_skipping(&da, &db, tolerance, &skip);
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
        Some("surface") => return swift_command(surface_command),
        Some("session") => kurven_cli::session::session_command(&args[1..]).map(|()| ExitCode::SUCCESS),
        Some("app") => return ExitCode::from(kurven_cli::app::app_command(&args[1..]).clamp(0, 255) as u8),
        Some("scene") => {
            return match scene_command(&args[1..]) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("kurven-cli: {error}");
                    ExitCode::from(1)
                }
            };
        }
        _ => Err(USAGE.into()),
    };
    result.unwrap_or_else(|message| {
        eprintln!("kurven-cli: {message}");
        ExitCode::from(2)
    })
}
