//! `kurven-cli references <fixtures> <dir>`: the reference drawings, baked by
//! the port as `Sources/kurven-cli/References.swift` bakes them in Swift, so
//! `kurven-cli compare` can hold each to Swift's (`crates/fixtures/swift`).
//!
//! Each is drawn by the path its `how` names, as in Swift: a bundle written
//! by `landscape` and drawn by `bake`, and the golden slope lines as `kurven-cli
//! fixtures` draws them. Written into `<dir>`: one `<name>.svg` per drawing
//! and `index.json`, in Swift's form (each drawing with how it was made, the
//! names of its non-empty stroke layers in the order the SVG's groups stand,
//! and their stroke counts).
//!
//! The port draws every drawing of Swift's list: the landscapes, the
//! surfaces (sn flat and in relief at 0.4 and at 1, a torus with a linear
//! curve, a periodic expression), the seven solids, the frame's two gallery
//! drawings and the composite of the cubic, a cube and a torus. Swift bakes
//! the two reliefs, which the coordinate image alone tests, with that image
//! rendered by the CPU rasterizer as the port renders every one.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use kurven_bake::{BakeOptions, bake};
use kurven_core::hidden_line::HiddenLine;
use kurven_core::json::{self, JsonObject, JsonValue};
use kurven_core::svg::{Strokes, Style};

use kurven_golden::gallery::{self, camera};

use crate::args::{Args, CliError};
use crate::commands::{landscape_command, load_drawing, write_svg};
use crate::surface::surface_command;

/// A drawing: its strokes and the name of the layer each stroke layer draws.
pub type Drawn = (Strokes, Vec<String>);

type Draw = Box<dyn Fn(&Path) -> Result<Drawn, CliError>>;

/// One drawing: its file stem, the commands it stands for, and how to draw
/// it into a scratch directory.
pub struct Reference {
    pub name: String,
    pub how: String,
    pub draw: Draw,
}

fn words(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

/// A bundle's drawing as `bake` draws it: under `preset` or its first, at
/// that preset's depth resolution and tolerance, its contours placed by its
/// function unless not `refine`.
pub fn baked(bundle: &Path, preset: Option<&str>, refine: bool) -> Result<Drawn, CliError> {
    let mut argv = vec!["bake".to_string(), bundle.to_string_lossy().into_owned()];
    if let Some(name) = preset {
        argv.extend(["--preset".to_string(), name.to_string()]);
    }
    if !refine {
        argv.push("--no-refine".to_string());
    }
    let drawing = load_drawing(&Args::new(&argv))?;
    let options = BakeOptions {
        resolution: drawing.preset.buffer,
        tiles: None,
        depth_tolerance: Some(drawing.preset.depth_tolerance),
        outline: None,
    };
    let result = bake(&drawing.scene, &options)?;
    let names = result.specs.iter().map(|s| s.name.clone()).collect();
    Ok((result.strokes, names))
}

/// What a reference's `how` adds for a surface Swift bakes with its
/// coordinate image rasterized on the CPU (References.swift's
/// `rasterizedHow`).
const RASTERIZED_HOW: &str = ", its coordinate image by SurfaceImage.rasterize in place of Metal's";

/// `surface ... --save B; bake B`.
pub fn surface(words: &[String], scratch: &Path, stem: &str) -> Result<Drawn, CliError> {
    let bundle = scratch.join(format!("{stem}.kurven"));
    let mut argv = vec!["surface".to_string()];
    argv.extend(words.iter().cloned());
    argv.extend(["--save".to_string(), bundle.to_string_lossy().into_owned()]);
    surface_command(&Args::new(&argv))?;
    baked(&bundle, None, true)
}

// MARK: the gallery, as docs/gallery/make.py sets it up

/// `landscape` as make.py calls it, then the manifest edited as make.py
/// edits it: one preset per camera, each the first preset with its name and
/// its projection's fields replaced, and the plateaus', walls' and risers'
/// hatching `gallery::HATCH` closer (`gallery::edit`).
pub fn gallery_landscape(
    scratch: &Path,
    stem: &str,
    words: &[String],
    cameras: &[(String, JsonObject)],
) -> Result<PathBuf, CliError> {
    let bundle = scratch.join(format!("{stem}.kurven"));
    let mut argv = vec!["landscape".to_string()];
    argv.extend(words.iter().cloned());
    argv.extend(["-o".to_string(), bundle.to_string_lossy().into_owned()]);
    landscape_command(&Args::new(&argv))?;
    let path = bundle.join("manifest.json");
    let text = std::fs::read_to_string(&path).map_err(|e| CliError(format!("{}: {e}", path.display())))?;
    let mut m = json::parse(&text)?.object("manifest")?.clone();
    gallery::edit(&mut m, stem, cameras).map_err(CliError)?;
    std::fs::write(&path, JsonValue::Object(m).canonical())
        .map_err(|e| CliError(format!("{}: {e}", path.display())))?;
    Ok(bundle)
}

/// A double as Swift's string interpolation writes it (`description`).
fn swift(x: f64) -> String {
    json::number(x)
}

fn plate(name: &str, how: String, draw: Draw) -> Reference {
    Reference { name: name.to_string(), how, draw }
}

pub fn gallery() -> Vec<Reference> {
    let gamma_words = words(&[
        "gamma(z)",
        "--re",
        "-4.5,4.5",
        "--im",
        "-2.5,2.5",
        "--res",
        "1000",
        "--bands",
        "-3.5:2.5,-2.5:3.0,-1.5:4.0,0.5:5.5",
        "--beyond",
        "4.0",
    ]);
    // cn's quarter periods by the AGM, as make.py computes them.
    let m = gallery::CN_MODULUS;
    let (k, kp) = gallery::cn_quarter_periods();
    let cn_words = vec![
        format!("cn(z, {})", swift(m)),
        "--re".to_string(),
        format!("{},{}", swift(-k), swift(5.0 * k)),
        "--im".to_string(),
        format!("0,{}", swift(3.0 * kp)),
        "--res".to_string(),
        "1200".to_string(),
        "--truncation".to_string(),
        "4".to_string(),
    ];
    let cubic_words =
        words(&["1/(z^3 - 1)", "--re", "-1.3,1.6", "--im", "-1.2,1.5", "--res", "300", "--truncation", "5"]);
    let corners: Vec<(String, JsonObject)> = [(-90, 0.5), (-90, -0.5), (90, 0.5), (90, -0.5)]
        .into_iter()
        .map(|(z, s)| (format!("t{z}s{}", swift(s)).replace('-', "m"), camera(s, -55.0, f64::from(z), true)))
        .collect();

    let mut references = Vec::new();
    let w = gamma_words.clone();
    references.push(plate(
        "gallery-gamma",
        format!(
            "docs/gallery/make.py: landscape {}, hatching x0.75, preset plate (shear -0.5, x -55, z -90); bake --preset plate",
            gamma_words.join(" ")
        ),
        Box::new(move |scratch| {
            let cameras = [("plate".to_string(), camera(-0.5, -55.0, -90.0, false))];
            let bundle = gallery_landscape(scratch, "gamma", &w, &cameras)?;
            baked(&bundle, Some("plate"), true)
        }),
    ));
    let w = cn_words.clone();
    references.push(plate(
        "gallery-cn",
        format!(
            "docs/gallery/make.py: landscape {}, hatching x0.75, preset plate (shear 0.51, x -63, z -90, flipX); bake --preset plate",
            cn_words.join(" ")
        ),
        Box::new(move |scratch| {
            let cameras = [("plate".to_string(), camera(0.51, -63.0, -90.0, true))];
            let bundle = gallery_landscape(scratch, "cn", &w, &cameras)?;
            baked(&bundle, Some("plate"), true)
        }),
    ));
    // make.py samples the cubic once and bakes each corner's preset from
    // that one bundle; so does this.
    let cubic: Rc<RefCell<Option<PathBuf>>> = Rc::new(RefCell::new(None));
    for (key, _) in &corners {
        let (sampled, key2, w, cameras) = (cubic.clone(), key.clone(), cubic_words.clone(), corners.clone());
        references.push(plate(
            &format!("gallery-cubic-{key}"),
            format!(
                "docs/gallery/make.py: landscape {}, hatching x0.75, the four corner presets; bake --preset {key}",
                cubic_words.join(" ")
            ),
            Box::new(move |scratch| {
                let known = sampled.borrow().clone();
                let bundle = match known {
                    Some(bundle) => bundle,
                    None => gallery_landscape(scratch, "cubic", &w, &cameras)?,
                };
                *sampled.borrow_mut() = Some(bundle.clone());
                baked(&bundle, Some(&key2), true)
            }),
        ));
    }
    let frame = words(&[
        "solid",
        "--kind",
        "frame",
        "--size",
        "2",
        "--hole",
        "2",
        "--depth",
        "0.8",
        "--x-angle",
        "-55",
        "--z-angle",
        "-60",
        "--shear",
        "0",
        "--width",
        "0.6",
    ]);
    let turing = words(&[
        "--hatching",
        "turing",
        "--angle",
        "30",
        "--spacing",
        "0.035",
        "--width-steps",
        "4",
        "--scale",
        "0.18",
        "--seed",
        "3",
        "--hatch-width",
        "0.5",
    ]);
    let w = frame.clone();
    references.push(plate(
        "gallery-frame-edges",
        format!("docs/gallery/make.py: surface {} --save B; bake B", frame.join(" ")),
        Box::new(move |scratch| surface(&w, scratch, "frame-edges")),
    ));
    let w: Vec<String> = frame.iter().chain(&turing).cloned().collect();
    references.push(plate(
        "gallery-frame-turing",
        format!("docs/gallery/make.py: surface {} --save B; bake B", w.join(" ")),
        Box::new(move |scratch| surface(&w, scratch, "frame-turing")),
    ));
    references
}

// MARK: the rest of the list

pub fn all(fixtures: &Path) -> Vec<Reference> {
    let mut references = gallery();
    // The native landscapes of expr/landscapes.json, drawn on their function.
    for stem in ["gamma", "cubic", "typed", "tower"] {
        let bundle = fixtures.join(format!("expr/landscape_{stem}.kurven"));
        references.push(plate(
            &format!("landscape-{stem}"),
            format!("bake tests/fixtures/expr/landscape_{stem}.kurven"),
            Box::new(move |_| baked(&bundle, None, true)),
        ));
    }
    // Each with whether the coordinate image tests it: sn in relief past
    // its reach, 1/3 (0.4, and 1, the most the range allows), bounds no
    // solid; the rest are sphere traced. The port renders every coordinate
    // image by the CPU rasterizer, and Swift bakes these with the same in
    // place of its Metal pass (References.swift's `rasterizedBake`), whose
    // float32 pass keeps another sheet at a few pixels where two sheets of
    // the relief tie in depth; their `how` says so.
    let surfaces: [(&str, &[&str], bool); 5] = [
        ("sn", &["sn"], false),
        ("sn-relief", &["sn", "--relief", "0.4"], true),
        ("sn-relief-1", &["sn", "--relief", "1"], true),
        ("torus-linear", &["torus", "--linear-curves", "2/5"], false),
        (
            "periodic",
            &["periodic", "--expression", "sn(z, 0.5)", "--periods", "7.416298709205487,3.7081493546027433"],
            false,
        ),
    ];
    for (name, list, rasterized) in surfaces {
        let list = words(list);
        references.push(plate(
            name,
            format!("surface {} --save B; bake B{}", list.join(" "), if rasterized { RASTERIZED_HOW } else { "" }),
            Box::new(move |scratch| surface(&list, scratch, name)),
        ));
    }
    // The slope lines of |f| golden (PR #60), drawn as `kurven-cli
    // fixtures` draws it: tests/fixtures/fields/landscape_slope_lines.svg.
    let slopes = fixtures.join("fields/landscape_slope_lines.kurven");
    references.push(plate(
        "slope-lines",
        "kurven-cli fixtures: tests/fixtures/fields/landscape_slope_lines.svg".to_string(),
        Box::new(move |_| {
            let drawing = load_drawing(&Args::new(&["bake".to_string(), slopes.to_string_lossy().into_owned()]))?;
            let layers = drawing.scene.visible_layers(None, None)?;
            let strokes = Strokes {
                layers: layers.iter().map(|(l, seen)| (Style::of(&l.spec), HiddenLine::pass(seen))).collect(),
            };
            Ok((strokes, layers.iter().map(|(l, _)| l.spec.name.clone()).collect()))
        }),
    ));
    for kind in ["tetrahedron", "cube", "octahedron", "dodecahedron", "icosahedron", "csaszar", "frame"] {
        references.push(plate(
            &format!("solid-{kind}"),
            format!("surface solid --kind {kind} --save B; bake B"),
            Box::new(move |scratch| surface(&words(&["solid", "--kind", kind]), scratch, &format!("solid-{kind}"))),
        ));
    }
    // The cubic landscape on its grid with the contract's cube and torus
    // standing on it: tests/fixtures/composite/three.svg.
    references.push(composite_three(fixtures));
    references
}

/// composite-three: tests/fixtures/composite/three.svg, byte for byte in
/// Swift, as `scene` draws it: the cubic landscape on its grid, a
/// polyhedron's bundle and a parametric surface's.
pub fn composite_three(fixtures: &Path) -> Reference {
    let parts: Vec<(String, kurven_core::composite::Placement)> = [
        ("expr/landscape_cubic.kurven", "0,0,0,1"),
        ("contract/polyhedron_cube.kurven", "-1.2,0.1,0.3,0.35"),
        ("contract/surface_torus.kurven", "1,0,1.6,0.35"),
    ]
    .iter()
    .map(|(p, at)| (fixtures.join(p).to_string_lossy().into_owned(), crate::scene::placement(at).expect("a placement")))
    .collect();
    plate(
        "composite-three",
        "scene expr/landscape_cubic.kurven contract/polyhedron_cube.kurven --at -1.2,0.1,0.3,0.35 \
         contract/surface_torus.kurven --at 1,0,1.6,0.35 --no-refine"
            .to_string(),
        Box::new(move |_| {
            let drawing = crate::scene::composite_drawing(&parts, None, false)?;
            Ok((drawing.strokes, drawing.layers.iter().map(|(l, _)| l.spec.name.clone()).collect()))
        }),
    )
}

/// Bake every reference whose name `only` holds (all of them for an empty
/// list) into `dir`, with `index.json`.
pub fn write(fixtures: &Path, dir: &Path, only: &[String]) -> Result<(), CliError> {
    let io = |p: &Path, e: std::io::Error| CliError(format!("{}: {e}", p.display()));
    std::fs::create_dir_all(dir).map_err(|e| io(dir, e))?;
    let scratch = std::env::temp_dir().join(format!("kurven-references-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).map_err(|e| io(&scratch, e))?;
    let result = (|| {
        let mut entries = Vec::new();
        let references = all(fixtures);
        if let Some(unknown) = only.iter().find(|n| !references.iter().any(|r| &&r.name == n)) {
            return Err(CliError(format!("no reference drawing called {unknown}")));
        }
        for reference in references.iter().filter(|r| only.is_empty() || only.contains(&r.name)) {
            let started = std::time::Instant::now();
            let (strokes, names) = (reference.draw)(&scratch)?;
            if names.len() != strokes.layers.len() {
                return Err(CliError(format!(
                    "{}: {} names for {} layers",
                    reference.name,
                    names.len(),
                    strokes.layers.len()
                )));
            }
            write_svg(&strokes, &dir.join(format!("{}.svg", reference.name)))?;
            // As the SVG has them: the layers with a stroke, in order.
            let drawn: Vec<usize> = (0..strokes.layers.len()).filter(|&i| !strokes.layers[i].1.is_empty()).collect();
            let mut entry = JsonObject::new();
            entry.insert("name".to_string(), JsonValue::String(reference.name.clone()));
            entry.insert("how".to_string(), JsonValue::String(reference.how.clone()));
            entry.insert(
                "layers".to_string(),
                JsonValue::Array(drawn.iter().map(|&i| JsonValue::String(names[i].clone())).collect()),
            );
            entry.insert(
                "paths".to_string(),
                JsonValue::Array(drawn.iter().map(|&i| JsonValue::Int(strokes.layers[i].1.count() as i64)).collect()),
            );
            entries.push(JsonValue::Object(entry));
            eprintln!("  {} ({:.2} s)", reference.name, started.elapsed().as_secs_f64());
        }
        let index: JsonObject = [("drawings".to_string(), JsonValue::Array(entries))].into_iter().collect();
        let path = dir.join("index.json");
        std::fs::write(&path, JsonValue::Object(index).pretty(1, false) + "\n").map_err(|e| io(&path, e))
    })();
    let _ = std::fs::remove_dir_all(&scratch);
    result
}

/// `references <fixtures> <dir> [--only NAME,...]`. `--only` is the port's:
/// Swift's command takes no options.
pub fn references_command(args: &Args) -> Result<(), CliError> {
    args.only("references", &["only"])?;
    let words = &args.positional[1..];
    let [fixtures, dir] = words else {
        return Err(CliError(
            "references wants <fixtures> <dir>: tests/fixtures and where to write the drawings".to_string(),
        ));
    };
    let only: Vec<String> = args
        .flag("only")
        .map(|o| o.split(',').filter(|s| !s.is_empty()).map(String::from).collect())
        .unwrap_or_default();
    let started = std::time::Instant::now();
    write(Path::new(fixtures), Path::new(dir), &only)?;
    println!("wrote the reference drawings under {dir} in {:.2} s", started.elapsed().as_secs_f64());
    Ok(())
}
