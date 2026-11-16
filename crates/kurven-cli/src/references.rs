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
//! The port draws the landscapes so far. Swift's list also holds sn flat and
//! in relief, a torus with a linear curve, a periodic expression and the
//! frame's two gallery drawings (parametric surfaces and solids built by
//! `surface`, Phase 4 and 5), the seven solids (Phase 5) and the composite
//! (Phase 5); they join this list with their phases.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use kurven_bake::{BakeOptions, bake};
use kurven_core::hidden_line::HiddenLine;
use kurven_core::json::{self, JsonObject, JsonValue};
use kurven_core::svg::{Strokes, Style};

use crate::args::{Args, CliError};
use crate::commands::{landscape_command, load_drawing, write_svg};

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

// MARK: the gallery, as docs/gallery/make.py sets it up

/// The plates hatch the plateaus, walls and risers this much closer than a
/// landscape does by default (make.py's `HATCH`).
pub const GALLERY_HATCH: f64 = 0.75;

/// `landscape` as make.py calls it, then the manifest edited as make.py
/// edits it: one preset per camera, each the first preset with its name and
/// its projection's fields replaced, and the plateaus', walls' and risers'
/// hatching `GALLERY_HATCH` closer.
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
    let (base, projection) = match m.get("presets").and_then(JsonValue::as_array).and_then(|p| p.first()) {
        Some(JsonValue::Object(base)) => match base.get("projection") {
            Some(JsonValue::Object(projection)) => (base.clone(), projection.clone()),
            _ => return Err(CliError(format!("{stem}: the landscape wrote no preset to start from"))),
        },
        _ => return Err(CliError(format!("{stem}: the landscape wrote no preset to start from"))),
    };
    let presets = cameras
        .iter()
        .map(|(name, fields)| {
            let mut preset = base.clone();
            preset.insert("name".to_string(), JsonValue::String(name.clone()));
            let mut merged = projection.clone();
            merged.extend(fields.iter().map(|(k, v)| (k.clone(), v.clone())));
            preset.insert("projection".to_string(), JsonValue::Object(merged));
            JsonValue::Object(preset)
        })
        .collect();
    m.insert("presets".to_string(), JsonValue::Array(presets));
    let Some(JsonValue::Array(layers)) = m.get("layers") else { return Err(CliError(format!("{stem}: no layers"))) };
    let layers = layers.iter().map(|layer| closer(layer).unwrap_or_else(|| layer.clone())).collect();
    m.insert("layers".to_string(), JsonValue::Array(layers));
    std::fs::write(&path, JsonValue::Object(m).canonical())
        .map_err(|e| CliError(format!("{}: {e}", path.display())))?;
    Ok(bundle)
}

/// A hatching layer on the plateaus, walls or risers, `GALLERY_HATCH`
/// closer; None for any other layer, which stays as it is.
fn closer(layer: &JsonValue) -> Option<JsonValue> {
    let mut l = layer.as_object()?.clone();
    let mut source = l.get("source")?.as_object()?.clone();
    if source.get("kind")?.as_str()? != "hatching" {
        return None;
    }
    let kind = source.get("faces")?.as_object()?.get("kind")?.as_str()?;
    if !["plateaus", "walls", "risers"].contains(&kind) {
        return None;
    }
    let spacing = match source.get("spacing")? {
        JsonValue::Double(s) => s * GALLERY_HATCH,
        JsonValue::Int(s) => *s as f64 * GALLERY_HATCH,
        _ => return None,
    };
    source.insert("spacing".to_string(), JsonValue::Double(spacing));
    l.insert("source".to_string(), JsonValue::Object(source));
    Some(JsonValue::Object(l))
}

pub fn camera(shear: f64, x_angle: f64, z_angle: f64, flip_x: bool) -> JsonObject {
    [
        ("shear", JsonValue::Double(shear)),
        ("xAngle", JsonValue::Double(x_angle)),
        ("zAngle", JsonValue::Double(z_angle)),
        ("flipX", JsonValue::Bool(flip_x)),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect()
}

/// The arithmetic-geometric mean, as make.py computes it for cn's quarter
/// periods.
pub fn agm(a0: f64, b0: f64) -> f64 {
    let (mut a, mut b) = (a0, b0);
    while (a - b).abs() > 1e-15 * a {
        (a, b) = ((a + b) / 2.0, (a * b).sqrt());
    }
    a
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
    let m: f64 = 0.64;
    let k = core::f64::consts::PI / (2.0 * agm(1.0, (1.0 - m).sqrt()));
    let kp = core::f64::consts::PI / (2.0 * agm(1.0, m.sqrt()));
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
    // PORT: phase 4 -- sn, sn-relief, torus-linear, periodic and the frame's
    // two gallery drawings (`surface ... --save B; bake B`).
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
    // PORT: phase 5 -- solid-{tetrahedron, cube, octahedron, dodecahedron,
    // icosahedron, csaszar, frame} and composite-three.
    references
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
