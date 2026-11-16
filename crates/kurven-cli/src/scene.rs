//! `scene` of `Sources/kurven-cli/main.swift`: several bundles drawn as one
//! composite (`sceneCommand`, `compositeDrawing`).

use std::path::Path;
use std::time::Instant;

use kurven_core::composite::{Composite, Placement};
use kurven_core::hidden_line::HiddenLine;
use kurven_core::polyline::{Layer, PolylineSet};
use kurven_core::simd::Vec3;
use kurven_core::spaces::ViewSpace;
use kurven_core::svg::{Strokes, Style};

use crate::args::{self, Args, CliError};
use crate::commands::{load_drawing, write_svg};

/// A composite's drawing: its parts' names, its layers tested, and their
/// strokes.
pub struct CompositeDrawing {
    pub names: Vec<String>,
    pub layers: Vec<(Layer, PolylineSet<ViewSpace>)>,
    pub strokes: Strokes,
}

/// The composite of bundles at their placements, under the first one's
/// camera (`preset`, or its first): each read as `bake` reads it, its
/// contours placed by its function unless not `refine`.
pub fn composite_drawing(
    bundles: &[(String, Placement)],
    preset: Option<&str>,
    refine: bool,
) -> Result<CompositeDrawing, CliError> {
    let mut parts = Vec::new();
    for (k, (path, placement)) in bundles.iter().enumerate() {
        let mut words = vec!["scene".to_string(), path.clone()];
        if k == 0
            && let Some(preset) = preset
        {
            words.extend(["--preset".to_string(), preset.to_string()]);
        }
        if !refine {
            words.push("--no-refine".to_string());
        }
        let drawing = load_drawing(&Args::new(&words))?;
        let name = drawing.url.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        parts.push((name, drawing.scene, *placement));
    }
    let camera = parts[0].1.camera;
    let names = parts.iter().map(|p| p.0.clone()).collect();
    let composite = Composite::new(parts, camera).map_err(|e| CliError(e.0))?;
    let layers = composite.visible_layers().map_err(|e| CliError(e.0))?;
    let strokes = Strokes {
        layers: layers.iter().map(|(layer, seen)| (Style::of(&layer.spec), HiddenLine::pass(seen))).collect(),
    };
    Ok(CompositeDrawing { names, layers, strokes })
}

/// `X,Y,Z` or `X,Y,Z,S` as a placement.
pub fn placement(text: &str) -> Result<Placement, CliError> {
    let numbers: Vec<Option<f64>> = text.split(',').map(args::double).collect();
    if !(numbers.len() == 3 || numbers.len() == 4) || numbers.iter().any(Option::is_none) {
        return Err(CliError(format!("--at wants X,Y,Z or X,Y,Z,S, got '{text}'")));
    }
    let n: Vec<f64> = numbers.into_iter().map(|x| x.unwrap_or(0.0)).collect();
    Ok(Placement::new(if n.len() == 4 { n[3] } else { 1.0 }, Vec3::new(n[0], n[1], n[2])))
}

/// `scene`: several bundles drawn as one composite (`Composite`), each
/// placed by the `--at` after it. Read off the command line in order, since
/// which bundle an `--at` places is where it stands; `argv` is the words
/// after `scene`.
pub fn scene_command(argv: &[String]) -> Result<(), CliError> {
    let mut bundles: Vec<(String, Option<String>)> = Vec::new();
    let (mut preset, mut output, mut refine) = (None::<String>, None::<String>, true);
    let mut i = 0;
    let value = |i: usize, name: &str| -> Result<String, CliError> {
        argv.get(i + 1).cloned().ok_or_else(|| CliError(format!("{name} wants a value")))
    };
    while i < argv.len() {
        match argv[i].as_str() {
            "--at" => {
                let Some(last) = bundles.last_mut() else {
                    return Err(CliError("--at places the bundle before it, and none is".to_string()));
                };
                if last.1.is_some() {
                    return Err(CliError(format!("{} is placed twice", last.0)));
                }
                last.1 = Some(value(i, "--at")?);
                i += 2;
            }
            "--preset" => {
                preset = Some(value(i, "--preset")?);
                i += 2;
            }
            "-o" => {
                output = Some(value(i, "-o")?);
                i += 2;
            }
            "--no-refine" => {
                refine = false;
                i += 1;
            }
            a if a.starts_with('-') => {
                return Err(CliError(format!("scene takes no {a}; it takes --at --preset --no-refine -o")));
            }
            a => {
                bundles.push((a.to_string(), None));
                i += 1;
            }
        }
    }
    if bundles.len() < 2 {
        return Err(CliError("scene draws two bundles or more".to_string()));
    }
    let Some(output) = output else { return Err(CliError("missing -o".to_string())) };
    let placed = bundles
        .iter()
        .map(|(path, at)| Ok((path.clone(), at.as_deref().map(placement).transpose()?.unwrap_or(Placement::IDENTITY))))
        .collect::<Result<Vec<_>, CliError>>()?;
    let started = Instant::now();
    let drawing = composite_drawing(&placed, preset.as_deref(), refine)?;
    let elapsed = started.elapsed();
    write_svg(&drawing.strokes, Path::new(&output))?;
    println!(
        "{} -> {}\n  strokes    {} paths, length {:.1}\n  took       {:.3} s",
        drawing.names.join(" + "),
        Path::new(&output).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        drawing.strokes.path_count(),
        drawing.strokes.total_length(),
        elapsed.as_secs_f64()
    );
    for (layer, seen) in &drawing.layers {
        println!(
            "    {:<14} {:<7} paths{}",
            layer.spec.name,
            seen.count(),
            if layer.spec.clipped { "" } else { "  unclipped" }
        );
    }
    Ok(())
}
