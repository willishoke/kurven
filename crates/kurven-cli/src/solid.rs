//! `surface solid` of `Sources/kurven-cli/main.swift`: a polyhedron by name,
//! baked straight to SVG or kept as a bundle (`namedSolid`, `solidCommand`,
//! `topology`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use kurven_bake::{BakeOptions, bake};
use kurven_core::camera::Camera;
use kurven_core::hatching::{FaceSelector, Hatching, Tone};
use kurven_core::manifest::{CameraPreset, HeightPolicy, LayerRole, LayerSource, LayerSpec, PolyhedronManifest};
use kurven_core::polyhedron_bundle::PolyhedronBundle;
use kurven_core::scene::{PreviewMode, Scene};
use kurven_core::simd::Vec3;
use kurven_core::solid::Solid;
use kurven_core::surface_request::SurfaceRequest;

use crate::args::{Args, CliError};
use crate::commands::write_svg;

/// A solid's boundary as a number: its genus when it is one shell, or else
/// how many shells and their Euler characteristic together.
pub fn topology(solid: &Solid) -> String {
    match solid.genus() {
        Some(g) => format!("genus {g}"),
        None => format!("{} shells, χ = {}", solid.shell_count(), solid.euler_characteristic()),
    }
}

/// One of the solids `surface solid` builds, by name (`Solid::named`).
pub fn named_solid(kind: &str, size: f64, hole: Option<f64>, depth: Option<f64>) -> Result<Solid, CliError> {
    Solid::named(kind, size, hole, depth).map_err(CliError)
}

/// The options `Args.projection` names.
const PROJECTION: [&str; 3] = ["shear", "x-angle", "z-angle"];

/// `surface solid`: a polyhedron, baked straight to SVG, or with `--save`
/// kept as a bundle of its vertices and loops, which `bake` draws again byte
/// for byte.
pub fn solid_command(args: &Args) -> Result<(), CliError> {
    let mut accepted: Vec<&str> = PROJECTION.to_vec();
    accepted.extend([
        "kind",
        "size",
        "hole",
        "depth",
        "width",
        "color",
        "resolution",
        "tiles",
        "output",
        "hatching",
        "angle",
        "spacing",
        "step",
        "width-steps",
        "seed",
        "scale",
        "hatch-width",
        "hatch-color",
        "save",
    ]);
    args.only("surface solid", &accepted)?;
    let kind = args.flag("kind").unwrap_or("dodecahedron").to_string();
    let size = args.positive("size")?.unwrap_or(2.0);
    let width = args.double("width")?.unwrap_or(0.6);
    let solid = named_solid(&kind, size, args.positive("hole")?, args.positive("depth")?)?;
    let mut projection = SurfaceRequest::PROJECTION;
    projection.shear = args.double("shear")?.unwrap_or(projection.shear);
    projection.x_angle = args.double("x-angle")?.unwrap_or(projection.x_angle);
    projection.z_angle = args.double("z-angle")?.unwrap_or(projection.z_angle);
    let camera = Camera::oblique(&projection);
    let mut edges = LayerSpec::new("edges", LayerRole::Outline, LayerSource::Edges, width, HeightPolicy::Surface);
    edges.color = args.flag("color").unwrap_or("#000000").to_string();
    let mut specs = vec![edges];
    if let Some(tone_name) = args.flag("hatching") {
        let tone = match tone_name {
            "uniform" => Tone::Uniform,
            "turing" => {
                let seed = args.int_or("seed", 1)?;
                if seed < 0 {
                    return Err(CliError(format!("--seed wants a whole number, 0 or more, got {seed}")));
                }
                Tone::Turing { scale: args.positive("scale")?.unwrap_or(size / 4.0), seed: seed as u64 }
            }
            "half-lambert" => Tone::HalfLambert { light: Vec3::new(-0.4, -0.6, 0.7) },
            other => {
                return Err(CliError(format!("unknown tone '{other}'; the tones are uniform, turing, half-lambert")));
            }
        };
        let spacing = args.positive("spacing")?.unwrap_or(size / 25.0);
        let step = args.positive("step")?;
        if let Some(step) = step
            && step < Hatching::FINEST_STEP * spacing
        {
            return Err(CliError(format!(
                "--step wants at least a millionth of --spacing, got {}",
                kurven_core::json::number(step)
            )));
        }
        let width_steps = args.int_or("width-steps", 1)?;
        if !(1..=PolyhedronManifest::MAX_WIDTH_STEPS).contains(&width_steps) {
            return Err(CliError(format!(
                "--width-steps wants 1 to {}, got {width_steps}",
                PolyhedronManifest::MAX_WIDTH_STEPS
            )));
        }
        let hatching = Hatching::new(
            args.double("angle")?.unwrap_or(45.0) * std::f64::consts::PI / 180.0,
            spacing,
            step,
            tone,
            width_steps,
        );
        let mut spec = LayerSpec::new(
            "hatching",
            LayerRole::Scaffold,
            LayerSource::Hatching { hatching, faces: FaceSelector::Solid },
            args.double("hatch-width")?.unwrap_or(width / 2.0),
            HeightPolicy::Surface,
        );
        spec.color = args.flag("hatch-color").or(args.flag("color")).unwrap_or("#000000").to_string();
        specs.push(spec);
    }
    // Refused by the rule a polyhedron's manifest is read by: a width, an
    // angle, Turing's scale, and what the hatching would lay.
    if let Some(why) = PolyhedronManifest::unsupported(&specs, &solid) {
        return Err(CliError(why));
    }
    // `--save PATH`: the bundle that keeps this solid, its layers, and the
    // camera and resolution it is baked at.
    let url = PathBuf::from(args.flag("save").map_or_else(|| format!("/native/{kind}.kurven"), String::from));
    let preset = CameraPreset {
        name: SurfaceRequest::preset().name,
        projection,
        depth_tolerance: 0.0,
        buffer: args.resolution(3000)?,
    };
    let bundle = PolyhedronBundle::new(url, solid.clone(), specs, vec![preset], &kind, "native");
    if args.flag("save").is_some() {
        bundle.write(&bundle.url)?;
        println!("{kind} -> {}", file_name(&bundle.url));
        if args.flag("output").is_none() {
            return Ok(());
        }
    }
    let layers = bundle.layers.clone();
    let scene = Scene::solid_scene(solid.clone(), layers.clone(), camera, PreviewMode::Drawing, 0.0);
    let output = PathBuf::from(args.string("output")?);
    let options = BakeOptions {
        resolution: args.resolution(3000)?,
        tiles: args.int("tiles")?,
        depth_tolerance: None,
        outline: None,
    };
    let started = Instant::now();
    let result = bake(&scene, &options)?;
    let elapsed = started.elapsed();
    write_svg(&result.strokes, &output)?;
    let drawn = &result.strokes.layers[0].1;
    let corners: Vec<Vec3> = solid.vertices.iter().map(|&p| camera.view.apply(p).v).collect();
    let whole = drawn
        .paths()
        .filter(|p| {
            solid
                .edges
                .iter()
                .any(|e| (p[0].v - corners[e.a]).length() < 1e-9 && (p[p.len() - 1].v - corners[e.b]).length() < 1e-9)
        })
        .count();
    let diagonals = solid.is_diagonal.iter().filter(|d| **d).count();
    let hatch_length: f64 = result.strokes.layers[1..].iter().map(|(_, p)| p.total_length()).sum();
    let hatch_lines: usize = result.strokes.layers[1..].iter().map(|(_, p)| p.count()).sum();
    let widths = result.strokes.layers.len() - 1;
    println!(
        "{kind} -> {}\n  solid      {} vertices, {} edges, {} faces{}, {}, {}\n  edges      {} spans drawn, {whole} of them whole edges, length {:.2}\n  hatching   {}\n  depth      none: the port bakes strokes only\n  took       {:.3} s",
        file_name(&output),
        solid.vertices.len(),
        solid.edges.len() - diagonals,
        solid.faces.len(),
        if diagonals > 0 { format!(" in {} convex pieces", solid.pieces.len()) } else { String::new() },
        topology(&solid),
        if solid.is_convex() {
            "convex: the sign test alone decides"
        } else {
            "not convex: edges are tested for occlusion"
        },
        drawn.count(),
        drawn.total_length(),
        if layers.len() > 1 {
            format!(
                "{hatch_lines} lines at {widths} width{}, length {hatch_length:.2}",
                if widths == 1 { "" } else { "s" }
            )
        } else {
            "none".to_string()
        },
        elapsed.as_secs_f64()
    );
    Ok(())
}

fn file_name(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// `surface`: the shape named first after the command. `surface solid` is
/// the polyhedron's (`solidCommand`); the parametric shapes wait for their
/// phase.
pub fn surface_command(args: &Args) -> Result<(), CliError> {
    let shape = args.positional.get(1).map_or("torus", String::as_str);
    if shape == "solid" {
        return solid_command(args);
    }
    // PORT: phase 4 -- `surface torus | sn | periodic | forced`, the
    // parametric surfaces (`surfaceCommand`).
    Err(CliError(format!("surface {shape} is a parametric surface, which the port does not draw yet (phase 4)")))
}
