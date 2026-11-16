//! `surface`: a parametric surface's drawing, as Swift's `surfaceCommand`
//! (`Sources/kurven-cli/main.swift`) builds it, saves it and bakes it; and the
//! layers it adds with `--lines-of-curvature` (`curvatureLayers`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use kurven_bake::{BakeOptions, bake};
use kurven_core::camera::Camera;
use kurven_core::hatching::FaceSelector;
use kurven_core::integral_curves::Limits;
use kurven_core::manifest::{CameraPreset, LayerRole, LayerSpec, ManifestError, PrincipalCurvature, SurfaceManifest};
use kurven_core::parametric::{ParamAxis, ParametricSurface};
use kurven_core::surface_request::{LinearCurves, Shape, Style, SurfaceRequest};
use kurven_math::{ContinuedFraction, format};
use kurven_surfaces::{DrawingError, SurfaceBundle, SurfaceDrawing, SurfacePreset};

use crate::args::{self, Args, CliError};
use crate::commands::{GIT_SHA, shading_layers, write_svg};

impl From<DrawingError> for CliError {
    fn from(e: DrawingError) -> CliError {
        // Swift's `catch ManifestError.unsupported(let why)` writes the
        // reason alone; every other error its description.
        match e {
            DrawingError::Manifest(ManifestError::Unsupported(why)) => CliError(why),
            other => CliError(other.to_string()),
        }
    }
}

impl From<kurven_surfaces::SurfaceBundleError> for CliError {
    fn from(e: kurven_surfaces::SurfaceBundleError) -> CliError {
        CliError(e.to_string())
    }
}

/// The options an oblique camera is set by (Swift's `Args.projection`).
pub const PROJECTION: [&str; 3] = ["shear", "x-angle", "z-angle"];
/// The options `previewOptions` reads.
pub const PREVIEW_OPTIONS: [&str; 3] = ["draft", "slope", "stroke-width"];
/// The flags that add isophotes and hatching along a coordinate to a curved face.
pub const SHADING: [&str; 8] =
    ["isophotes", "light", "isophote-width", "hatching", "angle", "spacing", "width-steps", "hatch-width"];
/// The flags that add lines of curvature to a surface.
pub const CURVATURE: [&str; 3] = ["lines-of-curvature", "curvature-seeds", "curvature-width"];

/// Swift's `split(separator:)`, which leaves out empty pieces.
fn pieces(text: &str, separator: char) -> Vec<&str> {
    text.split(separator).filter(|s| !s.is_empty()).collect()
}

/// `layers` with `added` drawn before the outlines that close the list, as
/// the window adds a layer.
pub fn adding(added: &[LayerSpec], layers: &[LayerSpec]) -> Vec<LayerSpec> {
    let outline = layers.iter().rev().take_while(|l| l.role == LayerRole::Outline).count();
    let mut out = layers.to_vec();
    let at = layers.len() - outline;
    out.splice(at..at, added.iter().cloned());
    out
}

/// The layers `--lines-of-curvature greater|lesser|both` adds to a surface
/// over `rectangle`: one family of lines of curvature, or both, each through
/// `--curvature-seeds` N seeds (24 by default) round a coordinate curve
/// (`LayerSpec::lines_of_curvature`), drawn `--curvature-width` wide, by
/// default `width`, the coordinate curves'. A flag with no layer to apply it
/// to is refused, and so is what the surface `request` builds cannot draw,
/// sn in relief, as its manifest refuses it.
pub fn curvature_layers(
    args: &Args,
    rectangle: (ParamAxis, ParamAxis),
    width: f64,
    request: &SurfaceRequest,
) -> Result<Vec<LayerSpec>, CliError> {
    let Some(text) = args.flag("lines-of-curvature") else {
        if let Some(orphan) = ["curvature-seeds", "curvature-width"].into_iter().find(|o| args.given(o)) {
            return Err(CliError(format!("--{orphan} applies to --lines-of-curvature, and none is given")));
        }
        return Ok(Vec::new());
    };
    let families = match text {
        "greater" => vec![PrincipalCurvature::Greater],
        "lesser" => vec![PrincipalCurvature::Lesser],
        "both" => vec![PrincipalCurvature::Greater, PrincipalCurvature::Lesser],
        _ => return Err(CliError(format!("--lines-of-curvature wants greater, lesser or both, not '{text}'"))),
    };
    let count = args.int_or("curvature-seeds", 24)?;
    let most = ParametricSurface::max_integral_curves(&Limits::default());
    if !(1..=most).contains(&count) {
        return Err(CliError(format!("--curvature-seeds wants 1 to {most} seeds, got {count}")));
    }
    let w = args.positive("curvature-width")?.unwrap_or(width);
    let layers: Vec<LayerSpec> =
        families.into_iter().map(|c| LayerSpec::lines_of_curvature(c, rectangle, count, w)).collect();
    if let Some(why) = layers.iter().find_map(|l| SurfaceManifest::unsupported(&l.source, request)) {
        return Err(CliError(why));
    }
    Ok(layers)
}

/// `surface [torus | sn | periodic | forced] [options]`: the drawing a
/// request builds, saved as a bundle with `--save`, baked to `-o` otherwise.
/// `surface solid` is a polyhedron's (Phase 5), and a preview (`--preview`,
/// a `.png` output) is the GPU's (Phase 7); both are refused.
pub fn surface_command(args: &Args) -> Result<(), CliError> {
    let shape = args.positional.get(1).map(String::as_str).unwrap_or("torus");
    if shape == "solid" {
        // PORT: phase 5 -- `solidCommand`: a named solid, its edges and hatching, saved or baked.
        return Err(CliError(
            "surface solid draws a polyhedron, which the port does not draw yet (phase 5)".to_string(),
        ));
    }
    let relief = shape == "sn" && args.flags.contains_key("relief");
    if args.flags.contains_key("relief") && ["torus", "periodic", "forced"].contains(&shape) {
        return Err(CliError(format!("--relief puts sn in relief; surface {shape} has none")));
    }
    // The switches, read before anything is built or written, so one given
    // a value is refused up front (`Args::switched_on`). `--conformal`: plain
    // sn in isothermal coordinates, on the torus its modulus pins
    // (`Shape::conformal_minor`).
    let conformal = args.switched_on("conformal")?;
    let soft = args.switched_on("soft")?;
    let preview = args.switched_on("preview")?;
    let _ = args.switched_on("draft")?;
    if conformal && (shape != "sn" || relief) {
        return Err(CliError(format!(
            "--conformal draws plain sn in isothermal coordinates; {}",
            if relief { "sn in relief has none".to_string() } else { format!("surface {shape} has none") }
        )));
    }
    if conformal && args.flags.contains_key("minor") {
        return Err(CliError(
            "--minor is pinned by the modulus under --conformal; give --modulus and --major".to_string(),
        ));
    }
    if relief {
        let mut accepted: Vec<&str> = PROJECTION.iter().chain(&SHADING).chain(&CURVATURE).copied().collect();
        accepted.extend([
            "relief",
            "modulus",
            "major",
            "minor",
            "repeat",
            "samples",
            "grid",
            "turn",
            "soft",
            "floor",
            "resolution",
            "tiles",
            "output",
            "save",
        ]);
        args.only("surface sn --relief", &accepted)?;
    } else {
        let own: Option<&[&str]> = match shape {
            "torus" => Some(&["coordinate-curves"]),
            "sn" => Some(&["modulus", "grid", "conformal"]),
            "periodic" => Some(&["expression", "periods", "grid"]),
            "forced" => {
                Some(&["system", "fit", "harmonics", "radial", "axial", "radius", "turns", "coordinate-curves"])
            }
            _ => None,
        };
        if let Some(own) = own {
            let mut accepted: Vec<&str> = own.to_vec();
            accepted.extend(PROJECTION.iter().chain(&PREVIEW_OPTIONS).chain(&SHADING).chain(&CURVATURE));
            accepted.extend([
                "major",
                "minor",
                "width",
                "samples",
                "linear-curves",
                "trajectory-width",
                "folds",
                "resolution",
                "tiles",
                "output",
                "preview",
                "width-px",
                "height-px",
                "mode",
                "save",
            ]);
            args.only(&format!("surface {shape}"), &accepted)?;
        }
    }
    let big_r = args.double("major")?.unwrap_or(2.0);
    let r = args.double("minor")?.unwrap_or(1.0);
    let width = args.double("width")?.unwrap_or(0.3);
    let counts = |text: &str| -> Result<(i64, i64), CliError> {
        let n: Vec<i64> = pieces(text, ',').into_iter().filter_map(args::int).collect();
        if n.len() != 2 {
            return Err(CliError("--coordinate-curves wants U,V, as in 36,18".to_string()));
        }
        Ok((n[0], n[1]))
    };
    // The flags, parsed into the request the window keeps; the drawing is
    // built from it by the same function either way.
    let mut request = match shape {
        "torus" => SurfaceRequest {
            lines: Some(counts(args.flag("coordinate-curves").unwrap_or("36,18"))?),
            ..SurfaceRequest::new(Shape::Torus { major: big_r, minor: r })
        },
        "sn" if relief => {
            // `surface sn --relief A`: the sn landscape in relief on its torus,
            // the surface displaced along its normal by A times |f| truncated at
            // the truncation. `--repeat N[,M]`: N periods round the ring and M
            // round the tube, the curves computed once and carried round by the
            // periods.
            let repeats: Vec<i64> =
                pieces(args.flag("repeat").unwrap_or("1"), ',').into_iter().filter_map(args::int).collect();
            if !(1..=2).contains(&repeats.len()) || !repeats.iter().all(|&k| k >= 1) {
                return Err(CliError("--repeat wants N or N,M whole periods, as in 8 or 8,2".to_string()));
            }
            SurfaceRequest::new(Shape::Relief {
                modulus: args.double("modulus")?.unwrap_or(0.64),
                major: big_r,
                minor: r,
                amplitude: args.double("relief")?.unwrap_or(0.0),
                repeats: (repeats[0], if repeats.len() > 1 { repeats[1] } else { 1 }),
                grid: args.int_or("grid", 600)?,
                turn: args.double("turn")?.unwrap_or(0.0),
                floor: args.double("floor")?,
                soft,
            })
        }
        "sn" => {
            let m = args.double("modulus")?.unwrap_or(0.64);
            SurfaceRequest::new(Shape::Sn {
                modulus: m,
                major: big_r,
                minor: if conformal { Shape::conformal_minor(m, big_r) } else { r },
                grid: args.int_or("grid", 600)?,
                conformal,
            })
        }
        "periodic" => {
            let periods: Vec<f64> =
                pieces(args.flag("periods").unwrap_or(""), ',').into_iter().filter_map(args::double).collect();
            if periods.len() != 2 {
                return Err(CliError("--periods wants the real and imaginary periods, as in 6.4,4.1".to_string()));
            }
            SurfaceRequest::new(Shape::Periodic {
                expression: args.string("expression")?.to_string(),
                periods: (periods[0], periods[1]),
                major: big_r,
                minor: r,
                grid: args.int_or("grid", 600)?,
            })
        }
        "forced" => {
            // A forced system's invariant torus, found in the spectrum of one
            // long run and fitted; drawn by revolution, with the trajectory on it.
            SurfaceRequest {
                lines: args.flag("coordinate-curves").map(counts).transpose()?,
                ..SurfaceRequest::new(Shape::Forced {
                    system: args.flag("system").unwrap_or("jerk").to_string(),
                    fit: args.double("fit")?.unwrap_or(8000.0),
                    harmonics: args.int_or("harmonics", 24)?,
                    radial: args.int_or("radial", 0)?,
                    axial: args.int_or("axial", 1)?,
                    radius: args.double("radius")?.unwrap_or(2.5),
                    turns: args.int_or("turns", 344)?,
                })
            }
        }
        _ => {
            return Err(CliError(format!(
                "unknown surface '{shape}'; the catalog has: torus, sn, periodic, forced, solid"
            )));
        }
    };
    // `--samples`: the grid round the ring; in relief, round the ring in
    // all, 1024 a period at most.
    request.grid = args.int_or("samples", if relief { 8192 } else { 1024 })?;
    // A linear curve: `--linear-curves 2/5`, `--linear-curves 0.618,24` for that many turns,
    // `--linear-curves 1/3,3,4` for four strands of it.
    if let Some(text) = args.flag("linear-curves") {
        let parts = pieces(text, ',');
        let slope = |s: &str| -> Option<f64> {
            let f: Vec<f64> = pieces(s, '/').into_iter().filter_map(args::double).collect();
            if f.len() == 2 && f[1] != 0.0 {
                Some(f[0] / f[1])
            } else if f.len() == 1 {
                Some(f[0])
            } else {
                None
            }
        };
        let wants = || CliError("--linear-curves wants SLOPE[,TURNS[,COUNT]], as in 2/5 or 0.618,24,3".to_string());
        if !(1..=3).contains(&parts.len()) {
            return Err(wants());
        }
        let slope = slope(parts[0]).ok_or_else(wants)?;
        let turns = if parts.len() > 1 { args::int(parts[1]).ok_or_else(wants)? } else { 24 };
        let count = if parts.len() > 2 { args::int(parts[2]).ok_or_else(wants)? } else { 1 };
        if !(turns >= 1 && count >= 1) {
            return Err(wants());
        }
        request.linear_curves = Some(LinearCurves { slope, turns, count });
    }
    // The outline and inner fold curves, derived for the camera; --folds 0
    // leaves them out.
    request.style = Style {
        lines: width,
        trajectory: args.double("trajectory-width")?.unwrap_or(0.08),
        folds: args.double("folds")?.unwrap_or(2.0 * width),
        linear_curves: width,
    };
    // A catalog entry by its own name, as the window opens it, so the two
    // keep the same request.
    if !relief && SurfacePreset::named(shape).is_some() {
        request.name = shape.to_string();
    }
    // Refused by the schema's rule, naming the flag that set the number.
    let flag_of = |key: &str| -> String {
        match key {
            "grid" => "--samples".to_string(),
            "shape.grid" => "--grid".to_string(),
            "shape.amplitude" => "--relief".to_string(),
            "shape.repeats" => "--repeat".to_string(),
            "lines" => "--coordinate-curves".to_string(),
            "linearCurves" => "--linear-curves".to_string(),
            "style.lines" | "style.linearCurves" => "--width".to_string(),
            "style.folds" => "--folds".to_string(),
            "style.trajectory" => "--trajectory-width".to_string(),
            other => format!("--{}", other.replace("shape.", "")),
        }
    };
    match request.check(&flag_of) {
        Ok(()) => {}
        Err(ManifestError::Invalid(why)) => return Err(CliError(why)),
        Err(other) => return Err(other.into()),
    }
    // `--lines-of-curvature`: refused before anything is built.
    curvature_layers(args, (ParamAxis::angle(2), ParamAxis::angle(2)), width, &request)?;
    let started = Instant::now();
    let mut drawing = SurfaceDrawing::build(&request, None, false)?;
    let built = started.elapsed().as_secs_f64();
    // `--isophotes`, `--hatching`: the surface's one face lit, its spacing
    // a 48th of the tube's turn by default.
    let (u, v) = (drawing.surface.u, drawing.surface.v);
    let mut shading = shading_layers(
        args,
        &FaceSelector::Surface,
        (u.range.length().abs() / 48.0, v.range.length().abs() / 48.0),
        false,
    )?;
    shading.extend(curvature_layers(args, (u, v), width, &request)?);
    if !shading.is_empty() {
        let specs: Vec<LayerSpec> = drawing.layers.iter().map(|l| l.spec.clone()).collect();
        drawing = drawing.restyled(&adding(&shading, &specs))?;
    }
    if let Shape::Sn { modulus, major, minor, conformal: true, .. } = &request.shape {
        println!(
            "sn(z, {}) conformal: the modulus pins the minor radius at {} round a ring of {} (r/R = {})",
            format::description(*modulus),
            kurven_surfaces::text::g(*minor, 12),
            format::description(*major),
            kurven_surfaces::text::g(minor / major, 12)
        );
    }
    if let Some(torus) = drawing.torus() {
        let rho = torus.rotation_number();
        let convergents: Vec<String> = ContinuedFraction::convergents(rho, 2000, 1e-12)
            .into_iter()
            .skip(1)
            .map(|c| format!("{}/{}", c.p, c.q))
            .collect();
        println!(
            "{}: forced at ω = {:.6}, own frequency Ω = {:.10}  (built in {built:.3} s)\n  fit       {} harmonics, \
             misses its trajectory by {:.2e} ({:.1e} of its size)\n  winds     at Ω/ω = {rho:.9} = {}\n            \
             nearly {}: drawn for those turns, it cuts the torus evenly\n  drawn     by revolution, {}",
            torus.system.name,
            torus.forcing(),
            torus.internal_frequency(),
            torus.fit.harmonics,
            torus.residual,
            torus.residual / torus.extent,
            ContinuedFraction::describe(&ContinuedFraction::quotients(rho, 8, 1e-12), true),
            convergents.join(", "),
            if drawing.embedded {
                "embedded: back faces are hidden"
            } else {
                "NOT embedded -- a slice crosses itself; back faces are kept"
            }
        );
    }
    let mut projection = SurfaceRequest::PROJECTION;
    projection.shear = args.double("shear")?.unwrap_or(projection.shear);
    projection.x_angle = args.double("x-angle")?.unwrap_or(projection.x_angle);
    projection.z_angle = args.double("z-angle")?.unwrap_or(projection.z_angle);
    let camera = Camera::oblique(&projection);
    // `--save PATH`: the bundle that keeps this drawing, as the window saves
    // one -- its request, its layers, and the camera and resolution it is
    // baked at -- so `bake PATH` draws what this does.
    if let Some(save) = args.flag("save") {
        let url = PathBuf::from(save);
        let base = SurfaceRequest::preset();
        let preset = CameraPreset {
            name: base.name.clone(),
            projection,
            depth_tolerance: base.depth_tolerance,
            buffer: args.resolution(base.buffer)?,
        };
        let bundle = SurfaceBundle::new(&url, drawing.clone(), vec![preset], GIT_SHA);
        bundle.write(&url, None)?;
        println!("{shape} -> {}  (built in {built:.3} s)", file_name(&url));
        if !args.flags.contains_key("output") {
            return Ok(());
        }
    }
    let output = PathBuf::from(args.string("output")?);
    if preview || output.extension().is_some_and(|e| e.eq_ignore_ascii_case("png")) {
        // PORT: phase 7 -- the realtime preview, one frame offscreen (`renderPreview`, PNG).
        return Err(CliError(
            "the preview is drawn by the GPU, which the port does not have yet (phase 7)".to_string(),
        ));
    }
    let options = BakeOptions {
        resolution: args.resolution(3000)?,
        tiles: args.int("tiles")?,
        depth_tolerance: None,
        outline: None,
    };
    let scene = drawing.scene(camera);
    let started = Instant::now();
    let result = bake(&scene, &options)?;
    let elapsed = started.elapsed().as_secs_f64();
    write_svg(&result.strokes, &output)?;
    let surface = &drawing.surface;
    let minor = request.get(kurven_core::surface_request::Field::Minor).unwrap_or(r);
    println!(
        "{shape}{} -> {}\n  grid    {}x{}{}\n  strokes    {} paths, length {:.1}\n  took       {elapsed:.3} s",
        if shape == "forced" { String::new() } else { format!(" R={} r={}", fmt_g(big_r), fmt_g(minor)) },
        file_name(&output),
        surface.u.samples,
        surface.v.samples,
        if surface.outward.is_none() { ", bounds no solid, back faces kept" } else { ", closed" },
        result.strokes.path_count(),
        result.strokes.total_length()
    );
    Ok(())
}

/// Swift's `fmt`, `%g`.
fn fmt_g(x: f64) -> String {
    kurven_surfaces::text::g(x, 6)
}

fn file_name(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}
