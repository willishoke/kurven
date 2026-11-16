//! The commands of `Sources/kurven-cli/main.swift` a landscape reaches:
//! `landscape`, `bake`, and the scene they read (`loadScene`,
//! `loadDrawing`), with the helpers they call.

use std::path::{Path, PathBuf};
use std::time::Instant;

use kurven_bake::{BakeOptions, bake};
use kurven_core::bundle::KurvenBundle;
use kurven_core::hatching::{FaceSelector, Hatching, Tone};
use kurven_core::json;
use kurven_core::manifest::{
    CameraPreset, ContourField, DirectionField, Domain, DomainAxis, HeightPolicy, Interval, LayerRole, LayerSource,
    LayerSpec, Manifest, PolyhedronManifest, Provenance, RealBand, Restriction, Seeds, Spacing, Truncation,
};
use kurven_core::scene::Scene;
use kurven_core::simd::Vec3;
use kurven_core::svg::{self, Strokes};
use kurven_landscape::catalog::{Catalog, LandscapeRequest};
use kurven_landscape::landscape::NativeLandscape;

use crate::args::{self, Args, CliError};

/// Swift's `NativeLandscape.build`'s default `gitSha`.
pub const GIT_SHA: &str = "native";

/// `String.trimmingCharacters(in: .whitespaces)`: tabs and the space
/// separators, not line breaks.
fn trimmed(s: &str) -> &str {
    s.trim_matches(|c: char| {
        c == '\t'
            || (c.is_whitespace() && !matches!(c, '\n' | '\r' | '\u{b}' | '\u{c}' | '\u{85}' | '\u{2028}' | '\u{2029}'))
    })
}

/// Swift's `split(separator:)`, which leaves out empty pieces.
fn pieces(text: &str, separator: char) -> Vec<&str> {
    text.split(separator).filter(|s| !s.is_empty()).collect()
}

/// `LO,HI` as an interval; the one place a pair of numbers is spelled on the
/// command line, so a landscape's window reads the way it is written down.
pub fn interval(text: &str, what: &str) -> Result<Interval, CliError> {
    let parts: Vec<Option<f64>> = pieces(text, ',').into_iter().map(|p| args::double(trimmed(p))).collect();
    match parts[..] {
        [Some(lo), Some(hi)] if lo < hi => Ok(Interval::new(lo, hi)),
        _ => Err(CliError(format!("--{what} wants LO,HI with LO < HI, got '{text}'"))),
    }
}

/// `--bands BELOW:HEIGHT,... --beyond H` as a truncation: a height per band of Re z.
pub fn real_bands(text: &str, beyond: Option<&str>) -> Result<Truncation, CliError> {
    let mut bands = Vec::new();
    for part in pieces(text, ',') {
        let pair: Vec<Option<f64>> = pieces(part, ':').into_iter().map(|p| args::double(trimmed(p))).collect();
        match pair[..] {
            [Some(below), Some(height)] if height > 0.0 => bands.push(RealBand { below, height }),
            _ => return Err(CliError(format!("--bands wants BELOW:HEIGHT,... with each HEIGHT > 0, got '{part}'"))),
        }
    }
    if bands.is_empty() || !bands.windows(2).all(|w| w[0].below < w[1].below) {
        return Err(CliError(format!("--bands wants its bands in increasing BELOW, got '{text}'")));
    }
    match beyond.and_then(args::double) {
        Some(z) if z > 0.0 => Ok(Truncation::RealBands { bands, beyond: z }),
        _ => Err(CliError("--bands needs --beyond Z, the height past the last band".to_string())),
    }
}

/// The layers `--isophotes` and `--hatching` add to a curved face
/// (Swift's `shadingLayers(_:on:spacing:spacingIsOwn:)`): isophotes, the
/// level sets of n · l at the listed levels, and a hatching along u
/// (`--angle 0`) or v (90), every `--spacing` of the other coordinate, by
/// default `spacing`'s for it, in `--width-steps` widths, both under
/// `--light X,Y,Z`. A flag with no layer to apply it to is refused, as is
/// what a manifest refuses. `spacing_is_own` says the command reads
/// `--spacing` itself, as `landscape` does for its plateaus.
pub fn shading_layers(
    args: &Args,
    faces: &FaceSelector,
    spacing: (f64, f64),
    spacing_is_own: bool,
) -> Result<Vec<LayerSpec>, CliError> {
    let isophotes = args.flags.contains_key("isophotes");
    let hatching = args.flags.contains_key("hatching");
    let mut orphans: Vec<&str> = Vec::new();
    if !(isophotes || hatching) {
        orphans.push("light");
    }
    if !isophotes {
        orphans.push("isophote-width");
    }
    if !hatching {
        orphans.extend(["angle", "width-steps", "hatch-width"]);
        if !spacing_is_own {
            orphans.push("spacing");
        }
    }
    if let Some(orphan) = orphans.into_iter().find(|o| args.given(o)) {
        let wants = match orphan {
            "light" => "--isophotes or --hatching",
            "isophote-width" => "--isophotes",
            _ => "--hatching",
        };
        return Err(CliError(format!("--{orphan} applies to {wants}, and none is given")));
    }
    let mut light = Vec3::new(-0.4, -0.6, 0.7);
    if let Some(text) = args.flag("light") {
        let l: Vec<f64> = pieces(text, ',').into_iter().filter_map(args::double).collect();
        if l.len() != 3 || !l.iter().all(|x| x.is_finite()) || l == [0.0, 0.0, 0.0] {
            return Err(CliError("--light wants X,Y,Z, a direction toward the light, as in -0.4,-0.6,0.7".to_string()));
        }
        light = Vec3::new(l[0], l[1], l[2]);
    }
    let mut layers = Vec::new();
    if let Some(text) = args.flag("isophotes") {
        let listed = pieces(text, ',');
        let levels: Vec<f64> = listed.iter().filter_map(|s| args::double(s)).collect();
        if levels.is_empty() || levels.len() != listed.len() || !levels.iter().all(|x| x.is_finite()) {
            return Err(CliError("--isophotes wants levels of n · l, as in -0.5,0,0.5".to_string()));
        }
        if let Some(why) = faces.unsupported_field(ContourField::Isophote) {
            return Err(CliError(why));
        }
        layers.push(LayerSpec::new(
            "isophotes",
            LayerRole::Scaffold,
            LayerSource::Contour {
                field: ContourField::Isophote,
                levels,
                restriction: Restriction::All,
                faces: faces.clone(),
                light: Some(light),
            },
            args.positive("isophote-width")?.unwrap_or(0.3),
            HeightPolicy::Surface,
        ));
    }
    if let Some(kind) = args.flag("hatching") {
        let tone = match kind {
            "uniform" => Tone::Uniform,
            "half-lambert" => Tone::HalfLambert { light },
            other => {
                return Err(CliError(format!(
                    "unknown tone '{other}' on a curved face; the tones are uniform, half-lambert"
                )));
            }
        };
        let width_steps = args.int_or("width-steps", 1)?;
        if !(1..=PolyhedronManifest::MAX_WIDTH_STEPS).contains(&width_steps) {
            return Err(CliError(format!(
                "--width-steps wants 1 to {}, got {width_steps}",
                PolyhedronManifest::MAX_WIDTH_STEPS
            )));
        }
        let angle = args.double("angle")?.unwrap_or(0.0) * core::f64::consts::PI / 180.0;
        let across = if DomainAxis::from_angle(angle) == Some(DomainAxis::Imag) { spacing.0 } else { spacing.1 };
        let hatching = Hatching::new(angle, args.positive("spacing")?.unwrap_or(across), None, tone, width_steps);
        if let Some(why) = faces.unsupported_hatching(&hatching) {
            return Err(CliError(why));
        }
        layers.push(LayerSpec::new(
            "hatching",
            LayerRole::Scaffold,
            LayerSource::Hatching { hatching, faces: faces.clone() },
            args.positive("hatch-width")?.unwrap_or(0.4),
            HeightPolicy::Surface,
        ));
    }
    Ok(layers)
}

/// The layer `--slope-lines C` adds to the graph: the slope lines of |f|,
/// seeded on the contour |f| = C at the angles of arg f `--seeds` lists, in
/// radians, by default every 30°, or every `--seed-spacing` along it, drawn
/// `--slope-width` wide. None without `--slope-lines`.
pub fn slope_layer(args: &Args) -> Result<Option<LayerSpec>, CliError> {
    let Some(text) = args.flag("slope-lines") else {
        if let Some(orphan) = ["seeds", "seed-spacing", "slope-width"].into_iter().find(|o| args.given(o)) {
            return Err(CliError(format!("--{orphan} applies to --slope-lines, and none is given")));
        }
        return Ok(None);
    };
    let level = match args::double(text) {
        Some(level) if level.is_finite() && level > 0.0 => level,
        _ => return Err(CliError(format!("--slope-lines wants a level of |f|, a positive number, not '{text}'"))),
    };
    let spacing = match (args.flag("seeds"), args.flag("seed-spacing")) {
        (Some(_), Some(_)) => {
            return Err(CliError("--seeds and --seed-spacing place the seeds two ways; give one".to_string()));
        }
        (Some(listed), None) => {
            let words = pieces(listed, ',');
            let angles: Vec<f64> = words.iter().filter_map(|s| args::double(s)).collect();
            if angles.is_empty() || angles.len() != words.len() || !angles.iter().all(|x| x.is_finite()) {
                return Err(CliError("--seeds wants angles of arg f in radians, as in -1.5,0,1.5".to_string()));
            }
            Spacing::Values(ContourField::Phase, angles)
        }
        (None, Some(_)) => Spacing::ArcLength(args.positive("seed-spacing")?.unwrap_or(f64::NAN)),
        (None, None) => Spacing::Values(
            ContourField::Phase,
            (0..12).map(|k| -core::f64::consts::PI + f64::from(k) * core::f64::consts::PI / 6.0).collect(),
        ),
    };
    let field = DirectionField::Slope(ContourField::Magnitude);
    let seeds = Seeds { field: ContourField::Magnitude, level, spacing };
    let faces = FaceSelector::Graph { tiled: false };
    if let Some(why) = LayerSource::unsupported_integral_curves(&field, &seeds, &faces, None) {
        return Err(CliError(why));
    }
    Ok(Some(LayerSpec::new(
        "slope_lines",
        LayerRole::Scaffold,
        LayerSource::IntegralCurves { field, seeds, faces },
        args.positive("slope-width")?.unwrap_or(0.3),
        HeightPolicy::Surface,
    )))
}

/// `layers` with `added` drawn before the outlines that close the list, as
/// the window adds a layer.
pub fn adding(added: Vec<LayerSpec>, layers: &[LayerSpec]) -> Vec<LayerSpec> {
    let outline = layers.iter().rev().take_while(|l| l.role == LayerRole::Outline).count();
    let mut out = layers.to_vec();
    let at = layers.len() - outline;
    out.splice(at..at, added);
    out
}

/// `landscape`: sample a function over a rectangle and write its bundle.
/// Prints the summary Swift prints, but for the layer sources, which are
/// named and not described (crates/FINDINGS.md).
pub fn landscape_command(args: &Args) -> Result<(), CliError> {
    args.only(
        "landscape",
        &[
            "function",
            "re",
            "im",
            "res",
            "truncation",
            "bands",
            "beyond",
            "no-truncation",
            "spacing",
            "repeat",
            "output",
            "isophotes",
            "light",
            "isophote-width",
            "hatching",
            "angle",
            "width-steps",
            "hatch-width",
            "slope-lines",
            "seeds",
            "seed-spacing",
            "slope-width",
        ],
    )?;
    let output = PathBuf::from(args.string("output")?);
    let catalog = Catalog::native();

    // The preset is the starting point and every flag is an override of it, so
    // `landscape --function gamma --res 1200` means what it looks like.
    let name = args.flag("function").unwrap_or(if args.positional.len() > 1 { "" } else { "rgamma" });
    let mut request = if let Some(preset) = catalog.preset(name) {
        LandscapeRequest::from_preset(preset, catalog.default_resolution)
    } else if let Some(expression) = args.positional.get(1) {
        LandscapeRequest::new(
            expression,
            Domain::new(Interval::new(-4.0, 4.0), Interval::new(-2.5, 2.5)),
            catalog.default_resolution,
        )
    } else {
        let named = if name.is_empty() { String::new() } else { format!("'{name}' ") };
        return Err(CliError(format!("no function {named}given; try 'catalog', or pass an expression")));
    };
    if let Some(re) = args.flag("re") {
        request.domain.real = interval(re, "re")?;
    }
    if let Some(im) = args.flag("im") {
        request.domain.imag = interval(im, "im")?;
    }
    request.resolution = args.int_or("res", request.resolution)?;
    if let Some(z) = args.positive("truncation")? {
        request.truncation = Some(Truncation::Uniform(z));
    }
    if let Some(bands) = args.flag("bands") {
        request.truncation = Some(real_bands(bands, args.flag("beyond"))?);
    }
    if args.switched_on("no-truncation")? {
        request.truncation = Some(Truncation::None);
    }
    if let Some(v) = args.positive("spacing")? {
        request.spacing = Some(v);
    }
    let slope = slope_layer(args)?;
    // Every flag is read and refused before f is sampled: Swift reads
    // `--isophotes`, `--hatching` and the seed spacing after the build, so a
    // mistyped flag cost a whole build to report. Nothing they decide reads
    // the samples. The hatching's default spacing is the plateaus', which
    // the manifest's default layers carry, so the layers are made again with
    // it once the bundle is built; the seed spacing is held to the grid the
    // build samples, the shape `grid_shape` gives it.
    let window_spacing =
        kurven_math::swift::max(request.domain.real.length().abs(), request.domain.imag.length().abs()) / 70.0;
    shading_layers(args, &FaceSelector::Graph { tiled: false }, (window_spacing, window_spacing), true)?;
    if let Some(slope) = &slope
        && let Some(shape) = NativeLandscape::grid_shape(request.domain, request.resolution)
        && let Some(why) =
            Manifest::unsupported(&slope.source, &request.domain, (shape.n_imag as i64, shape.n_real as i64))
    {
        return Err(CliError(format!("--seed-spacing: {why}")));
    }
    let repeat = args.int_or("repeat", 1)?;

    // The bundle is written as its manifest and its two grids, which a
    // refiner changes nothing in (`KurvenBundle::write`), so the build
    // attaches none: Swift's builds one and places every layer through it,
    // to be thrown away. The bytes written are the same.
    //
    // --repeat N builds it N times over, which is how a profiler gets a long
    // enough look at one build.
    for _ in 1..repeat {
        NativeLandscape::build(&request, GIT_SHA, false)?;
    }
    let started = Instant::now();
    let mut bundle = NativeLandscape::build(&request, GIT_SHA, false)?;
    let sampled = started.elapsed();
    // `--isophotes`, `--hatching`: the graph lit, the hatching at the
    // plateaus' spacing unless `--spacing` says.
    let plateau_spacing = bundle.manifest.layers.iter().find_map(|spec| match &spec.source {
        LayerSource::Hatching { hatching, faces: FaceSelector::Plateaus { .. } } => Some(hatching.spacing),
        _ => None,
    });
    let spacing = plateau_spacing.unwrap_or(window_spacing);
    let shading = shading_layers(args, &FaceSelector::Graph { tiled: false }, (spacing, spacing), true)?;
    let mut added = shading;
    added.extend(slope);
    if !added.is_empty() {
        let mut m = bundle.manifest.clone();
        m.layers = adding(added, &m.layers);
        bundle = bundle.restyled(&m);
    }
    bundle.write(&output)?;
    let m = &bundle.manifest;
    println!(
        "{} -> {} (in {:.3} s)\n  {}x{} samples, truncation {:?}",
        m.provenance.function,
        file_name(&output),
        sampled.as_secs_f64(),
        m.height.shape.1,
        m.height.shape.0,
        m.truncation
    );
    for spec in &m.layers {
        println!("    {}", spec.name);
    }
    Ok(())
}

fn file_name(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// A landscape's bundle, its contours placed by its function unless
/// `--no-refine`, under `--preset` or its first, and its scene.
pub fn load_scene(args: &Args) -> Result<(KurvenBundle, CameraPreset, Scene), CliError> {
    let Some(path) = args.positional.get(1) else { return Err(CliError("which bundle?".to_string())) };
    let mut bundle = KurvenBundle::read(path)?;
    // The contours placed by the function, as the window draws them
    // (`Document.refineContours`), whenever the bundle names a function to
    // place them by. `--no-refine` draws the grid's contours instead, which
    // is the stage the fixtures pin.
    if !args.switched_on("no-refine")?
        && let Some(refiner) = NativeLandscape::refiner_for_bundle(&bundle)
    {
        bundle = bundle.refined(Some(refiner.refine_hook()));
    }
    let preset = if let Some(name) = args.flag("preset") {
        bundle.manifest.preset(name)?.clone()
    } else if let Some(first) = bundle.manifest.presets.first() {
        first.clone()
    } else {
        return Err(CliError("bundle has no presets; pass --preset once one exists".to_string()));
    };
    let scene = Scene::from_bundle(&bundle, &preset);
    Ok((bundle, preset, scene))
}

/// Whether a bundle's manifest holds `key` at its top level: how Swift tells
/// a surface's bundle (`surface`) and a polyhedron's (`polyhedron`) from a
/// landscape's (`SurfaceBundle.holdsSurface`, `PolyhedronBundle.holdsPolyhedron`).
fn holds(url: &Path, key: &str) -> Result<bool, CliError> {
    let file = url.join("manifest.json");
    if !file.exists() {
        return Ok(false);
    }
    let text = std::fs::read_to_string(&file).map_err(|e| CliError(format!("{}: {e}", file.display())))?;
    Ok(json::parse(&text)?.object("Manifest")?.contains_key(key))
}

/// The drawing a bundle describes: where it was read from, its
/// provenance, the preset and the scene (Swift's `loadDrawing`).
pub struct Drawing {
    pub url: PathBuf,
    pub provenance: Provenance,
    pub preset: CameraPreset,
    pub scene: Scene,
}

/// The scene a bundle of any kind describes, under `--preset` or its first
/// preset: a landscape's (`load_scene`) or a parametric surface's, built
/// from its request. A polyhedron's is refused until Phase 5.
pub fn load_drawing(args: &Args) -> Result<Drawing, CliError> {
    let Some(path) = args.positional.get(1) else { return Err(CliError("which bundle?".to_string())) };
    let url = Path::new(path);
    if holds(url, "polyhedron")? {
        // PORT: phase 5 -- `PolyhedronBundle.read`: the solid built from its
        // vertices and loops, its layers drawn on it, and `scene(preset:)`.
        return Err(CliError(format!(
            "{path} holds a polyhedron, which the port does not draw yet (phase 5: PolyhedronBundle)"
        )));
    }
    if holds(url, "surface")? {
        // A parametric surface's bundle, built from its request.
        let surface = kurven_surfaces::SurfaceBundle::read(url)?;
        let presets = &surface.manifest.presets;
        let preset = match args.flag("preset") {
            Some(name) => surface.manifest.preset(name)?.clone(),
            None => presets
                .first()
                .cloned()
                .ok_or_else(|| CliError("bundle has no presets; pass --preset once one exists".to_string()))?,
        };
        return Ok(Drawing {
            url: surface.url.clone(),
            provenance: surface.manifest.provenance.clone(),
            scene: surface.scene(&preset),
            preset,
        });
    }
    let (bundle, preset, scene) = load_scene(args)?;
    Ok(Drawing { url: bundle.url.clone(), provenance: bundle.manifest.provenance.clone(), preset, scene })
}

/// `bake`: a bundle's drawing to SVG, as Swift writes it.
pub fn bake_command(args: &Args) -> Result<(), CliError> {
    let mut accepted = Args::SCENE.to_vec();
    accepted.extend(["resolution", "tiles", "depth-tolerance", "outline", "dump", "output"]);
    // Refused whether given a value or not, before `only` would call either
    // a bare option wanting a value.
    if args.switches.contains("dump") {
        return Err(CliError("--dump wants a value".to_string()));
    }
    args.only("bake", &accepted)?;
    let drawing = load_drawing(args)?;
    let output = PathBuf::from(args.string("output")?);
    let options = BakeOptions {
        resolution: args.resolution(drawing.preset.buffer)?,
        tiles: args.int("tiles")?,
        depth_tolerance: Some(args.double("depth-tolerance")?.unwrap_or(drawing.preset.depth_tolerance)),
        outline: args.double("outline")?,
    };
    if !drawing.provenance.is_reproducible() {
        eprintln!(
            "note: this bundle was contoured with {} chunks, so its contours were stitched in thread-completion \
             order and are not reproducible.",
            drawing.provenance.cpu_count
        );
    }
    let started = Instant::now();
    let result = bake(&drawing.scene, &options)?;
    let elapsed = started.elapsed();
    write_svg(&result.strokes, &output)?;
    // Each stroke layer by the layer it draws: a scene's layer, one width
    // of a solid's hatching ("hatching 2/4"), or past them the outline,
    // traced from the depth buffer and carried by no layer.
    let names: Vec<String> = (0..result.strokes.layers.len())
        .map(|i| result.specs.get(i).map_or_else(|| "outline".to_string(), |s| s.name.clone()))
        .collect();
    if let Some(prefix) = args.flag("dump") {
        dump(prefix, &drawing.scene, &result, &names, &options)?;
    }
    println!(
        "{} -> {}  [{}]\n  depth      {}\n  strokes    {} paths, length {:.1}\n  took       {:.3} s",
        file_name(&drawing.url),
        file_name(&output),
        drawing.preset.name,
        match &result.surface {
            Some(image) => format!(
                "{}x{} in {} pass{}",
                image.frame().rows,
                image.frame().cols,
                result.tiles * result.tiles,
                if result.tiles == 1 { "" } else { "es" }
            ),
            None => "none: the port bakes a landscape's strokes only".to_string(),
        },
        result.strokes.path_count(),
        result.strokes.total_length(),
        elapsed.as_secs_f64()
    );
    for (index, (style, paths)) in result.strokes.layers.iter().enumerate() {
        println!(
            "    {:<14} {:<7} paths  lw {}{}",
            names[index],
            paths.count(),
            style.width,
            result.specs.get(index).map_or("  from the depth buffer", |s| if s.clipped { "" } else { "  unclipped" })
        );
    }
    Ok(())
}

/// `bake --dump PREFIX`: the strokes as arrays, so a comparison is on
/// geometry rather than on rasterized pixels -- each layer's vertices and
/// offsets as float32 npy, a width's slash no part of a file name -- and
/// the depth frame as JSON. The frame is the one Swift's bake covers the
/// drawing with, `DepthFrame::covering` its view bounds at the resolution,
/// which needs no depth pass: a parametric bake's own, a landscape's or a
/// solid's computed here.
fn dump(
    prefix: &str,
    scene: &Scene,
    result: &kurven_bake::Bake,
    names: &[String],
    options: &BakeOptions,
) -> Result<(), CliError> {
    for (index, (_, paths)) in result.strokes.layers.iter().enumerate() {
        let name = names[index].replace('/', "of");
        let flat: Vec<f32> = paths.vertices.iter().flat_map(|v| [v.x() as f32, v.y() as f32]).collect();
        kurven_core::npy::write_f32(&flat, &[paths.vertices.len(), 2], format!("{prefix}.{name}.npy"))
            .map_err(|e| CliError(e.to_string()))?;
        let offsets: Vec<f32> = paths.offsets.iter().map(|&o| o as f32).collect();
        kurven_core::npy::write_f32(&offsets, &[offsets.len()], format!("{prefix}.{name}.idx.npy"))
            .map_err(|e| CliError(e.to_string()))?;
    }
    let frame = match &result.surface {
        Some(image) => image.frame(),
        None => {
            let bounds =
                scene.view_bounds()?.ok_or_else(|| CliError(kurven_bake::BakeError::EmptyScene.to_string()))?;
            kurven_core::depth::DepthFrame::covering(&bounds, options.resolution)
        }
    };
    let meta = json::object([
        (
            "axis0",
            json::JsonValue::Array(vec![
                json::JsonValue::Double(frame.axis0.lo),
                json::JsonValue::Double(frame.axis0.hi),
            ]),
        ),
        (
            "axis1",
            json::JsonValue::Array(vec![
                json::JsonValue::Double(frame.axis1.lo),
                json::JsonValue::Double(frame.axis1.hi),
            ]),
        ),
        ("shape", json::JsonValue::Array(vec![json::JsonValue::Int(frame.rows), json::JsonValue::Int(frame.cols)])),
        ("depthTolerance", json::JsonValue::Double(options.depth_tolerance.unwrap_or(scene.depth_tolerance))),
        ("layers", json::JsonValue::Array(names.iter().map(|n| json::JsonValue::String(n.clone())).collect())),
    ]);
    let path = format!("{prefix}.frame.json");
    std::fs::write(&path, meta.canonical() + "\n").map_err(|e| CliError(format!("{path}: {e}")))
}

/// `SVG.render(strokes)` with its defaults, written as Swift's bake writes it.
pub fn write_svg(strokes: &Strokes, output: &Path) -> Result<(), CliError> {
    let text = svg::render(strokes, svg::FIGURE_WIDTH, svg::MARGIN, svg::DECIMALS);
    std::fs::write(output, text).map_err(|e| CliError(format!("{}: {e}", output.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_dump_or_outline_wants_a_value() {
        for (words, bare) in [
            (vec!["bake", "b.kurven", "--dump", "-o", "x.svg"], "dump"),
            (vec!["bake", "b.kurven", "-o", "x.svg", "--outline"], "outline"),
        ] {
            let e = bake_command(&Args::new(&words)).unwrap_err().0;
            assert_eq!(e, format!("--{bare} wants a value"), "{words:?}");
        }
    }

    #[test]
    fn intervals_and_bands_read_as_swift_reads_them() {
        assert_eq!(interval("-4.5, 4.5", "re").unwrap(), Interval::new(-4.5, 4.5));
        assert_eq!(interval("1,,2", "re").unwrap(), Interval::new(1.0, 2.0));
        assert_eq!(interval("2,1", "im").unwrap_err().0, "--im wants LO,HI with LO < HI, got '2,1'");
        let t = real_bands("-3.5:2.5,-2.5:3.0", Some("4.0")).unwrap();
        assert_eq!(
            t,
            Truncation::RealBands {
                bands: vec![RealBand { below: -3.5, height: 2.5 }, RealBand { below: -2.5, height: 3.0 }],
                beyond: 4.0
            }
        );
        assert!(real_bands("1:2,0:3", Some("4")).unwrap_err().0.contains("increasing BELOW"));
        assert!(real_bands("1:0", Some("4")).unwrap_err().0.contains("each HEIGHT > 0"));
        assert_eq!(real_bands("1:2", None).unwrap_err().0, "--bands needs --beyond Z, the height past the last band");
    }
}
