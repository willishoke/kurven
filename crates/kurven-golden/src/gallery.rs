//! The gallery's landscapes as `docs/gallery/make.py` sets them up, shared by
//! `kurven-cli references` (which bakes them through `landscape` and `bake`,
//! as Swift's `References.swift` does) and the wasm module's timing export
//! (which builds and bakes them in memory, with no file to write).

use kurven_bake::{Bake, BakeOptions, bake};
use kurven_core::json::{JsonObject, JsonValue};
use kurven_core::manifest::{Domain, Interval, Manifest, RealBand, Truncation};
use kurven_core::scene::Scene;
use kurven_landscape::catalog::LandscapeRequest;
use kurven_landscape::landscape::NativeLandscape;

/// The plates hatch the plateaus, walls and risers this much closer than a
/// landscape does by default (make.py's `HATCH`).
pub const HATCH: f64 = 0.75;

/// A camera's projection fields, as make.py writes them over a preset's.
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

/// cn's modulus in the gallery, and its quarter periods K and K' by the AGM.
pub const CN_MODULUS: f64 = 0.64;

pub fn cn_quarter_periods() -> (f64, f64) {
    let m = CN_MODULUS;
    let k = core::f64::consts::PI / (2.0 * agm(1.0, (1.0 - m).sqrt()));
    let kp = core::f64::consts::PI / (2.0 * agm(1.0, m.sqrt()));
    (k, kp)
}

/// A manifest edited as make.py edits it: one preset per camera, each the
/// first preset with its name and its projection's fields replaced, and the
/// plateaus', walls' and risers' hatching `HATCH` closer. `stem` names the
/// drawing in a refusal.
pub fn edit(m: &mut JsonObject, stem: &str, cameras: &[(String, JsonObject)]) -> Result<(), String> {
    let (base, projection) = match m.get("presets").and_then(JsonValue::as_array).and_then(|p| p.first()) {
        Some(JsonValue::Object(base)) => match base.get("projection") {
            Some(JsonValue::Object(projection)) => (base.clone(), projection.clone()),
            _ => return Err(format!("{stem}: the landscape wrote no preset to start from")),
        },
        _ => return Err(format!("{stem}: the landscape wrote no preset to start from")),
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
    let Some(JsonValue::Array(layers)) = m.get("layers") else { return Err(format!("{stem}: no layers")) };
    let layers = layers.iter().map(|layer| closer(layer).unwrap_or_else(|| layer.clone())).collect();
    m.insert("layers".to_string(), JsonValue::Array(layers));
    Ok(())
}

/// A hatching layer on the plateaus, walls or risers, `HATCH` closer; None
/// for any other layer, which stays as it is.
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
        JsonValue::Double(s) => s * HATCH,
        JsonValue::Int(s) => *s as f64 * HATCH,
        _ => return None,
    };
    source.insert("spacing".to_string(), JsonValue::Double(spacing));
    l.insert("source".to_string(), JsonValue::Object(source));
    Some(JsonValue::Object(l))
}

/// gallery-gamma or gallery-cn built, edited and baked in memory under its
/// plate preset: `landscape` and `bake` with no bundle written between
/// them. None for another name. For timing; the references are baked
/// through the files, as Swift bakes them.
pub fn bake_plate(name: &str) -> Option<Bake> {
    let (request, plate) = match name {
        "gallery-gamma" => {
            let band = |below, height| RealBand { below, height };
            let request = LandscapeRequest {
                truncation: Some(Truncation::RealBands {
                    bands: vec![band(-3.5, 2.5), band(-2.5, 3.0), band(-1.5, 4.0), band(0.5, 5.5)],
                    beyond: 4.0,
                }),
                ..LandscapeRequest::new(
                    "gamma(z)",
                    Domain::new(Interval::new(-4.5, 4.5), Interval::new(-2.5, 2.5)),
                    1000,
                )
            };
            (request, camera(-0.5, -55.0, -90.0, false))
        }
        "gallery-cn" => {
            let (k, kp) = cn_quarter_periods();
            let request = LandscapeRequest {
                truncation: Some(Truncation::Uniform(4.0)),
                ..LandscapeRequest::new(
                    &format!("cn(z, {})", kurven_core::json::number(CN_MODULUS)),
                    Domain::new(Interval::new(-k, 5.0 * k), Interval::new(0.0, 3.0 * kp)),
                    1200,
                )
            };
            (request, camera(0.51, -63.0, -90.0, true))
        }
        _ => return None,
    };
    let bundle = NativeLandscape::build(&request, "native", true).ok()?;
    let JsonValue::Object(mut m) = bundle.manifest.json() else { return None };
    edit(&mut m, name, &[("plate".to_string(), plate)]).ok()?;
    let manifest = Manifest::from_json(&JsonValue::Object(m)).ok()?;
    let bundle = bundle.restyled(&manifest);
    let preset = bundle.manifest.preset("plate").ok()?.clone();
    let scene = Scene::from_bundle(&bundle, &preset);
    let options = BakeOptions { depth_tolerance: Some(preset.depth_tolerance), ..BakeOptions::new(preset.buffer) };
    bake(&scene, &options).ok()
}
