//! Someone else's description of a drawing, held within bounds before
//! anything is built from it: a `.kurven` bundle opened from a file (Phase
//! 8, step 7; Swift opens any bundle a user chooses) and a link
//! (`protocol::linkable`, step 3). One set of checks serves both, each with
//! its `Limits`.
//!
//! What a description can ask for and what bounds it:
//!
//! - a file's bytes, by the page's caps on the file and its entries (a grid
//!   or a dumped layer is as large as its file);
//! - a truncation: finite, at most `bands` bands;
//! - a parametric surface's request: Swift's own bounds (`SurfaceRequest
//!   ::check`, which reading a `SurfaceManifest` applies: the function's
//!   samples to 4096, the fit, the harmonics, curves of at most 2^24
//!   vertices, at most 2^25 samples), as Swift and kurven-cli write and
//!   read them, so every bundle our tools write opens;
//! - every layer: at most `layers`; a contour at most `levels` levels, every
//!   number of its restriction finite and at most `terms` terms; a hatching
//!   at most `strokes` lines across the drawing's extent (a landscape's
//!   window, a solid's box, a surface's rectangle); slope lines at most
//!   `seeds` seeds across it; coordinate and linear curves within Swift's
//!   vertex bound; every number finite. A solid's hatching reads its tone
//!   every `step` along each line, which `PolyhedronManifest` holds to
//!   2^20 readings in all (Swift's bound); no other face reads it.
//!
//! A link is held to what the controls reach (`LINK`: a strokes slider's
//! 400, its seeds' 400); a file to a bound on work ten times that (`FILE`),
//! since kurven-cli writes hatchings and seeds finer than a slider reaches.
//! Each refusal says which number and why, and touches nothing.

use kurven_core::bundle::{Source, holds_key_in};
use kurven_core::manifest::{
    LayerSource, LayerSpec, Manifest, PolyhedronManifest, Restriction, Spacing, SurfaceManifest, Truncation,
};
use kurven_core::surface_request::SurfaceRequest;

use crate::sliders::reach;

/// How much a description may ask for.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// Layers in all.
    pub layers: usize,
    /// Levels of one contour layer (the catalog's families list 40 at most).
    pub levels: usize,
    /// Terms of one contour's restriction, intersections unfolded.
    pub terms: usize,
    /// A hatching's lines across the drawing's extent.
    pub strokes: f64,
    /// Slope lines' seeds, listed or spaced across the extent.
    pub seeds: f64,
    /// A truncation's bands (the plates have four).
    pub bands: usize,
}

/// A link's: what the controls reach.
pub const LINK: Limits =
    Limits { layers: 64, levels: 1000, terms: 16, strokes: reach::STROKES.1 as f64, seeds: 400.0, bands: 64 };

/// A file's: a bound on work, ten times a link's strokes and five its
/// seeds, for what kurven-cli writes past the sliders (`--spacing`,
/// `--seeds`).
pub const FILE: Limits = Limits { strokes: 4000.0, seeds: 2000.0, ..LINK };

/// The most samples round a surface's ring a file may ask for: twice what
/// `kurven-cli surface sn --relief` builds (8192); Swift's `check` allows
/// 65,536, and bounds the samples in all by 2^25, which no ring past 8192
/// with its tube at half meets.
pub const MAX_GRID: i64 = 16384;

fn finite(x: f64) -> bool {
    x.is_finite()
}

/// Whether a light is a direction to normalize: not zero, not infinite,
/// its squared length a positive number (`Tone::has_direction`, which a
/// manifest's and a link's isophote are read by). A light without one has
/// no direction to read an isophote or a tone under, and its sliders'
/// angles would be NaN.
pub fn usable_light(l: kurven_core::simd::Vec3) -> bool {
    kurven_core::hatching::Tone::has_direction(l)
}

/// Whether the bundle staged as `source` may be opened: its manifest read
/// as its kind's schema, every number in it within `FILE` or Swift's own
/// bounds, and a camera preset to open at. Nothing is built.
pub fn openable(source: &Source) -> Result<(), String> {
    if !source.has("manifest.json") {
        return Err("a bundle holds a manifest.json, and this has none".to_string());
    }
    let json = source.manifest_json().map_err(|e| e.to_string())?;
    every_number_finite(&json, "the manifest")?;
    if holds_key_in(source, "polyhedron").map_err(|e| e.to_string())? {
        let m = PolyhedronManifest::from_json(&json).map_err(|e| e.to_string())?;
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for v in &m.solid.vertices {
            for (k, x) in [v.x(), v.y(), v.z()].into_iter().enumerate() {
                lo[k] = kurven_math::swift::min(lo[k], x);
                hi[k] = kurven_math::swift::max(hi[k], x);
            }
        }
        let extent = (0..3).map(|k| hi[k] - lo[k]).fold(0.0, kurven_math::swift::max);
        return layers(&m.layers, extent, &FILE);
    }
    if holds_key_in(source, "surface").map_err(|e| e.to_string())? {
        let m = SurfaceManifest::from_json(&json).map_err(|e| e.to_string())?;
        surface(&m.surface)?;
        // A surface's curves are spaced across its parameter rectangle,
        // whose sides a periodic function's periods give, which may be
        // negative: measured as lengths.
        let (u, v) = kurven_surfaces::drawing::SurfaceRequestExt::rectangle(&m.surface);
        if !(finite(u) && finite(v)) {
            return Err("the surface's parameter rectangle is not finite".to_string());
        }
        return layers(&m.layers, kurven_math::swift::max(u.abs(), v.abs()), &FILE);
    }
    let m = Manifest::from_json(&json).map_err(|e| e.to_string())?;
    // Opened, a landscape is fitted to its first preset; one without is a
    // failure the opening would only find after it had begun.
    if m.presets.is_empty() {
        return Err("the landscape declares no camera presets: it has no view to open at".to_string());
    }
    truncation(&m.truncation, &FILE)?;
    let d = m.domain;
    window(d.real.lo, d.real.hi, "Re")?;
    window(d.imag.lo, d.imag.hi, "Im")?;
    layers(&m.layers, kurven_math::swift::max(d.real.length(), d.imag.length()), &FILE)
}

/// Every number in `json` finite. The reader takes a number past a double's
/// range (`1e400`) as infinite, which no field of a manifest or a link is
/// meant to hold: a camera's angle, a tile's matrix, a wall's corner, a
/// depth tolerance written back into the state as infinite or NaN is a
/// value the canonical writer refuses -- a trap in the wasm module. So a
/// description holding one is refused whole, before any field is read.
pub fn every_number_finite(json: &kurven_core::json::JsonValue, what: &str) -> Result<(), String> {
    use kurven_core::json::JsonValue;
    fn walk(v: &JsonValue) -> bool {
        match v {
            JsonValue::Double(x) => x.is_finite(),
            JsonValue::Array(a) => a.iter().all(walk),
            JsonValue::Object(o) => o.values().all(walk),
            _ => true,
        }
    }
    if walk(json) { Ok(()) } else { Err(format!("{what} holds a number past a double's range")) }
}

/// One side of a window: an interval, lo below hi, finite each way. One
/// read the other way round would measure every spacing against a negative
/// extent.
pub fn window(lo: f64, hi: f64, axis: &str) -> Result<(), String> {
    if finite(lo) && finite(hi) && lo < hi && finite(hi - lo) {
        Ok(())
    } else {
        Err(format!("the window has {axis} from {lo:?} to {hi:?}, which is no interval"))
    }
}

/// A truncation: finite, at most `limits.bands` bands.
pub fn truncation(t: &Truncation, limits: &Limits) -> Result<(), String> {
    let ok = match t {
        Truncation::None => true,
        Truncation::Uniform(h) => finite(*h),
        Truncation::RealBands { bands, beyond } => {
            bands.len() <= limits.bands && bands.iter().all(|b| finite(b.below) && finite(b.height)) && !beyond.is_nan()
        }
    };
    if ok { Ok(()) } else { Err(format!("the truncation is not finite, or has more than {} bands", limits.bands)) }
}

/// A surface's request, past Swift's `check`: its ring within `MAX_GRID`.
fn surface(r: &SurfaceRequest) -> Result<(), String> {
    if !(8..=MAX_GRID).contains(&r.grid) {
        return Err(format!(
            "the surface's grid is {} samples round its ring; a file may ask for 8 to {MAX_GRID}",
            r.grid
        ));
    }
    Ok(())
}

/// A contour's restriction: every bound finite, and no more than
/// `limits.terms` terms, intersections unfolded.
fn restriction(r: &Restriction, limits: &Limits) -> Result<(), String> {
    fn walk(r: &Restriction, terms: &mut usize) -> bool {
        *terms += 1;
        match r {
            Restriction::All | Restriction::Footprint | Restriction::BelowTruncation => true,
            Restriction::Band { lo, hi, .. } => finite(*lo) && finite(*hi),
            Restriction::Intersection(all) => all.iter().all(|r| walk(r, terms)),
        }
    }
    let mut terms = 0;
    if !walk(r, &mut terms) {
        return Err("a band of its restriction is not finite".to_string());
    }
    if terms > limits.terms {
        return Err(format!("its restriction has {terms} terms, more than {}", limits.terms));
    }
    Ok(())
}

/// Every layer within `limits`, against `extent`, the longest side of what
/// its spacings are measured across.
pub fn layers(specs: &[LayerSpec], extent: f64, limits: &Limits) -> Result<(), String> {
    if !(finite(extent) && extent >= 0.0) {
        return Err(format!("the drawing's extent is {extent}, which is no length"));
    }
    if specs.len() > limits.layers {
        return Err(format!("it carries {} layers, more than {}", specs.len(), limits.layers));
    }
    let vertices = SurfaceRequest::MAX_VERTICES;
    for layer in specs {
        let name = &layer.name;
        let fine = match &layer.source {
            LayerSource::File { .. } | LayerSource::TruncationEdge { .. } | LayerSource::FoldCurves => true,
            LayerSource::Trajectory | LayerSource::Edges => true,
            LayerSource::Contour { levels, light, restriction: r, .. } => {
                if levels.len() > limits.levels {
                    return Err(format!("{name} lists {} levels, more than {}", levels.len(), limits.levels));
                }
                restriction(r, limits).map_err(|why| format!("{name}: {why}"))?;
                levels.iter().all(|&x| finite(x)) && light.is_none_or(usable_light)
            }
            LayerSource::Hatching { hatching, .. } => {
                let s = hatching.spacing;
                if !(finite(s) && s > 0.0 && (extent / s).round() <= limits.strokes) {
                    return Err(format!(
                        "{name} is hatched at a spacing of {s}: more than {} strokes across the drawing",
                        limits.strokes
                    ));
                }
                finite(hatching.angle) && finite(hatching.step)
            }
            LayerSource::WallEdges { edges, base } => {
                let mut seen = std::collections::BTreeSet::new();
                finite(*base) && edges.len() <= 4 && edges.iter().all(|&e| (0..4).contains(&e) && seen.insert(e))
            }
            // Swift's bound on a surface's curves (`check`): a count of
            // curves, or of a linear curve's turns and strands, past 2^24
            // vertices of a layer is past what one holds.
            LayerSource::CoordinateCurves { u, v } => {
                if !((1..=vertices).contains(u) && (1..=vertices).contains(v)) {
                    return Err(format!("{name} draws {u} by {v} coordinate curves, past {vertices}"));
                }
                true
            }
            LayerSource::LinearCurves { slope, turns, count } => {
                if !((1..=vertices).contains(turns)
                    && (1..=vertices).contains(count)
                    && (*turns as f64) * (*count as f64) <= vertices as f64)
                {
                    return Err(format!("{name} draws {count} strands of {turns} turns, past {vertices} vertices"));
                }
                finite(*slope)
            }
            LayerSource::IntegralCurves { seeds, .. } => {
                let many = match &seeds.spacing {
                    Spacing::Values(_, values) => {
                        values.len() as f64 > limits.seeds || !values.iter().all(|&x| finite(x))
                    }
                    Spacing::ArcLength(s) => !(finite(*s) && *s > 0.0 && extent / s <= limits.seeds),
                };
                if many {
                    return Err(format!("{name} is seeded more finely than {} seeds across the drawing", limits.seeds));
                }
                finite(seeds.level)
            }
        };
        if !fine {
            return Err(format!("{name} holds a number that is not finite, or an edge no wall has"));
        }
    }
    Ok(())
}
