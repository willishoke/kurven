//! A `.kurven` bundle opened from a file: someone else's bytes, held to what
//! the controls reach before anything is built from them, as a link is
//! (`protocol::linkable`). The page's addition (Phase 8, step 7); Swift
//! opens any bundle a user chooses.
//!
//! What a bundle can ask for and what bounds it:
//!
//! - its bytes, by the page's caps on the file and its entries (a grid or a
//!   dumped layer is as large as its file);
//! - a landscape's truncation: finite, at most `LINK_BANDS` bands;
//! - a parametric surface's request: Swift's `check` (which `SurfaceManifest`
//!   applies) and, past it, the controls' reach -- the function's samples
//!   (120 to 1200), the fit's length and harmonics, the trajectory's turns,
//!   the coordinate curves' counts, the linear curve's turns and strands --
//!   and at most `MAX_GRID` samples round the ring, a bound on work;
//! - every layer: at most `LINK_LAYERS`; a contour at most `LINK_LEVELS`
//!   levels; a hatching at most a strokes slider's 400 lines across the
//!   bundle's extent (a landscape's window, a solid's box); slope lines at
//!   most `LINK_SEEDS` seeds; coordinate and linear curves within their
//!   controls; every number finite.
//!
//! Each refusal says which number and why, and touches nothing.

use kurven_core::bundle::{Source, holds_key_in};
use kurven_core::manifest::{
    LayerSource, LayerSpec, Manifest, PolyhedronManifest, Spacing, SurfaceManifest, Truncation,
};
use kurven_core::surface_request::{Field, Shape, SurfaceRequest};

use crate::sliders::reach;
use crate::surface_controls::{LINE_COUNTS, control};

/// The most layers a bundle opened from a file may carry.
pub const LINK_LAYERS: usize = 64;
/// The most levels one contour layer may list (a bound on work).
pub const LINK_LEVELS: usize = 1000;
/// The most seeds a slope-line layer may list, or be spaced to across the
/// bundle's extent.
pub const LINK_SEEDS: f64 = 400.0;
/// The most bands a truncation may have.
pub const LINK_BANDS: usize = 64;
/// The most samples round a surface's ring a file may ask for: twice what
/// `kurven-cli surface sn --relief` builds (8192), a bound on work (Swift's
/// `check` allows 65,536).
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

/// The coordinate curves' counts within their control (1...200 each).
pub fn line_counts(u: i64, v: i64) -> Result<(), String> {
    let (lo, hi, _) = LINE_COUNTS;
    if (lo..=hi).contains(&u) && (lo..=hi).contains(&v) {
        Ok(())
    } else {
        Err(format!("coordinate curves {u} by {v}; each count reaches {lo} to {hi}"))
    }
}

/// Whether the bundle staged as `source` may be opened: its manifest read
/// as its kind's schema, and every number in it within the controls' reach
/// or a stated bound on work. Nothing is built.
pub fn openable(source: &Source) -> Result<(), String> {
    if !source.has("manifest.json") {
        return Err("a bundle holds a manifest.json, and this has none".to_string());
    }
    let json = source.manifest_json().map_err(|e| e.to_string())?;
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
        return layers(&m.layers, extent);
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
        return layers(&m.layers, kurven_math::swift::max(u.abs(), v.abs()));
    }
    let m = Manifest::from_json(&json).map_err(|e| e.to_string())?;
    truncation(&m.truncation)?;
    // A window is an interval each way, lo below hi: one read the other way
    // round would measure every spacing against a negative extent.
    let d = m.domain;
    for (axis, i) in [("Re", d.real), ("Im", d.imag)] {
        if !(finite(i.lo) && finite(i.hi) && i.lo < i.hi && finite(i.hi - i.lo)) {
            return Err(format!("the window has {axis} from {} to {}, which is no interval", i.lo, i.hi));
        }
    }
    layers(&m.layers, kurven_math::swift::max(d.real.length(), d.imag.length()))
}

fn truncation(t: &Truncation) -> Result<(), String> {
    let ok = match t {
        Truncation::None => true,
        Truncation::Uniform(h) => finite(*h),
        Truncation::RealBands { bands, beyond } => {
            bands.len() <= LINK_BANDS && bands.iter().all(|b| finite(b.below) && finite(b.height)) && !beyond.is_nan()
        }
    };
    if ok { Ok(()) } else { Err(format!("the truncation is not finite, or has more than {LINK_BANDS} bands")) }
}

/// A count within a control's reach.
fn within(n: f64, field: Field, what: &str) -> Result<(), String> {
    let c = control(field);
    if (c.range.0..=c.range.1).contains(&n) {
        Ok(())
    } else {
        Err(format!("the surface's {what} is {n}; its control reaches {} to {}", c.range.0, c.range.1))
    }
}

fn surface(r: &SurfaceRequest) -> Result<(), String> {
    if !(8..=MAX_GRID).contains(&r.grid) {
        return Err(format!(
            "the surface's grid is {} samples round its ring; a file may ask for 8 to {MAX_GRID}",
            r.grid
        ));
    }
    match &r.shape {
        Shape::Sn { grid, .. } | Shape::Periodic { grid, .. } | Shape::Relief { grid, .. } => {
            within(*grid as f64, Field::Grid, "function's samples")?;
        }
        Shape::Forced { fit, harmonics, turns, .. } => {
            within(*fit, Field::Fit, "fit length")?;
            within(*harmonics as f64, Field::Harmonics, "harmonics")?;
            within(*turns as f64, Field::TrajectoryTurns, "trajectory's turns")?;
        }
        Shape::Torus { .. } => {}
    }
    if let Some((u, v)) = r.lines {
        line_counts(u, v).map_err(|why| format!("the surface asks for {why}"))?;
    }
    if let Some(w) = r.linear_curves {
        if !finite(w.slope) {
            return Err("the linear curve's slope is not finite".to_string());
        }
        within(w.turns as f64, Field::Turns, "linear curve's turns")?;
        within(w.count as f64, Field::Strands, "linear curve's strands")?;
    }
    Ok(())
}

/// Every layer within a bound, against `extent`, the longest side of what
/// its spacings are measured across.
pub fn layers(specs: &[LayerSpec], extent: f64) -> Result<(), String> {
    if !(finite(extent) && extent >= 0.0) {
        return Err(format!("the drawing's extent is {extent}, which is no length"));
    }
    if specs.len() > LINK_LAYERS {
        return Err(format!("the bundle carries {} layers, more than {LINK_LAYERS}", specs.len()));
    }
    let strokes = reach::STROKES.1 as f64;
    for layer in specs {
        let name = &layer.name;
        let fine = match &layer.source {
            LayerSource::File { .. } | LayerSource::TruncationEdge { .. } | LayerSource::FoldCurves => true,
            LayerSource::Trajectory | LayerSource::Edges => true,
            LayerSource::Contour { levels, light, .. } => {
                if levels.len() > LINK_LEVELS {
                    return Err(format!("{name} lists {} levels, more than {LINK_LEVELS}", levels.len()));
                }
                levels.iter().all(|&x| finite(x)) && light.is_none_or(usable_light)
            }
            LayerSource::Hatching { hatching, .. } => {
                let s = hatching.spacing;
                if !(finite(s) && s > 0.0 && (extent / s).round() <= strokes) {
                    return Err(format!(
                        "{name} is hatched at a spacing of {s}: more than {strokes} strokes across the drawing, as a \
                         strokes slider reaches"
                    ));
                }
                finite(hatching.angle)
            }
            LayerSource::WallEdges { edges, base } => {
                let mut seen = std::collections::BTreeSet::new();
                finite(*base) && edges.len() <= 4 && edges.iter().all(|&e| (0..4).contains(&e) && seen.insert(e))
            }
            LayerSource::CoordinateCurves { u, v } => {
                line_counts(*u, *v).map_err(|why| format!("{name} draws {why}"))?;
                true
            }
            LayerSource::LinearCurves { slope, turns, count } => {
                within(*turns as f64, Field::Turns, "linear curve's turns")?;
                within(*count as f64, Field::Strands, "linear curve's strands")?;
                finite(*slope)
            }
            LayerSource::IntegralCurves { seeds, .. } => {
                let many = match &seeds.spacing {
                    Spacing::Values(_, values) => {
                        values.len() as f64 > LINK_SEEDS || !values.iter().all(|&x| finite(x))
                    }
                    Spacing::ArcLength(s) => !(finite(*s) && *s > 0.0 && extent / s <= LINK_SEEDS),
                };
                if many {
                    return Err(format!("{name} is seeded more finely than {LINK_SEEDS} seeds across the drawing"));
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
