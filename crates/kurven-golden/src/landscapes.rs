//! The landscapes' golden sets (Phase 3): two small landscapes sampled,
//! placed on their function, tested for visibility and baked, all in
//! memory, so the wasm module bakes them too. Each is written twice: as the
//! SVG the bake writes (to four decimals, what a reader looks at), and as
//! each layer's stroke count, total length and a hash of every vertex's
//! bits, which holds native and wasm to the bit.
//!
//! Kept small (a coarse grid each), since the wasm module bakes them under
//! Node in every check.

use kurven_bake::{Bake, BakeOptions, bake};
use kurven_core::manifest::{Domain, Interval, RealBand, Truncation};
use kurven_core::scene::Scene;
use kurven_core::svg;
use kurven_landscape::catalog::LandscapeRequest;
use kurven_landscape::landscape::NativeLandscape;

use crate::{Golden, bits, fnv, leak};

pub fn all() -> Vec<Golden> {
    let mut out = Vec::new();
    for (name, request) in requests() {
        let baked = baked(&request);
        out.push(Golden {
            path: leak(format!("landscape/{name}.svg")),
            bytes: svg::render(&baked.strokes, svg::FIGURE_WIDTH, svg::MARGIN, svg::DECIMALS).into_bytes(),
        });
        out.push(Golden { path: leak(format!("landscape/{name}.txt")), bytes: summary(&request, &baked).into_bytes() });
    }
    out
}

/// The cubic of the gallery at a coarse grid, truncated at a uniform
/// height; and gamma truncated by bands of Re z, as the gallery's plate is.
fn requests() -> Vec<(&'static str, LandscapeRequest)> {
    let cubic = LandscapeRequest {
        truncation: Some(Truncation::Uniform(5.0)),
        ..LandscapeRequest::new("1/(z^3 - 1)", Domain::new(Interval::new(-1.3, 1.6), Interval::new(-1.2, 1.5)), 64)
    };
    let band = |below, height| RealBand { below, height };
    let gamma = LandscapeRequest {
        truncation: Some(Truncation::RealBands {
            bands: vec![band(-3.5, 2.5), band(-2.5, 3.0), band(-1.5, 4.0), band(0.5, 5.5)],
            beyond: 4.0,
        }),
        ..LandscapeRequest::new("gamma(z)", Domain::new(Interval::new(-4.5, 4.5), Interval::new(-2.5, 2.5)), 90)
    };
    vec![("cubic", cubic), ("gamma-bands", gamma)]
}

/// The request built on its function and baked under its first preset, as
/// `landscape` then `bake` draw it.
fn baked(request: &LandscapeRequest) -> Bake {
    let bundle = NativeLandscape::build(request, "golden", true).expect("the golden landscape builds");
    let preset = bundle.manifest.presets[0].clone();
    let scene = Scene::from_bundle(&bundle, &preset);
    let options = BakeOptions { depth_tolerance: Some(preset.depth_tolerance), ..BakeOptions::new(preset.buffer) };
    bake(&scene, &options).expect("the golden landscape bakes")
}

fn summary(request: &LandscapeRequest, baked: &Bake) -> String {
    let mut out = format!(
        "# {} over {} at {}: each layer's strokes, vertices, total length, and an FNV-1a hash of its offsets and vertex bits.\n",
        request.expression,
        request.domain.json().canonical(),
        request.resolution
    );
    out += &layers(baked);
    out
}

/// Each stroke layer's name, counts, length bits and vertex hash, a line each.
pub(crate) fn layers(baked: &Bake) -> String {
    let mut out = String::new();
    for (spec, (_, paths)) in baked.specs.iter().zip(&baked.strokes.layers) {
        let offsets = paths.offsets.iter().map(|&o| o as u64);
        let vertices = paths.vertices.iter().flat_map(|v| [v.x().to_bits(), v.y().to_bits(), v.v.z.to_bits()]);
        out += &format!(
            "{} paths {} vertices {} length {} hash {:016x}\n",
            spec.name,
            paths.count(),
            paths.vertices.len(),
            bits(paths.total_length()),
            fnv(offsets.chain(vertices))
        );
    }
    out
}
