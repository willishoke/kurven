//! The parametric surfaces' golden sets (Phase 4): small drawings of each
//! kind of surface `kurven-cli surface` builds, baked in memory, so the wasm
//! module bakes them too; and the coordinate image the CPU rasterizer draws,
//! as bits. Each drawing is written as its SVG and as each layer's counts,
//! length and vertex hash (`landscapes::summary`'s form).
//!
//! - `sn`: sn(z, 0.64) on the round torus, by sphere tracing;
//! - `sn-relief`: sn in relief at 0.4, the displaced torus's neighbourhood;
//! - `torus-linear`: the round torus with its coordinate curves and a 2/5
//!   linear curve;
//! - `torus-bare`: the same torus with no tubular neighbourhood, so its
//!   curves are tested by the coordinate image, which is the one bake whose
//!   strokes depend on the rasterizer;
//! - `raster.txt`: that image itself, its depth and coordinates hashed.
//!
//! Kept small (coarse grids, a 600-pixel buffer), since the wasm module
//! bakes them under Node in every check.

use std::sync::Arc;

use kurven_bake::{Bake, BakeOptions, bake};
use kurven_core::camera::Camera;
use kurven_core::depth::DepthFrame;
use kurven_core::manifest::{HeightPolicy, LayerRole, LayerSource, LayerSpec};
use kurven_core::parametric::ParametricSurface;
use kurven_core::polyline::Layer;
use kurven_core::scene::{PreviewMode, Scene};
use kurven_core::surface_image::SurfaceImage;
use kurven_core::surface_request::{LinearCurves, Shape, SurfaceRequest};
use kurven_core::svg;
use kurven_surfaces::SurfaceDrawing;

use crate::{Golden, bits, fnv, leak};

pub fn all() -> Vec<Golden> {
    let mut out = Vec::new();
    for (name, scene) in scenes() {
        let baked = bake(&scene, &BakeOptions::new(600)).expect("the golden surface bakes");
        out.push(Golden {
            path: leak(format!("surface/{name}.svg")),
            bytes: svg::render(&baked.strokes, svg::FIGURE_WIDTH, svg::MARGIN, svg::DECIMALS).into_bytes(),
        });
        out.push(Golden { path: leak(format!("surface/{name}.txt")), bytes: summary(name, &baked).into_bytes() });
    }
    out.push(Golden { path: "surface/raster.txt", bytes: raster().into_bytes() });
    out
}

/// A request at small grids, drawn as `surface` draws it, from the
/// surface's own projection.
fn drawn(mut request: SurfaceRequest) -> Scene {
    request.grid = 256;
    SurfaceDrawing::build(&request, None, false)
        .expect("the golden surface builds")
        .scene(Camera::oblique(&SurfaceRequest::PROJECTION))
}

/// The torus of `torus-linear` without its tubular neighbourhood: the same
/// map and grid, declared to bound a solid, so it has an outside and no
/// sphere tracing.
fn bare_torus() -> ParametricSurface {
    let torus = ParametricSurface::torus(2.0, 1.0, (96, 48)).expect("the torus");
    ParametricSurface::new(torus.u, torus.v, true, None, None, torus.map.clone()).expect("the bare torus")
}

fn scenes() -> Vec<(&'static str, Scene)> {
    let sn =
        drawn(SurfaceRequest::new(Shape::Sn { modulus: 0.64, major: 2.0, minor: 1.0, grid: 120, conformal: false }));
    let relief = drawn(SurfaceRequest::new(Shape::Relief {
        modulus: 0.64,
        major: 2.0,
        minor: 1.0,
        amplitude: 0.4,
        repeats: (1, 1),
        grid: 120,
        turn: 0.0,
        floor: None,
        soft: false,
    }));
    let linear = drawn(SurfaceRequest {
        lines: Some((12, 6)),
        linear_curves: Some(LinearCurves { slope: 0.4, turns: 24, count: 1 }),
        ..SurfaceRequest::new(Shape::Torus { major: 2.0, minor: 1.0 })
    });
    let bare = bare_torus();
    let curves = bare.coordinate_curves((12, 6), 384);
    let layers = vec![
        Layer {
            spec: LayerSpec::new(
                "lines",
                LayerRole::Scaffold,
                LayerSource::CoordinateCurves { u: 12, v: 6 },
                0.3,
                HeightPolicy::Surface,
            ),
            paths: curves,
        },
        Layer {
            spec: LayerSpec::new("folds", LayerRole::Outline, LayerSource::FoldCurves, 0.6, HeightPolicy::Surface),
            paths: kurven_core::polyline::PolylineSet::empty(),
        },
    ];
    let bare_scene =
        Scene::parametric_scene(bare, layers, Camera::oblique(&SurfaceRequest::PROJECTION), PreviewMode::Drawing, 0.0);
    vec![("sn", sn), ("sn-relief", relief), ("torus-linear", linear), ("torus-bare", bare_scene)]
}

fn summary(name: &str, baked: &Bake) -> String {
    let mut out = format!(
        "# {name}: each layer's strokes, vertices, total length, and an FNV-1a hash of its offsets and vertex bits.\n"
    );
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

/// The bare torus's coordinate image at 300 pixels, from the surface's own
/// projection and a sheared one: how many pixels are covered, and a hash of
/// every depth's and coordinate's bits (a NaN as one word, whatever its
/// payload, so the hash holds on every platform).
fn raster() -> String {
    let surface = Arc::new(bare_torus());
    let mut out = String::from(
        "# The bare torus's coordinate image: covered pixels, and FNV-1a of the depth bits and the coordinate bits.\n",
    );
    for (name, shear) in [("oblique", 0.0), ("sheared", 0.5)] {
        let projection = kurven_core::manifest::ObliqueProjection { shear, ..SurfaceRequest::PROJECTION };
        let view = Camera::oblique(&projection).view;
        let points: Vec<_> = surface.positions.iter().map(|&p| view.apply(p)).collect();
        let bounds = kurven_core::spaces::AABB::of(&points).expect("a grid");
        let image = SurfaceImage::rasterize(&surface, &view, DepthFrame::covering(&bounds, 300));
        let word = |x: f32| if x.is_nan() { 0x7fc0_0000u64 } else { u64::from(x.to_bits()) };
        let covered = image.depth.values.iter().filter(|&&v| v > image.depth.empty).count();
        let depth = fnv(image.depth.values.iter().map(|&v| word(v)));
        let coords = fnv(image.coords.iter().flat_map(|c| [word(c[0]), word(c[1])]));
        out += &format!("{name} covered {covered} depth {depth:016x} coords {coords:016x}\n");
    }
    out
}

/// Bakes a reference surface drawing in memory, at the size `kurven-cli
/// surface NAME` draws it: 0 is `surface sn`, 1 `surface sn --relief 0.4`.
/// For timing the engine native and under Node (`bench.mjs`).
pub fn bake_reference(which: u32) -> Option<Bake> {
    let request = match which {
        0 => SurfaceRequest {
            name: "sn".to_string(),
            ..SurfaceRequest::new(Shape::Sn { modulus: 0.64, major: 2.0, minor: 1.0, grid: 600, conformal: false })
        },
        1 => SurfaceRequest {
            grid: 8192,
            ..SurfaceRequest::new(Shape::Relief {
                modulus: 0.64,
                major: 2.0,
                minor: 1.0,
                amplitude: 0.4,
                repeats: (1, 1),
                grid: 600,
                turn: 0.0,
                floor: None,
                soft: false,
            })
        },
        _ => return None,
    };
    let drawing = SurfaceDrawing::build(&request, None, false).ok()?;
    let scene = drawing.scene(Camera::oblique(&SurfaceRequest::PROJECTION));
    bake(&scene, &BakeOptions::new(3000)).ok()
}
