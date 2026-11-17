//! The polyhedra's golden sets (Phase 5), all in memory, so the wasm module
//! bakes them too: the seven named solids as `surface solid` draws them, the
//! frame under Turing's pattern in four widths, the dodecahedron under
//! half-Lambert in three, and a composite of the cubic landscape on its grid
//! with a cube and a frame standing on it, with their contact lines. Each as
//! its SVG and as each layer's counts, length and vertex hash
//! (`landscapes::layers`).

use kurven_bake::{Bake, BakeOptions, bake};
use kurven_core::camera::Camera;
use kurven_core::composite::{Composite, Placement};
use kurven_core::hatching::{FaceSelector, Hatching, Tone};
use kurven_core::hidden_line::HiddenLine;
use kurven_core::manifest::{Domain, HeightPolicy, Interval, LayerRole, LayerSource, LayerSpec, Truncation};
use kurven_core::polyhedron_bundle::PolyhedronBundle;
use kurven_core::scene::{PreviewMode, Scene};
use kurven_core::simd::Vec3;
use kurven_core::solid::{Platonic, Solid};
use kurven_core::surface_request::SurfaceRequest;
use kurven_core::svg::{self, Strokes, Style};
use kurven_landscape::catalog::LandscapeRequest;
use kurven_landscape::landscape::NativeLandscape;

use crate::Golden;
use crate::landscapes::layers;
use crate::leak;

pub fn all() -> Vec<Golden> {
    let mut out = Vec::new();
    let mut add = |name: &str, what: String, baked: Bake| {
        out.push(Golden {
            path: leak(format!("polyhedra/{name}.svg")),
            bytes: svg::render(&baked.strokes, svg::FIGURE_WIDTH, svg::MARGIN, svg::DECIMALS).into_bytes(),
        });
        out.push(Golden {
            path: leak(format!("polyhedra/{name}.txt")),
            bytes: format!(
                "# {what}: each layer's strokes, vertices, total length, and an FNV-1a hash of its offsets and vertex bits.\n{}",
                layers(&baked)
            )
            .into_bytes(),
        });
    };
    // As `surface solid` builds them (`Solid::named`), at its default size.
    for kind in Platonic::ALL.iter().map(|p| p.name()).chain(["csaszar", "frame"]) {
        add(kind, format!("surface solid --kind {kind}"), drawn(named(kind, None), None));
    }
    let turing = Hatching::new(30f64.to_radians_exact(), 0.035, None, Tone::Turing { scale: 0.18, seed: 3 }, 4);
    add(
        "frame-turing",
        "surface solid --kind frame --size 2 --hole 2 --depth 0.8 --hatching turing --angle 30 --spacing 0.035 \
         --width-steps 4 --scale 0.18 --seed 3"
            .to_string(),
        drawn(named("frame", Some(0.8)), Some(turing)),
    );
    let lambert =
        Hatching::new(45f64.to_radians_exact(), 0.08, None, Tone::HalfLambert { light: Vec3::new(-0.4, -0.6, 0.7) }, 3);
    add(
        "dodecahedron-half-lambert",
        "surface solid --kind dodecahedron --hatching half-lambert --width-steps 3".to_string(),
        drawn(named("dodecahedron", None), Some(lambert)),
    );
    add(
        "composite",
        "the cubic at 48 on its grid, a cube at (-1.2, 0.1, 0.3) scaled 0.35 and a frame at (0.3, -1.3, 0.2) \
         scaled 0.3, under the cubic's first preset"
            .to_string(),
        composite(),
    );
    out
}

/// One of the drawings above, by name, baked in memory: for timing the
/// engine natively and under Node (`crates/kurven-wasm/bench.mjs`).
/// "dodecahedron" is `surface solid`'s default, "frame-turing" the gallery's
/// frame (`--x-angle -55 --z-angle -60 --shear 0` aside), "composite" the
/// cubic with a cube and a frame on it.
pub fn bake_named(name: &str) -> Option<Bake> {
    Some(match name {
        "dodecahedron" => drawn(named("dodecahedron", None), None),
        "frame-turing" => drawn(
            named("frame", Some(0.8)),
            Some(Hatching::new(30f64.to_radians_exact(), 0.035, None, Tone::Turing { scale: 0.18, seed: 3 }, 4)),
        ),
        "composite" => composite(),
        _ => return None,
    })
}

/// Degrees to radians as Swift writes it, `deg * .pi / 180`.
trait Radians {
    fn to_radians_exact(self) -> f64;
}

impl Radians for f64 {
    fn to_radians_exact(self) -> f64 {
        self * std::f64::consts::PI / 180.0
    }
}

/// A solid `surface solid` builds at its default size 2 (`Solid::named`),
/// a frame `depth` thick where given.
fn named(kind: &str, depth: Option<f64>) -> Solid {
    Solid::named(kind, 2.0, None, depth).expect("a named solid")
}

/// Edges, and `hatching` on the faces where given, as `surface solid` draws
/// them under its default camera.
fn drawn(solid: Solid, hatching: Option<Hatching>) -> Bake {
    let mut specs = vec![LayerSpec::new("edges", LayerRole::Outline, LayerSource::Edges, 0.6, HeightPolicy::Surface)];
    if let Some(hatching) = hatching {
        specs.push(LayerSpec::new(
            "hatching",
            LayerRole::Scaffold,
            LayerSource::Hatching { hatching, faces: FaceSelector::Solid },
            0.3,
            HeightPolicy::Surface,
        ));
    }
    let preset = SurfaceRequest::preset();
    let bundle = PolyhedronBundle::new("/golden".into(), solid, specs, vec![preset.clone()], "golden", "golden");
    bake(&bundle.scene(&preset), &BakeOptions::new(preset.buffer)).expect("the golden solid bakes")
}

/// The cubic on its grid with a cube and a frame on it, as `scene` draws a
/// composite.
fn composite() -> Bake {
    let request = LandscapeRequest {
        truncation: Some(Truncation::Uniform(5.0)),
        ..LandscapeRequest::new("1/(z^3 - 1)", Domain::new(Interval::new(-1.3, 1.6), Interval::new(-1.2, 1.5)), 48)
    };
    let bundle = NativeLandscape::build(&request, "golden", false).expect("the golden landscape builds");
    let preset = bundle.manifest.presets[0].clone();
    let landscape = Scene::from_bundle(&bundle, &preset);
    let camera = landscape.camera;
    let solid_scene = |solid: Solid| {
        let edges = LayerSpec::new("edges", LayerRole::Outline, LayerSource::Edges, 0.6, HeightPolicy::Surface);
        let bundle =
            PolyhedronBundle::new("/golden".into(), solid, vec![edges], vec![preset.clone()], "golden", "golden");
        Scene::solid_scene(
            bundle.solid().clone(),
            bundle.layers.clone(),
            Camera::oblique(&preset.projection),
            PreviewMode::Drawing,
            0.0,
        )
    };
    let parts = vec![
        ("cubic".to_string(), landscape, Placement::IDENTITY),
        (
            "cube".to_string(),
            solid_scene(Solid::platonic(Platonic::Cube, 1.0).resting(0)),
            Placement::new(0.35, Vec3::new(-1.2, 0.1, 0.3)),
        ),
        ("frame".to_string(), solid_scene(Solid::frame(2.0, 1.0, 0.6)), Placement::new(0.3, Vec3::new(0.3, -1.3, 0.2))),
    ];
    let composite = Composite::new(parts, camera).expect("the golden composite stands");
    let visible = composite.visible_layers().expect("the golden composite draws");
    let strokes = Strokes {
        layers: visible.iter().map(|(layer, seen)| (Style::of(&layer.spec), HiddenLine::pass(seen))).collect(),
    };
    Bake {
        strokes,
        specs: visible.into_iter().map(|(layer, _)| layer.spec).collect(),
        frame: None,
        surface: None,
        tiles: 1,
    }
}

#[cfg(test)]
mod tests {
    /// The bench's drawings, timed natively in one thread, against
    /// `bench.mjs` under Node: `cargo test --release -p kurven-golden --
    /// --ignored --nocapture`.
    #[test]
    #[ignore = "a measure, not a check"]
    fn bake_times() {
        for name in ["dodecahedron", "frame-turing", "composite"] {
            let started = std::time::Instant::now();
            let baked = super::bake_named(name).expect("a drawing");
            eprintln!("{name}: {} strokes in {:.3} s", baked.strokes.path_count(), started.elapsed().as_secs_f64());
        }
    }
}
