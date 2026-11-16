//! What the visibility suites share: a landscape built as
//! `NativeLandscape.build` builds it (its bundle needs the refiner, so it
//! lives here rather than in kurven-core), and the cameras the Swift suites
//! look through.

#![allow(dead_code)]

use std::path::PathBuf;

use kurven_core::bundle::KurvenBundle;
use kurven_core::camera::{rotation_x, rotation_z};
use kurven_core::grid::Grid2D;
use kurven_core::json::{JsonObject, JsonValue};
use kurven_core::manifest::{Affine2, CameraPreset, GridRef, Manifest, Occluder, Provenance, Region, Walls};
use kurven_core::npy::NpyDType;
use kurven_core::simd::{Mat3, Vec3};
use kurven_core::spaces::{Angle, Transform, ViewSpace, WorldSpace};
use kurven_core::surface::Surface;
use kurven_landscape::{ContourRefiner, LandscapeRequest, NativeLandscape};
use kurven_math::Expression;

/// `NativeLandscape.build(request, refine:)`: the grids sampled and
/// quantized, the manifest's defaults, and with `refine` the refiner's hook,
/// every described layer derived through it.
pub fn build(request: &LandscapeRequest, refine: bool) -> KurvenBundle {
    let compiled = Expression::compile(&request.expression).unwrap();
    let canonical = compiled.expression().to_string();
    let samples = NativeLandscape::sample(&compiled, request.domain, request.resolution).unwrap();
    let shape = samples.shape;
    let domain = request.domain;
    let truncation =
        request.truncation.clone().unwrap_or_else(|| NativeLandscape::default_truncation(&samples.magnitude));
    let density = NativeLandscape::wall_density(domain, shape).unwrap();
    let perimeter = NativeLandscape::rectangle(domain, &density);
    let layers = request.layers.clone().unwrap_or_else(|| {
        NativeLandscape::default_layers(
            domain,
            &truncation,
            NativeLandscape::ceiling(&truncation, &samples.magnitude),
            request.spacing,
            true,
        )
    });
    let step = 1.max(shape.n_real.max(shape.n_imag) as i64 / NativeLandscape::OCCLUDER_RESOLUTION);
    let grid_shape = (shape.n_imag as i64, shape.n_real as i64);
    let mut params = JsonObject::new();
    params.insert("expression".into(), JsonValue::String(canonical.clone()));
    params.insert("name".into(), JsonValue::String(request.name.clone()));
    params.insert("resolution".into(), JsonValue::Int(request.resolution));
    params.insert("nReal".into(), JsonValue::Int(shape.n_real as i64));
    params.insert("nImag".into(), JsonValue::Int(shape.n_imag as i64));
    let manifest = Manifest::new(
        domain,
        GridRef { file: "height.npy".into(), shape: grid_shape, dtype: NpyDType::Float32 },
        Some(GridRef { file: "phase.npy".into(), shape: grid_shape, dtype: NpyDType::Float32 }),
        truncation.clone(),
        Occluder {
            step,
            tiles: vec![Affine2::IDENTITY],
            walls: Walls::Perimeter { perimeter, base: 0.0 },
            region: Region::Full,
            base: 0.0,
        },
        layers,
        vec![CameraPreset {
            name: "oblique".into(),
            projection: NativeLandscape::PROJECTION,
            depth_tolerance: NativeLandscape::DEFAULT_DEPTH_TOLERANCE,
            buffer: NativeLandscape::DEFAULT_BUFFER,
        }],
        Provenance { function: canonical, example: "function".into(), params, cpu_count: 1, git_sha: "native".into() },
    );
    let height = Grid2D::new(shape.n_real, shape.n_imag, domain, NativeLandscape::quantize(&samples.magnitude));
    let phase = Grid2D::new(shape.n_real, shape.n_imag, domain, NativeLandscape::quantize(&samples.phase));
    let surface = Surface::new(height, Some(phase), truncation, false);
    let refine = refine.then(|| ContourRefiner::new(compiled, domain, shape.n_real, shape.n_imag).refine_hook());
    let name = if request.name.is_empty() { "landscape".to_string() } else { request.name.clone() };
    KurvenBundle::described(PathBuf::from(format!("/native/{name}.kurven")), manifest, surface, refine)
}

/// The test suites' camera: `rotationX(elevation) * rotationZ(azimuth) * shear`,
/// its rows the view's.
pub fn test_camera(elevation: f64, azimuth: f64, shear: f64) -> Transform<WorldSpace, ViewSpace> {
    let s = Mat3::from_rows(Vec3::new(1.0, shear, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0));
    let m = rotation_x(Angle::from_degrees(elevation)) * rotation_z(Angle::from_degrees(azimuth)) * s;
    Transform::from_rows((m.row(0), m.row(1), m.row(2)), Vec3::ZERO)
}

/// The cameras the stress tests orbit through: from nearly overhead to a
/// degree off edge-on, round the compass, some sheared as the plates are.
pub fn stress_cameras() -> Vec<(String, Transform<WorldSpace, ViewSpace>)> {
    let mut out = Vec::new();
    for elevation in [-80.0, -55.0, -30.0, -10.0, -3.0, -1.0] {
        for azimuth in [0.0, 37.0, 90.0, 135.0, 200.0, 289.0] {
            out.push((
                format!("el {}° az {}°", -elevation as i64, azimuth as i64),
                test_camera(elevation, azimuth, 0.0),
            ));
        }
    }
    out.push(("el 55° sheared".into(), test_camera(-55.0, -20.0, 0.5)));
    out.push(("el 5° sheared".into(), test_camera(-5.0, 70.0, 0.5)));
    out
}
