//! The core values' golden sets (Phase 2): the camera and navigation,
//! marching squares, canonical JSON and the bundle schema, the npy and SVG
//! writers, on fixed inputs computed here, so the wasm module computes them
//! too. Floats are written as bits ([`bits`]), so native and wasm are held
//! to the bit, not to a decimal.

use kurven_core::camera::Camera;
use kurven_core::contour;
use kurven_core::grid::Grid2D;
use kurven_core::hatching::{FaceSelector, Hatching, Tone};
use kurven_core::json::{self, JsonObject, JsonValue};
use kurven_core::manifest::{
    Affine2, BoundaryPerimeter, CameraPreset, ContourField, Domain, GridRef, HeightPolicy, Interval, LayerRole,
    LayerSource, LayerSpec, Manifest, ObliqueProjection, Occluder, PerimeterEdge, PolyhedronManifest, Provenance,
    RealBand, Region, Restriction, SurfaceManifest, Truncation, Walls,
};
use kurven_core::navigation::{Framing, Gesture, Navigator, Orbit, Viewport};
use kurven_core::npy::{self, NpyDType};
use kurven_core::polyline::PolylineSet;
use kurven_core::simd::{Vec2, Vec3};
use kurven_core::solid::Solid;
use kurven_core::spaces::{AABB, Angle, DomainSpace, P2, P3, PageSpace, ViewSpace, WorldSpace};
use kurven_core::splitmix::SplitMix;
use kurven_core::surface_request::{Shape, SurfaceRequest};
use kurven_core::svg::{self, Strokes, Style};
use kurven_landscape::NativeLandscape;
use kurven_math::real;

use crate::{Golden, bits};

pub fn all() -> Vec<Golden> {
    vec![
        Golden { path: "core/camera.txt", bytes: camera_table().into_bytes() },
        Golden { path: "core/navigation.txt", bytes: navigation_table().into_bytes() },
        Golden { path: "core/contour.txt", bytes: contour_table().into_bytes() },
        Golden { path: "core/numbers.txt", bytes: number_table().into_bytes() },
        Golden { path: "core/manifests.json", bytes: manifests().into_bytes() },
        Golden { path: "core/ripple.npy", bytes: npy::encode_f32(&ripple(33, 25).values, &[25, 33]) },
        Golden { path: "core/ripple.svg", bytes: drawing().into_bytes() },
    ]
}

fn projections() -> [(&'static str, ObliqueProjection); 3] {
    ObliqueProjection::PLATES
}

/// Points spread over a box, by Swift's SplitMix (the suite's generator).
fn points(seed: u64, n: usize) -> Vec<P3<WorldSpace>> {
    let mut g = SplitMix::new(seed);
    (0..n).map(|_| P3::new(g.next(-8.0, 8.0), g.next(-30.0, 30.0), g.next(0.0, 6.0))).collect()
}

fn vec3(v: Vec3) -> String {
    format!("{} {} {}", bits(v.x), bits(v.y), bits(v.z))
}

/// Each projection's view matrix, by columns, and sixteen points through it.
fn camera_table() -> String {
    let mut out = String::from(
        "# Camera::oblique for the plates' projections: the view matrix by columns, then world -> view.\n",
    );
    for (name, projection) in projections() {
        let camera = Camera::oblique(&projection);
        out += &format!("{name}\n");
        for (k, c) in camera.view.m.columns.iter().enumerate() {
            out += &format!("  column {k}: {} {} {} {}\n", bits(c.x), bits(c.y), bits(c.z), bits(c.w));
        }
        for p in points(7, 16) {
            out += &format!("  {} -> {}\n", vec3(p.v), vec3(camera.view.apply(p).v));
        }
    }
    out
}

fn navigator_line(n: &Navigator) -> String {
    format!(
        "target {} azimuth {} elevation {} distance {} center {} {} upp {}",
        vec3(n.orbit.target.v),
        bits(n.orbit.azimuth.radians),
        bits(n.orbit.elevation.radians),
        n.orbit.distance.map_or("none".to_string(), bits),
        bits(n.framing.center.x()),
        bits(n.framing.center.y()),
        bits(n.framing.units_per_pixel)
    )
}

/// A navigator through every kind of gesture, its state after each, and a
/// pixel taken back to the world at the end.
fn navigation_table() -> String {
    let v = Viewport::new(1200, 800);
    let bounds = AABB::<ViewSpace>::new(Vec3::new(-4.0, -2.0, 0.0), Vec3::new(6.0, 3.0, 4.0));
    let gestures = [
        Gesture::Orbit(Vec2::new(37.0, -11.5)),
        Gesture::Pan(Vec2::new(-20.0, 13.0)),
        Gesture::Zoom { factor: 1.7, at: Vec2::new(300.0, 650.0) },
        Gesture::Roll(0.3),
        Gesture::Retarget(P3::new(2.5, -7.0, 3.25)),
        Gesture::Orbit(Vec2::new(-500.0, 400.0)),
        Gesture::Fit(bounds),
        Gesture::Project { field_of_view: Some(Angle::from_degrees(50.0)), bounds },
        Gesture::Pan(Vec2::new(15.0, -40.0)),
        Gesture::Zoom { factor: 2.0, at: Vec2::new(600.0, 400.0) },
        Gesture::Project { field_of_view: None, bounds },
        Gesture::Preset(projections()[2].1, bounds),
    ];
    let mut out = String::from("# Navigator::applying over a fixed sequence of gestures, the state after each.\n");
    let mut n = Navigator::new(
        Orbit::matching(&projections()[0].1, P3::new(0.0, 0.0, 0.0)),
        Framing::new(P2::new(0.0, 0.0), 0.01),
    );
    out += &format!("start {}\n", navigator_line(&n));
    for g in &gestures {
        n = n.applying(g, v);
        out += &format!("{g:?}\n  {}\n", navigator_line(&n));
    }
    let p = n.world_point_at_pixel(Vec2::new(417.25, 233.5), 1.5, v);
    out += &format!("world point at pixel (417.25, 233.5), depth 1.5: {}\n", vec3(p.v));
    let frame = n.framing.frame(v);
    out += &format!(
        "frame axis0 {} {} axis1 {} {}\n",
        bits(frame.axis0.lo),
        bits(frame.axis0.hi),
        bits(frame.axis1.lo),
        bits(frame.axis1.hi)
    );
    out
}

/// sin(3x) cos(2y) + x/4 on a grid over [-2, 2] x [-1.5, 1.5], in float32:
/// loops, open runs and saddles.
fn ripple(nx: usize, ny: usize) -> Grid2D<f32> {
    let domain = Domain::new(Interval::new(-2.0, 2.0), Interval::new(-1.5, 1.5));
    let mut values = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let x = -2.0 + 4.0 * i as f64 / (nx - 1) as f64;
            let y = -1.5 + 3.0 * j as f64 / (ny - 1) as f64;
            values.push((real::sin(3.0 * x) * real::cos(2.0 * y) + x / 4.0) as f32);
        }
    }
    Grid2D::new(nx, ny, domain, values)
}

fn paths_text(paths: &[Vec<P2<DomainSpace>>]) -> String {
    let mut out = String::new();
    for (k, path) in paths.iter().enumerate() {
        out += &format!("  path {k}: {} vertices\n", path.len());
        for p in path {
            out += &format!("    {} {}\n", bits(p.x()), bits(p.y()));
        }
    }
    out
}

/// Marching squares on the ripple at several levels, and the branch cut of
/// a phase grid.
fn contour_table() -> String {
    let grid = ripple(33, 25);
    let mut out = String::from("# contour::levels on the ripple grid, then contour::cut of a phase grid.\n");
    for (level, paths) in contour::levels(&grid, &[-0.8, -0.25, 0.0, 0.3, 0.9]) {
        out += &format!("level {}: {} paths\n{}", bits(level), paths.len(), paths_text(&paths));
    }
    // arg of (x + iy)(x - 1 + iy), whose cut runs from each zero. Stored as
    // a landscape stores its phase, by `NativeLandscape::quantize`, which
    // clamps into the samples' range: on the column x = 0.5 the sum is π to
    // rounding, either sign, and is kept at Swift's `Float.pi`, under π, as
    // the engine's grids are, not at π rounded up.
    let domain = Domain::new(Interval::new(-1.0, 2.0), Interval::new(-1.0, 1.0));
    let (nx, ny) = (25, 17);
    let mut phase = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let x = -1.0 + 3.0 * i as f64 / (nx - 1) as f64;
            let y = -1.0 + 2.0 * j as f64 / (ny - 1) as f64 + 0.03;
            // The sum of two angles in (-π, π], wrapped back into it.
            let mut a = real::atan2(y, x) + real::atan2(y, x - 1.0);
            if a > core::f64::consts::PI {
                a -= 2.0 * core::f64::consts::PI;
            } else if a <= -core::f64::consts::PI {
                a += 2.0 * core::f64::consts::PI;
            }
            phase.push(a);
        }
    }
    let cut = contour::cut(&Grid2D::new(nx, ny, domain, NativeLandscape::quantize(&phase)));
    out += &format!("cut: {} paths\n{}", cut.len(), paths_text(&cut));
    out
}

/// Doubles as the canonical writer puts them down, across the decades
/// and both of Swift's thresholds (1e-4 and 2^53).
fn number_table() -> String {
    let mut out = String::from("# json::number: bits, then the text the canonical writer writes.\n");
    let mut xs = vec![0.0, -0.0, 5.0, -79.5, 0.51, 1.0 / 3.0, 0.1, 9_007_199_254_740_992.0, 9_007_199_254_740_994.0];
    for e in -12..=20 {
        let base = real::pow(10.0, f64::from(e));
        xs.extend([base, 1.5 * base, -base / 7.0, base * core::f64::consts::PI]);
    }
    xs.extend([f64::MIN_POSITIVE, f64::from_bits(1), f64::MAX, f64::EPSILON]);
    for x in xs {
        out += &format!("{} {}\n", bits(x), json::number(x));
    }
    out
}

fn provenance(example: &str) -> Provenance {
    let mut params = JsonObject::new();
    params.insert("res".to_string(), JsonValue::Int(64));
    params.insert("note".to_string(), json::string("Γ(z) ζ\u{1}"));
    Provenance {
        function: "golden".to_string(),
        example: example.to_string(),
        params,
        cpu_count: 1,
        git_sha: "golden".to_string(),
    }
}

/// Manifests of every kind, built in code, as the canonical writer writes
/// them, one per line, each read back to the same document.
fn manifests() -> String {
    let perimeter = BoundaryPerimeter::new(vec![
        PerimeterEdge { start: P2::new(-2.0, -1.5), end: P2::new(2.0, -1.5), density: 40 },
        PerimeterEdge { start: P2::new(2.0, -1.5), end: P2::new(2.0, 1.5), density: 30 },
        PerimeterEdge { start: P2::new(2.0, 1.5), end: P2::new(-2.0, 1.5), density: 40 },
        PerimeterEdge { start: P2::new(-2.0, 1.5), end: P2::new(-2.0, -1.5), density: 30 },
    ]);
    let preset =
        CameraPreset { name: "recip".to_string(), projection: projections()[0].1, depth_tolerance: 0.02, buffer: 2000 };
    let landscape = Manifest::new(
        Domain::new(Interval::new(-2.0, 2.0), Interval::new(-1.5, 1.5)),
        GridRef { file: "height.npy".to_string(), shape: (25, 33), dtype: NpyDType::Float32 },
        Some(GridRef { file: "phase.npy".to_string(), shape: (25, 33), dtype: NpyDType::Float32 }),
        Truncation::RealBands {
            bands: vec![RealBand { below: -0.5, height: 2.5 }, RealBand { below: 1.0, height: 1e-5 }],
            beyond: 9_007_199_254_740_994.0,
        },
        Occluder {
            step: 2,
            tiles: vec![Affine2::IDENTITY, Affine2 { a: -1.0, b: 0.0, tx: 4.0, c: 0.0, d: 1.0, ty: 0.0 }],
            walls: Walls::Perimeter { perimeter: perimeter.clone(), base: -0.25 },
            region: Region::Inside(perimeter),
            base: -0.25,
        },
        vec![
            LayerSpec::new(
                "mag",
                LayerRole::Magnitude,
                LayerSource::Contour {
                    field: ContourField::Magnitude,
                    levels: vec![0.25, 0.5, 1.0 / 3.0],
                    restriction: Restriction::Intersection(vec![Restriction::Footprint, Restriction::BelowTruncation]),
                    faces: FaceSelector::Graph { tiled: true },
                    light: None,
                },
                0.35,
                HeightPolicy::Surface,
            ),
            LayerSpec::new(
                "walls",
                LayerRole::Scaffold,
                LayerSource::wall_hatching(vec![0, 1], 0.1, true, -0.25, 0.0),
                0.2,
                HeightPolicy::Surface,
            ),
            LayerSpec::new(
                "isophotes",
                LayerRole::Outline,
                LayerSource::Contour {
                    field: ContourField::Isophote,
                    levels: vec![0.0, 0.5],
                    restriction: Restriction::All,
                    faces: FaceSelector::Graph { tiled: false },
                    light: Some(Vec3::new(1.0, -1.0, 2.0)),
                },
                0.3,
                HeightPolicy::Surface,
            ),
        ],
        vec![preset.clone()],
        provenance("function"),
    );
    let surface = SurfaceManifest::new(
        {
            let mut r = SurfaceRequest::new(Shape::Relief {
                modulus: 0.5,
                major: 2.0,
                minor: 1.0,
                amplitude: 0.4,
                repeats: (2, 1),
                grid: 120,
                turn: 0.25,
                floor: None,
                soft: false,
            });
            r.grid = 256;
            r
        },
        vec![LayerSpec::new(
            "ang",
            LayerRole::Phase,
            LayerSource::Contour {
                field: ContourField::Phase,
                levels: vec![0.0, core::f64::consts::FRAC_PI_2],
                restriction: Restriction::BelowTruncation,
                faces: FaceSelector::Surface,
                light: None,
            },
            0.35,
            HeightPolicy::Surface,
        )],
        vec![SurfaceRequest::preset()],
        provenance("surface"),
    );
    let cube_vertices: Vec<P3<WorldSpace>> =
        (0..8).map(|k| P3::new((k & 1) as f64, ((k >> 1) & 1) as f64 * 1.5, ((k >> 2) & 1) as f64 * 0.75)).collect();
    let cube = Solid::new(
        cube_vertices,
        vec![
            vec![vec![0, 2, 3, 1]],
            vec![vec![4, 5, 7, 6]],
            vec![vec![0, 1, 5, 4]],
            vec![vec![2, 6, 7, 3]],
            vec![vec![0, 4, 6, 2]],
            vec![vec![1, 3, 7, 5]],
        ],
    );
    let polyhedron = PolyhedronManifest::new(
        cube,
        vec![
            LayerSpec::new("edges", LayerRole::Outline, LayerSource::Edges, 0.6, HeightPolicy::Surface),
            LayerSpec::new(
                "hatching",
                LayerRole::Scaffold,
                LayerSource::Hatching {
                    hatching: Hatching::new(0.5, 0.05, Some(0.01), Tone::Turing { scale: 0.18, seed: u64::MAX - 2 }, 4),
                    faces: FaceSelector::Solid,
                },
                0.5,
                HeightPolicy::Surface,
            ),
        ],
        vec![preset],
        provenance("polyhedron"),
    );
    let texts = [landscape.canonical_json(), surface.canonical_json(), polyhedron.canonical_json()];
    for text in &texts {
        let back = kurven_core::manifest::BundleManifest::from_json(&json::parse(text).expect("canonical JSON parses"))
            .expect("a written manifest reads back");
        assert_eq!(&back.canonical_json(), text, "a written manifest reads back to the same document");
    }
    texts.concat()
}

/// The ripple's contours lifted onto its graph, through the recip camera,
/// written as SVG: the core's half of a drawing, end to end.
fn drawing() -> String {
    let grid = ripple(33, 25);
    let camera = Camera::oblique(&projections()[0].1);
    let mut layers = Vec::new();
    for (k, (level, paths)) in contour::levels(&grid, &[-0.5, 0.0, 0.5]).into_iter().enumerate() {
        let lifted: Vec<Vec<P3<PageSpace>>> = paths
            .iter()
            .map(|path| {
                path.iter()
                    .map(|p| {
                        let q = camera.page_coordinates(P3::new(p.x(), p.y(), level + 1.0));
                        P3::new(q.x(), q.y(), 0.0)
                    })
                    .collect()
            })
            .collect();
        let style = Style { color: ["#000000", "#204080", "#802020"][k].to_string(), width: 0.35 + 0.1 * k as f64 };
        layers.push((style, PolylineSet::from_paths(&lifted)));
    }
    svg::render(&Strokes { layers }, svg::FIGURE_WIDTH, svg::MARGIN, svg::DECIMALS)
}
