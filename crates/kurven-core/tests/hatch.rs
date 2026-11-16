//! `Sources/kurven-test/main.swift`'s `hatchTests`: the four hatching
//! sources against the golden strokes of `tests/fixtures/hatch`, and what
//! the plateaus' hatching is for, with Swift's tolerances.
//!
//! Every height in the fixture is read out of the float32 grid the fixture
//! ships, so this is not a tolerance test of two approximations: it is the
//! same arithmetic on the same samples, and the answers are expected to
//! agree to the last few bits.

use kurven_core::fixtures;
use kurven_core::grid::Grid2D;
use kurven_core::hatch::HatchContext;
use kurven_core::hatching::{FaceSelector, Hatching, Tone, Turing};
use kurven_core::json::{self, JsonObject, JsonValue, Members};
use kurven_core::manifest::{
    Affine2, BoundaryPerimeter, Domain, DomainAxis, HeightPolicy, LayerRole, LayerSource, LayerSpec, Occluder, Region,
    Restriction, Truncation, Walls,
};
use kurven_core::polyline::PolylineSet;
use kurven_core::simd::{Vec2, Vec3};
use kurven_core::spaces::{P3, WorldSpace};
use kurven_core::surface::Surface;
use kurven_math::swift;

struct Fixture {
    index: JsonObject,
    grid: Grid2D<f32>,
    perimeter: BoundaryPerimeter,
}

/// "hatch: the fixture loads": everything else reads it.
fn fixture() -> Fixture {
    let index = fixtures::shared_json("hatch/index.json").as_object().expect("hatch/index.json is an object").clone();
    let domain = Domain::from_json(index.value("domain", "hatch").unwrap()).unwrap();
    let shape = index.value("shape", "hatch").unwrap().object("hatch.shape").unwrap();
    let values = fixtures::shared_npy("hatch/height.npy").floats().unwrap();
    let grid = Grid2D::new(
        shape.int("nx", "hatch.shape").unwrap() as usize,
        shape.int("ny", "hatch.shape").unwrap() as usize,
        domain,
        values,
    );
    let perimeter = BoundaryPerimeter::from_json(index.value("perimeter", "hatch").unwrap()).unwrap();
    Fixture { index, grid, perimeter }
}

impl Fixture {
    fn truncation(&self, name: &str) -> Truncation {
        let truncations = self.index.value("truncation", "hatch").unwrap().object("hatch.truncations").unwrap();
        Truncation::from_json(truncations.value(name, "hatch.truncations").unwrap()).unwrap()
    }

    fn surface(&self, truncation: &str) -> Surface {
        Surface::new(self.grid.clone(), None, self.truncation(truncation), false)
    }

    /// Each case: its name, its source, its truncation's name, and its
    /// committed strokes.
    fn cases(&self) -> Vec<(String, LayerSource, String, PolylineSet<WorldSpace>)> {
        self.index
            .array("cases", "hatch")
            .unwrap()
            .iter()
            .map(|c| {
                let c = c.object("hatch.case").unwrap();
                let verts = fixtures::shared_npy(&format!("hatch/{}", c.string("vertices", "hatch.case").unwrap()))
                    .rows3()
                    .unwrap();
                let offsets = fixtures::shared_npy(&format!("hatch/{}", c.string("offsets", "hatch.case").unwrap()))
                    .ints()
                    .unwrap();
                let expected = PolylineSet::new(
                    verts.iter().map(|v| P3::new(v[0], v[1], v[2])).collect(),
                    offsets.iter().map(|&o| o as usize).collect(),
                    None,
                );
                (
                    c.string("name", "hatch.case").unwrap(),
                    LayerSource::from_json(c.value("source", "hatch.case").unwrap()).unwrap(),
                    c.string("truncation", "hatch.case").unwrap(),
                    expected,
                )
            })
            .collect()
    }
}

fn segment_midpoints(paths: &PolylineSet<WorldSpace>) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for path in paths.paths() {
        for w in path.windows(2) {
            let m = (w[0].v + w[1].v) / 2.0;
            out.push([m.x, m.y, m.z]);
        }
    }
    out
}

/// How many midpoints of one set have no counterpart in the other within
/// 1e-9, and the worst distance among them (Swift's `strayed`).
fn strayed(a: &[[f64; 3]], b: &[[f64; 3]]) -> (usize, f64) {
    let directed = |from: &[[f64; 3]], to: &[[f64; 3]]| -> (usize, f64) {
        let (mut count, mut worst) = (0, 0.0f64);
        for p in from {
            let mut best = f64::INFINITY;
            for q in to {
                let d = (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) + (p[2] - q[2]) * (p[2] - q[2]);
                best = best.min(d);
            }
            let d = best.sqrt();
            if d > 1e-9 {
                count += 1;
                worst = worst.max(d);
            }
        }
        (count, worst)
    };
    let (ca, wa) = directed(a, b);
    let (cb, wb) = directed(b, a);
    (ca + cb, wa.max(wb))
}

#[test]
fn the_fixture_loads() {
    let f = fixture();
    assert_eq!((f.grid.width, f.grid.height), (37, 23));
    assert_eq!(f.perimeter.edges().len(), 4);
    assert_eq!(f.cases().len(), 12);
}

/// A stroke is a stroke, vertex for vertex -- both sides walked the same
/// samples in the same order, held to 1e-9. A truncation edge is a
/// marching-squares contour, and a closed loop has no first vertex, so it is
/// compared as the set of segments it draws, every midpoint within 1e-9 of
/// one in the other.
#[test]
fn every_source_reproduces_the_committed_strokes() {
    let f = fixture();
    let context = HatchContext::with_perimeter(Some(f.perimeter.clone()));
    let mut failures = Vec::new();
    for (name, source, truncation, expected) in f.cases() {
        let surface = f.surface(&truncation);
        let Some(derived) = surface.hatch(&source, &context, None) else {
            failures.push(format!("{name} is not a hatching"));
            continue;
        };
        if derived.count() != expected.count() {
            failures.push(format!("{name}: {} strokes, derived {}", expected.count(), derived.count()));
            continue;
        }
        if matches!(source, LayerSource::TruncationEdge { .. }) {
            let (a, b) = (segment_midpoints(&derived), segment_midpoints(&expected));
            let (unmatched, worst) = strayed(&a, &b);
            if !(a.len() == b.len() && unmatched == 0) {
                failures
                    .push(format!("{name}: {unmatched} of {} segments unmatched, worst {worst}", a.len() + b.len()));
            }
            continue;
        }
        let mut worst = 0.0f64;
        let mut shape_ok = true;
        for i in 0..derived.count() {
            let (a, b) = (derived.path(i), expected.path(i));
            if a.len() != b.len() {
                shape_ok = false;
                break;
            }
            for (p, q) in a.iter().zip(b) {
                worst = worst.max((p.v - q.v).abs().reduce_max());
            }
        }
        if !(shape_ok && worst < 1e-9) {
            failures.push(format!(
                "{name}: {} strokes, {}",
                derived.count(),
                if shape_ok { format!("max |Δ| = {worst}") } else { "a stroke has a different length".into() }
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Plateau hatching is defined by its expansion: the contours of Im z, for
/// lines along the real axis, or of Re z, on the plateaus, at levels that
/// divide the domain across into whole gaps, the two on its edges left out.
/// Written out here rather than asked of `plateau_hatching`, and held to the
/// committed strokes bit for bit.
#[test]
fn a_plateaus_hatching_is_the_contours_of_a_coordinate_on_the_plateaus_bit_for_bit() {
    let f = fixture();
    let occluder = Occluder {
        step: 1,
        tiles: vec![Affine2::IDENTITY],
        walls: Walls::Perimeter { perimeter: f.perimeter.clone(), base: 0.0 },
        region: Region::Full,
        base: 0.0,
    };
    let mut checked = 0;
    for (name, source, truncation, expected) in f.cases() {
        let LayerSource::Hatching { hatching, faces: FaceSelector::Plateaus { tiled } } = source else { continue };
        let Some(axis) = DomainAxis::from_angle(hatching.angle) else { continue };
        let across = if axis == DomainAxis::Real { f.grid.domain.imag } else { f.grid.domain.real };
        let n = (((across.hi - across.lo) / hatching.spacing + 0.5).floor()) as usize + 1;
        let levels: Vec<f64> =
            (1..(n - 1)).map(|k| across.lo + (across.hi - across.lo) * k as f64 / (n - 1) as f64).collect();
        let spec = LayerSpec::new(
            &name,
            LayerRole::Scaffold,
            LayerSource::Contour {
                field: if axis == DomainAxis::Real {
                    kurven_core::manifest::ContourField::V
                } else {
                    kurven_core::manifest::ContourField::U
                },
                levels: levels.clone(),
                restriction: Restriction::All,
                faces: FaceSelector::Plateaus { tiled },
                light: None,
            },
            0.2,
            HeightPolicy::Level,
        );
        let contours = f.surface(&truncation).curves(&spec, &occluder, None);
        assert!(
            contours == expected,
            "{name}: the contours at {} levels on the plateaus: {} paths, {} vertices; committed {}, {}",
            levels.len(),
            contours.count(),
            contours.vertices.len(),
            expected.count(),
            expected.vertices.len()
        );
        checked += 1;
    }
    assert_eq!(checked, 4);
}

/// What the hatching is for, held to arithmetic rather than to Python: the
/// top's strokes split its width into equal whole gaps, and every one of
/// them stands where a wall across it stands a stroke, so the hatching turns
/// the corner instead of missing it. 0.3 does not divide the fixture's
/// width, which is the case the rule exists for.
#[test]
fn a_plateaus_hatching_divides_its_width_and_meets_the_walls() {
    let f = fixture();
    let surface = f.surface("uniform");
    let context = HatchContext::with_perimeter(Some(f.perimeter.clone()));
    let spacing = 0.3;
    let across = f.grid.domain.imag;
    let width = across.hi - across.lo;
    let gaps = ((width / spacing + 0.5).floor() as i64) as f64;
    let top = surface
        .hatch(&LayerSource::plateau_hatching(DomainAxis::Real, spacing, false), &context, None)
        .expect("the plateau hatching derives");
    let mut rows: Vec<f64> = top.paths().filter_map(|p| p.first().map(|v| v.y())).collect();
    rows.sort_by(|a, b| a.partial_cmp(b).unwrap());
    rows.dedup();
    assert!(
        !rows.is_empty() && width / spacing != gaps,
        "the top is ruled, at a spacing that does not divide it: {} rows, {} spacings across",
        rows.len(),
        width / spacing
    );
    let off_grid = rows
        .iter()
        .map(|c| {
            let k = (c - across.lo) / width * gaps;
            (k - k.round()).abs()
        })
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(off_grid < 1e-9, "every row stands on a whole gap of width / {gaps}: worst {off_grid} of a gap");

    let crossing: Vec<usize> = (0..f.perimeter.edges().len())
        .filter(|&i| f.perimeter.edges()[i].start.x() == f.perimeter.edges()[i].end.x())
        .collect();
    assert!(!crossing.is_empty());
    for i in crossing {
        let wall = surface
            .hatch(&LayerSource::wall_hatching(vec![i as i64], spacing, false, 0.0, 0.0), &context, None)
            .unwrap_or_else(|| panic!("the wall on edge {i} derives"));
        let posts: Vec<f64> = wall.paths().filter_map(|p| p.first().map(|v| v.y())).collect();
        let missed = rows
            .iter()
            .map(|c| posts.iter().map(|p| (p - c).abs()).fold(f64::INFINITY, f64::min))
            .fold(0.0f64, f64::max);
        assert!(
            posts.len() == gaps as usize - 1 && missed < 1e-12,
            "the wall on edge {i} stands a stroke under every row: {} strokes, worst miss {missed}",
            posts.len()
        );
    }
}

/// Hatchings.swift's "hatching: one rule for a manifest's step and angles,
/// on walls as on plateaus".
#[test]
fn one_rule_for_a_manifests_step_and_angles_on_walls_as_on_plateaus() {
    let source = |faces: &str, angle: &str| {
        let text = format!(
            r#"{{"kind": "hatching", "faces": {faces}, "angle": {angle}, "spacing": 0.25,
                 "widthSteps": 1, "tone": {{"kind": "uniform"}}}}"#
        );
        LayerSource::from_json(&kurven_core::json::parse(&text).unwrap())
    };
    let walls = r#"{"kind": "walls", "edges": [0]}"#;
    let plateaus = r#"{"kind": "plateaus"}"#;
    for (faces, angle) in [(walls, "1.5707963267948966"), (plateaus, "0")] {
        let Ok(LayerSource::Hatching { hatching: h, .. }) = source(faces, angle) else { panic!("{faces} decodes") };
        assert_eq!(h.step, h.spacing, "a step left out is the spacing on {faces}");
    }
    assert!(
        matches!(source(walls, "1.5708"), Ok(LayerSource::Hatching { faces: FaceSelector::Walls { .. }, .. })),
        "π/2 written as 1.5708 decodes as vertical on a wall"
    );
    assert!(
        source(plateaus, "1.5708").unwrap().plateau_hatch_axis() == Some(DomainAxis::Imag)
            && source(plateaus, "0.0001").unwrap().plateau_hatch_axis() == Some(DomainAxis::Real),
        "and on a plateau it is along the imaginary axis, as a near 0 is along the real"
    );
    assert!(source(walls, "1.58").is_err(), "an angle a hundredth off vertical is refused on a wall");
    assert!(source(plateaus, "0.01").is_err(), "and a hundredth off an axis on a plateau");
}

/// Hatchings.swift's "hatching: Turing's pattern is a function of the seed,
/// mean zero, half positive".
#[test]
fn turings_pattern_is_a_function_of_the_seed_mean_zero_half_positive() {
    let p = Vec2::new(3.1, -2.7);
    assert!(
        Turing::value(p, 7) == Turing::value(p, 7) && Turing::value(p, 7) != Turing::value(p, 8),
        "the same point under the same seed is the same value, under another seed another"
    );
    let (mut total, mut positive, mut n) = (0.0, 0, 0);
    let (mut lo, mut hi) = (1.0f64, -1.0f64);
    for i in 0..120 {
        for j in 0..120 {
            let v = Turing::value(Vec2::new(i as f64 * 0.5, j as f64 * 0.5), 3);
            total += v;
            n += 1;
            if v > 0.0 {
                positive += 1;
            }
            lo = swift::min(lo, v);
            hi = swift::max(hi, v);
        }
    }
    let (mean, share) = (total / n as f64, positive as f64 / n as f64);
    assert!(
        mean.abs() < 0.05 && (share - 0.5).abs() < 0.05 && lo < -0.5 && hi > 0.5,
        "over a 60 by 60 window the mean is near zero and half the pattern is positive: mean {mean:.3}, {:.0}% \
         positive, range [{lo:.2}, {hi:.2}]",
        share * 100.0
    );
}

/// Hatchings.swift's "hatching: a tone or a step that cannot be read is
/// refused at decode".
#[test]
fn a_tone_or_a_step_that_cannot_be_read_is_refused_at_decode() {
    let hatching = |tone: &str, step: &str| -> Result<Hatching, String> {
        let text = format!(r#"{{"angle": 0, "spacing": 0.1, "widthSteps": 1{step}, "tone": {tone}}}"#);
        let json = json::parse(&text).map_err(|e| format!("{e:?}"))?;
        let o = json.object("hatching").map_err(|e| format!("{e:?}"))?;
        Hatching::from_fields(o).map_err(|e| format!("{e:?}"))
    };
    assert!(hatching(r#"{"kind": "turing", "scale": 0.5, "seed": 1}"#, "").is_ok(), "a well-formed hatching decodes");
    for scale in ["0", "-0.5"] {
        assert!(
            hatching(&format!(r#"{{"kind": "turing", "scale": {scale}, "seed": 1}}"#), "").is_err(),
            "Turing's tone at scale {scale} is refused"
        );
    }
    assert!(
        hatching(r#"{"kind": "halfLambert", "light": [0, 0, 0]}"#, "").is_err(),
        "a light of no direction is refused"
    );
    for step in ["0", "-0.1", "1e-300"] {
        assert!(
            hatching(r#"{"kind": "uniform"}"#, &format!(r#", "step": {step}"#)).is_err(),
            "a step of {step} is refused"
        );
    }
}

/// Hatchings.swift's "hatching: a hatching and a hatching layer round-trip
/// through their fields".
#[test]
fn a_hatching_and_a_hatching_layer_round_trip_through_their_fields() {
    let hatchings = [
        Hatching::plain(0.4, 0.1),
        Hatching::new(-1.2, 0.05, Some(0.01), Tone::Turing { scale: 0.7, seed: 0xDEAD_BEEF_0000_0001 }, 5),
        Hatching::new(0.0, 0.2, None, Tone::HalfLambert { light: Vec3::new(0.1, -0.3, 0.9) }, 2),
    ];
    for hatching in hatchings {
        let back = Hatching::from_fields(&hatching.fields()).unwrap();
        assert!(back == hatching, "{} decodes to itself", JsonValue::Object(hatching.fields()).canonical());
    }
    let sources = [
        LayerSource::wall_hatching(vec![0, 2], 0.3, true, -0.5, 0.02),
        LayerSource::plateau_hatching(DomainAxis::Imag, 0.4, true),
    ];
    for source in sources {
        assert!(
            LayerSource::from_json(&source.json()).unwrap() == source,
            "{} decodes to itself",
            source.json().canonical()
        );
    }
}
