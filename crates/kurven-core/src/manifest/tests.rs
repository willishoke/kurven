//! The contract suites of `Sources/kurven-test/main.swift`.

use std::path::PathBuf;

use super::*;
use crate::fixtures;

/// The contract bundles, sorted by path, as Swift lists them.
fn contract_bundles() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(fixtures::shared("contract"))
        .expect("tests/fixtures/contract")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "kurven"))
        .collect();
    out.sort();
    out
}

/// "contract: every fixture manifest round trips byte for byte"
#[test]
fn every_fixture_manifest_round_trips_byte_for_byte() {
    let bundles = contract_bundles();
    assert_eq!(bundles.len(), 18, "eighteen fixture bundles are present");
    for b in &bundles {
        let name = b.file_name().unwrap().to_string_lossy();
        let text = std::fs::read_to_string(b.join("manifest.json")).unwrap();
        let manifest =
            BundleManifest::from_json(&json::parse(&text).unwrap()).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            manifest.canonical_json() == text,
            "{name} round trips byte for byte:\n{}\n{text}",
            manifest.canonical_json()
        );
    }
}

/// Each kind of manifest decodes as its own kind, and a landscape's
/// reader refuses the other two.
#[test]
fn bundles_decode_as_their_kind() {
    let mut kinds = [0; 3];
    for b in contract_bundles() {
        let v = json::parse(&std::fs::read_to_string(b.join("manifest.json")).unwrap()).unwrap();
        match BundleManifest::from_json(&v).unwrap() {
            BundleManifest::Landscape(m) => {
                kinds[0] += 1;
                assert_eq!(m.layers.len(), v.get("layers").unwrap().as_array().unwrap().len());
            }
            BundleManifest::Surface(_) => {
                kinds[1] += 1;
                assert!(Manifest::from_json(&v).is_err());
            }
            BundleManifest::Polyhedron(p) => {
                kinds[2] += 1;
                assert!(Manifest::from_json(&v).is_err());
                assert!(!p.solid.pieces.is_empty());
            }
        }
    }
    assert_eq!(kinds, [5, 6, 7], "landscapes, surfaces, polyhedra");
}

/// "contract: what a reader cannot represent is refused, not guessed at"
#[test]
fn what_a_reader_cannot_represent_is_refused_not_guessed_at() {
    let good = fixtures::shared_json("contract/empty.kurven/manifest.json").as_object().unwrap().clone();
    let mutated = |change: &dyn Fn(&mut JsonObject)| -> JsonValue {
        let mut d = good.clone();
        change(&mut d);
        JsonValue::Object(d)
    };
    let occluder = |key: &str, v: JsonValue| -> JsonValue {
        mutated(&|d| {
            let mut o = d["occluder"].as_object().cloned().unwrap_or_default();
            o.insert(key.to_string(), v.clone());
            d.insert("occluder".to_string(), JsonValue::Object(o));
        })
    };
    assert!(Manifest::from_json(&JsonValue::Object(good.clone())).is_ok(), "the fixture itself decodes");
    assert!(
        Manifest::from_json(&mutated(&|d| {
            d.insert("schema".into(), JsonValue::Int(999));
        }))
        .is_err(),
        "rejects an unsupported schema"
    );
    assert!(
        Manifest::from_json(&mutated(&|d| {
            d.insert("schema".into(), JsonValue::Int(1));
        }))
        .is_err(),
        "rejects schema 1, whose names schema 2 replaced"
    );
    let error = Manifest::from_json(&mutated(&|d| {
        d.insert("schema".into(), JsonValue::Int(2));
    }))
    .unwrap_err();
    assert!(
        error == SchemaError::Manifest(ManifestError::UnsupportedSchema { schema: 2, supported: 3 })
            && error.to_string() == "manifest: schema 2 is not the supported 3",
        "rejects schema 2, whose layers name no face, saying so: {error}"
    );
    let with = |source: LayerSource, key: &str, v: JsonValue| -> JsonValue {
        let JsonValue::Object(mut o) = source.json() else { unreachable!() };
        o.insert(key.to_string(), v);
        JsonValue::Object(o)
    };
    assert!(
        LayerSource::from_json(&with(
            LayerSource::wall_hatching(vec![0], 0.1, false, 0.0, 0.0),
            "step",
            JsonValue::Double(0.1)
        ))
        .is_err(),
        "rejects a step on a wall's hatching: a line there is its two ends"
    );
    assert!(
        LayerSource::from_json(&with(
            LayerSource::WallEdges { edges: vec![0], base: 0.0 },
            "step",
            JsonValue::Double(0.1)
        ))
        .is_err(),
        "rejects a step on a wall's edges"
    );
    let walls = FaceSelector::Walls { edges: vec![0], end_lines: false, base: 0.0, top_offset: 0.0 };
    assert!(
        LayerSource::from_json(&LayerSource::Hatching { hatching: Hatching::plain(0.3, 0.1), faces: walls }.json())
            .is_err(),
        "rejects hatching on walls that is not vertical"
    );
    let turing = Hatching::new(0.0, 0.1, None, Tone::Turing { scale: 1.0, seed: 1 }, 1);
    assert!(
        LayerSource::from_json(
            &LayerSource::Hatching { hatching: turing, faces: FaceSelector::Plateaus { tiled: false } }.json()
        )
        .is_err(),
        "rejects a tone other than uniform on plateaus"
    );
    assert!(
        LayerSource::from_json(
            &LayerSource::Hatching { hatching: Hatching::plain(0.3, 0.1), faces: FaceSelector::Solid }.json()
        )
        .is_err(),
        "rejects hatching on a solid's faces: a manifest is a heightfield's"
    );
    let axes = JsonValue::Array(["imag", "real", "magnitude"].iter().map(|s| string(s)).collect());
    assert!(
        Manifest::from_json(&mutated(&|d| {
            d.insert("axes".into(), axes.clone());
        }))
        .is_err(),
        "rejects a bundle in another axis order"
    );
    assert!(
        Manifest::from_json(&mutated(&|d| {
            d.insert("truncation".into(), object([("kind", string("logarithmic"))]));
        }))
        .is_err(),
        "rejects an unknown truncation kind"
    );
    assert!(
        Manifest::from_json(&occluder("walls", object([("kind", string("spline"))]))).is_err(),
        "rejects an unknown wall kind"
    );
    assert!(
        Manifest::from_json(&occluder("region", object([("kind", string("mask"))]))).is_err(),
        "rejects an unknown region kind"
    );
    assert!(
        Manifest::from_json(&mutated(&|d| {
            d.remove("domain");
        }))
        .is_err(),
        "rejects a manifest with no domain"
    );
}

/// "core: point-in-polygon scans along x, matching kurven.bundle"
#[test]
fn point_in_polygon_scans_along_x() {
    // zeta's staircase, in world (x = real, y = imag).
    let corners: Vec<P2<WorldSpace>> = [
        (-6.0, 28.0),
        (-6.0, 0.0),
        (-2.0, 0.0),
        (-2.0, -5.0),
        (-0.5, -5.0),
        (-0.5, -14.0),
        (0.5, -14.0),
        (0.5, -28.0),
        (8.0, -28.0),
        (8.0, 28.0),
    ]
    .iter()
    .map(|&(x, y)| P2::new(x, y))
    .collect();
    assert!(point_in_polygon(P2::new(4.0, 0.0), &corners), "deep inside");
    assert!(!point_in_polygon(P2::new(-4.0, -20.0), &corners), "in the notch");
    assert!(point_in_polygon(P2::new(-4.0, 10.0), &corners), "in the pole arm");
    assert!(!point_in_polygon(P2::new(-9.0, 0.0), &corners), "left of everything");
    assert!(!point_in_polygon(P2::new(0.0, 40.0), &corners), "past the top");
}

/// "core: truncations are a property of the model, and bands read in order"
#[test]
fn truncations_are_a_property_of_the_model_and_bands_read_in_order() {
    assert_eq!(Truncation::None.height_at(3.0), f64::INFINITY, "no truncation is no truncation");
    assert_eq!(Truncation::Uniform(5.0).height_at(-100.0), 5.0, "a uniform truncation ignores x");
    let bands = Truncation::RealBands {
        bands: vec![
            RealBand { below: -3.5, height: 8.0 },
            RealBand { below: -2.5, height: 6.0 },
            RealBand { below: -1.5, height: 4.0 },
        ],
        beyond: f64::INFINITY,
    };
    assert!(
        bands.height_at(-4.0) == 8.0
            && bands.height_at(-3.0) == 6.0
            && bands.height_at(-2.0) == 4.0
            && bands.height_at(0.0) == f64::INFINITY,
        "the first matching band wins, and past the last there is none"
    );
    // Gamma's calm region: past the spires there is still a truncation.
    let calm = Truncation::RealBands {
        bands: vec![RealBand { below: -3.5, height: 2.5 }, RealBand { below: 0.5, height: 5.5 }],
        beyond: 3.0,
    };
    assert!(
        calm.height_at(-4.0) == 2.5 && calm.height_at(0.0) == 5.5 && calm.height_at(2.0) == 3.0,
        "and `beyond` is the truncation where no band matches"
    );
    assert_eq!(calm.height_at_from(-3.5, Side::Left), 2.5, "the band from the left ends there");
    assert_eq!(calm.band_at(-3.5), Some(1));
    let spans = calm.bands(-5.0, 1.0);
    assert_eq!(
        spans.iter().map(|s| (s.band, s.lo, s.hi)).collect::<Vec<_>>(),
        [(0, -5.0, -3.5), (1, -3.5, 0.5), (2, 0.5, 1.0)]
    );
    assert_eq!(calm.highest(), 5.5);
}

/// A manifest built in code writes the document it reads back.
#[test]
fn a_built_manifest_round_trips() {
    let m = Manifest::new(
        Domain::new(Interval::new(-2.0, 2.0), Interval::new(-1.5, 1.5)),
        GridRef { file: "height.npy".into(), shape: (3, 4), dtype: NpyDType::Float32 },
        None,
        Truncation::Uniform(5.0),
        Occluder { step: 1, tiles: vec![Affine2::IDENTITY], walls: Walls::None, region: Region::Full, base: 0.0 },
        vec![LayerSpec::new(
            "mag",
            LayerRole::Magnitude,
            LayerSource::Contour {
                field: ContourField::Magnitude,
                levels: vec![0.5, 1.0],
                restriction: Restriction::BelowTruncation,
                faces: FaceSelector::Graph { tiled: false },
                light: None,
            },
            0.35,
            HeightPolicy::Surface,
        )],
        vec![],
        Provenance {
            function: "z".into(),
            example: "".into(),
            params: JsonObject::new(),
            cpu_count: 1,
            git_sha: "golden".into(),
        },
    );
    let text = m.canonical_json();
    let back = Manifest::from_json(&json::parse(&text).unwrap()).unwrap();
    assert_eq!(back, m);
    assert_eq!(back.canonical_json(), text);
    assert!(m.preset("recip").is_err());
    assert_eq!(m.layer("mag").unwrap().name, "mag");
}

/// A region's footprint test is the corners' point-in-polygon, point for
/// point, scanned in place.
#[test]
fn a_regions_footprint_is_its_corners_polygon() {
    let corners: [(f64, f64); 6] = [(-6.0, 28.0), (-6.0, 0.0), (-2.0, 0.0), (-2.0, -5.0), (8.0, -28.0), (8.0, 28.0)];
    let edges = (0..corners.len())
        .map(|k| {
            let (a, b) = (corners[k], corners[(k + 1) % corners.len()]);
            PerimeterEdge { start: P2::new(a.0, a.1), end: P2::new(b.0, b.1), density: 1 }
        })
        .collect();
    let region = Region::Inside(BoundaryPerimeter::new(edges));
    let polygon: Vec<P2<WorldSpace>> = corners.iter().map(|&(x, y)| P2::new(x, y)).collect();
    for i in -20..=20 {
        for j in -40..=40 {
            let p = P2::new(f64::from(i) * 0.5, f64::from(j) * 0.75);
            assert_eq!(region.contains(p), point_in_polygon(p, &polygon), "{p:?}");
        }
    }
    assert!(Region::Full.contains(P2::new(1e9, -1e9)));
}
