//! The landscape's derived numbers, the catalog and the sampling, held to
//! the Swift suites and to the committed bundles of `tests/fixtures/expr`.

use kurven_core::bundle::KurvenBundle;
use kurven_core::fixtures;
use kurven_core::grid::Grid2D;
use kurven_core::json::{JsonValue, Members};
use kurven_core::manifest::{Domain, Interval, Manifest, Truncation, Walls};
use kurven_landscape::{Catalog, LandscapeRequest, LandscapeStyle, LanguageFunction, NativeLandscape};
use kurven_math::Expression;

/// `Expressions.swift`'s "landscape: the derived numbers".
#[test]
fn the_derived_numbers() {
    // A truncation is chosen only when there is a spire to cut.
    let flat: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
    assert_eq!(NativeLandscape::default_truncation(&flat), Truncation::None, "a bounded function is not truncated");
    let mut spire = flat.clone();
    spire[500] = 400.0;
    match NativeLandscape::default_truncation(&spire) {
        Truncation::Uniform(z) => assert!((z - 1.0).abs() < 1e-12, "a spire is cut near the 99th percentile: {z}"),
        t => panic!("a spire is cut near the 99th percentile: {t:?}"),
    }
    let domain = Domain::new(Interval::new(-2.0, 2.0), Interval::new(-1.0, 1.0));
    let shape = NativeLandscape::grid_shape(domain, 100).unwrap();
    assert!(shape.n_real == 100 && shape.n_imag == 50, "the grid keeps its cells square: {shape:?}");
    let phase = NativeLandscape::phase_minor();
    assert!(
        phase.len() == 8 && (phase[0] + 2.6179938779914944).abs() < 1e-15,
        "the minor phase levels are numpy's linspace, bit for bit: {}",
        phase[0]
    );
}

/// `main.swift`'s "landscape: the derived numbers are the committed ones".
#[test]
fn the_derived_numbers_are_the_committed_ones() {
    let index = fixtures::shared_json("hatch/index.json");
    let style = index.object("hatch").unwrap().value("style", "hatch").unwrap().object("hatch.style").unwrap();
    let mut nice = 0;
    for n in style.array("nice", "style").unwrap() {
        let n = n.object("style.nice").unwrap();
        let x = n.double("x", "style.nice").unwrap();
        let up = n.double("up", "style.nice").unwrap();
        let down = n.double("down", "style.nice").unwrap();
        assert_eq!(LandscapeStyle::nice(x, false), up, "nice({x})");
        assert_eq!(LandscapeStyle::nice(x, true), down, "nice({x}, down)");
        nice += 1;
    }
    assert!(nice > 0);
    for l in style.array("levels", "style").unwrap() {
        let l = l.object("style.levels").unwrap();
        let ceiling = l.double("ceiling", "style.levels").unwrap();
        let (major, minor) = LandscapeStyle::magnitude_levels(ceiling);
        let want_major = l.doubles("major", "style.levels").unwrap();
        let want_minor = l.doubles("minor", "style.levels").unwrap();
        let same = major.len() == want_major.len()
            && minor.len() == want_minor.len()
            && major.iter().zip(&want_major).all(|(a, b)| (a - b).abs() < 1e-9)
            && minor.iter().zip(&want_minor).all(|(a, b)| (a - b).abs() < 1e-9);
        assert!(
            same,
            "at ceiling {ceiling}: {}/{} vs {}/{}",
            major.len(),
            minor.len(),
            want_major.len(),
            want_minor.len()
        );
    }
}

/// `Expressions.swift`'s "expression: the language table is the frozen one",
/// read through the catalog as Swift reads it.
#[test]
fn the_language_table_is_the_frozen_one() {
    let index = fixtures::shared_json("expr/language.json");
    let language = index.object("language").unwrap().value("language", "language").unwrap().object("language").unwrap();
    let functions: Vec<LanguageFunction> = language
        .array("functions", "language")
        .unwrap()
        .iter()
        .map(|f| LanguageFunction::from_json(f).unwrap())
        .collect();
    let mine = &Catalog::native().functions;
    let names = |fs: &[LanguageFunction]| fs.iter().map(|f| f.name.clone()).collect::<Vec<_>>();
    assert_eq!(names(&functions), names(mine), "the same functions, in the same order");
    for (theirs, ours) in functions.iter().zip(mine) {
        assert_eq!(theirs, ours, "with the same arities, aliases and help");
    }
    let constants: Vec<String> = language
        .array("constants", "language")
        .unwrap()
        .iter()
        .map(|c| c.object("constant").unwrap().string("name", "constant").unwrap())
        .collect();
    assert_eq!(constants, Catalog::native().constants, "and the same constants");
    assert_eq!(language.string("variable", "language").unwrap(), "z", "and z is the variable");
}

#[test]
fn the_catalog_is_kurven_landscapes() {
    let catalog = Catalog::native();
    let names: Vec<&str> = catalog.presets.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "rgamma", "gamma", "zeta", "cn", "tan", "sin", "exp_inv", "cubic", "moebius", "sqrt", "log", "erf",
            "besselj", "digamma"
        ]
    );
    assert_eq!(catalog.default_resolution, 600);
    assert_eq!(catalog.default_buffer, 4000);
    for preset in &catalog.presets {
        assert!(Expression::compile(&preset.expression).is_ok(), "{} compiles", preset.name);
        let again = kurven_landscape::FunctionPreset::from_json(&preset.json()).unwrap();
        assert_eq!(&again, preset, "{} round trips", preset.name);
    }
    let tan = catalog.preset("tan").unwrap();
    assert_eq!(tan.truncation(), Truncation::Uniform(5.0));
    assert_eq!(catalog.preset("sqrt").unwrap().truncation(), Truncation::None);
    assert!(catalog.preset("nothing").is_none());
    let request = LandscapeRequest::from_preset(tan, 240);
    assert_eq!(request.name, "tan");
    assert_eq!(request.truncation, Some(Truncation::Uniform(5.0)));
    assert!(request.samples_as(&LandscapeRequest { truncation: None, ..request.clone() }));
    assert_eq!(LandscapeRequest::from_json(&request.json()).unwrap(), request, "a request round trips");
}

/// The parts of `NativeLandscape::build`, one by one, against the
/// committed bundles: each request parsed as Swift parses it, sampled and
/// quantized, against the committed grids at Swift's 1e-6 relative; and the
/// manifest's derived numbers (the grid's shape, the truncation, the walls'
/// perimeter, the occluder's step, the layers) against the committed
/// manifest. `built_here_it_is_the_committed_bundle` holds the whole.
#[test]
fn the_committed_landscapes_sample_and_derive_alike() {
    let index = fixtures::shared_json("expr/landscapes.json");
    let entries = index.object("landscapes").unwrap().array("landscapes", "landscapes").unwrap();
    assert_eq!(entries.len(), 4);
    for entry in entries {
        let e = entry.object("landscape").unwrap();
        let name = e.string("bundle", "landscape").unwrap();
        let spec = LandscapeRequest::from_json(e.value("spec", "landscape").unwrap()).unwrap();
        let text = std::fs::read_to_string(fixtures::shared(&format!("expr/{name}/manifest.json"))).unwrap();
        let theirs = Manifest::from_json(&kurven_core::json::parse(&text).unwrap()).unwrap();

        let compiled = Expression::compile(&spec.expression).unwrap();
        let samples = NativeLandscape::sample(&compiled, spec.domain, spec.resolution).unwrap();
        let shape = samples.shape;
        assert_eq!(theirs.height.shape, (shape.n_imag as i64, shape.n_real as i64), "{name}: the shape");
        assert_eq!(theirs.domain, spec.domain, "{name}: the domain");
        for (file, values) in [("height", &samples.magnitude), ("phase", &samples.phase)] {
            let ours = NativeLandscape::quantize(values);
            let want = fixtures::shared_npy(&format!("expr/{name}/{file}.npy")).floats().unwrap();
            assert_eq!(ours.len(), want.len(), "{name}: {file}'s count");
            let mut worst = 0.0f64;
            let mut off = 0;
            for (&x, &y) in ours.iter().zip(&want) {
                let err = f64::from((x - y).abs()) / f64::from(y.abs()).max(1.0);
                worst = worst.max(err);
                if err > 1e-6 {
                    off += 1;
                }
            }
            assert_eq!(off, 0, "{name}: the same {file} samples (worst {worst:.1e})");
        }

        let truncation =
            spec.truncation.clone().unwrap_or_else(|| NativeLandscape::default_truncation(&samples.magnitude));
        assert_eq!(theirs.truncation, truncation, "{name}: the truncation");
        let density = NativeLandscape::wall_density(spec.domain, shape).unwrap();
        let perimeter = NativeLandscape::rectangle(spec.domain, &density);
        assert_eq!(theirs.occluder.walls, Walls::Perimeter { perimeter, base: 0.0 }, "{name}: the walls");
        let step = 1.max(shape.n_real.max(shape.n_imag) as i64 / NativeLandscape::OCCLUDER_RESOLUTION);
        assert_eq!(theirs.occluder.step, step, "{name}: the occluder's step");
        let layers = spec.layers.clone().unwrap_or_else(|| {
            NativeLandscape::default_layers(
                spec.domain,
                &truncation,
                NativeLandscape::ceiling(&truncation, &samples.magnitude),
                spec.spacing,
                true,
            )
        });
        assert_eq!(theirs.layers, layers, "{name}: the layers");
        assert_eq!(theirs.layers.len() as i64, e.int("layers", "landscape").unwrap(), "{name}: the layer count");
        assert_eq!(theirs.presets[0].projection, NativeLandscape::PROJECTION, "{name}: the projection");
        let p = &theirs.provenance.params;
        assert_eq!(p.get("nReal"), Some(&JsonValue::Int(shape.n_real as i64)), "{name}: nReal");
        assert_eq!(p.get("nImag"), Some(&JsonValue::Int(shape.n_imag as i64)), "{name}: nImag");
        assert_eq!(theirs.provenance.function, compiled.expression(), "{name}: the canonical expression");
    }
}

/// `Expressions.swift`'s "landscape: built here, it is the committed
/// bundle": each committed bundle's request built here without a refiner,
/// under the committed git sha, against the bundle read from disk: the same
/// manifest, byte for byte; the same samples, to Swift's 1e-6 relative; and
/// every layer deriving the same curves, path for path and in length to 1e-6
/// relative. Both sides derive their layers here, so until Hatch.swift and
/// the fold curves land, the hatchings, the truncation edge and the wall
/// edges are the stub's on both.
#[test]
fn built_here_it_is_the_committed_bundle() {
    let index = fixtures::shared_json("expr/landscapes.json");
    let entries = index.object("landscapes").unwrap().array("landscapes", "landscapes").unwrap();
    assert_eq!(entries.len(), 4);
    for entry in entries {
        let e = entry.object("landscape").unwrap();
        let name = e.string("bundle", "landscape").unwrap();
        let spec = LandscapeRequest::from_json(e.value("spec", "landscape").unwrap()).unwrap();
        let theirs = KurvenBundle::read(fixtures::shared(&format!("expr/{name}"))).unwrap();
        let ours = NativeLandscape::build(&spec, &theirs.manifest.provenance.git_sha, false).unwrap();
        let label = format!("{name} ({})", spec.expression);

        let (a, b) = (ours.manifest.canonical_json(), theirs.manifest.canonical_json());
        let at = a.bytes().zip(b.bytes()).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
        assert!(a == b, "{label}: the same manifest; first difference at byte {at}: {:?} vs {:?}", &a[at..], &b[at..]);

        let compare = |a: Option<&Grid2D<f32>>, b: Option<&Grid2D<f32>>, what: &str| {
            let (Some(a), Some(b)) = (a, b) else {
                assert!(a.is_none() && b.is_none(), "{label}: {what}: both present or both absent");
                return;
            };
            assert_eq!(a.values.len(), b.values.len(), "{label}: {what}: the same shape");
            let (mut worst, mut off) = (0.0f64, 0);
            for (&x, &y) in a.values.iter().zip(&b.values) {
                let err = f64::from((x - y).abs()) / f64::from(y.abs()).max(1.0);
                worst = worst.max(err);
                if err > 1e-6 {
                    off += 1;
                }
            }
            assert_eq!(off, 0, "{label}: {what}: the same samples (worst {worst:.1e})");
        };
        compare(Some(&ours.surface.height), Some(&theirs.surface.height), "height");
        compare(ours.surface.phase.as_deref(), theirs.surface.phase.as_deref(), "phase");

        let mut curves_differ = Vec::new();
        assert_eq!(ours.layers.len(), theirs.layers.len(), "{label}: the same layers");
        for (mine, committed) in ours.layers.iter().zip(&theirs.layers) {
            let (a, b) = (mine.paths.total_length(), committed.paths.total_length());
            if mine.paths.count() != committed.paths.count() || (a - b).abs() > 1e-6 * b.max(1.0) {
                curves_differ.push(format!(
                    "{} {}/{} paths, length {a:.6}/{b:.6}",
                    mine.spec.name,
                    mine.paths.count(),
                    committed.paths.count()
                ));
            }
        }
        assert!(curves_differ.is_empty(), "{label}: and every layer derives to the same curves: {curves_differ:?}");
    }
}

#[test]
fn linspace_and_rounding_are_numpys_and_pythons() {
    assert_eq!(NativeLandscape::linspace(0.0, 1.0, 0), Vec::<f64>::new());
    assert_eq!(NativeLandscape::linspace(2.0, 3.0, 1), vec![2.0]);
    let l = NativeLandscape::linspace(-1.0, 1.0, 7);
    assert_eq!(l[6], 1.0);
    assert_eq!(l[1], -1.0 + 1.0 * (2.0 / 6.0));
    assert_eq!(NativeLandscape::py_round(2.5), Some(2));
    assert_eq!(NativeLandscape::py_round(3.5), Some(4));
    assert_eq!(NativeLandscape::py_round(f64::NAN), None, "where Swift's Int traps");
    let nan = Domain::new(Interval::new(f64::NAN, 1.0), Interval::new(0.0, 1.0));
    assert_eq!(NativeLandscape::grid_shape(nan, 100), None);
    let point = Domain::new(Interval::new(1.0, 1.0), Interval::new(0.0, 0.0));
    assert_eq!(NativeLandscape::grid_shape(point, 1).map(|s| (s.n_real, s.n_imag)), Some((2, 2)));
    // quantize keeps a NaN and never rounds past the float64 range.
    let q = NativeLandscape::quantize(&[0.1, f64::NAN, std::f64::consts::PI, -std::f64::consts::PI]);
    assert!(q[1].is_nan());
    assert!(f64::from(q[2]) <= std::f64::consts::PI && f64::from(q[3]) >= -std::f64::consts::PI);
}
