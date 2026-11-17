//! Bundles.swift's parametric-surface suites (`surfaceBundleTests`) and the
//! surface half of main.swift's "contract: bundles decode into consistent
//! values", with Swift's tolerances throughout. The polyhedron suites are
//! Phase 5's.
//!
//! A parametric surface's bundle stores the request that builds it, not
//! samples of it (`SurfaceManifest`): the map is code. So a bundle is held to
//! the request it keeps -- its bake is the bake of the drawing built straight
//! from the request, as `kurven-cli surface` builds it -- and its layers to
//! the drawing as it was kept, restyled or not.

mod common;

use common::par_map;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use kurven_bake::{BakeOptions, bake};
use kurven_core::bundle::KurvenBundle;
use kurven_core::camera::Camera;
use kurven_core::fixtures;
use kurven_core::hatching::{FaceSelector, Hatching};
use kurven_core::json::{self, JsonObject, JsonValue};
use kurven_core::manifest::{
    BundleManifest, ContourField, HeightPolicy, LayerRole, LayerSource, LayerSpec, Manifest, Restriction,
    SurfaceManifest,
};
use kurven_core::polyline::{Layer, PolylineSet};
use kurven_core::spaces::WorldSpace;
use kurven_core::surface_request::{Field, LinearCurves, Shape, SurfaceRequest, surface_key};
use kurven_core::svg;
use kurven_landscape::torus::{PeriodicOptions, PeriodicTorus};
use kurven_math::real;
use kurven_surfaces::{SurfaceBundle, SurfaceDrawing};

/// Swift's `Check.expect`s of one suite: each failure kept, and the suite
/// failed once at its end with all of them.
#[derive(Default)]
struct Checks(Vec<String>);

impl Checks {
    fn expect(&mut self, ok: bool, what: impl Into<String>) {
        if !ok {
            self.0.push(what.into());
        }
    }

    fn done(self) {
        assert!(self.0.is_empty(), "{}", self.0.join("\n"));
    }
}

/// A fresh directory under the system's temporary one, named with the
/// process id and `tag`, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("kurven-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        Scratch(dir)
    }

    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn build(request: &SurfaceRequest, previous: Option<&SurfaceDrawing>) -> SurfaceDrawing {
    SurfaceDrawing::build(request, previous, false).unwrap_or_else(|e| panic!("{request:?} builds: {e}"))
}

/// `SurfaceBundle(url:drawing:)`: the request's one preset, `gitSha` "native".
fn bundle(url: &Path, drawing: SurfaceDrawing) -> SurfaceBundle {
    SurfaceBundle::new(url, drawing, vec![SurfaceRequest::preset()], "native")
}

fn read(url: &Path) -> SurfaceBundle {
    SurfaceBundle::read(url).unwrap_or_else(|e| panic!("{} reads: {e}", url.display()))
}

fn specs(layers: &[Layer]) -> Vec<&LayerSpec> {
    layers.iter().map(|l| &l.spec).collect()
}

fn paths_of(layers: &[Layer]) -> Vec<&PolylineSet<WorldSpace>> {
    layers.iter().map(|l| &l.paths).collect()
}

fn paths<'a>(d: &'a SurfaceDrawing, name: &str) -> Option<&'a PolylineSet<WorldSpace>> {
    d.layers.iter().find(|l| l.spec.name == name).map(|l| &l.paths)
}

fn contour_source(field: ContourField, levels: &[f64], restriction: Restriction, faces: FaceSelector) -> LayerSource {
    LayerSource::Contour { field, levels: levels.to_vec(), restriction, faces, light: None }
}

fn svg_of(scene: &kurven_core::scene::Scene, options: &BakeOptions) -> String {
    let baked = bake(scene, options).expect("a bake");
    svg::render(&baked.strokes, svg::FIGURE_WIDTH, svg::MARGIN, svg::DECIMALS)
}

fn relief(amplitude: f64, repeats: (i64, i64), grid: i64, turn: f64, floor: Option<f64>, soft: bool) -> Shape {
    Shape::Relief { modulus: 0.64, major: 2.0, minor: 1.0, amplitude, repeats, grid, turn, floor, soft }
}

fn sn_shape(grid: i64) -> Shape {
    Shape::Sn { modulus: 0.64, major: 2.0, minor: 1.0, grid, conformal: false }
}

/// "bundles: a parametric surface's bundle bakes what its request bakes, for every surface"
#[test]
fn a_parametric_surfaces_bundle_bakes_what_its_request_bakes_for_every_surface() {
    let mut checks = Checks::default();
    // One request of each shape `kurven-cli surface` builds, coarser than
    // its defaults: the round trip, not the picture, is under test.
    let at = |shape: Shape, grid: i64| SurfaceRequest { grid, ..SurfaceRequest::new(shape) };
    let requests: Vec<(&str, SurfaceRequest)> = vec![
        (
            "torus",
            SurfaceRequest {
                lines: Some((36, 18)),
                linear_curves: Some(LinearCurves { slope: 0.4, turns: 24, count: 1 }),
                name: "torus".to_string(),
                ..at(Shape::Torus { major: 2.0, minor: 1.0 }, 256)
            },
        ),
        ("sn", SurfaceRequest { name: "sn".to_string(), ..at(sn_shape(200), 256) }),
        (
            "periodic",
            at(
                Shape::Periodic {
                    expression: "sn(z, 0.5)".to_string(),
                    periods: (7.416298709205487, 3.7081493546027433),
                    major: 2.0,
                    minor: 1.0,
                    grid: 200,
                },
                256,
            ),
        ),
        (
            "forced",
            SurfaceRequest {
                lines: Some((12, 6)),
                name: "forced".to_string(),
                ..at(
                    Shape::Forced {
                        system: "jerk".to_string(),
                        fit: 8000.0,
                        harmonics: 24,
                        radial: 0,
                        axial: 1,
                        radius: 2.5,
                        turns: 30,
                    },
                    256,
                )
            },
        ),
        ("relief", at(relief(0.4, (1, 1), 200, 0.0, None, false), 512)),
        ("relief, repeated, turned, floored and soft", at(relief(0.3, (2, 1), 160, 0.25, Some(0.5), true), 512)),
    ];
    let options = BakeOptions::new(1000);
    let directory = Scratch::new("bundles");
    // Each request built, baked, saved, reopened and baked again on its
    // own, independent of the others: side by side, the checks in order.
    let outcomes = par_map(&requests, |(label, request)| {
        let drawing = build(request, None);
        // The request's bake and the bundle's round trip, side by side.
        let scene = drawing.scene(Camera::oblique(&SurfaceRequest::PROJECTION));
        std::thread::scope(|scope| {
            let direct = scope.spawn(|| svg_of(&scene, &options));
            let url = directory.join(&format!("{label}.kurven"));
            bundle(&url, drawing).write(&url, None).unwrap_or_else(|e| panic!("{label}: written: {e}"));
            let text = std::fs::read_to_string(url.join("manifest.json")).expect("the manifest written");
            let read = read(&url);
            let from_bundle = svg_of(&read.scene(&read.manifest.presets[0]), &options);
            (direct.join().unwrap(), text, read, from_bundle)
        })
    });
    for ((label, request), (direct, text, read, from_bundle)) in requests.iter().zip(outcomes) {
        let strokes = direct.matches("<polyline").count();
        println!("{label}: {strokes} strokes");
        checks.expect(
            read.manifest.surface == *request
                && read.manifest.canonical_json() == text
                && strokes > 10
                && from_bundle == direct,
            format!(
                "{label}: saved, reopened and baked, the bundle bakes the request's SVG byte for byte: {strokes} \
                 strokes; request kept {}, manifest byte for byte {}, SVGs {} and {} bytes",
                read.manifest.surface == *request,
                read.manifest.canonical_json() == text,
                direct.len(),
                from_bundle.len()
            ),
        );
    }
    checks.done();
}

/// "bundles: open a parametric surface's bundle, restyle a layer, save it, and reopen it"
#[test]
fn open_a_parametric_surfaces_bundle_restyle_a_layer_save_it_and_reopen_it() {
    let mut checks = Checks::default();
    // The document's transitions on a surface, as the window makes them
    // (`Document.load`, a level slider, `saveBundle`), without the window.
    let directory = Scratch::new("restyle");
    let request = SurfaceRequest { grid: 256, name: "sn".to_string(), ..SurfaceRequest::new(sn_shape(200)) };
    let saved = directory.join("sn.kurven");
    bundle(&saved, build(&request, None)).write(&saved, None).expect("sn written");
    let opened = read(&saved);

    // Restyle: |f| at other levels and a thinner phase, and a contour of
    // v and of the height drawn on the surface besides.
    let mut layers = opened.manifest.layers.clone();
    let major = layers.iter().position(|l| l.name == "mag_major").expect("a layer mag_major");
    let minor = layers.iter().position(|l| l.name == "ang_minor").expect("a layer ang_minor");
    layers[major].source =
        contour_source(ContourField::Magnitude, &[0.5, 1.5], Restriction::All, FaceSelector::Surface);
    layers[minor].width = 0.1;
    layers.push(LayerSpec::new(
        "v",
        LayerRole::Scaffold,
        contour_source(ContourField::V, &[1.0, 2.0], Restriction::All, FaceSelector::Surface),
        0.2,
        HeightPolicy::Surface,
    ));
    layers.push(LayerSpec::new(
        "height",
        LayerRole::Scaffold,
        contour_source(ContourField::Height, &[0.5], Restriction::All, FaceSelector::Surface),
        0.2,
        HeightPolicy::Surface,
    ));
    let restyled = opened.drawing.restyled(&layers).expect("the restyle");
    let again = directory.join("restyled.kurven");
    opened.write(&again, Some(&layers)).expect("the restyled bundle written");
    let reopened = read(&again);

    checks.expect(
        restyled.content == opened.drawing.content && restyled.surface.positions == opened.drawing.surface.positions,
        "a restyle keeps the surface and its identity",
    );
    let fresh = PeriodicTorus::jacobi_sn(
        0.64,
        0.0,
        &PeriodicOptions {
            major: 2.0,
            minor: 1.0,
            resolution: 200,
            grid: 256,
            contours: Some(vec![layers[major].clone()]),
            ..PeriodicOptions::default()
        },
    )
    .expect("sn at those levels")
    .layers[0]
        .paths
        .clone();
    checks.expect(
        paths(&restyled, "mag_major") == Some(&fresh)
            && paths(&restyled, "mag_major") != paths(&opened.drawing, "mag_major"),
        "|f| at other levels is the function's contours there, placed as sn's own are",
    );
    checks.expect(
        paths(&restyled, "ang_minor") == paths(&opened.drawing, "ang_minor")
            && paths(&restyled, "ang_major") == paths(&opened.drawing, "ang_major"),
        "a layer restyled in width alone, and a layer not restyled, keep their curves",
    );
    let period_v = opened.drawing.surface.v.range.length();
    let on_v = paths(&restyled, "v").is_some_and(|p| {
        p.count() == 2
            && p.coords.as_ref().is_some_and(|coords| {
                coords.iter().all(|c| {
                    [1.0, 2.0].iter().any(|&l| (c.y() - l - period_v * ((c.y() - l) / period_v).round()).abs() < 1e-9)
                })
            })
    });
    let on_equator = paths(&restyled, "height")
        .is_some_and(|p| p.count() == 2 && p.vertices.iter().all(|v| (v.z() - 0.5).abs() < 1e-9));
    checks.expect(
        on_v && on_equator,
        "a contour of v at two levels is two coordinate circles, and of the height 0.5 two parallels",
    );
    checks.expect(
        reopened.manifest.layers == layers
            && reopened.manifest.surface == request
            && specs(&reopened.drawing.layers) == specs(&restyled.layers)
            && paths_of(&reopened.drawing.layers) == paths_of(&restyled.layers),
        "saved and reopened, the restyled bundle draws what was on screen",
    );

    // An edit of the request after a restyle is the drawing built from
    // nothing: what it reuses is what its request drew.
    let mut bigger = request.clone();
    bigger.set(Field::Major, 2.5);
    let edited = build(&bigger, Some(&reopened.drawing));
    let built = build(&bigger, None);
    checks.expect(
        specs(&edited.layers) == specs(&built.layers) && paths_of(&edited.layers) == paths_of(&built.layers),
        "a request edit after a restyle builds the request's drawing, as from nothing",
    );

    // A repeated relief restyled draws from the function as it was
    // sampled once, carried round the periods and through the relief:
    // what building it afresh with those layers draws.
    let relief_request =
        SurfaceRequest { grid: 256, ..SurfaceRequest::new(relief(0.3, (2, 1), 120, 0.25, None, false)) };
    let base = build(&relief_request, None);
    let mut relief_layers: Vec<LayerSpec> = base.layers.iter().map(|l| l.spec.clone()).collect();
    let mag = relief_layers.iter().position(|l| l.name == "mag_major").expect("a layer mag_major");
    relief_layers[mag].source =
        contour_source(ContourField::Magnitude, &[0.75, 1.25], Restriction::All, FaceSelector::Surface);
    let wanted = PeriodicTorus::jacobi_sn(
        0.64,
        0.25,
        &PeriodicOptions {
            major: 2.0,
            minor: 1.0,
            resolution: 120,
            grid: 256,
            truncated: true,
            contours: Some(vec![relief_layers[mag].clone()]),
            ..PeriodicOptions::default()
        },
    )
    .and_then(|d| d.repeated(2, 1, 256))
    .and_then(|d| d.relief(0.3, 1.0))
    .expect("the relief at those levels")
    .layers[0]
        .paths
        .clone();
    let restyled_relief = base.restyled(&relief_layers).expect("the relief restyled");
    checks.expect(
        paths(&restyled_relief, "mag_major") == Some(&wanted),
        "a repeated relief restyled is its contours at those levels, carried round and displaced",
    );
    checks.done();
}

/// "bundles: a request edit keeps the layers a bundle kept, and a slider's levels keep their range"
#[test]
fn a_request_edit_keeps_the_layers_a_bundle_kept_and_a_sliders_levels_keep_their_range() {
    let mut checks = Checks::default();
    // `Document`'s edit of a surface's request: built from the drawing
    // before it, with the layers its bundle kept (`SurfaceDrawing.keeping`).
    let opened = read(&fixtures::shared("contract/surface.kurven"));
    let kept = opened.manifest.layers.clone();
    let edit = |change: &dyn Fn(&mut SurfaceRequest)| -> SurfaceDrawing {
        let mut request = opened.manifest.surface.clone();
        change(&mut request);
        build(&request, Some(&opened.drawing)).keeping(&kept, None).expect("the edit keeps the layers")
    };
    let wider = edit(&|r| r.set(Field::Major, 2.5));
    let wider_specs: Vec<LayerSpec> = wider.layers.iter().map(|l| l.spec.clone()).collect();
    checks.expect(
        wider_specs == kept,
        "after the radii move, every layer is the one the bundle kept: the contours of u, v and the height, |f| and \
         arg f at the bundle's levels, the coordinate curves, the linear curve",
    );
    let fresh = build(&wider.request, None).restyled(&kept).expect("the restyle");
    checks.expect(
        paths_of(&wider.layers) == paths_of(&fresh.layers),
        "and draws what the edited request draws with those layers, built afresh",
    );
    let fewer = edit(&|r| {
        r.lines = Some((4, 2));
        r.linear_curves = None;
    });
    let names: Vec<&str> = fewer.layers.iter().map(|l| l.spec.name.as_str()).collect();
    checks.expect(
        fewer.layers.iter().find(|l| l.spec.name == "lines").map(|l| &l.spec.source)
            == Some(&LayerSource::CoordinateCurves { u: 4, v: 2 })
            && !names.contains(&"linearCurves")
            && names.contains(&"u")
            && names.contains(&"mag_major"),
        format!(
            "the coordinate curves and the linear curve are the request's: moved with it, and gone with it: {names:?}"
        ),
    );
    let added = edit(&|r| r.linear_curves = Some(LinearCurves { slope: 0.25, turns: 4, count: 2 }));
    checks.expect(
        added.layers.iter().find(|l| l.spec.name == "linearCurves").map(|l| &l.spec.source)
            == Some(&LayerSource::LinearCurves { slope: 0.25, turns: 4, count: 2 }),
        "a linear curve the request moves is drawn as it now says",
    );

    // The relief is a field of sn in relief alone: listed and set
    // there, and nil, listed nowhere and set to nothing, elsewhere.
    let mut relief_request = SurfaceRequest::new(relief(0.3, (1, 1), 120, 0.0, None, false));
    relief_request.set(Field::Relief, 0.4);
    let raised = SurfaceRequest::new(relief(0.4, (1, 1), 120, 0.0, None, false));
    let mut plain = SurfaceRequest::new(sn_shape(120));
    let before = plain.clone();
    plain.set(Field::Relief, 0.4);
    checks.expect(
        relief_request == raised
            && relief_request.fields().last() == Some(&Field::Relief)
            && plain == before
            && plain.get(Field::Relief).is_none()
            && !plain.fields().contains(&Field::Relief)
            && opened.manifest.surface.get(Field::Relief).is_none(),
        "the relief slider sets a relief's amplitude, and is no field of any other shape",
    );
    // A control's value is refused, not ignored or trapped on, where the
    // request would refuse it; −0 is set and saved as 0.
    let refused = |field: Field, value: f64, request: &SurfaceRequest| -> Option<String> {
        request.setting(field, value, &surface_key).err().map(|e| e.to_string())
    };
    let says = |r: Option<String>, words: &str| r.is_some_and(|r| r.contains(words));
    let zeroed = raised.setting(Field::Relief, -0.0, &surface_key).expect("−0 is an amplitude");
    let negative_zero = SurfaceRequest::new(relief(-0.0, (1, 1), 120, -0.0, None, false)).json().canonical();
    checks.expect(
        says(refused(Field::Relief, f64::NAN, &raised), "amplitude must be from 0 to 1, not nan")
            && says(refused(Field::Relief, 1.5, &raised), "amplitude must be from 0 to 1, not 1.5")
            && says(refused(Field::Modulus, 1.5, &raised), "modulus must lie strictly between 0 and 1")
            && says(refused(Field::Grid, 1e300, &raised), "must be 8 to 4096 samples")
            && says(refused(Field::Major, f64::INFINITY, &before), "major must be a finite number, not inf")
            && raised.setting(Field::Relief, 0.5, &surface_key).ok().and_then(|r| r.get(Field::Relief)) == Some(0.5)
            && zeroed.get(Field::Relief).is_some_and(|x| !real::sign_minus(x))
            && !negative_zero.contains("-0"),
        format!(
            "a value the request refuses is refused by name, a huge count with no trap, and −0 is 0: {:?}, {:?}, \
             {:?}, {:?}, {:?}; −0 set as {:?}; written {negative_zero}",
            refused(Field::Relief, f64::NAN, &raised),
            refused(Field::Relief, 1.5, &raised),
            refused(Field::Modulus, 1.5, &raised),
            refused(Field::Grid, 1e300, &raised),
            refused(Field::Major, f64::INFINITY, &before),
            zeroed.get(Field::Relief)
        ),
    );

    // A level slider resamples a family over its own range, and never to
    // fewer than two levels, so a family saved at them keeps its range.
    let family = LayerSpec::new(
        "mag",
        LayerRole::Magnitude,
        contour_source(ContourField::Magnitude, &[0.5, 1.0, 3.0], Restriction::All, FaceSelector::Surface),
        0.35,
        HeightPolicy::Surface,
    );
    let one = KurvenBundle::levels_resampled(&family, 1);
    let mut saved = family.clone();
    saved.source = contour_source(
        ContourField::Magnitude,
        &one.clone().unwrap_or_default(),
        Restriction::All,
        FaceSelector::Surface,
    );
    let six = KurvenBundle::levels_resampled(&saved, 6);
    checks.expect(
        one == Some(vec![0.5, 3.0])
            && six.as_ref().and_then(|l| l.first()) == Some(&0.5)
            && six.as_ref().and_then(|l| l.last()) == Some(&3.0),
        format!(
            "a slider at one level draws the family's two ends, and saved, the family keeps its range: {one:?}, {six:?}"
        ),
    );
    checks.expect(
        KurvenBundle::levels_resampled(&family, 6) == Some(vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
        format!("six levels are six even steps over the range: {:?}", KurvenBundle::levels_resampled(&family, 6)),
    );
    checks.done();
}

/// `.object([...])` with string keys.
fn obj(members: &[(&str, JsonValue)]) -> JsonObject {
    members.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

/// `dict.merging(other) { $1 }`.
fn merging(a: &JsonObject, b: &[(&str, JsonValue)]) -> JsonObject {
    let mut out = a.clone();
    out.extend(b.iter().map(|(k, v)| (k.to_string(), v.clone())));
    out
}

/// The fixture's request with another shape.
fn surface_shape(manifest: &JsonObject, shape: &JsonObject) -> JsonValue {
    let mut s = manifest.get("surface").and_then(JsonValue::as_object).cloned().unwrap_or_default();
    s.insert("shape".to_string(), JsonValue::Object(shape.clone()));
    JsonValue::Object(s)
}

/// "bundles: a malformed parametric surface's manifest is refused, saying why"
#[test]
fn a_malformed_parametric_surfaces_manifest_is_refused_saying_why() {
    let mut checks = Checks::default();
    let good: JsonObject = fixtures::shared_json("contract/surface.kurven/manifest.json")
        .as_object()
        .expect("the fixture is an object")
        .clone();
    checks.expect(SurfaceManifest::from_json(&JsonValue::Object(good.clone())).is_ok(), "the fixture itself is read");
    let mutated = |change: &dyn Fn(&mut JsonObject)| -> JsonValue {
        let mut d = good.clone();
        change(&mut d);
        JsonValue::Object(d)
    };
    let surface = |change: &dyn Fn(&mut JsonObject)| -> JsonValue {
        mutated(&|d| {
            let mut s = d.get("surface").and_then(JsonValue::as_object).cloned().unwrap_or_default();
            change(&mut s);
            d.insert("surface".to_string(), JsonValue::Object(s));
        })
    };
    let shape = |shape: JsonObject| -> JsonValue {
        surface(&|s| {
            s.insert("shape".to_string(), JsonValue::Object(shape.clone()));
        })
    };
    let layer = |source: LayerSource| -> JsonValue {
        let spec = LayerSpec::new("x", LayerRole::Scaffold, source, 0.2, HeightPolicy::Surface);
        mutated(&|d| {
            d.insert("layers".to_string(), JsonValue::Array(vec![spec.json()]));
        })
    };
    let mut refuses = |what: &str, words: &str, json: JsonValue| match BundleManifest::from_json(&json) {
        Ok(_) => checks.expect(false, format!("{what}: decoded")),
        Err(e) => {
            let e = e.to_string();
            checks.expect(e.contains(words), format!("{what}: {e}"));
        }
    };
    let (d, i, s) = (JsonValue::Double, JsonValue::Int, json::string);
    let sn = obj(&[("kind", s("sn")), ("modulus", d(0.64)), ("major", d(2.0)), ("minor", d(1.0)), ("grid", i(120))]);
    let torus = obj(&[("kind", s("torus")), ("major", d(2.0)), ("minor", d(1.0))]);
    let forced = obj(&[
        ("kind", s("forced")),
        ("system", s("jerk")),
        ("fit", d(8000.0)),
        ("harmonics", i(24)),
        ("radial", i(1)),
        ("axial", i(1)),
        ("radius", d(2.5)),
        ("turns", i(30)),
    ]);
    let relief = obj(&[
        ("kind", s("relief")),
        ("modulus", d(0.64)),
        ("major", d(2.0)),
        ("minor", d(1.0)),
        ("amplitude", d(0.4)),
        ("repeats", JsonValue::Array(vec![i(0), i(1)])),
        ("grid", i(120)),
        ("turn", d(0.0)),
        ("floor", JsonValue::Null),
        ("soft", JsonValue::Bool(false)),
    ]);
    let periodic = |expression: &str, periods: [f64; 2]| {
        obj(&[
            ("kind", s("periodic")),
            ("expression", s(expression)),
            ("periods", json::doubles(&periods)),
            ("major", d(2.0)),
            ("minor", d(1.0)),
            ("grid", i(120)),
        ])
    };
    let contour = |field: ContourField, restriction: Restriction, faces: FaceSelector| {
        contour_source(field, &[1.0], restriction, faces)
    };
    refuses(
        "refuses a modulus outside (0, 1)",
        "modulus must lie strictly between 0 and 1, not 1.5",
        shape(merging(&sn, &[("modulus", d(1.5))])),
    );
    refuses(
        "refuses a radius that is not positive",
        "surface.shape.major must be a positive number, not -1.0",
        shape(merging(&sn, &[("major", d(-1.0))])),
    );
    refuses(
        "refuses a grid too coarse to span its rectangle",
        "surface.grid must be 8 to 65536 samples, not 4",
        surface(&|s| {
            s.insert("grid".to_string(), i(4));
        }),
    );
    refuses(
        "refuses a shape it does not know, naming the ones it does",
        "'klein' is not one of",
        shape(obj(&[("kind", s("klein"))])),
    );
    refuses(
        "refuses a forced torus revolved on one state component twice",
        "surface.shape.radial and surface.shape.axial are both state component 1",
        shape(forced.clone()),
    );
    refuses(
        "refuses a period that is not positive",
        "surface.shape.periods must be a positive number, not 0.0",
        shape(periodic("sn(z, 0.5)", [0.0, 1.0])),
    );
    refuses("refuses an empty expression", "expression is empty", shape(periodic(" ", [1.0, 1.0])));
    refuses(
        "refuses a relief repeated less than once",
        "surface.shape.repeats must be 1 to 16 whole periods each way, not 0",
        shape(relief.clone()),
    );
    refuses(
        "refuses a negative count of coordinate curves",
        "surface.lines must be 0 to 16777216 coordinate curves, not -1",
        surface(&|s| {
            s.insert("lines".to_string(), JsonValue::Object(obj(&[("u", i(-1)), ("v", i(6))])));
        }),
    );
    refuses(
        "refuses a linear curve of no turns",
        "at least one turn of one strand",
        surface(&|s| {
            s.insert(
                "linearCurves".to_string(),
                JsonValue::Object(obj(&[("slope", d(0.4)), ("turns", i(0)), ("count", i(1))])),
            );
        }),
    );
    refuses(
        "refuses a negative stroke width",
        "surface.style.folds must be a number not below 0, not -1.0",
        surface(&|s| {
            let mut style = s.get("style").and_then(JsonValue::as_object).cloned().unwrap_or_default();
            style.insert("folds".to_string(), d(-1.0));
            s.insert("style".to_string(), JsonValue::Object(style));
        }),
    );
    refuses(
        "refuses a request with no style",
        "no key 'style'",
        surface(&|s| {
            s.remove("style");
        }),
    );
    refuses(
        "refuses hatching on a parametric surface along no coordinate",
        "runs along a coordinate",
        layer(LayerSource::Hatching { hatching: Hatching::plain(0.3, 0.1), faces: FaceSelector::Surface }),
    );
    refuses(
        "refuses wall edges on a parametric surface",
        "it is one face, with no walls",
        layer(LayerSource::WallEdges { edges: vec![0], base: 0.0 }),
    );
    refuses(
        "refuses a dumped layer on a parametric surface",
        "drawn from its request, not stored",
        layer(LayerSource::File { vertices: "v.npy".to_string(), offsets: "o.npy".to_string() }),
    );
    refuses(
        "refuses a contour on a face the surface does not have",
        "the surface is one face",
        layer(contour(ContourField::U, Restriction::All, FaceSelector::Graph { tiled: false })),
    );
    let magnitude = LayerSpec::new(
        "x",
        LayerRole::Scaffold,
        contour(ContourField::Magnitude, Restriction::All, FaceSelector::Surface),
        0.2,
        HeightPolicy::Surface,
    );
    refuses(
        "refuses |f| on a surface that paints no function",
        "paints no function",
        mutated(&|m| {
            m.insert("surface".to_string(), surface_shape(&good, &torus));
            m.insert("layers".to_string(), JsonValue::Array(vec![magnitude.json()]));
        }),
    );
    refuses(
        "refuses a trajectory on a torus no flow winds round",
        "no flow winds round",
        layer(LayerSource::Trajectory),
    );
    refuses(
        "refuses a contour kept below a truncation sn does not have",
        "only in relief",
        layer(contour_source(ContourField::Phase, &[0.0], Restriction::BelowTruncation, FaceSelector::Surface)),
    );
    refuses(
        "refuses a contour kept to a footprint the surface does not have",
        "the surface has neither",
        layer(contour(ContourField::U, Restriction::Footprint, FaceSelector::Surface)),
    );
    refuses(
        "refuses a manifest holding both geometries",
        "a manifest holds one geometry",
        mutated(&|m| {
            m.insert("domain".to_string(), JsonValue::Object(JsonObject::new()));
        }),
    );
    refuses(
        "refuses a parametric surface stating a landscape's axes",
        "with a landscape's axes",
        mutated(&|m| {
            m.insert("axes".to_string(), JsonValue::Array(vec![s("real"), s("imag"), s("magnitude")]));
        }),
    );
    refuses(
        "refuses a surface that is no request",
        "SurfaceRequest is not an object",
        mutated(&|m| {
            m.insert("surface".to_string(), JsonValue::Null);
        }),
    );
    refuses(
        "reads a manifest with neither geometry as a landscape's, and refuses it",
        "no key 'axes'",
        mutated(&|m| {
            m.remove("surface");
            m.remove("layers");
        }),
    );
    refuses(
        "refuses two layers of one name: a layer is kept across an edit by its name",
        "two layers named 'mag_major'",
        mutated(&|m| {
            if let Some(JsonValue::Array(layers)) = m.get("layers").cloned() {
                let mut twice = layers.clone();
                twice.push(layers[0].clone());
                m.insert("layers".to_string(), JsonValue::Array(twice));
            }
        }),
    );

    // Nothing a small bundle says traps or costs without bound: every
    // count that drives work is bounded, and a number that is not an
    // integer an `Int` holds is refused, not trapped on.
    refuses(
        "refuses a grid that is no integer an Int holds",
        "'grid' in SurfaceRequest should be an integer",
        surface(&|s| {
            s.insert("grid".to_string(), d(1e30));
        }),
    );
    refuses(
        "refuses a fractional grid",
        "'grid' in SurfaceRequest should be an integer",
        surface(&|s| {
            s.insert("grid".to_string(), d(64.5));
        }),
    );
    refuses(
        "refuses a grid of no samples",
        "surface.grid must be 8 to 65536 samples, not 0",
        surface(&|s| {
            s.insert("grid".to_string(), i(0));
        }),
    );
    refuses(
        "refuses a negative function grid",
        "surface.shape.grid must be 8 to 4096 samples, not -5",
        shape(merging(&sn, &[("grid", i(-5))])),
    );
    refuses(
        "refuses a function grid finer than it holds",
        "surface.shape.grid must be 8 to 4096",
        shape(merging(&sn, &[("grid", i(5000))])),
    );
    refuses(
        "refuses a surface of more samples than it holds",
        "more than the 33554432 a surface holds",
        mutated(&|m| {
            let mut surface = surface_shape(&good, &torus);
            if let JsonValue::Object(s) = &mut surface {
                s.insert("grid".to_string(), i(16384));
            }
            m.insert("surface".to_string(), surface);
            m.insert("layers".to_string(), JsonValue::Array(Vec::new()));
        }),
    );
    refuses(
        "refuses a relief repeated more than it holds",
        "must be 1 to 16 whole periods",
        shape(merging(&relief, &[("repeats", JsonValue::Array(vec![i(17), i(1)]))])),
    );
    // Swift's `\(floor)` and `\(amplitude)`: a Double's description.
    for (floor, written) in [(-1.0, "-1.0"), (3.0, "3.0"), (4.0, "4.0"), (-1e300, "-1e+300")] {
        refuses(
            &format!("refuses a relief floored at {written}"),
            &format!("surface.shape.floor must be at least 0 and below the truncation height 3.0, not {written}"),
            shape(merging(&relief, &[("repeats", JsonValue::Array(vec![i(1), i(1)])), ("floor", d(floor))])),
        );
    }
    for (amplitude, written) in [(-0.1, "-0.1"), (1.5, "1.5"), (f64::INFINITY, "inf"), (f64::NAN, "nan")] {
        refuses(
            &format!("refuses a relief of amplitude {written}"),
            &format!("surface.shape.amplitude must be from 0 to 1, not {written}"),
            shape(merging(&relief, &[("repeats", JsonValue::Array(vec![i(1), i(1)])), ("amplitude", d(amplitude))])),
        );
    }
    refuses(
        "refuses a fit longer than a run holds",
        "surface.shape.fit must be at most 100000",
        shape(merging(&forced, &[("radial", i(0)), ("fit", d(1e6))])),
    );
    refuses(
        "refuses more harmonics than a fit holds",
        "surface.shape.harmonics must be 1 to 64",
        shape(merging(&forced, &[("radial", i(0)), ("harmonics", i(100))])),
    );
    let long = merging(&forced, &[("radial", i(0)), ("turns", i(1_000_000))]);
    refuses(
        "refuses a trajectory of more vertices than a layer holds",
        "vertices for the trajectory",
        mutated(&|m| {
            m.insert("surface".to_string(), surface_shape(&good, &long));
            m.insert("layers".to_string(), JsonValue::Array(Vec::new()));
        }),
    );
    refuses(
        "refuses coordinate curves of more vertices than a layer holds",
        "vertices for 100000 + 6 coordinate curves",
        surface(&|s| {
            s.insert("lines".to_string(), JsonValue::Object(obj(&[("u", i(100_000)), ("v", i(6))])));
        }),
    );
    refuses(
        "refuses a coordinate-curve layer of a negative count",
        "coordinate curves must be counts",
        layer(LayerSource::CoordinateCurves { u: -1, v: 6 }),
    );
    refuses(
        "refuses a coordinate-curve layer of more vertices than a layer holds",
        "a layer that would take",
        layer(LayerSource::CoordinateCurves { u: 100_000, v: 6 }),
    );
    refuses(
        "refuses a linear-curve layer of no turns",
        "at least one turn of one strand",
        layer(LayerSource::LinearCurves { slope: 0.4, turns: 0, count: 1 }),
    );
    refuses(
        "refuses a linear-curve layer of no strands",
        "at least one turn of one strand",
        layer(LayerSource::LinearCurves { slope: 0.4, turns: 3, count: 0 }),
    );
    refuses(
        "refuses a linear curve too steep for a layer to hold",
        "vertices, more than the 16777216",
        layer(LayerSource::LinearCurves { slope: 1e9, turns: 1, count: 1 }),
    );
    refuses(
        "refuses a contour of u kept below a truncation",
        "only |f| and arg f are truncated",
        layer(contour(ContourField::U, Restriction::BelowTruncation, FaceSelector::Surface)),
    );
    refuses(
        "refuses schema 2",
        "schema 2 is not the supported 3",
        mutated(&|m| {
            m.insert("schema".to_string(), i(2));
        }),
    );
    let text = std::fs::read_to_string(fixtures::shared("contract/surface.kurven/manifest.json"))
        .expect("the fixture's manifest");
    for (what, number) in [("NaN", "NaN"), ("a number past a double", "1e400")] {
        let raw = text.replace("\"major\":2.0", &format!("\"major\":{number}"));
        checks.expect(
            raw != text && json::parse(&raw).ok().is_none_or(|j| BundleManifest::from_json(&j).is_err()),
            format!("refuses {what} for a radius"),
        );
    }
    checks.expect(
        Manifest::from_json(&JsonValue::Object(good.clone())).is_err(),
        "a landscape's reader refuses a parametric surface, naming its reader",
    );

    // What only building finds is refused when the bundle is read.
    let directory = Scratch::new("refusals");
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let mut unreadable = |what: &str, words: &str, json: JsonValue| {
        let url = directory.join(&format!("{}.kurven", NEXT.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(&url).expect("a bundle directory");
        std::fs::write(url.join("manifest.json"), json.canonical() + "\n").expect("a manifest written");
        match SurfaceBundle::read(&url) {
            Ok(_) => checks.expect(false, format!("{what}: read")),
            Err(e) => {
                let e = e.to_string();
                checks.expect(e.contains(words), format!("{what}: {e}"));
            }
        }
    };
    unreadable(
        "refuses a surface with no camera preset to open or bake at",
        "no camera preset",
        mutated(&|m| {
            m.insert("presets".to_string(), JsonValue::Array(Vec::new()));
        }),
    );
    let lorenz = merging(&forced, &[("system", s("lorenz")), ("radial", i(0))]);
    unreadable(
        "refuses a forced system the catalog does not have, by name",
        "unknown system 'lorenz'",
        mutated(&|m| {
            m.insert("surface".to_string(), surface_shape(&good, &lorenz));
            m.insert("layers".to_string(), JsonValue::Array(Vec::new()));
        }),
    );
    let broken = periodic("sn(z,", [1.0, 1.0]);
    unreadable(
        "refuses an expression that does not parse",
        "expression",
        mutated(&|m| {
            m.insert("surface".to_string(), surface_shape(&good, &broken));
        }),
    );
    checks.done();
}

/// "contract: bundles decode into consistent values", the parametric
/// surfaces' half: every contract bundle holding a `surface`, its request
/// built and every layer drawn on it. (The landscapes' half is in
/// kurven-core's and kurven-landscape's suites; the polyhedra's is Phase 5's.)
#[test]
fn contract_surface_bundles_decode_into_consistent_values() {
    let mut checks = Checks::default();
    let mut bundles: Vec<PathBuf> = std::fs::read_dir(fixtures::shared("contract"))
        .expect("the contract fixtures")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "kurven"))
        .collect();
    bundles.sort();
    let mut surfaces = 0;
    for url in &bundles {
        if !SurfaceBundle::holds_surface(url).unwrap_or_else(|e| panic!("{}: {e}", url.display())) {
            continue;
        }
        surfaces += 1;
        let name = url.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        // A parametric surface's: its request built, and every layer
        // drawn on it.
        let bundle = read(url);
        let layers = &bundle.drawing.layers;
        let ok = layers.iter().map(|l| &l.spec).eq(bundle.manifest.layers.iter())
            && layers.iter().all(|l| {
                l.paths.offsets.last() == Some(&l.paths.vertices.len())
                    && l.paths.coords.as_ref().map_or(0, |c| c.len()) == l.paths.vertices.len()
            })
            && layers.iter().filter(|l| l.spec.source != LayerSource::FoldCurves).all(|l| l.paths.count() > 0);
        checks.expect(
            ok,
            format!(
                "{name} decodes consistently, every layer drawn on its surface: {:?}",
                layers.iter().map(|l| (&l.spec.name, l.paths.count())).collect::<Vec<_>>()
            ),
        );
    }
    checks.expect(surfaces > 0, "the contract set holds parametric surfaces");
    checks.done();
}
