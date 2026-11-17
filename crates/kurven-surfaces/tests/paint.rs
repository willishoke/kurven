//! Paint.swift's suites: painting a parametric surface (`paintTests`).
//!
//! What the app's "Curves on" section does to a surface's request and
//! layers, without the window: a function painted on the plain torus, and a
//! request edit that carries a field's contours over by its class
//! (`FieldClass`). The window's own chain -- pick, paint, bake, undo -- is
//! held to the CLI by `scripts/check.sh`.

use kurven_core::hatching::FaceSelector;
use kurven_core::manifest::{ContourField, FieldClass, HeightPolicy, LayerRole, LayerSource, LayerSpec, Restriction};
use kurven_core::polyline::{Layer, PolylineSet};
use kurven_core::spaces::WorldSpace;
use kurven_core::surface_request::{Field, Shape, SurfaceRequest};
use kurven_landscape::torus::PeriodicTorus;
use kurven_surfaces::{SurfaceDrawing, SurfacePreset, SurfaceRequestExt};

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

fn build(request: &SurfaceRequest, previous: Option<&SurfaceDrawing>) -> SurfaceDrawing {
    SurfaceDrawing::build(request, previous, false).unwrap_or_else(|e| panic!("{request:?} builds: {e}"))
}

fn specs(d: &SurfaceDrawing) -> Vec<LayerSpec> {
    d.layers.iter().map(|l| l.spec.clone()).collect()
}

fn paths_of(layers: &[Layer]) -> Vec<&PolylineSet<WorldSpace>> {
    layers.iter().map(|l| &l.paths).collect()
}

fn paths<'a>(d: &'a SurfaceDrawing, name: &str) -> Option<&'a PolylineSet<WorldSpace>> {
    d.layers.iter().find(|l| l.spec.name == name).map(|l| &l.paths)
}

/// "paint: sn painted on the plain torus over its period rectangle draws surface sn"
#[test]
fn sn_painted_on_the_plain_torus_over_its_period_rectangle_draws_surface_sn() {
    let mut checks = Checks::default();
    let torus = SurfacePreset::named("torus").expect("the catalog's torus").request;
    let sn = SurfacePreset::named("sn").expect("the catalog's sn").request;
    let (Shape::Sn { modulus: m, .. }, Some(periods)) = (&sn.shape, sn.periods()) else {
        panic!("the catalog's sn has a modulus and a period rectangle");
    };
    let mut painted = torus.painting(&format!("sn(z, {m})"), periods).expect("sn paints on the torus");
    painted.lines = None;
    let built = build(&sn, None);
    let mut families = PeriodicTorus::contours(3.0, false);
    families.extend(specs(&built).into_iter().filter(|s| s.source == LayerSource::FoldCurves));
    let drawn = build(&painted, None).restyled(&families).expect("the restyle");
    checks.expect(
        specs(&drawn) == specs(&built)
            && paths_of(&drawn.layers) == paths_of(&built.layers)
            && drawn.surface.positions == built.surface.positions,
        format!(
            "its four families at the catalog's levels, and the folds, are surface sn's, curve for curve: {:?}",
            built.layers.iter().map(|l| l.paths.count()).collect::<Vec<_>>()
        ),
    );
    checks.expect(
        painted.shape
            == Shape::Periodic { expression: "sn(z, 0.64)".to_string(), periods, major: 2.0, minor: 1.0, grid: 600 }
            && painted.grid == torus.grid
            && painted.name.is_empty(),
        format!("the torus keeps its radii and grid, and f is sampled as the catalog's sn is: {:?}", painted.shape),
    );

    let refusal = |request: &SurfaceRequest, expression: &str, p: (f64, f64)| -> Option<String> {
        request.painting(expression, p).err().map(|e| e.to_string())
    };
    let forced = SurfacePreset::named("forced").expect("the catalog's forced torus").request;
    let refusals = [
        refusal(&torus, "sn(z,", periods),
        refusal(&torus, "frob(z)", periods),
        refusal(&torus, "  ", periods),
        refusal(&torus, "sn(z, 0.64)", (-1.0, 2.0)),
        refusal(&torus, "sn(z, 0.64)", (1.0, f64::INFINITY)),
        refusal(&forced, "sn(z, 0.64)", periods),
    ];
    let says = |i: usize, words: &str| refusals[i].as_ref().is_some_and(|r| r.contains(words));
    checks.expect(
        says(0, "stops in the middle")
            && says(1, "no 'frob'")
            && refusals[2].as_deref() == Some("the expression is empty")
            && says(3, "a period must be a positive")
            && refusals[4].is_some()
            && says(5, "forced torus"),
        format!(
            "an expression that does not parse, a period that is not positive, a forced torus: refused, saying why: \
             {refusals:?}"
        ),
    );
    checks.done();
}

/// A contour of `field` at `levels` on the surface's one face.
fn contour(field: ContourField, levels: &[f64]) -> LayerSpec {
    LayerSpec::new(
        field.raw(),
        LayerRole::Scaffold,
        LayerSource::Contour {
            field,
            levels: levels.to_vec(),
            restriction: Restriction::All,
            faces: FaceSelector::Surface,
            light: None,
        },
        0.2,
        HeightPolicy::Surface,
    )
}

/// "paint: a request edit carries a coordinate's contours over, and contours the height again"
#[test]
fn a_request_edit_carries_a_coordinates_contours_over_and_contours_the_height_again() {
    let mut checks = Checks::default();
    // The plain torus with a contour of each class of field the plain
    // routine draws: its rectangle keeps its samples when the radii move.
    let request = SurfaceRequest {
        grid: 128,
        lines: Some((12, 6)),
        name: "torus".to_string(),
        ..SurfaceRequest::new(Shape::Torus { major: 2.0, minor: 1.0 })
    };
    let first = build(&request, None);
    let mut kept = specs(&first);
    kept.extend([
        contour(ContourField::U, &[0.5, 2.0]),
        contour(ContourField::V, &[1.0, 4.0]),
        contour(ContourField::Height, &[0.25]),
    ]);
    let drawn = first.restyled(&kept).expect("the restyle");
    let mut wider = request.clone();
    wider.set(Field::Major, 2.5);
    wider.set(Field::Minor, 1.25);
    let edited = build(&wider, Some(&drawn)).keeping(&kept, Some(&drawn)).expect("the edit keeps the layers");
    let fresh = build(&wider, None).restyled(&kept).expect("the restyle");
    checks.expect(
        specs(&edited) == specs(&fresh) && paths_of(&edited.layers) == paths_of(&fresh.layers),
        "every layer is what the edited request draws afresh, curve for curve",
    );
    checks.expect(
        paths(&edited, "u").map(|p| &p.coords) == paths(&drawn, "u").map(|p| &p.coords)
            && paths(&edited, "u").map(|p| &p.vertices) != paths(&drawn, "u").map(|p| &p.vertices)
            && paths(&edited, "height").map(|p| &p.coords) != paths(&drawn, "height").map(|p| &p.coords),
        "u keeps its parameter coordinates and moves in space; the height's level set is a new one",
    );
    let mut fewer = request.clone();
    fewer.lines = Some((6, 3));
    let restyled_only = build(&fewer, Some(&drawn)).keeping(&kept, Some(&drawn)).expect("the edit keeps the layers");
    let fewer_fresh = build(&fewer, None).keeping(&kept, None).expect("the layers kept");
    checks.expect(
        restyled_only.content == drawn.content
            && paths(&restyled_only, "height") == paths(&drawn, "height")
            && paths_of(&restyled_only.layers) == paths_of(&fewer_fresh.layers),
        "an edit that keeps the surface keeps the height's contours, and draws what it draws afresh",
    );
    checks.expect(
        ContourField::ALL.iter().map(|f| f.field_class()).collect::<Vec<_>>()
            == [
                FieldClass::Function,
                FieldClass::Function,
                FieldClass::Coordinate,
                FieldClass::Coordinate,
                FieldClass::Embedding,
                FieldClass::Embedding,
            ],
        "|f| and arg f are the function's, u and v coordinates, the height and an isophote the embedding's",
    );

    // sn's torus sizes its tube's samples by its radii, so a radius
    // moves its rectangle's samples and a coordinate is contoured again.
    let sn = SurfaceRequest {
        grid: 128,
        name: "sn".to_string(),
        ..SurfaceRequest::new(Shape::Sn { modulus: 0.64, major: 2.0, minor: 1.0, grid: 120, conformal: false })
    };
    let sn_first = build(&sn, None);
    let mut sn_kept = specs(&sn_first);
    sn_kept.push(contour(ContourField::V, &[1.0, 2.0]));
    let sn_drawn = sn_first.restyled(&sn_kept).expect("the restyle");
    let mut fatter = sn.clone();
    fatter.set(Field::Minor, 1.4);
    let sn_edited =
        build(&fatter, Some(&sn_drawn)).keeping(&sn_kept, Some(&sn_drawn)).expect("the edit keeps the layers");
    let sn_fresh = build(&fatter, None).restyled(&sn_kept).expect("the restyle");
    checks.expect(
        sn_edited.surface.v != sn_drawn.surface.v && paths_of(&sn_edited.layers) == paths_of(&sn_fresh.layers),
        "where the rectangle's samples move, v is contoured again: what the edit draws afresh",
    );
    checks.done();
}
