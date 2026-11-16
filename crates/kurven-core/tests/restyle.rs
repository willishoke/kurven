//! Restyling: the edit path a truncation slider takes. The port of
//! `main.swift`'s `restyle:` suite.

use kurven_core::bundle::KurvenBundle;
use kurven_core::fixtures;
use kurven_core::manifest::Truncation;

/// The highest vertex of a layer's curves, or 0.
fn height(b: &KurvenBundle, name: &str) -> f64 {
    b.layer(name).ok().map(|l| l.paths.vertices.iter().map(|v| v.z()).fold(f64::NEG_INFINITY, f64::max)).unwrap_or(0.0)
}

/// "restyle: a new truncation re-derives the curves and keeps the landscape",
/// less its wall hatch, which is Hatch.swift's (see the ignored test below).
#[test]
fn a_new_truncation_re_derives_the_curves_and_keeps_the_landscape() {
    let bundle = KurvenBundle::read(fixtures::shared("contract/landscape.kurven")).expect("the landscape fixture");
    let mut manifest = bundle.manifest.clone();
    manifest.truncation = Truncation::Uniform(0.9);
    let restyled = bundle.restyled(&manifest);

    assert!(
        restyled.surface.height.values == bundle.surface.height.values,
        "the samples are untouched -- this is a restyling, not a resampling"
    );
    assert_eq!(restyled.surface.truncation, Truncation::Uniform(0.9), "the truncation is the new one");
    let walls = restyled.walls();
    let top = walls.vertices.iter().map(|v| v.z()).fold(f64::NEG_INFINITY, f64::max);
    assert!(!walls.vertices.is_empty() && top <= 0.9 + 1e-9, "the walls stop at the new truncation: {top}");

    // A dumped layer cannot be re-derived, so restyling must leave it be
    // rather than quietly emptying it.
    let dumped = KurvenBundle::read(fixtures::shared("contract/uniform_mesh.kurven")).expect("the mesh fixture");
    let mut other = dumped.manifest.clone();
    other.truncation = Truncation::Uniform(1.0);
    let kept_a = dumped.layer("mag").unwrap().paths.count();
    let kept_b = dumped.restyled(&other).layer("mag").unwrap().paths.count();
    assert!(kept_a == kept_b && kept_a > 0, "a dumped layer survives a restyling unchanged: {kept_a} paths");
}

/// "restyle: a new truncation re-derives the curves and keeps the landscape",
/// its wall hatch.
#[test]
#[ignore = "PORT: phase 3 stub (Hatch.swift): the wall hatch derives nothing until Hatch.swift is ported"]
fn a_new_truncation_lowers_the_wall_hatch() {
    let bundle = KurvenBundle::read(fixtures::shared("contract/landscape.kurven")).expect("the landscape fixture");
    let mut manifest = bundle.manifest.clone();
    manifest.truncation = Truncation::Uniform(0.9);
    let restyled = bundle.restyled(&manifest);
    // Lowering the truncation lowers every wall top edge, so the wall hatch is
    // strictly shorter.
    assert!(height(&restyled, "wall_hatch") <= 0.9 + 1e-9, "the wall hatch stops at the new truncation");
    assert!(height(&bundle, "wall_hatch") > 0.9, "and reached higher before it");
}
