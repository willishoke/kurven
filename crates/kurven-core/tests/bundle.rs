//! Reading and writing landscape bundles: the landscape half of `main.swift`'s
//! "contract: bundles decode into consistent values" (the surfaces' and the
//! polyhedra's halves are Phases 4 and 5), the native landscapes of
//! `tests/fixtures/expr`, and a written bundle read back.

use std::path::{Path, PathBuf};

use kurven_core::bundle::KurvenBundle;
use kurven_core::fixtures;
use kurven_core::json;
use kurven_core::manifest::{BundleManifest, Walls};
use kurven_core::scene::Scene;

fn bundles(dir: &str) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(fixtures::shared(dir))
        .expect("a fixture directory")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "kurven"))
        .collect();
    out.sort();
    out
}

fn is_landscape(url: &Path) -> bool {
    let v = json::parse(&std::fs::read_to_string(url.join("manifest.json")).unwrap()).unwrap();
    matches!(BundleManifest::from_json(&v).unwrap(), BundleManifest::Landscape(_))
}

fn consistent(bundle: &KurvenBundle) -> bool {
    let m = &bundle.manifest;
    let mut ok = bundle.surface.height.width as i64 == m.height.shape.1
        && bundle.surface.height.height as i64 == m.height.shape.0
        && bundle.surface.phase.is_none() == m.phase.is_none()
        && bundle.layers.len() == m.layers.len();
    for layer in &bundle.layers {
        ok = ok && layer.paths.offsets.last().copied().unwrap_or(0) == layer.paths.vertices.len();
        for i in 0..layer.paths.count() {
            ok = ok && layer.paths.path(i).len() >= 2;
        }
    }
    if let Walls::Mesh { .. } = m.occluder.walls {
        let walls = bundle.walls();
        ok = ok
            && !walls.is_empty()
            && walls.triangles.iter().all(|t| (t.iter().copied().max().unwrap() as usize) < walls.vertices.len());
    }
    ok
}

/// "contract: bundles decode into consistent values", its landscapes.
#[test]
fn landscape_bundles_decode_into_consistent_values() {
    let landscapes: Vec<PathBuf> = bundles("contract").into_iter().filter(|p| is_landscape(p)).collect();
    assert_eq!(landscapes.len(), 5, "five landscape bundles");
    for url in landscapes {
        let bundle = KurvenBundle::read(&url).unwrap_or_else(|e| panic!("{}: {e}", url.display()));
        assert!(consistent(&bundle), "{} decodes consistently", url.display());
    }
}

/// The native landscapes the reference drawings were baked from read, and
/// build their scenes, occluders and faces.
#[test]
fn the_native_landscapes_read_and_build_their_scenes() {
    let landscapes = bundles("expr");
    assert_eq!(landscapes.len(), 4, "gamma, cubic, typed, tower");
    for url in landscapes {
        let bundle = KurvenBundle::read(&url).unwrap_or_else(|e| panic!("{}: {e}", url.display()));
        assert!(consistent(&bundle), "{} decodes consistently", url.display());
        let scene = Scene::from_bundle(&bundle, &bundle.manifest.presets[0]);
        let faces = scene.geometry().faces();
        assert!(!faces.is_empty(), "{}: a landscape has faces", url.display());
        assert!(faces.windows(2).all(|w| w[0].id <= w[1].id), "{}: the faces come in FaceID's order", url.display());
        // The described contours of |f| and arg f come from the grids here,
        // as Swift draws them without a refiner.
        let drawn = bundle.layers.iter().filter(|l| l.paths.count() > 0).count();
        assert!(drawn > 0, "{}: some layer is drawn from the grids", url.display());
    }
}

/// A described bundle written and read back is the same bundle: its
/// manifest canonical, its grids to the bit, its layers drawn again alike.
#[test]
fn a_written_bundle_reads_back_the_same() {
    let bundle = KurvenBundle::read(fixtures::shared("contract/landscape.kurven")).unwrap();
    let dir = std::env::temp_dir().join(format!("kurven-bundle-{}", std::process::id()));
    let out = dir.join("landscape.kurven");
    bundle.write(&out).expect("a described bundle writes");
    let back = KurvenBundle::read(&out).expect("and reads back");
    assert_eq!(back.manifest, bundle.manifest, "the manifest");
    assert!(back.surface.height.values == bundle.surface.height.values, "the heights to the bit");
    assert_eq!(
        back.surface.phase.as_ref().map(|p| p.values.clone()),
        bundle.surface.phase.as_ref().map(|p| p.values.clone()),
        "the phases to the bit"
    );
    assert_eq!(back.layers, bundle.layers, "the layers drawn again alike");
    std::fs::remove_dir_all(&dir).ok();

    // A bundle with dumped layers is refused, not written without them.
    let dumped = KurvenBundle::read(fixtures::shared("contract/uniform_mesh.kurven")).unwrap();
    assert!(dumped.write(dir.join("dumped.kurven")).is_err(), "a bundle with dumped layers is refused");
    std::fs::remove_dir_all(&dir).ok();
}
