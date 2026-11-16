//! The composites Swift's `kurven-cli fixtures` writes under
//! `tests/fixtures/composite` (`GoldenFixtures.composites`), drawn by the
//! port's `scene` path (`composite_drawing`) and held to Swift's SVGs by
//! `compare` at the floor the reference drawings are held to: 1.5e-4 page
//! units of Hausdorff distance (the SVG's four decimals, on the diagonal)
//! and 1e-5 of each layer's length.

use kurven_cli::compare::{Tolerance, compare};
use kurven_cli::scene::{composite_drawing, placement};
use kurven_cli::svg;
use kurven_core::composite::Placement;
use kurven_core::fixtures;

fn held(name: &str, parts: &[(&str, Placement)]) {
    let bundles: Vec<(String, Placement)> =
        parts.iter().map(|(p, at)| (fixtures::shared(p).to_string_lossy().into_owned(), *at)).collect();
    let drawing = composite_drawing(&bundles, None, false).unwrap_or_else(|e| panic!("{name}: {e}"));
    let dir = std::env::temp_dir().join(format!("kurven-composite-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let ours = dir.join(format!("{name}.svg"));
    kurven_cli::commands::write_svg(&drawing.strokes, &ours).unwrap();
    let a = svg::read(fixtures::shared(&format!("composite/{name}.svg"))).unwrap();
    let b = svg::read(&ours).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let report = compare(&a, &b, Tolerance { distance: 1.5e-4, length: 1e-5 });
    assert!(report.same(), "composite {name} against Swift's:\n{report}");
}

#[test]
fn a_cube_behind_a_tower_is_swifts() {
    held(
        "cube",
        &[
            ("expr/landscape_cubic.kurven", Placement::IDENTITY),
            ("contract/polyhedron_cube.kurven", placement("-1.2,0.1,0.3,0.35").unwrap()),
        ],
    );
}

#[test]
fn a_frame_sunk_into_a_slope_is_swifts() {
    held(
        "frame",
        &[
            ("expr/landscape_cubic.kurven", Placement::IDENTITY),
            ("contract/polyhedron_frame.kurven", placement("0.3,-1.3,0.2,0.3").unwrap()),
        ],
    );
}

#[test]
fn a_torus_ringing_a_tower_is_swifts() {
    held(
        "torus",
        &[
            ("expr/landscape_cubic.kurven", Placement::IDENTITY),
            ("contract/surface_torus.kurven", placement("1,0,1.6,0.35").unwrap()),
        ],
    );
}

#[test]
fn the_cubic_with_a_cube_and_a_torus_is_swifts() {
    held(
        "three",
        &[
            ("expr/landscape_cubic.kurven", Placement::IDENTITY),
            ("contract/polyhedron_cube.kurven", placement("-1.2,0.1,0.3,0.35").unwrap()),
            ("contract/surface_torus.kurven", placement("1,0,1.6,0.35").unwrap()),
        ],
    );
}

#[test]
fn a_placement_reads_as_swift_reads_it() {
    assert_eq!(placement("1,2,3").unwrap(), Placement::new(1.0, kurven_core::simd::Vec3::new(1.0, 2.0, 3.0)));
    assert_eq!(placement("1,2,3,0.5").unwrap().scale, 0.5);
    assert_eq!(placement("1,2").unwrap_err().0, "--at wants X,Y,Z or X,Y,Z,S, got '1,2'");
    assert!(
        kurven_cli::scene::scene_command(&["a.kurven".to_string(), "-o".to_string(), "x.svg".to_string()])
            .unwrap_err()
            .0
            .contains("two bundles or more")
    );
}
