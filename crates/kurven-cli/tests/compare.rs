//! `compare` on the Swift reference drawings: each against itself is at
//! distance zero; moved by less than the tolerance it passes, with the
//! distance it was moved; moved by more, or with a stroke taken out, it
//! fails.

use kurven_cli::compare::{Tolerance, compare, compare_skipping};
use kurven_cli::svg::{self, Drawing};
use kurven_core::fixtures;

fn references() -> Vec<(String, Drawing)> {
    let index = fixtures::port_json("swift/index.json");
    let drawings = index.get("drawings").and_then(|d| d.as_array()).expect("index.json lists drawings");
    drawings
        .iter()
        .map(|entry| {
            let name = entry.get("name").and_then(|n| n.as_str()).expect("a name").to_string();
            let drawing = svg::read(fixtures::port(&format!("swift/{name}.svg"))).expect("a reference reads");
            (name, drawing)
        })
        .collect()
}

fn moved(d: &Drawing, dx: f64, dy: f64) -> Drawing {
    let mut d = d.clone();
    for layer in &mut d.layers {
        for path in &mut layer.paths {
            for p in path.iter_mut() {
                *p = [p[0] + dx, p[1] + dy];
            }
        }
    }
    d
}

#[test]
fn every_reference_against_itself_is_at_zero() {
    let all = references();
    assert_eq!(all.len(), 26);
    for (name, d) in &all {
        let report = compare(d, d, Tolerance::default());
        assert!(report.same(), "{name}: {report}");
        assert!(report.notes.is_empty(), "{name}");
        for layer in &report.layers {
            assert_eq!(layer.hausdorff.hi, 0.0, "{name}");
            assert_eq!(layer.relative_length(), 0.0, "{name}");
        }
    }
}

/// A translation by t moves every point by t, so the distance is at most t;
/// and the search must find no more than that, to within its eps.
#[test]
fn a_drawing_moved_less_than_the_tolerance_passes_and_more_fails() {
    let tolerance = Tolerance::default();
    let eps = tolerance.distance / 64.0;
    for (name, d) in references()
        .iter()
        .filter(|(n, _)| ["gallery-gamma", "sn", "solid-frame", "composite-three", "slope-lines"].contains(&n.as_str()))
    {
        let t = tolerance.distance / 2.0;
        let report = compare(d, &moved(d, 0.6 * t, 0.8 * t), tolerance);
        assert!(report.same(), "{name} moved by {t}: {report}");
        for layer in &report.layers {
            assert!(layer.hausdorff.hi <= t + eps, "{name}: {:?} past the move {t}", layer.hausdorff);
            assert!(layer.hausdorff.lo > 0.0, "{name}: a moved layer at distance 0");
        }
        let far = 3.0 * tolerance.distance;
        let report = compare(d, &moved(d, far, 0.0), tolerance);
        assert!(!report.same(), "{name} moved by {far} passed");
    }
}

/// A missing run is the failure the gate exists for. The longest stroke of a
/// polyhedron's edges taken out: its midpoint is far from every other edge.
#[test]
fn a_drawing_with_a_stroke_taken_out_fails() {
    for (name, d) in references().iter().filter(|(n, _)| n.starts_with("solid-") || n == "landscape-cubic") {
        let mut cut = d.clone();
        let layer = &mut cut.layers[0];
        let longest = (0..layer.paths.len())
            .max_by(|&i, &j| {
                kurven_cli::compare::length(&layer.paths[i..=i])
                    .partial_cmp(&kurven_cli::compare::length(&layer.paths[j..=j]))
                    .expect("a stroke's length is a number")
            })
            .unwrap();
        layer.paths.remove(longest);
        let report = compare(d, &cut, Tolerance::default());
        assert!(!report.same(), "{name} without its longest stroke passed: {report}");
        assert_eq!(report.notes.len(), 1, "{name}: the count difference is noted");
    }
}

/// A different number of layers, or a layer of another width, fails before
/// any distance is measured.
#[test]
fn layers_must_match() {
    let all = references();
    let (_, d) = all.iter().find(|(n, _)| n == "gallery-gamma").unwrap();
    let mut fewer = d.clone();
    fewer.layers.pop();
    assert!(!compare(d, &fewer, Tolerance::default()).same());
    let mut wider = d.clone();
    wider.layers[0].stroke_width *= 1.01;
    assert!(!compare(d, &wider, Tolerance::default()).same());
}

/// A skipped layer is measured and reported but left out of the verdict;
/// every other layer is still held, and so are the skipped layer's colour
/// and width.
#[test]
fn a_skipped_layer_is_reported_and_not_held() {
    let all = references();
    let (_, d) = all.iter().find(|(n, _)| n == "landscape-cubic").expect("landscape-cubic");
    let mut cut = d.clone();
    cut.layers[1].paths.remove(0);
    let report = compare_skipping(d, &cut, Tolerance::default(), &[1]);
    assert!(report.same(), "{report}");
    assert!(report.notes.iter().any(|n| n.contains("(skipped)")), "{report}");
    assert!(report.notes.iter().any(|n| n.contains("strokes against")), "{report}");
    cut.layers[2].paths.remove(0);
    assert!(!compare_skipping(d, &cut, Tolerance::default(), &[1]).same());
    let mut recoloured = d.clone();
    recoloured.layers[1].color = "#ff0000".to_string();
    assert!(!compare_skipping(d, &recoloured, Tolerance::default(), &[1]).same());
}
