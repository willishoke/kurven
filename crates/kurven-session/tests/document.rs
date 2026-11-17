//! The document's edits and their undo, on the session gate's inputs
//! (`crates/fixtures/swift/session/inputs`, written by Swift's kurven-cli).
//! What the edits make is held to Swift's app by the gate itself
//! (`scripts/check-web.sh`, "the session's edit scripts"); these hold the
//! document's own rules: one undo step an edit, a refusal leaves none, the
//! two speeds of edit, the git sha an undo puts back.

use std::path::{Path, PathBuf};

use kurven_core::bundle::Source;
use kurven_core::manifest::{LayerSource, Truncation};
use kurven_core::surface_request::Field;
use kurven_session::Document;
use kurven_session::sliders::Slider;

fn input(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/swift/session/inputs").join(format!("{name}.kurven"))
}

fn open(name: &str) -> Document {
    let mut d = Document::new();
    d.load_from(&Source::Directory(&input(name)));
    d.settle();
    assert!(d.scene().is_some(), "{name} opens");
    d
}

#[test]
fn a_truncation_is_one_undo_step_and_undoing_it_puts_back_the_bundles_styling() {
    let mut d = open("gamma");
    let before = d.edited().unwrap();
    d.edit_truncation(Truncation::Uniform(3.0), "Truncation");
    d.settle();
    let after = d.edited().unwrap();
    assert_eq!(after.truncation, Some(Truncation::Uniform(3.0)));
    assert_eq!(d.undo_stack().undo_name(), Some("Truncation"));
    // Two speeds: a truncation samples nothing, so the bundle keeps its git
    // sha, and the grids are the ones opened.
    assert_eq!(after.git_sha, before.git_sha);
    assert!(d.undo());
    d.settle();
    assert_eq!(d.edited().unwrap(), before);
    assert!(d.redo());
    d.settle();
    assert_eq!(d.edited().unwrap(), after);
    assert!(!d.redo());
}

#[test]
fn a_window_edit_samples_again_and_records_this_process_and_undo_puts_the_opened_sha_back() {
    let mut d = open("gamma");
    let opened = d.edited().unwrap().git_sha;
    let slider = Slider::Landscape { label: "rMax".to_string() };
    slider.set_landscape(&mut d, |r| r.domain.real.hi = 2.0);
    d.settle();
    assert_eq!(d.bundle().unwrap().manifest.domain.real.hi, 2.0);
    assert_eq!(d.edited().unwrap().git_sha, "native");
    assert_eq!(d.bundle().unwrap().manifest.provenance.git_sha, "native");
    d.undo();
    d.settle();
    assert_eq!(d.bundle().unwrap().manifest.domain.real.hi, 4.5);
    assert_eq!(d.bundle().unwrap().manifest.provenance.git_sha, opened);
}

#[test]
fn a_drag_is_one_step_and_samples_a_draft_until_it_is_let_go() {
    let mut d = open("gamma");
    d.draft_resolution = 40;
    let slider = Slider::Landscape { label: "rMin".to_string() };
    // Taken hold of: sampled at the draft resolution, its contours the grid's.
    slider.editing_changed(&mut d, true);
    d.settle();
    let width = d.bundle().unwrap().surface.height.width;
    assert!(width <= 40, "a draft is sampled at 40, not {width}");
    for lo in [-4.0, -3.5, -3.0] {
        slider.set_landscape(&mut d, |r| r.domain.real.lo = lo);
        d.settle();
    }
    // Let go: sampled in full, one undo step for the whole drag.
    slider.editing_changed(&mut d, false);
    d.settle();
    assert_eq!(d.bundle().unwrap().manifest.domain.real.lo, -3.0);
    assert!(d.bundle().unwrap().surface.height.width > 40);
    assert_eq!(d.undo_stack().steps(), 1);
}

#[test]
fn a_relief_past_what_the_surface_takes_is_refused_and_leaves_no_step() {
    let mut d = open("relief");
    let before = d.edited().unwrap();
    let slider = Slider::SurfaceField { field: Field::Relief, label: "Relief".to_string() };
    slider.set(&mut d, 7.0);
    d.settle();
    assert!(d.paint_status.is_some(), "the refusal is said");
    assert_eq!(d.edited().unwrap(), before);
    assert!(!d.undo_stack().can_undo());
}

#[test]
fn levels_on_a_surface_are_drawn_at_their_count_and_saved_at_it() {
    let mut d = open("sn");
    let slider = Slider::Levels { name: "mag_major".to_string() };
    slider.editing_changed(&mut d, true);
    slider.set(&mut d, 5.0);
    d.settle();
    slider.editing_changed(&mut d, false);
    d.settle();
    let saved = d.drawn_layers();
    let LayerSource::Contour { levels, .. } = &saved.iter().find(|l| l.name == "mag_major").unwrap().source else {
        panic!("a contour")
    };
    assert_eq!(levels.len(), 5);
    assert_eq!(d.undo_stack().undo_name(), Some("Levels of mag_major"));
}

#[test]
fn a_function_the_language_refuses_is_refused_and_changes_nothing() {
    let mut d = open("torus");
    let before = d.edited().unwrap();
    assert!(d.add_function("sin(", (6.0, 6.0)).is_err());
    d.settle();
    assert_eq!(d.edited().unwrap(), before);
    assert!(!d.undo_stack().can_undo());
}
