//! The suites of `Sources/kurven-test/Hatchings.swift` about a solid's
//! hatching, with Swift's tolerances. (Its suites on Turing's pattern and
//! on decoding are `kurven-core/tests/hatch.rs`.)
//!
//! Hatching on a solid is a set of curves on its faces: laid over each face
//! in closed form, ended on the tone's own level set by bisection, and
//! tested for visibility by the same occlusion test as the edges. So it is
//! held to the same oracle -- a point-by-point ray cast -- and to the same
//! property: nothing in it has a pixel, so the bake at 64 is the bake at
//! 2048 to the bit.

// Swift's `guard x > 0 else` refuses a NaN, and so does `if !(x > 0.0)`.
#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::manual_range_contains)]

mod common;

use std::collections::BTreeSet;

use common::{par_map, piece_meshes, ray_visible, stress_cameras};
use kurven_bake::{BakeOptions, bake};
use kurven_core::camera::Camera;
use kurven_core::hatching::{FaceSelector, Hatching, Tone};
use kurven_core::manifest::{HeightPolicy, LayerRole, LayerSource, LayerSpec, ObliqueProjection};
use kurven_core::polyline::Layer;
use kurven_core::scene::{PreviewMode, Scene};
use kurven_core::simd::Vec3;
use kurven_core::solid::{Platonic, Solid};
use kurven_math::{real, swift};

/// Whether `p`, in the plane of face `f` of `s`, lies in one of its pieces.
fn on_face(s: &Solid, f: usize, p: Vec3) -> bool {
    let face = s.frames[f];
    if !((p - face.origin).dot(face.normal).abs() < 1e-9) {
        return false;
    }
    s.faces[f].iter().any(|&piece| {
        let poly = &s.pieces[piece];
        (0..poly.len()).all(|k| {
            let (a, b) = (s.vertices[poly[k]].v, s.vertices[poly[(k + 1) % poly.len()]].v);
            !((b - a).cross(p - a).dot(face.normal) < -1e-9)
        })
    })
}

/// Whether `p` lies on a side of one of face `f`'s pieces.
fn on_piece_boundary(s: &Solid, f: usize, p: Vec3) -> bool {
    s.faces[f].iter().any(|&piece| {
        let poly = &s.pieces[piece];
        (0..poly.len()).any(|k| {
            let (a, b) = (s.vertices[poly[k]].v, s.vertices[poly[(k + 1) % poly.len()]].v);
            let t = (p - a).dot(b - a) / (b - a).length_squared();
            t >= -1e-9 && t <= 1.0 + 1e-9 && (p - (a + t * (b - a))).length() < 1e-6
        })
    })
}

/// "hatching: a hatch line lies on its face, runs unbroken across a
/// diagonal, and ends on the tone's own level set"
#[test]
fn a_hatch_line_lies_on_its_face_runs_unbroken_across_a_diagonal_and_ends_on_the_tones_own_level_set() {
    let s = Solid::frame(4.0, 2.0, 1.0);
    let lines = s.hatch_lines(&Hatching::plain(0.6, 0.07));
    let (mut in_plane, mut cross_diagonal, mut ends_on_diagonal) = (true, 0, 0);
    // The front annulus, face 0, and the diagonals of its partition.
    let front = 0;
    let diagonals: Vec<(Vec3, Vec3)> = (0..s.edges.len())
        .filter(|&k| s.is_diagonal[k] && s.face_index_of_piece(s.edges[k].left) == front)
        .map(|k| (s.vertices[s.edges[k].a].v, s.vertices[s.edges[k].b].v))
        .collect();
    let on_diagonal = |p: Vec3| {
        diagonals.iter().any(|&(a, b)| {
            let t = (p - a).dot(b - a) / (b - a).length_squared();
            t > 1e-6 && t < 1.0 - 1e-6 && (p - (a + t * (b - a))).length() < 1e-9
        })
    };
    for line in &lines {
        if line.path.iter().any(|v| !on_face(&s, line.face, v.v)) {
            in_plane = false;
        }
        if !(line.face == front && line.path.len() == 2) {
            continue;
        }
        ends_on_diagonal += line.path.iter().filter(|v| on_diagonal(v.v)).count();
        // Crosses a diagonal: the diagonal's ends lie on either side of the
        // hatch line and the crossing is within it.
        let (a, b) = (line.path[0].v, line.path[1].v);
        for &(p, q) in &diagonals {
            let n = (b - a).cross(s.normals[0]);
            let (fp, fq) = ((p - a).dot(n), (q - a).dot(n));
            if (fp > 0.0) == (fq > 0.0) {
                continue;
            }
            let x = p + (q - p) * (fp / (fp - fq));
            let t = (x - a).dot(b - a) / (b - a).length_squared();
            if t > 1e-6 && t < 1.0 - 1e-6 {
                cross_diagonal += 1;
            }
        }
    }
    eprintln!("frame: {} hatch lines, {cross_diagonal} crossings, {ends_on_diagonal} ends on a diagonal", lines.len());
    assert!(
        in_plane && lines.len() > 100,
        "every hatch line lies on its face, within its pieces: {} hatch lines",
        lines.len()
    );
    assert!(
        cross_diagonal > 0 && ends_on_diagonal == 0,
        "on the front annulus hatch lines cross the diagonals between its pieces whole, and none ends on one: \
         {cross_diagonal} crossings, {ends_on_diagonal} ends on a diagonal"
    );

    // With Turing's tone and one width, an end away from the pieces' sides
    // is a zero of the tone, to the last bit. With several widths it lies on
    // a step.
    let cube = Solid::platonic(Platonic::Cube, 2.0).resting(0);
    for width_steps in [1, 4] {
        let hatching = Hatching::new(0.3, 0.03, None, Tone::Turing { scale: 0.6, seed: 11 }, width_steps);
        let (mut interior, mut off_step, mut out_of_range) = (0, 0, 0);
        for line in cube.hatch_lines(&hatching) {
            if line.width_step < 1 || line.width_step > width_steps {
                out_of_range += 1;
            }
            let face = cube.frames[line.face];
            let tone = hatching.tone.value(&face);
            for v in &line.path {
                if on_piece_boundary(&cube, line.face, v.v) {
                    continue;
                }
                interior += 1;
                let w = tone(face.coordinates(v.v));
                // The end is where the width steps: at a threshold
                // k / widthSteps, within the bisection's resolution.
                if !(0..=width_steps).any(|k| (w - k as f64 / width_steps as f64).abs() < 1e-7) {
                    off_step += 1;
                }
            }
        }
        eprintln!(
            "cube, {width_steps} widths: {interior} interior ends, {off_step} off a step, {out_of_range} out of range"
        );
        assert!(
            interior > 50 && off_step == 0 && out_of_range == 0,
            "{}: {interior} interior ends, {off_step} off a step, {out_of_range} out of range",
            if width_steps == 1 {
                "with one width, every interior end is a zero of Turing's tone".to_string()
            } else {
                format!("with {width_steps} widths, every interior end lies on a step and every width is in range")
            }
        );
    }
}

/// "hatching: a change of width is bisected to neighbouring numbers,
/// however far from the origin"
///
/// A thousand units out, one ulp of the parameter is 1.1e-13: a bisection
/// that stops at a fixed tolerance below that never stops.
#[test]
fn a_change_of_width_is_bisected_to_neighbouring_numbers_however_far_from_the_origin() {
    let cut = 600.2345;
    let graded = Solid::graded(600.0, 600.5, 0.1, &|x| if x < cut { 1 } else { 2 });
    let end = graded.first().map_or(0.0, |g| g.1);
    assert!(
        graded.len() == 2 && graded[1].0 == end && end >= cut && end.next_down() < cut,
        "a step from 1 to 2 at 600.2345 ends the first interval on the first number past it: {} intervals",
        graded.len()
    );
    let frame = Solid::frame(2000.0, 1998.0, 1.0);
    let lines = frame.hatch_lines(&Hatching::new(0.0, 5.0, None, Tone::Turing { scale: 0.5, seed: 1 }, 1));
    eprintln!("a frame 2000 across: {} hatch lines", lines.len());
    assert!(
        !lines.is_empty(),
        "a frame 2000 across, its strips one unit wide, is hatched with Turing's tone: {} hatch lines",
        lines.len()
    );
}

/// "hatching: a hatch line along an edge is left to the edge, and one along
/// a diagonal is drawn"
#[test]
fn a_hatch_line_along_an_edge_is_left_to_the_edge_and_one_along_a_diagonal_is_drawn() {
    let s = Solid::frame(4.0, 2.0, 1.0);
    // The front's frame has u along x and v along y, so at angle 0 and
    // spacing 0.5 lines fall on the outer edges y = ±2, along the hole's
    // edges y = ±1, and on the walls' top and bottom edges.
    let on_edge = |p: Vec3| {
        (0..s.edges.len()).any(|k| {
            if s.is_diagonal[k] {
                return false;
            }
            let (a, b) = (s.vertices[s.edges[k].a].v, s.vertices[s.edges[k].b].v);
            let t = (p - a).dot(b - a) / (b - a).length_squared();
            t >= -1e-9 && t <= 1.0 + 1e-9 && (p - (a + t * (b - a))).length() < 1e-9
        })
    };
    let lines = s.hatch_lines(&Hatching::plain(0.0, 0.5));
    let along = lines.iter().filter(|l| on_edge(0.5 * (l.path[0].v + l.path[1].v))).count();
    let front = s.face_index_of_piece(0);
    let mut at_hole: Vec<(f64, f64)> = lines
        .iter()
        .filter(|l| l.face == front && (l.path[0].y() - 1.0).abs() < 1e-9)
        .map(|l| (swift::min(l.path[0].x(), l.path[1].x()), swift::max(l.path[0].x(), l.path[1].x())))
        .collect();
    // Swift's `sorted { $0.0 < $1.0 }`: stable, by the first.
    at_hole.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let split = at_hole.len() == 2
        && (at_hole[0].0 + 2.0).abs() < 1e-9
        && (at_hole[0].1 + 1.0).abs() < 1e-9
        && (at_hole[1].0 - 1.0).abs() < 1e-9
        && (at_hole[1].1 - 2.0).abs() < 1e-9;
    eprintln!("{along} along an edge, y = 1 drawn over {at_hole:?}");
    assert!(
        along == 0 && split,
        "no hatch line runs along an edge; y = 1 on the front is drawn on either side of the hole, not across it: \
         {along} along an edge, y = 1 drawn over {at_hole:?}"
    );
    // At 45° the line through the front's centre lies on two of its
    // diagonals, which are not edges: it is drawn along both.
    let diagonal: Vec<_> = s
        .hatch_lines(&Hatching::plain(std::f64::consts::PI / 4.0, 0.5))
        .into_iter()
        .filter(|l| l.face == front && l.path.iter().all(|p| (p.x() - p.y()).abs() < 1e-9))
        .collect();
    let length = diagonal.iter().fold(0.0, |acc, l| acc + (l.path[0].v - l.path[1].v).length());
    eprintln!("{} lines along the diagonals, length {length}", diagonal.len());
    assert!(
        diagonal.len() == 2 && (length - 2.0 * real::sqrt(2.0)).abs() < 1e-9,
        "a hatch line along the front's diagonals is drawn along both: {} lines, length {length}",
        diagonal.len()
    );
}

/// "hatching: a face seen edge-on draws no hatching"
#[test]
fn a_face_seen_edge_on_draws_no_hatching() {
    let s = Solid::frame(4.0, 2.0, 1.0);
    let lines = s.hatch_lines(&Hatching::plain(0.5, 0.08));
    let (mut edge_on, mut drawn) = (0, 0);
    for (_, view) in stress_cameras() {
        let occlusion = s.occlusion(&view);
        for i in 0..s.faces.len() {
            if !(s.normals[s.faces[i][0]].dot(occlusion.to_eye).abs() <= 1e-9) {
                continue;
            }
            edge_on += 1;
            let paths: Vec<_> = lines.iter().filter(|l| l.face == i).map(|l| l.path.clone()).collect();
            drawn += s.visible_curves(&paths, i, &occlusion).len();
        }
    }
    eprintln!("{edge_on} faces edge-on, {drawn} drawn");
    assert!(
        edge_on > 0 && drawn == 0,
        "under the stress cameras {edge_on} faces are seen edge-on, and none draws a hatch line: {drawn} drawn"
    );
}

/// "hatching: tested by occlusion, hatching agrees with a ray cast, through
/// the frame's hole included"
#[test]
fn tested_by_occlusion_hatching_agrees_with_a_ray_cast_through_the_frames_hole_included() {
    let s = Solid::frame(4.0, 2.0, 1.0);
    let hatching = Hatching::plain(0.5, 0.08);
    let meshes = piece_meshes(&s);
    let lines = s.hatch_lines(&hatching);
    let cameras = stress_cameras();
    let per_camera = par_map(&cameras, |(name, view)| {
        let d = -view.projection_direction();
        let seen = s.visible_hatching(&hatching, view);
        assert!(seen.len() == 1, "one width");
        let visible = &seen[0].2;
        let (mut compared, mut wrong, mut cut) = (0, 0, 0);
        let mut example = String::new();
        // Every visible segment as (face, a, b).
        let mut segments: Vec<(usize, Vec3, Vec3)> = Vec::new();
        for p in visible.paths() {
            for w in p.windows(2) {
                let (a, b) = (w[0].v, w[1].v);
                let Some(f) = (0..s.faces.len()).find(|&f| on_face(&s, f, a) && on_face(&s, f, b)) else {
                    wrong += 1;
                    continue;
                };
                segments.push((f, a, b));
            }
        }
        for line in &lines {
            // A face seen exactly edge-on projects to a line, which the ray
            // cast cannot decide; it draws no hatching (tested above).
            if !(s.normals[s.faces[line.face][0]].dot(d).abs() > 1e-9) {
                continue;
            }
            let (a, b) = (line.path[0].v, line.path[1].v);
            let own = &s.faces[line.face];
            let mut runs: Vec<(f64, f64)> = Vec::new();
            for &(f, p, q) in &segments {
                if f != line.face {
                    continue;
                }
                let len2 = (b - a).length_squared();
                let (t0, t1) = ((p - a).dot(b - a) / len2, (q - a).dot(b - a) / len2);
                if (p - (a + t0 * (b - a))).length() < 1e-9
                    && (q - (a + t1 * (b - a))).length() < 1e-9
                    && t0 >= -1e-9
                    && t1 <= 1.0 + 1e-9
                {
                    runs.push((t0, t1));
                    if t0 > 1e-9 || t1 < 1.0 - 1e-9 {
                        cut += 1;
                    }
                }
            }
            for k in 0..20 {
                let t = (k as f64 + 0.5) / 20.0;
                if runs.iter().any(|x| (t - x.0).abs() < 1e-6 || (t - x.1).abs() < 1e-6) {
                    continue;
                }
                let p = a + t * (b - a);
                let seen_by_occlusion = runs.iter().any(|x| t > x.0 && t < x.1);
                let truth = ray_visible(&s, &meshes, p, own, d);
                compared += 1;
                if truth != seen_by_occlusion {
                    wrong += 1;
                    if example.is_empty() {
                        example = format!(
                            "e.g. {name} face {} t={t:.3} truth {}",
                            line.face,
                            if truth { "seen" } else { "hidden" }
                        );
                    }
                }
            }
        }
        (compared, wrong, cut, example)
    });
    let (mut compared, mut wrong, mut cut) = (0, 0, 0);
    let mut example = String::new();
    for (c, w, k, ex) in per_camera {
        compared += c;
        wrong += w;
        cut += k;
        if example.is_empty() {
            example = ex;
        }
    }
    eprintln!("{cut} spans are parts of lines, {wrong} of {compared} sampled points disagree");
    assert!(
        cut > 0,
        "under {} cameras some hatch lines are drawn in part: the front hides the walls in the hole: {cut} spans are \
         parts of lines",
        cameras.len()
    );
    assert!(
        wrong == 0 && compared > 20_000,
        "and every hatch line agrees with a point-by-point ray cast against the pieces: {wrong} of {compared} \
         sampled points disagree {example}"
    );
}

/// "hatching: on a convex solid facing is the whole test, and the bake is
/// the same at every resolution"
///
/// Swift bakes with Metal; the port with `kurven_bake::bake`, which renders
/// no depth buffer and draws every stroke from the scene's own tests, as
/// Swift's strokes are drawn.
#[test]
fn on_a_convex_solid_facing_is_the_whole_test_and_the_bake_is_the_same_at_every_resolution() {
    let s = Solid::platonic(Platonic::Dodecahedron, 2.0).resting(0);
    // A face is about 1.4 across; at this scale that is a dozen Turing grid
    // spacings, enough spots for every width to occur on it. At half a unit
    // a face sits inside a single spot.
    let hatching = Hatching::new(1.1, 0.05, None, Tone::Turing { scale: 0.12, seed: 5 }, 3);
    let camera = Camera::oblique(&ObliqueProjection {
        shear: 0.3,
        x_angle: -55.0,
        z_angle: -70.0,
        flip_x: false,
        y_scale: None,
    });
    let d = -camera.view.projection_direction();
    let (mut whole_or_absent, mut widths_seen) = (true, BTreeSet::new());
    let lines = s.hatch_lines(&hatching);
    for (w, all, visible) in s.visible_hatching(&hatching, &camera.view) {
        widths_seen.insert(w);
        // Each face's hatch lines are all kept, whole and to the bit, or all
        // dropped, by its facing.
        for line in lines.iter().filter(|l| l.width_step == w) {
            let facing = s.normals[s.faces[line.face][0]].dot(d) > 0.0;
            let kept = visible.paths().any(|p| p == line.path.as_slice());
            if kept != facing {
                whole_or_absent = false;
            }
        }
        if visible.count() > all.count() {
            whole_or_absent = false;
        }
    }
    eprintln!("widths {widths_seen:?}, {} lines", lines.len());
    assert!(
        whole_or_absent && widths_seen == BTreeSet::from([1, 2, 3]),
        "every hatch line on a facing face is drawn whole and every other dropped, at all three widths: widths \
         {widths_seen:?}"
    );
    let layers = vec![
        Layer {
            spec: LayerSpec::new("edges", LayerRole::Outline, LayerSource::Edges, 0.5, HeightPolicy::Surface),
            paths: s.edge_paths(),
        },
        Layer {
            spec: LayerSpec::new(
                "hatching",
                LayerRole::Scaffold,
                LayerSource::Hatching { hatching, faces: FaceSelector::Solid },
                0.3,
                HeightPolicy::Surface,
            ),
            paths: s.hatch_paths(&hatching),
        },
    ];
    let scene = Scene::solid_scene(s.clone(), layers.clone(), camera, PreviewMode::Drawing, 0.0);
    let coarse = bake(&scene, &BakeOptions::new(64)).expect("the bake at 64");
    let fine = bake(&scene, &BakeOptions::new(2048)).expect("the bake at 2048");
    let same = coarse.strokes.layers.len() == fine.strokes.layers.len()
        && coarse.strokes.layers.iter().zip(&fine.strokes.layers).all(|(a, b)| a.1 == b.1 && a.0 == b.0);
    let widths: Vec<f64> = coarse.strokes.layers.iter().map(|l| l.0.width).collect();
    let expected = [0.5, 0.1, 0.2, 0.3];
    let widths_right =
        widths.len() == expected.len() && widths.iter().zip(expected).all(|(a, b)| (a - b).abs() < 1e-12);
    eprintln!("{} layers, widths {widths:?}, {} strokes", coarse.strokes.layers.len(), coarse.strokes.path_count());
    assert!(
        same && coarse.strokes.layers.len() == 4 && widths_right,
        "the bake has the edges and a layer per width step at a third, two thirds and the width, the same at 64 px \
         and 2048 px: {} layers, widths {widths:?}",
        coarse.strokes.layers.len()
    );
    let mut unclipped = layers[1].spec.clone();
    unclipped.clipped = false;
    let visible = Scene::solid_scene(
        s.clone(),
        vec![Layer { spec: unclipped, paths: s.hatch_paths(&hatching) }],
        camera,
        PreviewMode::Drawing,
        0.0,
    )
    .visible_layers(None, None)
    .expect("a solid's layers");
    let drawn: usize = visible.iter().map(|(_, p)| p.count()).sum();
    assert!(
        visible.len() == 3 && drawn == lines.len(),
        "unclipped, every hatch line is drawn, in a layer per width step: {drawn} of {} lines in {} layers",
        lines.len(),
        visible.len()
    );
}
