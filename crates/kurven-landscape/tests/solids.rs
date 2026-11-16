//! `Sources/kurven-test/Solids.swift`: the solids, every suite of
//! `solidTests` and `solidPlacementTests`, with Swift's tolerances.
//!
//! A polyhedron's visibility is the resolution test in its purest form: an
//! edge is seen where one of its faces faces the eye and nothing stands in
//! front, decided per edge by two signs and, on a body that can stand in its
//! own way, by testing the edge for occlusion by the pieces in closed form.
//! Nothing here has a pixel in it, so a bake at 64 is the bake at 4096 to the
//! bit -- and the convex case is held to a rule simpler still, the two signs
//! alone.
//!
//! The oracles are brute force, as in Swift; where a suite casts a ray per
//! sample under every stress camera, the cameras are spread over threads
//! (`common::par_map`), each camera's verdicts independent of the others'.

// Swift's `guard x > 0 else` refuses a NaN, and so does `if !(x > 0.0)`;
// the oracles keep Swift's guards as they are written.
#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::manual_range_contains, clippy::type_complexity)]

mod common;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::f64::consts::PI;
use std::time::{Duration, Instant};

use common::{first_hit, par_map, piece_meshes, ray_visible, stress_cameras};
use kurven_bake::{BakeOptions, bake};
use kurven_core::camera::Camera;
use kurven_core::hatching::{FaceSelector, Hatching, Tone};
use kurven_core::manifest::{HeightPolicy, LayerRole, LayerSource, LayerSpec, ObliqueProjection};
use kurven_core::mesh::Mesh;
use kurven_core::polyline::Layer;
use kurven_core::scene::{PreviewMode, Scene};
use kurven_core::simd::{Mat3, Vec2, Vec3, Vec4};
use kurven_core::solid::{Edge, Platonic, Solid};
use kurven_core::spaces::{P3, Transform, ViewSpace, WorldSpace};
use kurven_core::surface_request::SurfaceRequest;
use kurven_math::{real, swift};

// MARK: - helpers

/// `simd_double3x3(simd_quatd(angle:axis:))`: the rotation by `angle` about
/// the unit `axis`, through the quaternion `(sin(angle/2) axis, cos(angle/2))`
/// and simd's quaternion-to-matrix formula, term for term.
fn quaternion_matrix(angle: f64, axis: Vec3) -> Mat3 {
    let s = real::sin(angle / 2.0);
    let q = Vec4::new(s * axis.x, s * axis.y, s * axis.z, real::cos(angle / 2.0));
    let (x, y, z, w) = (q.x, q.y, q.z, q.w);
    Mat3::from_columns(
        Vec3::new(x * x - y * y - z * z + w * w, 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)),
        Vec3::new(2.0 * (x * y - z * w), y * y - z * z + w * w - x * x, 2.0 * (y * z + x * w)),
        Vec3::new(2.0 * (z * x + y * w), 2.0 * (y * z - x * w), z * z + w * w - x * x - y * y),
    )
}

/// The tilt the far-from-the-origin suites turn their solids by.
fn tilt() -> Mat3 {
    quaternion_matrix(0.7, Vec3::new(1.0, 2.0, 3.0).normalized())
}

/// The loops of a solid as its constructor takes them.
fn index_loops(loops: &[Vec<Vec<usize>>], offset: usize, reversed: bool) -> Vec<Vec<Vec<i64>>> {
    loops
        .iter()
        .map(|f| {
            f.iter()
                .map(|l| {
                    let mut l: Vec<i64> = l.iter().map(|&k| (k + offset) as i64).collect();
                    if reversed {
                        l.reverse();
                    }
                    l
                })
                .collect()
        })
        .collect()
}

/// Whether two triangles cross: an edge of one passes through the open
/// interior of the other. Triangles sharing an edge cannot cross; ones
/// sharing a vertex are tested away from it.
fn triangles_cross(a: &[Vec3], b: &[Vec3], shared: usize) -> bool {
    if !(shared < 2) {
        return false;
    }
    let through = |p: Vec3, q: Vec3, t: &[Vec3]| -> bool {
        let n = (t[1] - t[0]).cross(t[2] - t[0]);
        let d = q - p;
        let den = n.dot(d);
        if !(den.abs() > 1e-12) {
            return false;
        }
        let s = n.dot(t[0] - p) / den;
        if !(s > 1e-9 && s < 1.0 - 1e-9) {
            return false;
        }
        let x = p + s * d;
        let (v0, v1, v2) = (t[1] - t[0], t[2] - t[0], x - t[0]);
        let (d00, d01, d11) = (v0.dot(v0), v0.dot(v1), v1.dot(v1));
        let (d20, d21) = (v2.dot(v0), v2.dot(v1));
        let den2 = d00 * d11 - d01 * d01;
        let v = (d11 * d20 - d01 * d21) / den2;
        let w = (d00 * d21 - d01 * d20) / den2;
        v > 1e-9 && w > 1e-9 && 1.0 - v - w > 1e-9
    };
    (0..3).any(|k| through(a[k], a[(k + 1) % 3], b)) || (0..3).any(|k| through(b[k], b[(k + 1) % 3], a))
}

/// A prism over a face in the plane `z = 0`, `depth` deep: the face's loops
/// as the plane gives them (the outer counter-clockwise, the holes
/// clockwise), its top, its bottom, and a wall along every run of sides in
/// one line -- one face, with the straight corners in its loop.
fn prism(loops: &[Vec<Vec2>], depth: f64) -> Solid {
    let mut vertices: Vec<P3<WorldSpace>> = Vec::new();
    let (mut top, mut bottom): (Vec<Vec<i64>>, Vec<Vec<i64>>) = (Vec::new(), Vec::new());
    let mut walls: Vec<Vec<Vec<i64>>> = Vec::new();
    for lp in loops {
        let (base, n) = (vertices.len(), lp.len());
        vertices.extend(lp.iter().map(|p| P3::new(p.x, p.y, depth)));
        vertices.extend(lp.iter().map(|p| P3::new(p.x, p.y, 0.0)));
        top.push((0..n).map(|k| (base + k) as i64).collect());
        bottom.push((0..n).rev().map(|k| (base + n + k) as i64).collect());
        let straight = |k: usize| -> bool {
            let (a, b, c) = (lp[(k + n - 1) % n], lp[k], lp[(k + 1) % n]);
            let (u, v) = (b - a, c - b);
            (u.x * v.y - u.y * v.x).abs() < 1e-12 && u.dot(v) > 0.0
        };
        let start = (0..n).find(|&k| !straight(k)).expect("a corner");
        let mut run = vec![start];
        for j in 1..=n {
            let k = (start + j) % n;
            run.push(k);
            if straight(k) {
                continue;
            }
            // Along the bottom, then back along the top.
            let mut wall: Vec<i64> = run.iter().map(|&r| (base + n + r) as i64).collect();
            wall.extend(run.iter().rev().map(|&r| (base + r) as i64));
            walls.push(vec![wall]);
            run = vec![k];
        }
    }
    let mut faces = vec![top, bottom];
    faces.extend(walls);
    Solid::new(vertices, faces)
}

/// A box 4 by 4 by 2 on `z = 0` with pits sunk in its top, each an inverted
/// pyramid 1 deep. `points` are the top's vertices, the box's corners first
/// (counter-clockwise from above); each mouth lists its corners among them,
/// counter-clockwise from above, and `top` is the top face's loops.
fn pitted_box(points: &[Vec2], mouths: &[Vec<usize>], top: Vec<Vec<i64>>) -> Solid {
    let m = points.len() as i64;
    let mut vertices: Vec<P3<WorldSpace>> = points.iter().map(|p| P3::new(p.x, p.y, 2.0)).collect();
    vertices.extend(points[..4].iter().map(|p| P3::new(p.x, p.y, 0.0)));
    let mut faces: Vec<Vec<Vec<i64>>> = vec![top, vec![vec![m, m + 3, m + 2, m + 1]]];
    for k in 0..4 {
        faces.push(vec![vec![m + k, m + (k + 1) % 4, (k + 1) % 4, k]]);
    }
    for mouth in mouths {
        let centre = mouth.iter().fold(Vec2::ZERO, |a, &i| a + points[i]) / mouth.len() as f64;
        vertices.push(P3::new(centre.x, centre.y, 1.0));
        let apex = (vertices.len() - 1) as i64;
        for k in 0..mouth.len() {
            faces.push(vec![vec![mouth[k] as i64, mouth[(k + 1) % mouth.len()] as i64, apex]]);
        }
    }
    Solid::new(vertices, faces)
}

/// What is wrong with a solid's partition, or nothing: per face, every piece
/// is convex and uses only the face's vertices, the pieces' areas add up to
/// the face's, no point of the face lies in two pieces or in none, and the
/// diagonals number pieces − 1 + holes, which is Euler's formula for a disc
/// with holes cut into discs.
fn partition_report(s: &Solid) -> String {
    let mut problems: Vec<String> = Vec::new();
    for i in 0..s.faces.len() {
        let (frame, members, loops) = (s.frames[i], &s.faces[i], &s.loops[i]);
        let plane = |k: usize| frame.coordinates(s.vertices[k].v);
        let area = |polygon: &[usize]| frame.area(&polygon.iter().map(|&k| s.vertices[k].v).collect::<Vec<_>>());
        let own: HashSet<usize> = loops.iter().flatten().copied().collect();
        let mut piece_area = 0.0;
        for piece in members.iter().map(|&m| &s.pieces[m]) {
            piece_area += area(piece);
            let set: HashSet<usize> = piece.iter().copied().collect();
            if !set.is_subset(&own) || set.len() != piece.len() {
                problems.push(format!("face {i}: piece {piece:?} is not made of the face's vertices"));
            }
            let n = piece.len();
            for k in 0..n {
                let (a, b, c) = (plane(piece[k]), plane(piece[(k + 1) % n]), plane(piece[(k + 2) % n]));
                let turn = (b - a).x * (c - b).y - (b - a).y * (c - b).x;
                if turn < -1e-12 {
                    problems.push(format!("face {i}: piece {piece:?} turns right at {}", piece[(k + 1) % n]));
                }
            }
        }
        let face_area = loops.iter().fold(0.0, |acc, l| acc + area(l));
        if (piece_area - face_area).abs() > 1e-9 * swift::max(1.0, face_area) {
            problems.push(format!("face {i}: the pieces cover {piece_area}, the face is {face_area}"));
        }
        // A grid of points, each in the face (even-odd over the loops) or
        // not, and in as many pieces; points near a side are left out.
        let near = |p: Vec2, polygon: &[usize]| -> bool {
            (0..polygon.len()).any(|k| {
                let (a, b) = (plane(polygon[k]), plane(polygon[(k + 1) % polygon.len()]));
                let t = swift::max(0.0, swift::min(1.0, (p - a).dot(b - a) / (b - a).length_squared()));
                (p - (a + t * (b - a))).length() < 1e-6
            })
        };
        let crossings = |p: Vec2, polygon: &[usize]| -> usize {
            (0..polygon.len())
                .filter(|&k| {
                    let (a, b) = (plane(polygon[k]), plane(polygon[(k + 1) % polygon.len()]));
                    if (a.y > p.y) == (b.y > p.y) {
                        return false;
                    }
                    p.x < a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y)
                })
                .count()
        };
        let all: Vec<Vec2> = loops.iter().flatten().map(|&k| plane(k)).collect();
        let lo = all.iter().fold(Vec2::splat(f64::INFINITY), |a, &b| a.min(b));
        let hi = all.iter().fold(Vec2::splat(f64::NEG_INFINITY), |a, &b| a.max(b));
        let pieces: Vec<&Vec<usize>> = members.iter().map(|&m| &s.pieces[m]).collect();
        let mut mismatches = 0;
        for a in 0..61 {
            for b in 0..61 {
                let p = lo + (hi - lo).mul_elements(Vec2::new(a as f64 + 0.37, b as f64 + 0.61)) / 61.0;
                if pieces.iter().any(|piece| near(p, piece)) {
                    continue;
                }
                let in_face = loops.iter().map(|l| crossings(p, l)).sum::<usize>() % 2 == 1;
                let in_pieces = pieces.iter().filter(|piece| crossings(p, piece) % 2 == 1).count();
                if in_pieces != usize::from(in_face) {
                    mismatches += 1;
                }
            }
        }
        if mismatches > 0 {
            problems.push(format!("face {i}: {mismatches} sample points in the wrong number of pieces"));
        }
        let diagonals =
            (0..s.edges.len()).filter(|&k| s.is_diagonal[k] && s.face_index_of_piece(s.edges[k].left) == i).count();
        if diagonals != members.len() - 1 + (loops.len() - 1) {
            problems.push(format!(
                "face {i}: {diagonals} diagonals for {} pieces and {} holes",
                members.len(),
                loops.len() - 1
            ));
        }
    }
    problems.into_iter().take(4).collect::<Vec<_>>().join("; ")
}

// MARK: - oracles

/// The edge of `s` that the segment `a -> b` lies along, with its index.
fn edge_under(a: Vec3, b: Vec3, s: &Solid) -> Option<(usize, Edge)> {
    for (k, e) in s.edges.iter().enumerate() {
        let (p, q) = (s.vertices[e.a].v, s.vertices[e.b].v);
        let len2 = (q - p).length_squared();
        let (t0, t1) = ((a - p).dot(q - p) / len2, (b - p).dot(q - p) / len2);
        // Both ends within the edge, not merely on its line: an edge can
        // run on in line with another.
        if (a - (p + t0 * (q - p))).length() < 1e-9
            && (b - (p + t1 * (q - p))).length() < 1e-9
            && swift::min(t0, t1) >= -1e-9
            && swift::max(t0, t1) <= 1.0 + 1e-9
        {
            return Some((k, *e));
        }
    }
    None
}

fn edge_fractions(a: Vec3, b: Vec3, e: Edge, s: &Solid) -> (f64, f64) {
    let (p, q) = (s.vertices[e.a].v, s.vertices[e.b].v);
    let len2 = (q - p).length_squared();
    ((a - p).dot(q - p) / len2, (b - p).dot(q - p) / len2)
}

/// Why the ray cast says what it says: the facing of the own pieces and
/// every other piece hit, with its parameter.
fn ray_report(s: &Solid, meshes: &[Mesh<WorldSpace>], p: Vec3, own: &[usize], d: Vec3) -> String {
    let mut sorted = own.to_vec();
    sorted.sort();
    let facing: Vec<String> = sorted.iter().map(|&f| format!("f{f}·d={:.3}", s.normals[f].dot(d))).collect();
    let mut hits: Vec<String> = Vec::new();
    for (f, mesh) in meshes.iter().enumerate() {
        if own.contains(&f) {
            continue;
        }
        if let Some(t) = first_hit(mesh, p, d, 1e-9) {
            hits.push(format!("f{f}@{t:.4}"));
        }
    }
    format!(
        "p=({:.4},{:.4},{:.4}) d=({:.3},{:.3},{:.3}) {} hits {}",
        p.x,
        p.y,
        p.z,
        d.x,
        d.y,
        d.z,
        facing.join(" "),
        hits.join(" ")
    )
}

/// Point-by-point agreement of an edge's drawn spans with the ray cast:
/// disagreements, comparisons, and an example of the first disagreement.
fn occlusion_agreement(
    s: &Solid,
    meshes: &[Mesh<WorldSpace>],
    e: Edge,
    spans: &[(f64, f64)],
    d: Vec3,
    camera: &str,
) -> (usize, usize, String) {
    let (a, b) = (s.vertices[e.a].v, s.vertices[e.b].v);
    let (mut wrong, mut compared) = (0, 0);
    let mut example = String::new();
    let own = [e.left, e.right];
    for k in 0..200 {
        let t = (k as f64 + 0.5) / 200.0;
        // Within a hair of a span's end the truth is at a boundary;
        // elsewhere it is decided.
        if spans.iter().any(|x| (t - x.0).abs() < 1e-6 || (t - x.1).abs() < 1e-6) {
            continue;
        }
        let p = a + t * (b - a);
        let seen_by_occlusion = spans.iter().any(|x| t > x.0 && t < x.1);
        let truth = ray_visible(s, meshes, p, &own, d);
        compared += 1;
        if truth != seen_by_occlusion {
            wrong += 1;
            if example.is_empty() {
                let spans: Vec<String> = spans.iter().map(|x| format!("[{:.4}, {:.4}]", x.0, x.1)).collect();
                example = format!(
                    "e.g. {camera} edge {}-{} t={t:.3} truth {} {} spans {}",
                    e.a,
                    e.b,
                    if truth { "seen" } else { "hidden" },
                    ray_report(s, meshes, p, &own, d),
                    spans.join(" ")
                );
            }
        }
    }
    (wrong, compared, example)
}

/// Each edge's drawn spans under `view`, as fractions of the edge, and how
/// many drawn spans lie on no edge and how many on a diagonal.
fn drawn_spans(s: &Solid, view: &Transform<WorldSpace, ViewSpace>) -> (HashMap<Edge, Vec<(f64, f64)>>, usize, usize) {
    let drawn = s.visible_edges(view);
    let mut spans: HashMap<Edge, Vec<(f64, f64)>> = HashMap::new();
    let (mut off_edges, mut on_diagonal) = (0, 0);
    for p in drawn.paths() {
        let (first, last) = (p[0].v, p[p.len() - 1].v);
        let Some((k, e)) = edge_under(first, last, s) else {
            off_edges += 1;
            continue;
        };
        if s.is_diagonal[k] {
            on_diagonal += 1;
        }
        spans.entry(e).or_default().push(edge_fractions(first, last, e, s));
    }
    (spans, off_edges, on_diagonal)
}

/// Point-by-point agreement of every edge's drawn spans with the ray cast,
/// under every stress camera: disagreements, comparisons, and an example.
/// An edge with a piece exactly edge-on lies on the outline, where whether
/// it is drawn is rounding, and is set aside.
fn edge_agreement(s: &Solid) -> (usize, usize, String) {
    let meshes = piece_meshes(s);
    let per_camera = par_map(&stress_cameras(), |(camera, view)| {
        let d = -view.projection_direction();
        let (spans, _, _) = drawn_spans(s, view);
        let (mut wrong, mut compared) = (0, 0);
        let mut example = String::new();
        for (k, e) in s.edges.iter().enumerate() {
            if s.is_diagonal[k] || !(s.normals[e.left].dot(d).abs() > 1e-9 && s.normals[e.right].dot(d).abs() > 1e-9) {
                continue;
            }
            let (w, c, ex) = occlusion_agreement(s, &meshes, *e, spans.get(e).map_or(&[][..], |v| &v[..]), d, camera);
            wrong += w;
            compared += c;
            if example.is_empty() {
                example = ex;
            }
        }
        (wrong, compared, example)
    });
    let (mut wrong, mut compared) = (0, 0);
    let mut example = String::new();
    for (w, c, ex) in per_camera {
        wrong += w;
        compared += c;
        if example.is_empty() {
            example = ex;
        }
    }
    (wrong, compared, example)
}

// MARK: - solidTests

/// "solids: the five Platonic solids come out of their coordinates alone"
#[test]
fn the_five_platonic_solids_come_out_of_their_coordinates_alone() {
    let expected = [
        (Platonic::Tetrahedron, 4, 6, 4, 3),
        (Platonic::Cube, 8, 12, 6, 4),
        (Platonic::Octahedron, 6, 12, 8, 3),
        (Platonic::Dodecahedron, 20, 30, 12, 5),
        (Platonic::Icosahedron, 12, 30, 20, 3),
    ];
    for (kind, v, e, f, sides) in expected {
        let s = Solid::platonic(kind, 1.0);
        let lengths: Vec<f64> = s.edges.iter().map(|e| (s.vertices[e.a].v - s.vertices[e.b].v).length()).collect();
        let longest = lengths.iter().copied().reduce(swift::max).unwrap();
        let shortest = lengths.iter().copied().reduce(swift::min).unwrap();
        let regular = (longest - shortest) < 1e-9
            && s.faces.len() == s.pieces.len()
            && s.pieces.iter().all(|p| p.len() == sides)
            && s.vertices.iter().all(|p| (p.v.length() - 1.0).abs() < 1e-9);
        assert!(
            s.vertices.len() == v
                && s.edges.len() == e
                && s.faces.len() == f
                && s.euler_characteristic() == 2
                && s.is_convex()
                && regular,
            "{}: {v} vertices, {e} edges, {f} {sides}-gons, χ = 2, convex, regular: {} vertices, {} edges, {} faces",
            kind.name(),
            s.vertices.len(),
            s.edges.len(),
            s.faces.len()
        );
    }
}

/// "solids: on a convex solid the sign test is the whole test"
///
/// Every piece drawn is a whole edge, and the edges drawn are exactly the
/// ones with a face toward the eye -- under every camera, including ones a
/// degree off edge-on to a face, where a single sign decides a whole
/// pentagon's worth of edges.
#[test]
fn on_a_convex_solid_the_sign_test_is_the_whole_test() {
    for kind in Platonic::ALL {
        let s = Solid::platonic(kind, 1.0).resting(0);
        let (mut cameras, mut whole, mut by_signs, mut drawn_some, mut undecided) = (0, true, true, false, 0);
        for (_, view) in stress_cameras() {
            cameras += 1;
            let d = -view.projection_direction();
            let drawn = s.visible_edges(&view);
            // An edge with a face exactly edge-on -- a cube square to the
            // camera -- lies on the outline, and its sign is rounding;
            // whether it is drawn is not a question with an answer.
            let decidable = |e: &Edge| s.normals[e.left].dot(d).abs() > 1e-9 && s.normals[e.right].dot(d).abs() > 1e-9;
            let wanted: HashSet<Edge> = s
                .edges
                .iter()
                .filter(|e| decidable(e) && (s.normals[e.left].dot(d) > 0.0 || s.normals[e.right].dot(d) > 0.0))
                .copied()
                .collect();
            undecided += s.edges.iter().filter(|e| !decidable(e)).count();
            let mut got: HashSet<Edge> = HashSet::new();
            for p in drawn.paths() {
                let (first, last) = (p[0].v, p[p.len() - 1].v);
                let Some(e) = s.edges.iter().find(|e| {
                    let (a, b) = (s.vertices[e.a].v, s.vertices[e.b].v);
                    let len = (b - a).length();
                    let t0 = (first - a).dot(b - a) / (len * len);
                    let t1 = (last - a).dot(b - a) / (len * len);
                    (first - (a + t0 * (b - a))).length() < 1e-9
                        && (last - (a + t1 * (b - a))).length() < 1e-9
                        && t0 >= -1e-9
                        && t1 <= 1.0 + 1e-9
                }) else {
                    whole = false;
                    continue;
                };
                if !decidable(e) {
                    continue;
                }
                if (first - s.vertices[e.a].v).length() > 1e-12 || (last - s.vertices[e.b].v).length() > 1e-12 {
                    whole = false;
                }
                got.insert(*e);
            }
            if got != wanted {
                by_signs = false;
            }
            if drawn.count() > 0 {
                drawn_some = true;
            }
        }
        assert!(
            whole && by_signs && drawn_some,
            "{}: under {cameras} cameras every drawn piece is a whole edge, and the set is the sign test's: {}",
            kind.name(),
            if undecided > 0 { format!("{undecided} edge-on cases set aside") } else { String::new() }
        );
    }
}

/// "solids: the drawing is the same at every resolution, to the bit"
///
/// Swift bakes with Metal; the port bakes with `kurven_bake::bake`, which
/// renders no depth buffer (crates/kurven-bake/src/lib.rs), so the suite's
/// second check, "and the depth pass rasterized its faces", has nothing to
/// ask here and is left out. The strokes are compared as Swift compares them.
#[test]
fn the_drawing_is_the_same_at_every_resolution_to_the_bit() {
    let s = Solid::platonic(Platonic::Dodecahedron, 2.0).resting(0);
    let camera = Camera::oblique(&ObliqueProjection {
        shear: 0.3,
        x_angle: -55.0,
        z_angle: -70.0,
        flip_x: false,
        y_scale: None,
    });
    let layer = Layer {
        spec: LayerSpec::new("edges", LayerRole::Outline, LayerSource::Edges, 0.5, HeightPolicy::Surface),
        paths: s.edge_paths(),
    };
    let scene = Scene::solid_scene(s, vec![layer], camera, PreviewMode::Drawing, 0.0);
    let coarse = bake(&scene, &BakeOptions::new(64)).expect("the bake at 64");
    let fine = bake(&scene, &BakeOptions::new(4096)).expect("the bake at 4096");
    eprintln!("dodecahedron: {} edges drawn", coarse.strokes.path_count());
    assert!(
        coarse.strokes.layers[0].1 == fine.strokes.layers[0].1 && coarse.strokes.path_count() > 10,
        "a dodecahedron bakes the same strokes at 64 px and 4096 px: {} edges drawn",
        coarse.strokes.path_count()
    );
}

/// "solids: the Császár polyhedron is the seven-vertex torus, and its edges
/// are tested for occlusion"
#[test]
fn the_csaszar_polyhedron_is_the_seven_vertex_torus_and_its_edges_are_tested_for_occlusion() {
    let s = Solid::csaszar();
    assert!(
        s.vertices.len() == 7 && s.edges.len() == 21 && s.faces.len() == 14 && s.euler_characteristic() == 0,
        "7 vertices, 21 edges, 14 triangles, χ = 0"
    );
    let pairs: HashSet<(usize, usize)> = s.edges.iter().map(|e| (e.a, e.b)).collect();
    assert!(pairs.len() == 21, "every pair of vertices is an edge: K₇ on the torus");
    assert!(!s.is_convex(), "it is not convex");
    // Embedded: no two faces cross.
    let mut crossings = 0;
    for i in 0..s.pieces.len() {
        for j in (i + 1)..s.pieces.len() {
            let a: HashSet<usize> = s.pieces[i].iter().copied().collect();
            let shared = s.pieces[j].iter().filter(|k| a.contains(k)).count();
            let pa: Vec<Vec3> = s.pieces[i].iter().map(|&k| s.vertices[k].v).collect();
            let pb: Vec<Vec3> = s.pieces[j].iter().map(|&k| s.vertices[k].v).collect();
            if triangles_cross(&pa, &pb, shared) {
                crossings += 1;
            }
        }
    }
    assert!(crossings == 0, "no two faces cross: the surface is embedded");
    // The abstract triangulation: vertices mod 7, faces {i, i+1, i+3}
    // and {i, i+2, i+3}, under some relabelling.
    let sorted = |mut f: Vec<usize>| {
        f.sort();
        f
    };
    let mut abstract_faces: BTreeSet<Vec<usize>> = BTreeSet::new();
    for i in 0..7 {
        abstract_faces.insert(sorted(vec![i, (i + 1) % 7, (i + 3) % 7]));
        abstract_faces.insert(sorted(vec![i, (i + 2) % 7, (i + 3) % 7]));
    }
    let face_set: BTreeSet<Vec<usize>> = s.pieces.iter().map(|p| sorted(p.clone())).collect();
    fn permute(
        k: usize,
        perm: &mut [usize; 7],
        abstract_faces: &BTreeSet<Vec<usize>>,
        face_set: &BTreeSet<Vec<usize>>,
    ) -> bool {
        if k == 7 {
            let image: BTreeSet<Vec<usize>> = abstract_faces
                .iter()
                .map(|f| {
                    let mut g: Vec<usize> = f.iter().map(|&x| perm[x]).collect();
                    g.sort();
                    g
                })
                .collect();
            return image == *face_set;
        }
        for i in k..7 {
            perm.swap(k, i);
            let found = permute(k + 1, perm, abstract_faces, face_set);
            perm.swap(k, i);
            if found {
                return true;
            }
        }
        false
    }
    let relabelled = permute(0, &mut [0, 1, 2, 3, 4, 5, 6], &abstract_faces, &face_set);
    assert!(relabelled, "its faces are the K₇ triangulation {{i, i+1, i+3}}, {{i, i+2, i+3}} mod 7");

    // The sign test is not sufficient here: under some cameras an edge
    // that passes it runs partly behind a face, and occlusion splits it.
    // Every span drawn is held to a brute-force sampling of the edge, point
    // by point against every face.
    let meshes = piece_meshes(&s);
    let cameras = stress_cameras();
    let per_camera = par_map(&cameras, |(name, view)| {
        let d = -view.projection_direction();
        let drawn = s.visible_edges(view);
        let mut split = 0;
        let mut pieces: HashMap<Edge, Vec<(f64, f64)>> = HashMap::new();
        for p in drawn.paths() {
            let (first, last) = (p[0].v, p[p.len() - 1].v);
            for e in &s.edges {
                let (a, b) = (s.vertices[e.a].v, s.vertices[e.b].v);
                let len = (b - a).length();
                let t0 = (first - a).dot(b - a) / (len * len);
                let t1 = (last - a).dot(b - a) / (len * len);
                let off = (first - (a + t0 * (b - a))).length() + (last - (a + t1 * (b - a))).length();
                if off < 1e-9 && t0 >= -1e-9 && t1 <= 1.0 + 1e-9 {
                    pieces.entry(*e).or_default().push((t0, t1));
                    if t0 > 1e-9 || t1 < 1.0 - 1e-9 {
                        split += 1;
                    }
                    break;
                }
            }
        }
        let (mut compared, mut wrong) = (0, 0);
        let mut example = String::new();
        for e in &s.edges {
            let passes = s.normals[e.left].dot(d) > 0.0 || s.normals[e.right].dot(d) > 0.0;
            let (a, b) = (s.vertices[e.a].v, s.vertices[e.b].v);
            let runs = pieces.get(e).map_or(&[][..], |v| &v[..]);
            for k in 0..200 {
                let t = (k as f64 + 0.5) / 200.0;
                // Within a hair of a piece's end the truth is at a
                // boundary; elsewhere it is decided.
                if runs.iter().any(|x| (t - x.0).abs() < 1e-6 || (t - x.1).abs() < 1e-6) {
                    continue;
                }
                let p = a + t * (b - a);
                let seen_by_occlusion = runs.iter().any(|x| t > x.0 && t < x.1);
                // The brute force: a hit on any face but the edge's own
                // two, which contain the point.
                let hit = meshes
                    .iter()
                    .enumerate()
                    .any(|(f, mesh)| f != e.left && f != e.right && first_hit(mesh, p, d, 1e-9).is_some());
                let truth = passes && !hit;
                compared += 1;
                if truth != seen_by_occlusion {
                    wrong += 1;
                    if example.is_empty() {
                        example = format!(
                            "e.g. {name} edge {}-{} t={t:.3} truth {}",
                            e.a,
                            e.b,
                            if truth { "seen" } else { "hidden" }
                        );
                    }
                }
            }
        }
        (split, compared, wrong, example)
    });
    let (mut split, mut compared, mut wrong) = (0, 0, 0);
    let mut example = String::new();
    for (sp, c, w, ex) in per_camera {
        split += sp;
        compared += c;
        wrong += w;
        if example.is_empty() {
            example = ex;
        }
    }
    eprintln!("Császár: {split} split spans, {wrong} of {compared} sampled points disagree");
    assert!(
        split > 0,
        "under {} cameras some edges are drawn in pieces: the sign test alone would be wrong: {split} pieces are \
         parts of edges",
        cameras.len()
    );
    assert!(
        wrong == 0 && compared > 10_000,
        "and every span agrees with a point-by-point ray cast against the faces: {wrong} of {compared} sampled \
         points disagree {example}"
    );
}

/// "solids: the picture frame is a torus of ten faces, two of them with a hole"
#[test]
fn the_picture_frame_is_a_torus_of_ten_faces_two_of_them_with_a_hole() {
    let s = Solid::frame(4.0, 2.0, 1.0);
    let holes: usize = s.loops.iter().map(|l| l.len() - 1).sum();
    let edge_count = s.edge_paths().count();
    assert!(
        s.vertices.len() == 16
            && edge_count == 24
            && s.faces.len() == 10
            && holes == 2
            && s.euler_characteristic() == 0,
        "16 vertices, 24 edges, 10 faces, 2 holes: 16 − 24 + 10 − 2 = 0, one shell of genus one: {} vertices, \
         {edge_count} edges, {} faces, {holes} holes",
        s.vertices.len(),
        s.faces.len()
    );
    assert!(!s.is_convex(), "it is not convex");
    // The annuli are cut into convex pieces; the walls are one piece each.
    let sizes: Vec<usize> = s.faces.iter().map(|f| f.len()).collect();
    assert!(
        sizes.iter().skip(2).all(|&n| n == 1) && sizes.iter().take(2).all(|&n| n == 4),
        "each wall is one piece, and each annulus four, the fewest: each of the hole's four reflex corners needs a \
         diagonal of its own, and with one hole there are as many pieces as diagonals: pieces per face {sizes:?}"
    );
    let partition = partition_report(&s);
    assert!(
        partition.is_empty(),
        "the pieces are convex, tile their faces, and the partition counts close: {partition}"
    );
    // Every diagonal joins two pieces of one face, and every edge two
    // pieces of different faces.
    let sorted = s
        .edges
        .iter()
        .enumerate()
        .all(|(k, e)| (s.face_index_of_piece(e.left) == s.face_index_of_piece(e.right)) == s.is_diagonal[k]);
    assert!(sorted, "a diagonal lies within a face and an edge between two");
}

/// "solids: a face with holes is cut into convex pieces that tile it"
///
/// Prisms over faces a convex partition has to work for: reflex corners,
/// several holes, a hole near the outer loop, straight corners.
#[test]
fn a_face_with_holes_is_cut_into_convex_pieces_that_tile_it() {
    let v = Vec2::new;
    let square = vec![v(-2.0, -2.0), v(2.0, -2.0), v(2.0, 2.0), v(-2.0, 2.0)];
    // Clockwise: a hole.
    let rect = |x0: f64, y0: f64, x1: f64, y1: f64| vec![v(x0, y0), v(x0, y1), v(x1, y1), v(x1, y0)];
    // Five teeth: four slots cut down from the top, eight reflex corners.
    let mut comb = vec![v(0.0, 0.0), v(9.0, 0.0), v(9.0, 3.0)];
    for x in [8.0, 6.0, 4.0, 2.0] {
        comb.extend([v(x, 3.0), v(x, 1.0), v(x - 1.0, 1.0), v(x - 1.0, 3.0)]);
    }
    comb.push(v(0.0, 3.0));
    let star: Vec<Vec2> = (0..10)
        .map(|k| {
            let r = if k % 2 == 0 { 3.0 } else { 1.4 };
            let a = k as f64 * PI / 5.0;
            v(r * real::cos(a), r * real::sin(a))
        })
        .collect();
    let shapes: Vec<(&str, Vec<Vec<Vec2>>)> = vec![
        ("an L", vec![vec![v(0.0, 0.0), v(3.0, 0.0), v(3.0, 1.0), v(1.0, 1.0), v(1.0, 3.0), v(0.0, 3.0)]]),
        ("a comb", vec![comb]),
        ("a square with two holes", vec![square.clone(), rect(-1.5, -1.0, -0.5, 1.0), rect(0.5, -1.0, 1.5, 1.0)]),
        ("a square with a hole by its outer loop", vec![square.clone(), rect(1.0, -1.9, 1.9, 1.9)]),
        ("a star with a triangular hole", vec![star, vec![v(0.5, 0.0), v(-0.3, -0.4), v(-0.3, 0.4)]]),
        (
            "a square with straight corners",
            vec![
                vec![
                    v(-2.0, -2.0),
                    v(0.0, -2.0),
                    v(2.0, -2.0),
                    v(2.0, 0.0),
                    v(2.0, 2.0),
                    v(0.0, 2.0),
                    v(-2.0, 2.0),
                    v(-2.0, 0.0),
                ],
                vec![v(-1.0, -1.0), v(-1.0, 0.0), v(-1.0, 1.0), v(1.0, 1.0), v(1.0, -1.0), v(0.0, -1.0)],
            ],
        ),
    ];
    for (name, loops) in &shapes {
        let s = prism(loops, 0.5);
        let holes = loops.len() as i64 - 1;
        let pieces = s.faces[0].len();
        assert!(
            s.euler_characteristic() == 2 - 2 * holes,
            "{name}: the prism over it closes, a solid of genus {holes}: χ = {}",
            s.euler_characteristic()
        );
        let partition = partition_report(&s);
        assert!(partition.is_empty(), "{name}: {pieces} convex pieces tile the face, no vertex added: {partition}");
        let (wrong, compared, example) = edge_agreement(&s);
        eprintln!("{name}: {pieces} pieces, {wrong} of {compared} sampled points disagree");
        assert!(
            wrong == 0 && compared > 10_000,
            "{name}: every edge's drawn spans agree with a point-by-point ray cast: {wrong} of {compared} sampled \
             points disagree {example}"
        );
        // The partition depends on the face, not on where the solid is:
        // the same face a hundredth the size and 1000 from the origin is
        // cut into as many pieces.
        let scaled: Vec<Vec<Vec2>> = loops.iter().map(|l| l.iter().map(|&p| p / 100.0).collect()).collect();
        let far = prism(&scaled, 0.005).transformed(Mat3::IDENTITY, Vec3::new(1000.0, 0.0, 0.0));
        let far_partition = partition_report(&far);
        assert!(
            far.faces[0].len() == pieces && far_partition.is_empty(),
            "{name}, a hundredth the size and 1000 from the origin: {pieces} pieces as well: {} pieces {far_partition}",
            far.faces[0].len()
        );
    }
}

/// "solids: a small solid far from the origin is held to what its coordinates hold"
///
/// A millimetre across and millions out, tilted so that no face lies along
/// an axis. A coordinate there is rounded to about 2e-10, a hundred times a
/// billionth of a millimetre face, so a face held to the billionth alone is
/// bent by rounding and refused.
#[test]
fn a_small_solid_far_from_the_origin_is_held_to_what_its_coordinates_hold() {
    let far = Vec3::new(1e6, -2e6, 3e6);
    let cube = Solid::platonic(Platonic::Cube, 1e-3).transformed(tilt(), far);
    assert!(cube.euler_characteristic() == 2 && cube.is_convex(), "a cube is a convex sphere there");
    let frame = Solid::frame(2e-3, 1e-3, 5e-4).transformed(tilt(), far);
    let partition = partition_report(&frame);
    let sizes: Vec<usize> = frame.faces.iter().map(|f| f.len()).collect();
    assert!(
        frame.euler_characteristic() == 0 && !frame.is_convex() && sizes.iter().take(2).all(|&n| n == 4),
        "a picture frame is a torus there, not convex, its annuli four pieces each: χ = {}, pieces per face {sizes:?}",
        frame.euler_characteristic()
    );
    assert!(partition.is_empty(), "and its pieces tile their faces: {partition}");
    // Rounding alone takes the cube's vertices further off their faces'
    // planes than a billionth of a face, the tolerance without a floor.
    let mut bent = 0.0;
    for (i, f) in cube.frames.iter().enumerate() {
        for &k in cube.loops[i].iter().flatten() {
            bent = swift::max(bent, (cube.vertices[k].v - f.origin).dot(f.normal).abs() / f.extent);
        }
    }
    assert!(bent > 1e-9, "rounding alone bends the cube's faces by more than a billionth: {bent} of a face");
    // Refusal traps, so a bent face is not built here; what can be held is
    // the tolerance a bend would be measured against.
    let loosest = cube
        .frames
        .iter()
        .chain(&frame.frames)
        .map(|f| f.flatness_tolerance / f.extent)
        .reduce(swift::max)
        .unwrap_or(0.0);
    assert!(
        loosest < 1e-4,
        "a face bent by a ten-thousandth of its size there is still refused: the loosest flatness tolerance is \
         {loosest} of its face"
    );
}

/// "solids: convexity is decided at the solid's own size, wherever it stands"
///
/// A prism over a pentagon whose fourth corner is pushed in by 1e-4, two
/// faces of 1.5 meeting at a turn of 1e-4: not convex, by a margin a
/// billionth of the distance from the origin a million out would swallow.
#[test]
fn convexity_is_decided_at_the_solids_own_size_wherever_it_stands() {
    let dent = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(3.0, 0.0),
        Vec2::new(3.0, 1.0),
        Vec2::new(1.5, 1.0 - 1e-4),
        Vec2::new(0.0, 1.0),
    ];
    let unit = Solid::platonic(Platonic::Cube, 1.0);
    for far in [Vec3::new(0.0, 0.0, 0.0), Vec3::new(1e6, -2e6, 3e6)] {
        let dented = prism(std::slice::from_ref(&dent), 1.0).transformed(tilt(), far);
        let cube = unit.transformed(tilt(), far);
        assert!(
            cube.is_convex() && !dented.is_convex(),
            "{} from the origin, a cube is convex and a prism dented by 1e-4 is not",
            far.length()
        );
    }
}

/// "solids: a face with thousands of corners is cut in reasonable time"
///
/// Ear clipping that went back to the first corner after every ear, and
/// merging that went back to the first side after every merge, took seconds
/// on these. A star, every other corner reflex: the pieces inside merge into
/// one polygon a triangle at a time. Timed with `Instant` where Swift times
/// with `ContinuousClock`, to Swift's second (a release build).
#[test]
fn a_face_with_thousands_of_corners_is_cut_in_reasonable_time() {
    let star: Vec<Vec2> = (0..2000)
        .map(|k| {
            let r = if k % 2 == 0 { 10.0 } else { 7.0 };
            let a = k as f64 * PI / 1000.0;
            Vec2::new(r * real::cos(a), r * real::sin(a))
        })
        .collect();
    // A comb of 1,000 teeth, as the comb above: its back merges a tooth at
    // a time.
    let mut comb = vec![Vec2::new(0.0, 0.0), Vec2::new(1999.0, 0.0), Vec2::new(1999.0, 3.0)];
    let mut x = 1998.0;
    while x >= 2.0 {
        comb.extend([Vec2::new(x, 3.0), Vec2::new(x, 1.0), Vec2::new(x - 1.0, 1.0), Vec2::new(x - 1.0, 3.0)]);
        x -= 2.0;
    }
    comb.push(Vec2::new(0.0, 3.0));
    // A strip wound into a spiral of some fifty turns: out along its outer
    // side, back along its inner. Its ears lie far round the ring.
    let turns: Vec<(Vec2, f64)> = (0..2000)
        .map(|k| {
            let t = k as f64 * 0.05;
            (Vec2::new(real::cos(3.0 * t), real::sin(3.0 * t)), 1.0 + t)
        })
        .collect();
    let mut spiral: Vec<Vec2> = turns.iter().map(|&(d, r)| d * (r + 0.4)).collect();
    spiral.extend(turns.iter().rev().map(|&(d, r)| d * r));
    for (name, face) in [
        ("a star of 2,000 corners", star),
        ("a comb of 1,000 teeth", comb),
        ("a spiral strip of 4,000 corners", spiral),
    ] {
        let start = Instant::now();
        let s = prism(&[face], 1.0);
        let elapsed = start.elapsed();
        let top: Vec<&Vec<usize>> = s.faces[0].iter().map(|&p| &s.pieces[p]).collect();
        let polygon = |l: &[usize]| l.iter().map(|&k| s.vertices[k].v).collect::<Vec<_>>();
        let area = top.iter().fold(0.0, |acc, piece| acc + s.frames[0].area(&polygon(piece)));
        let face_area = s.frames[0].area(&polygon(&s.loops[0][0]));
        assert!(
            elapsed < Duration::from_secs(1) && (area - face_area).abs() <= 1e-9 * face_area,
            "{name}: a prism over it is built, its top cut into {} pieces that cover it: {elapsed:?}, area {area} of \
             {face_area}",
            top.len()
        );
        eprintln!("{name}: {} pieces in {elapsed:?}", top.len());
    }
}

/// "solids: a face that touches itself at a vertex is one loop through it twice"
///
/// A box 4 by 4 by 2 with pits sunk in its top. Where a pit's mouth meets
/// the top face's outer loop, or another pit's mouth, the top face's
/// boundary is one loop that passes the shared vertex twice: H counts the
/// boundary's components, and Euler–Poincaré holds. Not prisms: over such a
/// face, the vertical edge at the shared vertex would bound four walls.
#[test]
fn a_face_that_touches_itself_at_a_vertex_is_one_loop_through_it_twice() {
    let v = Vec2::new;
    let corners = [v(0.0, 0.0), v(4.0, 0.0), v(4.0, 4.0), v(0.0, 4.0)];
    let with = |more: &[Vec2]| -> Vec<Vec2> { corners.iter().chain(more).copied().collect() };
    let cases: Vec<(&str, Solid, usize)> = vec![
        // A pit with a mouth corner on the box's corner 0: the outer loop
        // runs round the box, back to 0, then round the mouth clockwise.
        (
            "a pit touching the outer loop",
            pitted_box(
                &with(&[v(1.5, 0.5), v(2.0, 2.0), v(0.5, 1.5)]),
                &[vec![0, 4, 5, 6]],
                vec![vec![0, 1, 2, 3, 0, 6, 5, 4]],
            ),
            0,
        ),
        // Two pits whose mouths share the corner 7: one hole, round the left
        // mouth and then the right, both clockwise.
        (
            "two pits touching each other",
            pitted_box(
                &with(&[v(1.4, 1.4), v(0.8, 2.0), v(1.4, 2.6), v(2.0, 2.0), v(2.6, 1.4), v(3.2, 2.0), v(2.6, 2.6)]),
                &[vec![5, 4, 7, 6], vec![7, 8, 9, 10]],
                vec![vec![0, 1, 2, 3], vec![7, 4, 5, 6, 7, 10, 9, 8]],
            ),
            1,
        ),
    ];
    for (name, s, holes) in &cases {
        let (vn, en, fn_) = (s.vertices.len(), s.edge_paths().count(), s.faces.len());
        assert!(
            s.euler_characteristic() == 2 && s.loops[0].len() == holes + 1,
            "{name}: {vn} − {en} + {fn_} − {holes} = 2, a sphere: χ = {}",
            s.euler_characteristic()
        );
        let partition = partition_report(s);
        assert!(partition.is_empty(), "{name}: {} convex pieces tile the top face: {partition}", s.faces[0].len());
        let (wrong, compared, example) = edge_agreement(s);
        eprintln!("{name}: {} pieces on top, {wrong} of {compared} sampled points disagree", s.faces[0].len());
        assert!(
            wrong == 0 && compared > 10_000,
            "{name}: every edge's drawn spans agree with a point-by-point ray cast: {wrong} of {compared} sampled \
             points disagree {example}"
        );
    }
}

/// "solids: on the frame no diagonal is drawn, and the back's inner edges
/// are seen through the hole in part"
#[test]
fn on_the_frame_no_diagonal_is_drawn_and_the_backs_inner_edges_are_seen_through_the_hole_in_part() {
    let s = Solid::frame(4.0, 2.0, 1.0);
    let meshes = piece_meshes(&s);
    let cameras = stress_cameras();
    let per_camera = par_map(&cameras, |(name, view)| {
        let d = -view.projection_direction();
        let (spans, off_edges, on_diagonal) = drawn_spans(&s, view);
        // The back's inner edges -- 4-5, 5-6, 6-7, 4-7 -- lie at the bottom
        // of the hole; from above they show only through the hole, and only
        // in part: the near side of the front cuts them.
        let mut through_hole = 0;
        for e in s.edges.iter().filter(|e| e.a >= 4 && e.a < 8 && e.b >= 4 && e.b < 8) {
            for &(t0, t1) in spans.get(e).map_or(&[][..], |v| &v[..]) {
                if t0 > 1e-9 || t1 < 1.0 - 1e-9 {
                    through_hole += 1;
                }
            }
        }
        // An edge with a face exactly edge-on -- a wall of the frame square
        // to the camera -- lies on the outline, and whether it is drawn is
        // decided by rounding: set aside, as for the Platonic solids.
        let (mut wrong, mut compared) = (0, 0);
        let mut example = String::new();
        for (k, e) in s.edges.iter().enumerate() {
            if s.is_diagonal[k] || !(s.normals[e.left].dot(d).abs() > 1e-9 && s.normals[e.right].dot(d).abs() > 1e-9) {
                continue;
            }
            let (w, c, ex) = occlusion_agreement(&s, &meshes, *e, spans.get(e).map_or(&[][..], |v| &v[..]), d, name);
            wrong += w;
            compared += c;
            if example.is_empty() {
                example = ex;
            }
        }
        (on_diagonal, off_edges, through_hole, wrong, compared, example)
    });
    let (mut on_diagonal, mut off_edges, mut through_hole, mut wrong, mut compared) = (0, 0, 0, 0, 0);
    let mut example = String::new();
    for (od, oe, th, w, c, ex) in per_camera {
        on_diagonal += od;
        off_edges += oe;
        through_hole += th;
        wrong += w;
        compared += c;
        if example.is_empty() {
            example = ex;
        }
    }
    eprintln!(
        "frame: {on_diagonal} on a diagonal, {off_edges} off the edges, {through_hole} through the hole, {wrong} of \
         {compared} disagree"
    );
    assert!(
        on_diagonal == 0 && off_edges == 0,
        "under {} cameras everything drawn lies on an edge, and nothing on a diagonal: {on_diagonal} spans on a \
         diagonal, {off_edges} on no side at all",
        cameras.len()
    );
    assert!(
        through_hole > 0,
        "the back's inner edges are drawn in part: seen through the hole, cut by the front: {through_hole} spans"
    );
    assert!(
        wrong == 0 && compared > 10_000,
        "and every span agrees with a point-by-point ray cast against the pieces: {wrong} of {compared} sampled \
         points disagree {example}"
    );
}

/// "solids: a curve on a face breaks where it goes out of sight, and is not
/// joined across"
///
/// A frame 2 deep, seen from low over -y along (0, 2, -1): picture
/// (x, y + 2z), depth 2z - y, every coordinate exact. The far wall of the
/// hole, y = 1, is hidden below z = 1 by the near half of the frame, whose
/// top inner edge projects onto it exactly there.
#[test]
fn a_curve_on_a_face_breaks_where_it_goes_out_of_sight_and_is_not_joined_across() {
    let s = Solid::frame(4.0, 2.0, 2.0);
    let view = Transform::<WorldSpace, ViewSpace>::from_rows(
        (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 2.0), Vec3::new(0.0, -1.0, 2.0)),
        Vec3::ZERO,
    );
    let occlusion = s.occlusion(&view);
    let wall = (0..s.faces.len()).find(|&i| {
        let f = s.frames[i];
        (f.normal - Vec3::new(0.0, -1.0, 0.0)).length() < 1e-12 && (f.origin.y - 1.0).abs() < 1e-12
    });
    let Some(wall) = wall else { panic!("the frame has a far inner wall") };
    // Down the wall to the line of sight's edge, on into the hidden part,
    // across it, and back up: seen, hidden, hidden, hidden, seen.
    let path: Vec<P3<WorldSpace>> = vec![
        P3::new(-0.5, 1.0, 2.0),
        P3::new(-0.5, 1.0, 1.0),
        P3::new(-0.5, 1.0, 0.5),
        P3::new(0.5, 1.0, 0.5),
        P3::new(0.5, 1.0, 1.0),
        P3::new(0.5, 1.0, 2.0),
    ];
    let drawn = s.visible_curves(std::slice::from_ref(&path), wall, &occlusion);
    assert!(
        drawn == vec![path[0..2].to_vec(), path[4..6].to_vec()],
        "a path on the far wall that dips out of sight is drawn as its two seen ends: {:?}",
        drawn.iter().map(|p| p.iter().map(|q| q.v).collect::<Vec<_>>()).collect::<Vec<_>>()
    );
}

/// "solids: loops that bound no solid are refused, saying why, not trapped on"
///
/// A cube 2 on a side, its corners 0-3 below and 4-7 above, each
/// counter-clockwise from above, and its faces outward; each case breaks it
/// in one way.
#[test]
fn loops_that_bound_no_solid_are_refused_saying_why_not_trapped_on() {
    let p = |x: f64, y: f64, z: f64| P3::<WorldSpace>::new(x, y, z);
    let corners: Vec<P3<WorldSpace>> = vec![
        p(0.0, 0.0, 0.0),
        p(2.0, 0.0, 0.0),
        p(2.0, 2.0, 0.0),
        p(0.0, 2.0, 0.0),
        p(0.0, 0.0, 2.0),
        p(2.0, 0.0, 2.0),
        p(2.0, 2.0, 2.0),
        p(0.0, 2.0, 2.0),
    ];
    let cube: Vec<Vec<Vec<i64>>> = vec![
        vec![vec![0, 3, 2, 1]],
        vec![vec![4, 5, 6, 7]],
        vec![vec![0, 1, 5, 4]],
        vec![vec![1, 2, 6, 5]],
        vec![vec![2, 3, 7, 6]],
        vec![vec![3, 0, 4, 7]],
    ];
    assert!(
        Solid::checking(corners.clone(), cube.clone()).ok().map(|s| s.euler_characteristic()) == Some(2),
        "the cube itself is a solid, χ = 2"
    );
    let refused =
        |what: &str, vertices: &[P3<WorldSpace>], faces: Vec<Vec<Vec<i64>>>, phrase: &str| match Solid::checking(
            vertices.to_vec(),
            faces,
        ) {
            Ok(_) => panic!("refuses {what}: built a solid"),
            Err(e) => assert!(format!("{e}").contains(phrase), "refuses {what}: {e}"),
        };
    let with = |face: usize, loops: Vec<Vec<i64>>| -> Vec<Vec<Vec<i64>>> {
        let mut faces = cube.clone();
        faces[face] = loops;
        faces
    };
    let plus = |more: &[P3<WorldSpace>]| -> Vec<P3<WorldSpace>> { corners.iter().chain(more).copied().collect() };
    // A square hole in the top, 0.5 to 1.5, for the cases about holes.
    let holed = plus(&[
        p(0.5, 0.5, 2.0),
        p(1.5, 0.5, 2.0),
        p(1.5, 1.5, 2.0),
        p(0.5, 1.5, 2.0),
        p(3.0, 0.5, 2.0),
        p(3.5, 0.5, 2.0),
        p(3.5, 1.0, 2.0),
        p(1.0, 0.0, 2.0),
        p(1.0, 0.5, 2.0),
    ]);
    refused("no faces", &corners, vec![], "has none");
    refused("a face with no loop", &corners, with(1, vec![]), "no loop");
    refused("a loop of two vertices", &corners, with(1, vec![vec![4, 5]]), "at least three");
    refused("a vertex index past the last", &corners, with(1, vec![vec![4, 5, 6, 8]]), "names vertex 8");
    refused("a negative vertex index", &corners, with(1, vec![vec![4, 5, 6, -1]]), "names vertex -1");
    let mut nan = corners.clone();
    nan[6] = p(2.0, 2.0, f64::NAN);
    refused("a coordinate that is not a number", &nan, cube.clone(), "not a finite point");
    let mut infinite = corners.clone();
    infinite[2] = p(f64::INFINITY, 2.0, 0.0);
    refused("an infinite coordinate", &infinite, cube.clone(), "not a finite point");
    refused("a side from a vertex to itself", &corners, with(1, vec![vec![4, 5, 5, 6, 7]]), "no length");
    refused(
        "a side between two vertices at one point",
        &plus(&[p(2.0, 0.0, 2.0)]),
        with(1, vec![vec![4, 5, 8, 6, 7]]),
        "no length",
    );
    refused(
        "two loops of a face sharing a vertex",
        &holed,
        with(1, vec![vec![4, 5, 6, 7], vec![8, 4, 10]]),
        "share vertex 4",
    );
    refused("a face of no area", &plus(&[p(4.0, 0.0, 2.0)]), with(1, vec![vec![4, 5, 8]]), "encloses no area");
    let mut bent = corners.clone();
    bent[6] = p(2.0, 2.0, 2.1);
    refused("a face that is not flat", &bent, cube.clone(), "not flat");
    refused(
        "a hole running the same way as its face",
        &holed,
        with(1, vec![vec![4, 5, 6, 7], vec![8, 9, 10, 11]]),
        "runs the same way",
    );
    refused("a hole outside its face", &holed, with(1, vec![vec![4, 5, 6, 7], vec![12, 14, 13]]), "not inside it");
    refused("a loop that crosses itself", &plus(&[p(1.0, -0.5, 2.0)]), with(1, vec![vec![4, 5, 6, 7, 8]]), "cross");
    refused(
        "a hole whose vertex lies on the outer loop's side",
        &holed,
        with(1, vec![vec![4, 5, 6, 7], vec![15, 16, 9]]),
        "lies on side 4-5",
    );
    refused("a vertex on no face", &plus(&[p(1e6, 1e6, 1e6)]), cube.clone(), "vertex 8 lies on no face");
    refused("an open surface", &corners, cube[..cube.len() - 1].to_vec(), "not closed");
    refused(
        "a face turned the other way from its neighbours",
        &corners,
        with(1, vec![vec![7, 6, 5, 4]]),
        "traversed twice the same way",
    );
    let inward: Vec<Vec<Vec<i64>>> =
        cube.iter().map(|f| f.iter().map(|l| l.iter().rev().copied().collect()).collect()).collect();
    refused("faces all turned inward", &corners, inward, "oriented inward");
    let split = plus(&[p(1.0, 0.0, 2.0), p(1.0, 2.0, 2.0)]);
    refused(
        "two faces in one plane",
        &split,
        vec![
            vec![vec![0, 3, 2, 1]],
            vec![vec![4, 8, 9, 7]],
            vec![vec![8, 5, 6, 9]],
            vec![vec![0, 1, 5, 8, 4]],
            vec![vec![1, 2, 6, 5]],
            vec![vec![2, 3, 7, 9, 6]],
            vec![vec![3, 0, 4, 7]],
        ],
        "in one plane",
    );
}

// MARK: - solidPlacementTests

/// "solids: a boundary of several shells has no one genus"
#[test]
fn a_boundary_of_several_shells_has_no_one_genus() {
    let cube = Solid::platonic(Platonic::Cube, 1.0);
    let frame = Solid::frame(4.0, 2.0, 0.8);
    let mut apart_vertices = cube.vertices.clone();
    apart_vertices.extend(cube.vertices.iter().map(|p| P3::from_vec(p.v + Vec3::new(5.0, 0.0, 0.0))));
    let mut apart_loops = index_loops(&cube.loops, 0, false);
    apart_loops.extend(index_loops(&cube.loops, 8, false));
    let apart = Solid::new(apart_vertices, apart_loops);
    // A cube 4 across with a cube 2 across hollowed out of it: the inner
    // shell's faces turned to face into the cavity.
    let mut hollow_vertices: Vec<P3<WorldSpace>> = cube.vertices.iter().map(|p| P3::from_vec(2.0 * p.v)).collect();
    hollow_vertices.extend(cube.vertices.iter().copied());
    let mut hollow_loops = index_loops(&cube.loops, 0, false);
    hollow_loops.extend(index_loops(&cube.loops, 8, true));
    let hollow = Solid::new(hollow_vertices, hollow_loops);
    assert!(
        cube.shell_count() == 1 && cube.genus() == Some(0) && frame.shell_count() == 1 && frame.genus() == Some(1),
        "a cube is one shell of genus 0, the frame one of genus 1"
    );
    assert!(
        apart.shell_count() == 2 && apart.genus().is_none() && apart.euler_characteristic() == 4,
        "two cubes apart are two shells, χ = 4, and no genus: {} shells, χ = {}",
        apart.shell_count(),
        apart.euler_characteristic()
    );
    assert!(
        hollow.shell_count() == 2 && hollow.genus().is_none() && hollow.euler_characteristic() == 4,
        "a cube with a cavity is two shells, χ = 4, and no genus: {} shells, χ = {}",
        hollow.shell_count(),
        hollow.euler_characteristic()
    );
}

/// "solids: a solid far from the origin draws what it draws at the origin"
///
/// The occlusion test's tolerances are fractions of the solid's own size in
/// view, not of its distance from the origin: a frame moved a million away
/// bakes the same edges and hatching, moved with it.
#[test]
fn a_solid_far_from_the_origin_draws_what_it_draws_at_the_origin() {
    let frame = Solid::frame(4.0, 2.0, 0.8);
    let hatching = Hatching::new(PI / 6.0, 0.035, None, Tone::Turing { scale: 0.18, seed: 3 }, 4);
    let camera = Camera::oblique(&SurfaceRequest::PROJECTION);
    let strokes = |solid: Solid| -> Vec<Vec<Vec3>> {
        let edges = LayerSpec::new("edges", LayerRole::Outline, LayerSource::Edges, 0.6, HeightPolicy::Surface);
        let hatch = LayerSpec::new(
            "hatching",
            LayerRole::Scaffold,
            LayerSource::Hatching { hatching, faces: FaceSelector::Solid },
            0.5,
            HeightPolicy::Surface,
        );
        let layers = vec![
            Layer { spec: edges, paths: solid.edge_paths() },
            Layer { spec: hatch, paths: solid.hatch_paths(&hatching) },
        ];
        let scene = Scene::solid_scene(solid, layers, camera, PreviewMode::Drawing, 0.0);
        let back = camera.view.inverse();
        scene
            .visible_layers(None, None)
            .expect("a solid's layers")
            .iter()
            .flat_map(|(_, seen)| {
                seen.paths().map(|p| p.iter().map(|&q| back.apply(q).v).collect()).collect::<Vec<_>>()
            })
            .collect()
    };
    let near = strokes(frame.clone());
    for shift in [Vec3::new(1e6, 0.0, 0.0), Vec3::new(-3e5, 2e5, 1e5)] {
        let far = strokes(frame.transformed(Mat3::IDENTITY, shift));
        let mut worst = 0.0;
        for (a, b) in near.iter().zip(&far) {
            if a.len() != b.len() {
                worst = f64::INFINITY;
                break;
            }
            for (&p, &q) in a.iter().zip(b) {
                worst = swift::max(worst, (p + shift - q).length());
            }
        }
        eprintln!("moved by {shift:?}: {} strokes near, {} far, the farthest {worst:e} off", near.len(), far.len());
        assert!(
            near.len() > 2000 && far.len() == near.len() && worst < 1e-6,
            "moved by {shift:?}, the frame's {} strokes are drawn, each moved with it: {} strokes, the farthest \
             {worst} off",
            near.len(),
            far.len()
        );
    }
}
