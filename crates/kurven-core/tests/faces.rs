//! A solid's faces: the port of the `faces:` suites of
//! `Sources/kurven-test/Faces.swift` that build faces without the refiner,
//! Hatch.swift or Scene's visibility, with Swift's tolerances.

// `if !(x > 0.0)` is Swift's `guard x > 0`, which refuses a NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use std::collections::BTreeMap;
use std::sync::Arc;

use kurven_core::bundle::KurvenBundle;
use kurven_core::camera::Camera;
use kurven_core::face::{Face, FaceCopy, FaceID};
use kurven_core::fixtures;
use kurven_core::hatching::{FaceKind, FaceSelector, Hatching};
use kurven_core::manifest::{
    ContourField, HeightPolicy, Interval, LayerRole, LayerSource, LayerSpec, ObliqueProjection, PolyhedronManifest,
    Restriction,
};
use kurven_core::parametric::{JetMap, ParamAxis, ParametricSurface, SurfaceJet};
use kurven_core::polyline::{Layer, PolylineSet};
use kurven_core::scalar_field::ScalarField;
use kurven_core::scene::{PreviewMode, Scene, SceneGeometry};
use kurven_core::simd::{Vec2, Vec3};
use kurven_core::solid::{Platonic, Solid};
use kurven_core::spaces::{P2, P3, ParamSpace, WorldSpace};
use kurven_math::{real, swift};

/// The parameter at which `copy` of `face` passes through `q`, if it does.
/// The graph and the plateaus lie over the domain, so a point's parameter is
/// its (x, y); every other face of a landscape or a polyhedron is flat with
/// an affine map, and a point's parameter is its projection onto the face's
/// plane. The map returning the point is what decides.
fn parameter(q: P3<WorldSpace>, face: &Face, copy: &FaceCopy) -> Option<P2<ParamSpace>> {
    let back = copy.tile.inverse()?;
    let local = back.apply(q).v;
    let p: P2<ParamSpace> = match face.id {
        FaceID::Graph | FaceID::Plateau { .. } => P2::new(local.x, local.y),
        _ => {
            let p0 = P2::<ParamSpace>::new(face.domain.u.range.lo, face.domain.v.range.lo);
            let jet = (face.map)(p0);
            let r = local - (jet.position - jet.du * p0.x() - jet.dv * p0.y());
            let (a, b, c) = (jet.du.dot(jet.du), jet.du.dot(jet.dv), jet.dv.dot(jet.dv));
            let det = a * c - b * b;
            if !(det > 0.0) {
                return None;
            }
            let (ru, rv) = (jet.du.dot(r), jet.dv.dot(r));
            P2::new((c * ru - b * rv) / det, (a * rv - b * ru) / det)
        }
    };
    let tolerance = 1e-9 * local.length().max(1.0);
    if !((copy.contains)(p) && ((face.map)(p).position - local).length() <= tolerance) {
        return None;
    }
    Some(p)
}

/// Every copy of a face that holds a point.
fn holders(q: P3<WorldSpace>, faces: &[Face]) -> Vec<FaceID> {
    faces.iter().flat_map(|f| f.copies.iter().filter_map(move |c| parameter(q, f, c).map(|_| f.id))).collect()
}

/// The outward normal of a copy at a parameter, in space: zero where the
/// face bounds no solid.
fn outward_normal(face: &Face, copy: &FaceCopy, p: P2<ParamSpace>) -> Vec3 {
    copy.place((face.map)(p)).normal() * copy.outward.unwrap_or(0.0)
}

fn spec(name: &str, role: LayerRole, source: LayerSource) -> LayerSpec {
    LayerSpec::new(name, role, source, 1.0, HeightPolicy::Surface)
}

fn faces_of(scene: &Scene) -> BTreeMap<String, Option<FaceKind>> {
    scene.layers().iter().map(|l| (l.spec.name.clone(), scene.geometry().face_of(&l.spec))).collect()
}

/// The contract's polyhedra, from their fixtures: the five Platonic solids,
/// the picture frame and Császár's torus (Swift builds them by name, which
/// is Phase 5's; the fixtures are those solids, as Swift wrote them).
fn polyhedra() -> Vec<(&'static str, Solid)> {
    ["tetrahedron", "cube", "octahedron", "dodecahedron", "icosahedron", "frame", "csaszar"]
        .into_iter()
        .map(|name| {
            let json = fixtures::shared_json(&format!("contract/polyhedron_{name}.kurven/manifest.json"));
            (name, PolyhedronManifest::from_json(&json).expect("a polyhedron manifest").solid)
        })
        .collect()
}

/// "faces: each layer says the kind of face its curves lie on"
///
/// The landscape is the contract's, as in Swift. Swift's parametric surface
/// is sn on its torus, Phase 4's; the rule reads only the layers' roles and
/// sources, so here a patch carries sn's layers. Its polyhedron is the
/// picture frame, here the contract's fixture of it.
#[test]
fn each_layer_says_the_kind_of_face_its_curves_lie_on() {
    let camera =
        Camera::oblique(&ObliqueProjection { shear: 0.0, x_angle: -55.0, z_angle: 30.0, flip_x: false, y_scale: None });
    let bundle = KurvenBundle::read(fixtures::shared("contract/landscape.kurven")).expect("the landscape fixture");
    let landscape = faces_of(&Scene::from_bundle(&bundle, &bundle.manifest.presets[0]));
    let on_landscape: BTreeMap<String, Option<FaceKind>> = [
        ("mag_major", Some(FaceKind::Graph)),
        ("truncation_edge", None),
        ("plateau_hatch", Some(FaceKind::Plateau)),
        ("plateau_hatch_tiled", Some(FaceKind::Plateau)),
        ("wall_edges", Some(FaceKind::Wall)),
        ("wall_hatch", Some(FaceKind::Wall)),
        ("folds", None),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    assert_eq!(
        landscape, on_landscape,
        "a landscape: contours on the graph, hatching on the plateaus and walls, the walls' edges on the walls; \
         the truncation edge and the folds on none"
    );

    let u = ParamAxis::new(Interval::new(0.0, 1.0), false, 4);
    let map: JetMap = Arc::new(|c: P2<ParamSpace>| {
        SurfaceJet::new(Vec3::new(c.x(), c.y(), 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0))
    });
    let patch = ParametricSurface::new(u, u, false, None, None, map).unwrap();
    let contour = |field| LayerSource::Contour {
        field,
        levels: vec![1.0],
        restriction: Restriction::All,
        faces: FaceSelector::Surface,
        light: None,
    };
    let layer = |s: LayerSpec| Layer { spec: s, paths: PolylineSet::empty() };
    let layers = vec![
        layer(spec("mag_major", LayerRole::Magnitude, contour(ContourField::Magnitude))),
        layer(spec("mag_minor", LayerRole::Magnitude, contour(ContourField::Magnitude))),
        layer(spec("ang_major", LayerRole::Phase, contour(ContourField::Phase))),
        layer(spec("ang_minor", LayerRole::Phase, contour(ContourField::Phase))),
        layer(spec("folds", LayerRole::Outline, LayerSource::FoldCurves)),
        layer(spec(
            "truncation edge",
            LayerRole::Outline,
            LayerSource::Contour {
                field: ContourField::Magnitude,
                levels: vec![2.0],
                restriction: Restriction::All,
                faces: FaceSelector::Graph { tiled: false },
                light: None,
            },
        )),
        // Folds are edge-on whatever their role.
        layer(spec("folds drawn as scaffold", LayerRole::Scaffold, LayerSource::FoldCurves)),
    ];
    let torus = faces_of(&Scene::parametric_scene(patch, layers, camera, PreviewMode::Drawing, 0.0));
    let on_torus: BTreeMap<String, Option<FaceKind>> = [
        ("mag_major", Some(FaceKind::Surface)),
        ("mag_minor", Some(FaceKind::Surface)),
        ("ang_major", Some(FaceKind::Surface)),
        ("ang_minor", Some(FaceKind::Surface)),
        ("folds", None),
        ("truncation edge", None),
        ("folds drawn as scaffold", None),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    assert_eq!(torus, on_torus, "a parametric surface: every curve on its one face but the folds and an outline");

    let frame = polyhedra().into_iter().find(|(n, _)| *n == "frame").unwrap().1;
    let hatching = Hatching::plain(0.5, 0.1);
    let solid = faces_of(&Scene::solid_scene(
        frame.clone(),
        vec![
            Layer { spec: spec("edges", LayerRole::Outline, LayerSource::Edges), paths: frame.edge_paths() },
            Layer {
                spec: spec(
                    "hatching",
                    LayerRole::Scaffold,
                    LayerSource::Hatching { hatching, faces: FaceSelector::Solid },
                ),
                paths: PolylineSet::empty(),
            },
        ],
        camera,
        PreviewMode::Drawing,
        0.0,
    ));
    let on_solid: BTreeMap<String, Option<FaceKind>> =
        [("edges".to_string(), None), ("hatching".to_string(), Some(FaceKind::Solid))].into_iter().collect();
    assert_eq!(solid, on_solid, "a polyhedron: its hatching on its faces, its edges on none");
}

/// "faces: a polyhedron's faces are its loops, and membership agrees with the convex pieces"
///
/// Over the contract's polyhedra (Swift names them; see `polyhedra`).
#[test]
fn a_polyhedrons_faces_are_its_loops_and_membership_agrees_with_the_convex_pieces() {
    for (name, s) in polyhedra() {
        let faces = SceneGeometry::Solid(s.clone()).faces();
        let (mut mismatches, mut untiled, mut inward, mut inside) = (0, 0, 0, 0);
        for (i, face) in faces.iter().enumerate() {
            let frame = s.frames[i];
            let copy = &face.copies[0];
            let plane = |k: usize| frame.coordinates(s.vertices[k].v);
            let pieces: Vec<(usize, Vec<Vec2>)> =
                s.faces[i].iter().map(|&piece| (piece, s.pieces[piece].iter().map(|&k| plane(k)).collect())).collect();
            let crossings = |p: Vec2, polygon: &[Vec2]| -> usize {
                (0..polygon.len())
                    .filter(|&k| {
                        let (a, b) = (polygon[k], polygon[(k + 1) % polygon.len()]);
                        (a.y > p.y) != (b.y > p.y) && p.x < a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y)
                    })
                    .count()
            };
            let near = |p: Vec2, polygon: &[Vec2]| -> bool {
                (0..polygon.len()).any(|k| {
                    let (a, b) = (polygon[k], polygon[(k + 1) % polygon.len()]);
                    let t = ((p - a).dot(b - a) / (b - a).length_squared()).clamp(0.0, 1.0);
                    (p - (a + t * (b - a))).length() < 1e-6
                })
            };
            // The partition suite's grid: each point in the face or not, and
            // in as many pieces; points near a side are left out.
            let all: Vec<Vec2> = s.loops[i].iter().flatten().map(|&k| plane(k)).collect();
            let lo = all.iter().fold(all[0], |a, &b| a.min(b));
            let hi = all.iter().fold(all[0], |a, &b| a.max(b));
            for a in 0..61 {
                for b in 0..61 {
                    let f = Vec2::new(a as f64 + 0.37, b as f64 + 0.61) / 61.0;
                    let p = lo + Vec2::new((hi - lo).x * f.x, (hi - lo).y * f.y);
                    if pieces.iter().any(|(_, c)| near(p, c)) {
                        continue;
                    }
                    let owners: Vec<&(usize, Vec<Vec2>)> =
                        pieces.iter().filter(|(_, c)| crossings(p, c) % 2 == 1).collect();
                    let in_face = (copy.contains)(P2::from_vec(p));
                    if owners.len() != usize::from(in_face) {
                        mismatches += 1;
                    }
                    let [owner] = owners.as_slice() else { continue };
                    // A point inside the face is in it and in no other face,
                    // and the face's map turns out of the solid there.
                    inside += 1;
                    if holders(P3::from_vec(frame.point(p)), &faces) != [FaceID::Solid { face: i }] {
                        untiled += 1;
                    }
                    if outward_normal(face, copy, P2::from_vec(p)).dot(s.normals[owner.0]) <= 0.0 {
                        inward += 1;
                    }
                }
            }
        }
        assert!(
            faces.len() == s.loops.len()
                && faces.iter().enumerate().all(|(i, f)| f.id == FaceID::Solid { face: i })
                && faces
                    .iter()
                    .all(|f| f.copies.len() == 1 && f.copies[0].tile == kurven_core::manifest::Affine2::IDENTITY),
            "{name}: a face per loop list, in order, each in one place: {} faces",
            faces.len()
        );
        assert!(
            mismatches == 0 && untiled == 0 && inward == 0 && inside > 1000,
            "{name}: membership by the loops agrees with the pieces, and each point is in its face alone: \
             {mismatches} disagree, {untiled} in another face too, {inward} facing in, of {inside} inside"
        );
    }
}

/// "faces: a parametric surface is one face, its whole parameter rectangle"
#[test]
fn a_parametric_surface_is_one_face_its_whole_parameter_rectangle() {
    let torus = ParametricSurface::torus(3.0, 1.0, (40, 20)).expect("the torus");
    let faces = SceneGeometry::Parametric(torus.clone()).faces();
    let mut agree = faces.len() == 1
        && faces[0].id == FaceID::Surface
        && faces[0].copies.len() == 1
        && faces[0].copies[0].outward == torus.outward;
    let (face, copy) = (&faces[0], &faces[0].copies[0]);
    for k in 0..50 {
        // Unwrapped coordinates too: the axes are periodic.
        let p = P2::<ParamSpace>::new(k as f64 * 0.37 - 4.0, k as f64 * 0.53 - 6.0);
        agree = agree && (copy.contains)(p) && copy.place((face.map)(p)) == (torus.map)(p);
    }
    assert!(agree, "the torus is one face, everywhere in it, mapped by the surface's own map");
}

/// "faces: a polyhedron face's trim loops are its loops, and a contour cut
/// at them is its hatching"
///
/// Hatching clips each line to the face's convex pieces, half-plane by
/// half-plane, and joins the intervals across the diagonals; a contour of
/// the coordinate across the hatching's angle, cut where it crosses the
/// face's loops, is the same lines. At an angle and a spacing that put no
/// line on a vertex or along a side. Over the seven solids built by name, as
/// Swift builds them.
#[test]
fn a_polyhedron_faces_trim_loops_are_its_loops_and_a_contour_cut_at_them_is_its_hatching() {
    let mut solids: Vec<(String, Solid)> =
        Platonic::ALL.iter().map(|&k| (k.name().to_string(), Solid::platonic(k, 1.0))).collect();
    solids.push(("the picture frame".to_string(), Solid::frame(2.0, 1.0, 0.8)));
    solids.push(("Császár's torus".to_string(), Solid::csaszar()));
    for (name, s) in &solids {
        let (mut lines, mut unmatched, mut worst, mut closed) = (0, 0, 0.0, true);
        for (i, face) in SceneGeometry::Solid(s.clone()).faces().iter().enumerate() {
            let (frame, copy) = (s.frames[i], &face.copies[0]);
            let Some(loops) = copy.trim_loops() else {
                closed = false;
                continue;
            };
            closed = closed && loops.len() == s.loops[i].len() && loops.iter().all(|l| l.first() == l.last());
            let (u, v) = (face.domain.u, face.domain.v);
            let extent = Vec2::new(u.range.length(), v.range.length()).length();
            let hatching = Hatching::plain(0.37, extent / 23.7);
            let per = Vec2::new(-real::sin(hatching.angle), real::cos(hatching.angle));
            // The hatch lines by level, in frame coordinates.
            let mut drawn: BTreeMap<i64, Vec<(Vec2, Vec2)>> = BTreeMap::new();
            for line in s.hatch_lines_on_face(&hatching, i) {
                let a = frame.coordinates(line.path[0].v);
                let b = frame.coordinates(line.path[line.path.len() - 1].v);
                drawn.entry((a.dot(per) / hatching.spacing).round() as i64).or_default().push((a, b));
            }
            let (Some(&first), Some(&last)) = (drawn.keys().next(), drawn.keys().next_back()) else { continue };
            let levels: Vec<f64> = (first..=last).map(|k| k as f64 * hatching.spacing).collect();
            let field = ScalarField::new(None, move |p| p.v.dot(per));
            for (level, runs) in face.contours(&field, &levels, copy) {
                let mut expected = drawn.get(&((level / hatching.spacing).round() as i64)).cloned().unwrap_or_default();
                lines += runs.len();
                for run in &runs {
                    let (a, b) = (run[0].v, run[run.len() - 1].v);
                    // The same segment, either way round: the first nearest,
                    // as Swift's `min(by:)` picks it.
                    let cost = |(p, q): (Vec2, Vec2)| {
                        swift::min((p - a).length() + (q - b).length(), (p - b).length() + (q - a).length())
                    };
                    let mut best: Option<usize> = None;
                    for j in 0..expected.len() {
                        if best.is_none_or(|k| cost(expected[j]) < cost(expected[k])) {
                            best = Some(j);
                        }
                    }
                    let Some(j) = best else {
                        unmatched += 1;
                        continue;
                    };
                    let (p, q) = expected.remove(j);
                    worst = swift::max(
                        worst,
                        swift::min(
                            swift::max((p - a).length(), (q - b).length()),
                            swift::max((p - b).length(), (q - a).length()),
                        ) / extent,
                    );
                }
                unmatched += expected.len();
            }
        }
        eprintln!("{name}: {unmatched} of {lines} unmatched, ends off by {worst:e} of the extent");
        assert!(
            closed && lines > 50 && unmatched == 0 && worst < 1e-12,
            "{name}: each face's trim loops are its loops, closed, and its lines cut at them are its hatching: \
             {}{unmatched} of {lines} unmatched, ends off by {worst} of the extent",
            if closed { "" } else { "loops not its own; " }
        );
    }
}
