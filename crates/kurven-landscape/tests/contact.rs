//! `Sources/kurven-test/Contact.swift`: where a part meets the terrain,
//! every suite, with Swift's tolerances.
//!
//! A composite's contact line (`Composite::contact`), held to the two
//! surfaces it lies on: every vertex, and every chord's middle, on the
//! landscape's drawn height (`Heightfield::drawn_height`, the grid's
//! triangles cut at the truncation, read apart from the composite's slab),
//! and on its part -- a polyhedron's face, the round torus. On a plane of the
//! grid a box's contact is the polygon the plane cuts from it, vertex by
//! vertex. A part that floats draws none. Its visibility is the composite
//! gate's (composite.rs).
//!
//! The round torus (`SurfaceBundle`, `ParametricSurface::torus`) is Phase
//! 4's: every suite or part of one that stands it in a composite is an
//! ignored test below, written out in full against the torus so that it
//! needs only the torus to run.

#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::type_complexity, clippy::manual_range_contains)]

mod common;
mod gate;

use std::collections::BTreeSet;
use std::f64::consts::PI;
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::par_map;
use gate::{WITH_TORUS, gate_compositions, landscape_cell, orthographic, polyhedron, spike, surface, torus_radii};
use kurven_core::camera::Camera;
use kurven_core::composite::{Composite, Placement};
use kurven_core::grid::Grid2D;
use kurven_core::manifest::{
    Affine2, BoundaryPerimeter, Domain, Interval, PerimeterEdge, RealBand, Region, Truncation, Walls,
};
use kurven_core::mesh::Mesh;
use kurven_core::parametric::{ParametricSurface, PortRefusal};
use kurven_core::scene::{Heightfield, PreviewMode, Scene, SceneGeometry};
use kurven_core::simd::{Mat3, Vec2, Vec3};
use kurven_core::solid::Solid;
use kurven_core::spaces::{P2, P3};
use kurven_core::surface::Surface;
use kurven_math::{real, swift};

// MARK: - helpers

/// The distance from `p` to the closed polygon `l` in the plane.
fn distance_to_loop(p: Vec2, l: &[Vec2]) -> f64 {
    (0..l.len())
        .map(|k| {
            let (a, b) = (l[k], l[(k + 1) % l.len()]);
            let ab = b - a;
            let t = swift::min(swift::max((p - a).dot(ab) / swift::max(ab.length_squared(), 1e-300), 0.0), 1.0);
            (p - (a + t * ab)).length()
        })
        .fold(f64::INFINITY, swift::min)
}

/// The distance from `p` to the polyline `path` in space.
fn distance_to_path(p: Vec3, path: &[Vec3]) -> f64 {
    path.windows(2)
        .map(|w| {
            let (a, b) = (w[0], w[1]);
            let ab = b - a;
            let t = swift::min(swift::max((p - a).dot(ab) / swift::max(ab.length_squared(), 1e-300), 0.0), 1.0);
            (p - (a + t * ab)).length()
        })
        .fold(f64::INFINITY, swift::min)
}

/// The distance from `p` to the nearest of `paths`.
fn distance_to_paths(p: Vec3, paths: &[Vec<Vec3>]) -> f64 {
    paths.iter().map(|path| distance_to_path(p, path)).fold(f64::INFINITY, swift::min)
}

/// The landscape's domain box, (x lo, x hi, y lo, y hi).
fn domain_box(h: &Heightfield) -> (f64, f64, f64, f64) {
    let d = h.surface.domain();
    (
        swift::min(d.real.lo, d.real.hi),
        swift::max(d.real.lo, d.real.hi),
        swift::min(d.imag.lo, d.imag.hi),
        swift::max(d.imag.lo, d.imag.hi),
    )
}

/// How far `l`, a point of a landscape's own coordinates, stands off its
/// drawn height, within its domain and footprint: the least gap over the
/// point and points a trillionth of a cell either way along x and y, so a
/// point on a step of the height -- a riser, a triangle with no top -- is
/// on the side it stands on. Infinite off the terrain.
fn terrain_gap(h: &Heightfield, l: Vec3) -> f64 {
    let e = 1e-12 * landscape_cell(h);
    let (xlo, xhi, ylo, yhi) = domain_box(h);
    [Vec2::new(0.0, 0.0), Vec2::new(e, 0.0), Vec2::new(-e, 0.0), Vec2::new(0.0, e), Vec2::new(0.0, -e)]
        .into_iter()
        .map(|step| {
            let p = Vec2::new(l.x, l.y) + step;
            if !((h.tiles.len() > 1 || (p.x >= xlo - e && p.x <= xhi + e && p.y >= ylo - e && p.y <= yhi + e))
                && h.region.contains(P2::new(p.x, p.y)))
            {
                return f64::INFINITY;
            }
            let z = h.drawn_height(P2::new(p.x, p.y));
            if z.is_finite() { (l.z - z).abs() } else { f64::INFINITY }
        })
        .fold(f64::INFINITY, swift::min)
}

/// Each landscape of a composite, with its offset.
fn landscapes(composite: &Composite) -> Vec<(&Heightfield, Vec3)> {
    composite.parts.iter().filter_map(|p| p.scene.heightfield().map(|h| (h, p.placement.offset))).collect()
}

/// How far a part's contact line strays from the terrain and from the part,
/// in world units, and how many points were held: each vertex, and each
/// chord's middle -- on the terrain always, on a polyhedron's face, not on
/// the torus, whose chords cut inside it.
fn contact_stray(composite: &Composite, i: usize) -> (usize, f64, f64) {
    let part = &composite.parts[i];
    let (s, o) = (part.placement.scale, part.placement.offset);
    let contact = composite.contact(i);
    let lands = landscapes(composite);
    let (mut points, mut terrain, mut on_part) = (0, 0.0, 0.0);
    for (k, path) in contact.paths.iter().enumerate() {
        let face = part.scene.solid().map(|solid| {
            let f = contact.faces[k];
            let frame = solid.frames[f];
            let loops: Vec<Vec<Vec2>> = solid.loops[f]
                .iter()
                .map(|l| l.iter().map(|&v| frame.coordinates(solid.vertices[v].v)).collect())
                .collect();
            (frame, loops, solid.face(f))
        });
        let mut held: Vec<(Vec3, bool)> = path.iter().map(|v| (v.v, true)).collect();
        held.extend(path.windows(2).map(|w| (0.5 * (w[0].v + w[1].v), false)));
        for (q, vertex) in held {
            points += 1;
            let w = s * q + o;
            terrain = swift::max(
                terrain,
                lands.iter().map(|&(h, offset)| terrain_gap(h, w - offset)).fold(f64::INFINITY, swift::min),
            );
            match part.scene.geometry() {
                SceneGeometry::Solid(_) => {
                    let (frame, loops, face) = face.as_ref().unwrap();
                    let c = frame.coordinates(q);
                    let off = if (face.copies[0].contains)(P2::from_vec(c)) {
                        0.0
                    } else {
                        loops.iter().map(|l| distance_to_loop(c, l)).fold(f64::INFINITY, swift::min)
                    };
                    on_part = swift::max(on_part, s * swift::max((q - frame.origin).dot(frame.normal).abs(), off));
                }
                SceneGeometry::Parametric(surface) if vertex => {
                    let (big, r) = torus_radii(surface);
                    let rho = (q.x * q.x + q.y * q.y).sqrt();
                    on_part = swift::max(on_part, s * (((rho - big) * (rho - big) + q.z * q.z).sqrt() - r).abs());
                }
                _ => {}
            }
        }
    }
    (points, terrain, on_part)
}

/// One grid of `contact_missed`: where it runs, how many steps each way,
/// which of its points count, and the world point at each.
struct Grid<'a> {
    lo: Vec2,
    hi: Vec2,
    n: (usize, usize),
    inside: Box<dyn Fn(Vec2) -> bool + 'a>,
    place: Box<dyn Fn(Vec2) -> Vec3 + 'a>,
}

/// What a contact line misses, found apart from it: on a grid over each
/// face of a polyhedron (`steps` to a side, inside the face, clear of its
/// edges by a millionth of the face) or over the torus's parameter
/// rectangle (`steps` along u, half as many along v), every grid edge along
/// which z − H changes sign continuously -- H the drawn height, read off
/// the landscape apart from the composite -- bisected onto the terrain. The
/// farthest of those points from the contact line, in world units, and how
/// many there were.
fn contact_missed(composite: &Composite, i: usize, steps: usize) -> (usize, f64, Vec3) {
    let part = &composite.parts[i];
    let (s, o) = (part.placement.scale, part.placement.offset);
    let lands = landscapes(composite);
    let contact: Vec<Vec<Vec3>> =
        composite.contact(i).paths.iter().map(|p| p.iter().map(|v| s * v.v + o).collect()).collect();
    // z − H at a world point over the first landscape that has terrain there.
    let excess = |w: Vec3| -> Option<f64> {
        for &(h, offset) in &lands {
            let l = w - offset;
            let (xlo, xhi, ylo, yhi) = domain_box(h);
            if !(l.x >= xlo && l.x <= xhi && l.y >= ylo && l.y <= yhi && h.region.contains(P2::new(l.x, l.y))) {
                continue;
            }
            let z = h.drawn_height(P2::new(l.x, l.y));
            if z.is_finite() {
                return Some(l.z - z);
            }
        }
        None
    };
    let cell = lands.iter().map(|&(h, _)| landscape_cell(h)).fold(f64::INFINITY, swift::min);
    let mut grids: Vec<Grid> = Vec::new();
    match part.scene.geometry() {
        SceneGeometry::Solid(solid) => {
            for f in 0..solid.faces.len() {
                let frame = solid.frames[f];
                let face = solid.face(f);
                let loops: Vec<Vec<Vec2>> = solid.loops[f]
                    .iter()
                    .map(|l| l.iter().map(|&v| frame.coordinates(solid.vertices[v].v)).collect())
                    .collect();
                let lo = Vec2::new(face.domain.u.range.lo, face.domain.v.range.lo);
                let hi = Vec2::new(face.domain.u.range.hi, face.domain.v.range.hi);
                let margin = 1e-6 * (hi - lo).length();
                let contains = face.copies[0].contains.clone();
                grids.push(Grid {
                    lo,
                    hi,
                    n: (steps, steps),
                    inside: Box::new(move |c| {
                        contains(P2::from_vec(c)) && loops.iter().all(|l| distance_to_loop(c, l) > margin)
                    }),
                    place: Box::new(move |c| s * frame.point(c) + o),
                });
            }
        }
        SceneGeometry::Parametric(surface) => {
            grids.push(Grid {
                lo: Vec2::new(surface.u.range.lo, surface.v.range.lo),
                hi: Vec2::new(surface.u.range.hi, surface.v.range.hi),
                n: (steps, steps / 2),
                inside: Box::new(|_| true),
                place: Box::new(move |c| s * (surface.map)(P2::from_vec(c)).position + o),
            });
        }
        SceneGeometry::Heightfield(_) => return (0, 0.0, Vec3::ZERO),
    }
    let mut crossings: Vec<Vec3> = Vec::new();
    for grid in &grids {
        let (nx, ny) = grid.n;
        let step = Vec2::new((grid.hi.x - grid.lo.x) / nx as f64, (grid.hi.y - grid.lo.y) / ny as f64);
        let at = |i: usize, j: usize| Vec2::new(grid.lo.x + i as f64 * step.x, grid.lo.y + j as f64 * step.y);
        let mut values: Vec<Option<f64>> = vec![None; (nx + 1) * (ny + 1)];
        for j in 0..=ny {
            for i in 0..=nx {
                if (grid.inside)(at(i, j)) {
                    values[j * (nx + 1) + i] = excess((grid.place)(at(i, j)));
                }
            }
        }
        for j in 0..=ny {
            for i in 0..=nx {
                for (di, dj) in [(1, 0), (0, 1)] {
                    if !(i + di <= nx && j + dj <= ny) {
                        continue;
                    }
                    let (Some(ga), Some(gb)) = (values[j * (nx + 1) + i], values[(j + dj) * (nx + 1) + i + di]) else {
                        continue;
                    };
                    if (ga > 0.0) == (gb > 0.0) {
                        continue;
                    }
                    let (mut a, mut b, mut fa) = (at(i, j), at(i + di, j + dj), ga);
                    let mut ok = true;
                    for _ in 0..70 {
                        let m = 0.5 * (a + b);
                        let Some(fm) = excess((grid.place)(m)) else {
                            ok = false;
                            break;
                        };
                        if (fm > 0.0) == (fa > 0.0) {
                            a = m;
                            fa = fm;
                        } else {
                            b = m;
                        }
                    }
                    // A crossing, not a step of H.
                    let w = (grid.place)(0.5 * (a + b));
                    if !ok {
                        continue;
                    }
                    let Some(e) = excess(w) else { continue };
                    if !(e.abs() <= 1e-9 * cell) {
                        continue;
                    }
                    crossings.push(w);
                }
            }
        }
    }
    let far = par_map(&crossings, |&c| distance_to_paths(c, &contact));
    // Swift's `max(by:)`: the first of the greatest.
    let mut best: Option<usize> = None;
    for k in 0..far.len() {
        if best.is_none_or(|b| far[b] < far[k]) {
            best = Some(k);
        }
    }
    match best {
        Some(k) => (crossings.len(), far[k], crossings[k]),
        None => (0, 0.0, Vec3::ZERO),
    }
}

/// The open ends of a part's contact paths that are not on a step of the
/// terrain -- its edge, a riser -- nor, on a polyhedron, on a face's
/// boundary: where a path stops short.
fn contact_loose_ends(composite: &Composite, i: usize) -> usize {
    let part = &composite.parts[i];
    let (s, o) = (part.placement.scale, part.placement.offset);
    if part.scene.parametric().is_none() {
        return 0;
    }
    let lands = landscapes(composite);
    let mut loose = 0;
    for path in &composite.contact(i).paths {
        let w: Vec<Vec3> = path.iter().map(|v| s * v.v + o).collect();
        let (first, last) = (w[0], w[w.len() - 1]);
        if !((first - last).length() > 1e-12) {
            continue;
        }
        for e in [first, last] {
            // Beside a step, some point a millionth of a cell away is off
            // the terrain or a thousandth of a cell off its height.
            let on_step = lands.iter().any(|&(h, offset)| {
                let cell = landscape_cell(h);
                let l = e - offset;
                let (xlo, xhi, ylo, yhi) = domain_box(h);
                (0..16).any(|k| {
                    let a = k as f64 * PI / 8.0;
                    let p = Vec2::new(l.x, l.y) + 1e-6 * cell * Vec2::new(real::cos(a), real::sin(a));
                    if !(p.x >= xlo && p.x <= xhi && p.y >= ylo && p.y <= yhi && h.region.contains(P2::new(p.x, p.y))) {
                        return true;
                    }
                    let z = h.drawn_height(P2::new(p.x, p.y));
                    !z.is_finite() || (z - l.z).abs() > 1e-3 * cell
                })
            });
            if !on_step {
                loose += 1;
            }
        }
    }
    loose
}

/// A seeded stream of numbers in [0, 1): a linear congruential generator.
struct SweepRandom {
    state: u64,
}

impl SweepRandom {
    fn next(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.state >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.next()
    }
}

type Height = Arc<dyn Fn(f64, f64) -> f64 + Send + Sync>;

/// A landscape of f sampled on an n by n grid over [-2, 2]², with samples
/// replaced as `patch` says.
fn grid_landscape(
    n: usize,
    truncation: Truncation,
    region: Region,
    patch: &[(usize, usize, f32)],
    camera: Camera,
    f: impl Fn(f64, f64) -> f64,
) -> Scene {
    let domain = Domain::new(Interval::new(-2.0, 2.0), Interval::new(-2.0, 2.0));
    let mut values: Vec<f32> = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            values.push(f(-2.0 + 4.0 * i as f64 / (n - 1) as f64, -2.0 + 4.0 * j as f64 / (n - 1) as f64) as f32);
        }
    }
    for &(i, j, v) in patch {
        values[j * n + i] = v;
    }
    Scene::heightfield_scene(
        Surface::new(Grid2D::new(n, n, domain, values), None, truncation, false),
        Mesh::empty(),
        vec![Affine2::IDENTITY],
        region,
        1,
        Vec::new(),
        camera,
        PreviewMode::Drawing,
        0.02,
        None,
        Walls::None,
    )
}

/// A solid's loops as `Solid::checking` takes them.
fn loops_of(s: &Solid) -> Vec<Vec<Vec<i64>>> {
    s.loops.iter().map(|f| f.iter().map(|l| l.iter().map(|&k| k as i64).collect()).collect()).collect()
}

/// A prism over an L, two units a side, standing on z = 0.
fn l_prism() -> Solid {
    let l = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)];
    let mut vertices: Vec<P3<_>> = l.iter().map(|&(x, y)| P3::new(x - 1.0, y - 1.0, 0.0)).collect();
    vertices.extend(l.iter().map(|&(x, y)| P3::new(x - 1.0, y - 1.0, 1.0)));
    let mut faces: Vec<Vec<Vec<i64>>> = vec![vec![(6..12).collect()], vec![(0..6).rev().collect()]];
    for k in 0..6i64 {
        faces.push(vec![vec![k, (k + 1) % 6, 6 + (k + 1) % 6, 6 + k]]);
    }
    Solid::checking(vertices, faces).expect("the L-prism is a solid")
}

/// A solid turned about x, then y, then z.
fn turned(s: &Solid, ax: f64, ay: f64, az: f64) -> Solid {
    let (cx, sx, cy, sy, cz, sz) =
        (real::cos(ax), real::sin(ax), real::cos(ay), real::sin(ay), real::cos(az), real::sin(az));
    let rx = Mat3::from_rows(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, cx, -sx), Vec3::new(0.0, sx, cx));
    let ry = Mat3::from_rows(Vec3::new(cy, 0.0, sy), Vec3::new(0.0, 1.0, 0.0), Vec3::new(-sy, 0.0, cy));
    let rz = Mat3::from_rows(Vec3::new(cz, -sz, 0.0), Vec3::new(sz, cz, 0.0), Vec3::new(0.0, 0.0, 1.0));
    let m = rz * ry * rx;
    Solid::checking(s.vertices.iter().map(|v| P3::from_vec(m * v.v)).collect(), loops_of(s))
        .expect("a turned solid is a solid")
}

fn solid_scene(s: Solid, camera: Camera) -> Scene {
    Scene::solid_scene(s, Vec::new(), camera, PreviewMode::Drawing, 0.0)
}

/// The sweep's camera, el 30° az 37°.
fn sweep_camera() -> Camera {
    orthographic(-30.0, 37.0, 0.0)
}

fn sinusoid(x: f64, y: f64) -> f64 {
    1.0 + 0.6 * real::sin(2.0 * x) * real::cos(3.0 * y)
}

/// The sweep's landscapes, each with the height a part is placed by.
fn sweep_landscapes() -> Vec<(String, Scene, Height)> {
    let camera = sweep_camera();
    let bands = Truncation::RealBands {
        bands: vec![RealBand { below: -0.53, height: 1.05 }, RealBand { below: 0.71, height: 0.8 }],
        beyond: 1.3,
    };
    let corners = [(-1.9, -1.9), (1.9, -1.9), (1.9, 0.3), (0.2, 0.3), (0.2, 1.9), (-1.9, 1.9)];
    let footprint = BoundaryPerimeter::new(
        (0..corners.len())
            .map(|k| {
                let (a, b) = (corners[k], corners[(k + 1) % corners.len()]);
                PerimeterEdge { start: P2::new(a.0, a.1), end: P2::new(b.0, b.1), density: 8 }
            })
            .collect(),
    );
    let f: Height = Arc::new(sinusoid);
    vec![
        ("the sinusoid".into(), grid_landscape(17, Truncation::None, Region::Full, &[], camera, sinusoid), f.clone()),
        (
            "the sinusoid in bands, on an L".into(),
            grid_landscape(17, bands, Region::Inside(footprint), &[], camera, sinusoid),
            f.clone(),
        ),
        (
            "the sinusoid with a sample not a number, truncated at 1.2".into(),
            grid_landscape(17, Truncation::Uniform(1.2), Region::Full, &[(8, 8, f32::NAN)], camera, sinusoid),
            f,
        ),
        // Slabs of no thickness: every sample at the base.
        (
            "a flat terrain".into(),
            grid_landscape(17, Truncation::None, Region::Full, &[], camera, |_, _| 0.5),
            Arc::new(|_, _| 0.5),
        ),
        (
            "a plateau".into(),
            grid_landscape(17, Truncation::Uniform(0.5), Region::Full, &[], camera, |_, _| 2.0),
            Arc::new(|_, _| 0.5),
        ),
        // The gate's spike, whose terrain is a rounding over its floor
        // away from the apex.
        ("the spike".into(), spike(), Arc::new(|x: f64, y: f64| real::exp(-20.0 * (x * x + y * y).sqrt()))),
    ]
}

/// One placement of the sweep: its name, its landscape, the part, whether
/// it is the torus, and where.
type SweepPlacement = (String, usize, Result<Scene, PortRefusal>, bool, Placement);

/// The placements the reviews found wanting, then a seeded sweep.
fn sweep_placements(landscapes: &[(String, Scene, Height)]) -> Vec<SweepPlacement> {
    let camera = sweep_camera();
    let torus = surface("torus");
    let cube = polyhedron("cube");
    let frame = polyhedron("frame");
    let turned_cube = turned(cube.solid().unwrap(), 0.4, 0.3, 0.7);
    let turned_l = turned(&l_prism(), 0.6, -0.5, 0.2);
    let solids: Vec<(&str, Solid)> = vec![
        ("the cube turned", turned_cube.clone()),
        ("the frame turned", turned(frame.solid().unwrap(), 1.1, 0.2, 0.5)),
        ("the L-prism", l_prism()),
        ("the L-prism turned", turned_l.clone()),
    ];
    let at = |scale: f64, x: f64, y: f64, z: f64| Placement::new(scale, Vec3::new(x, y, z));
    let mut out: Vec<SweepPlacement> = vec![
        (
            "the L-prism with a face on a grid line".into(),
            0,
            Ok(solid_scene(l_prism(), camera)),
            false,
            at(0.6, 0.1, -0.2, 0.6),
        ),
        ("the frame with a face on a grid line".into(), 0, Ok(frame.clone()), false, at(0.6, 0.1, -0.2, 0.6)),
    ];
    for (name, land, placement) in [
        ("the torus where a crossing left its piece", 0, at(0.4018, 0.5374, -0.7111, 0.9796)),
        ("the torus where a crossing was not solved", 0, at(0.5887, -0.2241, 0.5246, 1.2583)),
        ("the torus grazing the terrain", 0, at(0.5, 0.3, -0.2, 0.7)),
        ("the torus over the terrain's edge", 0, at(0.35, 2.0, 0.1, 0.9)),
        ("the torus far over the terrain's edge", 0, at(0.6, 2.3, 0.1, 0.9)),
        ("the torus over the steps", 1, at(0.35, 0.0, 0.0, 0.8)),
        ("the torus over the sample not a number", 2, at(0.35, 0.0, 0.0, 0.8)),
    ] {
        out.push((name.into(), land, torus.clone(), true, placement));
    }
    for (land, z) in [(3, 0.3), (3, 0.45), (4, 0.3), (4, 0.45)] {
        out.push((
            "the cube turned, on a slab of no thickness".into(),
            land,
            Ok(solid_scene(turned_cube.clone(), camera)),
            false,
            at(0.4, 0.1, 0.1, z),
        ));
        out.push((
            "the L-prism turned, on a slab of no thickness".into(),
            land,
            Ok(solid_scene(turned_l.clone(), camera)),
            false,
            at(0.4, 0.1, 0.1, z),
        ));
    }
    if landscapes.len() > 5 {
        for (x, y) in [(1.5, 1.5), (1.0, -1.2), (0.6, 0.5), (0.3, 0.2)] {
            out.push((
                "the cube turned, where the spike is a rounding over its floor".into(),
                5,
                Ok(solid_scene(turned_cube.clone(), camera)),
                false,
                at(0.2, x, y, -0.15),
            ));
        }
    }
    let mut random = SweepRandom { state: 99 };
    for k in 0..180 {
        let land = k % landscapes.len();
        let h = &landscapes[land].2;
        let x = random.range(-1.2, 1.2);
        let y = random.range(-1.2, 1.2);
        if k % 2 == 0 {
            let scale = random.range(0.2, 0.6);
            let z = h(x, y) + random.range(-0.3, 0.3);
            out.push(("a torus".into(), land, torus.clone(), true, at(scale, x, y, z)));
        } else {
            let (name, s) = &solids[(k / 2) % solids.len()];
            let scale = random.range(0.3, 0.9);
            let z = h(x, y) - random.range(-0.2, 0.5) * scale;
            out.push((name.to_string(), land, Ok(solid_scene(s.clone(), camera)), false, at(scale, x, y, z)));
        }
    }
    out
}

/// What the sweep found over the placements it ran.
#[derive(Default)]
struct Sweep {
    met: usize,
    points: usize,
    terrain: f64,
    on_part: f64,
    crossings: usize,
    missed: usize,
    loose: usize,
    farthest: f64,
    example: String,
    poly_farthest: f64,
    poly_example: String,
    stray_example: String,
}

/// The sweep over the placements `torus` selects (the torus's, or the
/// polyhedra's), each composite apart, spread over threads, the verdicts
/// summed in order.
fn sweep(torus: bool) -> (Sweep, Duration) {
    let landscapes = sweep_landscapes();
    let placements: Vec<SweepPlacement> = sweep_placements(&landscapes).into_iter().filter(|p| p.3 == torus).collect();
    let camera = sweep_camera();
    let start = Instant::now();
    let indices: Vec<usize> = (0..placements.len()).collect();
    // Each placement's stray, its miss and its loose ends, or None where
    // the part meets no terrain.
    let found = par_map(&indices, |&k| {
        let (name, land, scene, _, placement) = &placements[k];
        let scene = scene.as_ref().unwrap_or_else(|e| panic!("{name}: {e}"));
        let composite = Composite::new(
            vec![
                ("landscape".into(), landscapes[*land].1.clone(), Placement::IDENTITY),
                ("part".into(), scene.clone(), *placement),
            ],
            camera,
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        if composite.contact(1).is_empty() {
            return None;
        }
        let h = composite.parts[0].scene.heightfield().unwrap();
        let cell = landscape_cell(h);
        let least = composite.parts[1].scene.parametric().map(|s| {
            // The solve's least step: a sixty-fourth of a square of a
            // quarter of a cell, at most an eighth of the tube.
            swift::min(cell / 4.0, torus_radii(s).1 * placement.scale / 8.0) / 64.0
        });
        Some((
            contact_stray(&composite, 1),
            contact_missed(&composite, 1, 240),
            contact_loose_ends(&composite, 1),
            1e-9 * cell,
            least,
        ))
    });
    let mut out = Sweep::default();
    for (k, f) in found.into_iter().enumerate() {
        let Some((stray, miss, loose, tolerance, least)) = f else { continue };
        let (name, land, _, _, placement) = &placements[k];
        let land_name = &landscapes[*land].0;
        out.met += 1;
        out.points += stray.0;
        // Each against a billionth of its landscape's cell.
        if stray.1 / tolerance > out.terrain || stray.2 / tolerance > out.on_part {
            out.stray_example = format!("{name} #{k} on {land_name}");
        }
        out.terrain = swift::max(out.terrain, stray.1 / tolerance);
        out.on_part = swift::max(out.on_part, stray.2 / tolerance);
        out.crossings += miss.0;
        if let Some(least) = least {
            if miss.1 / least > out.farthest {
                out.farthest = miss.1 / least;
                let (o, a) = (placement.offset, miss.2);
                out.example = format!(
                    "{name} #{k} on {land_name} (scale {:.4} at ({:.4}, {:.4}, {:.4})), the crossing at ({:.5}, \
                     {:.5}, {:.5})",
                    placement.scale, o.x, o.y, o.z, a.x, a.y, a.z
                );
            }
        } else {
            out.poly_farthest = swift::max(out.poly_farthest, miss.1 / tolerance);
            if miss.1 > tolerance {
                out.missed += 1;
                if out.poly_example.is_empty() {
                    out.poly_example = format!("{name} #{k} on {land_name}, at {:?}", miss.2);
                }
            }
        }
        out.loose += loose;
    }
    (out, start.elapsed())
}

/// The contact lines of the gate's compositions, each held to the
/// terrain, its part and the crossings found apart from it, under the
/// landscape's camera: the failures, as text.
fn contact_lines(torus: bool) -> Vec<String> {
    let (cameras, compositions) = gate_compositions();
    let mut failures = Vec::new();
    for (name, parts) in &compositions {
        if WITH_TORUS.contains(&name.as_str()) != torus {
            continue;
        }
        let parts = parts.as_ref().unwrap_or_else(|e| panic!("{name}: {e}"));
        let composite = Composite::new(parts.clone(), cameras[0].1).unwrap_or_else(|e| panic!("{name}: {e}"));
        // A billionth of the landscape's cell, as the cut starts a line of
        // sight a billionth of a cell along it (`Slab`).
        let tolerance =
            landscapes(&composite).iter().map(|&(h, _)| 1e-9 * landscape_cell(h)).fold(f64::INFINITY, swift::min);
        let (mut points, mut terrain, mut on_part, mut crossings, mut farthest) = (0, 0.0, 0.0, 0, 0.0);
        for i in 0..composite.parts.len() {
            let stray = contact_stray(&composite, i);
            points += stray.0;
            terrain = swift::max(terrain, stray.1);
            on_part = swift::max(on_part, stray.2);
            // A polyhedron's contact line is exact: it passes through
            // every crossing found apart from it.
            if composite.parts[i].scene.solid().is_some() && stray.0 > 0 {
                let miss = contact_missed(&composite, i, 240);
                crossings += miss.0;
                farthest = swift::max(farthest, miss.1);
            }
        }
        let drawn = composite
            .visible_layers()
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .iter()
            .filter(|l| l.0.spec.name == "contact")
            .count();
        let text = format!(
            "{points} points, off the terrain by {terrain:.1e}, off the part by {on_part:.1e}, {crossings} crossings \
             the farthest {farthest:.1e} off it, within {tolerance:.1e}, {drawn} contact layers"
        );
        eprintln!("{name}: {text}");
        if points == 0 {
            if drawn != 0 {
                failures.push(format!("{name}: no part meets the terrain, and there is no contact line: {text}"));
            }
        } else if !(terrain <= tolerance && on_part <= tolerance && farthest <= tolerance && drawn >= 1) {
            failures.push(format!(
                "{name}: every point of the contact line is on the terrain and on its part, and it misses no \
                 crossing: {text}"
            ));
        }
    }
    failures
}

// MARK: - the suites

/// "composite: a contact line lies on the terrain and on its part", the
/// eight gate compositions with no torus, under the landscape's camera:
/// where a part meets the terrain, every vertex and chord middle of its
/// contact line within a billionth of the landscape's cell of the terrain
/// and of its part, every crossing found apart from a polyhedron's line
/// (`contact_missed`, 240 steps a face) within it of the line, and the
/// line drawn; where none does, no contact layer.
#[test]
fn a_contact_line_lies_on_the_terrain_and_on_its_part() {
    let start = Instant::now();
    let failures = contact_lines(false);
    eprintln!("in {:?}", start.elapsed());
    assert!(failures.is_empty(), "{failures:#?}");
}

/// "composite: a contact line lies on the terrain and on its part", the
/// three gate compositions that stand the round torus in a composite
/// (`SurfaceBundle`, Phase 4): the torus ringing a tower, which meets
/// nothing and draws no contact layer; the torus sunk into the foot of a
/// tower; all three moved. The torus's vertices within a billionth of the
/// cell of the tube and the terrain, the cube's line through every
/// crossing, the line drawn (`visible_contact` on a parametric surface,
/// also Phase 4).
#[test]
fn a_contact_line_lies_on_the_terrain_and_on_its_part_with_the_round_torus() {
    let failures = contact_lines(true);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// "composite: a box sunk into a plane of the grid meets it in the polygon
/// the plane cuts": z = 1 + x/4 + y/8 on a grid of 32 cells over [-2, 2]²,
/// every sample a float to the bit, so the drawn triangles are the plane;
/// the cube at scale 0.35 at (0.1, -0.2, 0.6). Alone and under a
/// truncation at 1, its contact is four paths on four faces, every vertex
/// within 1e-12 of the polygon the plane cuts (the feet on the terrain, and
/// where the truncation's edge crosses a side), every corner of it met
/// within 1e-12, and its length the perimeter's to 1e-12 of it.
#[test]
fn a_box_sunk_into_a_plane_of_the_grid_meets_it_in_the_polygon_the_plane_cuts() {
    let n = 33;
    let plane = |x: f64, y: f64| 1.0 + x / 4.0 + y / 8.0;
    let cube = polyhedron("cube");
    let placement = Placement::new(0.35, Vec3::new(0.1, -0.2, 0.6));
    // The cube's vertical edges, where they stand, counter-clockwise.
    let mut feet: Vec<Vec2> = Vec::new();
    for v in &cube.solid().unwrap().vertices {
        let w = placement.scale * v.v + placement.offset;
        let p = Vec2::new(w.x, w.y);
        if !feet.iter().any(|&q| (q - p).length() < 1e-9) {
            feet.push(p);
        }
    }
    let centre = feet.iter().fold(Vec2::new(0.0, 0.0), |a, &b| a + b) / feet.len() as f64;
    let angle = |p: Vec2| real::atan2(p.y - centre.y, p.x - centre.x);
    feet.sort_by(|a, b| angle(*a).partial_cmp(&angle(*b)).unwrap());
    let mut failures = Vec::new();
    for (what, truncation) in
        [("the plane", Truncation::None), ("the plane under a truncation at 1", Truncation::Uniform(1.0))]
    {
        let top = if let Truncation::Uniform(t) = truncation { t } else { f64::INFINITY };
        let landscape = grid_landscape(n, truncation, Region::Full, &[], cube.camera, plane);
        let height = |p: Vec2| swift::min(plane(p.x, p.y), top);
        // The polygon: the feet on the terrain, and where the truncation's
        // edge crosses a side between two of them.
        let mut polygon: Vec<Vec3> = Vec::new();
        for k in 0..feet.len() {
            let (a, c) = (feet[k], feet[(k + 1) % feet.len()]);
            polygon.push(Vec3::new(a.x, a.y, height(a)));
            let (ea, ec) = (plane(a.x, a.y) - top, plane(c.x, c.y) - top);
            if ea.is_finite() && (ea > 0.0) != (ec > 0.0) {
                let x = a + (c - a) * (ea / (ea - ec));
                polygon.push(Vec3::new(x.x, x.y, top));
            }
        }
        let composite = Composite::new(
            vec![("plane".into(), landscape, Placement::IDENTITY), ("cube".into(), cube.clone(), placement)],
            cube.camera,
        )
        .unwrap();
        let contact = composite.contact(1);
        let paths: Vec<Vec<Vec3>> = contact
            .paths
            .iter()
            .map(|p| p.iter().map(|v| placement.scale * v.v + placement.offset).collect())
            .collect();
        let vertices: Vec<Vec3> = paths.iter().flatten().copied().collect();
        let tolerance = 1e-12;
        let mut closed = polygon.clone();
        closed.push(polygon[0]);
        let off_polygon = if vertices.is_empty() {
            f64::INFINITY
        } else {
            vertices.iter().map(|&v| distance_to_path(v, &closed)).fold(f64::NEG_INFINITY, swift::max)
        };
        let missed = polygon
            .iter()
            .map(|&e| vertices.iter().map(|&v| (v - e).length()).fold(f64::INFINITY, swift::min))
            .fold(f64::NEG_INFINITY, swift::max);
        let length: f64 = paths.iter().map(|p| p.windows(2).map(|w| (w[1] - w[0]).length()).sum::<f64>()).sum();
        let perimeter: f64 = (0..polygon.len()).map(|k| (polygon[(k + 1) % polygon.len()] - polygon[k]).length()).sum();
        let sides = contact.faces.iter().collect::<BTreeSet<_>>().len();
        let text = format!(
            "{} paths on {sides} faces, vertices off it by {off_polygon:.1e}, its corners missed by {missed:.1e}, \
             length {length:.15} of {perimeter:.15}",
            contact.paths.len()
        );
        eprintln!("{what}: {text}");
        if !(off_polygon <= tolerance
            && missed <= tolerance
            && (length - perimeter).abs() <= 1e-12 * perimeter
            && sides == 4
            && contact.paths.len() == 4)
        {
            failures.push(format!("{what} meets the box in its {}-gon, vertex by vertex: {text}", polygon.len()));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The floating suite's check of one placement on the cubic landscape.
fn floats(what: &str, scene: Scene, placement: Placement, meets: bool) -> Option<String> {
    let landscape = gate::cubic_landscape();
    let composite = Composite::new(
        vec![("landscape".into(), landscape.clone(), Placement::IDENTITY), ("part".into(), scene, placement)],
        landscape.camera,
    )
    .unwrap();
    let layers = composite.visible_layers().unwrap().iter().filter(|l| l.0.spec.name == "contact").count();
    let paths = composite.contact(1).paths.len();
    let ok = composite.contact(1).is_empty() != meets && (layers == 0) != meets && composite.contact(0).is_empty();
    (!ok).then(|| format!("{what} has {} contact line: {paths} paths, {layers} layers", if meets { "a" } else { "no" }))
}

/// "composite: a part that floats over the terrain has no contact line",
/// the cube's two placements on the cubic landscape: at scale 0.35 at
/// (-0.5, 0.5, 5.5), over the truncation at 5, no contact and no contact
/// layer; let down to (-0.5, 0.5, 0), a contact line and its layer; the
/// landscape meeting nothing either way. The torus ringing a tower is
/// `the_round_torus_ringing_a_tower_has_no_contact_line` (ignored).
#[test]
fn a_part_that_floats_over_the_terrain_has_no_contact_line() {
    let cube = polyhedron("cube");
    let failures: Vec<String> = [
        ("a cube over the towers' tops", Placement::new(0.35, Vec3::new(-0.5, 0.5, 5.5)), false),
        ("the cube let down onto the terrain", Placement::new(0.35, Vec3::new(-0.5, 0.5, 0.0)), true),
    ]
    .into_iter()
    .filter_map(|(what, placement, meets)| floats(what, cube.clone(), placement, meets))
    .collect();
    assert!(failures.is_empty(), "{failures:#?}");
}

/// "composite: a part that floats over the terrain has no contact line",
/// the round torus (contract/surface_torus.kurven, Phase 4) ringing a tower
/// at scale 0.35 at (1, 0, 1.6), clear of it: no contact and no contact
/// layer, and the landscape meeting nothing.
#[test]
fn the_round_torus_ringing_a_tower_has_no_contact_line() {
    let torus = surface("torus").unwrap_or_else(|e| panic!("torus: {e}"));
    let failure =
        floats("the round torus ringing a tower", torus, Placement::new(0.35, Vec3::new(1.0, 0.0, 1.6)), false);
    assert!(failure.is_none(), "{failure:?}");
}

/// "composite: a sweep of parts over landscapes meets the terrain on both
/// surfaces, and misses nothing", its polyhedra: the L-prism and the frame
/// with a face on a grid line, the turned cube and L-prism on slabs of no
/// thickness and on the spike where it is a rounding over its floor, and
/// the odd placements of the seeded sweep (the turned cube, the turned
/// frame, the L-prism and the turned L-prism, at scales 0.3 to 0.9), over
/// the sinusoid, the sinusoid in bands on an L, the sinusoid with a sample
/// not a number under a truncation at 1.2, a flat terrain, a plateau and
/// the spike. Every vertex and chord middle of every contact line within a
/// billionth of its landscape's cell of the terrain and of the part, and a
/// polyhedron's line through every crossing found apart from it, over more
/// than 100 000 crossings. Swift's count of parts met, over 150, counts the
/// tori too, and is held with them in
/// `a_sweep_of_tori_over_landscapes_meets_the_terrain_and_misses_nothing`
/// (ignored); the polyhedra's count is printed.
#[test]
fn a_sweep_of_parts_over_landscapes_meets_the_terrain_on_both_surfaces_and_misses_nothing() {
    let (s, elapsed) = sweep(false);
    let stray = format!(
        "{} parts met, {} points, off the terrain by {:.2e} and off the part by {:.2e} of a billionth of the cell, \
         {elapsed:?}{}",
        s.met,
        s.points,
        s.terrain,
        s.on_part,
        if s.stray_example.is_empty() { String::new() } else { format!(", the worst {}", s.stray_example) }
    );
    let crossings = format!(
        "{} crossings, the farthest {:.2e} of a billionth of the cell off it{}",
        s.crossings,
        s.poly_farthest,
        if s.poly_example.is_empty() { String::new() } else { format!(", e.g. {}", s.poly_example) }
    );
    eprintln!("polyhedra: {stray}\npolyhedra: {crossings}");
    assert!(
        s.terrain <= 1.0 && s.on_part <= 1.0,
        "every vertex and chord middle of the polyhedra's contact lines lies on the terrain and on the part: {stray}"
    );
    assert!(
        s.missed == 0 && s.crossings > 100_000,
        "a polyhedron's contact line passes through every crossing found apart from it: {crossings}"
    );
}

/// "composite: a sweep of parts over landscapes meets the terrain on both
/// surfaces, and misses nothing", its tori (contract/surface_torus.kurven,
/// Phase 4): the seven named placements (where a crossing left its piece,
/// where one was not solved, grazing the terrain, over and far over its
/// edge, over the steps, over the sample not a number) and the even
/// placements of the seeded sweep, at scales 0.2 to 0.6. With the
/// polyhedra, over 150 parts met; every vertex of the torus's line within a
/// billionth of the cell of the terrain and of the tube; its line within
/// the solve's least step (a sixty-fourth of the lesser of a quarter cell
/// and an eighth of the tube) of every crossing found on a 240 by 120 grid
/// of its parameter rectangle; and no end of it but on a step of the
/// terrain.
#[test]
fn a_sweep_of_tori_over_landscapes_meets_the_terrain_and_misses_nothing() {
    let (polyhedra, _) = sweep(false);
    let (tori, elapsed) = sweep(true);
    let met = polyhedra.met + tori.met;
    assert!(
        met > 150 && tori.terrain <= 1.0 && tori.on_part <= 1.0,
        "every vertex and chord middle of {met} parts' contact lines lies on the terrain and on the part: torus off \
         the terrain by {:.2e} and off the part by {:.2e}, {elapsed:?}, the worst {}",
        tori.terrain,
        tori.on_part,
        tori.stray_example
    );
    assert!(
        tori.farthest <= 1.0,
        "the torus's contact line passes within the solve's least step of every crossing found apart from it: the \
         farthest {:.2} of the least step off it, at {}",
        tori.farthest,
        tori.example
    );
    assert!(
        tori.loose == 0,
        "the torus's contact line ends only on a step of the terrain: {} ends elsewhere",
        tori.loose
    );
}

/// "composite: a torus just touching a level terrain draws nothing, or the
/// whole of its circles" (`ParametricSurface::torus`, Phase 4): R 2, r 0.75,
/// at scale 0.4 at (0.13, 0.07, z), its crown or its bottom 1e-13 to 1e-6
/// either side of a flat terrain or a plateau at 0.5. Within a billionth of
/// the cell (2.5e-10) nothing; past it, two closed paths, each within 1e-12
/// of its circle ring ± √(tube² − (tube − eps)²) about the axis at height
/// 0.5, and as long as the circle to 1e-3.
#[test]
fn a_torus_just_touching_a_level_terrain_draws_nothing_or_the_whole_of_its_circles() {
    let camera = sweep_camera();
    let landscapes = sweep_landscapes();
    let surface = ParametricSurface::torus(2.0, 0.75, (64, 32)).unwrap_or_else(|e| panic!("{e}"));
    let scene = Scene::parametric_scene(surface, Vec::new(), camera, PreviewMode::Drawing, 0.0);
    let (s, tolerance) = (0.4, 1e-9 * 0.25);
    let (r, ring) = (0.75 * s, 2.0 * s);
    let axis = Vec2::new(0.13, 0.07);
    let (mut wrong, mut rungs) = (Vec::new(), 0);
    for (what, land) in [("a flat terrain", &landscapes[3].1), ("a plateau", &landscapes[4].1)] {
        for (end, sign) in [("crown", 1.0), ("bottom", -1.0)] {
            for exponent in -13..=-6 {
                let eps = real::pow(10.0, exponent as f64);
                // The crown eps over 0.5, or the bottom eps under it.
                let z = 0.5 - sign * r + sign * eps;
                let offset = Vec3::new(0.13, 0.07, z);
                let composite = Composite::new(
                    vec![
                        ("landscape".into(), land.clone(), Placement::IDENTITY),
                        ("torus".into(), scene.clone(), Placement::new(s, offset)),
                    ],
                    camera,
                )
                .unwrap();
                let mut paths: Vec<Vec<Vec3>> =
                    composite.contact(1).paths.iter().map(|p| p.iter().map(|v| s * v.v + offset).collect()).collect();
                rungs += 1;
                if eps <= tolerance {
                    if !paths.is_empty() {
                        wrong.push(format!("{what}, {end} {eps}: {} paths", paths.len()));
                    }
                    continue;
                }
                // Each path closed, on one circle all round, and its chords
                // as long as the circle's but for their sagitta.
                let half = (r * r - (r - eps) * (r - eps)).sqrt();
                let mut good = paths.len() == 2;
                let from_axis = |p: Vec3| (Vec2::new(p.x, p.y) - axis).length();
                paths.sort_by(|a, b| from_axis(a[0]).partial_cmp(&from_axis(b[0])).unwrap());
                for (path, rho) in paths.iter().zip([ring - half, ring + half]) {
                    let closed = (path[0] - path[path.len() - 1]).length() <= 1e-12;
                    let on_circle =
                        path.iter().all(|&p| (from_axis(p) - rho).abs() <= 1e-12 && (p.z - 0.5).abs() <= 1e-12);
                    let length: f64 = path.windows(2).map(|w| (w[1] - w[0]).length()).sum();
                    good = good && closed && on_circle && (length - 2.0 * PI * rho).abs() <= 1e-3 * 2.0 * PI * rho;
                }
                if !good {
                    wrong.push(format!("{what}, {end} {eps:.0e}: {} paths", paths.len()));
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "from 1e-13 to 1e-6 either side of touching, each rung draws nothing or both circles whole: {rungs} rungs, \
         e.g. {:?}",
        &wrong[..wrong.len().min(3)]
    );
}

/// "composite: a torus's contact costs with the curve's length, not the
/// area under it" (`ParametricSurface::torus` and the torus fixture, Phase
/// 4): a torus of R 2 and r 1e-3 and 1e-4 at scale 0.5 on the sinusoid; the
/// fixture torus at scale 1e3 with its outer wall across, and its top
/// through, the sinusoid on grids of 257 and 1025; and grazing a plateau on
/// a grid of 1025 by 1e-6. Each meets the terrain in under 10 s, off it by
/// at most a billionth of the cell.
#[test]
fn a_torus_contact_costs_with_the_curves_length_not_the_area_under_it() {
    let camera = sweep_camera();
    let landscapes = sweep_landscapes();
    let fine = |n: usize| grid_landscape(n, Truncation::None, Region::Full, &[], camera, sinusoid);
    let plateau = grid_landscape(1025, Truncation::Uniform(0.5), Region::Full, &[], camera, |_, _| 2.0);
    let fixture = surface("torus").unwrap_or_else(|e| panic!("torus: {e}"));
    let mut cases: Vec<(String, Scene, Scene, Placement)> = Vec::new();
    for r in [1e-3, 1e-4] {
        let torus = ParametricSurface::torus(2.0, r, (64, 32)).unwrap_or_else(|e| panic!("{e}"));
        cases.push((
            format!("a torus of R 2 and r {r} at scale 0.5"),
            landscapes[0].1.clone(),
            Scene::parametric_scene(torus, Vec::new(), camera, PreviewMode::Drawing, 0.0),
            Placement::new(0.5, Vec3::new(0.1, 0.05, 1.0)),
        ));
    }
    for n in [257, 1025] {
        let land = fine(n);
        cases.push((
            format!("the torus at scale 1e3, its outer wall across a grid of {n}"),
            land.clone(),
            fixture.clone(),
            Placement::new(1e3, Vec3::new(-2.75e3, 0.0, 1.0)),
        ));
        cases.push((
            format!("the torus at scale 1e3, its top through a grid of {n}"),
            land,
            fixture.clone(),
            Placement::new(1e3, Vec3::new(-2e3, 0.0, 1.0 - 750.0)),
        ));
    }
    cases.push((
        "the torus at scale 1e3 grazing a plateau on a grid of 1025".into(),
        plateau,
        fixture.clone(),
        Placement::new(1e3, Vec3::new(-2e3, 0.0, 0.5 - 750.0 + 1e-6)),
    ));
    let mut failures = Vec::new();
    for (what, land, part, placement) in cases {
        let tolerance = 1e-9 * landscape_cell(land.heightfield().unwrap());
        let composite = Composite::new(
            vec![("landscape".into(), land, Placement::IDENTITY), ("torus".into(), part, placement)],
            camera,
        )
        .unwrap();
        let start = Instant::now();
        let count = composite.contact(1).paths.len();
        let elapsed = start.elapsed();
        let stray = contact_stray(&composite, 1);
        if !(count > 0 && elapsed < Duration::from_secs(10) && stray.1 <= tolerance) {
            failures.push(format!(
                "{what} meets the terrain in reasonable time: {count} paths, off the terrain by {:.1e}, {elapsed:?}",
                stray.1
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// "composite: a face flush with the terrain draws the same at a rounding
/// either side": the L-prism at scale 0.5 at (0.1, 0.1, z), standing on a
/// flat terrain at 0.5 and on a plateau at 0.5, flush (z = 0.5) and 1e-15
/// over and under, meets it only along its edges and draws no contact; sunk
/// a little (z = 0.4) into the flat terrain, its sides meet it all round
/// the L, length 8 in its own units to 1e-12.
#[test]
fn a_face_flush_with_the_terrain_draws_the_same_at_a_rounding_either_side() {
    let camera = sweep_camera();
    let flat = grid_landscape(17, Truncation::None, Region::Full, &[], camera, |_, _| 0.5);
    let truncated = grid_landscape(17, Truncation::Uniform(0.5), Region::Full, &[], camera, |_, _| 2.0);
    let prism = solid_scene(l_prism(), camera);
    let drawn = |land: &Scene, z: f64| -> Vec<Vec<Vec3>> {
        Composite::new(
            vec![
                ("landscape".into(), land.clone(), Placement::IDENTITY),
                ("part".into(), prism.clone(), Placement::new(0.5, Vec3::new(0.1, 0.1, z))),
            ],
            camera,
        )
        .unwrap()
        .contact(1)
        .paths
        .iter()
        .map(|p| p.iter().map(|v| v.v).collect())
        .collect()
    };
    let mut failures = Vec::new();
    for (what, land) in [("on a flat terrain", &flat), ("on a plateau", &truncated)] {
        let (flush, over, under) = (drawn(land, 0.5), drawn(land, 0.5 + 1e-15), drawn(land, 0.5 - 1e-15));
        if !(flush.is_empty() && over.is_empty() && under.is_empty()) {
            failures.push(format!(
                "an L-prism standing {what}, flush and a rounding either way, meets it only along its edges, and \
                 draws no contact: {}, {} and {} paths",
                flush.len(),
                over.len(),
                under.len()
            ));
        }
    }
    let sunk = drawn(&flat, 0.4);
    let length: f64 = sunk.iter().map(|p| p.windows(2).map(|w| (w[1] - w[0]).length()).sum::<f64>()).sum();
    eprintln!("sunk: length {length} of 8");
    if !((length - 8.0).abs() <= 1e-12) {
        failures.push(format!(
            "sunk a little, its sides meet the terrain all round the L: length {length} in its own units, of 8"
        ));
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Where the Swift looks wrong, pinned (crates/FINDINGS.md, "the floor of a
/// footprint that is not convex"): `Slab.floorSpan` takes the hull of the
/// spans it finds in each of the footprint's convex pieces, so a stretch of
/// a segment whose lines of sight cross the floor's plane in the notch
/// between two pieces, where there is no floor, is reported hidden. A flat
/// terrain at 1 (its slab's base and top) on an L of [-1, 1]² less the
/// quarter x > 0, y > 0, seen from below along +z; the segment at z = 0 from
/// (-0.5, 0.8) to (0.8, -0.5) is under the L for s ≤ 0.5/1.3 and s ≥
/// 0.8/1.3, and under the notch between. Swift and the port hide all of it.
#[test]
fn the_floor_of_a_footprint_that_is_not_convex_hides_the_notch_as_swift_does() {
    let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (-1.0, 1.0)];
    let footprint = BoundaryPerimeter::new(
        (0..corners.len())
            .map(|k| {
                let (a, b) = (corners[k], corners[(k + 1) % corners.len()]);
                PerimeterEdge { start: P2::new(a.0, a.1), end: P2::new(b.0, b.1), density: 8 }
            })
            .collect(),
    );
    let scene = grid_landscape(17, Truncation::None, Region::Inside(footprint), &[], sweep_camera(), |_, _| 1.0);
    let slab = kurven_core::slab::Slab::new(scene.heightfield().unwrap(), "L").unwrap();
    assert_eq!((slab.base, slab.top), (1.0, 1.0));
    let (a, b, up) = (Vec3::new(-0.5, 0.8, 0.0), Vec3::new(0.8, -0.5, 0.0), Vec3::new(0.0, 0.0, 1.0));
    let floor = slab.copies.iter().filter_map(|copy| {
        let flat = |p: Vec3, moved| kurven_core::slab::Slab::flat(copy, p, moved);
        slab.floor_span(flat(a, true), flat(b - a, false), flat(up, false), copy)
    });
    let spans: Vec<(f64, f64)> = floor.collect();
    let seen = Composite::seen(slab.hidden(a, b, up));
    let (leaves, enters) = (0.5 / 1.3, 0.8 / 1.3);
    assert!(
        spans.len() == 1 && spans[0].0 <= 1e-12 && spans[0].1 >= 1.0 - 1e-12 && seen.is_empty(),
        "the floor's span is the hull of the L's two pieces, [0, 1], and nothing of the segment is seen, though \
         its lines of sight between s = {leaves:.4} and {enters:.4} cross no floor: spans {spans:?}, seen {seen:?}"
    );
}
