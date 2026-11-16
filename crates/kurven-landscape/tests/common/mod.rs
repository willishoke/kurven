//! What the visibility suites share: a landscape built as
//! `NativeLandscape.build` builds it (its bundle needs the refiner, so it
//! lives here rather than in kurven-core), and the cameras the Swift suites
//! look through.

#![allow(dead_code, clippy::neg_cmp_op_on_partial_ord, clippy::manual_range_contains)]

use kurven_core::bundle::KurvenBundle;
use kurven_core::camera::{rotation_x, rotation_z};
use kurven_core::grid::Grid2D;
use kurven_core::manifest::{Affine2, Domain, Interval, Region, Truncation, Walls};
use kurven_core::mesh::Mesh;
use kurven_core::scene::Heightfield;
use kurven_core::simd::{Mat3, Vec3};
use kurven_core::spaces::{Angle, P3, Transform, ViewSpace, WorldSpace};
use kurven_core::splitmix::SplitMix;
use kurven_core::surface::Surface;
use kurven_landscape::{LandscapeRequest, NativeLandscape};
use kurven_math::swift;

/// `NativeLandscape.build(request, refine:)`, as Swift's suites call it:
/// the git sha "native", the error unwrapped.
pub fn build(request: &LandscapeRequest, refine: bool) -> KurvenBundle {
    NativeLandscape::build(request, "native", refine).unwrap_or_else(|e| panic!("{}: {e:?}", request.expression))
}

/// The test suites' camera: `rotationX(elevation) * rotationZ(azimuth) * shear`,
/// its rows the view's.
pub fn test_camera(elevation: f64, azimuth: f64, shear: f64) -> Transform<WorldSpace, ViewSpace> {
    let s = Mat3::from_rows(Vec3::new(1.0, shear, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0));
    let m = rotation_x(Angle::from_degrees(elevation)) * rotation_z(Angle::from_degrees(azimuth)) * s;
    Transform::from_rows((m.row(0), m.row(1), m.row(2)), Vec3::ZERO)
}

/// The cameras the stress tests orbit through: from nearly overhead to a
/// degree off edge-on, round the compass, some sheared as the plates are.
pub fn stress_cameras() -> Vec<(String, Transform<WorldSpace, ViewSpace>)> {
    let mut out = Vec::new();
    for elevation in [-80.0, -55.0, -30.0, -10.0, -3.0, -1.0] {
        for azimuth in [0.0, 37.0, 90.0, 135.0, 200.0, 289.0] {
            out.push((
                format!("el {}° az {}°", -elevation as i64, azimuth as i64),
                test_camera(elevation, azimuth, 0.0),
            ));
        }
    }
    out.push(("el 55° sheared".into(), test_camera(-55.0, -20.0, 0.5)));
    out.push(("el 5° sheared".into(), test_camera(-5.0, 70.0, 0.5)));
    out
}

/// `items.map(body)` over the machine's threads, in order.
pub fn par_map<T: Sync, R: Send>(items: &[T], body: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = items.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles: Vec<_> =
            items.chunks(chunk).map(|c| scope.spawn(|| c.iter().map(&body).collect::<Vec<R>>())).collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

/// The ray from `origin` along `direction` against every triangle of
/// `mesh`: the parameter of the first hit past `after`, or None. Brute
/// force, as an oracle should be: nothing about the heightfield's structure
/// is used, only its triangles.
pub fn first_hit(mesh: &Mesh<WorldSpace>, origin: Vec3, direction: Vec3, after: f64) -> Option<f64> {
    let mut best: Option<f64> = None;
    for t in &mesh.triangles {
        let (a, b, c) =
            (mesh.vertices[t[0] as usize].v, mesh.vertices[t[1] as usize].v, mesh.vertices[t[2] as usize].v);
        // Möller–Trumbore.
        let (e1, e2) = (b - a, c - a);
        let p = direction.cross(e2);
        let det = e1.dot(p);
        if !(det.abs() > 1e-14) {
            continue;
        }
        let inv = 1.0 / det;
        let s = origin - a;
        let u = s.dot(p) * inv;
        if !(u >= -1e-12 && u <= 1.0 + 1e-12) {
            continue;
        }
        let q = s.cross(e1);
        let v = direction.dot(q) * inv;
        if !(v >= -1e-12 && u + v <= 1.0 + 1e-12) {
            continue;
        }
        let hit = e2.dot(q) * inv;
        if !(hit > after) {
            continue;
        }
        if best.is_none_or(|b| hit < b) {
            best = Some(hit);
        }
    }
    best
}

/// The drawn solid of a one-tile, uncut heightfield as explicit triangles:
/// the grid's cells as the rasterizer splits them, clamped at the
/// vertices, except that a triangle the truncation cuts is replaced by its truncation edge
/// pieces (`Mesh::truncation_edge_mesh`), so the surface is exactly min(interpolated, truncation);
/// plus wall curtains down to `base` and a floor. This is the boundary of
/// what the depth pass draws, and a line of sight that is inside the solid
/// anywhere crosses one of these triangles.
pub fn drawn_triangles(h: &Heightfield, base: f64) -> Mesh<WorldSpace> {
    let s = &h.surface;
    let g = &*s.height;
    let step = h.step.max(1);
    let nx = g.width.div_ceil(step);
    let ny = g.height.div_ceil(step);
    let at = |i: usize, j: usize| -> P3<WorldSpace> {
        let p = g.position(i * step, j * step);
        P3::new(p.x(), p.y(), swift::min(f64::from(g.at(i * step, j * step)), s.truncation.height_at(p.x())))
    };
    let excess = |i: usize, j: usize| -> f64 {
        let p = g.position(i * step, j * step);
        f64::from(g.at(i * step, j * step)) - s.truncation.height_at(p.x())
    };
    let mut vertices: Vec<P3<WorldSpace>> = Vec::new();
    let mut triangles: Vec<[i32; 3]> = Vec::new();
    let mut tri = |a: P3<WorldSpace>, b: P3<WorldSpace>, c: P3<WorldSpace>| {
        let k = vertices.len() as i32;
        vertices.extend([a, b, c]);
        triangles.push([k, k + 1, k + 2]);
    };
    // A triangle the truncation cuts: a corner over it and a corner under it.
    let cut = |corners: [(usize, usize); 3]| -> bool {
        let e = corners.map(|(i, j)| excess(i, j));
        e.iter().any(|&x| x > 0.0) && e.iter().any(|&x| x < 0.0)
    };
    for j in 0..(ny - 1) {
        for i in 0..(nx - 1) {
            if !cut([(i, j), (i + 1, j), (i, j + 1)]) {
                tri(at(i, j), at(i + 1, j), at(i, j + 1));
            }
            if !cut([(i + 1, j), (i + 1, j + 1), (i, j + 1)]) {
                tri(at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
            }
        }
    }
    // Walls round the four sides, up to the drawn top edge -- which, where a
    // boundary cell straddles the truncation, bends at the truncation edge -- and the floor.
    let raw = |i: usize, j: usize| -> P3<WorldSpace> {
        let p = g.position(i * step, j * step);
        P3::new(p.x(), p.y(), f64::from(g.at(i * step, j * step)))
    };
    let mut curtain = |i0: usize, j0: usize, i1: usize, j1: usize| {
        let (a, b) = (raw(i0, j0), raw(i1, j1));
        let (ca, cb) = (s.truncation.height_at(a.x()), s.truncation.height_at(b.x()));
        let mut top_edge = vec![P3::<WorldSpace>::new(a.x(), a.y(), swift::min(a.z(), ca))];
        if (a.z() - ca > 0.0) != (b.z() - cb > 0.0) && ca.is_finite() && cb.is_finite() {
            let t = (a.z() - ca) / ((a.z() - ca) - (b.z() - cb));
            top_edge.push(P3::new(a.x() + (b.x() - a.x()) * t, a.y() + (b.y() - a.y()) * t, ca + (cb - ca) * t));
        }
        top_edge.push(P3::new(b.x(), b.y(), swift::min(b.z(), cb)));
        for w in top_edge.windows(2) {
            let (p, q) = (w[0], w[1]);
            let (p0, q0) = (P3::new(p.x(), p.y(), base), P3::new(q.x(), q.y(), base));
            tri(p, q, p0);
            tri(q, q0, p0);
        }
    };
    for i in 0..(nx - 1) {
        curtain(i, 0, i + 1, 0);
        curtain(i, ny - 1, i + 1, ny - 1);
    }
    for j in 0..(ny - 1) {
        curtain(0, j, 0, j + 1);
        curtain(nx - 1, j, nx - 1, j + 1);
    }
    let (c00, c10, c01, c11) = (at(0, 0), at(nx - 1, 0), at(0, ny - 1), at(nx - 1, ny - 1));
    let floor = |p: P3<WorldSpace>| P3::new(p.x(), p.y(), base);
    tri(floor(c00), floor(c10), floor(c01));
    tri(floor(c10), floor(c11), floor(c01));
    let truncation_edge = Mesh::truncation_edge_mesh(s, step, &h.region, &h.tiles);
    Mesh::concat(&[Mesh::new(vertices, triangles), truncation_edge])
}

/// A small heightfield with random heights, for the grid tests.
pub fn random_heightfield(cells: usize, seed: u64, truncation: Truncation, amplitude: f64) -> Heightfield {
    let n = cells + 1;
    let mut rng = SplitMix::new(seed);
    let domain = Domain::new(Interval::new(-1.0, 1.0), Interval::new(-1.5, 0.5));
    let values: Vec<f32> = (0..n * n).map(|_| rng.next(0.0, amplitude) as f32).collect();
    let surface = Surface::new(Grid2D::new(n, n, domain, values), None, truncation, false);
    Heightfield::new(surface, Mesh::empty(), vec![Affine2::IDENTITY], Region::Full, 1, None, Walls::None)
}
