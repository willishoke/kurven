//! What the visibility suites share: a landscape built as
//! `NativeLandscape.build` builds it (its bundle needs the refiner, so it
//! lives here rather than in kurven-core), the cameras the Swift suites
//! look through, and the ray cast a polyhedron's visibility is held to.

#![allow(dead_code, clippy::neg_cmp_op_on_partial_ord, clippy::manual_range_contains)]

use kurven_core::bundle::KurvenBundle;
use kurven_core::camera::{rotation_x, rotation_z};
use kurven_core::grid::Grid2D;
use kurven_core::manifest::{Affine2, Domain, Interval, Region, Truncation, Walls};
use kurven_core::mesh::Mesh;
use kurven_core::scene::Heightfield;
use kurven_core::simd::{Mat3, Vec2, Vec3};
use kurven_core::solid::Solid;
use kurven_core::spaces::{Angle, DomainSpace, P2, P3, Transform, ViewSpace, WorldSpace};
use kurven_core::splitmix::SplitMix;
use kurven_core::surface::{Magnitude, Surface};
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

/// One triangle as `first_hit` tests it, with what does not depend on the
/// ray's origin worked out once, by the same operations in the same order:
/// its hit is `first_hit`'s, to the bit.
#[derive(Clone, Copy)]
struct Prepared {
    a: Vec3,
    e1: Vec3,
    e2: Vec3,
    p: Vec3,
    inv: f64,
}

impl Prepared {
    /// `first_hit`'s test of this triangle past its determinant's.
    #[inline]
    fn hit(&self, origin: Vec3, direction: Vec3, after: f64) -> Option<f64> {
        let s = origin - self.a;
        let u = s.dot(self.p) * self.inv;
        if !(u >= -1e-12 && u <= 1.0 + 1e-12) {
            return None;
        }
        let q = s.cross(self.e1);
        let v = direction.dot(q) * self.inv;
        if !(v >= -1e-12 && u + v <= 1.0 + 1e-12) {
            return None;
        }
        let hit = self.e2.dot(q) * self.inv;
        if !(hit > after) {
            return None;
        }
        Some(hit)
    }
}

/// `first_hit` against one mesh along one direction, asked of many origins:
/// the same brute force, over only the triangles that can answer. Each
/// triangle is projected along the direction onto a plane, and bucketed by
/// a grid there over its box, widened by a bound on how far outside it
/// Möller–Trumbore's rounding and tolerances can still take a hit; an
/// origin is tested against its cell's triangles alone. A triangle whose
/// bound is wide (nearly edge-on) is tested from every origin, and an
/// origin far outside the mesh against every triangle, so no triangle that
/// `first_hit` would hit is left out, and the least hit of the same set of
/// hits, computed the same way, is the same double. With
/// `KURVEN_RAY_INDEX_CHECK` set, every answer is checked against
/// `first_hit` itself.
pub struct RayIndex {
    direction: Vec3,
    prepared: Vec<Prepared>,
    /// Triangles tested from every origin.
    always: Vec<u32>,
    /// The plane's axes, and the grid on it: its low corner, its cell, its
    /// size, and each cell's triangles (`starts` into `items`).
    axes: (Vec3, Vec3),
    lo: Vec2,
    hi: Vec2,
    cell: Vec2,
    n: (usize, usize),
    starts: Vec<u32>,
    items: Vec<u32>,
    /// Origins farther than `reach` from `centre` are tested against every
    /// triangle: the widening assumes them within it.
    centre: Vec3,
    reach: f64,
    check: Option<Mesh<WorldSpace>>,
}

impl RayIndex {
    pub fn new(mesh: &Mesh<WorldSpace>, direction: Vec3) -> RayIndex {
        let eps = f64::EPSILON;
        let mut prepared = Vec::new();
        for t in &mesh.triangles {
            let (a, b, c) =
                (mesh.vertices[t[0] as usize].v, mesh.vertices[t[1] as usize].v, mesh.vertices[t[2] as usize].v);
            let (e1, e2) = (b - a, c - a);
            let p = direction.cross(e2);
            let det = e1.dot(p);
            // `first_hit` skips it, whatever the origin.
            if !(det.abs() > 1e-14) {
                continue;
            }
            prepared.push(Prepared { a, e1, e2, p, inv: 1.0 / det });
        }
        let (mut blo, mut bhi) = (Vec3::splat(f64::INFINITY), Vec3::splat(f64::NEG_INFINITY));
        for v in &mesh.vertices {
            blo = blo.min(v.v);
            bhi = bhi.max(v.v);
        }
        let centre = if blo.x.is_finite() { 0.5 * (blo + bhi) } else { Vec3::ZERO };
        let radius = if blo.x.is_finite() { 0.5 * (bhi - blo).length() } else { 0.0 };
        let reach = 4.0 * (radius + centre.length()) + 1e-300;
        let along = direction.length();
        let unit = direction / along;
        let ax = unit.cross(if unit.z.abs() < 0.9 { Vec3::new(0.0, 0.0, 1.0) } else { Vec3::new(1.0, 0.0, 0.0) });
        let ax = ax.normalized();
        let ay = unit.cross(ax);
        // The farthest an origin within reach and a vertex lie from 0, and
        // so the most an origin lies from a vertex.
        let far = centre.length() + reach;
        let span = reach + radius;
        let project = |x: Vec3| Vec2::new(x.dot(ax), x.dot(ay));
        let mut always = Vec::new();
        let mut boxes: Vec<Option<(Vec2, Vec2)>> = Vec::with_capacity(prepared.len());
        for (k, t) in prepared.iter().enumerate() {
            let (a, b, c) = (t.a, t.a + t.e1, t.a + t.e2);
            let (l1, l2, l3) = (t.e1.length(), t.e2.length(), (c - b).length());
            let det = 1.0 / t.inv;
            // How far below 0 or above 1 a barycentric the rounding of
            // `s · p`, `d · (s × e1)` and the determinant lets through, on
            // top of the 1e-12 allowed: each within a few ulps of the
            // magnitudes summed, here 64.
            let slack = 1e-12 * (1.0 + 1e-6)
                + 64.0 * eps * along * (span * swift::max(l1, l2) + l1 * l2 + span * (l1 + l2)) / det.abs();
            // A barycentric off by `slack` is a point off the triangle by at
            // most three times it along its longest side; and the
            // projections of the origin and of the corners round by a few
            // ulps of their size.
            let widen = 4.0 * slack * swift::max(swift::max(l1, l2), l3) + 64.0 * eps * (far + l1 + l2) * (1.0 + along);
            let (pa, pb, pc) = (project(a), project(b), project(c));
            let lo = pa.min(pb).min(pc) - Vec2::splat(widen);
            let hi = pa.max(pb).max(pc) + Vec2::splat(widen);
            if !(widen.is_finite() && widen < 1e-3 * span && lo.x.is_finite() && hi.x.is_finite()) {
                always.push(k as u32);
                boxes.push(None);
            } else {
                boxes.push(Some((lo, hi)));
            }
        }
        let (mut lo, mut hi) = (Vec2::splat(f64::INFINITY), Vec2::splat(f64::NEG_INFINITY));
        let mut count = 0usize;
        for b in boxes.iter().flatten() {
            lo = lo.min(b.0);
            hi = hi.max(b.1);
            count += 1;
        }
        let side = ((2.0 * (count as f64).sqrt()).ceil() as usize).clamp(1, 1024);
        let n = if count == 0 { (1, 1) } else { (side, side) };
        let cell = if count == 0 {
            Vec2::splat(1.0)
        } else {
            Vec2::new(swift::max((hi.x - lo.x) / n.0 as f64, 1e-300), swift::max((hi.y - lo.y) / n.1 as f64, 1e-300))
        };
        let mut index = RayIndex {
            direction,
            prepared,
            always,
            axes: (ax, ay),
            lo,
            hi,
            cell,
            n,
            starts: Vec::new(),
            items: Vec::new(),
            centre,
            reach,
            check: std::env::var_os("KURVEN_RAY_INDEX_CHECK").map(|_| mesh.clone()),
        };
        let mut counts = vec![0u32; n.0 * n.1 + 1];
        for b in boxes.iter().flatten() {
            let ((i0, j0), (i1, j1)) = (index.key(b.0), index.key(b.1));
            for j in j0..=j1 {
                for i in i0..=i1 {
                    counts[j * n.0 + i + 1] += 1;
                }
            }
        }
        for k in 1..counts.len() {
            counts[k] += counts[k - 1];
        }
        let mut fill = counts.clone();
        let mut items = vec![0u32; counts[n.0 * n.1] as usize];
        for (k, b) in boxes.iter().enumerate() {
            let Some(b) = b else { continue };
            let ((i0, j0), (i1, j1)) = (index.key(b.0), index.key(b.1));
            for j in j0..=j1 {
                for i in i0..=i1 {
                    let c = j * n.0 + i;
                    items[fill[c] as usize] = k as u32;
                    fill[c] += 1;
                }
            }
        }
        index.starts = counts;
        index.items = items;
        index
    }

    /// The grid's cell of a point on the plane, clamped to the grid:
    /// monotone in each coordinate, so a point in a box is keyed between its
    /// corners' keys.
    fn key(&self, x: Vec2) -> (usize, usize) {
        let i = ((x.x - self.lo.x) / self.cell.x).floor();
        let j = ((x.y - self.lo.y) / self.cell.y).floor();
        let clamp = |k: f64, n: usize| -> usize {
            if !(k > 0.0) {
                0
            } else if k >= (n - 1) as f64 {
                n - 1
            } else {
                k as usize
            }
        };
        (clamp(i, self.n.0), clamp(j, self.n.1))
    }

    /// Each candidate triangle's hit from `origin`, until `stop` says so.
    fn visit(&self, origin: Vec3, after: f64, mut stop: impl FnMut(f64) -> bool) {
        let d = self.direction;
        if !((origin - self.centre).length() <= self.reach) {
            for t in &self.prepared {
                if let Some(h) = t.hit(origin, d, after)
                    && stop(h)
                {
                    return;
                }
            }
            return;
        }
        for &k in &self.always {
            if let Some(h) = self.prepared[k as usize].hit(origin, d, after)
                && stop(h)
            {
                return;
            }
        }
        let x = Vec2::new(origin.dot(self.axes.0), origin.dot(self.axes.1));
        if !(x.x >= self.lo.x && x.x <= self.hi.x && x.y >= self.lo.y && x.y <= self.hi.y) {
            return;
        }
        let (i, j) = self.key(x);
        let c = j * self.n.0 + i;
        for &k in &self.items[self.starts[c] as usize..self.starts[c + 1] as usize] {
            if let Some(h) = self.prepared[k as usize].hit(origin, d, after)
                && stop(h)
            {
                return;
            }
        }
    }

    fn checked<R: PartialEq + std::fmt::Debug>(
        &self,
        origin: Vec3,
        after: f64,
        got: R,
        want: impl Fn(Option<f64>) -> R,
    ) -> R {
        if let Some(mesh) = &self.check {
            let brute = want(first_hit(mesh, origin, self.direction, after));
            assert_eq!(got, brute, "the ray index and first_hit from {origin:?} along {:?}", self.direction);
        }
        got
    }

    /// `first_hit(mesh, origin, direction, after)`, the same double.
    pub fn first_hit(&self, origin: Vec3, direction: Vec3, after: f64) -> Option<f64> {
        if direction.to_array().map(f64::to_bits) != self.direction.to_array().map(f64::to_bits) {
            panic!("a ray index asked along {direction:?}, built along {:?}", self.direction);
        }
        let mut best: Option<f64> = None;
        self.visit(origin, after, |hit| {
            if best.is_none_or(|b| hit < b) {
                best = Some(hit);
            }
            false
        });
        self.checked(origin, after, best.map(f64::to_bits), |b| b.map(f64::to_bits));
        best
    }

    /// `first_hit(mesh, origin, direction, after).is_some()`, stopping at
    /// the first hit.
    pub fn hits(&self, origin: Vec3, direction: Vec3, after: f64) -> bool {
        if direction.to_array().map(f64::to_bits) != self.direction.to_array().map(f64::to_bits) {
            panic!("a ray index asked along {direction:?}, built along {:?}", self.direction);
        }
        let mut any = false;
        self.visit(origin, after, |_| {
            any = true;
            true
        });
        self.checked(origin, after, any, |b| b.is_some())
    }

    pub fn direction(&self) -> Vec3 {
        self.direction
    }
}

fn same_direction(a: Vec3, b: Vec3) -> bool {
    a.to_array().map(f64::to_bits) == b.to_array().map(f64::to_bits)
}

/// Meshes cast against from many origins, along one direction as a rule:
/// a `RayIndex` of each, built along the first direction asked, and
/// `first_hit` itself along any other.
pub struct RayCache {
    meshes: Vec<Mesh<WorldSpace>>,
    built: std::sync::OnceLock<Vec<RayIndex>>,
}

impl RayCache {
    pub fn new(meshes: Vec<Mesh<WorldSpace>>) -> RayCache {
        RayCache { meshes, built: std::sync::OnceLock::new() }
    }

    pub fn meshes(&self) -> &[Mesh<WorldSpace>] {
        &self.meshes
    }

    /// The indexes, when they were built along `direction`.
    fn along(&self, direction: Vec3) -> Option<&[RayIndex]> {
        let built = self.built.get_or_init(|| self.meshes.iter().map(|m| RayIndex::new(m, direction)).collect());
        (built.is_empty() || same_direction(built[0].direction, direction)).then_some(built.as_slice())
    }

    /// `first_hit(mesh, origin, direction, after).is_none()` for every mesh.
    pub fn clear(&self, origin: Vec3, direction: Vec3, after: f64) -> bool {
        match self.along(direction) {
            Some(indexes) => indexes.iter().all(|r| !r.hits(origin, direction, after)),
            None => self.meshes.iter().all(|m| first_hit(m, origin, direction, after).is_none()),
        }
    }

    /// `first_hit` against the one mesh `k`.
    pub fn first_hit(&self, k: usize, origin: Vec3, direction: Vec3, after: f64) -> Option<f64> {
        match self.along(direction) {
            Some(indexes) => indexes[k].first_hit(origin, direction, after),
            None => first_hit(&self.meshes[k], origin, direction, after),
        }
    }
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

#[derive(Clone, Copy, PartialEq)]
pub struct DenseGap {
    pub over_f: f64,
    pub over_truncation: f64,
    pub cosine: f64,
}

/// Whether f hides `p` from the eye, by walking the line of sight in steps of
/// a sixty-fourth of a cell and bisecting each crossing of the surface it
/// finds, under the traversal's own meaning of the depth tolerance: in its own surface's
/// neighbourhood from the start, the line is forgiven the depth tolerance
/// perpendicular to the surface; once it has been outside, being inside at
/// or beyond the depth tolerance's reach is occlusion. The oracle for `GridTraversal`
/// with the function at hand. (Swift's trace is left out: it only prints.)
#[allow(clippy::too_many_arguments)]
pub fn densely_hidden(
    h: &Heightfield,
    magnitude: &Magnitude,
    p: Vec3,
    toward: Vec3,
    depth_tolerance: f64,
    per_unit: f64,
    cell: f64,
) -> Option<f64> {
    let reach = depth_tolerance / swift::max(per_unit, 1e-12);
    let mut top = f64::NEG_INFINITY;
    h.for_each_sample(|q| top = swift::max(top, q.z()));
    let d = toward;
    let mut t_max = f64::INFINITY;
    if d.z > 0.0 {
        t_max = swift::max((top - p.z) / d.z, 0.0);
    } else if d.z == 0.0 && p.z > top {
        return None;
    }
    assert_eq!(h.tiles.len(), 1, "the dense cast reads f on the fundamental tile only");
    let e = 1e-4 * cell;
    let dom = h.surface.domain();
    let (blo, bhi) = (
        Vec2::new(swift::min(dom.real.lo, dom.real.hi), swift::min(dom.imag.lo, dom.imag.hi)),
        Vec2::new(swift::max(dom.real.lo, dom.real.hi), swift::max(dom.imag.lo, dom.imag.hi)),
    );
    // The line's gaps over f and over the truncation at `t`, and f's cosine; None
    // outside the region.
    let gaps = |t: f64| -> Option<DenseGap> {
        let q = P2::<WorldSpace>::new(p.x + t * d.x, p.y + t * d.y);
        // Nothing is drawn outside the domain's box: the traversal stops at the
        // grid's edge, and f beyond it is not the solid.
        if !(q.x() >= blo.x && q.x() <= bhi.x && q.y() >= blo.y && q.y() <= bhi.y && h.region.contains(q)) {
            return None;
        }
        let dq = P2::<DomainSpace>::new(q.x(), q.y());
        let u = magnitude(dq);
        let truncation_height = h.surface.truncation.height_at(dq.x());
        let hx = (magnitude(P2::new(dq.x() + e, dq.y())) - magnitude(P2::new(dq.x() - e, dq.y()))) / (2.0 * e);
        let hy = (magnitude(P2::new(dq.x(), dq.y() + e)) - magnitude(P2::new(dq.x(), dq.y() - e))) / (2.0 * e);
        let cosine = 1.0 / (1.0 + hx * hx + hy * hy).sqrt();
        let z = p.z + t * d.z;
        Some(DenseGap {
            over_f: z - if u.is_nan() { f64::INFINITY } else { u },
            over_truncation: z - truncation_height,
            cosine: if cosine.is_nan() { 0.0 } else { cosine },
        })
    };
    let inside =
        |g: Option<DenseGap>, z_eps: f64| -> bool { g.is_some_and(|g| g.over_f < z_eps && g.over_truncation < z_eps) };
    let mut outside_yet = false;
    let t_eps = 1e-9 * cell;
    // As the traversal: a point on the solid's boundary -- the domain's edge --
    // is tested only from where the line goes next.
    let strictly_inside = p.x > blo.x + 1e-9 && p.x < bhi.x - 1e-9 && p.y > blo.y + 1e-9 && p.y < bhi.y - 1e-9;
    let mut hidden = |t: f64, g: Option<DenseGap>| -> bool {
        let z_eps = if t > t_eps { 0.0 } else { 1e-9 * cell };
        let Some(g) = g.filter(|_| inside(g, z_eps)) else {
            // As the traversal: outside by less than the depth tolerance is within the
            // surface's skin, not outside it.
            if let Some(g) = g {
                let clearance = if g.over_f > g.over_truncation { g.over_f * g.cosine } else { g.over_truncation };
                if clearance <= depth_tolerance {
                    return false;
                }
            }
            outside_yet = true;
            return false;
        };
        if !(t > t_eps || strictly_inside) {
            return false;
        }
        if outside_yet {
            return t + t_eps >= reach;
        }
        let depth = if g.over_f > g.over_truncation { g.over_f * g.cosine } else { g.over_truncation };
        depth + depth_tolerance < 0.0
    };
    // The line leaves the region eventually; walk until it has.
    let step = cell / 64.0;
    let mut t = 0.0;
    let mut previous = gaps(0.0);
    if hidden(0.0, previous) {
        return Some(0.0);
    }
    let mut left = 0;
    while t < t_max && left < 100_000 {
        let tn = t + step;
        let g = gaps(tn);
        // A crossing of the surface between the two: bisect it and test
        // just inside.
        if inside(previous, 0.0) != inside(g, 0.0) {
            let (mut lo, mut hi) = (t, tn);
            let inside_lo = inside(previous, 0.0);
            for _ in 0..50 {
                let m = 0.5 * (lo + hi);
                if inside(gaps(m), 0.0) == inside_lo {
                    lo = m;
                } else {
                    hi = m;
                }
            }
            let tc = if inside_lo { lo } else { hi };
            if hidden(tc, gaps(tc)) {
                return Some(tc);
            }
        }
        // Leaving the region between the two: the traversal tests the exit
        // itself, so the cast does too -- bisected, and tested a hair
        // inside, where a line passing under a top edge is inside the solid --
        // before the sample beyond it, which is off the region.
        if previous.is_some() && g.is_none() {
            let (mut lo, mut hi) = (t, tn);
            for _ in 0..50 {
                let m = 0.5 * (lo + hi);
                if gaps(m).is_some() {
                    lo = m;
                } else {
                    hi = m;
                }
            }
            if hidden(lo, gaps(lo)) {
                return Some(lo);
            }
        }
        if hidden(tn, g) {
            return Some(tn);
        }
        if g.is_none() && previous.is_none() {
            left += 1;
            if left > 64 {
                break;
            }
        } else {
            left = 0;
        }
        previous = g;
        t = tn;
    }
    None
}

/// A piece of a solid as triangles, fanned, for the brute-force ray cast
/// (Solids.swift's `pieceMesh`).
pub fn piece_mesh(s: &Solid, f: usize) -> Mesh<WorldSpace> {
    let piece = &s.pieces[f];
    Mesh::new(
        piece.iter().map(|&k| s.vertices[k]).collect(),
        (1..piece.len() - 1).map(|k| [0, k as i32, k as i32 + 1]).collect(),
    )
}

/// Every piece of a solid as its `piece_mesh`, by index.
pub fn piece_meshes(s: &Solid) -> Vec<Mesh<WorldSpace>> {
    (0..s.pieces.len()).map(|f| piece_mesh(s, f)).collect()
}

/// Whether a point of the solid's surface, lying in the pieces `own`, is
/// seen along `d`: one of its own pieces faces the eye and no other piece is
/// hit by the ray from it (Solids.swift's `rayVisible`). The brute force
/// every occlusion test is held to. `meshes` are the solid's `piece_meshes`,
/// made once rather than per point, as Swift makes them.
pub fn ray_visible(s: &Solid, meshes: &[Mesh<WorldSpace>], p: Vec3, own: &[usize], d: Vec3) -> bool {
    if !own.iter().any(|&f| s.normals[f].dot(d) > 0.0) {
        return false;
    }
    for (f, mesh) in meshes.iter().enumerate() {
        if own.contains(&f) {
            continue;
        }
        if first_hit(mesh, p, d, 1e-9).is_some() {
            return false;
        }
    }
    true
}

/// Whether the line from `origin` along `direction`, up to distance
/// `up_to`, passes under some triangle of `mesh` by more than
/// `depth_tolerance`, perpendicular to that triangle (Stress.swift's
/// `underSomeTriangle`; stress.rs keeps its own copy). Brute force over the
/// triangles, sharing nothing with the traversal.
pub fn under_some_triangle(
    mesh: &Mesh<WorldSpace>,
    origin: Vec3,
    direction: Vec3,
    depth_tolerance: f64,
    start_inside: bool,
    up_to: f64,
) -> bool {
    let (o, d) = (Vec2::new(origin.x, origin.y), Vec2::new(direction.x, direction.y));
    for t in &mesh.triangles {
        let (a, b, c) =
            (mesh.vertices[t[0] as usize].v, mesh.vertices[t[1] as usize].v, mesh.vertices[t[2] as usize].v);
        let n = (b - a).cross(c - a);
        let len = n.length();
        // A wall, seen from above, has no footprint and is no one's triangle.
        if !(len > 0.0 && n.z.abs() > 1e-12 * len) {
            continue;
        }
        let cosine = n.z.abs() / len;
        let corners = [Vec2::new(a.x, a.y), Vec2::new(b.x, b.y), Vec2::new(c.x, c.y)];
        let sign = if n.z > 0.0 { 1.0 } else { -1.0 };
        let (mut lo, mut hi) = (0.0f64, f64::INFINITY);
        let mut empty = false;
        for k in 0..3 {
            let (p, q) = (corners[k], corners[(k + 1) % 3]);
            let e = q - p;
            let f0 = sign * (e.x * (o.y - p.y) - e.y * (o.x - p.x));
            let f1 = sign * (e.x * d.y - e.y * d.x);
            if f1.abs() < 1e-15 {
                if f0 < -1e-12 {
                    empty = true;
                    break;
                }
                continue;
            }
            let root = -f0 / f1;
            if f1 > 0.0 {
                lo = swift::max(lo, root);
            } else {
                hi = swift::min(hi, root);
            }
        }
        hi = swift::min(hi, up_to);
        if empty || !(lo <= hi) || !(hi.is_finite() || lo.is_finite()) {
            continue;
        }
        let gap = |s: f64| -> f64 {
            let p = origin + s * direction;
            let z = a.z - (n.x * (p.x - a.x) + n.y * (p.y - a.y)) / n.z;
            (p.z - z) * cosine
        };
        for s in [lo, hi] {
            if s.is_finite() && (s > 1e-12 || start_inside) && gap(s) + depth_tolerance < 0.0 {
                return true;
            }
        }
    }
    false
}

/// Stress.swift's `exactlyHidden`, the oracle for the traversal (stress.rs
/// keeps its own copy): a point that starts inside the solid (strictly over
/// the domain) is in its own surface's neighbourhood until the line first
/// crosses out, hidden there only under some triangle by more than the
/// depth tolerance; after that, any crossing of the boundary beyond the
/// depth tolerance's reach along the line is occlusion.
#[allow(clippy::too_many_arguments)]
pub fn exactly_hidden(
    mesh: &Mesh<WorldSpace>,
    p: Vec3,
    toward: Vec3,
    depth_tolerance: f64,
    per_unit: f64,
    strictly_inside: bool,
    cell: f64,
) -> bool {
    let reach = depth_tolerance / per_unit;
    let mut exit = 0.0;
    if under_some_triangle(mesh, p, Vec3::new(0.0, 0.0, 1.0), -1e-9 * cell, true, 0.0) {
        exit = first_hit(mesh, p, toward, 1e-10).unwrap_or(f64::INFINITY);
    }
    if under_some_triangle(mesh, p, toward, depth_tolerance, strictly_inside, exit) {
        return true;
    }
    first_hit(mesh, p, toward, swift::max(exit, reach) + 1e-10).is_some()
}

/// The view depth per unit of travel along the line of sight.
pub fn per_unit(view: &Transform<WorldSpace, ViewSpace>, toward: Vec3) -> f64 {
    let c = view.m.columns;
    Vec3::new(c[0].z, c[1].z, c[2].z).dot(toward).abs()
}
