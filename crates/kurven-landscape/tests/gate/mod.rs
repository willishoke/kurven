//! The composite gate (`Sources/kurven-test/Composite.swift`, lines
//! 1-760, and the helpers Contact.swift shares with it): a composite's every
//! curve, held point by point to the brute-force casts the stress, solids
//! and occlusion suites run, each against its own geometry where it stands
//! -- a landscape's drawn triangles, a polyhedron's pieces, the round
//! torus's tube walked a step at a time. A point is seen exactly when its own
//! geometry shows it -- the geometry drawn alone, which is held to its own
//! cast here too -- and no other geometry's cast meets the line of sight
//! from it.
//!
//! Shared by composite.rs, contact.rs and isophotes.rs, each of which
//! declares `mod common;` before `mod gate;`.
//!
//! The round torus (`SurfaceBundle`, `ParametricSurface::torus`) is Phase
//! 4's: `surface` refuses, so the compositions that stand it in a composite
//! come back refused (`gate_compositions`), and the suites hold them in
//! ignored tests. The torus's casts and own truths are ported here all the
//! same, so those tests need only the torus.

#![allow(dead_code, clippy::neg_cmp_op_on_partial_ord, clippy::type_complexity, clippy::too_many_arguments)]

use std::sync::Arc;

use crate::common::{
    build, drawn_triangles, exactly_hidden, first_hit, par_map, piece_meshes, ray_visible, test_camera,
};
use kurven_core::bundle::KurvenBundle;
use kurven_core::camera::{Camera, Projection};
use kurven_core::composite::{Composite, Part, Placement};
use kurven_core::contact::ContactCurves;
use kurven_core::fixtures;
use kurven_core::hatching::{FaceKind, FaceSelector};
use kurven_core::hidden_line::HeightfieldVisibility;
use kurven_core::manifest::{Domain, Interval, LayerRole, LayerSource, LayerSpec, Truncation, Walls};
use kurven_core::mesh::Mesh;
use kurven_core::parametric::{ParametricSurface, PortRefusal};
use kurven_core::polyhedron_bundle::PolyhedronBundle;
use kurven_core::polyline::{Layer, PolylineSet};
use kurven_core::scene::{Heightfield, Scene, SceneGeometry};
use kurven_core::simd::{Vec2, Vec3};
use kurven_core::spaces::{AABB, P2, P3, Transform, ViewSpace, WorldSpace};
use kurven_landscape::LandscapeRequest;
use kurven_math::swift;

/// One geometry of a composite as the gate sees it: whether the line of
/// sight from a world point toward the eye meets it, by brute force; None
/// where the answer is a rounding (the line grazes it, or the point lies on
/// it).
pub type Cast = Arc<dyn Fn(Vec3, Vec3) -> Option<bool> + Send + Sync>;

/// A box: its low and high corners.
pub type Box3 = (Vec3, Vec3);

/// The decimated grid's cell of a landscape, the larger of its sides
/// (Contact.swift's `landscapeCell`).
pub fn landscape_cell(h: &Heightfield) -> f64 {
    let g = &*h.surface.height;
    let d = h.surface.domain();
    swift::max(d.real.length().abs() / (g.width - 1) as f64, d.imag.length().abs() / (g.height - 1) as f64)
        * h.step.max(1) as f64
}

/// Two directions across `d`, for nudging a point off the line it lies on.
pub fn across_sight(d: Vec3) -> [Vec3; 4] {
    let a = d.cross(if d.z.abs() < 0.9 { Vec3::new(0.0, 0.0, 1.0) } else { Vec3::new(1.0, 0.0, 0.0) }).normalized();
    let b = d.cross(a);
    [a, -a, b, -b]
}

/// A cast against triangles: clear when the ray meets none past a
/// billionth of `size`; None when a nudge of a ten-millionth across the line
/// of sight changes that. A point inside a closed mesh meets it on the way
/// out.
///
/// With `bx`, the box a landscape's slab fills, a point on one of its
/// faces -- a wall, the floor, the box's top -- within a billionth of
/// `size` is on the solid's boundary, seen from outside: it is cast from a
/// millionth outside that face, where nudges cannot carry it in.
pub fn triangle_cast(
    meshes: Vec<Mesh<WorldSpace>>,
    size: f64,
    bx: Option<Box3>,
    model: impl Fn(Vec3) -> Vec3 + Send + Sync + 'static,
) -> Cast {
    Arc::new(move |p, to_eye| {
        let (q, face) = match bx {
            Some(b) => outside(model(p), b, size),
            None => (model(p), None),
        };
        let clear = |x: Vec3| meshes.iter().all(|m| first_hit(m, x, to_eye, 1e-9 * size).is_none());
        let truth = clear(q);
        // On a face, nudged only along it.
        let nudges: Vec<Vec3> = across_sight(to_eye)
            .into_iter()
            .filter_map(|v| match face {
                None => Some(v),
                Some(n) => {
                    let w = v - v.dot(n) * n;
                    if w.length() > 1e-6 { Some(w.normalized()) } else { None }
                }
            })
            .collect();
        if nudges.iter().all(|&v| clear(q + 1e-7 * size * v) == truth) { Some(truth) } else { None }
    })
}

/// A point within a billionth of `size` of a face of `bx`, and inside the
/// box's extent across it, moved a hundred-millionth of `size` out through
/// that face, with the face's outward normal; any other point as it is.
pub fn outside(q: Vec3, bx: Box3, size: f64) -> (Vec3, Option<Vec3>) {
    let tol = 1e-9 * size;
    let (lo, hi) = (bx.0.to_array(), bx.1.to_array());
    let qa = q.to_array();
    if !(0..3).all(|k| qa[k] >= lo[k] - tol && qa[k] <= hi[k] + tol) {
        return (q, None);
    }
    for axis in 0..3 {
        let mut n = [0.0; 3];
        if (qa[axis] - lo[axis]).abs() <= tol {
            n[axis] = -1.0;
        } else if (qa[axis] - hi[axis]).abs() <= tol {
            n[axis] = 1.0;
        }
        let n = Vec3::from_array(n);
        if n != Vec3::ZERO {
            return (q + 1e-8 * size * n, Some(n));
        }
    }
    (q, None)
}

/// The box a landscape's slab fills in its own coordinates: its domain,
/// from its base up to its highest sample.
pub fn slab_box(h: &Heightfield) -> Box3 {
    let d = h.surface.domain();
    let mut top = f64::NEG_INFINITY;
    h.for_each_sample(|q| top = swift::max(top, q.z()));
    let base = if let Walls::Perimeter { base, .. } = &h.walls { *base } else { 0.0 };
    (
        Vec3::new(swift::min(d.real.lo, d.real.hi), swift::min(d.imag.lo, d.imag.hi), base),
        Vec3::new(swift::max(d.real.lo, d.real.hi), swift::max(d.imag.lo, d.imag.hi), top),
    )
}

/// The corners of a landscape's skyline, in its own coordinates: every
/// node of its decimated grid at its drawn height, every point where a
/// triangle's side crosses the truncation, and the floor's corners. A
/// peak or a notch of what it hides is the image of one of these.
pub fn skyline_corners(h: &Heightfield) -> Vec<Vec3> {
    let mut out: Vec<Vec3> = Vec::new();
    h.for_each_sample(|v| out.push(v.v));
    let s = &h.surface;
    let g = &*s.height;
    let step = h.step.max(1);
    let (nx, ny) = (g.width.div_ceil(step), g.height.div_ceil(step));
    let node = |i: usize, j: usize| -> Vec3 {
        let p = g.position(i * step, j * step);
        Vec3::new(p.x(), p.y(), f64::from(g.at(i * step, j * step)))
    };
    for j in 0..ny {
        for i in 0..nx {
            let a = node(i, j);
            let neighbours = [
                (i + 1 < nx).then(|| node(i + 1, j)),
                (j + 1 < ny).then(|| node(i, j + 1)),
                (i + 1 < nx && j > 0).then(|| node(i + 1, j - 1)),
            ];
            for b in neighbours.into_iter().flatten() {
                let ea = a.z - s.truncation.height_at(a.x);
                let eb = b.z - s.truncation.height_at(b.x);
                if !((ea > 0.0) != (eb > 0.0) && ea.is_finite() && eb.is_finite()) {
                    continue;
                }
                let t = ea / (ea - eb);
                let x = a + t * (b - a);
                out.push(Vec3::new(x.x, x.y, s.truncation.height_at(x.x)));
            }
        }
    }
    let (lo, hi) = slab_box(h);
    for x in [lo.x, hi.x] {
        for y in [lo.y, hi.y] {
            out.push(Vec3::new(x, y, lo.z));
        }
    }
    h.tiles
        .iter()
        .flat_map(|tile| {
            out.iter().map(move |&v| {
                let w = tile.apply(P3::from_vec(v));
                Vec3::new(w.x(), w.y(), v.z)
            })
        })
        .collect()
}

/// The clearance of `x` from the round torus of radii `big`, `r` about the z
/// axis: its distance from the core circle less the tube's radius.
fn torus_clearance(big: f64, r: f64, x: Vec3) -> f64 {
    let rho = (x.x * x.x + x.y * x.y).sqrt();
    ((rho - big) * (rho - big) + x.z * x.z).sqrt() - r
}

/// A cast against a round torus of radii `big`, `r` about the z axis: the
/// line walked a thousandth of the tube at a time through the ball that
/// holds the torus, hidden where it is ever inside the tube; None where the
/// least clearance seen is within a millionth of the tube, where the walk's
/// step decides, or the point lies within two of sphere tracing's least
/// steps of it (a hundred-thousandth of the ball), where that step decides.
pub fn torus_cast(big: f64, r: f64, model: impl Fn(Vec3) -> Vec3 + Send + Sync + 'static) -> Cast {
    Arc::new(move |p, to_eye| {
        let q = model(p);
        let ball = big + r;
        let b = q.dot(to_eye);
        let c = q.dot(q) - ball * ball;
        let disc = b * b - c;
        let mut least = torus_clearance(big, r, q);
        if least.abs() < 2e-5 * (big + r) {
            return None;
        }
        if least < 0.0 {
            return Some(false);
        }
        if disc > 0.0 {
            let exit = -b + disc.sqrt();
            let mut t = swift::max(-b - disc.sqrt(), 0.0);
            while t < exit {
                let g = torus_clearance(big, r, q + t * to_eye);
                least = swift::min(least, g);
                if g < -1e-6 * r {
                    return Some(false);
                }
                t += 1e-3 * r;
            }
        }
        if least.abs() > 1e-6 * r { Some(true) } else { None }
    })
}

/// The cast from a point on a round torus of radii `big`, `r` about the z
/// axis, facing the eye: the line walked a thousandth of the tube at a time
/// from a ten-thousandth of the tube along it, through the ball that holds
/// the torus; hidden where it is ever inside the tube by a millionth of it.
/// None where it starts inside -- the point is on a fold, or the line grazes
/// a concave stretch -- or where, once it turns back toward the tube, it
/// comes within a millionth of it, where the walk's step decides.
/// `torus_cast` casts a point off the torus; on it, its clearance is zero
/// and it says nothing.
pub fn torus_own_cast(big: f64, r: f64) -> Cast {
    Arc::new(move |q, to_eye| {
        let ball = big + r;
        let b = q.dot(to_eye);
        let c = q.dot(q) - ball * ball;
        let exit = -b + swift::max(b * b - c, 0.0).sqrt();
        let mut t = 1e-4 * r;
        let mut previous = torus_clearance(big, r, q + t * to_eye);
        if !(previous > 0.0) {
            return None;
        }
        let (mut rising, mut least) = (true, f64::INFINITY);
        while t < exit {
            t += 1e-3 * r;
            let g = torus_clearance(big, r, q + t * to_eye);
            if g < -1e-6 * r {
                return Some(false);
            }
            if rising && g < previous {
                rising = false;
            }
            if !rising {
                least = swift::min(least, g);
            }
            previous = g;
        }
        if least > 1e-6 * r { Some(true) } else { None }
    })
}

/// The brute-force cast against a part as it stands.
pub fn composite_cast(part: &Part) -> Cast {
    let (s, o) = (part.placement.scale, part.placement.offset);
    let model = move |p: Vec3| (p - o) / s;
    match part.scene.geometry() {
        SceneGeometry::Heightfield(h) => {
            let bx = slab_box(h);
            triangle_cast(vec![drawn_triangles(h, bx.0.z)], 4.0, Some(bx), model)
        }
        SceneGeometry::Solid(solid) => {
            let size = solid.vertices.iter().map(|v| v.v.length()).fold(f64::NEG_INFINITY, swift::max);
            triangle_cast(piece_meshes(solid), size, None, model)
        }
        SceneGeometry::Parametric(surface) => {
            let (big, r) = torus_radii(surface);
            torus_cast(big, r, model)
        }
    }
}

/// A round torus's radii, read off its neighbourhood: the ball that holds
/// it has radius R + r, and the centre stands R − r off the tube. A
/// composite takes no other parametric surface.
pub fn torus_radii(s: &ParametricSurface) -> (f64, f64) {
    let e = s.neighbourhood.as_ref().expect("a composite's parametric surface has a tubular neighbourhood");
    let d0 = (e.foot)(e.centre).1;
    ((e.radius + d0) / 2.0, (e.radius - d0) / 2.0)
}

/// Polylines in view space, indexed by the cells of a grid over the page
/// their segments cross: which segment passes through a point, and where
/// the runs end.
pub struct SegmentIndex {
    pub lo: Vec2,
    pub cell: f64,
    pub n: usize,
    pub buckets: Vec<Vec<(Vec3, Vec3)>>,
    pub ends: Vec<Vec<Vec3>>,
}

impl SegmentIndex {
    pub fn new(paths: &PolylineSet<ViewSpace>, lo: Vec2, size: f64, n: usize) -> SegmentIndex {
        let mut index = SegmentIndex {
            lo,
            cell: size / n as f64,
            n,
            buckets: vec![Vec::new(); n * n],
            ends: vec![Vec::new(); n * n],
        };
        for path in paths.paths() {
            for w in path.windows(2) {
                let (a, b) = (w[0], w[1]);
                let (i0, j0) = index.key(Vec2::new(swift::min(a.x(), b.x()), swift::min(a.y(), b.y())));
                let (i1, j1) = index.key(Vec2::new(swift::max(a.x(), b.x()), swift::max(a.y(), b.y())));
                for i in i0..=i1 {
                    for j in j0..=j1 {
                        index.buckets[i * n + j].push((a.v, b.v));
                    }
                }
            }
            for e in [path[0], path[path.len() - 1]] {
                let (i, j) = index.key(Vec2::new(e.x(), e.y()));
                index.ends[i * n + j].push(e.v);
            }
        }
        index
    }

    pub fn key(&self, p: Vec2) -> (usize, usize) {
        let k = (p - self.lo) / self.cell;
        let top = self.n as i64 - 1;
        ((k.x.floor() as i64).clamp(0, top) as usize, (k.y.floor() as i64).clamp(0, top) as usize)
    }

    /// Whether a segment passes within `tolerance` of `p`, in space.
    pub fn holds(&self, p: Vec3, tolerance: f64) -> bool {
        let (i, j) = self.key(Vec2::new(p.x, p.y));
        self.buckets[i * self.n + j].iter().any(|&(a, b)| {
            let ab = b - a;
            let l2 = ab.length_squared();
            let t = if l2 > 0.0 { swift::min(swift::max((p - a).dot(ab) / l2, 0.0), 1.0) } else { 0.0 };
            (p - (a + t * ab)).length() <= tolerance
        })
    }

    /// The nearest run end to `p`, within the cells about it.
    pub fn nearest_end(&self, p: Vec3) -> f64 {
        let (i, j) = self.key(Vec2::new(p.x, p.y));
        let mut best = f64::INFINITY;
        for di in -1i64..=1 {
            for dj in -1i64..=1 {
                let (a, b) = (i as i64 + di, j as i64 + dj);
                if !(a >= 0 && a < self.n as i64 && b >= 0 && b < self.n as i64) {
                    continue;
                }
                for &e in &self.ends[a as usize * self.n + b as usize] {
                    best = swift::min(best, (e - p).length());
                }
            }
        }
        best
    }

    pub fn all_ends(&self) -> Vec<Vec3> {
        self.ends.iter().flatten().copied().collect()
    }

    /// The segments that cross the cell `p` is in.
    pub fn segments_near(&self, p: Vec3) -> &[(Vec3, Vec3)] {
        let (i, j) = self.key(Vec2::new(p.x, p.y));
        &self.buckets[i * self.n + j]
    }
}

/// One upward triangle of a landscape's top: its corners and unit normal.
#[derive(Clone, Copy)]
pub struct TopTriangle {
    pub a: Vec3,
    pub b: Vec3,
    pub c: Vec3,
    pub n: Vec3,
}

/// A landscape's drawn triangles (`solid`), and those that face up -- its
/// top -- bucketed over (x, y): the top's normals at a point on it, and a
/// cast from such a point.
pub struct TerrainTops {
    pub solid: Mesh<WorldSpace>,
    pub triangles: Vec<TopTriangle>,
    pub lo: Vec2,
    pub cell: Vec2,
    pub n: usize,
    pub buckets: Vec<Vec<usize>>,
    /// The landscape's size, the larger side of its box, and how far off a
    /// triangle's plane a point of it may lie: a billionth of its cell, the
    /// composite's tolerance.
    pub size: f64,
    pub tolerance: f64,
}

impl TerrainTops {
    pub fn new(h: &Heightfield, n: usize) -> TerrainTops {
        let bx = slab_box(h);
        let mesh = drawn_triangles(h, bx.0.z);
        let size = swift::max(bx.1.x - bx.0.x, bx.1.y - bx.0.y);
        let tolerance = 1e-9 * landscape_cell(h);
        let mut triangles = Vec::new();
        for t in &mesh.triangles {
            let (a, b, c) =
                (mesh.vertices[t[0] as usize].v, mesh.vertices[t[1] as usize].v, mesh.vertices[t[2] as usize].v);
            let mut normal = (b - a).cross(c - a);
            if !(normal.length() > 0.0) {
                continue;
            }
            normal = normal.normalized();
            if normal.z < 0.0 {
                normal = -normal;
            }
            if !(normal.z > 1e-9) {
                continue;
            }
            triangles.push(TopTriangle { a, b, c, n: normal });
        }
        let mut lo3 = Vec3::splat(f64::INFINITY);
        let mut hi3 = Vec3::splat(f64::NEG_INFINITY);
        for t in &triangles {
            for x in [t.a, t.b, t.c] {
                lo3 = lo3.min(x);
                hi3 = hi3.max(x);
            }
        }
        let lo = Vec2::new(lo3.x, lo3.y);
        let cell = Vec2::new(swift::max(hi3.x - lo3.x, 1e-9) / n as f64, swift::max(hi3.y - lo3.y, 1e-9) / n as f64);
        let mut buckets = vec![Vec::new(); n * n];
        for (k, t) in triangles.iter().enumerate() {
            let tlo = t.a.min(t.b).min(t.c);
            let thi = t.a.max(t.b).max(t.c);
            // Swift's `Int(_:)` truncates toward zero.
            let i0 = (((tlo.x - lo.x) / cell.x) as i64 - 1).max(0) as usize;
            let i1 = (((thi.x - lo.x) / cell.x) as i64 + 1).min(n as i64 - 1) as usize;
            let j0 = (((tlo.y - lo.y) / cell.y) as i64 - 1).max(0) as usize;
            let j1 = (((thi.y - lo.y) / cell.y) as i64 + 1).min(n as i64 - 1) as usize;
            for j in j0..=j1 {
                for i in i0..=i1 {
                    buckets[j * n + i].push(k);
                }
            }
        }
        TerrainTops { solid: mesh, triangles, lo, cell, n, buckets, size, tolerance }
    }

    /// The cast from a point on the top, in the landscape's own coordinates
    /// (`model`): hidden where every triangle of the top it lies on faces
    /// away from the eye, since its line of sight goes into the solid there;
    /// where every one faces the eye, clear when the line meets no triangle
    /// past a billionth of the landscape's size, as `triangle_cast` casts.
    /// None where they disagree or one is edge-on, off the top, and where a
    /// nudge of a ten-millionth of its size along the top and across the line
    /// of sight changes the verdict.
    pub fn cast(self: &Arc<Self>, model: impl Fn(Vec3) -> Vec3 + Send + Sync + 'static) -> Cast {
        let tops = Arc::clone(self);
        Arc::new(move |p, to_eye| {
            let q = model(p);
            let normals = tops.normals(q);
            if normals.is_empty() {
                return None;
            }
            let facing: Vec<f64> = normals.iter().map(|n| n.dot(to_eye)).collect();
            if facing.iter().all(|&f| f < -1e-9) {
                return Some(false);
            }
            if !facing.iter().all(|&f| f > 1e-9) {
                return None;
            }
            let clear = |x: Vec3| first_hit(&tops.solid, x, to_eye, 1e-9 * tops.size).is_none();
            let truth = clear(q);
            let nudges: Vec<Vec3> = normals
                .iter()
                .flat_map(|&n| {
                    let v = to_eye.cross(n);
                    if v.length() > 1e-9 { vec![v.normalized(), -v.normalized()] } else { Vec::new() }
                })
                .collect();
            if nudges.iter().all(|&v| clear(q + 1e-7 * tops.size * v) == truth) { Some(truth) } else { None }
        })
    }

    /// The normals of every triangle of the top `q` lies on, within the
    /// tolerance.
    pub fn normals(&self, q: Vec3) -> Vec<Vec3> {
        // Swift's `Int(_:)` truncates toward zero.
        let i = ((q.x - self.lo.x) / self.cell.x) as i64;
        let j = ((q.y - self.lo.y) / self.cell.y) as i64;
        let n = self.n as i64;
        if !(i >= 0 && i < n && j >= 0 && j < n) {
            return Vec::new();
        }
        let mut out = Vec::new();
        for &k in &self.buckets[(j * n + i) as usize] {
            let t = self.triangles[k];
            // Barycentric in (x, y), and the plane's height there.
            let e1 = Vec2::new(t.b.x - t.a.x, t.b.y - t.a.y);
            let e2 = Vec2::new(t.c.x - t.a.x, t.c.y - t.a.y);
            let r = Vec2::new(q.x - t.a.x, q.y - t.a.y);
            let det = e1.x * e2.y - e1.y * e2.x;
            if det == 0.0 {
                continue;
            }
            let u = (r.x * e2.y - r.y * e2.x) / det;
            let v = (e1.x * r.y - e1.y * r.x) / det;
            if !(u >= -1e-9 && v >= -1e-9 && u + v <= 1.0 + 1e-9) {
                continue;
            }
            let z = t.a.z + u * (t.b.z - t.a.z) + v * (t.c.z - t.a.z);
            if (z - q.z).abs() <= self.tolerance {
                out.push(t.n);
            }
        }
        out
    }
}

/// What the gate found for one composition.
#[derive(Debug, Clone, Default)]
pub struct CompositeVerdict {
    pub compared: usize,
    pub wrong: usize,
    pub hidden_by_other: usize,
    pub undecided: usize,
    pub outside_alone: usize,
    pub own_compared: usize,
    pub own_wrong: usize,
    /// Of the own comparisons, those on the torus its cast says are seen:
    /// what shows the cast gives a verdict on a point facing the eye.
    pub torus_seen: usize,
    /// Of the comparisons, those on a contact line, and of them those
    /// another part hides.
    pub contact_compared: usize,
    pub contact_hidden: usize,
    /// Samples of a contact line on no landscape's top: each a failure.
    pub off_terrain: usize,
    pub example: String,
    pub own_example: String,
}

impl CompositeVerdict {
    /// Another camera's verdict added in, the first examples kept.
    pub fn add(&mut self, v: CompositeVerdict) {
        self.compared += v.compared;
        self.wrong += v.wrong;
        self.undecided += v.undecided;
        self.hidden_by_other += v.hidden_by_other;
        self.outside_alone += v.outside_alone;
        self.own_compared += v.own_compared;
        self.own_wrong += v.own_wrong;
        self.torus_seen += v.torus_seen;
        self.contact_compared += v.contact_compared;
        self.contact_hidden += v.contact_hidden;
        self.off_terrain += v.off_terrain;
        if self.example.is_empty() {
            self.example = v.example;
        }
        if self.own_example.is_empty() {
            self.own_example = v.own_example;
        }
    }
}

/// Whether a layer is a composite's contact line (`Composite::visible_contact`).
pub fn is_contact(spec: &LayerSpec) -> bool {
    spec.name == "contact" && spec.source == LayerSource::File { vertices: String::new(), offsets: String::new() }
}

/// What one sample of the gate found: whether the casts decided, what
/// they said, whether the composite drew it, and whether a contact line's
/// sample was on no landscape's top.
#[derive(Clone, Copy)]
struct Sampled {
    decided: bool,
    truth: bool,
    drawn: bool,
    off_terrain: bool,
}

const UNDECIDED: Sampled = Sampled { decided: false, truth: false, drawn: false, off_terrain: false };

/// The gate for one composite under one camera. Each layer's composite
/// runs are checked to lie on its runs drawn alone, and points sampled
/// densely along those, and on either side of every run end of both, are
/// drawn in the composite exactly when no other part's cast meets their
/// line of sight. Each part drawn alone is held to its own cast: a
/// polyhedron's edges and hatching anywhere, a landscape's graph and the
/// torus's curves at their vertices, where their own tests decide.
pub fn composite_verdict(composite: &Composite, casts: &[Cast]) -> Result<CompositeVerdict, String> {
    let mut verdict = CompositeVerdict::default();
    let drawn = composite.visible_layers().map_err(|e| e.0)?;
    // Each landscape's skyline corners and cell, in view space: where a
    // peak or a notch of what it hides can be.
    let marks: Vec<(Vec<Vec3>, f64)> = composite
        .parts
        .iter()
        .map(|part| {
            let SceneGeometry::Heightfield(h) = part.scene.geometry() else { return (Vec::new(), 0.0) };
            let view = part.scene.camera.view;
            let corners = skyline_corners(h).into_iter().map(|c| view.apply(P3::from_vec(c)).v).collect();
            (corners, landscape_cell(h))
        })
        .collect();
    let view = composite.camera.view;
    let world = view.inverse();
    let to_eye = -view.projection_direction();
    let terrain_tops: Vec<Option<Arc<TerrainTops>>> = composite
        .parts
        .iter()
        .map(|part| match part.scene.geometry() {
            SceneGeometry::Heightfield(h) => Some(Arc::new(TerrainTops::new(h, 128))),
            _ => None,
        })
        .collect();
    let mut index = 0;
    for (i, part) in composite.parts.iter().enumerate() {
        let contact = composite.contact(i);
        let mut alone = part.scene.visible_layers(None, None).map_err(|e| e.to_string())?;
        alone.extend(composite.visible_contact(i).map_err(|e| e.to_string())?);
        // A point of a contact line lies on one landscape's top: cast
        // against that one from where it lies (`TerrainTops::cast`), and
        // against the others as any point is.
        let other_parts: Vec<usize> = (0..casts.len()).filter(|&j| j != i).collect();
        let others: Vec<&Cast> = other_parts.iter().map(|&j| &casts[j]).collect();
        let top_casts: Vec<Option<Cast>> = other_parts
            .iter()
            .map(|&j| {
                let o = composite.parts[j].placement.offset;
                terrain_tops[j].as_ref().map(|tops| tops.cast(move |p| p - o))
            })
            .collect();
        let owner = |w: Vec3| -> Option<usize> {
            (0..other_parts.len()).find(|&k| {
                let j = other_parts[k];
                terrain_tops[j]
                    .as_ref()
                    .is_some_and(|tops| !tops.normals(w - composite.parts[j].placement.offset).is_empty())
            })
        };
        let part_view = part.scene.camera.view;
        for (layer, seen) in &alone {
            let at = index;
            index += 1;
            if !(at < drawn.len() && drawn[at].0.spec == layer.spec) {
                verdict.wrong += 1;
                verdict.example = format!("layers out of order at {}", layer.spec.name);
                continue;
            }
            let composed = &drawn[at].1;
            if !layer.spec.clipped {
                if composed != seen {
                    verdict.wrong += 1;
                    verdict.example = format!("{}, unclipped, was cut", layer.spec.name);
                }
                continue;
            }
            let Some(bounds) = AABB::of(seen.vertices.iter().chain(&composed.vertices)) else { continue };
            let size = swift::max(swift::max(bounds.size().x, bounds.size().y), 1e-9);
            let lo = Vec2::new(bounds.lo.x, bounds.lo.y) - Vec2::splat(1e-6 * size);
            let alone_index = SegmentIndex::new(seen, lo, 1.00001 * size, 400);
            let composed_index = SegmentIndex::new(composed, lo, 1.00001 * size, 400);
            let tolerance = 1e-9 * size;
            for v in &composed.vertices {
                if !alone_index.holds(v.v, tolerance) {
                    verdict.outside_alone += 1;
                }
            }
            // The samples: every segment of the runs drawn alone, densely,
            // and a hair either side of every run end.
            let h = size / 2000.0;
            let mut samples: Vec<(Vec3, f64)> = Vec::new();
            let mut ends = alone_index.all_ends();
            ends.extend(composed_index.all_ends());
            for path in seen.paths() {
                for w in path.windows(2) {
                    let (a, b) = (w[0].v, w[1].v);
                    let ab = b - a;
                    let length = ab.length();
                    if !(length > 0.0) {
                        continue;
                    }
                    let m = ((length / h).ceil() as i64).max(2);
                    for j in 0..=m {
                        samples.push((a + ab * (j as f64 / m as f64), length));
                    }
                }
            }
            for &e in &ends {
                // The alone segments the end lies on.
                for &(a, b) in alone_index.segments_near(e) {
                    let ab = b - a;
                    let l2 = ab.length_squared();
                    if !(l2 > 0.0) {
                        continue;
                    }
                    let t = (e - a).dot(ab) / l2;
                    if !((-1e-9..=1.0 + 1e-9).contains(&t) && (e - (a + t * ab)).length() <= tolerance) {
                        continue;
                    }
                    for d in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1] {
                        for u in [t - d, t + d] {
                            if u > 0.0 && u < 1.0 {
                                samples.push((a + ab * u, l2.sqrt()));
                            }
                        }
                    }
                }
            }
            // A hair either side of where each skyline corner of another
            // geometry crosses a segment on the page: a peak's tip hides a
            // span, and a notch shows a slit, as short as you like there,
            // and a missed one leaves no run end to sample beside.
            for (j, (corners, cell)) in marks.iter().enumerate() {
                if j == i || corners.is_empty() {
                    continue;
                }
                let cell = *cell;
                let points: Vec<P3<ViewSpace>> = corners.iter().flat_map(|&c| [P3::from_vec(c); 2]).collect();
                let offsets: Vec<usize> = (0..=corners.len()).map(|k| 2 * k).collect();
                let near = SegmentIndex::new(&PolylineSet::new(points, offsets, None), lo, 1.00001 * size, 200);
                for path in seen.paths() {
                    for w in path.windows(2) {
                        let (a, b) = (w[0], w[1]);
                        let e = Vec2::new(b.x() - a.x(), b.y() - a.y());
                        let l2 = e.length_squared();
                        if !(l2 > 0.0) {
                            continue;
                        }
                        let (i0, j0) =
                            near.key(Vec2::new(swift::min(a.x(), b.x()), swift::min(a.y(), b.y())) - Vec2::splat(cell));
                        let (i1, j1) =
                            near.key(Vec2::new(swift::max(a.x(), b.x()), swift::max(a.y(), b.y())) + Vec2::splat(cell));
                        let mut close: Vec<Vec3> = Vec::new();
                        for u in i0..=i1 {
                            for w in j0..=j1 {
                                close.extend(near.buckets[u * near.n + w].iter().map(|s| s.0));
                            }
                        }
                        let length = l2.sqrt();
                        for c in close {
                            let r = Vec2::new(c.x - a.x(), c.y - a.y());
                            let t = r.dot(e) / l2;
                            // Within a cell of the segment on the page, and
                            // in front of it: a corner behind hides nothing.
                            if !(t > -cell / length
                                && t < 1.0 + cell / length
                                && (r.x * e.y - r.y * e.x).abs() / length <= cell
                                && c.z > a.z() + (b.z() - a.z()) * swift::min(swift::max(t, 0.0), 1.0))
                            {
                                continue;
                            }
                            let fine = (-16..=16).map(|k| k as f64 * cell / 16.0 / length);
                            for dt in fine.chain([1e-9, -1e-9, 1e-7, -1e-7, 1e-5, -1e-5]) {
                                if t + dt > 0.0 && t + dt < 1.0 {
                                    samples.push((a.v + (b.v - a.v) * (t + dt), length));
                                }
                            }
                        }
                    }
                }
            }
            let on_terrain = is_contact(&layer.spec);
            let results: Vec<Sampled> = par_map(&samples, |&(p, length)| {
                // At a run end the truth is a boundary.
                if swift::min(alone_index.nearest_end(p), composed_index.nearest_end(p))
                    < swift::max(1e-6 * length, 1e-12 * size)
                {
                    return UNDECIDED;
                }
                let w = world.apply(P3::from_vec(p)).v;
                let mut on: Option<usize> = None;
                if on_terrain {
                    match owner(w) {
                        Some(k) => on = Some(k),
                        None => return Sampled { off_terrain: true, ..UNDECIDED },
                    }
                }
                let mut truth = true;
                for (k, &c) in others.iter().enumerate() {
                    let c = if on == Some(k) { top_casts[k].as_ref().expect("the owner is a landscape") } else { c };
                    match c(w, to_eye) {
                        None => return UNDECIDED,
                        Some(clear) => {
                            if !clear {
                                truth = false;
                            }
                        }
                    }
                }
                Sampled { decided: true, truth, drawn: composed_index.holds(p, tolerance), off_terrain: false }
            });
            for (&(p, _), r) in samples.iter().zip(&results) {
                if r.off_terrain {
                    verdict.off_terrain += 1;
                    if verdict.example.is_empty() {
                        let w = world.apply(P3::from_vec(p));
                        verdict.example = format!(
                            " e.g. {} of {} at ({:.4}, {:.4}, {:.4}) is on no landscape's top",
                            layer.spec.name,
                            part.name,
                            w.x(),
                            w.y(),
                            w.z()
                        );
                    }
                }
                if !r.decided {
                    verdict.undecided += 1;
                    continue;
                }
                verdict.compared += 1;
                if !r.truth {
                    verdict.hidden_by_other += 1;
                }
                if on_terrain {
                    verdict.contact_compared += 1;
                    if !r.truth {
                        verdict.contact_hidden += 1;
                    }
                }
                if r.truth != r.drawn {
                    verdict.wrong += 1;
                    if verdict.example.is_empty() {
                        let w = world.apply(P3::from_vec(p));
                        verdict.example = format!(
                            " e.g. {} of {} at ({:.4}, {:.4}, {:.4}): the casts say {}",
                            layer.spec.name,
                            part.name,
                            w.x(),
                            w.y(),
                            w.z(),
                            if r.truth { "seen" } else { "hidden" }
                        );
                    }
                }
            }
            // The part alone, held to its own cast.
            let alone_holds = |q: Vec3| -> Option<bool> {
                let p = part_view.apply(P3::from_vec(q)).v;
                if alone_index.nearest_end(p) < 1e-6 * size {
                    return None;
                }
                Some(alone_index.holds(p, tolerance))
            };
            let parametric = matches!(part.scene.geometry(), SceneGeometry::Parametric(_));
            for (q, truth) in own_truths(part, layer, &part_view, &contact) {
                let Some(truth) = truth else { continue };
                let Some(shown) = alone_holds(q) else { continue };
                verdict.own_compared += 1;
                if truth && parametric {
                    verdict.torus_seen += 1;
                }
                if truth != shown {
                    verdict.own_wrong += 1;
                    if verdict.own_example.is_empty() {
                        verdict.own_example = format!(
                            " e.g. {} of {} at ({:.4}, {:.4}, {:.4}): its cast says {}",
                            layer.spec.name,
                            part.name,
                            q.x,
                            q.y,
                            q.z,
                            if truth { "seen" } else { "hidden" }
                        );
                    }
                }
            }
        }
    }
    if index != drawn.len() {
        verdict.wrong += 1;
        verdict.example = format!("{} layers drawn, {index} alone", drawn.len());
    }
    Ok(verdict)
}

/// Points of a part's own curves, in its own coordinates, with what its
/// own brute-force cast says of each (None: undecided): a polyhedron's edges
/// and hatch lines sampled anywhere along them, as the solids suite samples
/// edges; a landscape's graph curves and the torus's curves at their
/// vertices, as the stress and surface suites test them. Its contact line
/// likewise: on a polyhedron sampled along each path, cast from its face,
/// on the torus at its vertices.
pub fn own_truths(
    part: &Part,
    layer: &Layer,
    view: &Transform<WorldSpace, ViewSpace>,
    contact: &ContactCurves,
) -> Vec<(Vec3, Option<bool>)> {
    let to_eye = -view.projection_direction();
    if is_contact(&layer.spec) {
        return match part.scene.geometry() {
            SceneGeometry::Solid(s) => {
                let meshes = piece_meshes(s);
                let mut points: Vec<(Vec3, Vec<usize>)> = Vec::new();
                for (path, &f) in contact.paths.iter().zip(&contact.faces) {
                    if !(s.normals[s.faces[f][0]].dot(to_eye).abs() > 1e-9) {
                        continue;
                    }
                    for w in path.windows(2) {
                        let (a, b) = (w[0].v, w[1].v);
                        for k in 0..8 {
                            points.push((a + (k as f64 + 0.5) / 8.0 * (b - a), s.faces[f].clone()));
                        }
                    }
                }
                par_map(&points, |(q, own)| (*q, Some(ray_visible(s, &meshes, *q, own, to_eye))))
            }
            SceneGeometry::Parametric(s) => {
                let (big, r) = torus_radii(s);
                let cast = torus_own_cast(big, r);
                let coords: Vec<P2<_>> = contact.coords.iter().flatten().copied().collect();
                par_map(&coords, |&c| {
                    let q = (s.map)(c).position;
                    match s.facing(c, -to_eye) {
                        Some(f) if f.abs() > 1e-9 => (q, if f > 0.0 { cast(q, to_eye) } else { Some(false) }),
                        _ => (q, None),
                    }
                })
            }
            SceneGeometry::Heightfield(_) => Vec::new(),
        };
    }
    match part.scene.geometry() {
        SceneGeometry::Solid(s) => {
            let mut lines: Vec<(Vec<Vec3>, Vec<usize>)> = Vec::new();
            match &layer.spec.source {
                LayerSource::Edges => {
                    for (k, e) in s.edges.iter().enumerate() {
                        if s.is_diagonal[k] {
                            continue;
                        }
                        if !(s.normals[e.left].dot(to_eye).abs() > 1e-9 && s.normals[e.right].dot(to_eye).abs() > 1e-9)
                        {
                            continue;
                        }
                        lines.push((vec![s.vertices[e.a].v, s.vertices[e.b].v], vec![e.left, e.right]));
                    }
                }
                LayerSource::Hatching { hatching, faces: FaceSelector::Solid } => {
                    // The width step a layer of one width carries in its
                    // name, "name k/n".
                    let width: Option<i64> = layer
                        .spec
                        .name
                        .split(' ')
                        .next_back()
                        .and_then(|w| if w.contains('/') { w.split('/').next()?.parse().ok() } else { None });
                    for line in s.hatch_lines(hatching) {
                        if !(width.is_none() || Some(line.width_step) == width) {
                            continue;
                        }
                        let pieces = s.faces[line.face].clone();
                        if !(s.normals[s.faces[line.face][0]].dot(to_eye).abs() > 1e-9) {
                            continue;
                        }
                        lines.push((line.path.iter().map(|p| p.v).collect(), pieces));
                    }
                }
                _ => {}
            }
            let meshes = piece_meshes(s);
            let mut points: Vec<(Vec3, &[usize])> = Vec::new();
            for (path, own) in &lines {
                for w in path.windows(2) {
                    let (a, b) = (w[0], w[1]);
                    for k in 0..40 {
                        points.push((a + (k as f64 + 0.5) / 40.0 * (b - a), own.as_slice()));
                    }
                }
            }
            par_map(&points, |&(q, own)| (q, Some(ray_visible(s, &meshes, q, own, to_eye))))
        }
        SceneGeometry::Heightfield(h) => {
            // On f the stress suite holds the traversal to its dense cast.
            if !(h.refine.is_none() && part.scene.geometry().face_of(&layer.spec) == Some(FaceKind::Graph)) {
                return Vec::new();
            }
            let mesh = drawn_triangles(h, 0.0);
            let vis = HeightfieldVisibility::new(h.clone(), view, part.scene.depth_tolerance);
            let per_unit = crate::common::per_unit(view, to_eye);
            let d = h.surface.domain();
            let g = &*h.surface.height;
            let cell =
                swift::max(d.real.length().abs() / (g.width - 1) as f64, d.imag.length().abs() / (g.height - 1) as f64);
            let (blo, bhi) = (
                Vec2::new(swift::min(d.real.lo, d.real.hi), swift::min(d.imag.lo, d.imag.hi)),
                Vec2::new(swift::max(d.real.lo, d.real.hi), swift::max(d.imag.lo, d.imag.hi)),
            );
            par_map(&layer.paths.vertices, |&v| {
                let f = vis.facing(v.xy());
                if !(f.abs() > 1e-9) {
                    return (v.v, None);
                }
                let strictly_inside =
                    v.x() > blo.x + 1e-9 && v.x() < bhi.x - 1e-9 && v.y() > blo.y + 1e-9 && v.y() < bhi.y - 1e-9;
                (
                    v.v,
                    Some(
                        f > 0.0
                            && !exactly_hidden(
                                &mesh,
                                v.v,
                                to_eye,
                                vis.depth_tolerance,
                                per_unit,
                                strictly_inside,
                                cell,
                            ),
                    ),
                )
            })
        }
        SceneGeometry::Parametric(s) => {
            if layer.spec.role == LayerRole::Outline {
                return Vec::new();
            }
            let Some(coords) = &layer.paths.coords else { return Vec::new() };
            let (big, r) = torus_radii(s);
            let cast = torus_own_cast(big, r);
            let indices: Vec<usize> = (0..coords.len()).collect();
            par_map(&indices, |&k| {
                let q = layer.paths.vertices[k].v;
                match s.facing(coords[k], -to_eye) {
                    Some(f) if f.abs() > 1e-9 => {
                        (q, if f > 0.0 { cast((s.map)(coords[k]).position, to_eye) } else { Some(false) })
                    }
                    _ => (q, None),
                }
            })
        }
    }
}

/// A contract polyhedron's scene under its first preset.
pub fn polyhedron(name: &str) -> Scene {
    let b = PolyhedronBundle::read(fixtures::shared(&format!("contract/polyhedron_{name}.kurven")))
        .unwrap_or_else(|e| panic!("polyhedron_{name}: {e:?}"));
    b.scene(&b.manifest.presets[0])
}

/// A contract surface's scene under its first preset
/// (`SurfaceBundle.read(...).scene(preset:)`): Phase 4's, refused.
pub fn surface(name: &str) -> Result<Scene, PortRefusal> {
    let _ = name;
    Err(PortRefusal { phase: 4, what: "SurfaceBundle.read (contract/surface_*.kurven)" })
}

/// The cubic landscape on its grid, `expr/landscape_cubic.kurven`, read as
/// it is (no refiner), under its first preset.
pub fn cubic_bundle() -> KurvenBundle {
    KurvenBundle::read(fixtures::shared("expr/landscape_cubic.kurven")).expect("landscape_cubic reads")
}

pub fn cubic_landscape() -> Scene {
    let bundle = cubic_bundle();
    Scene::from_bundle(&bundle, &bundle.manifest.presets[0])
}

/// The gate's spike, exp(-20 |z|), on a grid of 64 cells: an apex no step
/// along a segment can be trusted to find.
pub fn spike() -> Scene {
    let mut request =
        LandscapeRequest::new("exp(-20*abs(z))", Domain::new(Interval::new(-2.0, 2.0), Interval::new(-2.0, 2.0)), 65);
    request.truncation = Some(Truncation::Uniform(5.0));
    let bundle = build(&request, false);
    Scene::from_bundle(&bundle, &bundle.manifest.presets[0])
}

/// An orthographic camera through `test_camera`.
pub fn orthographic(elevation: f64, azimuth: f64, shear: f64) -> Camera {
    Camera { view: test_camera(elevation, azimuth, shear), projection: Projection::Orthographic { oblique: None } }
}

pub type Parts = Vec<(String, Scene, Placement)>;

/// The gate's compositions that stand the round torus in a composite
/// (Phase 4), by name.
pub const WITH_TORUS: [&str; 3] =
    ["the round torus ringing a tower", "the round torus sunk into the foot of a tower", "all three, moved"];

/// The composite gate's compositions on the cubic landscape's grid and the
/// spike's, and its three cameras. A composition that stands the round
/// torus in it comes back as the torus's refusal (Phase 4).
pub fn gate_compositions() -> (Vec<(String, Camera)>, Vec<(String, Result<Parts, PortRefusal>)>) {
    // The cubic landscape's grid: its solid is the drawn triangles, which
    // the cast is against. Three towers at the cube roots of unity.
    let landscape = cubic_landscape();
    let (cube, frame) = (polyhedron("cube"), polyhedron("frame"));
    let torus = surface("torus");
    let cube_min_y = cube.solid().unwrap().vertices.iter().map(|v| v.y()).fold(f64::INFINITY, swift::min);
    let spike = spike();
    let cameras = vec![
        ("the landscape's".to_string(), landscape.camera),
        ("el 30° az 37°".to_string(), orthographic(-30.0, 37.0, 0.0)),
        ("el 55° sheared".to_string(), orthographic(-55.0, 160.0, 0.5)),
    ];
    let id = Placement::IDENTITY;
    let at = |scale: f64, x: f64, y: f64, z: f64| Placement::new(scale, Vec3::new(x, y, z));
    let part = |name: &str, scene: &Scene, placement: Placement| (name.to_string(), scene.clone(), placement);
    let with_torus = |rest: Vec<(String, Scene, Placement)>, at: usize, placement: Placement| {
        torus.clone().map(|t| {
            let mut parts = rest;
            parts.insert(at, ("torus".to_string(), t, placement));
            parts
        })
    };
    let compositions: Vec<(&str, Result<Parts, PortRefusal>)> = vec![
        (
            "a cube standing on the landscape behind a tower",
            Ok(vec![part("landscape", &landscape, id), part("cube", &cube, at(0.35, -1.2, 0.1, 0.3))]),
        ),
        (
            "the round torus ringing a tower",
            with_torus(vec![part("landscape", &landscape, id)], 1, at(0.35, 1.0, 0.0, 1.6)),
        ),
        (
            "the round torus sunk into the foot of a tower",
            with_torus(vec![part("landscape", &landscape, id)], 1, at(0.3, 0.4, 1.3, 0.3)),
        ),
        (
            "a frame sunk into the slope of a tower",
            Ok(vec![part("landscape", &landscape, id), part("frame", &frame, at(0.3, 0.3, -1.3, 0.2))]),
        ),
        // Moved together, the landscape too.
        (
            "all three, moved",
            with_torus(
                vec![
                    part("landscape", &landscape, at(1.0, 0.4, 0.2, -0.1)),
                    part("cube", &cube, at(0.35, -0.8, 0.3, 0.2)),
                ],
                2,
                at(0.35, 1.4, 0.2, 1.5),
            ),
        ),
        // Under the slab's floor at its front, its front face a hair
        // inside the front wall: lines of sight from the face's lower
        // part leave the footprint before they reach the floor, and
        // from its upper part come up through it.
        (
            "a cube under the landscape's base",
            Ok(vec![part("landscape", &landscape, id), part("cube", &cube, at(0.35, 0.0, -1.56, -0.86))]),
        ),
        // Standing on the base, sunk into the slab, its front face in
        // the plane of the front wall, to rounding: on the slab's
        // boundary, seen from outside.
        (
            "a cube flush against the landscape's front wall",
            Ok(vec![
                part("landscape", &landscape, id),
                part("cube", &cube, at(0.35, 0.0, -2.0 - 0.35 * cube_min_y, 0.0)),
            ]),
        ),
        (
            "a cube seen through the notch between two towers",
            Ok(vec![part("landscape", &landscape, id), part("cube", &cube, at(0.4, -0.2, 1.8, 0.5))]),
        ),
        (
            "a cube over the towers, where a tower's corner hid a hatch line's span",
            Ok(vec![part("landscape", &landscape, id), part("cube", &cube, at(0.35, -1.0, 1.0, 1.0))]),
        ),
        (
            "a cube behind the spike's apex",
            Ok(vec![part("landscape", &spike, id), part("cube", &cube, at(1.0, -0.3712, 1.162, -0.249))]),
        ),
        // Two landscapes side by side, the cube standing across the gap:
        // each contact point lies on one of them, and the other hides it as
        // it hides any point.
        (
            "two landscapes side by side, a cube across the gap",
            Ok(vec![
                part("landscape", &landscape, id),
                part("spike", &spike, at(1.0, 4.1, 0.0, 0.0)),
                part("cube", &cube, at(0.35, 2.05, 0.1, -0.05)),
            ]),
        ),
    ];
    (cameras, compositions.into_iter().map(|(n, p)| (n.to_string(), p)).collect())
}
