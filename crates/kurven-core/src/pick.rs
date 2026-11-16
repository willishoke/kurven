//! Picking a face, and the boundary of a face's copy: the port of
//! `Sources/KurvenCore/Pick.swift`.
//!
//! Ported in Phase 3 because a face's boundary (`Surface::top_boundary`) cuts
//! and chains its pieces with `Face::pieces`, `snapped` and `chained`, and a
//! parametric surface's face reads its foot and seams from here.

use std::collections::{HashMap, HashSet};
use std::fmt;

use kurven_math::{real, swift};

use crate::face::{Face, FaceCopy, FaceID};
use crate::parametric::{ParamAxis, ParametricSurface};
use crate::scene::SceneGeometry;
use crate::simd::{Vec2, Vec3};
use crate::spaces::{DomainSpace, P2, P3, ParamSpace, WorldSpace};

/// The face a point of space lies on, as `Face::pick` finds it: which face,
/// which copy of it, the point of its parameter plane the point stands at
/// (`Face::foot`), and how far the point is from it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FacePick {
    pub face: FaceID,
    /// The copy's place in `Face::copies`: on a tiled landscape, the tile.
    pub copy: usize,
    pub parameter: P2<ParamSpace>,
    pub distance: f64,
}

impl fmt::Display for FacePick {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.copy == 0 { write!(f, "{}", self.face) } else { write!(f, "{} on tile {}", self.face, self.copy) }
    }
}

impl fmt::Display for FaceID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FaceID::Graph => write!(f, "graph"),
            FaceID::Plateau { band } => write!(f, "plateau {band}"),
            FaceID::Riser { band } => write!(f, "riser {band}"),
            FaceID::TileBoundaryRiser { tile, facing } => {
                write!(f, "riser at the tile boundary, tile {tile} over {facing}")
            }
            FaceID::Wall { edge } => write!(f, "wall {edge}"),
            FaceID::Surface => write!(f, "surface"),
            FaceID::Solid { face } => write!(f, "face {face}"),
        }
    }
}

impl Face {
    /// The face a point of space lies on, of `faces`, and the copy: what a
    /// click picks.
    ///
    /// Each copy of each face is asked where the point stands on it (`foot`,
    /// through the copy's tile carried back), and how far the point is from
    /// the face there, along the face's normal: exactly for a flat face and
    /// a parametric surface, whose feet are the nearest points, and to first
    /// order on the graph, whose foot is the point of the domain under it. A
    /// copy is a candidate where the distance is at most `tolerance` and its
    /// trimmed domain holds the foot, or a point `tie` from it
    /// (`FaceCopy::holds`), so a point on a crease is in both faces that
    /// meet there.
    ///
    /// The nearest candidate is the face, and the tie rule settles a point
    /// as near two: candidates within `tie` of the nearest are equally near,
    /// and the first of them in the order the faces come (`FaceID`) wins,
    /// then the copy first in the tiles. `None` where no face is within
    /// `tolerance`.
    pub fn pick(p: P3<WorldSpace>, faces: &[Face], tolerance: f64, tie: f64) -> Option<FacePick> {
        let mut picks: Vec<FacePick> = Vec::new();
        for face in faces {
            for (k, copy) in face.copies.iter().enumerate() {
                let Some(back) = copy.tile.inverse() else { continue };
                let q = (face.foot)(back.apply(p).v);
                let jet = copy.place((face.map)(q));
                let n = jet.normal();
                let length = n.length();
                let offset = p.v - jet.position;
                let distance = if length > 0.0 { offset.dot(n).abs() / length } else { offset.length() };
                if !(distance <= tolerance && copy.holds(q, tie)) {
                    continue;
                }
                picks.push(FacePick { face: face.id, copy: k, parameter: q, distance });
            }
        }
        // `map(\.distance).min()`: the first of the least, by `<`.
        let mut nearest: Option<f64> = None;
        for p in &picks {
            if nearest.is_none_or(|n| p.distance < n) {
                nearest = Some(p.distance);
            }
        }
        let nearest = nearest?;
        let mut best: Option<FacePick> = None;
        for p in picks.into_iter().filter(|p| p.distance <= nearest + tie) {
            if best.is_none_or(|b| (p.face, p.copy) < (b.face, b.copy)) {
                best = Some(p);
            }
        }
        best
    }

    /// The boundary of one copy of the face, as curves of its parameter
    /// plane: where its trimmed domain ends. `map`, then the copy's `place`,
    /// carry them into space.
    ///
    /// A polyhedron face's are its loops. The graph's and a plateau's are
    /// read even-odd off the graph's trim loops, the drawn truncation edge
    /// among them, cut to the footprint where the copy stands
    /// (`Surface::top_boundary`). A wall's runs along its bottom, up its ends
    /// and along its top edge as its layer draws it, and a riser's round each
    /// stretch where it stands, from its bottom up to its top
    /// (`Surface::column_loops`). A parametric surface's are its parameter
    /// rectangle's edges, and on a periodic axis the seam, where the
    /// rectangle is glued: it bounds nothing, but it is where the face's
    /// chart ends (`ParametricSurface::seams`).
    pub fn boundary(&self, copy: &FaceCopy) -> Vec<Vec<P2<ParamSpace>>> {
        (self.bounding)(copy.tile).into_iter().filter(|c| c.len() >= 2).collect()
    }

    /// The segments of `curves`, each cut where another crosses it or ends
    /// on it, or passes a vertex within `eps` of it: the vertex itself is
    /// the cut, so the pieces of two curves that run along each other share
    /// their ends.
    pub fn pieces(curves: &[Vec<Vec2>], eps: f64) -> Vec<(Vec2, Vec2)> {
        let mut segments: Vec<(Vec2, Vec2)> = Vec::new();
        for curve in curves {
            for w in curve.windows(2) {
                let (a, b) = (w[0], w[1]);
                if a != b && a.x.is_finite() && a.y.is_finite() && b.x.is_finite() && b.y.is_finite() {
                    segments.push((a, b));
                }
            }
        }
        if segments.is_empty() {
            return Vec::new();
        }
        // A grid of buckets over the segments' box, about one segment a
        // bucket, each segment filed under every bucket its box meets.
        let mut lo = Vec2::splat(f64::INFINITY);
        let mut hi = Vec2::splat(f64::NEG_INFINITY);
        for &(a, b) in &segments {
            lo = lo.min(a.min(b));
            hi = hi.max(a.max(b));
        }
        let side = ((segments.len() as f64).sqrt() as usize).max(1);
        let size = ((hi - lo) / side as f64).max(Vec2::splat(swift::max(eps, 1e-300)));
        let cell = |p: Vec2| -> (usize, usize) {
            let c = Vec2::new((p.x - lo.x) / size.x, (p.y - lo.y) / size.y);
            let clamp = |x: f64| -> usize { swift::int(x.floor()).map_or(0, |i| i.clamp(0, side as i64 - 1) as usize) };
            (clamp(c.x), clamp(c.y))
        };
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); side * side];
        for (i, &(a, b)) in segments.iter().enumerate() {
            let c0 = cell(a.min(b) - Vec2::splat(eps));
            let c1 = cell(a.max(b) + Vec2::splat(eps));
            for y in c0.1..=c1.1 {
                for x in c0.0..=c1.0 {
                    buckets[y * side + x].push(i);
                }
            }
        }
        let cross = |a: Vec2, b: Vec2| a.x * b.y - a.y * b.x;
        let mut stamp: Vec<Option<usize>> = vec![None; segments.len()];
        let mut out: Vec<(Vec2, Vec2)> = Vec::new();
        for (i, &(a, b)) in segments.iter().enumerate() {
            let d = b - a;
            let length2 = d.length_squared();
            let length = length2.sqrt();
            let mut cuts: Vec<(f64, Vec2)> = Vec::new();
            // A point within `eps` of the segment, strictly between its ends.
            let on_segment = |cuts: &mut Vec<(f64, Vec2)>, p: Vec2| {
                let t = (p - a).dot(d) / length2;
                if !(t * length > eps && (1.0 - t) * length > eps && cross(d, p - a).abs() / length <= eps) {
                    return;
                }
                cuts.push((t, p));
            };
            let c0 = cell(a.min(b) - Vec2::splat(eps));
            let c1 = cell(a.max(b) + Vec2::splat(eps));
            for y in c0.1..=c1.1 {
                for x in c0.0..=c1.0 {
                    for &j in &buckets[y * side + x] {
                        if j == i || stamp[j] == Some(i) {
                            continue;
                        }
                        stamp[j] = Some(i);
                        let (c, e) = segments[j];
                        on_segment(&mut cuts, c);
                        on_segment(&mut cuts, e);
                        let f = e - c;
                        let det = cross(d, f);
                        if det == 0.0 {
                            continue;
                        }
                        let r = c - a;
                        let t = cross(r, f) / det;
                        let u = cross(r, d) / det;
                        if !(t > 0.0 && t < 1.0 && u > 0.0 && u < 1.0) {
                            continue;
                        }
                        cuts.push((t, a + t * d));
                    }
                }
            }
            // Swift's sort by `<` on t; no t is a NaN here.
            cuts.sort_by(|l, r| l.0.partial_cmp(&r.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut start = a;
            let mut last = 0.0;
            for (t, p) in cuts {
                if (t - last) * length > eps && (1.0 - t) * length > eps {
                    out.push((start, p));
                    start = p;
                    last = t;
                }
            }
            out.push((start, b));
        }
        out
    }

    /// Segments with their ends taken onto one point wherever two lie
    /// within `eps`: two curves cut at one crossing land a rounding apart.
    /// A segment left with no length, and the second of two with the same
    /// ends either way round, are dropped.
    pub fn snapped(segments: &[(Vec2, Vec2)], eps: f64) -> Vec<(Vec2, Vec2)> {
        let mut cells: HashMap<(i64, i64), Vec<Vec2>> = HashMap::new();
        // Swift's `Int64(x)` traps past its range; the port files such a
        // point in a cell of its own at the end of the range.
        let index = |x: f64| -> i64 { swift::int((x / eps).floor()).unwrap_or(i64::MAX) };
        let mut snap = |p: Vec2| -> Vec2 {
            let c = (index(p.x), index(p.y));
            for dx in -1i64..=1 {
                for dy in -1i64..=1 {
                    if let Some(q) = cells
                        .get(&(c.0.wrapping_add(dx), c.1.wrapping_add(dy)))
                        .and_then(|v| v.iter().find(|q| (**q - p).length() <= eps))
                    {
                        return *q;
                    }
                }
            }
            cells.entry(c).or_default().push(p);
            p
        };
        let mut seen: HashSet<[u64; 4]> = HashSet::new();
        let mut out = Vec::new();
        for &(a, b) in segments {
            let (p, q) = (snap(a), snap(b));
            if p == q {
                continue;
            }
            let key = if p.x < q.x || (p.x == q.x && p.y < q.y) { [p.x, p.y, q.x, q.y] } else { [q.x, q.y, p.x, p.y] };
            if seen.insert(key.map(bits)) {
                out.push((p, q));
            }
        }
        out
    }

    /// Segments joined end to end into polylines, at ends they share
    /// exactly and no other segment does.
    pub fn chained(segments: &[(Vec2, Vec2)]) -> Vec<Vec<Vec2>> {
        let key = |p: Vec2| (bits(p.x), bits(p.y));
        let mut at: HashMap<(u64, u64), Vec<usize>> = HashMap::new();
        for (i, &(a, b)) in segments.iter().enumerate() {
            at.entry(key(a)).or_default().push(i);
            at.entry(key(b)).or_default().push(i);
        }
        let mut used = vec![false; segments.len()];
        let mut out: Vec<Vec<Vec2>> = Vec::new();
        // The next unused segment from `p`, where `p` joins exactly two.
        let next = |used: &[bool], p: Vec2| -> Option<usize> {
            let ends = at.get(&key(p))?;
            if ends.len() != 2 {
                return None;
            }
            ends.iter().copied().find(|&j| !used[j])
        };
        let extend = |used: &mut Vec<bool>, path: &mut Vec<Vec2>| {
            while let Some(j) = next(used, path[path.len() - 1]) {
                used[j] = true;
                let (a, b) = segments[j];
                let last = path[path.len() - 1];
                path.push(if a == last { b } else { a });
            }
        };
        for (i, &(a, b)) in segments.iter().enumerate() {
            if used[i] {
                continue;
            }
            used[i] = true;
            let mut forward = vec![a, b];
            extend(&mut used, &mut forward);
            let mut backward = vec![a];
            if forward.last() != Some(&a) {
                extend(&mut used, &mut backward);
            }
            let mut path: Vec<Vec2> = backward[1..].iter().rev().copied().collect();
            path.extend(forward);
            out.push(path);
        }
        out
    }
}

/// A double's bits as a hash key, with -0 read as +0, as Swift's `==` (and
/// so its `Hashable`) has them equal. No NaN reaches a key: `pieces` drops
/// segments that are not finite.
fn bits(x: f64) -> u64 {
    if x == 0.0 { 0 } else { x.to_bits() }
}

impl FaceCopy {
    /// Whether the copy's trimmed domain holds `p` or a point `tie` from it
    /// in one of sixteen directions round it: a point on its boundary, or at
    /// a corner of it no sharper than 22.5°, is held.
    pub fn holds(&self, p: P2<ParamSpace>, tie: f64) -> bool {
        if (self.contains)(p) {
            return true;
        }
        if !(tie > 0.0) {
            return false;
        }
        (0..16).any(|k| {
            let a = k as f64 * core::f64::consts::PI / 8.0;
            (self.contains)(P2::from_vec(p.v + tie * Vec2::new(real::cos(a), real::sin(a))))
        })
    }
}

/// A picked face's boundary in space, with each vertex's point of its
/// parameter plane.
pub type PickedBoundary = (Vec<Vec<P3<WorldSpace>>>, Vec<Vec<P2<ParamSpace>>>);

impl SceneGeometry {
    /// The face of the solid the scene draws that a point lies on, and the
    /// copy (`Face::pick`): on a landscape, a plateau where the point is at
    /// the truncation's height and |f| reaches it, a riser or a wall where it
    /// stands on one, the graph otherwise, on the tile the inverse tile map
    /// finds; on a parametric surface its one face; on a polyhedron the face
    /// whose plane holds the point and whose loops contain it. `faces` are
    /// the scene's (`faces`), passed where they are kept, so the trim loops a
    /// boundary needs are made once. Ties are settled within a billionth of
    /// `scale`.
    pub fn face_at(&self, p: P3<WorldSpace>, tolerance: f64, faces: Option<&[Face]>) -> Option<FacePick> {
        let owned;
        let faces = match faces {
            Some(f) => f,
            None => {
                owned = self.faces();
                &owned
            }
        };
        Face::pick(p, faces, tolerance, 1e-9 * self.scale())
    }

    /// The boundary of the picked face's copy, in space (`Face::boundary`),
    /// with each vertex's point of the face's parameter plane. Each of its
    /// segments is cut at the face's parameter spacing, then halved until
    /// its image lies within a hundredth of `scale` of the chord, twelve
    /// deep, so the curve follows a curved face between the boundary's own
    /// vertices: the graph along the domain's edge, where its boundary has a
    /// vertex only at the corners.
    pub fn boundary_of(&self, pick: &FacePick, faces: Option<&[Face]>) -> PickedBoundary {
        let owned;
        let faces = match faces {
            Some(f) => f,
            None => {
                owned = self.faces();
                &owned
            }
        };
        let Some(face) = faces.iter().find(|f| f.id == pick.face) else { return (Vec::new(), Vec::new()) };
        if pick.copy >= face.copies.len() {
            return (Vec::new(), Vec::new());
        }
        let copy = &face.copies[pick.copy];
        let tolerance = 0.01 * self.scale();
        let spacing = swift::min(face.domain.u.spacing(), face.domain.v.spacing());
        let point = |c: Vec2| -> Vec3 { copy.place((face.map)(P2::from_vec(c))).position };
        #[allow(clippy::too_many_arguments)]
        fn subdivide(
            point: &dyn Fn(Vec2) -> Vec3,
            tolerance: f64,
            cs: &mut Vec<Vec2>,
            ps: &mut Vec<Vec3>,
            a: Vec2,
            pa: Vec3,
            b: Vec2,
            pb: Vec3,
            depth: usize,
        ) {
            let m = 0.5 * (a + b);
            let pm = point(m);
            if depth < 12 && (pm - 0.5 * (pa + pb)).length() > tolerance {
                subdivide(point, tolerance, cs, ps, a, pa, m, pm, depth + 1);
                subdivide(point, tolerance, cs, ps, m, pm, b, pb, depth + 1);
            } else {
                cs.push(b);
                ps.push(pb);
            }
        }
        let mut paths = Vec::new();
        let mut coords = Vec::new();
        for curve in face.boundary(copy) {
            let mut cs = vec![curve[0].v];
            let mut ps = vec![point(curve[0].v)];
            for w in curve.windows(2) {
                let (a, b) = (w[0], w[1]);
                let n = if spacing > 0.0 {
                    swift::int(((b.v - a.v).length() / spacing).ceil()).map_or(4096, |k| k.clamp(1, 4096) as usize)
                } else {
                    1
                };
                for k in 1..=n {
                    let (c0, p0) = (cs[cs.len() - 1], ps[ps.len() - 1]);
                    let c1 = if k == n { b.v } else { a.v + (b.v - a.v) * (k as f64 / n as f64) };
                    subdivide(&point, tolerance, &mut cs, &mut ps, c0, p0, c1, point(c1), 0);
                }
            }
            paths.push(ps.into_iter().map(P3::from_vec).collect());
            coords.push(cs.into_iter().map(P2::from_vec).collect());
        }
        (paths, coords)
    }

    /// The length the scene's tolerances are fractions of: a landscape's
    /// cell, a parametric surface's grid spacing, the spacing of its
    /// neighbouring samples at the widest, a polyhedron's largest face.
    pub fn scale(&self) -> f64 {
        match self {
            SceneGeometry::Heightfield(h) => h.surface.cell(),
            SceneGeometry::Parametric(s) => s.grid_spacing(),
            SceneGeometry::Solid(s) => {
                let mut best: Option<f64> = None;
                for f in &s.frames {
                    if best.is_none_or(|b| b < f.extent) {
                        best = Some(f.extent);
                    }
                }
                best.unwrap_or(1.0)
            }
        }
    }

    /// How far a point the preview's depth buffer gives can stand from the
    /// face under it, with `units_per_pixel` the size of a pixel: four
    /// pixels, for the depth's precision and the raster's, and where the
    /// depth pass does not draw the faces themselves, how far the surface it
    /// draws stands off them over the point. On a landscape that is the gap
    /// between the grid's triangles, decimated and truncated as the depth
    /// pass draws them, and the solid's top the faces bound
    /// (`Heightfield::surface_height`): the greater of the gaps to the
    /// triangles truncated after they interpolate (`Heightfield::drawn_height`)
    /// and before, as the shader truncates its vertices
    /// (`Surface::clamped_height`), where the point lies within the pixels of
    /// either: a point on a wall's curtain or on walls given as a mesh is
    /// held to the pixels alone. On a parametric surface twice the gap
    /// between its grid, interpolated, and its map, at the point's nearest
    /// parameter. A polyhedron's depth pass draws its faces.
    pub fn pick_tolerance(&self, p: P3<WorldSpace>, units_per_pixel: f64) -> f64 {
        let pixels = 4.0 * units_per_pixel;
        match self {
            SceneGeometry::Heightfield(h) => {
                let top = h.surface_height(p.xy());
                let q = h.surface.in_tile(P2::<DomainSpace>::new(p.x(), p.y()), &h.tiles);
                let drawn = h.drawn_height(p.xy());
                let clamped = h.surface.clamped_height(q, h.step);
                // Only a point on the terrain the depth pass draws stands off the
                // faces by its gap; one on a wall's curtain or a mesh does not.
                if !(swift::min((p.z() - drawn).abs(), (p.z() - clamped).abs()) <= pixels) {
                    return pixels;
                }
                let gap = swift::max((drawn - top).abs(), (clamped - top).abs());
                pixels + if gap.is_finite() { gap } else { 0.0 }
            }
            SceneGeometry::Parametric(s) => {
                let c = s.nearest_parameter(p.v);
                pixels + 2.0 * (s.interpolate(c).v - (s.map)(c).position).length()
            }
            SceneGeometry::Solid(_) => pixels,
        }
    }
}

impl ParametricSurface {
    /// The point of the parameter plane whose image is nearest `p`: from the
    /// tubular neighbourhood's foot where the surface has one, else the
    /// nearest grid sample, by Gauss-Newton steps on the distance, each cut
    /// back until it brings the image nearer, a periodic coordinate taken
    /// into its range and a bounded one held to it.
    pub fn nearest_parameter(&self, p: Vec3) -> P2<ParamSpace> {
        let into = |x: f64, axis: &ParamAxis| -> f64 {
            let lo = swift::min(axis.range.lo, axis.range.hi);
            let hi = swift::max(axis.range.lo, axis.range.hi);
            if !axis.periodic {
                return swift::min(swift::max(x, lo), hi);
            }
            let period = hi - lo;
            x - period * ((x - lo) / period).floor()
        };
        let wrapped = |c: Vec2| Vec2::new(into(c.x, &self.u), into(c.y, &self.v));
        let mut c: Vec2 = match &self.neighbourhood {
            Some(n) => (n.foot)(p).0.v,
            None => {
                let mut best = f64::INFINITY;
                let mut at = 0usize;
                for (k, q) in self.positions.iter().enumerate() {
                    let d = (q.v - p).length_squared();
                    if d < best {
                        best = d;
                        at = k;
                    }
                }
                Vec2::new(
                    self.u.coordinate((at % self.u.samples) as i64),
                    self.v.coordinate((at / self.u.samples) as i64),
                )
            }
        };
        let mut r = (self.map)(P2::from_vec(c)).position - p;
        for _ in 0..40 {
            let jet = (self.map)(P2::from_vec(c));
            let a = jet.du.dot(jet.du);
            let b = jet.du.dot(jet.dv);
            let e = jet.dv.dot(jet.dv);
            let det = a * e - b * b;
            if !(det > 0.0) {
                break;
            }
            let gu = jet.du.dot(r);
            let gv = jet.dv.dot(r);
            let mut step = -Vec2::new((e * gu - b * gv) / det, (a * gv - b * gu) / det);
            let reach = Vec2::new(4.0 * self.u.spacing(), 4.0 * self.v.spacing());
            // `simd_clamp(step, -reach, reach)`.
            step = Vec2::new(
                swift::min(swift::max(step.x, -reach.x), reach.x),
                swift::min(swift::max(step.y, -reach.y), reach.y),
            );
            let mut moved = false;
            for _ in 0..20 {
                let next = wrapped(c + step);
                let rn = (self.map)(P2::from_vec(next)).position - p;
                if rn.length_squared() < r.length_squared() {
                    c = next;
                    r = rn;
                    moved = true;
                    break;
                }
                step *= 0.5;
            }
            let ratio = Vec2::new(step.x.abs() / self.u.spacing(), step.y.abs() / self.v.spacing());
            if !(moved && ratio.reduce_max() > 1e-14) {
                break;
            }
        }
        P2::from_vec(c)
    }

    /// The edges of the parameter rectangle, each at the other axis's
    /// samples, at least 64 of them: on a periodic axis the one seam where the
    /// rectangle is glued, on a bounded axis both of its ends.
    pub fn seams(&self) -> Vec<Vec<P2<ParamSpace>>> {
        let run = |axis: &ParamAxis| -> Vec<f64> {
            let n = axis.samples.max(64);
            (0..=n).map(|k| axis.range.lo + axis.range.length() * k as f64 / n as f64).collect()
        };
        let us = if self.u.periodic { vec![self.u.range.lo] } else { vec![self.u.range.lo, self.u.range.hi] };
        let vs = if self.v.periodic { vec![self.v.range.lo] } else { vec![self.v.range.lo, self.v.range.hi] };
        let mut out: Vec<Vec<P2<ParamSpace>>> =
            us.iter().map(|&x| run(&self.v).into_iter().map(|y| P2::new(x, y)).collect()).collect();
        out.extend(vs.iter().map(|&y| run(&self.u).into_iter().map(|x| P2::new(x, y)).collect()));
        out
    }

    /// The widest spacing between neighbouring samples of the grid.
    pub fn grid_spacing(&self) -> f64 {
        let mut widest = 0.0;
        for j in 0..self.v.samples as i64 {
            for i in 0..self.u.samples as i64 {
                let p = self.position(i, j).v;
                if i + 1 < self.u.samples as i64 || self.u.periodic {
                    widest = swift::max(widest, (self.position(i + 1, j).v - p).length());
                }
                if j + 1 < self.v.samples as i64 || self.v.periodic {
                    widest = swift::max(widest, (self.position(i, j + 1).v - p).length());
                }
            }
        }
        widest
    }
}
