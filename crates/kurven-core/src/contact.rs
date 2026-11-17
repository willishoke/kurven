//! Where a part of a composite meets a landscape's terrain: the port of
//! `Sources/KurvenCore/Contact.swift`.
//!
//! The curve on the part's surface where its point w has w.z = H(w.x, w.y),
//! with H the terrain's height as the composite's cut has it (`Slab`), so the
//! curve lies where the part's strokes, cut by the landscape, end.
//!
//! H is affine over each piece of the terrain's top (`TopPiece`): the slab's
//! prisms, each split where the triangle's plane meets the truncation. The
//! curve is solved piece by piece, so every point of it lies on both
//! surfaces and every chord in one piece, on the terrain; where H steps --
//! a riser, the terrain's edge, the footprint's -- the curve ends on the
//! step.
//!
//! - On a polyhedron each face is solved exactly: over each prism, the face's
//!   points on the terrain are the sides of the convex polygon the face
//!   leaves inside the prism that lie on its top. The pieces are joined end
//!   to end and clipped to the face's trim loops (`clipped`).
//! - On the round torus, over each piece, the curve is where the torus
//!   meets the piece's plane. On a level piece it is closed form: the
//!   torus stands upright, so the curve is circles about its axis, clipped
//!   to the piece. On any other it is the zero set of the torus's distance
//!   on the plane, which has Lipschitz constant 1 in space
//!   (`TubularNeighbourhood.foot`): blocks of cells, then the piece, then
//!   squares are passed over where the bound shows no zero, and split
//!   otherwise, down to squares a quarter of a cell, at most an eighth of
//!   the tube's radius (`zeros`), so the work goes with the curve's length.
//!   Each zero is bisected to the last bit. A loop smaller than a
//!   sixty-fourth of a square can be missed.
//!
//! Contact within the scene's tolerance (`Slab.start`) of a polyhedron
//! face's boundary, along it, is the part's edge and is not contact; a face
//! within the tolerance of the terrain's plane all over meets it nowhere.
//! So with the torus: a level piece it reaches past by no more than the
//! tolerance, or sinks into by no more, it meets nowhere.
//!
//! A prism with no top -- a corner of its triangle not a number, and no
//! truncation over it -- is no terrain, and nothing meets it.
//!
//! The curves are in the part's own coordinates, on a polyhedron with the
//! face each lies on, on a surface with each vertex's coordinate.

use std::collections::{BTreeSet, HashMap, HashSet};

use kurven_math::{real, swift};

use crate::composite::{Composite, Occluder, Placement};
use crate::hidden_line::HiddenLine;
use crate::manifest::{HeightPolicy, LayerRole, LayerSource, LayerSpec};
use crate::parametric::{ParametricSurface, PortRefusal};
use crate::polyline::{Layer, PolylineSet};
use crate::scalar_field::{Trim, clipped};
use crate::scene::SceneGeometry;
use crate::simd::{Vec2, Vec3};
use crate::slab::{HalfSpace, Slab, SlabCopy};
use crate::solid::Solid;
use crate::spaces::{P2, P3, ParamSpace, ViewSpace, WorldSpace};
use crate::visibility::SurfaceVisibility;

/// Where a part meets the landscapes: paths in the part's own coordinates,
/// on a polyhedron with the face each lies on, on a surface with each
/// path's surface coordinates.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ContactCurves {
    pub paths: Vec<Vec<P3<WorldSpace>>>,
    /// The face each path lies on, on a polyhedron; empty on a surface.
    pub faces: Vec<usize>,
    /// Each path's surface coordinates, on a surface; empty on a polyhedron.
    pub coords: Vec<Vec<P2<ParamSpace>>>,
}

impl ContactCurves {
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }
}

impl Composite {
    /// Where part `i` meets each landscape of the composite (`ContactCurves`).
    /// Empty for a landscape, and for a part that meets none.
    pub fn contact(&self, i: usize) -> ContactCurves {
        let mut out = ContactCurves::default();
        let part = &self.parts[i];
        let placement = part.placement;
        for (j, occluder) in self.occluders.iter().enumerate() {
            if j == i {
                continue;
            }
            let Occluder::Slab { slab, offset } = occluder else { continue };
            match part.scene.geometry() {
                SceneGeometry::Heightfield(_) => return ContactCurves::default(),
                SceneGeometry::Solid(s) => {
                    for f in 0..s.faces.len() {
                        for path in slab.contact_on_face(f, s, placement, *offset) {
                            out.paths.push(path);
                            out.faces.push(f);
                        }
                    }
                }
                SceneGeometry::Parametric(s) => {
                    let Some(e) = &s.neighbourhood else { continue };
                    for path in slab.contact_on_surface(s, placement, *offset) {
                        // Into the part's own coordinates, with the coordinate
                        // of each point, unwrapped along the path.
                        let points: Vec<Vec3> =
                            path.iter().map(|&p| (p + *offset - placement.offset) / placement.scale).collect();
                        let mut coords: Vec<Vec2> = points.iter().map(|&p| (e.foot)(p).0.v).collect();
                        for k in 1..coords.len() {
                            for (axis, period) in [(0, s.u.period()), (1, s.v.period())] {
                                let Some(period) = period else { continue };
                                let turns = ((coords[k][axis] - coords[k - 1][axis]) / period).round();
                                coords[k][axis] -= period * turns;
                            }
                        }
                        out.paths.push(points.into_iter().map(P3::from_vec).collect());
                        out.coords.push(coords.into_iter().map(P2::from_vec).collect());
                    }
                }
            }
        }
        out
    }

    /// Part `i`'s contact curves as a layer, drawn in the part's outline
    /// style, and in view space as the part's own geometry shows them: on a
    /// polyhedron each path tested on its face (`Solid::visible_curves`), on
    /// a surface by its coordinates (`clipOnChords`). `None` where the part
    /// meets no landscape. The layer's curves are carried as given, as a
    /// file layer's are.
    pub fn visible_contact(&self, i: usize) -> Result<Option<(Layer, PolylineSet<ViewSpace>)>, PortRefusal> {
        let contact = self.contact(i);
        if contact.is_empty() {
            return Ok(None);
        }
        let scene = &self.parts[i].scene;
        let view = scene.camera.view;
        // The part's outline style: its first outline layer's.
        let style = scene.layers().iter().find(|l| l.spec.role == LayerRole::Outline).map(|l| &l.spec);
        let mut spec = LayerSpec::new(
            "contact",
            LayerRole::Outline,
            LayerSource::File { vertices: String::new(), offsets: String::new() },
            style.map_or(0.5, |s| s.width),
            HeightPolicy::Surface,
        );
        spec.color = style.map_or_else(|| "#000000".to_string(), |s| s.color.clone());
        match scene.geometry() {
            SceneGeometry::Solid(s) => {
                let occlusion = s.occlusion(&view);
                let mut seen: Vec<Vec<P3<WorldSpace>>> = Vec::new();
                let mut done = BTreeSet::new();
                for &f in &contact.faces {
                    if !done.insert(f) {
                        continue;
                    }
                    let on: Vec<Vec<P3<WorldSpace>>> = (0..contact.paths.len())
                        .filter(|&k| contact.faces[k] == f)
                        .map(|k| contact.paths[k].clone())
                        .collect();
                    seen.extend(s.visible_curves(&on, f, &occlusion));
                }
                Ok(Some((
                    Layer { spec, paths: PolylineSet::from_paths(&contact.paths) },
                    PolylineSet::from_paths(&seen).mapped(&view),
                )))
            }
            SceneGeometry::Parametric(surface) => {
                let paths = PolylineSet::from_paths_with_coords(&contact.paths, &contact.coords);
                // A composite refuses a surface tested by its coordinate
                // image (`Composite::new`), so a part's surface here has a
                // tubular neighbourhood, as Swift's precondition asks.
                let visibility = SurfaceVisibility::sphere_tracing(surface.clone(), view)
                    .expect("a composite's parametric part has a tubular neighbourhood");
                let seen = Composite::clip_on_chords(&paths.mapped(&view), &visibility);
                Ok(Some((Layer { spec, paths }, seen)))
            }
            SceneGeometry::Heightfield(_) => Ok(None),
        }
    }
}

impl Composite {
    /// Curves on a surface whose chords lie on a landscape's terrain, cut
    /// where the surface hides them, as `HiddenLine::clip_on` cuts them --
    /// each vertex tested where it lies on the surface, and a segment along
    /// which the verdict changes bisected on the test at the coordinates
    /// along it -- but each cut placed on the chord, at the fraction of the
    /// way the bisection found, so that what is drawn stays on the terrain:
    /// the point of the surface there is off it by the chord's sagitta.
    /// One implementation with `HiddenLine::clip_on`'s, the cut placed on
    /// the chord (`clip_on_runs`); Swift tests the vertices in parallel, here
    /// in turn.
    pub fn clip_on_chords(paths: &PolylineSet<ViewSpace>, visibility: &SurfaceVisibility) -> PolylineSet<ViewSpace> {
        if paths.coords.is_none() {
            return PolylineSet::empty();
        }
        // Sphere tracing asks for no image, and the curves carry coordinates:
        // neither of `clip_on`'s refusals can be met.
        let runs = HiddenLine::clip_on_runs(paths, visibility, false, false, true)
            .expect("curves with coordinates, tested by sphere tracing");
        PolylineSet::from_paths(&runs)
    }
}

/// A piece of a landscape's terrain over which its height is affine: a
/// convex polygon of the landscape's own (x, y), counter-clockwise, and the
/// height over it, g · p + h. One of a prism's tops (`Slab::tops`).
#[derive(Debug, Clone, PartialEq)]
pub struct TopPiece {
    pub polygon: Vec<Vec2>,
    pub gradient: Vec2,
    pub constant: f64,
}

impl TopPiece {
    pub fn height(&self, p: Vec2) -> f64 {
        self.gradient.dot(p) + self.constant
    }

    /// The point of the terrain over `p`.
    pub fn point(&self, p: Vec2) -> Vec3 {
        Vec3::new(p.x, p.y, self.height(p))
    }

    /// Whether `p` lies in the polygon, or within `slack` of it.
    pub fn contains(&self, p: Vec2, slack: f64) -> bool {
        let n = self.polygon.len();
        (0..n).all(|k| {
            let (a, b) = (self.polygon[k], self.polygon[(k + 1) % n]);
            let e = b - a;
            e.x * (p.y - a.y) - e.y * (p.x - a.x) >= -slack * e.length()
        })
    }
}

/// A segment's two ends.
pub type Segment = (Vec3, Vec3);

/// The range of the terrain's height over a block of cells.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZRange {
    pub lo: f64,
    pub hi: f64,
}

/// Swift's `Int(x)` of a rounded double, `None` where it would trap.
fn int(x: f64) -> Option<i64> {
    swift::int(x)
}

/// The shoelace sum, twice the signed area.
fn area2(polygon: &[Vec2]) -> f64 {
    let mut area = 0.0;
    for k in 0..polygon.len() {
        let (p, q) = (polygon[k], polygon[(k + 1) % polygon.len()]);
        area += p.x * q.y - q.x * p.y;
    }
    area
}

impl Slab {
    /// The tops of the prisms over one triangle of cell (i, j) in one copy
    /// (`prisms`): each prism's sides bound a convex polygon, and over it
    /// the top is the least of its upward half-spaces, the triangle's plane
    /// and the truncation; the polygon is split where they meet. A prism
    /// with no top -- a corner of its triangle not a number, and no
    /// truncation over it -- has none.
    pub fn tops(&self, i: usize, j: usize, upper: bool, copy: &SlabCopy) -> Vec<TopPiece> {
        let (fi, fj) = (i as f64, j as f64);
        let square =
            vec![Vec2::new(fi, fj), Vec2::new(fi + 1.0, fj), Vec2::new(fi + 1.0, fj + 1.0), Vec2::new(fi, fj + 1.0)];
        let back = copy.linear.inverse();
        // A point of index coordinates in the landscape's (x, y).
        let xy = |q: Vec2| back.apply(q - copy.offset);
        // A reflected copy turns the polygon over.
        let turned = copy.linear.determinant() < 0.0;
        let mut out = Vec::new();
        for half_spaces in self.prisms(i, j, upper, copy) {
            let mut polygon = square.clone();
            for h in half_spaces.iter().filter(|h| h.n.z == 0.0) {
                polygon = Slab::clip2d(&polygon, &|p| h.n.x * p.x + h.n.y * p.y - h.c);
                if polygon.len() < 3 {
                    break;
                }
            }
            if polygon.len() < 3 {
                continue;
            }
            // Each top as an affine height over (u, v): a · q + b.
            let ups: Vec<(Vec2, f64)> = half_spaces
                .iter()
                .filter(|h| h.n.z > 0.0)
                .map(|h| (-Vec2::new(h.n.x, h.n.y) / h.n.z, h.c / h.n.z))
                .collect();
            for (k, top) in ups.iter().enumerate() {
                let mut region = polygon.clone();
                for (l, other) in ups.iter().enumerate() {
                    if l == k {
                        continue;
                    }
                    region = Slab::clip2d(&region, &|p| (top.0 - other.0).dot(p) + top.1 - other.1);
                    if region.len() < 3 {
                        break;
                    }
                }
                // A polygon with no area -- a corner or a side two pieces
                // share -- is none.
                if !(region.len() >= 3 && area2(&region) > 1e-18) {
                    continue;
                }
                // Into (x, y): q = L p + o.
                let mut mapped: Vec<Vec2> = region.iter().map(|&q| xy(q)).collect();
                if turned {
                    mapped.reverse();
                }
                out.push(TopPiece {
                    polygon: mapped,
                    gradient: copy.linear.transpose().apply(top.0),
                    constant: top.0.dot(copy.offset) + top.1,
                });
            }
        }
        out
    }

    /// The cells of a copy a box of the landscape's (x, y) can reach, a
    /// cell beyond on every side; `None` where it reaches none.
    pub fn cells(&self, lo: Vec2, hi: Vec2, copy: &SlabCopy) -> Option<((i64, i64), (i64, i64))> {
        let corners: Vec<Vec2> = [lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y)]
            .iter()
            .map(|&c| copy.linear.apply(c) + copy.offset)
            .collect();
        let fold = |f: fn(f64, f64) -> f64, g: fn(&Vec2) -> f64| corners.iter().map(g).reduce(f).expect("corners");
        let (ulo, uhi) = (fold(swift::min, |p| p.x), fold(swift::max, |p| p.x));
        let (vlo, vhi) = (fold(swift::min, |p| p.y), fold(swift::max, |p| p.y));
        if !(self.nx >= 2 && self.ny >= 2 && ulo.is_finite() && uhi.is_finite() && vlo.is_finite() && vhi.is_finite()) {
            return None;
        }
        let i0 = int(swift::max(ulo.floor() - 1.0, 0.0))?;
        let i1 = int(swift::min(uhi.ceil(), (self.nx - 2) as f64))?;
        let j0 = int(swift::max(vlo.floor() - 1.0, 0.0))?;
        let j1 = int(swift::min(vhi.ceil(), (self.ny - 2) as f64))?;
        if i0 > i1 || j0 > j1 {
            return None;
        }
        Some(((i0, i1), (j0, j1)))
    }

    /// Where face `f` of `solid`, standing at `placement`, meets the terrain
    /// of the landscape standing at `offset`: paths in the solid's own
    /// coordinates, clipped to the face's trim loops.
    ///
    /// The face's plane is X(a, b) = o + a e₁ + b e₂ in the landscape's own
    /// coordinates, (a, b) its frame's (`FaceFrame`). Over each prism the
    /// plane's points inside it are a convex polygon of (a, b), the face's
    /// rectangle, a little larger than the face, clipped by the prism's
    /// half-spaces; the polygon's sides on the triangle's plane or the
    /// truncation are where X.z = H. A top the face lies within the scene's
    /// tolerance of, all over its rectangle, clips nothing and draws no side.
    /// A stretch of the clipped paths within the tolerance of a side of the
    /// face's loops, along it, is that edge of the solid, and is left out.
    pub fn contact_on_face(
        &self,
        f: usize,
        solid: &Solid,
        placement: Placement,
        offset: Vec3,
    ) -> Vec<Vec<P3<WorldSpace>>> {
        let frame = solid.frames[f];
        let face = solid.face(f);
        let s = placement.scale;
        let o = s * frame.origin + placement.offset - offset;
        let (e1, e2) = (s * frame.u, s * frame.v);
        let (ua, va) = (face.domain.u, face.domain.v);
        let extent = swift::max(ua.range.length(), va.range.length());
        let pad = 0.01 * extent;
        let lo = Vec2::new(ua.range.lo - pad, va.range.lo - pad);
        let hi = Vec2::new(ua.range.hi + pad, va.range.hi + pad);
        let rectangle = vec![lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y)];
        let mut segments: Vec<Segment> = Vec::new();
        for copy in &self.copies {
            let big_o = Slab::flat(copy, o, true);
            let big_a = Slab::flat(copy, e1, false);
            let big_b = Slab::flat(copy, e2, false);
            // The rectangle's box in the landscape's (x, y).
            let corners: Vec<Vec3> = rectangle.iter().map(|c| o + c.x * e1 + c.y * e2).collect();
            let fold = |f: fn(f64, f64) -> f64, g: fn(&Vec3) -> f64| corners.iter().map(g).reduce(f).expect("corners");
            let xlo = Vec2::new(fold(swift::min, |p| p.x), fold(swift::min, |p| p.y));
            let xhi = Vec2::new(fold(swift::max, |p| p.x), fold(swift::max, |p| p.y));
            let Some(((i0, i1), (j0, j1))) = self.cells(xlo, xhi, copy) else { continue };
            for j in j0..=j1 {
                for i in i0..=i1 {
                    for upper in [false, true] {
                        // Without the floor, which bounds no top, and on a
                        // slab as thin as a rounding would leave the face
                        // nothing.
                        for prism in self.prisms(i as usize, j as usize, upper, copy) {
                            let half_spaces: Vec<HalfSpace> = prism.into_iter().filter(|h| h.n.z >= 0.0).collect();
                            segments.extend(
                                Slab::sides_on_tops(&rectangle, &half_spaces, big_o, big_a, big_b, hi - lo, self.start)
                                    .into_iter()
                                    .map(|(p, q)| (Vec3::new(p.x, p.y, 0.0), Vec3::new(q.x, q.y, 0.0))),
                            );
                        }
                    }
                }
            }
        }
        let paths: Vec<Vec<P2<ParamSpace>>> = Slab::chains(&segments, 1e-9 * extent)
            .into_iter()
            .map(|c| c.into_iter().map(|p| P2::new(p.x, p.y)).collect())
            .collect();
        let Some(loops) = face.trim_loops().cloned() else { return Vec::new() };
        let trim = Trim { trim_loops: Some(loops.clone()), ..Trim::default() };
        let clipped_paths =
            clipped(vec![(0.0, paths)], face.domain, &trim).into_iter().next().map_or_else(Vec::new, |c| c.1);
        // Off the face's edges: a segment within the tolerance of one side,
        // all along it, is that side.
        let tolerance = self.start / s;
        let sides: Vec<(Vec2, Vec2)> =
            loops.iter().flat_map(|l| l.windows(2).map(|w| (w[0].v, w[1].v)).collect::<Vec<_>>()).collect();
        let along_side = |a: Vec2, b: Vec2| -> bool {
            sides.iter().any(|&(p, q)| {
                let e = q - p;
                let l2 = e.length_squared();
                if !(l2 > 0.0) {
                    return false;
                }
                let off = |x: Vec2| {
                    let t = swift::min(swift::max((x - p).dot(e) / l2, 0.0), 1.0);
                    (x - (p + t * e)).length()
                };
                off(a) <= tolerance && off(b) <= tolerance
            })
        };
        let mut out = Vec::new();
        for path in clipped_paths {
            let mut run: Vec<P3<WorldSpace>> = Vec::new();
            for w in path.windows(2) {
                let (a, b) = (w[0], w[1]);
                if along_side(a.v, b.v) {
                    if run.len() >= 2 {
                        out.push(std::mem::take(&mut run));
                    }
                    run.clear();
                    continue;
                }
                if run.is_empty() {
                    run = vec![P3::from_vec(frame.point(a.v))];
                }
                run.push(P3::from_vec(frame.point(b.v)));
            }
            if run.len() >= 2 {
                out.push(run);
            }
        }
        out
    }

    /// The sides of `polygon` clipped by `half_spaces` (n · X ≤ c, X =
    /// origin + a A + b B) that lie on a half-space bounding the prism from
    /// above. A top the plane lies within `tolerance` of, all over the
    /// polygon's box of size `size`, clips nothing; a side parallel to the
    /// plane keeps all of the polygon or none. (Swift's `Slab.tops(_:_:
    /// origin:_:_:size:tolerance:)`, renamed beside `tops` of a cell.)
    pub fn sides_on_tops(
        polygon: &[Vec2],
        half_spaces: &[HalfSpace],
        origin: Vec3,
        a: Vec3,
        b: Vec3,
        size: Vec2,
        tolerance: f64,
    ) -> Vec<(Vec2, Vec2)> {
        let mut polygon = polygon.to_vec();
        let centre = 0.5 * (polygon[0] + polygon[2]);
        // The half-space each side lies on; -1 for the rectangle's own.
        let mut sides: Vec<i64> = vec![-1; polygon.len()];
        for (k, h) in half_spaces.iter().enumerate() {
            let (n, c) = (h.n, h.c);
            let k0 = n.dot(origin) - c;
            let ka = n.dot(a);
            let kb = n.dot(b);
            let reach = ka.abs() * size.x + kb.abs() * size.y;
            if n.z > 0.0 {
                // A top: n · X − c over n.z is how far over it X stands.
                let most = (k0 + ka * centre.x + kb * centre.y).abs() + 0.5 * reach;
                if most <= tolerance * n.z {
                    continue;
                }
            } else if reach <= 1e-12 * (n.dot(origin).abs() + c.abs() + n.length()) {
                // Parallel: all of the polygon or none of it.
                if k0 > 0.0 {
                    return Vec::new();
                }
                continue;
            }
            let mut out = Vec::new();
            let mut out_sides = Vec::new();
            for e in 0..polygon.len() {
                let (p, q) = (polygon[e], polygon[(e + 1) % polygon.len()]);
                let fp = k0 + ka * p.x + kb * p.y;
                let fq = k0 + ka * q.x + kb * q.y;
                if fp <= 0.0 {
                    out.push(p);
                    out_sides.push(sides[e]);
                    if fq > 0.0 {
                        out.push(p + (q - p) * (fp / (fp - fq)));
                        out_sides.push(k as i64);
                    }
                } else if fq <= 0.0 {
                    out.push(p + (q - p) * (fp / (fp - fq)));
                    out_sides.push(sides[e]);
                }
            }
            polygon = out;
            sides = out_sides;
            if polygon.len() < 3 {
                return Vec::new();
            }
        }
        (0..polygon.len())
            .filter_map(|e| {
                if sides[e] < 0 || !(half_spaces[sides[e] as usize].n.z > 0.0) {
                    return None;
                }
                let (p, q) = (polygon[e], polygon[(e + 1) % polygon.len()]);
                if p == q { None } else { Some((p, q)) }
            })
            .collect()
    }

    /// Segments joined where their ends meet within `tolerance`: a path
    /// runs through every point where exactly two segments meet and ends
    /// wherever another number do; a closed one ends where it begins. A
    /// segment found twice, along a side two pieces share, is one.
    pub fn chains(segments: &[Segment], tolerance: f64) -> Vec<Vec<Vec3>> {
        let mut points: Vec<Vec3> = Vec::new();
        let mut buckets: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
        // Swift's `Int64` of a coordinate over the tolerance traps where it
        // is not finite or does not fit; such a point is a node of its own.
        let key = |p: Vec3| -> Option<[i64; 3]> {
            Some([int((p.x / tolerance).floor())?, int((p.y / tolerance).floor())?, int((p.z / tolerance).floor())?])
        };
        let mut node = |p: Vec3, points: &mut Vec<Vec3>| -> usize {
            let Some(k) = key(p) else {
                points.push(p);
                return points.len() - 1;
            };
            for dx in -1i64..=1 {
                for dy in -1i64..=1 {
                    for dz in -1i64..=1 {
                        let q = [k[0].wrapping_add(dx), k[1].wrapping_add(dy), k[2].wrapping_add(dz)];
                        if let Some(list) = buckets.get(&q) {
                            for &n in list {
                                if (points[n] - p).length() <= tolerance {
                                    return n;
                                }
                            }
                        }
                    }
                }
            }
            points.push(p);
            buckets.entry(k).or_default().push(points.len() - 1);
            points.len() - 1
        };
        let mut edges: Vec<(usize, usize)> = Vec::new();
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        for &(p, q) in segments {
            let a = node(p, &mut points);
            let b = node(q, &mut points);
            if a == b || !seen.insert((a.min(b), a.max(b))) {
                continue;
            }
            edges.push((a, b));
        }
        let mut at: Vec<Vec<usize>> = vec![Vec::new(); points.len()];
        for (e, &(a, b)) in edges.iter().enumerate() {
            at[a].push(e);
            at[b].push(e);
        }
        let mut used = vec![false; edges.len()];
        let mut out: Vec<Vec<Vec3>> = Vec::new();
        let walk = |start: usize, first: usize, used: &mut Vec<bool>, out: &mut Vec<Vec<Vec3>>| {
            let mut path = vec![points[start]];
            let (mut here, mut e) = (start, first);
            loop {
                used[e] = true;
                here = if edges[e].0 == here { edges[e].1 } else { edges[e].0 };
                path.push(points[here]);
                if at[here].len() != 2 || here == start {
                    break;
                }
                let Some(&next) = at[here].iter().find(|&&x| !used[x]) else { break };
                e = next;
            }
            out.push(path);
        };
        for (n, ends) in at.iter().enumerate() {
            if ends.len() == 2 {
                continue;
            }
            for &e in ends {
                if !used[e] {
                    walk(n, e, &mut used, &mut out);
                }
            }
        }
        for e in 0..edges.len() {
            if !used[e] {
                walk(edges[e].0, e, &mut used, &mut out);
            }
        }
        out
    }

    /// Where `surface` -- the round torus about its z axis, upright as a
    /// placement leaves it, whose distance has Lipschitz constant 1 in space
    /// -- standing at `placement`, meets the terrain of the landscape
    /// standing at `offset`: paths of the landscape's own coordinates, each
    /// point on both surfaces and each chord in one piece of the terrain
    /// (`ContactCurves`).
    pub fn contact_on_surface(
        &self,
        surface: &ParametricSurface,
        placement: Placement,
        offset: Vec3,
    ) -> Vec<Vec<Vec3>> {
        let Some(e) = &surface.neighbourhood else { return Vec::new() };
        if self.nx < 2 || self.ny < 2 {
            return Vec::new();
        }
        let s = placement.scale;
        let shift = placement.offset - offset;
        // The torus's distance from a point of the landscape's coordinates.
        let distance = |x: Vec3| s * (e.foot)((x - shift) / s).1;
        let centre = s * e.centre + shift;
        let radius = s * e.radius;
        // The tube's radius and the core's: the ball that holds the torus
        // reaches R + r, and its centre stands R − r off the tube's core.
        let tube = 0.5 * (radius - s * (e.foot)(e.centre).1);
        let ring = radius - tube;
        let side = swift::min(self.cell / 4.0, tube / 8.0);
        let least = side / 64.0;
        // Where a closed loop of the curve over a piece can be: round a
        // critical point of the distance on the piece's plane. The distance
        // is smooth but on the axis and the core circle, and elsewhere its
        // gradient points away from the core; so its critical points on a
        // plane with normal n are on the axis, on the core circle, or at
        // C(θ) + t n with the core's C(θ) turned toward ±n's horizontal part.
        let critical = |piece: &TopPiece| -> Vec<Vec2> {
            let up = Vec3::new(-piece.gradient.x, -piece.gradient.y, 1.0);
            let norm = up.length();
            let n = up / norm;
            let c = piece.constant / norm;
            let core = centre;
            let flat = Vec2::new(n.x, n.y);
            let mut out: Vec<Vec3> = Vec::new();
            if n.z.abs() > 0.0 {
                out.push(Vec3::new(core.x, core.y, (c - n.x * core.x - n.y * core.y) / n.z));
            }
            let on = |theta: f64| core + ring * Vec3::new(real::cos(theta), real::sin(theta), 0.0);
            if flat.length() > 0.0 {
                let theta = real::atan2(flat.y, flat.x);
                for t in [theta, theta + std::f64::consts::PI] {
                    let p = on(t);
                    out.push(p + (c - n.dot(p)) * n);
                }
                // The core circle in the plane: n · (core + ring θ̂) = c.
                let amplitude = ring * flat.length();
                let d = c - n.dot(core);
                if d.abs() <= amplitude {
                    let phase = real::atan2(flat.y, flat.x);
                    let turn = real::acos(d / amplitude);
                    out.extend([on(phase + turn), on(phase - turn)]);
                }
            }
            out.into_iter().map(|p| Vec2::new(p.x, p.y)).collect()
        };
        let lo = Vec2::new(centre.x - radius, centre.y - radius);
        let hi = Vec2::new(centre.x + radius, centre.y + radius);
        // The cells the torus can meet, found from blocks of cells: a block
        // is passed over where the torus is farther from the middle of the
        // box that holds its terrain than the box's half-diagonal.
        let ranges = self.cell_ranges();
        let mut cells: Vec<(usize, i64, i64)> = Vec::new();
        for (k, copy) in self.copies.iter().enumerate() {
            let Some(((ilo, ihi), (jlo, jhi))) = self.cells(lo, hi, copy) else { continue };
            let back = copy.linear.inverse();
            let reaches = |i0: i64, i1: i64, j0: i64, j1: i64, z: ZRange| -> bool {
                if !(z.lo.is_finite() && z.hi.is_finite()) {
                    return true;
                }
                let corners: Vec<Vec2> = [
                    Vec2::new(i0 as f64, j0 as f64),
                    Vec2::new((i1 + 1) as f64, j0 as f64),
                    Vec2::new(i0 as f64, (j1 + 1) as f64),
                    Vec2::new((i1 + 1) as f64, (j1 + 1) as f64),
                ]
                .iter()
                .map(|&q| back.apply(q - copy.offset))
                .collect();
                let clo = corners.iter().fold(corners[0], |a, &p| a.min(p));
                let chi = corners.iter().fold(corners[0], |a, &p| a.max(p));
                let blo = Vec3::new(clo.x, clo.y, z.lo);
                let bhi = Vec3::new(chi.x, chi.y, z.hi);
                distance(0.5 * (blo + bhi)).abs() <= 0.5 * (bhi - blo).length() * (1.0 + 1e-9)
            };
            // Depth first, as Swift's recursion descends.
            let mut stack: Vec<(usize, i64, i64)> = vec![(ranges.len() - 1, 0, 0)];
            while let Some((level, bi, bj)) = stack.pop() {
                let size = 1i64 << level;
                let i0 = (bi * size).max(ilo);
                let i1 = ((bi + 1) * size - 1).min(ihi);
                let j0 = (bj * size).max(jlo);
                let j1 = ((bj + 1) * size - 1).min(jhi);
                if i0 > i1 || j0 > j1 {
                    continue;
                }
                let (w, z) = (&ranges[level].0, &ranges[level].1);
                if !reaches(i0, i1, j0, j1, z[(bj * *w as i64 + bi) as usize]) {
                    continue;
                }
                if level == 0 {
                    cells.push((k, i0, j0));
                    continue;
                }
                // Pushed in reverse, so they are visited in Swift's order.
                for (di, dj) in [(1, 1), (0, 1), (1, 0), (0, 0)] {
                    stack.push((level - 1, 2 * bi + di, 2 * bj + dj));
                }
            }
        }
        // The pieces over them that the torus can meet.
        let mut pieces: Vec<TopPiece> = Vec::new();
        for &(copy, i, j) in &cells {
            for upper in [false, true] {
                for piece in self.tops(i as usize, j as usize, upper, &self.copies[copy]) {
                    let points: Vec<Vec3> = piece.polygon.iter().map(|&p| piece.point(p)).collect();
                    let middle = points.iter().fold(Vec3::ZERO, |a, &p| a + p) / points.len() as f64;
                    let reach = points.iter().map(|&p| (p - middle).length()).reduce(swift::max).expect("points");
                    if distance(middle).abs() <= reach * (1.0 + 1e-9) + 1e-300 {
                        pieces.push(piece);
                    }
                }
            }
        }
        let tolerance = self.start;
        let mut found: Vec<Segment> = Vec::new();
        for piece in &pieces {
            if piece.gradient == Vec2::ZERO {
                found.extend(Slab::circles(
                    piece,
                    Vec2::new(centre.x, centre.y),
                    centre.z,
                    ring,
                    tube,
                    least,
                    tolerance,
                ));
            } else {
                found.extend(Slab::zeros(&distance, piece, side, least, (lo, hi), &critical(piece)));
            }
        }
        Slab::chains(&found, self.start)
    }

    /// The range of the terrain's height over each cell, and over blocks of
    /// 2ᵏ by 2ᵏ cells, level by level, up to one block: what bounds the top
    /// over a cell, the least of its triangles' planes and the truncation,
    /// from below and above. Unbounded over a cell with a corner not a
    /// number. Each level is its width in blocks and the ranges by rows.
    pub fn cell_ranges(&self) -> Vec<(usize, Vec<ZRange>)> {
        let least = self.bands.iter().map(|b| b.height).reduce(swift::min).unwrap_or(f64::INFINITY);
        let (cx, cy) = (self.nx - 1, self.ny - 1);
        let nx = self.nx;
        let mut z = vec![ZRange { lo: 0.0, hi: 0.0 }; cx * cy];
        for j in 0..cy {
            for i in 0..cx {
                let h = [
                    self.heights[j * nx + i],
                    self.heights[j * nx + i + 1],
                    self.heights[(j + 1) * nx + i],
                    self.heights[(j + 1) * nx + i + 1],
                ];
                z[j * cx + i] = if h.iter().all(|x| x.is_finite()) {
                    let hmin = h.iter().copied().reduce(swift::min).expect("four");
                    let hmax = h.iter().copied().reduce(swift::max).expect("four");
                    ZRange { lo: swift::min(hmin, least), hi: hmax }
                } else {
                    ZRange { lo: f64::NEG_INFINITY, hi: f64::INFINITY }
                };
            }
        }
        let mut levels = vec![(cx, z)];
        let (mut w, mut hgt) = (cx, cy);
        while w > 1 || hgt > 1 {
            let (w2, h2) = (w.div_ceil(2), hgt.div_ceil(2));
            let below = &levels[levels.len() - 1].1;
            let mut up = vec![ZRange { lo: f64::INFINITY, hi: f64::NEG_INFINITY }; w2 * h2];
            for j in 0..hgt {
                for i in 0..w {
                    let a = below[j * w + i];
                    let k = (j / 2) * w2 + i / 2;
                    up[k] = ZRange { lo: swift::min(up[k].lo, a.lo), hi: swift::max(up[k].hi, a.hi) };
                }
            }
            levels.push((w2, up));
            w = w2;
            hgt = h2;
        }
        levels
    }

    /// Where a level piece, z = h over its polygon, meets the torus about a
    /// vertical axis through `axis`, its core of radius `ring` at `height`
    /// and its tube of radius `tube`: the circles about the axis of radius
    /// ring ± √(tube² − (h − height)²), clipped to the polygon, each arc cut
    /// into chords whose sagitta is under half of `least`. A plane the torus
    /// reaches past by no more than `tolerance`, or not at all, meets it
    /// nowhere.
    pub fn circles(
        piece: &TopPiece,
        axis: Vec2,
        height: f64,
        ring: f64,
        tube: f64,
        least: f64,
        tolerance: f64,
    ) -> Vec<Segment> {
        let h = piece.constant;
        let dz = h - height;
        if !(tube - dz.abs() > tolerance) {
            return Vec::new();
        }
        let half = (tube * tube - dz * dz).sqrt();
        let polygon = &piece.polygon;
        let mut out = Vec::new();
        for rho in [ring - half, ring + half] {
            if !(rho > 0.0) {
                continue;
            }
            let at = |t: f64| axis + rho * Vec2::new(real::cos(t), real::sin(t));
            // Where the circle crosses each side, solved on the side, and
            // its angle about the axis.
            let mut crossings: Vec<(f64, Vec2)> = Vec::new();
            for k in 0..polygon.len() {
                let a = polygon[k];
                let e = polygon[(k + 1) % polygon.len()] - a;
                let w = a - axis;
                // |w + t e|² = ρ²: t² (e·e) + 2t (w·e) + (w·w − ρ²) = 0.
                let qa = e.dot(e);
                let qb = w.dot(e);
                let qc = (w.length() - rho) * (w.length() + rho);
                if !(qa > 0.0) {
                    continue;
                }
                let disc = qb * qb - qa * qc;
                if !(disc >= 0.0) {
                    continue;
                }
                let root = disc.sqrt();
                // The roots without cancellation.
                let q = -(qb + if qb >= 0.0 { root } else { -root });
                for t in [q / qa, if q != 0.0 { qc / q } else { 0.0 }] {
                    if (0.0..=1.0).contains(&t) {
                        let p = a + t * e;
                        crossings.push((real::atan2(p.y - axis.y, p.x - axis.x), p));
                    }
                }
            }
            crossings.sort_by(|x, y| swift::ascending(x.0, y.0));
            if crossings.is_empty() {
                crossings = vec![(0.0, at(0.0))];
            }
            // The arcs between consecutive crossings whose middle is inside.
            for k in 0..crossings.len() {
                let (t0, p0) = crossings[k];
                let (t1, p1) = if k + 1 < crossings.len() {
                    crossings[k + 1]
                } else {
                    (crossings[0].0 + 2.0 * std::f64::consts::PI, crossings[0].1)
                };
                if !(t1 - t0 > 0.0 && piece.contains(at(0.5 * (t0 + t1)), 0.0)) {
                    continue;
                }
                let step = 2.0 * real::acos(swift::max(1.0 - 0.5 * least / rho, -1.0));
                let count = int(((t1 - t0) / step).ceil()).map_or(1, |n| n.max(1));
                let mut previous = p0;
                for m in 1..=count {
                    let next = if m == count { p1 } else { at(t0 + (t1 - t0) * m as f64 / count as f64) };
                    out.push((Vec3::new(previous.x, previous.y, h), Vec3::new(next.x, next.y, h)));
                    previous = next;
                }
            }
        }
        out
    }

    /// The zero set of a function with Lipschitz constant 1 in space on a
    /// piece's plane over its polygon, as segments. The polygon, clipped to
    /// `bx`, is passed over where the bound shows no zero and split in four
    /// otherwise, down to squares of `side`. Each side of such a square is
    /// searched for every zero, its halves while the bound allows one, down
    /// to `least`, and each zero bisected to the last bit. A closed loop of
    /// the zero set lies round a critical point of f, so a square that holds
    /// none of `critical` holds arcs alone, each from one zero on its sides
    /// to another: with none it holds no zero, and with two it draws the
    /// chord between them where Newton's step from each of its quarters to
    /// the zero set is within half of `least`. A square with more, or a chord
    /// that strays, or a critical point, is split in four, down to `least`,
    /// where the zeros on its sides are joined in pairs in turn. So every arc
    /// is found that crosses a side of a square apart from its neighbours by
    /// `least`, and every loop larger than `least`.
    pub fn zeros(
        f: &dyn Fn(Vec3) -> f64,
        piece: &TopPiece,
        side: f64,
        least: f64,
        bx: (Vec2, Vec2),
        critical: &[Vec2],
    ) -> Vec<Segment> {
        let zeros = Zeros { f, piece, side, least, critical };
        let mut cut = piece.polygon.clone();
        cut = Slab::clip2d(&Slab::clip2d(&cut, &|p| bx.0.x - p.x), &|p| p.x - bx.1.x);
        cut = Slab::clip2d(&Slab::clip2d(&cut, &|p| bx.0.y - p.y), &|p| p.y - bx.1.y);
        if cut.len() >= 3 { zeros.inside(&cut) } else { Vec::new() }
    }
}

/// `Slab.zeros`'s nested functions, which recurse: its arguments held once.
struct Zeros<'a> {
    f: &'a dyn Fn(Vec3) -> f64,
    piece: &'a TopPiece,
    side: f64,
    least: f64,
    critical: &'a [Vec2],
}

impl Zeros<'_> {
    fn value(&self, p: Vec2) -> f64 {
        (self.f)(self.piece.point(p))
    }

    /// Every zero along a → b, in order from a.
    fn along(&self, a: Vec2, fa: f64, b: Vec2, fb: f64) -> Vec<Vec2> {
        let length = (self.piece.point(a) - self.piece.point(b)).length();
        if fa.abs() + fb.abs() > length * (1.0 + 1e-9) {
            return Vec::new();
        }
        if (a - b).length() <= self.least {
            if (fa > 0.0) == (fb > 0.0) {
                return Vec::new();
            }
            let (mut under, mut over) = if fa > 0.0 { (b, a) } else { (a, b) };
            for _ in 0..80 {
                let m = 0.5 * (under + over);
                if m == under || m == over {
                    break;
                }
                if self.value(m) > 0.0 {
                    over = m;
                } else {
                    under = m;
                }
            }
            return vec![under];
        }
        let m = 0.5 * (a + b);
        let fm = self.value(m);
        let mut out = self.along(a, fa, m, fm);
        out.extend(self.along(m, fm, b, fb));
        out
    }

    /// Whether Newton's step from p to the zero set, along f's gradient
    /// in the plane's (x, y), is within half of `least`.
    fn near(&self, p: Vec2) -> bool {
        let v = self.value(p);
        let h = 1e-3 * self.least;
        let g = Vec2::new(
            self.value(p + Vec2::new(h, 0.0)) - self.value(p - Vec2::new(h, 0.0)),
            self.value(p + Vec2::new(0.0, h)) - self.value(p - Vec2::new(0.0, h)),
        ) / (2.0 * h);
        v.abs() <= 0.5 * self.least * g.length()
    }

    fn split(&self, cut: &[Vec2]) -> Vec<Vec<Vec2>> {
        let clo = cut.iter().fold(cut[0], |a, &p| a.min(p));
        let chi = cut.iter().fold(cut[0], |a, &p| a.max(p));
        let mid = 0.5 * (clo + chi);
        [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
            .into_iter()
            .filter_map(|(sx, sy): (f64, f64)| {
                let quarter = Slab::clip2d(&Slab::clip2d(cut, &|p| sx * (mid.x - p.x)), &|p| sy * (mid.y - p.y));
                (quarter.len() >= 3).then_some(quarter)
            })
            .collect()
    }

    fn inside(&self, cut: &[Vec2]) -> Vec<Segment> {
        let piece = self.piece;
        let points: Vec<Vec3> = cut.iter().map(|&p| piece.point(p)).collect();
        let middle = cut.iter().fold(Vec2::ZERO, |a, &p| a + p) / cut.len() as f64;
        let reach = points.iter().map(|&p| (p - piece.point(middle)).length()).reduce(swift::max).expect("corners");
        if !(self.value(middle).abs() <= reach * (1.0 + 1e-9)) {
            return Vec::new();
        }
        let clo = cut.iter().fold(cut[0], |a, &p| a.min(p));
        let chi = cut.iter().fold(cut[0], |a, &p| a.max(p));
        let size = swift::max(chi.x - clo.x, chi.y - clo.y);
        if size > self.side * (1.0 + 1e-9) {
            return self.split(cut).iter().flat_map(|q| self.inside(q)).collect();
        }
        let values: Vec<f64> = cut.iter().map(|&p| self.value(p)).collect();
        let mut crossings: Vec<Vec2> = Vec::new();
        for k in 0..cut.len() {
            let l = (k + 1) % cut.len();
            crossings.extend(self.along(cut[k], values[k], cut[l], values[l]));
        }
        let least = self.least;
        let loops = self.critical.iter().any(|&c| {
            let lo = clo - Vec2::splat(least);
            let hi = chi + Vec2::splat(least);
            // simd_min(chi + least, simd_max(c, clo - least)) == c
            Vec2::new(swift::min(hi.x, swift::max(c.x, lo.x)), swift::min(hi.y, swift::max(c.y, lo.y))) == c
        });
        if !loops && crossings.is_empty() {
            return Vec::new();
        }
        if crossings.len() == 2 && (!loops || size <= least * 2.0) {
            let (a, b) = (crossings[0], crossings[1]);
            if size <= least * 2.0 || [0.25, 0.5, 0.75].iter().all(|&t| self.near(a + t * (b - a))) {
                return vec![(piece.point(a), piece.point(b))];
            }
        }
        if size <= least * 2.0 {
            return (0..crossings.len().saturating_sub(1))
                .step_by(2)
                .map(|k| (piece.point(crossings[k]), piece.point(crossings[k + 1])))
                .collect();
        }
        self.split(cut).iter().flat_map(|q| self.inside(q)).collect()
    }
}
