//! A landscape's solid on its grid, as another geometry's curves meet it in
//! a composite: the port of `Sources/KurvenCore/Slab.swift`.
//!
//! The slab under the drawn surface, down to its base, as convex prisms, and
//! the spans of a segment whose lines of sight pass through one, in closed
//! form.
//!
//! Over each of the grid's triangles, in each copy, the solid is the points
//! over the triangle, above the base, under the triangle's plane and under
//! the truncation: an intersection of half-spaces, so convex, split where a
//! band boundary crosses the triangle and, under a footprint, by the
//! footprint's convex pieces. The lines of sight from a segment `a -> b`
//! sweep the strip a + s (b − a) + t d, 0 ≤ s ≤ 1, t ≥ 0, and each prism's
//! half-spaces are half-planes of (s, t): clipped by them, the strip leaves a
//! convex polygon, and where it has area the lines through its span of s pass
//! through the prism's interior. A line that only grazes a prism leaves a
//! polygon of no area and hides nothing, and a point on the slab's boundary
//! -- a wall, the floor, the top -- is hidden only where its line goes in:
//! the polygon starts a billionth of a cell along the line. The prisms a
//! strip can meet are found by walking its footprint over the grid, row by
//! row.
//!
//! The prisms are the surface the traversal reads on the grid
//! (`Surface.drawnSurface(at:step:)`): each cell split along the diagonal
//! from its (1, 0) corner to its (0, 1) corner, the raw samples interpolated
//! and cut at the truncation, over the copies' boxes and the footprint.

use kurven_math::swift;

use crate::composite::CompositeError;
use crate::convex_partition;
use crate::manifest::{Region, Walls};
use crate::scene::Heightfield;
use crate::simd::{Vec2, Vec3};
use crate::spaces::P3;

/// `simd_double2x2`, by rows: `[r0; r1]`, applied to a column vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat2 {
    pub r0: Vec2,
    pub r1: Vec2,
}

impl Mat2 {
    pub fn from_rows(r0: Vec2, r1: Vec2) -> Mat2 {
        Mat2 { r0, r1 }
    }

    /// The product with a vector, column by column as simd sums it.
    pub fn apply(self, v: Vec2) -> Vec2 {
        Vec2::new(self.r0.x * v.x + self.r0.y * v.y, self.r1.x * v.x + self.r1.y * v.y)
    }

    pub fn times(self, o: Mat2) -> Mat2 {
        let c0 = self.apply(Vec2::new(o.r0.x, o.r1.x));
        let c1 = self.apply(Vec2::new(o.r0.y, o.r1.y));
        Mat2::from_rows(Vec2::new(c0.x, c1.x), Vec2::new(c0.y, c1.y))
    }

    pub fn transpose(self) -> Mat2 {
        Mat2::from_rows(Vec2::new(self.r0.x, self.r1.x), Vec2::new(self.r0.y, self.r1.y))
    }

    pub fn determinant(self) -> f64 {
        self.r0.x * self.r1.y - self.r0.y * self.r1.x
    }

    /// `simd_inverse`: the adjugate over the determinant.
    pub fn inverse(self) -> Mat2 {
        let det = self.determinant();
        Mat2::from_rows(Vec2::new(self.r1.y / det, -self.r0.y / det), Vec2::new(-self.r1.x / det, self.r0.x / det))
    }
}

/// A half-space `n · x ≤ c`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HalfSpace {
    pub n: Vec3,
    pub c: f64,
}

fn hs(n: Vec3, c: f64) -> HalfSpace {
    HalfSpace { n, c }
}

/// One convex piece of the footprint in a copy's index coordinates: its
/// half-planes, and its box.
#[derive(Debug, Clone)]
pub struct FootPiece {
    pub half_planes: Vec<HalfSpace>,
    pub lo: Vec2,
    pub hi: Vec2,
}

/// One copy of the grid: world (x, y) to the decimated grid's index
/// coordinates (u, v), and the footprint's convex pieces there.
#[derive(Debug, Clone)]
pub struct SlabCopy {
    pub linear: Mat2,
    pub offset: Vec2,
    pub pieces: Vec<FootPiece>,
    /// The copy's box corners in world space, for how far a line can run
    /// in it.
    pub corners: Vec<Vec3>,
}

/// A truncation band as a stretch of u, with its height.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlabBand {
    pub lo: f64,
    pub hi: f64,
    pub height: f64,
}

#[derive(Debug, Clone)]
pub struct Slab {
    pub copies: Vec<SlabCopy>,
    pub nx: usize,
    pub ny: usize,
    /// The decimated grid's samples, untruncated, row by row.
    pub heights: Vec<f64>,
    pub base: f64,
    pub top: f64,
    /// The truncation's bands as stretches of u, with their heights.
    pub bands: Vec<SlabBand>,
    /// The decimated grid's cell, the larger of its sides in the
    /// landscape's own units.
    pub cell: f64,
    /// Where a line starts to count: a billionth of a cell along it.
    pub start: f64,
}

/// Swift's `Int(x.rounded(.down))` and its kin where a non-finite value
/// would trap: here `None`.
fn floor_int(x: f64) -> Option<i64> {
    swift::int(x.floor())
}

impl Slab {
    pub fn new(h: &Heightfield, name: &str) -> Result<Slab, CompositeError> {
        let s = &h.surface;
        let g = &*s.height;
        let step = h.step.max(1);
        let nx = g.width.div_ceil(step);
        let ny = g.height.div_ceil(step);
        let mut heights = vec![0.0; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                heights[j * nx + i] = f64::from(g.at(i * step, j * step));
            }
        }
        let (mut top, mut floor) = (f64::NEG_INFINITY, f64::INFINITY);
        h.for_each_sample(|p| {
            top = swift::max(top, p.z());
            floor = swift::min(floor, p.z());
        });
        for v in &h.occluder.vertices {
            floor = swift::min(floor, v.z());
        }
        let base = if let Walls::Perimeter { base, .. } = &h.walls { *base } else { floor };
        let d = g.domain;
        let su = (g.width as f64 - 1.0) / (d.real.length() * step as f64);
        let sv = (g.height as f64 - 1.0) / (d.imag.length() * step as f64);
        let cell = swift::max((1.0 / su).abs(), (1.0 / sv).abs());
        let start = 1e-9 * cell;
        let u = |x: f64| (x - d.real.lo) * su;
        let spans = s.truncation.bands(swift::min(d.real.lo, d.real.hi), swift::max(d.real.lo, d.real.hi));
        let bands = if spans.is_empty() {
            vec![SlabBand { lo: f64::NEG_INFINITY, hi: f64::INFINITY, height: f64::INFINITY }]
        } else {
            // In order along u, the first and last open to either side.
            let mut b: Vec<SlabBand> = spans
                .iter()
                .map(|span| SlabBand {
                    lo: swift::min(u(span.lo), u(span.hi)),
                    hi: swift::max(u(span.lo), u(span.hi)),
                    height: span.height,
                })
                .collect();
            b.sort_by(|x, y| swift::ascending(x.lo, y.lo));
            b[0].lo = f64::NEG_INFINITY;
            let last = b.len() - 1;
            b[last].hi = f64::INFINITY;
            b
        };
        // The footprint's convex pieces in world (x, y).
        let mut footprint: Option<Vec<Vec<Vec2>>> = None;
        if let Region::Inside(perimeter) = &h.region {
            let mut corners: Vec<Vec2> = perimeter.edges().iter().map(|e| e.start.v).collect();
            let mut area = 0.0;
            for k in 0..corners.len() {
                let (p, q) = (corners[k], corners[(k + 1) % corners.len()]);
                area += p.x * q.y - q.x * p.y;
            }
            if area < 0.0 {
                corners.reverse();
            }
            let indices: Vec<usize> = (0..corners.len()).collect();
            let pieces = convex_partition::pieces(&[indices], &|k| corners[k], 1e-12 * area.abs(), 0)
                .map_err(|e| CompositeError(format!("{name}: its footprint is not one simple polygon ({e})")))?;
            footprint = Some(pieces.iter().map(|p| p.iter().map(|&k| corners[k]).collect()).collect());
        }
        let domain_corners = [
            Vec2::new(d.real.lo, d.imag.lo),
            Vec2::new(d.real.hi, d.imag.lo),
            Vec2::new(d.real.hi, d.imag.hi),
            Vec2::new(d.real.lo, d.imag.hi),
        ];
        let mut copies = Vec::new();
        for tile in &h.tiles {
            let Some(back) = tile.inverse() else {
                return Err(CompositeError(format!("{name}: a tile has no inverse")));
            };
            let scale = Mat2::from_rows(Vec2::new(su, 0.0), Vec2::new(0.0, sv));
            let linear = scale.times(Mat2::from_rows(Vec2::new(back.a, back.b), Vec2::new(back.c, back.d)));
            let offset = scale.apply(Vec2::new(back.tx, back.ty) - Vec2::new(d.real.lo, d.imag.lo));
            // A world half-plane n · w ≤ c in (u, v): w = L⁻¹ (uv − o).
            let inverse = linear.inverse();
            let pieces = footprint
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .map(|polygon| {
                    let mut planes = Vec::new();
                    for k in 0..polygon.len() {
                        let (p, q) = (polygon[k], polygon[(k + 1) % polygon.len()]);
                        let n = Vec2::new(q.y - p.y, p.x - q.x); // outward of a counter-clockwise piece
                        let m = inverse.transpose().apply(n);
                        planes.push(hs(Vec3::new(m.x, m.y, 0.0), n.dot(p) + m.dot(offset)));
                    }
                    let mapped: Vec<Vec2> = polygon.iter().map(|&p| linear.apply(p) + offset).collect();
                    let lo = mapped.iter().fold(mapped[0], |a, &p| a.min(p));
                    let hi = mapped.iter().fold(mapped[0], |a, &p| a.max(p));
                    FootPiece { half_planes: planes, lo, hi }
                })
                .collect();
            let corners = domain_corners
                .iter()
                .flat_map(|c| {
                    let w = tile.apply(P3::new(c.x, c.y, 0.0));
                    [Vec3::new(w.x(), w.y(), base), Vec3::new(w.x(), w.y(), top)]
                })
                .collect();
            copies.push(SlabCopy { linear, offset, pieces, corners });
        }
        Ok(Slab { copies, nx, ny, heights, base, top, bands, cell, start })
    }

    /// A point or vector of the landscape's coordinates in a copy's index
    /// coordinates, its height kept; moved by the offset when a point.
    pub fn flat(copy: &SlabCopy, p: Vec3, moved: bool) -> Vec3 {
        let q = copy.linear.apply(Vec2::new(p.x, p.y)) + if moved { copy.offset } else { Vec2::ZERO };
        Vec3::new(q.x, q.y, p.z)
    }

    /// The spans of the segment `a -> b`, in the landscape's own
    /// coordinates, whose lines of sight along the unit `d` pass through
    /// the slab's interior.
    pub fn hidden(&self, a: Vec3, b: Vec3, d: Vec3) -> Vec<(f64, f64)> {
        let mut out = Vec::new();
        let e = b - a;
        for copy in &self.copies {
            let big_a = Slab::flat(copy, a, true);
            let big_e = Slab::flat(copy, e, false);
            let big_d = Slab::flat(copy, d, false);
            let reach =
                copy.corners.iter().map(|&c| (c - a).length()).reduce(swift::max).expect("corners") + e.length();
            let (t0, t1) = (self.start, 1.01 * reach + 1.0);
            let bx = [
                hs(Vec3::new(-1.0, 0.0, 0.0), 0.0),
                hs(Vec3::new(1.0, 0.0, 0.0), (self.nx - 1) as f64),
                hs(Vec3::new(0.0, -1.0, 0.0), 0.0),
                hs(Vec3::new(0.0, 1.0, 0.0), (self.ny - 1) as f64),
                hs(Vec3::new(0.0, 0.0, -1.0), -self.base),
                hs(Vec3::new(0.0, 0.0, 1.0), self.top),
            ];
            let swept = Slab::clip(&bx, big_a, big_e, big_d, t0, t1);
            if swept.is_empty() {
                continue;
            }
            if let Some(span) = self.floor_span(big_a, big_e, big_d, copy) {
                out.push(span);
            }
            // The strip's footprint on the grid, walked row by row.
            let foot: Vec<Vec2> = swept
                .iter()
                .map(|q| Vec2::new(big_a.x + q.x * big_e.x + q.y * big_d.x, big_a.y + q.x * big_e.y + q.y * big_d.y))
                .collect();
            let vmin = foot.iter().map(|p| p.y).reduce(swift::min).expect("foot");
            let vmax = foot.iter().map(|p| p.y).reduce(swift::max).expect("foot");
            // Where Swift's `Int` traps, on a footprint that is not finite,
            // the copy is passed over.
            let (Some(vlo), Some(vhi)) = (floor_int(vmin), swift::int(vmax.ceil())) else { continue };
            let j0 = vlo.max(0);
            let j1 = (vhi - 1).max(j0).min(self.ny as i64 - 2);
            if j0 > j1 {
                continue;
            }
            for j in j0..=j1 {
                let row = Slab::clip2d_axis(&foot, 1, j as f64, (j + 1) as f64);
                if row.is_empty() {
                    continue;
                }
                let umin = row.iter().map(|p| p.x).reduce(swift::min).expect("row");
                let umax = row.iter().map(|p| p.x).reduce(swift::max).expect("row");
                let (Some(ulo), Some(uhi)) = (floor_int(umin), swift::int(umax.ceil())) else { continue };
                let i0 = ulo.max(0);
                let i1 = (uhi - 1).max(i0).min(self.nx as i64 - 2);
                if i0 > i1 {
                    continue;
                }
                for i in i0..=i1 {
                    for upper in [false, true] {
                        for half_spaces in self.prisms(i as usize, j as usize, upper, copy) {
                            let polygon = Slab::clip(&half_spaces, big_a, big_e, big_d, t0, t1);
                            if let Some(span) = Slab::span(&polygon) {
                                out.push(span);
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// The span of a segment whose lines of sight cross the floor, the
    /// slab's face at the base, over the copy's box and footprint: a sheet
    /// of the boundary, opaque where the slab over it is as thin as a
    /// rounding -- where the surface comes down to the base -- as much as
    /// where it is thick. The line from a + s e crosses the base at a point
    /// affine in s, so the span is where that point is in each of the
    /// floor's convex pieces, and the crossing is past the line's start.
    pub fn floor_span(&self, a: Vec3, e: Vec3, d: Vec3, copy: &SlabCopy) -> Option<(f64, f64)> {
        if !(d.z.abs() > 1e-12) {
            return None;
        }
        // t(s) = t0 + s t1 where the line meets the base; the point there
        // is p0 + s p1 in (u, v).
        let tb0 = (self.base - a.z) / d.z;
        let tb1 = -e.z / d.z;
        let p0 = Vec2::new(a.x + tb0 * d.x, a.y + tb0 * d.y);
        let p1 = Vec2::new(e.x + tb1 * d.x, e.y + tb1 * d.y);
        let bx = vec![
            hs(Vec3::new(-1.0, 0.0, 0.0), 0.0),
            hs(Vec3::new(1.0, 0.0, 0.0), (self.nx - 1) as f64),
            hs(Vec3::new(0.0, -1.0, 0.0), 0.0),
            hs(Vec3::new(0.0, 1.0, 0.0), (self.ny - 1) as f64),
        ];
        let sets: Vec<Vec<HalfSpace>> = if copy.pieces.is_empty() {
            vec![bx]
        } else {
            copy.pieces.iter().map(|p| bx.iter().chain(&p.half_planes).copied().collect()).collect()
        };
        let mut best: Option<(f64, f64)> = None;
        for planes in sets {
            // Each half-plane n · p ≤ c, and t(s) ≥ start, as k0 + s k1 ≤ 0.
            let (mut lo, mut hi) = (0.0, 1.0);
            let ks = planes
                .iter()
                .map(|h| {
                    let n = Vec2::new(h.n.x, h.n.y);
                    (n.dot(p0) - h.c, n.dot(p1))
                })
                .chain(std::iter::once((self.start - tb0, -tb1)));
            for (k0, k1) in ks {
                if k1.abs() < 1e-300 {
                    if k0 > 0.0 {
                        lo = 1.0;
                        hi = 0.0;
                    }
                    continue;
                }
                let root = -k0 / k1;
                if k1 > 0.0 {
                    hi = swift::min(hi, root);
                } else {
                    lo = swift::max(lo, root);
                }
            }
            if hi > lo {
                best = Some(best.map_or((lo, hi), |b| (swift::min(b.0, lo), swift::max(b.1, hi))));
            }
        }
        best
    }

    /// The convex prisms over one triangle of cell (i, j), in index
    /// coordinates: the lower triangle has the cell's (0, 0) corner, the
    /// upper its (1, 1) corner.
    pub fn prisms(&self, i: usize, j: usize, upper: bool, copy: &SlabCopy) -> Vec<Vec<HalfSpace>> {
        let nx = self.nx;
        let h00 = self.heights[j * nx + i];
        let h10 = self.heights[j * nx + i + 1];
        let h01 = self.heights[(j + 1) * nx + i];
        let h11 = self.heights[(j + 1) * nx + i + 1];
        let (fi, fj) = (i as f64, j as f64);
        let mut common = vec![hs(Vec3::new(0.0, 0.0, -1.0), -self.base)];
        if upper {
            common.extend([
                hs(Vec3::new(1.0, 0.0, 0.0), fi + 1.0),
                hs(Vec3::new(0.0, 1.0, 0.0), fj + 1.0),
                hs(Vec3::new(-1.0, -1.0, 0.0), -(fi + fj + 1.0)),
            ]);
            // z ≤ h11 + (i + 1 − u)(h01 − h11) + (j + 1 − v)(h10 − h11)
            let gu = -(h01 - h11);
            let gv = -(h10 - h11);
            if [h11, h01, h10].iter().all(|x| x.is_finite()) {
                common.push(hs(Vec3::new(-gu, -gv, 1.0), h11 + (fi + 1.0) * (h01 - h11) + (fj + 1.0) * (h10 - h11)));
            }
        } else {
            common.extend([
                hs(Vec3::new(-1.0, 0.0, 0.0), -fi),
                hs(Vec3::new(0.0, -1.0, 0.0), -fj),
                hs(Vec3::new(1.0, 1.0, 0.0), fi + fj + 1.0),
            ]);
            // z ≤ h00 + (u − i)(h10 − h00) + (v − j)(h01 − h00)
            let gu = h10 - h00;
            let gv = h01 - h00;
            if [h00, h10, h01].iter().all(|x| x.is_finite()) {
                common.push(hs(Vec3::new(-gu, -gv, 1.0), h00 - fi * gu - fj * gv));
            }
        }
        let mut out = Vec::new();
        for band in &self.bands {
            if !(band.lo < fi + 1.0 && band.hi > fi) {
                continue;
            }
            let mut planes = common.clone();
            if band.lo.is_finite() {
                planes.push(hs(Vec3::new(-1.0, 0.0, 0.0), -band.lo));
            }
            if band.hi.is_finite() {
                planes.push(hs(Vec3::new(1.0, 0.0, 0.0), band.hi));
            }
            if band.height.is_finite() {
                planes.push(hs(Vec3::new(0.0, 0.0, 1.0), band.height));
            }
            if copy.pieces.is_empty() {
                out.push(planes);
            } else {
                for piece in &copy.pieces {
                    if piece.lo.x <= fi + 1.0 && piece.hi.x >= fi && piece.lo.y <= fj + 1.0 && piece.hi.y >= fj {
                        out.push(planes.iter().chain(&piece.half_planes).copied().collect());
                    }
                }
            }
        }
        out
    }

    /// The part of the strip a + s e + t d, 0 ≤ s ≤ 1, t0 ≤ t ≤ t1, inside
    /// every half-space n · x ≤ c: a convex polygon in (s, t), empty where
    /// none is left.
    pub fn clip(half_spaces: &[HalfSpace], a: Vec3, e: Vec3, d: Vec3, t0: f64, t1: f64) -> Vec<Vec2> {
        let mut polygon = vec![Vec2::new(0.0, t0), Vec2::new(1.0, t0), Vec2::new(1.0, t1), Vec2::new(0.0, t1)];
        for h in half_spaces {
            let k = h.c - h.n.dot(a);
            let ks = h.n.dot(e);
            let kt = h.n.dot(d);
            polygon = Slab::clip2d(&polygon, &|q| ks * q.x + kt * q.y - k);
            if polygon.len() < 3 {
                return Vec::new();
            }
        }
        polygon
    }

    /// A polygon clipped to where the affine `f` is not positive.
    pub fn clip2d(polygon: &[Vec2], f: &dyn Fn(Vec2) -> f64) -> Vec<Vec2> {
        let mut out = Vec::with_capacity(polygon.len() + 1);
        for k in 0..polygon.len() {
            let (p, q) = (polygon[k], polygon[(k + 1) % polygon.len()]);
            let (fp, fq) = (f(p), f(q));
            if fp <= 0.0 {
                out.push(p);
            }
            if (fp <= 0.0) != (fq <= 0.0) {
                out.push(p + (q - p) * (fp / (fp - fq)));
            }
        }
        out
    }

    /// A polygon clipped to `lo ≤ x[axis] ≤ hi`.
    pub fn clip2d_axis(polygon: &[Vec2], axis: usize, lo: f64, hi: f64) -> Vec<Vec2> {
        Slab::clip2d(&Slab::clip2d(polygon, &|p| lo - p[axis]), &|p| p[axis] - hi)
    }

    /// The span of s a clipped strip covers, where it has area: a polygon
    /// whose area is a rounding of its box's is a line that grazes.
    pub fn span(polygon: &[Vec2]) -> Option<(f64, f64)> {
        if polygon.len() < 3 {
            return None;
        }
        let (mut lo, mut hi, mut area) = (polygon[0], polygon[0], 0.0);
        for k in 0..polygon.len() {
            let (p, q) = (polygon[k], polygon[(k + 1) % polygon.len()]);
            lo = lo.min(p);
            hi = hi.max(p);
            area += p.x * q.y - q.x * p.y;
        }
        let extent = hi - lo;
        if !(extent.x > 0.0 && extent.y > 0.0 && area.abs() > 2e-10 * extent.x * extent.y) {
            return None;
        }
        Some((swift::max(lo.x, 0.0), swift::min(hi.x, 1.0)))
    }
}
