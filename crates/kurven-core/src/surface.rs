//! The landscape's surface: |f| and arg f on their grids, the truncation, and
//! what is derived from them. The port of `Sources/KurvenCore/Surface.swift`.
//!
//! [`ContourRefine`] is the one type the refiner (`kurven-landscape`'s
//! `ContourRefiner`) and the surface share. In Swift `KurvenCore` has the
//! grids and not the function, and `KurvenLandscape` builds a `ContourRefine`
//! of closures; the same split holds here, so there is no edge from
//! `kurven-core` to `kurven-landscape`, and no crate cycle.
//!
//! [`Surface`] holds its grids behind `Arc`: Swift's arrays are shared until
//! written, so a face's closures capture the surface by value for nothing;
//! here a clone of the surface shares its samples the same way.

use std::ops::RangeInclusive;
use std::sync::Arc;

use kurven_math::swift;

use crate::grid::Grid2D;
use crate::hatching::FaceSelector;
use crate::manifest::{
    Affine2, ContourField, Domain, DomainAxis, HeightPolicy, LayerSource, Region, Restriction, Side, Truncation,
};
use crate::polyline::PolylineSet;
use crate::scalar_field::{self, Contoured, ParamPaths, ScalarField, Trim};
use crate::simd::{Vec2, Vec3};
use crate::spaces::{DomainSpace, P2, P3, ParamSpace, WorldSpace};

/// Moves the paths of one level of one field onto the function's own level
/// set (`ContourRefine.Contours`). See [`ContourRefine::contours`].
pub type Contours = Arc<
    dyn Fn(ContourField, f64, &[Vec<P2<DomainSpace>>], f64, RangeInclusive<f64>) -> Vec<Vec<P2<DomainSpace>>>
        + Send
        + Sync,
>;
/// The rings at a level of |f| round the zeros the grid could not see
/// (`ContourRefine.Rings`). See [`ContourRefine::rings`].
pub type Rings = Arc<dyn Fn(f64, &[Vec<P2<DomainSpace>>]) -> Vec<Vec<P2<DomainSpace>>> + Send + Sync>;
/// |f| at a domain point, untruncated, as the grid would hold it.
pub type Magnitude = Arc<dyn Fn(P2<DomainSpace>) -> f64 + Send + Sync>;
/// arg f at a domain point, in (−π, π].
pub type Phase = Arc<dyn Fn(P2<DomainSpace>) -> f64 + Send + Sync>;
/// A zero of f within a few cells of a domain point, or none.
pub type Zero = Arc<dyn Fn(P2<DomainSpace>) -> Option<P2<DomainSpace>> + Send + Sync>;
/// The same refiner with nothing shared (`ContourRefine.fork`).
pub type Fork = Arc<dyn Fn() -> ContourRefine + Send + Sync>;

/// What a consumer that has the function supplies, so the curves can be
/// placed by f rather than by the grid. Core has the grids and not the
/// function; `kurven-landscape`'s `ContourRefiner` builds one of these.
#[derive(Clone)]
pub struct ContourRefine {
    /// Moves the paths of one level of one field onto the function's own
    /// level set, given the field, the level, the marching-squares paths in
    /// domain space, and the ceiling: the height above which the curves are
    /// cut away (`Restriction.belowTruncation`), so a stretch of a phase line
    /// wholly above it need not be placed. Infinity places all of it. It is
    /// how |f| and arg f place their contours as scalar fields:
    /// `Surface.field(of:refine:ceiling:)` builds each a `place` that calls
    /// this with the tag and the ceiling closed over. It is also how the
    /// truncation edge is placed, and the last argument is for that: the span
    /// of Re z the curves are kept to, a band's for a run of a banded
    /// truncation edge, otherwise [`ContourRefine::EVERYWHERE`]. It places the
    /// paths it is given and nothing more: the rings the grid could not see
    /// are `rings`'.
    pub contours: Contours,
    /// The rings at a level of |f| round the zeros of f that the grid could
    /// not see, given the curves already placed at that level: a zero one of
    /// them turns round has its curve. Asked once a level, so a level placed
    /// in parts, as a banded truncation edge is, has each ring once. None for
    /// a consumer with no zeros to seed rings from. Given at every
    /// construction, so a wrapper that hands on `contours` decides about
    /// these too.
    pub rings: Option<Rings>,
    /// With it, the top edge of a cut face gains a vertex at each minimum of
    /// |f| between two grid nodes (`Surface.topEdge`), and a contour that ends
    /// on a zero of f ends at the zero's height (`Surface.derive`). None for a
    /// consumer that can place contours and not heights.
    pub magnitude: Option<Magnitude>,
    /// arg f at a domain point, in (−π, π], and not a number where f is zero
    /// or not finite: where seeds of integral curves are placed at listed
    /// angles of arg f, and the slope of arg f read (`Seeds`,
    /// `Surface.integralCurves(of:seeds:on:in:refine:)`). None for a consumer
    /// with no function.
    pub phase: Option<Phase>,
    /// With it, a fold curve that ends beside a zero of f is carried to the
    /// zero (`Heightfield.foldCurves`), as the refiner carries the phase lines.
    pub zero: Option<Zero>,
    /// How far a chord may depart from the curve it stands for, in cells: the
    /// refiner's own tolerance, which the top edge of a cut face and the fold
    /// curves are subdivided to as well, so every curve of the drawing is a
    /// curve to the same tolerance.
    pub tolerance: f64,
    /// The same refiner with nothing shared: f compiled again, into closures
    /// of its own. In Swift each thread of a parallel visibility test takes
    /// one, because closures' shared reference counts were the cores writing
    /// one cache line in turn. The port is single-threaded, so it only keeps
    /// the shape; None, a refiner that cannot be forked, is shared.
    pub fork: Option<Fork>,
}

impl ContourRefine {
    /// The span of Re z that keeps a curve nowhere (`ContourRefine.everywhere`).
    pub const EVERYWHERE: RangeInclusive<f64> = f64::NEG_INFINITY..=f64::INFINITY;

    /// Swift's `init(contours:rings:magnitude:phase:zero:tolerance:fork:)`
    /// with its defaults: no magnitude, phase, zero or fork, and a tolerance
    /// of 0.02 cells.
    pub fn new(contours: Contours, rings: Option<Rings>) -> ContourRefine {
        ContourRefine { contours, rings, magnitude: None, phase: None, zero: None, tolerance: 0.02, fork: None }
    }

    /// A copy for one thread: `fork`'s, or this one.
    pub fn forked(&self) -> ContourRefine {
        match &self.fork {
            Some(fork) => fork(),
            None => self.clone(),
        }
    }
}

impl std::fmt::Debug for ContourRefine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContourRefine").field("tolerance", &self.tolerance).finish_non_exhaustive()
    }
}

// MARK: - the surface

/// The landscape: |f| on a grid, plus how it is truncated.
///
/// `height` is unclamped, and the truncation is a separate value. That separation is
/// what lets a consumer change the truncation without resampling -- the truncation is a
/// drawing decision about how much of a pole to show, and the samples do not
/// depend on it.
#[derive(Debug, Clone)]
pub struct Surface {
    pub height: Arc<Grid2D<f32>>,
    pub phase: Option<Arc<Grid2D<f32>>>,
    pub truncation: Truncation,
    /// True when the samples came from a cache rather than an evaluator, in
    /// which case the Python side looks heights up by nearest pixel and this
    /// side must too, or the walls sit at a different top edge than the drawing's.
    pub cached: bool,
}

/// `Surface.drawnSurface(at:step:)`'s three numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawnSurface {
    /// The grid's interpolated height.
    pub interpolated: f64,
    /// The truncation over the point.
    pub truncation_height: f64,
    /// The cosine of the interpolating triangle's tilt.
    pub cosine: f64,
}

/// Where two tiles' copies of the domain meet edge to edge: the stretch
/// of space the two images share along a side of each, with the copies
/// on either side of it. Each pair of invertible tiles meets along one
/// at most, their images being convex and apart; tiles whose images
/// overlap, or touch at a corner, meet along none.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileBoundary {
    /// The two tiles, by their place in the list, the lesser first.
    pub first: usize,
    pub second: usize,
    /// Where the stretch begins and ends: from its end with the lesser x,
    /// or y where they agree, so the tile boundary is the same whichever
    /// tile is listed first.
    pub start: P2<WorldSpace>,
    pub end: P2<WorldSpace>,
}

impl TileBoundary {
    pub fn length(&self) -> f64 {
        (self.end.v - self.start.v).length()
    }

    /// The point `s` along the tile boundary from its start, keeping a
    /// tile boundary's x or y where it runs along an axis.
    pub fn point(&self, s: f64) -> P2<WorldSpace> {
        let t = s / self.length();
        let (start, end) = (self.start, self.end);
        P2::new(
            if start.x() == end.x() { start.x() } else { start.x() + (end.x() - start.x()) * t },
            if start.y() == end.y() { start.y() } else { start.y() + (end.y() - start.y()) * t },
        )
    }

    /// The unit normal to the left of the tile boundary's direction.
    pub fn normal(&self) -> Vec2 {
        let d = (self.end.v - self.start.v) / self.length();
        Vec2::new(-d.y, d.x)
    }
}

/// A point of the domain read as a point of the world plane, for a tile.
fn world(p: P2<DomainSpace>) -> P3<WorldSpace> {
    P3::new(p.x(), p.y(), 0.0)
}

impl Surface {
    pub fn new(height: Grid2D<f32>, phase: Option<Grid2D<f32>>, truncation: Truncation, cached: bool) -> Surface {
        Surface { height: Arc::new(height), phase: phase.map(Arc::new), truncation, cached }
    }

    pub fn domain(&self) -> Domain {
        self.height.domain
    }

    /// The clamped heightfield -- the surface the occluder is meshed from.
    pub fn clamped(&self) -> Grid2D<f32> {
        let height = &*self.height;
        match &self.truncation {
            Truncation::None => height.clone(),
            Truncation::Uniform(z) => {
                let truncation_height = *z as f32;
                Grid2D::new(
                    height.width,
                    height.height,
                    height.domain,
                    height.values.iter().map(|&v| if truncation_height < v { truncation_height } else { v }).collect(),
                )
            }
            Truncation::RealBands { .. } => {
                let mut out = height.values.clone();
                for x in 0..height.width {
                    let truncation_height = self.truncation_height_at(height.position(x, 0).x()) as f32;
                    for y in 0..height.height {
                        let v = out[y * height.width + x];
                        out[y * height.width + x] = if truncation_height < v { truncation_height } else { v };
                    }
                }
                Grid2D::new(height.width, height.height, height.domain, out)
            }
        }
    }

    /// |f| at an arbitrary domain point, unclamped.
    pub fn magnitude_at(&self, p: P2<DomainSpace>) -> f64 {
        if self.cached { self.height.nearest(p) } else { self.height.sample(p) }
    }

    /// The height a wall rises to: `magnitude`, truncated. `Surface.height_at`.
    pub fn height_at(&self, p: P2<DomainSpace>) -> f64 {
        swift::min(self.magnitude_at(p), self.truncation_height_at(p.x()))
    }

    /// The truncation height over Re z = x on the landscape, read from the
    /// right (`truncationHeight(atX:)`). See [`Surface::truncation_height_at_from`].
    pub fn truncation_height_at(&self, x: f64) -> f64 {
        self.truncation_height_at_from(x, Side::Right)
    }

    /// The truncation height over Re z = x on the landscape: the band x lies
    /// in (`Truncation::height_at`), but on the domain's right edge the
    /// one-sided limit from the left, from inside the domain. A band that
    /// starts on the right edge lies beyond the domain, and nothing of the
    /// landscape stands in it: the graph, the plateaus, the depth pass and
    /// the right wall end with the band before it, and there is no riser
    /// there.
    ///
    /// Elsewhere the limit is taken from `side`, the right unless a wall
    /// reads it: a wall on a band boundary reads the band on its
    /// footprint's side (`Surface::footprint_side`).
    ///
    /// The right edge is read at the domain's own end, not at the grid's
    /// last column, which its spacing can put a rounding past it.
    pub fn truncation_height_at_from(&self, x: f64, side: Side) -> f64 {
        if !matches!(self.truncation, Truncation::RealBands { .. }) {
            return self.truncation.height_at(x);
        }
        if self.is_on_right_edge(x) {
            self.truncation.height_at_from(self.right_edge(), Side::Left)
        } else {
            self.truncation.height_at_from(x, side)
        }
    }

    /// The band over Re z = x on the landscape, read from the right.
    pub fn band_at(&self, x: f64) -> Option<usize> {
        self.band_at_from(x, Side::Right)
    }

    /// The band over Re z = x on the landscape, as
    /// `truncation_height_at_from` reads it: on the domain's right edge, the
    /// band before it.
    pub fn band_at_from(&self, x: f64, side: Side) -> Option<usize> {
        if !matches!(self.truncation, Truncation::RealBands { .. }) {
            return self.truncation.band_at(x);
        }
        if self.is_on_right_edge(x) {
            self.truncation.band_at_from(self.right_edge(), Side::Left)
        } else {
            self.truncation.band_at_from(x, side)
        }
    }

    /// Re z of the domain's right edge, its greater end.
    pub fn right_edge(&self) -> f64 {
        let real = self.domain().real;
        swift::max(real.lo, real.hi)
    }

    /// Whether Re z = x is the domain's right edge: its greater end, or the
    /// grid's last column there, which its spacing can put a rounding off it.
    pub fn is_on_right_edge(&self, x: f64) -> bool {
        let real = self.domain().real;
        if !(real.lo <= real.hi) {
            return x == real.lo;
        }
        x == real.hi || x == self.height.position(self.height.width - 1, 0).x()
    }

    /// Every `step`-th sample, truncated and lifted, without materializing a grid.
    ///
    /// The bounds pass folds over three million points on elliptic and six
    /// million on zeta; it wants none of them kept. `clamped().decimated()`
    /// would allocate both intermediates to hand them over one at a time.
    pub fn for_each_sample(&self, step: usize, mut body: impl FnMut(P3<WorldSpace>)) {
        let g = &*self.height;
        let stride = step.max(1);
        let columns: Vec<usize> = (0..g.width).step_by(stride).collect();
        // The truncation depends only on x, so it is one lookup per sampled column
        // rather than one per sample -- which is the whole difference between
        // `Truncation::RealBands` costing nothing and costing a branch per point.
        let column_truncation: Vec<f64> =
            columns.iter().map(|&x| self.truncation_height_at(g.position(x, 0).x())).collect();
        for y in (0..g.height).step_by(stride) {
            for (i, &x) in columns.iter().enumerate() {
                let p = g.position(x, y);
                body(P3::new(p.x(), p.y(), swift::min(f64::from(g.at(x, y)), column_truncation[i])));
            }
        }
    }

    /// The height of a *tiled* landscape at a world point.
    ///
    /// The grid covers one fundamental tile; the drawing covers its images under
    /// the occluder's affine maps. A point out on the drawing is therefore not in
    /// the grid, and asking the grid about it returns whatever is nearest the
    /// edge -- which for elliptic, whose cutout runs across all nineteen tiles
    /// and whose tile edge is a pole, is a spire-high wall standing along the
    /// whole boundary.
    ///
    /// The point is mapped back through the tile it belongs to, chosen as the
    /// *nearest* rather than the containing one. Containment is too strict to be
    /// useful here: elliptic samples its tile on `[-K + eps, -eps]` to keep off
    /// the poles, so the cutout perimeter -- drawn at the exact quarter-periods
    /// -- lies a hair outside every tile image, and an exact test matches
    /// nothing at all.
    pub fn height_at_tiles(&self, p: P2<DomainSpace>, tiles: &[Affine2]) -> f64 {
        self.height_at(self.in_tile(p, tiles))
    }

    /// |f| at a world point of a tiled landscape, untruncated:
    /// `height_at_tiles` before the truncation.
    pub fn magnitude_at_tiles(&self, p: P2<DomainSpace>, tiles: &[Affine2]) -> f64 {
        self.magnitude_at(self.in_tile(p, tiles))
    }

    /// The grid's spacings along each axis.
    fn spacings(&self) -> (f64, f64) {
        let g = &*self.height;
        (
            g.domain.real.length().abs() / g.width.saturating_sub(1).max(1) as f64,
            g.domain.imag.length().abs() / g.height.saturating_sub(1).max(1) as f64,
        )
    }

    /// The outward normal of the truncated surface over a world point of a
    /// tiled landscape: `(-∂h/∂x, -∂h/∂y, 1)`, not normalized, by central
    /// differences a grid spacing wide, the height read through whichever
    /// tile the point is in. The differences run along the grid's own axes
    /// as the tile under the point carries them into place, a cell of each
    /// axis's own length, and the slope they give is carried back into
    /// world coordinates: so a copy's normal is the copy of the normal, a
    /// reflected copy's turned with it, and a turned copy's read off the
    /// grid's chords, not a cell of one axis's length along the other. On a
    /// plateau the differences vanish and the normal is up.
    ///
    /// With the function (`refine`), the slope is f's own, read by
    /// differences a step small against the cell and large against
    /// rounding, as the refiner reads it: the curves on the surface are placed
    /// on f to the last bit, and a facing read off the grid a cell wide
    /// tests it with a ruler a cell long -- a ring ends a stencil's width
    /// short of the fold on a convex graph and shows a stencil's width past
    /// it on a hump, whatever the grid. Without the function, the grid
    /// is the surface and its chords are the slope.
    pub fn normal(&self, p: P2<WorldSpace>, tiles: &[Affine2], refine: Option<&ContourRefine>) -> Vec3 {
        let (dx, dy) = self.spacings();
        if let Some(magnitude) = refine.and_then(|r| r.magnitude.as_ref()) {
            let step = 1e-4 * swift::max(dx, dy);
            let h = |x: f64, y: f64| -> f64 {
                let q = self.in_tile(P2::new(x, y), tiles);
                let u = magnitude(q);
                swift::min(if u.is_nan() { f64::INFINITY } else { u }, self.truncation_height_at(q.x()))
            };
            let hx = (h(p.x() + step, p.y()) - h(p.x() - step, p.y())) / (2.0 * step);
            let hy = (h(p.x(), p.y() + step) - h(p.x(), p.y() - step)) / (2.0 * step);
            if hx.is_finite() && hy.is_finite() {
                return Vec3::new(-hx, -hy, 1.0);
            }
        }
        let h = |q: Vec2| -> f64 { self.height_at_tiles(P2::new(q.x, q.y), tiles) };
        // The grid's axes in place, and the slope along each.
        let back = self.tile_under(P2::new(p.x(), p.y()), tiles).1;
        let forward = back.inverse().unwrap_or(Affine2::IDENTITY);
        let u = Vec2::new(forward.a, forward.c) * dx;
        let v = Vec2::new(forward.b, forward.d) * dy;
        let hu = (h(p.v + u) - h(p.v - u)) / (2.0 * dx);
        let hv = (h(p.v + v) - h(p.v - v)) / (2.0 * dy);
        Vec3::new(-(back.a * hu + back.c * hv), -(back.b * hu + back.d * hv), 1.0)
    }

    /// The normal of the graph over `p`: f's own slope, untruncated, read as
    /// `normal` reads it, so a fold is traced on the one surface the graph is
    /// and continues below the truncation, where the truncation edge cuts it.
    /// The truncated normal without the function, or where f's slope is not
    /// finite.
    pub fn graph_normal(&self, p: P2<WorldSpace>, tiles: &[Affine2], refine: Option<&ContourRefine>) -> Vec3 {
        let Some(magnitude) = refine.and_then(|r| r.magnitude.as_ref()) else {
            return self.normal(p, tiles, refine);
        };
        let (dx, dy) = self.spacings();
        let step = 1e-4 * swift::max(dx, dy);
        let h = |x: f64, y: f64| -> f64 {
            let u = magnitude(self.in_tile(P2::new(x, y), tiles));
            if u.is_nan() { f64::INFINITY } else { u }
        };
        let hx = (h(p.x() + step, p.y()) - h(p.x() - step, p.y())) / (2.0 * step);
        let hy = (h(p.x(), p.y() + step) - h(p.x(), p.y() - step)) / (2.0 * step);
        if !(hx.is_finite() && hy.is_finite()) {
            return self.normal(p, tiles, refine);
        }
        Vec3::new(-hx, -hy, 1.0)
    }

    /// The point of the fundamental tile a world point is the image of.
    pub fn in_tile(&self, p: P2<DomainSpace>, tiles: &[Affine2]) -> P2<DomainSpace> {
        self.tile_under(p, tiles).0
    }

    /// The tile under a world point: the point of the fundamental tile it is
    /// the image of (`in_tile`), and the inverse of the tile, which
    /// carries it back. The identity for one tile, or where none is
    /// invertible.
    pub fn tile_under(&self, p: P2<DomainSpace>, tiles: &[Affine2]) -> (P2<DomainSpace>, Affine2) {
        if tiles.len() <= 1 {
            return (p, Affine2::IDENTITY);
        }
        let d = self.domain();
        let (xlo, xhi) = (swift::min(d.real.lo, d.real.hi), swift::max(d.real.lo, d.real.hi));
        let (ylo, yhi) = (swift::min(d.imag.lo, d.imag.hi), swift::max(d.imag.lo, d.imag.hi));
        let mut best: Option<(P2<DomainSpace>, Affine2)> = None;
        let mut best_distance = f64::INFINITY;
        for tile in tiles {
            let Some(back) = tile.inverse() else { continue };
            let q = back.apply(world(p));
            // How far outside the domain box this tile puts the point; zero
            // when it is inside.
            let dx = swift::max(xlo - q.x(), 0.0) + swift::max(q.x() - xhi, 0.0);
            let dy = swift::max(ylo - q.y(), 0.0) + swift::max(q.y() - yhi, 0.0);
            let distance = dx + dy;
            if distance < best_distance {
                best_distance = distance;
                best = Some((P2::new(q.x(), q.y()), back));
                if distance == 0.0 {
                    break;
                }
            }
        }
        best.unwrap_or((p, Affine2::IDENTITY))
    }

    /// A point of the domain's rectangle, or a rounding off it: taken into
    /// it, and onto an edge within `edge_tolerance` of one. A point of a tile
    /// boundary carried back through either tile lands a rounding either side
    /// of the domain's edge, and the far edge reads the band before it only
    /// on the edge itself.
    pub fn onto_domain(&self, q: P2<WorldSpace>) -> P2<DomainSpace> {
        let d = self.domain();
        let tolerance = self.edge_tolerance();
        let onto = |v: f64, a: f64, b: f64| -> f64 {
            let (lo, hi) = (swift::min(a, b), swift::max(a, b));
            let c = swift::min(swift::max(v, lo), hi);
            if (c - lo).abs() <= tolerance {
                lo
            } else if (c - hi).abs() <= tolerance {
                hi
            } else {
                c
            }
        };
        P2::new(onto(q.x(), d.real.lo, d.real.hi), onto(q.y(), d.imag.lo, d.imag.hi))
    }

    /// The tile boundaries between `tiles`' copies of the domain, in the
    /// order of their pairs of tiles. A copy's side and another's lie along
    /// one line within `edge_tolerance`, overlapping by more than that, with
    /// the copies' centres either side.
    pub fn tile_boundaries(&self, tiles: &[Affine2]) -> Vec<TileBoundary> {
        if tiles.len() <= 1 {
            return Vec::new();
        }
        let d = self.domain();
        let (xlo, xhi) = (swift::min(d.real.lo, d.real.hi), swift::max(d.real.lo, d.real.hi));
        let (ylo, yhi) = (swift::min(d.imag.lo, d.imag.hi), swift::max(d.imag.lo, d.imag.hi));
        let corners = [Vec2::new(xlo, ylo), Vec2::new(xhi, ylo), Vec2::new(xhi, yhi), Vec2::new(xlo, yhi)];
        let centre = 0.5 * Vec2::new(xlo + xhi, ylo + yhi);
        let image = |tile: &Affine2, p: Vec2| -> Vec2 { tile.apply(P3::new(p.x, p.y, 0.0)).xy().v };
        let tolerance = self.edge_tolerance();
        let mut out = Vec::new();
        for i in 0..tiles.len() {
            if tiles[i].inverse().is_none() {
                continue;
            }
            for j in (i + 1)..tiles.len() {
                if tiles[j].inverse().is_none() {
                    continue;
                }
                let ci = image(&tiles[i], centre);
                let cj = image(&tiles[j], centre);
                'sides: for e in 0..4 {
                    let a = image(&tiles[i], corners[e]);
                    let b = image(&tiles[i], corners[(e + 1) % 4]);
                    let length = (b - a).length();
                    if !(length > tolerance) {
                        continue;
                    }
                    let u = (b - a) / length;
                    let n = Vec2::new(-u.y, u.x);
                    if !((ci - a).dot(n) * (cj - a).dot(n) < 0.0) {
                        continue;
                    }
                    for f in 0..4 {
                        let p = image(&tiles[j], corners[f]);
                        let q = image(&tiles[j], corners[(f + 1) % 4]);
                        if !((p - a).dot(n).abs() <= tolerance && (q - a).dot(n).abs() <= tolerance) {
                            continue;
                        }
                        let sp = (p - a).dot(u);
                        let sq = (q - a).dot(u);
                        let lo = swift::max(0.0, swift::min(sp, sq));
                        let hi = swift::min(length, swift::max(sp, sq));
                        if !(hi - lo > tolerance) {
                            continue;
                        }
                        // From the end with the lesser x, or y, whichever
                        // tile's side it is read along.
                        let mut ends =
                            [if lo == 0.0 { a } else { a + u * lo }, if hi == length { b } else { a + u * hi }];
                        if (ends[1].x, ends[1].y) < (ends[0].x, ends[0].y) {
                            ends.swap(0, 1);
                        }
                        out.push(TileBoundary {
                            first: i,
                            second: j,
                            start: P2::from_vec(ends[0]),
                            end: P2::from_vec(ends[1]),
                        });
                        break 'sides;
                    }
                }
            }
        }
        out
    }

    /// The top of the copy on a tile over a point of a tile boundary, `back`
    /// the tile's inverse: min(|f|, h(x)) at the point of the domain's edge
    /// the tile boundary point is the copy of, `magnitude` reading |f| there.
    pub fn tile_boundary_top(
        &self,
        w: P2<WorldSpace>,
        back: Affine2,
        magnitude: &dyn Fn(P2<DomainSpace>) -> f64,
    ) -> f64 {
        let q = self.onto_domain(back.apply(P3::new(w.x(), w.y(), 0.0)).xy());
        swift::min(magnitude(q), self.truncation_height_at(q.x()))
    }

    /// Whether a copy's top over a tile boundary stands over another's: by
    /// more than a billionth of the lower, a rounding of the point two tiles
    /// carry back, as copies reflected across the tile boundary do.
    pub fn stands(higher: f64, lower: f64) -> bool {
        higher > lower + 1e-9 * swift::max(1.0, lower.abs())
    }

    /// The height a wall rises to at its foot `p`, `t` of the way along its
    /// edge from `a` to `b`, `inward` its normal into the footprint: the
    /// truncated surface over the foot, read through the tile under it, as
    /// `height_at_tiles` reads it, but with the truncation's one-sided
    /// limit from the footprint's side (`Surface::footprint_side`).
    #[allow(clippy::too_many_arguments)] // Swift's signature, argument for argument
    pub fn wall_height(
        &self,
        p: P2<DomainSpace>,
        t: f64,
        a: P2<WorldSpace>,
        b: P2<WorldSpace>,
        inward: Vec2,
        tiles: &[Affine2],
        through: Option<Affine2>,
    ) -> f64 {
        let (point, truncation_height) = self.wall_foot(p, t, b.v - a.v, inward, tiles, through);
        swift::min(self.magnitude_at(point), truncation_height)
    }

    /// A wall's foot `p`, `t` of the way along its edge, `along` the edge's
    /// direction and `inward` its normal into the footprint, as the domain
    /// reads it: the point of the domain the foot is the image of, carried
    /// back through the tile its footprint is in (`wall_tile`), and the
    /// truncation over it, read at its x there (`wall_foot_x`) from the
    /// footprint's side (`Surface::footprint_side`). Every reader of a wall's
    /// top reads it so: the curtain, the top edge, the corners and the wall
    /// face.
    ///
    /// A foot where the wall crosses a tile boundary stands on two copies,
    /// and the top edge reads it through each (`through`): carried back by
    /// that tile, onto the domain's edge.
    pub fn wall_foot(
        &self,
        p: P2<DomainSpace>,
        t: f64,
        along: Vec2,
        inward: Vec2,
        tiles: &[Affine2],
        through: Option<Affine2>,
    ) -> (P2<DomainSpace>, f64) {
        let (q, back) = match through {
            Some(back) => (self.onto_domain(back.apply(P3::new(p.x(), p.y(), 0.0)).xy()), back),
            None => self.wall_tile(p, t, along, inward, tiles),
        };
        let side = Surface::footprint_side(t, along, inward, back);
        (q, self.truncation_height_at_from(self.wall_foot_x(q.x(), t, along, back), side))
    }

    /// The tile a wall's foot `p` is read through, and the foot carried back
    /// through it: the tile under the footprint beside the foot, a billionth
    /// of a cell in along `inward`, and at an end of the wall along the wall
    /// as well. A wall on a tile boundary stands on both tiles; it bounds the
    /// copy its footprint is in, and the tile under the foot itself would be
    /// whichever comes first.
    pub fn wall_tile(
        &self,
        p: P2<DomainSpace>,
        t: f64,
        along: Vec2,
        inward: Vec2,
        tiles: &[Affine2],
    ) -> (P2<DomainSpace>, Affine2) {
        if tiles.len() <= 1 {
            return (p, Affine2::IDENTITY);
        }
        let length = along.length();
        let into = if length > 0.0 {
            along / length
                * (if t == 0.0 {
                    1.0
                } else if t == 1.0 {
                    -1.0
                } else {
                    0.0
                })
        } else {
            Vec2::ZERO
        };
        let back = self.tile_under(P2::from_vec(p.v + self.edge_tolerance() * (inward + into)), tiles).1;
        let q = back.apply(world(p));
        (P2::new(q.x(), q.y()), back)
    }

    /// Re z of a wall's foot carried back into the domain by `back`, taken
    /// onto a band boundary within `edge_tolerance` of it where the wall
    /// runs parallel to the imaginary axis, as one standing on a boundary
    /// does, or ends: carried back through a tile, a foot
    /// on the boundary's image lands a rounding either side of it (the wall
    /// at 4.6 under x ↦ 5 - x comes back at 0.40000000000000036), and would
    /// read the band on whichever side it fell before its footprint's side
    /// could count. A foot between the ends of a wall that crosses a
    /// boundary keeps its x, so the wall steps where it crosses.
    pub fn wall_foot_x(&self, x: f64, t: f64, along: Vec2, back: Affine2) -> f64 {
        let Truncation::RealBands { bands, .. } = &self.truncation else { return x };
        if !(t == 0.0 || t == 1.0 || Surface::is_parallel_to_imaginary_axis(along, back)) {
            return x;
        }
        let tolerance = self.edge_tolerance();
        bands.iter().map(|b| b.below).find(|&b| (x - b).abs() <= tolerance).unwrap_or(x)
    }

    /// Whether an edge running along `along` is parallel to the imaginary
    /// axis once carried back into the domain by `back`: the carried
    /// direction's part along Re z is a rounding of its length. Not exactly
    /// zero: a tile written from cos and sin turns a quarter with cos(π/2) =
    /// 6.1e-17, and a wall on a boundary's image there carries back
    /// crossing it at that slope.
    pub fn is_parallel_to_imaginary_axis(along: Vec2, back: Affine2) -> bool {
        let carried = Vec2::new(back.a * along.x + back.b * along.y, back.c * along.x + back.d * along.y);
        carried.x.abs() <= 1e-12 * carried.length()
    }

    /// The side of Re z a wall's foot reads the truncation from, at `t` of
    /// the way along its edge, `along` the edge's direction and `inward` its
    /// normal into the footprint, both carried back into the domain by
    /// `back`: the side the footprint is on. Along a wall parallel to the
    /// imaginary axis, as one on a band boundary is, that is the way `inward`
    /// points; at an end of a wall that is not, the way the wall goes, so its
    /// top runs on to its end in its own band. Elsewhere, on a wall that
    /// crosses a boundary, the right, as `Truncation::height_at` reads it, so
    /// the wall steps there. The sides differ only on a boundary.
    pub fn footprint_side(t: f64, along: Vec2, inward: Vec2, back: Affine2) -> Side {
        let carried = |d: Vec2| back.a * d.x + back.b * d.y;
        let across = carried(along);
        let x = if Surface::is_parallel_to_imaginary_axis(along, back) {
            carried(inward)
        } else if t == 0.0 {
            across
        } else if t == 1.0 {
            -across
        } else {
            0.0
        };
        if x < 0.0 { Side::Left } else { Side::Right }
    }

    /// The height as the depth pass draws it over a domain point: the grid
    /// decimated by `step`, each cell as the two triangles the rasterizer
    /// splits it into -- from corner (1, 0) to corner (0, 1) -- then truncated.
    /// `height_at` interpolates bilinearly, which is what the Python plate
    /// placed its curves by and the fixtures pin; this is what the curves are drawn
    /// against, and the two differ within a cell by its twist.
    pub fn drawn_height(&self, p: P2<DomainSpace>, step: usize) -> f64 {
        let drawn = self.drawn_surface(p, step);
        swift::min(drawn.interpolated, drawn.truncation_height)
    }

    /// The decimated grid's cell under a point, and where in it: the lower
    /// corner and the fractions along each axis. `None` where Swift's
    /// `Int(fx)` traps, on a point that is not a number.
    fn drawn_cell(&self, p: P2<DomainSpace>, step: usize, nx: usize, ny: usize) -> Option<(usize, usize, f64, f64)> {
        let f = self.height.index(p) / step as f64;
        let fx = swift::min(swift::max(f.x, 0.0), (nx - 1) as f64);
        let fy = swift::min(swift::max(f.y, 0.0), (ny - 1) as f64);
        let i = (swift::int(fx)? as usize).min(nx - 2);
        let j = (swift::int(fy)? as usize).min(ny - 2);
        Some((i, j, fx - i as f64, fy - j as f64))
    }

    /// The height the depth pass's triangles draw over a point: as
    /// `drawn_height`, but each sample clamped to the truncation over it
    /// before the triangle interpolates them, as the shader clamps its
    /// vertices. In a cell the truncation crosses it lies under
    /// `drawn_height`, which interpolates and then clamps. Not a number
    /// where Swift traps, at a point that is not one.
    pub fn clamped_height(&self, p: P2<DomainSpace>, step: usize) -> f64 {
        let g = &*self.height;
        let step = step.max(1);
        let nx = g.width.div_ceil(step);
        let ny = g.height.div_ceil(step);
        if nx < 2 || ny < 2 {
            return swift::min(f64::from(g.at(0, 0)), self.truncation_height_at(p.x()));
        }
        let Some((i, j, u, v)) = self.drawn_cell(p, step, nx, ny) else { return f64::NAN };
        let sample = |a: usize, b: usize| -> f64 {
            swift::min(
                f64::from(g.at(a * step, b * step)),
                self.truncation_height_at(g.position(a * step, b * step).x()),
            )
        };
        let (h00, h10, h01, h11) = (sample(i, j), sample(i + 1, j), sample(i, j + 1), sample(i + 1, j + 1));
        let h = if u + v <= 1.0 {
            h00 + u * (h10 - h00) + v * (h01 - h00)
        } else {
            h11 + (1.0 - u) * (h01 - h11) + (1.0 - v) * (h10 - h11)
        };
        if h.is_nan() { f64::INFINITY } else { h }
    }

    /// `drawn_height` as the two numbers it is the lesser of -- the grid's
    /// interpolated height and the truncation over the point -- with the cosine of
    /// the interpolating triangle's tilt, `1 / sqrt(1 + |∇h|²)`, which turns a
    /// vertical gap to that triangle into a perpendicular distance.
    pub fn drawn_surface(&self, p: P2<DomainSpace>, step: usize) -> DrawnSurface {
        let g = &*self.height;
        let step = step.max(1);
        let nx = g.width.div_ceil(step);
        let ny = g.height.div_ceil(step);
        let truncation_height = self.truncation_height_at(p.x());
        if nx < 2 || ny < 2 {
            return DrawnSurface { interpolated: f64::from(g.at(0, 0)), truncation_height, cosine: 1.0 };
        }
        let (sx, sy) = self.spacings();
        let dx = sx * step as f64;
        let dy = sy * step as f64;
        let Some((i, j, u, v)) = self.drawn_cell(p, step, nx, ny) else {
            return DrawnSurface { interpolated: f64::NAN, truncation_height, cosine: f64::NAN };
        };
        let h00 = f64::from(g.at(i * step, j * step));
        let h10 = f64::from(g.at((i + 1) * step, j * step));
        let h01 = f64::from(g.at(i * step, (j + 1) * step));
        let h11 = f64::from(g.at((i + 1) * step, (j + 1) * step));
        let lower = u + v <= 1.0;
        let h = if lower {
            h00 + u * (h10 - h00) + v * (h01 - h00)
        } else {
            h11 + (1.0 - u) * (h01 - h11) + (1.0 - v) * (h10 - h11)
        };
        let hx = (if lower { h10 - h00 } else { h11 - h01 }) / dx;
        let hy = (if lower { h01 - h00 } else { h11 - h10 }) / dy;
        let cosine = 1.0 / (1.0 + hx * hx + hy * hy).sqrt();
        DrawnSurface {
            interpolated: if h.is_nan() { f64::INFINITY } else { h },
            truncation_height,
            cosine: if cosine.is_nan() { 0.0 } else { cosine },
        }
    }

    /// Lift a domain point onto the (truncated) surface.
    pub fn lift(&self, p: P2<DomainSpace>) -> P3<WorldSpace> {
        P3::new(p.x(), p.y(), self.height_at(p))
    }

    /// Lift a domain-space path onto the surface, by policy.
    ///
    /// `Surface.lift_contours`' three height policies, and the reason a bundle
    /// records which one a layer used: a magnitude isocontour sits at exactly
    /// its own level, a phase contour sits wherever the surface is, and an
    /// unclamped lift is a third thing again. A consumer regenerating a layer
    /// cannot guess which was meant.
    pub fn lift_path(&self, path: &[P2<DomainSpace>], policy: HeightPolicy, level: f64) -> Vec<P3<WorldSpace>> {
        path.iter()
            .map(|&p| match policy {
                HeightPolicy::Surface => P3::new(p.x(), p.y(), self.height_at(p)),
                HeightPolicy::Level => P3::new(p.x(), p.y(), level),
                HeightPolicy::Magnitude => P3::new(p.x(), p.y(), self.magnitude_at(p)),
            })
            .collect()
    }

    /// Whether a lifted vertex survives a `Restriction`.
    pub fn admits(&self, p: P3<WorldSpace>, restriction: &Restriction, region: &Region) -> bool {
        match restriction {
            Restriction::All => true,
            Restriction::Footprint => region.contains(p.xy()),
            Restriction::BelowTruncation => p.z() <= self.truncation_height_at(p.x()),
            Restriction::Band { axis, lo, hi } => match axis {
                DomainAxis::Real => p.x() > *lo && p.x() < *hi,
                DomainAxis::Imag => p.y() > *lo && p.y() < *hi,
            },
            Restriction::Intersection(all) => all.iter().all(|r| self.admits(p, r, region)),
        }
    }

    /// Where the segment from `a`, below the truncation, to `b`, over it, reaches the
    /// truncation -- on the truncation edge, at exactly the truncation height. `None` when it does not,
    /// or when there is no truncation.
    ///
    /// Solved on the lifted heights, linearly: that is the same interpolation
    /// the truncation edge is contoured by, so the crossing lands on the drawn truncation edge to
    /// within a cell's bend.
    ///
    /// Where a band boundary lies between them the truncation steps there,
    /// and the segment is followed band by band from `a`, each band's
    /// truncation constant: it reaches the truncation inside a band, solved
    /// as above in that band, or it reaches a boundary under its own band's
    /// truncation and over the next one's, and ends there, at its own
    /// height, on the riser's top -- where the graph meets the riser, not
    /// on either plateau. `height` is f's height there, where the caller
    /// has the function; the segment's own otherwise. The end is placed on
    /// `a`'s side of the boundary, a rounding short of it going right, since
    /// the boundary belongs to the band right of it, so it is read in `a`'s
    /// band. An end on a boundary is read in the band the segment runs into
    /// from it, so a step at that end is a riser like any other: `a` there,
    /// over the band it runs into, is on the riser's top already, and the
    /// run ends at it.
    pub fn truncation_crossing(
        &self,
        a: P3<WorldSpace>,
        b: P3<WorldSpace>,
        height: Option<&Magnitude>,
    ) -> Option<P3<WorldSpace>> {
        let ea = a.z() - self.truncation_height_at(a.x());
        let eb = b.z() - self.truncation_height_at(b.x());
        if !(ea.is_finite() && eb.is_finite() && ea <= 0.0 && eb > 0.0) {
            return None;
        }
        if matches!(self.truncation, Truncation::RealBands { .. }) && a.x() != b.x() {
            let rightward = a.x() < b.x();
            // The bands the segment runs through, in order from `a`. An end
            // on a boundary belongs to the band right of it, which is not
            // the band the segment runs in where it leaves `a` going left or
            // reaches `b` going right: the truncation steps at that end.
            let spans = self.truncation.bands(swift::min(a.x(), b.x()), swift::max(a.x(), b.x()));
            let ordered: Vec<_> = if rightward { spans } else { spans.into_iter().rev().collect() };
            if let (Some(near), Some(far)) = (ordered.first(), ordered.last())
                && (ordered.len() > 1
                    || near.height != self.truncation_height_at(a.x())
                    || far.height != self.truncation_height_at(b.x()))
            {
                // The point of the segment at `x`.
                let at = |x: f64| -> P3<WorldSpace> {
                    let t = (x - a.x()) / (b.x() - a.x());
                    P3::new(x, a.y() + (b.y() - a.y()) * t, a.z() + (b.z() - a.z()) * t)
                };
                // The segment's end on a riser's top, read in `span`.
                let on_riser_top = |end: P3<WorldSpace>, span_height: f64| -> P3<WorldSpace> {
                    let z = height.map_or(end.z(), |h| h(P2::new(end.x(), end.y())));
                    P3::new(end.x(), end.y(), if z.is_finite() { swift::min(z, span_height) } else { end.z() })
                };
                let mut start = a;
                for (k, span) in ordered.iter().enumerate() {
                    let last = k == ordered.len() - 1;
                    // The near side of the boundary this band ends at.
                    let end = if last { b } else { at(if rightward { span.hi.next_down() } else { span.lo }) };
                    let es = start.z() - span.height;
                    let ee = end.z() - span.height;
                    if !(es.is_finite() && ee.is_finite()) {
                        return None;
                    }
                    // Over this band's truncation from the start: only at
                    // `a`, on a boundary the truncation steps down from into
                    // the segment. `a` is on the riser's top, and the run
                    // ends there.
                    if !(es <= 0.0) {
                        return None;
                    }
                    if ee > 0.0 {
                        let t = es / (es - ee);
                        let x = start.x() + (end.x() - start.x()) * t;
                        // Kept in this band, where the solve lands a
                        // rounding past its end.
                        let kept = if rightward { swift::min(x, end.x()) } else { swift::max(x, end.x()) };
                        return Some(P3::new(kept, start.y() + (end.y() - start.y()) * t, span.height));
                    }
                    if last {
                        // Under this band's truncation all the way to `b`,
                        // which is over its own: `b` is on a boundary, going
                        // right, where the truncation steps down.
                        if !rightward {
                            return None;
                        }
                        return Some(on_riser_top(at(b.x().next_down()), span.height));
                    }
                    if end.z() > ordered[k + 1].height {
                        return Some(on_riser_top(end, span.height));
                    }
                    start = end;
                }
                return None;
            }
        }
        let t = ea / (ea - eb);
        let x = a.x() + (b.x() - a.x()) * t;
        Some(P3::new(x, a.y() + (b.y() - a.y()) * t, self.truncation_height_at(x)))
    }

    /// |f| or arg f, as `tag` names it, as a scalar field on the graph,
    /// contoured on its grid and placed on f by `refine`
    /// (`ContourRefine::contours`), which is handed the tag and `ceiling`, the
    /// height above which a layer's curves are cut away. Without a refiner
    /// the grid's paths are the contours. `None` for arg f on a landscape with
    /// no phase grid.
    ///
    /// arg f is circle-valued and has no period here: the refiner owns its
    /// branch cut, the level ±π. Beside a zero of f a grid edge can span
    /// more than π of phase, which the refiner keeps by testing the edge's
    /// midpoint on f, and the rule a circle-valued field is contoured by --
    /// no crossing on an edge whose samples differ by more than half a
    /// period -- would drop it. So the refiner is handed the grid's own
    /// paths, the spurious crossings along the wrap included, and at the
    /// branch cut the zero level of arg(-f) (`contour::cut`), which it places
    /// there. Without a refiner the branch cut has no contour: nothing would
    /// drop the wrap's crossings. `None` for a field not f's, which the refiner
    /// does not place.
    pub fn field(&self, tag: ContourField, refine: Option<&ContourRefine>, ceiling: f64) -> Option<ScalarField> {
        let samples: Arc<Grid2D<f32>> = match tag {
            ContourField::Magnitude => self.height.clone(),
            ContourField::Phase => self.phase.clone()?,
            ContourField::U | ContourField::V | ContourField::Height | ContourField::Isophote => return None,
        };
        let refine = refine.cloned();
        let grid = samples.clone();
        Some(ScalarField::placed(
            samples,
            Arc::new(move |level: f64, paths: &[Vec<P2<ParamSpace>>]| -> ParamPaths {
                let is_cut = tag == ContourField::Phase && crate::contour::is_cut(level);
                let Some(refine) = &refine else {
                    return if is_cut { Vec::new() } else { paths.to_vec() };
                };
                let lines: Vec<Vec<P2<DomainSpace>>> = if is_cut {
                    crate::contour::cut(&grid)
                } else {
                    paths.iter().map(|p| p.iter().map(|q| P2::from_vec(q.v)).collect()).collect()
                };
                let mut placed = (refine.contours)(tag, level, &lines, ceiling, ContourRefine::EVERYWHERE);
                if tag == ContourField::Magnitude
                    && let Some(rings) = &refine.rings
                {
                    let more = rings(level, &placed);
                    placed.extend(more);
                }
                placed.iter().map(|p| p.iter().map(|q| P2::from_vec(q.v)).collect()).collect()
            }),
        ))
    }

    /// Every level of a described layer of |f| or arg f on the graph,
    /// contoured, lifted, filtered and tiled.
    ///
    /// A path that leaves the kept region and comes back is split where it left,
    /// not closed across the gap. Filtering vertices and keeping the survivors
    /// as one polyline welds the far side of a contour to the near side; on the
    /// zeta plate that drew straight chords up to half the width of the domain
    /// across ground the cutout had deliberately removed.
    ///
    /// Where a path leaves by going *over the truncation*, the split is at the truncation
    /// itself: the run gains a vertex on the truncation edge (`truncation_crossing`) rather than
    /// ending at its last vertex under it, which on a pole's graph is up to
    /// half a world unit short. Only with a refiner: without one the curves are
    /// exactly what Python derives from the same grids, which ends at the
    /// last sample, and the fixtures hold it to that.
    ///
    /// The contours are the field's (`field`), asked for through
    /// `scalar_field::levels` as any field's are, over the graph's parameter
    /// rectangle (`parameter_rectangle`). Not over its trimmed domain: the
    /// restriction says which part of the domain a layer keeps, and a run is
    /// carried to the truncation edge by `truncation_crossing`, as below. A
    /// layer kept to the footprint is cut to it exactly, a tiled layer's copy
    /// on each tile where it lands. A phase level of ±π is the branch cut,
    /// where arg f jumps, and the grid has no contour at it. With a refiner
    /// it is drawn once, after the other levels, as one line of curves the
    /// way the Jahnke-Emde plates draw it.
    pub fn derive(
        &self,
        source: &LayerSource,
        policy: HeightPolicy,
        region: &Region,
        tiles: &[Affine2],
        refine: Option<&ContourRefine>,
    ) -> PolylineSet<WorldSpace> {
        let LayerSource::Contour { field: tag, levels, restriction, faces: FaceSelector::Graph { tiled }, .. } = source
        else {
            return PolylineSet::empty();
        };
        let (tag, tiled) = (*tag, *tiled);
        let Some(rectangle) = self.parameter_rectangle() else { return PolylineSet::empty() };
        if !tag.is_field_of_f() {
            return PolylineSet::empty();
        }
        let carry = refine.is_some() && restriction.requires_below_truncation();
        // A run carried onto a riser's top ends at f's own height there, as
        // its vertices are lifted; a level line at its level.
        let riser_height = if policy == HeightPolicy::Level { None } else { refine.and_then(|r| r.magnitude.as_ref()) };
        // Curves lifted to |f| and kept below the truncation are cut away wherever |f|
        // is above it, so the refiner need not place it there: on zeta and
        // 1/Γ that was nineteen phase vertices of every twenty. A tenth over
        // the truncation is room for a chord whose ends lie above it and whose middle
        // dips under. Curves lifted to the truncated surface are drawn on the plateau's
        // top, and placed all over.
        let ceiling = if restriction.requires_below_truncation() && policy == HeightPolicy::Magnitude {
            1.1 * self.truncation.highest()
        } else {
            f64::INFINITY
        };
        let Some(scalar) = self.field(tag, refine, ceiling) else { return PolylineSet::empty() };

        // A refiner has the function the grid was sampled from and moves
        // the vertices onto its true level set, in domain space, before
        // the lift: the grid decides which contours exist, f decides
        // exactly where they run. Every level at once, as `scalar_field::levels`
        // hands them on.
        //
        // Marching squares runs at the branch cut too, and its paths are
        // dropped for `contour::cut`'s: about 0.3-0.6 ms at res 1000, under
        // 1% of the refined work. Skipping it would want the branch cut as a
        // field of its own and a second `levels` call.
        let contoured = scalar_field::levels(&scalar, &tag.levels(levels), rectangle, &Trim::default());
        let magnitude = refine.and_then(|r| r.magnitude.as_ref());

        // The runs of the contours a restriction keeps, lifted: a path cut
        // where a vertex leaves the kept region, and carried to the
        // truncation edge where it leaves over the truncation.
        let kept = |contoured: &Contoured, region: &Region| -> Vec<Vec<P3<WorldSpace>>> {
            let mut paths: Vec<Vec<P3<WorldSpace>>> = Vec::new();
            for (level, placed) in contoured {
                let level = *level;
                for line in placed {
                    let line: Vec<P2<DomainSpace>> = line.iter().map(|p| P2::from_vec(p.v)).collect();
                    // Lifted onto f where the function is at hand: a phase line
                    // lifted to the grid's height stands a chord's error off
                    // the surface the rings, the folds and the top edge are drawn
                    // on, and the test hides it there or shows it floating.
                    let mut lifted: Vec<P3<WorldSpace>> = match magnitude {
                        Some(magnitude) if policy != HeightPolicy::Level => line
                            .iter()
                            .map(|&p| {
                                let u = magnitude(p);
                                if !u.is_finite() {
                                    return self.lift_path(&[p], policy, level)[0];
                                }
                                P3::new(
                                    p.x(),
                                    p.y(),
                                    if policy == HeightPolicy::Surface {
                                        swift::min(u, self.truncation_height_at(p.x()))
                                    } else {
                                        u
                                    },
                                )
                            })
                            .collect(),
                        _ => self.lift_path(&line, policy, level),
                    };
                    // A run the refiner carried to a zero of f ends at the zero's
                    // own height. The grid has no sample there: its chord over a
                    // pit's floor bottoms out at the nearest sample's height, a
                    // cell's rise above the zero, and lifting the end to the
                    // chord would hang the fan of phase lines that far above the
                    // top edge that f draws under them. Only an end that *is* a zero,
                    // to rounding against the grid's own height, so a vertex the
                    // grid places a hair under f is left on the drawn surface.
                    if policy != HeightPolicy::Level
                        && let Some(magnitude) = magnitude
                        && !lifted.is_empty()
                    {
                        let last = lifted.len() - 1;
                        let ends: &[usize] = if last == 0 { &[0] } else { &[0, last] };
                        for &i in ends {
                            let u = magnitude(line[i]);
                            if u.is_finite() && u <= 1e-6 * lifted[i].z() {
                                lifted[i] = P3::new(lifted[i].x(), lifted[i].y(), u);
                            }
                        }
                    }
                    let mut run: Vec<P3<WorldSpace>> = Vec::new();
                    let mut previous: Option<(P3<WorldSpace>, bool)> = None;
                    for &v in &lifted {
                        let admitted = self.admits(v, restriction, region);
                        if admitted {
                            if carry
                                && let Some((pv, false)) = previous
                                && let Some(edge) = self.truncation_crossing(v, pv, riser_height)
                                && self.admits(edge, restriction, region)
                            {
                                run.push(edge);
                            }
                            run.push(v);
                        } else {
                            if carry
                                && let Some((pv, true)) = previous
                                && let Some(edge) = self.truncation_crossing(pv, v, riser_height)
                                && self.admits(edge, restriction, region)
                            {
                                run.push(edge);
                            }
                            if run.len() >= 2 {
                                paths.push(std::mem::take(&mut run));
                            }
                            run.clear();
                        }
                        previous = Some((v, admitted));
                    }
                    if run.len() >= 2 {
                        paths.push(run);
                    }
                }
            }
            paths
        };

        // The footprint is where a copy stands in space, so each copy, on
        // each tile or in place, is clipped to it there, exactly, as a
        // face's copy is (`Face::contours`): cut where the footprint's
        // perimeter, carried back through the tile, crosses a contour, and
        // kept where the footprint holds the copy's point. What else the
        // restriction asks is read on the domain, where the copy came from:
        // a band of Re z or Im z is the function's, and the truncation is
        // read at the point's own x through the tile under it, which is the
        // domain's.
        if restriction.requires_footprint() && matches!(region, Region::Inside(_)) {
            let identity = [Affine2::IDENTITY];
            let copies: &[Affine2] = if tiled { tiles } else { &identity };
            let mut all: Vec<Vec<P3<WorldSpace>>> = Vec::new();
            for tile in copies {
                let Some((perimeter, contains)) = Surface::footprint(region, *tile) else { continue };
                let clipped = scalar_field::clipped(
                    contoured.clone(),
                    rectangle,
                    &Trim { contains: Some(contains), cuts: perimeter, ..Trim::default() },
                );
                let paths = kept(&clipped, &Region::Full);
                if tiled {
                    all.extend(paths.into_iter().map(|p| p.into_iter().map(|v| tile.apply(v)).collect()));
                } else {
                    all.extend(paths);
                }
            }
            return PolylineSet::from_paths(&all);
        }
        let paths = kept(&contoured, region);
        if !tiled {
            return PolylineSet::from_paths(&paths);
        }
        // Replicated by the same maps the occluder instances the heightfield
        // with, so a tiled contour cannot drift from the tile it lies on.
        let all: Vec<Vec<P3<WorldSpace>>> = tiles
            .iter()
            .flat_map(|tile| paths.iter().map(move |p| p.iter().map(|&v| tile.apply(v)).collect()))
            .collect();
        PolylineSet::from_paths(&all)
    }
}

impl ContourField {
    /// The levels a layer's contours are asked for: as listed, but for arg f
    /// a level of ±π is the branch cut, asked once, as π, after the rest.
    pub fn levels(self, listed: &[f64]) -> Vec<f64> {
        if self != ContourField::Phase {
            return listed.to_vec();
        }
        let mut plain: Vec<f64> = listed.iter().copied().filter(|&l| !crate::contour::is_cut(l)).collect();
        if plain.len() < listed.len() {
            plain.push(core::f64::consts::PI);
            plain
        } else {
            listed.to_vec()
        }
    }
}
