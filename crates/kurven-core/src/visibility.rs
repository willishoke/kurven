//! Hidden lines for curves drawn on a parametric surface: the port of the
//! types of `Sources/KurvenCore/Visibility.swift`. What it does is Phase 4's
//! (a parametric surface's visibility), each method a typed refusal; the
//! facing, which `ParametricSurface::facing` already answers, and sphere
//! tracing (`SphereTracing`) are whole.

use std::sync::Arc;

use kurven_math::swift;

use crate::hidden_line::{HiddenLine, fold_crossing};
use crate::occlusion::{OcclusionTest, SphereTracing};
use crate::parametric::{ParamAxis, ParametricSurface};
use crate::polyline::PolylineSet;
use crate::simd::{Vec2, Vec3};
use crate::spaces::{P2, P3, PageSpace, ParamSpace, Transform, ViewSpace, WorldSpace};
use crate::surface_image::SurfaceImage;

/// Hidden lines for curves drawn on a parametric surface, decided by *where on
/// the surface* the curves are rather than by how deep it is.
///
/// Two tests, and each covers the other's blind spot:
///
/// - **Facing.** On a surface that bounds a solid, curves whose outward normal
///   faces away from the eye is hidden, full stop. This is the test the
///   second cannot make: across a fold the front and the back of the surface
///   are neighbours in coordinates, so "front-most is my neighbourhood" holds
///   on both sides of it.
/// - **Nothing in front.** With the solid as a point test (`TubularNeighbourhood`), the
///   line of sight from the curves are sphere-traced against it, stepping by the gap
///   over its Lipschitz constant so that no step can pass through the
///   solid: exact, with no pixel in it. Without one, the surface in front of
///   the curve, at the curve's pixel or one beside it, must be the curve's own
///   neighbourhood of the surface. A second sheet in front is far away in
///   coordinates however close it is in depth, which is where a depth depth tolerance
///   fails; but a raster is right only to its resolution, and relief puts
///   the curves where that runs out.
///
/// Where visibility changes along a segment -- at a fold, or where the
/// segment passes behind something -- the segment is cut exactly there, by
/// bisection on the exact test, and the pixel grid has no say in where.
#[derive(Debug, Clone)]
pub struct SurfaceVisibility {
    pub surface: ParametricSurface,
    /// The coordinate image, which the front-most test reads; None when the
    /// sphere tracing tests alone, since it asks nothing of a raster.
    /// Shared, not copied: a 3000-pixel image is about a hundred megabytes,
    /// and the scene makes a visibility for every layer it tests.
    pub image: Option<Arc<SurfaceImage>>,
    pub view: Transform<WorldSpace, ViewSpace>,
    /// The direction lines of sight travel, in world space.
    pub sight: Vec3,
    /// The occlusion test, where the surface has a tubular neighbourhood;
    /// None where the coordinate image stands in for one.
    pub tracing: Option<SphereTracing>,
}

/// Swift's precondition of `init(sphereTracing:view:)`, as a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoNeighbourhood;

impl SurfaceVisibility {
    pub fn new(
        surface: ParametricSurface,
        image: impl Into<Arc<SurfaceImage>>,
        view: Transform<WorldSpace, ViewSpace>,
    ) -> SurfaceVisibility {
        let sight = view.projection_direction();
        let tracing = SphereTracing::new(&surface, sight);
        SurfaceVisibility { surface, image: Some(image.into()), view, sight, tracing }
    }

    /// Visibility by sphere tracing alone, with no image: for a surface with an
    /// `TubularNeighbourhood`, tested anywhere without rendering anything first.
    /// Only a surface with a tubular neighbourhood is tested by sphere
    /// tracing alone; any other is refused.
    pub fn sphere_tracing(
        surface: ParametricSurface,
        view: Transform<WorldSpace, ViewSpace>,
    ) -> Result<SurfaceVisibility, NoNeighbourhood> {
        let sight = view.projection_direction();
        let tracing = SphereTracing::new(&surface, sight).ok_or(NoNeighbourhood)?;
        Ok(SurfaceVisibility { surface, image: None, view, sight, tracing: Some(tracing) })
    }

    /// Whether the curves are tested by sphere tracing against the solid, or by the
    /// coordinate image.
    pub fn traces_spheres(&self) -> bool {
        self.tracing.is_some()
    }

    /// Positive facing the eye, negative facing away, None on a surface that
    /// bounds no solid.
    pub fn facing(&self, c: P2<ParamSpace>) -> Option<f64> {
        self.surface.facing(c, self.sight)
    }

    /// Whether nothing of the solid stands between the surface point `q` and
    /// the eye: sphere tracing (`SphereTracing`). Clear, with no
    /// neighbourhood to trace: the coordinate image is asked instead.
    pub fn is_clear(&self, q: Vec3) -> bool {
        self.tracing.as_ref().is_none_or(|t| t.is_clear(P3::from_vec(q)))
    }

    /// How far the front-most coordinate moves per pixel at `(row, col)`, per
    /// axis of the parameter rectangle.
    ///
    /// Per screen axis, the smaller of the two one-sided differences, and the
    /// larger over the two screen axes -- the same shape as the preview's
    /// `depth_slope`, for the same reason: a pixel beside an occlusion edge
    /// has one neighbour on another sheet, and counting that jump as the
    /// surface's own rate would make the tolerance swallow the other sheet.
    pub fn footprint(&self, row: i64, col: i64) -> Vec2 {
        let Some(image) = &self.image else { return Vec2::ZERO };
        let Some(here) = image.coordinate(row, col) else { return Vec2::ZERO };
        let frame = image.frame();
        let rate = |dr: i64, dc: i64| -> Vec2 {
            let mut best: Option<Vec2> = None;
            for s in [-1i64, 1] {
                let (r, c) = (row + s * dr, col + s * dc);
                if !(r >= 0 && r < frame.rows && c >= 0 && c < frame.cols) {
                    continue;
                }
                let Some(there) = image.coordinate(r, c) else { continue };
                let d = Vec2::new(
                    self.surface.u.difference(there.x(), here.x()).abs(),
                    self.surface.v.difference(there.y(), here.y()).abs(),
                );
                best = Some(best.map_or(d, |b| b.swift_min(d)));
            }
            best.unwrap_or(Vec2::ZERO)
        };
        rate(1, 0).swift_max(rate(0, 1))
    }

    /// Whether the front-most surface beside view point `p` is the surface at
    /// `c`.
    ///
    /// The curves are within about a pixel of the nearest pixel's sample point,
    /// so on its own sheet the coordinates there differ from `c` by at most
    /// about one footprint. Two footprints, plus one grid cell for the
    /// sheet's own piecewise-linear rasterization, is the tolerance; another
    /// sheet is not within it except where the two meet, which is a fold and
    /// `facing` decides it. Nothing drawn in all nine pixels: nothing in
    /// front. Outside the frame: occluded, as `DepthImage::depth` has it.
    ///
    /// A pixel holding another sheet hides the point only where that sheet
    /// is in front of it. The image draws each sheet as its grid, a chord of
    /// the surface, and along a crease on a sheet bent sharply in view the
    /// chord can fall pixels short of the crease, so that all nine pixels
    /// hold only what lies behind it. So each such pixel's sheet is followed
    /// from the pixel's coordinate onto the point's line of sight
    /// (`sheet_lies_behind`), and the point is seen if every one of them
    /// meets it behind the point. Swift's precondition that there is an
    /// image is [`NoImage`] here.
    pub fn is_front_most(&self, c: P2<ParamSpace>, p: P3<ViewSpace>) -> Result<bool, NoImage> {
        let Some(image) = &self.image else { return Err(NoImage) };
        let frame = image.frame();
        let i = frame.index(p.xy());
        if !(i[0] >= 0 && i[0] < frame.rows && i[1] >= 0 && i[1] < frame.cols) {
            return Ok(false);
        }
        let (u, v) = (self.surface.u, self.surface.v);
        let mut others: Vec<P2<ParamSpace>> = Vec::new();
        for dr in -1..=1 {
            for dc in -1..=1 {
                let (r, col) = (i[0] + dr, i[1] + dc);
                if !(r >= 0 && r < frame.rows && col >= 0 && col < frame.cols) {
                    continue;
                }
                let Some(q) = image.coordinate(r, col) else { continue };
                let f = self.footprint(r, col);
                if u.difference(q.x(), c.x()).abs() <= 2.0 * f.x + u.spacing()
                    && v.difference(q.y(), c.y()).abs() <= 2.0 * f.y + v.spacing()
                {
                    return Ok(true);
                }
                others.push(q);
            }
        }
        Ok(others.iter().all(|&q| self.sheet_lies_behind(q, p)))
    }

    /// Whether the sheet through surface coordinate `q` meets the line of
    /// sight through view point `p` behind `p`.
    ///
    /// Newton's method on the map followed by the view, from `q`, for the
    /// coordinate whose view x and y are `p`'s; the sheet's depth there
    /// decides. A solve that leaves the two grid cells about `q`, or the
    /// rectangle along a bounded axis, has found some other sheet or none,
    /// and so does one that does not settle: either way the sheet is not
    /// shown to be behind, and hides the point as the image has it.
    pub fn sheet_lies_behind(&self, q: P2<ParamSpace>, p: P3<ViewSpace>) -> bool {
        let scale = swift::max(swift::max(swift::max(p.x().abs(), p.y().abs()), p.z().abs()), 1.0);
        let near = |axis: &ParamAxis, x: f64, x0: f64| -> bool {
            axis.difference(x, x0).abs() <= 2.0 * axis.spacing()
                && (axis.periodic || (x >= axis.range.lo && x <= axis.range.hi))
        };
        let mut c = q.v;
        for _ in 0..30 {
            let jet = (self.surface.map)(P2::from_vec(c));
            let at = self.view.apply(P3::from_vec(jet.position));
            let r = Vec2::new(at.x() - p.x(), at.y() - p.y());
            if r.length() <= 1e-12 * scale {
                return at.z() < p.z() - 1e-9 * scale;
            }
            // The columns of the Jacobian, the map's partials carried into the view.
            let a = self.view.apply(P3::from_vec(jet.position + jet.du)).v - at.v;
            let b = self.view.apply(P3::from_vec(jet.position + jet.dv)).v - at.v;
            let det = a.x * b.y - a.y * b.x;
            if !(det != 0.0 && det.is_finite()) {
                return false;
            }
            c -= Vec2::new(r.x * b.y - r.y * b.x, a.x * r.y - a.y * r.x) / det;
            if !(near(&self.surface.u, c.x, q.x()) && near(&self.surface.v, c.y, q.y())) {
                return false;
            }
        }
        false
    }

    /// Curves at view point `p`, lying on the surface at `c`.
    pub fn is_visible(&self, p: P3<ViewSpace>, c: P2<ParamSpace>) -> Result<bool, NoImage> {
        self.is_visible_facing(p, c, self.facing(c))
    }

    fn is_visible_facing(&self, p: P3<ViewSpace>, c: P2<ParamSpace>, f: Option<f64>) -> Result<bool, NoImage> {
        if let Some(f) = f
            && f <= 0.0
        {
            return Ok(false);
        }
        if self.traces_spheres() {
            return Ok(self.is_clear((self.surface.map)(c).position));
        }
        self.is_front_most(c, p)
    }

    /// Curves at surface coordinate `c`, placed by the map: whether it is seen,
    /// and where it is in the view. `on_folds` asks only sphere tracing, for
    /// curves on the folds, where the facing is zero by construction.
    fn test(&self, c: P2<ParamSpace>, on_folds: bool) -> Result<(bool, P3<ViewSpace>), NoImage> {
        let world = (self.surface.map)(c).position;
        let p = self.view.apply(P3::from_vec(world));
        if !on_folds
            && let Some(f) = self.facing(c)
            && f <= 0.0
        {
            return Ok((false, p));
        }
        let seen = if self.traces_spheres() { self.is_clear(world) } else { self.is_front_most(c, p)? };
        Ok((seen, p))
    }

    /// The point of a crease beside `c`, a point of the chord between two of
    /// its vertices: where the facing jumps, across the chord within half
    /// its length either way; `c` where no jump is in reach. The vertices
    /// are on the crease but the chord between them crosses a face, which
    /// can face away where the crease is seen: a convex truncation edge is seen where
    /// either of its faces is.
    pub fn onto_crease(&self, c: P2<ParamSpace>, chord: Vec2) -> P2<ParamSpace> {
        let scale = Vec2::new(self.surface.u.spacing(), self.surface.v.spacing());
        let along = chord.div_elements(scale);
        let cells = along.length();
        if !(cells > 0.0) {
            return c;
        }
        let n = (Vec2::new(-along.y, along.x) / cells).mul_elements(scale);
        let mut a = c.v - 0.5 * cells * n;
        let mut b = c.v + 0.5 * cells * n;
        let (Some(fa), Some(fb)) = (self.facing(P2::from_vec(a)), self.facing(P2::from_vec(b))) else { return c };
        for _ in 0..60 {
            let m = 0.5 * (a + b);
            let Some(fm) = self.facing(P2::from_vec(m)) else { return c };
            if (fm - fa).abs() < (fm - fb).abs() {
                a = m;
            } else {
                b = m;
            }
        }
        // A jump, not a slope: a hundredth of the span holds most of the change.
        let m = 0.5 * (a + b);
        let across = |t: f64| -> f64 {
            (self.facing(P2::from_vec(m + t * n)).unwrap_or(0.0) - self.facing(P2::from_vec(m - t * n)).unwrap_or(0.0))
                .abs()
        };
        if across(1e-3) > 0.5 * across(1e-1) { P2::from_vec(m) } else { c }
    }

    /// Where along the straight segment `a -> b` in coordinates the surface
    /// turns edge-on, as a fraction of the way; None when both ends face the
    /// same way, or the surface has no outside.
    ///
    /// Bisection on the exact normal, to the last bit of the fraction: the
    /// facing is continuous along the segment and its sign is all that is
    /// asked of it, so nothing faster is also as certain.
    pub fn fold_crossing(&self, a: P2<ParamSpace>, b: P2<ParamSpace>) -> Option<f64> {
        let fa = self.facing(a)?;
        let fb = self.facing(b)?;
        self.fold_crossing_facing(a, fa, b, fb)
    }

    fn fold_crossing_facing(&self, a: P2<ParamSpace>, fa: f64, b: P2<ParamSpace>, fb: f64) -> Option<f64> {
        if (fa > 0.0) == (fb > 0.0) {
            return None;
        }
        Some(fold_crossing(fa, fb, |t| self.facing(P2::from_vec(a.v + (b.v - a.v) * t)).unwrap_or(0.0)))
    }
}

/// Swift's precondition that the front-most test has a coordinate image,
/// as a value: a visibility made by `sphere_tracing` has none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoImage;

impl std::fmt::Display for NoImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the front-most test needs the coordinate image")
    }
}

impl std::error::Error for NoImage {}

impl HiddenLine {
    /// Hidden-line removal for curves that lie on a parametric surface and
    /// carries its coordinates on it (`clip(_:on:onFolds:onCrease:)`; Swift's
    /// `on_folds` and `on_crease` default to false).
    ///
    /// The runs are `clip`'s -- hidden vertices split a path, and runs of
    /// fewer than two vertices draw nothing -- with one difference: a segment
    /// along which the verdict changes gains a vertex where it changes,
    /// which belongs to the run on its visible side. Visible by sphere tracing,
    /// that vertex is found by bisection on the whole test, so it lies on
    /// the fold or on the outline of whatever came in front, whichever
    /// cut the curves. Visible by the coordinate image, a segment that crosses a
    /// fold is cut at the fold, found on the exact normal, and that vertex is
    /// tested by the front-most test alone, since on the fold the facing is
    /// zero by construction.
    ///
    /// Curves that lie on the folds themselves -- `ParametricSurface::fold_curves`
    /// -- is edge-on everywhere by construction, so its facing is noise about
    /// zero and must not be asked: pass `on_folds` and it is tested by the
    /// sphere tracing or the front-most test alone. Curves on a crease -- a truncation edge --
    /// is tested so too, and `on_crease` says so: there a cut is bisected on
    /// the crease itself, each point moved onto it from the chord.
    ///
    /// Swift tests every vertex, and every cut, in parallel; here in turn,
    /// each independent of the others. Its preconditions -- curves that
    /// carry coordinates, and an image where the surface is not sphere
    /// traced -- are [`ClipOnError`] here.
    pub fn clip_on(
        paths: &PolylineSet<ViewSpace>,
        visibility: &SurfaceVisibility,
        on_folds: bool,
        on_crease: bool,
    ) -> Result<PolylineSet<PageSpace>, ClipOnError> {
        let Some(coords) = &paths.coords else { return Err(ClipOnError::NoCoordinates) };
        let n = coords.len();
        let facing: Vec<Option<f64>> =
            if on_folds { vec![None; n] } else { coords.iter().map(|&c| visibility.facing(c)).collect() };
        let mut kept = Vec::with_capacity(n);
        for k in 0..n {
            kept.push(visibility.is_visible_facing(paths.vertices[k], coords[k], facing[k])?);
        }
        let spheres = visibility.traces_spheres();
        // Every segment along which the verdict changes, and the vertex
        // that goes in at the change. By sphere tracing, bisection on the test
        // itself; by the image, the fold crossing if there is one.
        let mut changes: Vec<usize> = Vec::new();
        for path in 0..paths.count() {
            let (lo, hi) = (paths.offsets[path], paths.offsets[path + 1]);
            for k in (lo + 1)..hi {
                let flips = match (facing[k - 1], facing[k]) {
                    (Some(a), Some(b)) => (a > 0.0) != (b > 0.0),
                    _ => false,
                };
                if kept[k - 1] != kept[k] || (!spheres && flips) {
                    changes.push(k);
                }
            }
        }
        let mut cut: Vec<Option<(P3<ViewSpace>, bool)>> = vec![None; n];
        for &k in &changes {
            let (a, b) = (coords[k - 1], coords[k]);
            cut[k] = if spheres {
                if kept[k - 1] == kept[k] {
                    None
                } else {
                    let (mut lo, mut hi) = (0.0f64, 1.0f64);
                    let mut at = if kept[k - 1] { paths.vertices[k - 1] } else { paths.vertices[k] };
                    while hi - lo > 1e-13 {
                        let mid = 0.5 * (lo + hi);
                        let c = P2::from_vec(a.v + (b.v - a.v) * mid);
                        let c = if on_crease { visibility.onto_crease(c, b.v - a.v) } else { c };
                        let (seen, p) = visibility.test(c, on_folds)?;
                        if seen == kept[k - 1] {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                        if seen {
                            at = p;
                        }
                    }
                    Some((at, true))
                }
            } else {
                match (facing[k - 1], facing[k]) {
                    (Some(fa), Some(fb)) => match visibility.fold_crossing_facing(a, fa, b, fb) {
                        Some(t) => {
                            let (pa, pb) = (paths.vertices[k - 1], paths.vertices[k]);
                            let v = P3::from_vec(pa.v + (pb.v - pa.v) * t);
                            let c = P2::from_vec(a.v + (b.v - a.v) * t);
                            Some((v, visibility.is_front_most(c, v)?))
                        }
                        None => None,
                    },
                    _ => None,
                }
            };
        }

        let mut out: Vec<Vec<P3<PageSpace>>> = Vec::new();
        let mut run: Vec<P3<PageSpace>> = Vec::new();
        let take =
            |v: P3<ViewSpace>, visible: bool, run: &mut Vec<P3<PageSpace>>, out: &mut Vec<Vec<P3<PageSpace>>>| {
                if visible {
                    run.push(P3::from_vec(v.v));
                } else {
                    if run.len() >= 2 {
                        out.push(std::mem::take(run));
                    }
                    run.clear();
                }
            };
        for path in 0..paths.count() {
            let (lo, hi) = (paths.offsets[path], paths.offsets[path + 1]);
            for k in lo..hi {
                if let Some((at, seen)) = cut[k] {
                    take(at, seen, &mut run, &mut out);
                }
                take(paths.vertices[k], kept[k], &mut run, &mut out);
            }
            if run.len() >= 2 {
                out.push(std::mem::take(&mut run));
            }
            run.clear();
        }
        Ok(PolylineSet::from_paths(&out))
    }
}

/// Why curves cannot be clipped on a surface: Swift's preconditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipOnError {
    /// Clipping on a surface needs the curves' surface coordinates.
    NoCoordinates,
    /// The front-most test needs the coordinate image.
    NoImage,
}

impl From<NoImage> for ClipOnError {
    fn from(_: NoImage) -> ClipOnError {
        ClipOnError::NoImage
    }
}

impl std::fmt::Display for ClipOnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClipOnError::NoCoordinates => write!(f, "clipping on a surface needs the curves' surface coordinates"),
            ClipOnError::NoImage => write!(f, "the front-most test needs the coordinate image"),
        }
    }
}

impl std::error::Error for ClipOnError {}
