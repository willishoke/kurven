//! Hidden lines for curves drawn on a parametric surface: the port of the
//! types of `Sources/KurvenCore/Visibility.swift`. What it does is Phase 4's
//! (a parametric surface's visibility), each method a typed refusal; the
//! facing, which `ParametricSurface::facing` already answers, and sphere
//! tracing (`SphereTracing`) are whole.

use crate::hidden_line::HiddenLine;
use crate::occlusion::{OcclusionTest, SphereTracing};
use crate::parametric::{ParametricSurface, PortRefusal};
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
    pub image: Option<SurfaceImage>,
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
        image: SurfaceImage,
        view: Transform<WorldSpace, ViewSpace>,
    ) -> SurfaceVisibility {
        let sight = view.projection_direction();
        let tracing = SphereTracing::new(&surface, sight);
        SurfaceVisibility { surface, image: Some(image), view, sight, tracing }
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
    pub fn footprint(&self, _row: i64, _col: i64) -> Result<Vec2, PortRefusal> {
        // PORT: phase 4 -- per screen axis the smaller of the two one-sided differences of the image's coordinates (`ParamAxis::difference`), and the larger over the two screen axes; zero without an image or a coordinate there.
        Err(PortRefusal { phase: 4, what: "SurfaceVisibility.footprint" })
    }

    /// Whether the front-most surface beside view point `p` is the surface at
    /// `c`.
    pub fn is_front_most(&self, _c: P2<ParamSpace>, _p: P3<ViewSpace>) -> Result<bool, PortRefusal> {
        // PORT: phase 4 -- the nine pixels about p's: true where one holds c's neighbourhood (two footprints and a grid cell per axis), false outside the frame, otherwise true only if every other sheet there lies behind p (`sheet(at:liesBehind:)`).
        Err(PortRefusal { phase: 4, what: "SurfaceVisibility.isFrontMost" })
    }

    /// Whether the sheet through surface coordinate `q` meets the line of
    /// sight through view point `p` behind `p`.
    pub fn sheet_lies_behind(&self, _q: P2<ParamSpace>, _p: P3<ViewSpace>) -> Result<bool, PortRefusal> {
        // PORT: phase 4 -- Newton's method on the map followed by the view, from q, for the coordinate whose view x and y are p's, at most thirty steps within two cells of q (and inside a bounded axis); the sheet's depth there decides.
        Err(PortRefusal { phase: 4, what: "SurfaceVisibility.sheet(at:liesBehind:)" })
    }

    /// Curves at view point `p`, lying on the surface at `c`.
    pub fn is_visible(&self, _p: P3<ViewSpace>, _c: P2<ParamSpace>) -> Result<bool, PortRefusal> {
        // PORT: phase 4 -- hidden where the facing is not positive; otherwise sphere tracing at the map's point where the surface has a neighbourhood, the front-most test otherwise.
        Err(PortRefusal { phase: 4, what: "SurfaceVisibility.isVisible" })
    }

    /// The point of a crease beside `c`, a point of the chord between two of
    /// its vertices: where the facing jumps, across the chord within half
    /// its length either way; `c` where no jump is in reach.
    pub fn onto_crease(&self, _c: P2<ParamSpace>, _chord: Vec2) -> Result<P2<ParamSpace>, PortRefusal> {
        // PORT: phase 4 -- sixty bisections across the chord toward the side whose facing is nearer, kept where a thousandth of the span holds more than half the change a tenth of it does.
        Err(PortRefusal { phase: 4, what: "SurfaceVisibility.ontoCrease" })
    }

    /// Where along the straight segment `a -> b` in coordinates the surface
    /// turns edge-on, as a fraction of the way; None when both ends face the
    /// same way, or the surface has no outside.
    pub fn fold_crossing(&self, _a: P2<ParamSpace>, _b: P2<ParamSpace>) -> Result<Option<f64>, PortRefusal> {
        // PORT: phase 4 -- `fold_crossing` (hidden_line.rs) on the exact facing along the segment, when the ends' facings differ in sign.
        Err(PortRefusal { phase: 4, what: "SurfaceVisibility.foldCrossing" })
    }
}

impl HiddenLine {
    /// Hidden-line removal for curves that lie on a parametric surface and
    /// carries its coordinates on it (`clip(_:on:onFolds:onCrease:)`; Swift's
    /// `on_folds` and `on_crease` default to false).
    pub fn clip_on(
        _paths: &PolylineSet<ViewSpace>,
        _visibility: &SurfaceVisibility,
        _on_folds: bool,
        _on_crease: bool,
    ) -> Result<PolylineSet<PageSpace>, PortRefusal> {
        // PORT: phase 4 -- every vertex tested (`isVisible`, the facing not asked on folds), every segment whose verdict or facing changes cut where it changes (by bisection on the whole test under sphere tracing, moved onto the crease when `on_crease`; at the fold crossing, tested by the front-most test, by the image), runs split as `clip`'s are.
        Err(PortRefusal { phase: 4, what: "HiddenLine.clip(_:on:)" })
    }
}
