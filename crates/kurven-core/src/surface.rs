//! The landscape's surface: |f| and arg f on their grids, the truncation, and
//! what is derived from them. The port of `Sources/KurvenCore/Surface.swift`.
//!
//! Phase 3 begins with [`ContourRefine`], the one type the refiner
//! (`kurven-landscape`'s `ContourRefiner`) and the surface share. In Swift
//! `KurvenCore` has the grids and not the function, and `KurvenLandscape`
//! builds a `ContourRefine` of closures; the same split holds here, so there
//! is no edge from `kurven-core` to `kurven-landscape`, and no crate cycle.

use std::ops::RangeInclusive;
use std::sync::Arc;

use crate::manifest::ContourField;
use crate::spaces::{DomainSpace, P2};

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
