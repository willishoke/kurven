//! Contours placed by the function rather than by the grid. The port of
//! `Sources/KurvenLandscape/Refine.swift`, but for `refiner(for: bundle)` and
//! `benchmarkRefinement`, which need a `KurvenBundle` (a later wave).
//!
//! Marching squares puts a contour vertex where *linear interpolation* between
//! two samples crosses the level, which is off the true level set by about
//! h² |f''| / |f'| -- nothing where f is nearly linear across a cell, and
//! growing like 1/r toward a pole or a zero, which is exactly where the plates
//! are densest. The Python pipeline answered that with a second, finer grid
//! in rectangles around the steep places, because there every extra sample
//! was scipy time. Here the function is a call, so the answer is to ask it:
//!
//!  1. **Snap.** Every vertex lies on a grid edge whose two samples bracket the
//!     level. A one-dimensional root find along that edge, against f itself,
//!     moves the vertex to the exact crossing, by Brent's method (`Root::brent`).
//!  2. **Subdivide.** Between two snapped vertices the contour may curve away
//!     from the chord. The residual at the chord's midpoint, divided by the
//!     gradient there, estimates how far off the chord is; where that exceeds
//!     a fraction of a cell, a new vertex is solved for along the normal and
//!     the two halves are checked in turn. Density ends up proportional to
//!     curvature, continuously, and with no chunk boundaries to stitch because
//!     there is only one contour pass.
//!
//! The grid still decides which contours exist and roughly where; f decides
//! exactly where. The lift stays the grid's, so the curves stay on the drawn
//! surface -- see `Surface.derive`.
//!
//! Phase contours wrap. Where arg f passes from π to -π, marching squares on
//! the wrapped grid interpolates straight through the jump and emits a crossing
//! for *every* level between: a bundle of spurious segments along each wrap
//! line, in the Python plates as much as here (a third of the gamma plate's
//! phase vertices, measured). Two samples cannot tell such a vertex from a
//! real one beside a zero, where the phase turns a full circle within a cell
//! and one edge can legitimately span more than π of it. So the phase is
//! tracked through the edge's midpoint, each half the short way round, and a
//! vertex is a crossing only where that path passes the level (`bracket`);
//! otherwise it is dropped and the path split there. Residuals of the phase
//! are taken the short way round throughout, so a level near π is as near to
//! a sample at -π as it is. A chord midpoint whose residual is a jump is
//! likewise left alone rather than solved toward the wrong branch.
//!
//! Every phase meets at a zero of f and at a pole, and a phase contour ends
//! at one or the other. The grid stops it at the last cell edge before; the
//! refiner, which has f, carries each run whose end lies within a cell or so
//! of a zero or a pole -- found by Newton from the end, on f or on 1/f -- to
//! the point itself (`ended`, `trace`).
//!
//! Swift runs the paths of a level, and the rows of a node search, with
//! `concurrentPerform`; each iteration writes only its own slot, so here they
//! run in turn, with the same answer.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::f64::consts::PI;
use std::ops::RangeInclusive;
use std::sync::Arc;
use std::time::Duration;

use kurven_core::bisection::{bisect, bisect_points};
use kurven_core::contour;
use kurven_core::manifest::{ContourField, Domain};
use kurven_core::scalar_field::short_way;
use kurven_core::spaces::{DomainSpace, P2};
use kurven_core::surface::ContourRefine;
use kurven_math::{Compiled, Complex, ExpressionError, Node, Root, real, swift};

use crate::landscape::{GridShape, NativeLandscape};

type P = P2<DomainSpace>;
/// Swift's `stopping` closures: the handoff disc a point lies in, if any.
type Stopping<'a> = Option<&'a dyn Fn(P) -> Option<usize>>;

/// Which scalar field of f a contour is solved on: Swift passes the
/// refiner's own closures (`magnitude`, `phase`, `turnedPhase`) as `g`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scalar {
    Magnitude,
    Phase,
    /// arg(-f): the phase turned a half turn, whose zero level is f's cut.
    TurnedPhase,
}

/// A run of placed vertices; whether each end is the curve's own, to be
/// followed on f and carried to a zero or pole, or a cut at the ceiling;
/// and, for an end cut at the handoff disc of a singular point, which
/// point, for its ray to be spliced on.
#[derive(Debug, Clone)]
pub struct Piece {
    pub run: Vec<P>,
    pub ends: Ends,
    pub met: Met,
}

/// Swift's `(start: Bool, end: Bool)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ends {
    pub start: bool,
    pub end: bool,
}

/// Swift's `(start: Int?, end: Int?)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Met {
    pub start: Option<usize>,
    pub end: Option<usize>,
}

/// A zero or pole of f with its normal form there: f(z) ≈ coefficient ·
/// (z − point)^order, the order negative at a pole.
///
/// This is the chart in which the curves round the point are trivial. In
/// w = f(z) a line of constant phase θ is the ray arg w = θ and a line
/// of constant magnitude is a circle, so near the point the drawing is
/// read off the normal form and carried back by Newton on f, and the
/// grid, which cannot see inside the cell the point lies in, is not
/// consulted there. For order k there are k rays at every level,
/// leaving at the angles (θ − arg coefficient + 2πj) / k, and the same
/// formula with k negative gives a pole's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SingularPoint {
    pub point: P,
    pub order: i64,
    pub coefficient: Complex,
}

/// One ray of a singular point's fan at a level, from the point out to
/// the handoff radius; `handoff` is false when it ended before that, on
/// the window's edge, at the ceiling or on another singular point, and
/// is then complete as it stands.
#[derive(Debug, Clone)]
struct Ray {
    run: Vec<P>,
    point: usize,
    handoff: bool,
}

/// The nodes x + i dx + (y + j dy)i, for i < columns and j < rows: the
/// grid's, or a finer grid's over one of its cells.
#[derive(Debug, Clone, Copy)]
struct Nodes {
    x: f64,
    y: f64,
    dx: f64,
    dy: f64,
    columns: usize,
    rows: usize,
}

impl Nodes {
    fn node(&self, i: usize, j: usize) -> P {
        P::new(self.x + i as f64 * self.dx, self.y + j as f64 * self.dy)
    }

    /// The cell whose lower corner is node (i, j), `n` steps a side.
    fn cell(&self, i: usize, j: usize, n: usize) -> Nodes {
        Nodes {
            x: self.x + i as f64 * self.dx,
            y: self.y + j as f64 * self.dy,
            dx: self.dx / n as f64,
            dy: self.dy / n as f64,
            columns: n + 1,
            rows: n + 1,
        }
    }
}

/// A magnitude line's chord is sought past these zeros (`subdivide`): the
/// run being placed, and the zeros.
struct Past<'a> {
    run: &'a [P],
    zeros: &'a [P],
}

/// The bracketed sub-edge `bracket` answers.
struct Bracket {
    a: P,
    b: P,
    ra: f64,
    rb: f64,
}

/// Contours placed by the function rather than by the grid (the module's
/// doc says how).
#[derive(Clone)]
pub struct ContourRefiner {
    /// f itself, for the roots the phase contours end at. Swift keeps f as
    /// four closures (`value`, `magnitude`, `phase`, `turnedPhase`); here
    /// they are the methods of those names over this one tape.
    compiled: Arc<Compiled>,
    pub grid: Domain,
    pub width: i64,
    pub height: i64,
    /// Subdivide while the chord is estimated to be further than this fraction
    /// of a cell from the level set.
    pub tolerance: f64,
    /// How many times one grid-level segment may be halved.
    pub max_depth: i64,
    /// Whether the curves are drawn at height |f|, as a landscape's is, so that a
    /// phase chord is also split where its height sags from the chord's
    /// (`subdivide`). Curves on a parametric surface have no height to sag in.
    pub lifted: bool,
    /// The height above which the curves are cut away, past which no line is
    /// followed or seeded: set for the duration of a `refine` call from its
    /// ceiling.
    ceiling: f64,
    /// The span of Re z a magnitude line is kept to where it is sought past a
    /// zero (`subdivide`): a band's, for a run of a banded truncation edge.
    /// Set for the duration of a `refine` call, like the ceiling.
    span: RangeInclusive<f64>,
}

impl std::fmt::Debug for ContourRefiner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContourRefiner")
            .field("expression", &self.compiled.expression())
            .field("grid", &self.grid)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("tolerance", &self.tolerance)
            .field("max_depth", &self.max_depth)
            .field("lifted", &self.lifted)
            .finish()
    }
}

impl ContourRefiner {
    /// How far from a run's end a zero is looked for, in cells. The grid's
    /// contour can stop two cells short where the zero's own cells pair
    /// their crossings the other way.
    pub const REACH: f64 = 3.0;

    /// How far from a singular point its rays are followed before the
    /// grid's curves take over, in cells: the handoff radius. The same
    /// distance a run's end is looked for a zero or pole from, so a piece
    /// cut back from the disc is never also carried onto its point.
    pub const HANDOFF: f64 = ContourRefiner::REACH;

    /// A refiner for `compiled` on a grid of `width` × `height` nodes over
    /// `domain`, with Swift's defaults: a tolerance of 0.02 cells, chords
    /// halved at most 5 times, lifted. Set the public fields to change them.
    pub fn new(compiled: Compiled, domain: Domain, width: usize, height: usize) -> ContourRefiner {
        ContourRefiner::with(compiled, domain, width, height, 0.02, 5, true)
    }

    /// Swift's `init(_:domain:width:height:tolerance:maxDepth:lifted:)`.
    pub fn with(
        compiled: Compiled,
        domain: Domain,
        width: usize,
        height: usize,
        tolerance: f64,
        max_depth: i64,
        lifted: bool,
    ) -> ContourRefiner {
        ContourRefiner {
            compiled: Arc::new(compiled),
            grid: domain,
            width: width as i64,
            height: height as i64,
            tolerance,
            max_depth,
            lifted,
            ceiling: f64::INFINITY,
            span: ContourRefine::EVERYWHERE,
        }
    }

    /// The expression f was compiled from, for a fork to compile again.
    pub fn node(&self) -> &Node {
        &self.compiled.node
    }

    /// f(z).
    pub fn value(&self, z: Complex) -> Complex {
        self.compiled.call(z)
    }

    /// |f|, clamped at `NativeLandscape::HUGE` and put there where f is not
    /// finite, as the grid was sampled.
    pub fn magnitude(&self, z: Complex) -> f64 {
        let v = self.value(z);
        if v.is_finite() { swift::min(v.magnitude(), NativeLandscape::HUGE) } else { NativeLandscape::HUGE }
    }

    /// arg f. The phase is undefined where f is not finite and where it has
    /// underflowed to zero: the argument of 0 + 0i is a number, but not
    /// one of f's. NaN there, which no bracket, snap or chord accepts,
    /// so beside a singularity the lines end where f's phase does.
    pub fn phase(&self, z: Complex) -> f64 {
        let v = self.value(z);
        if v.is_finite() && v.magnitude() > 0.0 { v.argument() } else { f64::NAN }
    }

    /// arg(-f): the phase turned a half turn, whose zero level is f's cut.
    pub fn turned_phase(&self, z: Complex) -> f64 {
        let v = self.value(z);
        if v.is_finite() && v.magnitude() > 0.0 { (-v).argument() } else { f64::NAN }
    }

    /// The scalar field `g` at z.
    pub fn g(&self, g: Scalar, z: Complex) -> f64 {
        match g {
            Scalar::Magnitude => self.magnitude(z),
            Scalar::Phase => self.phase(z),
            Scalar::TurnedPhase => self.turned_phase(z),
        }
    }

    fn at(&self, g: Scalar, p: P) -> f64 {
        self.g(g, Complex::new(p.x(), p.y()))
    }

    fn dx(&self) -> f64 {
        self.grid.real.length() / (self.width - 1).max(1) as f64
    }

    fn dy(&self) -> f64 {
        self.grid.imag.length() / (self.height - 1).max(1) as f64
    }

    /// The cell size the tolerances are measured against.
    pub fn cell(&self) -> f64 {
        swift::max(self.dx().abs(), self.dy().abs())
    }

    /// An angle brought into [-π, π]: the short way round.
    pub fn wrapped(angle: f64) -> f64 {
        short_way(angle, 2.0 * PI)
    }

    /// How far a sample is from the level: for the phase, the short way round.
    fn residual(&self, sample: f64, level: f64, cut: bool) -> f64 {
        if cut { ContourRefiner::wrapped(sample - level) } else { sample - level }
    }

    /// Whether a domain point is inside the grid's window, to rounding: the
    /// zeros of a function sampled on a half-plane sit on its edge.
    fn in_window(&self, p: P) -> bool {
        let (r, i, slack) = (self.grid.real, self.grid.imag, 1e-9 * self.cell());
        p.x() >= swift::min(r.lo, r.hi) - slack
            && p.x() <= swift::max(r.lo, r.hi) + slack
            && p.y() >= swift::min(i.lo, i.hi) - slack
            && p.y() <= swift::max(i.lo, i.hi) + slack
    }

    /// The scalar field a contour of `f` is solved on. Swift traps
    /// (`preconditionFailure`) for a field that is not f's (u, v, the
    /// height, an isophote); here that is None, and the callers refuse it.
    pub fn field(f: ContourField) -> Option<Scalar> {
        match f {
            ContourField::Magnitude => Some(Scalar::Magnitude),
            ContourField::Phase => Some(Scalar::Phase),
            ContourField::U | ContourField::V | ContourField::Height | ContourField::Isophote => None,
        }
    }

    /// The scalar field and level a contour is solved on. A phase level of
    /// ±π is the branch cut, which arg f as a scalar field
    /// (`Surface.field(of:refine:ceiling:)`) hands on as the zero level of
    /// arg(-f) (`Contour.cut`), and is placed on that.
    pub fn field_level(f: ContourField, level: f64) -> Option<(Scalar, f64)> {
        if f == ContourField::Phase && contour::is_cut(level) {
            return Some((Scalar::TurnedPhase, 0.0));
        }
        Some((ContourRefiner::field(f)?, level))
    }

    /// `Surface.derive`'s hook, and `Surface.topEdge`'s and
    /// `Heightfield.foldCurves`'s: the contours placed, |f| for the top edge's
    /// minima and the zeros' heights, and the zeros the fold curves are
    /// carried to. The magnitude is the one the grid was sampled with,
    /// clamped at `NativeLandscape::HUGE` beside a pole. Swift's `refine`
    /// property.
    ///
    /// Remembering (`ContourRefine::remembering`): a bundle carries this hook
    /// through every restyle of its grids, and a restyle asks again for
    /// curves it has already placed.
    pub fn refine_hook(&self) -> ContourRefine {
        self.hook(self.singular_points()).remembering()
    }

    /// The hook over these singular points, forking into a refiner of its
    /// own: f compiled again from its tree, so the fork's closures share
    /// nothing with these (`ContourRefine.fork`).
    pub fn hook(&self, points: Vec<SingularPoint>) -> ContourRefine {
        let me = Arc::new(self.clone());
        let points = Arc::new(points);
        let zeros: Vec<P> = points.iter().filter(|s| s.order > 0).map(|s| s.point).collect();
        let rings: Option<kurven_core::surface::Rings> = if zeros.is_empty() {
            None
        } else {
            let me = me.clone();
            Some(Arc::new(move |level, existing| me.seeded_rings(level, &zeros, existing)))
        };
        let contours: kurven_core::surface::Contours = {
            let (me, points) = (me.clone(), points.clone());
            Arc::new(move |field, level, paths, ceiling, span| me.refine(field, level, paths, &points, ceiling, span))
        };
        let magnitude: kurven_core::surface::Magnitude = {
            let me = me.clone();
            Arc::new(move |p| me.magnitude(Complex::new(p.x(), p.y())))
        };
        let phase: kurven_core::surface::Phase = {
            let me = me.clone();
            Arc::new(move |p| me.phase(Complex::new(p.x(), p.y())))
        };
        let zero: kurven_core::surface::Zero = {
            let me = me.clone();
            Arc::new(move |p| me.zero_near(p))
        };
        let fork: kurven_core::surface::Fork = {
            let me = me.clone();
            Arc::new(move || {
                let mut again = (*me).clone();
                again.compiled = Arc::new(Compiled::new(me.compiled.node.clone()));
                again.ceiling = f64::INFINITY;
                again.span = ContourRefine::EVERYWHERE;
                again.hook((*points).clone())
            })
        };
        ContourRefine {
            contours,
            rings,
            magnitude: Some(magnitude),
            phase: Some(phase),
            zero: Some(zero),
            tolerance: self.tolerance,
            fork: Some(fork),
        }
    }

    /// The grid's paths at one level of one field, placed on f. The rings
    /// round the zeros that the grid could not see are the hook's apart
    /// (`seeded_rings`), asked once a level.
    ///
    /// A phase path is placed only where it can be drawn: along the
    /// stretches with a vertex at or under `ceiling`, the height above which
    /// its curves are cut away (`ContourRefine::contours`). A magnitude line is
    /// level, and wholly below the truncation or wholly over it.
    ///
    /// Swift's defaults: no points, an infinite ceiling, the span
    /// `ContourRefine::EVERYWHERE`. A field that is not f's, on which Swift
    /// traps, has its paths handed back as the grid placed them.
    pub fn refine(
        &self,
        field: ContourField,
        theta: f64,
        paths: &[Vec<P>],
        points: &[SingularPoint],
        ceiling: f64,
        span: RangeInclusive<f64>,
    ) -> Vec<Vec<P>> {
        let Some((g, level)) = ContourRefiner::field_level(field, theta) else { return paths.to_vec() };
        let cut = field == ContourField::Phase;
        let ceiling = if cut { ceiling } else { f64::INFINITY };
        // Nothing is followed or seeded above this call's ceiling, nor sought
        // outside its span: a copy of the refiner carries them. (Swift
        // recurses on the copy when they differ, which never ends for a NaN
        // ceiling; the copy is always made here.)
        let mut under = self.clone();
        under.ceiling = ceiling;
        under.span = span;
        under.refine_under(theta, paths, points, g, level, cut, ceiling)
    }

    #[allow(clippy::too_many_arguments)]
    fn refine_under(
        &self,
        theta: f64,
        paths: &[Vec<P>],
        points: &[SingularPoint],
        g: Scalar,
        level: f64,
        cut: bool,
        ceiling: f64,
    ) -> Vec<Vec<P>> {
        let zeros: Vec<P> = points.iter().filter(|s| s.order > 0).map(|s| s.point).collect();
        // Every path snapped and subdivided into its pieces.
        let all: Vec<Piece> = paths.iter().flat_map(|path| self.pieces(path, level, g, cut, ceiling, &zeros)).collect();
        if !cut {
            return all.into_iter().map(|p| p.run).collect();
        }
        // The fan at each zero and pole, from the point's normal form: its
        // rays at this level, each followed from the point out to the
        // handoff radius. The grid's pieces are cut back from those discs
        // and the rays spliced on, so the curves nearest a singular point are
        // f's, with as many rays as the point's order says, whatever the
        // grid resolved there.
        let rays: Vec<Ray> =
            (0..points.len()).flat_map(|i| self.seeded_rays(i, points, theta, level, g, ceiling)).collect();
        // Swift collects these in a `Set` and walks it in its per-process
        // order; here in ascending order of the point, so which disc a point
        // in two discs is in is fixed (crates/FINDINGS.md).
        let seeded: BTreeSet<usize> = rays.iter().filter(|r| r.handoff).map(|r| r.point).collect();
        let discs: Vec<(usize, P)> = seeded.iter().map(|&i| (i, points[i].point)).collect();
        let radius = ContourRefiner::HANDOFF * self.cell();
        let stopping = |p: P| discs.iter().find(|d| apart(p, d.1) < radius).map(|d| d.0);
        // A phase piece is followed on f beyond its ends, to the zero or
        // pole it runs into, to the window's edge, or to a disc. First once
        // over: a piece lying along another is a copy of a stretch of it,
        // and following it would copy the rest.
        let trimmed: Vec<Piece> = all.iter().flat_map(|p| self.trimmed(p, &discs)).collect();
        let kept = self.drawn_once(trimmed, |p| &p.run);
        let followed: Vec<Piece> = kept
            .into_iter()
            .map(|piece| {
                let run = self.ended(&piece.run, level, g, piece.ends);
                self.traced(Piece { run, ends: piece.ends, met: piece.met }, level, g, Some(&stopping))
            })
            .collect();
        let joined = self.spliced(&rays, followed, points, level, g, &stopping);
        // And once more: two pieces of one curve, split where a chord could
        // not be placed, have each been followed over the other's ground.
        self.drawn_once(joined, |r| r)
    }

    /// The runs drawn once: a run every vertex of which lies within a tenth
    /// of a cell of another run -- a longer one, or one as long and earlier
    /// -- is a copy of a stretch of it, and is dropped.
    ///
    /// A stretch comes to be placed twice in two ways. Through the cell of a
    /// pole, marching squares links the ray at a level into the bundle of
    /// spurious crossings along the cut, and beside the pole those crossings
    /// lie on edges the ray crosses too, so the snap puts them on the ray.
    /// And a chord that could not be placed splits a run in two, after which
    /// each half is followed on f over the other's ground. Distinct curves at
    /// one level meet only at a zero, a pole or a saddle, so a run lying
    /// wholly along another is never a curve of its own.
    pub fn drawn_once<T>(&self, items: Vec<T>, run: impl Fn(&T) -> &Vec<P>) -> Vec<T> {
        if items.len() <= 1 {
            return items;
        }
        let cell = self.cell();
        let near = 0.1 * cell;
        // Swift's `Int` traps on a vertex that is not finite; such a vertex
        // is binned in one cell of its own here (crates/FINDINGS.md).
        let key = |p: P| -> (i64, i64) {
            (
                swift::int((p.x() / cell).floor()).unwrap_or(i64::MIN),
                swift::int((p.y() / cell).floor()).unwrap_or(i64::MIN),
            )
        };
        // Every segment, binned by the cells its ends fall in. A segment is
        // at most about a cell long, so the three-by-three block of cells
        // round a vertex holds every segment within a tenth of a cell of it.
        let runs: Vec<&Vec<P>> = items.iter().map(&run).collect();
        let mut segments: HashMap<(i64, i64), Vec<(usize, usize)>> = HashMap::new();
        for (i, r) in runs.iter().enumerate() {
            for k in 0..r.len().saturating_sub(1) {
                let (a, b) = (key(r[k]), key(r[k + 1]));
                segments.entry(a).or_default().push((i, k));
                if b != a {
                    segments.entry(b).or_default().push((i, k));
                }
            }
        }
        let covered = |i: usize| -> bool {
            let r = runs[i];
            r.iter().all(|&p| {
                let c = key(p);
                for dx in -1i64..=1 {
                    for dy in -1i64..=1 {
                        let Some(list) = segments.get(&(c.0.wrapping_add(dx), c.1.wrapping_add(dy))) else {
                            continue;
                        };
                        for &(run, at) in list {
                            if run == i {
                                continue;
                            }
                            let other = runs[run];
                            if !(other.len() > r.len() || (other.len() == r.len() && run < i)) {
                                continue;
                            }
                            if distance(p, other[at], other[at + 1]) <= near {
                                return true;
                            }
                        }
                    }
                }
                false
            })
        };
        let keep: Vec<bool> = (0..items.len()).map(|i| !covered(i)).collect();
        items.into_iter().zip(keep).filter(|(_, k)| *k).map(|(t, _)| t).collect()
    }

    // MARK: rings the grid cannot see

    /// The zeros and the poles of f in the window. Every node that is the
    /// least of |f| among its neighbours is taken to the zero beside it by
    /// Newton, and every one that is the greatest to the pole beside it by
    /// Newton on 1/f: |f| has no local minimum but at a zero and no local
    /// maximum but at a pole (the maximum principle), and a node on the
    /// window's edge that is one with no point beside it fails Newton's test.
    ///
    /// The nodes do not see every point. A zero between two poles within a
    /// cell has no least node beside it, and from the least node beside a
    /// zero Newton can step out of reach past a pole. So what was found is
    /// held to the argument principle: the winding number of f round a cell
    /// is the number of zeros in it less the number of poles, with order
    /// (`unaccounted`). A cell where it is not what was found in the cell is
    /// searched the same way on a grid nine times finer, and a cell of that
    /// grid in turn: an odd number, so that neither a cell's centre nor the
    /// middle of an edge, where a symmetric arrangement puts a point, is a
    /// node, from which Newton, on the point to rounding, could not show the
    /// fall its test asks for. Found once per refiner.
    pub fn zeros_and_poles(&self) -> (Vec<P>, Vec<P>) {
        let (mut zeros, mut poles) = (Vec::new(), Vec::new());
        let nodes = Nodes {
            x: self.grid.real.lo,
            y: self.grid.imag.lo,
            dx: self.dx(),
            dy: self.dy(),
            columns: self.width.max(0) as usize,
            rows: self.height.max(0) as usize,
        };
        self.search(nodes, 2, &mut zeros, &mut poles);
        (zeros, poles)
    }

    /// The zeros and poles found from `nodes`, added to those already found,
    /// and, `depth` times over, from a finer grid over each cell they do
    /// not account for.
    fn search(&self, nodes: Nodes, depth: i64, zeros: &mut Vec<P>, poles: &mut Vec<P>) {
        if nodes.columns < 2 || nodes.rows < 2 {
            return;
        }
        let mut values = Vec::with_capacity(nodes.columns * nodes.rows);
        for j in 0..nodes.rows {
            for i in 0..nodes.columns {
                let p = nodes.node(i, j);
                values.push(self.value(Complex::new(p.x(), p.y())));
            }
        }
        let (minima, maxima) = self.extremes(&values, &nodes);
        self.rooted(&minima, |p| self.zero_near(p), zeros);
        self.rooted(&maxima, |p| self.pole_near(p), poles);
        if depth <= 0 {
            return;
        }
        for (i, j) in self.unaccounted(&values, &nodes, zeros, poles) {
            self.search(nodes.cell(i, j, 9), depth - 1, zeros, poles);
        }
    }

    /// The zeros and poles of f in the window, each with its order and
    /// leading coefficient. The order is the winding number of f round a
    /// small circle about the point, small against the distance to any
    /// other; the coefficient is the mean of f(z) / (z − point)^order round
    /// a smaller one. The circle's radius is half a cell, or a quarter of the
    /// way to the nearest other point, where it holds the point alone as far
    /// as winding numbers can tell: where the winding number round one an
    /// eighth the size is another, points the search did not find are inside
    /// -- an accumulation of zeros, as sin(1/z)'s at 0 -- and the circle is
    /// shrunk eightfold until two in turn agree. A point round which they
    /// never do, or whose winding number does not agree with the way it was
    /// found, or round which f is not finite and nonzero, is left out, and
    /// its lines are then carried onto it as before (`ended`).
    pub fn singular_points(&self) -> Vec<SingularPoint> {
        let (zeros, poles) = self.zeros_and_poles();
        let found: Vec<(P, i64)> = zeros.iter().map(|&z| (z, 1)).chain(poles.iter().map(|&p| (p, -1))).collect();
        let cell = self.cell();
        let mut out = Vec::new();
        for (i, &(p, sign)) in found.iter().enumerate() {
            let mut nearest = f64::INFINITY;
            for (j, &(q, _)) in found.iter().enumerate() {
                if j != i {
                    nearest = swift::min(nearest, apart(p, q));
                }
            }
            let mut r = swift::min(0.5 * cell, 0.25 * nearest);
            if !(r > 0.0) {
                continue;
            }
            let mut read = self.winding_number(p, r);
            let mut agreed = false;
            loop {
                let inner = self.winding_number(p, r / 8.0);
                if inner == read {
                    agreed = true;
                    break;
                }
                r /= 8.0;
                read = inner;
                if !(r >= 1e-6 * cell) {
                    break;
                }
            }
            let Some(order) = read else { continue };
            if !agreed || order == 0 || (order > 0) != (sign > 0) {
                continue;
            }
            let c = Complex::new(p.x(), p.y());
            let rc = swift::min(1e-2 * cell, 0.5 * r);
            let (mut sum, mut count) = (Complex::ZERO, 0usize);
            for k in 0..8 {
                let t = 2.0 * PI * (k as f64 + 0.5) / 8.0;
                let w = Complex::new(rc * real::cos(t), rc * real::sin(t));
                let v = self.value(c + w);
                if !(v.is_finite() && v.magnitude() > 0.0) {
                    continue;
                }
                let mut power = Complex::ONE;
                for _ in 0..order.unsigned_abs() {
                    power *= w;
                }
                let a = if order > 0 { v / power } else { v * power };
                if !a.is_finite() {
                    continue;
                }
                sum += a;
                count += 1;
            }
            if count == 0 {
                continue;
            }
            out.push(SingularPoint {
                point: p,
                order,
                coefficient: Complex::new(sum.re / count as f64, sum.im / count as f64),
            });
        }
        out
    }

    /// The winding number of f round the circle of radius `r` about `p`,
    /// read at 64 points, an arc between two that turns more than a quarter
    /// turn halved as a cell's edge is (`turn`), or None where f is not
    /// finite and nonzero on it or the turn cannot be followed.
    pub fn winding_number(&self, p: P, r: f64) -> Option<i64> {
        let c = Complex::new(p.x(), p.y());
        let at = |t: f64| -> f64 {
            let v = self.value(c + Complex::new(r * real::cos(t), r * real::sin(t)));
            if v.is_finite() && v.magnitude() > 0.0 { v.argument() } else { f64::NAN }
        };
        /// The turn of arg f along the arc from t0 to t1.
        fn turn(at: &dyn Fn(f64) -> f64, t0: f64, a0: f64, t1: f64, a1: f64, depth: i64) -> Option<f64> {
            if !(a0.is_finite() && a1.is_finite()) {
                return None;
            }
            let d = ContourRefiner::wrapped(a1 - a0);
            if d.abs() <= PI / 2.0 {
                return Some(d);
            }
            if depth <= 0 {
                return None;
            }
            let tm = 0.5 * (t0 + t1);
            let am = at(tm);
            let first = turn(at, t0, a0, tm, am, depth - 1)?;
            let second = turn(at, tm, am, t1, a1, depth - 1)?;
            Some(first + second)
        }
        let n = 64;
        let (mut total, mut t0, mut a0) = (0.0, 0.0, at(0.0));
        for k in 1..=n {
            let t1 = 2.0 * PI * k as f64 / n as f64;
            let a1 = at(t1);
            total += turn(&at, t0, a0, t1, a1, 10)?;
            t0 = t1;
            a0 = a1;
        }
        swift::int((total / (2.0 * PI)).round())
    }

    /// The points `find` takes `nodes` to, added to `found` unless already
    /// in it.
    fn rooted(&self, nodes: &[P], find: impl Fn(P) -> Option<P>, found: &mut Vec<P>) {
        let cell = self.cell();
        for &node in nodes {
            let Some(z) = find(node) else { continue };
            if found.iter().any(|&f| apart(f, z) <= 1e-6 * cell) {
                continue;
            }
            found.push(z);
        }
    }

    /// The nodes that are the least, and the greatest, of |f| among their
    /// neighbours, from f's `values` there. |f| is clamped at `HUGE` beside a
    /// pole, as `magnitude` clamps it, so a pole's node is a greatest one.
    fn extremes(&self, values: &[Complex], nodes: &Nodes) -> (Vec<P>, Vec<P>) {
        let (width, height) = (nodes.columns, nodes.rows);
        let values: Vec<f64> =
            values
                .iter()
                .map(|v| {
                    if v.is_finite() { swift::min(v.magnitude(), NativeLandscape::HUGE) } else { NativeLandscape::HUGE }
                })
                .collect();
        let (mut minima, mut maxima) = (Vec::new(), Vec::new());
        for j in 0..height {
            for i in 0..width {
                let v = values[j * width + i];
                if !v.is_finite() {
                    continue;
                }
                let (mut least, mut greatest) = (true, true);
                for dj in -1i64..=1 {
                    for di in -1i64..=1 {
                        if di == 0 && dj == 0 {
                            continue;
                        }
                        let (ni, nj) = (i as i64 + di, j as i64 + dj);
                        if ni < 0 || ni >= width as i64 || nj < 0 || nj >= height as i64 {
                            continue;
                        }
                        let u = values[nj as usize * width + ni as usize];
                        if u < v {
                            least = false;
                        }
                        if u > v {
                            greatest = false;
                        }
                    }
                }
                if least {
                    minima.push(nodes.node(i, j));
                }
                if greatest && v > 0.0 {
                    maxima.push(nodes.node(i, j));
                }
            }
        }
        (minima, maxima)
    }

    /// The cells of `nodes` that the zeros and poles found do not account
    /// for: those round which the winding number of f, from its `values` at
    /// the nodes, is not the sum of the orders of the points found in the
    /// cell, or cannot be read. Each order is read round a circle of radius
    /// a thousandth of a cell, or a quarter of the way to the nearest other
    /// point, so that a point found does not stand for one beside it that
    /// was not.
    fn unaccounted(&self, values: &[Complex], nodes: &Nodes, zeros: &[P], poles: &[P]) -> Vec<(usize, usize)> {
        let (width, height) = (nodes.columns, nodes.rows);
        let phases: Vec<f64> =
            values.iter().map(|v| if v.is_finite() && v.magnitude() > 0.0 { v.argument() } else { f64::NAN }).collect();
        // The turn of arg f along every edge, from node k to the next one
        // across and the next one up; NaN where it cannot be followed.
        let mut across = vec![f64::NAN; width * height];
        let mut up = vec![f64::NAN; width * height];
        for j in 0..height {
            for i in 0..width {
                let (k, p) = (j * width + i, nodes.node(i, j));
                if i + 1 < width {
                    across[k] = self.turn(p, phases[k], nodes.node(i + 1, j), phases[k + 1], 10).unwrap_or(f64::NAN);
                }
                if j + 1 < height {
                    up[k] = self.turn(p, phases[k], nodes.node(i, j + 1), phases[k + width], 10).unwrap_or(f64::NAN);
                }
            }
        }
        // The orders of what was found in each cell, summed, and the cells
        // where one cannot be read. A point whose order is not the
        // winding number round a circle of radius half a step, as
        // `singular_points` reads it on the grid, has another inside the
        // circle, and the cells the circle meets are searched.
        let all: Vec<P> = zeros.iter().chain(poles.iter()).copied().collect();
        let step = swift::max(nodes.dx.abs(), nodes.dy.abs());
        let cell = self.cell();
        let mut found: HashMap<(i64, i64), i64> = HashMap::new();
        let mut unread: HashSet<(i64, i64)> = HashSet::new();
        // Keyed (j, i), so that it iterates as Swift's result is sorted.
        let mut out: BTreeSet<(i64, i64)> = BTreeSet::new();
        let (w1, h1) = (width as i64 - 1, height as i64 - 1);
        for (k, &p) in all.iter().enumerate() {
            let i = ((p.x() - nodes.x) / nodes.dx).floor();
            let j = ((p.y() - nodes.y) / nodes.dy).floor();
            if !(i >= 0.0 && i < w1 as f64 && j >= 0.0 && j < h1 as f64) {
                continue;
            }
            let mut nearest = f64::INFINITY;
            for (l, &q) in all.iter().enumerate() {
                if l != k {
                    nearest = swift::min(nearest, apart(p, q));
                }
            }
            let key = (i as i64, j as i64);
            let order = self.winding_number(p, swift::min(1e-3 * cell, 0.25 * nearest));
            match order {
                Some(o) => *found.entry(key).or_insert(0) += o,
                None => {
                    unread.insert(key);
                }
            }
            if self.winding_number(p, swift::min(0.5 * step, 0.25 * nearest)) == order {
                continue;
            }
            for dj in -1..=1 {
                for di in -1..=1 {
                    if (0..w1).contains(&(key.0 + di)) && (0..h1).contains(&(key.1 + dj)) {
                        out.insert((key.1 + dj, key.0 + di));
                    }
                }
            }
        }
        // Counter-clockwise in i and j is clockwise in the plane where one
        // axis runs backwards.
        let sense = if nodes.dx * nodes.dy > 0.0 { 1.0 } else { -1.0 };
        // Nothing is read where f is not finite and nonzero at a corner:
        // where it overflows, say, which no point explains.
        let read = |k: usize| {
            phases[k].is_finite()
                && phases[k + 1].is_finite()
                && phases[k + width].is_finite()
                && phases[k + width + 1].is_finite()
        };
        // The winding number round the cell whose lower corner is node k,
        // the turns arg f makes round it, or None where the turn along one
        // of its edges cannot be read.
        let turns = |k: usize| -> Option<i64> {
            let angle = across[k] + up[k + 1] - across[k + width] - up[k];
            if angle.is_finite() { swift::int((sense * angle / (2.0 * PI)).round()) } else { None }
        };
        for j in 0..height.saturating_sub(1) {
            for i in 0..width.saturating_sub(1) {
                if !read(j * width + i) {
                    continue;
                }
                let w = turns(j * width + i);
                let key = (i as i64, j as i64);
                if w.is_none() || (w != Some(0) && found.get(&key).copied() != w) {
                    out.insert((key.1, key.0));
                }
            }
        }
        // And where a point was found but the winding number is zero, or an
        // order cannot be read.
        let keys: BTreeSet<(i64, i64)> = found.keys().chain(unread.iter()).copied().collect();
        for key in keys {
            let k = key.1 as usize * width + key.0 as usize;
            if !read(k) {
                continue;
            }
            if unread.contains(&key) || turns(k) != found.get(&key).copied() {
                out.insert((key.1, key.0));
            }
        }
        out.into_iter().map(|(j, i)| (i as usize, j as usize)).collect()
    }

    /// The turn of arg f along the segment from `a` to `b`, given its phase
    /// at each end: the short way round where that is at most a quarter
    /// turn, and otherwise the sum of its halves' turns, halved again at
    /// most `depth` times. None where the phase is not f's on the way, or still
    /// turns too far: a zero or a pole on the segment or all but on it.
    fn turn(&self, a: P, pa: f64, b: P, pb: f64, depth: i64) -> Option<f64> {
        if !(pa.is_finite() && pb.is_finite()) {
            return None;
        }
        let t = ContourRefiner::wrapped(pb - pa);
        if t.abs() <= PI / 2.0 {
            return Some(t);
        }
        if depth <= 0 {
            return None;
        }
        let m = P::new(0.5 * (a.x() + b.x()), 0.5 * (a.y() + b.y()));
        let pm = self.phase(Complex::new(m.x(), m.y()));
        let first = self.turn(a, pa, m, pm, depth - 1)?;
        let second = self.turn(m, pm, b, pb, depth - 1)?;
        Some(first + second)
    }

    /// The rays at level `theta` of the singular point `index` of `points`,
    /// read off its normal form: each started half a cell out along its
    /// direction, moved onto the level set, joined to the point by a chord
    /// placed like any other, and followed on f to the handoff radius.
    /// Under a finite ceiling a pole has no rays, its fan being cut away
    /// with everything over the truncation, and a zero's ray ends where it climbs
    /// over the ceiling, complete as it stands.
    fn seeded_rays(
        &self,
        index: usize,
        points: &[SingularPoint],
        theta: f64,
        level: f64,
        g: Scalar,
        ceiling: f64,
    ) -> Vec<Ray> {
        let s = points[index];
        if !(s.order != 0 && (s.order > 0 || !ceiling.is_finite())) {
            return Vec::new();
        }
        let cell = self.cell();
        let (z0, radius) = (s.point, ContourRefiner::HANDOFF * cell);
        let mut out = Vec::new();
        for j in 0..s.order.unsigned_abs() {
            let phi = (theta - s.coefficient.argument() + 2.0 * PI * j as f64) / s.order as f64;
            let r0 = 0.5 * cell;
            let q0 = P::new(z0.x() + r0 * real::cos(phi), z0.y() + r0 * real::sin(phi));
            if !self.in_window(q0) {
                continue;
            }
            let Some(p1) = self.onto(q0, level, g, 0.25 * cell) else { continue };
            let mut chord: Vec<Vec<P>> = vec![Vec::new()];
            self.subdivide(z0, p1, level, g, true, 0, None, &mut chord);
            if chord.len() != 1 {
                continue;
            }
            let mut run = vec![z0];
            run.extend(chord.swap_remove(0));
            run.push(p1);
            let stop = |p: P| if apart(p, z0) >= radius { Some(index) } else { None };
            let (tail, _) = self.trace(p1, z0, level, g, Some(&stop));
            run.extend(tail);
            let tip = run[run.len() - 1];
            let handoff = apart(tip, z0) >= radius && !(self.magnitude(Complex::new(tip.x(), tip.y())) > ceiling);
            out.push(Ray { run, point: index, handoff });
        }
        out
    }

    /// The piece cut back from the handoff discs: its vertices inside one
    /// are dropped, and an end so made is marked with the point it meets,
    /// for that point's ray to be spliced on. A piece that traversing
    /// squares led through a singular point's cell, from one ray onto
    /// another, comes apart there into the two.
    fn trimmed(&self, piece: &Piece, discs: &[(usize, P)]) -> Vec<Piece> {
        if discs.is_empty() {
            return vec![piece.clone()];
        }
        let radius = ContourRefiner::HANDOFF * self.cell();
        let inside = |p: P| discs.iter().find(|d| apart(p, d.1) < radius).map(|d| d.0);
        let mut out = Vec::new();
        let mut run: Vec<P> = Vec::new();
        let (mut follow, mut met) = (piece.ends.start, piece.met.start);
        for &p in &piece.run {
            if let Some(k) = inside(p) {
                if run.len() >= 2 {
                    out.push(Piece {
                        run: std::mem::take(&mut run),
                        ends: Ends { start: follow, end: false },
                        met: Met { start: met, end: Some(k) },
                    });
                }
                run.clear();
                follow = false;
                met = Some(k);
            } else {
                run.push(p);
            }
        }
        if run.len() >= 2 {
            out.push(Piece {
                run,
                ends: Ends { start: follow, end: piece.ends.end },
                met: Met { start: met, end: piece.met.end },
            });
        }
        out
    }

    /// The chord from `a` to `b`, with its interior vertices, when it can be
    /// placed on the phase level (`spliced`'s `chord`).
    fn chord(&self, a: P, b: P, level: f64, g: Scalar) -> Option<Vec<P>> {
        let mut runs: Vec<Vec<P>> = vec![Vec::new()];
        self.subdivide(a, b, level, g, true, 0, None, &mut runs);
        if runs.len() == 1 { runs.pop() } else { None }
    }

    /// The rays spliced onto the runs that meet their discs, and every ray
    /// nothing meets followed on from its handoff.
    ///
    /// A run's end inside a point's disc lies on one of the point's rays,
    /// the one leaving at the nearest angle; the two are joined by a chord
    /// placed like any other, so the curves are one curve from the point out.
    /// A ray no run meets -- the grid drew nothing along it, as it may on a
    /// coarse grid -- is followed on f to wherever it ends, and if that
    /// is another point's disc it is spliced there in turn. An end that no
    /// ray fits is carried onto the point as before.
    fn spliced(
        &self,
        rays: &[Ray],
        runs: Vec<Piece>,
        points: &[SingularPoint],
        level: f64,
        g: Scalar,
        stopping: &dyn Fn(P) -> Option<usize>,
    ) -> Vec<Vec<P>> {
        let mut used = vec![false; rays.len()];
        // The unused handoff ray of point `k` leaving toward `end`, by angle.
        let ray = |used: &[bool], k: usize, end: P| -> Option<usize> {
            let center = points[k].point;
            let angle = real::atan2(end.y() - center.y(), end.x() - center.x());
            let mut best: Option<(usize, f64)> = None;
            for (i, r) in rays.iter().enumerate() {
                if !(r.point == k && r.handoff && !used[i]) {
                    continue;
                }
                let tip = r.run[r.run.len() - 1];
                let off =
                    ContourRefiner::wrapped(real::atan2(tip.y() - center.y(), tip.x() - center.x()) - angle).abs();
                if best.is_none_or(|b| off < b.1) {
                    best = Some((i, off));
                }
            }
            let best = best?;
            if !(best.1 <= PI / (2 * points[k].order.abs()) as f64) {
                return None;
            }
            Some(best.0)
        };
        // `run` with the ray of point `k` spliced onto its start, or carried onto the point.
        let headed = |used: &mut Vec<bool>, run: Vec<P>, k: usize| -> Vec<P> {
            if let Some(i) = ray(used, k, run[0])
                && let Some(bridge) = self.chord(rays[i].run[rays[i].run.len() - 1], run[0], level, g)
            {
                used[i] = true;
                let mut out = rays[i].run.clone();
                out.extend(bridge);
                out.extend(run);
                return out;
            }
            let center = points[k].point;
            if let Some(bridge) = self.chord(center, run[0], level, g) {
                let mut out = vec![center];
                out.extend(bridge);
                out.extend(run);
                return out;
            }
            run
        };
        let reversed = |mut v: Vec<P>| {
            v.reverse();
            v
        };
        let mut out = Vec::new();
        for piece in runs {
            if piece.run.len() < 2 {
                continue;
            }
            let mut run = piece.run;
            if let Some(k) = piece.met.start {
                run = headed(&mut used, run, k);
            }
            if let Some(k) = piece.met.end {
                run = reversed(headed(&mut used, reversed(run), k));
            }
            out.push(run);
        }
        // Rays nothing met, followed on; one that reaches another disc is
        // spliced there, to a ray not yet used.
        for i in 0..rays.len() {
            if used[i] {
                continue;
            }
            let mut run = rays[i].run.clone();
            if rays[i].handoff {
                used[i] = true;
                let (tail, met) = self.trace(run[run.len() - 1], run[run.len() - 2], level, g, Some(stopping));
                run.extend(tail);
                if let Some(k) = met
                    && k != rays[i].point
                {
                    run = reversed(headed(&mut used, reversed(run), k));
                }
            }
            out.push(run);
        }
        out
    }

    /// How far round `z` the path turns, in radians: ±2π for a loop round it.
    pub fn turn_around(path: &[P], z: P) -> f64 {
        let mut total = 0.0;
        for w in path.windows(2) {
            let (a, b) = (w[0], w[1]);
            let ta = real::atan2(a.y() - z.y(), a.x() - z.x());
            let tb = real::atan2(b.y() - z.y(), b.x() - z.x());
            total += ContourRefiner::wrapped(tb - ta);
        }
        total
    }

    /// The rings at one level that the grid cannot see: around a zero, at a
    /// level below |f| at every corner of the cell the zero lies in, no node
    /// lies inside the loop and marching squares draws nothing, though the
    /// loop exists -- around a simple zero every such level is one closed
    /// curve. The grid decides which contours exist only where it can see
    /// them; at a zero, f decides. Each ring is seeded from the zero as a
    /// circle of radius level / |f'|, every vertex moved onto the level along
    /// its ray by bisection, the loop subdivided to the tolerance like any
    /// other, and cut to the window where it leaves it, each crossing solved
    /// along the window's edge. A level the grid did see -- a path already
    /// turning round the zero -- is left to the grid, and a ring round two
    /// zeros is seeded from the first.
    pub fn seeded_rings(&self, level: f64, zeros: &[P], existing: &[Vec<P>]) -> Vec<Vec<P>> {
        if !(level > 0.0 && level.is_finite()) {
            return Vec::new();
        }
        let (dx, dy, cell) = (self.dx(), self.dy(), self.cell());
        let mut out: Vec<Vec<P>> = Vec::new();
        for &z0 in zeros {
            let fx = (z0.x() - self.grid.real.lo) / dx;
            let fy = (z0.y() - self.grid.imag.lo) / dy;
            // Swift's `Int` traps on a zero off the grid's numbers; such a
            // zero is passed over here.
            let (Some(fi), Some(fj)) = (swift::int(fx.floor()), swift::int(fy.floor())) else { continue };
            let i = fi.max(0).min(self.width - 2);
            let j = fj.max(0).min(self.height - 2);
            let mut lowest = f64::INFINITY;
            for (ci, cj) in [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)] {
                lowest = swift::min(
                    lowest,
                    self.magnitude(Complex::new(
                        self.grid.real.lo + ci as f64 * dx,
                        self.grid.imag.lo + cj as f64 * dy,
                    )),
                );
            }
            if !(level < lowest) {
                continue;
            }
            if existing.iter().chain(out.iter()).any(|path| ContourRefiner::turn_around(path, z0).abs() > PI / 2.0) {
                continue;
            }
            let c = Complex::new(z0.x(), z0.y());
            let h = 1e-6 * cell;
            let d = (self.value(c + h) - self.value(c - h)) / (2.0 * h);
            if !(d.is_finite() && d.magnitude() > 0.0) {
                continue;
            }
            let r0 = level / d.magnitude();
            if !(r0.is_finite() && r0 < 2.0 * cell) {
                continue;
            }
            let n = 24;
            let mut ring: Vec<P> = Vec::new();
            for k in 0..n {
                let theta = 2.0 * PI * k as f64 / n as f64;
                let dir = Complex::new(real::cos(theta), real::sin(theta));
                let excess = |r: f64| self.magnitude(c + r * dir) - level;
                // Bracket the crossing along the ray from the zero, where
                // |f| is below the level, outward to where it is above.
                let mut hi = r0;
                let mut tries = 0;
                while excess(hi) <= 0.0 && tries < 12 {
                    hi *= 2.0;
                    tries += 1;
                }
                if !(excess(hi) > 0.0 && hi <= 4.0 * cell) {
                    ring.clear();
                    break;
                }
                let (holds, fails) = bisect(0.0, hi, |t| !(excess(t) > 0.0));
                let r = 0.5 * (holds + fails);
                ring.push(P::new(c.re + r * dir.re, c.im + r * dir.im));
            }
            if ring.len() != n {
                continue;
            }
            ring.push(ring[0]);
            let mut refined: Vec<Vec<P>> = vec![vec![ring[0]]];
            for w in ring.windows(2) {
                self.subdivide(w[0], w[1], level, Scalar::Magnitude, false, 0, None, &mut refined);
                let last = refined.len() - 1;
                refined[last].push(w[1]);
            }
            if refined.len() == 1 {
                out.extend(self.clipped_to_window(&refined[0], level, true));
            } else {
                // Broken where a chord could not be placed: its pieces, open.
                for piece in refined.iter().filter(|p| p.len() >= 2) {
                    out.extend(self.clipped_to_window(piece, level, false));
                }
            }
        }
        out
    }

    /// A loop cut to the window: the runs of it inside, each ending where
    /// it crosses the window's edge, the crossing solved on the edge itself
    /// so that it lies on the level as the grid's own edge vertices do. A
    /// loop wholly inside is returned closed. Swift's `closed` defaults to
    /// true.
    fn clipped_to_window(&self, ring: &[P], level: f64, closed: bool) -> Vec<Vec<P>> {
        let inside: Vec<bool> = ring.iter().map(|&p| self.in_window(p)).collect();
        if inside.iter().all(|&b| b) {
            return vec![ring.to_vec()];
        }
        let cell = self.cell();
        // The level's crossing of the window's edge between an inside vertex
        // and an outside one.
        let crossing = |a: P, b: P| -> Option<P> {
            let p = P::from_vec(bisect_points(a.v, b.v, |v| self.in_window(P::from_vec(v))).0);
            // Along whichever edge it is on, solve |f| = level by bisection
            // between the chord point and a step either way.
            let (r, im) = (self.grid.real, self.grid.imag);
            let on_row = (p.y() - im.lo).abs() <= 1e-6 * cell || (p.y() - im.hi).abs() <= 1e-6 * cell;
            let on_column = (p.x() - r.lo).abs() <= 1e-6 * cell || (p.x() - r.hi).abs() <= 1e-6 * cell;
            if !(on_row || on_column) {
                return Some(p);
            }
            let at = |s: f64| if on_row { P::new(p.x() + s, p.y()) } else { P::new(p.x(), p.y() + s) };
            let excess = |s: f64| {
                let q = at(s);
                self.magnitude(Complex::new(q.x(), q.y())) - level
            };
            let (mut slo, mut shi) = (-1e-3 * cell, 1e-3 * cell);
            let mut tries = 0;
            while (excess(slo) > 0.0) == (excess(shi) > 0.0) && tries < 10 {
                slo *= 2.0;
                shi *= 2.0;
                tries += 1;
            }
            if (excess(slo) > 0.0) == (excess(shi) > 0.0) {
                return Some(p);
            }
            let elo = excess(slo);
            let (holds, fails) = bisect(slo, shi, |s| (excess(s) > 0.0) == (elo > 0.0));
            Some(at(0.5 * (holds + fails)))
        };
        // A closed loop: start the walk at an outside vertex so no run is
        // split in two. An open run is walked from its first vertex.
        let n = if closed { ring.len().wrapping_sub(1) } else { ring.len() };
        let start = if closed { inside.iter().position(|&b| !b) } else { Some(0) };
        let Some(start) = start else { return vec![ring.to_vec()] };
        if !(n >= 2 && n != usize::MAX) {
            return vec![ring.to_vec()];
        }
        let mut runs: Vec<Vec<P>> = Vec::new();
        let mut run: Vec<P> = Vec::new();
        let last = if closed { n } else { n - 1 };
        for step in 0..=last {
            let k = (start + step) % n;
            let v = ring[k];
            if inside[k] {
                if run.is_empty()
                    && step > 0
                    && let Some(c) = crossing(v, ring[(start + step - 1) % n])
                {
                    run.push(c);
                }
                run.push(v);
            } else if !run.is_empty() {
                if let Some(c) = crossing(run[run.len() - 1], v) {
                    run.push(c);
                }
                if run.len() >= 2 {
                    runs.push(std::mem::take(&mut run));
                }
                run.clear();
            }
        }
        if run.len() >= 2 {
            runs.push(run);
        }
        runs
    }

    // MARK: one path

    /// One grid path, placed: possibly several pieces, where wrap vertices
    /// were dropped or a chord could not be placed, and only its stretches
    /// that reach under `ceiling`.
    ///
    /// A segment is placed when either of its grid vertices has |f| at or
    /// under the ceiling, so the vertex just over the truncation is placed too and
    /// the cut at the truncation edge (`Surface.truncationCrossing`) is the one it always was.
    /// A stretch of segments wholly over it is left out: its curves are cut away
    /// whatever its vertices are. The ends this makes are not ends of the
    /// curve, so they are not followed on f (`traced`) or carried to a zero.
    pub fn pieces(&self, path: &[P], level: f64, g: Scalar, cut: bool, ceiling: f64, zeros: &[P]) -> Vec<Piece> {
        if path.len() < 2 {
            return Vec::new();
        }
        if !ceiling.is_finite() {
            return self.placed(path, level, g, cut, true, true, zeros);
        }
        let n = path.len() - 1;
        // Not over: a vertex where |f| is not a number is the snap's to test.
        let low: Vec<bool> = path.iter().map(|p| !(self.magnitude(Complex::new(p.x(), p.y())) > ceiling)).collect();
        let wanted: Vec<bool> = (0..n).map(|k| low[k] || low[k + 1]).collect();
        if !wanted.contains(&false) {
            return self.placed(path, level, g, cut, true, true, zeros);
        }
        if !wanted.contains(&true) {
            return Vec::new();
        }
        let (mut vertices, mut want) = (path.to_vec(), wanted.clone());
        let closed = path[0] == path[n];
        if closed {
            // Walked from the start of a stretch, so none is split in two.
            // There is one: the stretches are some and not all of a loop.
            let Some(s) = (0..n).find(|&k| wanted[k] && !wanted[(k + n - 1) % n]) else { return Vec::new() };
            vertices = (0..=n).map(|k| path[(s + k) % n]).collect();
            want = (0..n).map(|k| wanted[(s + k) % n]).collect();
        }
        let mut out = Vec::new();
        let mut k = 0;
        while k < n {
            if !want[k] {
                k += 1;
                continue;
            }
            let mut e = k;
            while e < n && want[e] {
                e += 1;
            }
            out.extend(self.placed(&vertices[k..=e], level, g, cut, !closed && k == 0, !closed && e == n, zeros));
            k = e;
        }
        out
    }

    /// A run of grid vertices placed on f, in pieces. `follow_start` and
    /// `follow_end` say whether the path's own ends are the curve's, to be
    /// followed on f and carried to a zero or pole, or a cut at the ceiling;
    /// an end made by a split is always the curve's. A magnitude line's chord
    /// is sought past the `zeros` (`subdivide`).
    #[allow(clippy::too_many_arguments)]
    pub fn placed(
        &self,
        path: &[P],
        level: f64,
        g: Scalar,
        cut: bool,
        follow_start: bool,
        follow_end: bool,
        zeros: &[P],
    ) -> Vec<Piece> {
        if path.len() < 2 {
            return Vec::new();
        }
        let closed = path[0] == path[path.len() - 1];
        let mut snapped: Vec<Option<P>> = path.iter().map(|&p| self.snap(p, level, g, cut)).collect();
        if closed {
            let last = snapped.len() - 1;
            snapped[last] = snapped[0];
        }
        let past_run: Vec<P> = snapped.iter().flatten().copied().collect();
        let past = if cut || zeros.is_empty() { None } else { Some(Past { run: &past_run, zeros }) };
        // The runs: a new one begins at a dropped wrap vertex and wherever a
        // chord could not be placed on the level set (`subdivide`), so that
        // nothing is drawn between two vertices the curve does not join.
        let mut runs: Vec<Vec<P>> = vec![Vec::new()];
        let mut previous: Option<P> = None;
        for &vertex in &snapped {
            let Some(vertex) = vertex else {
                runs.push(Vec::new());
                previous = None;
                continue;
            };
            if let Some(previous) = previous {
                self.subdivide(previous, vertex, level, g, cut, 0, past.as_ref(), &mut runs);
            }
            let last = runs.len() - 1;
            runs[last].push(vertex);
            previous = Some(vertex);
        }
        // Which runs begin at the path's first vertex and end at its last:
        // only those ends can be cuts at the ceiling.
        let headed = snapped[0].is_some();
        let tailed = snapped[snapped.len() - 1].is_some();
        let count = runs.len();
        runs.into_iter()
            .enumerate()
            .filter(|(_, run)| run.len() >= 2)
            .map(|(i, run)| Piece {
                run,
                ends: Ends { start: follow_start || i != 0 || !headed, end: follow_end || i != count - 1 || !tailed },
                met: Met::default(),
            })
            .collect()
    }

    /// A phase run followed on f beyond either end that lies inside the
    /// window and on no zero. Marching squares ends a run where it cannot
    /// bracket the level on a grid edge: beside an essential singularity
    /// the phase turns many times within a cell, and the lines stopped a
    /// cell short of the origin they run into. From the last vertex the
    /// grid placed, the curve is followed on f: a step along the
    /// tangent, corrected onto the level set along the gradient, kept only
    /// when the chord to it can be placed (`subdivide`) and halved
    /// otherwise, until the curve leaves the window, where it ends on the
    /// edge, or the step has shrunk to nothing at a singularity.
    pub fn traced(&self, mut piece: Piece, level: f64, g: Scalar, stopping: Stopping<'_>) -> Piece {
        if piece.run.len() < 2 || piece.run[0] == piece.run[piece.run.len() - 1] {
            return piece;
        }
        let cell = self.cell();
        let interior = |p: P| {
            let (r, i, slack) = (self.grid.real, self.grid.imag, 1e-6 * cell);
            p.x() > swift::min(r.lo, r.hi) + slack
                && p.x() < swift::max(r.lo, r.hi) - slack
                && p.y() > swift::min(i.lo, i.hi) + slack
                && p.y() < swift::max(i.lo, i.hi) - slack
        };
        // On a zero or a pole, where the line is already at its end.
        let on_singular_point = |p: P| {
            let u = self.magnitude(Complex::new(p.x(), p.y()));
            !(u.is_finite() && u > 1e-9 && u < NativeLandscape::HUGE)
        };
        let n = piece.run.len();
        if piece.ends.end && interior(piece.run[n - 1]) && !on_singular_point(piece.run[n - 1]) {
            let (tail, met) = self.trace(piece.run[n - 1], piece.run[n - 2], level, g, stopping);
            piece.run.extend(tail);
            piece.met.end = met;
        }
        if piece.ends.start && interior(piece.run[0]) && !on_singular_point(piece.run[0]) {
            let (mut head, met) = self.trace(piece.run[0], piece.run[1], level, g, stopping);
            head.reverse();
            head.extend(std::mem::take(&mut piece.run));
            piece.run = head;
            piece.met.start = met;
        }
        piece
    }

    /// Onto the zero or pole `z`, within `REACH` of `p`, when the chord to it
    /// can be placed: every phase line through either ends on it
    /// (`trace`'s `carried`).
    fn carried(&self, z: Option<P>, p: P, level: f64, g: Scalar, out: &mut Vec<P>) -> bool {
        let Some(z) = z else { return false };
        if !(apart(z, p) <= ContourRefiner::REACH * self.cell()) {
            return false;
        }
        let Some(between) = self.chord(p, z, level, g) else { return false };
        out.extend(between);
        out.push(z);
        true
    }

    /// The vertices the curve through `p0` continues with, away from
    /// `before`, and the handoff disc it ended in, if any.
    pub fn trace(&self, p0: P, before: P, level: f64, g: Scalar, stopping: Stopping<'_>) -> (Vec<P>, Option<usize>) {
        let cell = self.cell();
        let mut out: Vec<P> = Vec::new();
        let (mut p, mut prev) = (p0, before);
        let mut h = 0.25 * cell;
        let (h_min, h_max) = (1e-7 * cell, 0.5 * cell);
        let mut met = None;
        let mut ended = false;
        let u0 = self.magnitude(Complex::new(p0.x(), p0.y()));
        // |f| the last time a zero or pole was looked for from the line.
        let mut marker = u0;
        for _ in 0..40000 {
            // A trace ends on a zero or a pole, on the window's edge, or
            // where its step has shrunk to nothing. The bound is a safety
            // net, not a budget: a line the grid lost beside its zero is
            // followed the whole way to its pole.
            if out.len() >= 20000 {
                break;
            }
            // Into the handoff disc of a singular point, where its ray takes
            // over (`spliced`).
            if let Some(stopping) = stopping
                && let Some(k) = stopping(p)
            {
                met = Some(k);
                ended = true;
                break;
            }
            // Over the ceiling, where the curves are cut away: the vertex just
            // over it is kept, for the cut at the truncation edge to be solved on.
            if self.magnitude(Complex::new(p.x(), p.y())) > self.ceiling {
                ended = true;
                break;
            }
            // Approaching a zero or a pole, the line ends on it rather than
            // stepping through it onto another ray, and rather than creeping
            // up to it: |f| doubles with every halving of the distance to a
            // simple pole and halves with it to a simple zero, so each time
            // |f| has doubled or halved since the last look, the point is
            // looked for, and the line is carried onto it from two cells out.
            let u = self.magnitude(Complex::new(p.x(), p.y()));
            if u <= marker / 2.0 {
                marker = u;
                if self.carried(self.zero_near(p), p, level, g, &mut out) {
                    ended = true;
                    break;
                }
            } else if u >= 2.0 * marker {
                marker = u;
                if self.carried(self.pole_near(p), p, level, g, &mut out) {
                    ended = true;
                    break;
                }
            }
            if u > 1e6 {
                break;
            }
            let (gx, gy) = self.gradient(g, p);
            let slope = (gx * gx + gy * gy).sqrt();
            if !(slope > 0.0 && slope.is_finite()) {
                break;
            }
            let (mut tx, mut ty) = (-gy / slope, gx / slope);
            if tx * (p.x() - prev.x()) + ty * (p.y() - prev.y()) < 0.0 {
                tx = -tx;
                ty = -ty;
            }
            let mut placed: Option<Vec<P>> = None;
            let mut next: Option<P> = None;
            while h >= h_min {
                let q0 = P::new(p.x() + h * tx, p.y() + h * ty);
                if let Some(q) = self.onto(q0, level, g, 0.5 * h)
                    && let Some(between) = self.chord(p, q, level, g)
                {
                    placed = Some(between);
                    next = Some(q);
                    break;
                }
                h *= 0.5;
            }
            let (Some(q), Some(between)) = (next, placed) else { break };
            if !self.in_window(q) {
                // Ended on the window's edge: the chord's crossing of the
                // edge, then the level solved along the edge itself.
                let e = P::from_vec(bisect_points(p.v, q.v, |v| self.in_window(P::from_vec(v))).0);
                let im = self.grid.imag;
                let on_row = (e.y() - im.lo).abs() <= 1e-6 * cell || (e.y() - im.hi).abs() <= 1e-6 * cell;
                let at = |t: f64| if on_row { P::new(e.x() + t, e.y()) } else { P::new(e.x(), e.y() + t) };
                let res = |t: f64| self.residual(self.at(g, at(t)), level, true);
                let (mut slo, mut shi, mut tries) = (-1e-3 * cell, 1e-3 * cell, 0);
                while (res(slo) > 0.0) == (res(shi) > 0.0) && tries < 8 {
                    slo *= 2.0;
                    shi *= 2.0;
                    tries += 1;
                }
                let mut end: Option<P> = None;
                if (res(slo) > 0.0) != (res(shi) > 0.0) && res(slo).abs() < PI / 2.0 && res(shi).abs() < PI / 2.0 {
                    let rlo = res(slo);
                    let (holds, fails) = bisect(slo, shi, |t| (res(t) > 0.0) == (rlo > 0.0));
                    end = Some(at(0.5 * (holds + fails)));
                }
                // Only an end on the level, that the surface has a height
                // at, reached by a chord that can be placed: a line that
                // meets the edge where the level is not crossed along it,
                // or runs into a singularity on the edge, stops at its last
                // vertex rather than ending off the level or jumping to the truncation.
                if let Some(end) = end {
                    let mut runs: Vec<Vec<P>> = vec![Vec::new()];
                    self.subdivide(p, end, level, g, true, 0, None, &mut runs);
                    let r_end = self.residual(self.at(g, end), level, true);
                    if r_end.is_finite()
                        && r_end.abs() <= 1e-9
                        && self.magnitude(Complex::new(end.x(), end.y())).is_finite()
                        && runs.len() == 1
                    {
                        out.extend(runs.swap_remove(0));
                        out.push(end);
                    }
                }
                ended = true;
                break;
            }
            let straight = between.is_empty();
            out.extend(between);
            out.push(q);
            prev = p;
            p = q;
            if straight {
                h = swift::min(h * 1.5, h_max);
            }
        }
        // Stopped inside the window: on the zero or pole beside it, if any.
        if !ended {
            let _ = self.carried(self.zero_near(p), p, level, g, &mut out)
                || self.carried(self.pole_near(p), p, level, g, &mut out);
        }
        (out, met)
    }

    /// `q0` moved onto the level set along the gradient, by Newton; None when
    /// it does not converge within `reach` of where it started.
    pub fn onto(&self, q0: P, level: f64, g: Scalar, reach: f64) -> Option<P> {
        let cell = self.cell();
        let mut q = q0;
        for _ in 0..30 {
            let r = self.residual(self.at(g, q), level, true);
            if !r.is_finite() {
                return None;
            }
            let (gx, gy) = self.gradient(g, q);
            let s2 = gx * gx + gy * gy;
            if !(s2 > 0.0 && s2.is_finite()) {
                return None;
            }
            let step = r / s2;
            q = P::new(q.x() - step * gx, q.y() - step * gy);
            if !(apart(q, q0) <= reach) {
                return None;
            }
            if step.abs() * s2.sqrt() <= 1e-12 * cell {
                return Some(q);
            }
        }
        None
    }

    /// A phase run carried to the zero or pole of f it ends beside, at
    /// either end.
    ///
    /// Every phase meets at a zero and at a pole, so a phase contour that
    /// stops within a cell or so of one was always going there: the grid's
    /// marching squares put its last vertex on the last cell edge it crossed.
    /// The point is found from the end by Newton, on f for a zero and on 1/f
    /// for a pole, accepted when the iteration converges without leaving the
    /// end's neighbourhood and lands on a value a million times smaller than
    /// the end's, and the new segment is subdivided like any other. A run
    /// that already ends on one, or that is closed, is left alone. Swift's
    /// `ended(atSingularPoints:level:g:ends:)`, whose `ends` default to both.
    pub fn ended(&self, run: &[P], level: f64, g: Scalar, ends: Ends) -> Vec<P> {
        if run.len() < 2 || run[0] == run[run.len() - 1] {
            return run.to_vec();
        }
        let cell = self.cell();
        // The zero or pole this end leads to, when the chord to it lies on
        // the level set -- its midpoint's phase within an eighth of a turn
        // of the level, which a different contour's end beside the same
        // point fails -- and the run does not already end there from the
        // other side, as a stub in a pit a cell wide would.
        let zero_for = |end: P, other: P| -> Option<P> {
            let z = self.zero_near(end).or_else(|| self.pole_near(end))?;
            let apart = ((z.x() - other.x()) * (z.x() - other.x()) + (z.y() - other.y()) * (z.y() - other.y())).sqrt();
            if !(apart > 1e-9 * cell) {
                return None;
            }
            let mid = Complex::new((z.x() + end.x()) / 2.0, (z.y() + end.y()) / 2.0);
            let r = self.residual(self.g(g, mid), level, true);
            if r.is_finite() && r.abs() <= PI / 4.0 { Some(z) } else { None }
        };
        let mut out = run.to_vec();
        if ends.start
            && let Some(z) = zero_for(out[0], out[out.len() - 1])
        {
            let mut lead: Vec<Vec<P>> = vec![Vec::new()];
            self.subdivide(z, out[0], level, g, true, 0, None, &mut lead);
            // Carried only when the whole chord to the zero could be placed.
            if lead.len() == 1 {
                let mut head = vec![z];
                head.extend(lead.swap_remove(0));
                head.extend(out);
                out = head;
            }
        }
        if ends.end
            && let Some(z) = zero_for(out[out.len() - 1], out[0])
        {
            let mut tail: Vec<Vec<P>> = vec![Vec::new()];
            self.subdivide(out[out.len() - 1], z, level, g, true, 0, None, &mut tail);
            if tail.len() == 1 {
                out.extend(tail.swap_remove(0));
                out.push(z);
            }
        }
        out
    }

    /// A zero of f within `REACH` cells of `p`, or None (`zero(near:)`).
    pub fn zero_near(&self, p: P) -> Option<P> {
        self.root(p, |z| self.value(z))
    }

    /// A pole of f within `REACH` cells of `p`, or None: a zero of 1/f, found
    /// the same way. Newton on 1/f is well conditioned where Newton on f is
    /// not, since 1/f is small and smooth beside a pole (`pole(near:)`).
    pub fn pole_near(&self, p: P) -> Option<P> {
        self.root(p, |z| self.reciprocal(z))
    }

    /// 1/f: zero where f is not finite, which is the pole itself, and not
    /// finite where f is zero.
    fn reciprocal(&self, z: Complex) -> Complex {
        let v = self.value(z);
        if !v.is_finite() {
            return Complex::ZERO;
        }
        if v.magnitude() > 0.0 { Complex::ONE / v } else { Complex::real(f64::INFINITY) }
    }

    /// A zero of `g` within `REACH` cells of `p`, by Newton from `p`, or None:
    /// accepted when the iteration converges without leaving that
    /// neighbourhood and lands on a value a million times smaller than at
    /// `p`, so that a step toward a far-off root, or a stall at a critical
    /// point, is not mistaken for one.
    ///
    /// At a root of order m, Newton's step shrinks only by (m − 1)/m each
    /// time, so a root of order three or more is not reached in forty
    /// steps, and by then it is nearer than the central difference's step,
    /// which no longer sees its slope. So an iteration still going after
    /// forty reads the derivative over a step small against its last, takes
    /// two more of Newton's steps, whose ratio says the order, and finishes
    /// by Schröder's step, m times Newton's, which converges at a root of
    /// order m as Newton's does at a simple one.
    fn root(&self, p: P, g: impl Fn(Complex) -> Complex) -> Option<P> {
        let cell = self.cell();
        let mut z = Complex::new(p.x(), p.y());
        let g0 = g(z);
        if !(g0.is_finite() && g0.magnitude() > 0.0) {
            return None;
        }
        // One step from z, m times Newton's, its derivative read over h;
        // the step's length, or None when it leaves the neighbourhood.
        let step = |z: &mut Complex, m: f64, h: f64| -> Option<f64> {
            let gz = g(*z);
            if !gz.is_finite() {
                return None;
            }
            let d = (g(*z + h) - g(*z - h)) / (2.0 * h);
            if !(d.is_finite() && d.squared_magnitude() > 0.0) {
                return None;
            }
            let mut step = gz / d;
            if m != 1.0 {
                step = Complex::new(m * step.re, m * step.im);
            }
            if !step.is_finite() {
                return None;
            }
            *z -= step;
            if !(Complex::new(z.re - p.x(), z.im - p.y()).magnitude() <= ContourRefiner::REACH * cell) {
                return None;
            }
            Some(step.magnitude())
        };
        // z, when g there is a million times smaller than at p.
        let accepted = |z: Complex| -> Option<P> {
            let at = g(z);
            if !(at.is_finite() && at.magnitude() <= 1e-6 * g0.magnitude()) {
                return None;
            }
            let q = P::new(z.re, z.im);
            if self.in_window(q) { Some(q) } else { None }
        };
        let h = 1e-6 * cell;
        let mut last = 0.0;
        for _ in 0..40 {
            let s = step(&mut z, 1.0, h)?;
            if s <= 1e-12 * cell {
                return accepted(z);
            }
            last = s;
        }
        let mut m = 1.0;
        for k in 0..12 {
            let s = step(&mut z, m, swift::max(swift::min(h, 1e-3 * last), 1e-10 * cell))?;
            if s <= 1e-12 * cell {
                return accepted(z);
            }
            if k == 1 {
                m = (1.0 / (1.0 - s / last)).round();
                if !(m.is_finite() && m >= 2.0) {
                    return None;
                }
            }
            last = s;
        }
        None
    }

    /// The sub-edge of `a -> b` on which the level is crossed, with the
    /// residuals at its ends, of opposite sign or zero; None when it is not
    /// crossed.
    ///
    /// For a field without a cut that is the whole edge when the residuals
    /// differ in sign. For the phase, the two samples alone cannot say: a
    /// difference of more than π is a wrap when the samples straddle the cut,
    /// and a genuine turn when the edge passes beside a zero. So the phase is
    /// tracked through the edge's midpoint, each half the short way round --
    /// a straight edge subtends less than a half turn from a simple zero, so
    /// each half is unambiguous -- and the level is crossed on whichever half
    /// that path passes a multiple of 2π of its residual.
    fn bracket(&self, a: P, b: P, level: f64, g: Scalar, cut: bool) -> Option<Bracket> {
        let (pa, pb) = (self.at(g, a), self.at(g, b));
        if !(pa.is_finite() && pb.is_finite()) {
            return None;
        }
        if !cut {
            let (ra, rb) = (pa - level, pb - level);
            return if ra == 0.0 || rb == 0.0 || real::sign_minus(ra) != real::sign_minus(rb) {
                Some(Bracket { a, b, ra, rb })
            } else {
                None
            };
        }
        let m = P::new((a.x() + b.x()) / 2.0, (a.y() + b.y()) / 2.0);
        let pm = self.at(g, m);
        if !pm.is_finite() {
            return None;
        }
        let ra = ContourRefiner::wrapped(pa - level);
        let rm = ra + ContourRefiner::wrapped(pm - pa);
        let rb = rm + ContourRefiner::wrapped(pb - pm);
        if let Some(k) = ContourRefiner::turn_between(ra, rm) {
            return Some(Bracket { a, b: m, ra: ra - k, rb: rm - k });
        }
        if let Some(k) = ContourRefiner::turn_between(rm, rb) {
            return Some(Bracket { a: m, b, ra: rm - k, rb: rb - k });
        }
        None
    }

    /// The multiple of 2π between two residuals, if there is one.
    fn turn_between(u: f64, v: f64) -> Option<f64> {
        let (lo, hi) = (swift::min(u, v), swift::max(u, v));
        let k = (lo / (2.0 * PI)).ceil() * 2.0 * PI;
        if k <= hi { Some(k) } else { None }
    }

    /// The exact crossing on the grid edge this vertex sits on; None for a
    /// vertex on a phase wrap, which is not a crossing of anything.
    pub fn snap(&self, p: P, level: f64, g: Scalar, cut: bool) -> Option<P> {
        let (dx, dy, cell) = (self.dx(), self.dy(), self.cell());
        let fx = (p.x() - self.grid.real.lo) / dx;
        let fy = (p.y() - self.grid.imag.lo) / dy;
        let on_column = (fx - fx.round()).abs() < 1e-6;
        let on_row = (fy - fy.round()).abs() < 1e-6;
        let (a, b);
        if on_row && !on_column {
            // A horizontal edge: between the two columns either side.
            // (Swift's `Int` traps where fx is not a number: left as it is.)
            let Some(i) = swift::int(fx.floor()) else { return Some(p) };
            if !(i >= 0 && i + 1 < self.width) {
                return Some(p);
            }
            a = P::new(self.grid.real.lo + i as f64 * dx, p.y());
            b = P::new(self.grid.real.lo + (i + 1) as f64 * dx, p.y());
        } else if on_column && !on_row {
            let Some(j) = swift::int(fy.floor()) else { return Some(p) };
            if !(j >= 0 && j + 1 < self.height) {
                return Some(p);
            }
            a = P::new(p.x(), self.grid.imag.lo + j as f64 * dy);
            b = P::new(p.x(), self.grid.imag.lo + (j + 1) as f64 * dy);
        } else if on_column && on_row {
            // On a node. Marching squares puts a vertex there when the sample
            // equals the level, which is right, and also when the far sample
            // is the truncation: exp(1/z) beside its essential singularity has a
            // node at 1e-109 next to one at 1e12, and the interpolated
            // crossing is 1e-12 of the way along the edge, which rounds onto
            // the node where f is flat and nowhere near the level. The edge is
            // not recorded, but if exactly one of the node's four edges
            // brackets the level, that is the one.
            let (Some(i), Some(j)) = (swift::int(fx.round()), swift::int(fy.round())) else { return Some(p) };
            let gp = self.residual(self.at(g, p), level, cut);
            if !gp.is_finite() {
                return None;
            }
            if cut && gp.abs() > PI / 2.0 {
                return Some(p);
            }
            let (gx, gy) = self.gradient(g, p);
            let slope = (gx * gx + gy * gy).sqrt();
            if !(slope > 0.0 && slope.is_finite() && gp.abs() / slope > 1e-2 * self.tolerance * cell) {
                return Some(p);
            }
            let mut bracketing: Vec<P> = Vec::new();
            for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let (ni, nj) = (i + di, j + dj);
                if !(ni >= 0 && ni < self.width && nj >= 0 && nj < self.height) {
                    continue;
                }
                let n = P::new(self.grid.real.lo + ni as f64 * dx, self.grid.imag.lo + nj as f64 * dy);
                if self.bracket(p, n, level, g, cut).is_some() {
                    bracketing.push(n);
                }
            }
            if bracketing.len() != 1 {
                return Some(p);
            }
            a = p;
            b = bracketing[0];
        } else {
            return Some(p); // nowhere a marching-squares vertex can be
        }
        let Some(edge) = self.bracket(a, b, level, g, cut) else {
            let ra = self.residual(self.at(g, a), level, cut);
            let rb = self.residual(self.at(g, b), level, cut);
            // An end where the field has no value -- a node on a zero, or
            // where f is not finite: the vertex was placed by a sample that
            // meant nothing, and is dropped; a phase line is carried onto
            // the zero from its last placed vertex instead.
            if !(ra.is_finite() && rb.is_finite()) {
                return None;
            }
            // The float32 grid saw a crossing that f, in double, does not:
            // the level set passes within rounding of a node. The nearer
            // endpoint is the honest position. For the phase, anything else
            // is a wrap, which is not a crossing of anything.
            if cut && swift::min(ra.abs(), rb.abs()) > 1e-9 {
                return None;
            }
            return Some(if ra.abs() <= rb.abs() { a } else { b });
        };
        let (a1, b1) = (edge.a, edge.b);
        // Brent's method on t in [0, 1] along the edge. A bracket like
        // exp(1/z)'s beside its essential singularity, 1e-109 at one end and
        // the 1e12 truncation at the other, defeats interpolation -- every step lands
        // beside the small end -- and there it bisects.
        let at = |t: f64| P::new(a1.x() + t * (b1.x() - a1.x()), a1.y() + t * (b1.y() - a1.y()));
        let r = |t: f64| self.residual(self.at(g, at(t)), level, cut);
        // An edge can cross the level more than once. The crossing meant is
        // the one marching squares put the vertex beside, so the bracket is
        // first split at the vertex, keeping the half that changes sign.
        let (mut ta, mut tb, mut ra, mut rb) = (0.0, 1.0, edge.ra, edge.rb);
        let mut t0 = (p.x() - a1.x()) * (b1.x() - a1.x()) + (p.y() - a1.y()) * (b1.y() - a1.y());
        t0 /= swift::max(
            (b1.x() - a1.x()) * (b1.x() - a1.x()) + (b1.y() - a1.y()) * (b1.y() - a1.y()),
            f64::MIN_POSITIVE,
        );
        if t0 > 0.0 && t0 < 1.0 && ra != 0.0 && rb != 0.0 {
            let r0 = r(t0);
            if !r0.is_finite() {
                return Some(p);
            }
            if r0 == 0.0 {
                return Some(at(t0));
            }
            if real::sign_minus(r0) == real::sign_minus(ra) {
                (ta, ra) = (t0, r0);
            } else {
                (tb, rb) = (t0, r0);
            }
        }
        let Some(t) = Root::brent(r, ta, tb, ra, rb, 1e-12, 1e-13 * swift::max(level.abs(), 1.0), 100) else {
            return Some(p);
        };
        Some(at(t))
    }

    /// The vertices placed between `a` and `b` go on the last run of
    /// `runs`. A chord that cannot be placed on the level set -- its
    /// midpoint's value is not finite or, on a phase line, more than a
    /// quarter turn off the level, so the chord spans a jump of the phase;
    /// the depth limit is reached with the error still above the tolerance;
    /// or no point of the level set is found along the chord's normal --
    /// breaks the run there, so that nothing is drawn between two vertices
    /// the curve does not join. Beside an essential singularity the grid
    /// brackets nothing and marching squares joins vertices of different
    /// branches; those chords used to be drawn as they were, as hooks and
    /// zigzags. A chord whose midpoint lies on a critical point of the
    /// field, where the level set crosses itself, has nothing to measure
    /// by and is kept.
    #[allow(clippy::too_many_arguments)]
    fn subdivide(
        &self,
        a: P,
        b: P,
        level: f64,
        g: Scalar,
        cut: bool,
        depth: i64,
        past: Option<&Past<'_>>,
        runs: &mut Vec<Vec<P>>,
    ) {
        let cell = self.cell();
        let tolerance = self.tolerance;
        let mid = P::new((a.x() + b.x()) / 2.0, (a.y() + b.y()) / 2.0);
        let mut r = self.residual(self.at(g, mid), level, cut);
        if !r.is_finite() {
            runs.push(Vec::new());
            return;
        }
        let chord = ((b.x() - a.x()) * (b.x() - a.x()) + (b.y() - a.y()) * (b.y() - a.y())).sqrt();
        if !(chord > 1e-9 * cell) {
            return;
        }
        let (gx, gy) = self.gradient(g, mid);
        let slope = (gx * gx + gy * gy).sqrt();
        if !(slope > 0.0 && slope.is_finite()) {
            return;
        }
        let mut error = r.abs() / slope;
        if cut {
            // A phase level repeats every turn: beside an essential
            // singularity the circles of one level crowd together, and a
            // chord from one circle to the next can have its midpoint on a
            // third, with a residual that says nothing. The phase is read
            // at eight points along the chord and unwrapped from one to
            // the next; a jump of more than a quarter turn between
            // neighbours is a chord the samples cannot follow, and a
            // residual that does not return to the level at the far end
            // is a chord that has wound round. Along a chord the samples
            // can follow, its departure from the level set is the largest
            // residual over the gradient at any sample, so a chord that
            // follows a small circle to the tolerance is placed although
            // the phase turns steeply beside it.
            // The ends themselves are not read: a chord's end may be a zero,
            // where the phase is undefined.
            let mut unwrapped = 0.0;
            for k in 1..=7 {
                let t = k as f64 / 8.0;
                let q = P::new(a.x() + t * (b.x() - a.x()), a.y() + t * (b.y() - a.y()));
                let rq = self.residual(self.at(g, q), level, true);
                if !rq.is_finite() {
                    runs.push(Vec::new());
                    return;
                }
                let mut d = rq - unwrapped;
                while d > PI {
                    d -= 2.0 * PI;
                }
                while d < -PI {
                    d += 2.0 * PI;
                }
                if !(d.abs() <= PI / 2.0) {
                    runs.push(Vec::new());
                    return;
                }
                unwrapped += d;
                if !(unwrapped.abs() <= PI / 2.0) {
                    runs.push(Vec::new());
                    return;
                }
                if k == 4 {
                    r = unwrapped;
                }
                let (qx, qy) = self.gradient(g, q);
                let s = (qx * qx + qy * qy).sqrt();
                if s > 0.0 && s.is_finite() {
                    error = swift::max(error, unwrapped.abs() / s);
                }
            }
        }
        // A phase line is a curve in space: across the plane it is placed to
        // the tolerance, but on a steep graph its height between two
        // vertices is not the chord's, and in projection the chords kink.
        // So a chord is also split where |f| at its midpoint departs from
        // the mean of |f| at its ends by more than the tolerance. A
        // magnitude line is level and has no height to depart in. Nor has a
        // chord with a pole at its end: the height there is whatever the
        // truncation's is, and no halving brings the midpoint's height to the mean
        // of a finite one and `huge`, so the test would refuse every chord
        // that carries a line onto its pole.
        let mut sag = 0.0;
        if cut && self.lifted {
            let ua = self.magnitude(Complex::new(a.x(), a.y()));
            let ub = self.magnitude(Complex::new(b.x(), b.y()));
            let um = self.magnitude(Complex::new(mid.x(), mid.y()));
            if ua.is_finite()
                && ub.is_finite()
                && um.is_finite()
                && ua < NativeLandscape::HUGE
                && ub < NativeLandscape::HUGE
            {
                sag = (um - 0.5 * (ua + ub)).abs();
            }
        }
        if !(error > tolerance * cell || sag > tolerance * cell) {
            return;
        }
        if depth >= self.max_depth {
            runs.push(Vec::new());
            return;
        }
        // Solve along the unit normal to the chord, never further than a
        // chord's length from the midpoint. Where the level curve crosses
        // itself at a critical point (|cn| = 1 at 0, |sin| = 1 at π/2,
        // 1/(z³ - 1) = 1 at 0 with f'' = 0 too) the residual along the
        // normal has a multiple root, which an iteration that does not
        // bracket creeps toward linearly and may stop short of.
        let (nx, ny) = (-(b.y() - a.y()) / chord, (b.x() - a.x()) / chord);
        let gn = gx * nx + gy * ny;
        if !(gn.abs() > 1e-3 * slope) {
            return;
        }
        let along = |t: f64| P::new(mid.x() + t * nx, mid.y() + t * ny);
        let res = |t: f64| self.residual(self.at(g, along(t)), level, cut);
        // So first look for a sign change: step toward the Newton step,
        // doubling until the residual changes sign or the step reaches the
        // chord's length. A chord split for its sag alone can lie on the
        // level set already, as a ray of a pole does, the phase constant
        // along it and |f| not; its Newton step is then zero, to rounding,
        // and no doubling makes it longer, so the midpoint is left to
        // Newton's step below.
        let r_now = r;
        let bracket = |toward: f64| -> Option<(f64, f64)> {
            let mut reach = (r_now / gn).abs();
            loop {
                let t = toward * swift::min(reach, chord);
                let rt = res(t);
                if !(rt.is_finite() && !(cut && rt.abs() > PI / 2.0)) {
                    return None;
                }
                if rt == 0.0 || real::sign_minus(rt) != real::sign_minus(r_now) {
                    return Some((t, rt));
                }
                if !(reach > 0.0 && reach < chord) {
                    return None;
                }
                reach *= 2.0;
            }
        };
        let toward = if r / gn > 0.0 { -1.0 } else { 1.0 };
        let end = bracket(toward);
        let mut found: Option<P> = None;
        if let Some((end_t, end_r)) = end {
            // A bracket, and Brent's method within it, whose multiplicity
            // bisection does not care about. The bracket is the certificate:
            // the level curve crosses the normal within xtol of the point.
            // The residual over the gradient could not say so at a saddle,
            // where the gradient vanishes at the very point sought. A phase
            // residual also changes sign across a wrap, so a point whose
            // residual is near a half turn is refused.
            if let Some(t) = Root::brent_default(res, 0.0, end_t, r, end_r, 1e-10 * cell)
                && !(cut && res(t).abs() > PI / 2.0)
            {
                found = Some(along(t));
            }
        } else {
            // No sign change: the residual only touches zero, as it does
            // along a line through a simple saddle (|cn| - 1 ~ y²/2 up the
            // imaginary axis), and there is nothing to bracket. Newton's
            // step, then the secant, accepting only a converged point: within
            // a hundredth of the tolerance by the estimate the measurement
            // uses, residual over the gradient at the point itself.
            let converged = |q: P, r: f64| -> bool {
                let (gx, gy) = self.gradient(g, q);
                let s = (gx * gx + gy * gy).sqrt();
                s > 0.0 && s.is_finite() && r.abs() / s <= 1e-2 * tolerance * cell
            };
            let (mut t0, mut r0) = (0.0, r);
            let mut t1 = -r / gn;
            let mut best: Option<(P, f64)> = None;
            for _ in 0..24 {
                if t1.abs() > chord {
                    break;
                }
                let q = along(t1);
                let r1 = res(t1);
                if !(r1.is_finite() && !(cut && r1.abs() > PI / 2.0)) {
                    break;
                }
                if best.is_none_or(|b| r1.abs() < b.1.abs()) {
                    best = Some((q, r1));
                }
                let stalled = (t1 - t0).abs() < 1e-10 * cell;
                if r1.abs() <= 1e-13 * swift::max(level.abs(), 1.0) || stalled {
                    if converged(q, r1) {
                        found = Some(q);
                    }
                    if found.is_some() || stalled {
                        break;
                    }
                }
                let denominator = r1 - r0;
                if denominator == 0.0 {
                    break;
                }
                let t2 = t1 - r1 * (t1 - t0) / denominator;
                t0 = t1;
                r0 = r1;
                t1 = t2;
            }
            if found.is_none()
                && let Some((q, rb)) = best
                && converged(q, rb)
            {
                found = Some(q);
            }
        }
        // Nor that way: a level of |f| can lie the other way, round the far
        // side of a zero, |f| falling to it and rising past it, while toward
        // Newton's step |f| turns back short of the level. The chord then
        // crosses the zero's hole, |f| under the level at its midpoint.
        // Sought only past one of `past`'s zeros, in the strip the chord
        // sweeps along its normal out to the point found, and kept only
        // where it is reached without crossing the run: past another chord
        // of the run it is on an arc the run already traverses, as round a
        // zero whose hole the grid drew, and a chord placed there would
        // double the run back over itself. And only within the span: a
        // band's edge past a zero by its boundary is the other band's. A
        // phase line's other side past a zero is another ray, and is not
        // sought.
        if found.is_none()
            && let Some(past) = past
            && r < 0.0
            && let Some((other_t, other_r)) = bracket(-toward)
            && let Some(t) = Root::brent_default(res, 0.0, other_t, r, other_r, 1e-10 * cell)
        {
            let q = along(t);
            // Which side of the line from s to e p lies on, by sign.
            let side = |p: P, s: P, e: P| (e.x() - s.x()) * (p.y() - s.y()) - (e.y() - s.y()) * (p.x() - s.x());
            // Whether z lies in the strip the chord sweeps out to q.
            let between = |z: P| {
                let across = ((z.x() - a.x()) * (b.x() - a.x()) + (z.y() - a.y()) * (b.y() - a.y())) / chord;
                let out = ((z.x() - mid.x()) * nx + (z.y() - mid.y()) * ny) * (if t > 0.0 { 1.0 } else { -1.0 });
                across >= 0.0 && across <= chord && out >= 0.0 && out <= t.abs()
            };
            // Whether the segment from the chord's midpoint to q crosses u-v,
            // a segment of the run other than the one the chord lies on.
            let crosses = |u: P, v: P| {
                if !(distance(mid, u, v) > 1e-9 * cell) {
                    return false;
                }
                (side(u, mid, q) > 0.0) != (side(v, mid, q) > 0.0) && (side(mid, u, v) > 0.0) != (side(q, u, v) > 0.0)
            };
            if self.span.contains(&q.x())
                && past.zeros.iter().any(|&z| between(z))
                && !past.run.windows(2).any(|w| crosses(w[0], w[1]))
            {
                found = Some(q);
            }
        }
        let Some(point) = found else { return };
        self.subdivide(a, point, level, g, cut, depth + 1, past, runs);
        let last = runs.len() - 1;
        runs[last].push(point);
        self.subdivide(point, b, level, g, cut, depth + 1, past, runs);
    }

    /// The gradient of g at p, by central differences at a step that is small
    /// against the cell and large against rounding.
    pub fn gradient(&self, g: Scalar, p: P) -> (f64, f64) {
        let step = 1e-4 * self.cell();
        (
            (self.at(g, P::new(p.x() + step, p.y())) - self.at(g, P::new(p.x() - step, p.y()))) / (2.0 * step),
            (self.at(g, P::new(p.x(), p.y() + step)) - self.at(g, P::new(p.x(), p.y() - step))) / (2.0 * step),
        )
    }

    // MARK: measuring

    /// The distance of each vertex from the level set, in cells: the residual
    /// over the gradient, first order. What the refinement is trying to make
    /// small, measured against f rather than against a finer grid. Truncated at
    /// a cell: the estimate is first order, and where f is flat (a grid
    /// vertex left on a node beside exp(1/z)'s essential singularity, where
    /// |f| is 1e-109) it runs to 1e106 and says only "far". NaN for every
    /// vertex of a field that is not f's (Swift traps).
    pub fn position_errors(&self, path: &[P], field: ContourField, level: f64) -> Vec<f64> {
        let Some((g, level)) = ContourRefiner::field_level(field, level) else { return vec![f64::NAN; path.len()] };
        let cell = self.cell();
        path.iter()
            .map(|&p| {
                let r = self.residual(self.at(g, p), level, field == ContourField::Phase);
                let (gx, gy) = self.gradient(g, p);
                let slope = (gx * gx + gy * gy).sqrt();
                if !(r.is_finite() && slope > 0.0 && slope.is_finite()) {
                    return f64::NAN;
                }
                if field == ContourField::Phase && r.abs() > PI / 2.0 {
                    return f64::NAN;
                }
                swift::min(r.abs() / slope / cell, 1.0)
            })
            .collect()
    }

    /// Where a chord departs from the level set between two vertices: the
    /// midpoint error, in cells, for each segment.
    pub fn chord_errors(&self, path: &[P], field: ContourField, level: f64) -> Vec<f64> {
        if path.len() < 2 {
            return Vec::new();
        }
        let mids: Vec<P> =
            path.windows(2).map(|w| P::new((w[0].x() + w[1].x()) / 2.0, (w[0].y() + w[1].y()) / 2.0)).collect();
        self.position_errors(&mids, field, level)
    }
}

fn apart(a: P, b: P) -> f64 {
    ((a.x() - b.x()) * (a.x() - b.x()) + (a.y() - b.y()) * (a.y() - b.y())).sqrt()
}

/// The distance from `p` to the segment `a`-`b`.
fn distance(p: P, a: P, b: P) -> f64 {
    let (abx, aby) = (b.x() - a.x(), b.y() - a.y());
    let l2 = abx * abx + aby * aby;
    let t = if l2 > 0.0 {
        swift::min(swift::max(((p.x() - a.x()) * abx + (p.y() - a.y()) * aby) / l2, 0.0), 1.0)
    } else {
        0.0
    };
    let (qx, qy) = (a.x() + t * abx - p.x(), a.y() + t * aby - p.y());
    (qx * qx + qy * qy).sqrt()
}

impl NativeLandscape {
    /// A refiner for the function behind a landscape, at its grid.
    pub fn refiner(expression: &str, domain: Domain, shape: GridShape) -> Result<ContourRefiner, ExpressionError> {
        Ok(ContourRefiner::new(kurven_math::Expression::compile(expression)?, domain, shape.n_real, shape.n_imag))
    }

    // PORT: phase 3 (Bundle.swift) — `refiner(for bundle: KurvenBundle)` and
    // `benchmarkRefinement` need `KurvenBundle` and `Surface`, ported in
    // another wave; they are built on `NativeLandscape::refiner` above,
    // `ContourRefiner::refine`, `position_errors`, `chord_errors` and
    // `summarize` below.

    /// The mean, the 95th percentile and the largest of the finite values.
    pub fn summarize(values: &[f64]) -> Summary {
        let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if finite.is_empty() {
            return Summary { mean: 0.0, p95: 0.0, max: 0.0 };
        }
        let mut sum = 0.0;
        for v in &finite {
            sum += v;
        }
        let mean = sum / finite.len() as f64;
        let k = swift::int(((finite.len() - 1) as f64 * 0.95).trunc()).unwrap_or(0).max(0) as usize;
        let p95 = finite[k.min(finite.len() - 1)];
        Summary { mean, p95, max: finite[finite.len() - 1] }
    }
}

/// Swift's `(mean: Double, p95: Double, max: Double)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    pub mean: f64,
    pub p95: f64,
    pub max: f64,
}

/// One contour layer, derived with and without refinement, timed and
/// measured (`NativeLandscape.RefinementReport`).
#[derive(Debug, Clone, PartialEq)]
pub struct RefinementReport {
    pub layer: String,
    pub field: ContourField,
    pub levels: usize,
    pub grid_vertices: usize,
    pub refined_vertices: usize,
    /// Grid vertices that sat on a phase wrap and were dropped.
    pub wrap_vertices: usize,
    pub grid_seconds: Duration,
    pub refine_seconds: Duration,
    /// Vertex error in cells: mean, 95th percentile, max -- before and after.
    pub grid_vertex: Summary,
    pub refined_vertex: Summary,
    /// Midpoint-of-chord error in cells, before and after.
    pub grid_chord: Summary,
    pub refined_chord: Summary,
}
