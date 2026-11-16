//! Scalar fields on a face's parameter rectangle, and their level sets
//! clipped to a trimmed domain: the port of
//! `Sources/KurvenCore/ScalarField.swift`.
//!
//! Swift's `Contour` extension (`Contour.levels(of:_:over:...)` and
//! `Contour.clipped`) is the free functions [`levels`] and [`clipped`]
//! here, next to marching squares' own `contour::levels`. Swift's defaulted
//! arguments -- the trim loops, `contains`, the cuts, `excess` and `lift` --
//! are the fields of [`Trim`], whose `Default` is Swift's defaults.

use std::sync::Arc;

use kurven_math::{real, swift};

use crate::bisection::bisect_points;
use crate::contour;
use crate::grid::Grid2D;
use crate::manifest::Domain;
use crate::parametric::{ParamAxis, ParamLift, ParamPredicate, ParamRectangle, ParamScalar};
use crate::simd::{Vec2, Vec3};
use crate::spaces::{P2, ParamSpace};

/// `d` taken the short way round a circle of `period`: the representative
/// of d modulo the period nearest zero, in [−period/2, period/2]. How a
/// circle-valued field's differences and residuals are read: arg f's, by
/// the refiner and the seeds of integral curves, and a coordinate's along a
/// periodic axis.
#[inline]
pub fn short_way(d: f64, period: f64) -> f64 {
    d - period * (d / period).round()
}

/// Paths of the parameter plane: a level's contours, a face's trim loops.
pub type ParamPaths = Vec<Vec<P2<ParamSpace>>>;
/// Level sets in level order, each with its paths (Swift's
/// `[(level: Double, paths: [[P2<ParamSpace>]])]`).
pub type Contoured = Vec<(f64, ParamPaths)>;

/// Moves the marching-squares paths of one level onto the field's own
/// level set: the hook through which a field with a routine of its own,
/// as |f| and arg f have the refiner, places its contours. Whatever else
/// the routine needs, such as which of f's fields it places or the
/// ceiling above which a layer's curves are cut away, it closes over.
pub type Place = Arc<dyn Fn(f64, &[Vec<P2<ParamSpace>>]) -> ParamPaths + Send + Sync>;

/// How a field's level sets are found.
#[derive(Clone)]
pub enum Contouring {
    /// Sampled on the parameter rectangle's grid, and each crossing
    /// bisected on the value: a field with no singular points. `period`
    /// is the field's period when it is circle-valued: 2π for an angle.
    Bisection { value: ParamScalar, period: Option<f64> },
    /// Contoured on `samples`, the grid the routine was built against,
    /// and the paths handed to `place`: a field with a routine of its
    /// own, which owns its branch cut if it has one (arg f's, as
    /// `Surface::field` says).
    Placed { samples: Arc<Grid2D<f32>>, place: Place },
}

/// A scalar field on a face's parameter rectangle: what a level set is taken
/// of. A coordinate, a height, |f| or arg f, the facing.
///
/// A field on a periodic axis takes the same value a period along it. A
/// *circle-valued* field, such as the coordinate along a periodic axis, is
/// defined up to whole multiples of its own period, and a level is too.
#[derive(Clone)]
pub struct ScalarField {
    pub contouring: Contouring,
}

impl ScalarField {
    /// A smooth field, contoured by bisection on its value.
    pub fn new(period: Option<f64>, value: impl Fn(P2<ParamSpace>) -> f64 + Send + Sync + 'static) -> ScalarField {
        ScalarField { contouring: Contouring::Bisection { value: Arc::new(value), period } }
    }

    /// A field whose contours are placed by a routine of its own, from
    /// marching squares on `samples`.
    pub fn placed(samples: Arc<Grid2D<f32>>, place: Place) -> ScalarField {
        ScalarField { contouring: Contouring::Placed { samples, place } }
    }
}

/// How a family of level sets is clipped and refined: Swift's defaulted
/// arguments of `Contour.levels(of:_:over:trimLoops:contains:cuts:excess:lift:)`.
/// The default trims nothing.
#[derive(Clone, Default)]
pub struct Trim {
    /// The closed curves that bound the trimmed domain even-odd.
    pub trim_loops: Option<ParamPaths>,
    /// Whatever the loops do not carry, such as a copy's footprint.
    pub contains: Option<ParamPredicate>,
    /// The curves where `contains` changes.
    pub cuts: ParamPaths,
    /// Negative inside the domain, zero on the edge the loops follow as
    /// chords.
    pub excess: Option<ParamScalar>,
    /// The face's point in space, where its parameter plane is the world's
    /// xy (only `levels` reads it).
    pub lift: Option<ParamLift>,
}

/// The level sets of a field over a parameter rectangle, in level order,
/// each vertex its surface coordinate.
///
/// A field contoured by bisection is sampled on the rectangle's grid and
/// contoured on its difference from the level, every crossing then
/// bisected to the last bit along its grid edge: the grid decides which
/// contours exist, the field where they run. A periodic axis is sampled
/// one cell past its end -- the first sample again, as `ParamAxis::sample`
/// reads it -- so a contour that crosses the seam is one path, and its
/// coordinates are unwrapped along it: a closed one ends a period past
/// its first vertex, as a coordinate curve does. A circle-valued field's
/// difference is taken the short way round, and it jumps by a period at
/// the level opposite; an edge whose two samples differ by more than
/// half a period is that jump, and has no crossing. A level the field
/// takes all along a bounded side of the rectangle draws nothing along it:
/// the side is the face's edge.
///
/// A field with a routine of its own is contoured on its samples, by
/// marching squares at each level, and the paths handed to the routine,
/// a level at a time (in Swift the levels in parallel; here in turn, with
/// the same answer). The samples are their own grid: nothing wraps or is
/// unwrapped there.
///
/// A path is clipped to a trimmed domain exactly, as hatching is clipped
/// to a polyhedron face's convex pieces: cut wherever a segment of it
/// crosses one of the trim loops or cuts, and each piece kept where the
/// domain holds its middle. The domain is what the loops bound even-odd
/// (`Face::trim_loops`), read by a ray along +u, and of that what
/// `contains` holds, for whatever the loops do not carry -- a copy's
/// footprint -- with `cuts` the curves where `contains` changes. The
/// pieces kept either side of a cut are joined. So a run ends on a loop
/// wherever the loop crosses it, between two vertices or through one,
/// and a hole narrower than a step still cuts the path. Where neither
/// is given nothing is trimmed; where a domain's boundary is left out of
/// the loops and the cuts, a path is kept or dropped a segment at a
/// time there. Loops that bound even-odd lie in the plane: no periodic
/// axis runs across them. On a periodic axis `contains` is asked at
/// unwrapped coordinates, and must answer the same a period along; the
/// cuts are met a period along too.
///
/// Where the domain's edge is a level set, the loops' chords only
/// approximate it, and `excess` gives it exactly: negative inside the
/// domain, zero on that edge, as |f| less the truncation height is on
/// the graph. A run is then kept where `excess` is negative, cut where it
/// changes sign, bisected along the chord to the last bit, and a run
/// cut short of that edge by a chord of the loops is carried on to it.
///
/// A field contoured by bisection has its chords refined; `lift` is the
/// face's point in space, given where the parameter plane is the world's
/// xy, as on the graph, so a chord's sag off the face is a length in the
/// cell's units.
pub fn levels(field: &ScalarField, levels: &[f64], rectangle: ParamRectangle, trim: &Trim) -> Contoured {
    let contoured: Contoured = match &field.contouring {
        Contouring::Bisection { value, period } => plain_levels(value, *period, levels, rectangle, trim.lift.as_ref()),
        Contouring::Placed { samples, place } => {
            let grid = contour::levels(samples, levels);
            let paths: Vec<ParamPaths> = grid
                .iter()
                .map(|(level, paths)| {
                    let inp: ParamPaths = paths.iter().map(|p| p.iter().map(|q| P2::from_vec(q.v)).collect()).collect();
                    place(*level, &inp)
                })
                .collect();
            levels.iter().copied().zip(paths).collect()
        }
    };
    clipped(contoured, rectangle, trim)
}

/// Level sets already found, clipped to a trimmed domain as [`levels`]
/// clips them: so a family contoured once can be clipped to each copy of
/// a face, a copy's footprint being where the copy stands.
pub fn clipped(contoured: Contoured, rectangle: ParamRectangle, trim: &Trim) -> Contoured {
    if trim.trim_loops.is_none() && trim.contains.is_none() {
        return contoured;
    }
    let (u, v) = (rectangle.u, rectangle.v);
    let mut all = trim.trim_loops.clone().unwrap_or_default();
    all.extend(trim.cuts.iter().cloned());
    let crossed = TrimIndex::new(&all, (u.period(), v.period()));
    let bounding = trim.trim_loops.as_ref().map(|l| TrimIndex::new(l, (None, None)));
    let contains = trim.contains.clone();
    let holds = |p: P2<ParamSpace>| -> bool {
        bounding.as_ref().is_none_or(|b| b.encloses(p.v)) && contains.as_ref().is_none_or(|c| c(p))
    };
    contoured
        .into_iter()
        .map(|(level, paths)| {
            let out: ParamPaths = paths
                .iter()
                .flat_map(|path| -> ParamPaths {
                    let (Some(first), Some(last)) = (path.first(), path.last()) else {
                        return Vec::new();
                    };
                    if path.len() < 2 {
                        return Vec::new();
                    }
                    // A closed path ends where it began, or a period along where
                    // it was unwrapped.
                    let closed = u.difference(last.x(), first.x()).abs() <= 1e-12 * u.range.length()
                        && v.difference(last.y(), first.y()).abs() <= 1e-12 * v.range.length();
                    trimmed(path, closed, &holds, trim.contains.as_ref(), &crossed, trim.excess.as_ref())
                })
                .collect();
            (level, out)
        })
        .collect()
}

fn plain_levels(
    value: &ParamScalar,
    period: Option<f64>,
    levels: &[f64],
    rectangle: ParamRectangle,
    lift: Option<&ParamLift>,
) -> Contoured {
    let (u, v) = (rectangle.u, rectangle.v);
    let cell = swift::max(u.spacing().abs(), v.spacing().abs());
    // Row by row in Swift's threads; here in turn.
    let samples: Vec<f64> = (0..u.samples * v.samples)
        .map(|k| value(P2::new(u.coordinate((k % u.samples) as i64), v.coordinate((k / u.samples) as i64))))
        .collect();
    let w = if u.periodic { u.samples + 1 } else { u.samples };
    let h = if v.periodic { v.samples + 1 } else { v.samples };
    let domain = Domain::new(u.range, v.range);

    // Each level is a grid of its own, the field less the level, so a
    // sample's sign is the field's own however near the level it lies.
    let lines = |level: f64| -> ParamPaths {
        let excess = |z: f64| -> f64 {
            match period {
                Some(p) => real::remainder(z - level, p),
                None => z - level,
            }
        };
        let mut g = vec![0.0f64; w * h];
        for j in 0..h {
            for i in 0..w {
                g[j * w + i] = excess(samples[v.sample(j as i64) * u.samples + u.sample(i as i64)]);
            }
        }
        let grid = Grid2D::new(w, h, domain, g.iter().map(|&x| x as f32).collect());
        let paths: ParamPaths = contour::lines_placed(&grid, 0.0, None, (u.periodic, v.periodic), |a, b| {
            let ga = g[a[1] * w + a[0]];
            let gb = g[b[1] * w + b[0]];
            if let Some(period) = period
                && (ga - gb).abs() > period / 2.0
            {
                return None;
            }
            let pa = Vec2::new(u.coordinate(a[0] as i64), v.coordinate(a[1] as i64));
            let pb = Vec2::new(u.coordinate(b[0] as i64), v.coordinate(b[1] as i64));
            // From the end the grid has at or under the level to the end
            // over it, so the bisection keeps the grid's own bracket. An
            // end on the level to the bit is the crossing, not half a
            // rounding past it: a coordinate's contour is on its level.
            let (under, over) = if grid.at(a[0], a[1]) > 0.0 { (pb, pa) } else { (pa, pb) };
            let (holds, fails) = bisect_points(under, over, |q| !(excess(value(P2::from_vec(q))) > 0.0));
            if excess(value(P2::from_vec(holds))) == 0.0 {
                return Some(P2::from_vec(holds));
            }
            Some(P2::from_vec(0.5 * (holds + fails)))
        });
        paths
            .iter()
            .flat_map(|p| off_edges(&unwrapped(p, u.period(), v.period()), rectangle))
            .map(|p| refined(&p, &|q| excess(value(q)), cell, period, lift))
            .collect()
    };
    levels.iter().map(|&level| (level, lines(level))).collect()
}

/// How far a chord of the plain routine may stray, in cells of the
/// parameter rectangle: the refiner's tolerance for |f| and arg f
/// (`Refiner.tolerance`), so the families drawn together on a face are
/// held to one bound. On the 64-sample landscape it is 1.3e-3, a third
/// of a pixel in a 1600-pixel drawing. On a parametric surface the
/// distance is Euclidean in (u, v), against the larger of the two
/// spacings, and there is no sag to measure.
pub const CHORD_TOLERANCE: f64 = 0.02;
/// How many times a chord may be halved: at most 31 vertices are added
/// between two of the grid's, as the refiner's `maxDepth`.
pub const CHORD_DEPTH: usize = 5;

/// A path with each chord split where it strays from the level set: its
/// midpoint's distance from it, estimated as the excess over the level
/// there over the excess's gradient, more than `CHORD_TOLERANCE` of a
/// cell; or, given `lift`, its midpoint in space more than that from the
/// face's point at the chord's midpoint, the chord's sag. The vertices
/// marching squares placed lie on the level set; between two, the curve
/// can turn within a cell, and on a steep graph its chord can sag far
/// off the face. A chord is split at the point of the level set on its
/// normal through its midpoint, sought either way from it, nearer first,
/// out to the chord's length, until the excess changes sign, then
/// bisected to the last bit, as a crossing is, and kept only where its
/// residual is small: not on a jump of the field. A chord split for its
/// sag alone is split at its midpoint where none is found there, the
/// midpoint being within the tolerance of the level set. Each half is
/// measured again, down to `CHORD_DEPTH` halvings. A chord is kept as it
/// is where its midpoint has no value or no gradient, where it strays
/// and no point of the level set is found along its normal, and past
/// the last halving.
fn refined(
    path: &[P2<ParamSpace>],
    excess: &dyn Fn(P2<ParamSpace>) -> f64,
    cell: f64,
    period: Option<f64>,
    lift: Option<&ParamLift>,
) -> Vec<P2<ParamSpace>> {
    if path.len() < 2 {
        return path.to_vec();
    }
    let tolerance = CHORD_TOLERANCE * cell;
    let step = 1e-4 * cell;
    let ctx = Refine { excess, tolerance, step, cell, period, lift };
    let mut out = vec![path[0]];
    let mut la = lift.map(|l| l(path[0]));
    for &b in &path[1..] {
        let lb = lift.map(|l| l(b));
        let a = out[out.len() - 1].v;
        ctx.split(&mut out, a, la, b.v, lb, 0);
        out.push(b);
        la = lb;
    }
    out
}

struct Refine<'a> {
    excess: &'a dyn Fn(P2<ParamSpace>) -> f64,
    tolerance: f64,
    step: f64,
    cell: f64,
    period: Option<f64>,
    lift: Option<&'a ParamLift>,
}

impl Refine<'_> {
    fn f(&self, p: Vec2) -> f64 {
        (self.excess)(P2::from_vec(p))
    }

    /// A circle-valued excess jumps by a period at the level opposite:
    /// past a quarter of one, a sign change is that jump.
    fn near(&self, r: f64) -> bool {
        r.is_finite() && self.period.is_none_or(|p| r.abs() < p / 4.0)
    }

    fn split(&self, out: &mut Vec<P2<ParamSpace>>, a: Vec2, la: Option<Vec3>, b: Vec2, lb: Option<Vec3>, depth: usize) {
        let chord = (a - b).length();
        if !(depth < CHORD_DEPTH && chord > 1e-9 * self.cell) {
            return;
        }
        let m = 0.5 * (a + b);
        let r = self.f(m);
        if !self.near(r) {
            return;
        }
        let normal = Vec2::new(a.y - b.y, b.x - a.x) / chord;
        let (mut error, mut gn, mut slope) = (0.0, 0.0, 0.0);
        if r != 0.0 {
            let step = self.step;
            let g = Vec2::new(
                self.f(m + Vec2::new(step, 0.0)) - self.f(m - Vec2::new(step, 0.0)),
                self.f(m + Vec2::new(0.0, step)) - self.f(m - Vec2::new(0.0, step)),
            ) / (2.0 * step);
            slope = g.length();
            if !(slope > 0.0 && slope.is_finite()) {
                return;
            }
            error = r.abs() / slope;
            gn = g.dot(normal);
        }
        let mut sag = 0.0;
        if let (Some(lift), Some(la), Some(lb)) = (self.lift, la, lb) {
            let d = (lift(P2::from_vec(m)) - 0.5 * (la + lb)).length();
            if d.is_finite() {
                sag = d;
            }
        }
        if !(error > self.tolerance || sag > self.tolerance) {
            return;
        }
        let mut point = m;
        if r != 0.0 {
            match self.on_level(m, r, gn, slope, normal, chord) {
                Some(q) => point = q,
                None => {
                    if error > self.tolerance {
                        return;
                    }
                }
            }
        }
        let lp = self.lift.map(|l| l(P2::from_vec(point)));
        self.split(out, a, la, point, lp, depth + 1);
        out.push(P2::from_vec(point));
        self.split(out, point, lp, b, lb, depth + 1);
    }

    /// The point of the level set on the chord's normal through its
    /// midpoint: the reach doubled from Newton's step out to the
    /// chord's length, each tried toward Newton's step and then away
    /// from it, since beside an extremum of the field the step points
    /// away from the level set. `None` for a chord across the level set
    /// rather than along it, with none within a chord's length, or
    /// where the bracket closes on a jump of the field rather than
    /// its level, its residual not small.
    fn on_level(&self, m: Vec2, r: f64, gn: f64, slope: f64, normal: Vec2, chord: f64) -> Option<Vec2> {
        if !(gn.abs() > 1e-3 * slope) {
            return None;
        }
        let placed = |q: Vec2| -> Option<Vec2> {
            let rq = self.f(q);
            (rq.is_finite() && rq.abs() <= 1e-6 * self.tolerance * slope).then_some(q)
        };
        let toward: f64 = if r / gn > 0.0 { -1.0 } else { 1.0 };
        let mut reach = swift::max((r / gn).abs(), 1e-6 * chord);
        loop {
            for t in [toward, -toward].map(|s| s * swift::min(reach, chord)) {
                let far = m + t * normal;
                let rt = self.f(far);
                if !self.near(rt) {
                    continue;
                }
                if rt == 0.0 {
                    return placed(far);
                }
                if (rt > 0.0) != (r > 0.0) {
                    // From the end at or under the level to the end
                    // over it, as a crossing is placed.
                    let (under, over) = if r > 0.0 { (far, m) } else { (m, far) };
                    let (holds, fails) = bisect_points(under, over, |q| !(self.f(q) > 0.0));
                    return placed(if self.f(holds) == 0.0 { holds } else { 0.5 * (holds + fails) });
                }
            }
            if !(reach < chord) {
                return None;
            }
            reach *= 2.0;
        }
    }
}

/// A path less its segments along a bounded side of the rectangle, cut
/// there. The rectangle is open: a level its field takes all along a side
/// draws nothing there, at the lower end of an axis as at the upper, since
/// the side is the face's edge, as a polyhedron's hatching leaves a line
/// along a side of its face to the edge (`Solid.hatchLines(_:onFace:)`).
/// Marching squares counts a sample on the level as under it, so a level
/// at an axis's lower end drew the side, and one at its upper end nothing.
fn off_edges(path: &[P2<ParamSpace>], rectangle: ParamRectangle) -> ParamPaths {
    let along = |axis: &ParamAxis, a: f64, b: f64| -> bool {
        if axis.periodic || a != b {
            return false;
        }
        a == axis.range.lo || a == axis.range.hi || a == axis.coordinate(0) || a == axis.coordinate(axis.cells() as i64)
    };
    let mut runs: ParamPaths = Vec::new();
    let mut run: Vec<P2<ParamSpace>> = Vec::new();
    for &p in path {
        if let Some(q) = run.last()
            && (along(&rectangle.u, q.x(), p.x()) || along(&rectangle.v, q.y(), p.y()))
        {
            if run.len() >= 2 {
                runs.push(std::mem::take(&mut run));
            }
            run.clear();
        }
        run.push(p);
    }
    if run.len() >= 2 {
        runs.push(run);
    }
    runs
}

/// A path whose every step is the short way round a periodic axis.
fn unwrapped(path: &[P2<ParamSpace>], pu: Option<f64>, pv: Option<f64>) -> Vec<P2<ParamSpace>> {
    let mut out = path.to_vec();
    for k in 1..out.len() {
        let mut c = out[k].v;
        let previous = out[k - 1].v;
        if let Some(pu) = pu {
            c.x -= pu * ((c.x - previous.x) / pu).round();
        }
        if let Some(pv) = pv {
            c.y -= pv * ((c.y - previous.y) / pv).round();
        }
        out[k] = P2::from_vec(c);
    }
    out
}

/// A run of a trimmed path, and the segments of the path its ends were cut
/// on, where a loop cut them (Swift's `(points:start:end:)`).
#[derive(Clone)]
struct Run {
    points: Vec<P2<ParamSpace>>,
    start: Option<i64>,
    end: Option<i64>,
}

/// The runs of a path inside the trimmed domain: the path cut where it
/// crosses a loop, segment by segment, each piece kept where `holds`
/// holds at its middle, and the pieces kept either side of a cut joined
/// into one run, with no vertex at the cut.
///
/// A closed path's last vertex is its first, or its first a period
/// along. Where it is kept through both, the run through its last vertex
/// and the run through its first are one run, so a loop is not split
/// where it happens to begin.
///
/// Given `excess`, whose zero level the loops follow as chords, the
/// loops' cuts are moved onto that level. Two runs the loops cut apart
/// where the path passes between a chord and the level are joined. An
/// end cut on a loop where `excess` is still negative stops short of the
/// level its chord stands for: it is carried on along the path to where
/// `excess` reaches 0, bisected on the segment where it does, so long as
/// the path crosses no loop and stays where `contains` holds on the way;
/// otherwise the cut was another edge's, and stands. Then each run is
/// kept where `excess` is negative and cut where it changes sign,
/// bisected along the segment to the last bit and ended on its negative
/// side: a run past the loops' chord, or one the loops missed, is cut
/// back to the level, and no vertex lies over it.
fn trimmed(
    path: &[P2<ParamSpace>],
    closed: bool,
    holds: &dyn Fn(P2<ParamSpace>) -> bool,
    contains: Option<&ParamPredicate>,
    loops: &TrimIndex,
    excess: Option<&ParamScalar>,
) -> ParamPaths {
    // Where an end of a run was cut on a loop: the segment of the path,
    // counted on round a closed path past its last, or back past its
    // first, so a run carried round where the path begins ends on
    // segment n + k for its k.
    let n = path.len() as i64 - 1;
    let shift = path[n as usize].v - path[0].v;
    let vertex = |j: i64| -> P2<ParamSpace> {
        let lap = if j >= 0 { j / n } else { -((n - 1 - j) / n) };
        P2::from_vec(path[(j - lap * n) as usize].v + lap as f64 * shift)
    };
    let contains_at = |p: P2<ParamSpace>| contains.is_none_or(|c| c(p));
    let mut runs: Vec<Run> = Vec::new();
    let mut run: Vec<P2<ParamSpace>> = Vec::new();
    let mut start: Option<i64> = None;
    // Whether the run under way ends at a cut, whether the path is kept
    // from its first vertex, and whether it crosses a loop at all.
    let (mut ends_at_cut, mut kept_from_start, mut crossed) = (false, false, false);
    for k in 0..n {
        let a = path[k as usize].v;
        let b = path[k as usize + 1].v;
        let cuts = loops.crossings(a, b);
        crossed = crossed || !cuts.is_empty();
        let mut from = a;
        let ends: Vec<Vec2> = cuts.iter().copied().chain(std::iter::once(b)).collect();
        for (i, &to) in ends.iter().enumerate() {
            if to == from {
                continue;
            }
            let previous_from = from;
            from = to;
            if !holds(P2::from_vec(0.5 * (previous_from + to))) {
                if run.len() >= 2 {
                    runs.push(Run { points: run.clone(), start, end: if ends_at_cut { Some(k) } else { None } });
                }
                run.clear();
                continue;
            }
            if run.is_empty() {
                run = vec![P2::from_vec(previous_from)];
                start = if i > 0 { Some(k) } else { None };
                if k == 0 && previous_from == a {
                    kept_from_start = true;
                }
            } else if ends_at_cut && run.len() >= 2 {
                run.pop();
            }
            run.push(P2::from_vec(to));
            ends_at_cut = i < cuts.len();
        }
    }
    let mut end: Option<i64> = None;
    if closed && kept_from_start && !run.is_empty() && !runs.is_empty() {
        let first = runs.remove(0);
        run.extend(first.points[1..].iter().map(|p| P2::from_vec(p.v + shift)));
        end = first.end.map(|e| e + n);
    }
    if run.len() >= 2 {
        runs.push(Run { points: run, start, end });
    }
    let Some(excess) = excess else {
        return runs.into_iter().map(|r| r.points).collect();
    };
    let ex = |p: P2<ParamSpace>| excess(p);
    // A path that crosses no loop and is held nowhere may lie between a
    // chord and the level, in the domain: it is read on `excess` whole,
    // where `contains` holds it, which does not change along it.
    if runs.is_empty() && !crossed && contains_at(P2::from_vec(0.5 * (path[0].v + path[1].v))) {
        runs = vec![Run { points: path.to_vec(), start: None, end: None }];
    }

    // The vertices that carry a run on from an end at `x`, cut on
    // segment `segment` of the path, out along it, forward or back, to
    // where `excess` reaches 0; `None` where it does not before the path
    // crosses a loop, leaves `contains` or ends. Round a closed path it
    // goes on past where the path begins, once round. On the cut's own
    // segment the loops are met beyond the cut, along the segment, so
    // the cut's own crossing, a rounding from it, is not met again.
    let carried = |x: Vec2, segment: i64, forward: bool| -> Option<Vec<P2<ParamSpace>>> {
        if !(ex(P2::from_vec(x)) < 0.0) {
            return None;
        }
        let s0 = vertex(segment).v;
        let s1 = vertex(segment + 1).v;
        let d = s1 - s0;
        let along = |q: Vec2| (q - s0).dot(d);
        let others = loops.crossings(s0, s1);
        // `indices.min(by:)`: the first of the nearest.
        let mut own: Option<usize> = None;
        for i in 0..others.len() {
            match own {
                None => own = Some(i),
                Some(o) => {
                    if (others[i] - x).length() < (others[o] - x).length() {
                        own = Some(i);
                    }
                }
            }
        }
        if let Some(own) = own {
            let beyond: &[Vec2] = if forward { &others[own + 1..] } else { &others[..own] };
            if beyond.iter().any(|&q| (along(q) > along(x)) == forward) {
                return None;
            }
        }
        let outward: Vec<i64> = if forward {
            ((segment + 1)..=(if closed { segment + n } else { n })).collect()
        } else {
            let through = if closed { segment - n + 1 } else { 0 };
            let mut v = Vec::new();
            let mut j = segment;
            while j >= through {
                v.push(j);
                j -= 1;
            }
            v
        };
        let mut previous = x;
        let mut added: Vec<P2<ParamSpace>> = Vec::new();
        for (k, w) in outward.iter().map(|&j| vertex(j).v).enumerate() {
            if w == previous {
                continue;
            }
            if !((k == 0 || loops.crossings(previous, w).is_empty()) && contains_at(P2::from_vec(0.5 * (previous + w))))
            {
                return None;
            }
            if !(ex(P2::from_vec(w)) < 0.0) {
                let edge = bisect_points(previous, w, |q| ex(P2::from_vec(q)) < 0.0).0;
                if added.last().map(|p| p.v) == Some(edge) {
                    return Some(added);
                }
                added.push(P2::from_vec(edge));
                return Some(added);
            }
            added.push(P2::from_vec(w));
            previous = w;
        }
        None
    };
    // Two runs cut apart where the path leaves the loops and comes back
    // between their chords and the level, through a sliver of the
    // domain, are one: the cuts and the path between them, its vertices
    // and the midpoints of its steps, are all where `excess` is negative
    // and `contains` holds. A path across a region outside the domain
    // that the loops caught has its cuts on that region's chords, over
    // the level. So are the last run and the first of a closed path,
    // across where it begins.
    let inside = |between: &[P2<ParamSpace>]| -> bool {
        between.windows(2).all(|w| {
            let (a, b) = (w[0], w[1]);
            let m = P2::from_vec(0.5 * (a.v + b.v));
            ex(a) < 0.0 && ex(m) < 0.0 && contains_at(m)
        }) && between.last().is_some_and(|&p| ex(p) < 0.0)
    };
    let mut joined: Vec<Run> = Vec::new();
    for next in runs {
        if let Some(previous) = joined.last()
            && let (Some(cut), Some(back)) = (previous.end, next.start)
            && cut <= back
            && let (Some(&x), Some(&y)) = (previous.points.last(), next.points.first())
        {
            let interior: Vec<P2<ParamSpace>> = ((cut + 1)..(back + 1)).map(vertex).collect();
            let mut between = vec![x];
            between.extend(interior.iter().copied());
            between.push(y);
            if inside(&between) {
                let last = joined.len() - 1;
                let mut points = joined[last].points[..joined[last].points.len() - 1].to_vec();
                points.extend(interior);
                points.extend(next.points[1..].iter().copied());
                joined[last] = Run { points, start: joined[last].start, end: next.end };
                continue;
            }
        }
        joined.push(next);
    }
    if closed
        && let (Some(last), Some(first)) = (joined.last().cloned(), joined.first().cloned())
        && let (Some(cut), Some(back)) = (last.end, first.start)
        && cut < back + n
        && let (Some(&x), Some(&y)) = (last.points.last(), first.points.first())
    {
        let interior: Vec<P2<ParamSpace>> = ((cut + 1)..(back + n + 1)).map(vertex).collect();
        let y_shifted = P2::from_vec(y.v + shift);
        let mut between = vec![x];
        between.extend(interior.iter().copied());
        between.push(y_shifted);
        if inside(&between) {
            let mut points = last.points[..last.points.len() - 1].to_vec();
            points.extend(interior);
            if joined.len() == 1 {
                points.push(y_shifted);
                joined[0] = Run { points, start: None, end: None };
            } else {
                points.extend(first.points[1..].iter().map(|p| P2::from_vec(p.v + shift)));
                let k = joined.len() - 1;
                joined[k] = Run { points, start: last.start, end: first.end.map(|e| e + n) };
                joined.remove(0);
            }
        }
    }
    let mut out: ParamPaths = Vec::new();
    for Run { points, start, end } in joined {
        let mut points = points;
        if let Some(end) = end
            && let Some(&last) = points.last()
            && let Some(more) = carried(last.v, end, true)
        {
            points.pop();
            points.extend(more);
        }
        if let Some(start) = start
            && let Some(&first) = points.first()
            && let Some(more) = carried(first.v, start, false)
        {
            points.remove(0);
            let mut front: Vec<P2<ParamSpace>> = more.into_iter().rev().collect();
            front.extend(points);
            points = front;
        }
        // A run round the whole of a closed path.
        let round = closed && points.len() >= 2 && points[points.len() - 1].v - points[0].v == shift;
        // Kept where `excess` is negative, cut where it changes sign. A
        // chord whose midpoint lies on the other side from both its ends
        // passes through a hole in the domain, or a region outside it,
        // between them, and is read at its midpoint too.
        let mut read: Vec<P2<ParamSpace>> = Vec::new();
        let mut inside_at: Vec<bool> = Vec::new();
        for &p in &points {
            let is_inside = ex(p) < 0.0;
            if let Some(&q) = read.last()
                && is_inside == inside_at[inside_at.len() - 1]
            {
                let m = P2::from_vec(0.5 * (q.v + p.v));
                if (ex(m) < 0.0) != is_inside {
                    read.push(m);
                    inside_at.push(!is_inside);
                }
            }
            read.push(p);
            inside_at.push(is_inside);
        }
        let points = read;
        let mut pieces: ParamPaths = Vec::new();
        let mut piece: Vec<P2<ParamSpace>> = Vec::new();
        let add = |piece: &mut Vec<P2<ParamSpace>>, p: Vec2| {
            if piece.last().map(|q| q.v) != Some(p) {
                piece.push(P2::from_vec(p));
            }
        };
        for (k, p) in points.iter().enumerate() {
            if k > 0 && inside_at[k] != inside_at[k - 1] {
                let (i, o) = if inside_at[k] { (p.v, points[k - 1].v) } else { (points[k - 1].v, p.v) };
                add(&mut piece, bisect_points(i, o, |q| ex(P2::from_vec(q)) < 0.0).0);
                if !inside_at[k] {
                    if piece.len() >= 2 {
                        pieces.push(std::mem::take(&mut piece));
                    }
                    piece.clear();
                }
            }
            if inside_at[k] {
                add(&mut piece, p.v);
            }
        }
        if piece.len() >= 2 {
            pieces.push(piece);
        }
        // Round a closed path, the piece through its end and the piece
        // through its start are one, as the runs are.
        if round && pieces.len() >= 2 && inside_at[0] && inside_at[inside_at.len() - 1] {
            let first = pieces.remove(0);
            let k = pieces.len() - 1;
            pieces[k].extend(first[1..].iter().map(|p| P2::from_vec(p.v + shift)));
        }
        out.extend(pieces);
    }
    out
}

/// The segments of a face's trim loops, bucketed on a grid over their
/// bounding box, so a segment of a contour, or a ray along +u, is met only
/// with those near it. On a periodic axis a segment is met with the loops a
/// period along too.
pub struct TrimIndex {
    starts: Vec<Vec2>,
    ends: Vec<Vec2>,
    lo: Vec2,
    hi: Vec2,
    cell: Vec2,
    columns: usize,
    rows: usize,
    buckets: Vec<Vec<u32>>,
    /// By rows alone, for the ray: each segment once in each row it spans.
    strips: Vec<Vec<u32>>,
    periods: (Option<f64>, Option<f64>),
}

impl TrimIndex {
    /// Each loop ends where it begins, or a period along; its segments are
    /// what is met, so a chain left open is met only where it runs.
    pub fn new(loops: &[Vec<P2<ParamSpace>>], periods: (Option<f64>, Option<f64>)) -> TrimIndex {
        let mut starts: Vec<Vec2> = Vec::new();
        let mut ends: Vec<Vec2> = Vec::new();
        for lp in loops {
            for w in lp.windows(2) {
                if w[0] != w[1] {
                    starts.push(w[0].v);
                    ends.push(w[1].v);
                }
            }
        }
        let mut lo = Vec2::new(f64::INFINITY, f64::INFINITY);
        let mut hi = -lo;
        for (a, b) in starts.iter().zip(&ends) {
            lo = lo.min(a.min(*b));
            hi = hi.max(a.max(*b));
        }
        let n = ((starts.len() as f64).sqrt() as usize).clamp(1, 1024);
        let extent = hi - lo;
        let cell = Vec2::new(
            if extent.x > 0.0 { extent.x / n as f64 } else { 1.0 },
            if extent.y > 0.0 { extent.y / n as f64 } else { 1.0 },
        );
        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); if starts.is_empty() { 0 } else { n * n }];
        let mut strips: Vec<Vec<u32>> = vec![Vec::new(); if starts.is_empty() { 0 } else { n }];
        for s in 0..starts.len() {
            let (i0, j0, i1, j1) = TrimIndex::cells(starts[s].min(ends[s]), starts[s].max(ends[s]), lo, cell, n, n);
            for j in j0..=j1 {
                strips[j].push(s as u32);
                for i in i0..=i1 {
                    buckets[j * n + i].push(s as u32);
                }
            }
        }
        TrimIndex { starts, ends, lo, hi, cell, columns: n, rows: n, buckets, strips, periods }
    }

    /// Whether the loops wind round `p` an odd number of times: a ray from
    /// it along +u crosses an odd number of their segments, each half open
    /// in v, so a ray through a vertex counts it once. The loops' own
    /// plane: no period is applied.
    pub fn encloses(&self, p: Vec2) -> bool {
        if self.starts.is_empty() || !(p.y >= self.lo.y && p.y <= self.hi.y) {
            return false;
        }
        let (_, j, _, _) = TrimIndex::cells(p, p, self.lo, self.cell, self.columns, self.rows);
        let mut inside = false;
        for &s in &self.strips[j] {
            let (a, b) = (self.starts[s as usize], self.ends[s as usize]);
            if (a.y > p.y) == (b.y > p.y) {
                continue;
            }
            if p.x < a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y) {
                inside = !inside;
            }
        }
        inside
    }

    /// The bucket of a coordinate. Swift's `Int(f)` traps on a coordinate
    /// that is not a number; the port files it in the first bucket.
    fn cells(a: Vec2, b: Vec2, lo: Vec2, cell: Vec2, columns: usize, rows: usize) -> (usize, usize, usize, usize) {
        let index = |x: f64, lo: f64, size: f64, count: usize| -> usize {
            let f = ((x - lo) / size).floor();
            if f < 0.0 {
                0
            } else if f >= count as f64 {
                count - 1
            } else {
                swift::int(f).map_or(0, |i| i as usize)
            }
        };
        (
            index(a.x, lo.x, cell.x, columns),
            index(a.y, lo.y, cell.y, rows),
            index(b.x, lo.x, cell.x, columns),
            index(b.y, lo.y, cell.y, rows),
        )
    }

    /// Where the segment from `p` to `q` crosses a trim loop, strictly
    /// between its ends, in order from `p`: each a point of the loop. Two
    /// crossings a rounding apart along the segment are one; a loop that runs
    /// along the segment crosses it at its own vertices.
    pub fn crossings(&self, p: Vec2, q: Vec2) -> Vec<Vec2> {
        if self.starts.is_empty() {
            return Vec::new();
        }
        // A rounding of the coordinates: how far off the segment a loop may
        // lie and still be met, so a loop along it is met however the two
        // were rounded.
        let pad = 1e-12 * swift::max(swift::max(p.abs().reduce_max(), q.abs().reduce_max()), (q - p).length());
        let qlo = p.min(q) - Vec2::splat(pad);
        let qhi = p.max(q) + Vec2::splat(pad);
        // The whole periods by which the loops are moved to meet the segment.
        let shifts = |period: Option<f64>, qlo: f64, qhi: f64, lo: f64, hi: f64| -> Vec<i64> {
            let Some(period) = period.filter(|&p| p > 0.0) else {
                return vec![0];
            };
            let (Some(a), Some(b)) =
                (swift::int(((qlo - hi) / period).ceil()), swift::int(((qhi - lo) / period).floor()))
            else {
                return Vec::new();
            };
            if a <= b { (a..=b).collect() } else { Vec::new() }
        };
        let us = shifts(self.periods.0, qlo.x, qhi.x, self.lo.x, self.hi.x);
        let vs = shifts(self.periods.1, qlo.y, qhi.y, self.lo.y, self.hi.y);
        let d = q - p;
        let dd = d.dot(d);
        if !(dd > 0.0) {
            return Vec::new();
        }
        let cross = |a: Vec2, b: Vec2| a.x * b.y - a.y * b.x;
        let mut found: Vec<(f64, Vec2)> = Vec::new();
        for &ku in &us {
            for &kv in &vs {
                let shift =
                    Vec2::new(ku as f64 * self.periods.0.unwrap_or(0.0), kv as f64 * self.periods.1.unwrap_or(0.0));
                let blo = qlo - shift;
                let bhi = qhi - shift;
                if !(blo.x <= self.hi.x && bhi.x >= self.lo.x && blo.y <= self.hi.y && bhi.y >= self.lo.y) {
                    continue;
                }
                let (i0, j0, i1, j1) = TrimIndex::cells(blo, bhi, self.lo, self.cell, self.columns, self.rows);
                let mut near: Vec<u32> = Vec::new();
                for j in j0..=j1 {
                    for i in i0..=i1 {
                        near.extend_from_slice(&self.buckets[j * self.columns + i]);
                    }
                }
                if i0 != i1 || j0 != j1 {
                    near.sort();
                }
                for (n, &s) in near.iter().enumerate() {
                    if n > 0 && near[n - 1] == s {
                        continue;
                    }
                    let a = self.starts[s as usize] + shift;
                    let b = self.ends[s as usize] + shift;
                    if !(a.min(b).x <= qhi.x && a.max(b).x >= qlo.x && a.min(b).y <= qhi.y && a.max(b).y >= qlo.y) {
                        continue;
                    }
                    let e = b - a;
                    let w = a - p;
                    let den = cross(d, e);
                    let length = dd.sqrt();
                    if den.abs() <= 1e-14 * length * e.length() {
                        // Parallel: crossed only where it runs along the
                        // segment, at the loop's own vertices.
                        if !(cross(w, d).abs() <= length * pad) {
                            continue;
                        }
                        for x in [a, b] {
                            let t = (x - p).dot(d) / dd;
                            if t > 0.0 && t < 1.0 {
                                found.push((t, x));
                            }
                        }
                        continue;
                    }
                    let t = cross(w, e) / den;
                    let u = cross(w, d) / den;
                    if !(t > 0.0 && t < 1.0 && (-1e-12..=1.0 + 1e-12).contains(&u)) {
                        continue;
                    }
                    found.push((t, a + u.clamp(0.0, 1.0) * e));
                }
            }
        }
        // Swift's tuple `<`: by t, then x, then y; a NaN compares false
        // throughout, as there.
        found.sort_by(|l, r| {
            let less = |a: &(f64, Vec2), b: &(f64, Vec2)| {
                if a.0 != b.0 {
                    a.0 < b.0
                } else if a.1.x != b.1.x {
                    a.1.x < b.1.x
                } else {
                    a.1.y < b.1.y
                }
            };
            if less(l, r) {
                std::cmp::Ordering::Less
            } else if less(r, l) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
        let mut out: Vec<Vec2> = Vec::new();
        let mut last = f64::NEG_INFINITY;
        for (t, point) in found {
            if t > 1e-12 && t < 1.0 - 1e-12 && t - last > 1e-12 {
                out.push(point);
                last = t;
            }
        }
        out
    }
}
