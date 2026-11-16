//! Integral curves of a vector field on a face: the port of
//! `Sources/KurvenCore/IntegralCurves.swift`.
//!
//! Swift's `IntegralCurve` is an enum used as a namespace; its nested types
//! are this module's [`End`], [`Section`], [`Limits`] and [`Traced`], read
//! as `integral_curves::End` where Swift reads `IntegralCurve.End`.
//!
//! Integral curves of a vector field on a face (do Carmo §3-4): the curves
//! whose tangent at every point is the field there. The slope lines of a
//! scalar field are the integral curves of its gradient (Cayley 1859, "On
//! contour and slope lines"); a linear curve on a torus is one of a constant
//! field, and so is a forced system's trajectory on its fitted torus.
//!
//! A field is a vector field (D9). Where its integral curves are known in
//! closed form, as a constant field's straight lines are, they are read off
//! it (`VectorField::flow`); otherwise they are integrated on the face's
//! parameter plane with RK4 at unit speed, from a seed both ways
//! (`Face::integral_curve`). A field is signed, as a slope is, or stands for
//! a field of line elements, whose sign means nothing, as the principal
//! directions do (`VectorField::signed`).
//!
//! The integrator keeps the heading continuous, by two rules that do not
//! meet. Of an unsigned field, every value sampled, each RK4 stage, is first
//! oriented to agree with the heading: of its two signs, the one within a
//! right angle of it (D9). Then, of either field, a step is refused and
//! halved where a stage turns too far from the heading the step began with:
//! a right angle or more for a signed field, which has turned against the
//! heading only where a singular point lies within the step, as a zero or a
//! pole of f does for the slope of |f|, so no step crosses one. An oriented
//! stage never turns a right angle, so for an unsigned field the bound is
//! half that, 45°: past it the line element is nearer perpendicular to the
//! heading than along it, its orientation no longer clear, and the step is
//! halved until it is, which keeps the step off an umbilic, where the
//! principal directions turn fast. Each step is RK4 taken once whole and
//! once in two halves, and is refused where the two disagree by more than
//! `accuracy`, where the chord to it sags off the curve by more than half
//! the refiner's tolerance, or, on the graph, where the lifted chord sags
//! off the graph by as much: so every chord of the polyline is within the
//! tolerance of the curve it stands for, on the face, and in space but where
//! the graph is steeper than that resolves. A step is halved for its sag in
//! space at most five times below the longest step
//! (`Limits::lifted_halvings`), as the plain contour routine halves a grid
//! chord at most five times, so a steep but smooth graph costs a bounded
//! number of vertices a cell and is followed to its end. A curve ends:
//!
//! - at a singular point, where the field is undefined or every step turns
//!   against the heading, so the steps refused there have shrunk below
//!   `least_step`: a zero of f, which the curve runs down into, carried to
//!   the point itself when the field can name it
//!   (`VectorField::singular_point`), or a pole, which it runs up into; or
//!   an umbilic, where the principal directions are undefined
//!   (`Face.principalDirections(of:on:)`, Phase 4);
//! - where it leaves the copy's trimmed domain, at the domain's edge, the
//!   footprint's perimeter or the truncation edge, bisected on membership
//!   (`FaceCopy::contains`) along the step to the last bit, so on the
//!   truncation edge it ends where |f| is the truncation height;
//! - back at its seed, a closed curve, which needs nothing traced backward:
//!   on a periodic face, at the seed's own image a whole number of periods
//!   along, across the seams;
//! - back on the level set it was seeded on, where it is traced to return
//!   to it, once round a closed face ([`Section`]), which needs nothing
//!   traced backward either;
//! - at `max_steps` steps each way, the step limit.
//!
//! Swift traces the seeds of a layer in parallel, each thread with a refiner
//! of its own (`Heightfield.parallelMap`, `ContourRefine.fork`); here they
//! are traced in turn with one fork, and each curve is the same.

use std::sync::Arc;

use kurven_math::{real, swift};

use crate::bisection::bisect;
use crate::face::{Face, FaceCopy};
use crate::hatch::HatchContext;
use crate::hatching::FaceSelector;
use crate::manifest::{Affine2, ContourField, DirectionField, LayerSource, Region, Seeds, Spacing};
use crate::parametric::{ParamScalar, SurfaceJet};
use crate::scalar_field::{ScalarField, short_way};
use crate::simd::{Vec2, Vec3};
use crate::spaces::{P2, P3, ParamSpace, WorldSpace};
use crate::surface::{ContourRefine, Surface};

/// The field at a point of the face's parameter plane, in its coordinates;
/// None where it is undefined or zero: at a singular point. Of an unsigned
/// field, either sign of the line element there.
pub type FieldValue = Arc<dyn Fn(P2<ParamSpace>) -> Option<Vec2> + Send + Sync>;
/// The point a time `t` along the integral curve through `p`.
pub type Flow = Arc<dyn Fn(P2<ParamSpace>, f64) -> P2<ParamSpace> + Send + Sync>;
/// The singular point an integral curve that ends beside `p` runs into.
pub type SingularPoint = Arc<dyn Fn(P2<ParamSpace>) -> Option<P2<ParamSpace>> + Send + Sync>;

/// A vector field on a face's parameter plane (the module's doc says how
/// its integral curves are traced, and where they end).
#[derive(Clone)]
pub struct VectorField {
    /// The field at a point of the face's parameter plane, in its
    /// coordinates; None where it is undefined or zero: at a singular point.
    /// Of an unsigned field, either sign of the line element there.
    pub value: FieldValue,
    /// The point a time `t` along the integral curve through `p`, where the
    /// curves are known in closed form; None where they are integrated.
    pub flow: Option<Flow>,
    /// The singular point an integral curve that ends beside `p` runs into,
    /// where the field knows its singular points: a zero of f, for the slope
    /// of |f|. None, or None from it, where it does not.
    pub singular_point: Option<SingularPoint>,
    /// Whether the field's sign means anything: false where it stands for a
    /// field of line elements, the principal directions, whose value has
    /// either sign at a point and is oriented by the heading as the curve is
    /// traced. A line field is a vector field so (D9); there is no type of
    /// its own.
    pub signed: bool,
}

impl VectorField {
    /// Swift's `init(value:flow:singularPoint:signed:)` with its defaults: no
    /// flow, no singular points, signed.
    pub fn new(value: FieldValue) -> VectorField {
        VectorField { value, flow: None, singular_point: None, signed: true }
    }

    /// The constant field `w`: its integral curves are the straight lines
    /// `p + t w`, in closed form, so a curve sampled at whole times is the
    /// line stepped by `w`, each coordinate `p + t · w` to the bit.
    pub fn constant(w: Vec2) -> VectorField {
        VectorField {
            value: Arc::new(move |_| Some(w)),
            flow: Some(Arc::new(move |p, t| P2::new(p.x() + t * w.x, p.y() + t * w.y))),
            singular_point: None,
            signed: true,
        }
    }
}

/// Why a traced integral curve ends (`IntegralCurve.End`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum End {
    /// At a singular point of the field: the steps refused there have
    /// shrunk below `Limits::least_step`.
    Singular,
    /// Where it leaves the copy's trimmed domain.
    Boundary,
    /// Back at its seed.
    Closed,
    /// Back across the level set its seed lies on, once round, where it
    /// was traced to return to it ([`Section`]). The seed is on it too, so
    /// such a curve is traced forward alone, and both its ends are
    /// `Returned`.
    Returned,
    /// At `Limits::max_steps`.
    StepLimit,
}

/// The level set a curve is traced back to, as the first return to a
/// Poincaré section is (`IntegralCurve.Section`): the seeds' own (`Seeds`),
/// a scalar field at a level, read the short way round a circle-valued
/// field's `period`. On a closed face, as a torus is, a curve with no
/// singular point to run into need never end, and one that does not close
/// winds round for ever. Ended at its first return, each seed draws its
/// curve once round, from that level set back to it, with nothing traced
/// backward, as a closed curve has nothing; traced both ways, each seed
/// would draw two turns, and the second of one seed's would run beside the
/// first of another's, a double line.
///
/// The return is counted on the field read continuously along the curve
/// from its seed, a circle-valued one unwrapped. Of a circle-valued field,
/// as a coordinate of a periodic axis is, it is where the field has run a
/// whole period from the seed's level, either way: a curve that turns back
/// across the section and winds round the other way returns too, and one
/// that only wanders back across it and on again does not. Of another
/// field, which cannot wind, it is where the curve crosses the level again
/// the way it left it: a meridian of the round torus seeded on its equators,
/// z = 0, crosses the inner one downward, runs on and closes.
///
/// A curve that returns away from its seed is followed on for
/// `Limits::closing_returns` more returns: if it closes on its seed within
/// them it is kept whole, closed, and otherwise cut at its first return. So
/// a closed curve that crosses the section up to that many times more than
/// once is drawn whole, as a (p, q) torus curve crossing u = 0 p times is
/// for p up to four; past that it is cut at its first return, a part of it.
/// Following on costs that many more turns of every curve that does not
/// close.
#[derive(Clone)]
pub struct Section {
    pub value: ParamScalar,
    pub level: f64,
    pub period: Option<f64>,
}

impl Section {
    pub fn new(value: ParamScalar, level: f64, period: Option<f64>) -> Section {
        Section { value, level, period }
    }

    /// The field less the level at `p`, the short way round.
    pub fn residual(&self, p: Vec2) -> f64 {
        let d = (self.value)(P2::from_vec(p)) - self.level;
        match self.period {
            Some(period) => short_way(d, period),
            None => d,
        }
    }
}

/// The tolerances a curve is traced to, lengths in cells of the face's
/// parameter rectangle (`Face::cell`). `Default` is Swift's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Limits {
    /// How far a chord may depart from the curve it stands for: the
    /// refiner's tolerance (`ContourRefine::tolerance`). A step's chord is
    /// held to half of it, on the face and lifted, and a curve closes where
    /// it passes within half of it of its seed.
    pub tolerance: f64,
    /// The longest step.
    pub largest_step: f64,
    /// The shortest step: one refused below it ends the curve at a singular
    /// point.
    pub least_step: f64,
    /// How far a whole RK4 step may land from the same step in two halves.
    pub accuracy: f64,
    /// The most steps each way from the seed.
    pub max_steps: i64,
    /// How many times a step may be halved below the longest for its
    /// chord's sag in space: past it the chord is kept.
    pub lifted_halvings: i64,
    /// How many more returns to its section a curve that comes back to it
    /// away from its seed is followed for, to see whether it closes on its
    /// seed ([`Section`]): a closed curve that crosses the section up to
    /// this many times more is drawn whole.
    pub closing_returns: i64,
}

impl Default for Limits {
    fn default() -> Limits {
        Limits {
            tolerance: 0.02,
            largest_step: 0.5,
            least_step: 1e-4,
            accuracy: 1e-7,
            max_steps: 20_000,
            lifted_halvings: 5,
            closing_returns: 3,
        }
    }
}

/// An integral curve traced from a seed both ways (`IntegralCurve.Traced`):
/// its polyline in the parameter plane, from its backward end to its forward
/// end, and why each end is where it is. A closed curve begins and ends at
/// its seed, both ends `Closed`; one that returns to its section begins at
/// its seed and ends back on the section, both ends `Returned`.
#[derive(Debug, Clone, PartialEq)]
pub struct Traced {
    pub path: Vec<P2<ParamSpace>>,
    pub backward: End,
    pub forward: End,
}

/// The curve as it ends: cut at its first return where it came back to its
/// section and did not close (`trace`'s `finish`).
fn finish(cut: &Option<Vec<Vec2>>, path: Vec<Vec2>, end: End) -> (Vec<Vec2>, End) {
    if let Some(cut) = cut
        && end != End::Closed
    {
        return (cut.clone(), End::Returned);
    }
    (path, end)
}

impl Face {
    /// The length the face's tolerances are measured in: the larger spacing
    /// of its parameter rectangle's grid. On the graph, a cell of the grid.
    pub fn cell(&self) -> f64 {
        swift::max(self.domain.u.spacing().abs(), self.domain.v.spacing().abs())
    }

    /// The integral curve of `field` through `seed` on `copy`, traced both
    /// ways from it ([`VectorField`]), or None where the seed is off the
    /// copy's trimmed domain or the field is undefined there. Coordinates run
    /// on along a periodic axis, unwrapped, so the polyline never jumps a
    /// period.
    ///
    /// Given a `section`, the level set the seed lies on, the curve ends at
    /// its first return to it, once round ([`Section`]), bisected onto it,
    /// unless it closes within a few more returns, and a curve that returns
    /// is not traced backward; where the forward way ends otherwise first,
    /// the backward way ends at its own first return. A return counts once
    /// the curve is twice the tolerance from its seed, and only where it
    /// crosses the section at its seed, at more than a hundredth of a radian,
    /// so a curve that runs along the level set it was seeded on, as a
    /// meridian of the round torus seeded on itself does, is not ended by
    /// rounding.
    ///
    /// Swift's defaults: `Limits::default()`, no section.
    pub fn integral_curve(
        &self,
        field: &VectorField,
        seed: P2<ParamSpace>,
        copy: &FaceCopy,
        limits: &Limits,
        section: Option<&Section>,
    ) -> Option<Traced> {
        if !(copy.contains)(seed) {
            return None;
        }
        let w = (field.value)(seed)?;
        if !(w.length() > 0.0 && w.length().is_finite()) {
            return None;
        }
        // Whether the curve crosses the section at its seed: the section's
        // gradient, by central differences a ten-thousandth of a cell wide,
        // against the field there.
        let crossing = section.and_then(|section| {
            let (h, x) = (1e-4 * self.cell(), seed.v);
            let g = Vec2::new(
                section.residual(x + Vec2::new(h, 0.0)) - section.residual(x - Vec2::new(h, 0.0)),
                section.residual(x + Vec2::new(0.0, h)) - section.residual(x - Vec2::new(0.0, h)),
            );
            let along = g.length() * w.length();
            if along > 0.0 && g.dot(w).abs() > 0.01 * along { Some(section) } else { None }
        });
        let (forward, forward_end) = self.trace(field, seed.v, 1.0, copy, limits, crossing);
        if forward_end == End::Closed || forward_end == End::Returned {
            return Some(Traced {
                path: forward.into_iter().map(P2::from_vec).collect(),
                backward: forward_end,
                forward: forward_end,
            });
        }
        let (backward, backward_end) = self.trace(field, seed.v, -1.0, copy, limits, crossing);
        let path: Vec<P2<ParamSpace>> =
            backward.into_iter().rev().chain(forward.into_iter().skip(1)).map(P2::from_vec).collect();
        Some(Traced { path, backward: backward_end, forward: forward_end })
    }

    /// One way of an integral curve from `seed`, along the field's own sign
    /// or against it: the polyline from the seed, and why it ends.
    fn trace(
        &self,
        field: &VectorField,
        seed: Vec2,
        sign: f64,
        copy: &FaceCopy,
        limits: &Limits,
        section: Option<&Section>,
    ) -> (Vec<Vec2>, End) {
        let cell = self.cell();
        let (largest, least) = (limits.largest_step * cell, limits.least_step * cell);
        let (accuracy, sag) = (limits.accuracy * cell, 0.5 * limits.tolerance * cell);
        let close_to = 0.5 * limits.tolerance * cell;
        let periods = Vec2::new(self.domain.u.period().unwrap_or(0.0), self.domain.v.period().unwrap_or(0.0));
        let lift = self.lift.clone();
        // The cosine a stage must exceed against the heading: a right
        // angle's for a signed field, 45°'s for a line field, its stages
        // oriented first.
        let turn = if field.signed { 0.0 } else { 0.5f64.sqrt() };
        let inside = |p: Vec2| (copy.contains)(P2::from_vec(p));
        // The unit direction of travel at `p`, None at a singular point; of
        // an unsigned field, oriented to agree with `heading`.
        let direction = |p: Vec2, heading: Option<Vec2>| -> Option<Vec2> {
            let w = (field.value)(P2::from_vec(p))?;
            let n = w.length();
            if !(n > 0.0 && n.is_finite()) {
                return None;
            }
            let d = sign * w / n;
            if !field.signed
                && let Some(heading) = heading
                && d.dot(heading) < 0.0
            {
                return Some(-d);
            }
            Some(d)
        };
        // One RK4 step of length `h` from `p`, whose direction is `k1`;
        // None where a stage is undefined or turns from `heading` as far as
        // `turn` allows or further.
        let step = |p: Vec2, h: f64, k1: Vec2, heading: Vec2| -> Option<Vec2> {
            let k2 = direction(p + 0.5 * h * k1, Some(heading))?;
            if !(k2.dot(heading) > turn) {
                return None;
            }
            let k3 = direction(p + 0.5 * h * k2, Some(heading))?;
            if !(k3.dot(heading) > turn) {
                return None;
            }
            let k4 = direction(p + h * k3, Some(heading))?;
            if !(k4.dot(heading) > turn) {
                return None;
            }
            let middle = k2 + k3;
            let sum = k1 + k4 + 2.0 * middle;
            Some(p + (h / 6.0) * sum)
        };
        // The singular point the curve ends at, beside `p`, where the field
        // names one on the copy.
        let ended = |path: Vec<Vec2>| -> Vec<Vec2> {
            let (Some(&p), Some(named)) = (path.last(), field.singular_point.as_ref()) else { return path };
            let Some(s) = named(P2::from_vec(p)) else { return path };
            if !(inside(s.v) && (s.v - p).length() <= limits.tolerance * cell && s.v != p) {
                return path;
            }
            let mut out = path;
            out.push(s.v);
            out
        };

        // The shortest step whose chord is still held to the face in space.
        let lifted_least = largest * real::pow(0.5, limits.lifted_halvings as f64);
        let mut path = vec![seed];
        let mut p = seed;
        let mut lifted: Option<Vec3> = None;
        let (mut h, mut steps, mut attempts) = (largest, 0i64, 0i64);
        let mut left = false;
        // The section's residual at the last vertex, and unwrapped along the
        // curve from its seed, so that on a circle-valued field it runs on
        // past a period; the side the curve left its seed for; how often it
        // has come back; and the curve cut at its first return, kept while
        // it is followed on to see whether it closes.
        let mut residual = section.map_or(0.0, |s| s.residual(seed));
        let mut unwrapped = residual;
        let mut departure = 0.0;
        let mut returns = 0i64;
        let mut cut: Option<Vec<Vec2>> = None;
        let lookahead = if sign > 0.0 { limits.closing_returns } else { 0 };
        let start = direction(seed, None);
        let mut heading = start;
        loop {
            if !(steps < limits.max_steps && attempts < 10 * limits.max_steps) {
                return finish(&cut, path, End::StepLimit);
            }
            attempts += 1;
            let Some(k1) = heading else { return finish(&cut, ended(path), End::Singular) };
            // The step whole, and in two halves; the halves are kept.
            let halves = (|| {
                let whole = step(p, h, k1, k1)?;
                let mid = step(p, 0.5 * h, k1, k1)?;
                let k1_mid = direction(mid, Some(k1))?;
                if !(k1_mid.dot(k1) > turn) {
                    return None;
                }
                let q = step(mid, 0.5 * h, k1_mid, k1)?;
                Some((whole, mid, k1_mid, q))
            })();
            let Some((whole, mid, k1_mid, q)) = halves else {
                h *= 0.5;
                if h < least {
                    return finish(&cut, ended(path), End::Singular);
                }
                continue;
            };
            let error = (q - whole).length();
            let mut sagged = (mid - 0.5 * (p + q)).length() > sag;
            if !sagged
                && let Some(lift) = &lift
                && h > lifted_least
            {
                let from = lifted.unwrap_or_else(|| lift(P2::from_vec(p)));
                lifted = Some(from);
                sagged = (lift(P2::from_vec(mid)) - 0.5 * (from + lift(P2::from_vec(q)))).length() > sag;
            }
            if !(error <= accuracy) || sagged {
                h *= 0.5;
                if h < least {
                    return finish(&cut, ended(path), End::Singular);
                }
                continue;
            }
            // Out of the trimmed domain within the step: the half that
            // leaves it bisected on membership, a step of that half's own
            // from its start, and the last point inside kept. At the half's
            // whole length the step is the half itself, outside, and at none
            // its start, inside.
            if !inside(mid) || !inside(q) {
                let (from, k) = if inside(mid) { (mid, k1_mid) } else { (p, k1) };
                if from == mid {
                    path.push(mid);
                }
                let t = bisect(0.0, 0.5 * h, |t| step(from, t, k, k1).map(inside).unwrap_or(false)).0;
                if t > 0.0
                    && let Some(end) = step(from, t, k, k1)
                {
                    path.push(end);
                }
                return finish(&cut, path, End::Boundary);
            }
            // Back at the seed, heading the way it set out: closed, on the
            // seed's own image a whole number of periods along.
            if sign > 0.0
                && left
                && let Some(start) = start
                && k1.dot(start) > 0.0
            {
                let mut image = seed;
                for axis in 0..2 {
                    if periods[axis] > 0.0 {
                        image[axis] += periods[axis] * ((q[axis] - seed[axis]) / periods[axis]).round();
                    }
                }
                let d = q - p;
                let t =
                    if d.dot(d) > 0.0 { swift::min(swift::max((image - p).dot(d) / d.dot(d), 0.0), 1.0) } else { 0.0 };
                if (p + t * d - image).length() <= close_to {
                    path.push(image);
                    return (path, End::Closed);
                }
            }
            // Back across the section it was seeded on, once round
            // ([`Section`]): the half that crosses it bisected onto it. The
            // residual is unwrapped from the last vertex across each half, a
            // step being far shorter than half a period.
            if let Some(section) = section {
                let period = section.period;
                let (p0, u0, so_far) = (residual, unwrapped, returns);
                // The unwrapped residual at a point a step from the last vertex.
                let along = |x: Vec2| -> f64 {
                    let r = section.residual(x);
                    match period {
                        Some(period) => u0 + short_way(r - p0, period),
                        None => r,
                    }
                };
                // Whether the unwrapped residual `r` is short of the next
                // return: a whole period more from the seed's level, either
                // way, on a circle-valued field; back across the level the
                // way the curve left it on another.
                let short = |r: f64| -> bool {
                    match period {
                        Some(period) => r.abs() < (so_far + 1) as f64 * period,
                        None => r * departure < 0.0 || departure == 0.0,
                    }
                };
                let next = along(q);
                if left && departure != 0.0 && !short(next) && short(u0) {
                    let first = !short(along(mid));
                    let (from, k) = if first { (p, k1) } else { (mid, k1_mid) };
                    let mut back = path.clone();
                    if !first {
                        back.push(mid);
                    }
                    let t = bisect(0.0, 0.5 * h, |t| step(from, t, k, k1).map(|x| short(along(x))).unwrap_or(false));
                    if let Some(end) = step(from, 0.5 * (t.0 + t.1), k, k1) {
                        back.push(end);
                    }
                    returns += 1;
                    if returns == 1 {
                        cut = Some(back);
                    }
                    if returns > lookahead {
                        // The cut was kept at the first return, before this.
                        return (cut.take().unwrap_or_default(), End::Returned);
                    }
                }
                residual = section.residual(q);
                unwrapped = next;
            }
            path.push(q);
            p = q;
            lifted = None;
            // The direction at q is the next step's first stage.
            heading = direction(q, Some(k1));
            steps += 1;
            if !left && (q - seed).length() > 4.0 * close_to {
                left = true;
                departure = if unwrapped > 0.0 {
                    1.0
                } else if unwrapped < 0.0 {
                    -1.0
                } else {
                    0.0
                };
            }
            let grow = if error > 0.0 { swift::min(2.0, 0.9 * real::pow(accuracy / error, 0.2)) } else { 2.0 };
            h = swift::min(largest, h * swift::max(grow, 1.0));
        }
    }

    /// The slope of a scalar field on the face: its gradient in the metric
    /// the face's map induces, `g⁻¹ dφ`, so its integral curves cross the
    /// field's level sets at right angles on the face, not merely in the
    /// parameter plane. `dφ` by central differences of `value` a
    /// ten-thousandth of a cell wide, as the graph's normal is read
    /// (`Surface::graph_normal`), each taken the short way round a field of
    /// `period`. Undefined where `value` is not a number, the gradient
    /// vanishes or the map's derivatives are degenerate.
    ///
    /// Swift's defaults: no period, no singular points.
    pub fn slope(&self, value: ParamScalar, period: Option<f64>, singular_point: Option<SingularPoint>) -> VectorField {
        let h = 1e-4 * self.cell();
        let (map, gradient) = (self.map.clone(), self.height_gradient.clone());
        let difference = move |a: f64, b: f64| match period {
            Some(period) => short_way(a - b, period),
            None => a - b,
        };
        VectorField {
            value: Arc::new(move |p| {
                let du = difference(value(P2::new(p.x() + h, p.y())), value(P2::new(p.x() - h, p.y()))) / (2.0 * h);
                let dv = difference(value(P2::new(p.x(), p.y() + h)), value(P2::new(p.x(), p.y() - h))) / (2.0 * h);
                if let Some(gradient) = &gradient {
                    return Face::raised_over_graph(Vec2::new(du, dv), gradient(p));
                }
                Face::raised(Vec2::new(du, dv), map(p))
            }),
            flow: None,
            singular_point,
            signed: true,
        }
    }

    /// The slope of the height, the map's z, read off the map's own jet:
    /// `dφ` is the z of its two derivatives. On the graph the height is |f|,
    /// and its slope lines are the lines of steepest ascent and descent; its
    /// jet is the height's derivatives alone there (`height_gradient`), the
    /// same numbers without its point.
    ///
    /// Swift's default: no singular points.
    pub fn height_slope(&self, singular_point: Option<SingularPoint>) -> VectorField {
        let value: FieldValue = match self.height_gradient.clone() {
            Some(gradient) => Arc::new(move |p| {
                let a = gradient(p);
                Face::raised_over_graph(a, a)
            }),
            None => {
                let map = self.map.clone();
                Arc::new(move |p| {
                    let jet = map(p);
                    Face::raised(Vec2::new(jet.du.z, jet.dv.z), jet)
                })
            }
        };
        VectorField { value, flow: None, singular_point, signed: true }
    }

    /// A differential raised to a vector over the graph of a height whose
    /// gradient is `a`, where the parameter plane is the world's xy
    /// (Swift's `raised(_:overGraphOf:)`): there g = I + a aᵀ, and
    /// g⁻¹ = (I + a⊥ a⊥ᵀ) / (1 + |a|²) for a⊥ = (−a_y, a_x), since
    /// |a|² I = a aᵀ + a⊥ a⊥ᵀ. Read so, nothing cancels however steep the
    /// graph: the adjugate's terms in |a|³ cancel to |a|, and on a pole's
    /// spire, |a| some 10⁶, turned the direction by a thousandth. The slope
    /// of the height itself, dφ = a, is a / (1 + |a|²) to the bit.
    pub fn raised_over_graph(d: Vec2, a: Vec2) -> Option<Vec2> {
        let perpendicular = Vec2::new(-a.y, a.x);
        let w = (d + perpendicular * perpendicular.dot(d)) / (1.0 + a.length_squared());
        if w == Vec2::ZERO || !w.x.is_finite() || !w.y.is_finite() {
            return None;
        }
        Some(w)
    }

    /// A differential raised to a vector by the first fundamental form of
    /// `jet`: `g⁻¹ dφ`. None where it is not finite, is zero, or `g` is
    /// singular.
    pub fn raised(d: Vec2, jet: SurfaceJet) -> Option<Vec2> {
        let (e, f, g) = (jet.du.dot(jet.du), jet.du.dot(jet.dv), jet.dv.dot(jet.dv));
        let det = e * g - f * f;
        if !(det > 0.0 && det.is_finite() && d.x.is_finite() && d.y.is_finite()) {
            return None;
        }
        let w = Vec2::new(g * d.x - f * d.y, e * d.y - f * d.x) / det;
        if w == Vec2::ZERO || !w.x.is_finite() || !w.y.is_finite() {
            return None;
        }
        Some(w)
    }
}

impl Seeds {
    /// The seeds along `paths`, polylines of the level set in the parameter
    /// plane, in order along each.
    ///
    /// At listed values of a second field, `value`, read the short way round
    /// a field of `period`, and the values the same modulo it taken once.
    /// Where a value falls strictly between a chord's two readings it is
    /// bisected along the chord to the last bit, a chord across the jump half
    /// a period from the value skipped; where a vertex reads the value
    /// exactly, the seed is the vertex, once, whether the field crosses the
    /// value there or touches it, and a closed path's last vertex, its first
    /// again, is not read twice: on a face periodic by `periods` (zero on a
    /// bounded axis), its first a whole number of periods along, as a level
    /// set that closes across a seam ends. Nothing without `value`.
    ///
    /// By arc length, measured on the curve in space, the polyline's vertices
    /// carried there by `position`: a path of length L takes round(L /
    /// spacing) seeds at the middles of as many equal stretches, so a piece
    /// of the level set shorter than half the spacing, a fragment the trimmed
    /// domain cuts off or a small ring, takes none, and no two seeds on one
    /// piece are closer than three quarters of the spacing. A path whose ratio
    /// is not a number or past ten million takes none: a layer's spacing is
    /// held to a cell at least (`Seeds::unsupported`). A negative ratio,
    /// which a decoded spacing cannot give, takes none here; Swift's range
    /// `0..<n` traps on it.
    ///
    /// Swift's defaults: no period, `periods` zero.
    pub fn points(
        &self,
        paths: &[Vec<P2<ParamSpace>>],
        value: Option<&dyn Fn(P2<ParamSpace>) -> f64>,
        period: Option<f64>,
        periods: Vec2,
        position: &dyn Fn(P2<ParamSpace>) -> Vec3,
    ) -> Vec<P2<ParamSpace>> {
        let mut seeds: Vec<P2<ParamSpace>> = Vec::new();
        match &self.spacing {
            Spacing::Values(_, listed) => {
                let Some(value) = value else { return Vec::new() };
                let residual = |x: f64, c: f64| match period {
                    Some(period) => short_way(x - c, period),
                    None => x - c,
                };
                let mut values: Vec<f64> = Vec::new();
                for &c in listed {
                    if !values.iter().any(|&v| residual(v, c).abs() <= 1e-12 * swift::max(1.0, c.abs())) {
                        values.push(c);
                    }
                }
                for path in paths.iter().filter(|p| p.len() >= 2) {
                    let read: Vec<f64> = path.iter().map(|&p| value(p)).collect();
                    let closed = Seeds::closes(path, periods);
                    for k in 0..path.len() {
                        if k > 0 {
                            let (a, b) = (path[k - 1].v, path[k].v);
                            for &c in &values {
                                let (ra, rb) = (residual(read[k - 1], c), residual(read[k], c));
                                if !(ra.is_finite()
                                    && rb.is_finite()
                                    && ra != 0.0
                                    && rb != 0.0
                                    && (ra > 0.0) != (rb > 0.0))
                                {
                                    continue;
                                }
                                if let Some(period) = period
                                    && (ra - rb).abs() > period / 2.0
                                {
                                    continue;
                                }
                                let t = bisect(0.0, 1.0, |t| {
                                    let r = residual(value(P2::from_vec(a + t * (b - a))), c);
                                    r.is_finite() && (r > 0.0) == (ra > 0.0)
                                });
                                seeds.push(P2::from_vec(a + 0.5 * (t.0 + t.1) * (b - a)));
                            }
                        }
                        if closed && k == path.len() - 1 {
                            continue;
                        }
                        for &c in &values {
                            if residual(read[k], c) == 0.0 {
                                seeds.push(path[k]);
                            }
                        }
                    }
                }
            }
            Spacing::ArcLength(spacing) => {
                for path in paths.iter().filter(|p| p.len() >= 2) {
                    let placed: Vec<Vec3> = path.iter().map(|&p| position(p)).collect();
                    let mut lengths = vec![0.0];
                    for k in 1..path.len() {
                        lengths.push(lengths[k - 1] + (placed[k] - placed[k - 1]).length());
                    }
                    let total = lengths[lengths.len() - 1];
                    let ratio = total / spacing;
                    if !(total > 0.0 && ratio.is_finite() && ratio < 1e7) {
                        continue;
                    }
                    // A ratio under ten million fits; a negative one counts no seed.
                    let n = swift::int(ratio.round()).unwrap_or(0).max(0);
                    let mut k = 1;
                    for i in 0..n {
                        let s = (i as f64 + 0.5) * total / n as f64;
                        while k < path.len() - 1 && lengths[k] < s {
                            k += 1;
                        }
                        let chord = lengths[k] - lengths[k - 1];
                        let t = if chord > 0.0 { (s - lengths[k - 1]) / chord } else { 0.0 };
                        seeds.push(P2::from_vec(path[k - 1].v + t * (path[k].v - path[k - 1].v)));
                    }
                }
            }
        }
        seeds
    }

    /// Whether a path ends where it begins, or a whole number of `periods`
    /// along on an axis whose period is not zero, to rounding: a trillionth
    /// of a period.
    pub fn closes(path: &[P2<ParamSpace>], periods: Vec2) -> bool {
        let (Some(a), Some(b)) = (path.first(), path.last()) else { return false };
        let (a, b) = (a.v, b.v);
        if a == b {
            return true;
        }
        for axis in 0..2 {
            let d = b[axis] - a[axis];
            if periods[axis] > 0.0 {
                if !(short_way(d, periods[axis]).abs() <= 1e-12 * periods[axis]) {
                    return false;
                }
            } else if d != 0.0 {
                return false;
            }
        }
        true
    }
}

/// A field on the graph by name (`graphIntegralCurves`' `named`): its scalar
/// field, to contour, and its value and period, to read along a curve.
type Named = (Option<ScalarField>, Option<ParamScalar>, Option<f64>);

impl Surface {
    /// The integral curves of `field` through `seeds` on `faces`, in world
    /// space: on the graph, each copy's (`graph_integral_curves`), lifted to
    /// |f|, f's own with a refiner, and carried to the copy's tile, on every
    /// tile where `tiled`. Nothing on another face, nor where the layer could
    /// not be decoded (`LayerSource::unsupported_integral_curves`).
    ///
    /// Swift's defaults: no refiner, `Limits::default()`.
    pub fn integral_curves(
        &self,
        field: &DirectionField,
        seeds: &Seeds,
        faces: &FaceSelector,
        context: &HatchContext,
        refine: Option<&ContourRefine>,
        limits: &Limits,
    ) -> Vec<Vec<P3<WorldSpace>>> {
        let FaceSelector::Graph { tiled } = faces else { return Vec::new() };
        let magnitude = self.face_magnitude(refine);
        let tiles = if *tiled { context.tiles.clone() } else { vec![Affine2::IDENTITY] };
        self.graph_integral_curves(field, seeds, &tiles, &context.region, refine, limits)
            .into_iter()
            .flat_map(|(tile, curves)| {
                let magnitude = magnitude.clone();
                curves.into_iter().map(move |curve| {
                    curve
                        .path
                        .iter()
                        .map(|p| tile.apply(P3::new(p.x(), p.y(), magnitude(P2::new(p.x(), p.y())))))
                        .collect::<Vec<_>>()
                })
            })
            .collect()
    }

    /// The integral curves of `field` on each copy of the graph, in its
    /// parameter plane, with why each ends, beside the copy's tile: one
    /// through each seed on the copy, the seeds placed along the copy's own
    /// level set of the seeds' field, clipped to its trimmed domain
    /// (`Face::contours`; `Seeds::points`). Copies that differ only in their
    /// tile, with no footprint to cut them, are traced once. The slope of |f|
    /// or of the height is read off the graph's own jet
    /// (`Face::height_slope`), so its normal and its slope lines are read off
    /// the same derivatives; of arg f, u or v by central differences
    /// (`Face::slope`). Both of f's fields end their slope lines on the zeros
    /// of f where the refiner finds one (`ContourRefine::zero`). arg f is
    /// read off f, so without a refiner that has it there are no seeds placed
    /// by it, and no slope of it. Nothing where the layer could not be
    /// decoded or its seeds lie closer than a cell of the grid
    /// (`LayerSource::unsupported_integral_curves`). Swift traces the seeds
    /// in parallel, each thread with a refiner of its own
    /// (`ContourRefine::forked`); here each copy's seeds are traced in turn
    /// on one fork, with the same curves.
    ///
    /// Swift's default: `Limits::default()`.
    pub fn graph_integral_curves(
        &self,
        field: &DirectionField,
        seeds: &Seeds,
        tiles: &[Affine2],
        region: &Region,
        refine: Option<&ContourRefine>,
        limits: &Limits,
    ) -> Vec<(Affine2, Vec<Traced>)> {
        let Some(face) = self.graph(tiles, region, refine) else { return Vec::new() };
        if LayerSource::unsupported_integral_curves(
            field,
            seeds,
            &FaceSelector::Graph { tiled: false },
            Some(face.cell()),
        )
        .is_some()
        {
            return Vec::new();
        }
        let magnitude = self.face_magnitude(refine);
        let phase: Option<ParamScalar> = refine
            .and_then(|r| r.phase.clone())
            .map(|phase| -> ParamScalar { Arc::new(move |p: P2<ParamSpace>| phase(P2::from_vec(p.v))) });
        // The field on a graph read with `refine`, a thread's own.
        let vectors = |face: &Face, refine: Option<&ContourRefine>| -> Option<VectorField> {
            let zero: Option<SingularPoint> = refine.and_then(|r| r.zero.clone()).map(|zero| -> SingularPoint {
                Arc::new(move |p: P2<ParamSpace>| zero(P2::from_vec(p.v)).map(|z| P2::from_vec(z.v)))
            });
            match field {
                DirectionField::Slope(ContourField::Magnitude) | DirectionField::Slope(ContourField::Height) => {
                    Some(face.height_slope(zero))
                }
                DirectionField::Slope(ContourField::Phase) => {
                    let phase = refine?.phase.clone()?;
                    Some(face.slope(
                        Arc::new(move |p: P2<ParamSpace>| phase(P2::from_vec(p.v))),
                        Some(2.0 * std::f64::consts::PI),
                        zero,
                    ))
                }
                DirectionField::Slope(ContourField::U) => {
                    Some(face.slope(Arc::new(|p: P2<ParamSpace>| p.x()), None, None))
                }
                DirectionField::Slope(ContourField::V) => {
                    Some(face.slope(Arc::new(|p: P2<ParamSpace>| p.y()), None, None))
                }
                // An isophote is refused above; lines of curvature lie on a
                // parametric surface alone, whose principal directions are
                // `LinesOfCurvature.swift`'s.
                DirectionField::Slope(ContourField::Isophote) | DirectionField::PrincipalDirection(_) => None,
            }
        };
        if vectors(&face, refine).is_none() {
            return Vec::new();
        }
        let named = |name: ContourField| -> Named {
            let m = magnitude.clone();
            let lifted: ParamScalar = Arc::new(move |p: P2<ParamSpace>| m(P2::new(p.x(), p.y())));
            match name {
                ContourField::U => (Some(ScalarField::new(None, |p| p.x())), Some(Arc::new(|p| p.x())), None),
                ContourField::V => (Some(ScalarField::new(None, |p| p.y())), Some(Arc::new(|p| p.y())), None),
                ContourField::Height => {
                    let l = lifted.clone();
                    (Some(ScalarField::new(None, move |p| l(p))), Some(lifted), None)
                }
                ContourField::Magnitude => {
                    (self.field(ContourField::Magnitude, refine, f64::INFINITY), Some(lifted), None)
                }
                ContourField::Phase => (
                    self.field(ContourField::Phase, refine, f64::INFINITY),
                    phase.clone(),
                    Some(2.0 * std::f64::consts::PI),
                ),
                ContourField::Isophote => (None, None, None),
            }
        };
        let (Some(scalar), Some(lift)) = (named(seeds.field).0, face.lift.clone()) else { return Vec::new() };
        let (at_value, at_period): (Option<ParamScalar>, Option<f64>) = match &seeds.spacing {
            Spacing::Values(of, _) => {
                let n = named(*of);
                (n.1, n.2)
            }
            Spacing::ArcLength(_) => (None, None),
        };
        let traced: Vec<usize> =
            if matches!(region, Region::Inside(_)) { (0..face.copies.len()).collect() } else { vec![0] };
        let curves: Vec<Vec<Traced>> = traced
            .iter()
            .map(|&k| {
                let level: Vec<Vec<P2<ParamSpace>>> = face
                    .contours(&scalar, &[seeds.level], &face.copies[k])
                    .into_iter()
                    .flat_map(|(_, paths)| paths)
                    .collect();
                let value = at_value.as_ref().map(|v| v.as_ref() as &dyn Fn(P2<ParamSpace>) -> f64);
                let points = seeds.points(&level, value, at_period, Vec2::ZERO, &|p| lift(p));
                // Swift's worker: a refiner of its own, f compiled again into
                // closures of its own (`ContourRefine.fork`), and the graph and
                // field on it.
                let own = refine.map(ContourRefine::forked);
                let worker = self
                    .graph(tiles, region, own.as_ref())
                    .and_then(|face| vectors(&face, own.as_ref()).map(|vectors| (face, vectors)));
                points
                    .into_iter()
                    .filter_map(|seed| {
                        let (face, vectors) = worker.as_ref()?;
                        face.integral_curve(vectors, seed, &face.copies[k], limits, None)
                    })
                    .filter(|curve| curve.path.len() >= 2)
                    .collect()
            })
            .collect();
        (0..face.copies.len()).map(|k| (face.copies[k].tile, curves[k.min(curves.len() - 1)].clone())).collect()
    }
}
