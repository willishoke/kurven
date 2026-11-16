//! Parametric surfaces: a map from a parameter rectangle into space, with its
//! derivatives. The port of `Sources/KurvenCore/Parametric.swift`.
//!
//! The whole file: the axes, the jets, the surface's constructors (its grid
//! and its orientation), its facing, its coordinate curves, its fold curves
//! exact and as the grid sees them, its grid's bilinear patch, the linear
//! curves, the round torus (in its angles or isothermal) and the surface
//! displaced along its normal. [`PortRefusal`], the port's typed refusal of
//! what a later phase brings, lives here because Phase 3 first needed it here.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use kurven_math::{real, swift};

use crate::bisection::bisect_points;
use crate::contour;
use crate::grid::Grid2D;
use crate::integral_curves::VectorField;
use crate::manifest::{Domain, Interval};
use crate::polyline::PolylineSet;
use crate::scalar_field::short_way;
use crate::simd::{Vec2, Vec3};
use crate::spaces::{P2, P3, ParamSpace, Transform, ViewSpace, WorldSpace};

/// A point of the parameter plane to a number: a field's value, a
/// membership's verdict as `ParamPredicate`, a truncation's excess.
pub type ParamScalar = Arc<dyn Fn(P2<ParamSpace>) -> f64 + Send + Sync>;
/// Whether a point of the parameter plane is held.
pub type ParamPredicate = Arc<dyn Fn(P2<ParamSpace>) -> bool + Send + Sync>;
/// A point of the parameter plane to a point of space.
pub type ParamLift = Arc<dyn Fn(P2<ParamSpace>) -> Vec3 + Send + Sync>;
/// A point of the parameter plane to a gradient there.
pub type ParamGradient = Arc<dyn Fn(P2<ParamSpace>) -> Vec2 + Send + Sync>;
/// A surface's map: the point of space at a point of the parameter plane,
/// with its two partial derivatives.
pub type JetMap = Arc<dyn Fn(P2<ParamSpace>) -> SurfaceJet + Send + Sync>;
/// A surface's jet of order two at a point of the parameter plane.
pub type SecondOrderMap = Arc<dyn Fn(P2<ParamSpace>) -> SecondOrderJet + Send + Sync>;

/// What the port has not brought across yet, said where it is asked for: the
/// phase that brings it, and what it is. A typed refusal, never a panic, so a
/// caller that reaches a deferred branch gets an answer it can report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortRefusal {
    pub phase: u8,
    pub what: &'static str,
}

impl fmt::Display for PortRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "not ported yet (phase {}): {}", self.phase, self.what)
    }
}

impl std::error::Error for PortRefusal {}

/// One direction of a parameter rectangle.
///
/// A periodic axis closes on itself: `samples` points at `lo + i * length /
/// samples`, and the last cell runs from the last sample back to the first. A
/// bounded axis has `samples` points spanning `[lo, hi]` inclusive, as a
/// `Grid2D` does. The difference is one cell, and getting it wrong leaves a
/// torus with a slit down one side or a sphere with a doubled seam.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamAxis {
    pub range: Interval,
    pub periodic: bool,
    pub samples: usize,
}

impl ParamAxis {
    /// Swift's `precondition(samples >= 2)` is an assertion here, as
    /// `Grid2D::new`'s is: every caller computes its count with a floor of 2.
    pub fn new(range: Interval, periodic: bool, samples: usize) -> ParamAxis {
        assert!(samples >= 2, "an axis needs at least two samples");
        ParamAxis { range, periodic, samples }
    }

    /// A whole turn, `[0, 2π)`.
    pub fn angle(samples: usize) -> ParamAxis {
        ParamAxis::new(Interval::new(0.0, 2.0 * core::f64::consts::PI), true, samples)
    }

    pub fn cells(&self) -> usize {
        if self.periodic { self.samples } else { self.samples - 1 }
    }

    pub fn spacing(&self) -> f64 {
        self.range.length() / self.cells() as f64
    }

    pub fn period(&self) -> Option<f64> {
        self.periodic.then(|| self.range.length())
    }

    /// The coordinate of grid index `i`, *unwrapped*: on a periodic axis
    /// `i == samples` is one period past `i == 0`, which is what lets a cell
    /// that closes the seam interpolate across it rather than back through
    /// the whole range. On a bounded axis it is a `Grid2D`'s position, to the
    /// bit, so a field on a landscape's face is sampled where its grid is.
    pub fn coordinate(&self, i: i64) -> f64 {
        if self.periodic {
            self.range.lo + i as f64 * self.spacing()
        } else {
            self.range.lo + self.range.length() * i as f64 / self.cells() as f64
        }
    }

    /// The stored sample an unwrapped grid index refers to.
    pub fn sample(&self, i: i64) -> usize {
        if self.periodic {
            let n = self.samples as i64;
            (((i % n) + n) % n) as usize
        } else {
            i as usize
        }
    }

    /// The signed difference `a - b` taken the short way round a periodic
    /// axis; plain subtraction on a bounded one.
    pub fn difference(&self, a: f64, b: f64) -> f64 {
        match self.period() {
            None => a - b,
            Some(period) => short_way(a - b, period),
        }
    }

    /// Whether a coordinate lies on the axis: anywhere on a periodic one,
    /// within the range, ends included, on a bounded one. (`Face.swift`'s
    /// extension.)
    pub fn covers(&self, x: f64) -> bool {
        self.periodic || {
            let (lo, hi) = (swift::min(self.range.lo, self.range.hi), swift::max(self.range.lo, self.range.hi));
            lo <= x && x <= hi
        }
    }
}

/// A parameter rectangle: Swift's `(u: ParamAxis, v: ParamAxis)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamRectangle {
    pub u: ParamAxis,
    pub v: ParamAxis,
}

impl ParamRectangle {
    pub fn new(u: ParamAxis, v: ParamAxis) -> ParamRectangle {
        ParamRectangle { u, v }
    }
}

/// A point of a surface with its two partial derivatives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceJet {
    pub position: Vec3,
    pub du: Vec3,
    pub dv: Vec3,
}

impl SurfaceJet {
    pub fn new(position: Vec3, du: Vec3, dv: Vec3) -> SurfaceJet {
        SurfaceJet { position, du, dv }
    }

    /// `du × dv`: the normal of the parametrization's own orientation, not
    /// normalized.
    pub fn normal(&self) -> Vec3 {
        self.du.cross(self.dv)
    }
}

/// A surface's second partial derivatives at a point, x_uu, x_uv and x_vv:
/// with the normal, its second fundamental form (do Carmo §3-3), which its
/// principal curvatures and directions are read from
/// (`Face.principalCurvatures(at:on:)`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SecondDerivatives {
    pub uu: Vec3,
    pub uv: Vec3,
    pub vv: Vec3,
}

/// A point of a surface with its first and second partial derivatives, its
/// jet of order two: what the first and second fundamental forms are read
/// from, in one evaluation of the map where it costs, as a Fourier series
/// does.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SecondOrderJet {
    pub jet: SurfaceJet,
    pub second: SecondDerivatives,
}

/// The coordinate whose normal line passes through a point, and the point's
/// signed distance along it (`TubularNeighbourhood.foot`).
pub type NeighbourhoodFoot = Arc<dyn Fn(Vec3) -> (P2<ParamSpace>, f64) + Send + Sync>;
/// How fast the foot's coordinate moves per unit of travel, given how far
/// outside and inside the base (`TubularNeighbourhood.spread`).
pub type Spread = Arc<dyn Fn(f64, f64) -> Vec2 + Send + Sync>;

/// The solid a surface bounds, for sphere tracing along a line of sight.
///
/// A point of space is inside the solid exactly when its signed distance
/// from the base surface, along the base's normal line through its foot, is
/// under the surface's normal offset from the base at that foot. The round torus
/// has its foot in closed form, and a surface displaced along the torus's
/// normal keeps that foot and adds the displacement as its normal offset, so
/// the test is a point test with no raster in it. The gap, distance less
/// normal offset, is positive outside, negative inside and zero on the surface,
/// and over the points outside the solid and within the shell of the base
/// it changes by at most `lipschitz` per unit of travel: sphere tracing, which
/// steps by the gap over that constant cannot step through the solid, and
/// a line of sight is clear exactly when every step along it is.
#[derive(Clone)]
pub struct TubularNeighbourhood {
    /// The surface coordinate whose normal line passes through a point,
    /// and the point's signed distance along it, positive outward.
    pub foot: NeighbourhoodFoot,
    /// How far the surface stands off the base along the normal at a
    /// coordinate, positive outward.
    pub normal_offset: ParamScalar,
    /// The normal offset's bounds: it is never more than `above` nor less than
    /// `-below`. Beyond `above` a point is outside, under `-below` inside.
    pub above: f64,
    pub below: f64,
    /// The most the gap changes per unit of travel, over the points outside
    /// the solid and within the shell.
    pub lipschitz: f64,
    /// How fast the foot's coordinate moves per unit of travel, per axis,
    /// over the points between `inside` under the base and `outside` over
    /// it: what a normal offset's gradient is multiplied by.
    pub spread: Spread,
    /// A ball containing the solid.
    pub centre: Vec3,
    pub radius: f64,
}

impl fmt::Debug for TubularNeighbourhood {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TubularNeighbourhood")
            .field("above", &self.above)
            .field("below", &self.below)
            .field("lipschitz", &self.lipschitz)
            .field("centre", &self.centre)
            .field("radius", &self.radius)
            .finish_non_exhaustive()
    }
}

/// Why a parametric surface cannot be built: Swift's preconditions, which
/// trap there.
#[derive(Debug, Clone, PartialEq)]
pub enum ParametricError {
    /// A surface declared to enclose a solid has no volume.
    NoVolume,
    /// A grid handed in whose count is not the axes' product.
    PositionCount { positions: usize, u: usize, v: usize },
    /// Grid normals handed in whose count is not the grid's.
    NormalCount { normals: usize, points: usize },
    /// A torus asked for over axes that are not both periodic, or
    /// isothermal coordinates on a torus that reaches its axis.
    NotATorus(&'static str),
}

impl fmt::Display for ParametricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParametricError::NoVolume => write!(f, "a surface declared to enclose a solid has no volume"),
            ParametricError::PositionCount { positions, u, v } => {
                write!(f, "{positions} positions for a {u}x{v} grid")
            }
            ParametricError::NormalCount { normals, points } => write!(f, "{normals} normals for {points} grid points"),
            ParametricError::NotATorus(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for ParametricError {}

/// A surface given as a map from a parameter rectangle into space.
///
/// The map is exact and carries its derivatives, because the questions asked
/// of it -- which way does it face here, and exactly where along this segment
/// does it turn edge-on -- are asked at the curves' own coordinates, and a normal
/// read off the nearest grid point flips sign between neighbours precisely
/// where the answer matters, at a fold. The grid is what gets rasterized;
/// the map is what gets asked.
///
/// A heightfield is the case `(u, v) -> (u, v, h(u, v))` with neither axis
/// periodic; a torus has both periodic.
#[derive(Clone)]
pub struct ParametricSurface {
    pub u: ParamAxis,
    pub v: ParamAxis,
    pub map: JetMap,
    /// The grid, `u` fastest: `positions[j * u.samples + i]`.
    pub positions: Arc<Vec<P3<WorldSpace>>>,
    /// `+1` when `du × dv` points out of the solid the surface bounds, `-1`
    /// when it points in; `None` when the surface bounds no solid.
    ///
    /// Only a closed, embedded, orientable surface bounds a solid, and only
    /// for such a surface is "faces away from the viewer" the same as
    /// "hidden": a line of sight that reaches the back of it must first have
    /// entered the solid through the front. That is a fact the caller knows
    /// about its geometry -- it cannot be read off samples -- so it is
    /// declared, and the orientation is then computed rather than guessed.
    pub outward: Option<f64>,
    /// The solid the surface bounds as a point test, for surfaces that have
    /// one in closed form; `None` otherwise, and the curves are then tested by
    /// the coordinate image.
    pub neighbourhood: Option<TubularNeighbourhood>,
    /// The map's jet of order two, its second partial derivatives with its
    /// first, where the surface has them in closed form, as the round torus
    /// and a fitted Fourier series do: what its lines of curvature are read
    /// from. `None` otherwise, and it has none drawn: a displaced surface's
    /// derivatives are central differences already, and sn in relief is
    /// creased where it is truncated.
    pub second_order_jet: Option<SecondOrderMap>,
}

impl fmt::Debug for ParametricSurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParametricSurface")
            .field("u", &self.u)
            .field("v", &self.v)
            .field("outward", &self.outward)
            .field("neighbourhood", &self.neighbourhood)
            .finish_non_exhaustive()
    }
}

impl ParametricSurface {
    /// The surface over `u` and `v`, its grid evaluated from the map.
    ///
    /// `encloses`: the surface is closed, embedded and orientable. Declaring
    /// it for a surface that passes through itself, or has a boundary, culls
    /// curves that are in plain view. Swift's precondition that such a
    /// surface has a volume is [`ParametricError::NoVolume`] here.
    pub fn new(
        u: ParamAxis,
        v: ParamAxis,
        encloses: bool,
        neighbourhood: Option<TubularNeighbourhood>,
        second_order_jet: Option<SecondOrderMap>,
        map: JetMap,
    ) -> Result<ParametricSurface, ParametricError> {
        let mut positions = Vec::with_capacity(u.samples * v.samples);
        // Six times the enclosed volume, by the divergence theorem over the
        // grid: the sum of p · (du × dv) du dv. Only its sign is used, and
        // the sign is robust at any resolution that resolves the surface.
        let mut volume = 0.0;
        for j in 0..v.samples {
            for i in 0..u.samples {
                let jet = map(P2::new(u.coordinate(i as i64), v.coordinate(j as i64)));
                positions.push(P3::from_vec(jet.position));
                volume += jet.position.dot(jet.normal());
            }
        }
        let outward = if encloses {
            if !(volume.is_finite() && volume.abs() > 0.0) {
                return Err(ParametricError::NoVolume);
            }
            Some(if volume > 0.0 { 1.0 } else { -1.0 })
        } else {
            None
        };
        Ok(ParametricSurface {
            u,
            v,
            map,
            positions: Arc::new(positions),
            outward,
            neighbourhood: if encloses { neighbourhood } else { None },
            second_order_jet,
        })
    }

    /// A surface whose grid is already computed -- because evaluating the
    /// map at every grid point is the expensive part and the caller did it
    /// once already, in parallel or for its own checks. The map still answers
    /// every question asked at an arbitrary point.
    ///
    /// The orientation comes from the grid's own central-difference
    /// normals rather than the map's: only its sign is used, and a grid
    /// fine enough to draw is fine enough for the sign of its volume.
    pub fn with_positions(
        u: ParamAxis,
        v: ParamAxis,
        encloses: bool,
        neighbourhood: Option<TubularNeighbourhood>,
        positions: Vec<P3<WorldSpace>>,
        second_order_jet: Option<SecondOrderMap>,
        map: JetMap,
    ) -> Result<ParametricSurface, ParametricError> {
        if positions.len() != u.samples * v.samples {
            return Err(ParametricError::PositionCount { positions: positions.len(), u: u.samples, v: v.samples });
        }
        let surface = |outward| ParametricSurface {
            u,
            v,
            map: map.clone(),
            positions: Arc::new(positions.clone()),
            outward,
            neighbourhood: if encloses { neighbourhood.clone() } else { None },
            second_order_jet: second_order_jet.clone(),
        };
        if !encloses {
            return Ok(surface(None));
        }
        let at = |i: i64, j: i64| -> Vec3 {
            let ii = if u.periodic { u.sample(i) } else { i.clamp(0, u.samples as i64 - 1) as usize };
            let jj = if v.periodic { v.sample(j) } else { j.clamp(0, v.samples as i64 - 1) as usize };
            positions[jj * u.samples + ii].v
        };
        let mut volume = 0.0;
        for j in 0..v.samples as i64 {
            for i in 0..u.samples as i64 {
                let du = at(i + 1, j) - at(i - 1, j);
                let dv = at(i, j + 1) - at(i, j - 1);
                volume += at(i, j).dot(du.cross(dv));
            }
        }
        if !(volume.is_finite() && volume.abs() > 0.0) {
            return Err(ParametricError::NoVolume);
        }
        Ok(surface(Some(if volume > 0.0 { 1.0 } else { -1.0 })))
    }

    pub fn position(&self, i: i64, j: i64) -> P3<WorldSpace> {
        self.positions[self.v.sample(j) * self.u.samples + self.u.sample(i)]
    }

    /// How squarely the surface faces the viewer at `c`: the cosine between
    /// the outward normal and the direction back toward the eye. Positive
    /// facing, negative facing away, zero on a fold. `None` for a surface
    /// that has no outside, where the question has no answer.
    pub fn facing(&self, c: P2<ParamSpace>, sight: Vec3) -> Option<f64> {
        let outward = self.outward?;
        let n = (self.map)(c).normal();
        let len = n.length();
        if !(len > 0.0) {
            return Some(0.0);
        }
        Some(-outward * n.dot(sight) / len)
    }

    /// Lines of constant `u` -- `u` of them, evenly spaced -- and of constant
    /// `v`, each drawn through `resolution` steps of the other coordinate
    /// with its surface coordinate beside every vertex.
    ///
    /// On a periodic axis the lines sit at `lo + k * length / count` and a
    /// line running along it closes, its last vertex one period past its
    /// first so the coordinates stay continuous; on a bounded axis they
    /// include both ends. Swift evaluates a line per thread; here they are
    /// evaluated in turn, and the answer is the same.
    pub fn coordinate_curves(&self, lines: (usize, usize), resolution: usize) -> PolylineSet<WorldSpace> {
        let place = |axis: &ParamAxis, k: usize, count: usize| -> f64 {
            axis.range.lo
                + axis.range.length() * k as f64
                    / (if axis.periodic { count } else { count.saturating_sub(1).max(1) }) as f64
        };
        let run = |axis: &ParamAxis| -> Vec<f64> {
            (0..=resolution).map(|s| axis.range.lo + axis.range.length() * s as f64 / resolution as f64).collect()
        };
        let mut coords: Vec<Vec<P2<ParamSpace>>> = Vec::new();
        for k in 0..lines.0 {
            let fixed = place(&self.u, k, lines.0);
            coords.push(run(&self.v).into_iter().map(|y| P2::new(fixed, y)).collect());
        }
        for k in 0..lines.1 {
            let fixed = place(&self.v, k, lines.1);
            coords.push(run(&self.u).into_iter().map(|x| P2::new(x, fixed)).collect());
        }
        let paths: Vec<Vec<P3<WorldSpace>>> =
            coords.iter().map(|line| line.iter().map(|&c| P3::from_vec((self.map)(c).position)).collect()).collect();
        PolylineSet::from_paths_with_coords(&paths, &coords)
    }

    /// The fold curves for lines of sight along `sight`: where the surface turns
    /// edge-on, `n · sight = 0`, which is where its outline and every inner
    /// outline lie.
    ///
    /// Traced by marching squares on that product over the parameter
    /// rectangle -- sampled one step past a periodic axis's end, so a fold
    /// crosses the seam instead of stopping at it -- and every vertex then
    /// moved onto the exact fold by bisection along the product's gradient.
    /// The grid decides which folds exist; the map decides where they run.
    /// Where the sign changes across a crease rather than through a zero
    /// -- a truncation edge, which the truncation edge's own layer draws -- the vertex is
    /// dropped and the line cut there. From a cut the fold is traced on in
    /// steps a sixteenth of a cell, each set onto the fold by the map, to
    /// where it meets the crease or another cut: a fold down a steep wall
    /// can run its whole visible length inside one cell, where the grid
    /// has no vertex for it. Where the grid has none of a fold at all,
    /// `creases` -- the truncation edges, in surface coordinates -- seed it: a fold
    /// meets a crease where the facing a hair to one side of it changes
    /// sign along it, and is traced from there both ways.
    /// They depend on the camera, so they are derived for one, never stored.
    ///
    /// Swift spreads the grid's rows, the cuts and the seeds over threads;
    /// here each is a loop in order, every iteration writing its own slot,
    /// so the answer is the same.
    pub fn fold_curves(
        &self,
        sight: Vec3,
        grid: Option<(usize, usize)>,
        creases: &[Vec<P2<ParamSpace>>],
    ) -> PolylineSet<WorldSpace> {
        let (u, v) = (self.u, self.v);
        let (nu, nv) = grid.unwrap_or((u.cells(), v.cells()));
        let map = &self.map;
        let edge_on = |c: Vec2| -> f64 {
            let n = map(P2::from_vec(c)).normal();
            let len = n.length();
            if len > 0.0 { n.dot(sight) / len } else { 0.0 }
        };
        let du = u.range.length() / nu as f64;
        let dv = v.range.length() / nv as f64;
        let mut values = Vec::with_capacity((nu + 1) * (nv + 1));
        for j in 0..=nv {
            for i in 0..=nu {
                values.push(edge_on(Vec2::new(u.range.lo + i as f64 * du, v.range.lo + j as f64 * dv)) as f32);
            }
        }
        let field = Grid2D::new(nu + 1, nv + 1, Domain::new(u.range, v.range), values);

        // Onto the fold: bracket a sign change along the gradient within a
        // cell either way, then bisect it to the last bit. A fold is a
        // zero the product passes through, changing in proportion to the
        // span across it; at a crease it changes all at once, even where a
        // differenced normal smooths the jump over a hair, so a span a
        // hundredth of the size holds most of it. That is no fold: None.
        // Nor is a vertex the map finds no zero beside, within a cell
        // either way: the grid saw a sign change the map does not have,
        // as across a crease, where a grid cell holds the truncation and the
        // wall.
        let h = 0.25 * swift::min(du, dv);
        let gradient = |c: Vec2, h: f64| -> Vec2 {
            Vec2::new(
                edge_on(Vec2::new(c.x + h, c.y)) - edge_on(Vec2::new(c.x - h, c.y)),
                edge_on(Vec2::new(c.x, c.y + h)) - edge_on(Vec2::new(c.x, c.y - h)),
            )
        };
        // The zero within `r` either way along the unit `e`, or None.
        let onto = |c: Vec2, e: Vec2, r: f64| -> Option<Vec2> {
            let d = e * r;
            let (a, b) = (c - d, c + d);
            let fa = edge_on(a);
            if (fa > 0.0) == (edge_on(b) > 0.0) {
                return None;
            }
            let (holds, fails) = bisect_points(a, b, |p| (edge_on(p) > 0.0) == (fa > 0.0));
            let m = 0.5 * (holds + fails);
            let across = |t: f64| (edge_on(m + t * d) - edge_on(m - t * d)).abs();
            if across(1e-3) > 0.5 * across(1e-1) { None } else { Some(m) }
        };
        let refine = |c: Vec2| -> Option<Vec2> {
            let g = gradient(c, h);
            let len = g.length();
            if !(len > 0.0) {
                return Some(c);
            }
            onto(c, g / len, swift::max(du, dv))
        };

        // The runs, and which of their ends a dropped vertex cut.
        struct Run {
            c: Vec<Vec2>,
            cut_start: bool,
            cut_end: bool,
        }
        let mut runs: Vec<Run> = Vec::new();
        for line in contour::lines(&field, 0.0, None) {
            let mut run: Vec<Vec2> = Vec::new();
            let mut cut = false;
            for p in line {
                if let Some(c) = refine(p.v) {
                    run.push(c);
                    continue;
                }
                if run.len() >= 2 {
                    runs.push(Run { c: std::mem::take(&mut run), cut_start: cut, cut_end: true });
                }
                run.clear();
                cut = true;
            }
            if run.len() >= 2 {
                runs.push(Run { c: run, cut_start: cut, cut_end: false });
            }
        }

        // On from each cut, a step at a time: along the fold's tangent, then
        // onto the fold along the gradient within a step either way. Where
        // the step no longer finds the fold it has met a crease, and the
        // last step is cut back by bisection to just short of it. It ends
        // at another cut it comes within a step and a half of, and where it
        // comes back onto fold the grid already has.
        let step = swift::min(du, dv) / 16.0;
        let lo = Vec2::new(u.range.lo - du, v.range.lo - dv);
        let hi = Vec2::new(u.range.hi + du, v.range.hi + dv);
        let scale = Vec2::new(du, dv);
        let mut ends: Vec<(usize, bool)> = Vec::new();
        for (i, r) in runs.iter().enumerate() {
            if r.cut_start {
                ends.push((i, false));
            }
            if r.cut_end {
                ends.push((i, true));
            }
        }
        let lines: Vec<Vec<Vec2>> = runs.iter().map(|r| r.c.clone()).collect();
        let end_point: Vec<Vec2> =
            ends.iter().map(|&(run, at_end)| lines[run][if at_end { lines[run].len() - 1 } else { 0 }]).collect();
        // Swift's `Int((c / d).rounded(.down))` traps on a coordinate that is
        // not a number; such a point is filed in no cell, so nothing is near it.
        let cell = |c: Vec2| -> Option<(i64, i64)> {
            Some((swift::int((c.x / du).floor())?, swift::int((c.y / dv).floor())?))
        };
        let mut by_cell: HashMap<(i64, i64), Vec<(usize, Vec2)>> = HashMap::new();
        for (i, r) in lines.iter().enumerate() {
            for &c in r {
                if let Some(k) = cell(c) {
                    by_cell.entry(k).or_default().push((i, c));
                }
            }
        }
        // Fold the grid has, within a quarter cell: any run's but this
        // one's own about where the trace set out.
        let covered = |z: Vec2, run: i64, p0: Vec2| -> bool {
            let Some((ox, oy)) = cell(z) else { return false };
            for a in -1..=1 {
                for b in -1..=1 {
                    let Some(list) = by_cell.get(&(ox.wrapping_add(a), oy.wrapping_add(b))) else { continue };
                    for &(qr, at) in list {
                        if (at - z).div_elements(scale).length() < 0.25
                            && (qr as i64 != run || (at - p0).div_elements(scale).length() > 0.5)
                        {
                            return true;
                        }
                    }
                }
            }
            false
        };
        // From `p0` on the fold, setting out along `heading`; `run` is the
        // run it leaves, -1 for none, and `k` the cut it leaves, -1 for none.
        let trace = |p0: Vec2, h0: Vec2, run: i64, k: i64| -> (Vec<Vec2>, Option<usize>) {
            let mut heading = h0;
            let mut p = p0;
            let mut out: Vec<Vec2> = Vec::new();
            let next = |p: Vec2, t: Vec2, tau: f64| -> Option<Vec2> {
                let q = p + tau * step * t;
                if !(q.x > lo.x && q.y > lo.y && q.x < hi.x && q.y < hi.y) {
                    return None;
                }
                let g = gradient(q, 0.25 * step);
                let len = g.length();
                if len > 0.0 { onto(q, g / len, step) } else { None }
            };
            for _ in 0..128 {
                let g = gradient(p, 0.25 * step);
                let len = g.length();
                if !(len > 0.0) {
                    break;
                }
                let mut t = Vec2::new(-g.y, g.x) / len;
                if t.dot(heading) < 0.0 {
                    t = -t;
                }
                let Some(z) = next(p, t, 1.0) else {
                    let (mut a, mut b) = (0.0, 1.0);
                    let mut last: Option<Vec2> = None;
                    for _ in 0..30 {
                        let mid = 0.5 * (a + b);
                        if let Some(z) = next(p, t, mid) {
                            a = mid;
                            last = Some(z);
                        } else {
                            b = mid;
                        }
                    }
                    if let Some(last) = last {
                        out.push(last);
                    }
                    break;
                };
                if let Some(j) =
                    (0..end_point.len()).find(|&j| j as i64 != k && (end_point[j] - z).length() < 1.5 * step)
                {
                    out.push(z);
                    out.push(end_point[j]);
                    return (out, Some(j));
                }
                if covered(z, run, p0) {
                    break;
                }
                out.push(z);
                heading = z - p;
                p = z;
            }
            (out, None)
        };
        let traced: Vec<(Vec<Vec2>, Option<usize>)> = (0..ends.len())
            .map(|k| {
                // The heading from a vertex half a cell back: two grid
                // vertices in one cell can come out of refinement reversed.
                let (run, at_end) = ends[k];
                let r = &lines[run];
                let p0 = end_point[k];
                let back: Vec<Vec2> = if at_end {
                    r.iter().rev().skip(1).copied().collect()
                } else {
                    r.iter().skip(1).copied().collect()
                };
                let behind =
                    back.iter().copied().find(|&q| (p0 - q).div_elements(scale).length() > 0.5).unwrap_or(back[0]);
                trace(p0, p0 - behind, run as i64, k as i64)
            })
            .collect();
        // Two cuts traced to each other are one piece of fold: keep one.
        for k in 0..ends.len() {
            if traced[k].0.is_empty() {
                continue;
            }
            if let Some(j) = traced[k].1
                && traced[j].1 == Some(k)
                && j < k
            {
                continue;
            }
            let pts = &traced[k].0;
            let (i, at_end) = ends[k];
            if at_end {
                runs[i].c.extend_from_slice(pts);
            } else {
                let mut joined: Vec<Vec2> = pts.iter().rev().copied().collect();
                joined.extend_from_slice(&runs[i].c);
                runs[i].c = joined;
            }
        }
        let mut coords: Vec<Vec<Vec2>> = runs.into_iter().map(|r| r.c).collect();

        // From the creases: wherever the facing a fiftieth of a cell to one
        // side of a crease changes sign between two of its vertices, a fold
        // meets it there. One already ended within a tenth of a cell is
        // drawn; any other is found on that line beside the crease and
        // traced both ways, to the crease and away from it.
        let mut drawn: HashMap<(i64, i64), Vec<Vec2>> = HashMap::new();
        for r in &coords {
            for &c in r {
                if let Some(k) = cell(c) {
                    drawn.entry(k).or_default().push(c);
                }
            }
        }
        let reached = |j: Vec2| -> bool {
            let Some((ox, oy)) = cell(j) else { return false };
            for a in -1..=1 {
                for b in -1..=1 {
                    if let Some(list) = drawn.get(&(ox.wrapping_add(a), oy.wrapping_add(b)))
                        && list.iter().any(|&q| (q - j).div_elements(scale).length() < 0.1)
                    {
                        return true;
                    }
                }
            }
            false
        };
        let mut seeds: Vec<(Vec2, Vec2, Vec2)> = Vec::new();
        for line in creases.iter().filter(|l| l.len() >= 2) {
            for i in 0..line.len() - 1 {
                let (a, b) = (line[i].v, line[i + 1].v);
                let along = (b - a).div_elements(scale);
                if !(along.length() > 0.0) {
                    continue;
                }
                let off = (0.02 * Vec2::new(-along.y, along.x) / along.length()).mul_elements(scale);
                for side in [1.0, -1.0] {
                    let fa = edge_on(a + side * off);
                    let fb = edge_on(b + side * off);
                    if (fa > 0.0) == (fb > 0.0) {
                        continue;
                    }
                    if !reached(a + fa / (fa - fb) * (b - a)) {
                        seeds.push((a + side * off, b + side * off, side * off));
                    }
                }
            }
        }
        for &(a, b, off) in &seeds {
            // Onto the fold along the line beside the crease, which
            // brackets it and never crosses the crease.
            let Some(p0) = onto(0.5 * (a + b), (b - a).normalized(), 0.5 * (b - a).length()) else { continue };
            let toward = trace(p0, -off, -1, -1).0;
            let away = trace(p0, off, -1, -1).0;
            let mut c: Vec<Vec2> = toward.into_iter().rev().collect();
            c.push(p0);
            c.extend(away);
            if c.len() >= 2 {
                coords.push(c);
            }
        }

        let coords: Vec<Vec<P2<ParamSpace>>> =
            coords.into_iter().map(|r| r.into_iter().map(P2::from_vec).collect()).collect();
        let paths: Vec<Vec<P3<WorldSpace>>> =
            coords.iter().map(|r| r.iter().map(|&c| P3::from_vec(map(c).position)).collect()).collect();
        PolylineSet::from_paths_with_coords(&paths, &coords)
    }

    /// The grid's unit normals, by central differences on the grid --
    /// one-sided at a bounded axis's ends -- `u` fastest, as `positions`.
    ///
    /// They do not depend on the camera, so they are computed once per
    /// surface and handed to `grid_fold_curves` for every camera after.
    /// Where the grid is degenerate the normal is zero.
    pub fn grid_normals(&self) -> Vec<[f32; 3]> {
        let (nu, nv) = (self.u.samples, self.v.samples);
        let at = |i: i64, j: i64| -> Vec3 {
            let ii = if self.u.periodic { self.u.sample(i) } else { i.clamp(0, nu as i64 - 1) as usize };
            let jj = if self.v.periodic { self.v.sample(j) } else { j.clamp(0, nv as i64 - 1) as usize };
            self.positions[jj * nu + ii].v
        };
        let mut normals = vec![[0.0f32; 3]; nu * nv];
        for j in 0..nv as i64 {
            for i in 0..nu as i64 {
                let n = (at(i + 1, j) - at(i - 1, j)).cross(at(i, j + 1) - at(i, j - 1));
                let len = n.length();
                if len > 0.0 {
                    let unit = n / len;
                    normals[j as usize * nu + i as usize] = [unit.x as f32, unit.y as f32, unit.z as f32];
                }
            }
        }
        normals
    }

    /// The fold curves as the grid sees them: `fold_curves` without asking
    /// the map anything.
    ///
    /// Marching squares on `n · sight` at the grid points, from normals
    /// computed once (`grid_normals`), and each vertex placed by bilinear
    /// interpolation of the grid -- no refinement onto the exact fold. So
    /// a fold is as accurate as the grid is fine, which at a drawing
    /// grid is well under a pixel, and costs a dot product per grid
    /// point: fast enough to follow the camera. The preview draws these; the
    /// bake keeps `fold_curves`, whose vertices are exact. The dot product is
    /// single precision, as simd's `Float` one is, summed in component order.
    pub fn grid_fold_curves(
        &self,
        normals: &[[f32; 3]],
        sight: Vec3,
    ) -> Result<PolylineSet<WorldSpace>, ParametricError> {
        if normals.len() != self.positions.len() {
            return Err(ParametricError::NormalCount { normals: normals.len(), points: self.positions.len() });
        }
        // One column and row past a periodic axis's end, as `fold_curves`
        // samples: the seam's cells close on the first samples.
        let (nu, nv, stride) = (self.u.cells(), self.v.cells(), self.u.samples);
        let s = [sight.x as f32, sight.y as f32, sight.z as f32];
        let mut values = vec![0.0f32; (nu + 1) * (nv + 1)];
        for j in 0..=nv {
            let row = self.v.sample(j as i64) * stride;
            for i in 0..=nu {
                let n = normals[row + self.u.sample(i as i64)];
                values[j * (nu + 1) + i] = n[0] * s[0] + n[1] * s[1] + n[2] * s[2];
            }
        }
        let field = Grid2D::new(nu + 1, nv + 1, Domain::new(self.u.range, self.v.range), values);
        let mut paths = Vec::new();
        let mut coords = Vec::new();
        for line in contour::lines(&field, 0.0, None) {
            let c: Vec<P2<ParamSpace>> = line.iter().map(|p| P2::new(p.x(), p.y())).collect();
            paths.push(c.iter().map(|&c| self.interpolate(c)).collect());
            coords.push(c);
        }
        Ok(PolylineSet::from_paths_with_coords(&paths, &coords))
    }

    /// The grid's bilinear patch at a coordinate: the surface as it is
    /// drawn, rather than as the map has it.
    ///
    /// The grid is what gets rasterized, so curves placed this way lie on
    /// the drawn surface to the last bit, and it costs four grid reads
    /// where the map may cost a Fourier series. A coordinate any number of
    /// periods past a periodic axis's range wraps; on a bounded axis it is
    /// clamped to the end cell.
    pub fn interpolate(&self, c: P2<ParamSpace>) -> P3<WorldSpace> {
        let Some((i, j, a, b)) = self.cell(c) else {
            return P3::new(f64::NAN, f64::NAN, f64::NAN);
        };
        let p00 = self.position(i, j).v;
        let p10 = self.position(i + 1, j).v;
        let p01 = self.position(i, j + 1).v;
        let p11 = self.position(i + 1, j + 1).v;
        P3::from_vec((1.0 - b) * ((1.0 - a) * p00 + a * p10) + b * ((1.0 - a) * p01 + a * p11))
    }

    /// The grid's normals (`grid_normals`) at a coordinate, by the same
    /// bilinear patch as `interpolate`, unit length; zero where the patch's
    /// corners cancel.
    ///
    /// Within a grid cell of the exact normal, and so, like a grid
    /// fold, within a pixel of where the facing changes at any drawing
    /// grid: enough for the preview, which asks a normal of every curve
    /// vertex and asks it again whenever the curves move. The bake keeps the
    /// map's. Single precision throughout, as Swift's `SIMD3<Float>`; zero,
    /// too, at a coordinate that is not a number, where Swift's `Int` traps.
    pub fn interpolate_normal(&self, c: P2<ParamSpace>, normals: &[[f32; 3]]) -> Result<[f32; 3], ParametricError> {
        if normals.len() != self.positions.len() {
            return Err(ParametricError::NormalCount { normals: normals.len(), points: self.positions.len() });
        }
        let Some((i, j, a, b)) = self.cell(c) else { return Ok([0.0; 3]) };
        let at = |i: i64, j: i64| normals[self.v.sample(j) * self.u.samples + self.u.sample(i)];
        let (fa, fb) = (a as f32, b as f32);
        let (n00, n10, n01, n11) = (at(i, j), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1));
        let mut n = [0.0f32; 3];
        for k in 0..3 {
            n[k] = (1.0 - fb) * ((1.0 - fa) * n00[k] + fa * n10[k]) + fb * ((1.0 - fa) * n01[k] + fa * n11[k]);
        }
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        Ok(if len > 0.0 { [n[0] / len, n[1] / len, n[2] / len] } else { [0.0; 3] })
    }

    /// The grid cell a coordinate falls in, and where in it: the lower
    /// corner's unwrapped indices and the fractions along each axis. A
    /// periodic axis wraps any number of periods away; a bounded one clamps
    /// to its end cell. `None` where Swift's `Int(f.rounded(.down))` traps,
    /// on a coordinate that is not a number or past 2^63 cells.
    fn cell(&self, c: P2<ParamSpace>) -> Option<(i64, i64, f64, f64)> {
        let split = |x: f64, axis: &ParamAxis| -> Option<(i64, f64)> {
            let f = (x - axis.range.lo) / axis.spacing();
            let mut k = swift::int(f.floor())?;
            if !axis.periodic {
                k = k.clamp(0, axis.cells() as i64 - 1);
            }
            Some((k, f - k as f64))
        };
        let (i, a) = split(c.x(), &self.u)?;
        let (j, b) = split(c.y(), &self.v)?;
        Some((i, j, a, b))
    }

    /// The turns after which a linear curve of `slope` closes -- the
    /// denominator of a fraction, to within rounding -- when that is within
    /// `turns`; `None` for a linear curve that runs out its turns open.
    pub fn linear_curve_closes(slope: f64, turns: usize) -> Option<usize> {
        if turns < 1 {
            return None;
        }
        (1..=turns).find(|&k| {
            let w = slope * k as f64;
            (w - w.round()).abs() <= 1e-9 * swift::max(1.0, w.abs())
        })
    }

    /// Linear curves: straight lines through the parameter rectangle, `slope`
    /// turns of `v` per turn of `u`, glued into curves on the torus.
    ///
    /// A linear curve is what a torus's coordinates make of a straight line, and
    /// the same line on every torus with these coordinates: it lives on the
    /// flat torus, and the map -- any map -- is what puts it in space. The
    /// coordinate curves are its two ends, slope zero and slope infinite; the
    /// trajectory of a forced system, in the coordinates of its fitted torus,
    /// is the one with the system's own rotation number.
    ///
    /// It is an integral curve of a constant field (`VectorField::constant`),
    /// read off its closed form, not integrated (D9): the field is one step
    /// of the curve, so the vertex at whole time m is the strand's start
    /// moved by m steps, `flow(start, m)`.
    ///
    /// `count` strands start at `start` -- the rectangle's origin unless
    /// given -- and evenly spaced from it along `v`, and each runs `turns`
    /// turns of `u` -- or, when `slope` is a fraction `p/q` with `q ≤
    /// turns`, the `q` turns after which it closes, so a rational linear curve is
    /// one closed (p, q) curve and an irrational one is a stretch of a curve
    /// that never closes. A vertex per grid cell along the steeper axis,
    /// placed by `interpolate`, so the curves lie on the drawn surface; each
    /// vertex carries its coordinate, continuous along a path and never more
    /// than eight turns from zero, so the preview's float32 still resolves a
    /// pixel's worth of it: a path is ended and the next begun from the same
    /// vertex every eight turns.
    ///
    /// Swift's preconditions (a torus, a finite slope, a turn and a strand)
    /// are assertions here, as `ParamAxis::new`'s is; a step count past
    /// 2^63, where Swift's `Int` traps, is one too. `SurfaceRequest::check`
    /// bounds every request's vertices first.
    pub fn linear_curves(
        &self,
        slope: f64,
        turns: usize,
        count: usize,
        start: Option<P2<ParamSpace>>,
    ) -> PolylineSet<WorldSpace> {
        let (u, v) = (self.u, self.v);
        assert!(u.periodic && v.periodic, "a linear curve is a curve on a torus");
        assert!(
            turns >= 1 && count >= 1 && slope.is_finite(),
            "a linear curve wants a finite slope and at least one turn of one strand"
        );
        let drawn = ParametricSurface::linear_curve_closes(slope, turns).unwrap_or(turns);
        let steep = swift::int((slope.abs() * v.cells() as f64).ceil()).expect("a linear curve's steps fit an Int");
        let steps = (u.cells() as i64).max(steep) as usize;
        let du = u.range.length() / steps as f64;
        let dv = slope * v.range.length() / steps as f64;
        let line = VectorField::constant(Vec2::new(du, dv)).flow.expect("a constant field's flow is in closed form");
        let span = 8.0 * u.range.length();
        let origin = start.unwrap_or(P2::new(u.range.lo, v.range.lo));
        let mut paths: Vec<Vec<P3<WorldSpace>>> = Vec::new();
        let mut coords: Vec<Vec<P2<ParamSpace>>> = Vec::new();
        for strand in 0..count {
            let v0 = origin.y() + v.range.length() * strand as f64 / count as f64;
            let mut path: Vec<P3<WorldSpace>> = Vec::new();
            let mut coord: Vec<P2<ParamSpace>> = Vec::new();
            let mut offset = Vec2::new(0.0, 0.0);
            let start = P2::new(origin.x(), v0);
            for m in 0..=(drawn * steps) {
                let c = line(start, m as f64);
                if !path.is_empty() && c.x() - origin.x() - offset.x > span {
                    // The seam of a path: the same vertex ends one and
                    // begins the next, its coordinates brought back near
                    // the rectangle's origin by whole periods.
                    let last = coord[coord.len() - 1].v + offset;
                    let last_point = path[path.len() - 1];
                    paths.push(std::mem::take(&mut path));
                    coords.push(std::mem::take(&mut coord));
                    offset = Vec2::new(
                        u.range.length() * ((c.x() - u.range.lo) / u.range.length()).floor(),
                        v.range.length() * ((c.y() - v.range.lo) / v.range.length()).floor(),
                    );
                    path = vec![last_point];
                    coord = vec![P2::from_vec(last - offset)];
                }
                path.push(self.interpolate(c));
                coord.push(P2::from_vec(c.v - offset));
            }
            paths.push(path);
            coords.push(coord);
        }
        PolylineSet::from_paths_with_coords(&paths, &coords)
    }

    /// The torus of revolution about the z axis: `u` around the axis, `v`
    /// around the tube.
    ///
    /// It bounds a solid exactly when the tube does not reach the axis,
    /// `major > minor`; the horn and spindle tori pass through themselves.
    pub fn torus(major: f64, minor: f64, samples: (usize, usize)) -> Result<ParametricSurface, ParametricError> {
        ParametricSurface::torus_round(major, minor, ParamAxis::angle(samples.0), ParamAxis::angle(samples.1))
    }

    /// The same torus over any two periodic axes: each axis's range is one
    /// turn, so a rectangle of periods -- a doubly periodic function's
    /// fundamental domain -- glues into it edge to edge.
    ///
    /// `isothermal`: `v` runs evenly round the tube in isothermal coordinates
    /// rather than in its angle θ. The round torus's metric is
    /// (R + r cos θ)² dφ² + r² dθ², and with w(θ) = ∫₀^θ r dθ′ / (R + r cos θ′)
    /// it is (R + r cos θ)² (dφ² + dw²), a multiple of the flat one (do Carmo
    /// §4-2). A whole turn of the tube is 2πr / √(R² − r²) in w, so the map
    /// is conformal where `v`'s range is r / √(R² − r²) of `u`'s. A torus
    /// that reaches its axis has no such coordinates.
    pub fn torus_over(
        major: f64,
        minor: f64,
        u: ParamAxis,
        v: ParamAxis,
        isothermal: bool,
    ) -> Result<ParametricSurface, ParametricError> {
        if isothermal {
            ParametricSurface::isothermal_torus(major, minor, u, v)
        } else {
            ParametricSurface::torus_round(major, minor, u, v)
        }
    }

    /// Swift's `torus(major:minor:u:v:)`. Its precondition that both axes
    /// are periodic is [`ParametricError::NotATorus`] here.
    pub fn torus_round(r_major: f64, r: f64, u: ParamAxis, v: ParamAxis) -> Result<ParametricSurface, ParametricError> {
        if !(u.periodic && v.periodic) {
            return Err(ParametricError::NotATorus("a torus is periodic both ways"));
        }
        let big_r = r_major;
        let (su0, sv0) = (u.range.lo, v.range.lo);
        let two_pi = 2.0 * core::f64::consts::PI;
        let (ku, kv) = (two_pi / u.range.length(), two_pi / v.range.length());
        // The solid: a point's foot is the nearest point of the core circle
        // turned toward it, its distance how far it stands from that circle
        // less the tube's radius. The ring angle moves fastest where the
        // ring is narrowest, the tube angle where the tube is.
        let neighbourhood = TubularNeighbourhood {
            foot: Arc::new(move |q: Vec3| {
                let mut a = real::atan2(q.y, q.x);
                let rho = real::sqrt(q.x * q.x + q.y * q.y);
                let mut b = real::atan2(q.z, rho - big_r);
                if a < 0.0 {
                    a += two_pi;
                }
                if b < 0.0 {
                    b += two_pi;
                }
                let tube = real::sqrt((rho - big_r) * (rho - big_r) + q.z * q.z);
                (P2::new(su0 + a / ku, sv0 + b / kv), tube - r)
            }),
            normal_offset: Arc::new(|_| 0.0),
            above: 0.0,
            below: 0.0,
            lipschitz: 1.0,
            spread: Arc::new(move |outside: f64, inside: f64| {
                Vec2::new(
                    1.0 / (ku * swift::max(big_r - r - outside, 1e-12)),
                    1.0 / (kv * swift::max(r - inside, 1e-12)),
                )
            }),
            centre: Vec3::ZERO,
            radius: big_r + r,
        };
        // Its jet of order two in closed form, for its lines of curvature:
        // the parallels and the meridians, as on any surface of revolution.
        let second: SecondOrderMap = Arc::new(move |c: P2<ParamSpace>| {
            let a = ku * (c.x() - su0);
            let b = kv * (c.y() - sv0);
            let (ca, sa, cb, sb) = (real::cos(a), real::sin(a), real::cos(b), real::sin(b));
            let ring = big_r + r * cb;
            SecondOrderJet {
                jet: SurfaceJet::new(
                    Vec3::new(ring * ca, ring * sa, r * sb),
                    ku * Vec3::new(-ring * sa, ring * ca, 0.0),
                    kv * Vec3::new(-r * sb * ca, -r * sb * sa, r * cb),
                ),
                second: SecondDerivatives {
                    uu: ku * ku * Vec3::new(-ring * ca, -ring * sa, 0.0),
                    uv: ku * kv * Vec3::new(r * sb * sa, -r * sb * ca, 0.0),
                    vv: kv * kv * Vec3::new(-r * cb * ca, -r * cb * sa, -r * sb),
                },
            }
        });
        let map: JetMap = Arc::new(move |c: P2<ParamSpace>| {
            let a = ku * (c.x() - su0);
            let b = kv * (c.y() - sv0);
            let (ca, sa, cb, sb) = (real::cos(a), real::sin(a), real::cos(b), real::sin(b));
            let ring = big_r + r * cb;
            SurfaceJet::new(
                Vec3::new(ring * ca, ring * sa, r * sb),
                ku * Vec3::new(-ring * sa, ring * ca, 0.0),
                kv * Vec3::new(-r * sb * ca, -r * sb * sa, r * cb),
            )
        });
        ParametricSurface::new(u, v, big_r > r, Some(neighbourhood), Some(second), map)
    }

    /// The round torus with `v` isothermal (`torus_over(..., true)`).
    ///
    /// With t = w √(R² − r²) / r, w's turn scaled to a whole one, the tube
    /// angle is θ = 2 atan(k tan(t/2)), k = √((R + r) / (R − r)). Written so,
    /// it is cut at t = π; it is computed as
    /// θ = t + 2 atan((k − 1) sin t / ((k + 1) − (k − 1) cos t)), the same angle,
    /// whose denominator is at least 2: θ − t is smooth and 2π-periodic, so θ
    /// runs on across the seam, and on unwrapped coordinates, with no branch
    /// to choose. Its inverse is the same with 1/k for k, and
    /// dθ/dt = (R + r cos θ) / √(R² − r²).
    fn isothermal_torus(big_r: f64, r: f64, u: ParamAxis, v: ParamAxis) -> Result<ParametricSurface, ParametricError> {
        if !(u.periodic && v.periodic) {
            return Err(ParametricError::NotATorus("a torus is periodic both ways"));
        }
        if !(big_r > r && r > 0.0) {
            return Err(ParametricError::NotATorus(
                "only a torus that does not reach its axis has isothermal coordinates",
            ));
        }
        let (su0, sv0) = (u.range.lo, v.range.lo);
        let two_pi = 2.0 * core::f64::consts::PI;
        let (ku, kv) = (two_pi / u.range.length(), two_pi / v.range.length());
        let s = real::sqrt((big_r - r) * (big_r + r));
        let k = real::sqrt((big_r + r) / (big_r - r));
        fn angle(t: f64, k: f64) -> f64 {
            t + 2.0 * real::atan((k - 1.0) * real::sin(t) / ((k + 1.0) - (k - 1.0) * real::cos(t)))
        }
        // The round torus's solid, its tube angle read back to t; t moves at
        // most k times as fast as θ, on the inner equator.
        let neighbourhood = TubularNeighbourhood {
            foot: Arc::new(move |q: Vec3| {
                let mut a = real::atan2(q.y, q.x);
                let rho = real::sqrt(q.x * q.x + q.y * q.y);
                let mut b = real::atan2(q.z, rho - big_r);
                if a < 0.0 {
                    a += two_pi;
                }
                if b < 0.0 {
                    b += two_pi;
                }
                let tube = real::sqrt((rho - big_r) * (rho - big_r) + q.z * q.z);
                (P2::new(su0 + a / ku, sv0 + angle(b, 1.0 / k) / kv), tube - r)
            }),
            normal_offset: Arc::new(|_| 0.0),
            above: 0.0,
            below: 0.0,
            lipschitz: 1.0,
            spread: Arc::new(move |outside: f64, inside: f64| {
                Vec2::new(1.0 / (ku * swift::max(big_r - r - outside, 1e-12)), k / (kv * swift::max(r - inside, 1e-12)))
            }),
            centre: Vec3::ZERO,
            radius: big_r + r,
        };
        // The tube angle at a coordinate, and its first two derivatives along v.
        let tube = move |c: P2<ParamSpace>| -> (f64, f64, f64) {
            let theta = angle(kv * (c.y() - sv0), k);
            let d = kv * (big_r + r * real::cos(theta)) / s;
            (theta, d, -kv * r * real::sin(theta) * d / s)
        };
        // The chain rule through θ(v): x_v = θ′ x_θ, x_uv = θ′ x_uθ,
        // x_vv = θ′² x_θθ + θ″ x_θ.
        let second: SecondOrderMap = Arc::new(move |c: P2<ParamSpace>| {
            let a = ku * (c.x() - su0);
            let (theta, d, dd) = tube(c);
            let (ca, sa, cb, sb) = (real::cos(a), real::sin(a), real::cos(theta), real::sin(theta));
            let ring = big_r + r * cb;
            let along = Vec3::new(-r * sb * ca, -r * sb * sa, r * cb);
            SecondOrderJet {
                jet: SurfaceJet::new(
                    Vec3::new(ring * ca, ring * sa, r * sb),
                    ku * Vec3::new(-ring * sa, ring * ca, 0.0),
                    d * along,
                ),
                second: SecondDerivatives {
                    uu: ku * ku * Vec3::new(-ring * ca, -ring * sa, 0.0),
                    uv: ku * d * Vec3::new(r * sb * sa, -r * sb * ca, 0.0),
                    vv: d * d * Vec3::new(-r * cb * ca, -r * cb * sa, -r * sb) + dd * along,
                },
            }
        });
        let map: JetMap = Arc::new(move |c: P2<ParamSpace>| {
            let a = ku * (c.x() - su0);
            let (theta, d, _) = tube(c);
            let (ca, sa, cb, sb) = (real::cos(a), real::sin(a), real::cos(theta), real::sin(theta));
            let ring = big_r + r * cb;
            SurfaceJet::new(
                Vec3::new(ring * ca, ring * sa, r * sb),
                ku * Vec3::new(-ring * sa, ring * ca, 0.0),
                d * Vec3::new(-r * sb * ca, -r * sb * sa, r * cb),
            )
        });
        ParametricSurface::new(u, v, true, Some(neighbourhood), Some(second), map)
    }

    /// This surface displaced along its normal: p ↦ p + amplitude · height(p) · n(p).
    ///
    /// A landscape is the plane displaced along its normal by |f|; the same
    /// sentence over any base surface puts the landscape on it. The result
    /// is the image of a section of the base's normal bundle under its
    /// normal exponential map, which is an embedding as long as the
    /// displacement stays under the base's reach -- for a round torus,
    /// min(r, R − r) -- and `encloses` is the caller's statement that it
    /// does. The jet is the chain rule, d(p + h n) = dp + dh ⊗ n + h dn,
    /// with dh and dn, the shape operator of the base, by central
    /// differences on the base's map.
    /// `ceiling` bounds the height, for the displaced solid's shell: the
    /// displacement is never more than `amplitude` times it.
    pub fn displaced(
        &self,
        height: ParamScalar,
        amplitude: f64,
        ceiling: f64,
        encloses: bool,
    ) -> Result<ParametricSurface, ParametricError> {
        let (u, v) = (self.u, self.v);
        let base = self.map.clone();
        let sign = self.outward.unwrap_or(1.0);
        let eu = 1e-5 * u.spacing();
        let ev = 1e-5 * v.spacing();
        let unit_normal = {
            let base = base.clone();
            move |c: P2<ParamSpace>| -> Vec3 {
                let n = base(c).normal();
                sign * n / swift::max(n.length(), f64::MIN_POSITIVE)
            }
        };
        let h = {
            let height = height.clone();
            move |c: P2<ParamSpace>| amplitude * height(c)
        };
        let map: JetMap = {
            let (base, unit_normal, h) = (base.clone(), unit_normal.clone(), h.clone());
            Arc::new(move |c: P2<ParamSpace>| {
                let jet = base(c);
                let n = unit_normal(c);
                let hc = h(c);
                let cu1 = P2::new(c.x() + eu, c.y());
                let cu0 = P2::new(c.x() - eu, c.y());
                let cv1 = P2::new(c.x(), c.y() + ev);
                let cv0 = P2::new(c.x(), c.y() - ev);
                let dnu = (unit_normal(cu1) - unit_normal(cu0)) / (2.0 * eu);
                let dnv = (unit_normal(cv1) - unit_normal(cv0)) / (2.0 * ev);
                let dhu = (h(cu1) - h(cu0)) / (2.0 * eu);
                let dhv = (h(cv1) - h(cv0)) / (2.0 * ev);
                SurfaceJet::new(jet.position + hc * n, jet.du + dhu * n + hc * dnu, jet.dv + dhv * n + hc * dnv)
            })
        };
        let displaced = ParametricSurface::new(u, v, encloses, None, None, map)?;
        // The displaced solid, when the base has one: the same foot, the
        // displacement added to the normal offset. The gap's Lipschitz constant
        // needs the displacement's steepest slope, read off the grid
        // just built -- the displacement at a grid point is what was
        // added to the base there -- in parameter units, times how fast
        // the foot's coordinate can move; with half as much again, for the
        // slope between grid points.
        let Some(e) = self.neighbourhood.as_ref().filter(|_| encloses) else { return Ok(displaced) };
        let (nu, nv) = (u.samples, v.samples);
        let mut normal_offsets = vec![0.0; nu * nv];
        for j in 0..nv {
            for i in 0..nu {
                let c = P2::new(u.coordinate(i as i64), v.coordinate(j as i64));
                normal_offsets[j * nu + i] = (displaced.positions[j * nu + i].v - base(c).position).dot(unit_normal(c));
            }
        }
        let (mut su, mut sv) = (0.0f64, 0.0f64);
        for j in 0..nv {
            for i in 0..nu {
                let t = normal_offsets[j * nu + i];
                if i + 1 < nu || u.periodic {
                    su = swift::max(su, (normal_offsets[j * nu + u.sample(i as i64 + 1)] - t).abs() / u.spacing());
                }
                if j + 1 < nv || v.periodic {
                    sv = swift::max(sv, (normal_offsets[v.sample(j as i64 + 1) * nu + i] - t).abs() / v.spacing());
                }
            }
        }
        let above = e.above + swift::max(amplitude, 0.0) * ceiling;
        let below = e.below + swift::max(-amplitude, 0.0) * ceiling;
        let spread = (e.spread)(above, below);
        let normal_offset = e.normal_offset.clone();
        let lifted = TubularNeighbourhood {
            foot: e.foot.clone(),
            normal_offset: Arc::new(move |c| normal_offset(c) + h(c)),
            above,
            below,
            lipschitz: e.lipschitz + 1.5 * (su * spread.x + sv * spread.y),
            spread: e.spread.clone(),
            centre: e.centre,
            radius: e.radius + above,
        };
        ParametricSurface::with_positions(
            u,
            v,
            true,
            Some(lifted),
            (*displaced.positions).clone(),
            None,
            displaced.map.clone(),
        )
    }
}

impl Transform<WorldSpace, ViewSpace> {
    /// The direction a line of sight travels through the world, unit length.
    ///
    /// Under an affine camera the eye looks along the one direction that
    /// changes neither view x nor view y -- the kernel of the first two rows
    /// -- signed so that travelling it takes view z down, away from the eye.
    /// That is *not* the gradient of view z unless the camera is orthonormal,
    /// and the oblique camera are not: they shear. Asking the gradient instead
    /// tilts every fold by the shear.
    pub fn projection_direction(&self) -> Vec3 {
        let c = self.m.columns;
        let row0 = Vec3::new(c[0].x, c[1].x, c[2].x);
        let row1 = Vec3::new(c[0].y, c[1].y, c[2].y);
        let row2 = Vec3::new(c[0].z, c[1].z, c[2].z);
        let mut d = row0.cross(row1).normalized();
        if row2.dot(d) > 0.0 {
            d = -d;
        }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axes_sample_and_wrap() {
        let a = ParamAxis::angle(8);
        assert_eq!(a.cells(), 8);
        assert_eq!(a.sample(-1), 7);
        assert_eq!(a.sample(8), 0);
        assert_eq!(a.coordinate(8), 2.0 * core::f64::consts::PI);
        let b = ParamAxis::new(Interval::new(0.0, 1.0), false, 11);
        assert_eq!(b.cells(), 10);
        assert_eq!(b.coordinate(10), 1.0);
        assert!(b.covers(1.0) && !b.covers(1.1) && a.covers(100.0));
        assert_eq!(ParametricSurface::linear_curve_closes(0.4, 10), Some(5));
        assert_eq!(ParametricSurface::linear_curve_closes(core::f64::consts::SQRT_2, 24), None);
    }

    #[test]
    fn a_plane_has_no_outside_and_a_patch_reads_its_grid() {
        let u = ParamAxis::new(Interval::new(0.0, 2.0), false, 3);
        let v = ParamAxis::new(Interval::new(0.0, 1.0), false, 2);
        let map: JetMap = Arc::new(|c: P2<ParamSpace>| {
            SurfaceJet::new(
                Vec3::new(c.x(), c.y(), c.x() * c.y()),
                Vec3::new(1.0, 0.0, c.y()),
                Vec3::new(0.0, 1.0, c.x()),
            )
        });
        let s = ParametricSurface::new(u, v, false, None, None, map.clone()).unwrap();
        assert_eq!(s.outward, None);
        assert_eq!(s.facing(P2::new(0.5, 0.5), Vec3::new(0.0, 0.0, -1.0)), None);
        assert_eq!(s.interpolate(P2::new(1.5, 0.5)), P3::new(1.5, 0.5, 0.75));
        let flat: JetMap = Arc::new(|c: P2<ParamSpace>| {
            SurfaceJet::new(Vec3::new(c.x(), c.y(), 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0))
        });
        assert_eq!(
            ParametricSurface::new(u, v, true, None, None, flat).unwrap_err(),
            ParametricError::NoVolume,
            "a patch of the plane through the origin encloses nothing"
        );
        assert_eq!(ParametricSurface::new(u, v, true, None, None, map).unwrap().outward, Some(-1.0));
    }
}
