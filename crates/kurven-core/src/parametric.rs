//! Parametric surfaces: a map from a parameter rectangle into space, with its
//! derivatives. The port of `Sources/KurvenCore/Parametric.swift`.
//!
//! Phase 3 ports every type of the file whole, and the behaviour a face, a
//! scalar field or a landscape reaches: the axes, the jets, the surface's
//! constructors (its grid and its orientation), its facing, its coordinate
//! curves and its grid's bilinear patch. What only a parametric surface
//! needs -- the fold curves and their grid normals, the linear curves, the
//! torus, the displaced surface -- is Phase 4's, and each is a typed refusal
//! ([`PortRefusal`]) until then.

use std::fmt;
use std::sync::Arc;

use kurven_math::swift;

use crate::manifest::Interval;
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
}

impl fmt::Display for ParametricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParametricError::NoVolume => write!(f, "a surface declared to enclose a solid has no volume"),
            ParametricError::PositionCount { positions, u, v } => {
                write!(f, "{positions} positions for a {u}x{v} grid")
            }
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

    /// The fold curves for lines of sight along `sight`, where the surface
    /// turns edge-on (`foldCurves(sight:grid:creases:)`).
    pub fn fold_curves(
        &self,
        _sight: Vec3,
        _grid: Option<(usize, usize)>,
        _creases: &[Vec<P2<ParamSpace>>],
    ) -> Result<PolylineSet<WorldSpace>, PortRefusal> {
        // PORT: phase 4 -- the fold curves of a parametric surface: marching squares on n · sight, each vertex bisected onto the fold, traced on from cuts and seeded from creases.
        Err(PortRefusal { phase: 4, what: "ParametricSurface.foldCurves" })
    }

    /// The grid's unit normals by central differences (`gridNormals`).
    pub fn grid_normals(&self) -> Result<Vec<[f32; 3]>, PortRefusal> {
        // PORT: phase 4 -- the grid's unit normals by central differences, one-sided at a bounded axis's ends, for the preview's grid folds.
        Err(PortRefusal { phase: 4, what: "ParametricSurface.gridNormals" })
    }

    /// The fold curves as the grid sees them (`gridFoldCurves`).
    pub fn grid_fold_curves(
        &self,
        _normals: &[[f32; 3]],
        _sight: Vec3,
    ) -> Result<PolylineSet<WorldSpace>, PortRefusal> {
        // PORT: phase 4 -- marching squares on n · sight at the grid points from gridNormals, each vertex placed by interpolate.
        Err(PortRefusal { phase: 4, what: "ParametricSurface.gridFoldCurves" })
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

    /// The grid's normals at a coordinate (`interpolateNormal(_:normals:)`).
    pub fn interpolate_normal(&self, _c: P2<ParamSpace>, _normals: &[[f32; 3]]) -> Result<[f32; 3], PortRefusal> {
        // PORT: phase 4 -- the grid normals at a coordinate by the bilinear patch of interpolate, in single precision, unit length.
        Err(PortRefusal { phase: 4, what: "ParametricSurface.interpolateNormal" })
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

    /// Linear curves on a torus (`linearCurves(slope:turns:count:start:)`).
    pub fn linear_curves(
        &self,
        _slope: f64,
        _turns: usize,
        _count: usize,
        _start: Option<P2<ParamSpace>>,
    ) -> Result<PolylineSet<WorldSpace>, PortRefusal> {
        // PORT: phase 4 -- straight lines through the parameter rectangle glued into curves on the torus, read off VectorField.constant's flow (IntegralCurves.swift).
        Err(PortRefusal { phase: 4, what: "ParametricSurface.linearCurves" })
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

    /// The torus of revolution about the z axis (`torus(major:minor:samples:)`).
    pub fn torus(_major: f64, _minor: f64, _samples: (usize, usize)) -> Result<ParametricSurface, PortRefusal> {
        // PORT: phase 4 -- the round torus over two whole turns, with its tubular neighbourhood and its jet of order two in closed form.
        Err(PortRefusal { phase: 4, what: "ParametricSurface.torus(major:minor:samples:)" })
    }

    /// The torus over any two periodic axes, isothermal or not
    /// (`torus(major:minor:u:v:isothermal:)`).
    pub fn torus_over(
        _major: f64,
        _minor: f64,
        _u: ParamAxis,
        _v: ParamAxis,
        _isothermal: bool,
    ) -> Result<ParametricSurface, PortRefusal> {
        // PORT: phase 4 -- the round torus over a rectangle of periods, with v in the tube angle or isothermal (θ = t + 2 atan((k − 1) sin t / ((k + 1) − (k − 1) cos t))).
        Err(PortRefusal { phase: 4, what: "ParametricSurface.torus(major:minor:u:v:isothermal:)" })
    }

    /// This surface displaced along its normal (`displaced(by:amplitude:ceiling:encloses:)`).
    pub fn displaced(
        &self,
        _height: ParamScalar,
        _amplitude: f64,
        _ceiling: f64,
        _encloses: bool,
    ) -> Result<ParametricSurface, PortRefusal> {
        // PORT: phase 4 -- p ↦ p + amplitude · height(p) · n(p), its jet by the chain rule with central differences, and the lifted tubular neighbourhood.
        Err(PortRefusal { phase: 4, what: "ParametricSurface.displaced" })
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
