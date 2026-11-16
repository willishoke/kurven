//! Forced systems and their invariant tori: the port of
//! `Sources/KurvenDynamics/Forced.swift`.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, LazyLock};

use kurven_core::parametric::{
    JetMap, ParamAxis, ParametricError, ParametricSurface, SecondDerivatives, SecondOrderJet, SecondOrderMap,
    SurfaceJet,
};
use kurven_core::polyline::PolylineSet;
use kurven_core::simd::{Vec2, Vec3};
use kurven_core::spaces::{P2, P3, ParamSpace, WorldSpace};
use kurven_math::frequency::Line;
use kurven_math::ode::{Failure, Tolerance};
use kurven_math::{FourierTorus, Frequency, Ode, Simd, Simd3, real, swift};

/// A forced system's right-hand side, `f(t, x)`.
pub type Field = Arc<dyn Fn(f64, Simd3) -> Simd3 + Send + Sync>;

/// A periodically forced system in three dimensions: `dx/dt = f(t, x)` with
/// `f` periodic in `t` at angular frequency `forcing`.
///
/// The forcing phase `θ = ωt` is one angle of any invariant torus the system
/// has, known exactly; the other is the system's own. That is what makes the
/// torus recoverable from one trajectory without reconstructing a surface
/// from points: every sample already knows where on the torus it is.
#[derive(Clone)]
pub struct ForcedSystem {
    pub name: String,
    pub forcing: f64,
    pub start: Simd3,
    /// How long to run before the trajectory is on its attractor.
    pub transient: f64,
    pub tolerance: Tolerance,
    pub field: Field,
}

impl fmt::Debug for ForcedSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ForcedSystem").field("name", &self.name).field("forcing", &self.forcing).finish_non_exhaustive()
    }
}

impl ForcedSystem {
    /// `toroidal_manifolds.ipynb`, "Higher Dimensional Torus": the jerk
    /// oscillator `x''' = -α x' x'' - (ω₀² + 3βx²) x' + A sin ωt`, from
    /// *Mem-elements for Neuromorphic Circuits with Artificial Intelligence
    /// Applications*, with the parameters and starting point the notebook
    /// drew.
    pub fn jerk_torus() -> ForcedSystem {
        ForcedSystem {
            name: "jerk".to_string(),
            forcing: 0.94,
            start: Simd([-1.02909432, -1.70628519, -0.18950172]),
            transient: 400.0,
            tolerance: Tolerance { relative: 1e-11, absolute: 1e-12, max_step: None },
            field: Arc::new(|t, u| {
                let (alpha, beta, amplitude, w0squared) = (0.6, 0.5975, 0.6, 0.5896);
                let (x, y, z) = (u.0[0], u.0[1], u.0[2]);
                Simd([y, z, -alpha * y * z - (w0squared + 3.0 * beta * x * x) * y + amplitude * real::sin(0.94 * t)])
            }),
        }
    }

    /// The systems by name (Swift's `catalog`), sorted as Swift's
    /// `catalog.keys.sorted()` lists them.
    pub fn catalog() -> &'static BTreeMap<String, ForcedSystem> {
        static CATALOG: LazyLock<BTreeMap<String, ForcedSystem>> =
            LazyLock::new(|| BTreeMap::from([("jerk".to_string(), ForcedSystem::jerk_torus())]));
        &CATALOG
    }
}

/// Why a forced torus cannot be found or drawn.
#[derive(Debug, Clone, PartialEq)]
pub enum DynamicsError {
    NoGenerator(String),
    NotATorus {
        residual: f64,
        extent: f64,
    },
    UnknownSystem {
        name: String,
        known: Vec<String>,
    },
    /// The integrator gave up (Swift's `ODE` error, rethrown).
    Ode(Failure),
    /// The surface could not be built: Swift's precondition, which traps.
    Surface(ParametricError),
}

impl fmt::Display for DynamicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DynamicsError::NoGenerator(why) => write!(f, "dynamics: no second frequency in the spectrum ({why})"),
            DynamicsError::UnknownSystem { name, known } => {
                write!(f, "dynamics: unknown system '{name}'; the catalog has: {}", known.join(", "))
            }
            DynamicsError::NotATorus { residual, extent } => write!(
                f,
                "dynamics: the fitted torus misses the trajectory by {}, {} of its size; the orbit is not on a torus \
                 at this order -- chaotic, locked to a closed orbit, or wanting more harmonics",
                crate::text::g(*residual, 3),
                crate::text::g(residual / extent, 2)
            ),
            DynamicsError::Ode(e) => write!(f, "{e:?}"),
            DynamicsError::Surface(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for DynamicsError {}

impl From<Failure> for DynamicsError {
    fn from(e: Failure) -> DynamicsError {
        DynamicsError::Ode(e)
    }
}

impl From<ParametricError> for DynamicsError {
    fn from(e: ParametricError) -> DynamicsError {
        DynamicsError::Surface(e)
    }
}

/// An invariant torus of a forced system, found in its spectrum and fitted
/// as a Fourier series in the forcing phase and the system's own.
#[derive(Debug, Clone)]
pub struct InvariantTorus {
    pub system: ForcedSystem,
    pub fit: FourierTorus<3>,
    /// The worst distance from the trajectory to the fitted torus at its own
    /// phases, and the trajectory's size, for scale.
    pub residual: f64,
    pub extent: f64,
    pub lines: Vec<Line>,
    /// The fitted samples: the trajectory after its transient.
    pub t0: f64,
    pub dt: f64,
    pub samples: Vec<Simd3>,
}

/// A run of the system: its states at `begin + k dt`.
#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    pub begin: f64,
    pub dt: f64,
    pub states: Vec<Simd3>,
}

impl Run {
    /// The first `count` states: the same run, shorter. Exact but for the
    /// last few samples, which a run that ends there reads off a step cut
    /// short at its end -- close to the integrator's tolerance, and so
    /// good for a draft.
    pub fn prefix(&self, count: usize) -> Run {
        Run { begin: self.begin, dt: self.dt, states: self.states.iter().take(count.max(2)).copied().collect() }
    }
}

fn lane_min(a: Simd3, b: Simd3) -> Simd3 {
    Simd([swift::min(a.0[0], b.0[0]), swift::min(a.0[1], b.0[1]), swift::min(a.0[2], b.0[2])])
}

fn lane_max(a: Simd3, b: Simd3) -> Simd3 {
    Simd([swift::max(a.0[0], b.0[0]), swift::max(a.0[1], b.0[1]), swift::max(a.0[2], b.0[2])])
}

/// Swift's `Int((x).rounded())` of a duration over a step, which traps on
/// what does not fit; a count that is not one is refused here.
fn steps(x: f64) -> Result<usize, DynamicsError> {
    swift::int(x.round())
        .filter(|&n| n >= 0)
        .map(|n| n as usize)
        .ok_or_else(|| DynamicsError::NoGenerator(format!("{x} steps is no count")))
}

impl InvariantTorus {
    pub fn forcing(&self) -> f64 {
        self.fit.frequencies.0
    }

    pub fn internal_frequency(&self) -> f64 {
        self.fit.frequencies.1
    }

    /// `Ω/ω`: turns of the system's own phase per turn of the forcing. In
    /// the fitted torus's coordinates the flow is a rigid rotation, so the
    /// trajectory is the straight line of this slope -- the linear curve at the
    /// rotation number -- and the number's continued fraction says which
    /// closed linear curves it is nearly, and after how many turns it cuts the
    /// torus most evenly.
    pub fn rotation_number(&self) -> f64 {
        self.internal_frequency() / self.forcing()
    }

    /// Where on the torus the fitted samples begin: the forcing phase at
    /// `t0`, brought into its turn, and the system's phase on the line
    /// through it.
    ///
    /// In the fitted coordinates the flow is `θ = ωt, φ = Ωt`: the straight
    /// line `φ = ρθ` through the origin, unwrapped. Bringing `θ` into its
    /// turn slides the point along that line by whole turns of `θ`, and
    /// each slides `φ` by `2πρ` -- so the start is `(θ₀, ρθ₀)`, not the two
    /// phases wrapped on their own, which would be a point of another leaf.
    pub fn start_phase(&self) -> P2<ParamSpace> {
        let turn = 2.0 * core::f64::consts::PI;
        let wrap = |x: f64| x - turn * (x / turn).floor();
        let theta = wrap(self.forcing() * self.t0);
        P2::new(theta, wrap(self.rotation_number() * theta))
    }

    /// Integrate past the transient, find the second frequency among the
    /// spectral lines of the first component, and fit. Fails -- rather than
    /// fitting something -- when the spectrum has no second generator, or
    /// when the fit misses its own trajectory by more than `acceptance` of
    /// its size: the self-check that tells a torus from a chaotic or locked
    /// orbit. Swift's defaults: duration 8000, dt 0.05, 24 harmonics,
    /// acceptance 1e-3.
    pub fn find(
        system: &ForcedSystem,
        duration: f64,
        dt: f64,
        harmonics: usize,
        acceptance: f64,
    ) -> Result<InvariantTorus, DynamicsError> {
        let skip = steps(system.transient / dt)?;
        let count = skip + steps(duration / dt)?;
        let field = system.field.clone();
        let (states, _) = Ode::sample(|t, x| field(t, x), 0.0, system.start, dt, count, system.tolerance)?;
        let samples: Vec<Simd3> = states[skip..].to_vec();
        let t0 = skip as f64 * dt;
        let first: Vec<f64> = samples.iter().map(|s| s.0[0]).collect();
        let lines = Frequency::lines(&first, t0, dt, 12);
        let Some(g) = Frequency::generator(&lines, system.forcing, 4, 1e-7) else {
            return Err(DynamicsError::NoGenerator("every line is a harmonic of the forcing".to_string()));
        };
        if !(g.explained >= 2) {
            return Err(DynamicsError::NoGenerator("no candidate explains another line".to_string()));
        }
        let fit = FourierTorus::fit(&samples, t0, dt, (system.forcing, g.frequency), harmonics);
        let residual = fit.residual(&samples, t0, dt, 3);
        let mut lo = Simd([f64::INFINITY; 3]);
        let mut hi = -lo;
        for &s in &samples {
            lo = lane_min(lo, s);
            hi = lane_max(hi, s);
        }
        let extent = (hi - lo).max();
        if !(residual <= acceptance * extent) {
            return Err(DynamicsError::NotATorus { residual, extent });
        }
        Ok(InvariantTorus { system: system.clone(), fit, residual, extent, lines, t0, dt, samples })
    }

    /// `find` at Swift's defaults but for the duration and harmonics.
    pub fn find_default(
        system: &ForcedSystem,
        duration: f64,
        harmonics: usize,
    ) -> Result<InvariantTorus, DynamicsError> {
        InvariantTorus::find(system, duration, 0.05, harmonics, 1e-3)
    }

    /// The torus as a parametric surface: `u` the forcing phase, `v` the
    /// system's own, both whole turns. The grid is evaluated once, and
    /// checked slice by slice: the surface is declared to bound a solid --
    /// and so to hide whatever faces away -- only when it is embedded.
    /// Swift's default grid is (1024, 512).
    pub fn surface(&self, e: &Revolution, grid: (usize, usize)) -> Result<(ParametricSurface, bool), DynamicsError> {
        self.surface_from(e, grid, &self.values(grid))
    }

    /// The fitted torus at every grid point, in state space, before any
    /// embedding: the expensive half of `surface`, a whole Fourier series per
    /// point. Kept, it lets a new revolution be placed without evaluating the
    /// series again. Swift splits the rows over the cores; here they are a
    /// loop, each point its own.
    pub fn values(&self, grid: (usize, usize)) -> Vec<Simd3> {
        let (u, v) = (ParamAxis::angle(grid.0), ParamAxis::angle(grid.1));
        let mut values = Vec::with_capacity(grid.0 * grid.1);
        for j in 0..grid.1 {
            let phi = v.coordinate(j as i64);
            for i in 0..grid.0 {
                values.push(self.fit.value(u.coordinate(i as i64), phi));
            }
        }
        values
    }

    /// The torus placed by `e` from its grid `values` (`values`): the cheap
    /// half of `surface`. Swift's precondition that there is a value a grid
    /// point is an assertion here.
    pub fn surface_from(
        &self,
        e: &Revolution,
        grid: (usize, usize),
        values: &[Simd3],
    ) -> Result<(ParametricSurface, bool), DynamicsError> {
        assert!(values.len() == grid.0 * grid.1, "{} values for a {}x{} grid", values.len(), grid.0, grid.1);
        let (u, v) = (ParamAxis::angle(grid.0), ParamAxis::angle(grid.1));
        let mut positions = Vec::with_capacity(grid.0 * grid.1);
        for j in 0..grid.1 {
            for i in 0..grid.0 {
                positions.push(P3::from_vec(e.place(values[j * grid.0 + i], u.coordinate(i as i64))));
            }
        }
        let embedded = Revolution::slices_are_simple(&positions, grid);
        let (fit, e2, e3) = (Arc::new(self.fit.clone()), e.clone(), e.clone());
        let fit2 = fit.clone();
        let second: SecondOrderMap = Arc::new(move |c: P2<ParamSpace>| e2.second_order_jet(&fit2, c.x(), c.y()));
        let map: JetMap = Arc::new(move |c: P2<ParamSpace>| e3.jet(&fit, c.x(), c.y()));
        let surface = ParametricSurface::with_positions(u, v, embedded, None, positions, Some(second), map)?;
        Ok((surface, embedded))
    }

    /// The trajectory as curves: the linear curve of `surface` at the rotation
    /// number, `turns` turns from the fitted samples' start.
    ///
    /// The fitted torus holds the trajectory to within `residual`, a
    /// fraction of a pixel at any drawing resolution, so nothing is
    /// integrated: the trajectory is the straight line `φ = Ωt, θ = ωt` in
    /// the torus's own coordinates, the integral curve in closed form of the
    /// constant field at the rotation number's slope (`VectorField::constant`),
    /// placed from the grid as any linear curve is. The integrated one,
    /// `trajectory_run`, is the check that this is so.
    pub fn trajectory(&self, surface: &ParametricSurface, turns: usize) -> PolylineSet<WorldSpace> {
        surface.linear_curves(self.rotation_number(), turns, 1, Some(self.start_phase()))
    }

    /// The integrated trajectory as curves, from `start` for `duration`, every
    /// `dt`: each vertex the integrated state placed at its own forcing
    /// phase, and its surface coordinate `(ωt, Ωt)`. What the plates draw
    /// is `trajectory`; this is what it is held to.
    pub fn trajectory_integrated(
        &self,
        e: &Revolution,
        start: Option<f64>,
        duration: f64,
        dt: f64,
    ) -> Result<PolylineSet<WorldSpace>, DynamicsError> {
        Ok(self.trajectory_run(e, &self.run(start, duration, dt)?))
    }

    /// The integrated states a trajectory is drawn from, in state space: the
    /// expensive half of `trajectory`, and the half no embedding changes.
    /// Swift's default `dt` is 0.01.
    pub fn run(&self, start: Option<f64>, duration: f64, dt: f64) -> Result<Run, DynamicsError> {
        let begin = start.unwrap_or(self.t0);
        let mut lead = self.system.start;
        let field = self.system.field.clone();
        if begin > 0.0 {
            Ode::integrate(
                |t, x| field(t, x),
                0.0,
                self.system.start,
                begin,
                self.system.tolerance,
                |step| lead = step.state(step.t1()),
            )?;
        }
        let count = steps(duration / dt)? + 1;
        let (states, _) = Ode::sample(|t, x| field(t, x), begin, lead, dt, count, self.system.tolerance)?;
        Ok(Run { begin, dt, states })
    }

    /// A run placed by `e`: the cheap half of `trajectory`.
    ///
    /// Coordinates are kept continuous along a path, so a segment never jumps
    /// a period, and small, so the preview's float32 can still resolve a
    /// pixel's worth of them: a path is ended, and the next begun from the
    /// same vertex, every eight turns of the forcing.
    pub fn trajectory_run(&self, e: &Revolution, run: &Run) -> PolylineSet<WorldSpace> {
        let (begin, dt) = (run.begin, run.dt);
        let turn = 2.0 * core::f64::consts::PI;
        let span = 8.0 * turn;
        let (forcing, internal) = (self.forcing(), self.internal_frequency());
        let phases = |k: usize| {
            let t = begin + k as f64 * dt;
            (forcing * t, internal * t)
        };
        let mut paths: Vec<Vec<P3<WorldSpace>>> = Vec::new();
        let mut coords: Vec<Vec<P2<ParamSpace>>> = Vec::new();
        let mut path: Vec<P3<WorldSpace>> = Vec::new();
        let mut coord: Vec<P2<ParamSpace>> = Vec::new();
        let mut offset = (0.0, 0.0);
        for (k, &x) in run.states.iter().enumerate() {
            let (theta, phi) = phases(k);
            if path.is_empty() || theta - offset.0 > span {
                // A new path, its coordinates brought back near zero; after
                // the first it begins from the previous path's last vertex,
                // so the curves do not break.
                offset = (turn * (theta / turn).floor(), turn * (phi / turn).floor());
                if let Some(&last) = path.last() {
                    paths.push(std::mem::take(&mut path));
                    coords.push(std::mem::take(&mut coord));
                    let (t0, p0) = phases(k - 1);
                    path = vec![last];
                    coord = vec![P2::new(t0 - offset.0, p0 - offset.1)];
                }
            }
            path.push(P3::from_vec(e.place(x, theta)));
            coord.push(P2::new(theta - offset.0, phi - offset.1));
        }
        if path.len() >= 2 {
            paths.push(path);
            coords.push(coord);
        }
        PolylineSet::from_paths_with_coords(&paths, &coords)
    }
}

/// The torus of a forced system drawn by revolution: the forcing phase `θ`
/// is a real angle about the vertical, and each θ-slice -- the invariant
/// circle the stroboscopic map leaves at that phase -- is drawn in its own
/// half-plane, two state components as radius and height.
///
/// Projected straight to three state coordinates the torus can collapse into
/// a ribbon that passes through itself, and nothing drawn on that can be
/// told front from back. By revolution each slice has a half-plane to
/// itself, so the torus is embedded exactly when every slice is a simple
/// closed curve on the positive side of the axis -- which is checked, not
/// assumed.
#[derive(Debug, Clone, PartialEq)]
pub struct Revolution {
    /// The state components drawn as radius and as height.
    pub radial: usize,
    pub axial: usize,
    /// The state point drawn at the ring's centre line, and state units per
    /// world unit.
    pub centre: Vec2,
    pub scale: f64,
    /// The ring's radius, in world units.
    pub radius: f64,
}

impl Revolution {
    /// Centred on the trajectory's range in the two components, scaled so
    /// the larger spans two world units, on a ring of `radius`. Swift's
    /// defaults: radial 0, axial 1, radius 2.5.
    pub fn fitting(torus: &InvariantTorus, radial: usize, axial: usize, radius: f64) -> Revolution {
        let mut lo = Vec2::splat(f64::INFINITY);
        let mut hi = -lo;
        for s in &torus.samples {
            let p = Vec2::new(s.0[radial], s.0[axial]);
            lo = Vec2::new(swift::min(lo.x, p.x), swift::min(lo.y, p.y));
            hi = Vec2::new(swift::max(hi.x, p.x), swift::max(hi.y, p.y));
        }
        let span = hi - lo;
        // `(hi - lo).max()`, simd's reduction (maxNum).
        let widest = if span.x > span.y || span.y.is_nan() { span.x } else { span.y };
        Revolution {
            radial,
            axial,
            centre: 0.5 * (lo + hi),
            scale: 2.0 / swift::max(widest, f64::MIN_POSITIVE),
            radius,
        }
    }

    /// A state at forcing phase `θ`, placed.
    pub fn place(&self, x: Simd3, theta: f64) -> Vec3 {
        let r = self.radius + self.scale * (x.0[self.radial] - self.centre.x);
        Vec3::new(r * real::cos(theta), r * real::sin(theta), self.scale * (x.0[self.axial] - self.centre.y))
    }

    /// The fitted torus through this embedding, with both partials.
    pub fn jet(&self, torus: &FourierTorus<3>, theta: f64, phi: f64) -> SurfaceJet {
        let k = torus.jet(theta, phi);
        let (radial, axial, scale) = (self.radial, self.axial, self.scale);
        let r = self.radius + scale * (k.value.0[radial] - self.centre.x);
        let (c, s) = (real::cos(theta), real::sin(theta));
        let dr_theta = scale * k.d_theta.0[radial];
        let dr_phi = scale * k.d_phi.0[radial];
        SurfaceJet::new(
            Vec3::new(r * c, r * s, scale * (k.value.0[axial] - self.centre.y)),
            Vec3::new(dr_theta * c - r * s, dr_theta * s + r * c, scale * k.d_theta.0[axial]),
            Vec3::new(dr_phi * c, dr_phi * s, scale * k.d_phi.0[axial]),
        )
    }

    /// The fitted torus's jet of order two through this embedding, in closed
    /// form, the series' own and in one pass over it
    /// (`FourierTorus::second_order_jet`): the point is (ρ cos θ, ρ sin θ, z),
    /// its radius ρ and height z the series' components scaled, so
    /// x_θθ = (ρ_θθ − ρ) (cos θ, sin θ, 0) + 2ρ_θ (−sin θ, cos θ, 0) + z_θθ e_z,
    /// and the rest likewise.
    pub fn second_order_jet(&self, torus: &FourierTorus<3>, theta: f64, phi: f64) -> SecondOrderJet {
        let k = torus.second_order_jet(theta, phi);
        let (radial, axial, scale) = (self.radial, self.axial, self.scale);
        let r = self.radius + scale * (k.value.0[radial] - self.centre.x);
        let (c, s) = (real::cos(theta), real::sin(theta));
        let (r_t, r_p) = (scale * k.d_theta.0[radial], scale * k.d_phi.0[radial]);
        let (r_tt, r_tp, r_pp) =
            (scale * k.theta_theta.0[radial], scale * k.theta_phi.0[radial], scale * k.phi_phi.0[radial]);
        let out = Vec3::new(c, s, 0.0);
        let around = Vec3::new(-s, c, 0.0);
        SecondOrderJet {
            jet: SurfaceJet::new(
                Vec3::new(r * c, r * s, scale * (k.value.0[axial] - self.centre.y)),
                r_t * out + r * around + Vec3::new(0.0, 0.0, scale * k.d_theta.0[axial]),
                r_p * out + Vec3::new(0.0, 0.0, scale * k.d_phi.0[axial]),
            ),
            second: SecondDerivatives {
                uu: (r_tt - r) * out + 2.0 * r_t * around + Vec3::new(0.0, 0.0, scale * k.theta_theta.0[axial]),
                uv: r_tp * out + r_p * around + Vec3::new(0.0, 0.0, scale * k.theta_phi.0[axial]),
                vv: r_pp * out + Vec3::new(0.0, 0.0, scale * k.phi_phi.0[axial]),
            },
        }
    }

    /// Whether every θ-slice of a grid is a simple closed curve on the
    /// positive side of the axis: the condition for the revolved torus to be
    /// embedded. Each slice is a closed polygon in its own (radius, height)
    /// half-plane; two of its edges that are not neighbours must not meet.
    /// Swift tests the slices in parallel; here in turn, each its own.
    pub fn slices_are_simple(positions: &[P3<WorldSpace>], grid: (usize, usize)) -> bool {
        (0..grid.0).all(|i| {
            // The radius along the slice's own half-plane, signed: a
            // point past the axis is at a negative radius, not at the
            // same distance on the far side, which `hypot` would say and
            // which would never fail the test below.
            let theta = ParamAxis::angle(grid.0).coordinate(i as i64);
            let (c, s) = (real::cos(theta), real::sin(theta));
            let slice: Vec<Vec2> = (0..grid.1)
                .map(|j| {
                    let p = positions[j * grid.0 + i];
                    Vec2::new(p.x() * c + p.y() * s, p.z())
                })
                .collect();
            slice.iter().all(|p| p.x > 0.0) && Revolution::is_simple_closed(&slice)
        })
    }

    /// A closed polygon with no two non-adjacent edges touching.
    pub fn is_simple_closed(p: &[Vec2]) -> bool {
        let n = p.len();
        if n < 3 {
            return false;
        }
        let cross = |o: Vec2, a: Vec2, b: Vec2| (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
        let within = |a: Vec2, b: Vec2, q: Vec2| {
            q.x >= swift::min(a.x, b.x)
                && q.x <= swift::max(a.x, b.x)
                && q.y >= swift::min(a.y, b.y)
                && q.y <= swift::max(a.y, b.y)
        };
        // Segments ab and cd meet: they straddle each other, or an end of
        // one lies exactly on the other.
        let meet = |a: Vec2, b: Vec2, c: Vec2, d: Vec2| -> bool {
            if !(swift::max(a.x, b.x) >= swift::min(c.x, d.x)
                && swift::max(c.x, d.x) >= swift::min(a.x, b.x)
                && swift::max(a.y, b.y) >= swift::min(c.y, d.y)
                && swift::max(c.y, d.y) >= swift::min(a.y, b.y))
            {
                return false;
            }
            let (d1, d2, d3, d4) = (cross(c, d, a), cross(c, d, b), cross(a, b, c), cross(a, b, d));
            if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
            {
                return true;
            }
            (d1 == 0.0 && within(c, d, a))
                || (d2 == 0.0 && within(c, d, b))
                || (d3 == 0.0 && within(a, b, c))
                || (d4 == 0.0 && within(a, b, d))
        };
        for i in 0..n - 2 {
            let (a, b) = (p[i], p[(i + 1) % n]);
            for j in (i + 2)..n {
                if i == 0 && j == n - 1 {
                    continue;
                }
                if meet(a, b, p[j], p[(j + 1) % n]) {
                    return false;
                }
            }
        }
        true
    }
}
