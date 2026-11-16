//! Ordinary differential equations: Dormand–Prince 5(4) with dense output.
//! The port of `ODE.swift`.
//!
//! The pair every general-purpose integrator is built on (MATLAB's `ode45`,
//! scipy's `RK45`, Hairer's `DOPRI5`): seven stages, the seventh reused as the
//! next step's first, a fifth-order solution propagated and a fourth-order
//! one for the error estimate, and Shampine's fourth-order interpolant
//! between steps. The interpolant is what makes sampling at fixed times free
//! and exact to the method's order, so the integrator chooses its own steps
//! and the caller still gets a uniform time series.
//!
//! Generic over the width of a [`Simd`] vector, so a three-dimensional system
//! is a `Simd3` and nothing is allocated per step.

use core::fmt;

use crate::format::description;
use crate::jacobi::ulp;
use crate::real;
use crate::simd::Simd;
use crate::swift;

pub enum Ode {}

/// Step-size control.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Tolerance {
    pub relative: f64,
    pub absolute: f64,
    /// A truncation on the step, or None for none. A system with a
    /// discontinuous right-hand side wants one, so a step cannot stride over
    /// the discontinuity and accept an error estimate that never saw it.
    pub max_step: Option<f64>,
}

impl Default for Tolerance {
    fn default() -> Self {
        Tolerance { relative: 1e-10, absolute: 1e-12, max_step: None }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Statistics {
    pub accepted: usize,
    pub rejected: usize,
    pub evaluations: usize,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Failure {
    /// The step fell below what the time's own precision can resolve.
    StepUnderflow {
        at: f64,
    },
    NonFinite {
        at: f64,
    },
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Failure::StepUnderflow { at } => write!(f, "ode: the step size underflowed at t = {}", description(*at)),
            Failure::NonFinite { at } => {
                write!(f, "ode: the solution stopped being finite at t = {}", description(*at))
            }
        }
    }
}

impl std::error::Error for Failure {}

// The Dormand–Prince tableau, exactly as published (J. R. Dormand and
// P. J. Prince, J. Comp. Appl. Math. 6, 1980), and Hairer's coefficients
// for the dense output (`contd5` in DOPRI5).
const C2: f64 = 1.0 / 5.0;
const C3: f64 = 3.0 / 10.0;
const C4: f64 = 4.0 / 5.0;
const C5: f64 = 8.0 / 9.0;
const A21: f64 = 1.0 / 5.0;
const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;
const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;
const A51: f64 = 19372.0 / 6561.0;
const A52: f64 = -25360.0 / 2187.0;
const A53: f64 = 64448.0 / 6561.0;
const A54: f64 = -212.0 / 729.0;
const A61: f64 = 9017.0 / 3168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46732.0 / 5247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5103.0 / 18656.0;
const A71: f64 = 35.0 / 384.0;
const A73: f64 = 500.0 / 1113.0;
const A74: f64 = 125.0 / 192.0;
const A75: f64 = -2187.0 / 6784.0;
const A76: f64 = 11.0 / 84.0;
// Fifth order minus fourth: the error estimate's weights.
const E1: f64 = 71.0 / 57600.0;
const E3: f64 = -71.0 / 16695.0;
const E4: f64 = 71.0 / 1920.0;
const E5: f64 = -17253.0 / 339200.0;
const E6: f64 = 22.0 / 525.0;
const E7: f64 = -1.0 / 40.0;
const D1: f64 = -12715105075.0 / 11282082432.0;
const D3: f64 = 87487479700.0 / 32700410799.0;
const D4: f64 = -10690763975.0 / 1880347072.0;
const D5: f64 = 701980252875.0 / 199316789632.0;
const D6: f64 = -1453857185.0 / 822651844.0;
const D7: f64 = 69997945.0 / 29380423.0;

/// One accepted step, as the interpolant needs it: the state at `t` is
/// `r1 + θ(r2 + (1-θ)(r3 + θ(r4 + (1-θ) r5)))` with `θ = (t - t0) / h`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Step<const N: usize> {
    pub t0: f64,
    pub h: f64,
    r1: Simd<N>,
    r2: Simd<N>,
    r3: Simd<N>,
    r4: Simd<N>,
    r5: Simd<N>,
}

impl<const N: usize> Step<N> {
    #[inline]
    pub fn state(&self, t: f64) -> Simd<N> {
        let s = (t - self.t0) / self.h;
        let s1 = 1.0 - s;
        self.r1 + s * (self.r2 + s1 * (self.r3 + s * (self.r4 + s1 * self.r5)))
    }

    #[inline]
    pub fn t1(&self) -> f64 {
        self.t0 + self.h
    }
}

/// One attempted step: the step with its interpolant, the last stage (the
/// next step's first), and the local error estimate.
struct Attempt<const N: usize> {
    step: Step<N>,
    k7: Simd<N>,
    error: Simd<N>,
}

impl Ode {
    /// One Dormand–Prince step of length `h` from `(t, y)`, given `k1 =
    /// f(t, y)`: the step with its interpolant, the last stage (the next
    /// step's first), and the local error estimate, fifth order minus fourth.
    #[inline]
    fn attempt<const N: usize>(
        f: &mut impl FnMut(f64, Simd<N>) -> Simd<N>,
        t: f64,
        y: Simd<N>,
        k1: Simd<N>,
        h: f64,
    ) -> Attempt<N> {
        let k2 = f(t + C2 * h, y + h * (A21 * k1));
        let k3 = f(t + C3 * h, y + h * (A31 * k1 + A32 * k2));
        let k4 = f(t + C4 * h, y + h * (A41 * k1 + A42 * k2 + A43 * k3));
        let k5 = f(t + C5 * h, y + h * (A51 * k1 + A52 * k2 + A53 * k3 + A54 * k4));
        let k6 = f(t + h, y + h * (A61 * k1 + A62 * k2 + A63 * k3 + A64 * k4 + A65 * k5));
        let y_new = y + h * (A71 * k1 + A73 * k3 + A74 * k4 + A75 * k5 + A76 * k6);
        let k7 = f(t + h, y_new);
        let dy = y_new - y;
        let r3 = h * k1 - dy;
        let step = Step {
            t0: t,
            h,
            r1: y,
            r2: dy,
            r3,
            r4: dy - h * k7 - r3,
            r5: h * (D1 * k1 + D3 * k3 + D4 * k4 + D5 * k5 + D6 * k6 + D7 * k7),
        };
        Attempt { step, k7, error: h * (E1 * k1 + E3 * k3 + E4 * k4 + E5 * k5 + E6 * k6 + E7 * k7) }
    }

    /// `steps` equal steps with no error control: the method's own order,
    /// laid bare, which is what a test of the tableau measures. Returns the
    /// state and the last step's error estimate.
    pub fn fixed<const N: usize>(
        mut f: impl FnMut(f64, Simd<N>) -> Simd<N>,
        t0: f64,
        y0: Simd<N>,
        t1: f64,
        steps: usize,
    ) -> (Simd<N>, Simd<N>) {
        let h = (t1 - t0) / steps as f64;
        let mut y = y0;
        let mut k1 = f(t0, y0);
        let mut error = Simd::ZERO;
        for i in 0..steps {
            let trial = Ode::attempt(&mut f, t0 + i as f64 * h, y, k1, h);
            y = trial.step.r1 + trial.step.r2;
            k1 = trial.k7;
            error = trial.error;
        }
        (y, error)
    }

    /// Integrate `dy/dt = f(t, y)` from `(t0, y0)` to `t1`, handing every
    /// accepted step to `body` in order.
    ///
    /// Step control is Hairer's: the error is the largest over components of
    /// `|err| / (atol + rtol * max(|y0|, |y1|))`, a step is accepted when that
    /// is at most one, and the next step is scaled by `0.9 err^(-1/5)`, held
    /// between a fifth and ten times, and not allowed to grow straight after
    /// a rejection. The first step is Hairer's `hinit` estimate.
    ///
    /// Swift's `body` may throw; no caller's does, and here it cannot.
    // `!(a > b)`, as Swift's `guard a > b`, so a NaN takes the refusal.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn integrate<const N: usize>(
        mut f: impl FnMut(f64, Simd<N>) -> Simd<N>,
        t0: f64,
        y0: Simd<N>,
        t1: f64,
        tol: Tolerance,
        mut body: impl FnMut(&Step<N>),
    ) -> Result<Statistics, Failure> {
        let mut stats = Statistics::default();
        if !(t1 > t0) {
            return Ok(stats);
        }
        let norm = |v: Simd<N>, scale: Simd<N>| (v.absolute() / scale).max();
        let scale =
            |a: Simd<N>, b: Simd<N>| tol.absolute + tol.relative * Simd::pointwise_max(a.absolute(), b.absolute());
        let hmax = swift::min(tol.max_step.unwrap_or(f64::INFINITY), t1 - t0);

        let mut t = t0;
        let mut y = y0;
        let mut k1 = f(t, y);
        stats.evaluations += 1;

        // hinit: a step whose first-order Euler move is about 1% of the
        // state, then revised from the second derivative it implies.
        let mut h = {
            let sc = scale(y, y);
            let d0 = norm(y, sc);
            let d1 = norm(k1, sc);
            let mut h0 = if d0 < 1e-5 || d1 < 1e-5 { 1e-6 } else { 0.01 * d0 / d1 };
            h0 = swift::min(h0, hmax);
            let k = f(t + h0, y + h0 * k1);
            stats.evaluations += 1;
            let d2 = norm(k - k1, sc) / h0;
            let h1 = if swift::max(d1, d2) <= 1e-15 {
                swift::max(1e-6, h0 * 1e-3)
            } else {
                real::pow(0.01 / swift::max(d1, d2), 1.0 / 5.0)
            };
            swift::min(swift::min(100.0 * h0, h1), hmax)
        };

        let mut rejected_last = false;
        while t < t1 {
            if t + h >= t1 {
                h = t1 - t;
            }
            if !(h > 16.0 * ulp(t).abs()) {
                return Err(Failure::StepUnderflow { at: t });
            }

            let trial = Ode::attempt(&mut f, t, y, k1, h);
            let y_new = trial.step.r1 + trial.step.r2;
            let k7 = trial.k7;
            stats.evaluations += 6;

            let err = norm(trial.error, scale(y, y_new));
            if !err.is_finite() {
                // An overflow in a trial stage is a step too long, not a
                // failure; a non-finite state after a tiny step is.
                if h < 1e-12 * swift::max(1.0, t.abs()) {
                    return Err(Failure::NonFinite { at: t });
                }
                h *= 0.1;
                rejected_last = true;
                stats.rejected += 1;
                continue;
            }
            let mut factor = 0.9 * real::pow(swift::max(err, 1e-10), -1.0 / 5.0);
            factor = swift::min(10.0, swift::max(0.2, factor));
            if err <= 1.0 {
                body(&trial.step);
                stats.accepted += 1;
                t += h;
                y = y_new;
                k1 = k7;
                if rejected_last {
                    factor = swift::min(factor, 1.0);
                }
                rejected_last = false;
            } else {
                stats.rejected += 1;
                factor = swift::min(factor, 1.0);
                rejected_last = true;
            }
            h = swift::min(h * factor, hmax);
        }
        Ok(stats)
    }

    /// The solution at `t0 + k dt` for `k = 0..<count`, read off the dense
    /// output: the steps are the integrator's, the samples are the caller's.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn sample<const N: usize>(
        f: impl FnMut(f64, Simd<N>) -> Simd<N>,
        t0: f64,
        y0: Simd<N>,
        dt: f64,
        count: usize,
        tolerance: Tolerance,
    ) -> Result<(Vec<Simd<N>>, Statistics), Failure> {
        assert!(dt > 0.0 && count >= 1);
        let mut out = Vec::with_capacity(count);
        out.push(y0);
        let end = t0 + dt * (count - 1) as f64;
        let mut last: Option<Step<N>> = None;
        let stats = Ode::integrate(f, t0, y0, end, tolerance, |step| {
            // Every sample time this step covers.
            while out.len() < count {
                let t = t0 + dt * out.len() as f64;
                if !(t <= step.t1()) {
                    break;
                }
                out.push(step.state(t));
            }
            last = Some(*step);
        })?;
        // The final step ends on `end` to rounding; a last sample that lands a
        // hair past it is read from that step all the same.
        while out.len() < count
            && let Some(step) = last
        {
            out.push(step.state(t0 + dt * out.len() as f64));
        }
        Ok((out, stats))
    }
}
