//! ODE, frequency analysis and the Fourier torus: the port of `odeTests` and
//! `frequencyTests` in `Sources/kurven-test/Dynamics.swift`, and the quarter
//! periods of `periodicTorusTests` in `Sources/kurven-test/Surfaces.swift`.
//!
//! Every check here is against something exact: a closed-form solution, a
//! conserved quantity, or the order the tableau is supposed to have. The
//! order test is the one a mistyped coefficient cannot pass -- the method
//! still converges with a wrong digit in it, just more slowly. The bounds are
//! Swift's.

use core::f64::consts::PI;

use kurven_math::fourier_torus::FourierTorus;
use kurven_math::frequency::Frequency;
use kurven_math::ode::{Ode, Tolerance};
use kurven_math::real::{cos, hypot, log2, pow, sin};
use kurven_math::{Complex, Jacobi, Simd, Simd2, Simd3, Simd4};

/// The suite's deterministic generator (`SplitMix` in
/// `Sources/kurven-test/main.swift`): the same cases every run.
struct SplitMix(u64);

impl SplitMix {
    fn next_bits(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * (self.next_bits() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
}

#[test]
fn dormand_prince_is_fifth_order_and_its_error_estimate_too() {
    // x'' = -36x, forced at 3t, over four time units: fast enough that the
    // error stays far above rounding while the step shrinks into the
    // asymptotic range, and time-dependent so no stage is right by symmetry.
    let (w, p) = (6.0, 3.0);
    let f = |t: f64, y: Simd2| Simd([y[1], -w * w * y[0] + sin(p * t)]);
    // From rest: x = (sin pt − (p/w) sin wt) / (w² − p²).
    let exact = |t: f64| (sin(p * t) - p / w * sin(w * t)) / (w * w - p * p);
    let mut errors = Vec::new();
    let mut estimates = Vec::new();
    for n in [100, 200, 400, 800, 1600] {
        let (state, last_error) = Ode::fixed(f, 0.0, Simd([0.0, 0.0]), 4.0, n);
        errors.push((state[0] - exact(4.0)).abs());
        // The last step's estimate is a local error, O(h⁵) for a 5(4) pair.
        estimates.push(last_error.length());
    }
    // Dormand and Prince made the fifth-order error constant small on
    // purpose, so at coarse steps the sixth-order term shows through and the
    // observed order comes down to five from above. A mistyped coefficient
    // pulls it below five and keeps it there.
    let orders: Vec<f64> = errors.windows(2).map(|e| log2(e[0] / e[1])).collect();
    let local: Vec<f64> = estimates.windows(2).map(|e| log2(e[0] / e[1])).collect();
    let falling = orders.windows(2).all(|o| o[0] >= o[1] - 0.05);
    let last = *orders.last().unwrap();
    assert!(
        *errors.last().unwrap() > 1e-14 && orders.iter().all(|&o| o > 4.9) && falling && (last - 5.0).abs() < 0.2,
        "halving the step divides the global error by 2⁵, in the limit: {orders:?}; finest error {:e}",
        errors.last().unwrap()
    );
    assert!(local.iter().all(|&o| (o - 5.0).abs() < 0.2), "the error estimate is O(h⁵) per step: {local:?}");
}

#[test]
fn closed_form_solutions_adaptively() {
    // The harmonic oscillator, a hundred time units: sixteen turns.
    let spring = |_: f64, y: Simd2| Simd([y[1], -y[0]]);
    let (mut worst, mut samples) = (0.0f64, 0);
    let stats = Ode::integrate(spring, 0.0, Simd([1.0, 0.0]), 100.0, Tolerance::default(), |step| {
        // The interpolant, at three points inside every step.
        for s in [0.25, 0.5, 0.75] {
            let t = step.t0 + s * step.h;
            worst = worst.max((step.state(t) - Simd([cos(t), -sin(t)])).length());
            samples += 1;
        }
    })
    .unwrap();
    assert!(
        worst < 1e-8,
        "the oscillator, read anywhere between steps: max error {worst} over {samples} interior points, {} steps, {} rejected",
        stats.accepted,
        stats.rejected
    );

    // Forced: x'' + x = sin 2t from rest is (2/3) sin t − (1/3) sin 2t. The
    // time argument is live, so a stage evaluated at the wrong c_i shows here
    // and nowhere in an autonomous test.
    let forced = |t: f64, y: Simd2| Simd([y[1], -y[0] + sin(2.0 * t)]);
    let (states, _) = Ode::sample(forced, 0.0, Simd([0.0, 0.0]), 0.05, 2001, Tolerance::default()).unwrap();
    let mut drift = 0.0f64;
    for (k, y) in states.iter().enumerate() {
        let t = 0.05 * k as f64;
        drift = drift.max((y[0] - (2.0 / 3.0 * sin(t) - 1.0 / 3.0 * sin(2.0 * t))).abs());
    }
    assert!(states.len() == 2001 && drift < 1e-8, "a forced oscillator, sampled every 0.05 to t = 100: {drift}");
}

#[test]
fn a_kepler_orbit_keeps_its_energy_and_angular_momentum() {
    // Eccentricity 0.6, ten periods: the close pass is where the step
    // control earns its keep.
    let kepler = |_: f64, s: Simd4| {
        let r2 = s[0] * s[0] + s[1] * s[1];
        let k = pow(r2, 1.5);
        Simd([s[2], s[3], -s[0] / k, -s[1] / k])
    };
    let e = 0.6;
    let start = Simd([1.0 - e, 0.0, 0.0, ((1.0 + e) / (1.0 - e)).sqrt()]);
    let energy = |s: Simd4| 0.5 * (s[2] * s[2] + s[3] * s[3]) - 1.0 / (s[0] * s[0] + s[1] * s[1]).sqrt();
    let momentum = |s: Simd4| s[0] * s[3] - s[1] * s[2];
    let mut last = start;
    let tolerance = Tolerance { relative: 1e-11, absolute: 1e-13, max_step: None };
    let stats = Ode::integrate(kepler, 0.0, start, 20.0 * PI, tolerance, |step| last = step.state(step.t1())).unwrap();
    let d_e = (energy(last) - energy(start)).abs() / energy(start).abs();
    let d_l = (momentum(last) - momentum(start)).abs() / momentum(start).abs();
    // Back where it started, too: the period is 2π for a = 1.
    let home = (last - start).length();
    assert!(
        d_e < 1e-8 && d_l < 1e-8 && home < 1e-6,
        "after ten periods: energy {d_e}, momentum {d_l}, distance from the start {home}; {} steps",
        stats.accepted
    );
}

#[test]
fn a_discontinuous_right_hand_side_with_a_step_truncation() {
    // y' = sign(sin t): a triangle wave. Its corners are where an unbounded
    // step would stride over the change and accept an estimate that never
    // saw it.
    let zigzag = |t: f64, _: Simd2| Simd([if sin(t) >= 0.0 { 1.0 } else { -1.0 }, 0.0]);
    let exact = |t: f64| {
        let p = kurven_math::real::fmod(t, 2.0 * PI);
        if p <= PI { p } else { 2.0 * PI - p }
    };
    let tolerance = Tolerance { max_step: Some(0.05), ..Tolerance::default() };
    let (states, _) = Ode::sample(zigzag, 0.0, Simd([0.0, 0.0]), 0.1, 301, tolerance).unwrap();
    let worst = states.iter().enumerate().map(|(k, y)| (y[0] - exact(0.1 * k as f64)).abs()).fold(0.0, f64::max);
    assert!(worst < 1e-6, "tracks the zigzag through every corner: max error {worst}");
}

#[test]
fn the_fft_is_the_dft() {
    let mut rng = SplitMix(3);
    let n = 256;
    let xr: Vec<f64> = (0..n).map(|_| rng.next(-1.0, 1.0)).collect();
    let xi: Vec<f64> = (0..n).map(|_| rng.next(-1.0, 1.0)).collect();
    let (mut re, mut im) = (xr.clone(), xi.clone());
    Frequency::fft(&mut re, &mut im);
    let mut worst = 0.0f64;
    for k in 0..n {
        let (mut sr, mut si) = (0.0, 0.0);
        for j in 0..n {
            let a = -2.0 * PI * ((j * k % n) as f64) / n as f64;
            sr += xr[j] * cos(a) - xi[j] * sin(a);
            si += xr[j] * sin(a) + xi[j] * cos(a);
        }
        worst = worst.max(hypot(sr - re[k], si - im[k]));
    }
    assert!(worst < 1e-11, "every bin, to rounding: max {worst}");
}

#[test]
fn lines_and_the_generator_of_a_toruss_spectrum() {
    // Lines at ω, Ω, ω + 2Ω and 2ω − Ω, with the strongest non-forcing line
    // not the generator itself -- the case that makes "take the strongest
    // line" wrong.
    let (omega, big_omega) = (0.94, 0.396_221_328_3);
    let (dt, n) = (0.05, 80_000);
    let x: Vec<f64> = (0..n)
        .map(|k| {
            let t = 400.0 + k as f64 * dt;
            cos(omega * t)
                + 0.3 * cos(big_omega * t + 0.3)
                + 0.8 * cos((omega + 2.0 * big_omega) * t + 1.0)
                + 0.1 * cos((2.0 * omega - big_omega) * t)
        })
        .collect();
    let lines = Frequency::lines(&x, 400.0, dt, 4);
    let truth = [omega, big_omega, omega + 2.0 * big_omega, 2.0 * omega - big_omega];
    let worst = truth
        .iter()
        .map(|f| lines.iter().map(|l| (l.frequency - f).abs()).fold(f64::INFINITY, f64::min))
        .fold(0.0, f64::max);
    assert!(lines.len() == 4 && worst < 1e-9, "each line is found to its exact frequency: worst {worst}");
    let g = Frequency::generator(&lines, omega, 4, 1e-7);
    assert!(
        g.is_some_and(|g| (g.frequency - big_omega).abs() < 1e-9 && g.explained == 3),
        "the generator is Ω though ω + 2Ω is the stronger line: {g:?}"
    );
}

#[test]
fn a_torus_that_is_a_fourier_series_is_fitted_exactly() {
    // K(θ, φ) of degree three in each angle, sampled along the line
    // (ω₁t, ω₂t): the fit at M = 4 has every true coefficient and should
    // reproduce K everywhere on the torus, not only along the samples.
    fn k(a: f64, b: f64) -> Simd3 {
        Simd([
            2.0 * cos(a) + 0.5 * cos(a + b) - 0.2 * sin(3.0 * b),
            2.0 * sin(a) + 0.3 * sin(2.0 * a - b) + 0.1,
            0.7 * sin(b) + 0.2 * cos(a - 3.0 * b),
        ])
    }
    let (w1, w2, dt) = (0.94, 0.396_221_328_3, 0.05);
    // What the averages cannot remove is a finite run's leakage between
    // lines, and the Hann² window makes that fall steeply with length:
    // measured at two lengths, it must.
    let errors = |n: usize| {
        let samples: Vec<Simd3> = (0..n)
            .map(|i| {
                let t = i as f64 * dt;
                k(w1 * t, w2 * t)
            })
            .collect();
        let fit = FourierTorus::fit(&samples, 0.0, dt, (w1, w2), 4);
        let mut rng = SplitMix(5);
        let (mut worst, mut slope) = (0.0f64, 0.0f64);
        for _ in 0..500 {
            let a = rng.next(0.0, 2.0 * PI);
            let b = rng.next(0.0, 2.0 * PI);
            let j = fit.jet(a, b);
            worst = worst.max((j.value - k(a, b)).length());
            let h = 1e-6;
            let da = (k(a + h, b) - k(a - h, b)) / (2.0 * h);
            let db = (k(a, b + h) - k(a, b - h)) / (2.0 * h);
            slope = slope.max((j.d_theta - da).length()).max((j.d_phi - db).length());
        }
        // Not in the Swift suite: the second-order jet's value and first
        // derivatives are the jet's, and the residual is small.
        let s = fit.second_order_jet(1.0, 2.0);
        let j = fit.jet(1.0, 2.0);
        assert_eq!((s.value, s.d_theta, s.d_phi), (j.value, j.d_theta, j.d_phi));
        (worst, slope, fit.residual(&samples, 0.0, dt, 3))
    };
    let short = errors(80_000);
    let long = errors(160_000);
    assert!(
        long.0 < 1e-8 && short.0 / long.0 > 8.0,
        "everywhere on the torus, not only on the line, and better with length: max {} at T = 4000, {} at T = 8000",
        short.0,
        long.0
    );
    assert!(long.1 < 1e-7, "and so are its derivatives: max {}", long.1);
    assert!(long.2 < 1e-8, "and it holds the trajectory it came from: residual {}", long.2);
}

#[test]
fn the_quarter_periods_are_the_complete_elliptic_integral() {
    assert!((Jacobi::quarter_period(0.0) - PI / 2.0).abs() < 1e-15, "K(0) = π/2");
    // K(1/2) = Γ(1/4)² / (4√π), to the last digit.
    let k_half = Jacobi::quarter_period(0.5);
    assert!((k_half - 1.854_074_677_301_371_9).abs() < 4e-16, "K(1/2) = Γ(1/4)² / 4√π: {k_half}");
    let m = 0.64;
    let (k, kp) = (Jacobi::quarter_period(m), Jacobi::quarter_period(1.0 - m));
    assert!((Jacobi::real(k, m).sn - 1.0).abs() < 1e-14, "sn(K) = 1");
    let mut rng = SplitMix(11);
    let mut worst = 0.0f64;
    for _ in 0..200 {
        let z = Complex::new(rng.next(-3.0, 3.0), rng.next(-1.0, 1.0));
        let s = Jacobi::complex(z, m).sn;
        for shifted in [Complex::new(z.re + 4.0 * k, z.im), Complex::new(z.re, z.im + 2.0 * kp)] {
            let t = Jacobi::complex(shifted, m).sn;
            worst = worst.max((t - s).magnitude() / s.magnitude().max(1.0));
        }
    }
    assert!(worst < 1e-11, "sn has periods 4K and 2iK': worst {worst}");
}
