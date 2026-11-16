//! An invariant torus as a two-dimensional Fourier series. The port of
//! `FourierTorus.swift`.

use crate::complex::Complex;
use crate::frequency::Frequency;
use crate::real;
use crate::scratch;
use crate::simd::Simd;

/// An invariant torus as a two-dimensional Fourier series,
/// `K(θ, φ) = Σ c_{mn} e^{i(mθ + nφ)}` over `|m|, |n| ≤ M`, fitted to a
/// quasi-periodic trajectory with basic frequencies `(ω₁, ω₂)`.
///
/// On the torus the trajectory is the straight line `(θ, φ) = (ω₁t, ω₂t)`,
/// and with incommensurate frequencies that line covers it evenly -- so a
/// coefficient is a time average, `c_{mn} = ⟨w(t) x(t) e^{-i(mω₁ + nω₂)t}⟩`,
/// weighted by the Hann² window so the finite run does not leak each line
/// into its neighbours. No linear system is solved; on the systems measured
/// this matches least squares at the same order, for a cost of one pass over
/// the samples per coefficient block.
///
/// A real signal's coefficients come in conjugate pairs, `c_{-m,-n} =
/// conj(c_{mn})`, so only `m ≥ 0` is stored: `K = Re Σ_{m≥0} μ_m Σ_n c_{mn}
/// e^{i(mθ+nφ)}` with `μ_0 = 1` and `μ_m = 2` otherwise.
#[derive(Clone, PartialEq, Debug)]
pub struct FourierTorus<const N: usize> {
    pub frequencies: (f64, f64),
    pub harmonics: usize,
    /// `c[(m * (2M + 1) + n + M)]`, real and imaginary parts, one lane per
    /// state component.
    re: Vec<Simd<N>>,
    im: Vec<Simd<N>>,
}

/// The torus at a point with its two partial derivatives.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Jet<const N: usize> {
    pub value: Simd<N>,
    pub d_theta: Simd<N>,
    pub d_phi: Simd<N>,
}

/// The torus at a point with its two first and three second partial
/// derivatives.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SecondOrderJet<const N: usize> {
    pub value: Simd<N>,
    pub d_theta: Simd<N>,
    pub d_phi: Simd<N>,
    pub theta_theta: Simd<N>,
    pub theta_phi: Simd<N>,
    pub phi_phi: Simd<N>,
}

impl<const N: usize> FourierTorus<N> {
    fn width(&self) -> usize {
        2 * self.harmonics + 1
    }

    /// The fit, over samples at `t0 + k dt`.
    ///
    /// Swift splits the samples over as many chunks as the machine has
    /// cores, so the order of its sums, and their last bits, depend on the
    /// machine. Here the sums run over the samples in order, in one pass, so
    /// the fit is the same everywhere (`crates/FINDINGS.md`).
    #[allow(clippy::needless_range_loop)]
    pub fn fit(samples: &[Simd<N>], t0: f64, dt: f64, frequencies: (f64, f64), harmonics: usize) -> FourierTorus<N> {
        let m_max = harmonics;
        let n = samples.len();
        let w = Frequency::window(n);
        let width = 2 * m_max + 1;
        let blocks = (m_max + 1) * width;
        let mut re = vec![Simd::ZERO; blocks];
        let mut im = vec![Simd::ZERO; blocks];
        let mut a = vec![Complex::ZERO; m_max + 1];
        let mut b = vec![Complex::ZERO; width];
        for k in 0..n {
            let t = t0 + k as f64 * dt;
            phasors(-frequencies.0 * t, -frequencies.1 * t, m_max, &mut a, &mut b);
            let x = w[k] * samples[k];
            for m in 0..=m_max {
                for j in 0..width {
                    let e = a[m] * b[j];
                    re[m * width + j] += e.re * x;
                    im[m * width + j] += e.im * x;
                }
            }
        }
        for i in 0..blocks {
            re[i] /= n as f64;
            im[i] /= n as f64;
        }
        FourierTorus { frequencies, harmonics, re, im }
    }

    /// The torus at `(θ, φ)`, with its two partial derivatives. Allocates
    /// nothing: the phasors live in the thread's scratch for the call, as
    /// Swift's live on the stack.
    #[allow(clippy::needless_range_loop)]
    pub fn jet(&self, theta: f64, phi: f64) -> Jet<N> {
        let m_max = self.harmonics;
        let width = self.width();
        scratch::with_complex(m_max + 1 + width, |buffer| {
            let (a, b) = buffer.split_at_mut(m_max + 1);
            phasors(theta, phi, m_max, a, b);
            let (mut value, mut d_theta, mut d_phi) = (Simd::ZERO, Simd::ZERO, Simd::ZERO);
            for m in 0..=m_max {
                let mu = if m == 0 { 1.0 } else { 2.0 };
                for j in 0..width {
                    let e = a[m] * b[j];
                    let i = m * width + j;
                    // Re(c e), and Re(i k c e) = -k Im(c e), lane by lane.
                    let real = e.re * self.re[i] - e.im * self.im[i];
                    let imag = e.re * self.im[i] + e.im * self.re[i];
                    value += mu * real;
                    d_theta -= (mu * m as f64) * imag;
                    d_phi -= (mu * (j as f64 - m_max as f64)) * imag;
                }
            }
            Jet { value, d_theta, d_phi }
        })
    }

    /// The torus at `(θ, φ)` with its two first and three second partial
    /// derivatives, in one pass over the series, exact as the series is: the
    /// first Re(i k c e), the second Re(−k k' c e), for the frequencies k, k'
    /// each takes.
    #[allow(clippy::needless_range_loop)]
    pub fn second_order_jet(&self, theta: f64, phi: f64) -> SecondOrderJet<N> {
        let m_max = self.harmonics;
        let width = self.width();
        scratch::with_complex(m_max + 1 + width, |buffer| {
            let (a, b) = buffer.split_at_mut(m_max + 1);
            phasors(theta, phi, m_max, a, b);
            let mut out = SecondOrderJet {
                value: Simd::ZERO,
                d_theta: Simd::ZERO,
                d_phi: Simd::ZERO,
                theta_theta: Simd::ZERO,
                theta_phi: Simd::ZERO,
                phi_phi: Simd::ZERO,
            };
            for m in 0..=m_max {
                let mu = if m == 0 { 1.0 } else { 2.0 };
                for j in 0..width {
                    let e = a[m] * b[j];
                    let i = m * width + j;
                    let real = e.re * self.re[i] - e.im * self.im[i];
                    let imag = e.re * self.im[i] + e.im * self.re[i];
                    let n = j as f64 - m_max as f64;
                    let (km, kn) = (mu * m as f64, mu * n);
                    out.value += mu * real;
                    out.d_theta -= km * imag;
                    out.d_phi -= kn * imag;
                    out.theta_theta -= (km * m as f64) * real;
                    out.theta_phi -= (km * n) * real;
                    out.phi_phi -= (kn * n) * real;
                }
            }
            out
        })
    }

    pub fn value(&self, theta: f64, phi: f64) -> Simd<N> {
        self.jet(theta, phi).value
    }

    /// The worst distance, lane-wise Euclidean over the first `lanes` lanes,
    /// between the samples and the torus at their own phases: how well the
    /// fit holds the trajectory it came from.
    pub fn residual(&self, samples: &[Simd<N>], t0: f64, dt: f64, lanes: usize) -> f64 {
        let mut worst = 0.0f64;
        for (k, sample) in samples.iter().enumerate() {
            let t = t0 + k as f64 * dt;
            let d = self.value(self.frequencies.0 * t, self.frequencies.1 * t) - *sample;
            let mut s = 0.0;
            for l in 0..lanes {
                s += d[l] * d[l];
            }
            let r = s.sqrt();
            // Swift's `max(w, r)`: `r >= w ? r : w`.
            if r >= worst {
                worst = r;
            }
        }
        worst
    }
}

/// `e^{imα}` for `0 ≤ m ≤ M` into `a`, and `e^{inβ}` for `-M ≤ n ≤ M`
/// into `b` (at `n + M`), by multiplication from one sine and cosine each.
#[inline]
fn phasors(alpha: f64, beta: f64, m_max: usize, a: &mut [Complex], b: &mut [Complex]) {
    let ea = Complex::new(real::cos(alpha), real::sin(alpha));
    let eb = Complex::new(real::cos(beta), real::sin(beta));
    a[0] = Complex::real(1.0);
    for m in 1..=m_max {
        a[m] = a[m - 1] * ea;
    }
    b[m_max] = Complex::real(1.0);
    let conj = Complex::new(eb.re, -eb.im);
    for n in 1..=m_max {
        b[m_max + n] = b[m_max + n - 1] * eb;
        b[m_max - n] = b[m_max - n + 1] * conj;
    }
}
