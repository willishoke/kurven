//! Frequency analysis of a quasi-periodic signal. The port of
//! `Frequency.swift`.
//!
//! A trajectory linear curve round an invariant torus is a sum of sinusoids at
//! the integer combinations `m ω₁ + n ω₂` of two basic frequencies. The FFT
//! finds each peak to within a bin; refining it -- maximizing the windowed
//! amplitude `|⟨w x e^{-iνt}⟩|` over ν, Laskar's NAFF -- finds it to many
//! orders of magnitude better, because the window makes the amplitude a
//! smooth function of ν with its maximum exactly on the line. That precision
//! is not a nicety: an error δ in a frequency drifts the phase by δ·T across
//! the run, and the torus fit reads coordinates from those phases.

use core::f64::consts::PI;

use crate::complex::Complex;
use crate::real;
use crate::swift;

pub enum Frequency {}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Line {
    /// Angular frequency.
    pub frequency: f64,
    pub amplitude: f64,
}

/// The generator of a torus's spectrum, and how many lines it explains.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Generator {
    pub frequency: f64,
    pub explained: usize,
}

impl Frequency {
    /// The Hann window squared, normalized to mean one. Its sidelobes fall
    /// fast enough that neighbouring lines do not pull a peak off its true
    /// frequency, and its squared form is what the torus fit averages with.
    pub fn window(count: usize) -> Vec<f64> {
        if count <= 1 {
            return vec![1.0];
        }
        let mut w: Vec<f64> = (0..count)
            .map(|k| {
                let s = k as f64 / (count - 1) as f64;
                let h = 1.0 - real::cos(2.0 * PI * s);
                h * h
            })
            .collect();
        let mean = w.iter().fold(0.0, |a, b| a + b) / count as f64;
        for x in &mut w {
            *x /= mean;
        }
        w
    }

    /// `⟨w x e^{-iνt}⟩` over samples at `t0 + k dt`: the windowed amplitude
    /// at angular frequency ν. The phasor is advanced by multiplication and
    /// renormalized every thousand samples, so its drift over a long series
    /// stays at rounding.
    pub fn amplitude(x: &[f64], w: &[f64], t0: f64, dt: f64, nu: f64) -> Complex {
        let step = Complex::new(real::cos(-nu * dt), real::sin(-nu * dt));
        let mut phase = Complex::new(real::cos(-nu * t0), real::sin(-nu * t0));
        let mut re = 0.0;
        let mut im = 0.0;
        for k in 0..x.len() {
            let a = w[k] * x[k];
            re += a * phase.re;
            im += a * phase.im;
            phase *= step;
            if k % 1024 == 1023 {
                let m = phase.magnitude();
                phase = Complex::new(phase.re / m, phase.im / m);
            }
        }
        let n = x.len() as f64;
        Complex::new(re / n, im / n)
    }

    /// In-place radix-2 FFT, forward (`e^{-i...}`); the length must be a
    /// power of two.
    pub fn fft(re: &mut [f64], im: &mut [f64]) {
        let n = re.len();
        assert!(n == im.len() && n > 0 && n & (n - 1) == 0, "FFT length must be 2^k");
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let angle = -2.0 * PI / len as f64;
            for start in (0..n).step_by(len) {
                for k in 0..(len / 2) {
                    let c = real::cos(angle * k as f64);
                    let s = real::sin(angle * k as f64);
                    let a = start + k;
                    let b = a + len / 2;
                    let tr = re[b] * c - im[b] * s;
                    let ti = re[b] * s + im[b] * c;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            len <<= 1;
        }
    }

    /// The strongest `count` spectral lines of `x`, each refined to its
    /// exact frequency.
    ///
    /// The windowed, mean-removed series is transformed at twice its length
    /// or more, the local maxima of the spectrum taken strongest first --
    /// no two within three bins of the series' own resolution, `2π/T` -- and
    /// each refined by golden-section search on the windowed amplitude over
    /// the bin either side.
    pub fn lines(x: &[f64], t0: f64, dt: f64, count: usize) -> Vec<Line> {
        let n = x.len();
        if n < 8 {
            return Vec::new();
        }
        let w = Frequency::window(n);
        let mean = w.iter().zip(x).fold(0.0, |acc, (w, x)| acc + w * x) / n as f64;
        let centred: Vec<f64> = x.iter().map(|x| x - mean).collect();

        let mut size = 1;
        while size < 2 * n {
            size <<= 1;
        }
        let mut re = vec![0.0; size];
        let mut im = vec![0.0; size];
        for k in 0..n {
            re[k] = w[k] * centred[k];
        }
        Frequency::fft(&mut re, &mut im);
        let bin_width = 2.0 * PI / (dt * size as f64);
        let magnitude: Vec<f64> = (0..=(size / 2)).map(|k| (re[k] * re[k] + im[k] * im[k]).sqrt()).collect();
        let mut maxima: Vec<usize> =
            (1..(size / 2)).filter(|&k| magnitude[k] > magnitude[k - 1] && magnitude[k] >= magnitude[k + 1]).collect();
        // Strongest first. Swift's sort is stable (since Swift 5; checked
        // with swiftc on this machine) and so is Rust's, so equal magnitudes
        // keep their order in both. `real::total_order` orders as Swift's `>` does on
        // every magnitude but NaN, which only a NaN sample makes; there Rust's
        // sort needs a total order, which `>` is not, and a NaN of either sign is the same NaN. The same holds for the
        // lines' final sort below.
        maxima.sort_by(|&a, &b| real::total_order(magnitude[b], magnitude[a]));

        let resolution = 2.0 * PI / (dt * n as f64);
        let mut found: Vec<Line> = Vec::new();
        for bin in maxima {
            let guess = bin as f64 * bin_width;
            if !found.iter().all(|l| (l.frequency - guess).abs() > 3.0 * resolution) {
                continue;
            }
            let amp = |nu: f64| Frequency::amplitude(&centred, &w, t0, dt, nu).magnitude();
            let mut a = guess - resolution;
            let mut b = guess + resolution;
            let g = (5.0f64.sqrt() - 1.0) / 2.0;
            let mut c = b - g * (b - a);
            let mut d = a + g * (b - a);
            let mut fc = amp(c);
            let mut fd = amp(d);
            while b - a > 1e-13 * swift::max(1.0, guess) {
                if fc > fd {
                    b = d;
                    d = c;
                    fd = fc;
                    c = b - g * (b - a);
                    fc = amp(c);
                } else {
                    a = c;
                    c = d;
                    fc = fd;
                    d = a + g * (b - a);
                    fd = amp(d);
                }
            }
            let nu = 0.5 * (a + b);
            found.push(Line { frequency: nu, amplitude: amp(nu) });
            if found.len() == count {
                break;
            }
        }
        found.sort_by(|a, b| real::total_order(b.amplitude, a.amplitude));
        found
    }

    /// The second basic frequency of a torus forced at ω, from its lines.
    ///
    /// Every line is `m ω + n Ω`. Reduced mod ω the harmonics of the forcing
    /// vanish and the rest are `n Ω mod ω`, so the generator is the reduced
    /// frequency of which the others are small multiples -- not simply the
    /// strongest line, which on a torus can as well be `2Ω` and would then
    /// describe half of it. Each candidate, from the strongest lines, is
    /// scored by how many lines it explains with `|n| ≤ max_multiple`, and
    /// ties go to the stronger. Returned in `(0, ω/2]`: `Ω` and `ω − Ω`
    /// generate the same torus with the second angle reversed. Swift's
    /// defaults are `max_multiple` 4 and `tolerance` 1e-7.
    pub fn generator(lines: &[Line], omega: f64, max_multiple: usize, tolerance: f64) -> Option<Generator> {
        let reduce = |nu: f64| {
            let r = real::fmod(nu, omega);
            if r < 0.0 { r + omega } else { r }
        };
        let tol = tolerance * omega;
        let reduced: Vec<f64> =
            lines.iter().map(|l| reduce(l.frequency)).filter(|&r| r > tol && omega - r > tol).collect(); // not a harmonic of ω
        let mut best: Option<Generator> = None;
        for &c in &reduced {
            let explained = reduced
                .iter()
                .filter(|&&r| {
                    (1..=max_multiple).any(|n| {
                        let k = reduce(n as f64 * c);
                        (k - r).abs() < n as f64 * tol || (omega - k - r).abs() < n as f64 * tol
                    })
                })
                .count();
            if best.is_none_or(|b| explained > b.explained) {
                best = Some(Generator { frequency: swift::min(c, omega - c), explained });
            }
        }
        best
    }
}
