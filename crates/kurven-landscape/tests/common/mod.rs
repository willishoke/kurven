//! What the face suites share: landscapes built on f as
//! `NativeLandscape.build(_:refine: true)` builds them (its grids and its
//! refiner; the bundle around them is the build's own, not ported yet), and
//! Swift's `Segments`, polylines bucketed for distance and even-odd
//! membership.

#![allow(dead_code)]

use kurven_core::grid::Grid2D;
use kurven_core::manifest::{Domain, Interval, RealBand, Truncation};
use kurven_core::simd::Vec2;
use kurven_core::surface::{ContourRefine, Surface};
use kurven_landscape::{ContourRefiner, NativeLandscape};
use kurven_math::Expression;

pub fn domain(r0: f64, r1: f64, i0: f64, i1: f64) -> Domain {
    Domain::new(Interval::new(r0, r1), Interval::new(i0, i1))
}

pub fn bands(bands: &[(f64, f64)], beyond: f64) -> Truncation {
    Truncation::RealBands { bands: bands.iter().map(|&(below, height)| RealBand { below, height }).collect(), beyond }
}

/// A landscape on f: its surface, as `NativeLandscape.build` makes it from
/// the float32 grids of |f| and arg f, and the refiner's hook.
pub struct Built {
    pub surface: Surface,
    pub refine: ContourRefine,
}

impl Built {
    pub fn magnitude(&self) -> &kurven_core::surface::Magnitude {
        self.refine.magnitude.as_ref().expect("the refiner has |f|")
    }
}

/// `NativeLandscape.build(LandscapeRequest(expression:domain:resolution:
/// truncation:), refine: true)`, its surface and refiner.
pub fn build(expression: &str, domain: Domain, resolution: i64, truncation: Truncation) -> Built {
    let compiled = Expression::compile(expression).unwrap_or_else(|e| panic!("{expression}: {e:?}"));
    let samples = NativeLandscape::sample(&compiled, domain, resolution).expect("the domain samples");
    let s = samples.shape;
    let height = Grid2D::new(s.n_real, s.n_imag, domain, NativeLandscape::quantize(&samples.magnitude));
    let phase = Grid2D::new(s.n_real, s.n_imag, domain, NativeLandscape::quantize(&samples.phase));
    let surface = Surface::new(height, Some(phase), truncation, false);
    let refine = ContourRefiner::new(compiled, domain, s.n_real, s.n_imag).refine_hook();
    Built { surface, refine }
}

/// Polylines as segments, bucketed so that a point is met only with those
/// near it: the distance to them within a radius, and even-odd membership by
/// a ray along +x (Swift's `Segments`).
pub struct Segments {
    a: Vec<Vec2>,
    b: Vec<Vec2>,
    lo: Vec2,
    size: f64,
    columns: usize,
    rows: usize,
    cells: Vec<Vec<usize>>,
    /// By rows alone, for the ray: each segment once in a row it spans.
    strips: Vec<Vec<usize>>,
}

impl Segments {
    pub fn new(paths: &[Vec<Vec2>], size: f64) -> Segments {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for path in paths {
            for w in path.windows(2) {
                if w[0] != w[1] {
                    a.push(w[0]);
                    b.push(w[1]);
                }
            }
        }
        let mut lo = Vec2::splat(f64::INFINITY);
        let mut hi = -lo;
        for (p, q) in a.iter().zip(&b) {
            lo = lo.min(p.min(*q));
            hi = hi.max(p.max(*q));
        }
        if a.is_empty() {
            lo = Vec2::ZERO;
            hi = Vec2::ZERO;
        }
        let count = |extent: f64| -> usize { (((extent / size).floor() as i64 + 1).clamp(1, 4096)) as usize };
        let (columns, rows) = (count(hi.x - lo.x), count(hi.y - lo.y));
        let mut cells = vec![Vec::new(); columns * rows];
        let mut strips = vec![Vec::new(); rows];
        for k in 0..a.len() {
            let (l, h) = (a[k].min(b[k]), a[k].max(b[k]));
            let (i0, i1) = (Segments::index(l.x, lo.x, size, columns), Segments::index(h.x, lo.x, size, columns));
            let (j0, j1) = (Segments::index(l.y, lo.y, size, rows), Segments::index(h.y, lo.y, size, rows));
            for j in j0..=j1 {
                strips[j].push(k);
                for i in i0..=i1 {
                    cells[j * columns + i].push(k);
                }
            }
        }
        Segments { a, b, lo, size, columns, rows, cells, strips }
    }

    fn index(x: f64, lo: f64, size: f64, count: usize) -> usize {
        let f = ((x - lo) / size).floor();
        if f < 0.0 {
            0
        } else if f >= count as f64 {
            count - 1
        } else {
            f as usize
        }
    }

    /// The distance from `p` to the nearest segment, or infinity where none
    /// lies within `r`.
    pub fn distance(&self, p: Vec2, r: f64) -> f64 {
        if self.a.is_empty() {
            return f64::INFINITY;
        }
        let (i0, i1) = (
            Segments::index(p.x - r, self.lo.x, self.size, self.columns),
            Segments::index(p.x + r, self.lo.x, self.size, self.columns),
        );
        let (j0, j1) = (
            Segments::index(p.y - r, self.lo.y, self.size, self.rows),
            Segments::index(p.y + r, self.lo.y, self.size, self.rows),
        );
        let mut best = f64::INFINITY;
        for j in j0..=j1 {
            for i in i0..=i1 {
                for &k in &self.cells[j * self.columns + i] {
                    let d = self.b[k] - self.a[k];
                    let t = (p - self.a[k]).dot(d) / d.length_squared();
                    let t = 0.0f64.max(1.0f64.min(t));
                    best = best.min((p - (self.a[k] + t * d)).length());
                }
            }
        }
        if best <= r { best } else { f64::INFINITY }
    }

    /// Whether the segments wind round `p` an odd number of times: a ray
    /// along +x crosses an odd number of them, each half open in y.
    pub fn inside(&self, p: Vec2) -> bool {
        if self.a.is_empty() || !(p.y >= self.lo.y && p.y <= self.lo.y + self.rows as f64 * self.size) {
            return false;
        }
        let mut inside = false;
        for &k in &self.strips[Segments::index(p.y, self.lo.y, self.size, self.rows)] {
            let (a, b) = (self.a[k], self.b[k]);
            if (a.y > p.y) == (b.y > p.y) {
                continue;
            }
            if p.x < a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y) {
                inside = !inside;
            }
        }
        inside
    }
}
