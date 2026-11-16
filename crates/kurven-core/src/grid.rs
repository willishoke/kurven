//! A regular 2D sample grid: the port of `Sources/KurvenCore/Grid.swift`.

use crate::manifest::{Domain, Interval};
use crate::simd::Vec2;
use crate::spaces::{DomainSpace, P2};

/// A regular 2D sample grid over a rectangle.
///
/// `values[y * width + x]`, with `x` indexing real and `y` indexing imag, which
/// is the orientation `height.npy` is written in and the orientation a texture
/// wants. Sample positions are the *vertices* of the grid: `x = 0` sits at
/// `domain.real.lo` and `x = width - 1` at `domain.real.hi`. That is the
/// convention `Surface` was sampled with, and getting it wrong shifts the whole
/// landscape by half a cell.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid2D<T> {
    pub width: usize,
    pub height: usize,
    pub domain: Domain,
    pub values: Vec<T>,
}

impl<T: Copy> Grid2D<T> {
    pub fn new(width: usize, height: usize, domain: Domain, values: Vec<T>) -> Grid2D<T> {
        assert!(values.len() == width * height, "grid is {width}x{height} but has {} values", values.len());
        Grid2D { width, height, domain, values }
    }

    /// The sample at column `x`, row `y`: Swift's `grid[x, y]`.
    pub fn at(&self, x: usize, y: usize) -> T {
        self.values[y * self.width + x]
    }

    /// The *shape* a `decimated(by:)` grid would have, without building it.
    ///
    /// The renderer needs four numbers from the decimated grid -- its two
    /// dimensions and its domain -- and nothing else, because the heights come
    /// from the texture. Materializing the grid to read them off costs a full
    /// copy of the samples: 43 ms per pass on zeta's 25-million-sample grid, to
    /// produce four doubles.
    pub fn decimated_extent(&self, step: usize) -> (usize, usize, Domain) {
        if step <= 1 {
            return (self.width, self.height, self.domain);
        }
        let w = self.width.div_ceil(step);
        let h = self.height.div_ceil(step);
        let dx = self.domain.real.length() / self.width.saturating_sub(1).max(1) as f64;
        let dy = self.domain.imag.length() / self.height.saturating_sub(1).max(1) as f64;
        (
            w,
            h,
            Domain::new(
                Interval::new(self.domain.real.lo, self.domain.real.lo + dx * ((w - 1) * step) as f64),
                Interval::new(self.domain.imag.lo, self.domain.imag.lo + dy * ((h - 1) * step) as f64),
            ),
        )
    }

    /// Keep every `step`-th sample in both directions. The domain is unchanged
    /// only if the last sample survives, so the decimated grid's domain is
    /// narrowed to the samples it actually kept.
    ///
    /// Prefer `decimated_extent` when only the shape is wanted.
    pub fn decimated(&self, step: usize) -> Grid2D<T> {
        if step <= 1 {
            return self.clone();
        }
        let xs: Vec<usize> = (0..self.width).step_by(step).collect();
        let ys: Vec<usize> = (0..self.height).step_by(step).collect();
        let mut out = Vec::with_capacity(xs.len() * ys.len());
        for &y in &ys {
            for &x in &xs {
                out.push(self.at(x, y));
            }
        }
        let dx = self.domain.real.length() / self.width.saturating_sub(1).max(1) as f64;
        let dy = self.domain.imag.length() / self.height.saturating_sub(1).max(1) as f64;
        Grid2D::new(
            xs.len(),
            ys.len(),
            Domain::new(
                Interval::new(self.domain.real.lo, self.domain.real.lo + dx * xs.last().copied().unwrap_or(0) as f64),
                Interval::new(self.domain.imag.lo, self.domain.imag.lo + dy * ys.last().copied().unwrap_or(0) as f64),
            ),
            out,
        )
    }
}

impl Grid2D<f32> {
    /// Fractional sample index of a domain point. Separated from `sample` so the
    /// mesh builder and the sampler cannot disagree about where sample `(i, j)`
    /// lives.
    pub fn index(&self, p: P2<DomainSpace>) -> Vec2 {
        Vec2::new(
            (p.x() - self.domain.real.lo) / self.domain.real.length().max(f64::MIN_POSITIVE)
                * (self.width as f64 - 1.0),
            (p.y() - self.domain.imag.lo) / self.domain.imag.length().max(f64::MIN_POSITIVE)
                * (self.height as f64 - 1.0),
        )
    }

    pub fn position(&self, x: usize, y: usize) -> P2<DomainSpace> {
        P2::new(
            self.domain.real.lo + self.domain.real.length() * x as f64 / self.width.saturating_sub(1).max(1) as f64,
            self.domain.imag.lo + self.domain.imag.length() * y as f64 / self.height.saturating_sub(1).max(1) as f64,
        )
    }

    /// Bilinear sample, clamped at the edges.
    pub fn sample(&self, p: P2<DomainSpace>) -> f64 {
        let f = self.index(p);
        let (w, h) = (self.width as i64, self.height as i64);
        let x0 = (f.x.floor() as i64).max(0).min(w - 1);
        let y0 = (f.y.floor() as i64).max(0).min(h - 1);
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);
        let tx = (f.x - x0 as f64).clamp(0.0, 1.0);
        let ty = (f.y - y0 as f64).clamp(0.0, 1.0);
        let at = |x: i64, y: i64| f64::from(self.at(x as usize, y as usize));
        let a = at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx;
        let b = at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx;
        a * (1.0 - ty) + b * ty
    }

    /// Nearest sample. `Surface.from_cache` on the Python side looks up this
    /// way, so a bundle built from a cache is reproduced by this, not by
    /// `sample`.
    pub fn nearest(&self, p: P2<DomainSpace>) -> f64 {
        let f = self.index(p);
        let (w, h) = (self.width as i64, self.height as i64);
        let x = (f.x.round() as i64).max(0).min(w - 1);
        let y = (f.y.round() as i64).max(0).min(h - 1);
        f64::from(self.at(x as usize, y as usize))
    }

    /// The finite samples' least and greatest; `(0, 0)` when there are none.
    pub fn range(&self) -> (f32, f32) {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in self.values.iter().filter(|v| v.is_finite()) {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if lo <= hi { (lo, hi) } else { (0.0, 0.0) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "core: decimation keeps the samples grid_mesh's stride keeps"
    #[test]
    fn decimation_keeps_the_samples_grid_meshs_stride_keeps() {
        let grid = Grid2D::new(
            7,
            5,
            Domain::new(Interval::new(0.0, 6.0), Interval::new(0.0, 4.0)),
            (0..35).map(|v| v as f32).collect(),
        );
        let d = grid.decimated(3);
        // Python: values[::3, ::3] -- columns 0, 3, 6 and rows 0, 3.
        assert!(d.width == 3 && d.height == 2 && d.values == [0.0, 3.0, 6.0, 21.0, 24.0, 27.0], "[::3, ::3]");
        assert!(d.domain.real.hi == 6.0 && d.domain.imag.hi == 3.0, "the domain narrows to the samples that survived");
        let (w, h, domain) = grid.decimated_extent(3);
        assert!(w == d.width && h == d.height && domain == d.domain, "the extent is the decimated grid's");
    }

    #[test]
    fn sampling() {
        let grid = Grid2D::new(
            3,
            2,
            Domain::new(Interval::new(0.0, 2.0), Interval::new(0.0, 1.0)),
            vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
        );
        assert_eq!(grid.sample(P2::new(0.5, 0.5)), 5.5);
        assert_eq!(grid.sample(P2::new(-5.0, 9.0)), 10.0, "clamped at the edges");
        assert_eq!(grid.nearest(P2::new(1.4, 0.6)), 11.0);
        assert_eq!(grid.position(2, 1), P2::new(2.0, 1.0));
        assert_eq!(grid.range(), (0.0, 12.0));
    }
}
