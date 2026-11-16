//! Small fixed-width vectors of doubles, for the ODE integrator and the
//! Fourier torus: what Swift's `SIMD2<Double>`, `SIMD3<Double>` and
//! `SIMD4<Double>` are to `ODE.swift` and `FourierTorus.swift`.
//!
//! Every operation is lane by lane with the scalar's own rounding, as Swift's
//! are; nothing is fused.

use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, Neg, Sub, SubAssign};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Simd<const N: usize>(pub [f64; N]);

pub type Simd2 = Simd<2>;
pub type Simd3 = Simd<3>;
pub type Simd4 = Simd<4>;

impl<const N: usize> Default for Simd<N> {
    fn default() -> Self {
        Simd([0.0; N])
    }
}

impl<const N: usize> Simd<N> {
    pub const ZERO: Simd<N> = Simd([0.0; N]);

    /// The largest lane: Swift's `max()` on a vector of floating-point
    /// lanes, a reduction by `Double.maximum` (IEEE maxNum): a NaN lane is
    /// passed over unless every lane is NaN, and of two equal lanes the later
    /// is kept.
    pub fn max(self) -> f64 {
        let mut m = self.0[0];
        for &y in &self.0[1..] {
            m = if m > y || y.is_nan() { m } else { y };
        }
        m
    }

    /// Each lane's magnitude: `v.replacing(with: -v, where: v .< 0)`, so a
    /// negative zero stays as it is.
    pub fn absolute(self) -> Simd<N> {
        Simd(self.0.map(|x| if x < 0.0 { -x } else { x }))
    }

    /// Lane by lane the larger: `a.replacing(with: b, where: a .< b)`.
    pub fn pointwise_max(a: Simd<N>, b: Simd<N>) -> Simd<N> {
        let mut out = a.0;
        for (o, &y) in out.iter_mut().zip(&b.0) {
            if *o < y {
                *o = y;
            }
        }
        Simd(out)
    }

    /// The Euclidean length, `simd_length`.
    pub fn length(self) -> f64 {
        let mut s = 0.0;
        for &x in &self.0 {
            s += x * x;
        }
        s.sqrt()
    }

    fn zip(self, b: Simd<N>, f: impl Fn(f64, f64) -> f64) -> Simd<N> {
        let mut out = self.0;
        for (o, &y) in out.iter_mut().zip(&b.0) {
            *o = f(*o, y);
        }
        Simd(out)
    }
}

impl<const N: usize> Index<usize> for Simd<N> {
    type Output = f64;
    fn index(&self, i: usize) -> &f64 {
        &self.0[i]
    }
}

impl<const N: usize> IndexMut<usize> for Simd<N> {
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.0[i]
    }
}

impl<const N: usize> Add for Simd<N> {
    type Output = Simd<N>;
    fn add(self, b: Simd<N>) -> Simd<N> {
        self.zip(b, |x, y| x + y)
    }
}

impl<const N: usize> Sub for Simd<N> {
    type Output = Simd<N>;
    fn sub(self, b: Simd<N>) -> Simd<N> {
        self.zip(b, |x, y| x - y)
    }
}

impl<const N: usize> Div for Simd<N> {
    type Output = Simd<N>;
    fn div(self, b: Simd<N>) -> Simd<N> {
        self.zip(b, |x, y| x / y)
    }
}

impl<const N: usize> Neg for Simd<N> {
    type Output = Simd<N>;
    fn neg(self) -> Simd<N> {
        Simd(self.0.map(|x| -x))
    }
}

impl<const N: usize> Mul<Simd<N>> for f64 {
    type Output = Simd<N>;
    fn mul(self, v: Simd<N>) -> Simd<N> {
        Simd(v.0.map(|x| self * x))
    }
}

impl<const N: usize> Mul<f64> for Simd<N> {
    type Output = Simd<N>;
    fn mul(self, s: f64) -> Simd<N> {
        Simd(self.0.map(|x| x * s))
    }
}

impl<const N: usize> Div<f64> for Simd<N> {
    type Output = Simd<N>;
    fn div(self, s: f64) -> Simd<N> {
        Simd(self.0.map(|x| x / s))
    }
}

/// `scalar + vector`, lane by lane: Swift's `tol.absolute + v`.
impl<const N: usize> Add<Simd<N>> for f64 {
    type Output = Simd<N>;
    fn add(self, v: Simd<N>) -> Simd<N> {
        Simd(v.0.map(|x| self + x))
    }
}

impl<const N: usize> AddAssign for Simd<N> {
    fn add_assign(&mut self, b: Simd<N>) {
        *self = *self + b;
    }
}

impl<const N: usize> SubAssign for Simd<N> {
    fn sub_assign(&mut self, b: Simd<N>) {
        *self = *self - b;
    }
}

impl<const N: usize> DivAssign<f64> for Simd<N> {
    fn div_assign(&mut self, s: f64) {
        *self = *self / s;
    }
}
