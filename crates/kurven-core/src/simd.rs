//! The small vectors and matrices Swift takes from `simd`: `SIMD2<Double>`,
//! `SIMD3<Double>`, `SIMD4<Double>`, `simd_double3x3` and `simd_double4x4`,
//! as [`Vec2`], [`Vec3`], [`Vec4`], [`Mat3`] and [`Mat4`].
//!
//! Matrices are stored by columns, as simd stores them, and multiply a
//! column vector on the right. Every product is written out in one fixed
//! order, column by column, unfused: simd on Apple silicon may contract
//! `a * b + c` into a fused multiply-add, the port never does (no FMA, so
//! native and wasm agree to the bit), and the two are compared to a
//! tolerance, never as bytes.

use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

use kurven_math::real;

macro_rules! vector {
    ($name:ident, $n:literal, $($field:ident : $i:literal),+) => {
        #[derive(Debug, Clone, Copy, PartialEq, Default)]
        pub struct $name {
            $(pub $field: f64,)+
        }

        impl $name {
            pub const ZERO: $name = $name { $($field: 0.0,)+ };

            pub const fn new($($field: f64),+) -> $name {
                $name { $($field,)+ }
            }

            pub const fn splat(x: f64) -> $name {
                $name { $($field: x,)+ }
            }

            pub fn to_array(self) -> [f64; $n] {
                [$(self.$field,)+]
            }

            pub fn from_array(a: [f64; $n]) -> $name {
                $name { $($field: a[$i],)+ }
            }

            /// `simd_dot`, summed in component order.
            pub fn dot(self, o: $name) -> f64 {
                let mut s = 0.0;
                $(s += self.$field * o.$field;)+
                s
            }

            /// `simd_length_squared`.
            pub fn length_squared(self) -> f64 {
                self.dot(self)
            }

            /// `simd_length`.
            pub fn length(self) -> f64 {
                real::sqrt(self.length_squared())
            }

            /// `simd_normalize`: the vector over its length.
            pub fn normalized(self) -> $name {
                self / self.length()
            }

            /// `simd_min`, component by component.
            pub fn min(self, o: $name) -> $name {
                $name { $($field: self.$field.min(o.$field),)+ }
            }

            /// `simd_max`, component by component.
            pub fn max(self, o: $name) -> $name {
                $name { $($field: self.$field.max(o.$field),)+ }
            }

            /// `simd_abs`.
            pub fn abs(self) -> $name {
                $name { $($field: self.$field.abs(),)+ }
            }

            /// `simd_reduce_max`.
            pub fn reduce_max(self) -> f64 {
                let mut m = f64::NEG_INFINITY;
                $(m = m.max(self.$field);)+
                m
            }

            /// `.sum()`.
            pub fn sum(self) -> f64 {
                let mut s = 0.0;
                $(s += self.$field;)+
                s
            }

            /// Component-wise product, simd's `*` of two vectors.
            pub fn mul_elements(self, o: $name) -> $name {
                $name { $($field: self.$field * o.$field,)+ }
            }
        }

        impl Add for $name {
            type Output = $name;
            fn add(self, o: $name) -> $name {
                $name { $($field: self.$field + o.$field,)+ }
            }
        }
        impl Sub for $name {
            type Output = $name;
            fn sub(self, o: $name) -> $name {
                $name { $($field: self.$field - o.$field,)+ }
            }
        }
        impl Neg for $name {
            type Output = $name;
            fn neg(self) -> $name {
                $name { $($field: -self.$field,)+ }
            }
        }
        impl Mul<f64> for $name {
            type Output = $name;
            fn mul(self, s: f64) -> $name {
                $name { $($field: self.$field * s,)+ }
            }
        }
        impl Mul<$name> for f64 {
            type Output = $name;
            fn mul(self, v: $name) -> $name {
                $name { $($field: self * v.$field,)+ }
            }
        }
        impl Div<f64> for $name {
            type Output = $name;
            fn div(self, s: f64) -> $name {
                $name { $($field: self.$field / s,)+ }
            }
        }
        impl AddAssign for $name {
            fn add_assign(&mut self, o: $name) {
                $(self.$field += o.$field;)+
            }
        }
        impl SubAssign for $name {
            fn sub_assign(&mut self, o: $name) {
                $(self.$field -= o.$field;)+
            }
        }
        impl MulAssign<f64> for $name {
            fn mul_assign(&mut self, s: f64) {
                $(self.$field *= s;)+
            }
        }
        impl DivAssign<f64> for $name {
            fn div_assign(&mut self, s: f64) {
                $(self.$field /= s;)+
            }
        }
        impl Index<usize> for $name {
            type Output = f64;
            fn index(&self, i: usize) -> &f64 {
                match i {
                    $($i => &self.$field,)+
                    _ => panic!("index {i} out of range"),
                }
            }
        }
        impl IndexMut<usize> for $name {
            fn index_mut(&mut self, i: usize) -> &mut f64 {
                match i {
                    $($i => &mut self.$field,)+
                    _ => panic!("index {i} out of range"),
                }
            }
        }
    };
}

vector!(Vec2, 2, x: 0, y: 1);
vector!(Vec3, 3, x: 0, y: 1, z: 2);
vector!(Vec4, 4, x: 0, y: 1, z: 2, w: 3);

impl Vec3 {
    /// `simd_cross`.
    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)
    }

    pub fn xy(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }
}

impl Vec4 {
    pub fn xyz(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}

/// `simd_double3x3`: three columns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    pub columns: [Vec3; 3],
}

impl Mat3 {
    pub const IDENTITY: Mat3 =
        Mat3 { columns: [Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)] };

    pub fn from_columns(c0: Vec3, c1: Vec3, c2: Vec3) -> Mat3 {
        Mat3 { columns: [c0, c1, c2] }
    }

    /// `simd_double3x3(rows:)`.
    pub fn from_rows(r0: Vec3, r1: Vec3, r2: Vec3) -> Mat3 {
        Mat3 { columns: [Vec3::new(r0.x, r1.x, r2.x), Vec3::new(r0.y, r1.y, r2.y), Vec3::new(r0.z, r1.z, r2.z)] }
    }

    pub fn transpose(self) -> Mat3 {
        let [c0, c1, c2] = self.columns;
        Mat3::from_rows(c0, c1, c2)
    }

    /// Row `i`.
    pub fn row(self, i: usize) -> Vec3 {
        Vec3::new(self.columns[0][i], self.columns[1][i], self.columns[2][i])
    }
}

impl Mul<Vec3> for Mat3 {
    type Output = Vec3;
    /// The columns weighted by the vector's components, summed in order.
    fn mul(self, v: Vec3) -> Vec3 {
        self.columns[0] * v.x + self.columns[1] * v.y + self.columns[2] * v.z
    }
}

impl Mul<Mat3> for Mat3 {
    type Output = Mat3;
    fn mul(self, o: Mat3) -> Mat3 {
        Mat3 { columns: [self * o.columns[0], self * o.columns[1], self * o.columns[2]] }
    }
}

/// `simd_double4x4`: four columns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    pub columns: [Vec4; 4],
}

impl Mat4 {
    pub const IDENTITY: Mat4 = Mat4 {
        columns: [
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        ],
    };

    pub fn from_columns(c0: Vec4, c1: Vec4, c2: Vec4, c3: Vec4) -> Mat4 {
        Mat4 { columns: [c0, c1, c2, c3] }
    }

    /// Element at row `r`, column `c`.
    pub fn at(&self, r: usize, c: usize) -> f64 {
        self.columns[c][r]
    }

    /// The inverse, by cofactors over the determinant, as `simd_inverse`
    /// computes it up to rounding. A singular matrix gives infinities or
    /// NaNs, as simd's does.
    pub fn inverse(self) -> Mat4 {
        let m = |r: usize, c: usize| self.at(r, c);
        // The 2x2 minors of the top two rows and of the bottom two.
        let s0 = m(0, 0) * m(1, 1) - m(1, 0) * m(0, 1);
        let s1 = m(0, 0) * m(1, 2) - m(1, 0) * m(0, 2);
        let s2 = m(0, 0) * m(1, 3) - m(1, 0) * m(0, 3);
        let s3 = m(0, 1) * m(1, 2) - m(1, 1) * m(0, 2);
        let s4 = m(0, 1) * m(1, 3) - m(1, 1) * m(0, 3);
        let s5 = m(0, 2) * m(1, 3) - m(1, 2) * m(0, 3);
        let c5 = m(2, 2) * m(3, 3) - m(3, 2) * m(2, 3);
        let c4 = m(2, 1) * m(3, 3) - m(3, 1) * m(2, 3);
        let c3 = m(2, 1) * m(3, 2) - m(3, 1) * m(2, 2);
        let c2 = m(2, 0) * m(3, 3) - m(3, 0) * m(2, 3);
        let c1 = m(2, 0) * m(3, 2) - m(3, 0) * m(2, 2);
        let c0 = m(2, 0) * m(3, 1) - m(3, 0) * m(2, 1);
        let det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
        let d = 1.0 / det;
        let r = [
            [
                (m(1, 1) * c5 - m(1, 2) * c4 + m(1, 3) * c3) * d,
                (-m(0, 1) * c5 + m(0, 2) * c4 - m(0, 3) * c3) * d,
                (m(3, 1) * s5 - m(3, 2) * s4 + m(3, 3) * s3) * d,
                (-m(2, 1) * s5 + m(2, 2) * s4 - m(2, 3) * s3) * d,
            ],
            [
                (-m(1, 0) * c5 + m(1, 2) * c2 - m(1, 3) * c1) * d,
                (m(0, 0) * c5 - m(0, 2) * c2 + m(0, 3) * c1) * d,
                (-m(3, 0) * s5 + m(3, 2) * s2 - m(3, 3) * s1) * d,
                (m(2, 0) * s5 - m(2, 2) * s2 + m(2, 3) * s1) * d,
            ],
            [
                (m(1, 0) * c4 - m(1, 1) * c2 + m(1, 3) * c0) * d,
                (-m(0, 0) * c4 + m(0, 1) * c2 - m(0, 3) * c0) * d,
                (m(3, 0) * s4 - m(3, 1) * s2 + m(3, 3) * s0) * d,
                (-m(2, 0) * s4 + m(2, 1) * s2 - m(2, 3) * s0) * d,
            ],
            [
                (-m(1, 0) * c3 + m(1, 1) * c1 - m(1, 2) * c0) * d,
                (m(0, 0) * c3 - m(0, 1) * c1 + m(0, 2) * c0) * d,
                (-m(3, 0) * s3 + m(3, 1) * s1 - m(3, 2) * s0) * d,
                (m(2, 0) * s3 - m(2, 1) * s1 + m(2, 2) * s0) * d,
            ],
        ];
        // r[row][col] of the inverse.
        Mat4 {
            columns: [
                Vec4::new(r[0][0], r[1][0], r[2][0], r[3][0]),
                Vec4::new(r[0][1], r[1][1], r[2][1], r[3][1]),
                Vec4::new(r[0][2], r[1][2], r[2][2], r[3][2]),
                Vec4::new(r[0][3], r[1][3], r[2][3], r[3][3]),
            ],
        }
    }
}

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;
    /// The columns weighted by the vector's components, summed in order.
    fn mul(self, v: Vec4) -> Vec4 {
        self.columns[0] * v.x + self.columns[1] * v.y + self.columns[2] * v.z + self.columns[3] * v.w
    }
}

impl Mul<Mat4> for Mat4 {
    type Output = Mat4;
    fn mul(self, o: Mat4) -> Mat4 {
        Mat4 { columns: [self * o.columns[0], self * o.columns[1], self * o.columns[2], self * o.columns[3]] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_and_columns() {
        let m = Mat3::from_rows(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0), Vec3::new(7.0, 8.0, 10.0));
        assert_eq!(m * Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 4.0, 7.0));
        assert_eq!(m.row(1), Vec3::new(4.0, 5.0, 6.0));
        assert_eq!(m.transpose().row(0), Vec3::new(1.0, 4.0, 7.0));
        assert_eq!(Mat3::IDENTITY * m, m);
    }

    #[test]
    fn inverse_of_an_affine_map() {
        let m = Mat4::from_columns(
            Vec4::new(2.0, 0.5, 0.0, 0.0),
            Vec4::new(-1.0, 3.0, 0.25, 0.0),
            Vec4::new(0.0, 1.0, 4.0, 0.0),
            Vec4::new(1.0, -2.0, 3.0, 1.0),
        );
        let p = m.inverse() * m;
        for c in 0..4 {
            for r in 0..4 {
                let want = if r == c { 1.0 } else { 0.0 };
                assert!((p.at(r, c) - want).abs() < 1e-14, "{r},{c}: {}", p.at(r, c));
            }
        }
    }
}
