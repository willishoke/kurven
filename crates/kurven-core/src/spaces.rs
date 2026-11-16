//! Coordinate spaces, as uninhabited types: the port of
//! `Sources/KurvenCore/Spaces.swift`.
//!
//! The dominant historical bug class in kurven is the `(imag, real)` column
//! order: every Python module documents it and several commits exist only to
//! fix having forgotten it. Here the convention is a type. A point in the
//! complex plane and a point on the drawing are different types, a transform
//! between them is a `Transform<A, B>`, and composing them in the wrong order
//! does not compile. The docstring is replaced by the compiler.

use std::fmt;
use std::marker::PhantomData;

use crate::simd::{Mat4, Vec2, Vec3, Vec4};

/// (real, imag) in the complex plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainSpace {}
/// (x = real, y = imag, z = |f|).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorldSpace {}
/// After the camera; z increases toward the viewer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewSpace {}
/// 2D output, SVG units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageSpace {}
/// (u, v) on a surface's parameter rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamSpace {}

/// A point in `S`.
pub struct P2<S> {
    pub v: Vec2,
    space: PhantomData<S>,
}

impl<S> P2<S> {
    pub const fn new(x: f64, y: f64) -> P2<S> {
        P2 { v: Vec2::new(x, y), space: PhantomData }
    }
    pub const fn from_vec(v: Vec2) -> P2<S> {
        P2 { v, space: PhantomData }
    }
    pub fn x(&self) -> f64 {
        self.v.x
    }
    pub fn y(&self) -> f64 {
        self.v.y
    }
}

pub struct P3<S> {
    pub v: Vec3,
    space: PhantomData<S>,
}

impl<S> P3<S> {
    pub const fn new(x: f64, y: f64, z: f64) -> P3<S> {
        P3 { v: Vec3::new(x, y, z), space: PhantomData }
    }
    pub const fn from_vec(v: Vec3) -> P3<S> {
        P3 { v, space: PhantomData }
    }
    pub fn x(&self) -> f64 {
        self.v.x
    }
    pub fn y(&self) -> f64 {
        self.v.y
    }
    pub fn z(&self) -> f64 {
        self.v.z
    }
    pub fn xy(&self) -> P2<S> {
        P2::new(self.v.x, self.v.y)
    }

    /// A point displaced by a world-space vector. Spelled out rather than
    /// operator-overloaded: a point plus a point is not a thing, and the
    /// phantom space is there to keep that from being expressible.
    pub fn plus(target: P3<S>, delta: Vec3) -> P3<S> {
        P3::from_vec(target.v + delta)
    }
}

// Derived impls would ask `S` for the trait too; the phantom space carries
// no value, so these are written by hand.
macro_rules! point_impls {
    ($p:ident) => {
        impl<S> Clone for $p<S> {
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<S> Copy for $p<S> {}
        impl<S> PartialEq for $p<S> {
            fn eq(&self, o: &Self) -> bool {
                self.v == o.v
            }
        }
        impl<S> fmt::Debug for $p<S> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}{:?}", stringify!($p), self.v.to_array())
            }
        }
    };
}
point_impls!(P2);
point_impls!(P3);

/// A linear-plus-translation map from `A` to `B`.
///
/// Stored column-major as a 4x4 (the simd convention), applied as
/// `m * (p, 1)`. `then` reads left to right, as Swift's `>>>` does: `world.then(view)`
/// is "do the world transform, then the view one", and the phantom
/// parameters make the only legal composition the one that type-checks.
pub struct Transform<A, B> {
    pub m: Mat4,
    spaces: PhantomData<(A, B)>,
}

impl<A, B> Clone for Transform<A, B> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<A, B> Copy for Transform<A, B> {}
impl<A, B> PartialEq for Transform<A, B> {
    fn eq(&self, o: &Self) -> bool {
        self.m == o.m
    }
}
impl<A, B> fmt::Debug for Transform<A, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Transform({:?})", self.m)
    }
}

impl<A, B> Transform<A, B> {
    pub const fn new(m: Mat4) -> Transform<A, B> {
        Transform { m, spaces: PhantomData }
    }

    pub const fn identity() -> Transform<A, B> {
        Transform::new(Mat4::IDENTITY)
    }

    /// Build from a 3x3 linear part in row-major order, plus a translation.
    pub fn from_rows(rows: (Vec3, Vec3, Vec3), translation: Vec3) -> Transform<A, B> {
        let (r0, r1, r2) = rows;
        let t = translation;
        Transform::new(Mat4::from_columns(
            Vec4::new(r0.x, r1.x, r2.x, 0.0),
            Vec4::new(r0.y, r1.y, r2.y, 0.0),
            Vec4::new(r0.z, r1.z, r2.z, 0.0),
            Vec4::new(t.x, t.y, t.z, 1.0),
        ))
    }

    /// The transform of a point: Swift's `callAsFunction`.
    pub fn apply(&self, p: P3<A>) -> P3<B> {
        let q = self.m * Vec4::new(p.v.x, p.v.y, p.v.z, 1.0);
        P3::new(q.x, q.y, q.z)
    }

    pub fn apply_all(&self, ps: &[P3<A>]) -> Vec<P3<B>> {
        ps.iter().map(|&p| self.apply(p)).collect()
    }

    pub fn inverse(&self) -> Transform<B, A> {
        Transform::new(self.m.inverse())
    }

    /// `a.then(b)`: apply `a`, then `b`. Swift's `a >>> b`.
    pub fn then<C>(&self, b: &Transform<B, C>) -> Transform<A, C> {
        Transform::new(b.m * self.m)
    }

    /// The matrix in single precision, column by column, as the GPU takes it.
    pub fn float4x4(&self) -> [[f32; 4]; 4] {
        self.m.columns.map(|c| c.to_array().map(|x| x as f32))
    }
}

/// An axis-aligned bounding box in `S`. An empty box is `None`.
#[allow(clippy::upper_case_acronyms)] // Swift's name, so the Swift is findable
pub struct AABB<S> {
    pub lo: Vec3,
    pub hi: Vec3,
    space: PhantomData<S>,
}

impl<S> Clone for AABB<S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S> Copy for AABB<S> {}
impl<S> PartialEq for AABB<S> {
    fn eq(&self, o: &Self) -> bool {
        self.lo == o.lo && self.hi == o.hi
    }
}
impl<S> fmt::Debug for AABB<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AABB({:?}, {:?})", self.lo.to_array(), self.hi.to_array())
    }
}

impl<S> AABB<S> {
    pub const fn new(lo: Vec3, hi: Vec3) -> AABB<S> {
        AABB { lo, hi, space: PhantomData }
    }

    /// The box of the points, or `None` when there are none.
    pub fn of<'a>(points: impl IntoIterator<Item = &'a P3<S>>) -> Option<AABB<S>>
    where
        S: 'a,
    {
        let mut lo = Vec3::splat(f64::INFINITY);
        let mut hi = Vec3::splat(f64::NEG_INFINITY);
        let mut any = false;
        for p in points {
            lo = lo.min(p.v);
            hi = hi.max(p.v);
            any = true;
        }
        any.then_some(AABB::new(lo, hi))
    }

    pub fn union(&self, other: &AABB<S>) -> AABB<S> {
        AABB::new(self.lo.min(other.lo), self.hi.max(other.hi))
    }

    pub fn size(&self) -> Vec3 {
        self.hi - self.lo
    }

    pub fn center(&self) -> Vec3 {
        (self.lo + self.hi) / 2.0
    }
}

/// Degrees, kept distinct from radians because the drawing parameters are stated
/// in degrees and every conversion is a chance to lose a factor of pi.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Angle {
    pub radians: f64,
}

impl Angle {
    pub const fn from_radians(radians: f64) -> Angle {
        Angle { radians }
    }
    pub fn from_degrees(degrees: f64) -> Angle {
        Angle { radians: degrees * core::f64::consts::PI / 180.0 }
    }
    pub fn degrees(&self) -> f64 {
        self.radians * 180.0 / core::f64::consts::PI
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "core: transforms compose left to right"
    #[test]
    fn transforms_compose_left_to_right() {
        let a = Transform::<WorldSpace, ViewSpace>::from_rows(
            (Vec3::new(2.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
            Vec3::new(1.0, 0.0, 0.0),
        );
        let b = Transform::<ViewSpace, PageSpace>::from_rows(
            (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 3.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
            Vec3::ZERO,
        );
        let p = P3::<WorldSpace>::new(1.0, 2.0, 3.0);
        assert_eq!(a.then(&b).apply(p), b.apply(a.apply(p)), "a then b is 'apply a, then b'");
        assert_eq!(a.then(&b).apply(p), P3::<PageSpace>::new(3.0, 6.0, 3.0));
    }

    #[test]
    fn boxes() {
        let ps = [P3::<WorldSpace>::new(1.0, -2.0, 0.0), P3::new(-1.0, 4.0, 2.0)];
        let b = AABB::of(&ps).unwrap();
        assert_eq!(b.lo, Vec3::new(-1.0, -2.0, 0.0));
        assert_eq!(b.center(), Vec3::new(0.0, 1.0, 1.0));
        assert!(AABB::<WorldSpace>::of(&[]).is_none());
    }
}
