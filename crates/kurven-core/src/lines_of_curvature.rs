//! Lines of curvature (do Carmo §3-2): the port of
//! `Sources/KurvenCore/LinesOfCurvature.swift`.
//!
//! The integral curves of a surface's principal directions, the directions in
//! which its normal curvature is greatest and least. The two families cross at
//! right angles in space. On a surface of revolution they are the meridians and
//! the parallels, so on the round torus they are its coordinate curves.
//!
//! The principal directions are a field of line elements, with no sign, so
//! they are a vector field whose integral curves are traced unsigned
//! (`VectorField::signed`, D9). They are undefined at an umbilic, where the
//! two principal curvatures are equal and every direction is principal, so
//! a line of curvature that runs into one ends there, at a singular point of
//! the field (`End::Singular`).

use std::sync::Arc;

use kurven_math::real;

use crate::face::{Face, FaceCopy};
use crate::hatching::FaceSelector;
use crate::integral_curves::{Limits, Section, Traced, VectorField};
use crate::manifest::{
    ContourField, DirectionField, HeightPolicy, LayerRole, LayerSource, LayerSpec, ManifestError, PrincipalCurvature,
    Seeds, Spacing,
};
use crate::parametric::{ParamAxis, ParamScalar, ParametricSurface, SecondDerivatives, SurfaceJet};
use crate::polyline::PolylineSet;
use crate::simd::{Vec2, Vec3};
use crate::spaces::{P2, P3, ParamSpace, WorldSpace};
use crate::surface_request::SurfaceRequest;

/// The principal curvatures at a point, the greater first, each one's
/// principal direction in the parameter plane, unit length in space, and
/// whether the point is an umbilic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Principal {
    pub greater: f64,
    pub lesser: f64,
    pub directions: (Vec2, Vec2),
    pub umbilic: bool,
}

impl Face {
    /// How near the two principal curvatures may come before a point is
    /// taken for an umbilic, where the principal directions are undefined:
    /// k₁ − k₂ under this fraction of |k₁| + |k₂|, well above the rounding
    /// of second derivatives in closed form, under which the directions
    /// would be noise. A planar point, k₁ = k₂ = 0, is an umbilic too.
    ///
    /// A line of curvature heading into an umbilic does not reach it. Round
    /// a generic umbilic the principal directions turn half a turn, one way
    /// or the other, for each turn round it (index ±½), so a step that would
    /// pass it samples a direction a right angle from the heading, and is
    /// refused at 45° and halved until it falls under the least step: the
    /// line ends at a singular point (`End::Singular`), short of the umbilic
    /// by about the least step.
    pub const UMBILIC_TOLERANCE: f64 = 1e-6;

    /// The principal curvatures at a point, the greater first, measured
    /// against the normal into the solid the copy bounds, so a convex
    /// surface's are positive; where it bounds none, against −(du × dv), the
    /// parametrization's own. With them, each one's principal direction in
    /// the parameter plane, unit length in space, and whether the point is
    /// an umbilic (`UMBILIC_TOLERANCE`). None where the face's map has no
    /// second derivatives (`second_order_jet`) or the point is singular.
    pub fn principal_curvatures(&self, p: P2<ParamSpace>, copy: &FaceCopy) -> Option<Principal> {
        let second = self.second_order_jet.as_ref()?;
        let j = second(p);
        Face::principal(j.jet, j.second, -copy.outward.unwrap_or(1.0))
    }

    /// The principal directions of `curvature` on `copy`, as an unsigned
    /// vector field (`VectorField::signed`): its integral curves are that
    /// family's lines of curvature. Undefined at an umbilic. None where the
    /// face's map has no second derivatives.
    pub fn principal_directions(&self, curvature: PrincipalCurvature, copy: &FaceCopy) -> Option<VectorField> {
        let second = self.second_order_jet.clone()?;
        let inward = -copy.outward.unwrap_or(1.0);
        // One evaluation of the map's jet of order two a sample.
        let mut field = VectorField::new(Arc::new(move |p| {
            let j = second(p);
            let k = Face::principal(j.jet, j.second, inward).filter(|k| !k.umbilic)?;
            Some(if curvature == PrincipalCurvature::Greater { k.directions.0 } else { k.directions.1 })
        }));
        field.signed = false;
        Some(field)
    }

    /// The principal curvatures and directions of a jet and its second
    /// derivatives, against the unit normal `inward · (du × dv)/|du × dv|`.
    pub fn principal(jet: SurfaceJet, second: SecondDerivatives, inward: f64) -> Option<Principal> {
        let n = jet.normal();
        let length = n.length();
        let du = jet.du.length();
        if !(length > 0.0 && length.is_finite() && du > 0.0) {
            return None;
        }
        let normal = inward * n / length;
        // An orthonormal frame of the tangent plane, e₁ along du and e₂ the
        // part of dv across it, as vectors of the parameter plane.
        let e1 = jet.du / du;
        let across = jet.dv - jet.dv.dot(e1) * e1;
        let w = across.length();
        if !(w > 0.0) {
            return None;
        }
        let p1 = Vec2::new(1.0 / du, 0.0);
        let p2 = Vec2::new(-jet.dv.dot(e1) / du, 1.0) / w;
        // The second fundamental form, L du² + 2M du dv + N dv², in it.
        let (l, m, nn) = (second.uu.dot(normal), second.uv.dot(normal), second.vv.dot(normal));
        let form = |x: Vec2, y: Vec2| l * x.x * y.x + m * (x.x * y.y + x.y * y.x) + nn * x.y * y.y;
        let (a, b, c) = (form(p1, p1), form(p1, p2), form(p2, p2));
        let mean = 0.5 * (a + c);
        let half = 0.5 * (a - c);
        let radius = real::sqrt(half * half + b * b);
        if !(mean.is_finite() && radius.is_finite()) {
            return None;
        }
        let angle = 0.5 * real::atan2(b, half);
        let (ca, sa) = (real::cos(angle), real::sin(angle));
        let (greater, lesser) = (mean + radius, mean - radius);
        Some(Principal {
            greater,
            lesser,
            directions: (ca * p1 + sa * p2, -sa * p1 + ca * p2),
            umbilic: 2.0 * radius <= Face::UMBILIC_TOLERANCE * (greater.abs() + lesser.abs()),
        })
    }
}

/// A field on a surface by name, and its period where it is circle-valued.
#[derive(Clone)]
pub struct NamedField {
    pub value: ParamScalar,
    pub period: Option<f64>,
}

impl ParametricSurface {
    /// The most seeds one layer of integral curves takes: each curve runs up
    /// to `max_steps` steps each way, and a layer holds
    /// `SurfaceRequest::MAX_VERTICES` vertices. A bound on the seeds, not a
    /// count of vertices, so approximate: a curve also takes the point its
    /// end is bisected to and the middle of its last step, and a vertex again
    /// at each split every eight turns, which at the bound is a few percent
    /// more.
    pub fn max_integral_curves(limits: &Limits) -> i64 {
        SurfaceRequest::MAX_VERTICES / (2 * limits.max_steps + 1)
    }

    /// A field on the surface by name, and its period where it is
    /// circle-valued: a coordinate, or the height, the map's z. None for f's
    /// fields and an isophote, which the map does not carry. What a contour
    /// of a coordinate or the height is taken of, and what seeds are placed
    /// by.
    pub fn field(&self, name: ContourField) -> Option<NamedField> {
        let map = self.map.clone();
        match name {
            ContourField::U => Some(NamedField { value: Arc::new(|p: P2<ParamSpace>| p.x()), period: self.u.period() }),
            ContourField::V => Some(NamedField { value: Arc::new(|p: P2<ParamSpace>| p.y()), period: self.v.period() }),
            ContourField::Height => Some(NamedField { value: Arc::new(move |p| map(p).position.z), period: None }),
            ContourField::Magnitude | ContourField::Phase | ContourField::Isophote => None,
        }
    }

    /// Where `seeds` lie on the surface's one face: along its level set of a
    /// coordinate or the height (`Face::contours`), at listed values of
    /// another or by arc length in space (`Seeds::points`). A coordinate
    /// along a periodic axis is read the short way round, so values the same
    /// modulo its period are one, and a level set that closes across a seam
    /// is read once round. Nothing where the seeds are placed by a field the
    /// surface does not carry.
    pub fn seed_points(&self, seeds: &Seeds) -> Vec<P2<ParamSpace>> {
        self.placed(seeds, None).0
    }

    /// The seeds' points, or none and how many there would be where they are
    /// placed by arc length and would be more than `most`: counted off the
    /// level set's length before any is placed.
    fn placed(&self, seeds: &Seeds, most: Option<i64>) -> (Vec<P2<ParamSpace>>, Option<i64>) {
        let face = self.face();
        let copy = &face.copies[0];
        let map = self.map.clone();
        let Some(on) = self.field(seeds.field) else { return (Vec::new(), None) };
        let value = on.value.clone();
        let level: Vec<Vec<P2<ParamSpace>>> = face
            .contours(&crate::scalar_field::ScalarField::new(on.period, move |p| value(p)), &[seeds.level], copy)
            .into_iter()
            .flat_map(|(_, paths)| paths)
            .collect();
        let mut at: Option<NamedField> = None;
        match &seeds.spacing {
            Spacing::Values(of, _) => {
                let Some(named) = self.field(*of) else { return (Vec::new(), None) };
                at = Some(named);
            }
            Spacing::ArcLength(spacing) => {
                // As `Seeds::points` counts them, round(L / spacing) a piece,
                // and past ten million, where it places none, too many as well.
                if let Some(most) = most {
                    let mut count = 0.0;
                    for path in level.iter().filter(|p| p.len() >= 2) {
                        let mut length = 0.0;
                        for w in path.windows(2) {
                            length += (map(w[0]).position - map(w[1]).position).length();
                        }
                        let ratio = length / spacing;
                        count += if ratio.is_finite() { ratio.round() } else { f64::INFINITY };
                    }
                    if count > most as f64 {
                        // Swift's `Int(count)`, which fits below 2^63.
                        return (Vec::new(), Some(if count.is_finite() { count as i64 } else { i64::MAX }));
                    }
                }
            }
        }
        let position = |p: P2<ParamSpace>| -> Vec3 { map(p).position };
        let value_fn = at.as_ref().map(|n| n.value.clone());
        let value_ref: Option<&dyn Fn(P2<ParamSpace>) -> f64> = value_fn.as_ref().map(|f| f.as_ref() as _);
        let periods = Vec2::new(self.u.period().unwrap_or(0.0), self.v.period().unwrap_or(0.0));
        (seeds.points(&level, value_ref, at.and_then(|n| n.period), periods, &position), None)
    }

    /// The level set `seeds` lie on as a section their curves return to
    /// (`Section`), or None where the surface does not carry its field.
    pub fn section(&self, seeds: &Seeds) -> Option<Section> {
        self.field(seeds.field).map(|f| Section::new(f.value, seeds.level, f.period))
    }

    /// The integral curves of `field` through `points` on the surface's one
    /// face, in its parameter plane, with why each ends
    /// (`Face::integral_curve`), to where it first returns to `section`
    /// where one is given. Nothing where the face has no such field: the
    /// principal directions where the map has no second derivatives, and a
    /// slope, which lies on the graph. Swift traces them in parallel; here
    /// in turn, each independent of the others.
    pub fn traced_integral_curves(
        &self,
        field: &DirectionField,
        points: &[P2<ParamSpace>],
        section: Option<&Section>,
        limits: &Limits,
    ) -> Vec<Traced> {
        let face = self.face();
        let copy = &face.copies[0];
        let DirectionField::PrincipalDirection(curvature) = field else { return Vec::new() };
        let Some(vectors) = face.principal_directions(*curvature, copy) else { return Vec::new() };
        points
            .iter()
            .filter_map(|&seed| face.integral_curve(&vectors, seed, copy, limits, section))
            .filter(|t| t.path.len() >= 2)
            .collect()
    }

    /// The integral curves of `field` through `seeds`, in space, each vertex
    /// on the map with its surface coordinate beside it, so they are tested
    /// as the coordinate curves are. A curve runs on across the seams, its
    /// coordinates unwrapped, and closes on its seed's image a whole number
    /// of periods along; one that does not close ends where it first comes
    /// back across the level set its seeds lie on, once round
    /// (`section`, `Section`). As a linear curve's, its coordinates are kept
    /// within eight turns of the rectangle, for the preview's float32: a path
    /// is ended and the next begun from the same vertex, brought back by
    /// whole periods, wherever it would run further.
    ///
    /// Refused where the layer could not be decoded on the surface
    /// (`LayerSource::unsupported_integral_curves`), and where its seeds are
    /// more than `max_integral_curves`, before any is traced; by arc length,
    /// before any is placed.
    pub fn integral_curves(
        &self,
        field: &DirectionField,
        seeds: &Seeds,
        limits: &Limits,
    ) -> Result<PolylineSet<WorldSpace>, ManifestError> {
        if let Some(why) = LayerSource::unsupported_integral_curves(field, seeds, &FaceSelector::Surface, None) {
            return Err(ManifestError::Unsupported(why));
        }
        let most = ParametricSurface::max_integral_curves(limits);
        let (points, over) = self.placed(seeds, Some(most));
        if let Some(count) = over.or(if points.len() as i64 > most { Some(points.len() as i64) } else { None }) {
            return Err(ManifestError::Unsupported(format!(
                "{count} seeds of integral curves, more than the {most} a layer holds at {} steps each way",
                limits.max_steps
            )));
        }
        let section = self.section(seeds);
        let traced = self.traced_integral_curves(field, &points, section.as_ref(), limits);
        let lo = Vec2::new(self.u.range.lo, self.v.range.lo);
        let periods = [self.u.period().unwrap_or(0.0), self.v.period().unwrap_or(0.0)];
        // The whole periods to take from `c` to bring it into the rectangle.
        let offset = |c: Vec2| -> Vec2 {
            let mut o = [0.0, 0.0];
            for axis in 0..2 {
                if periods[axis] > 0.0 {
                    o[axis] = periods[axis] * ((c[axis] - lo[axis]) / periods[axis]).floor();
                }
            }
            Vec2::new(o[0], o[1])
        };
        let mut coords: Vec<Vec<P2<ParamSpace>>> = Vec::new();
        for curve in &traced {
            let mut path: Vec<P2<ParamSpace>> = Vec::new();
            let mut o = offset(curve.path[0].v);
            for c in &curve.path {
                let local = c.v - o - lo;
                let far = (0..2).any(|a| periods[a] > 0.0 && local[a].abs() > 8.0 * periods[a]);
                if far && let Some(&last) = path.last() {
                    coords.push(std::mem::take(&mut path));
                    let next = offset(c.v);
                    path = vec![P2::from_vec(last.v + o - next)];
                    o = next;
                }
                path.push(P2::from_vec(c.v - o));
            }
            coords.push(path);
        }
        let paths: Vec<Vec<P3<WorldSpace>>> =
            coords.iter().map(|line| line.iter().map(|&c| P3::from_vec((self.map)(c).position)).collect()).collect();
        Ok(PolylineSet::from_paths_with_coords(&paths, &coords))
    }
}

impl LayerSpec {
    /// One family of lines of curvature on a surface over `rectangle`, as
    /// the window adds it and `surface --lines-of-curvature` draws it, named
    /// `lines_of_curvature_greater` or `_lesser`: the greater family seeded
    /// on the coordinate curve v = its range's start, at `count` values of u
    /// evenly round, the lesser on u = its start at `count` values of v. On
    /// a torus of revolution the greater family is the meridians and the
    /// lesser the parallels, so each seed draws one of them, crossing the
    /// curve it is seeded on.
    pub fn lines_of_curvature(
        curvature: PrincipalCurvature,
        rectangle: (ParamAxis, ParamAxis),
        count: i64,
        width: f64,
    ) -> LayerSpec {
        let greater = curvature == PrincipalCurvature::Greater;
        let (along, across) = if greater { (rectangle.0, rectangle.1) } else { (rectangle.1, rectangle.0) };
        let n = count.max(1);
        let values: Vec<f64> = (0..n).map(|k| along.range.lo + along.range.length() * k as f64 / n as f64).collect();
        let seeds = if greater {
            Seeds { field: ContourField::V, level: across.range.lo, spacing: Spacing::Values(ContourField::U, values) }
        } else {
            Seeds { field: ContourField::U, level: across.range.lo, spacing: Spacing::Values(ContourField::V, values) }
        };
        LayerSpec::new(
            &format!("lines_of_curvature_{}", curvature.raw()),
            LayerRole::Scaffold,
            LayerSource::IntegralCurves {
                field: DirectionField::PrincipalDirection(curvature),
                seeds,
                faces: FaceSelector::Surface,
            },
            width,
            HeightPolicy::Surface,
        )
    }
}
