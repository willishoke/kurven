//! The camera: the port of `Sources/KurvenCore/Camera.swift`.

use kurven_math::real;

use crate::manifest::ObliqueProjection;
use crate::navigation::Orbit;
use crate::simd::{Mat3, Mat4, Vec3, Vec4};
use crate::spaces::{Angle, P2, P3, PageSpace, Transform, ViewSpace, WorldSpace};

/// How view space becomes the drawing.
///
/// A sum type rather than a struct with optional fields: the Jahnke-Emde
/// oblique shear is a property of an orthographic drawing and means nothing
/// under perspective, so "shear under perspective" is unrepresentable instead
/// of a runtime check nobody remembers to write.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Projection {
    Orthographic { oblique: Option<Oblique> },
    Perspective { fov_y: Angle },
}

/// The foreshortening shear the plates use: in the pre-rotation frame,
/// `along -= shear * across`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Oblique {
    pub shear: f64,
}

/// A camera: one world-to-view transform and how that view is projected.
///
/// `view` is *derived*, never edited. Changing where you are looking from means
/// producing a new `Camera`; the renderer's only per-frame work is uploading
/// this matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub view: Transform<WorldSpace, ViewSpace>,
    pub projection: Projection,
}

impl Camera {
    /// The exact Python oblique camera, in world coordinates.
    ///
    /// `Projection.apply` consumes `(imag, real, z)` and, in order: scales
    /// column 1 by `yScale`, negates column 0 when `flipX`, subtracts
    /// `shear * column0` from column 1, then applies Rz followed by Rx. In world
    /// order column 0 is `y` and column 1 is `x`, so the whole of it -- axis
    /// exchange included -- is one 3x3:
    ///
    /// ```text
    ///     pre = [ 0            fx           0 ]     fx = flipX ? -1 : 1
    ///           [ ys   -shear * fx          0 ]     ys = yScale ?? 1
    ///           [ 0             0           1 ]
    ///
    ///     view = Rx(xAngle) * Rz(zAngle) * pre
    /// ```
    ///
    /// The three factors are `Orbit`'s elevation, azimuth and `DrawingConvention`, so
    /// this is `Orbit(matching:).camera` and a preset is a starting point for
    /// navigation rather than a fixed picture.
    ///
    /// Folding the exchange in here is what keeps it out of everywhere else:
    /// the arrays a bundle carries are already world order, and no stage between
    /// the decoder and the drawing touches a column index again.
    pub fn oblique(p: &ObliqueProjection) -> Camera {
        Orbit::matching(p, P3::new(0.0, 0.0, 0.0)).camera()
    }

    /// The drawing's 2D coordinates of a world point: the first two view
    /// components under orthographic projection.
    pub fn page_coordinates(&self, p: P3<WorldSpace>) -> P2<PageSpace> {
        let v = self.view.apply(p);
        match self.projection {
            Projection::Orthographic { .. } => P2::new(v.x(), v.y()),
            Projection::Perspective { fov_y } => {
                // Not reachable from a bundle preset; kept total so a future
                // navigation camera cannot silently fall through to orthographic.
                // A visible point has z < 0, so this divides by the least
                // normal, not by its distance: Swift's, ported as it is
                // (FINDINGS). Nothing calls it under perspective.
                let f = 1.0 / real::tan(fov_y.radians / 2.0);
                let d = v.z().max(f64::MIN_POSITIVE);
                P2::new(v.x() * f / d, v.y() * f / d)
            }
        }
    }

    /// The view-to-clip matrix for a perspective camera.
    ///
    /// View z increases toward the viewer, so the eye is at the origin looking
    /// down -z and a visible point has `z < 0`. The `w` row is therefore `-z`,
    /// and the divide the hardware performs is by distance along the view axis.
    /// That convention is the same one the depth buffer uses -- larger z is
    /// nearer, which is what makes a MAX blend hidden-surface removal -- so
    /// nothing downstream changes when the projection does.
    ///
    /// `near` exists only to keep geometry behind the eye from wrapping around;
    /// nothing reads the clip-space depth, because depth travels in the colour
    /// attachment. (Swift's default is 0.01, and its body never reads it.)
    pub fn perspective_clip(fov_y: Angle, aspect: f64) -> Mat4 {
        let f = 1.0 / real::tan(fov_y.radians / 2.0);
        // Columns, not rows: see `DepthFrame::metal_clip`.
        Mat4::from_columns(
            Vec4::new(f / aspect.max(f64::MIN_POSITIVE), 0.0, 0.0, 0.0),
            Vec4::new(0.0, f, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, -1.0), // w = -z
            Vec4::new(0.0, 0.0, 0.5, 0.0),  // constant depth
        )
    }

    /// True when this camera cannot be baked.
    ///
    /// A bake clips per vertex against a depth buffer it looks up by an affine
    /// map from view coordinates to pixels; under perspective that map depends
    /// on depth and the lookup is wrong. It could be fixed by projecting first
    /// and indexing the projected point -- but there is no perspective
    /// `Projection` on the Python side, so a perspective bake would be the one
    /// artifact this design cannot check against its oracle. Refusing is the
    /// honest form of that.
    pub fn is_perspective(&self) -> bool {
        matches!(self.projection, Projection::Perspective { .. })
    }
}

/// Active rotation about x, matching `scipy.spatial.transform.Rotation.from_euler("x", a)`.
pub fn rotation_x(a: Angle) -> Mat3 {
    let c = real::cos(a.radians);
    let s = real::sin(a.radians);
    Mat3::from_rows(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, c, -s), Vec3::new(0.0, s, c))
}

/// Active rotation about z, matching `Rotation.from_euler("z", a)`.
pub fn rotation_z(a: Angle) -> Mat3 {
    let c = real::cos(a.radians);
    let s = real::sin(a.radians);
    Mat3::from_rows(Vec3::new(c, -s, 0.0), Vec3::new(s, c, 0.0), Vec3::new(0.0, 0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;
    use crate::json::JsonValue;
    use crate::navigation::Orbit;

    /// "camera: each projection preset lands on the committed projection".
    /// The projections were scipy's Rotation composed as the Python plates
    /// composed it until the golden set was regenerated from Camera.drawing
    /// (within 7.1e-15 of them). Swift's tolerance, 1e-9: the cosines and
    /// sines are the libm crate's here and Darwin's there.
    #[test]
    fn each_projection_preset_lands_on_the_committed_projection() {
        for name in ["recip", "elliptic", "zeta"] {
            let spec = fixtures::shared_json(&format!("camera/{name}.json"));
            let projection = ObliqueProjection::from_json(spec.get("projection").unwrap()).unwrap();
            let world = fixtures::shared_npy(&format!("camera/{name}.points.npy")).rows3().unwrap();
            let want = fixtures::shared_npy(&format!("camera/{name}.projected.npy")).rows3().unwrap();
            let camera = Camera::oblique(&projection);
            let mut worst = 0.0f64;
            for (w, expected) in world.iter().zip(&want) {
                let got = camera.view.apply(P3::new(w[0], w[1], w[2])).v;
                worst = worst.max((got - Vec3::from_array(*expected)).abs().reduce_max());
            }
            assert!(world.len() == want.len() && worst < 1e-9, "{name}: worst |Δ| = {worst} over {}", world.len());
            assert!(matches!(spec, JsonValue::Object(_)));
        }
    }

    /// Swift's perspective branch divides by max(z, least normal), but a
    /// visible point has z < 0, so it divides by 2.2e-308: a point a unit
    /// off the target, ten in front of the eye, lands about 1e308 out. Ported
    /// as it is (FINDINGS, "Where the Swift looks wrong"); nothing reaches it.
    #[test]
    fn perspective_page_coordinates_divide_by_the_least_normal_as_swift_does() {
        let projection =
            ObliqueProjection { shear: 0.5, x_angle: -55.0, z_angle: -90.0, flip_x: true, y_scale: None };
        let mut orbit = Orbit::matching(&projection, P3::new(0.0, 0.0, 0.0));
        orbit.distance = Some(10.0);
        orbit.field_of_view = Angle::from_degrees(50.0);
        let camera = orbit.camera();
        let p = camera.view.inverse().apply(P3::new(1.0, 0.0, -10.0));
        let v = camera.view.apply(p);
        assert!((v.x() - 1.0).abs() < 1e-12 && (v.z() + 10.0).abs() < 1e-12, "in front of the eye, z < 0");
        let page = camera.page_coordinates(p);
        let f = 1.0 / real::tan(Angle::from_degrees(50.0).radians / 2.0);
        assert_eq!(page.x(), v.x() * f / f64::MIN_POSITIVE, "divided by the least normal, not by -z = 10");
        assert!(page.x() > 9e307);
    }

    /// "camera: the projection matrix is the composition it claims to be"
    #[test]
    fn the_projection_matrix_is_the_composition_it_claims_to_be() {
        let projection =
            ObliqueProjection { shear: 0.5, x_angle: -55.0, z_angle: -90.0, flip_x: true, y_scale: Some(0.75) };
        let camera = Camera::oblique(&projection);
        let p = P3::<WorldSpace>::new(1.5, -2.25, 3.0);
        // The same thing spelled out in the library's own column order.
        let mut legacy = Vec3::new(p.y(), p.x(), p.z()); // (imag, real, z)
        legacy.y *= 0.75; // yScale scales real
        legacy.x *= -1.0; // flipX flips imag
        legacy.y -= 0.5 * legacy.x; // shear, on the flipped column
        let rotated = rotation_x(Angle::from_degrees(-55.0)) * (rotation_z(Angle::from_degrees(-90.0)) * legacy);
        assert!(
            (camera.view.apply(p).v - rotated).abs().reduce_max() < 1e-12,
            "folding the axis exchange into one matrix changes nothing"
        );
        let Projection::Orthographic { oblique } = camera.projection else {
            panic!("an oblique camera is orthographic");
        };
        assert_eq!(oblique.map(|o| o.shear), Some(0.5), "the oblique shear survives as a camera property");
    }
}
