//! Navigation as a pure function: the port of
//! `Sources/KurvenCore/Navigation.swift`.

use kurven_math::real;

use crate::camera::{Camera, Oblique, Projection, rotation_x, rotation_z};
use crate::depth::{DepthFrame, RasterOrder};
use crate::manifest::{Interval, ObliqueProjection};
use crate::simd::{Mat3, Vec2, Vec3};
use crate::spaces::{AABB, Angle, P2, P3, Transform, ViewSpace, WorldSpace};

/// Everything in an oblique projection that is not a rotation.
///
/// `ObliqueProjection` mixes two unrelated things: where the camera is (`zAngle`,
/// `xAngle`) and what kind of drawing this is (the axis convention, the x flip,
/// the y scale, the Jahnke-Emde oblique shear). Only the first pair should move
/// when you orbit. Separating them is what lets a preset be a starting point for
/// navigation rather than a fixed picture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawingConvention {
    pub shear: f64,
    pub flip_x: bool,
    pub y_scale: Option<f64>,
}

impl DrawingConvention {
    pub fn new(projection: &ObliqueProjection) -> DrawingConvention {
        DrawingConvention { shear: projection.shear, flip_x: projection.flip_x, y_scale: projection.y_scale }
    }

    /// The world-to-pre-rotation matrix: the axis exchange, the flip, the y
    /// scale and the shear, in the order `Projection.apply` performs them.
    ///
    /// ```text
    ///     pre = [ 0            fx           0 ]     fx = flipX ? -1 : 1
    ///           [ ys   -shear * fx          0 ]     ys = yScale ?? 1
    ///           [ 0             0           1 ]
    /// ```
    pub fn matrix(&self) -> Mat3 {
        let fx = if self.flip_x { -1.0 } else { 1.0 };
        let ys = self.y_scale.unwrap_or(1.0);
        Mat3::from_rows(Vec3::new(0.0, fx, 0.0), Vec3::new(ys, -self.shear * fx, 0.0), Vec3::new(0.0, 0.0, 1.0))
    }
}

/// A CAD-style camera: what it looks at, and from where.
///
/// Orbiting a landscape means turning about the world's vertical axis and
/// tilting toward it, which is exactly the `Rz` then `Rx` the plates already
/// apply. So an orbit camera is not a new kind of camera -- it is the drawing
/// camera with its two angles made adjustable and a target subtracted first,
/// and `Orbit::matching` followed by `.camera()` reproduces `Camera::oblique`
/// exactly. That identity is what makes "load a preset, then navigate away from
/// it" a continuous motion rather than a jump.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Orbit {
    /// The world point held at the centre of the turn.
    pub target: P3<WorldSpace>,
    /// Rotation about the world vertical.
    pub azimuth: Angle,
    /// Tilt toward the viewer.
    pub elevation: Angle,
    pub style: DrawingConvention,
    /// How far back the eye sits, under perspective. `None` is orthographic,
    /// where there is no eye position -- only a direction and a rectangle.
    pub distance: Option<f64>,
    pub field_of_view: Angle,
}

impl Orbit {
    /// An orthographic orbit with the default field of view, 45°.
    pub fn new(target: P3<WorldSpace>, azimuth: Angle, elevation: Angle, style: DrawingConvention) -> Orbit {
        Orbit { target, azimuth, elevation, style, distance: None, field_of_view: Angle::from_degrees(45.0) }
    }

    pub fn is_perspective(&self) -> bool {
        self.distance.is_some()
    }

    /// The orbit that reproduces a drawing preset. With the target at the
    /// origin (Swift's default) this gives back exactly `Camera::oblique(projection)`.
    pub fn matching(projection: &ObliqueProjection, target: P3<WorldSpace>) -> Orbit {
        Orbit::new(
            target,
            Angle::from_degrees(projection.z_angle),
            Angle::from_degrees(projection.x_angle),
            DrawingConvention::new(projection),
        )
    }

    pub fn camera(&self) -> Camera {
        let m = rotation_x(self.elevation) * rotation_z(self.azimuth) * self.style.matrix();
        // The target is subtracted in world space, before the drawing transform,
        // so it lands on the view origin whatever the style does to the axes.
        let mut t = m * (-self.target.v);
        let rows = (m.row(0), m.row(1), m.row(2));
        let Some(distance) = self.distance else {
            return Camera {
                view: Transform::from_rows(rows, t),
                projection: Projection::Orthographic { oblique: Some(Oblique { shear: self.style.shear }) },
            };
        };
        // The eye sits at the view origin looking down -z, so the target is
        // pushed to z = -distance. That is the same z convention the depth
        // buffer uses -- larger is nearer -- so nothing downstream changes when
        // the projection does.
        t.z -= distance;
        Camera {
            view: Transform::from_rows(rows, t),
            // No oblique: the Jahnke-Emde shear is a property of an
            // orthographic drawing, and the ADT will not let it be otherwise.
            projection: Projection::Perspective { fov_y: self.field_of_view },
        }
    }

    /// Elevation is clamped rather than wrapped: past vertical the landscape
    /// turns inside out and the horizon flips, which reads as the input having
    /// broken rather than as a viewpoint anyone wanted. In degrees.
    pub const ELEVATION_LIMIT_DEGREES: f64 = 89.5;
}

/// The pixel raster a preview draws into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Viewport {
    pub width: i64,
    pub height: i64,
}

impl Viewport {
    pub fn new(width: i64, height: i64) -> Viewport {
        Viewport { width: width.max(1), height: height.max(1) }
    }
    pub fn aspect(&self) -> f64 {
        self.width as f64 / self.height as f64
    }
}

/// Where the viewport sits in view space, and how much of it it shows.
///
/// Pan and zoom live here rather than in the camera. A camera says which way the
/// landscape faces; sliding the paper around under it and holding a magnifier
/// over it are operations on the paper. Keeping them apart means the orbit is
/// three numbers that mean what they say, and it means zoom-toward-the-cursor is
/// a two-line calculation on a rectangle instead of a correction applied to a
/// matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Framing {
    /// The view point at the centre of the viewport.
    pub center: P2<ViewSpace>,
    /// View-space units per pixel -- one number, so pixels are square.
    pub units_per_pixel: f64,
}

impl Framing {
    pub fn new(center: P2<ViewSpace>, units_per_pixel: f64) -> Framing {
        Framing { center, units_per_pixel: units_per_pixel.max(f64::MIN_POSITIVE) }
    }

    /// The view point under a pixel. Pixel y is measured from the top, view y
    /// grows upward, so one of them is negated exactly once, here.
    pub fn view_point(&self, p: Vec2, v: Viewport) -> P2<ViewSpace> {
        P2::new(
            self.center.x() + (p.x - v.width as f64 / 2.0) * self.units_per_pixel,
            self.center.y() - (p.y - v.height as f64 / 2.0) * self.units_per_pixel,
        )
    }

    pub fn pixel(&self, p: P2<ViewSpace>, v: Viewport) -> Vec2 {
        Vec2::new(
            (p.x() - self.center.x()) / self.units_per_pixel + v.width as f64 / 2.0,
            (self.center.y() - p.y()) / self.units_per_pixel + v.height as f64 / 2.0,
        )
    }

    /// The raster this framing draws into: screen order, square pixels, one
    /// sample per pixel.
    pub fn frame(&self, v: Viewport) -> DepthFrame {
        let half_x = self.units_per_pixel * (v.width - 1) as f64 / 2.0;
        let half_y = self.units_per_pixel * (v.height - 1) as f64 / 2.0;
        DepthFrame::new(
            // Row 0 is the top of the picture, so axis0 descends.
            Interval::new(self.center.y() + half_y, self.center.y() - half_y),
            Interval::new(self.center.x() - half_x, self.center.x() + half_x),
            v.height,
            v.width,
        )
        .with_order(RasterOrder::Screen)
    }

    /// The framing that shows all of `bounds`, with a margin (Swift's
    /// default is 0.04).
    pub fn fitting(bounds: &AABB<ViewSpace>, v: Viewport, margin: f64) -> Framing {
        let size = bounds.size();
        let scale = 1.0 + 2.0 * margin;
        let upp = (size.x * scale / v.width as f64).max(size.y * scale / v.height as f64).max(f64::MIN_POSITIVE);
        Framing::new(P2::new(bounds.center().x, bounds.center().y), upp)
    }

    /// Swift's default margin for `fitting`.
    pub const FIT_MARGIN: f64 = 0.04;

    /// Zoom, keeping the view point under `pixel` where it is. Under an
    /// orthographic camera that is the whole of "zoom toward the cursor"; there
    /// is no dolly and nothing to correct afterwards.
    pub fn zoomed(&self, factor: f64, pixel: Vec2, v: Viewport) -> Framing {
        let anchor = self.view_point(pixel, v);
        let upp = self.units_per_pixel / factor.max(f64::MIN_POSITIVE);
        Framing::new(
            P2::new(
                anchor.x() - (pixel.x - v.width as f64 / 2.0) * upp,
                anchor.y() + (pixel.y - v.height as f64 / 2.0) * upp,
            ),
            upp,
        )
    }

    /// Drag the paper by a pixel delta.
    pub fn panned(&self, d: Vec2) -> Framing {
        Framing::new(
            P2::new(self.center.x() - d.x * self.units_per_pixel, self.center.y() + d.y * self.units_per_pixel),
            self.units_per_pixel,
        )
    }
}

/// What the input handling produces. Not what it does.
///
/// A `Gesture` is a value; `Navigator::applying` folds it. Input handling becomes
/// "turn an event into one of these", navigation becomes a pure function, and
/// undo is a list of `Navigator`s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Gesture {
    /// Drag to turn, in pixels.
    Orbit(Vec2),
    /// Drag to slide the paper, in pixels.
    Pan(Vec2),
    /// Scroll or pinch, anchored at a pixel.
    Zoom { factor: f64, at: Vec2 },
    /// Trackpad rotation, in radians.
    Roll(f64),
    /// Look at a world point, keeping its position on screen.
    Retarget(P3<WorldSpace>),
    /// Jump to a preset, refit to the scene.
    Preset(ObliqueProjection, AABB<ViewSpace>),
    /// Show all of it.
    Fit(AABB<ViewSpace>),
    /// Switch projection. `None` is orthographic; a field of view is
    /// perspective, which is a way of navigating rather than a drawing convention --
    /// the plates are orthographic and oblique, and a perspective bake would be
    /// the one artifact with no Python oracle to check it against.
    Project { field_of_view: Option<Angle>, bounds: AABB<ViewSpace> },
}

/// The camera state a preview draws from: an orbit and a framing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Navigator {
    pub orbit: Orbit,
    pub framing: Framing,
    /// Degrees of turn per pixel dragged.
    pub sensitivity: f64,
}

impl Navigator {
    /// A navigator at Swift's default sensitivity, 0.35° a pixel.
    pub fn new(orbit: Orbit, framing: Framing) -> Navigator {
        Navigator { orbit, framing, sensitivity: 0.35 }
    }

    pub fn camera(&self) -> Camera {
        self.orbit.camera()
    }

    /// Fold a gesture. Pure: the same navigator and gesture always give the same
    /// result, which is why the property tests below are worth anything.
    pub fn applying(&self, g: &Gesture, v: Viewport) -> Navigator {
        let mut out = *self;
        let orbit = &self.orbit;
        match *g {
            Gesture::Orbit(d) => {
                out.orbit.azimuth = Angle::from_degrees(orbit.azimuth.degrees() - d.x * self.sensitivity);
                let tilt = orbit.elevation.degrees() + d.y * self.sensitivity;
                out.orbit.elevation =
                    Angle::from_degrees(tilt.clamp(-Orbit::ELEVATION_LIMIT_DEGREES, Orbit::ELEVATION_LIMIT_DEGREES));
            }
            Gesture::Pan(d) => {
                if orbit.is_perspective() {
                    // Move what is being looked at, not the paper: dragging under
                    // perspective means walking sideways, and the target is where
                    // the walk is centred.
                    let inverse = self.camera().view.inverse();
                    let scale = orbit.distance.unwrap_or(1.0) * 2.0 * real::tan(orbit.field_of_view.radians / 2.0)
                        / v.height.max(1) as f64;
                    let origin = inverse.apply(P3::new(0.0, 0.0, 0.0));
                    let moved = inverse.apply(P3::new(-d.x * scale, d.y * scale, 0.0));
                    out.orbit.target = P3::plus(orbit.target, moved.v - origin.v);
                } else {
                    out.framing = self.framing.panned(d);
                }
            }
            Gesture::Zoom { factor, at } => {
                if let Some(distance) = orbit.distance {
                    // Under perspective there is no rectangle to shrink: coming
                    // closer *is* the zoom, and it changes the picture rather than
                    // scaling it, which is the whole reason to have this camera.
                    out.orbit.distance = Some((distance / factor.max(f64::MIN_POSITIVE)).max(1e-4));
                } else {
                    out.framing = self.framing.zoomed(factor, at, v);
                }
            }
            Gesture::Roll(radians) => {
                out.orbit.azimuth = Angle::from_radians(orbit.azimuth.radians + radians);
            }
            Gesture::Retarget(p) => {
                // Keep the point where it is on screen: re-targeting moves the
                // centre of rotation, not the picture. Moving the target moves
                // the view origin, so the same world point lands somewhere else
                // in view space; shifting the framing centre by that difference
                // cancels it exactly and the point stays under the cursor.
                // Anything else makes a double-click jump, which reads as a
                // misclick.
                let before = self.camera().view.apply(p).xy();
                out.orbit.target = p;
                let after = out.camera().view.apply(p).xy();
                out.framing = Framing::new(
                    P2::new(
                        self.framing.center.x() + after.x() - before.x(),
                        self.framing.center.y() + after.y() - before.y(),
                    ),
                    self.framing.units_per_pixel,
                );
            }
            Gesture::Preset(ref projection, ref bounds) => {
                out.orbit = Orbit::matching(projection, orbit.target);
                out.framing = Framing::fitting(bounds, v, Framing::FIT_MARGIN);
            }
            Gesture::Fit(ref bounds) => {
                out.framing = Framing::fitting(bounds, v, Framing::FIT_MARGIN);
            }
            Gesture::Project { field_of_view, ref bounds } => {
                let Some(field_of_view) = field_of_view else {
                    out.orbit.distance = None;
                    out.framing = Framing::fitting(bounds, v, Framing::FIT_MARGIN);
                    return out;
                };
                out.orbit.field_of_view = field_of_view;
                // Under perspective the picture is centred on the target, not on a
                // rectangle -- so the target has to move to the middle of what is
                // being looked at. The bounds are in view space; the rotation does
                // not change here, so taking their centre back through the current
                // camera gives the world point that will land at the view origin.
                out.orbit.target = self.camera().view.inverse().apply(P3::from_vec(bounds.center()));
                // Stand back far enough that they subtend the field of view.
                let size = bounds.size();
                let half = (size.x / v.aspect().max(f64::MIN_POSITIVE)).max(size.y) / 2.0;
                out.orbit.distance = Some((half / real::tan(field_of_view.radians / 2.0)).max(1e-3) + size.z.max(0.0));
            }
        }
        out
    }

    pub fn applying_all(&self, gestures: &[Gesture], v: Viewport) -> Navigator {
        gestures.iter().fold(*self, |n, g| n.applying(g, v))
    }

    /// The world point under a pixel position at a view depth: the
    /// position's view coordinates, which the framing gives exactly, and the
    /// depth, taken back through the camera.
    pub fn world_point_at_pixel(&self, p: Vec2, depth: f64, v: Viewport) -> P3<WorldSpace> {
        let q = self.framing.view_point(p, v);
        self.camera().view.inverse().apply(P3::new(q.x(), q.y(), depth))
    }

    /// The depth texel a pixel position falls in: the one whose square holds
    /// it, pixel positions running from a texel's corner.
    pub fn texel(p: Vec2) -> [i64; 2] {
        [p.x.floor() as i64, p.y.floor() as i64]
    }

    /// The world point a depth texel holds: its centre, at the depth the
    /// depth pass sampled there. The raster's column 0 is at its left edge
    /// plus half a pixel (`Framing::frame`), so texel i's centre is pixel
    /// position i + 1/2; placed at the clicked position instead, a point
    /// stands up to half a pixel off the surface whose depth it carries, and
    /// at an outline it can land nearer the face behind. What a click reads
    /// off the preview's depth buffer
    /// (`MetalRenderer.worldPoint(atPixel:navigator:viewport:)`).
    pub fn world_point_at_texel(&self, t: [i64; 2], depth: f64, v: Viewport) -> P3<WorldSpace> {
        self.world_point_at_pixel(Vec2::new(t[0] as f64 + 0.5, t[1] as f64 + 0.5), depth, v)
    }
}

#[cfg(test)]
mod tests {
    //! The navigation suites of `Sources/kurven-test/main.swift`, with
    //! Swift's tolerances.

    use super::*;

    /// A deterministic generator. The property tests want the same hundred cases
    /// every run, so a failure is reproducible rather than a story about last
    /// Tuesday. Swift's `SplitMix`, to the bit.
    pub(crate) struct SplitMix {
        state: u64,
    }

    impl SplitMix {
        pub(crate) fn new(seed: u64) -> SplitMix {
            SplitMix { state: seed }
        }
        pub(crate) fn next_bits(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        pub(crate) fn next(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * (self.next_bits() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
        }
    }

    fn projections() -> [(&'static str, ObliqueProjection); 3] {
        [
            ("recip", ObliqueProjection { shear: 0.5, x_angle: -55.0, z_angle: -90.0, flip_x: true, y_scale: None }),
            (
                "elliptic",
                ObliqueProjection { shear: 0.51, x_angle: -63.0, z_angle: -90.0, flip_x: true, y_scale: None },
            ),
            (
                "zeta",
                ObliqueProjection { shear: -0.18, x_angle: -79.5, z_angle: -90.0, flip_x: true, y_scale: Some(0.75) },
            ),
        ]
    }

    fn origin() -> P3<WorldSpace> {
        P3::new(0.0, 0.0, 0.0)
    }

    /// The bridge between "a drawing" and "a thing you can orbit". If this fails,
    /// loading a preset jumps rather than starting where the drawing is.
    #[test]
    fn an_orbit_matching_a_preset_is_the_oblique_camera() {
        let mut rng = SplitMix::new(4);
        for (name, projection) in projections() {
            let a = Camera::oblique(&projection).view;
            let b = Orbit::matching(&projection, origin()).camera().view;
            let mut worst = 0.0f64;
            for _ in 0..200 {
                let p = P3::<WorldSpace>::new(rng.next(-8.0, 8.0), rng.next(-30.0, 30.0), rng.next(0.0, 6.0));
                worst = worst.max((a.apply(p).v - b.apply(p).v).abs().reduce_max());
            }
            assert_eq!(worst, 0.0, "{name}: worst |Δ| = {worst}");
        }
    }

    #[test]
    fn orbiting_and_orbiting_back_is_the_identity() {
        let v = Viewport::new(1200, 800);
        let mut rng = SplitMix::new(9);
        let mut worst = 0.0f64;
        for (_, projection) in projections() {
            let mut n = Navigator::new(Orbit::matching(&projection, origin()), Framing::new(P2::new(0.0, 0.0), 0.01));
            let start = n;
            let mut back = Vec::new();
            for _ in 0..12 {
                // Stay clear of the elevation clamp, which is deliberately not
                // invertible -- past vertical the landscape turns inside out.
                let d = Vec2::new(rng.next(-40.0, 40.0), rng.next(-6.0, 6.0));
                n = n.applying(&Gesture::Orbit(d), v);
                back.push(Gesture::Orbit(-d));
            }
            back.reverse();
            n = n.applying_all(&back, v);
            worst = worst.max((n.orbit.azimuth.degrees() - start.orbit.azimuth.degrees()).abs());
            worst = worst.max((n.orbit.elevation.degrees() - start.orbit.elevation.degrees()).abs());
        }
        assert!(worst < 1e-9, "azimuth and elevation return exactly: worst |Δ| = {worst} degrees");
    }

    #[test]
    fn zoom_holds_the_point_under_the_cursor() {
        let v = Viewport::new(1440, 900);
        let mut rng = SplitMix::new(17);
        let mut worst = 0.0f64;
        for _ in 0..200 {
            let center = P2::new(rng.next(-5.0, 5.0), rng.next(-5.0, 5.0));
            let n = Navigator::new(
                Orbit::matching(&projections()[0].1, origin()),
                Framing::new(center, rng.next(0.001, 0.05)),
            );
            let at = Vec2::new(rng.next(0.0, 1440.0), rng.next(0.0, 900.0));
            let before = n.framing.view_point(at, v);
            let after = n.applying(&Gesture::Zoom { factor: rng.next(0.2, 5.0), at }, v).framing.view_point(at, v);
            worst = worst.max((after.x() - before.x()).abs().max((after.y() - before.y()).abs()));
        }
        assert!(worst < 1e-9, "the anchor does not drift: worst |Δ| = {worst}");
    }

    #[test]
    fn pan_is_invertible_and_zoom_independent() {
        let v = Viewport::new(800, 600);
        let n = Navigator::new(Orbit::matching(&projections()[1].1, origin()), Framing::new(P2::new(1.5, -2.0), 0.02));
        let d = Vec2::new(37.0, -19.0);
        let there = n.applying(&Gesture::Pan(d), v).applying(&Gesture::Pan(-d), v);
        assert_eq!(there.framing, n.framing, "panning back lands where it started");

        // A pixel drag moves the picture by that many pixels whatever the zoom.
        let zoomed = n.applying(&Gesture::Zoom { factor: 4.0, at: Vec2::new(400.0, 300.0) }, v);
        let moved = zoomed.applying(&Gesture::Pan(Vec2::new(100.0, 0.0)), v);
        let dx = (moved.framing.center.x() - zoomed.framing.center.x()) / zoomed.framing.units_per_pixel;
        assert!((dx + 100.0).abs() < 1e-9, "100 px of drag is 100 px of travel: moved {} px", -dx);
    }

    #[test]
    fn pixels_are_square_and_round_trip() {
        let v = Viewport::new(1237, 811);
        let f = Framing::new(P2::new(-3.0, 7.0), 0.013);
        let frame = f.frame(v);
        let upp = frame.units_per_pixel();
        assert!(
            (upp.x - upp.y).abs() < 1e-12 && (upp.x - 0.013).abs() < 1e-12,
            "the raster has square pixels of the size asked for: {} x {}",
            upp.x,
            upp.y
        );
        assert!(
            frame.rows == 811 && frame.cols == 1237 && frame.order == RasterOrder::Screen,
            "one sample per pixel, in screen order"
        );

        let mut worst = 0.0f64;
        let mut rng = SplitMix::new(23);
        for _ in 0..500 {
            let px = Vec2::new(rng.next(0.0, 1237.0), rng.next(0.0, 811.0));
            let back = f.pixel(f.view_point(px, v), v);
            worst = worst.max((back - px).abs().reduce_max());
        }
        assert!(worst < 1e-9, "pixel -> view -> pixel is the identity: worst |Δ| = {worst}");

        // The top-left pixel really is the top left: view y decreasing downward.
        let top_left = f.view_point(Vec2::new(0.0, 0.0), v);
        let bottom_right = f.view_point(Vec2::new(1236.0, 810.0), v);
        assert!(top_left.x() < bottom_right.x() && top_left.y() > bottom_right.y(), "the picture is the right way up");
    }

    #[test]
    fn retargeting_turns_about_a_new_point_without_moving_it() {
        let v = Viewport::new(1000, 700);
        let n = Navigator::new(Orbit::matching(&projections()[2].1, origin()), Framing::new(P2::new(0.4, -1.1), 0.03));
        let p = P3::<WorldSpace>::new(2.5, -7.0, 3.25);
        let before = n.framing.pixel(n.camera().view.apply(p).xy(), v);
        let after = n.applying(&Gesture::Retarget(p), v);
        let moved = after.framing.pixel(after.camera().view.apply(p).xy(), v);
        assert!((moved - before).abs().reduce_max() < 1e-9, "the clicked point stays under the cursor");
        assert_eq!(after.orbit.target, p, "and becomes the centre of rotation");

        // Orbiting now leaves it alone, which is the point of re-targeting.
        let turned = after.applying(&Gesture::Orbit(Vec2::new(120.0, 25.0)), v);
        let still = turned.framing.pixel(turned.camera().view.apply(p).xy(), v);
        let drift = (still - before).abs().reduce_max();
        assert!(drift < 1e-6, "and stays put through a turn: moved {drift} px");
    }

    #[test]
    fn perspective_is_a_way_of_looking_not_a_projection_convention() {
        let v = Viewport::new(1200, 800);
        let projection = projections()[1].1;
        let bounds = AABB::<ViewSpace>::new(Vec3::new(-4.0, -2.0, 0.0), Vec3::new(6.0, 3.0, 4.0));
        let ortho = Navigator::new(Orbit::matching(&projection, origin()), Framing::new(P2::new(0.0, 0.0), 0.01));
        assert!(!ortho.orbit.is_perspective() && !ortho.camera().is_perspective(), "a preset camera is orthographic");

        let persp = ortho.applying(&Gesture::Project { field_of_view: Some(Angle::from_degrees(50.0)), bounds }, v);
        assert!(
            persp.camera().is_perspective() && persp.orbit.distance.is_some(),
            "and .project makes one that is not"
        );

        // The oblique shear is a property of an orthographic drawing. The ADT
        // makes "shear under perspective" unrepresentable; this checks the
        // camera actually takes that route rather than carrying it along.
        assert!(matches!(persp.camera().projection, Projection::Perspective { .. }), "with no oblique shear to carry");

        // What it is looking at ends up in the middle.
        let centre = persp.camera().view.apply(persp.orbit.target);
        assert!(
            centre.x().abs() < 1e-9 && centre.y().abs() < 1e-9,
            "the target lands at the view origin: ({}, {})",
            centre.x(),
            centre.y()
        );
        assert!(centre.z() < 0.0, "with the landscape in front of the eye: z = {}", centre.z());

        // Nearer things are bigger. That is the entire visible difference, and
        // it is what the orthographic camera cannot do.
        let clip = Camera::perspective_clip(persp.orbit.field_of_view, v.aspect());
        let width_on_screen = |z: f64| {
            let h = clip * crate::simd::Vec4::new(1.0, 0.0, z, 1.0);
            (h.x / h.w).abs()
        };
        assert!(width_on_screen(-5.0) > width_on_screen(-50.0), "the same span is wider when it is nearer");

        // And going back to orthographic restores the framing.
        let back = persp.applying(&Gesture::Project { field_of_view: None, bounds }, v);
        assert!(!back.camera().is_perspective(), "and .project(nil) goes back");

        // Zoom means coming closer, not shrinking a rectangle.
        let closer = persp.applying(&Gesture::Zoom { factor: 2.0, at: Vec2::new(600.0, 400.0) }, v);
        assert!(
            closer.orbit.distance.unwrap() < persp.orbit.distance.unwrap() && closer.framing == persp.framing,
            "zooming moves the eye and leaves the framing alone"
        );
    }

    #[test]
    fn fitting_shows_everything_with_a_margin() {
        let v = Viewport::new(900, 600);
        let bounds = AABB::<ViewSpace>::new(Vec3::new(-4.0, -2.0, 0.0), Vec3::new(6.0, 3.0, 1.0));
        let f = Framing::fitting(&bounds, v, Framing::FIT_MARGIN);
        for c in [Vec2::new(-4.0, -2.0), Vec2::new(6.0, -2.0), Vec2::new(-4.0, 3.0), Vec2::new(6.0, 3.0)] {
            let px = f.pixel(P2::new(c.x, c.y), v);
            assert!(
                px.x >= 0.0 && px.x <= 900.0 && px.y >= 0.0 && px.y <= 600.0,
                "every corner of the bounds lands on screen"
            );
        }
        assert!((f.center.x() - 1.0).abs() < 1e-12 && (f.center.y() - 0.5).abs() < 1e-12, "centred on the bounds");
    }
}
