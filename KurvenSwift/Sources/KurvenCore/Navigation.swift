import Foundation
import simd

/// Everything in a plate projection that is not a rotation.
///
/// `PlateProjection` mixes two unrelated things: where the camera is (`zAngle`,
/// `xAngle`) and what kind of drawing this is (the axis convention, the x flip,
/// the y scale, the Jahnke-Emde oblique shear). Only the first pair should move
/// when you orbit. Separating them is what lets a preset be a starting point for
/// navigation rather than a fixed picture.
public struct PlateStyle: Sendable, Equatable {
    public var shear: Double
    public var flipX: Bool
    public var yScale: Double?

    public init(shear: Double, flipX: Bool, yScale: Double?) {
        self.shear = shear; self.flipX = flipX; self.yScale = yScale
    }

    public init(_ plate: PlateProjection) {
        self.init(shear: plate.shear, flipX: plate.flipX, yScale: plate.yScale)
    }

    /// The world-to-pre-rotation matrix: the axis exchange, the flip, the y
    /// scale and the shear, in the order `Projection.apply` performs them.
    ///
    ///     pre = [ 0            fx           0 ]     fx = flipX ? -1 : 1
    ///           [ ys   -shear * fx          0 ]     ys = yScale ?? 1
    ///           [ 0             0           1 ]
    public var matrix: simd_double3x3 {
        let fx = flipX ? -1.0 : 1.0
        let ys = yScale ?? 1.0
        return simd_double3x3(rows: [
            SIMD3(0, fx, 0),
            SIMD3(ys, -shear * fx, 0),
            SIMD3(0, 0, 1),
        ])
    }
}

/// A CAD-style camera: what it looks at, and from where.
///
/// Orbiting a landscape means turning about the world's vertical axis and
/// tilting toward it, which is exactly the `Rz` then `Rx` the plates already
/// apply. So an orbit camera is not a new kind of camera -- it is the plate
/// camera with its two angles made adjustable and a target subtracted first,
/// and `Orbit(matching:)` followed by `.camera` reproduces `Camera.plate`
/// exactly. That identity is what makes "load a preset, then navigate away from
/// it" a continuous motion rather than a jump.
public struct Orbit: Sendable, Equatable {
    /// The world point held at the centre of the turn.
    public var target: P3<WorldSpace>
    /// Rotation about the world vertical.
    public var azimuth: Angle
    /// Tilt toward the viewer.
    public var elevation: Angle
    public var style: PlateStyle

    public init(target: P3<WorldSpace>, azimuth: Angle, elevation: Angle,
                style: PlateStyle) {
        self.target = target; self.azimuth = azimuth
        self.elevation = elevation; self.style = style
    }

    /// The orbit that reproduces a plate preset. With the default target this
    /// gives back exactly `Camera.plate(plate)`.
    public init(matching plate: PlateProjection,
                target: P3<WorldSpace> = P3(0, 0, 0)) {
        self.init(target: target,
                  azimuth: Angle(degrees: plate.zAngle),
                  elevation: Angle(degrees: plate.xAngle),
                  style: PlateStyle(plate))
    }

    public var camera: Camera {
        let m = rotationX(elevation) * rotationZ(azimuth) * style.matrix
        // The target is subtracted in world space, before the plate transform,
        // so it lands on the view origin whatever the style does to the axes.
        let t = m * (-target.v)
        return Camera(
            view: Transform(rows: (m.transpose.columns.0,
                                   m.transpose.columns.1,
                                   m.transpose.columns.2),
                            translation: t),
            projection: .orthographic(oblique: Oblique(shear: style.shear)))
    }

    /// Elevation is clamped rather than wrapped: past vertical the landscape
    /// turns inside out and the horizon flips, which reads as the input having
    /// broken rather than as a viewpoint anyone wanted.
    public static let elevationLimit = Angle(degrees: 89.5)
}

/// The pixel raster a preview draws into.
public struct Viewport: Sendable, Equatable {
    public var width: Int
    public var height: Int
    public init(width: Int, height: Int) {
        self.width = max(width, 1); self.height = max(height, 1)
    }
    public var aspect: Double { Double(width) / Double(height) }
}

/// Where the viewport sits in view space, and how much of it it shows.
///
/// Pan and zoom live here rather than in the camera. A camera says which way the
/// landscape faces; sliding the paper around under it and holding a magnifier
/// over it are operations on the paper. Keeping them apart means the orbit is
/// three numbers that mean what they say, and it means zoom-toward-the-cursor is
/// a two-line calculation on a rectangle instead of a correction applied to a
/// matrix.
public struct Framing: Sendable, Equatable {
    /// The view point at the centre of the viewport.
    public var center: P2<ViewSpace>
    /// View-space units per pixel -- one number, so pixels are square.
    public var unitsPerPixel: Double

    public init(center: P2<ViewSpace>, unitsPerPixel: Double) {
        self.center = center
        self.unitsPerPixel = max(unitsPerPixel, .leastNormalMagnitude)
    }

    /// The view point under a pixel. Pixel y is measured from the top, view y
    /// grows upward, so one of them is negated exactly once, here.
    public func viewPoint(atPixel p: SIMD2<Double>, in v: Viewport) -> P2<ViewSpace> {
        P2(center.x + (p.x - Double(v.width) / 2) * unitsPerPixel,
           center.y - (p.y - Double(v.height) / 2) * unitsPerPixel)
    }

    public func pixel(of p: P2<ViewSpace>, in v: Viewport) -> SIMD2<Double> {
        SIMD2((p.x - center.x) / unitsPerPixel + Double(v.width) / 2,
              (center.y - p.y) / unitsPerPixel + Double(v.height) / 2)
    }

    /// The raster this framing draws into: screen order, square pixels, one
    /// sample per pixel.
    public func frame(_ v: Viewport) -> DepthFrame {
        let halfX = unitsPerPixel * Double(v.width - 1) / 2
        let halfY = unitsPerPixel * Double(v.height - 1) / 2
        return DepthFrame(
            // Row 0 is the top of the picture, so axis0 descends.
            axis0: Interval(lo: center.y + halfY, hi: center.y - halfY),
            axis1: Interval(lo: center.x - halfX, hi: center.x + halfX),
            rows: v.height, cols: v.width, order: .screen)
    }

    /// The framing that shows all of `bounds`, with a margin.
    public static func fitting(_ bounds: AABB<ViewSpace>, in v: Viewport,
                               margin: Double = 0.04) -> Framing {
        let size = bounds.size
        let scale = 1 + 2 * margin
        let upp = max(size.x * scale / Double(v.width),
                      size.y * scale / Double(v.height),
                      .leastNormalMagnitude)
        return Framing(center: P2(bounds.center.x, bounds.center.y), unitsPerPixel: upp)
    }

    /// Zoom, keeping the view point under `pixel` where it is. Under an
    /// orthographic camera that is the whole of "zoom toward the cursor"; there
    /// is no dolly and nothing to correct afterwards.
    public func zoomed(by factor: Double, about pixel: SIMD2<Double>,
                       in v: Viewport) -> Framing {
        let anchor = viewPoint(atPixel: pixel, in: v)
        let upp = unitsPerPixel / max(factor, .leastNormalMagnitude)
        return Framing(
            center: P2(anchor.x - (pixel.x - Double(v.width) / 2) * upp,
                       anchor.y + (pixel.y - Double(v.height) / 2) * upp),
            unitsPerPixel: upp)
    }

    /// Drag the paper by a pixel delta.
    public func panned(byPixels d: SIMD2<Double>) -> Framing {
        Framing(center: P2(center.x - d.x * unitsPerPixel,
                           center.y + d.y * unitsPerPixel),
                unitsPerPixel: unitsPerPixel)
    }
}

/// What the input handling produces. Not what it does.
///
/// A `Gesture` is a value; `Navigator.applying` folds it. Input handling becomes
/// "turn an NSEvent into one of these", navigation becomes a pure function, and
/// undo is a list of `Navigator`s.
public enum Gesture: Sendable, Equatable {
    /// Drag to turn, in pixels.
    case orbit(SIMD2<Double>)
    /// Drag to slide the paper, in pixels.
    case pan(SIMD2<Double>)
    /// Scroll or pinch, anchored at a pixel.
    case zoom(factor: Double, at: SIMD2<Double>)
    /// Trackpad rotation, in radians.
    case roll(Double)
    /// Look at a world point, keeping its position on screen.
    case retarget(P3<WorldSpace>)
    /// Jump to a preset, refit to the scene.
    case preset(PlateProjection, AABB<ViewSpace>)
    /// Show all of it.
    case fit(AABB<ViewSpace>)
}

/// The camera state a preview draws from: an orbit and a framing.
public struct Navigator: Sendable, Equatable {
    public var orbit: Orbit
    public var framing: Framing
    /// Degrees of turn per pixel dragged.
    public var sensitivity: Double

    public init(orbit: Orbit, framing: Framing, sensitivity: Double = 0.35) {
        self.orbit = orbit; self.framing = framing; self.sensitivity = sensitivity
    }

    public var camera: Camera { orbit.camera }

    /// Fold a gesture. Pure: the same navigator and gesture always give the same
    /// result, which is why the property tests below are worth anything.
    public func applying(_ g: Gesture, in v: Viewport) -> Navigator {
        var out = self
        switch g {
        case .orbit(let d):
            out.orbit.azimuth = Angle(degrees: orbit.azimuth.degrees - d.x * sensitivity)
            let tilt = orbit.elevation.degrees + d.y * sensitivity
            out.orbit.elevation = Angle(
                degrees: min(max(tilt, -Orbit.elevationLimit.degrees),
                             Orbit.elevationLimit.degrees))
        case .pan(let d):
            out.framing = framing.panned(byPixels: d)
        case .zoom(let factor, let at):
            out.framing = framing.zoomed(by: factor, about: at, in: v)
        case .roll(let radians):
            out.orbit.azimuth = Angle(radians: orbit.azimuth.radians + radians)
        case .retarget(let p):
            // Keep the point where it is on screen: re-targeting moves the
            // centre of rotation, not the picture. Anything else makes a
            // double-click jump, which reads as a misclick.
            // Moving the target moves the view origin, so the same world point
            // lands somewhere else in view space; shifting the framing centre by
            // that difference cancels it exactly and the point stays under the
            // cursor. Anything else makes a double-click jump, which reads as a
            // misclick.
            let before = camera.view(p).xy
            out.orbit.target = p
            let after = out.camera.view(p).xy
            out.framing = Framing(
                center: P2(framing.center.x + after.x - before.x,
                           framing.center.y + after.y - before.y),
                unitsPerPixel: framing.unitsPerPixel)
        case .preset(let plate, let bounds):
            out.orbit = Orbit(matching: plate, target: orbit.target)
            out.framing = .fitting(bounds, in: v)
        case .fit(let bounds):
            out.framing = .fitting(bounds, in: v)
        }
        return out
    }

    public func applying(_ gestures: [Gesture], in v: Viewport) -> Navigator {
        gestures.reduce(self) { $0.applying($1, in: v) }
    }
}
