import Foundation
import simd

/// How view space becomes the plate.
///
/// A sum type rather than a struct with optional fields: the Jahnke-Emde
/// oblique shear is a property of an orthographic drawing and means nothing
/// under perspective, so "shear under perspective" is unrepresentable instead
/// of a runtime check nobody remembers to write.
public enum Projection: Sendable, Equatable {
    case orthographic(oblique: Oblique?)
    case perspective(fovY: Angle)
}

/// The foreshortening shear the plates use: in the pre-rotation frame,
/// `along -= shear * across`.
public struct Oblique: Sendable, Equatable {
    public var shear: Double
    public init(shear: Double) { self.shear = shear }
}

/// A camera: one world-to-view transform and how that view is projected.
///
/// `view` is *derived*, never edited. Changing where you are looking from means
/// producing a new `Camera`; the renderer's only per-frame work is uploading
/// this matrix.
public struct Camera: Sendable, Equatable {
    public var view: Transform<WorldSpace, ViewSpace>
    public var projection: Projection

    public init(view: Transform<WorldSpace, ViewSpace>, projection: Projection) {
        self.view = view; self.projection = projection
    }

    /// The exact Python plate camera, in world coordinates.
    ///
    /// `Projection.apply` consumes `(imag, real, z)` and, in order: scales
    /// column 1 by `yScale`, negates column 0 when `flipX`, subtracts
    /// `shear * column0` from column 1, then applies Rz followed by Rx. In world
    /// order column 0 is `y` and column 1 is `x`, so the whole of it -- axis
    /// exchange included -- is one 3x3:
    ///
    ///     pre = [ 0            fx           0 ]     fx = flipX ? -1 : 1
    ///           [ ys   -shear * fx          0 ]     ys = yScale ?? 1
    ///           [ 0             0           1 ]
    ///
    ///     view = Rx(xAngle) * Rz(zAngle) * pre
    ///
    /// The three factors are `Orbit`'s elevation, azimuth and `PlateStyle`, so
    /// this is `Orbit(matching:).camera` and a preset is a starting point for
    /// navigation rather than a fixed picture.
    ///
    /// Folding the exchange in here is what keeps it out of everywhere else:
    /// the arrays a bundle carries are already world order, and no stage between
    /// the decoder and the plate touches a column index again.
    public static func plate(_ p: PlateProjection) -> Camera {
        Orbit(matching: p).camera
    }

    /// The plate's 2D coordinates of a world point: the first two view
    /// components under orthographic projection.
    public func plateCoordinates(of p: P3<WorldSpace>) -> P2<PlateSpace> {
        let v = view(p)
        switch projection {
        case .orthographic:
            return P2(v.x, v.y)
        case .perspective(let fovY):
            // Not reachable from a bundle preset; kept total so a future
            // navigation camera cannot silently fall through to orthographic.
            let f = 1 / tan(fovY.radians / 2)
            let d = max(v.z, .leastNormalMagnitude)
            return P2(v.x * f / d, v.y * f / d)
        }
    }
}

/// Active rotation about x, matching `scipy.spatial.transform.Rotation.from_euler("x", a)`.
public func rotationX(_ a: Angle) -> simd_double3x3 {
    let c = cos(a.radians), s = sin(a.radians)
    return simd_double3x3(rows: [SIMD3(1, 0, 0), SIMD3(0, c, -s), SIMD3(0, s, c)])
}

/// Active rotation about z, matching `Rotation.from_euler("z", a)`.
public func rotationZ(_ a: Angle) -> simd_double3x3 {
    let c = cos(a.radians), s = sin(a.radians)
    return simd_double3x3(rows: [SIMD3(c, -s, 0), SIMD3(s, c, 0), SIMD3(0, 0, 1)])
}


// MARK: - perspective

public extension Camera {
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
    /// attachment.
    static func perspectiveClip(fovY: Angle, aspect: Double,
                                near: Double = 0.01) -> simd_double4x4 {
        let f: Double = 1 / tan(fovY.radians / 2)
        let zero: Double = 0
        // Columns, not rows: see `DepthFrame.metalClip`.
        let col0 = SIMD4<Double>(f / max(aspect, .leastNormalMagnitude), zero, zero, zero)
        let col1 = SIMD4<Double>(zero, f, zero, zero)
        let col2 = SIMD4<Double>(zero, zero, zero, -1)    // w = -z
        let col3 = SIMD4<Double>(zero, zero, 0.5, zero)   // constant depth
        return simd_double4x4(col0, col1, col2, col3)
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
    var isPerspective: Bool {
        if case .perspective = projection { return true }
        return false
    }
}
