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
    /// Folding the exchange in here is what keeps it out of everywhere else:
    /// the arrays a bundle carries are already world order, and no stage between
    /// the decoder and the plate touches a column index again.
    public static func plate(_ p: PlateProjection) -> Camera {
        let fx = p.flipX ? -1.0 : 1.0
        let ys = p.yScale ?? 1.0
        let pre = simd_double3x3(rows: [
            SIMD3(0, fx, 0),
            SIMD3(ys, -p.shear * fx, 0),
            SIMD3(0, 0, 1),
        ])
        let rot = rotationX(Angle(degrees: p.xAngle)) * rotationZ(Angle(degrees: p.zAngle))
        let m = rot * pre
        return Camera(
            view: Transform(rows: (m.transpose.columns.0,
                                   m.transpose.columns.1,
                                   m.transpose.columns.2)),
            projection: .orthographic(oblique: Oblique(shear: p.shear)))
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
