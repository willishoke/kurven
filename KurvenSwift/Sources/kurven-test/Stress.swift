import Foundation
import simd
import KurvenCore
import KurvenMetal
import KurvenBake
import KurvenLandscape

// MARK: - visibility under stress
//
// The principle these suites hold the hidden-line rules to: visibility is a
// function of the geometry alone, never of the output resolution. A rule
// that consults a pixel fails this by construction -- a pixel's depth span
// on a steep flank or an edge-on wall exceeds any margin, and the span
// changes with the resolution -- and the rules here are built so that no
// pixel is consulted: the facing of the surface the ink lies on, and the
// march up the sight line through the drawn solid (`SightMarch`), which is
// exact because the drawn solid is piecewise linear. So the suites ask the
// strongest questions such a rule can be asked: the same bake at every
// depth resolution, the same drawing scaled when the plate is scaled, and
// on a lattice too coarse for any approximation to hide in, agreement with
// a brute-force ray cast against the triangles the depth pass draws.

/// The ray from `origin` along `direction` against every triangle of
/// `mesh`: the parameter of the first hit past `after`, or nil. Brute
/// force, as an oracle should be: nothing about the heightfield's structure
/// is used, only its triangles.
func firstHit(_ mesh: Mesh<WorldSpace>, from origin: SIMD3<Double>,
              along direction: SIMD3<Double>, after: Double) -> Double? {
    var best: Double?
    for t in mesh.triangles {
        let a = mesh.vertices[Int(t.x)].v, b = mesh.vertices[Int(t.y)].v, c = mesh.vertices[Int(t.z)].v
        // Möller–Trumbore.
        let e1 = b - a, e2 = c - a
        let p = simd_cross(direction, e2)
        let det = simd_dot(e1, p)
        guard abs(det) > 1e-14 else { continue }
        let inv = 1 / det
        let s = origin - a
        let u = simd_dot(s, p) * inv
        guard u >= -1e-12, u <= 1 + 1e-12 else { continue }
        let q = simd_cross(s, e1)
        let v = simd_dot(direction, q) * inv
        guard v >= -1e-12, u + v <= 1 + 1e-12 else { continue }
        let hit = simd_dot(e2, q) * inv
        guard hit > after else { continue }
        if best == nil || hit < best! { best = hit }
    }
    return best
}

/// The cameras the stress tests orbit through: from nearly overhead to a
/// degree off edge-on, round the compass, some sheared as the plates are.
func stressCameras() -> [(name: String, view: Transform<WorldSpace, ViewSpace>)] {
    var out: [(String, Transform<WorldSpace, ViewSpace>)] = []
    for elevation in [-80.0, -55, -30, -10, -3, -1] {
        for azimuth in [0.0, 37, 90, 135, 200, 289] {
            out.append(("el \(Int(-elevation))° az \(Int(azimuth))°",
                        testCamera(elevation: elevation, azimuth: azimuth)))
        }
    }
    out.append(("el 55° sheared", testCamera(elevation: -55, azimuth: -20, shear: 0.5)))
    out.append(("el 5° sheared", testCamera(elevation: -5, azimuth: 70, shear: 0.5)))
    return out
}
