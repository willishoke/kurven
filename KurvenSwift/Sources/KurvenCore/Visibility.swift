import Foundation
import simd

/// Hidden lines for ink drawn on a parametric surface, decided by *where on
/// the surface* the ink is rather than by how deep it is.
///
/// Two tests, and each covers the other's blind spot:
///
/// - **Front-most.** The surface in front of the ink, at the ink's pixel or
///   one beside it, is the ink's own neighbourhood of the surface. A second
///   sheet in front is far away in coordinates however close it is in depth,
///   which is where a depth margin fails.
/// - **Facing.** On a surface that bounds a solid, ink whose outward normal
///   faces away from the eye is hidden, full stop. This is the test the first
///   cannot make: across a fold the front and the back of the surface are
///   neighbours in coordinates, so "front-most is my neighbourhood" holds on
///   both sides of it.
///
/// Where a segment crosses a fold, it is cut at the fold itself, found by root
/// finding on the exact normal. Visibility changes exactly there, and the
/// pixel grid has no say in where.
public struct SurfaceVisibility: Sendable {
    public let surface: ParametricSurface
    public let image: SurfaceImage
    /// The direction sight lines travel, in world space.
    public let sight: SIMD3<Double>

    public init(surface: ParametricSurface, image: SurfaceImage,
                view: Transform<WorldSpace, ViewSpace>) {
        self.surface = surface; self.image = image; self.sight = view.sightLine
    }

    /// Positive facing the eye, negative facing away, nil on a surface that
    /// bounds no solid.
    public func facing(_ c: P2<ParamSpace>) -> Double? {
        surface.facing(c, sight: sight)
    }

    /// How far the front-most coordinate moves per pixel at `(row, col)`, per
    /// axis of the parameter rectangle.
    ///
    /// Per screen axis, the smaller of the two one-sided differences, and the
    /// larger over the two screen axes -- the same shape as the preview's
    /// `depth_slope`, for the same reason: a pixel beside an occlusion edge
    /// has one neighbour on another sheet, and counting that jump as the
    /// surface's own rate would make the tolerance swallow the other sheet.
    public func footprint(row: Int, col: Int) -> SIMD2<Double> {
        guard let here = image.coordinate(row: row, col: col) else { return .zero }
        let frame = image.frame
        func rate(_ dr: Int, _ dc: Int) -> SIMD2<Double> {
            var best: SIMD2<Double>?
            for s in [-1, 1] {
                let r = row + s * dr, c = col + s * dc
                guard r >= 0, r < frame.rows, c >= 0, c < frame.cols,
                      let there = image.coordinate(row: r, col: c) else { continue }
                let d = SIMD2(abs(surface.u.difference(there.x, here.x)),
                              abs(surface.v.difference(there.y, here.y)))
                best = best.map { simd_min($0, d) } ?? d
            }
            return best ?? .zero
        }
        return simd_max(rate(1, 0), rate(0, 1))
    }

    /// Whether the front-most surface beside view point `p` is the surface at
    /// `c`.
    ///
    /// The ink is within about a pixel of the nearest pixel's sample point,
    /// so on its own sheet the coordinates there differ from `c` by at most
    /// about one footprint. Two footprints, plus one lattice cell for the
    /// sheet's own piecewise-linear rasterization, is the tolerance; another
    /// sheet is not within it except where the two meet, which is a fold and
    /// `facing` decides it. Nothing drawn in all nine pixels: nothing in
    /// front. Outside the frame: occluded, as `DepthImage.depth(under:)` has
    /// it.
    public func isFrontMost(_ c: P2<ParamSpace>, at p: P2<ViewSpace>) -> Bool {
        let frame = image.frame
        let i = frame.index(of: p)
        guard i.x >= 0, i.x < frame.rows, i.y >= 0, i.y < frame.cols else { return false }
        var covered = false
        for dr in -1...1 {
            for dc in -1...1 {
                let r = i.x + dr, col = i.y + dc
                guard r >= 0, r < frame.rows, col >= 0, col < frame.cols,
                      let q = image.coordinate(row: r, col: col) else { continue }
                covered = true
                let f = footprint(row: r, col: col)
                if abs(surface.u.difference(q.x, c.x)) <= 2 * f.x + surface.u.spacing,
                   abs(surface.v.difference(q.y, c.y)) <= 2 * f.y + surface.v.spacing {
                    return true
                }
            }
        }
        return !covered
    }

    /// Ink at view point `p`, lying on the surface at `c`.
    public func isVisible(_ p: P3<ViewSpace>, at c: P2<ParamSpace>) -> Bool {
        isVisible(p, at: c, facing: facing(c))
    }

    func isVisible(_ p: P3<ViewSpace>, at c: P2<ParamSpace>, facing f: Double?) -> Bool {
        if let f, f <= 0 { return false }
        return isFrontMost(c, at: p.xy)
    }

    /// Where along the straight segment `a -> b` in coordinates the surface
    /// turns edge-on, as a fraction of the way; nil when both ends face the
    /// same way, or the surface has no outside.
    ///
    /// Bisection on the exact normal, to the last bit of the fraction: the
    /// facing is continuous along the segment and its sign is all that is
    /// asked of it, so nothing faster is also as certain.
    public func foldCrossing(from a: P2<ParamSpace>, to b: P2<ParamSpace>) -> Double? {
        guard let fa = facing(a), let fb = facing(b) else { return nil }
        return foldCrossing(from: a, fa, to: b, fb)
    }

    func foldCrossing(from a: P2<ParamSpace>, _ fa: Double,
                      to b: P2<ParamSpace>, _ fb: Double) -> Double? {
        guard (fa > 0) != (fb > 0) else { return nil }
        var lo = 0.0, hi = 1.0
        let frontAtLo = fa > 0
        while hi - lo > 1e-13 {
            let mid = 0.5 * (lo + hi)
            let f = facing(P2(a.v + (b.v - a.v) * mid)) ?? 0
            if (f > 0) == frontAtLo { lo = mid } else { hi = mid }
        }
        return 0.5 * (lo + hi)
    }
}

public extension HiddenLine {
    /// Hidden-line removal for ink that lies on a parametric surface and
    /// carries its coordinates on it.
    ///
    /// The runs are `clip`'s -- hidden vertices split a path, and runs of
    /// fewer than two vertices draw nothing -- with one difference: a segment
    /// that crosses a fold gains a vertex exactly on the fold, which belongs
    /// to the run on its visible side. That vertex is judged by the front-most
    /// test alone, since on the fold the facing is zero by construction.
    static func clip(_ paths: PolylineSet<ViewSpace>,
                     on visibility: SurfaceVisibility) -> PolylineSet<PlateSpace> {
        guard let coords = paths.coords else {
            preconditionFailure("clipping on a surface needs the ink's surface coordinates")
        }
        let facing = coords.map { visibility.facing($0) }
        var out: [[P3<PlateSpace>]] = []
        var run: [P3<PlateSpace>] = []
        func take(_ v: P3<ViewSpace>, _ visible: Bool) {
            if visible {
                run.append(P3(v.v))
            } else {
                if run.count >= 2 { out.append(run) }
                run = []
            }
        }
        for path in 0..<paths.count {
            let lo = paths.offsets[path], hi = paths.offsets[path + 1]
            for k in lo..<hi {
                if k > lo, let fa = facing[k - 1], let fb = facing[k],
                   let t = visibility.foldCrossing(from: coords[k - 1], fa, to: coords[k], fb) {
                    let a = paths.vertices[k - 1], b = paths.vertices[k]
                    let v = P3<ViewSpace>(a.v + (b.v - a.v) * t)
                    let c = P2<ParamSpace>(coords[k - 1].v + (coords[k].v - coords[k - 1].v) * t)
                    take(v, visibility.isFrontMost(c, at: v.xy))
                }
                take(paths.vertices[k],
                     visibility.isVisible(paths.vertices[k], at: coords[k], facing: facing[k]))
            }
            if run.count >= 2 { out.append(run) }
            run = []
        }
        return PolylineSet(paths: out)
    }
}
