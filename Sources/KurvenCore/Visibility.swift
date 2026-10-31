import Foundation
import simd

/// Hidden lines for curves drawn on a parametric surface, decided by *where on
/// the surface* the curves are rather than by how deep it is.
///
/// Two tests, and each covers the other's blind spot:
///
/// - **Facing.** On a surface that bounds a solid, curves whose outward normal
///   faces away from the eye is hidden, full stop. This is the test the
///   second cannot make: across a fold the front and the back of the surface
///   are neighbours in coordinates, so "front-most is my neighbourhood" holds
///   on both sides of it.
/// - **Nothing in front.** With the solid as a point test (`TubularNeighbourhood`), the
///   line of sight from the curves are sphere-traced against it, stepping by the gap
///   over its Lipschitz constant so that no step can pass through the
///   solid: exact, with no pixel in it. Without one, the surface in front of
///   the curve, at the curve's pixel or one beside it, must be the curve's own
///   neighbourhood of the surface. A second sheet in front is far away in
///   coordinates however close it is in depth, which is where a depth depth tolerance
///   fails; but a raster is right only to its resolution, and relief puts
///   the curves where that runs out.
///
/// Where visibility changes along a segment -- at a fold, or where the
/// segment passes behind something -- the segment is cut exactly there, by
/// bisection on the exact test, and the pixel grid has no say in where.
public struct SurfaceVisibility: Sendable {
    public let surface: ParametricSurface
    /// The coordinate image, which the front-most test reads; nil when the
    /// sphere tracing tests alone, since it asks nothing of a raster.
    public let image: SurfaceImage?
    public let view: Transform<WorldSpace, ViewSpace>
    /// The direction lines of sight travel, in world space.
    public let sight: SIMD3<Double>

    public init(surface: ParametricSurface, image: SurfaceImage,
                view: Transform<WorldSpace, ViewSpace>) {
        self.surface = surface; self.image = image; self.view = view
        self.sight = view.projectionDirection
    }

    /// Visibility by sphere tracing alone, with no image: for a surface with an
    /// `TubularNeighbourhood`, tested anywhere without rendering anything first.
    public init(sphereTracing surface: ParametricSurface, view: Transform<WorldSpace, ViewSpace>) {
        precondition(surface.neighbourhood != nil, "only a surface with a tubular neighbourhood is tested by sphere tracing alone")
        self.surface = surface; self.image = nil; self.view = view
        self.sight = view.projectionDirection
    }

    /// Whether the curves are tested by sphere tracing against the solid, or by the
    /// coordinate image.
    public var tracesSpheres: Bool { surface.neighbourhood != nil }

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
        guard let image, let here = image.coordinate(row: row, col: col) else { return .zero }
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
    /// The curves are within about a pixel of the nearest pixel's sample point,
    /// so on its own sheet the coordinates there differ from `c` by at most
    /// about one footprint. Two footprints, plus one grid cell for the
    /// sheet's own piecewise-linear rasterization, is the tolerance; another
    /// sheet is not within it except where the two meet, which is a fold and
    /// `facing` decides it. Nothing drawn in all nine pixels: nothing in
    /// front. Outside the frame: occluded, as `DepthImage.depth(under:)` has
    /// it.
    public func isFrontMost(_ c: P2<ParamSpace>, at p: P2<ViewSpace>) -> Bool {
        guard let image else { preconditionFailure("the front-most test needs the coordinate image") }
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

    /// Whether nothing of the solid stands between the surface point `q` and
    /// the eye: sphere tracing.
    ///
    /// Up the line of sight from `q`, stepping by the gap over the solid's
    /// Lipschitz constant, which cannot reach the solid, until the line has
    /// left the ball that holds it. Above the shell the gap is at least the
    /// height over it, and that is the step, with no normal offset asked. The
    /// first step is a small fraction of the solid's size: the curves lie on
    /// the surface to rounding, where the gap is zero and says nothing, and
    /// from a point facing the eye the line is outside from there on. On a
    /// fold the line is tangent, and the gap grows or shrinks with the
    /// surface's curvature along it: a convex fold is seen, a concave one
    /// lies behind the surface it folds into, and the first step decides
    /// which. A step shorter than that fraction is never taken, so an
    /// occluder thinner than it along the line can be stepped over; it is a
    /// hundred-thousandth of the solid, under any pixel.
    public func isClear(_ q: SIMD3<Double>) -> Bool {
        guard let e = surface.neighbourhood else { return true }
        let d = -sight
        let o = q - e.centre
        let b = simd_dot(o, d), c = simd_dot(o, o) - e.radius * e.radius
        let disc = b * b - c
        guard disc > 0 else { return true }
        let exit = -b + disc.squareRoot()
        let eps = 1e-9 * e.radius, least = 1e-5 * e.radius
        var t = least
        while t < exit {
            let p = q + t * d
            let (coord, distance) = e.foot(p)
            let over = distance - e.above
            if over > least { t += over; continue }
            if distance < -e.below - eps { return false }
            let gap = distance - e.normalOffset(coord)
            if gap < -eps { return false }
            t += max(gap / e.lipschitz, least)
        }
        return true
    }

    /// Curves at view point `p`, lying on the surface at `c`.
    public func isVisible(_ p: P3<ViewSpace>, at c: P2<ParamSpace>) -> Bool {
        isVisible(p, at: c, facing: facing(c))
    }

    func isVisible(_ p: P3<ViewSpace>, at c: P2<ParamSpace>, facing f: Double?) -> Bool {
        if let f, f <= 0 { return false }
        if tracesSpheres { return isClear(surface.map(c).position) }
        return isFrontMost(c, at: p.xy)
    }

    /// Curves at surface coordinate `c`, placed by the map: whether it is seen,
    /// and where it is in the view. `facing` nil asks only sphere tracing, for
    /// curves on the folds, where the facing is zero by construction.
    func test(_ c: P2<ParamSpace>, onFolds: Bool) -> (seen: Bool, at: P3<ViewSpace>) {
        let world = surface.map(c).position
        let p = view(P3<WorldSpace>(world))
        if !onFolds, let f = facing(c), f <= 0 { return (false, p) }
        return (tracesSpheres ? isClear(world) : isFrontMost(c, at: p.xy), p)
    }

    /// The point of a crease beside `c`, a point of the chord between two of
    /// its vertices: where the facing jumps, across the chord within half
    /// its length either way; `c` where no jump is in reach. The vertices
    /// are on the crease but the chord between them crosses a face, which
    /// can face away where the crease is seen: a convex truncation edge is seen where
    /// either of its faces is.
    public func ontoCrease(_ c: P2<ParamSpace>, chord: SIMD2<Double>) -> P2<ParamSpace> {
        let scale = SIMD2(surface.u.spacing, surface.v.spacing)
        let along = chord / scale
        let cells = simd_length(along)
        guard cells > 0 else { return c }
        let n = SIMD2(-along.y, along.x) / cells * scale
        var a = c.v - 0.5 * cells * n, b = c.v + 0.5 * cells * n
        guard let fa = facing(P2(a)), let fb = facing(P2(b)) else { return c }
        for _ in 0..<60 {
            let m = 0.5 * (a + b)
            guard let fm = facing(P2(m)) else { return c }
            if abs(fm - fa) < abs(fm - fb) { a = m } else { b = m }
        }
        // A jump, not a slope: a hundredth of the span holds most of the change.
        let m = 0.5 * (a + b)
        func across(_ t: Double) -> Double {
            abs((facing(P2(m + t * n)) ?? 0) - (facing(P2(m - t * n)) ?? 0))
        }
        return across(1e-3) > 0.5 * across(1e-1) ? P2(m) : c
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
    /// Hidden-line removal for curves that lie on a parametric surface and
    /// carries its coordinates on it.
    ///
    /// The runs are `clip`'s -- hidden vertices split a path, and runs of
    /// fewer than two vertices draw nothing -- with one difference: a segment
    /// along which the verdict changes gains a vertex where it changes,
    /// which belongs to the run on its visible side. Visible by sphere tracing,
    /// that vertex is found by bisection on the whole test, so it lies on
    /// the fold or on the outline of whatever came in front, whichever
    /// cut the curves. Visible by the coordinate image, a segment that crosses a
    /// fold is cut at the fold, found on the exact normal, and that vertex is
    /// tested by the front-most test alone, since on the fold the facing is
    /// zero by construction.
    ///
    /// Curves that lie on the folds themselves -- `ParametricSurface.foldCurves`
    /// -- is edge-on everywhere by construction, so its facing is noise about
    /// zero and must not be asked: pass `onFolds` and it is tested by the
    /// sphere tracing or the front-most test alone. Curves on a crease -- a truncation edge --
    /// is tested so too, and `onCrease` says so: there a cut is bisected on
    /// the crease itself, each point moved onto it from the chord.
    ///
    /// Every vertex is tested in parallel, and so is every cut: sphere tracing is
    /// hundreds of evaluations of the normal offset, and a cut is forty sphere-traces.
    static func clip(_ paths: PolylineSet<ViewSpace>, on visibility: SurfaceVisibility,
                     onFolds: Bool = false, onCrease: Bool = false) -> PolylineSet<PageSpace> {
        guard let coords = paths.coords else {
            preconditionFailure("clipping on a surface needs the curves' surface coordinates")
        }
        let n = coords.count
        let facing: [Double?] = onFolds ? [Double?](repeating: nil, count: n)
                                       : Heightfield.parallelMap(coords) { visibility.facing($0) }
        let kept: [Bool] = Heightfield.parallelMap(Array(0..<n), chunk: 32) { k in
            visibility.isVisible(paths.vertices[k], at: coords[k], facing: facing[k])
        }
        // Every segment along which the verdict changes, and the vertex
        // that goes in at the change. By sphere tracing, bisection on the test
        // itself; by the image, the fold crossing if there is one.
        var changes: [Int] = []
        for path in 0..<paths.count {
            let lo = paths.offsets[path], hi = paths.offsets[path + 1]
            for k in (lo + 1)..<hi where kept[k - 1] != kept[k]
                || (!visibility.tracesSpheres && facing[k - 1] != nil && facing[k] != nil
                    && (facing[k - 1]! > 0) != (facing[k]! > 0)) {
                changes.append(k)
            }
        }
        let cuts: [(at: P3<ViewSpace>, seen: Bool)?] =
            Heightfield.parallelMap(changes, chunk: 1) { k -> (at: P3<ViewSpace>, seen: Bool)? in
                let a = coords[k - 1], b = coords[k]
                if visibility.tracesSpheres {
                    guard kept[k - 1] != kept[k] else { return nil }
                    var lo = 0.0, hi = 1.0
                    var at = kept[k - 1] ? paths.vertices[k - 1] : paths.vertices[k]
                    while hi - lo > 1e-13 {
                        let mid = 0.5 * (lo + hi)
                        let c = P2<ParamSpace>(a.v + (b.v - a.v) * mid)
                        let j = visibility.test(onCrease ? visibility.ontoCrease(c, chord: b.v - a.v) : c,
                                                onFolds: onFolds)
                        if j.seen == kept[k - 1] { lo = mid } else { hi = mid }
                        if j.seen { at = j.at }
                    }
                    return (at, true)
                }
                guard let fa = facing[k - 1], let fb = facing[k],
                      let t = visibility.foldCrossing(from: a, fa, to: b, fb) else { return nil }
                let pa = paths.vertices[k - 1], pb = paths.vertices[k]
                let v = P3<ViewSpace>(pa.v + (pb.v - pa.v) * t)
                let c = P2<ParamSpace>(a.v + (b.v - a.v) * t)
                return (v, visibility.isFrontMost(c, at: v.xy))
            }
        var cut = [(at: P3<ViewSpace>, seen: Bool)?](repeating: nil, count: n)
        for (i, k) in changes.enumerated() { cut[k] = cuts[i] }

        var out: [[P3<PageSpace>]] = []
        var run: [P3<PageSpace>] = []
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
                if let c = cut[k] { take(c.at, c.seen) }
                take(paths.vertices[k], kept[k])
            }
            if run.count >= 2 { out.append(run) }
            run = []
        }
        return PolylineSet(paths: out)
    }
}
