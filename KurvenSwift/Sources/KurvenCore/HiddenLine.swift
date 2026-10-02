import Foundation
import simd

/// Hidden-line removal: a pure function of a depth image and some polylines.
///
/// A line-for-line port of `outline.clip_hidden_lines`. A vertex is visible when
/// `z + margin > buffer[pixel]`; hidden vertices split a path into runs; runs
/// shorter than two vertices draw nothing and are dropped. Because it takes a
/// `DepthImage` *value*, it is tested against a Python-dumped buffer and
/// Python-dumped polylines with no GPU anywhere in the loop -- which is what
/// makes "the bake reproduces the plate" a checkable claim rather than a
/// rendering that looks about right.
///
/// The realtime preview does the same test per *fragment* in a shader, which can
/// disagree with this on sub-pixel runs. That disagreement is deliberate and
/// bounded: the preview is for navigating, the bake is the artifact, and the
/// bake is this.
/// The two questions asked of ink on a heightfield: does the surface face
/// the eye where the ink lies, and is anything in front of it.
///
/// The second is the depth test with its margin, as ever. The first is what
/// `SurfaceVisibility` asks of a parametric surface, and a heightfield can
/// answer it too, since it bounds the solid under it: the normal is the
/// gradient, read off the grid. Without it, where a steep flank turns away
/// from the eye its back lies within the margin of its front for a stretch,
/// and the ink there -- the other family of contours, wrapping round the
/// back -- shows through as ticks along the silhouette that zoom magnifies.
public struct HeightfieldVisibility: Sendable {
    public let heightfield: Heightfield
    public let depth: DepthImage
    public let margin: Double
    /// The direction sight lines travel, in world space.
    public let sight: SIMD3<Double>

    public init(heightfield: Heightfield, depth: DepthImage, margin: Double,
                view: Transform<WorldSpace, ViewSpace>) {
        self.heightfield = heightfield; self.depth = depth; self.margin = margin
        self.sight = view.sightLine
    }

    /// How squarely the surface faces the viewer over `p`: positive facing,
    /// negative facing away, zero on a fold.
    public func facing(_ p: P2<WorldSpace>) -> Double {
        let n = heightfield.normal(at: p)
        return -simd_dot(n, sight) / simd_length(n)
    }

    /// The depth test: nothing closer than the margin is in front of `v`.
    public func isInFront(_ v: P3<ViewSpace>) -> Bool {
        v.z + margin > depth.depth(under: v.xy)
    }

    /// Where along the straight segment `a -> b` over the plane the surface
    /// turns edge-on, as a fraction of the way; nil when both ends face the
    /// same way. Bisection on the normal, to the last bit of the fraction.
    public func foldCrossing(from a: P2<WorldSpace>, _ fa: Double,
                             to b: P2<WorldSpace>, _ fb: Double) -> Double? {
        guard (fa > 0) != (fb > 0) else { return nil }
        var lo = 0.0, hi = 1.0
        let frontAtLo = fa > 0
        while hi - lo > 1e-13 {
            let mid = 0.5 * (lo + hi)
            let f = facing(P2(a.v + (b.v - a.v) * mid))
            if (f > 0) == frontAtLo { lo = mid } else { hi = mid }
        }
        return 0.5 * (lo + hi)
    }
}

public enum HiddenLine {
    public static func clip(_ paths: PolylineSet<ViewSpace>, against depth: DepthImage,
                            margin: Double) -> PolylineSet<PlateSpace> {
        var out: [[P3<PlateSpace>]] = []
        for i in 0..<paths.count {
            var run: [P3<PlateSpace>] = []
            for v in paths[path: i] {
                if v.z + margin > depth.depth(under: v.xy) {
                    run.append(P3(v.x, v.y, v.z))
                } else {
                    if run.count >= 2 { out.append(run) }
                    run = []
                }
            }
            if run.count >= 2 { out.append(run) }
        }
        return PolylineSet(paths: out)
    }

    /// Hidden-line removal for ink lying on a heightfield: the depth test
    /// above, and before it the facing test a heightfield allows.
    ///
    /// A heightfield bounds the solid under it, so ink on a part of the
    /// surface that faces away from the eye is hidden, full stop -- the
    /// depth test's margin has no say. Where a segment crosses the fold it
    /// is cut at the fold itself, found by bisection on the normal, and the
    /// new vertex belongs to the run on the visible side, judged by depth
    /// alone. This is `clip(_:on:onFolds:)`'s shape with the front-most
    /// test replaced by the depth test: a heightfield has no coordinate
    /// image, but its normal is a gradient away.
    ///
    /// `world` is `paths` before the camera, index for index.
    public static func clip(_ paths: PolylineSet<ViewSpace>, world: PolylineSet<WorldSpace>,
                            on visibility: HeightfieldVisibility) -> PolylineSet<PlateSpace> {
        precondition(paths.vertices.count == world.vertices.count
                     && paths.offsets == world.offsets,
                     "the view and world paths of a layer are the same paths")
        let facing = world.vertices.map { visibility.facing(P2($0.x, $0.y)) }
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
                if k > lo, let t = visibility.foldCrossing(
                    from: P2(world.vertices[k - 1].x, world.vertices[k - 1].y), facing[k - 1],
                    to: P2(world.vertices[k].x, world.vertices[k].y), facing[k]) {
                    let a = paths.vertices[k - 1], b = paths.vertices[k]
                    let v = P3<ViewSpace>(a.v + (b.v - a.v) * t)
                    take(v, visibility.isInFront(v))
                }
                take(paths.vertices[k], facing[k] > 0 && visibility.isInFront(paths.vertices[k]))
            }
            if run.count >= 2 { out.append(run) }
            run = []
        }
        return PolylineSet(paths: out)
    }

    /// Ink that is not depth-tested: a cut-face hatch lies *in* the wall it
    /// hatches, so testing it would erase about half of it to no purpose.
    /// Passing it through this rather than casting keeps "which space am I in"
    /// a decision the type system witnesses.
    public static func pass(_ paths: PolylineSet<ViewSpace>) -> PolylineSet<PlateSpace> {
        PolylineSet(vertices: paths.vertices.map { P3<PlateSpace>($0.v) },
                    offsets: paths.offsets)
    }
}

public extension DepthImage {
    /// The outline of the drawn region -- `outline.extract_outline`.
    ///
    /// Binarize coverage into a signed field and take its zero level set. The
    /// Python version does this by handing the binarized buffer to matplotlib's
    /// contouring and mapping the result back out of pixel space; with marching
    /// squares in Core it is four lines, and it needs no plotting library to
    /// find the edge of a picture.
    ///
    /// None of the three plates draws one -- they are bounded by their own wall
    /// ink -- so the bake offers it rather than assuming it.
    func silhouette() -> PolylineSet<PlateSpace> {
        guard frame.rows >= 2, frame.cols >= 2 else { return .empty }
        let coverage = (0..<(frame.rows * frame.cols)).map { i -> Float in
            values[i] > empty ? 1 : -1
        }
        // A grid whose sample positions are the frame's own lattice, so a
        // contour vertex comes out in view units without a second mapping.
        let grid = Grid2D(
            width: frame.cols, height: frame.rows,
            domain: Domain(real: Interval(lo: frame.axis1.lo, hi: frame.axis1.hi),
                           imag: Interval(lo: frame.axis0.lo, hi: frame.axis0.hi)),
            values: coverage)

        let paths = Contour.lines(of: grid, level: 0).map { line in
            line.map { p -> P3<PlateSpace> in
                // `p.x` is the column value and `p.y` the row value; which view
                // component each is depends on the frame's raster order.
                switch frame.order {
                case .buffer: return P3(p.y, p.x, 0)
                case .screen: return P3(p.x, p.y, 0)
                }
            }
        }
        return PolylineSet(paths: paths)
    }
}
