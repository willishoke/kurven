import Foundation
import simd

/// Hidden-line removal: a pure function of a depth image and some polylines.
///
/// A line-for-line port of `outline.clip_hidden_lines`. A vertex is visible when
/// `z + depthTolerance > buffer[pixel]`; hidden vertices split a path into runs; runs
/// shorter than two vertices draw nothing and are dropped. Because it takes a
/// `DepthImage` *value*, it is tested against a Python-dumped buffer and
/// Python-dumped polylines with no GPU anywhere in the loop -- which is what
/// makes "the bake reproduces the drawing" a checkable claim rather than a
/// rendering that looks about right.
///
/// The realtime preview does the same test per *fragment* in a shader, which can
/// disagree with this on sub-pixel runs. That disagreement is deliberate and
/// bounded: the preview is for navigating, the bake is the artifact, and the
/// bake is this.
/// The two questions asked of curves on a heightfield: does the surface face
/// the eye where the curves lie, and is anything in front of it.
///
/// The first is what `SurfaceVisibility` asks of a parametric surface, and a
/// heightfield can answer it too, since it bounds the solid under it: the
/// normal is the gradient, read off the grid. Without it, where a steep
/// graph turns away from the eye its back lies within any depth tolerance of its
/// front for a stretch, and the curves there -- the other family of contours,
/// wrapping round the back -- shows through as ticks along the outline.
/// The second is the traversal up the line of sight (`GridTraversal`): the exact
/// question the depth buffer approximates per pixel, asked per vertex with
/// no pixel in it.
public struct HeightfieldVisibility: Sendable {
    public let heightfield: Heightfield
    public let depthTolerance: Double
    /// The direction lines of sight travel, in world space.
    public let sight: SIMD3<Double>
    let traversal: GridTraversal

    public init(heightfield: Heightfield, view: Transform<WorldSpace, ViewSpace>, depthTolerance: Double) {
        self.heightfield = heightfield
        self.traversal = GridTraversal(heightfield, view: view, depthTolerance: depthTolerance)
        self.depthTolerance = self.traversal.depthTolerance
        self.sight = view.projectionDirection
    }

    /// How squarely the surface faces the viewer over `p`: positive facing,
    /// negative facing away, zero on a fold.
    public func facing(_ p: P2<WorldSpace>) -> Double {
        let n = heightfield.normal(at: p)
        return -simd_dot(n, sight) / simd_length(n)
    }

    /// Whether nothing of the solid stands between `p` and the eye, to
    /// within the depth tolerance.
    public func isClear(_ p: P3<WorldSpace>) -> Bool { traversal.unoccluded(p) }

    /// Why `p` is hidden or clear: every sample the traversal took along the
    /// line of sight from it and what it made of each, one line per sample.
    public func explain(_ p: P3<WorldSpace>) -> String {
        var lines: [String] = []
        let clear = traversal.unoccluded(p) { lines.append($0) }
        lines.append(clear ? "clear" : "hidden")
        return lines.joined(separator: "\n")
    }

    /// The depth tolerance as a distance along the line of sight: an occluder must
    /// stand in front of the curves by more than this.
    public var toleranceAlongProjection: Double { traversal.reach }

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
                            depthTolerance: Double) -> PolylineSet<PageSpace> {
        var out: [[P3<PageSpace>]] = []
        for i in 0..<paths.count {
            var run: [P3<PageSpace>] = []
            for v in paths[path: i] {
                if v.z + depthTolerance > depth.depth(under: v.xy) {
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

    /// Curves that are not depth-tested: unclipped curves, and every layer of a
    /// heightfield or a solid, which arrives already tested by its geometry
    /// (`Scene.visibleLayers`). Passing it through this rather than casting
    /// keeps "which space am I in" a decision the type system witnesses.
    public static func pass(_ paths: PolylineSet<ViewSpace>) -> PolylineSet<PageSpace> {
        PolylineSet(vertices: paths.vertices.map { P3<PageSpace>($0.v) },
                    offsets: paths.offsets)
    }
}

public extension DepthImage {
    /// The outline of the drawn region -- `outline.extract_outline`.
    ///
    /// Binarize coverage into a signed field and take its zero level set. The
    /// Python version does this by handing the binarized buffer to matplotlib's
    /// contouring and mapping the result back out of pixel space; with traversing
    /// squares in Core it is four lines, and it needs no plotting library to
    /// find the edge of a picture.
    ///
    /// None of the three plates draws one -- they are bounded by their own wall
    /// curves -- so the bake offers it rather than assuming it.
    func outline() -> PolylineSet<PageSpace> {
        guard frame.rows >= 2, frame.cols >= 2 else { return .empty }
        let coverage = (0..<(frame.rows * frame.cols)).map { i -> Float in
            values[i] > empty ? 1 : -1
        }
        // A grid whose sample positions are the frame's own grid, so a
        // contour vertex comes out in view units without a second mapping.
        let grid = Grid2D(
            width: frame.cols, height: frame.rows,
            domain: Domain(real: Interval(lo: frame.axis1.lo, hi: frame.axis1.hi),
                           imag: Interval(lo: frame.axis0.lo, hi: frame.axis0.hi)),
            values: coverage)

        let paths = Contour.lines(of: grid, level: 0).map { line in
            line.map { p -> P3<PageSpace> in
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
