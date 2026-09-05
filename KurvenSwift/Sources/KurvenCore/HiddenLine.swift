import Foundation

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
