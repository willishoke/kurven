import Foundation
import simd

/// The mapping between view coordinates and depth-buffer pixels.
///
/// This is `kurven.zbuffer.ZBuffer`'s mapping, verbatim, including the `0.01`
/// nudge: samples are at lattice *vertices*, so the coordinate `axis0.lo` lands
/// at pixel 0 and `axis0.hi` at pixel `rows - 1`, and the nudge keeps a
/// coordinate exactly on `lo` from flooring to -1 under float jitter. Anything
/// else here is a half-pixel shift between what the renderer drew and what the
/// clipper reads, which shows up as ink leaking through the surface along
/// silhouettes and nowhere else.
///
/// `axis0` is the first view component and indexes *rows*; `axis1` is the second
/// and indexes columns. That is the Python convention and the reason `ZBuffer`
/// documents that it "doesn't care about x/y semantics" -- here the phantom
/// space says which is which.
public struct DepthFrame: Sendable, Equatable {
    public var axis0: Interval
    public var axis1: Interval
    public var rows: Int
    public var cols: Int
    public var nudge: Double

    public init(axis0: Interval, axis1: Interval, rows: Int, cols: Int,
                nudge: Double = 0.01) {
        self.axis0 = axis0; self.axis1 = axis1
        self.rows = rows; self.cols = cols; self.nudge = nudge
    }

    /// The frame that covers `bounds` at `resolution` squared.
    public init(covering bounds: AABB<ViewSpace>, resolution: Int) {
        self.init(axis0: Interval(lo: bounds.lo.x, hi: bounds.hi.x),
                  axis1: Interval(lo: bounds.lo.y, hi: bounds.hi.y),
                  rows: resolution, cols: resolution)
    }

    public func index(of p: P2<ViewSpace>) -> SIMD2<Int> {
        SIMD2(Int((nudge + Double(rows - 1) * (p.x - axis0.lo) / axis0.length).rounded(.down)),
              Int((nudge + Double(cols - 1) * (p.y - axis1.lo) / axis1.length).rounded(.down)))
    }

    /// Whether this frame's lattice contains a view point.
    public func contains(_ p: P2<ViewSpace>) -> Bool {
        let i = index(of: p)
        return i.x >= 0 && i.x < rows && i.y >= 0 && i.y < cols
    }

    /// The sub-frame covering tile `(i, j)` of an `n x n` split.
    ///
    /// Split on *pixel* boundaries, not coordinate boundaries: the tile's
    /// lattice is a contiguous slice of the whole plate's lattice, so a
    /// coordinate lands on the same pixel whether it was rendered whole or in
    /// pieces. Splitting the coordinate range instead would give each tile its
    /// own lattice and leave seams wherever two disagreed by a fraction of a
    /// pixel.
    public func tile(_ i: Int, _ j: Int, of n: Int) -> DepthFrame {
        guard n > 1 else { return self }
        let r0 = rows * i / n, r1 = rows * (i + 1) / n
        let c0 = cols * j / n, c1 = cols * (j + 1) / n
        let a0 = axis0.length / Double(rows - 1)
        let a1 = axis1.length / Double(cols - 1)
        return DepthFrame(
            axis0: Interval(lo: axis0.lo + a0 * Double(r0),
                            hi: axis0.lo + a0 * Double(r1 - 1)),
            axis1: Interval(lo: axis1.lo + a1 * Double(c0),
                            hi: axis1.lo + a1 * Double(c1 - 1)),
            rows: r1 - r0, cols: c1 - c0, nudge: nudge)
    }

    public func coordinate(row: Int, col: Int) -> P2<ViewSpace> {
        P2(Double(row) / Double(rows - 1) * axis0.length + axis0.lo,
           Double(col) / Double(cols - 1) * axis1.length + axis1.lo)
    }

    /// The affine map from a view coordinate to Metal's normalized device
    /// coordinates that puts each pixel's *sample point* exactly on the lattice
    /// vertex `coordinate(row:col:)` names.
    ///
    /// Metal samples pixel `i` at its center, `(2i + 1) / N - 1` in NDC. Solving
    /// for the coordinate that lands there gives the scale and offset below.
    /// The Python GPU path does not do this -- it maps the coordinate range
    /// straight onto [-1, 1], which puts every sample half a pixel off the
    /// lattice its own `index_to_coord` describes. Matching the lattice instead
    /// is what makes the rendered depth image and the clipper's lookups the
    /// same numbers rather than nearly the same.
    /// - Note: `scale.x`/`offset.x` map `axis1` (the second view component) to
    ///   NDC x, and `scale.y`/`offset.y` map `axis0` to NDC y -- Python's
    ///   `ZBuffer` indexes rows by the first view component, so that is the one
    ///   that must run down the image. The y pair is negated because Metal puts
    ///   NDC +1 at the *top* of the render target, i.e. at row 0, while the
    ///   OpenGL path the Python side uses puts it at the bottom; negating keeps
    ///   row `r` reading the coordinate `coordinate(row: r, col:)` names.
    public var metalNDC: (scale: SIMD2<Double>, offset: SIMD2<Double>) {
        let sx = 2 * Double(cols - 1) / (Double(cols) * axis1.length)
        let sy = 2 * Double(rows - 1) / (Double(rows) * axis0.length)
        let ox = 1 / Double(cols) - 1 - sx * axis1.lo
        let oy = 1 / Double(rows) - 1 - sy * axis0.lo
        return (SIMD2(sx, -sy), SIMD2(ox, -oy))
    }
}

/// A rendered depth buffer as a value.
///
/// Empty pixels are `-.infinity`, matching the Python fill, so the visibility
/// test needs no separate coverage mask: nothing is ever behind nothing.
public struct DepthImage: Sendable {
    public let frame: DepthFrame
    public let values: [Float]

    public init(frame: DepthFrame, values: [Float]) {
        precondition(values.count == frame.rows * frame.cols,
                     "depth image is \(frame.rows)x\(frame.cols) but has \(values.count) values")
        self.frame = frame; self.values = values
    }

    public subscript(row: Int, col: Int) -> Double { Double(values[row * frame.cols + col]) }

    /// The stored depth under a view point, or `+.infinity` outside the frame --
    /// which reads as "occluded by the edge of the world", the same convention
    /// `clip_hidden_lines` uses for out-of-buffer lookups.
    public func depth(under p: P2<ViewSpace>) -> Double {
        let i = frame.index(of: p)
        guard i.x >= 0, i.x < frame.rows, i.y >= 0, i.y < frame.cols else { return .infinity }
        return self[i.x, i.y]
    }
}
