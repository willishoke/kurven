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
public enum RasterOrder: Sendable, Equatable {
    /// Rows index view x and columns index view y -- `kurven.zbuffer.ZBuffer`'s
    /// convention. The bake must use this: it is the lattice the Python oracle
    /// describes, and a buffer is a lookup table, never something anyone looks
    /// at, so its orientation is free to be inconvenient.
    case buffer
    /// Rows index view y downward and columns index view x -- the picture the
    /// right way up, with view x running across the screen as it does in the
    /// SVG. What a preview wants, and the same rectangle either way.
    case screen
}

public struct DepthFrame: Sendable, Equatable {
    /// The value at row 0 and at row `rows - 1`. Directed: for `.screen` order
    /// `lo` is the *top* of the picture, so `hi < lo`. Every formula below is
    /// sign-agnostic, which is why the direction can live in the data instead of
    /// in a branch.
    public var axis0: Interval
    /// The value at column 0 and at column `cols - 1`.
    public var axis1: Interval
    public var rows: Int
    public var cols: Int
    public var nudge: Double
    public var order: RasterOrder

    public init(axis0: Interval, axis1: Interval, rows: Int, cols: Int,
                nudge: Double = 0.01, order: RasterOrder = .buffer) {
        self.axis0 = axis0; self.axis1 = axis1
        self.rows = rows; self.cols = cols; self.nudge = nudge; self.order = order
    }

    /// The frame that covers `bounds` at `resolution` squared, in buffer order.
    public init(covering bounds: AABB<ViewSpace>, resolution: Int) {
        self.init(axis0: Interval(lo: bounds.lo.x, hi: bounds.hi.x),
                  axis1: Interval(lo: bounds.lo.y, hi: bounds.hi.y),
                  rows: resolution, cols: resolution)
    }

    /// The view components that drive rows and columns, in this order.
    public func components(of p: P2<ViewSpace>) -> (row: Double, col: Double) {
        switch order {
        case .buffer: (p.x, p.y)
        case .screen: (p.y, p.x)
        }
    }

    public func index(of p: P2<ViewSpace>) -> SIMD2<Int> {
        let c = components(of: p)
        return SIMD2(
            Int((nudge + Double(rows - 1) * (c.row - axis0.lo) / axis0.length).rounded(.down)),
            Int((nudge + Double(cols - 1) * (c.col - axis1.lo) / axis1.length).rounded(.down)))
    }

    /// Pixel size in view units along each screen axis. Equal for a frame a
    /// `Framing` built, which is what keeps preview pixels square.
    public var unitsPerPixel: SIMD2<Double> {
        SIMD2(abs(axis1.length) / Double(max(cols - 1, 1)),
              abs(axis0.length) / Double(max(rows - 1, 1)))
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
        let a = Double(row) / Double(rows - 1) * axis0.length + axis0.lo
        let b = Double(col) / Double(cols - 1) * axis1.length + axis1.lo
        switch order {
        case .buffer: return P2(a, b)
        case .screen: return P2(b, a)
        }
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
    /// The view-to-clip matrix for this frame, under an orthographic camera.
    ///
    /// Places each pixel's *sample point* exactly on the lattice vertex
    /// `coordinate(row:col:)` names. Metal samples pixel `i` at its centre,
    /// `(2i + 1) / N - 1` in NDC; solving for the coordinate that lands there
    /// gives the scale and offset below. The Python GPU path does not do this --
    /// it maps the coordinate range straight onto [-1, 1], half a pixel off the
    /// lattice its own `index_to_coord` describes -- so this is half a pixel
    /// closer to the CPU oracle than Python's own GPU path is.
    ///
    /// The y row is negated because Metal puts NDC +1 at the *top* of the render
    /// target, which is row 0, while the OpenGL path the Python side uses puts
    /// it at the bottom. Which view component drives which screen axis is the
    /// frame's business too, and folded in here: the bake rasterizes in
    /// ZBuffer's order so it can be compared against Python, the preview in
    /// screen order so it looks like the plate.
    ///
    /// A 4x4, so it can stand where a perspective matrix stands; its `w` row is
    /// constant, because an orthographic map has no divide.
    public var metalClip: simd_double4x4 {
        let sx = 2 * Double(cols - 1) / (Double(cols) * axis1.length)
        let sy = 2 * Double(rows - 1) / (Double(rows) * axis0.length)
        let ox = 1 / Double(cols) - 1 - sx * axis1.lo
        let oy = 1 / Double(rows) - 1 - sy * axis0.lo
        let zero: Double = 0
        let a: Double, b: Double, c: Double, d: Double
        switch order {
        case .buffer:                  // ndc.x from view.y, ndc.y from view.x
            a = zero; b = sx; c = -sy; d = zero
        case .screen:                  // ndc.x from view.x, ndc.y from view.y
            a = sx; b = zero; c = zero; d = -sy
        }
        // Columns, which is simd's own layout.
        let col0 = SIMD4<Double>(a, c, zero, zero)
        let col1 = SIMD4<Double>(b, d, zero, zero)
        let col2 = SIMD4<Double>(zero, zero, zero, zero)
        let col3 = SIMD4<Double>(ox, -oy, 0.5, 1)
        return simd_double4x4(col0, col1, col2, col3)
    }
}

/// A rendered depth buffer as a value.
///
/// `empty` is the value stored where nothing was drawn -- `-.infinity` for a
/// buffer built the way Python builds one, and a large negative sentinel for one
/// that came off the GPU, because Metal cannot clear a float attachment to
/// infinity. The two are interchangeable for the visibility test (`z + margin`
/// beats either for any real depth), so carrying the sentinel rather than
/// rewriting it costs nothing and saves a pass over the whole buffer: on a
/// 16000-square bake that pass was 400 ms of the 1600 the bake took.
///
/// Coverage is the one question that does distinguish them, and it asks
/// `isCovered` rather than testing for a magic value.
public struct DepthImage: Sendable {
    public let frame: DepthFrame
    public let values: [Float]
    /// The value meaning "nothing was drawn here".
    public let empty: Float

    public init(frame: DepthFrame, values: [Float], empty: Float = -.infinity) {
        precondition(values.count == frame.rows * frame.cols,
                     "depth image is \(frame.rows)x\(frame.cols) but has \(values.count) values")
        self.frame = frame; self.values = values; self.empty = empty
    }

    public subscript(row: Int, col: Int) -> Double { Double(values[row * frame.cols + col]) }

    /// Whether anything was drawn at a pixel. With a MAX blend a written value
    /// is `max(empty, z)`, so "covered" is exactly "greater than empty".
    public func isCovered(row: Int, col: Int) -> Bool {
        values[row * frame.cols + col] > empty
    }

    /// The stored depth under a view point, or `+.infinity` outside the frame --
    /// which reads as "occluded by the edge of the world", the same convention
    /// `clip_hidden_lines` uses for out-of-buffer lookups.
    public func depth(under p: P2<ViewSpace>) -> Double {
        let i = frame.index(of: p)
        guard i.x >= 0, i.x < frame.rows, i.y >= 0, i.y < frame.cols else { return .infinity }
        return self[i.x, i.y]
    }
}
