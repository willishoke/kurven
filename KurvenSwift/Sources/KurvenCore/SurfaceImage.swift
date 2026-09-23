import Foundation
import simd

/// A depth image that also records *which* surface point is front-most at each
/// pixel.
///
/// Depth answers "is anything in front of this ink?" by a margin, and a margin
/// has no right value where two sheets pass within it of each other or where
/// the surface is steep enough in view that one pixel spans more depth than
/// the margin allows. The coordinate answers a different question -- "is the
/// thing in front of this ink the ink's own piece of surface?" -- and that one
/// has no margin in it.
public struct SurfaceImage: Sendable {
    public let depth: DepthImage
    /// The surface coordinates of the front-most point at each pixel, row
    /// major like `depth.values`; NaN where nothing was drawn. Unwrapped as
    /// rasterized, so compare them with `ParamAxis.difference`.
    public let coords: [SIMD2<Float>]

    public init(depth: DepthImage, coords: [SIMD2<Float>]) {
        precondition(coords.count == depth.values.count,
                     "\(coords.count) coordinates for \(depth.values.count) pixels")
        self.depth = depth; self.coords = coords
    }

    public var frame: DepthFrame { depth.frame }

    /// The front-most surface coordinate at a pixel, or nil where nothing was
    /// drawn.
    public func coordinate(row: Int, col: Int) -> P2<ParamSpace>? {
        let c = coords[row * frame.cols + col]
        return c.x.isNaN ? nil : P2(Double(c.x), Double(c.y))
    }
}

public extension SurfaceImage {
    /// Rasterize a parametric surface's lattice on the CPU.
    ///
    /// The reference the GPU coordinate pass is held to, as the Python
    /// Z-buffer is for the depth pass. Two triangles per lattice cell, sampled
    /// at the frame's lattice points -- the same points `metalClip` puts pixel
    /// centres on -- keeping the greatest view z, and carrying the
    /// interpolated surface coordinate of whichever triangle won.
    static func rasterize(_ s: ParametricSurface, view: Transform<WorldSpace, ViewSpace>,
                          frame: DepthFrame) -> SurfaceImage {
        let nu = s.u.cells, nv = s.v.cells
        let (rows, cols) = (frame.rows, frame.cols)
        let rowScale = Double(rows - 1) / frame.axis0.length
        let colScale = Double(cols - 1) / frame.axis1.length

        // Every lattice vertex once, as (row, col, view z) in continuous pixel
        // units, with its unwrapped coordinate beside it.
        var pixel: [SIMD3<Double>] = []
        var param: [SIMD2<Double>] = []
        pixel.reserveCapacity((nu + 1) * (nv + 1))
        param.reserveCapacity((nu + 1) * (nv + 1))
        for j in 0...nv {
            for i in 0...nu {
                let q = view(s.position(i, j))
                let c = frame.components(of: q.xy)
                pixel.append(SIMD3((c.row - frame.axis0.lo) * rowScale,
                                   (c.col - frame.axis1.lo) * colScale, q.z))
                param.append(SIMD2(s.u.coordinate(i), s.v.coordinate(j)))
            }
        }

        var depth = [Float](repeating: -.infinity, count: rows * cols)
        var coords = [SIMD2<Float>](repeating: SIMD2(.nan, .nan), count: rows * cols)

        func edge(_ a: SIMD3<Double>, _ b: SIMD3<Double>, _ r: Double, _ c: Double) -> Double {
            (b.x - a.x) * (c - a.y) - (r - a.x) * (b.y - a.y)
        }
        func triangle(_ k0: Int, _ k1: Int, _ k2: Int) {
            let p0 = pixel[k0], p1 = pixel[k1], p2 = pixel[k2]
            let area = edge(p0, p1, p2.x, p2.y)
            guard abs(area) > 1e-12 else { return }
            let r0 = max(0, Int(min(p0.x, p1.x, p2.x).rounded(.up)))
            let r1 = min(rows - 1, Int(max(p0.x, p1.x, p2.x).rounded(.down)))
            let c0 = max(0, Int(min(p0.y, p1.y, p2.y).rounded(.up)))
            let c1 = min(cols - 1, Int(max(p0.y, p1.y, p2.y).rounded(.down)))
            guard r0 <= r1, c0 <= c1 else { return }
            let t0 = param[k0], t1 = param[k1], t2 = param[k2]
            for r in r0...r1 {
                let rr = Double(r)
                for c in c0...c1 {
                    let cc = Double(c)
                    let w0 = edge(p1, p2, rr, cc) / area
                    let w1 = edge(p2, p0, rr, cc) / area
                    let w2 = 1 - w0 - w1
                    // Inclusive: a sample on a shared edge is claimed by both
                    // triangles and the greater depth decides, as MAX does.
                    guard w0 >= -1e-12, w1 >= -1e-12, w2 >= -1e-12 else { continue }
                    let z = w0 * p0.z + w1 * p1.z + w2 * p2.z
                    let k = r * cols + c
                    if Float(z) > depth[k] {
                        depth[k] = Float(z)
                        let t = w0 * t0 + w1 * t1 + w2 * t2
                        coords[k] = SIMD2(Float(t.x), Float(t.y))
                    }
                }
            }
        }

        let stride = nu + 1
        for j in 0..<nv {
            for i in 0..<nu {
                let a = j * stride + i, b = a + 1, c = a + stride, d = c + 1
                triangle(a, b, c)
                triangle(b, d, c)
            }
        }
        return SurfaceImage(depth: DepthImage(frame: frame, values: depth), coords: coords)
    }
}
