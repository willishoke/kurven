import Foundation
import simd

/// The landscape: |f| on a grid, plus how it is truncated.
///
/// `height` is unclamped, and the cap is a separate value. That separation is
/// what lets a consumer change the cap without resampling -- the cap is a
/// drawing decision about how much of a pole to show, and the samples do not
/// depend on it.
public struct Surface: Sendable {
    public let height: Grid2D<Float>
    public let phase: Grid2D<Float>?
    public let caps: Caps
    /// True when the samples came from a cache rather than an evaluator, in
    /// which case the Python side looks heights up by nearest pixel and this
    /// side must too, or the walls sit at a different crest than the plate's.
    public let cached: Bool

    public init(height: Grid2D<Float>, phase: Grid2D<Float>?, caps: Caps,
                cached: Bool = false) {
        self.height = height; self.phase = phase; self.caps = caps; self.cached = cached
    }

    public var domain: Domain { height.domain }

    /// The clamped heightfield -- the surface the occluder is meshed from.
    public func clamped() -> Grid2D<Float> {
        switch caps {
        case .none:
            return height
        case .uniform(let z):
            let cap = Float(z)
            return Grid2D(width: height.width, height: height.height,
                          domain: height.domain, values: height.values.map { min($0, cap) })
        case .realBands:
            var out = height.values
            for x in 0..<height.width {
                let cap = Float(caps.height(atX: height.position(x: x, y: 0).x))
                for y in 0..<height.height { out[y * height.width + x] = min(out[y * height.width + x], cap) }
            }
            return Grid2D(width: height.width, height: height.height,
                          domain: height.domain, values: out)
        }
    }

    /// |f| at an arbitrary domain point, unclamped.
    public func magnitude(at p: P2<DomainSpace>) -> Double {
        cached ? height.nearest(p) : height.sample(p)
    }

    /// The height a wall rises to: `magnitude`, capped. `Surface.height_at`.
    public func height(at p: P2<DomainSpace>) -> Double {
        min(magnitude(at: p), caps.height(atX: p.x))
    }

    /// Every `step`-th sample, capped and lifted, without materializing a grid.
    ///
    /// The bounds pass folds over three million points on elliptic and six
    /// million on zeta; it wants none of them kept. `clamped().decimated()`
    /// would allocate both intermediates to hand them over one at a time.
    public func forEachSample(step: Int, _ body: (P3<WorldSpace>) -> Void) {
        let g = height
        let columns = Array(stride(from: 0, to: g.width, by: max(step, 1)))
        // The cap depends only on x, so it is one lookup per sampled column
        // rather than one per sample -- which is the whole difference between
        // `Caps.realBands` costing nothing and costing a branch per point.
        let columnCap = columns.map { caps.height(atX: g.position(x: $0, y: 0).x) }
        for y in stride(from: 0, to: g.height, by: max(step, 1)) {
            for (i, x) in columns.enumerated() {
                let p = g.position(x: x, y: y)
                body(P3(p.x, p.y, min(Double(g[x, y]), columnCap[i])))
            }
        }
    }

    /// Lift a domain point onto the (capped) surface.
    public func lift(_ p: P2<DomainSpace>) -> P3<WorldSpace> {
        P3(p.x, p.y, height(at: p))
    }
}
