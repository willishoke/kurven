import Foundation
import simd

/// A regular 2D sample grid over a rectangle.
///
/// `values[y * width + x]`, with `x` indexing real and `y` indexing imag, which
/// is the orientation `height.npy` is written in and the orientation a texture
/// wants. Sample positions are the *vertices* of the lattice: `x = 0` sits at
/// `domain.real.lo` and `x = width - 1` at `domain.real.hi`. That is the
/// convention `Surface` was sampled with, and getting it wrong shifts the whole
/// landscape by half a cell.
public struct Grid2D<T>: Sendable where T: Sendable {
    public let width: Int
    public let height: Int
    public let domain: Domain
    public let values: [T]

    public init(width: Int, height: Int, domain: Domain, values: [T]) {
        precondition(values.count == width * height,
                     "grid is \(width)x\(height) but has \(values.count) values")
        self.width = width; self.height = height
        self.domain = domain; self.values = values
    }

    public subscript(x: Int, y: Int) -> T { values[y * width + x] }

    /// The *shape* a `decimated(by:)` grid would have, without building it.
    ///
    /// The renderer needs four numbers from the decimated lattice -- its two
    /// dimensions and its domain -- and nothing else, because the heights come
    /// from the texture. Materializing the grid to read them off costs a full
    /// copy of the samples: 43 ms per pass on zeta's 25-million-sample grid, to
    /// produce four doubles.
    public func decimatedExtent(by step: Int) -> (width: Int, height: Int, domain: Domain) {
        guard step > 1 else { return (width, height, domain) }
        let w = (width + step - 1) / step
        let h = (height + step - 1) / step
        let dx = domain.real.length / Double(max(width - 1, 1))
        let dy = domain.imag.length / Double(max(height - 1, 1))
        return (w, h, Domain(
            real: Interval(lo: domain.real.lo,
                           hi: domain.real.lo + dx * Double((w - 1) * step)),
            imag: Interval(lo: domain.imag.lo,
                           hi: domain.imag.lo + dy * Double((h - 1) * step))))
    }

    /// Keep every `step`-th sample in both directions. The domain is unchanged
    /// only if the last sample survives, so the decimated grid's domain is
    /// narrowed to the samples it actually kept.
    ///
    /// Prefer `decimatedExtent(by:)` when only the shape is wanted.
    public func decimated(by step: Int) -> Grid2D<T> {
        guard step > 1 else { return self }
        let xs = stride(from: 0, to: width, by: step).map { $0 }
        let ys = stride(from: 0, to: height, by: step).map { $0 }
        var out: [T] = []
        out.reserveCapacity(xs.count * ys.count)
        for y in ys { for x in xs { out.append(self[x, y]) } }
        let dx = domain.real.length / Double(max(width - 1, 1))
        let dy = domain.imag.length / Double(max(height - 1, 1))
        return Grid2D(width: xs.count, height: ys.count,
                      domain: Domain(
                          real: Interval(lo: domain.real.lo,
                                         hi: domain.real.lo + dx * Double(xs.last ?? 0)),
                          imag: Interval(lo: domain.imag.lo,
                                         hi: domain.imag.lo + dy * Double(ys.last ?? 0))),
                      values: out)
    }
}

public extension Grid2D<Float> {
    /// Fractional sample index of a domain point. Separated from `sample` so the
    /// mesh builder and the sampler cannot disagree about where sample `(i, j)`
    /// lives.
    func index(of p: P2<DomainSpace>) -> SIMD2<Double> {
        SIMD2((p.x - domain.real.lo) / max(domain.real.length, .leastNormalMagnitude)
                * Double(width - 1),
              (p.y - domain.imag.lo) / max(domain.imag.length, .leastNormalMagnitude)
                * Double(height - 1))
    }

    func position(x: Int, y: Int) -> P2<DomainSpace> {
        P2(domain.real.lo + domain.real.length * Double(x) / Double(max(width - 1, 1)),
           domain.imag.lo + domain.imag.length * Double(y) / Double(max(height - 1, 1)))
    }

    /// Bilinear sample, clamped at the edges.
    func sample(_ p: P2<DomainSpace>) -> Double {
        let f = index(of: p)
        let x0 = min(max(Int(f.x.rounded(.down)), 0), width - 1)
        let y0 = min(max(Int(f.y.rounded(.down)), 0), height - 1)
        let x1 = min(x0 + 1, width - 1)
        let y1 = min(y0 + 1, height - 1)
        let tx = min(max(f.x - Double(x0), 0), 1)
        let ty = min(max(f.y - Double(y0), 0), 1)
        let a = Double(self[x0, y0]) * (1 - tx) + Double(self[x1, y0]) * tx
        let b = Double(self[x0, y1]) * (1 - tx) + Double(self[x1, y1]) * tx
        return a * (1 - ty) + b * ty
    }

    /// Nearest sample. `Surface.from_cache` on the Python side looks up this
    /// way, so a bundle built from a cache is reproduced by this, not by
    /// `sample`.
    func nearest(_ p: P2<DomainSpace>) -> Double {
        let f = index(of: p)
        let x = min(max(Int(f.x.rounded()), 0), width - 1)
        let y = min(max(Int(f.y.rounded()), 0), height - 1)
        return Double(self[x, y])
    }

    var range: ClosedRange<Float> {
        var lo = Float.infinity, hi = -Float.infinity
        for v in values where v.isFinite { lo = min(lo, v); hi = max(hi, v) }
        return lo <= hi ? lo...hi : 0...0
    }
}
