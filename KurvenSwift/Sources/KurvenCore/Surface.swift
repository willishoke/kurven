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

    /// The height of a *tiled* landscape at a world point.
    ///
    /// The grid covers one fundamental tile; the plate covers its images under
    /// the occluder's affine maps. A point out on the plate is therefore not in
    /// the grid, and asking the grid about it returns whatever is nearest the
    /// edge -- which for elliptic, whose cutout runs across all nineteen tiles
    /// and whose tile edge is a pole, is a spire-high wall standing along the
    /// whole boundary.
    ///
    /// The point is mapped back through the tile it belongs to, chosen as the
    /// *nearest* rather than the containing one. Containment is too strict to be
    /// useful here: elliptic samples its tile on `[-K + eps, -eps]` to keep off
    /// the poles, so the cutout perimeter -- drawn at the exact quarter-periods
    /// -- lies a hair outside every tile image, and an exact test matches
    /// nothing at all.
    func height(at p: P2<DomainSpace>, tiles: [Affine2]) -> Double {
        guard tiles.count > 1 else { return height(at: p) }
        let d = domain
        let (xlo, xhi) = (min(d.real.lo, d.real.hi), max(d.real.lo, d.real.hi))
        let (ylo, yhi) = (min(d.imag.lo, d.imag.hi), max(d.imag.lo, d.imag.hi))

        var best: P2<DomainSpace>?
        var bestDistance = Double.infinity
        for tile in tiles {
            guard let back = tile.inverse else { continue }
            let q = back(P3<WorldSpace>(p.x, p.y, 0))
            // How far outside the domain box this tile puts the point; zero
            // when it is inside.
            let dx = max(xlo - q.x, 0) + max(q.x - xhi, 0)
            let dy = max(ylo - q.y, 0) + max(q.y - yhi, 0)
            let distance = dx + dy
            if distance < bestDistance {
                bestDistance = distance
                best = P2<DomainSpace>(q.x, q.y)
                if distance == 0 { break }
            }
        }
        return height(at: best ?? p)
    }

    /// Lift a domain point onto the (capped) surface.
    public func lift(_ p: P2<DomainSpace>) -> P3<WorldSpace> {
        P3(p.x, p.y, height(at: p))
    }
}

// MARK: - deriving ink from the grids

public extension Surface {
    /// Lift a domain-space path onto the surface, by policy.
    ///
    /// `Surface.lift_contours`' three height policies, and the reason a bundle
    /// records which one a layer used: a magnitude isocontour sits at exactly
    /// its own level, a phase contour sits wherever the surface is, and an
    /// unclamped lift is a third thing again. A consumer regenerating a layer
    /// cannot guess which was meant.
    func lift(_ path: some Sequence<P2<DomainSpace>>, policy: HeightPolicy,
              level: Double) -> [P3<WorldSpace>] {
        path.map { p in
            switch policy {
            case .surface: P3(p.x, p.y, height(at: p))
            case .level: P3(p.x, p.y, level)
            case .magnitude: P3(p.x, p.y, magnitude(at: p))
            }
        }
    }

    /// Whether a lifted vertex survives a `Keep`.
    func admits(_ p: P3<WorldSpace>, _ keep: Keep, region: Region) -> Bool {
        switch keep {
        case .all: true
        case .region: region.contains(p.xy)
        case .belowCap: p.z <= caps.height(atX: p.x)
        case .band(let axis, let lo, let hi):
            switch axis {
            case .real: p.x > lo && p.x < hi
            case .imag: p.y > lo && p.y < hi
            }
        case .every(let all): all.allSatisfy { admits(p, $0, region: region) }
        }
    }

    /// Every level of a described layer, contoured, lifted, filtered and tiled.
    ///
    /// A path that leaves the kept region and comes back is split where it left,
    /// not closed across the gap. Filtering vertices and keeping the survivors
    /// as one polyline welds the far side of a contour to the near side; on the
    /// zeta plate that drew straight chords up to half the width of the domain
    /// across ground the cutout had deliberately removed.
    func derive(_ source: LayerSource, policy: HeightPolicy, region: Region,
                tiles: [Affine2]) -> PolylineSet<WorldSpace> {
        guard case .contour(let field, let levels, let keep, let tiled) = source else {
            return .empty
        }
        let grid: Grid2D<Float>
        switch field {
        case .magnitude: grid = height
        case .phase:
            guard let phase else { return .empty }
            grid = phase
        }

        var paths: [[P3<WorldSpace>]] = []
        for level in levels {
            for line in Contour.lines(of: grid, level: level) {
                let lifted = lift(line, policy: policy, level: level)
                var run: [P3<WorldSpace>] = []
                for v in lifted {
                    if admits(v, keep, region: region) {
                        run.append(v)
                    } else {
                        if run.count >= 2 { paths.append(run) }
                        run = []
                    }
                }
                if run.count >= 2 { paths.append(run) }
            }
        }
        guard tiled else { return PolylineSet(paths: paths) }
        // Replicated by the same maps the occluder instances the heightfield
        // with, so a tiled contour cannot drift from the tile it lies on.
        return PolylineSet(paths: tiles.flatMap { tile in
            paths.map { $0.map { tile($0) } }
        })
    }
}
