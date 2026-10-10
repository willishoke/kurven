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
        height(at: inTile(p, tiles: tiles))
    }

    /// |f| at a world point of a tiled landscape, uncapped: `height(at:tiles:)`
    /// before the cap.
    func magnitude(at p: P2<DomainSpace>, tiles: [Affine2]) -> Double {
        magnitude(at: inTile(p, tiles: tiles))
    }

    /// The point of the fundamental tile a world point is the image of.
    func inTile(_ p: P2<DomainSpace>, tiles: [Affine2]) -> P2<DomainSpace> {
        guard tiles.count > 1 else { return p }
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
        return best ?? p
    }

    /// The height as the depth pass draws it over a domain point: the grid
    /// decimated by `step`, each cell as the two triangles the rasterizer
    /// splits it into -- from corner (1, 0) to corner (0, 1) -- then capped.
    /// `height(at:)` interpolates bilinearly, which is what the Python plate
    /// placed its ink by and the fixtures pin; this is what the ink is drawn
    /// against, and the two differ within a cell by its twist.
    func drawnHeight(at p: P2<DomainSpace>, step: Int) -> Double {
        let drawn = drawnSurface(at: p, step: step)
        return min(drawn.interpolated, drawn.cap)
    }

    /// `drawnHeight` as the two numbers it is the lesser of -- the lattice's
    /// interpolated height and the cap over the point -- with the cosine of
    /// the interpolating facet's tilt, `1 / sqrt(1 + |∇h|²)`, which turns a
    /// vertical gap to that facet into a perpendicular distance.
    func drawnSurface(at p: P2<DomainSpace>, step: Int)
        -> (interpolated: Double, cap: Double, cosine: Double)
    {
        let g = height
        let step = max(step, 1)
        let nx = (g.width + step - 1) / step, ny = (g.height + step - 1) / step
        let cap = caps.height(atX: p.x)
        guard nx >= 2, ny >= 2 else { return (Double(g[0, 0]), cap, 1) }
        let dx = abs(g.domain.real.length) / Double(max(g.width - 1, 1)) * Double(step)
        let dy = abs(g.domain.imag.length) / Double(max(g.height - 1, 1)) * Double(step)
        let f = g.index(of: p) / Double(step)
        let fx = min(max(f.x, 0), Double(nx - 1)), fy = min(max(f.y, 0), Double(ny - 1))
        let i = min(Int(fx), nx - 2), j = min(Int(fy), ny - 2)
        let u = fx - Double(i), v = fy - Double(j)
        let h00 = Double(g[i * step, j * step]), h10 = Double(g[(i + 1) * step, j * step])
        let h01 = Double(g[i * step, (j + 1) * step]), h11 = Double(g[(i + 1) * step, (j + 1) * step])
        let lower = u + v <= 1
        let h = lower
            ? h00 + u * (h10 - h00) + v * (h01 - h00)
            : h11 + (1 - u) * (h01 - h11) + (1 - v) * (h10 - h11)
        let hx = (lower ? h10 - h00 : h11 - h01) / dx
        let hy = (lower ? h01 - h00 : h11 - h10) / dy
        let cosine = 1 / (1 + hx * hx + hy * hy).squareRoot()
        return (h.isNaN ? .infinity : h, cap, cosine.isNaN ? 0 : cosine)
    }

    /// Lift a domain point onto the (capped) surface.
    public func lift(_ p: P2<DomainSpace>) -> P3<WorldSpace> {
        P3(p.x, p.y, height(at: p))
    }
}

// MARK: - deriving ink from the grids

/// What a consumer that has the function supplies, so the ink can be placed by
/// f rather than by the grid. Core has the grids and not the function;
/// `KurvenLandscape`'s `ContourRefiner` builds one of these.
public struct ContourRefine: Sendable {
    /// Moves the paths of one level of one field onto the function's own
    /// level set, given the field, the level and the marching-squares paths
    /// in domain space.
    public typealias Contours = @Sendable (ContourField, Double, [[P2<DomainSpace>]])
        -> [[P2<DomainSpace>]]
    /// |f| at a domain point, uncapped, as the grid would hold it.
    public typealias Magnitude = @Sendable (P2<DomainSpace>) -> Double
    /// A zero of f within a few cells of a domain point, or nil.
    public typealias Zero = @Sendable (P2<DomainSpace>) -> P2<DomainSpace>?

    public let contours: Contours
    /// With it, the crest of a cut face gains a vertex at each minimum of |f|
    /// between two grid nodes (`Surface.crest`), and a contour that ends on a
    /// zero of f ends at the zero's height (`Surface.derive`). Nil for a
    /// consumer that can place contours and not heights.
    public let magnitude: Magnitude?
    /// With it, a fold line that ends beside a zero of f is carried to the
    /// zero (`Heightfield.foldLines`), as the refiner carries the phase lines.
    public let zero: Zero?
    /// How far a chord may depart from the curve it stands for, in cells:
    /// the refiner's own tolerance, which the crest of a cut face and the
    /// fold lines are subdivided to as well, so every curve of the drawing
    /// is a curve to the same tolerance.
    public let tolerance: Double

    public init(contours: @escaping Contours, magnitude: Magnitude? = nil,
                zero: Zero? = nil, tolerance: Double = 0.02) {
        self.tolerance = tolerance
        self.contours = contours; self.magnitude = magnitude; self.zero = zero
    }
}

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

    /// Where the segment from `a`, under the cap, to `b`, over it, reaches the
    /// cap -- on the rim, at exactly the cap's height. Nil when it does not,
    /// or when there is no cap.
    ///
    /// Solved on the lifted heights, linearly: that is the same interpolation
    /// the rim is contoured by, so the crossing lands on the drawn rim to
    /// within a cell's bend.
    func capCrossing(from a: P3<WorldSpace>, to b: P3<WorldSpace>) -> P3<WorldSpace>? {
        let ea = a.z - caps.height(atX: a.x), eb = b.z - caps.height(atX: b.x)
        guard ea.isFinite, eb.isFinite, ea <= 0, eb > 0 else { return nil }
        let t = ea / (ea - eb)
        let x = a.x + (b.x - a.x) * t
        return P3(x, a.y + (b.y - a.y) * t, caps.height(atX: x))
    }

    /// Every level of a described layer, contoured, lifted, filtered and tiled.
    ///
    /// A path that leaves the kept region and comes back is split where it left,
    /// not closed across the gap. Filtering vertices and keeping the survivors
    /// as one polyline welds the far side of a contour to the near side; on the
    /// zeta plate that drew straight chords up to half the width of the domain
    /// across ground the cutout had deliberately removed.
    ///
    /// Where a path leaves by going *over the cap*, the split is at the cap
    /// itself: the run gains a vertex on the rim (`capCrossing`) rather than
    /// ending at its last vertex under it, which on a pole's flank is up to
    /// half a world unit short. Only with a refiner: without one the ink is
    /// exactly what Python derives from the same grids, which ends at the
    /// last sample, and the fixtures hold it to that.
    ///
    /// A phase level of ±π is the cut, where arg f jumps, and the grid has no
    /// contour at it. With a refiner it is drawn once, as one line of ink the
    /// way the Jahnke-Emde plates draw it (`Contour.cut`), under the level π
    /// so the refiner knows to place it on arg(-f) = 0.
    func derive(_ source: LayerSource, policy: HeightPolicy, region: Region,
                tiles: [Affine2], refine: ContourRefine? = nil) -> PolylineSet<WorldSpace> {
        guard case .contour(let field, let levels, let keep, let tiled) = source else {
            return .empty
        }
        let grid: Grid2D<Float>
        var contoured: [(level: Double, paths: [[P2<DomainSpace>]])]
        switch field {
        case .magnitude:
            grid = height
            contoured = Contour.levels(of: grid, levels)
        case .phase:
            guard let phase else { return .empty }
            grid = phase
            let plain = levels.filter { !Contour.isCut($0) }
            contoured = Contour.levels(of: grid, plain)
            if refine != nil, plain.count < levels.count {
                contoured.append((.pi, Contour.cut(of: grid)))
            }
        }
        let carry = refine != nil && keep.keepsBelowCap

        var paths: [[P3<WorldSpace>]] = []
        for (level, lines) in contoured {
            // A refiner has the function the grid was sampled from and moves
            // the vertices onto its true level set, in domain space, before
            // the lift: the grid decides which contours exist, f decides
            // exactly where they run.
            let refined = refine.map { r in r.contours(field, level, lines) } ?? lines
            for line in refined {
                // Lifted onto f where the function is at hand: a phase line
                // lifted to the lattice's height stands a chord's error off
                // the surface the rings, the folds and the crest are drawn
                // on, and the judge hides it there or shows it floating.
                var lifted: [P3<WorldSpace>]
                if policy != .level, let magnitude = refine?.magnitude {
                    lifted = line.map { p in
                        let u = magnitude(p)
                        guard u.isFinite else { return lift([p], policy: policy, level: level)[0] }
                        return P3(p.x, p.y, policy == .surface ? min(u, caps.height(atX: p.x)) : u)
                    }
                } else {
                    lifted = lift(line, policy: policy, level: level)
                }
                // A run the refiner carried to a zero of f ends at the zero's
                // own height. The grid has no sample there: its chord over a
                // pit's floor bottoms out at the nearest sample's height, a
                // cell's rise above the zero, and lifting the end to the
                // chord would hang the fan of phase lines that far above the
                // crest that f draws under them. Only an end that *is* a zero,
                // to rounding against the grid's own height, so a vertex the
                // grid places a hair under f is left on the drawn surface.
                if policy != .level, let magnitude = refine?.magnitude, !lifted.isEmpty {
                    for i in Set([0, lifted.count - 1]) {
                        let u = magnitude(line[i])
                        if u.isFinite, u <= 1e-6 * lifted[i].z {
                            lifted[i] = P3(lifted[i].x, lifted[i].y, u)
                        }
                    }
                }
                var run: [P3<WorldSpace>] = []
                var previous: (vertex: P3<WorldSpace>, admitted: Bool)?
                for v in lifted {
                    let admitted = admits(v, keep, region: region)
                    if admitted {
                        if carry, let p = previous, !p.admitted,
                           let rim = capCrossing(from: v, to: p.vertex),
                           admits(rim, keep, region: region) {
                            run.append(rim)
                        }
                        run.append(v)
                    } else {
                        if carry, let p = previous, p.admitted,
                           let rim = capCrossing(from: p.vertex, to: v),
                           admits(rim, keep, region: region) {
                            run.append(rim)
                        }
                        if run.count >= 2 { paths.append(run) }
                        run = []
                    }
                    previous = (v, admitted)
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
