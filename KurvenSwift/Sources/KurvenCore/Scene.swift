import Foundation
import simd

/// The identity of a piece of content.
///
/// GPU resources are a memo keyed by *what* is being drawn, and "what" cannot be
/// a hash: hashing a hundred megabytes of heightfield every frame costs more
/// than re-uploading it. So identity is assigned once, when the content is
/// built, and carried. Two scenes with the same `ContentID` are the same
/// geometry seen from possibly different places.
public struct ContentID: Hashable, Sendable {
    private let raw: UUID
    public init() { raw = UUID() }
}

/// A heightfield over the complex plane, as the plates draw it: the surface,
/// instanced once per tile, cut to a region, with its wall curtains.
public struct Heightfield: Sendable {
    public let surface: Surface
    /// Wall curtains, and the cap rims where the heightfield's own triangles
    /// cut the corner (`Mesh.capRim`), in world coordinates.
    public let occluder: Mesh<WorldSpace>
    /// Heightfield instances; always at least the identity.
    public let tiles: [Affine2]
    /// The footprint each instance is clipped to.
    public let region: Region
    /// Subsampling step for the heightfield when it is rasterized.
    public let step: Int

    public init(surface: Surface, occluder: Mesh<WorldSpace>, tiles: [Affine2],
                region: Region = .full, step: Int) {
        self.surface = surface; self.occluder = occluder; self.tiles = tiles
        self.region = region; self.step = step
    }

    /// Every vertex of the decimated, capped heightfield, once per tile.
    ///
    /// The lattice is `Surface.grid_mesh`'s `self.clamped[::step, ::step]`. Not
    /// materialized: elliptic's is three million points, zeta's six, and the
    /// caller only ever folds over them.
    public func forEachSample(_ body: (P3<WorldSpace>) -> Void) {
        for tile in tiles {
            surface.forEachSample(step: step) { body(tile($0)) }
        }
    }

    /// The outward normal of the capped surface over a world point:
    /// `(-∂h/∂x, -∂h/∂y, 1)`, not normalized, by central differences a grid
    /// spacing wide, the height read through whichever tile the point is in.
    ///
    /// The solid is what lies under the graph, so outward is up, and a
    /// sight line that reaches a point of the surface facing away from the
    /// eye has entered the solid already: such a point is hidden, whatever
    /// the depth buffer says to within its margin. That is the test the
    /// margin cannot make -- where a steep flank turns away, the back of it
    /// lies within the margin of the front for a stretch that zoom
    /// magnifies, and back-face ink shows there as ticks.
    ///
    /// Differences in world coordinates, so a tile's reflection is already
    /// in them; on the cap the differences vanish and the normal is up.
    public func normal(at p: P2<WorldSpace>) -> SIMD3<Double> {
        let g = surface.height
        let dx = abs(g.domain.real.length) / Double(max(g.width - 1, 1))
        let dy = abs(g.domain.imag.length) / Double(max(g.height - 1, 1))
        func h(_ x: Double, _ y: Double) -> Double {
            surface.height(at: P2<DomainSpace>(x, y), tiles: tiles)
        }
        let hx = (h(p.x + dx, p.y) - h(p.x - dx, p.y)) / (2 * dx)
        let hy = (h(p.x, p.y + dy) - h(p.x, p.y - dy)) / (2 * dy)
        return SIMD3(-hx, -hy, 1)
    }

    /// The fold lines under a camera: where the surface turns edge-on,
    /// `n · sight = 0`, which is where its outline and every inner silhouette
    /// lie -- the far edge of a pit, the crest of a flank seen from the side.
    /// The 1933 plates draw them, and the notebooks drew them from a
    /// thresholded matrix of gradients, contoured. This is that, with the
    /// sight line as the threshold: the facing is sampled on the lattice,
    /// its zero level traced by marching squares, and, when `refined`, every
    /// vertex moved onto the exact fold by bisection along the facing's
    /// gradient -- the same facing `HeightfieldVisibility` cuts ink by, so
    /// the two agree on where the surface turns. Nothing on a cap, which is
    /// flat and faces up and has a rim of its own, and nothing outside the
    /// region. They depend on the camera, so they are derived for one, never
    /// stored.
    public func foldLines(view: Transform<WorldSpace, ViewSpace>,
                          refined: Bool = true) -> PolylineSet<WorldSpace> {
        let sight = view.sightLine
        let g = surface.height
        let nx = g.width, ny = g.height
        guard nx >= 2, ny >= 2 else { return .empty }
        func facing(_ p: P2<WorldSpace>) -> Double {
            let n = normal(at: p)
            let len = simd_length(n)
            return len > 0 ? -simd_dot(n, sight) / len : 0
        }
        let cell = max(abs(g.domain.real.length) / Double(nx - 1),
                       abs(g.domain.imag.length) / Double(ny - 1))
        let h = 0.25 * cell
        /// Onto the fold: bracket a sign change along the gradient within a
        /// cell either way, then bisect it to the last bit.
        func onto(_ c: P2<WorldSpace>) -> P2<WorldSpace> {
            let gx = facing(P2(c.x + h, c.y)) - facing(P2(c.x - h, c.y))
            let gy = facing(P2(c.x, c.y + h)) - facing(P2(c.x, c.y - h))
            let len = (gx * gx + gy * gy).squareRoot()
            guard len > 0 else { return c }
            let d = SIMD2(gx, gy) / len * cell
            var a = SIMD2(c.x, c.y) - d, b = SIMD2(c.x, c.y) + d
            var fa = facing(P2(a.x, a.y))
            guard (fa > 0) != (facing(P2(b.x, b.y)) > 0) else { return c }
            for _ in 0..<60 {
                let m = 0.5 * (a + b)
                let fm = facing(P2(m.x, m.y))
                if (fm > 0) == (fa > 0) { a = m; fa = fm } else { b = m }
            }
            let m = 0.5 * (a + b)
            return P2(m.x, m.y)
        }

        var paths: [[P3<WorldSpace>]] = []
        let d = g.domain
        for tile in tiles {
            // The tile's image of the domain, as a box: the tiles are
            // reflections and translations, so it is one.
            let corners = [(d.real.lo, d.imag.lo), (d.real.hi, d.imag.lo),
                           (d.real.lo, d.imag.hi), (d.real.hi, d.imag.hi)]
                .map { tile(P3<WorldSpace>($0.0, $0.1, 0)) }
            let lo = SIMD2(corners.map(\.x).min()!, corners.map(\.y).min()!)
            let hi = SIMD2(corners.map(\.x).max()!, corners.map(\.y).max()!)
            let sx = (hi.x - lo.x) / Double(nx - 1), sy = (hi.y - lo.y) / Double(ny - 1)
            var values = [Float](repeating: 0, count: nx * ny)
            values.withUnsafeMutableBufferPointer { out in
                // Each row writes only its own slots, so sharing is safe.
                nonisolated(unsafe) let base = out.baseAddress!
                DispatchQueue.concurrentPerform(iterations: ny) { j in
                    for i in 0..<nx {
                        base[j * nx + i] = Float(facing(P2(lo.x + Double(i) * sx,
                                                           lo.y + Double(j) * sy)))
                    }
                }
            }
            let field = Grid2D(width: nx, height: ny,
                               domain: Domain(real: Interval(lo: lo.x, hi: hi.x),
                                              imag: Interval(lo: lo.y, hi: hi.y)),
                               values: values)
            for line in Contour.lines(of: field, level: 0) {
                var run: [P3<WorldSpace>] = []
                for v in line {
                    var p = P2<WorldSpace>(v.x, v.y)
                    if refined { p = onto(p) }
                    let z = surface.height(at: P2<DomainSpace>(p.x, p.y), tiles: tiles)
                    if region.contains(p) && z < surface.caps.height(atX: p.x) - 1e-9 {
                        run.append(P3(p.x, p.y, z))
                    } else {
                        if run.count >= 2 { paths.append(run) }
                        run = []
                    }
                }
                if run.count >= 2 { paths.append(run) }
            }
        }
        return PolylineSet(paths: paths)
    }

    /// The height of the surface *as the depth pass draws it* over a world
    /// point: the lattice decimated by `step`, each cell as the two triangles
    /// the rasterizer splits it into, capped, through the tile the point is
    /// in. This is the solid every piece of ink is judged against, so the
    /// judge and the rasterizer cannot disagree about where the surface is.
    public func drawnHeight(at p: P2<WorldSpace>) -> Double {
        surface.drawnHeight(at: surface.inTile(P2<DomainSpace>(p.x, p.y), tiles: tiles), step: step)
    }

    /// The fold lines that can be seen: `foldLines`, less what the surface
    /// hides. A fold cannot be judged by a depth buffer: the surface is
    /// edge-on there, so its depth changes by the whole flank within one
    /// pixel, and the pixel's value is the near side's, in front of the fold
    /// by far more than any margin. It is judged by the solid instead: a fold
    /// vertex is hidden when the sight line from it to the eye passes under
    /// the surface (`SightMarch`). `margin` is the plate's hidden-line
    /// margin, which keeps a vertex on the surface it belongs to from hiding
    /// behind that surface's own rounding.
    public func visibleFolds(view: Transform<WorldSpace, ViewSpace>, margin: Double,
                             refined: Bool = true) -> PolylineSet<WorldSpace> {
        let folds = foldLines(view: view, refined: refined)
        guard !folds.vertices.isEmpty else { return folds }
        let march = SightMarch(self, view: view, margin: margin)
        return Self.runs(of: folds) { march.unoccluded($0) }
    }

    /// Ink that lies in a cut face -- the wall's hatch and its outline --
    /// less what cannot be seen. A wall is a plane the plate camera often
    /// sees nearly edge-on, and the depth buffer cannot judge ink on such a
    /// face: one pixel spans the whole receding wall, keeps the near side's
    /// depth, and erases most of the hatch, leaving spots. So wall ink is
    /// judged as the folds are: hidden where the wall it lies in faces away
    /// from the eye -- it is then behind the solid the wall bounds -- and
    /// otherwise hidden only where the sight line from it passes under the
    /// surface. A post at a corner lies in two walls and shows if either
    /// does. Which wall a vertex lies in is read off its position against
    /// the box of each tile, since every wall of a described landscape
    /// stands on a tile's edge.
    ///
    /// The crest is the exception: it is the surface's own edge as much as
    /// the wall's top, and the far edge of a plate is in plain view over a
    /// wall that faces away. A vertex at the surface's height is judged by
    /// the march alone, which hides it exactly when the surface in front of
    /// it does.
    public func visibleWallInk(_ paths: PolylineSet<WorldSpace>,
                               view: Transform<WorldSpace, ViewSpace>,
                               margin: Double) -> PolylineSet<WorldSpace> {
        guard !paths.vertices.isEmpty else { return paths }
        let sight = view.sightLine
        let march = SightMarch(self, view: view, margin: margin)
        let d = surface.domain
        let slack = 1e-6 * march.cell
        let s = surface, t = tiles
        let boxes: [(lo: SIMD2<Double>, hi: SIMD2<Double>)] = tiles.map { tile in
            let corners = [(d.real.lo, d.imag.lo), (d.real.hi, d.imag.lo),
                           (d.real.lo, d.imag.hi), (d.real.hi, d.imag.hi)]
                .map { tile(P3<WorldSpace>($0.0, $0.1, 0)) }
            return (SIMD2(corners.map(\.x).min()!, corners.map(\.y).min()!),
                    SIMD2(corners.map(\.x).max()!, corners.map(\.y).max()!))
        }
        return Self.runs(of: paths) { p in
            let h = s.height(at: P2<DomainSpace>(p.x, p.y), tiles: t)
            let onCrest = p.z >= h - 1e-9 * max(1, abs(h))
            /// Whether some wall this point lies in faces the eye.
            var onAnyWall = false, facesEye = false
            for box in boxes {
                let walls: [(on: Bool, outward: SIMD3<Double>)] = [
                    (abs(p.x - box.lo.x) <= slack, SIMD3(-1, 0, 0)),
                    (abs(p.x - box.hi.x) <= slack, SIMD3(1, 0, 0)),
                    (abs(p.y - box.lo.y) <= slack, SIMD3(0, -1, 0)),
                    (abs(p.y - box.hi.y) <= slack, SIMD3(0, 1, 0)),
                ]
                for wall in walls where wall.on {
                    onAnyWall = true
                    if -simd_dot(wall.outward, sight) > 0 { facesEye = true }
                }
            }
            // Ink that stands in no wall is left to the march alone.
            return (onCrest || facesEye || !onAnyWall) && march.unoccluded(p)
        }
    }

    /// Ink that lies on the surface itself -- the contours of |f| and arg f,
    /// lifted onto it -- less what cannot be seen.
    ///
    /// Two tests, as a parametric surface has. The surface bounds the solid
    /// under it, so where it faces away from the eye the ink on it is hidden,
    /// full stop, and no margin has a say: this is the test a depth margin
    /// cannot make, since where a steep flank turns away its back lies within
    /// the margin of its front for a stretch, and the other family of
    /// contours would show through there as ticks along the silhouette.
    /// Where it faces the eye, the ink is hidden exactly where the sight
    /// line from it passes under the surface in front (`SightMarch`). A
    /// segment that crosses a fold is cut on the fold itself, found by
    /// bisection on the normal -- the same facing the fold lines are traced
    /// from, so a contour ends where the silhouette is drawn -- and the new
    /// vertex belongs to the run on the visible side, judged by the march
    /// alone, since on the fold the facing is zero by construction.
    public func visibleSurfaceInk(_ paths: PolylineSet<WorldSpace>,
                                  view: Transform<WorldSpace, ViewSpace>,
                                  margin: Double) -> PolylineSet<WorldSpace> {
        guard !paths.vertices.isEmpty else { return paths }
        let vis = HeightfieldVisibility(heightfield: self, view: view, margin: margin)
        let vertices = paths.vertices
        let facing = Self.parallelMap(vertices) { vis.facing(P2($0.x, $0.y)) }
        // The segments whose ends face opposite ways, each cut on the fold.
        var crossing: [Int] = []
        for path in 0..<paths.count {
            for k in (paths.offsets[path] + 1)..<paths.offsets[path + 1]
            where (facing[k - 1] > 0) != (facing[k] > 0) { crossing.append(k) }
        }
        let onFold = Self.parallelMap(crossing) { k -> P3<WorldSpace>? in
            let a = vertices[k - 1], b = vertices[k]
            guard let t = vis.foldCrossing(from: a.xy, facing[k - 1], to: b.xy, facing[k]) else {
                return nil
            }
            return P3(a.v + (b.v - a.v) * t)
        }
        var fold = [P3<WorldSpace>?](repeating: nil, count: vertices.count)
        for (i, k) in crossing.enumerated() { fold[k] = onFold[i] }
        let clear = Self.parallelMap(Array(vertices.indices)) { k in
            facing[k] > 0 && vis.isClear(vertices[k])
        }
        let foldClear = Self.parallelMap(fold) { $0.map { vis.isClear($0) } ?? false }

        var out: [[P3<WorldSpace>]] = []
        var run: [P3<WorldSpace>] = []
        func take(_ v: P3<WorldSpace>, _ visible: Bool) {
            if visible {
                run.append(v)
            } else {
                if run.count >= 2 { out.append(run) }
                run = []
            }
        }
        for path in 0..<paths.count {
            for k in paths.offsets[path]..<paths.offsets[path + 1] {
                if let f = fold[k] { take(f, foldClear[k]) }
                take(vertices[k], clear[k])
            }
            if run.count >= 2 { out.append(run) }
            run = []
        }
        return PolylineSet(paths: out)
    }

    /// Any other ink -- the rim of a cap, the hatch across it, a dumped
    /// scaffold -- less what the solid hides: judged by the march alone,
    /// since such ink lies on a crease or a flat top where the surface's
    /// facing is not the ink's.
    public func visibleInk(_ paths: PolylineSet<WorldSpace>,
                           view: Transform<WorldSpace, ViewSpace>,
                           margin: Double) -> PolylineSet<WorldSpace> {
        guard !paths.vertices.isEmpty else { return paths }
        let march = SightMarch(self, view: view, margin: margin)
        return Self.runs(of: paths) { march.unoccluded($0) }
    }

    /// The runs of each path whose vertices pass `keep`, split where one
    /// does not; runs shorter than two vertices draw nothing and are dropped.
    /// `keep` is asked of every vertex in parallel, since the march behind it
    /// is the expensive part of a bake.
    static func runs(of paths: PolylineSet<WorldSpace>,
                     keep: @Sendable (P3<WorldSpace>) -> Bool) -> PolylineSet<WorldSpace> {
        let kept = parallelMap(paths.vertices, keep)
        var out: [[P3<WorldSpace>]] = []
        for i in 0..<paths.count {
            var run: [P3<WorldSpace>] = []
            for k in paths.offsets[i]..<paths.offsets[i + 1] {
                if kept[k] {
                    run.append(paths.vertices[k])
                } else {
                    if run.count >= 2 { out.append(run) }
                    run = []
                }
            }
            if run.count >= 2 { out.append(run) }
        }
        return PolylineSet(paths: out)
    }

    /// `items.map(body)`, in parallel chunks.
    public static func parallelMap<T: Sendable, R: Sendable>(_ items: [T],
                                                      _ body: @Sendable (T) -> R) -> [R] {
        let n = items.count
        guard n > 0 else { return [] }
        let chunk = 512
        return [R](unsafeUninitializedCapacity: n) { buffer, initialized in
            // Each chunk writes only its own slots, so sharing is safe.
            nonisolated(unsafe) let base = buffer.baseAddress!
            DispatchQueue.concurrentPerform(iterations: (n + chunk - 1) / chunk) { c in
                for k in (c * chunk)..<min((c + 1) * chunk, n) {
                    (base + k).initialize(to: body(items[k]))
                }
            }
            initialized = n
        }
    }
}

/// A march up a sight line through a heightfield's solid: the question the
/// depth pass answers for every pixel, asked for one point, exactly.
///
/// Exactly, because the drawn surface is piecewise linear: the lattice's
/// cells, two triangles each, capped. Along a straight line the gap between
/// the line and that surface is therefore linear between the places where
/// the line crosses a lattice row, a lattice column or a cell's diagonal, and
/// its least value lies at one of them. So the march visits those crossings
/// and no other points -- in the lattice's own index coordinates, where the
/// rows, columns and diagonals are the integer levels of `u`, `v` and
/// `u + v` -- and a point is hidden when at some crossing beyond the margin
/// the line is under the surface. Nothing is sampled: a sliver thinner than
/// any step is still a crossing. The answer is a function of the geometry
/// alone, which is what makes a bake the same drawing at every depth
/// resolution.
///
/// The margin means two things, and the march keeps them apart. A point of
/// ink lies on a surface it was placed on by the function or by the grid,
/// and the drawn surface -- the lattice's facets -- misses that by an
/// interpolation gap, so the point may start a little inside the solid.
/// That is forgiven *perpendicular* to the facet, by the margin: a vertical
/// gap times the facet's cosine, still linear along the line over one
/// facet. Measured vertically it would be a horizontal tolerance of
/// margin / slope, and on a cone of slope 24 the rings came out dashed;
/// measured along the sight line it forgives nothing at a fold, where the
/// line runs along the surface for a long way within a hair of it, and the
/// folds vanished. Then, once the line has been outside the solid, anything
/// it enters is an occluder and is judged *along the sight line*, as the
/// depth buffer judged it: hidden when the line is still inside the solid
/// at the margin's distance toward the eye, or beyond. A wedge of rock
/// that is thin across but long along the line -- the side of a pit seen
/// through its notch -- hides what is behind it, as it should.
///
/// A point strictly over the domain is judged where it stands as well. A
/// point on a wall -- at the domain's edge -- is on the solid's boundary
/// and is judged only from where the line goes next, so the foot of a wall
/// that faces the eye, whose line leaves the domain at once, is seen.
///
/// Built once per camera, since it scans the samples for the solid's extent
/// and inverts the tiles.
struct SightMarch: Sendable {
    let heightfield: Heightfield
    let toward: SIMD3<Double>
    /// The hidden-line margin, in view depth. Infinite disarms the march.
    let margin: Double
    /// The margin as a distance along the sight line: an occluder must
    /// stand in front by more than this.
    let reach: Double
    /// The drawn lattice's spacing, the larger of the two directions.
    let cell: Double
    let top: Double
    let floor: Double
    /// Per tile: world to the tile's lattice index coordinates, and the
    /// lattice's last index each way.
    let lattices: [(linear: simd_double2x2, offset: SIMD2<Double>, extent: SIMD2<Double>)]
    /// Per tile: world back to the fundamental domain, for the nearest-tile
    /// height read (`Surface.inTile`, with the inverses taken once).
    let backs: [Affine2]
    let step: Int

    init(_ heightfield: Heightfield, view: Transform<WorldSpace, ViewSpace>, margin: Double) {
        self.heightfield = heightfield
        self.toward = -view.sightLine
        self.margin = margin
        // View depth per unit of travel along the sight line: the camera's
        // third row dotted with the direction, which under a sheared plate
        // camera is not one.
        let c = view.m.columns
        let perUnit = abs(simd_dot(SIMD3(c.0.z, c.1.z, c.2.z), toward))
        self.reach = margin / max(perUnit, 1e-12)
        let g = heightfield.surface.height
        let step = max(heightfield.step, 1)
        self.step = step
        let nx = (g.width + step - 1) / step, ny = (g.height + step - 1) / step
        let dx = g.domain.real.length / Double(max(g.width - 1, 1)) * Double(step)
        let dy = g.domain.imag.length / Double(max(g.height - 1, 1)) * Double(step)
        cell = max(abs(dx), abs(dy))
        var top = -Double.infinity, floor = Double.infinity
        heightfield.forEachSample { top = max(top, $0.z); floor = min(floor, $0.z) }
        for v in heightfield.occluder.vertices { floor = min(floor, v.z) }
        self.top = top; self.floor = floor
        var lattices: [(linear: simd_double2x2, offset: SIMD2<Double>, extent: SIMD2<Double>)] = []
        var backs: [Affine2] = []
        for tile in heightfield.tiles {
            let back = tile.inverse ?? .identity
            backs.append(back)
            lattices.append((
                linear: simd_double2x2(rows: [SIMD2(back.a / dx, back.b / dx),
                                              SIMD2(back.c / dy, back.d / dy)]),
                offset: SIMD2((back.tx - g.domain.real.lo) / dx, (back.ty - g.domain.imag.lo) / dy),
                extent: SIMD2(Double(nx - 1), Double(ny - 1))))
        }
        self.lattices = lattices; self.backs = backs
    }

    /// The drawn surface over a world point, read through the nearest tile:
    /// the lattice's interpolated height before the cap, the cap there, and
    /// the cosine of the facet's tilt. The drawn height is the lesser of the
    /// first two.
    func drawnSurface(at p: SIMD2<Double>) -> (interpolated: Double, cap: Double, cosine: Double) {
        let s = heightfield.surface
        guard backs.count > 1 else {
            return s.drawnSurface(at: P2<DomainSpace>(p.x, p.y), step: step)
        }
        let d = s.domain
        let (xlo, xhi) = (min(d.real.lo, d.real.hi), max(d.real.lo, d.real.hi))
        let (ylo, yhi) = (min(d.imag.lo, d.imag.hi), max(d.imag.lo, d.imag.hi))
        var best = P2<DomainSpace>(p.x, p.y)
        var bestDistance = Double.infinity
        for back in backs {
            let q = back(P3<WorldSpace>(p.x, p.y, 0))
            let distance = max(xlo - q.x, 0) + max(q.x - xhi, 0) + max(ylo - q.y, 0) + max(q.y - yhi, 0)
            if distance < bestDistance {
                bestDistance = distance
                best = P2<DomainSpace>(q.x, q.y)
                if distance == 0 { break }
            }
        }
        return s.drawnSurface(at: best, step: step)
    }

    /// Whether nothing of the solid stands between `p` and the eye.
    func unoccluded(_ p: P3<WorldSpace>) -> Bool {
        let d = toward
        let pxy = SIMD2(p.x, p.y), dxy = SIMD2(d.x, d.y)
        // How far up the line there can be anything: once above the highest
        // sample, or below the lowest, nothing is.
        var tMax = Double.infinity
        if d.z > 0 {
            tMax = max((top - p.z) / d.z, 0)
        } else if d.z < 0 {
            tMax = max((floor - p.z) / d.z, 0)
        } else if p.z > top || p.z < floor {
            return true
        }
        let tEps = 1e-9 * cell
        /// The line's height over the drawn surface at distance `t` along
        /// it: over the interpolated facet and over the cap, and the facet's
        /// cosine; nil where the point is outside the region, over which
        /// nothing is drawn.
        func gaps(_ t: Double) -> (overFacet: Double, overCap: Double, cosine: Double)? {
            let q = pxy + t * dxy
            guard heightfield.region.contains(P2<WorldSpace>(q.x, q.y)) else { return nil }
            let (interpolated, cap, cosine) = drawnSurface(at: q)
            let z = p.z + t * d.z
            return (z - interpolated, z - cap, cosine)
        }
        // The cap's bands, if any, as levels of the lattice's u where the
        // cap steps: a crossing of the drawn surface too.
        var bandEdges: [Double] = []
        if case .realBands(let bands, _) = heightfield.surface.caps {
            let g = heightfield.surface.height
            let dx = g.domain.real.length / Double(max(g.width - 1, 1)) * Double(step)
            bandEdges = bands.map { ($0.below - g.domain.real.lo) / dx }
        }
        // Every crossing of a lattice row, column or diagonal -- the
        // integer levels of u, v and u + v -- and of a cap band's edge, over
        // every tile the line passes, with the ends of each passage; and
        // the margin's reach, which is where occlusion starts to count.
        var ts: [Double] = []
        var strictlyInside = false
        for lattice in lattices {
            let u0 = lattice.linear * pxy + lattice.offset
            let du = lattice.linear * dxy
            var ta = 0.0, tb = tMax
            var misses = false
            for axis in 0..<2 {
                if abs(du[axis]) < 1e-15 {
                    if u0[axis] < 0 || u0[axis] > lattice.extent[axis] { misses = true }
                    continue
                }
                let t1 = -u0[axis] / du[axis], t2 = (lattice.extent[axis] - u0[axis]) / du[axis]
                ta = max(ta, min(t1, t2)); tb = min(tb, max(t1, t2))
            }
            guard !misses, ta <= tb, tb.isFinite else { continue }
            if u0.x > 1e-9, u0.x < lattice.extent.x - 1e-9,
               u0.y > 1e-9, u0.y < lattice.extent.y - 1e-9 { strictlyInside = true }
            ts.append(ta); ts.append(tb)
            for family in 0..<3 {
                let f0 = family < 2 ? u0[family] : u0.x + u0.y
                let df = family < 2 ? du[family] : du.x + du.y
                guard abs(df) > 1e-15 else { continue }
                let fa = f0 + ta * df, fb = f0 + tb * df
                let kLo = Int((min(fa, fb) + 1e-9).rounded(.up))
                let kHi = Int((max(fa, fb) - 1e-9).rounded(.down))
                if kLo <= kHi {
                    for k in kLo...kHi { ts.append((Double(k) - f0) / df) }
                }
                if family == 0 {
                    for level in bandEdges {
                        let t = (level - f0) / df
                        if t > ta, t < tb { ts.append(t) }
                    }
                }
            }
            if reach.isFinite, reach > ta, reach < tb { ts.append(reach) }
        }
        guard !ts.isEmpty else { return true }
        ts.sort()
        var unique: [Double] = []
        for t in ts where unique.last.map({ t - $0 > tEps }) ?? true { unique.append(t) }
        ts = unique
        let values = ts.map { gaps($0) }

        // The walk. In its own surface's neighbourhood -- from the start,
        // for as long as it has not yet been outside -- the line is
        // forgiven the margin perpendicular to the facet over it. Once it
        // has been outside, being inside at or beyond the margin's reach is
        // occlusion.
        var outsideYet = false
        var startJudged = false
        /// Hidden at `t`, judged under a facet of the given cosine; `g` is
        /// the gap there, nil beyond the region (outside, then).
        func hidden(_ t: Double, _ g: (overFacet: Double, overCap: Double, cosine: Double)?,
                    cosine: Double) -> Bool {
            var judged = t > tEps
            if !judged, strictlyInside, !startJudged { judged = true; startJudged = true }
            // At the start, a point on the surface to rounding is in its own
            // surface's neighbourhood, not outside it.
            let zEps = t > tEps ? 0 : 1e-9 * cell
            guard let g, g.overFacet < zEps, g.overCap < zEps else {
                // Outside here, the start included: a point on the boundary
                // that stands above the surface is outside from the start,
                // and what its line then enters is an occluder.
                outsideYet = true
                return false
            }
            guard judged else { return false }
            if outsideYet { return t + tEps >= reach }
            let depth = g.overFacet > g.overCap ? g.overFacet * cosine : g.overCap
            return depth + margin < 0
        }
        for k in 0..<ts.count {
            let t = ts[k], g = values[k]
            // A point on a lattice line lies under two facets: judged by
            // the facet behind it and the one ahead.
            let before = k > 0 ? gaps(0.5 * (ts[k - 1] + t))?.cosine : nil
            let after = k + 1 < ts.count ? gaps(0.5 * (t + ts[k + 1]))?.cosine : nil
            let cosines = [before, after].compactMap { $0 }
            for cosine in cosines.isEmpty ? [g?.cosine ?? 1] : cosines {
                if hidden(t, g, cosine: cosine) { return false }
            }
            // The rim between this crossing and the next, where the facet
            // meets the cap: a crease, judged under both.
            if k + 1 < ts.count, let ga = g, let gb = values[k + 1] {
                let ea = ga.overFacet - ga.overCap, eb = gb.overFacet - gb.overCap
                if (ea > 0) != (eb > 0), ea != eb {
                    let rim = t + (ts[k + 1] - t) * ea / (ea - eb)
                    if rim > tEps, let gr = gaps(rim) {
                        if hidden(rim, gr, cosine: after ?? gr.cosine) { return false }
                        if hidden(rim, gr, cosine: 1) { return false }
                    }
                }
            }
        }
        return true
    }
}

/// What the depth pass draws.
///
/// A sum rather than a surface with optional parts: a heightfield has tiles,
/// a region, walls and a cap, and a parametric surface has none of them --
/// what it has instead is a coordinate at every point, which is what its ink
/// is judged by. Each renderer, bound and bake says what it does with each.
public enum SceneGeometry: Sendable {
    case heightfield(Heightfield)
    case parametric(ParametricSurface)
}

/// An immutable snapshot of everything a frame needs.
///
/// The renderer's signature is `renderDepth(scene:frame:)`: a frame is a
/// function of a value, not of accumulated state. Moving the camera means
/// producing a `Scene` that differs in one field, and the GPU buffers behind it
/// are a memo keyed by `content`.
///
/// The geometry is `let` and the camera is `var`, deliberately. If the surface
/// could be reassigned in place, `content` would go stale and the memo would
/// serve the wrong texture -- so the type makes the memo's premise true rather
/// than documenting it. Producing different geometry means producing a new
/// `Scene`, which mints a new `ContentID`.
public struct Scene: Sendable {
    /// Identifies the *geometry*: everything the depth pass draws.
    public let content: ContentID
    /// Identifies the *ink*. Separate from `content` because editing a level
    /// set changes every stroke and none of the landscape, and rebuilding a
    /// hundred-megabyte height texture to move a contour would make the slider
    /// unusable on exactly the plates where it is most interesting.
    public let ink: ContentID
    public let geometry: SceneGeometry
    public let layers: [Layer]

    public var camera: Camera
    public var mode: PreviewMode
    /// Hidden-line margin, in world units of depth, for ink judged by depth.
    public var margin: Double

    /// A heightfield scene.
    public init(surface: Surface, occluder: Mesh<WorldSpace>, tiles: [Affine2],
                region: Region = .full, step: Int, layers: [Layer], camera: Camera,
                mode: PreviewMode = .plate, margin: Double) {
        self.init(content: ContentID(), ink: ContentID(),
                  geometry: .heightfield(Heightfield(surface: surface, occluder: occluder,
                                                     tiles: tiles, region: region, step: step)),
                  layers: layers, camera: camera, mode: mode, margin: margin)
    }

    /// A parametric surface, with ink on it.
    ///
    /// Ink that carries surface coordinates is judged by them; ink that does
    /// not falls back to depth, at `margin`.
    public init(surface: ParametricSurface, layers: [Layer], camera: Camera,
                mode: PreviewMode = .plate, margin: Double) {
        self.init(content: ContentID(), ink: ContentID(), geometry: .parametric(surface),
                  layers: layers, camera: camera, mode: mode, margin: margin)
    }

    /// The scene a bundle describes under one of its presets.
    public init(bundle: KurvenBundle, preset: CameraPreset) {
        self.init(surface: bundle.surface,
                  occluder: bundle.occluder(),
                  tiles: bundle.manifest.occluder.tiles,
                  region: bundle.manifest.occluder.region,
                  step: bundle.manifest.occluder.step,
                  layers: bundle.layers,
                  camera: .plate(preset.plate),
                  margin: preset.margin)
    }

    /// The heightfield, when that is what this scene draws.
    public var heightfield: Heightfield? {
        if case .heightfield(let h) = geometry { return h }
        return nil
    }

    /// The parametric surface, when that is what this scene draws.
    public var parametric: ParametricSurface? {
        if case .parametric(let s) = geometry { return s }
        return nil
    }

    /// The same content, looked at from somewhere else. Keeps both identities,
    /// so nothing at all is rebuilt -- which is the whole point of separating
    /// the camera from the content.
    public func looking(_ camera: Camera) -> Scene {
        var out = self; out.camera = camera; return out
    }

    /// The same landscape, drawn with different ink. Keeps `content` and mints
    /// a new `ink`, so the heightfield stays uploaded and only the line buffer
    /// is rebuilt.
    public func drawing(_ layers: [Layer]) -> Scene {
        Scene(content: content, ink: ContentID(), geometry: geometry,
              layers: layers, camera: camera, mode: mode, margin: margin)
    }

    private init(content: ContentID, ink: ContentID, geometry: SceneGeometry,
                 layers: [Layer], camera: Camera, mode: PreviewMode, margin: Double) {
        self.content = content; self.ink = ink; self.geometry = geometry
        self.layers = layers; self.camera = camera; self.mode = mode; self.margin = margin
    }

    /// Every layer's ink in view space, in declaration (draw) order, judged
    /// for this camera wherever the geometry can judge it exactly.
    ///
    /// Ink that depends on the camera -- the fold lines of a surface, of
    /// either kind -- is derived here, for this camera, rather than
    /// carried. On a heightfield every clipped layer is
    /// judged here, by the facing of what it lies on and by the solid
    /// (`SightMarch`), and the depth buffer has no part in it: the drawing is
    /// a function of the geometry and the camera, never of a resolution. A
    /// parametric surface's ink is judged by the bake, which has the
    /// surface's coordinate image; its fold lines come judged by
    /// construction. This is the bake's work, not the preview's: it is a
    /// march per vertex, and the preview keeps its depth test. `margin` is
    /// the hidden-line margin to judge by; the scene's own when not given.
    public func judgedLayers(margin: Double? = nil) -> [(Layer, PolylineSet<ViewSpace>)] {
        let view = camera.view
        let margin = margin ?? self.margin
        return layers.map { layer in
            switch geometry {
            case .parametric(let s):
                if case .foldLines = layer.spec.source {
                    return (layer, s.foldLines(sight: view.sightLine).mapped(view))
                }
                return (layer, layer.paths.mapped(view))
            case .heightfield(let h):
                let judged: PolylineSet<WorldSpace>
                if case .foldLines = layer.spec.source {
                    judged = h.visibleFolds(view: view, margin: margin)
                } else if !layer.spec.clipped {
                    judged = layer.paths
                } else if layer.spec.source.isWallInk {
                    judged = h.visibleWallInk(layer.paths, view: view, margin: margin)
                } else if layer.spec.liesOnSurface {
                    judged = h.visibleSurfaceInk(layer.paths, view: view, margin: margin)
                } else {
                    judged = h.visibleInk(layer.paths, view: view, margin: margin)
                }
                return (layer, judged.mapped(view))
            }
        }
    }

    /// The view-space extent the depth buffer covers: every heightfield sample,
    /// the walls, and every *clipped* layer.
    ///
    /// This is `ZBuffer(xs.min(), xs.max(), ys.min(), ys.max(), ...)` on the
    /// Python side, and it has to be the same extent or the two rasterize onto
    /// different pixel lattices and every comparison downstream is measuring the
    /// framing rather than the drawing. Two consequences of matching it exactly:
    ///
    /// - The heightfield is scanned sample by sample, not bounded by its box.
    ///   The box of a rotated landscape is looser than the hull of its samples,
    ///   and the difference is visible at bake resolution.
    /// - Masked-out samples still count. `build_occluder` emits every lattice
    ///   vertex and drops only *triangles*, so `occ_rot` includes the vertices
    ///   inside zeta's cutout even though nothing references them. Excluding
    ///   them here would be more principled and would not be the same picture.
    ///
    /// Unclipped ink is excluded, as it is in Python: it is drawn but never
    /// looked up, so letting it stretch the frame would spend depth resolution
    /// on nothing and would make the buffer depend on decoration.
    public func viewBounds() -> AABB<ViewSpace>? {
        var lo = SIMD3<Double>(repeating: .infinity)
        var hi = SIMD3<Double>(repeating: -.infinity)
        var any = false
        func add(_ p: P3<ViewSpace>) {
            lo = simd_min(lo, p.v); hi = simd_max(hi, p.v); any = true
        }
        switch geometry {
        case .heightfield(let h):
            for p in h.occluder.vertices { add(camera.view(p)) }
            h.forEachSample { add(camera.view($0)) }
        case .parametric(let s):
            for p in s.positions { add(camera.view(p)) }
        }
        // The ink as carried, before any judging: judging is the bake's
        // expensive step and would be paid twice, and ink the judge drops is
        // still ink the frame was sized to.
        for layer in layers where layer.spec.clipped {
            for v in layer.paths.vertices { add(camera.view(v)) }
        }
        // A parametric surface's folds are placed on the map, not the
        // lattice, and can lie a hair outside its hull: they are derived here
        // so the frame holds them, since they are judged by the frame.
        if let s = parametric, layers.contains(where: { layer in
            if case .foldLines = layer.spec.source { return true }
            return false
        }) {
            for v in s.foldLines(sight: camera.view.sightLine).vertices { add(camera.view(v)) }
        }
        return any ? AABB(lo: lo, hi: hi) : nil
    }

    /// A cheap approximate view-space bound, from a coarse subsample.
    ///
    /// `viewBounds()` folds over every heightfield sample because the bake has
    /// to frame the picture exactly the way Python does. Nothing interactive can
    /// afford that -- a "fit to window" that scans six million points is not a
    /// fit, it is a stall -- and nothing interactive needs it, because a frame
    /// that is a fraction of a percent loose is a frame nobody can see is loose.
    ///
    /// The bounding *box* would be cheaper still and is much worse: the box of a
    /// rotated landscape is far bigger than the hull of its samples, so fitting
    /// to it leaves the picture small and off-centre. Subsampling keeps the
    /// shape of the hull and only loses its last few percent.
    public func quickBounds(budget: Int = 20_000) -> AABB<ViewSpace>? {
        var lo = SIMD3<Double>(repeating: .infinity)
        var hi = SIMD3<Double>(repeating: -.infinity)
        var any = false
        func add(_ p: P3<ViewSpace>) {
            lo = simd_min(lo, p.v); hi = simd_max(hi, p.v); any = true
        }

        switch geometry {
        case .heightfield(let h):
            let g = h.surface.height
            let perTile = max(budget / max(h.tiles.count, 1), 16)
            let side = max(Int(Double(perTile).squareRoot()), 4)
            let stride = max(h.step, max(g.width / side, g.height / side))
            for tile in h.tiles {
                h.surface.forEachSample(step: stride) { add(camera.view(tile($0))) }
            }
            for v in h.occluder.vertices { add(camera.view(v)) }
        case .parametric(let s):
            let stride = max(s.positions.count / max(budget, 1), 1)
            for i in Swift.stride(from: 0, to: s.positions.count, by: stride) {
                add(camera.view(s.positions[i]))
            }
        }
        return any ? AABB(lo: lo, hi: hi) : nil
    }
}

/// What a preview draws.
public enum PreviewMode: Sendable, Equatable {
    /// The plate: white occluding surface, black hidden-line ink.
    case plate
    /// A lit heightfield, for orientation.
    case shaded(Lighting)
    /// The depth attachment itself -- the substitute for a frame-capture
    /// viewer, and the reason not having Xcode costs nothing here.
    case depth
}

public struct Lighting: Sendable, Equatable {
    public var direction: SIMD3<Float>
    public var ambient: Float
    public init(direction: SIMD3<Float> = SIMD3(0.4, -0.6, 0.7), ambient: Float = 0.25) {
        self.direction = simd_normalize(direction); self.ambient = ambient
    }
}
