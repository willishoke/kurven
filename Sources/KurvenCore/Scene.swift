import Foundation
import Synchronization
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
    /// Wall curtains, and the truncation edges where the heightfield's own triangles
    /// cut the corner (`Mesh.truncationEdgeMesh`), in world coordinates.
    public let occluder: Mesh<WorldSpace>
    /// Heightfield instances; always at least the identity.
    public let tiles: [Affine2]
    /// The footprint each instance is clipped to.
    public let region: Region
    /// Subsampling step for the heightfield when it is rasterized.
    public let step: Int
    /// The function behind the grids, when a consumer attached one to the
    /// bundle: the fold curves are carried to its zeros.
    public let refine: ContourRefine?

    public init(surface: Surface, occluder: Mesh<WorldSpace>, tiles: [Affine2],
                region: Region = .full, step: Int, refine: ContourRefine? = nil) {
        self.surface = surface; self.occluder = occluder; self.tiles = tiles
        self.region = region; self.step = step; self.refine = refine
    }

    /// The same heightfield for one thread of a parallel visibility test: its
    /// function forked (`ContourRefine.fork`), its grids shared.
    public func forked() -> Heightfield {
        Heightfield(surface: surface, occluder: occluder, tiles: tiles, region: region,
                    step: step, refine: refine?.forked())
    }

    /// The same heightfield without its function: traced and tested on the
    /// grid, whose chords are a height read each where f is an
    /// evaluation. What a preview can afford for a camera that is still
    /// moving.
    public var onGrid: Heightfield {
        Heightfield(surface: surface, occluder: occluder, tiles: tiles,
                    region: region, step: step, refine: nil)
    }

    /// Every vertex of the decimated, truncated heightfield, once per tile.
    ///
    /// The grid is `Surface.grid_mesh`'s `self.clamped[::step, ::step]`. Not
    /// materialized: elliptic's is three million points, zeta's six, and the
    /// caller only ever folds over them.
    public func forEachSample(_ body: (P3<WorldSpace>) -> Void) {
        for tile in tiles {
            surface.forEachSample(step: step) { body(tile($0)) }
        }
    }

    /// The outward normal of the truncated surface over a world point:
    /// `(-∂h/∂x, -∂h/∂y, 1)`, not normalized, by central differences a grid
    /// spacing wide, the height read through whichever tile the point is in.
    ///
    /// The solid is what lies under the graph, so outward is up, and a
    /// line of sight that reaches a point of the surface facing away from the
    /// eye has entered the solid already: such a point is hidden, whatever
    /// the depth buffer says to within its depth tolerance. That is the test the
    /// depth tolerance cannot make -- where a steep graph turns away, the back of it
    /// lies within the depth tolerance of the front for a stretch that zoom
    /// magnifies, and back-face curves show there as ticks.
    ///
    /// Differences in world coordinates, so a tile's reflection is already
    /// in them; on the plateau the differences vanish and the normal is up.
    ///
    /// With the function at hand (`refine`), the slope is f's own, read by
    /// differences a step small against the cell and large against
    /// rounding, as the refiner reads it: the curves on the surface are placed
    /// on f to the last bit, and a facing read off the grid a cell wide
    /// tests it with a ruler a cell long -- a ring ends a stencil's width
    /// short of the fold on a convex graph and shows a stencil's width past
    /// it on a hump, whatever the grid. Without the function, the grid
    /// is the surface and its chords are the slope.
    public func normal(at p: P2<WorldSpace>) -> SIMD3<Double> {
        let g = surface.height
        let dx = abs(g.domain.real.length) / Double(max(g.width - 1, 1))
        let dy = abs(g.domain.imag.length) / Double(max(g.height - 1, 1))
        if let magnitude = refine?.magnitude {
            let step = 1e-4 * max(dx, dy)
            func h(_ x: Double, _ y: Double) -> Double {
                let q = surface.inTile(P2<DomainSpace>(x, y), tiles: tiles)
                let u = magnitude(q)
                return min(u.isNaN ? .infinity : u, surface.truncation.height(atX: q.x))
            }
            let hx = (h(p.x + step, p.y) - h(p.x - step, p.y)) / (2 * step)
            let hy = (h(p.x, p.y + step) - h(p.x, p.y - step)) / (2 * step)
            if hx.isFinite, hy.isFinite { return SIMD3(-hx, -hy, 1) }
        }
        func h(_ x: Double, _ y: Double) -> Double {
            surface.height(at: P2<DomainSpace>(x, y), tiles: tiles)
        }
        let hx = (h(p.x + dx, p.y) - h(p.x - dx, p.y)) / (2 * dx)
        let hy = (h(p.x, p.y + dy) - h(p.x, p.y - dy)) / (2 * dy)
        return SIMD3(-hx, -hy, 1)
    }

    /// The normal of the graph over `p`: f's own slope, untruncated, read as
    /// `normal(at:)` reads it, so a fold is traced on the one surface the
    /// graph is and continues below the truncation, where the truncation edge cuts it. The
    /// truncated normal without the function, or where f's slope is not
    /// finite.
    func graphNormal(at p: P2<WorldSpace>) -> SIMD3<Double> {
        guard let magnitude = refine?.magnitude else { return normal(at: p) }
        let g = surface.height
        let dx = abs(g.domain.real.length) / Double(max(g.width - 1, 1))
        let dy = abs(g.domain.imag.length) / Double(max(g.height - 1, 1))
        let step = 1e-4 * max(dx, dy)
        func h(_ x: Double, _ y: Double) -> Double {
            let u = magnitude(surface.inTile(P2<DomainSpace>(x, y), tiles: tiles))
            return u.isNaN ? .infinity : u
        }
        let hx = (h(p.x + step, p.y) - h(p.x - step, p.y)) / (2 * step)
        let hy = (h(p.x, p.y + step) - h(p.x, p.y - step)) / (2 * step)
        guard hx.isFinite, hy.isFinite else { return normal(at: p) }
        return SIMD3(-hx, -hy, 1)
    }

    /// The fold curves under a camera: where the surface turns edge-on,
    /// `n · sight = 0`, which is where its outline and every inner outline
    /// lie -- the far edge of a pit, the top edge of a graph seen from the side.
    /// The 1933 plates draw them, and the notebooks drew them from a
    /// thresholded matrix of gradients, contoured. This is that, with the
    /// line of sight as the threshold: the facing is sampled on the grid,
    /// its zero level traced by marching squares, and, when `refined`, every
    /// vertex moved onto the exact fold by bisection along the facing's
    /// gradient -- the same facing `HeightfieldVisibility` cuts curves by, so
    /// the two agree on where the surface turns. Nothing on a plateau, which is
    /// flat and faces up and has a truncation edge of its own, and nothing outside the
    /// region. They depend on the camera, so they are derived for one, never
    /// stored.
    ///
    /// A fold that ends beside a zero of f was going there: the outline
    /// of a pit runs down its graph into the zero, the apex of the cone.
    /// The grid loses it a cell or two short, where its chord across the
    /// floor flattens the facing out. With the function at hand (`refine`),
    /// an open run's end is carried to the zero the refiner finds from it,
    /// at |f|'s own height there -- the floor -- as the phase lines are.
    public func foldCurves(view: Transform<WorldSpace, ViewSpace>,
                          refined: Bool = true) -> PolylineSet<WorldSpace> {
        let g = surface.height
        let nx = g.width, ny = g.height
        guard nx >= 2, ny >= 2 else { return .empty }
        let cell = max(abs(g.domain.real.length) / Double(nx - 1),
                       abs(g.domain.imag.length) / Double(ny - 1))
        /// A thread's placer: its own fork of this heightfield, on the
        /// tile's grid.
        @Sendable func placer(lo: SIMD2<Double>, sx: Double, sy: Double) -> FoldPlacer {
            FoldPlacer(heightfield: forked(), sight: view.projectionDirection, cell: cell, refined: refined,
                       lo: lo, sx: sx, sy: sy)
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
            let worker: @Sendable () -> FoldPlacer = { placer(lo: lo, sx: sx, sy: sy) }
            // A row at a time, each worker with its own f (`forked`): every
            // node's facing is three evaluations of f.
            let rows = Self.parallelMap(Array(0..<ny), chunk: 1, worker: worker) { w, j in
                (0..<nx).map { i in
                    Float(w.facing(P2(lo.x + Double(i) * sx, lo.y + Double(j) * sy)))
                }
            }
            let values = Array(rows.joined())
            let field = Grid2D(width: nx, height: ny,
                               domain: Domain(real: Interval(lo: lo.x, hi: hi.x),
                                              imag: Interval(lo: lo.y, hi: hi.y)),
                               values: values)
            // Only on the fundamental tile: the zero is found on the
            // function, which is sampled there.
            let fundamental = tile == .identity
            paths += worker().place(Contour.lines(of: field, level: 0), fundamental: fundamental,
                                    worker: worker)
        }
        return PolylineSet(paths: paths)
    }

    /// The height of the surface the curves are tested against, over a world
    /// point: f's own, truncated, read through the tile the point is in, when
    /// the function is at hand; the grid's interpolated height otherwise.
    public func surfaceHeight(at p: P2<WorldSpace>) -> Double {
        if let magnitude = refine?.magnitude {
            let q = surface.inTile(P2<DomainSpace>(p.x, p.y), tiles: tiles)
            let u = magnitude(q)
            if u.isFinite { return min(u, surface.truncation.height(atX: q.x)) }
        }
        return surface.height(at: P2<DomainSpace>(p.x, p.y), tiles: tiles)
    }

    /// The height of the surface *as the depth pass draws it* over a world
    /// point: the grid decimated by `step`, each cell as the two triangles
    /// the rasterizer splits it into, truncated, through the tile the point is
    /// in. This is the solid every curve is tested against, so the
    /// test and the rasterizer cannot disagree about where the surface is.
    public func drawnHeight(at p: P2<WorldSpace>) -> Double {
        surface.drawnHeight(at: surface.inTile(P2<DomainSpace>(p.x, p.y), tiles: tiles), step: step)
    }

    /// An open fold run carried to the zero of f either end leads to, when
    /// the refiner finds one within reach that the run does not already
    /// touch, inside the region: the new vertex is the zero at |f|'s height.
    /// `cell` is the grid spacing, which is the only length the test of
    /// "already touches" is measured against, so a scaled drawing carries the
    /// scaled fold.
    func endedAtZeros(_ run: [P3<WorldSpace>], cell: Double) -> [P3<WorldSpace>] {
        guard let refine, let zero = refine.zero, let magnitude = refine.magnitude,
              run.count >= 2, run.first != run.last else { return run }
        func carried(_ end: P3<WorldSpace>, other: P3<WorldSpace>) -> P3<WorldSpace>? {
            guard let z = zero(P2(end.x, end.y)), region.contains(P2<WorldSpace>(z.x, z.y)) else { return nil }
            let toEnd = ((z.x - end.x) * (z.x - end.x) + (z.y - end.y) * (z.y - end.y)).squareRoot()
            let toOther = ((z.x - other.x) * (z.x - other.x) + (z.y - other.y) * (z.y - other.y)).squareRoot()
            guard toEnd > 1e-9 * cell, toEnd < toOther else { return nil }
            let u = magnitude(z)
            guard u.isFinite else { return nil }
            return P3(z.x, z.y, min(u, surface.truncation.height(atX: z.x)))
        }
        var out = run
        if let z = carried(out[0], other: out[out.count - 1]) { out.insert(z, at: 0) }
        if let z = carried(out[out.count - 1], other: out[0]) { out.append(z) }
        return out
    }

    /// The fold curves that can be seen: `foldCurves`, less what the surface
    /// hides. A fold cannot be tested by a depth buffer: the surface is
    /// edge-on there, so its depth changes by the whole graph within one
    /// pixel, and the pixel's value is the near side's, in front of the fold
    /// by far more than any depth tolerance. It is tested by the solid instead: a fold
    /// vertex is hidden when the line of sight from it to the eye passes under
    /// the surface (`GridTraversal`). `depthTolerance` is the drawing's hidden-line
    /// depth tolerance, which keeps a vertex on the surface it belongs to from hiding
    /// behind that surface's own rounding.
    ///
    /// With the function at hand, curves on the surface are tested where the
    /// surface is (`onSurface`) and the depth tolerance is rounding: a fold vertex
    /// lies on f to the last bit, and forgiving the tolerance instead draws
    /// a stub of a bowl's fold beside the cut, hidden by less than the
    /// tolerance, almost on the top edge.
    ///
    /// `resolution` is how finely a fold's end behind the solid is placed,
    /// as a length in the view plane (`runs(of:)`); zero, to the last bit.
    public func visibleFolds(view: Transform<WorldSpace, ViewSpace>, depthTolerance: Double,
                             refined: Bool = true, resolution: Double = 0) -> PolylineSet<WorldSpace> {
        let folds = foldCurves(view: view, refined: refined)
        guard !folds.vertices.isEmpty else { return folds }
        let traversal = GridTraversal(self, view: view, depthTolerance: surfaceDepthTolerance(depthTolerance))
        return Self.runs(of: folds, worker: {
            let m = traversal.forked(), h = m.heightfield
            return (holds: { m.unoccluded(h.onSurface($0)) }, lift: { _, _, q in q })
        }, stop: Self.stop(resolution, view))
    }

    /// The fraction of a segment `a -> b` to which `turning` bisects for curves
    /// placed to `resolution` in the view plane: the segment's own length on
    /// the screen, in resolutions, inverted. A preview places a run's end to
    /// a fraction of its pixel, and a traversal per halving is milliseconds on
    /// f; bisected to the last bit, as a bake is, a fold of cn cost three
    /// quarters of a second a camera for ends no pixel can show.
    static func stop(_ resolution: Double, _ view: Transform<WorldSpace, ViewSpace>)
        -> @Sendable (P3<WorldSpace>, P3<WorldSpace>) -> Double
    {
        guard resolution > 0 else { return { _, _ in 0 } }
        return { a, b in
            let pa = view(a), pb = view(b)
            let length = ((pb.x - pa.x) * (pb.x - pa.x) + (pb.y - pa.y) * (pb.y - pa.y)).squareRoot()
            return length > 0 ? resolution / length : 0
        }
    }

    /// The depth tolerance curves on the surface are tested with: the drawing's against the
    /// grid, whose chords are that far from f; rounding against f, since
    /// the curves are tested on f itself.
    func surfaceDepthTolerance(_ depthTolerance: Double) -> Double {
        guard refine?.magnitude != nil else { return depthTolerance }
        let g = surface.height
        let cell = max(abs(g.domain.real.length) / Double(max(g.width - 1, 1)),
                       abs(g.domain.imag.length) / Double(max(g.height - 1, 1)))
        return 1e-9 * cell
    }

    /// A point of curves on the surface, at the surface's own height over it:
    /// where it is tested. A ring's vertex is on f to a hundredth of the
    /// tolerance and a point of its chord to the tolerance; both are tested
    /// on f, so that a point of a ring at a fold is tested exactly as the
    /// fold is, and the ring's end and the fold's end agree.
    func onSurface(_ p: P3<WorldSpace>) -> P3<WorldSpace> {
        guard refine?.magnitude != nil else { return p }
        let z = surfaceHeight(at: P2(p.x, p.y))
        return z.isFinite ? P3(p.x, p.y, z) : p
    }

    /// Curves that lie in a cut face -- the wall's hatch and its outline --
    /// less what cannot be seen. A wall is a plane the oblique camera often
    /// sees nearly edge-on, and the depth buffer cannot test curves on such a
    /// face: one pixel spans the whole receding wall, keeps the near side's
    /// depth, and erases most of the hatch, leaving spots. So wall curves are
    /// tested as the folds are: hidden where the wall it lies in faces away
    /// from the eye -- it is then behind the solid the wall bounds -- and
    /// otherwise hidden only where the line of sight from it passes under the
    /// surface. A vertical edge at a corner lies in two walls and shows if either
    /// does. Which wall a vertex lies in is read off its position against
    /// the box of each tile, since every wall of a described landscape
    /// stands on a tile's edge.
    ///
    /// The top edge is the exception: it is the surface's own edge as much as
    /// the wall's top, and the far edge of a drawing is in plain view over a
    /// wall that faces away. A vertex at the surface's height is tested by
    /// the traversal alone, which hides it exactly when the surface in front of
    /// it does.
    public func visibleWallCurves(_ paths: PolylineSet<WorldSpace>,
                               view: Transform<WorldSpace, ViewSpace>,
                               depthTolerance: Double, resolution: Double = 0) -> PolylineSet<WorldSpace> {
        guard !paths.vertices.isEmpty else { return paths }
        let sight = view.projectionDirection
        let traversal = GridTraversal(self, view: view, depthTolerance: depthTolerance)
        let d = surface.domain
        let slack = 1e-6 * traversal.cell
        let boxes: [(lo: SIMD2<Double>, hi: SIMD2<Double>)] = tiles.map { tile in
            let corners = [(d.real.lo, d.imag.lo), (d.real.hi, d.imag.lo),
                           (d.real.lo, d.imag.hi), (d.real.hi, d.imag.hi)]
                .map { tile(P3<WorldSpace>($0.0, $0.1, 0)) }
            return (SIMD2(corners.map(\.x).min()!, corners.map(\.y).min()!),
                    SIMD2(corners.map(\.x).max()!, corners.map(\.y).max()!))
        }
        let cell = traversal.cell
        /// Whether the point is tested by the traversal at all: on the top edge, in
        /// a wall that faces the eye, or in no wall. Curves that stands in no
        /// wall is left to the traversal alone.
        @Sendable func onTopEdge(_ hf: Heightfield, _ p: P3<WorldSpace>) -> Bool {
            let h = hf.surfaceHeight(at: P2(p.x, p.y))
            return p.z >= h - 1e-9 * max(cell, abs(h))
        }
        @Sendable func admitted(_ hf: Heightfield, _ p: P3<WorldSpace>) -> Bool {
            let onTopEdge = onTopEdge(hf, p)
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
            return onTopEdge || facesEye || !onAnyWall
        }
        // A top edge run ends where the traversal turns along the top edge itself: the
        // chord between two top edge vertices lies under a concave top edge, and a
        // point under the top edge is in the wall, which, facing away, is
        // never admitted -- so bisected along the chord the run began at
        // the next vertex, half a cell late, and the far edge of a drawing
        // came out from behind a tower with a stub missing beside the
        // outline's cusp. Visible on the surface, the chord's points are
        // top edge points.
        return Self.runs(of: paths, worker: {
            let m = traversal.forked(), h = m.heightfield
            return (holds: { p in admitted(h, p) && m.unoccluded(p) },
                    lift: { a, b, q in onTopEdge(h, a) && onTopEdge(h, b) ? h.onSurface(q) : q })
        }, bisectable: { a, b in admitted(self, a) && admitted(self, b) },
           stop: Self.stop(resolution, view))
    }

    /// Curves that lie on the surface itself -- the contours of |f| and arg f,
    /// lifted onto it -- less what cannot be seen.
    ///
    /// Two tests, as a parametric surface has. The surface bounds the solid
    /// under it, so where it faces away from the eye the curves on it are hidden,
    /// full stop, and no depth tolerance has a say: this is the test a depth depth tolerance
    /// cannot make, since where a steep graph turns away its back lies within
    /// the depth tolerance of its front for a stretch, and the other family of
    /// contours would show through there as ticks along the outline.
    /// Where it faces the eye, the curves are hidden exactly where the sight
    /// line from it passes under the surface in front (`GridTraversal`). A
    /// segment that crosses a fold is cut on the fold itself, found by
    /// bisection on the normal -- the same facing the fold curves are traced
    /// from, so a contour ends where the outline is drawn -- and the new
    /// vertex belongs to the run on the visible side, tested by the traversal
    /// alone, since on the fold the facing is zero by construction.
    public func visibleSurfaceCurves(_ paths: PolylineSet<WorldSpace>,
                                  view: Transform<WorldSpace, ViewSpace>,
                                  depthTolerance: Double) -> PolylineSet<WorldSpace> {
        guard !paths.vertices.isEmpty else { return paths }
        let vis = HeightfieldVisibility(heightfield: self, view: view, depthTolerance: surfaceDepthTolerance(depthTolerance))
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
            facing[k] > 0 && vis.isClear(self.onSurface(vertices[k]))
        }
        let foldClear = Self.parallelMap(fold) { $0.map { vis.isClear(self.onSurface($0)) } ?? false }

        // The sequence each path is tested as: its vertices with the fold
        // crossings between them, each with its verdict and whether the
        // traversal alone decided it (a back-facing vertex is hidden by the
        // fold, and the run already ends on the fold crossing).
        var sequence: [[(v: P3<WorldSpace>, visible: Bool, byTraversal: Bool)]] = []
        for path in 0..<paths.count {
            var seq: [(v: P3<WorldSpace>, visible: Bool, byTraversal: Bool)] = []
            for k in paths.offsets[path]..<paths.offsets[path + 1] {
                if let f = fold[k] { seq.append((f, foldClear[k], true)) }
                seq.append((vertices[k], clear[k], facing[k] > 0))
            }
            sequence.append(seq)
        }
        // Where a run ends between a visible point and one the traversal hides,
        // it ends where the traversal turns, by bisection along the segment: a
        // ring passing behind a top edge ends on the top edge as drawn, not a
        // segment short of it.
        var crossings: [(path: Int, k: Int)] = []
        for (i, seq) in sequence.enumerated() {
            for k in 1..<max(seq.count, 1) where seq[k - 1].visible != seq[k].visible
                && (seq[k - 1].visible ? seq[k].byTraversal : seq[k - 1].byTraversal) {
                crossings.append((i, k))
            }
        }
        let visibilitySequence = sequence
        let turned = Self.parallelMap(crossings) { c -> P3<WorldSpace>? in
            let a = visibilitySequence[c.path][c.k - 1], b = visibilitySequence[c.path][c.k]
            return Self.turning(from: a.v, a.visible, to: b.v, holds: { vis.isClear(self.onSurface($0)) })
        }
        var turn: [[P3<WorldSpace>?]] = sequence.map { [P3<WorldSpace>?](repeating: nil, count: $0.count) }
        for (i, c) in crossings.enumerated() { turn[c.path][c.k] = turned[i] }

        var out: [[P3<WorldSpace>]] = []
        var run: [P3<WorldSpace>] = []
        for (i, seq) in sequence.enumerated() {
            for (k, item) in seq.enumerated() {
                if item.visible {
                    if let t = turn[i][k] { run.append(t) }
                    run.append(item.v)
                } else {
                    if let t = turn[i][k] { run.append(t) }
                    if run.count >= 2 { out.append(run) }
                    run = []
                }
            }
            if run.count >= 2 { out.append(run) }
            run = []
        }
        return PolylineSet(paths: out)
    }

    /// Any other curves -- the truncation edge of a truncation, the hatch across it, a dumped
    /// scaffold -- less what the solid hides: tested by the traversal alone,
    /// since such curves lie on a crease or a flat top where the surface's
    /// facing is not the curves'.
    public func visibleCurves(_ paths: PolylineSet<WorldSpace>,
                           view: Transform<WorldSpace, ViewSpace>,
                           depthTolerance: Double) -> PolylineSet<WorldSpace> {
        guard !paths.vertices.isEmpty else { return paths }
        let traversal = GridTraversal(self, view: view, depthTolerance: depthTolerance)
        return Self.runs(of: paths, worker: {
            let m = traversal.forked()
            return (holds: { m.unoccluded($0) }, lift: { _, _, q in q })
        })
    }

    /// The runs of each path whose vertices pass `holds`, split where one
    /// does not; runs shorter than two vertices draw nothing and are dropped.
    /// `holds` is asked of every vertex in parallel, since the traversal behind it
    /// is the expensive part of a bake.
    ///
    /// Where a run ends between a kept vertex and a dropped one, it ends
    /// where `holds` turns, found by bisection along the segment to the last
    /// bit: a ring passing behind a top edge ends on the top edge as drawn, not at
    /// its last vertex in front of it. Only across a segment `bisectable`
    /// admits: the traversal's verdict turns at a point of the segment, but a
    /// test of where the vertex *is* -- in a wall that faces the eye, on the
    /// top edge -- does not, and bisecting it would draw the sliver of a stroke
    /// between the top edge as drawn and f.
    /// `lift` places a point of the chord `a -> b` where the curves between
    /// the two vertices actually runs, for the bisection of a run's end:
    /// for most curves the chord itself. `stop` is the fraction of the segment
    /// the bisection may stop at (`stop(_:_:)`): none, by default.
    ///
    /// `holds` and `lift` are each worker thread's own, made by `worker` once
    /// a thread (`parallelMap(_:chunk:worker:_:)`): a traversal forked for the
    /// thread shares nothing of f with the others.
    static func runs(of paths: PolylineSet<WorldSpace>,
                     worker: @Sendable () -> (holds: Predicate, lift: Lift),
                     bisectable: @Sendable (P3<WorldSpace>, P3<WorldSpace>) -> Bool = { _, _ in true },
                     stop: @Sendable (P3<WorldSpace>, P3<WorldSpace>) -> Double = { _, _ in 0 })
        -> PolylineSet<WorldSpace>
    {
        // A few vertices to a thread at a time, and one crossing: a predicate that
        // traverses a line of sight on f is milliseconds a vertex, and at the
        // default chunk a fold of a thousand vertices was four threads' work.
        let kept = parallelMap(paths.vertices, chunk: 4, worker: worker) { w, v in w.holds(v) }
        let vertices = paths.vertices
        var crossings: [Int] = []
        for path in 0..<paths.count {
            for k in (paths.offsets[path] + 1)..<paths.offsets[path + 1]
            where kept[k - 1] != kept[k] && bisectable(vertices[k - 1], vertices[k]) {
                crossings.append(k)
            }
        }
        let turned = parallelMap(crossings, chunk: 1, worker: worker) { w, k -> P3<WorldSpace>? in
            let a = vertices[k - 1], b = vertices[k]
            return Self.turning(from: a, kept[k - 1], to: b, holds: w.holds, until: stop(a, b)) {
                w.lift(a, b, $0)
            }
        }
        var turn = [P3<WorldSpace>?](repeating: nil, count: vertices.count)
        for (i, k) in crossings.enumerated() { turn[k] = turned[i] }
        var out: [[P3<WorldSpace>]] = []
        for i in 0..<paths.count {
            var run: [P3<WorldSpace>] = []
            for k in paths.offsets[i]..<paths.offsets[i + 1] {
                if kept[k] {
                    if let t = turn[k] { run.append(t) }
                    run.append(paths.vertices[k])
                } else {
                    if let t = turn[k] { run.append(t) }
                    if run.count >= 2 { out.append(run) }
                    run = []
                }
            }
            if run.count >= 2 { out.append(run) }
        }
        return PolylineSet(paths: out)
    }

    typealias Predicate = @Sendable (P3<WorldSpace>) -> Bool
    typealias Lift = @Sendable (P3<WorldSpace>, P3<WorldSpace>, P3<WorldSpace>) -> P3<WorldSpace>

    /// Where `holds` turns on the segment from `a` to `b`, given it holds at
    /// `a` iff `keptA`: bisection to the last bit of the fraction, or to
    /// `until` of it, returning the last point at which it still holds on
    /// the kept side.
    static func turning(from a: P3<WorldSpace>, _ keptA: Bool, to b: P3<WorldSpace>,
                        holds: @Sendable (P3<WorldSpace>) -> Bool, until: Double = 0,
                        lift: (P3<WorldSpace>) -> P3<WorldSpace> = { $0 }) -> P3<WorldSpace>? {
        var lo = 0.0, hi = 1.0
        while hi - lo > max(until, 1e-13) {
            let mid = 0.5 * (lo + hi)
            if holds(lift(P3(a.v + (b.v - a.v) * mid))) == keptA { lo = mid } else { hi = mid }
        }
        // Turned within rounding of the kept end: the kept vertex is where
        // the curve ends, and a second vertex there would be a run of nothing.
        let t = keptA ? lo : 1 - hi
        guard t > 1e-12 else { return nil }
        return lift(P3(a.v + (b.v - a.v) * (keptA ? lo : hi)))
    }

    /// `items.map(body)`, in parallel chunks.
    /// `chunk` items go to a thread at a time: hundreds for a vertex's
    /// worth of work, one for a level's.
    public static func parallelMap<T: Sendable, R: Sendable>(_ items: [T], chunk: Int = 512,
                                                      _ body: @Sendable (T) -> R) -> [R] {
        let n = items.count
        guard n > 0 else { return [] }
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

    /// `items.map(body)` in parallel, each worker thread with a state of its
    /// own, made by `worker` once a thread rather than once a chunk: a
    /// thread's forked traversal (`Heightfield.forked`). Chunks are handed out
    /// as threads come free, so a slow vertex holds up only its own chunk.
    public static func parallelMap<T: Sendable, R: Sendable, W>(
        _ items: [T], chunk: Int, worker: @Sendable () -> W, _ body: @Sendable (W, T) -> R
    ) -> [R] {
        let n = items.count
        guard n > 0 else { return [] }
        let chunks = (n + chunk - 1) / chunk
        let workers = min(chunks, ProcessInfo.processInfo.activeProcessorCount)
        let next = Atomic<Int>(0)
        return [R](unsafeUninitializedCapacity: n) { buffer, initialized in
            // Each chunk writes only its own slots, so sharing is safe.
            nonisolated(unsafe) let base = buffer.baseAddress!
            DispatchQueue.concurrentPerform(iterations: workers) { _ in
                let state = worker()
                while true {
                    let c = next.wrappingAdd(1, ordering: .relaxed).oldValue
                    guard c < chunks else { break }
                    for k in (c * chunk)..<min((c + 1) * chunk, n) {
                        (base + k).initialize(to: body(state, items[k]))
                    }
                }
            }
            initialized = n
        }
    }
}

/// A traversal up a line of sight through a heightfield's solid: the question the
/// depth pass answers for every pixel, asked for one point, exactly.
///
/// Exactly, because the drawn surface is piecewise linear: the grid's
/// cells, two triangles each, truncated. Along a straight line the gap between
/// the line and that surface is therefore linear between the places where
/// the line crosses a grid row, a grid column or a cell's diagonal, and
/// its least value lies at one of them. So the traversal visits those crossings
/// and no other points -- in the grid's own index coordinates, where the
/// rows, columns and diagonals are the integer levels of `u`, `v` and
/// `u + v` -- and a point is hidden when at some crossing beyond the depth tolerance
/// the line is under the surface. Nothing is sampled: a sliver thinner than
/// any step is still a crossing. The answer is a function of the geometry
/// alone, which is what makes a bake the same drawing at every depth
/// resolution.
///
/// The depth tolerance means two things, and the traversal keeps them apart. A point of
/// curves lie on a surface it was placed on by the function or by the grid,
/// and the drawn surface -- the grid's triangles -- misses that by an
/// interpolation gap, so the point may start a little inside the solid.
/// That is forgiven *perpendicular* to the triangle, by the depth tolerance: a vertical
/// gap times the triangle's cosine, still linear along the line over one
/// triangle. Measured vertically it would be a horizontal tolerance of
/// depth tolerance / slope, and on a cone of slope 24 the rings came out dashed;
/// measured along the line of sight it forgives nothing at a fold, where the
/// line runs along the surface for a long way within a hair of it, and the
/// folds vanished. Then, once the line has been outside the solid, anything
/// it enters is an occluder and is tested *along the line of sight*, as the
/// depth buffer tested it: hidden when the line is still inside the solid
/// at the depth tolerance's distance toward the eye, or beyond. A wedge of rock
/// that is thin across but long along the line -- the side of a pit seen
/// through its notch -- hides what is behind it, as it should.
///
/// A point strictly over the domain is tested where it stands as well. A
/// point on a wall -- at the domain's edge -- is on the solid's boundary
/// and is tested only from where the line goes next, so the bottom edge of a wall
/// that faces the eye, whose line leaves the domain at once, is seen.
///
/// Built once per camera, since it scans the samples for the solid's extent
/// and inverts the tiles.
struct GridTraversal: Sendable {
    private(set) var heightfield: Heightfield

    /// The same traversal for one thread: its heightfield's function forked.
    func forked() -> GridTraversal {
        var m = self
        m.heightfield = heightfield.forked()
        return m
    }

    let toward: SIMD3<Double>
    /// The hidden-line depth tolerance, in view depth. Infinite disarms the traversal.
    let depthTolerance: Double
    /// The depth tolerance as a distance along the line of sight: an occluder must
    /// stand in front by more than this.
    let reach: Double
    /// The drawn grid's spacing, the larger of the two directions.
    let cell: Double
    let top: Double
    let floor: Double
    /// Per tile: world to the tile's grid index coordinates, and the
    /// grid's last index each way.
    let grids: [(linear: simd_double2x2, offset: SIMD2<Double>, extent: SIMD2<Double>)]
    /// Per tile: world back to the fundamental domain, for the nearest-tile
    /// height read (`Surface.inTile`, with the inverses taken once).
    let backs: [Affine2]
    let step: Int

    init(_ heightfield: Heightfield, view: Transform<WorldSpace, ViewSpace>, depthTolerance: Double) {
        self.heightfield = heightfield
        self.toward = -view.projectionDirection
        // View depth per unit of travel along the line of sight: the camera's
        // third row dotted with the direction, which under a sheared drawing
        // camera is not one.
        let c = view.m.columns
        let perUnit = abs(simd_dot(SIMD3(c.0.z, c.1.z, c.2.z), toward))
        let g = heightfield.surface.height
        let step = max(heightfield.step, 1)
        self.step = step
        let nx = (g.width + step - 1) / step, ny = (g.height + step - 1) / step
        let dx = g.domain.real.length / Double(max(g.width - 1, 1)) * Double(step)
        let dy = g.domain.imag.length / Double(max(g.height - 1, 1)) * Double(step)
        let cell = max(abs(dx), abs(dy))
        self.cell = cell
        // The depth tolerance forgives a curve its distance from the solid it is
        // tested against. Against the grid that is the chord's departure
        // from f, the drawing's depth tolerance. Against f it is the refiner's own
        // tolerance: a vertex is on f to a hundredth of it, and a chord
        // between two vertices departs from the curve by at most the
        // tolerance of a cell, measured across the curve -- which is the
        // perpendicular distance the traversal tests by, since along a level
        // line f is flat. A drawing-scale depth tolerance would let a ring show
        // through a wall for the depth tolerance's reach past the top edge.
        let bounded = heightfield.refine.map { min(depthTolerance, $0.tolerance * cell) } ?? depthTolerance
        self.depthTolerance = bounded
        self.reach = bounded / max(perUnit, 1e-12)
        var top = -Double.infinity, floor = Double.infinity
        heightfield.forEachSample { top = max(top, $0.z); floor = min(floor, $0.z) }
        for v in heightfield.occluder.vertices { floor = min(floor, v.z) }
        self.top = top; self.floor = floor
        var grids: [(linear: simd_double2x2, offset: SIMD2<Double>, extent: SIMD2<Double>)] = []
        var backs: [Affine2] = []
        for tile in heightfield.tiles {
            let back = tile.inverse ?? .identity
            backs.append(back)
            grids.append((
                linear: simd_double2x2(rows: [SIMD2(back.a / dx, back.b / dx),
                                              SIMD2(back.c / dy, back.d / dy)]),
                offset: SIMD2((back.tx - g.domain.real.lo) / dx, (back.ty - g.domain.imag.lo) / dy),
                extent: SIMD2(Double(nx - 1), Double(ny - 1))))
        }
        self.grids = grids; self.backs = backs
    }

    /// The drawn surface over a world point, read through the nearest tile:
    /// the grid's interpolated height before the truncation, the truncation there, and
    /// the cosine of the triangle's tilt. The drawn height is the lesser of the
    /// first two.
    func drawnSurface(at p: SIMD2<Double>) -> (interpolated: Double, truncationHeight: Double, cosine: Double) {
        let s = heightfield.surface
        guard backs.count > 1 else {
            return surface(at: P2<DomainSpace>(p.x, p.y))
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
        return surface(at: best)
    }

    /// The surface over a point of the fundamental tile: f's own height and
    /// the cosine of its tilt, by differences a step small against the cell,
    /// when the function is at hand; the grid's triangle otherwise. The curves
    /// is placed on f, so the solid it is tested against is f: a ring seen
    /// through a notch is hidden where f's top edge hides it, not where the
    /// chord across the nearest two nodes would.
    func surface(at q: P2<DomainSpace>) -> (interpolated: Double, truncationHeight: Double, cosine: Double) {
        let s = heightfield.surface
        guard let magnitude = heightfield.refine?.magnitude else {
            return s.drawnSurface(at: q, step: step)
        }
        let (u, slope) = functionSurface(at: q, magnitude: magnitude)
        let cosine = 1 / (1 + simd_dot(slope, slope)).squareRoot()
        return (u, s.truncation.height(atX: q.x), cosine.isNaN ? 0 : cosine)
    }

    /// f's height over a point of the fundamental tile and its slope there,
    /// by differences a step small against the cell.
    func functionSurface(at q: P2<DomainSpace>, magnitude: ContourRefine.Magnitude)
        -> (height: Double, slope: SIMD2<Double>)
    {
        let u = magnitude(q)
        let e = 1e-4 * cell
        // One-sided, from the value already in hand: three evaluations of
        // f per sample rather than five, at a step small enough that the
        // difference from the centred slope is below what the cosine or
        // the rate is used for.
        let hx = (magnitude(P2(q.x + e, q.y)) - u) / e
        let hy = (magnitude(P2(q.x, q.y + e)) - u) / e
        return (u.isNaN ? .infinity : u, SIMD2(hx, hy))
    }

    /// Whether nothing of the solid stands between `p` and the eye. With
    /// `trace`, every sample the traversal takes and what it made of it, for
    /// asking why a vertex is hidden.
    func unoccluded(_ p: P3<WorldSpace>, trace: ((String) -> Void)? = nil) -> Bool {
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
        /// it: over the interpolated triangle and over the truncation, and the triangle's
        /// cosine; nil where the point is outside the region, over which
        /// nothing is drawn.
        /// With the function, the sample also carries the rate the gap over
        /// f changes along the line, from the same three evaluations, for
        /// finding where the gap peaks or dips between two crossings.
        /// Through a reflected tile the slope's sign flips with the
        /// reflection; only its product with the line's direction is needed,
        /// and both are read in the same frame.
        let magnitude = heightfield.refine?.magnitude
        typealias Sample = (overTriangle: Double, overTruncation: Double, cosine: Double, rate: Double)
        func gaps(_ t: Double) -> Sample? {
            let q = pxy + t * dxy
            guard heightfield.region.contains(P2<WorldSpace>(q.x, q.y)) else { return nil }
            let z = p.z + t * d.z
            if let magnitude {
                let dq = heightfield.surface.inTile(P2<DomainSpace>(q.x, q.y), tiles: heightfield.tiles)
                let (u, slope) = functionSurface(at: dq, magnitude: magnitude)
                let cosine = 1 / (1 + simd_dot(slope, slope)).squareRoot()
                let truncationHeight = heightfield.surface.truncation.height(atX: dq.x)
                return (z - u, z - truncationHeight, cosine.isNaN ? 0 : cosine, d.z - simd_dot(slope, dxy))
            }
            let (interpolated, truncationHeight, cosine) = drawnSurface(at: q)
            return (z - interpolated, z - truncationHeight, cosine, 0)
        }
        // The truncation's bands, if any, as levels of the grid's u where the
        // truncation steps: a crossing of the drawn surface too.
        var bandEdges: [Double] = []
        if case .realBands(let bands, _) = heightfield.surface.truncation {
            let g = heightfield.surface.height
            let dx = g.domain.real.length / Double(max(g.width - 1, 1)) * Double(step)
            bandEdges = bands.map { ($0.below - g.domain.real.lo) / dx }
        }
        // Every crossing of a grid row, column or diagonal -- the
        // integer levels of u, v and u + v -- and of a truncation band's edge, over
        // every tile the line passes, with the ends of each passage; and
        // the depth tolerance's reach, which is where occlusion starts to count.
        var ts: [Double] = []
        var strictlyInside = false
        for grid in grids {
            let u0 = grid.linear * pxy + grid.offset
            let du = grid.linear * dxy
            var ta = 0.0, tb = tMax
            var misses = false
            for axis in 0..<2 {
                if abs(du[axis]) < 1e-15 {
                    if u0[axis] < 0 || u0[axis] > grid.extent[axis] { misses = true }
                    continue
                }
                let t1 = -u0[axis] / du[axis], t2 = (grid.extent[axis] - u0[axis]) / du[axis]
                ta = max(ta, min(t1, t2)); tb = min(tb, max(t1, t2))
            }
            guard !misses, ta <= tb, tb.isFinite else { continue }
            if u0.x > 1e-9, u0.x < grid.extent.x - 1e-9,
               u0.y > 1e-9, u0.y < grid.extent.y - 1e-9 { strictlyInside = true }
            ts.append(ta); ts.append(tb)
            // The diagonals split the grid's cells into the triangles the
            // depth pass draws; f has no triangles, and its brackets are the
            // rows and columns.
            for family in 0..<(magnitude == nil ? 3 : 2) {
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
        // forgiven the depth tolerance perpendicular to the triangle over it. Once it
        // has been outside, being inside at or beyond the depth tolerance's reach is
        // occlusion.
        var outsideYet = false
        var startForgiven = false
        /// Hidden at `t`, tested under a triangle of the given cosine; `g` is
        /// the gap there, nil beyond the region (outside, then).
        func hidden(_ t: Double, _ g: Sample?, cosine: Double) -> Bool {
            var visible = t > tEps
            if !visible, strictlyInside, !startForgiven { visible = true; startForgiven = true }
            // At the start, a point on the surface to rounding is in its own
            // surface's neighbourhood, not outside it. Further along, a line
            // under the surface by rounding alone is not inside it: the last
            // sample lies at the highest height by construction, and over a
            // truncation at that height the sign of its gap is a rounding error.
            let zEps = t > tEps ? -1e-9 * cell : 1e-9 * cell
            guard let g, g.overTriangle < zEps, g.overTruncation < zEps else {
                // Outside here, the start included: a point on the boundary
                // that stands above the surface is outside from the start,
                // and what its line then enters is an occluder. Against f,
                // outside means outside by more than the depth tolerance: a line that
                // grazes out of the surface by less than its skin and back
                // in has not left it.
                if let g, magnitude != nil {
                    let clearance = g.overTriangle > g.overTruncation ? g.overTriangle * cosine : g.overTruncation
                    if clearance <= depthTolerance {
                        trace?(String(format: "t %.4g cells: within the skin, %.3e over f", t / cell, g.overTriangle))
                        return false
                    }
                }
                outsideYet = true
                trace?(String(format: "t %.4g cells: outside (gap %@)", t / cell,
                              g.map { String(format: "%+.3e over f, %+.3e over truncation", $0.overTriangle, $0.overTruncation) } ?? "off the region"))
                return false
            }
            guard visible else {
                trace?(String(format: "t %.4g cells: inside by %.3e, not tested at the start", t / cell, -g.overTriangle))
                return false
            }
            if outsideYet {
                trace?(String(format: "t %.4g cells: inside by %.3e (below the truncation by %.3e) after being outside, reach %.4g cells: %@",
                              t / cell, -g.overTriangle, -g.overTruncation, reach / cell, t + tEps >= reach ? "hidden" : "forgiven"))
                return t + tEps >= reach
            }
            let depth = g.overTriangle > g.overTruncation ? g.overTriangle * cosine : g.overTruncation
            trace?(String(format: "t %.4g cells: inside by %.3e in its own surface's neighbourhood, depth %.3e, depth tolerance %.3e: %@",
                          t / cell, -g.overTriangle, depth, depthTolerance, depth + depthTolerance < 0 ? "hidden" : "forgiven"))
            return depth + depthTolerance < 0
        }
        trace?(String(format: "from (%.4f, %.4f, %.4f), %d samples to %.4g cells", p.x, p.y, p.z, ts.count, (ts.last ?? 0) / cell))
        var previousMiddle: Sample?
        for k in 0..<ts.count {
            let t = ts[k], g = values[k]
            // A point on a grid line lies under two triangles: tested by
            // the triangle behind it and the one ahead. Under f the cosine is
            // continuous and the sample's own serves. The sample at the
            // middle of the interval ahead is taken once, for this and for
            // the search of the interval.
            let middle: Sample? = k + 1 < ts.count ? gaps(0.5 * (t + ts[k + 1])) : nil
            if magnitude == nil {
                let cosines = [previousMiddle?.cosine, middle?.cosine].compactMap { $0 }
                for cosine in cosines.isEmpty ? [g?.cosine ?? 1] : cosines {
                    if hidden(t, g, cosine: cosine) { return false }
                }
            } else if hidden(t, g, cosine: g?.cosine ?? 1) { return false }
            previousMiddle = middle
            // With the function, a dip of f under the line between this
            // crossing and the next: where the gap over f falls at one end
            // and rises at the other it has a least value between, found by
            // golden section; under the line, it is tested where it is.
            // With the function, the gap over f between this crossing and
            // the next is not the triangle's straight line: it can rise above
            // the surface's skin and fall back under it, or dip under and
            // come back out, between two samples that see neither. The gap
            // and its rate are read at both ends and at the middle; a
            // greatest value is bracketed where the gap rises at one end and
            // falls at the other, or the middle is above both ends, and a
            // least value likewise; each is found by golden section. The
            // greatest, if its clearance exceeds the depth tolerance, is the line
            // leaving the surface; the least is tested where it is, after
            // the greatest if that came first. A point just in front of a
            // fold is the case: its line leaves the surface, passes under it
            // and comes out again before the cut.
            if magnitude != nil, k + 1 < ts.count, let ra = g, let rb = values[k + 1], let rm = middle {
                let ta = t, tb = ts[k + 1], tm = 0.5 * (ta + tb)
                let phi = (5.0.squareRoot() - 1) / 2
                func gapOverF(_ x: Double) -> Double { gaps(x)?.overTriangle ?? .infinity }
                /// The t of the least (or, negated, the greatest) gap on a span.
                func extremum(_ lo0: Double, _ hi0: Double, sign: Double) -> Double {
                    var lo = lo0, hi = hi0
                    var c = hi - phi * (hi - lo), e = lo + phi * (hi - lo)
                    var fc = sign * gapOverF(c), fe = sign * gapOverF(e)
                    for _ in 0..<60 {
                        if fc < fe { hi = e; e = c; fe = fc; c = hi - phi * (hi - lo); fc = sign * gapOverF(c) }
                        else { lo = c; c = e; fc = fe; e = lo + phi * (hi - lo); fe = sign * gapOverF(e) }
                        if hi - lo < tEps { break }
                    }
                    return 0.5 * (lo + hi)
                }
                var peak: Double?
                if rm.overTriangle > max(ra.overTriangle, rb.overTriangle) { peak = extremum(ta, tb, sign: -1) }
                else if ra.rate > 0, rm.rate < 0 { peak = extremum(ta, tm, sign: -1) }
                else if rm.rate > 0, rb.rate < 0 { peak = extremum(tm, tb, sign: -1) }
                var dip: Double?
                if rm.overTriangle < min(ra.overTriangle, rb.overTriangle) { dip = extremum(ta, tb, sign: 1) }
                else if ra.rate < 0, rm.rate > 0 { dip = extremum(ta, tm, sign: 1) }
                else if rm.rate < 0, rb.rate > 0 { dip = extremum(tm, tb, sign: 1) }
                else if ra.rate < 0, rb.rate > 0 { dip = extremum(ta, tb, sign: 1) }
                func test(_ tx: Double) -> Bool {
                    guard let gx = gaps(tx) else { return false }
                    trace?(String(format: "extremum at %.4g cells between %.4g and %.4g: gap %+.3e", tx / cell, ta / cell, tb / cell, gx.overTriangle))
                    return hidden(tx, gx, cosine: gx.cosine)
                }
                for tx in [peak, dip].compactMap({ $0 }).sorted() where test(tx) { return false }
            }
            // The truncation edge between this crossing and the next, where the triangle
            // meets the truncation: a crease, tested under both.
            if k + 1 < ts.count, let ga = g, let gb = values[k + 1] {
                let ea = ga.overTriangle - ga.overTruncation, eb = gb.overTriangle - gb.overTruncation
                if (ea > 0) != (eb > 0), ea != eb {
                    var truncationEdge = t + (ts[k + 1] - t) * ea / (ea - eb)
                    if heightfield.refine?.magnitude != nil {
                        // On f the excess is not linear between crossings:
                        // bisect the sign change to the last bit.
                        var lo = t, hi = ts[k + 1], elo = ea
                        for _ in 0..<60 {
                            let m = 0.5 * (lo + hi)
                            guard let gm = gaps(m) else { break }
                            let em = gm.overTriangle - gm.overTruncation
                            if (em > 0) == (elo > 0) { lo = m; elo = em } else { hi = m }
                        }
                        truncationEdge = 0.5 * (lo + hi)
                    }
                    if truncationEdge > tEps, let gr = gaps(truncationEdge) {
                        if hidden(truncationEdge, gr, cosine: middle?.cosine ?? gr.cosine) { return false }
                        if hidden(truncationEdge, gr, cosine: 1) { return false }
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
/// a region, walls and a truncation, and a parametric surface has none of them --
/// what it has instead is a coordinate at every point, which is what its curves
/// is tested by. Each renderer, bound and bake says what it does with each.
public enum SceneGeometry: Sendable {
    case heightfield(Heightfield)
    case parametric(ParametricSurface)
    /// A polyhedron, whose curves are its edges (`Solid`).
    case solid(Solid)
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
    /// Identifies the *curves*. Separate from `content` because editing a level
    /// set changes every stroke and none of the landscape, and rebuilding a
    /// hundred-megabyte height texture to move a contour would make the slider
    /// unusable on exactly the plates where it is most interesting.
    public let curves: ContentID
    public let geometry: SceneGeometry
    public let layers: [Layer]

    public var camera: Camera
    public var mode: PreviewMode
    /// Hidden-line depth tolerance, in world units of depth, for curves tested by depth.
    public var depthTolerance: Double
    /// How finely the ends of tested curves are placed, as a length in the view
    /// plane: a window sets a fraction of its pixel, since curves placed more
    /// finely than a pixel looks the same and costs a traversal per halving.
    /// Zero, the default and a bake's, places them to the last bit.
    public var resolution: Double = 0

    /// A heightfield scene.
    public init(surface: Surface, occluder: Mesh<WorldSpace>, tiles: [Affine2],
                region: Region = .full, step: Int, layers: [Layer], camera: Camera,
                mode: PreviewMode = .drawing, depthTolerance: Double, refine: ContourRefine? = nil) {
        self.init(content: ContentID(), curves: ContentID(),
                  geometry: .heightfield(Heightfield(surface: surface, occluder: occluder,
                                                     tiles: tiles, region: region, step: step,
                                                     refine: refine)),
                  layers: layers, camera: camera, mode: mode, depthTolerance: depthTolerance)
    }

    /// A parametric surface, with curves on it.
    ///
    /// Curves that carry surface coordinates are tested by them; curves that do
    /// not falls back to depth, at `depthTolerance`.
    public init(surface: ParametricSurface, layers: [Layer], camera: Camera,
                mode: PreviewMode = .drawing, depthTolerance: Double) {
        self.init(content: ContentID(), curves: ContentID(), geometry: .parametric(surface),
                  layers: layers, camera: camera, mode: mode, depthTolerance: depthTolerance)
    }

    /// A solid, with its edges as curves.
    public init(solid: Solid, layers: [Layer], camera: Camera,
                mode: PreviewMode = .drawing, depthTolerance: Double = 0) {
        self.init(content: ContentID(), curves: ContentID(), geometry: .solid(solid),
                  layers: layers, camera: camera, mode: mode, depthTolerance: depthTolerance)
    }

    /// The scene a bundle describes under one of its presets.
    public init(bundle: KurvenBundle, preset: CameraPreset) {
        self.init(surface: bundle.surface,
                  occluder: bundle.occluder(),
                  tiles: bundle.manifest.occluder.tiles,
                  region: bundle.manifest.occluder.region,
                  step: bundle.manifest.occluder.step,
                  layers: bundle.layers,
                  camera: .oblique(preset.projection),
                  depthTolerance: preset.depthTolerance,
                  refine: bundle.refine)
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

    /// The solid, when that is what this scene draws.
    public var solid: Solid? {
        if case .solid(let s) = geometry { return s }
        return nil
    }

    /// The same content, looked at from somewhere else. Keeps both identities,
    /// so nothing at all is rebuilt -- which is the whole point of separating
    /// the camera from the content.
    public func looking(_ camera: Camera) -> Scene {
        var out = self; out.camera = camera; return out
    }

    /// The same landscape, drawn with different curves. Keeps `content` and mints
    /// a new `curves`, so the heightfield stays uploaded and only the line buffer
    /// is rebuilt.
    public func drawing(_ layers: [Layer]) -> Scene {
        Scene(content: content, curves: ContentID(), geometry: geometry,
              layers: layers, camera: camera, mode: mode, depthTolerance: depthTolerance)
    }

    private init(content: ContentID, curves: ContentID, geometry: SceneGeometry,
                 layers: [Layer], camera: Camera, mode: PreviewMode, depthTolerance: Double) {
        self.content = content; self.curves = curves; self.geometry = geometry
        self.layers = layers; self.camera = camera; self.mode = mode; self.depthTolerance = depthTolerance
    }

    /// The creases of the surface's curves, in its coordinates: every outline
    /// that is not the folds -- a truncation edge. A fold that meets one is traced
    /// from it, where the grid may hold none of the fold.
    public var creases: [[P2<ParamSpace>]] {
        layers.flatMap { layer -> [[P2<ParamSpace>]] in
            if case .foldCurves = layer.spec.source { return [] }
            guard layer.spec.role == .outline, let coords = layer.paths.coords else { return [] }
            return (0..<layer.paths.count).map {
                Array(coords[layer.paths.offsets[$0]..<layer.paths.offsets[$0 + 1]])
            }
        }
    }

    /// Every layer's curves in view space, in declaration (draw) order, tested
    /// for this camera wherever the geometry can test it exactly.
    ///
    /// Curves that depend on the camera -- the fold curves of a surface, of
    /// either kind, and the edges of a solid -- is derived here, for this
    /// camera, rather than carried. On a heightfield every clipped layer is
    /// tested here, by the facing of what it lies on and by the solid
    /// (`GridTraversal`), and the depth buffer has no part in it: the drawing is
    /// a function of the geometry and the camera, never of a resolution. A
    /// parametric surface's curves are tested by the bake, which has the
    /// surface's coordinate image; its fold curves come tested by
    /// construction. This is the bake's work, not the preview's: it is a
    /// traversal per vertex, and the preview keeps its depth test. `depthTolerance` is
    /// the hidden-line depth tolerance to test by; the scene's own when not given.
    public func visibleLayers(depthTolerance: Double? = nil) -> [(Layer, PolylineSet<ViewSpace>)] {
        let view = camera.view
        let depthTolerance = depthTolerance ?? self.depthTolerance
        return layers.flatMap { layer -> [(Layer, PolylineSet<ViewSpace>)] in
            switch geometry {
            case .parametric(let s):
                if case .foldCurves = layer.spec.source {
                    return [(layer, s.foldCurves(sight: view.projectionDirection, creases: creases).mapped(view))]
                }
                return [(layer, layer.paths.mapped(view))]
            case .heightfield(let h):
                let visible: PolylineSet<WorldSpace>
                if case .foldCurves = layer.spec.source {
                    visible = h.visibleFolds(view: view, depthTolerance: depthTolerance)
                } else if !layer.spec.clipped {
                    visible = layer.paths
                } else if layer.spec.source.liesOnWall {
                    visible = h.visibleWallCurves(layer.paths, view: view, depthTolerance: depthTolerance)
                } else if layer.spec.liesOnSurface {
                    visible = h.visibleSurfaceCurves(layer.paths, view: view, depthTolerance: depthTolerance)
                } else {
                    visible = h.visibleCurves(layer.paths, view: view, depthTolerance: depthTolerance)
                }
                return [(layer, visible.mapped(view))]
            case .solid(let s):
                if case .edges = layer.spec.source, layer.spec.clipped {
                    return [(layer, s.visibleEdges(view: view).mapped(view))]
                }
                if case .hatching(let hatching, .solid) = layer.spec.source {
                    // One layer per width step, at that fraction of the
                    // width: every hatch line when unclipped, the visible
                    // spans otherwise.
                    let steps = layer.spec.clipped
                        ? s.visibleHatching(hatching, view: view)
                        : s.hatchLinesByWidth(hatching).map {
                            let all = PolylineSet(paths: $0.lines.map(\.path))
                            return (widthStep: $0.widthStep, all: all, visible: all)
                        }
                    return steps.map { w, all, drawn in
                        var spec = layer.spec
                        spec.width = layer.spec.width * Double(w) / Double(hatching.widthSteps)
                        if hatching.widthSteps > 1 { spec.name = "\(layer.spec.name) \(w)/\(hatching.widthSteps)" }
                        return (Layer(spec: spec, paths: all), drawn.mapped(view))
                    }
                }
                return [(layer, layer.paths.mapped(view))]
            }
        }
    }

    /// The view-space extent the depth buffer covers: every heightfield sample,
    /// the walls, and every *clipped* layer.
    ///
    /// This is `ZBuffer(xs.min(), xs.max(), ys.min(), ys.max(), ...)` on the
    /// Python side, and it has to be the same extent or the two rasterize onto
    /// different pixel grids and every comparison downstream is measuring the
    /// framing rather than the drawing. Two consequences of matching it exactly:
    ///
    /// - The heightfield is scanned sample by sample, not bounded by its box.
    ///   The box of a rotated landscape is looser than the hull of its samples,
    ///   and the difference is visible at bake resolution.
    /// - Masked-out samples still count. `build_occluder` emits every grid
    ///   vertex and drops only *triangles*, so `occ_rot` includes the vertices
    ///   inside zeta's cutout even though nothing references them. Excluding
    ///   them here would be more principled and would not be the same picture.
    ///
    /// Unclipped curves are excluded, as they are in Python: they are drawn but never
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
        case .solid(let s):
            for p in s.vertices { add(camera.view(p)) }
        }
        // The curves as carried, before any testing visibility: testing visibility is the bake's
        // expensive step and would be paid twice, and curves the test drops are
        // still curves the frame was sized to.
        for layer in layers where layer.spec.clipped {
            for v in layer.paths.vertices { add(camera.view(v)) }
        }
        // A parametric surface's folds are placed on the map, not the
        // grid, and can lie a hair outside its hull: they are derived here
        // so the frame holds them, since they are tested by the frame.
        if let s = parametric, layers.contains(where: { layer in
            if case .foldCurves = layer.spec.source { return true }
            return false
        }) {
            for v in s.foldCurves(sight: camera.view.projectionDirection, creases: creases).vertices { add(camera.view(v)) }
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
        case .solid(let s):
            for p in s.vertices { add(camera.view(p)) }
        }
        return any ? AABB(lo: lo, hi: hi) : nil
    }
}

/// What a preview draws.
public enum PreviewMode: Sendable, Equatable {
    /// The drawing: white occluding surface, black hidden-line curves.
    case drawing
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

/// A heightfield's fold curves placed on f, for one thread: a fork of the
/// heightfield, and the grid of the tile the lines were traced on.
///
/// Placing a vertex on the fold is sixty bisections of four evaluations of
/// f each, the chord between two vertices as many again, and a crossing of
/// the truncation edge sixty of sixty. Each is independent of every other, so `place`
/// takes the lines in stages -- every vertex, then every truncation edge crossing, then
/// every chord, then every run's ends -- and runs each stage in parallel
/// over its pieces, each thread with a placer of its own. The pieces come
/// back in the order they went in, so the fold is the same whatever the
/// thread count: a line placed on one core and a line placed on eight are
/// the same vertices.
struct FoldPlacer: Sendable {
    let heightfield: Heightfield
    /// The direction lines of sight travel, in world space.
    let sight: SIMD3<Double>
    /// The grid spacing, the larger of the two directions.
    let cell: Double
    /// Whether vertices are moved onto the exact fold, or left where the
    /// grid's chords put them.
    let refined: Bool
    /// The tile's grid: its corner and spacing.
    let lo: SIMD2<Double>
    let sx: Double, sy: Double

    var surface: Surface { heightfield.surface }
    var region: Region { heightfield.region }
    /// With the function at hand the facing is the graph's, untruncated: the
    /// fold is a property of the one surface the graph is, and runs on
    /// below the truncation, where the truncation cuts it at the truncation edge. The truncated facing
    /// jumps at the crease, and its zero there is the crease, not the fold.
    var onF: Bool { heightfield.refine?.magnitude != nil }

    /// How squarely the surface faces the viewer over `p`: positive facing,
    /// negative facing away, zero on the fold.
    func facing(_ p: P2<WorldSpace>) -> Double {
        let n = onF ? heightfield.graphNormal(at: p) : heightfield.normal(at: p)
        let len = simd_length(n)
        return len > 0 ? -simd_dot(n, sight) / len : 0
    }

    /// |f| over a world point, untruncated, through its tile; the grid's
    /// height without the function.
    func graphHeight(_ p: P2<WorldSpace>) -> Double {
        guard let magnitude = heightfield.refine?.magnitude else { return heightfield.surfaceHeight(at: p) }
        let u = magnitude(surface.inTile(P2<DomainSpace>(p.x, p.y), tiles: heightfield.tiles))
        return u.isNaN ? .infinity : u
    }

    /// The surface's height over a fold vertex: f's, truncated, with the
    /// function at hand, so the fold is drawn at the height the rings that
    /// end on it are drawn at; the grid's otherwise.
    func height(_ p: P2<WorldSpace>) -> Double { heightfield.surfaceHeight(at: p) }

    /// A traced vertex placed: moved onto the fold, at the surface's
    /// height, and whether it lies in the region and on the graph rather
    /// than below the truncation.
    struct Vertex: Sendable {
        let p: P2<WorldSpace>
        let z: Double
        let inRegion: Bool
        let onGraph: Bool
    }

    func vertex(_ v: P2<DomainSpace>) -> Vertex {
        var p = P2<WorldSpace>(v.x, v.y)
        if refined { p = ontoEdge(p) }
        let z = height(p)
        let truncationHeight = surface.truncation.height(atX: p.x)
        let inRegion = region.contains(p)
        return Vertex(p: p, z: z, inRegion: inRegion, onGraph: inRegion && z.isFinite && z < truncationHeight)
    }

    /// The traced lines placed on the fold, in stages, each in parallel.
    /// Nothing on a plateau, which faces up and has a truncation edge of its own. With the
    /// function at hand the fold is traced on the graph alone and runs on
    /// below the truncation: where a run passes under the truncation edge it ends on the truncation edge,
    /// at the fold's own crossing of |f| = h(x), so a tower's outline
    /// reaches the truncation and arrives along the fold. On the fundamental
    /// tile, where the function is sampled, each run is then subdivided to
    /// the refiner's tolerance and carried to the zero it was going to.
    func place(_ lines: [[P2<DomainSpace>]], fundamental: Bool,
               worker: @Sendable () -> FoldPlacer) -> [[P3<WorldSpace>]] {
        // Every vertex of every line onto the fold.
        var offsets = [0]
        var flat: [P2<DomainSpace>] = []
        for line in lines { flat += line; offsets.append(flat.count) }
        let vertices = Heightfield.parallelMap(flat, chunk: 8, worker: worker) { w, v in w.vertex(v) }
        // Between a vertex on the graph and the next below the truncation, the
        // fold's crossing of the truncation edge.
        var pairs: [Int] = []
        for i in lines.indices {
            for k in (offsets[i] + 1)..<offsets[i + 1] {
                let a = vertices[k - 1], b = vertices[k]
                if onF, a.inRegion, b.inRegion, a.onGraph != b.onGraph,
                   surface.truncation.height(atX: b.p.x).isFinite { pairs.append(k) }
            }
        }
        let crossings = Heightfield.parallelMap(pairs, chunk: 1, worker: worker) { w, k -> P3<WorldSpace>? in
            let a = vertices[k - 1], b = vertices[k]
            return b.onGraph ? w.ontoTruncationEdge(from: b.p, to: a.p) : w.ontoTruncationEdge(from: a.p, to: b.p)
        }
        var truncationEdge = [P3<WorldSpace>?](repeating: nil, count: flat.count)
        for (i, k) in pairs.enumerated() { truncationEdge[k] = crossings[i] }
        // The runs: a line cut where it leaves the graph, each piece
        // ending on the truncation edge it passed under.
        var runs: [[P3<WorldSpace>]] = []
        for i in lines.indices {
            var run: [P3<WorldSpace>] = []
            func close() {
                if run.count >= 2 { runs.append(run) }
                run = []
            }
            for k in offsets[i]..<offsets[i + 1] {
                let v = vertices[k]
                if let r = truncationEdge[k] { run.append(r) }
                if v.onGraph { run.append(P3(v.p.x, v.p.y, v.z)) } else { close() }
            }
            close()
        }
        guard fundamental else { return runs }
        // Every chord of every run subdivided, then each run's ends carried
        // to their zeros.
        if refined, let tolerance = heightfield.refine?.tolerance {
            var chords: [SIMD2<Int>] = []
            for (r, run) in runs.enumerated() {
                for k in 0..<(run.count - 1) { chords.append(SIMD2(r, k)) }
            }
            let whole = runs
            let between = Heightfield.parallelMap(chords, chunk: 4, worker: worker) { w, c -> [P3<WorldSpace>] in
                var out: [P3<WorldSpace>] = []
                w.subdivide(whole[c.x][c.y], whole[c.x][c.y + 1], tolerance: tolerance, depth: 0, into: &out)
                return out
            }
            var next = 0
            for r in runs.indices {
                var out = [whole[r][0]]
                for k in 0..<(whole[r].count - 1) {
                    out += between[next]
                    next += 1
                    out.append(whole[r][k + 1])
                }
                runs[r] = out
            }
        }
        let subdivided = runs
        return Heightfield.parallelMap(subdivided, chunk: 1, worker: worker) { w, run in
            w.heightfield.endedAtZeros(run, cell: w.cell)
        }
    }

    /// Between two vertices on the fold, the fold itself: the chord's
    /// midpoint moved onto the fold along the chord's normal, kept when it
    /// moved further than the refiner's tolerance, and the two halves
    /// likewise -- the contour refiner's rule, so a fold is a curve to the
    /// same tolerance the rings that end on it are.
    func subdivide(_ a: P3<WorldSpace>, _ b: P3<WorldSpace>, tolerance: Double, depth: Int,
                   into out: inout [P3<WorldSpace>]) {
        guard depth < 5 else { return }
        let chord = ((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y)).squareRoot()
        guard chord > 1e-9 * cell else { return }
        let mid = SIMD2(0.5 * (a.x + b.x), 0.5 * (a.y + b.y))
        let n = SIMD2(-(b.y - a.y), b.x - a.x) / chord
        var lo = mid - 0.5 * chord * n, hi = mid + 0.5 * chord * n
        var flo = facing(P2(lo.x, lo.y))
        guard (flo > 0) != (facing(P2(hi.x, hi.y)) > 0) else { return }
        for _ in 0..<60 {
            let m = 0.5 * (lo + hi)
            let fm = facing(P2(m.x, m.y))
            if (fm > 0) == (flo > 0) { lo = m; flo = fm } else { hi = m }
        }
        let q = 0.5 * (lo + hi)
        let p = P2<WorldSpace>(q.x, q.y)
        guard region.contains(p) else { return }
        let z = height(p)
        guard z.isFinite, z < surface.truncation.height(atX: p.x) - 1e-9 else { return }
        let v = P3<WorldSpace>(p.x, p.y, z)
        // The fold is a curve in space: its departure from the chord is
        // measured there, across and up.
        guard simd_length(v.v - 0.5 * (a.v + b.v)) > tolerance * cell else { return }
        subdivide(a, v, tolerance: tolerance, depth: depth + 1, into: &out)
        out.append(v)
        subdivide(v, b, tolerance: tolerance, depth: depth + 1, into: &out)
    }

    /// Onto the fold, along the gradient: bracket a sign change within a
    /// cell either way, then bisect it to the last bit. The fallback, for a
    /// vertex that lies on no grid edge.
    func onto(_ c: P2<WorldSpace>) -> P2<WorldSpace> {
        let h = 0.25 * cell
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

    /// The grid edges a marching-squares vertex lies on: one, or two for
    /// a vertex on a node.
    func gridEdges(through c: P2<WorldSpace>) -> [(SIMD2<Double>, SIMD2<Double>)] {
        let g = surface.height
        let nx = g.width, ny = g.height
        let fx = (c.x - lo.x) / sx, fy = (c.y - lo.y) / sy
        let onColumn = abs(fx - fx.rounded()) < 1e-6, onRow = abs(fy - fy.rounded()) < 1e-6
        var ends: [(SIMD2<Double>, SIMD2<Double>)] = []
        if onColumn, fy < Double(ny - 1) {
            let i = Int(fx.rounded()), j = Int(fy.rounded(.down))
            ends.append((SIMD2(lo.x + Double(i) * sx, lo.y + Double(j) * sy),
                         SIMD2(lo.x + Double(i) * sx, lo.y + Double(j + 1) * sy)))
        }
        if onRow, fx < Double(nx - 1) {
            let i = Int(fx.rounded(.down)), j = Int(fy.rounded())
            ends.append((SIMD2(lo.x + Double(i) * sx, lo.y + Double(j) * sy),
                         SIMD2(lo.x + Double(i + 1) * sx, lo.y + Double(j) * sy)))
        }
        return ends
    }

    /// Onto the fold, along the grid edge the vertex lies on: a
    /// marching-squares vertex is the chord's crossing of an edge whose two
    /// nodes face opposite ways, so the edge brackets the fold by
    /// construction, and bisecting it to the last bit lands on the fold, as
    /// the contour refiner lands on f along the same edges. Bracketing
    /// along the gradient instead can miss where the facing is sharp -- a
    /// pit one cell wide -- and leave the vertex on the chord.
    func ontoEdge(_ c: P2<WorldSpace>) -> P2<WorldSpace> {
        for (a0, b0) in gridEdges(through: c) {
            var a = a0, b = b0
            var fa = facing(P2(a.x, a.y))
            let fb = facing(P2(b.x, b.y))
            guard (fa > 0) != (fb > 0) else { continue }
            for _ in 0..<60 {
                let m = 0.5 * (a + b)
                let fm = facing(P2(m.x, m.y))
                if (fm > 0) == (fa > 0) { a = m; fa = fm } else { b = m }
            }
            let m = 0.5 * (a + b)
            return P2(m.x, m.y)
        }
        return onto(c)
    }

    /// Where the fold crosses the truncation edge, between a vertex on the graph and
    /// one below the truncation: the chord's points moved onto the fold along the
    /// chord's normal, as `subdivide` moves a midpoint, and the one where
    /// |f| reaches the truncation found by bisection to the last bit. The
    /// grid's crossing of the crease is where a grid edge meets the
    /// truncation edge, which is not where the fold does: joined to the graph by one
    /// straight chord a cell long, the fold cut across the truncation's projection
    /// on its way to the truncation edge.
    func ontoTruncationEdge(from a: P2<WorldSpace>, to b: P2<WorldSpace>) -> P3<WorldSpace>? {
        let chord = ((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y)).squareRoot()
        guard chord > 1e-9 * cell else { return nil }
        let n = SIMD2(-(b.y - a.y), b.x - a.x) / chord
        func onFold(_ t: Double) -> P2<WorldSpace> {
            let m = SIMD2(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y))
            var lo = m - 0.5 * chord * n, hi = m + 0.5 * chord * n
            var flo = facing(P2(lo.x, lo.y))
            guard (flo > 0) != (facing(P2(hi.x, hi.y)) > 0) else { return P2(m.x, m.y) }
            for _ in 0..<60 {
                let c = 0.5 * (lo + hi)
                let fc = facing(P2(c.x, c.y))
                if (fc > 0) == (flo > 0) { lo = c; flo = fc } else { hi = c }
            }
            let c = 0.5 * (lo + hi)
            return P2(c.x, c.y)
        }
        func excess(_ t: Double) -> Double {
            let q = onFold(t)
            return graphHeight(q) - surface.truncation.height(atX: q.x)
        }
        var lo = 0.0, hi = 1.0
        guard excess(lo) < 0, excess(hi) >= 0 else { return nil }
        for _ in 0..<60 {
            let m = 0.5 * (lo + hi)
            if excess(m) < 0 { lo = m } else { hi = m }
        }
        let q = onFold(0.5 * (lo + hi))
        let truncationHeight = surface.truncation.height(atX: q.x)
        guard region.contains(q), truncationHeight.isFinite else { return nil }
        return P3(q.x, q.y, truncationHeight)
    }
}
