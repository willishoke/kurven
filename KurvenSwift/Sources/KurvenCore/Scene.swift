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

    /// The fold lines that can be seen: `foldLines`, less what the surface
    /// hides. A fold cannot be judged by the depth buffer: the surface is
    /// edge-on there, so its depth changes by the whole flank within one
    /// pixel, and the pixel's value is the near side's, in front of the fold
    /// by far more than any margin. It is judged by the solid instead: a fold
    /// vertex is hidden when the sight line from it to the eye passes under
    /// the surface, which is a march up that line in half-cell steps until
    /// it is above everything -- the question the depth pass answers for
    /// every pixel, asked here for every vertex, exactly. `margin` is the
    /// plate's hidden-line margin, which keeps a vertex on the surface it
    /// belongs to from hiding behind that surface's own rounding.
    public func visibleFolds(view: Transform<WorldSpace, ViewSpace>, margin: Double,
                             refined: Bool = true) -> PolylineSet<WorldSpace> {
        let folds = foldLines(view: view, refined: refined)
        guard !folds.vertices.isEmpty else { return folds }
        let toward = -view.sightLine
        let g = surface.height
        let cell = max(abs(g.domain.real.length) / Double(max(g.width - 1, 1)),
                       abs(g.domain.imag.length) / Double(max(g.height - 1, 1)))
        var top = -Double.infinity
        var lo = SIMD2(Double.infinity, Double.infinity), hi = -lo
        forEachSample {
            top = max(top, $0.z)
            lo = simd_min(lo, SIMD2($0.x, $0.y)); hi = simd_max(hi, SIMD2($0.x, $0.y))
        }
        let rise = toward.z
        func unoccluded(_ p: P3<WorldSpace>) -> Bool {
            // Nothing can stand in the way of a sight line that does not
            // descend toward the solid, once it has left this point.
            guard rise > 0 else { return true }
            let step = 0.5 * cell / max(simd_length(SIMD2(toward.x, toward.y)), 1e-12)
            var t = step
            while true {
                let q = p.v + t * toward
                // Above everything, or out past the box of every tile, which
                // is convex and so is not re-entered: open air from here on.
                if q.z > top { return true }
                if q.x < lo.x || q.x > hi.x || q.y < lo.y || q.y > hi.y { return true }
                if q.z + margin < surface.height(at: P2<DomainSpace>(q.x, q.y), tiles: tiles) {
                    return false
                }
                t += step
            }
        }
        var out: [[P3<WorldSpace>]] = []
        for i in 0..<folds.count {
            var run: [P3<WorldSpace>] = []
            for v in folds[path: i] {
                if unoccluded(v) {
                    run.append(v)
                } else {
                    if run.count >= 2 { out.append(run) }
                    run = []
                }
            }
            if run.count >= 2 { out.append(run) }
        }
        return PolylineSet(paths: out)
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

    /// Every layer's vertices in view space, in declaration (draw) order.
    ///
    /// Ink that depends on the camera -- the fold lines of a surface, of
    /// either kind -- is derived here, for this camera, rather than carried.
    public func projectedLayers() -> [(Layer, PolylineSet<ViewSpace>)] {
        layers.map { layer in
            if case .foldLines = layer.spec.source {
                if let s = parametric {
                    return (layer, s.foldLines(sight: camera.view.sightLine).mapped(camera.view))
                }
                if let h = heightfield {
                    // Already judged for visibility, by the solid: the bake
                    // passes these through.
                    return (layer, h.visibleFolds(view: camera.view, margin: margin).mapped(camera.view))
                }
            }
            return (layer, layer.paths.mapped(camera.view))
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
        for (layer, projected) in projectedLayers() where layer.spec.clipped {
            for v in projected.vertices { add(v) }
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
