import Foundation
import simd

/// Where a geometry stands in a composite: scaled uniformly about the origin,
/// then translated. A similarity, so a line of sight in world space is a line
/// of sight in the geometry's own coordinates, and each occlusion test is
/// asked there, exactly as when the geometry stands alone.
public struct Placement: Sendable, Equatable {
    public var scale: Double
    public var offset: SIMD3<Double>

    public init(scale: Double = 1, offset: SIMD3<Double> = .zero) {
        self.scale = scale; self.offset = offset
    }

    public static let identity = Placement()

    /// From the geometry's own coordinates into world space.
    public var transform: Transform<WorldSpace, WorldSpace> {
        Transform(rows: (SIMD3(scale, 0, 0), SIMD3(0, scale, 0), SIMD3(0, 0, scale)), translation: offset)
    }

    /// A world point in the geometry's own coordinates.
    public func back(_ p: P3<WorldSpace>) -> P3<WorldSpace> { P3((p.v - offset) / scale) }
}

/// Why geometries cannot stand in one composite.
public struct CompositeError: Error, CustomStringConvertible, Equatable {
    public let description: String
    public init(_ description: String) { self.description = description }
}

/// Several geometries in one scene, under one camera: each a `Scene` of its
/// own -- its faces, its curves, its depth tolerance -- standing at its
/// placement. Every clipped curve is tested against every geometry's
/// occlusion test (`OcclusionTest`): its own, exactly as when it stands
/// alone (`Scene.visible(_:depthTolerance:image:)`), and then each other's,
/// which forgives it nothing (`OcclusionTest.isClear(ofAnotherSolid:)`).
/// Curves of different geometries that cross on the page are cut only where
/// one hides the other, never joined.
///
/// Only a geometry with an exact occlusion test stands in a composite: a
/// landscape, a polyhedron, or a parametric surface with a tubular
/// neighbourhood. The coordinate image says only whether a point is the
/// front-most of its own surface, nothing about another solid, so a surface
/// tested by it -- the forced torus -- is refused. No depth buffer stands in
/// between solids: the drawing is a function of the geometry and the camera.
public struct Composite: Sendable {
    public struct Part: Sendable {
        /// What the part is called in a refusal: its bundle's name.
        public let name: String
        /// The geometry and its curves, looking through the composite's
        /// camera from where the part stands (`placement`).
        public let scene: Scene
        public let placement: Placement
    }

    public let parts: [Part]
    public let camera: Camera

    /// The parts, each `scene` taken with its geometry, layers and depth
    /// tolerance (its camera is the composite's), standing at its placement.
    public init(_ parts: [(name: String, scene: Scene, placement: Placement)], camera: Camera) throws {
        guard !camera.isPerspective else {
            throw CompositeError("a composite is drawn under a parallel projection, and this camera is perspective")
        }
        self.camera = camera
        self.parts = try parts.map { name, scene, placement in
            guard placement.scale > 0, placement.scale.isFinite,
                  placement.offset.x.isFinite, placement.offset.y.isFinite, placement.offset.z.isFinite else {
                throw CompositeError("\(name): a placement is a positive scale and a finite offset")
            }
            switch scene.geometry {
            case .parametric(let s) where s.neighbourhood == nil:
                throw CompositeError("""
                    \(name) is a surface with no exact occlusion test (it has no tubular neighbourhood): \
                    its coordinate image says only what of the surface itself is in front, nothing about \
                    another solid, so it cannot stand in a composite. It bakes alone.
                    """)
            case .heightfield where placement.scale != 1:
                throw CompositeError("""
                    \(name): a landscape stands at scale 1 in a composite, since its depth tolerance and \
                    its function's tolerance are lengths in its own domain; scale the other geometries instead
                    """)
            default: break
            }
            let view = placement.transform >>> camera.view
            return Part(name: name, scene: scene.looking(Camera(view: view, projection: camera.projection)),
                        placement: placement)
        }
    }

    /// Every part's layers in view space, in order, tested for the camera:
    /// each by its own geometry as alone, then cut wherever another part
    /// hides it. An unclipped layer is drawn whole.
    public func visibleLayers() -> [(Layer, PolylineSet<ViewSpace>)] {
        let occluders = parts.map(Occluder.init)
        let world = camera.view.inverse
        return parts.indices.flatMap { i -> [(Layer, PolylineSet<ViewSpace>)] in
            let others = occluders.indices.filter { $0 != i }.map { occluders[$0] }
            return parts[i].scene.visibleLayers().map { layer, seen in
                guard layer.spec.clipped, !others.isEmpty else { return (layer, seen) }
                return (layer, Self.cut(seen, by: others, world: world))
            }
        }
    }

    /// How a part hides the points of another's curves. A polyhedron's
    /// pieces hide the spans of a straight segment in closed form, asked of
    /// the segment's own ends in view space, where the part's occlusion is
    /// built (`Solid.Occlusion`). The other tests are asked of points
    /// (`OcclusionTest.isClear(ofAnotherSolid:)`), every `step` along a
    /// segment in world space and at its ends, and a change of verdict is
    /// bisected on the test.
    struct Occluder: Sendable {
        let pieces: Solid.Occlusion?
        /// A test for each worker thread, taking a world point.
        let points: (@Sendable () -> @Sendable (P3<WorldSpace>) -> Bool)?
        let step: Double

        init(_ part: Part) {
            let view = part.scene.camera.view, placement = part.placement
            switch part.scene.geometry {
            case .solid(let s):
                pieces = s.occlusion(view: view); points = nil; step = .infinity
            case .heightfield(let h):
                let test = HeightfieldVisibility(heightfield: h, view: view, depthTolerance: part.scene.depthTolerance)
                pieces = nil
                points = {
                    let mine = test.forked()
                    return { mine.isClear(ofAnotherSolid: placement.back($0)) }
                }
                // A quarter of a cell: the traversal's solid is linear
                // between its crossings, a cell apart.
                step = 0.25 * test.traversal.cell * placement.scale
            case .parametric(let s):
                // Refused at the composite's door without one.
                let test = SphereTracing(s, sight: view.projectionDirection)!
                pieces = nil
                points = { { test.isClear(ofAnotherSolid: placement.back($0)) } }
                step = test.neighbourhood.radius / 256 * placement.scale
            }
        }
    }

    /// The runs of `paths` (view space) that no occluder hides: each
    /// segment cut to the spans every occluder leaves, a run kept across a
    /// vertex where the spans either side meet it, and a run of fewer than
    /// two vertices dropped. `world` takes a view point back to world space.
    static func cut(_ paths: PolylineSet<ViewSpace>, by occluders: [Occluder],
                    world: Transform<ViewSpace, WorldSpace>) -> PolylineSet<ViewSpace> {
        guard !paths.vertices.isEmpty else { return paths }
        let pointed = occluders.filter { $0.points != nil }
        let step = pointed.map(\.step).min() ?? .infinity
        let worker: @Sendable () -> @Sendable (P3<WorldSpace>) -> Bool = {
            let tests = pointed.map { $0.points!() }
            return { p in tests.allSatisfy { $0(p) } }
        }
        let vertices = paths.vertices
        // Every vertex's verdict under the point tests, once.
        let clear = Heightfield.parallelMap(vertices, chunk: 16, worker: worker) { test, v in test(world(v)) }
        // Every segment's seen spans, as fractions of the way along it.
        var segments: [Int] = []
        for path in 0..<paths.count {
            for k in (paths.offsets[path] + 1)..<paths.offsets[path + 1] { segments.append(k) }
        }
        let spans = Heightfield.parallelMap(segments, chunk: 8, worker: worker) { test, k -> [(Double, Double)] in
            let a = vertices[k - 1], b = vertices[k]
            var hidden: [(Double, Double)] = []
            for o in occluders {
                if let pieces = o.pieces { hidden += pieces.hidden(from: a, to: b, excluding: { _ in false }) }
            }
            if !pointed.isEmpty {
                let wa = world(a), wb = world(b)
                let n = max(1, Int((simd_length(wb.v - wa.v) / step).rounded(.up)))
                func at(_ t: Double) -> P3<WorldSpace> { P3(wa.v + (wb.v - wa.v) * t) }
                var t0 = 0.0, c0 = clear[k - 1]
                for i in 1...n {
                    let t1 = i == n ? 1 : Double(i) / Double(n)
                    let c1 = i == n ? clear[k] : test(at(t1))
                    if c0 != c1 {
                        // Where the verdict turns, to the last bit.
                        var lo = t0, hi = t1
                        while hi - lo > 1e-13 {
                            let mid = 0.5 * (lo + hi)
                            if test(at(mid)) == c0 { lo = mid } else { hi = mid }
                        }
                        if c0 { hidden.append((hi, t1)) } else { hidden.append((t0, lo)) }
                    } else if !c0 {
                        hidden.append((t0, t1))
                    }
                    t0 = t1; c0 = c1
                }
            }
            return seen(after: hidden)
        }
        var out: [[P3<ViewSpace>]] = []
        var s = 0
        for path in 0..<paths.count {
            var run: [P3<ViewSpace>] = []
            var openAtEnd = false
            for k in (paths.offsets[path] + 1)..<paths.offsets[path + 1] {
                let a = vertices[k - 1], b = vertices[k]
                let continuing = openAtEnd
                openAtEnd = false
                for (lo, hi) in spans[s] where hi - lo > 1e-12 {
                    let start = along(a, b, lo), end = along(a, b, hi)
                    if continuing, lo == 0, !run.isEmpty {
                        run.append(end)
                    } else {
                        if run.count >= 2 { out.append(run) }
                        run = [start, end]
                    }
                    openAtEnd = hi == 1
                }
                if !openAtEnd {
                    if run.count >= 2 { out.append(run) }
                    run = []
                }
                s += 1
            }
            if run.count >= 2 { out.append(run) }
        }
        return PolylineSet(paths: out)
    }

    /// The spans of [0, 1] left by `hidden`, in order.
    static func seen(after hidden: [(Double, Double)]) -> [(Double, Double)] {
        var t = 0.0
        var spans: [(Double, Double)] = []
        for (a, b) in hidden.sorted(by: { $0.0 < $1.0 }) {
            if a > t { spans.append((t, a)) }
            t = max(t, b)
        }
        if t < 1 { spans.append((t, 1)) }
        return spans
    }

    /// The point `t` of the way from `a` to `b`, the ends to the bit.
    static func along(_ a: P3<ViewSpace>, _ b: P3<ViewSpace>, _ t: Double) -> P3<ViewSpace> {
        t == 0 ? a : (t == 1 ? b : P3(a.v + t * (b.v - a.v)))
    }
}
