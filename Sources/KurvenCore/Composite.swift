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

    /// The scales a placement takes, and how far from the origin its offset
    /// reaches on each axis. Within them a geometry's arithmetic in world
    /// space stays far from overflow and from the rounding of its distance
    /// to the origin, and the steps along a segment stay countable.
    public static let scales: ClosedRange<Double> = 1e-3...1e3
    public static let reach = 1e4

    /// Why this placement is out of range; nil when it is in it.
    public var refusal: String? {
        guard scale.isFinite, Placement.scales.contains(scale) else {
            return "a placement's scale is between \(Placement.scales.lowerBound) and \(Placement.scales.upperBound), not \(scale)"
        }
        guard (0..<3).allSatisfy({ offset[$0].isFinite && abs(offset[$0]) <= Placement.reach }) else {
            return "a placement's offset is within \(Placement.reach) of the origin on each axis, not "
                + "(\(offset.x), \(offset.y), \(offset.z))"
        }
        return nil
    }

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
/// Another geometry cuts a curve's segments where their lines of sight meet
/// it (`Occluder`): a polyhedron and a landscape on its grid in closed form,
/// the round torus by the clearances of its sphere tracing.
///
/// Only a geometry that cuts another's segments exactly stands in a
/// composite: a landscape on its grid, a polyhedron, or the round torus. The
/// coordinate image says only whether a point is the front-most of its own
/// surface, nothing about another solid, so a surface tested by it -- the
/// forced torus -- is refused; so is a displaced torus, whose sphere tracing
/// steps by a Lipschitz constant read off its grid rather than proved, and a
/// landscape on its function, whose slope nothing bounds. No depth buffer
/// stands in between solids: the drawing is a function of the geometry and
/// the camera.
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
    let occluders: [Occluder]

    /// The parts, each `scene` taken with its geometry, layers and depth
    /// tolerance (its camera is the composite's), standing at its placement.
    public init(_ parts: [(name: String, scene: Scene, placement: Placement)], camera: Camera) throws {
        guard !camera.isPerspective else {
            throw CompositeError("a composite is drawn under a parallel projection, and this camera is perspective")
        }
        self.camera = camera
        self.parts = try parts.map { name, scene, placement in
            if let why = placement.refusal { throw CompositeError("\(name): \(why)") }
            switch scene.geometry {
            case .parametric(let s) where s.neighbourhood == nil:
                throw CompositeError("""
                    \(name) is a surface with no exact occlusion test (it has no tubular neighbourhood): \
                    its coordinate image says only what of the surface itself is in front, nothing about \
                    another solid, so it cannot stand in a composite. It bakes alone.
                    """)
            case .parametric(let s) where s.neighbourhood!.above != 0 || s.neighbourhood!.below != 0:
                throw CompositeError("""
                    \(name) is a displaced torus: its sphere tracing steps by a Lipschitz constant read \
                    off its grid, not proved, so it cannot hide another solid's curves exactly, and it \
                    cannot stand in a composite. It bakes alone.
                    """)
            case .heightfield(let h) where h.refine?.magnitude != nil:
                throw CompositeError("""
                    \(name) is a landscape on its function: nothing bounds f's slope, so it cannot cut \
                    another solid's curves exactly, and it cannot stand in a composite. It stands on its \
                    grid (kurven-cli scene --no-refine), or bakes alone.
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
        occluders = try self.parts.map(Occluder.init)
    }

    /// Every part's layers in view space, in order, tested for the camera:
    /// each by its own geometry as alone, then cut wherever another part
    /// hides it. After a part's own layers comes its contact with the
    /// landscapes, where it has one (`visibleContact(of:)`), cut the same
    /// way: a landscape hides its own contact curve only where the line of
    /// sight goes into its slab, as it hides any point on its boundary
    /// (`Slab`). An unclipped layer is drawn whole. Throws where a segment
    /// would be asked at more points against one part than
    /// `Occluder.steps`.
    public func visibleLayers() throws -> [(Layer, PolylineSet<ViewSpace>)] {
        let world = camera.view.inverse
        return try parts.indices.flatMap { i -> [(Layer, PolylineSet<ViewSpace>)] in
            let others = occluders.indices.filter { $0 != i }.map { occluders[$0] }
            let own = parts[i].scene.visibleLayers() + (visibleContact(of: i).map { [$0] } ?? [])
            return try own.map { layer, seen in
                guard layer.spec.clipped, !others.isEmpty else { return (layer, seen) }
                return (layer, try Self.cut(seen, by: others, world: world))
            }
        }
    }

    /// How a part hides the segments of another's curves: the spans of a
    /// segment whose lines of sight meet the part, as fractions of the way
    /// along it. Each works only where those lines can meet the part at all,
    /// through the box or the ball that holds it.
    enum Occluder: Sendable {
        /// A polyhedron's pieces, in closed form, asked of the segment's ends
        /// in view space, where its occlusion is built.
        case pieces(Solid.Occlusion)
        /// A landscape on its grid, in closed form (`Slab`), in its own
        /// coordinates: world space less its offset.
        case slab(Slab, offset: SIMD3<Double>)
        /// The round torus: sphere tracing's clearances
        /// (`SphereTracing.clearance(ofAnotherSolid:)`), each a stretch of the
        /// segment with one verdict.
        case tube(SphereTracing, placement: Placement)

        /// The most points a segment is asked at against one part: past it,
        /// the drawing is refused rather than cut.
        static let steps = 1_000_000

        init(_ part: Part) throws {
            let view = part.scene.camera.view, placement = part.placement
            switch part.scene.geometry {
            case .solid(let s):
                self = .pieces(s.occlusion(view: view))
            case .heightfield(let h):
                // On its grid: one on its function is refused at the door.
                self = .slab(try Slab(h, name: part.name), offset: placement.offset)
            case .parametric(let s):
                // Refused at the composite's door without one.
                self = .tube(SphereTracing(s, sight: view.projectionDirection)!, placement: placement)
            }
        }

        /// The spans of the segment `a -> b` (view space; `world` takes it
        /// back to world space) whose lines of sight, along `toward`, meet
        /// this part.
        func hidden(from a: P3<ViewSpace>, to b: P3<ViewSpace>,
                    world: Transform<ViewSpace, WorldSpace>, toward: SIMD3<Double>) throws -> [(Double, Double)] {
            switch self {
            case .pieces(let occlusion):
                return occlusion.hidden(from: a, to: b, excluding: { _ in false })
            case .slab(let slab, let offset):
                return slab.hidden(from: world(a).v - offset, to: world(b).v - offset, toward: toward)
            case .tube(let tracing, let placement):
                let wa = placement.back(world(a)).v, wb = placement.back(world(b)).v
                let e = tracing.neighbourhood
                guard let (s0, s1) = Composite.sweep(ofBall: e.centre, radius: 1.01 * e.radius,
                                                     from: wa, to: wb, toward: toward) else { return [] }
                let length = simd_length(wb - wa), least = 1e-5 * e.radius
                func clearance(_ s: Double) -> Double { tracing.clearance(ofAnotherSolid: P3(wa + (wb - wa) * s)) }
                guard length > 0 else { return clearance(0) < 0 ? [(0, 1)] : [] }
                var hidden: [(Double, Double)] = []
                var s = s0, m = clearance(s0), steps = 0
                while s < s1 {
                    // Every point within the clearance keeps the verdict, so
                    // the next point to ask is the clearance along; a clearance
                    // under the least step is a verdict to that resolution.
                    let next = min(s + max(abs(m), least) / length, s1)
                    let mn = clearance(next)
                    if (m < 0) != (mn < 0) {
                        var lo = s, hi = next
                        while hi - lo > 1e-13 {
                            let mid = 0.5 * (lo + hi)
                            if (clearance(mid) < 0) == (m < 0) { lo = mid } else { hi = mid }
                        }
                        hidden.append(m < 0 ? (s, lo) : (hi, next))
                    } else if m < 0 {
                        hidden.append((s, next))
                    }
                    s = next; m = mn
                    steps += 1
                    guard steps <= Occluder.steps else { throw Composite.tooFine(Double(steps)) }
                }
                return hidden
            }
        }
    }

    static func tooFine(_ count: Double) -> CompositeError {
        CompositeError("a segment would be asked at \(Int(min(count, 1e18))) points against one geometry, "
                       + "past the \(Occluder.steps) a composite allows")
    }

    /// The fractions of the segment `a -> b` whose lines of sight, along
    /// the unit `d`, pass within `radius` of `centre`: where the distance
    /// from the centre to the line through the point is at most the radius,
    /// a quadratic in the fraction. Those that meet the ball ahead of the
    /// point are among them.
    static func sweep(ofBall centre: SIMD3<Double>, radius: Double, from a: SIMD3<Double>, to b: SIMD3<Double>,
                      toward d: SIMD3<Double>) -> (Double, Double)? {
        // The part of (a + s e - centre) across d.
        let w0 = a - centre, e = b - a
        let p0 = w0 - simd_dot(w0, d) * d, pe = e - simd_dot(e, d) * d
        let qa = simd_dot(pe, pe), qb = 2 * simd_dot(p0, pe), qc = simd_dot(p0, p0) - radius * radius
        if qa <= 1e-24 * radius * radius { return qc <= 0 ? (0, 1) : nil }
        let disc = qb * qb - 4 * qa * qc
        guard disc >= 0 else { return nil }
        let root = disc.squareRoot()
        let lo = max((-qb - root) / (2 * qa), 0), hi = min((-qb + root) / (2 * qa), 1)
        return lo <= hi ? (lo, hi) : nil
    }

    /// The runs of `paths` (view space) that no occluder hides: each
    /// segment cut to the spans every occluder leaves, a run kept across a
    /// vertex where the spans either side meet it, and a run of fewer than
    /// two vertices dropped. `world` takes a view point back to world space.
    static func cut(_ paths: PolylineSet<ViewSpace>, by occluders: [Occluder],
                    world: Transform<ViewSpace, WorldSpace>) throws -> PolylineSet<ViewSpace> {
        guard !paths.vertices.isEmpty else { return paths }
        // Toward the eye: view depth grows toward it.
        let toward = simd_normalize(world(P3<ViewSpace>(0, 0, 1)).v - world(P3<ViewSpace>(0, 0, 0)).v)
        let vertices = paths.vertices
        var segments: [Int] = []
        for path in 0..<paths.count {
            for k in (paths.offsets[path] + 1)..<paths.offsets[path + 1] { segments.append(k) }
        }
        let results = Heightfield.parallelMap(segments, chunk: 4, worker: { occluders }) {
            mine, k -> Result<[(Double, Double)], CompositeError> in
            let a = vertices[k - 1], b = vertices[k]
            do {
                var hidden: [(Double, Double)] = []
                for o in mine { hidden += try o.hidden(from: a, to: b, world: world, toward: toward) }
                return .success(seen(after: hidden))
            } catch let error as CompositeError {
                return .failure(error)
            } catch {
                return .failure(CompositeError("\(error)"))
            }
        }
        let spans = try results.map { try $0.get() }
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
