import Foundation
import simd
import KurvenCore
import KurvenDynamics
import KurvenLandscape

// MARK: - several geometries in one scene
//
// A composite's every curve, held point by point to the brute-force casts
// the stress, solids and occlusion suites run, each against its own
// geometry where it stands: a landscape's drawn triangles, a polyhedron's
// pieces, the round torus's tube walked a step at a time. A point is seen
// exactly when its own geometry shows it -- the geometry drawn alone, which
// is held to its own cast here too -- and no other geometry's cast meets the
// line of sight from it.

/// One geometry of a composite as the gate sees it: whether the line of
/// sight from a world point toward the eye meets it, by brute force; nil
/// where the answer is a rounding (the line grazes it, or the point lies on
/// it).
typealias Cast = @Sendable (SIMD3<Double>, SIMD3<Double>) -> Bool?

/// Two directions across `d`, for nudging a point off the line it lies on.
func acrossSight(_ d: SIMD3<Double>) -> [SIMD3<Double>] {
    let a = simd_normalize(simd_cross(d, abs(d.z) < 0.9 ? SIMD3(0, 0, 1) : SIMD3(1, 0, 0)))
    let b = simd_cross(d, a)
    return [a, -a, b, -b]
}

/// A cast against triangles: clear when the ray meets none past a
/// billionth of `size`; nil when a nudge of a ten-millionth across the line
/// of sight changes that. A point inside a closed mesh meets it on the way
/// out.
func triangleCast(_ meshes: [Mesh<WorldSpace>], size: Double,
                  model: @escaping @Sendable (SIMD3<Double>) -> SIMD3<Double>) -> Cast {
    { p, toEye in
        func clear(_ q: SIMD3<Double>) -> Bool {
            meshes.allSatisfy { firstHit($0, from: model(q), along: toEye, after: 1e-9 * size) == nil }
        }
        let truth = clear(p)
        return acrossSight(toEye).allSatisfy { clear(p + 1e-7 * size * $0) == truth } ? truth : nil
    }
}

/// A cast against a round torus of radii `R`, `r` about the z axis: the line
/// walked a thousandth of the tube at a time through the ball that holds the
/// torus, hidden where it is ever inside the tube; nil where the least
/// clearance seen is within a millionth of the tube, where the walk's step
/// decides, or the point lies within two of sphere tracing's least steps of
/// it (a hundred-thousandth of the ball), where that step decides.
func torusCast(R: Double, r: Double, model: @escaping @Sendable (SIMD3<Double>) -> SIMD3<Double>) -> Cast {
    { p, toEye in
        let q = model(p)
        func clearance(_ x: SIMD3<Double>) -> Double {
            let rho = (x.x * x.x + x.y * x.y).squareRoot()
            return ((rho - R) * (rho - R) + x.z * x.z).squareRoot() - r
        }
        let ball = R + r
        let b = simd_dot(q, toEye), c = simd_dot(q, q) - ball * ball
        let disc = b * b - c
        var least = clearance(q)
        if abs(least) < 2e-5 * (R + r) { return nil }
        if least < 0 { return false }
        if disc > 0 {
            let exit = -b + disc.squareRoot()
            var t = max(-b - disc.squareRoot(), 0)
            while t < exit {
                let g = clearance(q + t * toEye)
                least = min(least, g)
                if g < -1e-6 * r { return false }
                t += 1e-3 * r
            }
        }
        return abs(least) > 1e-6 * r ? true : nil
    }
}

/// A cast against a landscape's function, as the stress suite's dense
/// cast walks it: the line from the point walked a sixty-fourth of a cell at
/// a time over the domain until it is above the highest sample, hidden where
/// it is ever under min(|f|, h), the point itself included; nil where the
/// least gap seen is within a ten-thousandth of a cell, where the walk's
/// step decides.
func functionCast(_ h: Heightfield, model: @escaping @Sendable (SIMD3<Double>) -> SIMD3<Double>) -> Cast {
    let magnitude = h.refine!.magnitude!
    let d = h.surface.domain, g = h.surface.height
    let lo = SIMD2(min(d.real.lo, d.real.hi), min(d.imag.lo, d.imag.hi))
    let hi = SIMD2(max(d.real.lo, d.real.hi), max(d.imag.lo, d.imag.hi))
    let cell = max(abs(d.real.length) / Double(g.width - 1), abs(d.imag.length) / Double(g.height - 1))
    var highest = -Double.infinity
    h.forEachSample { highest = max(highest, $0.z) }
    let top = highest
    let truncation = h.surface.truncation
    return { p, toEye in
        let q = model(p)
        // The stretch of the line over the domain's box, below the top.
        var ta = 0.0, tb = toEye.z > 0 ? max((top - q.z) / toEye.z, 0) : 1e3
        for axis in 0..<2 {
            let o = axis == 0 ? q.x : q.y, v = axis == 0 ? toEye.x : toEye.y
            let (a, b) = (lo[axis], hi[axis])
            if abs(v) < 1e-15 {
                if o < a || o > b { return true }
                continue
            }
            let t1 = (a - o) / v, t2 = (b - o) / v
            ta = max(ta, min(t1, t2)); tb = min(tb, max(t1, t2))
        }
        guard ta <= tb else { return true }
        var least = Double.infinity
        var t = ta
        while t <= tb {
            let x = q + t * toEye
            let u = magnitude(P2<DomainSpace>(x.x, x.y))
            let gap = x.z - min(u.isNaN ? .infinity : u, truncation.height(atX: x.x))
            least = min(least, gap)
            if gap < -1e-4 * cell { return false }
            t += cell / 64
        }
        return abs(least) > 1e-4 * cell ? true : nil
    }
}

/// The brute-force cast against a part as it stands.
func compositeCast(_ part: Composite.Part) -> Cast {
    let s = part.placement.scale, o = part.placement.offset
    let model: @Sendable (SIMD3<Double>) -> SIMD3<Double> = { ($0 - o) / s }
    switch part.scene.geometry {
    case .heightfield(let h):
        if h.refine?.magnitude != nil { return functionCast(h, model: model) }
        return triangleCast([drawnTriangles(h, base: 0)], size: 4, model: model)
    case .solid(let solid):
        let size = solid.vertices.map { simd_length($0.v) }.max()!
        return triangleCast(solid.pieces.indices.map { pieceMesh(solid, $0) }, size: size, model: model)
    case .parametric:
        // The torus fixture's radii; nothing else stands in these composites.
        return torusCast(R: 2, r: 0.75, model: model)
    }
}

/// Polylines in view space, indexed by the cells of a grid over the page
/// their segments cross: which segment passes through a point, and where
/// the runs end.
struct SegmentIndex {
    let lo: SIMD2<Double>, cell: Double, n: Int
    var buckets: [[(a: SIMD3<Double>, b: SIMD3<Double>)]]
    var ends: [[SIMD3<Double>]]

    init(_ paths: PolylineSet<ViewSpace>, lo: SIMD2<Double>, size: Double, n: Int = 400) {
        self.lo = lo; self.cell = size / Double(n); self.n = n
        buckets = Array(repeating: [], count: n * n)
        ends = Array(repeating: [], count: n * n)
        for i in 0..<paths.count {
            let path = paths[path: i]
            for (a, b) in zip(path, path.dropFirst()) {
                let (i0, j0) = key(SIMD2(min(a.x, b.x), min(a.y, b.y)))
                let (i1, j1) = key(SIMD2(max(a.x, b.x), max(a.y, b.y)))
                for i in i0...i1 { for j in j0...j1 { buckets[i * n + j].append((a.v, b.v)) } }
            }
            for e in [path.first!, path.last!] {
                let (i, j) = key(SIMD2(e.x, e.y))
                ends[i * n + j].append(e.v)
            }
        }
    }

    func key(_ p: SIMD2<Double>) -> (Int, Int) {
        let k = (p - lo) / cell
        return (min(max(Int(k.x.rounded(.down)), 0), n - 1), min(max(Int(k.y.rounded(.down)), 0), n - 1))
    }

    /// Whether a segment passes within `tolerance` of `p`, in space.
    func holds(_ p: SIMD3<Double>, tolerance: Double) -> Bool {
        let (i, j) = key(SIMD2(p.x, p.y))
        return buckets[i * n + j].contains { a, b in
            let ab = b - a, l2 = simd_length_squared(ab)
            let t = l2 > 0 ? min(max(simd_dot(p - a, ab) / l2, 0), 1) : 0
            return simd_length(p - (a + t * ab)) <= tolerance
        }
    }

    /// The nearest run end to `p`, within the cells about it.
    func nearestEnd(_ p: SIMD3<Double>) -> Double {
        let (i, j) = key(SIMD2(p.x, p.y))
        var best = Double.infinity
        for di in -1...1 {
            for dj in -1...1 {
                let a = i + di, b = j + dj
                guard a >= 0, a < n, b >= 0, b < n else { continue }
                for e in ends[a * n + b] { best = min(best, simd_length(e - p)) }
            }
        }
        return best
    }

    var allEnds: [SIMD3<Double>] { ends.flatMap { $0 } }

    /// The segments that cross the cell `p` is in.
    func segments(near p: SIMD3<Double>) -> [(a: SIMD3<Double>, b: SIMD3<Double>)] {
        let (i, j) = key(SIMD2(p.x, p.y))
        return buckets[i * n + j]
    }
}

/// What the gate found for one composition.
struct CompositeVerdict {
    var compared = 0, wrong = 0, hiddenByOther = 0, undecided = 0, outsideAlone = 0
    var ownCompared = 0, ownWrong = 0
    var example = "", ownExample = ""
}

/// The gate for one composite under one camera. Each layer's composite
/// runs are checked to lie on its runs drawn alone, and points sampled
/// densely along those, and on either side of every run end of both, are
/// drawn in the composite exactly when no other part's cast meets their
/// line of sight. Each part drawn alone is held to its own cast: a
/// polyhedron's edges and hatching anywhere, a landscape's graph and the
/// torus's curves at their vertices, where their own tests decide.
func compositeVerdict(_ composite: Composite, casts: [Cast]) -> CompositeVerdict {
    var verdict = CompositeVerdict()
    let drawn = composite.visibleLayers()
    let view = composite.camera.view, world = view.inverse
    let toEye = -view.projectionDirection
    var index = 0
    for (i, part) in composite.parts.enumerated() {
        let alone = part.scene.visibleLayers()
        let others = casts.indices.filter { $0 != i }.map { casts[$0] }
        let partView = part.scene.camera.view
        for (layer, seen) in alone {
            defer { index += 1 }
            guard index < drawn.count, drawn[index].0.spec == layer.spec else {
                verdict.wrong += 1; verdict.example = "layers out of order at \(layer.spec.name)"; continue
            }
            let composed = drawn[index].1
            guard layer.spec.clipped else {
                if composed != seen { verdict.wrong += 1; verdict.example = "\(layer.spec.name), unclipped, was cut" }
                continue
            }
            guard let bounds = AABB(seen.vertices + composed.vertices) else { continue }
            let size = max(bounds.size.x, bounds.size.y, 1e-9)
            let lo = SIMD2(bounds.lo.x, bounds.lo.y) - 1e-6 * size
            let aloneIndex = SegmentIndex(seen, lo: lo, size: 1.00001 * size)
            let composedIndex = SegmentIndex(composed, lo: lo, size: 1.00001 * size)
            let tolerance = 1e-9 * size
            for v in composed.vertices where !aloneIndex.holds(v.v, tolerance: tolerance) {
                verdict.outsideAlone += 1
            }
            // The samples: every segment of the runs drawn alone, densely,
            // and a hair either side of every run end.
            let h = size / 2000
            var samples: [(p: SIMD3<Double>, length: Double)] = []
            let ends = aloneIndex.allEnds + composedIndex.allEnds
            for k in 0..<seen.count {
                let path = seen[path: k]
                for (a, b) in zip(path, path.dropFirst()) {
                    let ab = b.v - a.v, length = simd_length(ab)
                    guard length > 0 else { continue }
                    let m = max(2, Int((length / h).rounded(.up)))
                    for j in 0...m { samples.append((a.v + ab * (Double(j) / Double(m)), length)) }
                }
            }
            for e in ends {
                // The alone segments the end lies on.
                for (a, b) in aloneIndex.segments(near: e) {
                    let ab = b - a, l2 = simd_length_squared(ab)
                    guard l2 > 0 else { continue }
                    let t = simd_dot(e - a, ab) / l2
                    guard t >= -1e-9, t <= 1 + 1e-9, simd_length(e - (a + t * ab)) <= tolerance else { continue }
                    for d in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1] {
                        for u in [t - d, t + d] where u > 0 && u < 1 {
                            samples.append((a + ab * u, l2.squareRoot()))
                        }
                    }
                }
            }
            let results = Heightfield.parallelMap(samples, chunk: 64) { sample -> (decided: Bool, truth: Bool, drawn: Bool) in
                let p = sample.p
                // At a run end the truth is a boundary.
                if min(aloneIndex.nearestEnd(p), composedIndex.nearestEnd(p)) < max(1e-6 * sample.length, 1e-12 * size) {
                    return (false, false, false)
                }
                let w = world(P3<ViewSpace>(p)).v
                var truth = true
                for cast in others {
                    guard let clear = cast(w, toEye) else { return (false, false, false) }
                    if !clear { truth = false }
                }
                return (true, truth, composedIndex.holds(p, tolerance: tolerance))
            }
            for (sample, r) in zip(samples, results) {
                guard r.decided else { verdict.undecided += 1; continue }
                verdict.compared += 1
                if !r.truth { verdict.hiddenByOther += 1 }
                if r.truth != r.drawn {
                    verdict.wrong += 1
                    if verdict.example.isEmpty {
                        let w = world(P3<ViewSpace>(sample.p))
                        verdict.example = String(format: " e.g. %@ of %@ at (%.4f, %.4f, %.4f): the casts say %@",
                                                 layer.spec.name, part.name, w.x, w.y, w.z, r.truth ? "seen" : "hidden")
                    }
                }
            }
            // The part alone, held to its own cast.
            let aloneHolds: (SIMD3<Double>) -> Bool? = { q in
                let p = partView(P3<WorldSpace>(q)).v
                if aloneIndex.nearestEnd(p) < 1e-6 * size { return nil }
                return aloneIndex.holds(p, tolerance: tolerance)
            }
            let own = ownTruths(part, layer: layer, view: partView)
            for (q, truth) in own {
                guard let truth, let shown = aloneHolds(q) else { continue }
                verdict.ownCompared += 1
                if truth != shown {
                    verdict.ownWrong += 1
                    if verdict.ownExample.isEmpty {
                        verdict.ownExample = String(format: " e.g. %@ of %@ at (%.4f, %.4f, %.4f): its cast says %@",
                                                    layer.spec.name, part.name, q.x, q.y, q.z, truth ? "seen" : "hidden")
                    }
                }
            }
        }
    }
    if index != drawn.count { verdict.wrong += 1; verdict.example = "\(drawn.count) layers drawn, \(index) alone" }
    return verdict
}

/// Points of a part's own curves, in its own coordinates, with what its
/// own brute-force cast says of each (nil: undecided): a polyhedron's edges
/// and hatch lines sampled anywhere along them, as the solids suite samples
/// edges; a landscape's graph curves and the torus's curves at their
/// vertices, as the stress and surface suites test them.
func ownTruths(_ part: Composite.Part, layer: Layer, view: Transform<WorldSpace, ViewSpace>) -> [(SIMD3<Double>, Bool?)] {
    let toEye = -view.projectionDirection
    switch part.scene.geometry {
    case .solid(let s):
        var lines: [(path: [SIMD3<Double>], own: Set<Int>)] = []
        if case .edges = layer.spec.source {
            for (k, e) in s.edges.enumerated() where !s.isDiagonal[k] {
                guard abs(simd_dot(s.normals[e.left], toEye)) > 1e-9, abs(simd_dot(s.normals[e.right], toEye)) > 1e-9 else { continue }
                lines.append(([s.vertices[e.a].v, s.vertices[e.b].v], [e.left, e.right]))
            }
        } else if case .hatching(let hatching, .solid) = layer.spec.source {
            let width = layer.spec.name.split(separator: " ").last.flatMap { w -> Int? in
                w.contains("/") ? Int(w.split(separator: "/")[0]) : nil
            }
            for line in s.hatchLines(hatching) where width == nil || line.widthStep == width {
                let pieces = Set(s.faces[line.face])
                guard abs(simd_dot(s.normals[s.faces[line.face][0]], toEye)) > 1e-9 else { continue }
                lines.append((line.path.map(\.v), pieces))
            }
        }
        var out: [(SIMD3<Double>, Bool?)] = []
        for (path, own) in lines {
            for (a, b) in zip(path, path.dropFirst()) {
                for k in 0..<40 {
                    let q = a + (Double(k) + 0.5) / 40 * (b - a)
                    out.append((q, rayVisible(s, at: q, own: own, toEye: toEye)))
                }
            }
        }
        return out
    case .heightfield(let h):
        // On f the stress suite holds the traversal to its dense cast.
        guard h.refine == nil, part.scene.geometry.face(of: layer.spec) == .graph else { return [] }
        let mesh = drawnTriangles(h, base: 0)
        let vis = HeightfieldVisibility(heightfield: h, view: view, depthTolerance: part.scene.depthTolerance)
        let c = view.m.columns
        let perUnit = abs(simd_dot(SIMD3(c.0.z, c.1.z, c.2.z), toEye))
        let d = h.surface.domain, g = h.surface.height
        let cell = max(abs(d.real.length) / Double(g.width - 1), abs(d.imag.length) / Double(g.height - 1))
        let box = (lo: SIMD2(min(d.real.lo, d.real.hi), min(d.imag.lo, d.imag.hi)),
                   hi: SIMD2(max(d.real.lo, d.real.hi), max(d.imag.lo, d.imag.hi)))
        let vertices = layer.paths.vertices
        return Heightfield.parallelMap(vertices) { v -> (SIMD3<Double>, Bool?) in
            let f = vis.facing(v.xy)
            guard abs(f) > 1e-9 else { return (v.v, nil) }
            let strictlyInside = v.x > box.lo.x + 1e-9 && v.x < box.hi.x - 1e-9
                && v.y > box.lo.y + 1e-9 && v.y < box.hi.y - 1e-9
            return (v.v, f > 0 && !exactlyHidden(mesh, from: v.v, along: toEye, depthTolerance: vis.depthTolerance,
                                               perUnit: perUnit, strictlyInside: strictlyInside, cell: cell))
        }
    case .parametric(let s):
        guard layer.spec.role != .outline, let coords = layer.paths.coords else { return [] }
        let cast = torusCast(R: 2, r: 0.75, model: { $0 })
        return Heightfield.parallelMap(Array(coords.indices)) { k -> (SIMD3<Double>, Bool?) in
            let q = layer.paths.vertices[k].v
            guard let f = s.facing(coords[k], sight: -toEye), abs(f) > 1e-9 else { return (q, nil) }
            guard f > 0 else { return (q, false) }
            return (q, cast(s.map(coords[k]).position, toEye))
        }
    }
}

func compositeTests() {
    let contract = Fixtures.dir.appendingPathComponent("contract")
    func polyhedron(_ name: String) throws -> Scene {
        let b = try PolyhedronBundle.read(at: contract.appendingPathComponent("polyhedron_\(name).kurven"))
        return b.scene(preset: b.manifest.presets[0])
    }
    func surface(_ name: String) throws -> Scene {
        let b = try SurfaceBundle.read(at: contract.appendingPathComponent("surface_\(name).kurven"))
        return b.scene(preset: b.manifest.presets[0])
    }

    Check.suite("composite: every curve is seen exactly where no geometry's brute-force cast meets its line of sight") {
        // The cubic landscape's grid: its solid is the drawn triangles, which
        // the cast is against. Three towers at the cube roots of unity.
        let bundle = try KurvenBundle.read(at: Fixtures.dir.appendingPathComponent("expr/landscape_cubic.kurven"))
        let landscape = Scene(bundle: bundle, preset: bundle.manifest.presets[0])
        let cube = try polyhedron("cube"), frame = try polyhedron("frame"), torus = try surface("torus")
        let cameras: [(String, Camera)] = [
            ("the landscape's", landscape.camera),
            ("el 30° az 37°", Camera(view: testCamera(elevation: -30, azimuth: 37), projection: .orthographic(oblique: nil))),
            ("el 55° sheared", Camera(view: testCamera(elevation: -55, azimuth: 160, shear: 0.5), projection: .orthographic(oblique: nil))),
        ]
        let compositions: [(String, [(String, Scene, Placement)])] = [
            ("a cube standing on the landscape behind a tower",
             [("landscape", landscape, .identity), ("cube", cube, Placement(scale: 0.35, offset: SIMD3(-1.2, 0.1, 0.3)))]),
            ("the round torus ringing a tower",
             [("landscape", landscape, .identity), ("torus", torus, Placement(scale: 0.35, offset: SIMD3(1, 0, 1.6)))]),
            ("the round torus sunk into the foot of a tower",
             [("landscape", landscape, .identity), ("torus", torus, Placement(scale: 0.3, offset: SIMD3(0.4, 1.3, 0.3)))]),
            ("a frame sunk into the slope of a tower",
             [("landscape", landscape, .identity), ("frame", frame, Placement(scale: 0.3, offset: SIMD3(0.3, -1.3, 0.2)))]),
            // Moved together, the landscape too.
            ("all three, moved",
             [("landscape", landscape, Placement(offset: SIMD3(0.4, 0.2, -0.1))),
              ("cube", cube, Placement(scale: 0.35, offset: SIMD3(-0.8, 0.3, 0.2))),
              ("torus", torus, Placement(scale: 0.35, offset: SIMD3(1.4, 0.2, 1.5)))]),
        ]
        // The same landscape with its function, whose solid is f: the
        // frame's curves held to the dense cast against f, under the
        // landscape's own camera.
        let refined = bundle.refined(by: NativeLandscape.refiner(for: bundle)?.refine)
        let onF = Scene(bundle: refined, preset: refined.manifest.presets[0])
        let clock = ContinuousClock()
        for (name, parts, cameras) in compositions.map({ ($0.0, $0.1, cameras) })
            + [("a frame sunk into the slope of a tower, on f",
                [("landscape", onF, .identity), ("frame", frame, Placement(scale: 0.3, offset: SIMD3(0.3, -1.3, 0.2)))],
                Array(cameras.prefix(1)))] {
            var total = CompositeVerdict()
            let elapsed = try clock.measure {
                for (_, camera) in cameras {
                    let composite = try Composite(parts, camera: camera)
                    let v = compositeVerdict(composite, casts: composite.parts.map(compositeCast))
                    total.compared += v.compared; total.wrong += v.wrong; total.undecided += v.undecided
                    total.hiddenByOther += v.hiddenByOther; total.outsideAlone += v.outsideAlone
                    total.ownCompared += v.ownCompared; total.ownWrong += v.ownWrong
                    if total.example.isEmpty { total.example = v.example }
                    if total.ownExample.isEmpty { total.ownExample = v.ownExample }
                }
            }
            Check.expect(total.wrong == 0 && total.outsideAlone == 0 && total.compared > 20_000
                         && total.hiddenByOther > total.compared / 50,
                         "\(name): drawn exactly where no other geometry's cast meets the line of sight",
                         "\(total.wrong) of \(total.compared) disagree, \(total.hiddenByOther) hidden by another, "
                         + "\(total.undecided) at a boundary, \(total.outsideAlone) vertices off the drawing alone, "
                         + "\(elapsed)\(total.example)")
            Check.expect(total.ownWrong == 0 && total.ownCompared > 2_000,
                         "\(name): each geometry alone is its own cast",
                         "\(total.ownWrong) of \(total.ownCompared) disagree\(total.ownExample)")
        }
    }

    Check.suite("composite: a surface with no exact occlusion test is refused") {
        let bundle = try KurvenBundle.read(at: Fixtures.dir.appendingPathComponent("expr/landscape_cubic.kurven"))
        let landscape = Scene(bundle: bundle, preset: bundle.manifest.presets[0])
        let forced = try surface("forced")
        do {
            _ = try Composite([("landscape", landscape, .identity), ("forced", forced, .identity)], camera: landscape.camera)
            Check.expect(false, "the forced torus is refused in a composite", "it was taken")
        } catch let error as CompositeError {
            Check.expect(error.description.contains("no exact occlusion test"),
                         "the forced torus is refused in a composite, saying why", error.description)
        }
        Check.expectThrows("a landscape scaled is refused") {
            _ = try Composite([("landscape", landscape, Placement(scale: 2)), ("cube", try polyhedron("cube"), .identity)],
                              camera: landscape.camera)
        }
    }
}
