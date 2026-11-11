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
///
/// With `box`, the box a landscape's slab fills, a point on one of its
/// faces -- a wall, the floor, the box's top -- within a billionth of
/// `size` is on the solid's boundary, seen from outside: it is cast from a
/// millionth outside that face, where nudges cannot carry it in.
func triangleCast(_ meshes: [Mesh<WorldSpace>], size: Double, box: (lo: SIMD3<Double>, hi: SIMD3<Double>)? = nil,
                  model: @escaping @Sendable (SIMD3<Double>) -> SIMD3<Double>) -> Cast {
    { p, toEye in
        let (q, face) = box.map { outside(model(p), of: $0, size: size) } ?? (model(p), nil)
        func clear(_ x: SIMD3<Double>) -> Bool {
            meshes.allSatisfy { firstHit($0, from: x, along: toEye, after: 1e-9 * size) == nil }
        }
        let truth = clear(q)
        // On a face, nudged only along it.
        let nudges = acrossSight(toEye).compactMap { v -> SIMD3<Double>? in
            guard let n = face else { return v }
            let w = v - simd_dot(v, n) * n
            return simd_length(w) > 1e-6 ? simd_normalize(w) : nil
        }
        return nudges.allSatisfy { clear(q + 1e-7 * size * $0) == truth } ? truth : nil
    }
}

/// A point within a billionth of `size` of a face of `box`, and inside the
/// box's extent across it, moved a hundred-millionth of `size` out through
/// that face, with the face's outward normal; any other point as it is.
func outside(_ q: SIMD3<Double>, of box: (lo: SIMD3<Double>, hi: SIMD3<Double>),
             size: Double) -> (SIMD3<Double>, SIMD3<Double>?) {
    let tol = 1e-9 * size
    guard (0..<3).allSatisfy({ q[$0] >= box.lo[$0] - tol && q[$0] <= box.hi[$0] + tol }) else { return (q, nil) }
    for axis in 0..<3 {
        var n = SIMD3<Double>.zero
        if abs(q[axis] - box.lo[axis]) <= tol { n[axis] = -1 } else if abs(q[axis] - box.hi[axis]) <= tol { n[axis] = 1 }
        if n != .zero { return (q + 1e-8 * size * n, n) }
    }
    return (q, nil)
}

/// The box a landscape's slab fills in its own coordinates: its domain,
/// from its base up to its highest sample.
func slabBox(_ h: Heightfield) -> (lo: SIMD3<Double>, hi: SIMD3<Double>) {
    let d = h.surface.domain
    var top = -Double.infinity
    h.forEachSample { top = max(top, $0.z) }
    var base = 0.0
    if case .perimeter(_, let b) = h.walls { base = b }
    return (SIMD3(min(d.real.lo, d.real.hi), min(d.imag.lo, d.imag.hi), base),
            SIMD3(max(d.real.lo, d.real.hi), max(d.imag.lo, d.imag.hi), top))
}

/// The corners of a landscape's skyline, in its own coordinates: every
/// node of its decimated grid at its drawn height, every point where a
/// triangle's side crosses the truncation, and the floor's corners. A
/// peak or a notch of what it hides is the image of one of these.
func skylineCorners(_ h: Heightfield) -> [SIMD3<Double>] {
    var out: [SIMD3<Double>] = []
    h.forEachSample { out.append($0.v) }
    let s = h.surface, g = s.height, step = max(h.step, 1)
    let nx = (g.width + step - 1) / step, ny = (g.height + step - 1) / step
    func node(_ i: Int, _ j: Int) -> SIMD3<Double> {
        let p = g.position(x: i * step, y: j * step)
        return SIMD3(p.x, p.y, Double(g[i * step, j * step]))
    }
    for j in 0..<ny {
        for i in 0..<nx {
            let a = node(i, j)
            for b in [i + 1 < nx ? node(i + 1, j) : nil, j + 1 < ny ? node(i, j + 1) : nil,
                      i + 1 < nx && j > 0 ? node(i + 1, j - 1) : nil].compactMap({ $0 }) {
                let ea = a.z - s.truncationHeight(atX: a.x), eb = b.z - s.truncationHeight(atX: b.x)
                guard (ea > 0) != (eb > 0), ea.isFinite, eb.isFinite else { continue }
                let t = ea / (ea - eb), x = a + t * (b - a)
                out.append(SIMD3(x.x, x.y, s.truncationHeight(atX: x.x)))
            }
        }
    }
    let box = slabBox(h)
    for x in [box.lo.x, box.hi.x] { for y in [box.lo.y, box.hi.y] { out.append(SIMD3(x, y, box.lo.z)) } }
    return h.tiles.flatMap { tile in out.map { v in SIMD3(tile(P3<WorldSpace>(v)).x, tile(P3<WorldSpace>(v)).y, v.z) } }
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

/// The cast from a point on a round torus of radii `R`, `r` about the z axis,
/// facing the eye: the line walked a thousandth of the tube at a time from a
/// ten-thousandth of the tube along it, through the ball that holds the
/// torus; hidden where it is ever inside the tube by a millionth of it. Nil
/// where it starts inside -- the point is on a fold, or the line grazes a
/// concave stretch -- or where, once it turns back toward the tube, it comes
/// within a millionth of it, where the walk's step decides. `torusCast`
/// casts a point off the torus; on it, its clearance is zero and it says
/// nothing.
func torusOwnCast(R: Double, r: Double) -> Cast {
    { q, toEye in
        func clearance(_ x: SIMD3<Double>) -> Double {
            let rho = (x.x * x.x + x.y * x.y).squareRoot()
            return ((rho - R) * (rho - R) + x.z * x.z).squareRoot() - r
        }
        let ball = R + r
        let b = simd_dot(q, toEye), c = simd_dot(q, q) - ball * ball
        let exit = -b + max(b * b - c, 0).squareRoot()
        var t = 1e-4 * r
        var previous = clearance(q + t * toEye)
        guard previous > 0 else { return nil }
        var rising = true, least = Double.infinity
        while t < exit {
            t += 1e-3 * r
            let g = clearance(q + t * toEye)
            if g < -1e-6 * r { return false }
            if rising, g < previous { rising = false }
            if !rising { least = min(least, g) }
            previous = g
        }
        return least > 1e-6 * r ? true : nil
    }
}

/// The brute-force cast against a part as it stands.
func compositeCast(_ part: Composite.Part) -> Cast {
    let s = part.placement.scale, o = part.placement.offset
    let model: @Sendable (SIMD3<Double>) -> SIMD3<Double> = { ($0 - o) / s }
    switch part.scene.geometry {
    case .heightfield(let h):
        let box = slabBox(h)
        return triangleCast([drawnTriangles(h, base: box.lo.z)], size: 4, box: box, model: model)
    case .solid(let solid):
        let size = solid.vertices.map { simd_length($0.v) }.max()!
        return triangleCast(solid.pieces.indices.map { pieceMesh(solid, $0) }, size: size, model: model)
    case .parametric(let surface):
        let (R, r) = torusRadii(surface)
        return torusCast(R: R, r: r, model: model)
    }
}

/// A round torus's radii, read off its neighbourhood: the ball that holds it
/// has radius R + r, and the centre stands R − r off the tube. A composite
/// takes no other parametric surface.
func torusRadii(_ s: ParametricSurface) -> (R: Double, r: Double) {
    let e = s.neighbourhood!
    let d0 = e.foot(e.centre).distance
    return ((e.radius + d0) / 2, (e.radius - d0) / 2)
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

/// A landscape's drawn triangles (`solid`), and those that face up -- its
/// top -- bucketed over (x, y): the top's normals at a point on it, and a
/// cast from such a point.
struct TerrainTops: Sendable {
    let solid: Mesh<WorldSpace>
    let triangles: [(a: SIMD3<Double>, b: SIMD3<Double>, c: SIMD3<Double>, n: SIMD3<Double>)]
    let lo: SIMD2<Double>, cell: SIMD2<Double>, n: Int
    let buckets: [[Int]]
    /// The landscape's size, the larger side of its box, and how far off a
    /// triangle's plane a point of it may lie: a billionth of its cell, the
    /// composite's tolerance.
    let size: Double, tolerance: Double

    init(_ h: Heightfield, n: Int = 128) {
        let box = slabBox(h)
        let mesh = drawnTriangles(h, base: box.lo.z)
        solid = mesh
        size = max(box.hi.x - box.lo.x, box.hi.y - box.lo.y)
        tolerance = 1e-9 * landscapeCell(h)
        var triangles: [(a: SIMD3<Double>, b: SIMD3<Double>, c: SIMD3<Double>, n: SIMD3<Double>)] = []
        for t in mesh.triangles {
            let a = mesh.vertices[Int(t.x)].v, b = mesh.vertices[Int(t.y)].v, c = mesh.vertices[Int(t.z)].v
            var normal = simd_cross(b - a, c - a)
            guard simd_length(normal) > 0 else { continue }
            normal = simd_normalize(normal)
            if normal.z < 0 { normal = -normal }
            guard normal.z > 1e-9 else { continue }
            triangles.append((a, b, c, normal))
        }
        let xs = triangles.flatMap { [$0.a, $0.b, $0.c] }
        let lo = xs.reduce(SIMD3(repeating: .infinity), simd_min), hi = xs.reduce(SIMD3(repeating: -.infinity), simd_max)
        self.lo = SIMD2(lo.x, lo.y); self.n = n
        cell = SIMD2(max(hi.x - lo.x, 1e-9), max(hi.y - lo.y, 1e-9)) / Double(n)
        var buckets = [[Int]](repeating: [], count: n * n)
        for (k, t) in triangles.enumerated() {
            let tlo = simd_min(simd_min(t.a, t.b), t.c), thi = simd_max(simd_max(t.a, t.b), t.c)
            let i0 = max(Int((tlo.x - lo.x) / cell.x) - 1, 0), i1 = min(Int((thi.x - lo.x) / cell.x) + 1, n - 1)
            let j0 = max(Int((tlo.y - lo.y) / cell.y) - 1, 0), j1 = min(Int((thi.y - lo.y) / cell.y) + 1, n - 1)
            for j in j0...j1 { for i in i0...i1 { buckets[j * n + i].append(k) } }
        }
        self.triangles = triangles; self.buckets = buckets
    }

    /// The cast from a point on the top, in the landscape's own
    /// coordinates (`model`): hidden where every triangle of the top it lies
    /// on faces away from the eye, since its line of sight goes into the
    /// solid there; where every one faces the eye, clear when the line
    /// meets no triangle past a billionth of the landscape's size, as
    /// `triangleCast` casts. Nil where they disagree or one is edge-on, off
    /// the top, and where a nudge of a ten-millionth of its size along the
    /// top and across the line of sight changes the verdict.
    func cast(model: @escaping @Sendable (SIMD3<Double>) -> SIMD3<Double>) -> Cast {
        { p, toEye in
            let q = model(p)
            let normals = self.normals(at: q)
            guard !normals.isEmpty else { return nil }
            let facing = normals.map { simd_dot($0, toEye) }
            if facing.allSatisfy({ $0 < -1e-9 }) { return false }
            guard facing.allSatisfy({ $0 > 1e-9 }) else { return nil }
            func clear(_ x: SIMD3<Double>) -> Bool { firstHit(solid, from: x, along: toEye, after: 1e-9 * size) == nil }
            let truth = clear(q)
            let nudges = normals.flatMap { n -> [SIMD3<Double>] in
                let v = simd_cross(toEye, n)
                return simd_length(v) > 1e-9 ? [simd_normalize(v), -simd_normalize(v)] : []
            }
            return nudges.allSatisfy { clear(q + 1e-7 * size * $0) == truth } ? truth : nil
        }
    }

    /// The normals of every triangle of the top `q` lies on, within the
    /// tolerance.
    func normals(at q: SIMD3<Double>) -> [SIMD3<Double>] {
        let i = Int((q.x - lo.x) / cell.x), j = Int((q.y - lo.y) / cell.y)
        guard i >= 0, i < n, j >= 0, j < n else { return [] }
        var out: [SIMD3<Double>] = []
        for k in buckets[j * n + i] {
            let t = triangles[k]
            // Barycentric in (x, y), and the plane's height there.
            let e1 = SIMD2(t.b.x - t.a.x, t.b.y - t.a.y), e2 = SIMD2(t.c.x - t.a.x, t.c.y - t.a.y)
            let r = SIMD2(q.x - t.a.x, q.y - t.a.y)
            let det = e1.x * e2.y - e1.y * e2.x
            guard det != 0 else { continue }
            let u = (r.x * e2.y - r.y * e2.x) / det, v = (e1.x * r.y - e1.y * r.x) / det
            guard u >= -1e-9, v >= -1e-9, u + v <= 1 + 1e-9 else { continue }
            let z = t.a.z + u * (t.b.z - t.a.z) + v * (t.c.z - t.a.z)
            if abs(z - q.z) <= tolerance { out.append(t.n) }
        }
        return out
    }
}

/// What the gate found for one composition.
struct CompositeVerdict {
    var compared = 0, wrong = 0, hiddenByOther = 0, undecided = 0, outsideAlone = 0
    var ownCompared = 0, ownWrong = 0
    /// Of the own comparisons, those on the torus its cast says are seen:
    /// what shows the cast gives a verdict on a point facing the eye.
    var torusSeen = 0
    /// Of the comparisons, those on a contact line, and of them those
    /// another part hides.
    var contactCompared = 0, contactHidden = 0
    /// Samples of a contact line on no landscape's top: each a failure.
    var offTerrain = 0
    var example = "", ownExample = ""
}

/// Whether a layer is a composite's contact line (`Composite.visibleContact(of:)`).
func isContact(_ spec: LayerSpec) -> Bool {
    spec.name == "contact" && spec.source == .file(vertices: "", offsets: "")
}

/// The gate for one composite under one camera. Each layer's composite
/// runs are checked to lie on its runs drawn alone, and points sampled
/// densely along those, and on either side of every run end of both, are
/// drawn in the composite exactly when no other part's cast meets their
/// line of sight. Each part drawn alone is held to its own cast: a
/// polyhedron's edges and hatching anywhere, a landscape's graph and the
/// torus's curves at their vertices, where their own tests decide.
func compositeVerdict(_ composite: Composite, casts: [Cast]) throws -> CompositeVerdict {
    var verdict = CompositeVerdict()
    let drawn = try composite.visibleLayers()
    // Each landscape's skyline corners and cell, in view space: where a
    // peak or a notch of what it hides can be.
    let marks: [(corners: [SIMD3<Double>], cell: Double)] = composite.parts.map { part in
        guard case .heightfield(let h) = part.scene.geometry else { return ([], 0) }
        let view = part.scene.camera.view
        let g = h.surface.height, d = h.surface.domain
        let cell = max(abs(d.real.length) / Double(g.width - 1), abs(d.imag.length) / Double(g.height - 1))
            * Double(max(h.step, 1))
        return (skylineCorners(h).map { view(P3<WorldSpace>($0)).v }, cell)
    }
    let view = composite.camera.view, world = view.inverse
    let toEye = -view.projectionDirection
    let terrainTops: [TerrainTops?] = composite.parts.map { part in
        guard case .heightfield(let h) = part.scene.geometry else { return nil }
        return TerrainTops(h)
    }
    var index = 0
    for (i, part) in composite.parts.enumerated() {
        let contact = composite.contact(of: i)
        let alone = part.scene.visibleLayers() + (composite.visibleContact(of: i).map { [$0] } ?? [])
        let others = casts.indices.filter { $0 != i }.map { casts[$0] }
        // A point of a contact line lies on one landscape's top: cast
        // against that one from where it lies (`TerrainTops.cast`), and
        // against the others as any point is.
        let otherParts = casts.indices.filter { $0 != i }
        let topCasts: [Cast?] = otherParts.map { j in
            guard let tops = terrainTops[j] else { return nil }
            let o = composite.parts[j].placement.offset
            return tops.cast(model: { $0 - o })
        }
        let owner: @Sendable (SIMD3<Double>) -> Int? = { w in
            otherParts.indices.first { k in
                guard let tops = terrainTops[otherParts[k]] else { return false }
                return !tops.normals(at: w - composite.parts[otherParts[k]].placement.offset).isEmpty
            }
        }
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
            // A hair either side of where each skyline corner of another
            // geometry crosses a segment on the page: a peak's tip hides a
            // span, and a notch shows a slit, as short as you like there,
            // and a missed one leaves no run end to sample beside.
            for (j, mark) in marks.enumerated() where j != i && !mark.corners.isEmpty {
                let near = SegmentIndex(PolylineSet(paths: mark.corners.map { [P3<ViewSpace>($0), P3<ViewSpace>($0)] }),
                                        lo: lo, size: 1.00001 * size, n: 200)
                for k in 0..<seen.count {
                    let path = seen[path: k]
                    for (a, b) in zip(path, path.dropFirst()) {
                        let e = SIMD2(b.x - a.x, b.y - a.y), l2 = simd_length_squared(e)
                        guard l2 > 0 else { continue }
                        var corners: [SIMD3<Double>] = []
                        let (i0, j0) = near.key(SIMD2(min(a.x, b.x), min(a.y, b.y)) - mark.cell)
                        let (i1, j1) = near.key(SIMD2(max(a.x, b.x), max(a.y, b.y)) + mark.cell)
                        for u in i0...i1 { for w in j0...j1 { corners += near.buckets[u * near.n + w].map(\.a) } }
                        let length = l2.squareRoot()
                        for c in corners {
                            let r = SIMD2(c.x - a.x, c.y - a.y)
                            let t = simd_dot(r, e) / l2
                            // Within a cell of the segment on the page, and
                            // in front of it: a corner behind hides nothing.
                            guard t > -mark.cell / length, t < 1 + mark.cell / length,
                                  abs(r.x * e.y - r.y * e.x) / length <= mark.cell,
                                  c.z > a.z + (b.z - a.z) * min(max(t, 0), 1) else { continue }
                            let fine = (-16...16).map { Double($0) * mark.cell / 16 / length }
                            for dt in fine + [1e-9, -1e-9, 1e-7, -1e-7, 1e-5, -1e-5] where t + dt > 0 && t + dt < 1 {
                                samples.append((a.v + (b.v - a.v) * (t + dt), length))
                            }
                        }
                    }
                }
            }
            let onTerrain = isContact(layer.spec)
            let results = Heightfield.parallelMap(samples, chunk: 64) {
                sample -> (decided: Bool, truth: Bool, drawn: Bool, offTerrain: Bool) in
                let p = sample.p
                // At a run end the truth is a boundary.
                if min(aloneIndex.nearestEnd(p), composedIndex.nearestEnd(p)) < max(1e-6 * sample.length, 1e-12 * size) {
                    return (false, false, false, false)
                }
                let w = world(P3<ViewSpace>(p)).v
                var cast = others
                if onTerrain {
                    guard let k = owner(w) else { return (false, false, false, true) }
                    cast[k] = topCasts[k]!
                }
                var truth = true
                for c in cast {
                    guard let clear = c(w, toEye) else { return (false, false, false, false) }
                    if !clear { truth = false }
                }
                return (true, truth, composedIndex.holds(p, tolerance: tolerance), false)
            }
            for (sample, r) in zip(samples, results) {
                if r.offTerrain {
                    verdict.offTerrain += 1
                    if verdict.example.isEmpty {
                        let w = world(P3<ViewSpace>(sample.p))
                        verdict.example = String(format: " e.g. %@ of %@ at (%.4f, %.4f, %.4f) is on no landscape's top",
                                                 layer.spec.name, part.name, w.x, w.y, w.z)
                    }
                }
                guard r.decided else { verdict.undecided += 1; continue }
                verdict.compared += 1
                if !r.truth { verdict.hiddenByOther += 1 }
                if isContact(layer.spec) {
                    verdict.contactCompared += 1
                    if !r.truth { verdict.contactHidden += 1 }
                }
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
            let own = ownTruths(part, layer: layer, view: partView, contact: contact)
            for (q, truth) in own {
                guard let truth, let shown = aloneHolds(q) else { continue }
                verdict.ownCompared += 1
                if truth, case .parametric = part.scene.geometry { verdict.torusSeen += 1 }
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
/// vertices, as the stress and surface suites test them. Its contact line
/// likewise: on a polyhedron sampled along each path, cast from its face,
/// on the torus at its vertices.
func ownTruths(_ part: Composite.Part, layer: Layer, view: Transform<WorldSpace, ViewSpace>,
               contact: ContactCurves) -> [(SIMD3<Double>, Bool?)] {
    let toEye = -view.projectionDirection
    if isContact(layer.spec) {
        switch part.scene.geometry {
        case .solid(let s):
            var out: [(SIMD3<Double>, Bool?)] = []
            for (path, f) in zip(contact.paths, contact.faces) {
                guard abs(simd_dot(s.normals[s.faces[f][0]], toEye)) > 1e-9 else { continue }
                let own = Set(s.faces[f])
                for (a, b) in zip(path, path.dropFirst()) {
                    for k in 0..<8 {
                        let q = a.v + (Double(k) + 0.5) / 8 * (b.v - a.v)
                        out.append((q, rayVisible(s, at: q, own: own, toEye: toEye)))
                    }
                }
            }
            return out
        case .parametric(let s):
            let (R, r) = torusRadii(s)
            let cast = torusOwnCast(R: R, r: r)
            let coords = Array(contact.coords.joined())
            return Heightfield.parallelMap(coords) { c -> (SIMD3<Double>, Bool?) in
                let q = s.map(c).position
                guard let f = s.facing(c, sight: -toEye), abs(f) > 1e-9 else { return (q, nil) }
                guard f > 0 else { return (q, false) }
                return (q, cast(q, toEye))
            }
        case .heightfield:
            return []
        }
    }
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
        let (R, r) = torusRadii(s)
        let cast = torusOwnCast(R: R, r: r)
        return Heightfield.parallelMap(Array(coords.indices)) { k -> (SIMD3<Double>, Bool?) in
            let q = layer.paths.vertices[k].v
            guard let f = s.facing(coords[k], sight: -toEye), abs(f) > 1e-9 else { return (q, nil) }
            guard f > 0 else { return (q, false) }
            return (q, cast(s.map(coords[k]).position, toEye))
        }
    }
}

/// The composite gate's compositions on the cubic landscape's grid and the
/// spike's, and its three cameras.
func gateCompositions() throws -> (cameras: [(String, Camera)], compositions: [(String, [(String, Scene, Placement)])]) {
    let contract = Fixtures.dir.appendingPathComponent("contract")
    func polyhedron(_ name: String) throws -> Scene {
        let b = try PolyhedronBundle.read(at: contract.appendingPathComponent("polyhedron_\(name).kurven"))
        return b.scene(preset: b.manifest.presets[0])
    }
    func surface(_ name: String) throws -> Scene {
        let b = try SurfaceBundle.read(at: contract.appendingPathComponent("surface_\(name).kurven"))
        return b.scene(preset: b.manifest.presets[0])
    }
    // The cubic landscape's grid: its solid is the drawn triangles, which
    // the cast is against. Three towers at the cube roots of unity.
    let bundle = try KurvenBundle.read(at: Fixtures.dir.appendingPathComponent("expr/landscape_cubic.kurven"))
    let landscape = Scene(bundle: bundle, preset: bundle.manifest.presets[0])
    let cube = try polyhedron("cube"), frame = try polyhedron("frame"), torus = try surface("torus")
    let cubeMinY = cube.solid!.vertices.map(\.y).min()!
    // A spike, exp(-20 |z|), on a grid of 64 cells: an apex no step
    // along a segment can be trusted to find.
    let spikeBundle = try NativeLandscape.build(
        LandscapeRequest(expression: "exp(-20*abs(z))",
                         domain: Domain(real: Interval(lo: -2, hi: 2), imag: Interval(lo: -2, hi: 2)),
                         resolution: 65, truncation: .uniform(5)), refine: false)
    let spike = Scene(bundle: spikeBundle, preset: spikeBundle.manifest.presets[0])
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
        // Under the slab's floor at its front, its front face a hair
        // inside the front wall: lines of sight from the face's lower
        // part leave the footprint before they reach the floor, and
        // from its upper part come up through it.
        ("a cube under the landscape's base",
         [("landscape", landscape, .identity), ("cube", cube, Placement(scale: 0.35, offset: SIMD3(0, -1.56, -0.86)))]),
        // Standing on the base, sunk into the slab, its front face in
        // the plane of the front wall, to rounding: on the slab's
        // boundary, seen from outside.
        ("a cube flush against the landscape's front wall",
         [("landscape", landscape, .identity), ("cube", cube, Placement(scale: 0.35, offset: SIMD3(0, -2 - 0.35 * cubeMinY, 0)))]),
        ("a cube seen through the notch between two towers",
         [("landscape", landscape, .identity), ("cube", cube, Placement(scale: 0.4, offset: SIMD3(-0.2, 1.8, 0.5)))]),
        ("a cube over the towers, where a tower's corner hid a hatch line's span",
         [("landscape", landscape, .identity), ("cube", cube, Placement(scale: 0.35, offset: SIMD3(-1, 1, 1)))]),
        ("a cube behind the spike's apex",
         [("landscape", spike, .identity), ("cube", cube, Placement(offset: SIMD3(-0.3712, 1.162, -0.249)))]),
        // Two landscapes side by side, the cube standing across the gap:
        // each contact point lies on one of them, and the other hides it as
        // it hides any point.
        ("two landscapes side by side, a cube across the gap",
         [("landscape", landscape, .identity), ("spike", spike, Placement(offset: SIMD3(4.1, 0, 0))),
          ("cube", cube, Placement(scale: 0.35, offset: SIMD3(2.05, 0.1, -0.05)))]),
    ]
    return (cameras, compositions)
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
        let (cameras, compositions) = try gateCompositions()
        // The compositions whose part meets the terrain: their contact line
        // is drawn, and partly hidden by another part. The others draw none.
        let touching: Set<String> = ["a cube standing on the landscape behind a tower",
                                     "the round torus sunk into the foot of a tower",
                                     "a frame sunk into the slope of a tower", "all three, moved",
                                     "a cube flush against the landscape's front wall",
                                     "a cube over the towers, where a tower's corner hid a hatch line's span",
                                     "a cube behind the spike's apex",
                                     "two landscapes side by side, a cube across the gap"]
        let clock = ContinuousClock()
        for (name, parts) in compositions {
            var total = CompositeVerdict()
            let elapsed = try clock.measure {
                for (_, camera) in cameras {
                    let composite = try Composite(parts, camera: camera)
                    let v = try compositeVerdict(composite, casts: composite.parts.map(compositeCast))
                    total.compared += v.compared; total.wrong += v.wrong; total.undecided += v.undecided
                    total.hiddenByOther += v.hiddenByOther; total.outsideAlone += v.outsideAlone
                    total.ownCompared += v.ownCompared; total.ownWrong += v.ownWrong; total.torusSeen += v.torusSeen
                    total.contactCompared += v.contactCompared; total.contactHidden += v.contactHidden
                    total.offTerrain += v.offTerrain
                    if total.example.isEmpty { total.example = v.example }
                    if total.ownExample.isEmpty { total.ownExample = v.ownExample }
                }
            }
            Check.expect(total.wrong == 0 && total.outsideAlone == 0 && total.offTerrain == 0 && total.compared > 20_000
                         && total.hiddenByOther > 0,
                         "\(name): drawn exactly where no other geometry's cast meets the line of sight",
                         "\(total.wrong) of \(total.compared) disagree, \(total.hiddenByOther) hidden by another, "
                         + "\(total.undecided) at a boundary, \(total.outsideAlone) vertices off the drawing alone, "
                         + "\(total.contactCompared) on a contact line, \(total.contactHidden) of them hidden by another, "
                         + "\(total.offTerrain) of them on no landscape's top, "
                         + "\(elapsed)\(total.example)")
            if touching.contains(name) {
                Check.expect(total.contactCompared > 1_000 && total.contactHidden > 0
                             && total.contactHidden < total.contactCompared,
                             "\(name): its contact line is drawn, and partly hidden by another part",
                             "\(total.contactCompared) compared, \(total.contactHidden) hidden")
            } else {
                Check.expect(total.contactCompared == 0, "\(name): no part meets the terrain, and no contact line is drawn")
            }
            let hasTorus = parts.contains { if case .parametric = $0.1.geometry { return true }; return false }
            Check.expect(total.ownWrong == 0 && total.ownCompared > 2_000 && (!hasTorus || total.torusSeen > 1_000),
                         "\(name): each geometry alone is its own cast",
                         "\(total.ownWrong) of \(total.ownCompared) disagree, \(total.torusSeen) seen on the torus\(total.ownExample)")
        }
    }

    Check.suite("composite: a landscape's test of another solid's point has a floor and a boundary") {
        // Under the landscape's own camera, whose lines of sight run toward
        // the front wall (−y) and up: a point a hair inside the wall, on it
        // and a hair outside, each under the wall's top, is seen; one under
        // the surface inside is not; one under the base whose line leaves
        // the footprint before it reaches the base is seen, and one whose
        // line comes up through the floor is not.
        let bundle = try KurvenBundle.read(at: Fixtures.dir.appendingPathComponent("expr/landscape_cubic.kurven"))
        for (name, b) in [("on the grid", bundle)] {
            let scene = Scene(bundle: b, preset: b.manifest.presets[0])
            let test = HeightfieldVisibility(heightfield: scene.heightfield!, view: scene.camera.view,
                                             depthTolerance: scene.depthTolerance)
            for (what, p, clear) in [("a hair inside the front wall", P3<WorldSpace>(0, -2 + 1e-12, 0.05), true),
                                     ("on the front wall", P3<WorldSpace>(0, -2, 0.05), true),
                                     ("a hair outside the front wall", P3<WorldSpace>(0, -2 - 1e-12, 0.05), true),
                                     ("under the surface inside", P3<WorldSpace>(0, -1.9, 0.05), false),
                                     ("under the base, its line leaving through the wall", P3<WorldSpace>(0, -1.95, -0.05), true),
                                     ("under the base, its line coming up through the floor", P3<WorldSpace>(0, -1.5, -0.05), false)] {
                Check.expect(test.isClear(ofAnotherSolid: p) == clear, "\(name): a point \(what) is \(clear ? "seen" : "hidden")")
            }
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
        // A landscape on its function: nothing bounds f's slope, so it
        // cannot cut another's curves exactly, and it is refused.
        let refined = bundle.refined(by: NativeLandscape.refiner(for: bundle)?.refine)
        let onF = Scene(bundle: refined, preset: refined.manifest.presets[0])
        do {
            _ = try Composite([("landscape", onF, .identity), ("cube", try polyhedron("cube"), .identity)], camera: onF.camera)
            Check.expect(false, "a landscape on its function is refused in a composite", "it was taken")
        } catch let error as CompositeError {
            Check.expect(error.description.contains("on its function"),
                         "a landscape on its function is refused in a composite, saying why", error.description)
        }
        // A displaced torus's sphere tracing steps by a Lipschitz constant
        // read off its grid, not proved: it is refused.
        let relief = try surface("relief")
        do {
            _ = try Composite([("landscape", landscape, .identity), ("relief", relief, .identity)], camera: landscape.camera)
            Check.expect(false, "a displaced torus is refused in a composite", "it was taken")
        } catch let error as CompositeError {
            Check.expect(error.description.contains("displaced"), "a displaced torus is refused in a composite, saying why",
                         error.description)
        }
        // At either end of the range of scales, drawn in reasonable time:
        // each part works only over the ball or box that holds it.
        let clock = ContinuousClock()
        for (what, part, placement) in [("the round torus at scale 1e-3", try surface("torus"), Placement(scale: 1e-3, offset: SIMD3(0, 0, 1))),
                                        ("the round torus at scale 1e3", try surface("torus"), Placement(scale: 1e3, offset: SIMD3(0, 0, 1))),
                                        ("a cube at scale 1e3", try polyhedron("cube"), Placement(scale: 1e3)),
                                        ("a cube at scale 1e-3", try polyhedron("cube"), Placement(scale: 1e-3, offset: SIMD3(1, 1, 1)))] {
            var drawn = false
            let elapsed = try clock.measure {
                drawn = try Composite([("landscape", landscape, .identity), ("part", part, placement)],
                                      camera: landscape.camera).visibleLayers().count > 0
            }
            Check.expect(drawn && elapsed < .seconds(20), "\(what) on the landscape is drawn in reasonable time", "\(elapsed)")
        }
        // Placements at either end of the range, and past it.
        let cube = try polyhedron("cube")
        for (what, placement, taken) in [("scale 1e-3", Placement(scale: 1e-3), true), ("scale 1e3", Placement(scale: 1e3), true),
                                         ("offset 1e4", Placement(offset: SIMD3(1e4, -1e4, 1e4)), true),
                                         ("scale 1e200", Placement(scale: 1e200), false),
                                         ("scale 1e-20", Placement(scale: 1e-20), false),
                                         ("scale 0.999e-3", Placement(scale: 0.999e-3), false),
                                         ("offset 1e300", Placement(offset: SIMD3(0, 1e300, 0)), false),
                                         ("offset 1.001e4", Placement(offset: SIMD3(0, 0, -1.001e4)), false)] {
            let made = try? Composite([("landscape", landscape, .identity), ("cube", cube, placement)], camera: landscape.camera)
            Check.expect((made != nil) == taken, "a cube placed at \(what) is \(taken ? "taken" : "refused")")
        }
    }
}
