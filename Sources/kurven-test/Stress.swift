import Foundation
import simd
import KurvenCore
import KurvenMetal
import KurvenBake
import KurvenLandscape

// MARK: - visibility under stress
//
// The principle these suites hold the hidden-line rules to: visibility is a
// function of the geometry alone, never of the output resolution. A rule
// that consults a pixel fails this by construction -- a pixel's depth span
// on a steep graph or an edge-on wall exceeds any depth tolerance, and the span
// changes with the resolution -- and the rules here are built so that no
// pixel is consulted: the facing of the surface the curves lie on, and the
// traversal up the line of sight through the drawn solid (`GridTraversal`), which is
// exact because the drawn solid is piecewise linear. So the suites ask the
// strongest questions such a rule can be asked: the same bake at every
// depth resolution, the same drawing scaled when the drawing is scaled, and
// on a grid too coarse for any approximation to hide in, agreement with
// a brute-force ray cast against the triangles the depth pass draws.

/// The ray from `origin` along `direction` against every triangle of
/// `mesh`: the parameter of the first hit past `after`, or nil. Brute
/// force, as an oracle should be: nothing about the heightfield's structure
/// is used, only its triangles.
func firstHit(_ mesh: Mesh<WorldSpace>, from origin: SIMD3<Double>,
              along direction: SIMD3<Double>, after: Double) -> Double? {
    var best: Double?
    for t in mesh.triangles {
        let a = mesh.vertices[Int(t.x)].v, b = mesh.vertices[Int(t.y)].v, c = mesh.vertices[Int(t.z)].v
        // Möller–Trumbore.
        let e1 = b - a, e2 = c - a
        let p = simd_cross(direction, e2)
        let det = simd_dot(e1, p)
        guard abs(det) > 1e-14 else { continue }
        let inv = 1 / det
        let s = origin - a
        let u = simd_dot(s, p) * inv
        guard u >= -1e-12, u <= 1 + 1e-12 else { continue }
        let q = simd_cross(s, e1)
        let v = simd_dot(direction, q) * inv
        guard v >= -1e-12, u + v <= 1 + 1e-12 else { continue }
        let hit = simd_dot(e2, q) * inv
        guard hit > after else { continue }
        if best == nil || hit < best! { best = hit }
    }
    return best
}

/// The drawn solid of a one-tile, uncut heightfield as explicit triangles:
/// the grid's cells as the rasterizer splits them, clamped at the
/// vertices, except that a triangle the truncation cuts is replaced by its truncation edge
/// pieces (`Mesh.truncationEdgeMesh`), so the surface is exactly min(interpolated, truncation);
/// plus wall curtains down to `base` and a floor. This is the boundary of
/// what the depth pass draws, and a line of sight that is inside the solid
/// anywhere crosses one of these triangles.
func drawnTriangles(_ h: Heightfield, base: Double) -> Mesh<WorldSpace> {
    let s = h.surface, g = s.height
    let step = max(h.step, 1)
    let nx = (g.width + step - 1) / step, ny = (g.height + step - 1) / step
    func at(_ i: Int, _ j: Int) -> P3<WorldSpace> {
        let p = g.position(x: i * step, y: j * step)
        return P3(p.x, p.y, min(Double(g[i * step, j * step]), s.truncation.height(atX: p.x)))
    }
    func excess(_ i: Int, _ j: Int) -> Double {
        let p = g.position(x: i * step, y: j * step)
        return Double(g[i * step, j * step]) - s.truncation.height(atX: p.x)
    }
    var vertices: [P3<WorldSpace>] = []
    var triangles: [SIMD3<Int32>] = []
    func tri(_ a: P3<WorldSpace>, _ b: P3<WorldSpace>, _ c: P3<WorldSpace>) {
        let k = Int32(vertices.count)
        vertices += [a, b, c]
        triangles.append(SIMD3(k, k + 1, k + 2))
    }
    /// A triangle the truncation cuts: a corner over it and a corner under it.
    func cut(_ corners: [(Int, Int)]) -> Bool {
        let e = corners.map { excess($0.0, $0.1) }
        return e.contains { $0 > 0 } && e.contains { $0 < 0 }
    }
    for j in 0..<(ny - 1) {
        for i in 0..<(nx - 1) {
            if !cut([(i, j), (i + 1, j), (i, j + 1)]) { tri(at(i, j), at(i + 1, j), at(i, j + 1)) }
            if !cut([(i + 1, j), (i + 1, j + 1), (i, j + 1)]) {
                tri(at(i + 1, j), at(i + 1, j + 1), at(i, j + 1))
            }
        }
    }
    // Walls round the four sides, up to the drawn top edge -- which, where a
    // boundary cell straddles the truncation, bends at the truncation edge -- and the floor.
    func raw(_ i: Int, _ j: Int) -> P3<WorldSpace> {
        let p = g.position(x: i * step, y: j * step)
        return P3(p.x, p.y, Double(g[i * step, j * step]))
    }
    func curtain(_ i0: Int, _ j0: Int, _ i1: Int, _ j1: Int) {
        let a = raw(i0, j0), b = raw(i1, j1)
        let ca = s.truncation.height(atX: a.x), cb = s.truncation.height(atX: b.x)
        var topEdge = [P3<WorldSpace>(a.x, a.y, min(a.z, ca))]
        if (a.z - ca > 0) != (b.z - cb > 0), ca.isFinite, cb.isFinite {
            let t = (a.z - ca) / ((a.z - ca) - (b.z - cb))
            topEdge.append(P3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, ca + (cb - ca) * t))
        }
        topEdge.append(P3(b.x, b.y, min(b.z, cb)))
        for (p, q) in zip(topEdge, topEdge.dropFirst()) {
            let p0 = P3<WorldSpace>(p.x, p.y, base), q0 = P3<WorldSpace>(q.x, q.y, base)
            tri(p, q, p0); tri(q, q0, p0)
        }
    }
    for i in 0..<(nx - 1) { curtain(i, 0, i + 1, 0); curtain(i, ny - 1, i + 1, ny - 1) }
    for j in 0..<(ny - 1) { curtain(0, j, 0, j + 1); curtain(nx - 1, j, nx - 1, j + 1) }
    let c00 = at(0, 0), c10 = at(nx - 1, 0), c01 = at(0, ny - 1), c11 = at(nx - 1, ny - 1)
    func floor(_ p: P3<WorldSpace>) -> P3<WorldSpace> { P3(p.x, p.y, base) }
    tri(floor(c00), floor(c10), floor(c01)); tri(floor(c10), floor(c11), floor(c01))
    let truncationEdge = Mesh.truncationEdgeMesh(of: s, step: step, region: h.region, tiles: h.tiles)
    return Mesh.concat([Mesh(vertices: vertices, triangles: triangles), truncationEdge])
}

/// Whether the line from `origin` along `direction`, up to distance
/// `upTo`, passes under some facet of `mesh` by more than `depthTolerance`,
/// perpendicular to that facet: for every triangle with a footprint, the
/// stretch of the line over the footprint, and the facet's gap at its ends,
/// which is where a linear gap is least. The start of the line counts only
/// when `startInside`. Brute force over the triangles, sharing nothing
/// with the traversal.
func underSomeTriangle(_ mesh: Mesh<WorldSpace>, from origin: SIMD3<Double>,
                    along direction: SIMD3<Double>, by depthTolerance: Double, startInside: Bool,
                    upTo: Double = .infinity) -> Bool {
    let o = SIMD2(origin.x, origin.y), d = SIMD2(direction.x, direction.y)
    for t in mesh.triangles {
        let a = mesh.vertices[Int(t.x)].v, b = mesh.vertices[Int(t.y)].v, c = mesh.vertices[Int(t.z)].v
        let n = simd_cross(b - a, c - a)
        let len = simd_length(n)
        // A wall, seen from above, has no footprint and is no one's facet.
        guard len > 0, abs(n.z) > 1e-12 * len else { continue }
        let cosine = abs(n.z) / len
        // The line clipped to the footprint: inside each edge's half-plane,
        // with the footprint's linear curve.
        let corners = [SIMD2(a.x, a.y), SIMD2(b.x, b.y), SIMD2(c.x, c.y)]
        let sign: Double = n.z > 0 ? 1 : -1
        var lo = 0.0, hi = Double.infinity
        var empty = false
        for k in 0..<3 {
            let p = corners[k], q = corners[(k + 1) % 3]
            let e = q - p
            let f0 = sign * (e.x * (o.y - p.y) - e.y * (o.x - p.x))
            let f1 = sign * (e.x * d.y - e.y * d.x)   // f(t) = f0 + t f1 >= 0
            if abs(f1) < 1e-15 {
                if f0 < -1e-12 { empty = true; break }
                continue
            }
            let root = -f0 / f1
            if f1 > 0 { lo = max(lo, root) } else { hi = min(hi, root) }
        }
        hi = min(hi, upTo)
        guard !empty, lo <= hi, hi.isFinite || lo.isFinite else { continue }
        func gap(_ s: Double) -> Double {
            let p = origin + s * direction
            let z = a.z - (n.x * (p.x - a.x) + n.y * (p.y - a.y)) / n.z
            return (p.z - z) * cosine
        }
        for s in [lo, hi] where s.isFinite && (s > 1e-12 || startInside) {
            if gap(s) + depthTolerance < 0 { return true }
        }
    }
    return false
}

/// Whether a point is under the drawn surface, or on it to within
/// `tolerance`: under some facet whose footprint holds it.
func underSurface(_ mesh: Mesh<WorldSpace>, _ p: SIMD3<Double>, tolerance: Double) -> Bool {
    underSomeTriangle(mesh, from: p, along: SIMD3(0, 0, 1), by: -tolerance, startInside: true, upTo: 0)
}

/// The oracle for the traversal, brute force over the drawn triangles and
/// sharing nothing with it. A point that starts inside the solid (strictly
/// over the domain) is in its own surface's neighbourhood until the line
/// first crosses out: there it is hidden only if it lies under some facet
/// by more than the depth tolerance, perpendicular to that facet. After that, any
/// crossing of the boundary beyond the depth tolerance's reach along the line is
/// occlusion. `perUnit` is view depth per unit along the line.
func exactlyHidden(_ mesh: Mesh<WorldSpace>, from p: SIMD3<Double>, along toward: SIMD3<Double>,
                   depthTolerance: Double, perUnit: Double, strictlyInside: Bool, cell: Double) -> Bool {
    let reach = depthTolerance / perUnit
    var exit = 0.0
    if underSurface(mesh, p, tolerance: 1e-9 * cell) {
        exit = firstHit(mesh, from: p, along: toward, after: 1e-10) ?? .infinity
    }
    if underSomeTriangle(mesh, from: p, along: toward, by: depthTolerance, startInside: strictlyInside, upTo: exit) {
        return true
    }
    return firstHit(mesh, from: p, along: toward, after: max(exit, reach) + 1e-10) != nil
}

/// A small heightfield with random heights, for the grid tests.
/// Whether f hides `p` from the eye, by walking the line of sight in steps of
/// a sixty-fourth of a cell and bisecting each crossing of the surface it
/// finds, under the traversal's own meaning of the depth tolerance: in its own surface's
/// neighbourhood from the start, the line is forgiven the depth tolerance
/// perpendicular to the surface; once it has been outside, being inside at
/// or beyond the depth tolerance's reach is occlusion. The oracle for `GridTraversal`
/// with the function at hand.
func denselyHidden(_ h: Heightfield, magnitude: ContourRefine.Magnitude, from p: SIMD3<Double>,
                   along toward: SIMD3<Double>, depthTolerance: Double, perUnit: Double, cell: Double,
                   trace: ((String) -> Void)? = nil) -> Double? {
    let reach = depthTolerance / max(perUnit, 1e-12)
    var top = -Double.infinity
    h.forEachSample { top = max(top, $0.z) }
    let d = toward
    var tMax = Double.infinity
    if d.z > 0 { tMax = max((top - p.z) / d.z, 0) } else if d.z == 0, p.z > top { return nil }
    precondition(h.tiles.count == 1, "the dense cast reads f on the fundamental tile only")
    let e: Double = 1e-4 * cell
    /// The line's gaps over f and over the truncation at `t`, and f's cosine; nil
    /// outside the region.
    let dom = h.surface.domain
    let box = (lo: SIMD2(min(dom.real.lo, dom.real.hi), min(dom.imag.lo, dom.imag.hi)),
               hi: SIMD2(max(dom.real.lo, dom.real.hi), max(dom.imag.lo, dom.imag.hi)))
    func gaps(_ t: Double) -> (overF: Double, overTruncation: Double, cosine: Double)? {
        let q = P2<WorldSpace>(p.x + t * d.x, p.y + t * d.y)
        // Nothing is drawn outside the domain's box: the traversal stops at the
        // grid's edge, and f beyond it is not the solid.
        guard q.x >= box.lo.x, q.x <= box.hi.x, q.y >= box.lo.y, q.y <= box.hi.y,
              h.region.contains(q) else { return nil }
        let dq = P2<DomainSpace>(q.x, q.y)
        let u: Double = magnitude(dq)
        let truncationHeight: Double = h.surface.truncation.height(atX: dq.x)
        let hx: Double = (magnitude(P2<DomainSpace>(dq.x + e, dq.y)) - magnitude(P2<DomainSpace>(dq.x - e, dq.y))) / (2 * e)
        let hy: Double = (magnitude(P2<DomainSpace>(dq.x, dq.y + e)) - magnitude(P2<DomainSpace>(dq.x, dq.y - e))) / (2 * e)
        let cosine = 1 / (1 + hx * hx + hy * hy).squareRoot()
        let z = p.z + t * d.z
        return (z - (u.isNaN ? .infinity : u), z - truncationHeight, cosine.isNaN ? 0 : cosine)
    }
    func inside(_ g: (overF: Double, overTruncation: Double, cosine: Double)?, zEps: Double) -> Bool {
        guard let g else { return false }
        return g.overF < zEps && g.overTruncation < zEps
    }
    var outsideYet = false
    let tEps = 1e-9 * cell
    // As the traversal: a point on the solid's boundary -- the domain's edge --
    // is tested only from where the line goes next.
    let strictlyInside = p.x > box.lo.x + 1e-9 && p.x < box.hi.x - 1e-9
        && p.y > box.lo.y + 1e-9 && p.y < box.hi.y - 1e-9
    func hidden(_ t: Double, _ g: (overF: Double, overTruncation: Double, cosine: Double)?) -> Bool {
        let zEps = t > tEps ? 0 : 1e-9 * cell
        guard inside(g, zEps: zEps), let g else {
            // As the traversal: outside by less than the depth tolerance is within the
            // surface's skin, not outside it.
            if let g {
                let clearance = g.overF > g.overTruncation ? g.overF * g.cosine : g.overTruncation
                if clearance <= depthTolerance { return false }
            }
            trace?(String(format: "cast t %.4g cells: outside (%@)", t / cell, g.map { String(format: "%+.3e over f", $0.overF) } ?? "off"))
            outsideYet = true; return false
        }
        guard t > tEps || strictlyInside else { return false }
        if outsideYet {
            trace?(String(format: "cast t %.4g cells: inside by %.3e after outside: %@", t / cell, -g.overF, t + tEps >= reach ? "hidden" : "forgiven"))
            return t + tEps >= reach
        }
        let depth = g.overF > g.overTruncation ? g.overF * g.cosine : g.overTruncation
        trace?(String(format: "cast t %.4g cells: inside by %.3e, cosine %.4f, depth %.3e, depth tolerance %.3e: %@", t / cell, -g.overF, g.cosine, depth, depthTolerance, depth + depthTolerance < 0 ? "hidden" : "forgiven"))
        return depth + depthTolerance < 0
    }
    // The line leaves the region eventually; walk until it has.
    let step = cell / 64
    var t = 0.0
    var previous = gaps(0)
    if hidden(0, previous) { return 0 }
    var left = 0
    while t < tMax, left < 100_000 {
        let tn = t + step
        let g = gaps(tn)
        // A crossing of the surface between the two: bisect it and test
        // just inside.
        if inside(previous, zEps: 0) != inside(g, zEps: 0) {
            var lo = t, hi = tn
            let insideLo = inside(previous, zEps: 0)
            for _ in 0..<50 {
                let m = 0.5 * (lo + hi)
                if inside(gaps(m), zEps: 0) == insideLo { lo = m } else { hi = m }
            }
            let tc = insideLo ? lo : hi
            if hidden(tc, gaps(tc)) { return tc }
        }
        // Leaving the region between the two: the traversal tests the exit
        // itself, so the cast does too -- bisected, and tested a hair
        // inside, where a line passing under a top edge is inside the solid --
        // before the sample beyond it, which is off the region.
        if previous != nil, g == nil {
            var lo = t, hi = tn
            for _ in 0..<50 {
                let m = 0.5 * (lo + hi)
                if gaps(m) != nil { lo = m } else { hi = m }
            }
            if hidden(lo, gaps(lo)) { return lo }
        }
        if hidden(tn, g) { return tn }
        if g == nil, previous == nil { left += 1; if left > 64 { break } } else { left = 0 }
        previous = g; t = tn
    }
    return nil
}

func randomHeightfield(cells: Int, seed: UInt64, truncation: Truncation, amplitude: Double = 3) -> Heightfield {
    let n = cells + 1
    let rng = SplitMix(seed: seed)
    let domain = Domain(real: Interval(lo: -1, hi: 1), imag: Interval(lo: -1.5, hi: 0.5))
    var values = [Float](repeating: 0, count: n * n)
    for k in values.indices { values[k] = Float(rng.next(0, amplitude)) }
    let surface = Surface(height: Grid2D(width: n, height: n, domain: domain, values: values),
                          phase: nil, truncation: truncation)
    return Heightfield(surface: surface, occluder: .empty, tiles: [.identity], step: 1)
}

/// The cameras the stress tests orbit through: from nearly overhead to a
/// degree off edge-on, round the compass, some sheared as the plates are.
func stressCameras() -> [(name: String, view: Transform<WorldSpace, ViewSpace>)] {
    var out: [(String, Transform<WorldSpace, ViewSpace>)] = []
    for elevation in [-80.0, -55, -30, -10, -3, -1] {
        for azimuth in [0.0, 37, 90, 135, 200, 289] {
            out.append(("el \(Int(-elevation))° az \(Int(azimuth))°",
                        testCamera(elevation: elevation, azimuth: azimuth)))
        }
    }
    out.append(("el 55° sheared", testCamera(elevation: -55, azimuth: -20, shear: 0.5)))
    out.append(("el 5° sheared", testCamera(elevation: -5, azimuth: 70, shear: 0.5)))
    return out
}

func stressTests() {
    Check.suite("stress: a heightfield bake is the same drawing at every depth resolution") {
        // The drawing with every kind of feature: plateaus below the truncation, cut
        // faces, pits to the floor, a long slope.
        let preset = Catalog.native.preset("rgamma")!
        let bundle = try NativeLandscape.build(LandscapeRequest(preset: preset, resolution: 240))
        let scene = Scene(bundle: bundle, preset: bundle.manifest.presets[0])
        let renderer = try MetalRenderer()
        let resolutions = [64, 256, 1024, 4096]
        var bakes: [Bake] = []
        let clock = ContinuousClock()
        let elapsed = try clock.measure {
            for r in resolutions { bakes.append(try renderer.bake(scene, options: BakeOptions(resolution: r))) }
        }
        let reference = bakes[0]
        Check.expect(reference.strokes.pathCount > 200 && reference.strokes.totalLength > 100,
                     "the drawing draws", "\(reference.strokes.pathCount) paths, \(String(format: "%.1f", reference.strokes.totalLength)) units of curves")
        for (r, b) in zip(resolutions, bakes).dropFirst() {
            var same = b.strokes.layers.count == reference.strokes.layers.count
            var differing: [String] = []
            for (k, (x, y)) in zip(reference.strokes.layers, b.strokes.layers).enumerated()
            where x.paths != y.paths {
                same = false; differing.append(scene.layers[k].spec.name)
            }
            Check.expect(same, "at \(r) px every layer is what it was at 64 px, vertex for vertex",
                         differing.isEmpty ? "" : "differs: " + differing.joined(separator: ", "))
        }
        Check.expect(elapsed < .seconds(60), "four bakes ran in reasonable time", "\(elapsed)")

        // The steep case: erf under a truncation of 14 over a four-unit window,
        // where the depth test lost a third of the contour curves to pixel
        // span. The drawing is still the same at every resolution, and the
        // curves the test removes are removed by the solid, not by a pixel.
        var steep = LandscapeRequest(preset: Catalog.native.preset("erf")!, resolution: 240)
        steep.domain = Domain(real: Interval(lo: -2, hi: 2), imag: Interval(lo: -2, hi: 2))
        steep.truncation = .uniform(14)
        let steepBundle = try NativeLandscape.build(steep)
        let steepScene = Scene(bundle: steepBundle, preset: steepBundle.manifest.presets[0])
        let coarse = try renderer.bake(steepScene, options: BakeOptions(resolution: 64))
        let fine = try renderer.bake(steepScene, options: BakeOptions(resolution: 4096))
        var same = true
        for (x, y) in zip(coarse.strokes.layers, fine.strokes.layers) where x.paths != y.paths { same = false }
        Check.expect(same, "the steep drawing too: 64 px and 4096 px bake the same strokes")
        // What the solid hides of the facing contour curves, and why: curves
        // behind a nearer graph, or curves under its own drawn surface by more
        // than the depth tolerance -- the grid's interpolation gap, which a finer
        // grid closes and the depth tolerance is for. Reported, since neither
        // number is a target; the floor only says the traversal is not
        // erasing the drawing.
        let h = steepScene.heightfield!
        let vis = HeightfieldVisibility(heightfield: h, view: steepScene.camera.view, depthTolerance: steepScene.depthTolerance)
        var kept = 0.0, facingOnly = 0.0, buried = 0, behind = 0
        for layer in steepScene.layers where layer.spec.liesOnSurface {
            kept += h.visibleSurfaceCurves(layer.paths, view: steepScene.camera.view, depthTolerance: steepScene.depthTolerance).totalLength
            facingOnly += h.visibleSurfaceCurves(layer.paths, view: steepScene.camera.view, depthTolerance: .infinity).totalLength
            for v in layer.paths.vertices where vis.facing(v.xy) > 0 && !vis.isClear(v) {
                if v.z < h.drawnHeight(at: v.xy) { buried += 1 } else { behind += 1 }
            }
        }
        Check.expect(kept > 0.5 * facingOnly,
                     "of the facing contour curves, the solid hides what lies behind a nearer graph",
                     String(format: "%.1f of %.1f units kept; %d hidden vertices above their own facet, %d under it",
                            kept, facingOnly, behind, buried))
    }

    Check.suite("stress: on a real drawing the traversal is the brute-force ray cast") {
        // rgamma's every facing contour vertex, the traversal against the
        // perpendicular-gap cast over the drawing's drawn triangles at the
        // drawing's depth tolerance, and against the depth buffer the bake used to
        // test by. The pit at -4, a cone of slope 24 cut by the front
        // wall, is reported on its own: its far wall through the notch lies
        // partly behind the pit's own side under the drawing's diagonal sight
        // line, which a depth buffer mostly misses -- the wall rasterizes
        // to slivers -- and which no vertical tolerance could test.
        let preset = Catalog.native.preset("rgamma")!
        let bundle = try NativeLandscape.build(LandscapeRequest(preset: preset, resolution: 240))
        let scene = Scene(bundle: bundle, preset: bundle.manifest.presets[0])
        // The grid test: the drawing without its function, whose solid is
        // the drawn triangles the cast is against. With the function the
        // test's solid is f (below).
        let withF = scene.heightfield!
        let h = Heightfield(surface: withF.surface, occluder: withF.occluder, tiles: withF.tiles,
                            region: withF.region, step: withF.step)
        let view = scene.camera.view
        let mesh = drawnTriangles(h, base: 0)
        let vis = HeightfieldVisibility(heightfield: h, view: view, depthTolerance: scene.depthTolerance)
        let toward = -view.projectionDirection
        let c = view.m.columns
        let perUnit = abs(simd_dot(SIMD3(c.0.z, c.1.z, c.2.z), toward))
        let d = h.surface.domain
        let box = (lo: SIMD2(min(d.real.lo, d.real.hi), min(d.imag.lo, d.imag.hi)),
                   hi: SIMD2(max(d.real.lo, d.real.hi), max(d.imag.lo, d.imag.hi)))
        let depth = try MetalRenderer().bake(scene, options: BakeOptions(resolution: 1000)).depth
        let g = h.surface.height
        let cell = max(abs(g.domain.real.length) / Double(g.width - 1), abs(g.domain.imag.length) / Double(g.height - 1))
        var total = 0, traversalHidden = 0, castHidden = 0, disagree = 0, depthHidden = 0
        var pit = (total: 0, traversal: 0, cast: 0, depth: 0)
        let clock = ContinuousClock()
        let elapsed = clock.measure {
            for layer in scene.layers where layer.spec.liesOnSurface {
                let vertices = layer.paths.vertices
                let facing = vertices.map { vis.facing($0.xy) > 0 }
                let traversal = vertices.map { vis.isClear($0) }
                let cast = Heightfield.parallelMap(vertices) { v in
                    let strictlyInside = v.x > box.lo.x + 1e-9 && v.x < box.hi.x - 1e-9
                        && v.y > box.lo.y + 1e-9 && v.y < box.hi.y - 1e-9
                    return !exactlyHidden(mesh, from: v.v, along: toward, depthTolerance: scene.depthTolerance,
                                          perUnit: perUnit, strictlyInside: strictlyInside, cell: cell)
                }
                for (k, v) in vertices.enumerated() where facing[k] {
                    total += 1
                    let q = view(v)
                    let byDepth = q.z + scene.depthTolerance > depth.depth(under: q.xy)
                    if !traversal[k] { traversalHidden += 1 }
                    if !cast[k] { castHidden += 1 }
                    if !byDepth { depthHidden += 1 }
                    if traversal[k] != cast[k] { disagree += 1 }
                    if v.x > -4.3, v.x < -3.7, v.y < 0.6 {
                        pit.total += 1
                        if !traversal[k] { pit.traversal += 1 }
                        if !cast[k] { pit.cast += 1 }
                        if !byDepth { pit.depth += 1 }
                    }
                }
            }
        }
        Check.expect(disagree == 0 && total > 5000,
                     "the traversal and the cast agree on every facing contour vertex",
                     "\(total) vertices; traversal hides \(traversalHidden), cast hides \(castHidden), the 1000 px depth buffer hid \(depthHidden)")
        Check.expect(pit.traversal == pit.cast && pit.traversal > pit.depth,
                     "in the pit at -4 both hide what the depth buffer missed",
                     "\(pit.total) facing vertices: traversal \(pit.traversal), cast \(pit.cast), depth buffer \(pit.depth)")
        Check.expect(elapsed < .seconds(60), "in reasonable time", "\(elapsed)")

        // With the function, the test's solid is f: the traversal samples the
        // line of sight at the grid's crossings and reads f there, and is
        // held to a cast that walks the line in steps of a sixty-fourth of
        // a cell and bisects every crossing of f. What the traversal can miss
        // is a dip of f within one cell along the line; the count is the
        // measure of that, on the real drawing.
        let visF = HeightfieldVisibility(heightfield: withF, view: view, depthTolerance: scene.depthTolerance)
        let magnitude = withF.refine!.magnitude!
        var totalF = 0, traversalF = 0, castF = 0, disagreeF = 0
        var examples = ""
        let elapsedF = clock.measure {
            for layer in scene.layers where layer.spec.liesOnSurface {
                let vertices = layer.paths.vertices
                let facing = vertices.map { visF.facing($0.xy) > 0 }
                let traversal = vertices.map { visF.isClear($0) }
                let cast = Heightfield.parallelMap(vertices) { v in
                    denselyHidden(withF, magnitude: magnitude, from: v.v, along: toward,
                                  depthTolerance: visF.depthTolerance, perUnit: perUnit, cell: cell)
                }
                for k in vertices.indices where facing[k] {
                    totalF += 1
                    if !traversal[k] { traversalF += 1 }
                    if cast[k] != nil { castF += 1 }
                    if traversal[k] != (cast[k] == nil) {
                        disagreeF += 1
                        if disagreeF == 1 {
                            let v = vertices[k]
                            var profile = "    gap over f along the line, every eighth of a cell:"
                            for i in 0...48 {
                                let t = Double(i) * cell / 8
                                let q = P2<DomainSpace>(v.x + t * toward.x, v.y + t * toward.y)
                                let u = magnitude(q)
                                profile += String(format: " %+.3f", v.z + t * toward.z - min(u, withF.surface.truncation.height(atX: q.x)))
                            }
                            print(profile)
                            print("    the traversal: " + visF.explain(v).replacingOccurrences(of: "\n", with: "\n    "))
                            _ = denselyHidden(withF, magnitude: magnitude, from: v.v, along: toward, depthTolerance: visF.depthTolerance,
                                              perUnit: perUnit, cell: cell) { print("    " + $0) }
                        }
                        if disagreeF <= 6 {
                            let v = vertices[k]
                            examples += String(format: " [(%.3f, %.3f, %.3f) truncation %.2f |f| %.4f: traversal %@, cast %@]",
                                               v.x, v.y, v.z, withF.surface.truncation.height(atX: v.x),
                                               magnitude(P2<DomainSpace>(v.x, v.y)),
                                               traversal[k] ? "clear" : "hidden",
                                               cast[k].map { String(format: "hides at t = %.4f cells", $0 / cell) } ?? "clear")
                        }
                    }
                }
            }
        }
        Check.expect(disagreeF == 0 && totalF > 5000,
                     "with the function, the traversal against f is the dense cast against f",
                     "\(totalF) vertices; traversal hides \(traversalF), cast hides \(castF), \(disagreeF) disagree" + examples)
        Check.expect(elapsedF < .seconds(120), "in reasonable time", "\(elapsedF)")
    }

    Check.suite("stress: scaling the drawing scales the drawing") {
        // The whole drawing -- heights, domain, truncations, walls, curves and depth tolerance --
        // scaled by a power of two, so every float scales exactly; the tested
        // curves must be the tested curves scaled, vertex for vertex. A rule with a
        // pixel or an absolute tolerance in it would not pass this; one that
        // is geometry alone cannot fail it.
        let preset = Catalog.native.preset("rgamma")!
        let bundle = try NativeLandscape.build(LandscapeRequest(preset: preset, resolution: 240))
        let scene = Scene(bundle: bundle, preset: bundle.manifest.presets[0])
        let h = scene.heightfield!
        let visible = scene.visibleLayers()
        for s in [8.0, 0.125] {
            let g = h.surface.height
            let domain = Domain(real: Interval(lo: g.domain.real.lo * s, hi: g.domain.real.hi * s),
                                imag: Interval(lo: g.domain.imag.lo * s, hi: g.domain.imag.hi * s))
            let truncation: Truncation
            switch h.surface.truncation {
            case .none: truncation = .none
            case .uniform(let z): truncation = .uniform(z * s)
            case .realBands(let bands, let beyond):
                truncation = .realBands(bands.map { RealBand(below: $0.below * s, height: $0.height * s) }, beyond: beyond * s)
            }
            let surface = Surface(height: Grid2D(width: g.width, height: g.height, domain: domain,
                                                 values: g.values.map { $0 * Float(s) }),
                                  phase: h.surface.phase, truncation: truncation)
            let occluder = Mesh(vertices: h.occluder.vertices.map { P3<WorldSpace>($0.v * s) },
                                triangles: h.occluder.triangles)
            let layers = scene.layers.map { layer in
                Layer(spec: layer.spec,
                      paths: PolylineSet(vertices: layer.paths.vertices.map { P3<WorldSpace>($0.v * s) },
                                         offsets: layer.paths.offsets))
            }
            // The scaled drawing is the scaled function: the refiner's |f| and
            // its zeros, read at the point scaled back, scaled up again. The
            // contours are not re-derived under the test, so they pass.
            let refine: ContourRefine? = h.refine.map { r -> ContourRefine in
                let magnitude: ContourRefine.Magnitude? = r.magnitude.map { m -> ContourRefine.Magnitude in
                    { (p: P2<DomainSpace>) -> Double in s * m(P2<DomainSpace>(p.x / s, p.y / s)) }
                }
                let zero: ContourRefine.Zero? = r.zero.map { z -> ContourRefine.Zero in
                    { (p: P2<DomainSpace>) -> P2<DomainSpace>? in
                        z(P2<DomainSpace>(p.x / s, p.y / s)).map { P2<DomainSpace>($0.x * s, $0.y * s) }
                    }
                }
                return ContourRefine(contours: { _, _, lines, _ in lines }, magnitude: magnitude, zero: zero)
            }
            let scaled = Scene(surface: surface, occluder: occluder, tiles: h.tiles, region: h.region,
                               step: h.step, layers: layers, camera: scene.camera, depthTolerance: scene.depthTolerance * s,
                               refine: refine)
            let visibleScaled = scaled.visibleLayers()
            var worst = 0.0, structure = true
            for ((layer, a), (_, b)) in zip(visible, visibleScaled) {
                guard a.offsets == b.offsets else {
                    structure = false
                    Check.expect(false, "x\(s): \(layer.spec.name) has the same runs",
                                 "\(a.count) vs \(b.count) paths, \(a.vertices.count) vs \(b.vertices.count) vertices")
                    continue
                }
                for (p, q) in zip(a.vertices, b.vertices) {
                    worst = max(worst, simd_length(p.v * s - q.v) / s)
                }
            }
            Check.expect(structure && worst < 1e-6,
                         "x\(s): every layer is the drawing scaled, vertex for vertex",
                         String(format: "worst |Δ| / scale %.1e over %d layers", worst, visible.count))
        }
    }

    Check.suite("stress: on a 4x4 grid the traversal is the brute-force ray cast, exactly") {
        // Heights drawn at random on a grid of four cells a side: every
        // cell is a different slope, every crossing is a crease, and no
        // approximation has anywhere to hide. The traversal is held to a ray
        // cast against the triangles the depth pass draws -- the grid's
        // cells as it splits them, the truncation edge pieces, the walls -- at
        // points on the drawn surface, on its grid lines, and anywhere in
        // the box; under every camera; at three depth tolerances. The oracle is
        // `exactlyHidden`: brute force over the triangles, with the depth tolerance
        // meaning what the traversal means by it, and at a depth tolerance of zero also
        // the plain question of whether the ray crosses a triangle at all.
        // The two must agree wherever the oracle's own answer is not within
        // rounding of changing.
        for (label, truncation) in [("uncut", Truncation.none), ("truncated at 2.2", .uniform(2.2))] {
            let field = randomHeightfield(cells: 4, seed: 0xC0FFEE, truncation: truncation)
            let base = -1.0
            let mesh = drawnTriangles(field, base: base)
            let rng = SplitMix(seed: 42)
            let d = field.surface.domain
            let box = (lo: SIMD2(min(d.real.lo, d.real.hi), min(d.imag.lo, d.imag.hi)),
                       hi: SIMD2(max(d.real.lo, d.real.hi), max(d.imag.lo, d.imag.hi)))
            var points: [P3<WorldSpace>] = []
            for _ in 0..<300 {
                let p = P2<WorldSpace>(rng.next(d.real.lo, d.real.hi), rng.next(d.imag.lo, d.imag.hi))
                points.append(P3(p.x, p.y, field.drawnHeight(at: p)))
            }
            for _ in 0..<100 {
                points.append(P3(rng.next(d.real.lo, d.real.hi), rng.next(d.imag.lo, d.imag.hi),
                                 rng.next(base, 4)))
            }
            // On the grid lines and diagonals exactly, where two facets meet.
            for _ in 0..<100 {
                let i = Double(Int(rng.next(0, 4.999))), t = rng.next(0, 4)
                let onX = rng.next(0, 1) < 0.5
                let u = onX ? i : t, v = onX ? t : i
                let p = P2<WorldSpace>(d.real.lo + u * d.real.length / 4, d.imag.lo + v * d.imag.length / 4)
                points.append(P3(p.x, p.y, field.drawnHeight(at: p)))
            }
            var compared = 0, disagree = 0, undecidable = 0
            var example = ""
            let clock = ContinuousClock()
            let elapsed = clock.measure {
                for (name, view) in stressCameras() {
                    let toward = -view.projectionDirection
                    let c = view.m.columns
                    let perUnit = abs(simd_dot(SIMD3(c.0.z, c.1.z, c.2.z), toward))
                    for depthTolerance in [0.0, 1e-6, 0.05] {
                        let vis = HeightfieldVisibility(heightfield: field, view: view, depthTolerance: depthTolerance)
                        if abs(vis.toleranceAlongProjection - depthTolerance / perUnit) > 1e-12 * (1 + depthTolerance) { disagree += 1 }
                        for p in points {
                            let strictlyInside = p.x > box.lo.x + 1e-9 && p.x < box.hi.x - 1e-9
                                && p.y > box.lo.y + 1e-9 && p.y < box.hi.y - 1e-9
                            func hidden(_ m: Double) -> Bool {
                                exactlyHidden(mesh, from: p.v, along: toward, depthTolerance: m,
                                              perUnit: perUnit, strictlyInside: strictlyInside, cell: 0.5)
                            }
                            let truth = hidden(depthTolerance)
                            if depthTolerance == 0,
                               (firstHit(mesh, from: p.v, along: toward, after: 1e-10) != nil) != truth {
                                // The two brute forces must agree with each
                                // other too, where neither is on a fence.
                                if hidden(-1e-9) == hidden(1e-9) { disagree += 1 }
                            }
                            // The oracle a hair either side of the depth tolerance: where
                            // it changes its mind, rounding decides and no
                            // answer is wrong.
                            guard hidden(depthTolerance - 1e-9) == truth, hidden(depthTolerance + 1e-9) == truth else {
                                undecidable += 1; continue
                            }
                            compared += 1
                            if vis.isClear(p) == truth {
                                disagree += 1
                                if disagree <= 6 {
                                    let under = underSurface(mesh, p.v, tolerance: 1e-9 * 0.5)
                                    let exit = under ? (firstHit(mesh, from: p.v, along: toward, after: 1e-10) ?? .infinity) : 0
                                    let perp = underSomeTriangle(mesh, from: p.v, along: toward, by: depthTolerance, startInside: strictlyInside, upTo: exit)
                                    let ray = firstHit(mesh, from: p.v, along: toward, after: max(exit, depthTolerance / perUnit) + 1e-10)
                                    print(String(format: "    DISAGREE %@ m=%g at (%.4f, %.4f, %.4f) inside %@: oracle %@ (under %@ exit %.4f perp %@ rayHit %@ reach %.4f) drawn %.4f",
                                                 name, depthTolerance, p.x, p.y, p.z, strictlyInside ? "yes" : "no", truth ? "hidden" : "seen",
                                                 under ? "y" : "n", exit, perp ? "y" : "n", ray.map { String(format: "%.4f", $0) } ?? "-",
                                                 depthTolerance / perUnit, field.drawnHeight(at: p.xy)))
                                }
                                if example.isEmpty {
                                    example = String(format: "e.g. %@ m=%g at (%.3f, %.3f, %.3f): oracle %@",
                                                     name, depthTolerance, p.x, p.y, p.z, truth ? "hidden" : "seen")
                                }
                            }
                        }
                    }
                }
            }
            Check.expect(disagree == 0 && compared > 40_000,
                         "\(label): the traversal agrees with the ray cast at every decidable point",
                         "\(disagree) of \(compared) disagree, \(undecidable) within rounding of the depth tolerance \(example)")
            Check.expect(elapsed < .seconds(60), "\(label): in reasonable time", "\(elapsed)")
        }
    }

    Check.suite("stress: the facing read off the grid converges on the drawn facets") {
        // The one approximation left in the predicate, measured rather than
        // hidden. Curves on the surface are cut by the central-difference normal
        // -- the normal the fold curves are traced from, so a contour ends on
        // the outline as drawn -- while the solid it is traversed through is
        // the triangulated grid, whose facets turn at grid lines. The
        // two can disagree about which way the surface faces only within a
        // cell of a fold, and that band shrinks with the cell: the
        // disagreement, as a fraction of random surface points, must fall
        // as the grid is refined and be small at a drawing grid. On a
        // 4x4 grid it is large, and that is a fact about central
        // differences on four cells, not a tolerance chosen to pass.
        func height(_ x: Double, _ y: Double) -> Double {
            2.5 * exp(-(x * x + 0.6 * y * y) / 0.3) + 0.4 * sin(3 * x) * cos(2 * y)
        }
        let view = testCamera(elevation: -40, azimuth: 30)
        let sight = view.projectionDirection
        var rates: [(cells: Int, rate: Double)] = []
        for cells in [4, 8, 16, 32, 64, 128] {
            let n = cells + 1
            let domain = Domain(real: Interval(lo: -1.5, hi: 1.5), imag: Interval(lo: -1.5, hi: 1.5))
            var values = [Float](repeating: 0, count: n * n)
            for j in 0..<n {
                for i in 0..<n {
                    values[j * n + i] = Float(height(-1.5 + 3 * Double(i) / Double(cells),
                                                     -1.5 + 3 * Double(j) / Double(cells)))
                }
            }
            let field = Heightfield(surface: Surface(height: Grid2D(width: n, height: n, domain: domain, values: values),
                                                     phase: nil, truncation: .none),
                                    occluder: .empty, tiles: [.identity], step: 1)
            let vis = HeightfieldVisibility(heightfield: field, view: view, depthTolerance: 0)
            let rng = SplitMix(seed: UInt64(cells))
            let cell = 3.0 / Double(cells)
            var differ = 0
            let total = 20_000
            for _ in 0..<total {
                let p = P2<WorldSpace>(rng.next(-1.4, 1.4), rng.next(-1.4, 1.4))
                // The facet's own normal: the drawn surface's one-sided slopes
                // a hair into the facet the point is in.
                let e = 1e-6 * cell
                let h0 = field.drawnHeight(at: p)
                let hx = (field.drawnHeight(at: P2(p.x + e, p.y)) - h0) / e
                let hy = (field.drawnHeight(at: P2(p.x, p.y + e)) - h0) / e
                let facet = -simd_dot(SIMD3(-hx, -hy, 1), sight)
                if (facet > 0) != (vis.facing(p) > 0) { differ += 1 }
            }
            rates.append((cells, Double(differ) / Double(total)))
        }
        let text = rates.map { String(format: "%d: %.2f%%", $0.cells, 100 * $0.rate) }.joined(separator: "  ")
        var falls = true
        for (a, b) in zip(rates, rates.dropFirst()) where b.rate > 0.75 * a.rate && a.rate > 0.002 { falls = false }
        Check.expect(falls, "the disagreement falls as the grid is refined", text)
        Check.expect(rates.last!.rate < 0.01, "and is under 1% of surface points at 128 cells", text)
    }
}
