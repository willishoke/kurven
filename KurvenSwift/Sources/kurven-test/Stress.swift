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
// on a steep flank or an edge-on wall exceeds any margin, and the span
// changes with the resolution -- and the rules here are built so that no
// pixel is consulted: the facing of the surface the ink lies on, and the
// march up the sight line through the drawn solid (`SightMarch`), which is
// exact because the drawn solid is piecewise linear. So the suites ask the
// strongest questions such a rule can be asked: the same bake at every
// depth resolution, the same drawing scaled when the plate is scaled, and
// on a lattice too coarse for any approximation to hide in, agreement with
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
/// the lattice's cells as the rasterizer splits them, clamped at the
/// vertices, except that a triangle the cap cuts is replaced by its rim
/// pieces (`Mesh.capRim`), so the surface is exactly min(interpolated, cap);
/// plus wall curtains down to `base` and a floor. This is the boundary of
/// what the depth pass draws, and a sight line that is inside the solid
/// anywhere crosses one of these triangles.
func drawnTriangles(_ h: Heightfield, base: Double) -> Mesh<WorldSpace> {
    let s = h.surface, g = s.height
    let step = max(h.step, 1)
    let nx = (g.width + step - 1) / step, ny = (g.height + step - 1) / step
    func at(_ i: Int, _ j: Int) -> P3<WorldSpace> {
        let p = g.position(x: i * step, y: j * step)
        return P3(p.x, p.y, min(Double(g[i * step, j * step]), s.caps.height(atX: p.x)))
    }
    func excess(_ i: Int, _ j: Int) -> Double {
        let p = g.position(x: i * step, y: j * step)
        return Double(g[i * step, j * step]) - s.caps.height(atX: p.x)
    }
    var vertices: [P3<WorldSpace>] = []
    var triangles: [SIMD3<Int32>] = []
    func tri(_ a: P3<WorldSpace>, _ b: P3<WorldSpace>, _ c: P3<WorldSpace>) {
        let k = Int32(vertices.count)
        vertices += [a, b, c]
        triangles.append(SIMD3(k, k + 1, k + 2))
    }
    /// A triangle the cap cuts: a corner over it and a corner under it.
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
    // Walls round the four sides, up to the drawn crest -- which, where a
    // boundary cell straddles the cap, bends at the rim -- and the floor.
    func raw(_ i: Int, _ j: Int) -> P3<WorldSpace> {
        let p = g.position(x: i * step, y: j * step)
        return P3(p.x, p.y, Double(g[i * step, j * step]))
    }
    func curtain(_ i0: Int, _ j0: Int, _ i1: Int, _ j1: Int) {
        let a = raw(i0, j0), b = raw(i1, j1)
        let ca = s.caps.height(atX: a.x), cb = s.caps.height(atX: b.x)
        var crest = [P3<WorldSpace>(a.x, a.y, min(a.z, ca))]
        if (a.z - ca > 0) != (b.z - cb > 0), ca.isFinite, cb.isFinite {
            let t = (a.z - ca) / ((a.z - ca) - (b.z - cb))
            crest.append(P3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, ca + (cb - ca) * t))
        }
        crest.append(P3(b.x, b.y, min(b.z, cb)))
        for (p, q) in zip(crest, crest.dropFirst()) {
            let p0 = P3<WorldSpace>(p.x, p.y, base), q0 = P3<WorldSpace>(q.x, q.y, base)
            tri(p, q, p0); tri(q, q0, p0)
        }
    }
    for i in 0..<(nx - 1) { curtain(i, 0, i + 1, 0); curtain(i, ny - 1, i + 1, ny - 1) }
    for j in 0..<(ny - 1) { curtain(0, j, 0, j + 1); curtain(nx - 1, j, nx - 1, j + 1) }
    let c00 = at(0, 0), c10 = at(nx - 1, 0), c01 = at(0, ny - 1), c11 = at(nx - 1, ny - 1)
    func floor(_ p: P3<WorldSpace>) -> P3<WorldSpace> { P3(p.x, p.y, base) }
    tri(floor(c00), floor(c10), floor(c01)); tri(floor(c10), floor(c11), floor(c01))
    let rim = Mesh.capRim(of: s, step: step, region: h.region, tiles: h.tiles)
    return Mesh.concat([Mesh(vertices: vertices, triangles: triangles), rim])
}

/// Whether the line from `origin` along `direction`, up to distance
/// `upTo`, passes under some facet of `mesh` by more than `margin`,
/// perpendicular to that facet: for every triangle with a footprint, the
/// stretch of the line over the footprint, and the facet's gap at its ends,
/// which is where a linear gap is least. The start of the line counts only
/// when `startInside`. Brute force over the triangles, sharing nothing
/// with the march.
func underSomeFacet(_ mesh: Mesh<WorldSpace>, from origin: SIMD3<Double>,
                    along direction: SIMD3<Double>, by margin: Double, startInside: Bool,
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
        // with the footprint's winding.
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
            if gap(s) + margin < 0 { return true }
        }
    }
    return false
}

/// Whether a point is under the drawn surface, or on it to within
/// `tolerance`: under some facet whose footprint holds it.
func underSurface(_ mesh: Mesh<WorldSpace>, _ p: SIMD3<Double>, tolerance: Double) -> Bool {
    underSomeFacet(mesh, from: p, along: SIMD3(0, 0, 1), by: -tolerance, startInside: true, upTo: 0)
}

/// The oracle for the march, brute force over the drawn triangles and
/// sharing nothing with it. A point that starts inside the solid (strictly
/// over the domain) is in its own surface's neighbourhood until the line
/// first crosses out: there it is hidden only if it lies under some facet
/// by more than the margin, perpendicular to that facet. After that, any
/// crossing of the boundary beyond the margin's reach along the line is
/// occlusion. `perUnit` is view depth per unit along the line.
func exactlyHidden(_ mesh: Mesh<WorldSpace>, from p: SIMD3<Double>, along toward: SIMD3<Double>,
                   margin: Double, perUnit: Double, strictlyInside: Bool, cell: Double) -> Bool {
    let reach = margin / perUnit
    var exit = 0.0
    if underSurface(mesh, p, tolerance: 1e-9 * cell) {
        exit = firstHit(mesh, from: p, along: toward, after: 1e-10) ?? .infinity
    }
    if underSomeFacet(mesh, from: p, along: toward, by: margin, startInside: strictlyInside, upTo: exit) {
        return true
    }
    return firstHit(mesh, from: p, along: toward, after: max(exit, reach) + 1e-10) != nil
}

/// A small heightfield with random heights, for the lattice tests.
func randomLattice(cells: Int, seed: UInt64, caps: Caps, amplitude: Double = 3) -> Heightfield {
    let n = cells + 1
    let rng = SplitMix(seed: seed)
    let domain = Domain(real: Interval(lo: -1, hi: 1), imag: Interval(lo: -1.5, hi: 0.5))
    var values = [Float](repeating: 0, count: n * n)
    for k in values.indices { values[k] = Float(rng.next(0, amplitude)) }
    let surface = Surface(height: Grid2D(width: n, height: n, domain: domain, values: values),
                          phase: nil, caps: caps)
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
        // The plate with every kind of feature: plateaus under the cap, cut
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
        Check.expect(reference.strokes.pathCount > 200 && reference.strokes.inkLength > 100,
                     "the plate draws", "\(reference.strokes.pathCount) paths, \(String(format: "%.1f", reference.strokes.inkLength)) units of ink")
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

        // The steep case: erf under a cap of 14 over a four-unit window,
        // where the depth test lost a third of the contour ink to pixel
        // span. The drawing is still the same at every resolution, and the
        // ink the judge removes is removed by the solid, not by a pixel.
        var steep = LandscapeRequest(preset: Catalog.native.preset("erf")!, resolution: 240)
        steep.domain = Domain(real: Interval(lo: -2, hi: 2), imag: Interval(lo: -2, hi: 2))
        steep.caps = .uniform(14)
        let steepBundle = try NativeLandscape.build(steep)
        let steepScene = Scene(bundle: steepBundle, preset: steepBundle.manifest.presets[0])
        let coarse = try renderer.bake(steepScene, options: BakeOptions(resolution: 64))
        let fine = try renderer.bake(steepScene, options: BakeOptions(resolution: 4096))
        var same = true
        for (x, y) in zip(coarse.strokes.layers, fine.strokes.layers) where x.paths != y.paths { same = false }
        Check.expect(same, "the steep plate too: 64 px and 4096 px bake the same strokes")
        // What the solid hides of the facing contour ink, and why: ink
        // behind a nearer flank, or ink under its own drawn surface by more
        // than the margin -- the lattice's interpolation gap, which a finer
        // lattice closes and the margin is for. Reported, since neither
        // number is a target; the floor only says the march is not
        // erasing the plate.
        let h = steepScene.heightfield!
        let vis = HeightfieldVisibility(heightfield: h, view: steepScene.camera.view, margin: steepScene.margin)
        var kept = 0.0, facingOnly = 0.0, buried = 0, behind = 0
        for layer in steepScene.layers where layer.spec.liesOnSurface {
            kept += h.visibleSurfaceInk(layer.paths, view: steepScene.camera.view, margin: steepScene.margin).inkLength
            facingOnly += h.visibleSurfaceInk(layer.paths, view: steepScene.camera.view, margin: .infinity).inkLength
            for v in layer.paths.vertices where vis.facing(v.xy) > 0 && !vis.isClear(v) {
                if v.z < h.drawnHeight(at: v.xy) { buried += 1 } else { behind += 1 }
            }
        }
        Check.expect(kept > 0.5 * facingOnly,
                     "of the facing contour ink, the solid hides what lies behind a nearer flank",
                     String(format: "%.1f of %.1f units kept; %d hidden vertices above their own facet, %d under it",
                            kept, facingOnly, behind, buried))
    }

    Check.suite("stress: on a real plate the march is the brute-force ray cast") {
        // rgamma's every facing contour vertex, the march against the
        // perpendicular-gap cast over the plate's drawn triangles at the
        // plate's margin, and against the depth buffer the bake used to
        // judge by. The pit at -4, a cone of slope 24 cut by the front
        // wall, is reported on its own: its far wall through the notch lies
        // partly behind the pit's own side under the plate's diagonal sight
        // line, which a depth buffer mostly misses -- the wall rasterizes
        // to slivers -- and which no vertical tolerance could judge.
        let preset = Catalog.native.preset("rgamma")!
        let bundle = try NativeLandscape.build(LandscapeRequest(preset: preset, resolution: 240))
        let scene = Scene(bundle: bundle, preset: bundle.manifest.presets[0])
        let h = scene.heightfield!
        let view = scene.camera.view
        let mesh = drawnTriangles(h, base: 0)
        let vis = HeightfieldVisibility(heightfield: h, view: view, margin: scene.margin)
        let toward = -view.sightLine
        let c = view.m.columns
        let perUnit = abs(simd_dot(SIMD3(c.0.z, c.1.z, c.2.z), toward))
        let d = h.surface.domain
        let box = (lo: SIMD2(min(d.real.lo, d.real.hi), min(d.imag.lo, d.imag.hi)),
                   hi: SIMD2(max(d.real.lo, d.real.hi), max(d.imag.lo, d.imag.hi)))
        let depth = try MetalRenderer().bake(scene, options: BakeOptions(resolution: 1000)).depth
        let g = h.surface.height
        let cell = max(abs(g.domain.real.length) / Double(g.width - 1), abs(g.domain.imag.length) / Double(g.height - 1))
        var total = 0, marchHidden = 0, castHidden = 0, disagree = 0, depthHidden = 0
        var pit = (total: 0, march: 0, cast: 0, depth: 0)
        let clock = ContinuousClock()
        let elapsed = clock.measure {
            for layer in scene.layers where layer.spec.liesOnSurface {
                let vertices = layer.paths.vertices
                let facing = vertices.map { vis.facing($0.xy) > 0 }
                let march = vertices.map { vis.isClear($0) }
                let cast = Heightfield.parallelMap(vertices) { v in
                    let strictlyInside = v.x > box.lo.x + 1e-9 && v.x < box.hi.x - 1e-9
                        && v.y > box.lo.y + 1e-9 && v.y < box.hi.y - 1e-9
                    return !exactlyHidden(mesh, from: v.v, along: toward, margin: scene.margin,
                                          perUnit: perUnit, strictlyInside: strictlyInside, cell: cell)
                }
                for (k, v) in vertices.enumerated() where facing[k] {
                    total += 1
                    let q = view(v)
                    let byDepth = q.z + scene.margin > depth.depth(under: q.xy)
                    if !march[k] { marchHidden += 1 }
                    if !cast[k] { castHidden += 1 }
                    if !byDepth { depthHidden += 1 }
                    if march[k] != cast[k] { disagree += 1 }
                    if v.x > -4.3, v.x < -3.7, v.y < 0.6 {
                        pit.total += 1
                        if !march[k] { pit.march += 1 }
                        if !cast[k] { pit.cast += 1 }
                        if !byDepth { pit.depth += 1 }
                    }
                }
            }
        }
        Check.expect(disagree == 0 && total > 5000,
                     "the march and the cast agree on every facing contour vertex",
                     "\(total) vertices; march hides \(marchHidden), cast hides \(castHidden), the 1000 px depth buffer hid \(depthHidden)")
        Check.expect(pit.march == pit.cast && pit.march > pit.depth,
                     "in the pit at -4 both hide what the depth buffer missed",
                     "\(pit.total) facing vertices: march \(pit.march), cast \(pit.cast), depth buffer \(pit.depth)")
        Check.expect(elapsed < .seconds(60), "in reasonable time", "\(elapsed)")
    }

    Check.suite("stress: scaling the plate scales the drawing") {
        // The whole plate -- heights, domain, caps, walls, ink and margin --
        // scaled by a power of two, so every float scales exactly; the judged
        // ink must be the judged ink scaled, vertex for vertex. A rule with a
        // pixel or an absolute tolerance in it would not pass this; one that
        // is geometry alone cannot fail it.
        let preset = Catalog.native.preset("rgamma")!
        let bundle = try NativeLandscape.build(LandscapeRequest(preset: preset, resolution: 240))
        let scene = Scene(bundle: bundle, preset: bundle.manifest.presets[0])
        let h = scene.heightfield!
        let judged = scene.judgedLayers()
        for s in [8.0, 0.125] {
            let g = h.surface.height
            let domain = Domain(real: Interval(lo: g.domain.real.lo * s, hi: g.domain.real.hi * s),
                                imag: Interval(lo: g.domain.imag.lo * s, hi: g.domain.imag.hi * s))
            let caps: Caps
            switch h.surface.caps {
            case .none: caps = .none
            case .uniform(let z): caps = .uniform(z * s)
            case .realBands(let bands, let beyond):
                caps = .realBands(bands.map { RealBand(below: $0.below * s, cap: $0.cap * s) }, beyond: beyond * s)
            }
            let surface = Surface(height: Grid2D(width: g.width, height: g.height, domain: domain,
                                                 values: g.values.map { $0 * Float(s) }),
                                  phase: h.surface.phase, caps: caps)
            let occluder = Mesh(vertices: h.occluder.vertices.map { P3<WorldSpace>($0.v * s) },
                                triangles: h.occluder.triangles)
            let layers = scene.layers.map { layer in
                Layer(spec: layer.spec,
                      paths: PolylineSet(vertices: layer.paths.vertices.map { P3<WorldSpace>($0.v * s) },
                                         offsets: layer.paths.offsets))
            }
            // The scaled plate is the scaled function: the refiner's |f| and
            // its zeros, read at the point scaled back, scaled up again. The
            // contours are not re-derived under the judge, so they pass.
            let refine: ContourRefine? = h.refine.map { r -> ContourRefine in
                let magnitude: ContourRefine.Magnitude? = r.magnitude.map { m -> ContourRefine.Magnitude in
                    { (p: P2<DomainSpace>) -> Double in s * m(P2<DomainSpace>(p.x / s, p.y / s)) }
                }
                let zero: ContourRefine.Zero? = r.zero.map { z -> ContourRefine.Zero in
                    { (p: P2<DomainSpace>) -> P2<DomainSpace>? in
                        z(P2<DomainSpace>(p.x / s, p.y / s)).map { P2<DomainSpace>($0.x * s, $0.y * s) }
                    }
                }
                return ContourRefine(contours: { _, _, lines in lines }, magnitude: magnitude, zero: zero)
            }
            let scaled = Scene(surface: surface, occluder: occluder, tiles: h.tiles, region: h.region,
                               step: h.step, layers: layers, camera: scene.camera, margin: scene.margin * s,
                               refine: refine)
            let judgedScaled = scaled.judgedLayers()
            var worst = 0.0, structure = true
            for ((layer, a), (_, b)) in zip(judged, judgedScaled) {
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
                         String(format: "worst |Δ| / scale %.1e over %d layers", worst, judged.count))
        }
    }

    Check.suite("stress: on a 4x4 lattice the march is the brute-force ray cast, exactly") {
        // Heights drawn at random on a lattice of four cells a side: every
        // cell is a different slope, every crossing is a crease, and no
        // approximation has anywhere to hide. The march is held to a ray
        // cast against the triangles the depth pass draws -- the lattice's
        // cells as it splits them, the cap's rim pieces, the walls -- at
        // points on the drawn surface, on its lattice lines, and anywhere in
        // the box; under every camera; at three margins. The oracle is
        // `exactlyHidden`: brute force over the triangles, with the margin
        // meaning what the march means by it, and at a margin of zero also
        // the plain question of whether the ray crosses a triangle at all.
        // The two must agree wherever the oracle's own answer is not within
        // rounding of changing.
        for (label, caps) in [("uncut", Caps.none), ("capped at 2.2", .uniform(2.2))] {
            let field = randomLattice(cells: 4, seed: 0xC0FFEE, caps: caps)
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
            // On the lattice lines and diagonals exactly, where two facets meet.
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
                    let toward = -view.sightLine
                    let c = view.m.columns
                    let perUnit = abs(simd_dot(SIMD3(c.0.z, c.1.z, c.2.z), toward))
                    for margin in [0.0, 1e-6, 0.05] {
                        let vis = HeightfieldVisibility(heightfield: field, view: view, margin: margin)
                        if abs(vis.marginAlongSight - margin / perUnit) > 1e-12 * (1 + margin) { disagree += 1 }
                        for p in points {
                            let strictlyInside = p.x > box.lo.x + 1e-9 && p.x < box.hi.x - 1e-9
                                && p.y > box.lo.y + 1e-9 && p.y < box.hi.y - 1e-9
                            func hidden(_ m: Double) -> Bool {
                                exactlyHidden(mesh, from: p.v, along: toward, margin: m,
                                              perUnit: perUnit, strictlyInside: strictlyInside, cell: 0.5)
                            }
                            let truth = hidden(margin)
                            if margin == 0,
                               (firstHit(mesh, from: p.v, along: toward, after: 1e-10) != nil) != truth {
                                // The two brute forces must agree with each
                                // other too, where neither is on a fence.
                                if hidden(-1e-9) == hidden(1e-9) { disagree += 1 }
                            }
                            // The oracle a hair either side of the margin: where
                            // it changes its mind, rounding decides and no
                            // answer is wrong.
                            guard hidden(margin - 1e-9) == truth, hidden(margin + 1e-9) == truth else {
                                undecidable += 1; continue
                            }
                            compared += 1
                            if vis.isClear(p) == truth {
                                disagree += 1
                                if disagree <= 6 {
                                    let under = underSurface(mesh, p.v, tolerance: 1e-9 * 0.5)
                                    let exit = under ? (firstHit(mesh, from: p.v, along: toward, after: 1e-10) ?? .infinity) : 0
                                    let perp = underSomeFacet(mesh, from: p.v, along: toward, by: margin, startInside: strictlyInside, upTo: exit)
                                    let ray = firstHit(mesh, from: p.v, along: toward, after: max(exit, margin / perUnit) + 1e-10)
                                    print(String(format: "    DISAGREE %@ m=%g at (%.4f, %.4f, %.4f) inside %@: oracle %@ (under %@ exit %.4f perp %@ rayHit %@ reach %.4f) drawn %.4f",
                                                 name, margin, p.x, p.y, p.z, strictlyInside ? "yes" : "no", truth ? "hidden" : "seen",
                                                 under ? "y" : "n", exit, perp ? "y" : "n", ray.map { String(format: "%.4f", $0) } ?? "-",
                                                 margin / perUnit, field.drawnHeight(at: p.xy)))
                                }
                                if example.isEmpty {
                                    example = String(format: "e.g. %@ m=%g at (%.3f, %.3f, %.3f): oracle %@",
                                                     name, margin, p.x, p.y, p.z, truth ? "hidden" : "seen")
                                }
                            }
                        }
                    }
                }
            }
            Check.expect(disagree == 0 && compared > 40_000,
                         "\(label): the march agrees with the ray cast at every decidable point",
                         "\(disagree) of \(compared) disagree, \(undecidable) within rounding of the margin \(example)")
            Check.expect(elapsed < .seconds(60), "\(label): in reasonable time", "\(elapsed)")
        }
    }

    Check.suite("stress: the facing read off the lattice converges on the drawn facets") {
        // The one approximation left in the predicate, measured rather than
        // hidden. Ink on the surface is cut by the central-difference normal
        // -- the normal the fold lines are traced from, so a contour ends on
        // the silhouette as drawn -- while the solid it is marched through is
        // the triangulated lattice, whose facets turn at lattice lines. The
        // two can disagree about which way the surface faces only within a
        // cell of a fold, and that band shrinks with the cell: the
        // disagreement, as a fraction of random surface points, must fall
        // as the lattice is refined and be small at a drawing lattice. On a
        // 4x4 lattice it is large, and that is a fact about central
        // differences on four cells, not a tolerance chosen to pass.
        func height(_ x: Double, _ y: Double) -> Double {
            2.5 * exp(-(x * x + 0.6 * y * y) / 0.3) + 0.4 * sin(3 * x) * cos(2 * y)
        }
        let view = testCamera(elevation: -40, azimuth: 30)
        let sight = view.sightLine
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
                                                     phase: nil, caps: .none),
                                    occluder: .empty, tiles: [.identity], step: 1)
            let vis = HeightfieldVisibility(heightfield: field, view: view, margin: 0)
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
        Check.expect(falls, "the disagreement falls as the lattice is refined", text)
        Check.expect(rates.last!.rate < 0.01, "and is under 1% of surface points at 128 cells", text)
    }
}
