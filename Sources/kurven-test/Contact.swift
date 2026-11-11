import Foundation
import simd
import KurvenCore
import KurvenDynamics
import KurvenLandscape

// MARK: - where a part meets the terrain
//
// A composite's contact line (`Composite.contact(of:)`), held to the two
// surfaces it lies on: every vertex, and every chord's middle, on the
// landscape's drawn height (`Heightfield.drawnHeight(at:)`, the grid's
// triangles cut at the truncation, read apart from the composite's slab), and
// on its part -- a polyhedron's face, the round torus. On a plane of the grid
// a box's contact is the polygon the plane cuts from it, vertex by vertex. A
// part that floats draws none. Its visibility is the composite gate's.

/// The decimated grid's cell of a landscape, the larger of its sides.
func landscapeCell(_ h: Heightfield) -> Double {
    let g = h.surface.height, d = h.surface.domain
    return max(abs(d.real.length) / Double(g.width - 1), abs(d.imag.length) / Double(g.height - 1))
        * Double(max(h.step, 1))
}

/// The distance from `p` to the closed polygon `loop` in the plane.
func distance(_ p: SIMD2<Double>, toLoop loop: [SIMD2<Double>]) -> Double {
    loop.indices.map { k in
        let a = loop[k], b = loop[(k + 1) % loop.count], ab = b - a
        let t = min(max(simd_dot(p - a, ab) / max(simd_length_squared(ab), 1e-300), 0), 1)
        return simd_length(p - (a + t * ab))
    }.min() ?? .infinity
}

/// The distance from `p` to the polyline `path` in space.
func distance(_ p: SIMD3<Double>, toPath path: [SIMD3<Double>]) -> Double {
    zip(path, path.dropFirst()).map { a, b in
        let ab = b - a
        let t = min(max(simd_dot(p - a, ab) / max(simd_length_squared(ab), 1e-300), 0), 1)
        return simd_length(p - (a + t * ab))
    }.min() ?? .infinity
}

/// How far `l`, a point of a landscape's own coordinates, stands off its
/// drawn height (`Heightfield.drawnHeight(at:)`), within its domain and
/// footprint: the least gap over the point and points a trillionth of a
/// cell either way along x and y, so a point on a step of the height -- a
/// riser, a triangle with no top -- is on the side it stands on. Infinite
/// off the terrain.
func terrainGap(_ h: Heightfield, _ l: SIMD3<Double>) -> Double {
    let d = h.surface.domain, e = 1e-12 * landscapeCell(h)
    let xlo = min(d.real.lo, d.real.hi), xhi = max(d.real.lo, d.real.hi)
    let ylo = min(d.imag.lo, d.imag.hi), yhi = max(d.imag.lo, d.imag.hi)
    return [SIMD2(0.0, 0.0), SIMD2(e, 0), SIMD2(-e, 0), SIMD2(0, e), SIMD2(0, -e)].map { step -> Double in
        let p = SIMD2(l.x, l.y) + step
        guard h.tiles.count > 1 || (p.x >= xlo - e && p.x <= xhi + e && p.y >= ylo - e && p.y <= yhi + e),
              h.region.contains(P2(p.x, p.y)) else { return .infinity }
        let z = h.drawnHeight(at: P2(p.x, p.y))
        return z.isFinite ? abs(l.z - z) : .infinity
    }.min()!
}

/// How far a part's contact line strays from the terrain and from the part,
/// in world units, and how many points were held: each vertex, and each
/// chord's middle -- on the terrain always, on a polyhedron's face, not on
/// the torus, whose chords cut inside it.
func contactStray(_ composite: Composite, part i: Int) -> (points: Int, terrain: Double, part: Double) {
    let part = composite.parts[i], s = part.placement.scale, o = part.placement.offset
    let contact = composite.contact(of: i)
    let landscapes: [(Heightfield, SIMD3<Double>)] = composite.parts.compactMap { p in
        guard case .heightfield(let h) = p.scene.geometry else { return nil }
        return (h, p.placement.offset)
    }
    var points = 0, terrain = 0.0, onPart = 0.0
    for (k, path) in contact.paths.enumerated() {
        let middles = zip(path, path.dropFirst()).map { 0.5 * ($0.v + $1.v) }
        for (q, vertex) in path.map({ ($0.v, true) }) + middles.map({ ($0, false) }) {
            points += 1
            let w = s * q + o
            terrain = max(terrain, landscapes.map { h, offset in terrainGap(h, w - offset) }.min()!)
            switch part.scene.geometry {
            case .solid(let solid):
                let f = contact.faces[k], frame = solid.frames[f]
                let c = frame.coordinates(q)
                let loops = solid.loops[f].map { $0.map { frame.coordinates(solid.vertices[$0].v) } }
                let off = solid.face(f).copies[0].contains(P2(c)) ? 0 : loops.map { distance(c, toLoop: $0) }.min()!
                onPart = max(onPart, s * max(abs(simd_dot(q - frame.origin, frame.normal)), off))
            case .parametric(let surface) where vertex:
                let (R, r) = torusRadii(surface)
                let rho = (q.x * q.x + q.y * q.y).squareRoot()
                onPart = max(onPart, s * abs(((rho - R) * (rho - R) + q.z * q.z).squareRoot() - r))
            default:
                break
            }
        }
    }
    return (points, terrain, onPart)
}

func contactTests() {
    Check.suite("composite: a contact line lies on the terrain and on its part") {
        let (cameras, compositions) = try gateCompositions()
        for (name, parts) in compositions {
            let composite = try Composite(parts, camera: cameras[0].1)
            // A billionth of the landscape's cell, as the cut starts a line
            // of sight a billionth of a cell along it (`Slab`).
            let tolerance = composite.parts.compactMap { p -> Double? in
                guard case .heightfield(let h) = p.scene.geometry else { return nil }
                return 1e-9 * landscapeCell(h)
            }.min()!
            var points = 0, terrain = 0.0, onPart = 0.0, crossings = 0, farthest = 0.0
            for i in composite.parts.indices {
                let stray = contactStray(composite, part: i)
                points += stray.points; terrain = max(terrain, stray.terrain); onPart = max(onPart, stray.part)
                // A polyhedron's contact line is exact: it passes through
                // every crossing found apart from it.
                if case .solid = composite.parts[i].scene.geometry, stray.points > 0 {
                    let miss = contactMissed(composite, part: i, steps: 240)
                    crossings += miss.crossings; farthest = max(farthest, miss.farthest)
                }
            }
            let drawn = try composite.visibleLayers().filter { $0.0.spec.name == "contact" }.count
            if points == 0 {
                Check.expect(drawn == 0, "\(name): no part meets the terrain, and there is no contact line")
            } else {
                Check.expect(terrain <= tolerance && onPart <= tolerance && farthest <= tolerance && drawn >= 1,
                             "\(name): every point of the contact line is on the terrain and on its part, and it misses no crossing",
                             String(format: "%d points, off the terrain by %.1e, off the part by %.1e, %d crossings the farthest %.1e off it, within %.1e",
                                    points, terrain, onPart, crossings, farthest, tolerance))
            }
        }
    }

    Check.suite("composite: a box sunk into a plane of the grid meets it in the polygon the plane cuts") {
        // z = 1 + x/4 + y/8 on a grid of 32 cells over [-2, 2]²: every
        // sample a float to the bit, so the drawn triangles are the plane.
        let n = 33
        let domain = Domain(real: Interval(lo: -2, hi: 2), imag: Interval(lo: -2, hi: 2))
        func plane(_ x: Double, _ y: Double) -> Double { 1 + x / 4 + y / 8 }
        var values: [Float] = []
        for j in 0..<n { for i in 0..<n { values.append(Float(plane(-2 + 4 * Double(i) / 32, -2 + 4 * Double(j) / 32))) } }
        let b = try PolyhedronBundle.read(at: Fixtures.dir.appendingPathComponent("contract/polyhedron_cube.kurven"))
        let cube = b.scene(preset: b.manifest.presets[0])
        let placement = Placement(scale: 0.35, offset: SIMD3(0.1, -0.2, 0.6))
        // The cube's vertical edges, where they stand, counter-clockwise.
        var feet: [SIMD2<Double>] = []
        for v in cube.solid!.vertices {
            let w = placement.scale * v.v + placement.offset
            let p = SIMD2(w.x, w.y)
            if !feet.contains(where: { simd_distance($0, p) < 1e-9 }) { feet.append(p) }
        }
        let centre = feet.reduce(.zero, +) / Double(feet.count)
        feet.sort { atan2($0.y - centre.y, $0.x - centre.x) < atan2($1.y - centre.y, $1.x - centre.x) }
        for (what, truncation) in [("the plane", Truncation.none), ("the plane under a truncation at 1", .uniform(1))] {
            let surface = Surface(height: Grid2D(width: n, height: n, domain: domain, values: values),
                                  phase: nil, truncation: truncation)
            let landscape = Scene(surface: surface, occluder: Mesh(vertices: [], triangles: []), tiles: [.identity],
                                  step: 1, layers: [], camera: cube.camera, depthTolerance: 0.02, walls: .none)
            let top: Double = if case .uniform(let t) = truncation { t } else { .infinity }
            func height(_ p: SIMD2<Double>) -> Double { min(plane(p.x, p.y), top) }
            // The polygon: the feet on the terrain, and where the
            // truncation's edge crosses a side between two of them.
            var polygon: [SIMD3<Double>] = []
            for k in feet.indices {
                let a = feet[k], c = feet[(k + 1) % feet.count]
                polygon.append(SIMD3(a.x, a.y, height(a)))
                let ea = plane(a.x, a.y) - top, ec = plane(c.x, c.y) - top
                if ea.isFinite, (ea > 0) != (ec > 0) {
                    let x = a + (c - a) * (ea / (ea - ec))
                    polygon.append(SIMD3(x.x, x.y, top))
                }
            }
            let composite = try Composite([("plane", landscape, .identity), ("cube", cube, placement)], camera: cube.camera)
            let contact = composite.contact(of: 1)
            let paths = contact.paths.map { $0.map { placement.scale * $0.v + placement.offset } }
            let vertices = paths.flatMap { $0 }
            let size = 1.0, tolerance = 1e-12 * size
            let offPolygon = vertices.map { distance($0, toPath: polygon + [polygon[0]]) }.max() ?? .infinity
            let missed = polygon.map { e in vertices.map { simd_distance($0, e) }.min() ?? .infinity }.max()!
            let length = paths.reduce(0) { $0 + zip($1, $1.dropFirst()).reduce(0) { $0 + simd_distance($1.0, $1.1) } }
            let perimeter = polygon.indices.reduce(0) { $0 + simd_distance(polygon[$1], polygon[($1 + 1) % polygon.count]) }
            let sides = Set(contact.faces).count
            Check.expect(offPolygon <= tolerance && missed <= tolerance && abs(length - perimeter) <= 1e-12 * perimeter
                         && sides == 4 && contact.paths.count == 4,
                         "\(what) meets the box in its \(polygon.count)-gon, vertex by vertex",
                         String(format: "%d paths on %d faces, vertices off it by %.1e, its corners missed by %.1e, "
                                + "length %.15f of %.15f", contact.paths.count, sides, offPolygon, missed, length, perimeter))
        }
    }

    Check.suite("composite: a part that floats over the terrain has no contact line") {
        let bundle = try KurvenBundle.read(at: Fixtures.dir.appendingPathComponent("expr/landscape_cubic.kurven"))
        let landscape = Scene(bundle: bundle, preset: bundle.manifest.presets[0])
        let contract = Fixtures.dir.appendingPathComponent("contract")
        let cube = try PolyhedronBundle.read(at: contract.appendingPathComponent("polyhedron_cube.kurven"))
        let torus = try SurfaceBundle.read(at: contract.appendingPathComponent("surface_torus.kurven"))
        // Over the truncation, at 5; the torus ringing a tower, clear of it;
        // and the same cube let down onto the terrain, which meets it.
        for (what, scene, placement, meets) in [
            ("a cube over the towers' tops", cube.scene(preset: cube.manifest.presets[0]),
             Placement(scale: 0.35, offset: SIMD3(-0.5, 0.5, 5.5)), false),
            ("the round torus ringing a tower", torus.scene(preset: torus.manifest.presets[0]),
             Placement(scale: 0.35, offset: SIMD3(1, 0, 1.6)), false),
            ("the cube let down onto the terrain", cube.scene(preset: cube.manifest.presets[0]),
             Placement(scale: 0.35, offset: SIMD3(-0.5, 0.5, 0)), true),
        ] {
            let composite = try Composite([("landscape", landscape, .identity), ("part", scene, placement)],
                                          camera: landscape.camera)
            let layers = try composite.visibleLayers().filter { $0.0.spec.name == "contact" }
            Check.expect(composite.contact(of: 1).isEmpty != meets && layers.isEmpty != meets
                         && composite.contact(of: 0).isEmpty,
                         "\(what) has \(meets ? "a" : "no") contact line",
                         "\(composite.contact(of: 1).paths.count) paths, \(layers.count) layers")
        }
    }
}

// MARK: - a sweep

/// A seeded stream of numbers in [0, 1): a linear congruential generator.
struct SweepRandom {
    var state: UInt64
    mutating func next() -> Double {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Double(state >> 11) / Double(1 << 53)
    }
    mutating func range(_ a: Double, _ b: Double) -> Double { a + (b - a) * next() }
}

/// A landscape of f sampled on an n by n grid over [-2, 2]², with samples
/// replaced as `patch` says.
func gridLandscape(n: Int = 17, truncation: Truncation = .none, region: Region = .full,
                   patch: [(Int, Int, Float)] = [], camera: Camera, f: (Double, Double) -> Double) -> Scene {
    let domain = Domain(real: Interval(lo: -2, hi: 2), imag: Interval(lo: -2, hi: 2))
    var values: [Float] = []
    for j in 0..<n { for i in 0..<n { values.append(Float(f(-2 + 4 * Double(i) / Double(n - 1), -2 + 4 * Double(j) / Double(n - 1)))) } }
    for (i, j, v) in patch { values[j * n + i] = v }
    let surface = Surface(height: Grid2D(width: n, height: n, domain: domain, values: values), phase: nil, truncation: truncation)
    return Scene(surface: surface, occluder: Mesh(vertices: [], triangles: []), tiles: [.identity], region: region,
                 step: 1, layers: [], camera: camera, depthTolerance: 0.02, walls: .none)
}

/// A prism over an L, two units a side, standing on z = 0.
func lPrism() throws -> Solid {
    let l: [SIMD2<Double>] = [SIMD2(0, 0), SIMD2(2, 0), SIMD2(2, 1), SIMD2(1, 1), SIMD2(1, 2), SIMD2(0, 2)]
    let vertices = l.map { P3<WorldSpace>($0.x - 1, $0.y - 1, 0) } + l.map { P3<WorldSpace>($0.x - 1, $0.y - 1, 1) }
    var faces: [[[Int]]] = [[Array(6..<12)], [Array((0..<6).reversed())]]
    for k in 0..<6 { faces.append([[k, (k + 1) % 6, 6 + (k + 1) % 6, 6 + k]]) }
    return try Solid(checking: vertices, faces: faces)
}

/// A solid turned about x, then y, then z.
func turned(_ s: Solid, _ ax: Double, _ ay: Double, _ az: Double) throws -> Solid {
    let rx = simd_double3x3(rows: [SIMD3(1, 0, 0), SIMD3(0, cos(ax), -sin(ax)), SIMD3(0, sin(ax), cos(ax))])
    let ry = simd_double3x3(rows: [SIMD3(cos(ay), 0, sin(ay)), SIMD3(0, 1, 0), SIMD3(-sin(ay), 0, cos(ay))])
    let rz = simd_double3x3(rows: [SIMD3(cos(az), -sin(az), 0), SIMD3(sin(az), cos(az), 0), SIMD3(0, 0, 1)])
    let m = rz * ry * rx
    return try Solid(checking: s.vertices.map { P3(m * $0.v) }, faces: s.loops)
}

/// The distance from `p` to the nearest of `paths`.
func distance(_ p: SIMD3<Double>, toPaths paths: [[SIMD3<Double>]]) -> Double {
    paths.map { distance(p, toPath: $0) }.min() ?? .infinity
}

/// What a contact line misses, found apart from it: on a grid over each
/// face of a polyhedron (`steps` to a side, inside the face, clear of its
/// edges by a millionth of the face) or over the torus's parameter
/// rectangle (`steps` along u, half as many along v), every grid edge along
/// which z − H changes sign continuously -- H the drawn height, read off
/// the landscape apart from the composite -- bisected onto the terrain. The
/// farthest of those points from the contact line, in world units, and how
/// many there were.
func contactMissed(_ composite: Composite, part i: Int, steps: Int) -> (crossings: Int, farthest: Double, at: SIMD3<Double>) {
    let part = composite.parts[i], s = part.placement.scale, o = part.placement.offset
    let landscapes: [(Heightfield, SIMD3<Double>)] = composite.parts.compactMap { p in
        guard case .heightfield(let h) = p.scene.geometry else { return nil }
        return (h, p.placement.offset)
    }
    let contact = composite.contact(of: i).paths.map { $0.map { s * $0.v + o } }
    // z − H at a world point over the first landscape that has terrain there.
    func excess(_ w: SIMD3<Double>) -> Double? {
        for (h, offset) in landscapes {
            let l = w - offset, d = h.surface.domain
            guard l.x >= min(d.real.lo, d.real.hi), l.x <= max(d.real.lo, d.real.hi),
                  l.y >= min(d.imag.lo, d.imag.hi), l.y <= max(d.imag.lo, d.imag.hi),
                  h.region.contains(P2(l.x, l.y)) else { continue }
            let z = h.drawnHeight(at: P2(l.x, l.y))
            if z.isFinite { return l.z - z }
        }
        return nil
    }
    let cell = landscapes.map { landscapeCell($0.0) }.min()!
    // Where each grid runs, which of its points count, and the world point
    // at each.
    var grids: [(lo: SIMD2<Double>, hi: SIMD2<Double>, n: SIMD2<Int>, inside: (SIMD2<Double>) -> Bool, place: (SIMD2<Double>) -> SIMD3<Double>)] = []
    switch part.scene.geometry {
    case .solid(let solid):
        for f in solid.faces.indices {
            let frame = solid.frames[f], face = solid.face(f)
            let loops = solid.loops[f].map { $0.map { frame.coordinates(solid.vertices[$0].v) } }
            let lo = SIMD2(face.domain.u.range.lo, face.domain.v.range.lo), hi = SIMD2(face.domain.u.range.hi, face.domain.v.range.hi)
            let margin = 1e-6 * simd_length(hi - lo)
            grids.append((lo, hi, SIMD2(steps, steps), { c in
                face.copies[0].contains(P2(c)) && loops.allSatisfy { distance(c, toLoop: $0) > margin }
            }, { c in s * frame.point(c) + o }))
        }
    case .parametric(let surface):
        grids.append((SIMD2(surface.u.range.lo, surface.v.range.lo), SIMD2(surface.u.range.hi, surface.v.range.hi),
                      SIMD2(steps, steps / 2), { _ in true }, { c in s * surface.map(P2(c)).position + o }))
    case .heightfield:
        return (0, 0, .zero)
    }
    var crossings: [SIMD3<Double>] = []
    for grid in grids {
        let step = (grid.hi - grid.lo) / SIMD2<Double>(grid.n)
        func at(_ i: Int, _ j: Int) -> SIMD2<Double> { grid.lo + SIMD2(Double(i), Double(j)) * step }
        var values = [Double?](repeating: nil, count: (grid.n.x + 1) * (grid.n.y + 1))
        for j in 0...grid.n.y { for i in 0...grid.n.x where grid.inside(at(i, j)) {
            values[j * (grid.n.x + 1) + i] = excess(grid.place(at(i, j)))
        } }
        for j in 0...grid.n.y {
            for i in 0...grid.n.x {
                for (di, dj) in [(1, 0), (0, 1)] where i + di <= grid.n.x && j + dj <= grid.n.y {
                    guard let ga = values[j * (grid.n.x + 1) + i], let gb = values[(j + dj) * (grid.n.x + 1) + i + di],
                          (ga > 0) != (gb > 0) else { continue }
                    var a = at(i, j), b = at(i + di, j + dj), fa = ga
                    var ok = true
                    for _ in 0..<70 {
                        let m = 0.5 * (a + b)
                        guard let fm = excess(grid.place(m)) else { ok = false; break }
                        if (fm > 0) == (fa > 0) { a = m; fa = fm } else { b = m }
                    }
                    // A crossing, not a step of H.
                    let w = grid.place(0.5 * (a + b))
                    guard ok, let e = excess(w), abs(e) <= 1e-9 * cell else { continue }
                    crossings.append(w)
                }
            }
        }
    }
    let far = Heightfield.parallelMap(crossings, chunk: 16) { distance($0, toPaths: contact) }
    guard let k = far.indices.max(by: { far[$0] < far[$1] }) else { return (0, 0, .zero) }
    return (crossings.count, far[k], crossings[k])
}

/// The open ends of a part's contact paths that are not on a step of the
/// terrain -- its edge, a riser -- nor, on a polyhedron, on a face's
/// boundary: where a path stops short.
func contactLooseEnds(_ composite: Composite, part i: Int) -> Int {
    let part = composite.parts[i], s = part.placement.scale, o = part.placement.offset
    guard case .parametric = part.scene.geometry else { return 0 }
    let landscapes: [(Heightfield, SIMD3<Double>)] = composite.parts.compactMap { p in
        guard case .heightfield(let h) = p.scene.geometry else { return nil }
        return (h, p.placement.offset)
    }
    var loose = 0
    for path in composite.contact(of: i).paths {
        let w = path.map { s * $0.v + o }
        guard simd_distance(w.first!, w.last!) > 1e-12 else { continue }
        for e in [w.first!, w.last!] {
            // Beside a step, some point a millionth of a cell away is off
            // the terrain or a thousandth of a cell off its height.
            let onStep = landscapes.contains { h, offset in
                let cell = landscapeCell(h), l = e - offset
                return (0..<16).contains { k in
                    let a = Double(k) * .pi / 8
                    let p = SIMD2(l.x, l.y) + 1e-6 * cell * SIMD2(cos(a), sin(a))
                    let d = h.surface.domain
                    guard p.x >= min(d.real.lo, d.real.hi), p.x <= max(d.real.lo, d.real.hi),
                          p.y >= min(d.imag.lo, d.imag.hi), p.y <= max(d.imag.lo, d.imag.hi),
                          h.region.contains(P2(p.x, p.y)) else { return true }
                    let z = h.drawnHeight(at: P2(p.x, p.y))
                    return !z.isFinite || abs(z - l.z) > 1e-3 * cell
                }
            }
            if !onStep { loose += 1 }
        }
    }
    return loose
}

func contactSweepTests() {
    let camera = Camera(view: testCamera(elevation: -30, azimuth: 37), projection: .orthographic(oblique: nil))
    func f(_ x: Double, _ y: Double) -> Double { 1 + 0.6 * sin(2 * x) * cos(3 * y) }
    let bands = Truncation.realBands([RealBand(below: -0.53, height: 1.05), RealBand(below: 0.71, height: 0.8)], beyond: 1.3)
    let corners: [SIMD2<Double>] = [SIMD2(-1.9, -1.9), SIMD2(1.9, -1.9), SIMD2(1.9, 0.3), SIMD2(0.2, 0.3), SIMD2(0.2, 1.9), SIMD2(-1.9, 1.9)]
    let footprint = BoundaryPerimeter(edges: corners.indices.map { k in
        PerimeterEdge(start: P2(corners[k].x, corners[k].y),
                      end: P2(corners[(k + 1) % corners.count].x, corners[(k + 1) % corners.count].y), density: 8)
    })
    func spikeHeight(_ x: Double, _ y: Double) -> Double { exp(-20 * (x * x + y * y).squareRoot()) }
    // Each with the height a part is placed by.
    var landscapes: [(name: String, scene: Scene, height: (Double, Double) -> Double)] = [
        ("the sinusoid", gridLandscape(camera: camera, f: f), f),
        ("the sinusoid in bands, on an L", gridLandscape(truncation: bands, region: .inside(footprint), camera: camera, f: f), f),
        ("the sinusoid with a sample not a number, truncated at 1.2",
         gridLandscape(truncation: .uniform(1.2), patch: [(8, 8, .nan)], camera: camera, f: f), f),
        // Slabs of no thickness: every sample at the base.
        ("a flat terrain", gridLandscape(camera: camera) { _, _ in 0.5 }, { _, _ in 0.5 }),
        ("a plateau", gridLandscape(truncation: .uniform(0.5), camera: camera) { _, _ in 2 }, { _, _ in 0.5 }),
    ]
    // The gate's spike, whose terrain is a rounding over its floor away
    // from the apex.
    if let spike = try? NativeLandscape.build(
        LandscapeRequest(expression: "exp(-20*abs(z))", domain: Domain(real: Interval(lo: -2, hi: 2), imag: Interval(lo: -2, hi: 2)),
                         resolution: 65, truncation: .uniform(5)), refine: false) {
        landscapes.append(("the spike", Scene(bundle: spike, preset: spike.manifest.presets[0]), spikeHeight))
    }

    Check.suite("composite: a sweep of parts over landscapes meets the terrain on both surfaces, and misses nothing") {
        let contract = Fixtures.dir.appendingPathComponent("contract")
        let torusBundle = try SurfaceBundle.read(at: contract.appendingPathComponent("surface_torus.kurven"))
        let torus = torusBundle.scene(preset: torusBundle.manifest.presets[0])
        let cube = try PolyhedronBundle.read(at: contract.appendingPathComponent("polyhedron_cube.kurven"))
        let frame = try PolyhedronBundle.read(at: contract.appendingPathComponent("polyhedron_frame.kurven"))
        let turnedCube = try turned(cube.scene(preset: cube.manifest.presets[0]).solid!, 0.4, 0.3, 0.7)
        let turnedL = try turned(try lPrism(), 0.6, -0.5, 0.2)
        let solids: [(String, Solid)] = [
            ("the cube turned", turnedCube),
            ("the frame turned", try turned(frame.scene(preset: frame.manifest.presets[0]).solid!, 1.1, 0.2, 0.5)),
            ("the L-prism", try lPrism()), ("the L-prism turned", turnedL),
        ]
        func solid(_ s: Solid) -> Scene { Scene(solid: s, layers: [], camera: camera) }
        // The placements the reviews found wanting, then a seeded sweep.
        var placements: [(String, Int, Scene, Placement)] = [
            ("the L-prism with a face on a grid line", 0, solid(try lPrism()), Placement(scale: 0.6, offset: SIMD3(0.1, -0.2, 0.6))),
            ("the frame with a face on a grid line", 0, frame.scene(preset: frame.manifest.presets[0]),
             Placement(scale: 0.6, offset: SIMD3(0.1, -0.2, 0.6))),
            ("the torus where a crossing left its piece", 0, torus, Placement(scale: 0.4018, offset: SIMD3(0.5374, -0.7111, 0.9796))),
            ("the torus where a crossing was not solved", 0, torus, Placement(scale: 0.5887, offset: SIMD3(-0.2241, 0.5246, 1.2583))),
            ("the torus grazing the terrain", 0, torus, Placement(scale: 0.5, offset: SIMD3(0.3, -0.2, 0.7))),
            ("the torus over the terrain's edge", 0, torus, Placement(scale: 0.35, offset: SIMD3(2.0, 0.1, 0.9))),
            ("the torus far over the terrain's edge", 0, torus, Placement(scale: 0.6, offset: SIMD3(2.3, 0.1, 0.9))),
            ("the torus over the steps", 1, torus, Placement(scale: 0.35, offset: SIMD3(0, 0, 0.8))),
            ("the torus over the sample not a number", 2, torus, Placement(scale: 0.35, offset: SIMD3(0, 0, 0.8))),
        ]
        for (land, z) in [(3, 0.3), (3, 0.45), (4, 0.3), (4, 0.45)] {
            placements.append(("the cube turned, on a slab of no thickness", land, solid(turnedCube), Placement(scale: 0.4, offset: SIMD3(0.1, 0.1, z))))
            placements.append(("the L-prism turned, on a slab of no thickness", land, solid(turnedL), Placement(scale: 0.4, offset: SIMD3(0.1, 0.1, z))))
        }
        if landscapes.count > 5 {
            for (x, y) in [(1.5, 1.5), (1.0, -1.2), (0.6, 0.5), (0.3, 0.2)] {
                placements.append(("the cube turned, where the spike is a rounding over its floor", 5, solid(turnedCube),
                                   Placement(scale: 0.2, offset: SIMD3(x, y, -0.15))))
            }
        }
        var random = SweepRandom(state: 99)
        for k in 0..<180 {
            let land = k % landscapes.count, h = landscapes[land].height
            let x = random.range(-1.2, 1.2), y = random.range(-1.2, 1.2)
            if k % 2 == 0 {
                let scale = random.range(0.2, 0.6)
                placements.append(("a torus", land, torus, Placement(scale: scale, offset: SIMD3(x, y, h(x, y) + random.range(-0.3, 0.3)))))
            } else {
                let (name, s) = solids[(k / 2) % solids.count]
                let scale = random.range(0.3, 0.9)
                placements.append((name, land, solid(s), Placement(scale: scale, offset: SIMD3(x, y, h(x, y) - random.range(-0.2, 0.5) * scale))))
            }
        }
        let clock = ContinuousClock()
        var met = 0, points = 0, terrain = 0.0, onPart = 0.0, crossings = 0, missed = 0, loose = 0
        var farthest = 0.0, example = "", polyFarthest = 0.0, polyExample = "", strayExample = ""
        let elapsed = try clock.measure {
            for (k, (name, land, scene, placement)) in placements.enumerated() {
                let composite = try Composite([("landscape", landscapes[land].scene, .identity), ("part", scene, placement)],
                                              camera: camera)
                guard !composite.contact(of: 1).isEmpty else { continue }
                met += 1
                guard case .heightfield(let h) = composite.parts[0].scene.geometry else { continue }
                let cell = landscapeCell(h), tolerance = 1e-9 * cell
                let stray = contactStray(composite, part: 1)
                points += stray.points
                // Each against a billionth of its landscape's cell.
                if stray.terrain / tolerance > terrain || stray.part / tolerance > onPart {
                    strayExample = "\(name) #\(k) on \(landscapes[land].name)"
                }
                terrain = max(terrain, stray.terrain / tolerance); onPart = max(onPart, stray.part / tolerance)
                let miss = contactMissed(composite, part: 1, steps: 240)
                crossings += miss.crossings
                if case .parametric(let surface) = scene.geometry {
                    // The solve's least step: a sixty-fourth of a square of
                    // a quarter of a cell, at most an eighth of the tube.
                    let least = min(cell / 4, torusRadii(surface).r * placement.scale / 8) / 64
                    if miss.farthest / least > farthest {
                        farthest = miss.farthest / least
                        example = String(format: "%@ #%d on %@ (scale %.4f at (%.4f, %.4f, %.4f)), the crossing at (%.5f, %.5f, %.5f)",
                                         name, k, landscapes[land].name, placement.scale, placement.offset.x, placement.offset.y,
                                         placement.offset.z, miss.at.x, miss.at.y, miss.at.z)
                    }
                } else {
                    polyFarthest = max(polyFarthest, miss.farthest / tolerance)
                    if miss.farthest > tolerance {
                        missed += 1
                        if polyExample.isEmpty { polyExample = "\(name) #\(k) on \(landscapes[land].name), at \(miss.at)" }
                    }
                }
                loose += contactLooseEnds(composite, part: 1)
            }
        }
        Check.expect(met > 150 && terrain <= 1 && onPart <= 1,
                     "every vertex and chord middle of \(met) parts' contact lines lies on the terrain and on the part",
                     String(format: "%d points, off the terrain by %.2g and off the part by %.2g of a billionth of the cell, %@%@",
                            points, terrain, onPart, "\(elapsed)", strayExample.isEmpty ? "" : ", the worst " + strayExample))
        Check.expect(missed == 0 && crossings > 100_000,
                     "a polyhedron's contact line passes through every crossing found apart from it",
                     String(format: "%d crossings, the farthest %.2g of a billionth of the cell off it%@", crossings, polyFarthest,
                            polyExample.isEmpty ? "" : ", e.g. " + polyExample))
        // A chord is drawn where Newton's step from each of its quarters to
        // the curve is within half the least step, a first-order estimate:
        // the curve is held within the least step of it.
        Check.expect(farthest <= 1,
                     "the torus's contact line passes within the solve's least step of every crossing found apart from it",
                     String(format: "the farthest %.2f of the least step off it, at %@", farthest, example))
        Check.expect(loose == 0, "the torus's contact line ends only on a step of the terrain", "\(loose) ends elsewhere")
    }

    Check.suite("composite: a torus just touching a level terrain draws nothing, or the whole of its circles") {
        // R 2, r 0.75, at scale 0.4, its crown a little over a flat terrain
        // or a plateau at 0.5, or its bottom a little into one: within a
        // billionth of the cell (2.5e-10) nothing; past it, two closed
        // circles about the axis, ring ± √(tube² − (h − z₀)²).
        let surface = ParametricSurface.torus(major: 2, minor: 0.75, samples: (u: 64, v: 32))
        let scene = Scene(surface: surface, layers: [], camera: camera, depthTolerance: 0)
        let s = 0.4, r = 0.75 * s, ring = 2 * s, tolerance = 1e-9 * 0.25
        var wrong: [String] = [], rungs = 0
        for (what, land) in [("a flat terrain", landscapes[3].scene), ("a plateau", landscapes[4].scene)] {
            for (end, sign) in [("crown", 1.0), ("bottom", -1.0)] {
                for exponent in -13 ... -6 {
                    let eps = pow(10, Double(exponent))
                    // The crown eps over 0.5, or the bottom eps under it.
                    let z = 0.5 - sign * r + sign * eps
                    let composite = try Composite([("landscape", land, .identity),
                                                   ("torus", scene, Placement(scale: s, offset: SIMD3(0.13, 0.07, z)))], camera: camera)
                    let paths = composite.contact(of: 1).paths.map { $0.map { s * $0.v + SIMD3(0.13, 0.07, z) } }
                    rungs += 1
                    if eps <= tolerance {
                        if !paths.isEmpty { wrong.append("\(what), \(end) \(eps): \(paths.count) paths") }
                        continue
                    }
                    // Each path closed, on one circle all round, and its chords
                    // as long as the circle's but for their sagitta.
                    let half = (r * r - (r - eps) * (r - eps)).squareRoot()
                    let axis = SIMD2(0.13, 0.07)
                    var good = paths.count == 2
                    for (path, rho) in zip(paths.sorted { simd_length(SIMD2($0[0].x, $0[0].y) - axis) < simd_length(SIMD2($1[0].x, $1[0].y) - axis) },
                                           [ring - half, ring + half]) {
                        let closed = simd_distance(path.first!, path.last!) <= 1e-12
                        let onCircle = path.allSatisfy { abs(simd_length(SIMD2($0.x, $0.y) - axis) - rho) <= 1e-12 && abs($0.z - 0.5) <= 1e-12 }
                        let length = zip(path, path.dropFirst()).reduce(0) { $0 + simd_distance($1.0, $1.1) }
                        good = good && closed && onCircle && abs(length - 2 * .pi * rho) <= 1e-3 * 2 * .pi * rho
                    }
                    if !good { wrong.append(String(format: "%@, %@ %.0e: %d paths", what, end, eps, paths.count)) }
                }
            }
        }
        Check.expect(wrong.isEmpty, "from 1e-13 to 1e-6 either side of touching, each rung draws nothing or both circles whole",
                     "\(rungs) rungs\(wrong.isEmpty ? "" : ", e.g. " + wrong.prefix(3).joined(separator: "; "))")
    }

    Check.suite("composite: a torus's contact costs with the curve's length, not the area under it") {
        let clock = ContinuousClock()
        var fine: [Int: Scene] = [:]
        for n in [257, 1025] { fine[n] = gridLandscape(n: n, camera: camera, f: f) }
        let plateau = gridLandscape(n: 1025, truncation: .uniform(0.5), camera: camera) { _, _ in 2 }
        let torus = try SurfaceBundle.read(at: Fixtures.dir.appendingPathComponent("contract/surface_torus.kurven"))
        let fixture = torus.scene(preset: torus.manifest.presets[0])
        var cases: [(String, Scene, Scene, Placement)] = []
        for r in [1e-3, 1e-4] {
            cases.append(("a torus of R 2 and r \(r) at scale 0.5", landscapes[0].scene,
                          Scene(surface: .torus(major: 2, minor: r, samples: (u: 64, v: 32)), layers: [], camera: camera, depthTolerance: 0),
                          Placement(scale: 0.5, offset: SIMD3(0.1, 0.05, 1.0))))
        }
        for n in [257, 1025] {
            cases.append(("the torus at scale 1e3, its outer wall across a grid of \(n)", fine[n]!, fixture,
                          Placement(scale: 1e3, offset: SIMD3(-2.75e3, 0, 1))))
            cases.append(("the torus at scale 1e3, its top through a grid of \(n)", fine[n]!, fixture,
                          Placement(scale: 1e3, offset: SIMD3(-2e3, 0, 1 - 750))))
        }
        cases.append(("the torus at scale 1e3 grazing a plateau on a grid of 1025", plateau, fixture,
                      Placement(scale: 1e3, offset: SIMD3(-2e3, 0, 0.5 - 750 + 1e-6))))
        for (what, land, part, placement) in cases {
            let composite = try Composite([("landscape", land, .identity), ("torus", part, placement)], camera: camera)
            var count = 0
            let elapsed = clock.measure { count = composite.contact(of: 1).paths.count }
            let stray = contactStray(composite, part: 1)
            guard case .heightfield(let h) = land.geometry else { continue }
            let tolerance = 1e-9 * landscapeCell(h)
            Check.expect(count > 0 && elapsed < .seconds(10) && stray.terrain <= tolerance,
                         "\(what) meets the terrain in reasonable time",
                         String(format: "%d paths, off the terrain by %.1e, %@", count, stray.terrain, "\(elapsed)"))
        }
    }

    Check.suite("composite: a face flush with the terrain draws the same at a rounding either side") {
        let flat = gridLandscape(camera: camera) { _, _ in 0.5 }
        let truncated = gridLandscape(truncation: .uniform(0.5), camera: camera) { _, _ in 2 }
        let prism = Scene(solid: try lPrism(), layers: [], camera: camera)
        func drawn(_ land: Scene, _ z: Double) throws -> [[SIMD3<Double>]] {
            try Composite([("landscape", land, .identity), ("part", prism, Placement(scale: 0.5, offset: SIMD3(0.1, 0.1, z)))],
                          camera: camera).contact(of: 1).paths.map { $0.map(\.v) }
        }
        for (what, land) in [("on a flat terrain", flat), ("on a plateau", truncated)] {
            let flush = try drawn(land, 0.5), over = try drawn(land, 0.5 + 1e-15), under = try drawn(land, 0.5 - 1e-15)
            Check.expect(flush.isEmpty && over.isEmpty && under.isEmpty,
                         "an L-prism standing \(what), flush and a rounding either way, meets it only along its edges, and draws no contact",
                         "\(flush.count), \(over.count) and \(under.count) paths")
        }
        let sunk = try drawn(flat, 0.4)
        let length = sunk.reduce(0) { $0 + zip($1, $1.dropFirst()).reduce(0) { $0 + simd_distance($1.0, $1.1) } }
        Check.expect(abs(length - 8) <= 1e-12, "sunk a little, its sides meet the terrain all round the L",
                     "length \(length) in its own units, of 8")
    }
}
