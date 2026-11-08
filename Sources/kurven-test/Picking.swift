import Foundation
import simd
import KurvenCore
import KurvenDynamics
import KurvenMetal

// MARK: - picking a face
//
// `face(at:)` asked without a window, as navigation is: a point sampled on a
// face, from its parameter plane through the face's own map and its copy's
// tile, returns that face and that copy; a point on a crease returns the face
// the tie rule names; a point taken to a pixel and back, through the camera
// mapping a click reads the depth buffer with, returns the same. Where the
// faces come from is the faces suite's: the contract fixtures and its
// landscapes, banded, notched, tiled, on f; the contract surfaces and
// polyhedra, the torus and the frame among them.

/// A sampled point: where it is, and the face and copy it was taken from.
private struct Sample {
    let world: P3<WorldSpace>
    let face: FaceID
    let copy: Int
}

/// The faces and copies a point lies on, by their own maps and trimmed
/// domains: each whose surface passes within `tie` of it over the point's
/// foot, where the copy's domain holds the foot or a point `tie` from it.
/// The tie rule's choice among them is the first, in the faces' order, then
/// the tiles'.
private func lyingOn(_ w: P3<WorldSpace>, _ faces: [Face], tie: Double) -> [(face: FaceID, copy: Int)] {
    faces.flatMap { face in
        face.copies.enumerated().compactMap { k, copy -> (FaceID, Int)? in
            guard let back = copy.tile.inverse else { return nil }
            let q = face.foot(back(w).v)
            guard copy.holds(q, within: tie), simd_length(copy.place(face.map(q)).position - w.v) <= tie else { return nil }
            return (face.id, k)
        }
    }
}

/// The first of them, by the tie rule.
private func first(_ lying: [(face: FaceID, copy: Int)]) -> (face: FaceID, copy: Int)? {
    lying.min { ($0.face, $0.copy) < ($1.face, $1.copy) }
}

/// Points of one copy of a face: a jittered grid over its parameter
/// rectangle, kept where the copy's trimmed domain holds them, `most` of them at most.
private func samples(of face: Face, copy k: Int, side: Int, most: Int, rng: SplitMix) -> [Sample] {
    let copy = face.copies[k]
    let (u, v) = face.domain
    let (u0, u1) = (min(u.range.lo, u.range.hi), max(u.range.lo, u.range.hi))
    let (v0, v1) = (min(v.range.lo, v.range.hi), max(v.range.lo, v.range.hi))
    var out: [Sample] = []
    for i in 0..<side {
        for j in 0..<side {
            let q = P2<ParamSpace>(u0 + (u1 - u0) * (Double(i) + rng.next(0.05, 0.95)) / Double(side),
                                   v0 + (v1 - v0) * (Double(j) + rng.next(0.05, 0.95)) / Double(side))
            guard copy.contains(q) else { continue }
            out.append(Sample(world: P3(copy.place(face.map(q)).position), face: face.id, copy: k))
        }
    }
    guard out.count > most else { return out }
    return (0..<most).map { out[$0 * out.count / most] }
}

/// What a sampling of one geometry found: points asked, those that came back
/// wrong, and the first of them.
private struct PickTally {
    var asked = 0, wrong = 0
    var example: String?
    mutating func check(_ got: FacePick?, _ face: FaceID, _ copy: Int, _ at: P3<WorldSpace>) {
        asked += 1
        guard got?.face != face || got?.copy != copy else { return }
        wrong += 1
        if example == nil { example = "\(at.v) on \(face) copy \(copy) picked \(got.map { "\($0)" } ?? "nothing")" }
    }
}

/// A navigator looking at a scene as the window first does: the orbit
/// matching `projection`, framed to fit the viewport.
private func navigator(for scene: Scene, projection: ObliqueProjection, viewport: Viewport) -> Navigator {
    var navigator = Navigator(orbit: Orbit(matching: projection),
                              framing: Framing(center: P2(0, 0), unitsPerPixel: 1))
    if let bounds = scene.looking(navigator.camera).quickBounds() {
        navigator.framing = .fitting(bounds, in: viewport)
    }
    return navigator
}

/// The point a click on the pixel under `p` reads: the pixel, and the view
/// depth there as the depth buffer holds it, in single precision.
private func throughAPixel(_ p: P3<WorldSpace>, _ navigator: Navigator, _ viewport: Viewport) -> P3<WorldSpace> {
    let q = navigator.camera.view(p)
    let pixel = navigator.framing.pixel(of: q.xy, in: viewport)
    return navigator.worldPoint(atPixel: pixel, depth: Double(Float(q.z)), in: viewport)
}

/// Where along `a` to `b` a predicate stops holding, to the last bit: it
/// holds at `a` and not at `b`.
private func edge(_ a: Double, _ b: Double, _ holds: (Double) -> Bool) -> Double {
    var (lo, hi) = (a, b)
    for _ in 0..<200 {
        let m = 0.5 * (lo + hi)
        guard m != lo, m != hi else { break }
        if holds(m) { lo = m } else { hi = m }
    }
    return lo
}

func pickTests() {
    let viewport = Viewport(width: 1200, height: 800)
    let projection = ObliqueProjection(shear: 0.5, xAngle: -55, zAngle: -90, flipX: true, yScale: nil)
    let landscapes: [(name: String, heightfield: Heightfield, faces: [Face])]
    do { landscapes = try faceSuiteLandscapes() } catch {
        Check.expect(false, "pick: the faces suite's landscapes load", "\(error)")
        return
    }

    Check.suite("pick: a point sampled on any face of a landscape returns that face, on the tile it was sampled on") {
        let rng = SplitMix(seed: 81)
        var total = PickTally()
        var kinds: [FaceKind: Int] = [:]
        var tileBoundaryRisers = 0, laterCopies = 0, emptyCopies = 0, overlapping = 0
        var everyCopyOfTheGraph = true
        for l in landscapes {
            let geometry = SceneGeometry.heightfield(l.heightfield)
            let tolerance = geometry.scale
            var tally = PickTally()
            for face in l.faces {
                for k in face.copies.indices {
                    let points = samples(of: face, copy: k, side: 48, most: 120, rng: rng)
                    if points.isEmpty {
                        emptyCopies += 1
                        if face.id == .graph, l.name.contains("uniform_mesh") { everyCopyOfTheGraph = false }
                        continue
                    }
                    kinds[face.id.kind, default: 0] += 1
                    if case .tileBoundaryRiser = face.id { tileBoundaryRisers += 1 }
                    if k > 0 { laterCopies += 1 }
                    for s in points {
                        // Where copies overlap, as the contract's two tiles do,
                        // a point of a plateau lies on both, and is the first's.
                        let lying = lyingOn(s.world, l.faces, tie: 1e-9 * geometry.scale)
                        guard lying.contains(where: { $0.face == s.face && $0.copy == s.copy }), let expected = first(lying) else {
                            tally.check(nil, s.face, s.copy, s.world)
                            continue
                        }
                        if expected.copy != s.copy { overlapping += 1 }
                        tally.check(geometry.face(at: s.world, within: tolerance, faces: l.faces), expected.face, expected.copy, s.world)
                    }
                }
            }
            if tally.wrong > 0 { print("        \(l.name): \(tally.wrong) of \(tally.asked); \(tally.example ?? "")") }
            total.asked += tally.asked; total.wrong += tally.wrong
            total.example = total.example ?? tally.example.map { "\(l.name): \($0)" }
        }
        Check.expect(total.wrong == 0, "every sampled point returns its face and copy, on \(landscapes.count) landscapes",
                     "\(total.wrong) of \(total.asked) wrong\(total.example.map { "; first: \($0)" } ?? "")")
        Check.expect([FaceKind.graph, .plateau, .riser, .wall].allSatisfy { kinds[$0, default: 0] > 0 }
                        && tileBoundaryRisers > 0 && laterCopies > 0 && everyCopyOfTheGraph,
                     "the graph, plateaus, risers at band and tile boundaries and walls were sampled, on every tile",
                     "copies sampled: \(kinds.sorted { "\($0.key)" < "\($1.key)" }.map { "\($0.key) \($0.value)" }.joined(separator: ", ")); "
                        + "\(tileBoundaryRisers) at tile boundaries, \(laterCopies) past the first tile, \(emptyCopies) empty; "
                        + "\(overlapping) points on an earlier copy too")
    }

    Check.suite("pick: a point on a crease belongs to the face listed first of the two it bounds") {
        let rng = SplitMix(seed: 82)
        var tops = PickTally(), bottoms = PickTally(), edges = PickTally(), tileBoundaries = PickTally()
        var misplaced: [String] = []
        for l in landscapes {
            let geometry = SceneGeometry.heightfield(l.heightfield)
            let tolerance = geometry.scale
            let cell = geometry.scale, tie = 1e-9 * cell
            func picked(_ w: P3<WorldSpace>) -> FacePick? { geometry.face(at: w, within: tolerance, faces: l.faces) }
            /// The tie rule's face at a crease point that lies on `face`'s
            /// copy, with the kinds it must be of.
            func expect(_ w: P3<WorldSpace>, on face: FaceID, _ copy: Int, kinds: Set<FaceKind>, _ tally: inout PickTally) {
                let lying = lyingOn(w, l.faces, tie: tie)
                guard lying.contains(where: { $0.face == face && $0.copy == copy }), let expected = first(lying),
                      kinds.contains(expected.face.kind) else {
                    misplaced.append("\(l.name): \(w.v) on \(face) copy \(copy) lies on \(lying.map { "\($0.face) \($0.copy)" })")
                    return
                }
                tally.check(picked(w), expected.face, expected.copy, w)
            }
            for face in l.faces {
                switch face.id.kind {
                case .wall, .riser:
                    // Up each vertical face at stations along it: its top,
                    // found by bisection on its own trimmed domain, lies on
                    // the face above, graph or plateau, which comes first; a
                    // riser's bottom on the face below.
                    for (k, copy) in face.copies.enumerated() {
                        let (u, v) = face.domain
                        for _ in 0..<12 {
                            let s = rng.next(min(u.range.lo, u.range.hi), max(u.range.lo, u.range.hi))
                            let z0 = v.range.lo, ceiling = v.range.hi + 1
                            guard copy.contains(P2(s, z0 + 1e-6 * cell)), !copy.contains(P2(s, ceiling)) else { continue }
                            let top = edge(z0 + 1e-6 * cell, ceiling) { copy.contains(P2(s, $0)) }
                            expect(P3(copy.place(face.map(P2(s, top))).position), on: face.id, k, kinds: [.graph, .plateau], &tops)
                            guard face.id.kind == .riser else { continue }
                            // A riser's bottom: the lesser height at a band
                            // boundary, the lower copy's top at a tile boundary.
                            let bottom = edge(top, z0 - 1) { copy.contains(P2(s, $0)) }
                            expect(P3(copy.place(face.map(P2(s, bottom))).position), on: face.id, k, kinds: [.graph, .plateau], &bottoms)
                        }
                    }
                case .graph:
                    // Across the truncation edge: from a point of the graph
                    // toward a point of a plateau on the same copy, where the
                    // graph ends at the plateau's height.
                    let plateaus = l.faces.filter { $0.id.kind == .plateau }
                    for (k, copy) in face.copies.enumerated() {
                        guard let back = copy.tile.inverse else { continue }
                        let inGraph = samples(of: face, copy: k, side: 24, most: 20, rng: rng)
                        for plateau in plateaus where k < plateau.copies.count {
                            let onPlateau = samples(of: plateau, copy: k, side: 24, most: 20, rng: rng)
                            for (a, b) in zip(inGraph, onPlateau) {
                                let pa = back(a.world).xy.v, pb = back(b.world).xy.v
                                let t = edge(0, 1) { copy.contains(P2(pa + $0 * (pb - pa))) }
                                let q = P2<ParamSpace>(pa + t * (pb - pa))
                                let h = plateau.map(q).position.z
                                // The truncation edge, not a riser's foot, a
                                // footprint's edge or the domain's.
                                guard abs(face.map(q).position.z - h) <= tie, plateau.copies[k].holds(q, within: tie) else { continue }
                                let w = P3<WorldSpace>(copy.place(SurfaceJet(position: SIMD3(q.x, q.y, h), du: .zero, dv: .zero)).position)
                                expect(w, on: plateau.id, k, kinds: [.graph], &edges)
                            }
                        }
                    }
                    // On the domain's edge of each copy, where another copy
                    // meets it with the same top: the first copy's.
                    guard face.copies.count > 1 else { continue }
                    let (u, v) = face.domain
                    for (k, copy) in face.copies.enumerated() {
                        for _ in 0..<16 {
                            let t = rng.next(0, 1)
                            let q: P2<ParamSpace> = switch Int(rng.next(0, 4)) {
                            case 0: P2(u.range.lo, v.range.lo + t * v.range.length)
                            case 1: P2(u.range.hi, v.range.lo + t * v.range.length)
                            case 2: P2(u.range.lo + t * u.range.length, v.range.lo)
                            default: P2(u.range.lo + t * u.range.length, v.range.hi)
                            }
                            guard copy.contains(q) else { continue }
                            let w = P3<WorldSpace>(copy.place(face.map(q)).position)
                            let graphs = lyingOn(w, l.faces, tie: tie).filter { $0.face == .graph }
                            guard graphs.count > 1 else { continue }
                            expect(w, on: .graph, k, kinds: [.graph], &tileBoundaries)
                        }
                    }
                default:
                    break
                }
            }
        }
        Check.expect(misplaced.isEmpty, "every crease point lies on the face it was found on, and the tie rule's face is of the kind it bounds",
                     "\(misplaced.count); first \(misplaced.first ?? "")")
        for (name, tally) in [("the top of a wall or a riser is the graph's or the plateau's above it", tops),
                              ("the bottom of a riser is the plateau's or the graph's below it", bottoms),
                              ("the truncation edge is the graph's", edges),
                              ("where two copies meet with the same top, the graph is the first copy's", tileBoundaries)] {
            Check.expect(tally.asked > 0 && tally.wrong == 0, name,
                         "\(tally.wrong) of \(tally.asked) wrong\(tally.example.map { "; first: \($0)" } ?? "")")
        }
    }

    // The contract's surfaces and polyhedra.
    var surfaces: [(String, Scene)] = []
    var polyhedra: [(String, Scene)] = []
    for url in Fixtures.contractBundles {
        if (try? SurfaceBundle.holdsSurface(at: url)) == true, let b = try? SurfaceBundle.read(at: url) {
            surfaces.append((url.lastPathComponent, b.scene(preset: b.manifest.presets[0])))
        } else if (try? PolyhedronBundle.holdsPolyhedron(at: url)) == true, let b = try? PolyhedronBundle.read(at: url) {
            polyhedra.append((url.lastPathComponent, b.scene(preset: b.manifest.presets[0])))
        }
    }

    Check.suite("pick: a point on a parametric surface returns its one face, at the point's own parameter") {
        let rng = SplitMix(seed: 83)
        Check.expect(surfaces.contains { $0.0 == "surface_torus.kurven" }, "the torus fixture is among them")
        for (name, scene) in surfaces {
            guard let surface = scene.parametric else { continue }
            let geometry = scene.geometry
            let scale = geometry.scale
            var wrong = 0, off = 0, far = 0, worst = 0.0
            for _ in 0..<200 {
                let c = P2<ParamSpace>(rng.next(surface.u.range.lo, surface.u.range.hi),
                                       rng.next(surface.v.range.lo, surface.v.range.hi))
                let jet = surface.map(c)
                let p = P3<WorldSpace>(jet.position)
                guard let pick = geometry.face(at: p, within: 1e-6 * scale), pick.face == .surface else { wrong += 1; continue }
                // The parameter found is a point the map takes to p.
                let d = simd_length(surface.map(pick.parameter).position - jet.position)
                worst = max(worst, d)
                if d > 1e-9 * scale { off += 1 }
                // A point a tenth of the grid's spacing off it, along its
                // normal, is on it within a fifth and on nothing within a
                // twentieth.
                let n = simd_normalize(jet.normal)
                let lifted = P3<WorldSpace>(jet.position + 0.1 * scale * n)
                if geometry.face(at: lifted, within: 0.2 * scale)?.face != .surface
                    || geometry.face(at: lifted, within: 0.05 * scale) != nil { far += 1 }
            }
            Check.expect(wrong == 0 && off == 0 && far == 0, "\(name): every point returns the surface",
                         "\(wrong) wrong, \(off) at another point (worst \(worst)), \(far) off it misread")
        }
    }

    Check.suite("pick: a point on a polyhedron's face returns that face; on an edge or a corner, the face of least index") {
        let rng = SplitMix(seed: 84)
        Check.expect(polyhedra.contains { $0.0 == "polyhedron_frame.kurven" }, "the frame fixture is among them")
        for (name, scene) in polyhedra {
            guard let solid = scene.solid else { continue }
            let geometry = scene.geometry, faces = geometry.faces
            let tolerance = 1e-6 * geometry.scale
            var tally = PickTally(), creases = PickTally()
            for face in faces {
                for s in samples(of: face, copy: 0, side: 24, most: 80, rng: rng) {
                    tally.check(geometry.face(at: s.world, within: tolerance, faces: faces), s.face, 0, s.world)
                }
            }
            // Along each edge, and at each vertex: the faces meeting there.
            for (k, e) in solid.edges.enumerated() where !solid.isDiagonal[k] {
                let first = min(solid.faceIndex(ofPiece: e.left), solid.faceIndex(ofPiece: e.right))
                for t in [rng.next(0, 1), rng.next(0, 1), 0.5] {
                    let w = P3<WorldSpace>(solid.vertices[e.a].v + t * (solid.vertices[e.b].v - solid.vertices[e.a].v))
                    creases.check(geometry.face(at: w, within: tolerance, faces: faces), .solid(face: first), 0, w)
                }
            }
            for (i, vertex) in solid.vertices.enumerated() {
                let meeting = solid.loops.indices.filter { f in solid.loops[f].contains { $0.contains(i) } }
                guard let first = meeting.min() else { continue }
                creases.check(geometry.face(at: vertex, within: tolerance, faces: faces), .solid(face: first), 0, vertex)
            }
            Check.expect(tally.asked > 0 && tally.wrong == 0, "\(name): every point of a face returns it",
                         "\(tally.wrong) of \(tally.asked) wrong\(tally.example.map { "; first: \($0)" } ?? "")")
            Check.expect(creases.wrong == 0, "\(name): every point of an edge or corner returns the face of least index",
                         "\(creases.wrong) of \(creases.asked) wrong\(creases.example.map { "; first: \($0)" } ?? "")")
        }
    }

    Check.suite("pick: a point taken to its pixel and back through the camera returns the same face") {
        let rng = SplitMix(seed: 85)
        var tally = PickTally()
        var asked = 0
        func run(_ scene: Scene, faces: [Face], copies: Bool) {
            let n = navigator(for: scene, projection: projection, viewport: viewport)
            let geometry = scene.geometry
            let upp = n.framing.unitsPerPixel
            for face in faces {
                for k in (copies ? Array(face.copies.indices) : [0]) {
                    for s in samples(of: face, copy: k, side: 16, most: 12, rng: rng) {
                        asked += 1
                        let expected = first(lyingOn(s.world, faces, tie: 1e-9 * geometry.scale)) ?? (s.face, s.copy)
                        let w = throughAPixel(s.world, n, viewport)
                        tally.check(geometry.face(at: w, within: geometry.pickTolerance(at: w, unitsPerPixel: upp), faces: faces),
                                    expected.face, expected.copy, s.world)
                    }
                }
            }
        }
        for l in landscapes {
            let scene = Scene(surface: l.heightfield.surface, occluder: .empty, tiles: l.heightfield.tiles,
                              region: l.heightfield.region, step: 1, layers: [], camera: .oblique(projection),
                              depthTolerance: 0, refine: l.heightfield.refine, walls: l.heightfield.walls)
            run(scene, faces: l.faces, copies: true)
        }
        for (_, scene) in surfaces + polyhedra { run(scene, faces: scene.geometry.faces, copies: false) }
        Check.expect(asked > 0 && tally.wrong == 0, "every face of every fixture, through its pixel",
                     "\(tally.wrong) of \(tally.asked) wrong\(tally.example.map { "; first: \($0)" } ?? "")")
    }

    Check.suite("pick: a picked face's boundary runs round its trimmed domain, closed") {
        let rng = SplitMix(seed: 86)
        var faced = 0, empty: [String] = [], offEdge: [String] = [], open: [String] = []
        var worst: [String: Double] = [:], worstAt: [String: String] = [:], worstSag = 0.0, sags: [String] = []
        var outside: [String] = []
        /// `straddled`, on the grid, says whether a point of space lies, on
        /// some tile, within 1.5 cells of a band boundary.
        func check(_ name: String, _ geometry: SceneGeometry, _ faces: [Face],
                   straddled: ((P3<WorldSpace>) -> Bool)? = nil) {
            let scale = geometry.scale
            for face in faces {
                for (k, copy) in face.copies.enumerated() {
                    guard let s = samples(of: face, copy: k, side: 24, most: 1, rng: rng).first,
                          let pick = geometry.face(at: s.world, within: 1e-6 * scale, faces: faces),
                          pick.face == face.id, pick.copy == k else { continue }
                    faced += 1
                    let paths = geometry.boundary(of: pick, faces: faces).paths
                    let coords = face.boundary(of: copy)
                    let label = "\(name): \(face.id) copy \(k)"
                    guard !paths.isEmpty else {
                        if straddled?(s.world) != true { empty.append(label) }
                        continue
                    }
                    // How far each vertex is from the edge of the copy's
                    // trimmed domain, read by its own membership: the least
                    // radius at which the domain holds a point of the circle
                    // round it, or the vertex itself, and leaves one out, in
                    // 64 directions. On the grid a banded truncation edge
                    // crosses a cell a band boundary straddles, up to 1.42
                    // cells off the face's step, and those cells are left
                    // out, as the trim suite leaves them.
                    // How far a vertex may stand from the edge of the copy's
                    // trimmed domain, in the scale's units (cells):
                    //  - a polyhedron's loops are its edges, to a billionth;
                    //  - on f, a vertex where the footprint or a band
                    //    boundary cuts a chord of the truncation edge lies on
                    //    the chord, which the refiner holds within its
                    //    tolerance, 0.02 cells, of |f| = h at its midpoint;
                    //    a quarter more for the rest of the chord;
                    //  - on the grid, membership reads the grid bilinearly,
                    //    and its level set in a cell stands off the chord
                    //    across it by up to half the cell's diagonal, beside
                    //    a saddle;
                    //  - a wall's top edge on the grid is solved linearly
                    //    between the wall's own stations, which the bundle
                    //    can space wider than a cell, so it stands off the
                    //    face's exact top by up to a station's spacing: the
                    //    drawn top edge's, which the boundary follows.
                    let kind = face.id.kind == .solid ? "solid" : straddled == nil ? "on f"
                        : face.id.kind == .wall ? "a wall on the grid" : "on the grid"
                    let allowed = kind == "solid" ? 1e-9 : kind == "on f" ? 0.025
                        : kind == "on the grid" ? 0.5.squareRoot() : max(0.5.squareRoot(), face.domain.u.spacing / scale)
                    if face.id != .surface {
                        func meets(_ q: P2<ParamSpace>, _ r: Double) -> Bool {
                            let around = [q] + (0..<64).map { k in
                                P2<ParamSpace>(q.v + r * scale * SIMD2(cos(Double(k) * .pi / 32), sin(Double(k) * .pi / 32)))
                            }
                            return around.contains(where: copy.contains) && around.contains { !copy.contains($0) }
                        }
                        let radii = [0, 1e-9, 1e-6, 1e-4, 1e-3, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
                        var far = 0.0, at = SIMD2<Double>.zero
                        for q in coords.joined() {
                            if let straddled, straddled(P3(copy.place(face.map(q)).position)) { continue }
                            guard let k = radii.indices.dropFirst().first(where: { meets(q, radii[$0]) }) else {
                                far = .infinity; at = q.v; continue
                            }
                            // Within the bin, to a hundredth of it.
                            var (lo, hi) = (radii[k - 1], radii[k])
                            if hi > far {
                                for _ in 0..<7 { let m = 0.5 * (lo + hi); if meets(q, m) { hi = m } else { lo = m } }
                            }
                            if hi > far { far = hi; at = q.v }
                        }
                        if far > worst[kind] ?? 0 { worst[kind] = far; worstAt[kind] = "\(label) at \(at)" }
                        if far > allowed { offEdge.append("\(label): \(far) cells at \(at)") }
                    }
                    // Every segment in space lies on the copy, at its
                    // quarters, within a fiftieth of the scale, over a foot
                    // its trimmed domain holds, within what a vertex is
                    // allowed: the curve follows a curved face between its
                    // vertices, and does not leave it.
                    if face.id != .surface, let back = copy.tile.inverse {
                        let r = allowed * scale
                        let leaves = paths.contains { path in
                            zip(path, path.dropFirst()).contains { a, b in
                                [0.25, 0.5, 0.75].contains { t in
                                    let w = P3<WorldSpace>(a.v + t * (b.v - a.v))
                                    if let straddled, straddled(w) { return false }
                                    let q = face.foot(back(w).v)
                                    // Anywhere in the disc of radius r round it.
                                    return !copy.contains(q) && ![0.125, 0.25, 0.5, 0.75, 1].contains { m in
                                        (0..<64).contains { k in
                                            copy.contains(P2(q.v + m * r * SIMD2(cos(Double(k) * .pi / 32), sin(Double(k) * .pi / 32))))
                                        }
                                    }
                                }
                            }
                        }
                        if leaves { outside.append(label) }
                    }
                    let sagging = paths.lazy.flatMap { path in zip(path, path.dropFirst()) }.flatMap { a, b in
                        [0.25, 0.5, 0.75].map { t -> Double in
                            // Off the surface the copy lies on, along its
                            // normal at the point's foot, whether or not the
                            // trimmed domain holds it.
                            let m = a.v + t * (b.v - a.v)
                            guard let back = copy.tile.inverse else { return .infinity }
                            let jet = copy.place(face.map(face.foot(back(P3(m)).v)))
                            let n = jet.normal
                            return simd_length(n) > 0 ? abs(simd_dot(m - jet.position, n)) / simd_length(n)
                                : simd_length(m - jet.position)
                        }
                    }.max() ?? 0
                    worstSag = max(worstSag, sagging / scale)
                    if sagging > 0.02 * scale { sags.append("\(label): \(sagging / scale)") }
                    // Every end of a curve meets another's, in space.
                    let size = max(scale, 1e-12), near = 1e-6 * size
                    let ends = paths.filter { simd_length($0[0].v - $0[$0.count - 1].v) > near }
                        .flatMap { [$0[0].v, $0[$0.count - 1].v] }
                    let loose = ends.indices.filter { i in
                        !ends.indices.contains { $0 != i && simd_length(ends[$0] - ends[i]) <= near }
                    }
                    if !loose.isEmpty { open.append("\(label): \(loose.count) loose ends, first \(ends[loose[0]])") }
                }
            }
        }
        for l in landscapes {
            let h = l.heightfield, cell = SceneGeometry.heightfield(h).scale
            let real = h.surface.domain.real, imag = h.surface.domain.imag
            let bounds = h.surface.truncation.bands(from: min(real.lo, real.hi), to: max(real.lo, real.hi)).map(\.hi)
            let straddled: ((P3<WorldSpace>) -> Bool)? = h.refine?.magnitude != nil ? nil : { w in
                h.tiles.contains { tile in
                    guard let back = tile.inverse else { return false }
                    let q = back(w)
                    return q.x >= min(real.lo, real.hi) - cell && q.x <= max(real.lo, real.hi) + cell
                        && q.y >= min(imag.lo, imag.hi) - cell && q.y <= max(imag.lo, imag.hi) + cell
                        && bounds.contains { abs(q.x - $0) <= 1.5 * cell }
                }
            }
            check(l.name, .heightfield(h), l.faces, straddled: straddled)
        }
        for (name, scene) in surfaces + polyhedra { check(name, scene.geometry, scene.geometry.faces) }
        // |f| = 1 + sin²(2πx)/2 on the unit square: along the domain's edge
        // the graph's boundary has a vertex only at the corners, and the
        // edge's midpoint lies on the chord between them, so halving alone
        // would draw the chord.
        let unit = Domain(real: Interval(lo: 0, hi: 1), imag: Interval(lo: 0, hi: 1))
        let wavy = Surface(height: Grid2D(width: 41, height: 41, domain: unit, values: (0..<1681).map { k in
            Float(1 + 0.5 * pow(sin(2 * Double.pi * Double(k % 41) / 40), 2))
        }), phase: nil, truncation: .uniform(10))
        let flat = Heightfield(surface: wavy, occluder: .empty, tiles: [.identity], step: 1, walls: .none)
        check("a wave whose midpoint is on the chord", .heightfield(flat), wavy.faces())
        print("        \(faced) faces picked; \(open.count) with loose ends; farthest vertex from the edge, in cells:")
        for (kind, d) in worst.sorted(by: { $0.key < $1.key }) { print("          \(kind): \(d), \(worstAt[kind] ?? "")") }
        for line in offEdge.prefix(12) { print("        \(line)") }
        for line in open.prefix(12) { print("        \(line)") }
        Check.expect(faced > 0 && empty.isEmpty, "every picked face has a boundary", empty.first ?? "")
        Check.expect(offEdge.isEmpty, "every vertex of it lies on the edge of the copy's trimmed domain: a polyhedron's to a billionth, on f within 0.025 cells, on the grid half a cell's diagonal, a wall's on the grid its stations' spacing",
                     "\(offEdge.count) faces stray; first \(offEdge.first ?? "")")
        Check.expect(open.isEmpty, "every boundary closes", "\(open.count) open; first \(open.first ?? "")")
        Check.expect(outside.isEmpty, "every segment of it stands over the copy's trimmed domain, within what a vertex is allowed",
                     "\(outside.count) faces leave it; first \(outside.first ?? "")")
        Check.expect(sags.isEmpty, "every segment of it in space lies on the face, within a fiftieth of the scale",
                     "worst \(worstSag); \(sags.count) faces sag; first \(sags.first ?? "")")
    }
}




func pickOnSolidTests() {
    Check.suite("pick: a picked face's boundary on a polyhedron is seen where its edges are, to the occlusion's precision") {
        for url in Fixtures.contractBundles where (try? PolyhedronBundle.holdsPolyhedron(at: url)) == true {
            let bundle = try PolyhedronBundle.read(at: url)
            let base = bundle.scene(preset: bundle.manifest.presets[0]), solid = bundle.solid
            let geometry = base.geometry, faces = geometry.faces, scale = geometry.scale
            var worst = 0.0, compared = 0
            for (azimuth, elevation) in [(-60.0, 35.0), (25.0, 20.0), (140.0, 55.0)] {
                var orbit = Orbit(matching: bundle.manifest.presets[0].projection)
                orbit.azimuth = Angle(degrees: azimuth); orbit.elevation = Angle(degrees: elevation)
                orbit.style = DrawingConvention(shear: 0, flipX: false, yScale: nil)
                let scene = base.looking(orbit.camera), view = scene.camera.view
                // The visible edges, each a span of one edge of the solid.
                let edges = solid.visibleEdges(view: view)
                let spans = (0..<edges.count).map { Array(edges[path: $0]) }
                for face in faces {
                    let loops = face.boundary(of: face.copies[0]).map { $0.map { face.copies[0].place(face.map($0)).position } }
                    let spec = LayerSpec(name: "selection", role: .scaffold, source: .file(vertices: "", offsets: ""),
                                         width: 1, heightPolicy: .surface)
                    let layer = Layer(spec: spec, paths: PolylineSet(paths: loops.map { $0.map(P3<WorldSpace>.init) }))
                    let seen = scene.visible(layer).first?.1 ?? .empty
                    func length(_ path: [SIMD3<Double>]) -> Double {
                        zip(path, path.dropFirst()).reduce(0) { $0 + simd_length($1.1 - $1.0) }
                    }
                    func onLoops(_ p: SIMD3<Double>) -> Bool {
                        loops.contains { loop in
                            zip(loop, loop.dropFirst()).contains { a, b in
                                let d = b - a, t = max(0, min(1, simd_dot(p - a, d) / simd_length_squared(d)))
                                return simd_length(p - (a + t * d)) <= 1e-9 * scale
                            }
                        }
                    }
                    let seenLength = (0..<seen.count).reduce(0.0) { sum, k in
                        sum + length(seen[path: k].map { view.inverse(P3<ViewSpace>($0.v)).v })
                    }
                    // The face's own edges' visible spans: those along its loops.
                    let edgeLength = spans.filter { span in
                        onLoops(span[0].v) && onLoops(span[span.count - 1].v) && onLoops(0.5 * (span[0].v + span[span.count - 1].v))
                    }.reduce(0.0) { $0 + length($1.map(\.v)) }
                    worst = max(worst, abs(seenLength - edgeLength) / scale)
                    compared += 1
                }
            }
            Check.expect(compared > 0 && worst < 1e-7, "\(url.lastPathComponent): every face's boundary, under three cameras",
                         "worst difference in length \(worst) of the largest face")
        }
    }
}

/// How far a point is from a triangle.
private func distance(_ p: SIMD3<Double>, _ a: SIMD3<Double>, _ b: SIMD3<Double>, _ c: SIMD3<Double>) -> Double {
    let n = simd_cross(b - a, c - a), area = simd_length(n)
    guard area > 0 else { return .infinity }
    let u = n / area, q = p - simd_dot(p - a, u) * u
    let inside = [(a, b), (b, c), (c, a)].allSatisfy { e0, e1 in simd_dot(simd_cross(e1 - e0, q - e0), u) >= 0 }
    if inside { return abs(simd_dot(p - a, u)) }
    return [(a, b), (b, c), (c, a)].map { e0, e1 in
        let d = e1 - e0, t = max(0, min(1, simd_dot(p - e0, d) / simd_length_squared(d)))
        return simd_length(p - (e0 + t * d))
    }.min() ?? .infinity
}

func pickFromDepthTests() {
    // The preview's own depth, picked as a click picks it: at a pixel, the
    // depth the frame left in the texel under it, at the texel's centre.
    // Every pixel the preview draws must pick a face, and the face must face
    // the eye, but for what the depth pass draws that no face is, which picks
    // nothing farther than the four pixels a pick allows any point:
    //  - walls given as a mesh (`uniform_mesh`'s skirt), which describe no
    //    face;
    //  - a wall's curtain where it stands over its face's top: the curtain is
    //    sampled at the wall's stations and runs straight between them, while
    //    the face's top follows the surface;
    //  - `bands_perimeter`'s far wall, seen from inside over the solid's top,
    //    which its footprint drops at the domain's edge.
    // The three are the drawing's, not the pick's, and are counted.
    Check.suite("pick: every pixel the preview draws picks a face, facing the eye, from the depth it left") {
        guard Check.gpu() != nil else { return }
        let renderer = try MetalRenderer()
        let viewport = Viewport(width: 1200, height: 800)
        let target = try renderer.makePreviewTarget(viewport)
        for url in Fixtures.contractBundles {
            let scene: Scene, preset: CameraPreset
            if (try? PolyhedronBundle.holdsPolyhedron(at: url)) == true {
                let b = try PolyhedronBundle.read(at: url); preset = b.manifest.presets[0]; scene = b.scene(preset: preset)
            } else if (try? SurfaceBundle.holdsSurface(at: url)) == true {
                continue
            } else {
                let b = try KurvenBundle.read(at: url)
                guard let first = b.manifest.presets.first else { continue }
                preset = first; scene = Scene(bundle: b, preset: preset)
            }
            let name = url.lastPathComponent
            // As the window first frames it (`Document`).
            var navigator = Navigator(orbit: Orbit(matching: preset.projection),
                                      framing: Framing(center: P2(0, 0), unitsPerPixel: 1))
            if let bounds = scene.looking(navigator.camera).quickBounds() {
                navigator.framing = .fitting(bounds, in: viewport)
            }
            try renderer.renderPreview(scene.looking(navigator.camera), navigator: navigator, viewport: viewport,
                                       options: PreviewOptions(), into: target)
            let geometry = scene.geometry, faces = geometry.faces, scale = geometry.scale
            let toEye = -navigator.camera.view.projectionDirection
            let walls = faces.filter { $0.id.kind == .wall }
            let mesh: Mesh<WorldSpace>? = {
                guard case .heightfield(let h) = geometry, case .mesh = h.walls else { return nil }
                return h.occluder
            }()
            /// On a wall's curtain over the wall face's top.
            func overAWall(_ w: P3<WorldSpace>) -> Bool {
                walls.contains { face in
                    let copy = face.copies[0], q = face.foot(w.v)
                    guard simd_length(copy.place(face.map(q)).position - w.v) <= 1e-4 * scale,
                          q.x >= 0, q.x <= face.domain.u.range.hi, !copy.contains(q) else { return false }
                    return copy.contains(P2(q.x, face.domain.v.range.lo + 1e-9 * scale))
                }
            }
            func onTheMesh(_ w: P3<WorldSpace>) -> Bool {
                guard let mesh else { return false }
                return mesh.triangles.contains { t in
                    distance(w.v, mesh.vertices[Int(t.x)].v, mesh.vertices[Int(t.y)].v, mesh.vertices[Int(t.z)].v) <= 1e-4 * scale
                }
            }
            var covered = 0, excused = 0, turnedAwayExcused = 0
            var empty: [String] = [], away: [String] = [], offFace: [String] = []
            for y in stride(from: 0, to: viewport.height, by: 4) {
                for x in stride(from: 0, to: viewport.width, by: 4) {
                    guard let w = renderer.worldPoint(atPixel: SIMD2(Double(x), Double(y)), navigator: navigator,
                                                      viewport: viewport) else { continue }
                    covered += 1
                    let tolerance = geometry.pickTolerance(at: w, unitsPerPixel: navigator.framing.unitsPerPixel)
                    let pick = geometry.face(at: w, within: tolerance, faces: faces)
                    // What no face is picks nothing.
                    if overAWall(w) || onTheMesh(w) {
                        if let pick, pick.distance > 4 * navigator.framing.unitsPerPixel { offFace.append("\(x),\(y) \(pick)") }
                        else { excused += 1 }
                        continue
                    }
                    guard let pick else {
                        empty.append("\(x),\(y) at \(w.v)")
                        continue
                    }
                    guard let face = faces.first(where: { $0.id == pick.face }) else { continue }
                    let copy = face.copies[pick.copy]
                    if let outward = copy.outward, simd_dot(copy.place(face.map(pick.parameter)).normal * outward, toEye) < 0 {
                        if name == "bands_perimeter.kurven", face.id.kind == .wall { turnedAwayExcused += 1 }
                        else { away.append("\(x),\(y) \(pick)") }
                    }
                }
            }
            Check.expect(covered > 0 && empty.isEmpty && away.isEmpty && offFace.isEmpty,
                         "\(name): every pixel drawn picks a face facing the eye, and none where no face is",
                         "\(covered) drawn, \(empty.count) pick nothing, \(away.count) a face turned away, "
                            + "\(offFace.count) a face where none is" + (offFace.first.map { " (first \($0))" } ?? "") + "; "
                            + "\(excused) on a curtain over its wall or a mesh, picking nothing past 4 px, \(turnedAwayExcused) on the far wall from inside"
                            + (empty.first.map { "; first empty \($0)" } ?? "") + (away.first.map { "; first away \($0)" } ?? ""))
        }
    }
}
