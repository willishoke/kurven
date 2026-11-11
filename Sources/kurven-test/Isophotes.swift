import Foundation
import simd
import KurvenCore
import KurvenLandscape
import KurvenDynamics

/// Isophotes and half-Lambert tone on curved faces (phase 11a), held to
/// closed forms: the round torus lit along z, where n · l = sin v; the graph
/// of exp(z), where n · l = 1 / √(1 + e^{2x}); flat faces, which have none;
/// and the visibility of both kinds of layer, held to the brute-force casts.
func isophoteTests() {
    var request = SurfaceRequest(.torus(major: 2, minor: 1))
    request.grid = 512
    let torus: ParametricSurface
    do {
        torus = try SurfaceDrawing.build(request).surface
    } catch {
        Check.suite("isophotes: the torus builds") { Check.expect(false, "the torus builds", "\(error)") }
        return
    }
    let face = torus.face, copy = torus.face.copies[0]
    let twoPi = 2 * Double.pi
    /// How far apart two angles are, round the circle.
    func apart(_ a: Double, _ b: Double) -> Double {
        let d = (a - b).truncatingRemainder(dividingBy: twoPi)
        return min(abs(d), twoPi - abs(d))
    }
    /// The round torus's outward unit normal at (u, v), in closed form.
    func normal(_ u: Double, _ v: Double) -> SIMD3<Double> { SIMD3(cos(v) * cos(u), cos(v) * sin(u), sin(v)) }

    Check.suite("isophotes: on the round torus lit along z, n · l = sin v, and the isophote at c is v = asin c or π − asin c") {
        // The convention first: u round the axis from 0, v round the tube
        // from the outer equator, both over [0, 2π), the outward normal the
        // closed form's, so n · z = sin v.
        Check.expect(torus.u.range.lo == 0 && torus.v.range.lo == 0 && abs(torus.u.range.length - twoPi) < 1e-15
                         && abs(torus.v.range.length - twoPi) < 1e-15 && torus.outward == 1,
                     "u and v run over [0, 2π), du × dv outward")
        var off = 0.0
        for i in 0..<64 {
            for j in 0..<64 {
                let p = P2<ParamSpace>(twoPi * (Double(i) + 0.37) / 64, twoPi * (Double(j) + 0.61) / 64)
                let n = face.outwardNormal(at: p, on: copy)!
                off = max(off, simd_length(n - normal(p.x, p.y)), abs(simd_dot(n, SIMD3(0, 0, 1)) - sin(p.y)))
            }
        }
        Check.expect(off <= 1e-15, "the outward normal is (cos v cos u, cos v sin u, sin v), so n · z = sin v",
                     String(format: "off by %.1e", off))
        // Lit along z, and along z at another length: the field normalizes
        // the light.
        let levels = [-0.999999, -0.9999, -0.99, -0.9, -0.5, -0.1, 0, 0.3, 0.7, 0.99, 0.9999, 0.999999]
        for light in [SIMD3<Double>(0, 0, 1), SIMD3(0, 0, 2.5)] {
            var worst = 0.0, closed = true, counts: [Int] = [], example = ""
            for (c, contour) in zip(levels, face.isophotes(light: light, levels, on: copy)) {
                precondition(contour.level == c)
                counts.append(contour.paths.count)
                let a = asin(c), targets = [a, Double.pi - a]
                for path in contour.paths {
                    for p in path {
                        let d = targets.map { apart(p.y, $0) }.min()!
                        if d > worst { worst = d; example = String(format: "c = %g at v = %.17g", c, p.y) }
                    }
                    // A whole turn round the axis, ending where it began.
                    let first = path.first!, last = path.last!
                    if abs(abs(last.x - first.x) - twoPi) > 1e-12 || abs(last.y - first.y) > 1e-12 { closed = false }
                }
            }
            Check.expect(worst <= 1e-12, "light \(light): every vertex of the isophote at c lies on v = asin c or π − asin c",
                         String(format: "worst %.1e rad", worst) + (example.isEmpty ? "" : ", " + example))
            Check.expect(counts.allSatisfy { $0 == 2 } && closed,
                         "light \(light): each isophote is the two circles, each a whole turn round the axis",
                         "paths per level \(counts)")
        }
        // At ±1 n · l touches its extreme on a circle, v = ±π/2; past it,
        // nothing. A level there draws nothing, by rule.
        let extremes = face.isophotes(light: SIMD3(0, 0, 1), [-1.5, -1, 1, 1.5], on: copy)
        Check.expect(extremes.count == 0, "at c = ±1 and past it, the isophote draws nothing",
                     "\(extremes.map { "\($0.level): \($0.paths.count)" })")
    }

    Check.suite("isophotes: a flat face has none, at any level, its own constant n · l included") {
        let light = SIMD3<Double>(-0.4, -0.6, 0.7), l = simd_normalize(light)
        var faces = 0, drawn = 0, fields = 0
        var kinds = Set<FaceKind>()
        for name in ["frame", "cube"] {
            let bundle = try PolyhedronBundle.read(at: Fixtures.dir.appendingPathComponent("contract/polyhedron_\(name).kurven"))
            let solid = bundle.solid
            for i in solid.faces.indices {
                let f = solid.face(i)
                faces += 1
                kinds.insert(f.id.kind)
                let c = simd_dot(solid.frames[i].normal, l)
                if f.isophote(light: light, on: f.copies[0]) != nil { fields += 1 }
                drawn += f.isophotes(light: light, [c, 0, 0.5, -0.5], on: f.copies[0]).flatMap(\.paths).count
            }
        }
        // A tower under two bands with walls: plateaus, a riser, walls.
        let built = try NativeLandscape.build(
            LandscapeRequest(expression: "1/(z - 0.45)",
                             domain: Domain(real: Interval(lo: -0.5, hi: 1.5), imag: Interval(lo: -1, hi: 1)),
                             resolution: 32, truncation: .realBands([RealBand(below: 0.5, height: 2)], beyond: 4)),
            refine: true)
        let context = Surface.HatchContext(built.manifest.occluder)
        let landscapeFaces = built.surface.faces(walls: built.manifest.occluder.walls, refine: built.refine)
        for f in landscapeFaces where f.isFlat {
            faces += 1
            kinds.insert(f.id.kind)
            let p = P2<ParamSpace>(f.domain.u.range.lo + 0.5 * f.domain.u.range.length,
                                   f.domain.v.range.lo + 0.5 * f.domain.v.range.length)
            let c = simd_dot(f.outwardNormal(at: p, on: f.copies[0]) ?? SIMD3(0, 0, 1), l)
            if f.isophote(light: light, on: f.copies[0]) != nil { fields += 1 }
            drawn += f.isophotes(light: light, [c, 0], on: f.copies[0]).flatMap(\.paths).count
        }
        for selector in [FaceSelector.plateaus(tiled: false), .walls(edges: [0, 1, 2, 3], endLines: false, base: 0, topOffset: 0),
                         .risers(tiled: false, endLines: false)] {
            drawn += built.surface.contours(of: .isophote, [0, 0.7, l.z], on: selector, restriction: .all, in: context,
                                            refine: built.refine, light: light).count
        }
        Check.expect(faces > 20 && kinds == [.plateau, .wall, .riser, .solid] && fields == 0 && drawn == 0,
                     "no isophote field and no curve on a polyhedron's faces, plateaus, walls or a riser",
                     "\(faces) faces of kinds \(kinds), \(fields) fields, \(drawn) curves")
        let graph = landscapeFaces.first { $0.id == .graph }!
        Check.expect(!graph.isFlat && graph.isophote(light: light, on: graph.copies[0]) != nil,
                     "the graph, curved, has one")
    }

    Check.suite("isophotes: on the graph of exp(z) lit along z, n · l = 1/√(1 + e^{2x}), and the isophote at c is Re z = ½ log(1/c² − 1)") {
        let built = try NativeLandscape.build(
            LandscapeRequest(expression: "exp(z)", domain: Domain(real: Interval(lo: -1, hi: 1), imag: Interval(lo: -1, hi: 1)),
                             resolution: 48, truncation: Truncation.none, layers: []),
            refine: true)
        let surface = built.surface, refine = built.refine
        let graph = surface.graph(refine: refine)!
        let g = graph.copies[0]
        let levels = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        var residual = 0.0, closed = 0.0, spans = true, counts: [Int] = []
        for contour in graph.isophotes(light: SIMD3(0, 0, 1), levels, on: g) {
            let c = contour.level, x = 0.5 * log(1 / (c * c) - 1)
            counts.append(contour.paths.count)
            for path in contour.paths {
                for p in path {
                    residual = max(residual, abs(simd_dot(graph.outwardNormal(at: p, on: g)!, SIMD3(0, 0, 1)) - c))
                    closed = max(closed, abs(p.x - x))
                }
                let ys = path.map(\.y)
                if abs(ys.min()! + 1) > 1e-12 || abs(ys.max()! - 1) > 1e-12 { spans = false }
            }
        }
        // The graph's slope is f's, read by central differences a
        // ten-thousandth of a cell wide (`Surface.graphNormal`), whose
        // rounding is about 1e-11 here: so n · l, read off the graph's own
        // jet, is c to that, and so is the closed form.
        Check.expect(residual <= 1e-10, "every vertex is on n · l = c, the normal read off the graph's own jet",
                     String(format: "worst %.1e", residual))
        Check.expect(closed <= 1e-10 && counts.allSatisfy { $0 == 1 } && spans,
                     "the isophote at c is the line Re z = ½ log(1/c² − 1), across the whole domain",
                     String(format: "worst %.1e off the line; ", closed) + "paths per level \(counts)")
        // Under a light at a slant the isophotes curve; still on n · l = c.
        // Held both to the graph's own jet and to its normal in closed form,
        // (−e^x, 0, 1)/√(1 + e^{2x}), which no code under test computes.
        let slant = SIMD3<Double>(1, 0.5, 2), l = simd_normalize(slant)
        var slanted = 0.0, slantedClosed = 0.0, vertices = 0
        for contour in graph.isophotes(light: slant, [0.5, 0.7, 0.85, 0.95], on: g) {
            for p in contour.paths.joined() {
                vertices += 1
                slanted = max(slanted, abs(simd_dot(graph.outwardNormal(at: p, on: g)!, l) - contour.level))
                let e = exp(p.x), n = SIMD3(-e, 0, 1) / (1 + e * e).squareRoot()
                slantedClosed = max(slantedClosed, abs(simd_dot(n, l) - contour.level))
            }
        }
        Check.expect(slanted <= 1e-10 && slantedClosed <= 1e-10 && vertices > 40,
                     "under a light at a slant, every vertex is on n · l = c, n the jet's and the closed form's",
                     String(format: "worst %.1e by the jet, %.1e by the closed form, over %d vertices",
                            slanted, slantedClosed, vertices))
        // Through the layer's own path: lifted onto |f|.
        let layer = surface.contours(of: .isophote, levels, on: .graph(tiled: false), restriction: .all,
                                     in: Surface.HatchContext(built.manifest.occluder), refine: refine,
                                     light: SIMD3(0, 0, 1))
        let lifted = layer.joined().map { abs($0.z - exp($0.x)) / exp($0.x) }.max() ?? .infinity
        Check.expect(layer.count == levels.count && lifted <= 1e-12, "the layer's curves lie on the graph, at |f|",
                     String(format: "%d paths, worst %.1e", layer.count, lifted))
    }

    Check.suite("isophotes and tone: on a tiled graph, each copy under its own normal, reflected and turned") {
        // exp(z) over [−1, 1]², copied by a reflection across Re z = 2 and
        // a quarter turn moved to Re z = 8: each copy's outward normal is
        // the closed form's, (−e^x, 0, 1)/√(1 + e^{2x}) at the point of the
        // domain it came from, its xy turned by the tile's linear part.
        let built = try NativeLandscape.build(
            LandscapeRequest(expression: "exp(z)", domain: Domain(real: Interval(lo: -1, hi: 1), imag: Interval(lo: -1, hi: 1)),
                             resolution: 48, truncation: Truncation.none, layers: []),
            refine: true)
        let tiles = [Affine2.identity, Affine2(a: -1, b: 0, tx: 4, c: 0, d: 1, ty: 0),
                     Affine2(a: 0, b: -1, tx: 8, c: 1, d: 0, ty: 0)]
        let context = Surface.HatchContext(perimeter: nil, region: .full, tiles: tiles)
        let light = SIMD3<Double>(0.6, -0.8, 1.2), l = simd_normalize(light)
        /// The tile a world point is a copy of, the point of the domain it
        /// came from, and the copy's outward normal there in closed form.
        func origin(_ w: P3<WorldSpace>) -> (tile: Int, p: SIMD2<Double>, n: SIMD3<Double>)? {
            for (k, tile) in tiles.enumerated() {
                let q = tile.inverse!(w)
                guard abs(q.x) <= 1 + 1e-9, abs(q.y) <= 1 + 1e-9 else { continue }
                let e = exp(q.x)
                let xy = SIMD2(tile.a * -e + tile.b * 0, tile.c * -e + tile.d * 0)
                return (k, SIMD2(q.x, q.y), SIMD3(xy.x, xy.y, 1) / (1 + e * e).squareRoot())
            }
            return nil
        }
        let levels = [0.3, 0.5, 0.7, 0.85]
        let isophotes = built.surface.contours(of: .isophote, levels, on: .graph(tiled: true), restriction: .all,
                                               in: context, refine: built.refine, light: light)
        var off = 0.0, offF = 0.0, lost = 0, perTile = [0, 0, 0]
        for path in isophotes {
            guard let first = origin(path[0]) else { lost += 1; continue }
            perTile[first.tile] += 1
            // The path's level: the nearest to its first vertex's n · l.
            let c = levels.min { abs($0 - simd_dot(first.n, l)) < abs($1 - simd_dot(first.n, l)) }!
            for w in path {
                guard let o = origin(w) else { lost += 1; continue }
                off = max(off, abs(simd_dot(o.n, l) - c))
                offF = max(offF, abs(w.z - exp(o.p.x)) / exp(o.p.x))
            }
        }
        Check.expect(off <= 1e-10 && offF <= 1e-12 && lost == 0 && perTile.allSatisfy { $0 > 0 },
                     "every isophote vertex of every copy is on n · l = c, the copy's normal in closed form",
                     String(format: "worst %.1e, off |f| by %.1e; paths per tile ", off, offF) + "\(perTile)")
        let hatching = Hatching(angle: 0, spacing: 0.05, tone: .halfLambert(light: light), widthSteps: 6)
        var wrong = 0, samples = 0, changes = 0, tilesHatched = Set<Int>()
        for (step, paths) in built.surface.graphHatching(hatching, tiled: true, in: context, refine: built.refine) {
            for path in paths {
                for (a, b) in zip(path, path.dropFirst()) {
                    for t in [0.0, 0.5, 1.0] {
                        guard let o = origin(P3(a.v + (b.v - a.v) * t)) else { wrong += 1; continue }
                        tilesHatched.insert(o.tile)
                        let nd = 6 * (1 - simd_dot(o.n, l)) / 2
                        guard abs(nd - nd.rounded()) > 1e-9 else { continue }
                        samples += 1
                        if Int(nd.rounded(.up)) != step { wrong += 1 }
                    }
                }
                for end in [path.first!, path.last!] {
                    guard let o = origin(end), abs(o.p.x) < 1 - 1e-12 else { continue }
                    let nd = 6 * (1 - simd_dot(o.n, l)) / 2
                    if abs(nd - nd.rounded()) <= 1e-10 { changes += 1 } else { wrong += 1 }
                }
            }
        }
        Check.expect(wrong == 0 && samples > 3000 && changes > 30 && tilesHatched == [0, 1, 2],
                     "the tone on every copy has the width step of the copy's darkness at every sample, changing on its isophotes",
                     "\(wrong) wrong of \(samples) samples, \(changes) changes, tiles \(tilesHatched.sorted())")
    }

    /// The torus's hatching held to the darkness in closed form: each piece
    /// along `alongU`'s coordinate, at one of the levels across, its width
    /// step the step of the darkness at every sample of it, and its ends,
    /// where the width changes, on an isophote.
    func toneVerdict(_ hatching: Hatching, alongU: Bool)
        -> (pieces: Int, offLine: Double, wrongStep: Int, samples: Int, offIsophote: Double, changes: Int, gaps: Double) {
        guard case .halfLambert(let light) = hatching.tone else { preconditionFailure() }
        let l = simd_normalize(light), n = Double(hatching.widthSteps)
        func darkness(_ p: P2<ParamSpace>) -> Double { (1 - simd_dot(normal(p.x, p.y), l)) / 2 }
        let levels = Face.hatchLevels(across: alongU ? torus.v : torus.u, spacing: hatching.spacing)
        var pieces = 0, offLine = 0.0, wrongStep = 0, samples = 0, offIsophote = 0.0, changes = 0
        var extent: [Int: Double] = [:]
        var ends: [Int: [Double]] = [:]
        for (step, set) in torus.coordinateHatching(hatching) {
            for k in 0..<set.count {
                let coords = Array(set.coords(path: k)!)
                pieces += 1
                let across = { (p: P2<ParamSpace>) in alongU ? p.y : p.x }, along = { (p: P2<ParamSpace>) in alongU ? p.x : p.y }
                let line = levels.indices.min { apart(across(coords[0]), levels[$0]) < apart(across(coords[0]), levels[$1]) }!
                offLine = max(offLine, coords.map { apart(across($0), levels[line]) }.max()!)
                extent[line, default: 0] += abs(along(coords.last!) - along(coords.first!))
                for (a, b) in zip(coords, coords.dropFirst()) {
                    for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
                        let d = darkness(P2(a.v + (b.v - a.v) * t))
                        // A sample on a step's boundary is the boundary's.
                        guard abs(n * d - (n * d).rounded()) > 1e-9 else { continue }
                        samples += 1
                        if Int((n * d).rounded(.up)) != step { wrongStep += 1 }
                    }
                }
                for end in [coords.first!, coords.last!] {
                    let nd = n * darkness(end)
                    if abs(nd - nd.rounded()) <= 1e-12 * n {
                        changes += 1
                    } else {
                        ends[line, default: []].append(along(end))
                    }
                }
            }
        }
        // Off an isophote, a piece ends only where its line does: a closed
        // line's two ends, a turn apart.
        for (_, e) in ends {
            if e.count > 2 || (e.count == 2 && apart(e[0], e[1]) > 1e-12) { offIsophote = .infinity }
        }
        let gaps = levels.indices.map { abs((extent[$0] ?? 0) - twoPi) }.max() ?? .infinity
        return (pieces, offLine, wrongStep, samples, offIsophote, changes, gaps)
    }

    Check.suite("tone: on the torus, half-Lambert hatching runs along a coordinate, its width the darkness's step at every sample, changing on the isophotes") {
        for (hatching, alongU) in [(Hatching(angle: 0, spacing: twoPi / 24, tone: .halfLambert(light: SIMD3(1, -1, 1.5)),
                                             widthSteps: 8), true),
                                   (Hatching(angle: .pi / 2, spacing: twoPi / 36, tone: .halfLambert(light: SIMD3(-0.4, -0.6, 0.7)),
                                             widthSteps: 5), false)] {
            let v = toneVerdict(hatching, alongU: alongU)
            let what = alongU ? "along u" : "along v"
            Check.expect(v.offLine <= 1e-12 && v.gaps <= 1e-9,
                         "\(what): every piece lies on its line of the other coordinate, and the pieces cover each line, a whole turn",
                         String(format: "%d pieces, off by %.1e, cover a turn to %.1e", v.pieces, v.offLine, v.gaps))
            Check.expect(v.wrongStep == 0 && v.samples > 10_000,
                         "\(what): at every sample a piece's width step is ⌈\(hatching.widthSteps) (1 − n · l)/2⌉, n the closed form's",
                         "\(v.wrongStep) of \(v.samples) samples wrong")
            Check.expect(v.offIsophote == 0 && v.changes > 100,
                         "\(what): the width changes only on an isophote n · l = 1 − 2k/\(hatching.widthSteps), within 1e-12",
                         "\(v.changes) piece ends on one")
        }
        // The bake: a layer per width step at k/N of the layer's width,
        // each holding that step's pieces.
        let hatching = Hatching(angle: 0, spacing: twoPi / 24, tone: .halfLambert(light: SIMD3(1, -1, 1.5)), widthSteps: 8)
        let spec = LayerSpec(name: "tone", role: .scaffold, source: .hatching(hatching, faces: .surface),
                             width: 1.5, heightPolicy: .surface)
        let scene = Scene(surface: torus, layers: [Layer(spec: spec, paths: .empty)],
                          camera: .oblique(SurfaceRequest.projection), depthTolerance: 0)
        let steps = torus.coordinateHatching(hatching)
        let layers = scene.visibleLayers()
        let widths = zip(layers, steps).allSatisfy { drawn, step in
            drawn.0.spec.width == 1.5 * Double(step.widthStep) / 8 && drawn.0.spec.name == "tone \(step.widthStep)/8"
                && drawn.0.paths == step.paths
        }
        Check.expect(layers.count == steps.count && steps.count >= 6 && widths,
                     "baked as a layer per width step, at k/8 of the layer's width, each with that step's pieces",
                     "\(layers.map { "\($0.0.spec.name) \($0.0.spec.width)" })")
    }

    Check.suite("tone: on the graph, along a coordinate at the plateaus' levels, cut where the width step changes") {
        let built = try NativeLandscape.build(
            LandscapeRequest(expression: "exp(z)", domain: Domain(real: Interval(lo: -1, hi: 1), imag: Interval(lo: -1, hi: 1)),
                             resolution: 48, truncation: Truncation.none, layers: []),
            refine: true)
        let light = SIMD3<Double>(1, 0.5, 2), l = simd_normalize(light)
        let hatching = Hatching(angle: 0, spacing: 0.05, tone: .halfLambert(light: light), widthSteps: 6)
        let graph = built.surface.graph(refine: built.refine)!
        let steps = built.surface.graphHatching(hatching, tiled: false, in: Surface.HatchContext(built.manifest.occluder),
                                                refine: built.refine)
        let levels = Face.hatchLevels(across: graph.domain.v, spacing: 0.05)
        var offLine = 0.0, wrong = 0, samples = 0, changes = 0, onF = 0.0
        for (step, paths) in steps {
            for path in paths {
                let y = levels.min { abs($0 - path[0].y) < abs($1 - path[0].y) }!
                offLine = max(offLine, path.map { abs($0.y - y) }.max()!)
                onF = max(onF, path.map { abs($0.z - exp($0.x)) / exp($0.x) }.max()!)
                for (a, b) in zip(path, path.dropFirst()) {
                    for t in [0.0, 0.5, 1.0] {
                        let p = a.v + (b.v - a.v) * t
                        let nl = simd_dot(graph.outwardNormal(at: P2(p.x, p.y), on: graph.copies[0])!, l)
                        let nd = 6 * (1 - nl) / 2
                        guard abs(nd - nd.rounded()) > 1e-9 else { continue }
                        samples += 1
                        if Int(nd.rounded(.up)) != step { wrong += 1 }
                    }
                }
                for end in [path.first!, path.last!] where abs(end.x) < 1 - 1e-12 {
                    let nl = simd_dot(graph.outwardNormal(at: P2(end.x, end.y), on: graph.copies[0])!, l)
                    let nd = 6 * (1 - nl) / 2
                    if abs(nd - nd.rounded()) <= 1e-10 { changes += 1 } else { wrong += 1 }
                }
            }
        }
        Check.expect(offLine <= 1e-15 && onF <= 1e-12 && levels.count == 39,
                     "every piece lies on its line Im z = c, on the graph at |f|",
                     String(format: "%d lines, off by %.1e, off |f| by %.1e", levels.count, offLine, onF))
        Check.expect(wrong == 0 && samples > 1000 && changes > 20,
                     "its width step is the darkness's at every sample, read off the graph's jet, and it changes only on an isophote, to the jet's rounding",
                     "\(wrong) wrong of \(samples) samples, \(changes) changes")
    }

    Check.suite("isophotes and tone: seen exactly where the torus's own cast and the grid landscape's cast say") {
        let isophotes = LayerSpec(name: "isophotes", role: .scaffold,
                                  source: .contour(field: .isophote, levels: [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75],
                                                   restriction: .all, faces: .surface, light: SIMD3(-0.4, -0.6, 0.7)),
                                  width: 0.3, heightPolicy: .surface)
        let tone = LayerSpec(name: "tone", role: .scaffold,
                             source: .hatching(Hatching(angle: 0, spacing: twoPi / 48, tone: .halfLambert(light: SIMD3(-0.4, -0.6, 0.7)),
                                                        widthSteps: 4), faces: .surface),
                             width: 1, heightPolicy: .surface)
        let drawing = try SurfaceDrawing.build(request).restyled([isophotes, tone])
        let camera = Camera.oblique(SurfaceRequest.projection)
        let torusScene = drawing.scene(camera: camera)
        let t = try compositeVerdict(try Composite([("torus", torusScene, .identity)], camera: camera),
                                     casts: [{ _, _ in true }])
        Check.expect(t.ownWrong == 0 && t.ownCompared > 20_000 && t.torusSeen > 5000,
                     "the torus's isophotes and tone, at every vertex, are drawn exactly where its own cast sees them",
                     "\(t.ownCompared) compared, \(t.torusSeen) seen, \(t.ownWrong) wrong" + t.ownExample)
        // The cubic landscape on its grid: its own normals, the grid's.
        var bundle = try KurvenBundle.read(at: Fixtures.dir.appendingPathComponent("expr/landscape_cubic.kurven"))
        var m = bundle.manifest
        m.layers = [
            LayerSpec(name: "isophotes", role: .scaffold,
                      source: .contour(field: .isophote, levels: [-0.6, -0.3, 0, 0.3, 0.6, 0.9], restriction: .all,
                                       faces: .graph(tiled: false), light: SIMD3(-0.4, -0.6, 0.7)),
                      width: 0.3, heightPolicy: .surface),
            LayerSpec(name: "tone", role: .scaffold,
                      source: .hatching(Hatching(angle: 0, spacing: 0.04, tone: .halfLambert(light: SIMD3(-0.4, -0.6, 0.7)),
                                                 widthSteps: 4), faces: .graph(tiled: false)),
                      width: 1, heightPolicy: .surface),
        ]
        bundle = bundle.restyled(m)
        let landscape = Scene(bundle: bundle, preset: bundle.manifest.presets[0])
        let g = try compositeVerdict(try Composite([("cubic", landscape, .identity)], camera: landscape.camera),
                                     casts: [{ _, _ in true }])
        Check.expect(g.ownWrong == 0 && g.ownCompared > 5000,
                     "the grid landscape's isophotes and tone, at every vertex, are drawn exactly where its cast sees them",
                     "\(g.ownCompared) compared, \(g.ownWrong) wrong" + g.ownExample)
    }

    Check.suite("isophotes and tone: on f, seen exactly where the dense cast against f says") {
        let built = try NativeLandscape.build(LandscapeRequest(preset: Catalog.native.preset("cubic")!, resolution: 64))
        var m = built.manifest
        m.layers = [
            LayerSpec(name: "isophotes", role: .scaffold,
                      source: .contour(field: .isophote, levels: [-0.6, -0.3, 0, 0.3, 0.6, 0.9], restriction: .all,
                                       faces: .graph(tiled: false), light: SIMD3(-0.4, -0.6, 0.7)),
                      width: 0.3, heightPolicy: .surface),
            LayerSpec(name: "tone", role: .scaffold,
                      source: .hatching(Hatching(angle: 0, spacing: 0.06, tone: .halfLambert(light: SIMD3(-0.4, -0.6, 0.7)),
                                                 widthSteps: 4), faces: .graph(tiled: false)),
                      width: 1, heightPolicy: .surface),
        ]
        let bundle = built.restyled(m)
        let scene = Scene(bundle: bundle, preset: bundle.manifest.presets[0])
        let h = scene.heightfield!, view = scene.camera.view
        let magnitude = h.refine!.magnitude!
        let vis = HeightfieldVisibility(heightfield: h, view: view, depthTolerance: scene.depthTolerance)
        let toward = -view.projectionDirection
        let c = view.m.columns
        let perUnit = abs(simd_dot(SIMD3(c.0.z, c.1.z, c.2.z), toward))
        let grid = h.surface.height
        let cell = max(abs(grid.domain.real.length) / Double(grid.width - 1), abs(grid.domain.imag.length) / Double(grid.height - 1))
        var compared = 0, wrong = 0, layers = 0, example = ""
        for (layer, seen) in scene.visibleLayers() {
            layers += 1
            guard let bounds = AABB(seen.vertices + layer.paths.mapped(view).vertices) else { continue }
            let size = max(bounds.size.x, bounds.size.y)
            let index = SegmentIndex(seen, lo: SIMD2(bounds.lo.x, bounds.lo.y) - 1e-6 * size, size: 1.00001 * size)
            let vertices = layer.paths.vertices
            let truths = Heightfield.parallelMap(vertices) { v -> Bool? in
                let f = vis.facing(v.xy)
                guard abs(f) > 1e-9 else { return nil }
                guard f > 0 else { return false }
                return denselyHidden(h, magnitude: magnitude, from: v.v, along: toward, depthTolerance: vis.depthTolerance,
                                     perUnit: perUnit, cell: cell) == nil
            }
            for (v, truth) in zip(vertices, truths) {
                let p = view(v).v
                guard let truth, index.nearestEnd(p) >= 1e-6 * size else { continue }
                compared += 1
                if truth != index.holds(p, tolerance: 1e-9 * size) {
                    wrong += 1
                    if example.isEmpty {
                        example = String(format: " e.g. %@ at (%.4f, %.4f, %.4f): the cast says %@", layer.spec.name,
                                         v.x, v.y, v.z, truth ? "seen" : "hidden")
                    }
                }
            }
        }
        Check.expect(wrong == 0 && compared > 3000 && layers == 5,
                     "every vertex of the isophotes and of each width of the tone is drawn exactly where the dense cast sees it",
                     "\(layers) layers, \(compared) compared, \(wrong) wrong" + example)
    }

    Check.suite("isophotes and tone: the manifest") {
        func source(_ text: String, on geometry: String = "landscape") throws -> LayerSource {
            switch geometry {
            case "surface":
                let empty = SurfaceManifest(surface: SurfaceRequest(.torus(major: 2, minor: 1)), layers: [], presets: [],
                                            provenance: Provenance(function: "", example: "", params: [:], cpuCount: 1,
                                                                   gitSha: ""))
                let layer = #"{"name": "x", "role": "scaffold", "width": 0.3, "heightPolicy": "surface", "source": "#
                    + text + "}"
                let m = try SurfaceManifest(json: JSONValue.parse(Data(empty.canonicalJSON
                    .replacingOccurrences(of: #""layers":[]"#, with: #""layers":["# + layer + "]").utf8)))
                return m.layers[0].source
            default:
                return try LayerSource(json: JSONValue.parse(Data(text.utf8)))
            }
        }
        func refused(_ text: String, on geometry: String = "landscape", saying words: String) -> Bool {
            do { _ = try source(text, on: geometry); return false } catch { return "\(error)".contains(words) }
        }
        let iso = #"{"kind": "contour", "field": "isophote", "levels": [0.5], "restriction": {"kind": "all"}, "light": [0, 0, 2], "faces": "#
        let graphIso = try source(iso + #"{"kind": "graph", "tiled": false}}"#)
        let surfaceIso = try source(iso + #"{"kind": "surface"}}"#, on: "surface")
        let tone = #"{"kind": "hatching", "angle": 0, "spacing": 0.1, "widthSteps": 4, "tone": {"kind": "halfLambert", "light": [1, 0, 1]}, "faces": "#
        let surfaceTone = try source(tone + #"{"kind": "surface"}}"#, on: "surface")
        let graphTone = try source(tone + #"{"kind": "graph", "tiled": true}}"#)
        var roundTrips = true
        for s in [graphIso, graphTone] where try source(s.json.canonical) != s { roundTrips = false }
        for s in [surfaceIso, surfaceTone] where try source(s.json.canonical, on: "surface") != s { roundTrips = false }
        Check.expect(graphIso.light == SIMD3(0, 0, 2) && surfaceTone.light == SIMD3(1, 0, 1) && roundTrips
                         && graphIso.json.canonical.contains(#""light":[0.0,0.0,2.0]"#)
                         && !surfaceTone.json.canonical.contains(#""step""#),
                     "an isophote keeps its light as written, a curved face's hatching writes no step, and both round-trip")
        let plain = LayerSource.contour(field: .u, levels: [1], restriction: .all, faces: .graph(tiled: false))
        Check.expect(!plain.json.canonical.contains("light"), "a contour of anything else writes no light")
        Check.expect(refused(iso.replacingOccurrences(of: #""light": [0, 0, 2], "#, with: "") + #"{"kind": "graph"}}"#,
                             saying: "light")
                         && refused(iso.replacingOccurrences(of: "[0, 0, 2]", with: "[0, 0, 0]") + #"{"kind": "graph"}}"#,
                                    saying: "not all zero")
                         && refused(iso.replacingOccurrences(of: "isophote", with: "u") + #"{"kind": "graph"}}"#,
                                    saying: "only an isophote has one")
                         && refused(iso + #"{"kind": "plateaus"}}"#, saying: "constant on a flat face")
                         && refused(iso + #"{"kind": "walls", "edges": [0]}}"#, saying: "constant on a flat face")
                         && refused(iso + #"{"kind": "risers"}}"#, saying: "constant on a flat face"),
                     "an isophote with no light, a light with no direction, a light on another field, and isophotes on a flat face are refused")
        Check.expect(refused(tone.replacingOccurrences(of: #""angle": 0"#, with: #""angle": 0.5"#) + #"{"kind": "graph"}}"#,
                             saying: "along a coordinate")
                         && refused(tone.replacingOccurrences(of: #""widthSteps": 4"#, with: #""widthSteps": 17"#)
                                    + #"{"kind": "surface"}}"#, on: "surface", saying: "17 widths")
                         && refused(#"{"kind": "hatching", "angle": 0, "spacing": 0.1, "widthSteps": 1, "tone": {"kind": "turing", "scale": 1, "seed": 1}, "faces": {"kind": "graph"}}"#,
                                    saying: "Turing")
                         && refused(tone.replacingOccurrences(of: #""angle": 0"#, with: #""angle": 0, "step": 0.1"#)
                                    + #"{"kind": "surface"}}"#, on: "surface", saying: "a step on a curved face")
                         && refused(tone + #"{"kind": "solid"}}"#, on: "surface", saying: "one face"),
                     "hatching on a curved face at a slant, in too many widths, under Turing's tone, with a step, or on another face is refused")
        // Read off the request before anything is built, as the window asks
        // before an edit is taken: the outward side, and the most vertices a
        // hatching may take.
        func relief(_ amplitude: Double, floor: Double? = nil) -> SurfaceRequest {
            SurfaceRequest(.relief(modulus: 0.64, major: 2, minor: 1, amplitude: amplitude, repeats: SIMD2(1, 1),
                                   grid: 120, turn: 0, floor: floor, soft: false))
        }
        let lit = LayerSource.contour(field: .isophote, levels: [0.5], restriction: .all, faces: .surface,
                                      light: SIMD3(0, 0, 1))
        let toneSource = LayerSource.hatching(Hatching(angle: 0, spacing: 0.1, tone: .halfLambert(light: SIMD3(0, 0, 1))),
                                              faces: .surface)
        Check.expect(relief(0.3).boundsSolid == true && relief(1 / 3).boundsSolid == false
                         && relief(0.4).boundsSolid == false && relief(0.4, floor: 1).boundsSolid == true
                         && SurfaceRequest(.torus(major: 1, minor: 1)).boundsSolid == false
                         && relief(0.3).unsupported(lit) == nil && relief(0.3).unsupported(toneSource) == nil
                         && relief(0.4).unsupported(lit)?.contains("bounds no solid") == true
                         && relief(0.4).unsupported(toneSource)?.contains("bounds no solid") == true
                         && relief(0.4).unsupported(.coordinateCurves(u: 4, v: 4)) == nil,
                     "a relief bounds a solid while |A| times the height it displaces stays under the tube's reach, "
                         + "and isophotes and half-Lambert tone are refused on one that does not")
        // And the rule is the build's: the surface built has an outward side
        // exactly where the request says it bounds a solid.
        var agree = 0, disagree: [String] = []
        for amplitude in [0.3, 1.0 / 3, 0.4] {
            for floor in [nil, 2.5] as [Double?] {
                var request = relief(amplitude, floor: floor)
                request.grid = 64
                let built = try SurfaceDrawing.build(request).surface.outward != nil
                if built == request.boundsSolid { agree += 1 } else { disagree.append("\(amplitude), \(floor ?? 0)") }
            }
        }
        Check.expect(disagree.isEmpty && agree == 6, "the relief built bounds a solid exactly where its request says",
                     "disagree at \(disagree)")
        let tooFine = LayerSource.hatching(Hatching(angle: .pi / 2, spacing: 1e-6), faces: .surface)
        let unlit = LayerSource.contour(field: .isophote, levels: [0.5], restriction: .all, faces: .surface)
        Check.expect(SurfaceRequest(.torus(major: 2, minor: 1)).unsupported(tooFine)?.contains("more than the") == true
                         && SurfaceRequest(.torus(major: 2, minor: 1)).unsupported(unlit)?.contains("no light") == true,
                     "hatching past a layer's vertices, and an isophote with no light, are refused before anything is built")
        var unlitRefused = false
        do {
            _ = try SurfaceDrawing.build(SurfaceRequest(.torus(major: 2, minor: 1), grid: 64))
                .restyled([LayerSpec(name: "x", role: .scaffold, source: unlit, width: 0.3, heightPolicy: .surface)])
        } catch { unlitRefused = "\(error)".contains("no light") }
        Check.expect(unlitRefused, "an isophote with no light, built in code, is refused, not drawn as the height")
        let negative = #"{"name": "x", "role": "scaffold", "width": -1, "heightPolicy": "surface", "source": {"kind": "contour", "field": "u", "levels": [0.5], "restriction": {"kind": "all"}, "faces": {"kind": "graph"}}}"#
        Check.expect((try? LayerSpec(json: JSONValue.parse(Data(negative.utf8)))) == nil,
                     "a layer of negative width is refused")
    }
}
