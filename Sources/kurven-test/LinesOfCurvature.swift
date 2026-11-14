import Foundation
import simd
import KurvenCore
import KurvenMath
import KurvenDynamics

/// Lines of curvature (phase 11c): the integral curves of a surface's
/// principal directions, an unsigned field, held to closed forms. On the
/// round torus, a surface of revolution, they are the meridians and the
/// parallels, its coordinate curves, closed after one turn across the seams.
/// On a triaxial ellipsoid they run into its four umbilics and end there, and
/// a seed on one draws nothing. On the forced torus, which has none, the two
/// families cross at right angles in space, and each line, which does not
/// close, ends where it first returns to the coordinate curve it was seeded
/// on. Then the manifest, and visibility as the coordinate curves have it.
func lineOfCurvatureTests() {
    let twoPi = 2 * Double.pi
    /// The round torus as the CLI draws it.
    let torus = ParametricSurface.torus(major: 2, minor: 1, samples: (1024, 512))

    Check.suite("lines of curvature: on the round torus, the principal curvatures and directions are the closed form's") {
        let face = torus.face, copy = face.copies[0]
        var curvatureOff = 0.0, directionOff = 0.0, count = 0
        for j in 0..<64 {
            for i in 0..<64 {
                let p = P2<ParamSpace>(twoPi * (Double(i) + 0.37) / 64, twoPi * (Double(j) + 0.61) / 64)
                guard let k = face.principalCurvatures(at: p, on: copy) else { continue }
                count += 1
                // The tube's 1/r, and the parallel's cos v / (R + r cos v),
                // both against the inward normal.
                curvatureOff = max(curvatureOff, abs(k.greater - 1), abs(k.lesser - cos(p.y) / (2 + cos(p.y))))
                let g = simd_normalize(k.directions.greater), l = simd_normalize(k.directions.lesser)
                directionOff = max(directionOff, abs(g.x), abs(l.y))
            }
        }
        Check.expect(count == 4096 && curvatureOff <= 1e-14 && directionOff <= 1e-14,
                     "the greater is the tube's 1/r along the meridian, the lesser cos v / (R + r cos v) along the parallel",
                     String(format: "%d points; curvatures off by %.1e, directions off their axes by %.1e",
                            count, curvatureOff, directionOff))
    }

    Check.suite("lines of curvature: on the round torus they are its coordinate curves, closed after one turn across the seams") {
        let cell = torus.face.cell
        // Each family from seeds on the coordinate curve across it, at values
        // the same modulo a turn once: 2π and −2π are 0 again. Seeds next to
        // the seams, and on them, so every line crosses one.
        let values = [0, 1.3, 3.0, twoPi - 0.01, -0.5, twoPi, -twoPi, 6.2831]
        for (curvature, along, on, levels) in [(PrincipalCurvature.greater, ContourField.u, ContourField.v, [0, 2.5, 6.25]),
                                               (.lesser, .v, .u, [0, 4.0, 6.27])] {
            var curves = 0, closed = 0, drift = 0.0, turnOff = 0.0, world = 0.0
            var seeds = 0
            for level in levels {
                let s = Seeds(on: on, at: level, .values(of: along, values))
                let points = torus.seedPoints(s)
                seeds += points.count
                let traced = torus.tracedIntegralCurves(of: .principalDirection(of: curvature), through: points,
                                                        returningTo: torus.section(s))
                for (seed, curve) in zip(points, traced) {
                    curves += 1
                    if curve.backward == .closed && curve.forward == .closed { closed += 1 }
                    // The fixed coordinate stays the seed's; the other runs
                    // one turn, to the seed's image a period along.
                    let fixed = curvature == .greater ? 0 : 1
                    for p in curve.path {
                        drift = max(drift, abs(p.v[fixed] - seed.v[fixed]))
                        // How far in space from the coordinate curve through
                        // the seed: the point with that one coordinate set.
                        var q = p.v
                        q[fixed] = seed.v[fixed]
                        world = max(world, simd_distance(torus.map(P2(q)).position, torus.map(p).position))
                    }
                    let run = curve.path.last!.v - curve.path.first!.v
                    turnOff = max(turnOff, abs(abs(run[1 - fixed]) - twoPi), abs(run[fixed]))
                }
            }
            // Six distinct values a level, three levels.
            Check.expect(seeds == 18 && curves == 18 && closed == 18 && drift <= 1e-12 && turnOff <= 1e-12,
                         "the \(curvature) family: each line is the coordinate curve \(along.rawValue) runs along, "
                             + "closed on its seed's image one turn along",
                         String(format: "%d seeds, %d curves, %d closed; worst %.1e rad (%.1e cell) off the coordinate "
                                    + "curve, %.1e in space; a turn off by %.1e",
                                seeds, curves, closed, drift, drift / cell, world, turnOff))
        }
        // In space, as a layer: every vertex on the map, with its coordinate.
        let layer = LayerSpec.linesOfCurvature(.lesser, over: (torus.u, torus.v), count: 6, width: 0.3)
        guard case .integralCurves(let field, let seeds, _) = layer.source else { return }
        let drawn = try torus.integralCurves(of: field, seeds: seeds)
        var onMap = true
        for k in 0..<drawn.count {
            for (v, c) in zip(drawn[path: k], drawn.coords(path: k)!) where v.v != torus.map(c).position { onMap = false }
        }
        Check.expect(drawn.count == 6 && onMap,
                     "drawn as a layer, each parallel is one path whose vertices are the map's at their coordinates",
                     "\(drawn.count) paths")
    }

    Check.suite("lines of curvature: a triaxial ellipsoid's run into its umbilics and end there; a seed on one draws nothing") {
        // x²/a² + y²/b² + z²/c² = 1, a > b > c, its poles on the middle axis
        // so the umbilics, in the plane of the longest and the shortest, lie
        // on the equator v = 0, where cos u = √((a² − b²)/(a² − c²)) up to
        // sign and sin u = √((b² − c²)/(a² − c²)) likewise.
        let (a, b, c) = (3.0, 2.0, 1.0)
        let ellipsoid = ParametricSurface(
            u: .angle(samples: 256), v: ParamAxis(Interval(lo: -.pi / 2, hi: .pi / 2), periodic: false, samples: 129),
            encloses: true,
            secondDerivatives: { p in
                let (cu, su, cv, sv) = (cos(p.x), sin(p.x), cos(p.y), sin(p.y))
                return SecondDerivatives(uu: SIMD3(-a * cv * cu, 0, -c * cv * su),
                                         uv: SIMD3(a * sv * su, 0, -c * sv * cu),
                                         vv: SIMD3(-a * cv * cu, -b * sv, -c * cv * su))
            }) { p in
                let (cu, su, cv, sv) = (cos(p.x), sin(p.x), cos(p.y), sin(p.y))
                return SurfaceJet(position: SIMD3(a * cv * cu, b * sv, c * cv * su),
                                  du: SIMD3(-a * cv * su, 0, c * cv * cu),
                                  dv: SIMD3(-a * sv * cu, b * cv, -c * sv * su))
            }
        let face = ellipsoid.face, copy = face.copies[0]
        let u0 = atan2(((b * b - c * c) / (a * a - c * c)).squareRoot(), ((a * a - b * b) / (a * a - c * c)).squareRoot())
        let umbilics = [u0, .pi - u0, .pi + u0, twoPi - u0].map { P2<ParamSpace>($0, 0) }
        let read = umbilics.compactMap { face.principalCurvatures(at: $0, on: copy) }
        let gap = read.map { abs($0.greater - $0.lesser) / abs($0.greater) }.max() ?? .infinity
        // At (a, 0, 0): the curvature across, along u, is a/c², along v a/b²,
        // both positive against the inward normal.
        let vertex = face.principalCurvatures(at: P2(0, 0), on: copy)
        Check.expect(read.count == 4 && read.allSatisfy(\.umbilic) && gap <= 1e-14
                         && vertex.map { abs($0.greater - a / (c * c)) <= 1e-14 && abs($0.lesser - a / (b * b)) <= 1e-14
                             && !$0.umbilic } == true,
                     "the four umbilics read as umbilics, and the vertex (a, 0, 0) as a/c² and a/b², convex",
                     String(format: "curvatures equal at them to %.1e", gap))
        // From (a, 0, 0) the greater family runs along the equator both
        // ways into the umbilics either side; from (0, 0, c) the lesser does,
        // into the same two from the other side.
        var ends: [IntegralCurve.End] = [], worst = 0.0, inSpace = 0.0
        for (curvature, seed, targets) in [(PrincipalCurvature.greater, P2<ParamSpace>(0, 0), [twoPi - u0, u0]),
                                           (.lesser, P2<ParamSpace>(.pi / 2, 0), [u0, .pi - u0])] {
            guard let field = face.principalDirections(of: curvature, on: copy),
                  let curve = face.integralCurve(of: field, through: seed, on: copy) else { ends.append(.boundary); continue }
            ends += [curve.backward, curve.forward]
            for end in [curve.path.first!, curve.path.last!] {
                let target = targets.map { P2<ParamSpace>($0, 0) }.min {
                    abs(shortWay($0.x - end.x, period: twoPi)) < abs(shortWay($1.x - end.x, period: twoPi))
                }!
                worst = max(worst, hypot(shortWay(end.x - target.x, period: twoPi), end.y - target.y))
                inSpace = max(inSpace, simd_distance(ellipsoid.map(end).position, ellipsoid.map(target).position))
            }
        }
        // Within the least step of it: 1e-4 cell, 2.5e-6 rad here.
        Check.expect(ends == [.singular, .singular, .singular, .singular] && worst <= 1e-5 && inSpace <= 2e-5,
                     "a line of either family heading into an umbilic ends at it, a singular point of the field, "
                         + "about the least step short",
                     String(format: "ends %@; worst %.1e rad, %.1e in space, from the umbilic", "\(ends.map { "\($0)" })", worst, inSpace))
        let field = face.principalDirections(of: .greater, on: copy)!
        let onOne = umbilics.compactMap { face.integralCurve(of: field, through: $0, on: copy) }
        let seeded = ellipsoid.tracedIntegralCurves(of: .principalDirection(of: .lesser),
                                                    through: ellipsoid.seedPoints(Seeds(on: .v, at: 0, .values(of: .u, [u0, 1.0]))))
        Check.expect(onOne.isEmpty && seeded.count == 1,
                     "a seed on an umbilic draws nothing, and its neighbour on the same level set draws its line",
                     "\(onOne.count) from the umbilics, \(seeded.count) of two seeds")
    }

    Check.suite("lines of curvature: on the forced torus the two families cross at right angles in space, each once round") {
        let drawing = try SurfaceDrawing.build(SurfaceRequest(.forced(system: "jerk", fit: 8000, harmonics: 24, radial: 0,
                                                                      axial: 1, radius: 2.5, turns: 1)))
        let surface = drawing.surface, face = surface.face, copy = face.copies[0]
        // No umbilic: the principal curvatures stay apart everywhere.
        var least = Double.infinity
        for j in 0..<128 {
            for i in 0..<256 {
                let p = P2<ParamSpace>(twoPi * Double(i) / 256, twoPi * Double(j) / 128)
                if let k = face.principalCurvatures(at: p, on: copy) { least = min(least, k.greater - k.lesser) }
            }
        }
        var families: [[[SIMD2<Double>]]] = []
        var returned = 0, curves = 0, offSection = 0.0
        for curvature in [PrincipalCurvature.greater, .lesser] {
            let layer = LayerSpec.linesOfCurvature(curvature, over: (surface.u, surface.v), count: 24, width: 0.3)
            guard case .integralCurves(let field, let seeds, _) = layer.source else { return }
            let traced = surface.tracedIntegralCurves(of: field, through: surface.seedPoints(seeds),
                                                      returningTo: surface.section(seeds))
            let fixed = curvature == .greater ? 1 : 0
            for curve in traced {
                curves += 1
                if curve.backward == .returned && curve.forward == .returned { returned += 1 }
                for end in [curve.path.first!, curve.path.last!] {
                    offSection = max(offSection, abs(shortWay(end.v[fixed], period: twoPi)))
                }
            }
            families.append(traced.map { $0.path.map(\.v) })
        }
        // Where a chord of one family crosses a chord of the other, each
        // carried into the period rectangle by whole turns, the angle between
        // them in space, each carried there by the map's derivatives: between
        // the chords themselves, which the sag rule lets turn from the curve
        // by up to 4s/L, 4.6° at the longest step; between the curves'
        // tangents there, each the chords' directions either side
        // interpolated to the crossing, as a chord's direction is its curve's
        // at its middle to second order; and between the field's directions,
        // orthogonal by construction.
        func wrap(_ p: SIMD2<Double>) -> SIMD2<Double> { p - twoPi * (p / twoPi).rounded(.down) }
        let bin = 0.05
        func bins(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> [SIMD2<Int>] {
            let lo = simd_min(a, b) / bin, hi = simd_max(a, b) / bin
            var out: [SIMD2<Int>] = []
            for x in Int(lo.x.rounded(.down))...Int(hi.x.rounded(.down)) {
                for y in Int(lo.y.rounded(.down))...Int(hi.y.rounded(.down)) { out.append(SIMD2(x, y)) }
            }
            return out
        }
        /// The tangent at fraction `s` along chord `i` of `path`, from the
        /// chords' directions at their middles.
        func tangent(_ path: [SIMD2<Double>], _ i: Int, _ s: Double) -> SIMD2<Double> {
            func chord(_ k: Int) -> SIMD2<Double> { path[k + 1] - path[k] }
            let j = s < 0.5 ? i - 1 : i + 1
            guard j >= 0, j < path.count - 1 else { return chord(i) }
            let here = simd_length(chord(i)), there = simd_length(chord(j))
            // From the middle of chord i to the middle of chord j, in length.
            let w = abs(s - 0.5) * here / (0.5 * (here + there))
            return (1 - w) * chord(i) / here + w * chord(j) / there
        }
        var index: [SIMD2<Int>: [(path: Int, chord: Int, a: SIMD2<Double>, b: SIMD2<Double>)]] = [:]
        for (n, path) in families[1].enumerated() {
            for i in 0..<(path.count - 1) {
                let a0 = wrap(path[i]), b0 = a0 + (path[i + 1] - path[i])
                for k in bins(a0, b0) { index[k, default: []].append((n, i, a0, b0)) }
            }
        }
        let greaterField = face.principalDirections(of: .greater, on: copy)!
        let lesserField = face.principalDirections(of: .lesser, on: copy)!
        var crossings = 0, chordWorst = 0.0, tangentWorst = 0.0, fieldWorst = 0.0
        for path in families[0] {
            for i in 0..<(path.count - 1) {
                let p = wrap(path[i]), r = path[i + 1] - path[i]
                var met = Set<SIMD2<Int>>()
                for k in bins(p, p + r) {
                    for other in index[k] ?? [] {
                        let c = other.a, e = other.b - other.a, den = r.x * e.y - r.y * e.x
                        guard den != 0 else { continue }
                        let s = ((c.x - p.x) * e.y - (c.y - p.y) * e.x) / den
                        let t = ((c.x - p.x) * r.y - (c.y - p.y) * r.x) / den
                        guard s >= 0, s < 1, t >= 0, t < 1, met.insert(SIMD2(other.path, other.chord)).inserted else { continue }
                        crossings += 1
                        let x = P2<ParamSpace>(p + s * r)
                        let jet = surface.map(x)
                        func space(_ w: SIMD2<Double>) -> SIMD3<Double> { simd_normalize(w.x * jet.du + w.y * jet.dv) }
                        chordWorst = max(chordWorst, asin(min(1, abs(simd_dot(space(r), space(e))))))
                        let along = tangent(path, i, s), across = tangent(families[1][other.path], other.chord, t)
                        tangentWorst = max(tangentWorst, asin(min(1, abs(simd_dot(space(along), space(across))))))
                        if let g = greaterField.value(x), let l = lesserField.value(x) {
                            fieldWorst = max(fieldWorst, asin(min(1, abs(simd_dot(space(g), space(l))))))
                        }
                    }
                }
            }
        }
        let degrees = 180 / Double.pi
        Check.expect(least > 0.3 && curves == 48 && returned == 48 && offSection <= 1e-12,
                     "the forced torus has no umbilic, and each of 24 lines of either family runs once round, from the "
                         + "coordinate curve it was seeded on back to it",
                     String(format: "k₁ − k₂ ≥ %.3f; %d curves, %d returned, ends off the section by %.1e",
                            least, curves, returned, offSection))
        // Each of 24 lines of one family crosses each of the other's about once.
        Check.expect(crossings > 500 && tangentWorst * degrees <= 0.3 && chordWorst * degrees <= 4 && fieldWorst <= 1e-12,
                     "where a line of one family crosses one of the other, they meet at right angles in space",
                     String(format: "%d crossings; off a right angle by at most %.3g° between the curves' tangents, "
                                + "%.3g° between their chords, %.1e rad between the field's directions",
                            crossings, tangentWorst * degrees, chordWorst * degrees, fieldWorst))
    }

    Check.suite("lines of curvature: the manifest") {
        let provenance = Provenance(function: "", example: "", params: [:], cpuCount: 1, gitSha: "")
        let head = #"{"kind": "integralCurves", "field": {"kind": "principalDirection", "of": "greater"}, "#
        let seeds = #""seeds": {"on": {"field": "v", "level": 0}, "at": {"kind": "values", "field": "u", "values": [0, 1, 2]}}, "#
        let source = head + seeds + #""faces": {"kind": "surface"}}"#
        /// A parametric surface's manifest of `request` holding the layer
        /// `text`, read.
        func read(_ text: String, on request: SurfaceRequest = SurfaceRequest(.torus(major: 2, minor: 1))) throws -> SurfaceManifest {
            let empty = SurfaceManifest(surface: request, layers: [], presets: [], provenance: provenance)
            let layer = #"{"name": "x", "role": "scaffold", "width": 0.3, "heightPolicy": "surface", "source": "# + text + "}"
            return try SurfaceManifest(json: JSONValue.parse(Data(empty.canonicalJSON
                .replacingOccurrences(of: #""layers":[]"#, with: #""layers":["# + layer + "]").utf8)))
        }
        func refused(_ text: String, on request: SurfaceRequest = SurfaceRequest(.torus(major: 2, minor: 1)),
                     saying words: String) -> Bool {
            do { _ = try read(text, on: request); return false } catch { return "\(error)".contains(words) }
        }
        // On a surface the height takes any sign: the torus's equators are
        // its level set z = 0, which a meridian crosses twice, the inner one
        // the other way, so it runs on to its seed and closes.
        let equators = try read(source.replacingOccurrences(of: #""on": {"field": "v", "level": 0}"#,
                                                             with: #""on": {"field": "height", "level": 0}"#))
        let crossing: Int
        if case .integralCurves(let field, let seeds, _)? = equators.layers.first?.source {
            crossing = torus.tracedIntegralCurves(of: field, through: torus.seedPoints(seeds), returningTo: torus.section(seeds))
                .filter { $0.forward == .closed }.count
        } else { crossing = 0 }
        // Each meridian crosses both, so each is drawn twice: seeds are not
        // told apart by the curves through them.
        Check.expect(crossing == 6, "seeds on the surface's height at 0, its two equators, at three values of u: "
                         + "three meridians, each from both equators, closed",
                     "\(crossing) closed")
        let m = try read(source)
        let again = try SurfaceManifest(json: JSONValue.parse(Data(m.canonicalJSON.utf8)))
        let expected = LayerSource.integralCurves(field: .principalDirection(of: .greater),
                                                  seeds: Seeds(on: .v, at: 0, .values(of: .u, [0, 1, 2])), faces: .surface)
        Check.expect(m.layers.first?.source == expected && again == m && expected.kind == "lines of curvature",
                     "lines of curvature on a parametric surface decode as written and round-trip")
        let relief = SurfaceRequest(.relief(modulus: 0.64, major: 2, minor: 1, amplitude: 0.4, repeats: SIMD2(1, 1), grid: 64,
                                            turn: 0, floor: nil, soft: false))
        let refusals: [(String, SurfaceRequest?, String)] = [
            (source.replacingOccurrences(of: #""of": "greater""#, with: #""of": "middle""#), nil, "not one of"),
            (source.replacingOccurrences(of: #""kind": "principalDirection", "of": "greater""#,
                                         with: #""kind": "slope", "of": "height""#), nil,
             "slope lines on the surface: they lie on the graph so far"),
            (source.replacingOccurrences(of: #""field": "u""#, with: #""field": "magnitude""#), nil,
             "seeds placed by magnitude on a parametric surface"),
            (source.replacingOccurrences(of: #""on": {"field": "v""#, with: #""on": {"field": "phase""#), nil,
             "seeds placed by phase on a parametric surface"),
            (source.replacingOccurrences(of: #""field": "u""#, with: #""field": "v""#), nil, "they are one function"),
            (source.replacingOccurrences(of: #"{"kind": "values", "field": "u", "values": [0, 1, 2]}"#,
                                         with: #"{"kind": "arcLength", "spacing": 0}"#), nil, "a spacing is positive"),
            (source.replacingOccurrences(of: #"{"kind": "surface"}"#, with: #"{"kind": "plateaus"}"#), nil,
             "integral curves on the plateaus of a parametric surface"),
            (source, relief, "lines of curvature on sn in relief"),
        ]
        let missed = refusals.filter { !refused($0.0, on: $0.1 ?? SurfaceRequest(.torus(major: 2, minor: 1)), saying: $0.2) }
            .map(\.2)
        Check.expect(missed.isEmpty,
                     "a curvature not named, a slope, seeds placed by f's fields or by the field they lie on, a spacing "
                         + "not positive, faces not the surface's, and sn in relief are refused, saying why",
                     "not refused: \(missed)")
        // A landscape's manifest takes none, on the graph or any face.
        func landscape(_ text: String, saying words: String) -> Bool {
            do { _ = try LayerSource(json: JSONValue.parse(Data(text.utf8))); return false } catch { return "\(error)".contains(words) }
        }
        Check.expect(landscape(source.replacingOccurrences(of: #"{"kind": "surface"}"#, with: #"{"kind": "graph", "tiled": false}"#),
                               saying: "lines of curvature on the graph: they lie on a parametric surface")
                         && landscape(source, saying: "this manifest holds a landscape"),
                     "a landscape's manifest refuses lines of curvature, on its graph or a surface")
        // More seeds than a layer's vertices could hold are refused before
        // any is traced.
        let most = ParametricSurface.maxIntegralCurves()
        let many = Seeds(on: .v, at: 0, .values(of: .u, (0..<(most + 1)).map { twoPi * Double($0) / Double(most + 1) }))
        let tooMany: Bool
        do { _ = try torus.integralCurves(of: .principalDirection(of: .greater), seeds: many); tooMany = false }
        catch { tooMany = "\(error)".contains("more than the \(most) a layer holds") }
        Check.expect(most == 419 && tooMany, "more than \(most) seeds, \(2 * 20_000 + 1) vertices each at most, are refused")
    }

    Check.suite("lines of curvature: on the round torus, seen exactly where its coordinate curves are") {
        let drawing = try SurfaceDrawing.build(SurfaceRequest(.torus(major: 2, minor: 1), lines: SIMD2(12, 0)))
        let s = drawing.surface
        let layer = LayerSpec.linesOfCurvature(.greater, over: (s.u, s.v), count: 12, width: 0.3)
        let scene = try drawing.restyled(drawing.layers.map(\.spec) + [layer]).scene(camera: .oblique(SurfaceRequest.projection))
        let seen = scene.visibleLayers()
        guard let lines = seen.first(where: { $0.0.spec.name == "lines" })?.1,
              let meridians = seen.first(where: { $0.0.spec.name == layer.name })?.1,
              let bounds = lines.bounds else { Check.expect(false, "both layers are drawn"); return }
        let size = max(bounds.size.x, bounds.size.y)
        func ends(_ set: PolylineSet<ViewSpace>) -> [SIMD2<Double>] {
            (0..<set.count).flatMap { k in [set[path: k].first!, set[path: k].last!].map { SIMD2($0.x, $0.y) } }
        }
        let a = ends(meridians), b = ends(lines)
        let off = a.map { p in b.map { simd_distance($0, p) }.min() ?? .infinity }.max() ?? .infinity
        let lengths = (meridians.totalLength, lines.totalLength)
        Check.expect(meridians.count == lines.count && meridians.count > 12 && off <= 1e-5 * size
                         && abs(lengths.0 - lengths.1) <= 1e-4 * lengths.1,
                     "the greater family's lines, seen and hidden by the same test, are the meridians as the coordinate curves draw them",
                     String(format: "%d and %d pieces seen; ends within %.1e of the drawing's size; lengths %.6f and %.6f",
                            meridians.count, lines.count, off / size, lengths.0, lengths.1))
    }
}
