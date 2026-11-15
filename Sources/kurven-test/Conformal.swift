import Foundation
import simd
import KurvenCore
import KurvenMath
import KurvenDynamics

/// sn on its conformal torus (phase 11e): isothermal coordinates round the
/// tube, on the one round torus whose flat rectangle has the aspect of sn's,
/// so the modulus pins the minor radius. A conformal map keeps angles, and
/// the level sets of |f| and arg f of an analytic f cross at right angles in
/// the plane, so on this torus they cross at right angles in space. Held two
/// ways that do not share a derivative: at the drawn contours' own crossings,
/// their tangents pushed through the map's jet, as the drawing computes it;
/// and the map's conformality from differences of its positions alone, with
/// the tube angle against a quadrature of ∫ r dθ / (R + r cos θ). Then the
/// drawn chords, which the turning of the contours bounds, and the request.
func conformalTorusTests() {
    let m = 0.64, R = 2.0
    let K = Jacobi.quarterPeriod(m), Kp = Jacobi.quarterPeriod(1 - m)
    let r = SurfaceRequest.Shape.conformalMinor(modulus: m, major: R)
    let conformal = SurfaceRequest(.sn(modulus: m, major: R, minor: r, grid: 600, conformal: true))
    /// The catalog's sn, on the same ring: the negative control.
    let plain = SurfaceRequest(.sn(modulus: m, major: R, minor: 1, grid: 600))

    Check.suite("conformal sn: the modulus pins r/R, with the real period round the ring") {
        // K and K′ as Jacobi computes them, through the pin's own copy of
        // the mean, to the bit, over the whole range of the modulus.
        var unequal: [Double] = []
        for m in [1e-15, 1e-9, 1e-4, 0.01, 0.1, 0.25, 0.5, 0.64, 0.75, 0.9, 0.99, 0.999999, 1 - 1e-12] {
            let a = Jacobi.quarterPeriod(1 - m) / (2 * Jacobi.quarterPeriod(m))
            if SurfaceRequest.Shape.conformalMinor(modulus: m, major: 3) != 3 * a / (1 + a * a).squareRoot() {
                unequal.append(m)
            }
        }
        Check.expect(unequal.isEmpty, "the pin reads K and K′ as Jacobi.quarterPeriod does, bit for bit",
                     "unequal at \(unequal)")
        let drawing = try SurfaceDrawing.build(conformal)
        let s = drawing.surface
        // Round the ring along u: z and the distance from the axis stay put
        // and the point turns a quarter about z in a quarter of 4K. Round the
        // tube along v: half of 2K′ takes the outer equator to the inner.
        let p0 = s.map(P2(0, 0)).position, pu = s.map(P2(K, 0)).position, pv = s.map(P2(0, Kp)).position
        let along = s.u.range.length == 4 * K && s.v.range.length == 2 * Kp
            && abs(pu.z - p0.z) < 1e-15 && abs(simd_length(pu) - simd_length(p0)) < 1e-14
            && abs(simd_dot(simd_normalize(pu), simd_normalize(p0))) < 1e-15
            && abs(p0.x - (R + r)) < 1e-15 && abs(pv.x - (R - r)) < 1e-14 && abs(pv.z) < 1e-14
        Check.expect(along, "4K runs round the ring and 2K′ round the tube, from the outer equator to the inner and back")
        // The flat torus's aspect, tube over ring, is the rectangle's.
        let aspect = r / ((R - r) * (R + r)).squareRoot(), want = Kp / (2 * K)
        Check.expect(abs(aspect / want - 1) < 1e-15 && abs(r / R - want / (1 + want * want).squareRoot()) < 1e-15,
                     "r / √(R² − r²) = K′ / (2K): r/R = a / √(1 + a²), a = K′ / (2K)",
                     String(format: "r/R = %.15f at m = %g; aspect off by %.1e", r / R, m, abs(aspect / want - 1)))
    }

    Check.suite("conformal sn: the tube angle is the closed form's, by quadrature, across the seam") {
        // w(θ) = ∫₀^θ r dθ′ / (R + r cos θ′) by Simpson's rule, against v
        // laid evenly in w: w = (v / 2K′) · 2πr / √(R² − r²). θ is read off the
        // position, not the closed form; unwrapped coordinates run a turn
        // past the seam.
        let s = try SurfaceDrawing.build(conformal).surface
        func w(_ theta: Double) -> Double {
            let n = 4000, h = theta / Double(n)
            func g(_ x: Double) -> Double { r / (R + r * cos(x)) }
            var sum = g(0) + g(theta)
            for i in 1..<n { sum += Double(i % 2 == 1 ? 4 : 2) * g(Double(i) * h) }
            return sum * h / 3
        }
        let turn = 2 * Double.pi * r / ((R - r) * (R + r)).squareRoot()
        var worst = 0.0
        let vs = (0...64).map { 2 * Kp * Double($0) / 64 } + [1e-12, 2 * Kp - 1e-12, Kp - 1e-9, Kp + 1e-9,
                                                              2 * Kp + 0.3, 4 * Kp - 1e-12, -0.4]
        for v in vs {
            let p = s.map(P2(0.7, v)).position
            var theta = atan2(p.z, (p.x * p.x + p.y * p.y).squareRoot() - R)
            if theta < 0 { theta += 2 * .pi }
            // Whole turns, from the coordinate, then the angle within one,
            // which by a seam can read a turn either side.
            let turns = (v / (2 * Kp)).rounded(.down), part = v / (2 * Kp) - turns
            if theta > .pi, part < 0.25 { theta -= 2 * .pi }
            if theta < .pi, part > 0.75 { theta += 2 * .pi }
            worst = max(worst, abs(w(theta) + turns * turn - v / (2 * Kp) * turn))
        }
        Check.expect(worst < 1e-10, "w of the tube angle at v is v's share of w's turn, 2πr / √(R² − r²)",
                     String(format: "%d coordinates, seams and unwrapped included; worst %.1e", vs.count, worst))
    }

    /// The crossings of a drawing's |f| contours with its arg f contours:
    /// each the point of the parameter rectangle where a chord of one meets
    /// a chord of the other, with the chords in space and how far each
    /// stroke turns at the chord's ends.
    struct Crossing {
        var c: SIMD2<Double>
        var chords: (SIMD3<Double>, SIMD3<Double>)
        var turning: Double
    }
    func crossings(_ drawing: SurfaceDrawing) -> [Crossing] {
        struct Chord { var a, b: SIMD2<Double>; var world: SIMD3<Double>; var turning: Double }
        func chords(_ prefix: String) -> [Chord] {
            var out: [Chord] = []
            for layer in drawing.layers where layer.spec.name.hasPrefix(prefix) {
                guard let coords = layer.paths.coords else { continue }
                let o = layer.paths.offsets, x = layer.paths.vertices
                for k in 0..<(o.count - 1) where o[k + 1] - o[k] >= 2 {
                    func turn(_ i: Int) -> Double {
                        guard i > o[k], i < o[k + 1] - 1 else { return 0 }
                        let d0 = simd_normalize(x[i].v - x[i - 1].v), d1 = simd_normalize(x[i + 1].v - x[i].v)
                        return acos(min(max(simd_dot(d0, d1), -1), 1))
                    }
                    for i in o[k]..<(o[k + 1] - 1) {
                        out.append(Chord(a: coords[i].v, b: coords[i + 1].v, world: x[i + 1].v - x[i].v,
                                         turning: max(turn(i), turn(i + 1))))
                    }
                }
            }
            return out
        }
        let magnitude = chords("mag_"), phase = chords("ang_")
        // Phase chords in buckets of the rectangle, each in every bucket its
        // box touches.
        let size = SIMD2(drawing.surface.u.range.length, drawing.surface.v.range.length)
        let lo = SIMD2(drawing.surface.u.range.lo, drawing.surface.v.range.lo)
        let n = SIMD2(256, 128)
        func bucket(_ p: SIMD2<Double>) -> SIMD2<Int> {
            let q = (p - lo) / size * SIMD2(Double(n.x), Double(n.y))
            return SIMD2(min(max(Int(q.x.rounded(.down)), 0), n.x - 1), min(max(Int(q.y.rounded(.down)), 0), n.y - 1))
        }
        var buckets = [[Int]](repeating: [], count: n.x * n.y)
        for (i, chord) in phase.enumerated() {
            let b0 = bucket(simd_min(chord.a, chord.b)), b1 = bucket(simd_max(chord.a, chord.b))
            for y in b0.y...b1.y { for x in b0.x...b1.x { buckets[y * n.x + x].append(i) } }
        }
        func cross(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double { a.x * b.y - a.y * b.x }
        var out: [Crossing] = []
        for chord in magnitude {
            let b0 = bucket(simd_min(chord.a, chord.b)), b1 = bucket(simd_max(chord.a, chord.b))
            var seen = Set<Int>()
            for y in b0.y...b1.y {
                for x in b0.x...b1.x {
                    for j in buckets[y * n.x + x] where seen.insert(j).inserted {
                        let other = phase[j]
                        let d = chord.b - chord.a, e = other.b - other.a, den = cross(d, e)
                        guard den != 0 else { continue }
                        let t = cross(other.a - chord.a, e) / den, s = cross(other.a - chord.a, d) / den
                        guard (0...1).contains(t), (0...1).contains(s) else { continue }
                        out.append(Crossing(c: chord.a + t * d, chords: (chord.world, other.world),
                                            turning: chord.turning + other.turning))
                    }
                }
            }
        }
        return out
    }
    /// How far from a right angle, in radians, the level sets of |sn| and
    /// arg sn through `c` cross in space on `surface`: their tangents in the
    /// parameter plane, each square to its field's gradient there, pushed
    /// through the map's jet. With g = sn′ / sn = cn dn / sn, the gradient
    /// of log |f| is (Re g, −Im g) and of arg f (Im g, Re g). Nil where f is
    /// zero, a pole, or critical.
    func offSquare(_ c: SIMD2<Double>, on surface: ParametricSurface) -> Double? {
        let f = Jacobi.complex(Complex(c.x, c.y), m)
        let g = f.cn * f.dn / f.sn
        guard g.isFinite, g.magnitude > 1e-8, g.magnitude < 1e8 else { return nil }
        let magnitude = SIMD2(g.re, -g.im), phase = SIMD2(g.im, g.re)
        let tm = SIMD2(-magnitude.y, magnitude.x), tp = SIMD2(-phase.y, phase.x)
        let jet = surface.map(P2(c))
        let a = tm.x * jet.du + tm.y * jet.dv, b = tp.x * jet.du + tp.y * jet.dv
        return abs(asin(min(max(simd_dot(a, b) / (simd_length(a) * simd_length(b)), -1), 1)))
    }

    Check.suite("conformal sn: wherever an |f| contour crosses an arg f contour, the angle in space is π/2") {
        let drawing = try SurfaceDrawing.build(conformal), control = try SurfaceDrawing.build(plain)
        let found = crossings(drawing)
        var worst = 0.0, counted = 0, seams = 0, inside = 0, outside = 0
        for x in found {
            guard let off = offSquare(x.c, on: drawing.surface) else { continue }
            counted += 1
            worst = max(worst, off)
            let u = x.c.x / (4 * K), v = x.c.y / (2 * Kp)
            if min(u, 1 - u, v, 1 - v) < 0.02 { seams += 1 }
            let p = drawing.surface.map(P2(x.c)).position
            let cosine = ((p.x * p.x + p.y * p.y).squareRoot() - R) / r
            if cosine < -0.9 { inside += 1 }
            if cosine > 0.9 { outside += 1 }
        }
        Check.expect(counted > 700 && seams > 50 && inside > 100 && outside > 50 && worst <= 1e-6,
                     "at every crossing of the drawn contours, over the torus, the tangents meet at π/2 in space",
                     String(format: "%d crossings (%d by a seam, %d on the tube's inside, %d outside); worst %.1e rad",
                            counted, seams, inside, outside, worst))
        // And through every point of a grid laid on the seams themselves,
        // on the equators and between.
        var dense = 0.0, points = 0
        for j in 0...96 {
            for i in 0...192 {
                let c = SIMD2(4 * K * Double(i) / 192, 2 * Kp * Double(j) / 96 + (j == 96 ? -1e-12 : 0))
                guard let off = offSquare(c, on: drawing.surface) else { continue }
                points += 1
                dense = max(dense, off)
            }
        }
        Check.expect(points > 18000 && dense <= 1e-6, "and through every point of a grid that lies on the seams",
                     String(format: "%d points; worst %.1e rad", points, dense))
        // The negative control: the catalog's sn on the same ring, its tube
        // in its angle.
        var missed = 0.0, controls = 0
        for x in crossings(control) {
            guard let off = offSquare(x.c, on: control.surface) else { continue }
            controls += 1
            missed = max(missed, off)
        }
        Check.expect(controls > 700 && missed > 0.3,
                     "on the catalog's torus, r = 1, the same contours cross far from square",
                     String(format: "%d crossings; worst %.3f rad (%.1f°)", controls, missed, missed * 180 / .pi))
        // The chords themselves. A chord's direction is the mean of its
        // curve's tangent over the arc it spans, so it lies within the arc's
        // turning of the tangent at any point of it; the turning is bounded
        // here by the most the stroke turns at either end of the chord.
        func chordsOff(_ x: Crossing) -> Double {
            let (a, b) = x.chords
            return abs(asin(min(max(simd_dot(a, b) / (simd_length(a) * simd_length(b)), -1), 1)))
        }
        // sn's critical points, sn′ = cn dn = 0, where |f| = 1 and arg f
        // = 0 or π both cross themselves and their strokes turn corners.
        let critical = [SIMD2(K, 0), SIMD2(3 * K, 0), SIMD2(K, Kp), SIMD2(3 * K, Kp)]
        func fromCritical(_ c: SIMD2<Double>) -> Double {
            critical.map { p in
                let d = c - p
                return simd_length(SIMD2(d.x - (d.x / (4 * K)).rounded() * 4 * K, d.y - (d.y / (2 * Kp)).rounded() * 2 * Kp))
            }.min()!
        }
        var over = 0, chordWorst = 0.0, worstFrom = 0.0, bound = 0.0, resolved = 0, resolvedWorst = 0.0
        var away = 0.0
        for x in found {
            let off = chordsOff(x)
            if off > chordWorst { chordWorst = off; worstFrom = fromCritical(x.c) }
            if fromCritical(x.c) > 0.05 * K { away = max(away, off) }
            bound = max(bound, x.turning)
            if off > x.turning + 1e-9 { over += 1 }
            if x.turning <= 2 * .pi / 180 { resolved += 1; resolvedWorst = max(resolvedWorst, off) }
        }
        let missedByChords = crossings(control).filter { chordsOff($0) > $0.turning + 0.05 }.count
        Check.expect(over == 0 && resolved > 200 && resolvedWorst < 2 * .pi / 180 && missedByChords > 200,
                     "the drawn chords cross square to within the strokes' turning there; the control's do not",
                     String(format: "%d crossings, worst %.3f° (%.1e from a critical point of sn), the strokes turning "
                            + "up to %.1f°; %.3f° farther than K/20 from one; %d where the strokes turn under 2°, "
                            + "worst %.3f°; %d control crossings off by 3° more than their turning", found.count,
                            chordWorst * 180 / .pi, worstFrom, bound * 180 / .pi, away * 180 / .pi, resolved,
                            resolvedWorst * 180 / .pi, missedByChords))
    }

    Check.suite("conformal sn: the map is conformal by differences of its positions alone") {
        let s = try SurfaceDrawing.build(conformal).surface, control = try SurfaceDrawing.build(plain).surface
        let hu = 1e-5 * 4 * K, hv = 1e-5 * 2 * Kp
        func differences(_ surface: ParametricSurface, _ c: SIMD2<Double>) -> (SIMD3<Double>, SIMD3<Double>) {
            func p(_ du: Double, _ dv: Double) -> SIMD3<Double> { surface.map(P2(c.x + du, c.y + dv)).position }
            return ((p(hu, 0) - p(-hu, 0)) / (2 * hu), (p(0, hv) - p(0, -hv)) / (2 * hv))
        }
        func offConformal(_ xu: SIMD3<Double>, _ xv: SIMD3<Double>) -> Double {
            max(abs(simd_dot(xu, xv)) / (simd_length(xu) * simd_length(xv)), abs(simd_length(xu) / simd_length(xv) - 1))
        }
        var worst = 0.0, jetOff = 0.0, controlWorst = 0.0, secondOff = 0.0, footOff = 0.0
        for j in 0...48 {
            for i in 0...48 {
                let c = SIMD2(4 * K * Double(i) / 48, 2 * Kp * Double(j) / 48)
                let (xu, xv) = differences(s, c)
                worst = max(worst, offConformal(xu, xv))
                let jet = s.map(P2(c))
                jetOff = max(jetOff, simd_length(jet.du - xu) / simd_length(xu), simd_length(jet.dv - xv) / simd_length(xv))
                let (yu, yv) = differences(control, c)
                controlWorst = max(controlWorst, offConformal(yu, yv))
                // The second derivatives, against central differences of
                // the first, as 11c held the round torus's.
                if let second = s.secondOrderJet {
                    let e = second(P2(c)).second
                    let uu = (s.map(P2(c.x + hu, c.y)).du - s.map(P2(c.x - hu, c.y)).du) / (2 * hu)
                    let uv = (s.map(P2(c.x, c.y + hv)).du - s.map(P2(c.x, c.y - hv)).du) / (2 * hv)
                    let vv = (s.map(P2(c.x, c.y + hv)).dv - s.map(P2(c.x, c.y - hv)).dv) / (2 * hv)
                    let scale = simd_length(xu) * simd_length(xu) / (R + r)
                    secondOff = max(secondOff, simd_length(e.uu - uu) / scale, simd_length(e.uv - uv) / scale,
                                    simd_length(e.vv - vv) / scale)
                } else {
                    secondOff = .infinity
                }
                // The solid's foot reads each point back to its coordinate.
                if let foot = s.neighbourhood?.foot(jet.position) {
                    let d = foot.coordinate.v - c
                    let wrapped = SIMD2(d.x - (d.x / (4 * K)).rounded() * 4 * K, d.y - (d.y / (2 * Kp)).rounded() * 2 * Kp)
                    footOff = max(footOff, simd_length(wrapped), abs(foot.distance))
                } else {
                    footOff = .infinity
                }
            }
        }
        Check.expect(worst < 1e-8 && controlWorst > 0.3,
                     "x_u · x_v = 0 and |x_u| = |x_v| by central differences of positions; not on the catalog's torus",
                     String(format: "worst %.1e over 2401 points, seams included; the control off by %.2f",
                            worst, controlWorst))
        Check.expect(jetOff < 1e-8 && secondOff < 1e-7,
                     "the jet the drawing pushes tangents through is the positions' differences, and so are its second derivatives",
                     String(format: "first off by %.1e, second by %.1e of their size", jetOff, secondOff))
        Check.expect(footOff < 1e-12, "the solid's foot reads each point back to its own coordinate, across the seams",
                     String(format: "worst %.1e", footOff))
    }

    Check.suite("conformal sn: the request pins the minor radius, saves only when on, and refuses the rest") {
        func refused(_ request: SurfaceRequest, _ field: SurfaceRequest.Field, _ value: Double) -> String? {
            do { _ = try request.setting(field, to: value); return nil } catch { return "\(error)" }
        }
        let on = try plain.setting(.conformal, to: 1)
        let wider = try conformal.setting(.modulus, to: 0.3), larger = try conformal.setting(.major, to: 3)
        Check.expect(on == conformal && on[.conformal] == 1 && plain[.conformal] == 0
                     && wider[.minor] == SurfaceRequest.Shape.conformalMinor(modulus: 0.3, major: R)
                     && larger[.minor] == SurfaceRequest.Shape.conformalMinor(modulus: m, major: 3)
                     && (try conformal.setting(.conformal, to: 0)) == SurfaceRequest(.sn(modulus: m, major: R, minor: r,
                                                                                          grid: 600))
                     && conformal.drafted().conformal && SurfaceRequest(.torus(major: 2, minor: 1))[.conformal] == nil,
                     "switched on, the modulus pins the minor radius, and a modulus or a major radius moves it")
        Check.expect(conformal.fields == [.modulus, .major, .grid] && plain.fields == [.modulus, .major, .minor, .grid],
                     "a conformal sn lists no minor radius among its sliders")
        Check.expect(refused(conformal, .minor, 1)?.contains("surface.shape.minor is pinned by the modulus on a conformal "
                                                             + "sn, to \(r) round a ring of 2.0, not 1.0") == true
                     && refused(plain, .conformal, 0.5)?.contains("surface.shape.conformal must be 1 (on) or 0 (off), not 0.5") == true
                     && refused(conformal, .modulus, 1.5)?.contains("modulus must lie strictly between 0 and 1") == true
                     && refused(conformal, .modulus, 1e-300)?.contains("too near 0 for sn to be drawn conformally") == true,
                     "a minor radius set outright, a switch at neither 0 nor 1, and a modulus it cannot pin by are refused")
        // Saved: the key only when on, and a bundle saved before it reads
        // as off. Read back, the same request.
        let text = conformal.json.canonical
        let back = try SurfaceRequest(json: JSONValue.parse(text))
        let old = try Fixtures.json("contract/surface.kurven/manifest.json")
        let saved = try SurfaceRequest(json: old["surface"]!)
        Check.expect(back == conformal && back.json.canonical == text && text.contains("\"conformal\":true")
                     && !plain.json.canonical.contains("conformal") && !saved.conformal,
                     "the switch is written only when on, read back exactly, and a bundle without it is off")
        func shape(_ change: (inout [String: JSONValue]) -> Void, of request: SurfaceRequest) -> JSONValue {
            var o = (try? request.json.object("request")) ?? [:]
            var s = (try? o["shape"]!.object("shape")) ?? [:]
            change(&s)
            o["shape"] = .object(s)
            return .object(o)
        }
        let relief = SurfaceRequest(.relief(modulus: m, major: R, minor: 1, amplitude: 0.3, repeats: SIMD2(1, 1),
                                            grid: 120, turn: 0, floor: nil, soft: false))
        var message: [String] = []
        for json in [shape({ $0["conformal"] = .bool(true) }, of: relief),
                     shape({ $0["conformal"] = .bool(false) }, of: SurfaceRequest(.torus(major: 2, minor: 1))),
                     shape({ $0["minor"] = .double(0.8) }, of: conformal),
                     shape({ $0["conformal"] = .int(1) }, of: plain)] {
            do { _ = try SurfaceRequest(json: json); message.append("read") } catch { message.append("\(error)") }
        }
        Check.expect(message[0].contains("conformal on a relief shape: only plain sn")
                     && message[1].contains("conformal on a torus shape") && message[2].contains("is pinned by the modulus")
                     && message[3].contains("should be a boolean"),
                     "read, conformal is refused on any other shape, with a minor radius it does not pin, or not a boolean",
                     message.joined(separator: " | "))
    }
}
