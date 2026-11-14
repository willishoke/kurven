import Foundation
import simd
import KurvenCore
import KurvenLandscape
import KurvenMath
import KurvenDynamics

/// Integral curves and slope lines (phase 11b), held to closed forms. Where
/// f is analytic and not zero, log f is analytic too, so the slope lines of
/// |f| are the level sets of arg f: seeded on a contour of |f| at listed
/// angles of arg f, each stays on the phase contour at its angle, to the
/// refiner's tolerance, on sn's landscape, a rational function's and exp(z)'s,
/// whose phase contours are the lines Im z = θ, tiled too. Each stopping rule
/// on its own: a zero of f, a pole, the domain's edge, the footprint, the
/// truncation edge, a closed curve and the step limit. The linear curves on a
/// torus are the constant field's integral curves in closed form; and the
/// manifest, and visibility against the dense cast.
func integralCurveTests() {
    let twoPi = 2 * Double.pi
    func wrapped(_ a: Double) -> Double { a - twoPi * (a / twoPi).rounded() }
    func landscape(_ expression: String, re: (Double, Double), im: (Double, Double), resolution: Int,
                   truncation: Truncation) throws -> KurvenBundle {
        try NativeLandscape.build(
            LandscapeRequest(expression: expression,
                             domain: Domain(real: Interval(lo: re.0, hi: re.1), imag: Interval(lo: im.0, hi: im.1)),
                             resolution: resolution, truncation: truncation, layers: []),
            refine: true)
    }
    /// The cell of a landscape's grid.
    func cell(_ surface: Surface) -> Double {
        let g = surface.height
        return max(abs(g.domain.real.length) / Double(g.width - 1), abs(g.domain.imag.length) / Double(g.height - 1))
    }

    /// Slope lines of |f| seeded on |f| = `level` at `angles` of arg f, held
    /// to the phase contours: the worst distance of a vertex or a chord's
    /// midpoint from the true level set arg f = θ, |arg f − θ| / |∇ arg f|
    /// (|∇ arg f| = |∇ log |f|| by Cauchy–Riemann, read by central
    /// differences), and of a vertex from the refiner's drawn contour at θ,
    /// both in cells; with how many curves and vertices, and why they end.
    func phaseVerdict(_ bundle: KurvenBundle, level: Double, angles: [Double], crossing: [Double] = [],
                      tiles: [Affine2] = [.identity])
        -> (curves: Int, vertices: Int, trueWorst: Double, vertexWorst: Double, drawnWorst: Double,
            ends: [IntegralCurve.End: Int],
            example: String) {
        let surface = bundle.surface, refine = bundle.refine!
        let magnitude = refine.magnitude!, phase = refine.phase!
        let c = cell(surface)
        let seeds = Seeds(on: .magnitude, at: level, .values(of: .phase, angles))
        let traced = surface.graphIntegralCurves(of: .slope(of: .magnitude), seeds: seeds, tiles: tiles,
                                                 region: .full, refine: refine)
        // The refiner's phase contours at each angle, as the drawing has them.
        let rectangle = surface.parameterRectangle!
        let drawn = Contour.levels(of: surface.field(of: .phase, refine: refine)!, angles, over: rectangle)
        func distance(to paths: [[P2<ParamSpace>]], _ p: SIMD2<Double>) -> Double {
            var best = Double.infinity
            for path in paths {
                for (a, b) in zip(path, path.dropFirst()) {
                    let d = b.v - a.v, dd = simd_dot(d, d)
                    let t = dd > 0 ? min(max(simd_dot(p - a.v, d) / dd, 0), 1) : 0
                    best = min(best, simd_distance(a.v + t * d, p))
                }
            }
            return best
        }
        func gradLog(_ p: SIMD2<Double>) -> Double {
            let h = 1e-6 * c
            func l(_ q: SIMD2<Double>) -> Double { log(magnitude(P2(q))) }
            return simd_length(SIMD2(l(p + SIMD2(h, 0)) - l(p - SIMD2(h, 0)), l(p + SIMD2(0, h)) - l(p - SIMD2(0, h)))) / (2 * h)
        }
        var curves = 0, vertices = 0, trueWorst = 0.0, vertexWorst = 0.0, drawnWorst = 0.0, example = "", drawnExample = ""
        var ends: [IntegralCurve.End: Int] = [:]
        for (_, set) in traced.prefix(1) {
            for curve in set {
                curves += 1
                ends[curve.backward, default: 0] += 1
                ends[curve.forward, default: 0] += 1
                let middle = curve.path[curve.path.count / 2].v
                let k = angles.indices.min { abs(wrapped(phase(P2(middle)) - angles[$0])) < abs(wrapped(phase(P2(middle)) - angles[$1])) }!
                let theta = angles[k], contour = drawn[k].paths
                var samples = curve.path.map(\.v)
                samples += zip(curve.path, curve.path.dropFirst()).map { 0.5 * ($0.v + $1.v) }
                // An end on a zero of f, where arg f is undefined: the
                // samples within a thousandth of a cell of it are not read.
                let zeros = [curve.path.first!.v, curve.path.last!.v].filter { magnitude(P2($0)) < 1e-9 }
                for (i, p) in samples.enumerated() {
                    guard !zeros.contains(where: { simd_distance($0, p) < 1e-3 * c }) else { continue }
                    let g = gradLog(p), m = magnitude(P2(p))
                    guard g.isFinite, g > 0, m > 0 else { continue }
                    let off = abs(wrapped(phase(P2(p)) - theta)) / g / c
                    if i < curve.path.count { vertexWorst = max(vertexWorst, off) }
                    if off > trueWorst {
                        trueWorst = off
                        example = String(format: "θ = %.3f at (%.5f, %.5f)", theta, p.x, p.y)
                    }
                    if i < curve.path.count, !crossing.contains(theta) {
                        vertices += 1
                        let d = distance(to: contour, p) / c
                        if d > drawnWorst {
                            drawnWorst = d
                            drawnExample = String(format: " drawn: θ = %.3f at (%.5f, %.5f), |f| %.3g", theta, p.x, p.y, m)
                        }
                    }
                }
            }
        }
        return (curves, vertices, trueWorst, vertexWorst, drawnWorst, ends, example + drawnExample)
    }

    Check.suite("integral curves: slope lines of |f| are the phase contours, to the refiner's tolerance, on sn, a rational function and exp(z)") {
        // arg sn = 0 runs through sn's critical points K + 2mK, where its
        // contours cross and the drawn ones leave the crossing open; a slope
        // line there ends at the critical point, on the true contour, and is
        // held to it alone.
        let angles = [-2.6, -2.0, -1.2, -0.5, 0.0, 0.4, 1.1, 1.9, 2.7]
        for (name, bundle, level, crossing, count) in [
            ("sn(z, 0.64)", try landscape("sn(z, 0.64)", re: (-4, 4), im: (-3.5, 3.5), resolution: 200,
                                          truncation: .uniform(3)), 0.6, [0.0], 34),
            ("(z^2 - 1)/(z^2 + 1)", try landscape("(z^2 - 1)/(z^2 + 1)", re: (-2.5, 2.5), im: (-2, 2), resolution: 160,
                                                  truncation: .uniform(4)), 0.5, [], 18),
            ("exp(z)", try landscape("exp(z)", re: (-1, 1), im: (-1, 1), resolution: 48, truncation: Truncation.none),
             1.0, [], 3),
        ] as [(String, KurvenBundle, Double, [Double], Int)] {
            let v = phaseVerdict(bundle, level: level, angles: angles, crossing: crossing)
            Check.expect(v.curves == count && v.trueWorst <= 0.02 && v.drawnWorst <= 0.02,
                         "\(name): every slope line seeded on |f| = \(level) at an angle of arg f stays on its phase contour",
                         String(format: "%d curves, %d vertices; worst %.1e cell off arg f = θ at a vertex, %.1e at a "
                                    + "chord's midpoint, %.1e off the drawn contour; ",
                                v.curves, v.vertices, v.vertexWorst, v.trueWorst, v.drawnWorst) + "ends \(v.ends) " + v.example)
        }
    }

    Check.suite("integral curves: each stopping rule, on its own") {
        // (z² − 1)/(z² + 1): zeros at ±1, poles at ±i.
        let rational = "(z^2 - 1)/(z^2 + 1)"
        let truncated = try landscape(rational, re: (-2.5, 2.5), im: (-2, 2), resolution: 160, truncation: .uniform(4))
        let h = truncated.surface, refine = truncated.refine!, magnitude = refine.magnitude!
        let c = cell(h)
        let seeds = Seeds(on: .magnitude, at: 0.5, .values(of: .phase, [-2.6, -1.2, 0.4, 1.9]))
        let curves = h.graphIntegralCurves(of: .slope(of: .magnitude), seeds: seeds, tiles: [.identity], region: .full,
                                           refine: refine)[0].curves
        // Downhill into a zero of f, carried to the zero itself; uphill to
        // the truncation edge, where |f| is the truncation height.
        var atZero = 0, onEdge = 0, wrong = 0
        var zeroOff = 0.0, edgeOff = 0.0
        for curve in curves {
            for (end, p) in [(curve.backward, curve.path.first!), (curve.forward, curve.path.last!)] {
                let m = magnitude(P2(p.v))
                switch end {
                case .singular:
                    atZero += 1
                    zeroOff = max(zeroOff, min(simd_distance(p.v, SIMD2(1, 0)), simd_distance(p.v, SIMD2(-1, 0))))
                case .boundary:
                    onEdge += 1
                    edgeOff = max(edgeOff, abs(m - 4) / 4)
                default: wrong += 1
                }
            }
        }
        Check.expect(curves.count == 8 && atZero == 8 && onEdge == 8 && wrong == 0 && zeroOff <= 1e-12 && edgeOff <= 1e-12,
                     "on (z² − 1)/(z² + 1) truncated at 4, each slope line runs down into a zero of f and up to the truncation edge, exactly",
                     String(format: "%d curves, %d at a zero (off by %.1e), %d on the edge (|f| off 4 by %.1e), %d otherwise",
                            curves.count, atZero, zeroOff, onEdge, edgeOff, wrong))
        // Untruncated, uphill into a pole: it ends within a thousandth of a
        // cell of it, where the step it would take shrinks below the least.
        let open = try landscape(rational, re: (-2.5, 2.5), im: (-2, 2), resolution: 160, truncation: Truncation.none)
        let poleCurves = open.surface.graphIntegralCurves(of: .slope(of: .magnitude), seeds: seeds, tiles: [.identity],
                                                          region: .full, refine: open.refine)[0].curves
        let poleOff = poleCurves.map { min(simd_distance($0.path.last!.v, SIMD2(0, 1)),
                                           simd_distance($0.path.last!.v, SIMD2(0, -1))) / c }.max() ?? .infinity
        // Into a pole the field turns against any step that would cross
        // it, so the steps shrink to the least, the curve at the pole: the
        // lifted chords' sag is refined only so far, and a steep spire does
        // not stop it.
        Check.expect(poleCurves.count == 8 && poleCurves.allSatisfy { $0.forward == .singular } && poleOff <= 1e-3,
                     "untruncated, each runs up into a pole of f and ends at it, under a thousandth of a cell short",
                     String(format: "%d curves, worst %.1e cell short; ends ", poleCurves.count, poleOff)
                         + "\(poleCurves.map(\.forward))")
        // exp(z): the slope lines are the lines Im z = θ, out to the
        // domain's edge both ways; and to a footprint's edge, a diamond,
        // where one cuts the copy.
        let exp = try landscape("exp(z)", re: (-1, 1), im: (-1, 1), resolution: 48, truncation: Truncation.none)
        let lines = exp.surface.graphIntegralCurves(of: .slope(of: .magnitude),
                                                    seeds: Seeds(on: .magnitude, at: 1, .values(of: .phase, [-0.7, 0.2, 0.9])),
                                                    tiles: [.identity], region: .full, refine: exp.refine)[0].curves
        let edgeEnds = lines.map { max(abs($0.path.first!.x + 1), abs($0.path.last!.x - 1)) }.max() ?? .infinity
        Check.expect(lines.count == 3 && lines.allSatisfy { $0.backward == .boundary && $0.forward == .boundary } && edgeEnds <= 1e-12,
                     "on exp(z) each runs from the domain's edge at Re z = −1 to its edge at Re z = 1, to the last bit",
                     String(format: "%d curves, ends off the edge by %.1e", lines.count, edgeEnds))
        let diamond = BoundaryPerimeter(edges: [(0.8, 0.0), (0.0, 0.8), (-0.8, 0.0), (0.0, -0.8)].enumerated().map { k, a in
            let b = [(0.8, 0.0), (0.0, 0.8), (-0.8, 0.0), (0.0, -0.8)][(k + 1) % 4]
            return PerimeterEdge(start: P2(a.0, a.1), end: P2(b.0, b.1), density: 8)
        })
        let cut = exp.surface.graphIntegralCurves(of: .slope(of: .magnitude),
                                                  seeds: Seeds(on: .magnitude, at: 1, .values(of: .phase, [-0.5, 0.2, 0.6])),
                                                  tiles: [.identity], region: .inside(diamond), refine: exp.refine)[0].curves
        let ends: [SIMD2<Double>] = cut.flatMap { [$0.path.first!.v, $0.path.last!.v] }
        let footprintOff = ends.map { (p: SIMD2<Double>) -> Double in abs(abs(p.x) + abs(p.y) - 0.8) }.max() ?? .infinity
        Check.expect(cut.count == 3 && cut.allSatisfy { $0.backward == .boundary && $0.forward == .boundary } && footprintOff <= 1e-12,
                     "cut to a footprint, each ends on the footprint's perimeter where the copy stands, to the last bit",
                     String(format: "%d curves, ends off the diamond by %.1e", cut.count, footprintOff))
        // The slope of arg f: its integral curves are the contours of |f|,
        // closed round the zeros of f. Seeded on arg f = 0.3 at |f| = 0.2
        // and 0.4, each closes on its seed, on |f| = c all round.
        let loops = h.graphIntegralCurves(of: .slope(of: .phase),
                                          seeds: Seeds(on: .phase, at: 0.3, .values(of: .magnitude, [0.2, 0.4])),
                                          tiles: [.identity], region: .full, refine: refine)[0].curves
        var loopOff = 0.0
        for loop in loops {
            let level = [0.2, 0.4].min { abs(magnitude(P2(loop.path[0].v)) - $0) < abs(magnitude(P2(loop.path[0].v)) - $1) }!
            for p in loop.path {
                let g = refine.magnitude!, e = 1e-6 * c
                let grad = simd_length(SIMD2(g(P2(p.x + e, p.y)) - g(P2(p.x - e, p.y)), g(P2(p.x, p.y + e)) - g(P2(p.x, p.y - e)))) / (2 * e)
                loopOff = max(loopOff, abs(magnitude(P2(p.v)) - level) / grad / c)
            }
        }
        Check.expect(loops.count == 4 && loops.allSatisfy { $0.backward == .closed && $0.forward == .closed && $0.path.first == $0.path.last }
                         && loopOff <= 0.02,
                     "the slope lines of arg f close round the zeros on their seeds, on the contours of |f|",
                     String(format: "%d loops, worst %.1e cell off |f| = c; ends ", loops.count, loopOff)
                         + "\(loops.map { "\($0.backward)/\($0.forward)" })")
        // The step limit: each way stops after so many steps.
        var capped = IntegralCurve.Limits()
        capped.maxSteps = 5
        let short = exp.surface.graphIntegralCurves(of: .slope(of: .magnitude),
                                                    seeds: Seeds(on: .magnitude, at: 1, .values(of: .phase, [0.2])),
                                                    tiles: [.identity], region: .full, refine: exp.refine, limits: capped)[0].curves
        Check.expect(short.count == 1 && short[0].backward == .stepLimit && short[0].forward == .stepLimit && short[0].path.count == 11,
                     "at the step limit each way ends after that many steps",
                     "\(short.map { "\($0.backward)/\($0.forward), \($0.path.count) vertices" })")
        // A seed off the trimmed domain, over the truncation, starts
        // nothing.
        let graph = h.graph(refine: refine)!
        let field = graph.heightSlope()
        Check.expect(graph.integralCurve(of: field, through: P2(0, 1.02), on: graph.copies[0]) == nil
                         && graph.integralCurve(of: field, through: P2(3, 0), on: graph.copies[0]) == nil
                         && graph.integralCurve(of: field, through: P2(0.5, 0.5), on: graph.copies[0]) != nil,
                     "a seed over the truncation, or off the domain, starts no curve")
    }

    Check.suite("integral curves: the slope of a coordinate is raised by the graph's metric, and a steep graph is followed to its end") {
        // The graph of |exp(z²/2)| = h = e^{(x² − y²)/2}: h_x = x h, h_y = −y h.
        // Its first fundamental form is g = I + ∇h ∇hᵀ, so the slope of u is
        // g⁻¹e₁ ∝ (1 + h_y², −h_x h_y) and of v g⁻¹e₂ ∝ (−h_x h_y, 1 + h_x²):
        // not along the axes where h_x h_y ≠ 0, as the plane's gradient is.
        let b = try landscape("exp(z^2/2)", re: (-1, 1), im: (-1, 1), resolution: 48, truncation: Truncation.none)
        let graph = b.surface.graph(refine: b.refine)!
        let su = graph.slope(of: { $0.x }), sv = graph.slope(of: { $0.y })
        var worst = 0.0, turned = 0.0, sampled = 0
        for i in 0..<9 {
            for j in 0..<9 {
                let x = -0.9 + 0.225 * Double(i), y = -0.9 + 0.225 * Double(j)
                let h = exp((x * x - y * y) / 2), hx = x * h, hy = -y * h
                for (field, closed) in [(su, SIMD2(1 + hy * hy, -hx * hy)), (sv, SIMD2(-hx * hy, 1 + hx * hx))] {
                    guard let w = field.value(P2(x, y)) else { worst = .infinity; continue }
                    sampled += 1
                    worst = max(worst, simd_length(simd_normalize(w) - simd_normalize(closed)))
                    turned = max(turned, 1 - abs(simd_normalize(closed).x == 0 ? 1 : simd_dot(simd_normalize(closed), simd_normalize(SIMD2(closed.x, 0)))))
                }
            }
        }
        // The general rule, the adjugate of g from the jet, agrees where the
        // graph is gentle.
        var general = 0.0
        for p in [P2<ParamSpace>(0.3, -0.4), P2(-0.7, 0.2), P2(0.5, 0.5)] {
            let jet = graph.map(p)
            for d in [SIMD2(1.0, 0), SIMD2(0, 1.0), SIMD2(0.3, -2)] {
                let x = Face.raised(d, overGraphOf: SIMD2(jet.du.z, jet.dv.z))!, y = Face.raised(d, jet)!
                general = max(general, simd_length(x - y) / simd_length(x))
            }
        }
        Check.expect(sampled == 162 && worst <= 1e-8 && turned > 0.05 && general <= 1e-12,
                     "on |exp(z²/2)|'s graph the slope of u is ∝ (1 + h_y², −h_x h_y) and of v ∝ (−h_x h_y, 1 + h_x²), g⁻¹ raised",
                     String(format: "%d samples, worst %.1e off the closed form's direction, which leaves its axis by up to %.2f; "
                                + "the jet's adjugate agrees to %.1e", sampled, worst, turned, general))
        // exp(20z): its slope lines are the lines Im z = θ, straight in the
        // plane, on a graph that rises to e^20. The chord's sag in space is
        // refined at most five halvings below the longest step, so the line
        // reaches the domain's edge at Re z = 1.
        let steep = try landscape("exp(20*z)", re: (-1, 1), im: (-0.3, 0.3), resolution: 48, truncation: Truncation.none)
        let line = steep.surface.graphIntegralCurves(of: .slope(of: .magnitude),
                                                     seeds: Seeds(on: .magnitude, at: 1, .values(of: .phase, [0])),
                                                     tiles: [.identity], region: .full, refine: steep.refine)[0].curves
        let c = cell(steep.surface)
        let offAxis = line.first.map { $0.path.map { abs($0.y) }.max()! } ?? .infinity
        Check.expect(line.count == 1 && line[0].backward == .boundary && line[0].forward == .boundary
                         && abs(line[0].path.first!.x + 1) <= 1e-12 && abs(line[0].path.last!.x - 1) <= 1e-12
                         && line[0].path.count <= Int(2 / c * 64) && offAxis <= 1e-9,
                     "on exp(20z) the slope line runs from Re z = −1 to Re z = 1, to the domain's edge both ways, its vertices bounded",
                     line.map { String(format: "%@/%@ from %.6f to %.6f, %d vertices (%.0f a cell), off Im z = 0 by %.1e",
                                       "\($0.backward)", "\($0.forward)", $0.path.first!.x, $0.path.last!.x,
                                       $0.path.count, Double($0.path.count) / (2 / c), offAxis) }.joined())
    }

    Check.suite("integral curves: seeds once each, and no closer than the grid resolves") {
        let rational = try landscape("(z^2 - 1)/(z^2 + 1)", re: (-2.5, 2.5), im: (-2, 2), resolution: 160,
                                     truncation: .uniform(4))
        func count(_ values: [Double]) -> Int {
            rational.surface.graphIntegralCurves(of: .slope(of: .magnitude), seeds: Seeds(on: .magnitude, at: 0.5, .values(of: .phase, values)),
                                                 tiles: [.identity], region: .full, refine: rational.refine)[0].curves.count
        }
        let one = count([1]), cut = count([.pi]), twice = [count([1, 1 + 2 * .pi]), count([.pi, -.pi]), count([1, 1])]
        Check.expect(one == 2 && cut == 2 && twice == [2, 2, 2],
                     "values the same modulo 2π seed once: |f| = 1/2 round each zero meets arg f = 1 once, and π once",
                     "arg f = 1: \(one), π: \(cut); 1 and 1 + 2π, π and −π, 1 twice: \(twice)")
        // A value read exactly at a vertex: once, whether the field crosses
        // it there (−, 0, +) or touches it (−, 0, −), and a closed path's
        // first vertex once.
        let line = [P2<ParamSpace>(0, 0), P2(1, 0), P2(2, 0)]
        let seedsAt = Seeds(on: .magnitude, at: 1, .values(of: .u, [0]))
        let touch = seedsAt.points(along: [line], value: { -($0.x - 1) * ($0.x - 1) }) { SIMD3($0.x, $0.y, 0) }
        let cross = seedsAt.points(along: [line], value: { $0.x - 1 }) { SIMD3($0.x, $0.y, 0) }
        let square = [P2<ParamSpace>(0, 0), P2(1, 0), P2(1, 1), P2(0, 1), P2(0, 0)]
        let corner = seedsAt.points(along: [square], value: { $0.x + $0.y }) { SIMD3($0.x, $0.y, 0) }
        Check.expect(touch == [P2(1, 0)] && cross == [P2(1, 0)] && corner == [P2(0, 0)],
                     "a value read at a vertex seeds the vertex once, touched or crossed, and a closed path's first vertex once",
                     "touched \(touch), crossed \(cross), closed \(corner)")
        // By arc length a piece of the level set takes round(L / spacing)
        // seeds: none under half the spacing; and a ratio no grid resolves
        // seeds nothing rather than trapping.
        let unit = [P2<ParamSpace>(0, 0), P2(1, 0)]
        func spaced(_ s: Double) -> Int {
            Seeds(on: .magnitude, at: 1, .arcLength(s)).points(along: [unit], value: nil) { SIMD3($0.x, $0.y, 0) }.count
        }
        Check.expect(spaced(3) == 0 && spaced(1.5) == 1 && spaced(0.25) == 4 && spaced(1e-300) == 0 && spaced(.leastNonzeroMagnitude) == 0,
                     "a unit stretch spaced 3 apart takes no seed, 1.5 one, a quarter four, and 1e-300 none, without trapping",
                     "\([3, 1.5, 0.25, 1e-300].map(spaced))")
        // Every path refuses seeds closer than a cell: the layer's own
        // check, given the cell, and the derivation, which draws nothing.
        let c = cell(rational.surface)
        let close = Seeds(on: .magnitude, at: 0.5, .arcLength(0.5 * c))
        let refused = LayerSource.unsupported(.slope(of: .magnitude), close, on: .graph(tiled: false), cell: c)
        let drawn = rational.surface.integralCurves(of: .slope(of: .magnitude), seeds: close, on: .graph(tiled: false),
                                                    in: Surface.HatchContext(perimeter: nil), refine: rational.refine)
        let far = rational.surface.integralCurves(of: .slope(of: .magnitude), seeds: Seeds(on: .magnitude, at: 0.5, .arcLength(2 * c)),
                                                  on: .graph(tiled: false), in: Surface.HatchContext(perimeter: nil),
                                                  refine: rational.refine)
        Check.expect(refused?.contains("closer than a cell of the grid") == true && drawn.isEmpty && !far.isEmpty,
                     "seeds closer than a cell are refused by the layer's check and draw nothing; two cells apart they draw",
                     "\(refused ?? "accepted"); \(drawn.count) and \(far.count) curves")
    }

    Check.suite("integral curves: seeds by arc length, and on a tiled graph each copy's own") {
        // exp(z) over [−1, 1]²: |f| = 1 is the segment Re z = 0, length 2
        // in space, so seeds a quarter apart are the middles of its eight
        // quarters, and the slope lines through them Im z = −7/8 ... 7/8.
        let expLandscape = try landscape("exp(z)", re: (-1, 1), im: (-1, 1), resolution: 48, truncation: Truncation.none)
        let even = expLandscape.surface.graphIntegralCurves(of: .slope(of: .magnitude), seeds: Seeds(on: .magnitude, at: 1, .arcLength(0.25)),
                                                   tiles: [.identity], region: .full, refine: expLandscape.refine)[0].curves
        let ys = even.map { $0.path[0].y }.sorted()
        let expected = (0..<8).map { (k: Int) -> Double in -1 + 0.25 * (Double(k) + 0.5) }
        let flat = even.flatMap { curve in curve.path.map { abs($0.y - curve.path[0].y) } }.max() ?? .infinity
        Check.expect(ys.count == 8 && zip(ys, expected).allSatisfy { abs($0 - $1) <= 1e-9 } && flat <= 1e-9,
                     "on exp(z), seeds a quarter apart along |f| = 1 are the middles of its quarters, each slope line level in Im z",
                     String(format: "%d seeds at Im z = ", ys.count) + ys.map { String(format: "%.4f", $0) }.joined(separator: ", ")
                         + String(format: "; Im z varies by %.1e along a line", flat))
        // Tiled by a reflection across Re z = 2 and a quarter turn moved to
        // Re z = 8: every copy's slope lines are its image of Im z = θ, at
        // |f| = e^x of the point of the domain it came from.
        let tiles = [Affine2.identity, Affine2(a: -1, b: 0, tx: 4, c: 0, d: 1, ty: 0),
                     Affine2(a: 0, b: -1, tx: 8, c: 1, d: 0, ty: 0)]
        let angles = [-0.7, 0.2, 0.9]
        let paths = expLandscape.surface.integralCurves(of: .slope(of: .magnitude),
                                               seeds: Seeds(on: .magnitude, at: 1, .values(of: .phase, angles)),
                                               on: .graph(tiled: true),
                                               in: Surface.HatchContext(perimeter: nil, region: .full, tiles: tiles),
                                               refine: expLandscape.refine)
        var off = 0.0, offZ = 0.0, lost = 0, perTile = [0, 0, 0]
        for path in paths {
            var tile: Int?
            for w in path {
                guard let k = tiles.indices.first(where: { k in
                    let q = tiles[k].inverse!(w)
                    return abs(q.x) <= 1 + 1e-9 && abs(q.y) <= 1 + 1e-9
                }) else { lost += 1; continue }
                if tile == nil { tile = k; perTile[k] += 1 }
                let q = tiles[k].inverse!(w)
                off = max(off, angles.map { abs(q.y - $0) }.min()!)
                offZ = max(offZ, abs(w.z - exp(q.x)) / exp(q.x))
            }
        }
        Check.expect(paths.count == 9 && perTile == [3, 3, 3] && lost == 0 && off <= 1e-9 && offZ <= 1e-12,
                     "on a tiled graph each copy's slope lines are its image of Im z = θ, lifted to |f|",
                     String(format: "%d paths, worst %.1e off Im z = θ, %.1e off |f|; per tile ", paths.count, off, offZ)
                         + "\(perTile)")
        // sn on two tiles, its phase contours held as on one.
        let sn = try landscape("sn(z, 0.64)", re: (-4, 4), im: (-3.5, 3.5), resolution: 200, truncation: .uniform(3))
        let twoTiles = [Affine2.identity, Affine2(a: 1, b: 0, tx: 8, c: 0, d: 1, ty: 0)]
        let tiled = sn.surface.integralCurves(of: .slope(of: .magnitude),
                                              seeds: Seeds(on: .magnitude, at: 0.6, .values(of: .phase, [-2.0, 1.1])),
                                              on: .graph(tiled: true),
                                              in: Surface.HatchContext(perimeter: nil, region: .full, tiles: twoTiles),
                                              refine: sn.refine)
        let phase = sn.refine!.phase!, snMagnitude = sn.refine!.magnitude!, snCell = cell(sn.surface)
        var snOff = 0.0, perCopy = [0, 0]
        for path in tiled {
            // The copy a path lies on, read at its middle, off the tile
            // boundary its ends can stand on.
            let k = path[path.count / 2].x > 4 ? 1 : 0
            perCopy[k] += 1
            for w in path {
                let q = twoTiles[k].inverse!(w)
                let p = SIMD2(q.x, q.y), e = 1e-6 * snCell
                func l(_ r: SIMD2<Double>) -> Double { log(snMagnitude(P2(r))) }
                let g = simd_length(SIMD2(l(p + SIMD2(e, 0)) - l(p - SIMD2(e, 0)), l(p + SIMD2(0, e)) - l(p - SIMD2(0, e)))) / (2 * e)
                guard snMagnitude(P2(p)) > 1e-9, g.isFinite, g > 0, 1 / g > 1e-3 * snCell else { continue }
                snOff = max(snOff, [-2.0, 1.1].map { abs(wrapped(phase(P2(p)) - $0)) }.min()! / g / snCell)
            }
        }
        Check.expect(perCopy[0] == perCopy[1] && perCopy[0] >= 4 && snOff <= 1e-5,
                     "sn's slope lines on two tiles, each carried back by its tile, lie on their phase contours on both",
                     String(format: "%d paths, worst %.1e cell off arg f = θ; per copy ", tiled.count, snOff) + "\(perCopy)")
    }

    Check.suite("integral curves: a linear curve is the constant field's, in closed form, to the bit") {
        let w = SIMD2<Double>(0.1, 0.37), field = VectorField.constant(w)
        let p = P2<ParamSpace>(0.25, -1.5)
        Check.expect((0...50).allSatisfy { m in field.flow!(p, Double(m)) == P2(p.x + Double(m) * w.x, p.y + Double(m) * w.y) }
                         && field.value(P2(3, 4)) == w,
                     "the constant field's flow is p + t w, each coordinate to the bit")
        var request = SurfaceRequest(.torus(major: 2, minor: 1))
        request.grid = 64
        let torus = try SurfaceDrawing.build(request).surface
        let slope = 0.4, turns = 5, count = 2
        let curves = torus.linearCurves(slope: slope, turns: turns, count: count)
        let steps = max(torus.u.cells, Int((abs(slope) * Double(torus.v.cells)).rounded(.up)))
        let du = torus.u.range.length / Double(steps), dv = slope * torus.v.range.length / Double(steps)
        var first: [P2<ParamSpace>] = []
        for k in 0..<curves.count { first += Array(curves.coords(path: k)!) }
        // The first strand's first path, before any seam: its vertex m is
        // the start moved by m steps.
        let strand = Array(curves.coords(path: 0)!)
        let exact = strand.indices.allSatisfy { m in
            strand[m] == P2(torus.u.range.lo + Double(m) * du, torus.v.range.lo + Double(m) * dv)
        }
        Check.expect(exact && strand.count > 10 && !first.isEmpty,
                     "the torus's linear curve is the constant field's integral curve, its vertices the closed form's to the bit",
                     "\(strand.count) vertices")
    }

    Check.suite("integral curves: the manifest") {
        func source(_ text: String) throws -> LayerSource { try LayerSource(json: JSONValue.parse(Data(text.utf8))) }
        func refused(_ text: String, saying words: String) -> Bool {
            do { _ = try source(text); return false } catch { return "\(error)".contains(words) }
        }
        let head = #"{"kind": "integralCurves", "field": {"kind": "slope", "of": "magnitude"}, "#
        let graph = #""faces": {"kind": "graph", "tiled": true}}"#
        let byValues = head + #""seeds": {"on": {"field": "magnitude", "level": 0.6}, "at": {"kind": "values", "field": "phase", "values": [-1, 0.5]}}, "# + graph
        let byLength = head + #""seeds": {"on": {"field": "u", "level": 0.25}, "at": {"kind": "arcLength", "spacing": 0.3}}, "# + graph
        let a = try source(byValues), b = try source(byLength)
        let aAgain = try source(a.json.canonical), bAgain = try source(b.json.canonical)
        Check.expect(a == .integralCurves(field: .slope(of: .magnitude),
                                          seeds: Seeds(on: .magnitude, at: 0.6, .values(of: .phase, [-1, 0.5])),
                                          faces: .graph(tiled: true))
                         && b == .integralCurves(field: .slope(of: .magnitude), seeds: Seeds(on: .u, at: 0.25, .arcLength(0.3)),
                                                 faces: .graph(tiled: true))
                         && aAgain == a && bAgain == b
                         && a.kind == "slope lines",
                     "integral curves seeded at values or by arc length decode as written and round-trip")
        let refusals: [(String, String)] = [
            (byValues.replacingOccurrences(of: #""level": 0.6"#, with: #""level": 0"#), "f's zeros or nothing"),
            (byValues.replacingOccurrences(of: #""level": 0.6"#, with: #""level": -1"#), "f's zeros or nothing"),
            (byValues.replacingOccurrences(of: "[-1, 0.5]", with: "[]"), "no values"),
            (byValues.replacingOccurrences(of: #""field": "phase""#, with: #""field": "magnitude""#), "they are one function"),
            (byValues.replacingOccurrences(of: #""field": "phase", "values": [-1, 0.5]"#,
                                           with: #""field": "height", "values": [0.5]"#), "they are one function"),
            (byValues.replacingOccurrences(of: #""on": {"field": "magnitude""#, with: #""on": {"field": "height""#)
                .replacingOccurrences(of: #""field": "phase", "values": [-1, 0.5]"#,
                                      with: #""field": "magnitude", "values": [0.5]"#), "they are one function"),
            (byValues.replacingOccurrences(of: #""on": {"field": "magnitude""#, with: #""on": {"field": "phase""#)
                .replacingOccurrences(of: #""field": "phase", "values": [-1, 0.5]"#,
                                      with: #""field": "height", "values": [0, 1]"#), "it is positive"),
            (byValues.replacingOccurrences(of: #""field": "phase""#, with: #""field": "isophote""#), "isophote"),
            (byValues.replacingOccurrences(of: #""of": "magnitude""#, with: #""of": "isophote""#), "no light"),
            (byValues.replacingOccurrences(of: #""of": "magnitude""#, with: #""of": "curl""#), "not one of"),
            (byValues.replacingOccurrences(of: #""kind": "slope""#, with: #""kind": "principal""#), "not one of"),
            (byLength.replacingOccurrences(of: #""spacing": 0.3"#, with: #""spacing": 0"#), "a spacing is positive"),
            (byLength.replacingOccurrences(of: #""spacing": 0.3"#, with: #""spacing": -2"#), "a spacing is positive"),
            (byLength.replacingOccurrences(of: #""kind": "arcLength""#, with: #""kind": "evenly""#), "not one of"),
            (byValues.replacingOccurrences(of: #"{"kind": "graph", "tiled": true}"#, with: #"{"kind": "plateaus"}"#),
             "they lie on the graph so far"),
            (byValues.replacingOccurrences(of: #"{"kind": "graph", "tiled": true}"#, with: #"{"kind": "walls", "edges": [0]}"#),
             "they lie on the graph so far"),
            (byValues.replacingOccurrences(of: #"{"kind": "graph", "tiled": true}"#, with: #"{"kind": "surface"}"#),
             "this manifest holds a landscape"),
        ]
        let missed = refusals.filter { !refused($0.0, saying: $0.1) }.map(\.1)
        Check.expect(missed.isEmpty,
                     "seeds on |f| or the height at 0 or below, at no values, at values of the function they lie on, by any name, or a height not "
                         + "positive, by an isophote, the slope of an isophote, a field or kind unknown, a spacing not positive, "
                         + "and integral curves off the graph are refused, saying why",
                     "not refused: \(missed)")
        let nan = byValues.replacingOccurrences(of: #""level": 0.6"#, with: #""level": 1e999"#)
        Check.expect((try? source(nan)) == nil, "a level that is not a number is refused")
        // A parametric surface's manifest and a polyhedron's take none.
        let empty = SurfaceManifest(surface: SurfaceRequest(.torus(major: 2, minor: 1)), layers: [], presets: [],
                                    provenance: Provenance(function: "", example: "", params: [:], cpuCount: 1, gitSha: ""))
        let layer = #"{"name": "x", "role": "scaffold", "width": 0.3, "heightPolicy": "surface", "source": "# + byValues + "}"
        let onSurface: Bool
        do {
            _ = try SurfaceManifest(json: JSONValue.parse(Data(empty.canonicalJSON
                .replacingOccurrences(of: #""layers":[]"#, with: #""layers":["# + layer + "]").utf8)))
            onSurface = false
        } catch { onSurface = "\(error)".contains("integral curves on a parametric surface") }
        Check.expect(onSurface, "a parametric surface's manifest refuses integral curves, saying they lie on the graph so far")
        // A landscape's manifest: seeds closer than a cell of its grid.
        let cubic = try Data(contentsOf: Fixtures.dir.appendingPathComponent("expr/landscape_cubic.kurven/manifest.json"))
        let m = try Manifest(json: JSONValue.parse(cubic))
        var withLayer = m
        withLayer.layers.append(LayerSpec(name: "slope", role: .scaffold,
                                          source: .integralCurves(field: .slope(of: .magnitude),
                                                                  seeds: Seeds(on: .magnitude, at: 1, .arcLength(1e-4)),
                                                                  faces: .graph(tiled: false)),
                                          width: 0.3, heightPolicy: .surface))
        let tooClose: Bool
        do { _ = try Manifest(json: JSONValue.parse(Data(withLayer.canonicalJSON.utf8))); tooClose = false }
        catch { tooClose = "\(error)".contains("closer than a cell of the grid") }
        withLayer.layers[withLayer.layers.count - 1].source = .integralCurves(
            field: .slope(of: .magnitude), seeds: Seeds(on: .magnitude, at: 1, .arcLength(0.5)), faces: .graph(tiled: false))
        let roundTrip = try Manifest(json: JSONValue.parse(Data(withLayer.canonicalJSON.utf8)))
        Check.expect(tooClose && roundTrip == withLayer,
                     "a landscape's manifest refuses seeds closer than a cell of its grid, and keeps them a cell apart or more")
    }

    Check.suite("integral curves: on f, seen exactly where the dense cast against f says") {
        let built = try NativeLandscape.build(LandscapeRequest(preset: Catalog.native.preset("cubic")!, resolution: 64))
        var m = built.manifest
        m.layers = [
            LayerSpec(name: "slope", role: .scaffold,
                      source: .integralCurves(field: .slope(of: .magnitude),
                                              seeds: Seeds(on: .magnitude, at: 0.5,
                                                           .values(of: .phase, (0..<12).map { -Double.pi + Double($0) * .pi / 6 })),
                                              faces: .graph(tiled: false)),
                      width: 0.3, heightPolicy: .surface),
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
        let cellSize = max(abs(grid.domain.real.length) / Double(grid.width - 1), abs(grid.domain.imag.length) / Double(grid.height - 1))
        var compared = 0, wrong = 0, seen = 0, example = ""
        for (layer, visible) in scene.visibleLayers() {
            guard let bounds = AABB(visible.vertices + layer.paths.mapped(view).vertices) else { continue }
            let size = max(bounds.size.x, bounds.size.y)
            let index = SegmentIndex(visible, lo: SIMD2(bounds.lo.x, bounds.lo.y) - 1e-6 * size, size: 1.00001 * size)
            let vertices = layer.paths.vertices
            let truths = Heightfield.parallelMap(vertices) { v -> Bool? in
                let f = vis.facing(v.xy)
                guard abs(f) > 1e-9 else { return nil }
                guard f > 0 else { return false }
                return denselyHidden(h, magnitude: magnitude, from: v.v, along: toward, depthTolerance: vis.depthTolerance,
                                     perUnit: perUnit, cell: cellSize) == nil
            }
            for (v, truth) in zip(vertices, truths) {
                let p = view(v).v
                guard let truth, index.nearestEnd(p) >= 1e-6 * size else { continue }
                compared += 1
                if truth { seen += 1 }
                if truth != index.holds(p, tolerance: 1e-9 * size) {
                    wrong += 1
                    if example.isEmpty {
                        example = String(format: " e.g. at (%.4f, %.4f, %.4f): the cast says %@", v.x, v.y, v.z,
                                         truth ? "seen" : "hidden")
                    }
                }
            }
        }
        Check.expect(wrong == 0 && compared > 1000 && seen > 300 && seen < compared,
                     "every vertex of the slope lines is drawn exactly where the dense cast sees it",
                     "\(compared) compared, \(seen) seen, \(wrong) wrong" + example)
    }
}
