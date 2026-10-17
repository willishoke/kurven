import Foundation
import Synchronization
import KurvenCore
import KurvenMath
import KurvenLandscape

// MARK: - 10. the expression language and its functions

/// The native evaluator against the oracles -- numpy for the elementary
/// functions, scipy.special where it has a complex form, mpmath where it does
/// not -- at every point of every fixture grid. `tests/oracles.py` names the
/// library behind each case, and each case records it.
///
/// The comparison is `|a - b| <= tol * max(|b|, 1)`: relative where the value
/// is large, absolute where it is small, so a zero of a function is not held to
/// a relative error it cannot have, and both sides agreeing that a point is a
/// pole is agreement. The tolerance is per case, stated in the fixture, and it
/// is a statement about the algorithm: 1e-12 where this side runs the
/// oracle's own method or one as good, 1e-9 for the Bessel and Airy families,
/// whose engines are different from AMOS and agree with it to about 1e-11
/// across these windows.
func expressionTests() {
    Check.suite("expression: the functions agree with numpy, scipy and mpmath on the fixture grids") {
        let index = try Fixtures.json("expr/index.json")
        let cases = try index.array("cases", "expr index")
        // Every function in the table has a case, or is listed as held to
        // something else; the oracle script refuses to write a table that
        // leaves one out, and this is the same rule read from the other side.
        let named = Set(try cases.map { try $0.object("case").string("name", "case") })
        let unheld: [String: JSONValue] = (try? index.value("unheld", "expr index")
                                                .object("unheld")) ?? [:]
        let uncovered = Catalog.native.functions.map(\.name).filter {
            !named.contains("table_\($0)") && unheld[$0] == nil
        }
        Check.expect(uncovered.isEmpty, "every function in the table has an oracle",
                     uncovered.joined(separator: ", "))
        var worstOverall = 0.0
        var failed = 0
        for entry in cases {
            let e = try entry.object("case")
            let name = try e.string("name", "case")
            let text = try e.string("expression", "case")
            let real = try e.doubles("real", "case"), imag = try e.doubles("imag", "case")
            let tol = try e.double("tol", "case")
            let nr = Int(real[2]), ni = Int(imag[2])
            let rs = NativeLandscape.linspace(real[0], real[1], nr)
            let is_ = NativeLandscape.linspace(imag[0], imag[1], ni)
            let want = try NPY.read(contentsOf: Fixtures.url("expr/\(try e.string("file", "case"))"))
                .complexPairs()
            let f = try KurvenMath.Expression.compile(text)
            var worst = 0.0
            var worstAt = Complex.zero
            var disagreements = 0
            for (i, r) in rs.enumerated() {
                for (j, im) in is_.enumerated() {
                    let z = Complex(r, im)
                    let a = f(z)
                    let k = 2 * (i * ni + j)
                    let b = Complex(want[k], want[k + 1])
                    // A NaN from the oracle is the library declining to answer
                    // (Γ at a pole it hit exactly), not a value to be held to;
                    // an infinity is a value, and both sides must agree on it.
                    if b.isNaN { continue }
                    if !a.isFinite || !b.isFinite {
                        if a.isFinite != b.isFinite {
                            disagreements += 1
                            if disagreements == 1 { worstAt = z; worst = .infinity }
                        }
                        continue
                    }
                    let err = (a - b).magnitude / max(b.magnitude, 1)
                    if err > worst { worst = err; worstAt = z }
                    if err > tol { disagreements += 1 }
                }
            }
            worstOverall = max(worstOverall, worst.isFinite ? worst : 0)
            if disagreements > 0 { failed += 1 }
            let oracle = (try? e.string("oracle", "case")) ?? "?"
            Check.expect(disagreements == 0, "\(name): \(text) against \(oracle)",
                         String(format: "worst %.1e at %@ (tol %g)%@", worst, "\(worstAt)", tol,
                                disagreements == 0 ? "" : ", \(disagreements) points over"))
        }
        Check.expect(failed == 0, "all \(cases.count) cases agree",
                     String(format: "worst %.1e", worstOverall))
    }

    // `language.json` is frozen and edited by hand: the readings two
    // reasonable people disagree about, the character each malformed
    // expression is refused at, and the function table. It was last written
    // by the Python implementation of the language, and it is the record of
    // what this one was ported to read.
    Check.suite("expression: the parser reads the frozen readings") {
        let index = try Fixtures.json("expr/language.json")
        let canonical = try index.value("canonical", "language").object("canonical")
        let grid = try index.value("grid", "language").object("grid")
        let gr = try grid.doubles("real", "grid"), gi = try grid.doubles("imag", "grid")
        let points = NativeLandscape.linspace(gr[0], gr[1], Int(gr[2])).flatMap { r in
            NativeLandscape.linspace(gi[0], gi[1], Int(gi[2])).map { Complex(r, $0) }
        }
        var wrong: [String] = []
        var unlike: [String] = []
        for (source, want) in canonical.sorted(by: { $0.key < $1.key }) {
            guard case .string(let wanted) = want else { wrong.append("\(source): not a string"); continue }
            let got = try KurvenMath.Expression.canonical(source)
            if got != wanted { wrong.append("\(source) -> \(got), frozen \(wanted)") }
            // And the canonical form is a fixed point that means the same thing.
            let again = try KurvenMath.Expression.canonical(got)
            if again != got { wrong.append("\(got) is not a fixed point (\(again))") }
            let f = try KurvenMath.Expression.compile(source)
            let g = try KurvenMath.Expression.compile(wanted)
            let apart = points.contains { z in
                let a = f(z), b = g(z)
                if a.isNaN && b.isNaN { return false }
                if !a.isFinite || !b.isFinite { return a.isFinite != b.isFinite }
                return (a - b).magnitude > 1e-12 * max(b.magnitude, 1)
            }
            if apart { unlike.append("\(source) / \(wanted)") }
        }
        Check.expect(wrong.isEmpty, "\(canonical.count) canonical spellings",
                     wrong.joined(separator: "; "))
        Check.expect(unlike.isEmpty, "and each spelling evaluates to its reading's values",
                     unlike.joined(separator: "; "))

        let errors = try index.array("errors", "language")
        var misplaced: [String] = []
        for entry in errors {
            let e = try entry.object("error")
            let text = try e.string("text", "error")
            let parses = (try? e.bool("parses", "error", default: false)) ?? false
            do {
                _ = try KurvenMath.Expression.parse(text)
                if !parses { misplaced.append("'\(text)' parsed") }
            } catch let error as ExpressionError {
                if parses { misplaced.append("'\(text)' was refused: \(error)"); continue }
                let position: Int? = if case .some(.int(let p)) = e["position"] { p } else { nil }
                let length = (try? e.int("length", "error")) ?? 1
                if error.position != position || error.length != length {
                    misplaced.append("'\(text)' refused at \(error.position.map(String.init) ?? "nil")"
                                     + "+\(error.length), frozen \(position.map(String.init) ?? "nil")"
                                     + "+\(length)")
                }
            }
        }
        Check.expect(misplaced.isEmpty, "\(errors.count) malformed expressions refused at the same character",
                     misplaced.joined(separator: "; "))
        do {
            _ = try KurvenMath.Expression.parse("gama(z)")
            Check.expect(false, "a near miss suggests the name it nearly is")
        } catch let error as ExpressionError {
            Check.expect(error.message.contains("gamma"),
                         "a near miss suggests the name it nearly is", error.message)
        }
    }

    Check.suite("expression: the language table is the frozen one") {
        let index = try Fixtures.json("expr/language.json")
        let language = try index.value("language", "language").object("language")
        let functions = try language.array("functions", "language").map(LanguageFunction.init(json:))
        let mine = Catalog.native.functions
        Check.expect(functions.map(\.name) == mine.map(\.name),
                     "the same \(functions.count) functions, in the same order",
                     functions.count == mine.count ? ""
                        : "\(mine.count) here: " + mine.map(\.name).joined(separator: " "))
        var differ: [String] = []
        for (theirs, ours) in zip(functions, mine) where theirs != ours {
            differ.append("\(theirs.name): arity \(theirs.arity)/\(ours.arity), aliases "
                          + "\(theirs.aliases)/\(ours.aliases), help \(theirs.help == ours.help ? "same" : "differs")")
        }
        Check.expect(differ.isEmpty, "with the same arities, aliases and help",
                     differ.joined(separator: "; "))
        let constants = try language.array("constants", "language").map {
            try $0.object("constant").string("name", "constant")
        }
        Check.expect(constants == Catalog.native.constants, "and the same constants",
                     constants.joined(separator: ", "))
        let variable = try language.string("variable", "language")
        Check.expect(variable == "z", "and z is the variable")
    }
}

// MARK: - 11. the native landscape

/// `NativeLandscape.build` against the bundles it wrote for the same specs
/// when they were last regenerated (`kurven-cli fixtures`): the same manifest,
/// the same grids, the same ink. The bundles are golden. They prove nothing
/// about correctness, only that a change to a landscape was made on purpose.
/// When the Python builder was retired, the grids written here were
/// byte-identical to its own and the manifests differed only in gitSha.
func landscapeTests() {
    Check.suite("landscape: built here, it is the committed bundle") {
        let index = try Fixtures.json("expr/landscapes.json")
        for entry in try index.array("landscapes", "landscapes") {
            let e = try entry.object("landscape")
            let name = try e.string("bundle", "landscape")
            let spec = try LandscapeRequest(json: e.value("spec", "landscape"))
            let bundle = Fixtures.url("expr/\(name)")
            let theirs = try KurvenBundle.read(at: bundle)
            let clock = ContinuousClock()
            var ours: KurvenBundle!
            let elapsed = try clock.measure {
                ours = try NativeLandscape.build(spec, gitSha: theirs.manifest.provenance.gitSha,
                                                 refine: false)
            }
            let shape = theirs.manifest.height.shape
            let label = "\(name) (\(spec.expression), \(shape.nx)x\(shape.ny), \(elapsed))"

            let same = ours.manifest.canonicalJSON == theirs.manifest.canonicalJSON
            Check.expect(same, "\(label): the same manifest",
                         same ? "" : firstDifference(theirs.manifest.canonicalJSON,
                                                     ours.manifest.canonicalJSON))

            func compare(_ a: Grid2D<Float>?, _ b: Grid2D<Float>?, _ what: String) {
                guard let a, let b else {
                    Check.expect(a == nil && b == nil, "\(what): both present or both absent")
                    return
                }
                guard a.values.count == b.values.count else {
                    Check.expect(false, "\(what): the same shape", "\(a.values.count) vs \(b.values.count)")
                    return
                }
                var worst = 0.0
                var off = 0
                for (x, y) in zip(a.values, b.values) {
                    let err = Double(abs(x - y)) / max(Double(abs(y)), 1)
                    worst = max(worst, err)
                    if err > 1e-6 { off += 1 }
                }
                Check.expect(off == 0, "\(what): the same samples",
                             String(format: "worst %.1e%@", worst, off == 0 ? "" : ", \(off) differ"))
            }
            compare(ours.surface.height, theirs.surface.height, "height")
            compare(ours.surface.phase, theirs.surface.phase, "phase")

            var inkDiffers: [String] = []
            for (mine, committed) in zip(ours.layers, theirs.layers) {
                let a = mine.paths.inkLength, b = committed.paths.inkLength
                if mine.paths.count != committed.paths.count || abs(a - b) > 1e-6 * max(b, 1) {
                    inkDiffers.append("\(mine.spec.name) \(mine.paths.count)/\(committed.paths.count) paths, "
                                      + String(format: "length %.6g/%.6g", a, b))
                }
            }
            Check.expect(inkDiffers.isEmpty, "and every layer derives to the same ink",
                         inkDiffers.joined(separator: "; "))
        }
    }

    Check.suite("landscape: the derived numbers") {
        // A cap is chosen only when there is a spire to cut.
        let flat = (0..<1000).map { Double($0) / 1000 }
        Check.expect(NativeLandscape.defaultCaps(flat) == .none,
                     "a bounded function is not truncated")
        var spire = flat
        spire[500] = 400
        if case .uniform(let z) = NativeLandscape.defaultCaps(spire) {
            Check.expect(abs(z - 1.0) < 1e-12, "a spire is cut near the 99th percentile", "\(z)")
        } else {
            Check.expect(false, "a spire is cut near the 99th percentile")
        }
        let domain = Domain(real: Interval(lo: -2, hi: 2), imag: Interval(lo: -1, hi: 1))
        let shape = NativeLandscape.gridShape(domain, resolution: 100)
        Check.expect(shape.nReal == 100 && shape.nImag == 50,
                     "the grid keeps its cells square", "\(shape)")
        let phase = NativeLandscape.phaseMinor
        Check.expect(phase.count == 8 && abs(phase[0] + 2.6179938779914944) < 1e-15,
                     "the minor phase levels are numpy's linspace, bit for bit",
                     "\(phase.first ?? 0)")
    }
}

/// Where two texts first part company, with a little context either side.
func firstDifference(_ a: String, _ b: String) -> String {
    let x = Array(a), y = Array(b)
    var i = 0
    while i < min(x.count, y.count), x[i] == y[i] { i += 1 }
    let from = max(0, i - 60), to = min(min(x.count, y.count), i + 60)
    return "at \(i): committed …\(String(x[from..<min(x.count, to)]))… built …\(String(y[from..<min(y.count, to)]))…"
}

// MARK: - 12. contour refinement

/// Contours placed by f: the vertices land on the level set, the chords stay
/// within tolerance of it, the phase wraps are gone, and a restyle keeps all
/// of that.
func refinementTests() {
    Check.suite("root: Brent's method converges where bisection would, and faster") {
        var calls = 0
        func counted(_ f: @escaping (Double) -> Double) -> (Double) -> Double {
            { x in calls += 1; return f(x) }
        }
        // Dottie's number: bisection needs forty halvings for 1e-12.
        let dottie = 0.7390851332151607
        let x = Root.brent(counted { cos($0) - $0 }, 0, 1, fa: 1, fb: cos(1) - 1, xtol: 1e-12)
        Check.expect(x.map { abs($0 - dottie) <= 1e-12 } ?? false && calls <= 10,
                     "cos x = x, to 1e-12 in a handful of evaluations",
                     "\(x ?? .nan) after \(calls)")
        // A bracket like exp(1/z)'s beside its essential singularity: -1 at
        // one end, 1e76 at the other, where interpolation lands beside the
        // small end every time and only bisection makes progress.
        calls = 0
        let steep: (Double) -> Double = { exp(250 * ($0 - 0.3)) - 1 }
        let y = Root.brent(counted(steep), 0, 1, fa: steep(0), fb: steep(1), xtol: 1e-12)
        Check.expect(y.map { abs($0 - 0.3) <= 1e-11 } ?? false && calls <= 120,
                     "a bracket of -1 against 1e76 still converges",
                     "\(y ?? .nan) after \(calls)")
        Check.expect(Root.brent({ $0 }, 0, 1, fa: 0, fb: 1, xtol: 1e-12) == 0,
                     "a zero at an end is that end")
        Check.expect(Root.brent({ $0 * $0 + 1 }, -1, 1, fa: 2, fb: 2, xtol: 1e-12) == nil,
                     "ends of one sign bracket nothing")
        Check.expect(Root.brent({ $0 < 0.5 ? -1 : .nan }, 0, 1, fa: -1, fb: 1, xtol: 1e-12) == nil,
                     "a value that is not finite ends the search")
    }

    Check.suite("refine: contours placed by the function") {
        let request = LandscapeRequest(preset: Catalog.native.preset("gamma")!, resolution: 300)
        let (bundle, reports) = try NativeLandscape.benchmarkRefinement(request)
        for r in reports {
            let what = "\(r.layer) (\(r.field.rawValue), \(r.levels) levels)"
            switch r.field {
            case .magnitude:
                // At rounding, but for a vertex solved at a saddle of |f|,
                // which is accepted within a hundredth of the tolerance.
                Check.expect(r.refinedVertex.max <= 1e-2 * 0.02 + 1e-9,
                             "\(what): every vertex is on the level set",
                             String(format: "max %.1e cells, was %.3f", r.refinedVertex.max, r.gridVertex.max))
                Check.expect(r.refinedChord.p95 <= 0.02 + 1e-9 && r.refinedChord.max <= 0.03,
                             "\(what): the chords stay within tolerance",
                             String(format: "p95 %.4f max %.4f cells, was max %.3f",
                                    r.refinedChord.p95, r.refinedChord.max, r.gridChord.max))
            case .phase:
                Check.expect(r.wrapVertices > 0, "\(what): the grid had vertices on the wrap",
                             "\(r.wrapVertices) of \(r.gridVertices)")
                // Within the float32 grid's own precision: an edge the grid
                // saw a crossing on but f does not is snapped to its nearer
                // end, which is where the level set passes within rounding.
                Check.expect(r.refinedVertex.max < 1e-3,
                             "\(what): and the refined vertices are all on the level set",
                             String(format: "max %.1e cells, was %.1f", r.refinedVertex.max, r.gridVertex.max))
            }
            Check.expect(r.refineSeconds < 1.0, "\(what): in reasonable time",
                         String(format: "%.1f ms for %d vertices", r.refineSeconds * 1e3, r.refinedVertices))
        }
        // A restyle re-derives through the refiner rather than losing it.
        let refined = try NativeLandscape.build(request)
        Check.expect(refined.refine != nil, "a native landscape carries its refiner")
        var manifest = refined.manifest
        manifest.caps = .uniform(3)
        let restyled = refined.restyled(manifest)
        Check.expect(restyled.refine != nil, "and a restyled one still does")
        let plain = try NativeLandscape.build(request, refine: false)
        let a = restyled.layers.first { $0.spec.name == "mag_major" }!.paths
        let b = plain.restyled(manifest).layers.first { $0.spec.name == "mag_major" }!.paths
        Check.expect(a.vertices.count != b.vertices.count || a.inkLength != b.inkLength,
                     "so its ink after the restyle is the refined ink, not the grid's",
                     "\(a.count) vs \(b.count) paths")
        _ = bundle
    }

    Check.suite("refine: phase contours end on the zeros of f") {
        // 1/Γ vanishes at the non-positive integers, which the rgamma window
        // holds from 0 down to -5 along its real edge. Every phase meets at
        // each, so each is where a fan of phase contours should end -- and
        // the grid's end at the last cell edge before it, up to a cell short.
        let preset = Catalog.native.preset("rgamma")!
        let bundle = try NativeLandscape.build(LandscapeRequest(preset: preset, resolution: 240))
        let refiner = NativeLandscape.refiner(for: bundle)!
        let cell = refiner.cell
        let zeros: [Double] = [0, -1, -2, -3, -4, -5]
        var onZero = [Int](repeating: 0, count: zeros.count)
        var shortOfZero = [Int](repeating: 0, count: zeros.count)
        var worst = 0.0
        for layer in bundle.layers where layer.spec.role == .phase {
            for i in 0..<layer.paths.count {
                let path = layer.paths[path: i]
                guard let first = path.first, let last = path.last else { continue }
                for (k, z0) in zeros.enumerated() {
                    let ends = [first, last].map {
                        (($0.x - z0) * ($0.x - z0) + $0.y * $0.y).squareRoot()
                    }
                    if ends.contains(where: { $0 <= 1e-9 * cell }) {
                        onZero[k] += 1
                    } else if let near = ends.min(), near <= 2 * cell {
                        // A path that comes this close and ends on neither end
                        // is the grid's, stopped short.
                        shortOfZero[k] += 1
                        worst = max(worst, near / cell)
                    }
                }
            }
        }
        // Twelve phase directions, 30 degrees apart; the window is the upper
        // half-plane, so five rays leave each zero into it, and the two along
        // the real axis lie on the window's edge.
        Check.expect(onZero.allSatisfy { $0 >= 5 },
                     "a fan of phase contours ends on each zero, to rounding",
                     "runs ending on 0, -1, ..., -5: \(onZero)")
        Check.expect(shortOfZero.allSatisfy { $0 == 0 },
                     "and none stops within two cells of a zero without reaching it",
                     String(format: "short: \(shortOfZero), the worst by %.2f cells", worst))
        Check.expect(bundle.layers.first { $0.spec.name == "ang_major" }!.paths.vertices
                        .allSatisfy { $0.z >= 0 },
                     "the zero's own vertex lifts to the floor and nothing below it")
    }

    Check.suite("refine: the branch cut is drawn once, as one line") {
        // 1/Γ is real and negative on (-1, 0), (-3, -2) and (-5, -4) along the
        // window's edge, and the cut curves leave the real axis from there
        // into the upper half-plane. The grid has no contour at ±π; the wrap
        // bundles the plain contours carry along the cut are spurious; the
        // refined plate draws the cut itself, once.
        let preset = Catalog.native.preset("rgamma")!
        let request = LandscapeRequest(preset: preset, resolution: 240)
        let bundle = try NativeLandscape.build(request)
        let refiner = NativeLandscape.refiner(for: bundle)!
        let cell = refiner.cell
        func cutPaths(_ b: KurvenBundle) -> [[P3<WorldSpace>]] {
            var out: [[P3<WorldSpace>]] = []
            for layer in b.layers where layer.spec.role == .phase {
                for i in 0..<layer.paths.count {
                    let path = Array(layer.paths[path: i])
                    // Every vertex on Im f = 0 with Re f < 0, but for the two
                    // ends: a zero, where the phase is anyone's, or a rim
                    // vertex, placed by interpolation within a cell's bend.
                    let onCut = path.filter { v in
                        let f = refiner.value(Complex(v.x, v.y))
                        return f.re < 0 && abs(f.im) <= 1e-6 * f.magnitude
                    }.count
                    if path.count >= 3 && onCut >= path.count - 2 { out.append(path) }
                }
            }
            return out
        }
        func apart(_ a: P3<WorldSpace>, _ b: P3<WorldSpace>) -> Double {
            ((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)).squareRoot()
        }
        func length(_ p: [P3<WorldSpace>]) -> Double {
            zip(p, p.dropFirst()).reduce(0) { $0 + apart($1.0, $1.1) }
        }
        let cuts = cutPaths(bundle).filter { length($0) > 10 * cell }
        Check.expect(!cuts.isEmpty, "the refined plate has the cut as ink",
                     "\(cuts.count) paths, lengths \(cuts.map { String(format: "%.2f", length($0)) })")
        // Once: no two cut paths run together.
        var doubled = 0
        for (i, a) in cuts.enumerated() {
            for b in cuts[(i + 1)...] {
                let near = a.filter { v in b.contains { apart($0, v) < 0.1 * cell } }.count
                if near > a.count / 2 { doubled += 1 }
            }
        }
        Check.expect(doubled == 0, "and no two of them run together", "\(doubled) pairs")
        let zeros: [Double] = [0, -1, -2, -3, -4, -5]
        let fromZero = cuts.filter { p in
            [p[0], p[p.count - 1]].contains { e in
                zeros.contains { abs(e.x - $0) <= 1e-9 * cell && abs(e.y) <= 1e-9 * cell }
            }
        }
        Check.expect(!fromZero.isEmpty, "a cut leaves a zero, as the phase does", "\(fromZero.count) of \(cuts.count)")
        let plain = try NativeLandscape.build(request, refine: false)
        Check.expect(cutPaths(plain).filter { length($0) > 10 * cell }.isEmpty,
                     "without a refiner the grid draws no cut, only its wrap bundles")
    }

    Check.suite("refine: a phase line is placed only where it can be drawn") {
        // Phase ink is lifted to |f| and cut away above the cap, and on tan
        // most of every phase line is above it, on the poles' spires. Placed
        // under a ceiling, the lines it draws are the same curves, and the
        // stretches over the cap are not solved for: they were the most of
        // the refiner's work on zeta and 1/Γ, for ink that was deleted.
        let request = LandscapeRequest(preset: Catalog.native.preset("tan")!, resolution: 240)
        let plain = try NativeLandscape.build(request, refine: false)
        let refiner = NativeLandscape.refiner(for: plain)!
        let cap = plain.manifest.caps.highest
        let cell = refiner.cell
        func drawn(_ p: P2<DomainSpace>) -> Bool { refiner.magnitude(Complex(p.x, p.y)) <= cap }
        func distance(_ v: P2<DomainSpace>, to runs: [[P2<DomainSpace>]]) -> Double {
            var best = Double.infinity
            for run in runs {
                for (p, q) in zip(run, run.dropFirst()) {
                    let dx = q.x - p.x, dy = q.y - p.y, l2 = dx * dx + dy * dy
                    let t = l2 > 0 ? max(0, min(1, ((v.x - p.x) * dx + (v.y - p.y) * dy) / l2)) : 0
                    best = min(best, ((p.x + t * dx - v.x) * (p.x + t * dx - v.x)
                                      + (p.y + t * dy - v.y) * (p.y + t * dy - v.y)).squareRoot())
                }
            }
            return best
        }
        var everywhere = 0, under = 0, lost = 0, added = 0
        for (level, paths) in Contour.levels(of: plain.surface.phase!, [-2.618, -1.047, 0.524, 2.094]) {
            let all = refiner.refine(field: .phase, level: level, paths)
            let capped = refiner.refine(field: .phase, level: level, paths, ceiling: 1.1 * cap)
            everywhere += all.reduce(0) { $0 + $1.count }
            under += capped.reduce(0) { $0 + $1.count }
            for run in all { for v in run where drawn(v) && distance(v, to: capped) > 0.05 * cell { lost += 1 } }
            for run in capped { for v in run where drawn(v) && distance(v, to: all) > 0.05 * cell { added += 1 } }
        }
        Check.expect(lost == 0 && added == 0,
                     "under the cap, the lines placed under a ceiling are the lines placed everywhere",
                     "\(lost) drawn vertices lost, \(added) added")
        // Once the lines over the poles' spires were most of the placing:
        // 1272 vertices under the ceiling of 18354 everywhere. Then every
        // stretch came to be placed once (`ContourRefiner.drawnOnce`) and
        // the rest fell to 2117, so the ceiling now saves the spires alone.
        Check.expect(under * 3 < everywhere * 2, "and a good part of the placing was over the cap",
                     "\(under) vertices placed, of \(everywhere)")
    }

    Check.suite("refine: a restyle places only the curves it has not placed before") {
        // The cap slider's path in the app: the same grids under a new cap.
        // Before the refiner remembered, every phase line of the landscape
        // was solved again on each step -- 4.5 s on zeta at 600², for ink
        // that differs from the step before only in where it is clipped.
        let request = LandscapeRequest(preset: Catalog.native.preset("rgamma")!, resolution: 240)
        let built = try NativeLandscape.build(request)
        let inner = built.refine!
        let solved = Mutex([ContourField: Int]())
        let counted = ContourRefine(contours: { field, level, paths, ceiling in
                                        solved.withLock { $0[field, default: 0] += 1 }
                                        return inner.contours(field, level, paths, ceiling)
                                    },
                                    magnitude: inner.magnitude, zero: inner.zero,
                                    tolerance: inner.tolerance).remembering()
        let bundle = KurvenBundle(url: built.url, manifest: built.manifest, surface: built.surface,
                                  layers: built.layers, wallMesh: built.wallMesh, refine: counted)
        let cap = LandscapeStyle.ceiling(of: built.manifest.caps)!
        var first = bundle.manifest, second = bundle.manifest
        // Two caps on one rung of the ceiling's ladder (`ContourRefine.rung`),
        // as a slider's neighbouring steps are.
        let near = [0.97, 0.95, 0.9].first {
            ContourRefine.rung(above: 1.1 * cap * 0.9 * $0) == ContourRefine.rung(above: 1.1 * cap * 0.9)
        }!
        first.caps = .uniform(cap * 0.9)
        second.caps = .uniform(cap * 0.9 * near)
        let once = bundle.restyled(first)
        let asked = solved.withLock { $0 }
        let twice = once.restyled(second)
        let again = solved.withLock { $0 }
        Check.expect(asked[.phase, default: 0] > 0, "the first restyle places the phase lines",
                     "\(asked)")
        Check.expect(again[.phase] == asked[.phase],
                     "and the next cap places none of them again",
                     "\(asked[.phase] ?? 0) then \(again[.phase] ?? 0)")
        // The ink is the refiner's answer, remembered or not, to the bit.
        let fresh = NativeLandscape.refiner(for: built)!.refine
        let cold = KurvenBundle(url: built.url, manifest: built.manifest, surface: built.surface,
                                layers: built.layers, wallMesh: built.wallMesh, refine: fresh)
            .restyled(second)
        Check.expect(twice.layers == cold.layers,
                     "and the ink is the ink a cold refiner draws under that cap",
                     "\(twice.layers.map(\.paths.vertices.count)) vs "
                        + "\(cold.layers.map(\.paths.vertices.count))")
    }
}

// MARK: - the numbers in an expression

/// `KurvenMath.Expression.literals` and `replacing`: a slider moves a number without
/// changing the shape of the expression around it.
func literalTests() {
    Check.suite("literals: every number in the source, with its place") {
        let one = KurvenMath.Expression.literals(in: "sin(z)cn(z, 0.64)")
        Check.expect(one == [KurvenMath.Expression.Literal(position: 12, length: 4, value: 0.64, text: "0.64")],
                     "cn(z, 0.64) has one literal, the modulus", "\(one)")

        let mixed = KurvenMath.Expression.literals(in: "-z^2 + 2^-0.5 z - 3")
        Check.expect(mixed.map(\.text) == ["2", "2", "-0.5", "3"]
                     && mixed.map(\.value) == [2, 2, -0.5, 3]
                     && mixed.map(\.position) == [3, 7, 9, 18],
                     "a minus after an operator is the literal's sign; after an operand it is subtraction",
                     "\(mixed.map { "\($0.text)@\($0.position)" })")

        let forms = KurvenMath.Expression.literals(in: "1e-3 z! + (-1)! + .5")
        Check.expect(forms.map(\.text) == ["1e-3", "-1", ".5"]
                     && forms.map(\.value) == [1e-3, -1, 0.5],
                     "exponent forms, a sign after '(', and a bare point", "\(forms.map(\.text))")

        Check.expect(KurvenMath.Expression.literals(in: "gamma(z)").isEmpty, "an expression with no numbers has none")
        Check.expect(KurvenMath.Expression.literals(in: "2 $ z").isEmpty, "text the tokenizer refuses has none")
    }

    Check.suite("literals: rewriting one keeps the expression's shape") {
        func moved(_ text: String, _ index: Int, _ value: Double) -> String {
            KurvenMath.Expression.replacing(KurvenMath.Expression.literals(in: text)[index], with: value, in: text)
        }
        Check.expect(moved("sin(z)cn(z, 0.64)", 0, 0.71) == "sin(z)cn(z, 0.71)",
                     "the modulus of cn moves in place")
        Check.expect(moved("2^-0.5", 1, 0.25) == "2^0.25",
                     "a signed literal crossing zero drops its sign")
        Check.expect(moved("2^0.25", 1, -0.5) == "2^-0.5",
                     "and an unsigned one after an operator gains one")
        Check.expect(moved("2!", 0, -1) == "(-1)!" && moved("(-1)!", 0, 2) == "(2)!",
                     "before a factorial a negative value is parenthesized")
        Check.expect(moved("z-0.5", 0, -0.3) == "z-(-0.3)",
                     "so subtraction of a negative reads as one")
        let parsed = try KurvenMath.Expression.parse("(-1)!")
        Check.expect(parsed == .factorial(.negate(.number(1))),
                     "which is the reading a slider must not lose: (-1)!, not -(1!)")
        let again = KurvenMath.Expression.literals(in: moved("z-0.5", 0, -0.3))
        Check.expect(again.count == 1 && again[0].value == -0.3 && again[0].text == "-0.3",
                     "and the literal is found again, sign and all, at the same index",
                     "\(again)")
        Check.expect(moved("2z + 3", 0, 2.5) == "2.5z + 3" && moved("2exp(z)", 0, 2.5) == "2.5exp(z)",
                     "juxtaposition survives, including before a name that starts with e")
        let f = try KurvenMath.Expression.compile(moved("z - 0.5", 0, -0.3))
        Check.expect((f(Complex(1)) - Complex(1.3)).magnitude < 1e-15,
                     "and the rewritten expression evaluates to what the number says")
    }

    Check.suite("literals: numbers are written the way the language writes them") {
        Check.expect(KurvenMath.Expression.formatNumber(2) == "2" && KurvenMath.Expression.formatNumber(-3) == "-3",
                     "integers without a point")
        Check.expect(KurvenMath.Expression.formatNumber(0.1 + 0.2, significant: 6) == "0.3",
                     "a slider's value rounded to what it means", KurvenMath.Expression.formatNumber(0.1 + 0.2, significant: 6))
        Check.expect(KurvenMath.Expression.formatNumber(1e-5) == "1e-05" && KurvenMath.Expression.formatNumber(0.64) == "0.64",
                     "and otherwise the shortest decimal that reads back")
    }

    Check.suite("literals: each number's role is what a control would call it") {
        func roles(_ text: String) -> [String] {
            KurvenMath.Expression.literals(in: text).map {
                KurvenMath.Expression.role(of: $0, in: text).label
            }
        }
        Check.expect(roles("sin(z)cn(z, 0.64)") == ["Modulus"], "the modulus of cn", "\(roles("sin(z)cn(z, 0.64)"))")
        Check.expect(roles("besselj(2, z)") == ["Order"], "the order of a Bessel function")
        Check.expect(roles("gamma(2)") == ["Argument"], "a number where z would go")
        Check.expect(roles("z^3 - 1") == ["Exponent", "Constant"], "an exponent and a term",
                     "\(roles("z^3 - 1"))")
        Check.expect(roles("2z + 1/z + z/2") == ["Coefficient", "Coefficient", "Divisor"],
                     "factors beside a name, over and under a slash", "\(roles("2z + 1/z + z/2"))")
        Check.expect(roles("cn(2z, -0.5)^-2") == ["Coefficient", "Modulus", "Exponent"],
                     "roles decide in order: a factor inside a call, a signed modulus, a signed exponent",
                     "\(roles("cn(2z, -0.5)^-2"))")
        Check.expect(roles("(z+1)(z-2)") == ["Constant", "Constant"],
                     "terms inside groups are constants", "\(roles("(z+1)(z-2)"))")
        Check.expect(KurvenMath.Language.function("cn")!.parameters == ["argument", "modulus"]
                     && KurvenMath.Language.function("besselk")!.parameters == ["order", "argument"]
                     && KurvenMath.Language.function("exp")!.parameters == ["argument"],
                     "the function table names its parameters")
    }
}
