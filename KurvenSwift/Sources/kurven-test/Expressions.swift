import Foundation
import KurvenCore
import KurvenMath
import KurvenLandscape

// MARK: - 10. the expression language and its functions

/// The native evaluator against the Python one, at every point of every
/// fixture grid.
///
/// The comparison is `|a - b| <= tol * max(|b|, 1)`: relative where the value
/// is large, absolute where it is small, so a zero of a function is not held to
/// a relative error it cannot have, and both sides agreeing that a point is a
/// pole is agreement. The tolerance is per case, stated in the fixture, and it
/// is a statement about the algorithm: 1e-12 where this side runs scipy's own
/// method, 1e-9 for the Bessel and Airy families, whose engines are different
/// from AMOS and agree with it to about 1e-11 across these windows.
func expressionTests() {
    Check.suite("expression: the functions agree with scipy on the fixture grids") {
        let index = try Fixtures.json("expr/index.json")
        let cases = try index.array("cases", "expr index")
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
                    // A NaN on the Python side is scipy declining to answer
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
            Check.expect(disagreements == 0, "\(name): \(text)",
                         String(format: "worst %.1e at %@ (tol %g)%@", worst, "\(worstAt)", tol,
                                disagreements == 0 ? "" : ", \(disagreements) points over"))
        }
        Check.expect(failed == 0, "all \(cases.count) cases agree",
                     String(format: "worst %.1e", worstOverall))
    }

    Check.suite("expression: the parser reads what Python reads") {
        let index = try Fixtures.json("expr/index.json")
        let canonical = try index.value("canonical", "expr index").object("canonical")
        var wrong: [String] = []
        for (source, want) in canonical.sorted(by: { $0.key < $1.key }) {
            guard case .string(let wanted) = want else { wrong.append("\(source): not a string"); continue }
            let got = try KurvenMath.Expression.canonical(source)
            if got != wanted { wrong.append("\(source) -> \(got), Python \(wanted)") }
            // And the canonical form is a fixed point that means the same thing.
            let again = try KurvenMath.Expression.canonical(got)
            if again != got { wrong.append("\(got) is not a fixed point (\(again))") }
        }
        Check.expect(wrong.isEmpty, "\(canonical.count) canonical spellings",
                     wrong.joined(separator: "; "))

        let errors = try index.array("errors", "expr index")
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
                                     + "+\(error.length), Python \(position.map(String.init) ?? "nil")"
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

    Check.suite("expression: the language table is the one Python reports") {
        let index = try Fixtures.json("expr/index.json")
        let language = try index.value("language", "expr index").object("language")
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

    Check.suite("expression: the native catalog is Python's, entry for entry") {
        let index = try Fixtures.json("expr/index.json")
        let presets = try index.array("catalog", "expr index").map(FunctionPreset.init(json:))
        let mine = Catalog.native.presets
        Check.expect(presets.map(\.name) == mine.map(\.name),
                     "the same \(presets.count) presets, in the same order",
                     mine.map(\.name).joined(separator: " "))
        var differ: [String] = []
        for (theirs, ours) in zip(presets, mine) where theirs != ours {
            differ.append(theirs.name)
        }
        Check.expect(differ.isEmpty, "with the same expressions, windows, caps and notes",
                     differ.joined(separator: ", "))
        let defaults = try index.value("defaults", "expr index").object("defaults")
        Check.expect(try defaults.int("resolution", "defaults") == Catalog.native.defaultResolution
                     && defaults.int("buffer", "defaults") == Catalog.native.defaultBuffer,
                     "and the same default resolution and bake size")
    }
}

// MARK: - 11. the native landscape

/// `NativeLandscape.build` against the bundles `kurven.landscape` wrote for
/// the same specs: the same manifest, the same grids, the same ink.
func landscapeTests() {
    Check.suite("landscape: built here, it is the bundle Python writes") {
        let index = try Fixtures.json("expr/index.json")
        for entry in try index.array("landscapes", "expr index") {
            let e = try entry.object("landscape")
            let name = try e.string("bundle", "landscape")
            let spec = try LandscapeRequest(json: e.value("spec", "landscape"))
            let theirs = try KurvenBundle.read(at: Fixtures.url("expr/\(name)"))
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
            for (mine, python) in zip(ours.layers, theirs.layers) {
                let a = mine.paths.inkLength, b = python.paths.inkLength
                if mine.paths.count != python.paths.count || abs(a - b) > 1e-6 * max(b, 1) {
                    inkDiffers.append("\(mine.spec.name) \(mine.paths.count)/\(python.paths.count) paths, "
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
    return "at \(i): python …\(String(x[from..<min(x.count, to)]))… swift …\(String(y[from..<min(y.count, to)]))…"
}

// MARK: - 12. contour refinement

/// Contours placed by f: the vertices land on the level set, the chords stay
/// within tolerance of it, the phase wraps are gone, and a restyle keeps all
/// of that.
func refinementTests() {
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
