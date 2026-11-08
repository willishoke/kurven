import Foundation
import simd
import KurvenCore
import KurvenMetal
import KurvenBake
import KurvenDynamics
import KurvenLandscape

// MARK: - a parametric surface in a bundle
//
// A parametric surface's bundle stores the request that builds it, not
// samples of it (`SurfaceManifest`): the map is code. So a bundle is held to
// the request it keeps -- its bake is the bake of the drawing built straight
// from the request, as `kurven-cli surface` builds it -- and its layers to
// the drawing as it was kept, restyled or not.

func surfaceBundleTests() {
    Check.suite("bundles: a parametric surface's bundle bakes what its request bakes, for every surface") {
        // One request of each shape `kurven-cli surface` builds, coarser than
        // its defaults: the round trip, not the picture, is under test.
        func sn(_ shape: SurfaceRequest.Shape) -> SurfaceRequest { SurfaceRequest(shape, grid: 256) }
        let requests: [(String, SurfaceRequest)] = [
            ("torus", SurfaceRequest(.torus(major: 2, minor: 1), grid: 256, lines: SIMD2(36, 18),
                                     linearCurves: SurfaceRequest.LinearCurves(slope: 0.4), name: "torus")),
            ("sn", SurfaceRequest(.sn(modulus: 0.64, major: 2, minor: 1, grid: 200), grid: 256, name: "sn")),
            ("periodic", sn(.periodic(expression: "sn(z, 0.5)", periods: SIMD2(7.416298709205487, 3.7081493546027433),
                                      major: 2, minor: 1, grid: 200))),
            ("forced", SurfaceRequest(.forced(system: "jerk", fit: 8000, harmonics: 24, radial: 0, axial: 1,
                                              radius: 2.5, turns: 30),
                                      grid: 256, lines: SIMD2(12, 6), name: "forced")),
            ("relief", SurfaceRequest(.relief(modulus: 0.64, major: 2, minor: 1, amplitude: 0.4, repeats: SIMD2(1, 1),
                                              grid: 200, turn: 0, floor: nil, soft: false), grid: 512)),
            ("relief, repeated, turned, floored and soft",
             SurfaceRequest(.relief(modulus: 0.64, major: 2, minor: 1, amplitude: 0.3, repeats: SIMD2(2, 1),
                                    grid: 160, turn: 0.25, floor: 0.5, soft: true), grid: 512)),
        ]
        let renderer = try MetalRenderer()
        let options = BakeOptions(resolution: 1000)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("kurven-bundles-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        for (label, request) in requests {
            let drawing = try SurfaceDrawing.build(request)
            let direct = SVG.render(try renderer.bake(drawing.scene(camera: .oblique(SurfaceRequest.projection)),
                                                      options: options).strokes)
            let url = directory.appendingPathComponent("\(label).kurven")
            try SurfaceBundle(url: url, drawing: drawing).write(to: url)
            let text = try String(contentsOf: url.appendingPathComponent("manifest.json"), encoding: .utf8)
            let read = try SurfaceBundle.read(at: url)
            let fromBundle = SVG.render(try renderer.bake(read.scene(preset: read.manifest.presets[0]),
                                                          options: options).strokes)
            let strokes = direct.components(separatedBy: "<polyline").count - 1
            Check.expect(read.manifest.surface == request && read.manifest.canonicalJSON == text
                         && strokes > 10 && fromBundle == direct,
                         "\(label): saved, reopened and baked, the bundle bakes the request's SVG byte for byte",
                         "\(strokes) strokes")
        }
    }

    Check.suite("bundles: open a parametric surface's bundle, restyle a layer, save it, and reopen it") {
        // The document's transitions on a surface, as the window makes them
        // (`Document.load`, a level slider, `saveBundle`), without the window.
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("kurven-restyle-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let request = SurfaceRequest(.sn(modulus: 0.64, major: 2, minor: 1, grid: 200), grid: 256, name: "sn")
        let saved = directory.appendingPathComponent("sn.kurven")
        try SurfaceBundle(url: saved, drawing: try SurfaceDrawing.build(request)).write(to: saved)
        let opened = try SurfaceBundle.read(at: saved)

        // Restyle: |f| at other levels and a thinner phase, and a contour of
        // v and of the height drawn on the surface besides.
        var layers = opened.manifest.layers
        let major = layers.firstIndex { $0.name == "mag_major" }!
        let minor = layers.firstIndex { $0.name == "ang_minor" }!
        layers[major].source = .contour(field: .magnitude, levels: [0.5, 1.5], restriction: .all, faces: .surface)
        layers[minor].width = 0.1
        layers.append(LayerSpec(name: "v", role: .scaffold,
                                source: .contour(field: .v, levels: [1, 2], restriction: .all, faces: .surface),
                                width: 0.2, heightPolicy: .surface))
        layers.append(LayerSpec(name: "height", role: .scaffold,
                                source: .contour(field: .height, levels: [0.5], restriction: .all, faces: .surface),
                                width: 0.2, heightPolicy: .surface))
        let restyled = try opened.drawing.restyled(layers)
        let again = directory.appendingPathComponent("restyled.kurven")
        try opened.write(to: again, layers: layers)
        let reopened = try SurfaceBundle.read(at: again)

        func paths(_ d: SurfaceDrawing, _ name: String) -> PolylineSet<WorldSpace>? {
            d.layers.first { $0.spec.name == name }?.paths
        }
        func paths(_ b: SurfaceBundle, _ name: String) -> PolylineSet<WorldSpace>? { paths(b.drawing, name) }
        Check.expect(restyled.content == opened.drawing.content
                     && restyled.surface.positions == opened.drawing.surface.positions,
                     "a restyle keeps the surface and its identity")
        let fresh = try PeriodicTorus.jacobiSN(modulus: 0.64, major: 2, minor: 1, resolution: 200, grid: 256,
                                               contours: [layers[major]]).layers[0].paths
        Check.expect(paths(restyled, "mag_major") == fresh && paths(restyled, "mag_major") != paths(opened, "mag_major"),
                     "|f| at other levels is the function's contours there, placed as sn's own are")
        Check.expect(paths(restyled, "ang_minor") == paths(opened, "ang_minor")
                     && paths(restyled, "ang_major") == paths(opened, "ang_major"),
                     "a layer restyled in width alone, and a layer not restyled, keep their curves")
        let v = paths(restyled, "v"), height = paths(restyled, "height")
        let periodV = opened.drawing.surface.v.range.length
        let onV = v.map { p in
            p.count == 2 && p.coords!.allSatisfy { c in
                [1.0, 2.0].contains { abs(c.y - $0 - periodV * ((c.y - $0) / periodV).rounded()) < 1e-9 }
            }
        } ?? false
        let onEquator = height.map { p in
            p.count == 2 && p.vertices.allSatisfy { abs($0.z - 0.5) < 1e-9 }
        } ?? false
        Check.expect(onV && onEquator,
                     "a contour of v at two levels is two coordinate circles, and of the height 0.5 two parallels")
        Check.expect(reopened.manifest.layers == layers && reopened.manifest.surface == request
                     && reopened.drawing.layers.map(\.spec) == restyled.layers.map(\.spec)
                     && reopened.drawing.layers.map(\.paths) == restyled.layers.map(\.paths),
                     "saved and reopened, the restyled bundle draws what was on screen")

        // An edit of the request after a restyle is the drawing built from
        // nothing: what it reuses is what its request drew.
        var bigger = request
        bigger[.major] = 2.5
        let edited = try SurfaceDrawing.build(bigger, reusing: reopened.drawing)
        let built = try SurfaceDrawing.build(bigger)
        Check.expect(edited.layers.map(\.spec) == built.layers.map(\.spec)
                     && edited.layers.map(\.paths) == built.layers.map(\.paths),
                     "a request edit after a restyle builds the request's drawing, as from nothing")

        // A repeated relief restyled draws from the function as it was
        // sampled once, carried round the periods and through the relief:
        // what building it afresh with those layers draws.
        let relief = SurfaceRequest(.relief(modulus: 0.64, major: 2, minor: 1, amplitude: 0.3, repeats: SIMD2(2, 1),
                                            grid: 120, turn: 0.25, floor: nil, soft: false), grid: 256)
        let base = try SurfaceDrawing.build(relief)
        var reliefLayers = base.layers.map(\.spec)
        let mag = reliefLayers.firstIndex { $0.name == "mag_major" }!
        reliefLayers[mag].source = .contour(field: .magnitude, levels: [0.75, 1.25], restriction: .all,
                                            faces: .surface)
        let wanted = try PeriodicTorus.jacobiSN(modulus: 0.64, major: 2, minor: 1, resolution: 120, grid: 256,
                                                truncated: true, turn: 0.25, contours: [reliefLayers[mag]])
            .repeated(real: 2, imag: 1, grid: 256).relief(amplitude: 0.3, reach: 1).layers[0].paths
        Check.expect(paths(try base.restyled(reliefLayers), "mag_major") == wanted,
                     "a repeated relief restyled is its contours at those levels, carried round and displaced")
    }

    Check.suite("bundles: a request edit keeps the layers a bundle kept, and a slider's levels keep their range") {
        // `Document`'s edit of a surface's request: built from the drawing
        // before it, with the layers its bundle kept (`SurfaceDrawing.keeping`).
        let opened = try SurfaceBundle.read(at: Fixtures.url("contract/surface.kurven"))
        let kept = opened.manifest.layers
        func edit(_ change: (inout SurfaceRequest) -> Void) throws -> SurfaceDrawing {
            var request = opened.manifest.surface
            change(&request)
            return try SurfaceDrawing.build(request, reusing: opened.drawing).keeping(kept)
        }
        let wider = try edit { $0[.major] = 2.5 }
        let specs = wider.layers.map(\.spec)
        Check.expect(specs == kept,
                     "after the radii move, every layer is the one the bundle kept: the contours of u, v and the "
                     + "height, |f| and arg f at the bundle's levels, the coordinate curves, the linear curve")
        let fresh = try SurfaceDrawing.build(wider.request).restyled(kept)
        Check.expect(wider.layers.map(\.paths) == fresh.layers.map(\.paths),
                     "and draws what the edited request draws with those layers, built afresh")
        let fewer = try edit { $0.lines = SIMD2(4, 2); $0.linearCurves = nil }
        let names = fewer.layers.map(\.spec.name)
        Check.expect(fewer.layers.first { $0.spec.name == "lines" }?.spec.source == .coordinateCurves(u: 4, v: 2)
                     && !names.contains("linearCurves") && names.contains("u") && names.contains("mag_major"),
                     "the coordinate curves and the linear curve are the request's: moved with it, and gone with it")
        let added = try edit { $0.linearCurves = SurfaceRequest.LinearCurves(slope: 0.25, turns: 4, count: 2) }
        Check.expect(added.layers.first { $0.spec.name == "linearCurves" }?.spec.source
                        == .linearCurves(slope: 0.25, turns: 4, count: 2),
                     "a linear curve the request moves is drawn as it now says")

        // A level slider resamples a family over its own range, and never to
        // fewer than two levels, so a family saved at them keeps its range.
        let family = LayerSpec(name: "mag", role: .magnitude,
                               source: .contour(field: .magnitude, levels: [0.5, 1, 3], restriction: .all,
                                                faces: .surface),
                               width: 0.35, heightPolicy: .surface)
        let one = KurvenBundle.levels(of: family, count: 1)
        var saved = family
        saved.source = .contour(field: .magnitude, levels: one ?? [], restriction: .all, faces: .surface)
        Check.expect(one == [0.5, 3] && KurvenBundle.levels(of: saved, count: 6)?.first == 0.5
                     && KurvenBundle.levels(of: saved, count: 6)?.last == 3,
                     "a slider at one level draws the family's two ends, and saved, the family keeps its range")
        Check.expect(KurvenBundle.levels(of: family, count: 6) == [0.5, 1, 1.5, 2, 2.5, 3],
                     "six levels are six even steps over the range")
    }

    Check.suite("bundles: a malformed parametric surface's manifest is refused, saying why") {
        let good = try Fixtures.json("contract/surface.kurven/manifest.json")
        Check.expect((try? SurfaceManifest(json: .object(good))) != nil, "the fixture itself is read")
        func mutated(_ change: (inout [String: JSONValue]) -> Void) -> JSONValue {
            var d = good; change(&d); return .object(d)
        }
        func surface(_ change: (inout [String: JSONValue]) -> Void) -> JSONValue {
            mutated { d in
                var s = (try? d["surface"]!.object("s")) ?? [:]
                change(&s)
                d["surface"] = .object(s)
            }
        }
        func shape(_ shape: [String: JSONValue]) -> JSONValue { surface { $0["shape"] = .object(shape) } }
        func layer(_ source: LayerSource) -> JSONValue {
            mutated { d in
                d["layers"] = .array([LayerSpec(name: "x", role: .scaffold, source: source, width: 0.2,
                                                heightPolicy: .surface).json])
            }
        }
        func refuses(_ what: String, saying words: String, _ json: JSONValue) {
            do {
                _ = try BundleManifest(json: json)
                Check.expect(false, what, "decoded")
            } catch {
                Check.expect("\(error)".contains(words), what, "\(error)")
            }
        }
        let sn: [String: JSONValue] = ["kind": .string("sn"), "modulus": .double(0.64), "major": .double(2),
                                       "minor": .double(1), "grid": .int(120)]
        let torus: [String: JSONValue] = ["kind": .string("torus"), "major": .double(2), "minor": .double(1)]
        let forced: [String: JSONValue] = ["kind": .string("forced"), "system": .string("jerk"),
                                           "fit": .double(8000), "harmonics": .int(24), "radial": .int(1),
                                           "axial": .int(1), "radius": .double(2.5), "turns": .int(30)]
        let relief: [String: JSONValue] = ["kind": .string("relief"), "modulus": .double(0.64), "major": .double(2),
                                           "minor": .double(1), "amplitude": .double(0.4),
                                           "repeats": .array([.int(0), .int(1)]), "grid": .int(120),
                                           "turn": .double(0), "floor": .null, "soft": .bool(false)]
        refuses("refuses a modulus outside (0, 1)", saying: "modulus must lie strictly between 0 and 1, not 1.5",
                shape(sn.merging(["modulus": .double(1.5)]) { $1 }))
        refuses("refuses a radius that is not positive", saying: "surface.shape.major must be a positive number, not -1.0",
                shape(sn.merging(["major": .double(-1)]) { $1 }))
        refuses("refuses a grid too coarse to span its rectangle", saying: "surface.grid must be 8 to 65536 samples, not 4",
                surface { $0["grid"] = .int(4) })
        refuses("refuses a shape it does not know, naming the ones it does", saying: "'klein' is not one of",
                shape(["kind": .string("klein")]))
        refuses("refuses a forced torus revolved on one state component twice", saying: "surface.shape.radial and surface.shape.axial are both state component 1",
                shape(forced))
        refuses("refuses a period that is not positive", saying: "surface.shape.periods must be a positive number, not 0.0",
                shape(["kind": .string("periodic"), "expression": .string("sn(z, 0.5)"),
                       "periods": .array([.double(0), .double(1)]), "major": .double(2), "minor": .double(1),
                       "grid": .int(120)]))
        refuses("refuses an empty expression", saying: "expression is empty",
                shape(["kind": .string("periodic"), "expression": .string(" "),
                       "periods": .array([.double(1), .double(1)]), "major": .double(2), "minor": .double(1),
                       "grid": .int(120)]))
        refuses("refuses a relief repeated less than once", saying: "surface.shape.repeats must be 1 to 16 whole periods each way, not 0",
                shape(relief))
        refuses("refuses a negative count of coordinate curves", saying: "surface.lines must be 0 to 16777216 coordinate curves, not -1",
                surface { $0["lines"] = .object(["u": .int(-1), "v": .int(6)]) })
        refuses("refuses a linear curve of no turns", saying: "at least one turn of one strand",
                surface { $0["linearCurves"] = .object(["slope": .double(0.4), "turns": .int(0), "count": .int(1)]) })
        refuses("refuses a negative stroke width", saying: "surface.style.folds must be a number not below 0, not -1.0",
                surface { s in
                    var style = (try? s["style"]!.object("style")) ?? [:]
                    style["folds"] = .double(-1)
                    s["style"] = .object(style)
                })
        refuses("refuses a request with no style", saying: "no key 'style'", surface { $0["style"] = nil })
        refuses("refuses hatching on a parametric surface", saying: "hatching on a parametric surface: it is curved",
                layer(.hatching(Hatching(angle: 0, spacing: 0.1), faces: .surface)))
        refuses("refuses wall edges on a parametric surface", saying: "it is one face, with no walls",
                layer(.wallEdges(edges: [0], base: 0)))
        refuses("refuses a dumped layer on a parametric surface", saying: "drawn from its request, not stored",
                layer(.file(vertices: "v.npy", offsets: "o.npy")))
        refuses("refuses a contour on a face the surface does not have", saying: "the surface is one face",
                layer(.contour(field: .u, levels: [1], restriction: .all, faces: .graph(tiled: false))))
        refuses("refuses |f| on a surface that paints no function", saying: "paints no function",
                mutated { d in
                    d["surface"] = surfaceShape(good, torus)
                    d["layers"] = .array([LayerSpec(name: "x", role: .scaffold,
                                                    source: .contour(field: .magnitude, levels: [1], restriction: .all,
                                                                     faces: .surface),
                                                    width: 0.2, heightPolicy: .surface).json])
                })
        refuses("refuses a trajectory on a torus no flow winds round", saying: "no flow winds round",
                layer(.trajectory))
        refuses("refuses a contour kept below a truncation sn does not have", saying: "only in relief",
                layer(.contour(field: .phase, levels: [0], restriction: .belowTruncation, faces: .surface)))
        refuses("refuses a contour kept to a footprint the surface does not have", saying: "the surface has neither",
                layer(.contour(field: .u, levels: [1], restriction: .footprint, faces: .surface)))
        refuses("refuses a manifest holding both geometries", saying: "a manifest holds one geometry",
                mutated { $0["domain"] = .object([:]) })
        refuses("refuses a parametric surface stating a landscape's axes", saying: "with a landscape's axes",
                mutated { $0["axes"] = .array(["real", "imag", "magnitude"].map(JSONValue.string)) })
        refuses("refuses a surface that is no request", saying: "SurfaceRequest is not an object",
                mutated { $0["surface"] = .null })
        refuses("reads a manifest with neither geometry as a landscape's, and refuses it", saying: "no key 'axes'",
                mutated { $0["surface"] = nil; $0["layers"] = nil })
        refuses("refuses two layers of one name: a layer is kept across an edit by its name",
                saying: "two layers named 'mag_major'",
                mutated { d in
                    if case .array(let layers) = d["layers"]! { d["layers"] = .array(layers + [layers[0]]) }
                })

        // Nothing a small bundle says traps or costs without bound: every
        // count that drives work is bounded, and a number that is not an
        // integer an `Int` holds is refused, not trapped on.
        refuses("refuses a grid that is no integer an Int holds", saying: "'grid' in SurfaceRequest should be an integer",
                surface { $0["grid"] = .double(1e30) })
        refuses("refuses a fractional grid", saying: "'grid' in SurfaceRequest should be an integer",
                surface { $0["grid"] = .double(64.5) })
        refuses("refuses a grid of no samples", saying: "surface.grid must be 8 to 65536 samples, not 0",
                surface { $0["grid"] = .int(0) })
        refuses("refuses a negative function grid", saying: "surface.shape.grid must be 8 to 4096 samples, not -5",
                shape(sn.merging(["grid": .int(-5)]) { $1 }))
        refuses("refuses a function grid finer than it holds", saying: "surface.shape.grid must be 8 to 4096",
                shape(sn.merging(["grid": .int(5000)]) { $1 }))
        refuses("refuses a surface of more samples than it holds", saying: "more than the 33554432 a surface holds",
                mutated { d in
                    d["surface"] = surfaceShape(good, torus)
                    d["layers"] = .array([])
                    if case .object(var s) = d["surface"]! { s["grid"] = .int(16384); d["surface"] = .object(s) }
                })
        refuses("refuses a relief repeated more than it holds", saying: "must be 1 to 16 whole periods",
                shape(relief.merging(["repeats": .array([.int(17), .int(1)])]) { $1 }))
        refuses("refuses a fit longer than a run holds", saying: "surface.shape.fit must be at most 100000",
                shape(forced.merging(["radial": .int(0), "fit": .double(1e6)]) { $1 }))
        refuses("refuses more harmonics than a fit holds", saying: "surface.shape.harmonics must be 1 to 64",
                shape(forced.merging(["radial": .int(0), "harmonics": .int(100)]) { $1 }))
        refuses("refuses a trajectory of more vertices than a layer holds", saying: "vertices for the trajectory",
                mutated { d in
                    d["surface"] = surfaceShape(good, forced.merging(["radial": .int(0), "turns": .int(1_000_000)]) { $1 })
                    d["layers"] = .array([])
                })
        refuses("refuses coordinate curves of more vertices than a layer holds", saying: "vertices for 100000 + 6 coordinate curves",
                surface { $0["lines"] = .object(["u": .int(100_000), "v": .int(6)]) })
        refuses("refuses a coordinate-curve layer of a negative count", saying: "coordinate curves must be counts",
                layer(.coordinateCurves(u: -1, v: 6)))
        refuses("refuses a coordinate-curve layer of more vertices than a layer holds", saying: "a layer that would take",
                layer(.coordinateCurves(u: 100_000, v: 6)))
        refuses("refuses a linear-curve layer of no turns", saying: "at least one turn of one strand",
                layer(.linearCurves(slope: 0.4, turns: 0, count: 1)))
        refuses("refuses a linear-curve layer of no strands", saying: "at least one turn of one strand",
                layer(.linearCurves(slope: 0.4, turns: 3, count: 0)))
        refuses("refuses a linear curve too steep for a layer to hold", saying: "vertices, more than the 16777216",
                layer(.linearCurves(slope: 1e9, turns: 1, count: 1)))
        refuses("refuses a contour of u kept below a truncation", saying: "only |f| and arg f are truncated",
                layer(.contour(field: .u, levels: [1], restriction: .belowTruncation, faces: .surface)))
        let text = try String(contentsOf: Fixtures.url("contract/surface.kurven/manifest.json"), encoding: .utf8)
        for (what, number) in [("NaN", "NaN"), ("a number past a double", "1e400")] {
            let raw = text.replacingOccurrences(of: "\"major\":2.0", with: "\"major\":\(number)")
            Check.expectThrows("refuses \(what) for a radius") {
                _ = try BundleManifest(json: JSONValue.parse(raw))
            }
        }
        refuses("refuses schema 2", saying: "schema 2 is not the supported 3", mutated { $0["schema"] = .int(2) })
        Check.expectThrows("a landscape's reader refuses a parametric surface, naming its reader") {
            _ = try Manifest(json: .object(good))
        }

        // What only building finds is refused when the bundle is read.
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("kurven-refusals-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        func unreadable(_ what: String, saying words: String, _ json: JSONValue) throws {
            let url = directory.appendingPathComponent("\(UUID().uuidString).kurven")
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            try (json.canonical + "\n").write(to: url.appendingPathComponent("manifest.json"),
                                             atomically: true, encoding: .utf8)
            do {
                _ = try SurfaceBundle.read(at: url)
                Check.expect(false, what, "read")
            } catch {
                Check.expect("\(error)".contains(words), what, "\(error)")
            }
        }
        try unreadable("refuses a surface with no camera preset to open or bake at", saying: "no camera preset",
                       mutated { $0["presets"] = .array([]) })
        try unreadable("refuses a forced system the catalog does not have, by name", saying: "unknown system 'lorenz'",
                       mutated { d in
                           d["surface"] = surfaceShape(good, forced.merging(["system": .string("lorenz"),
                                                                             "radial": .int(0)]) { $1 })
                           d["layers"] = .array([])
                       })
        try unreadable("refuses an expression that does not parse", saying: "expression",
                       mutated { d in
                           d["surface"] = surfaceShape(good, ["kind": .string("periodic"), "expression": .string("sn(z,"),
                                                              "periods": .array([.double(1), .double(1)]),
                                                              "major": .double(2), "minor": .double(1),
                                                              "grid": .int(120)])
                       })
    }
}

/// The fixture's request with another shape.
private func surfaceShape(_ manifest: [String: JSONValue], _ shape: [String: JSONValue]) -> JSONValue {
    var s = (try? manifest["surface"]!.object("s")) ?? [:]
    s["shape"] = .object(shape)
    return .object(s)
}

// MARK: - a polyhedron in a bundle
//
// A polyhedron's bundle stores its geometry, the vertices and loops
// `Solid(vertices:faces:)` takes (`PolyhedronManifest`). So a bundle is held
// to the solid it keeps -- its bake is the bake of the same solid drawn
// straight, as `kurven-cli surface solid` draws it -- and its reader to
// refusing, with the reason, whatever bounds no solid or would cost without
// bound.

func polyhedronBundleTests() {
    Check.suite("bundles: a polyhedron's bundle bakes what its solid bakes, for every kind and tone") {
        let csaszar = Solid.csaszar
        let reach = csaszar.vertices.map { simd_length(SIMD2($0.x, $0.y)) }.max()!
        let frame = Solid.frame(outer: 4, inner: 2, depth: 0.8)
        let solids: [(String, Solid)] = Solid.Platonic.allCases.map {
            ($0.rawValue, Solid.platonic($0, circumradius: 2).resting(onFace: 0))
        } + [("csaszar", csaszar.transformed(rotation: matrix_identity_double3x3 * (2 / reach))), ("frame", frame)]
        let light = SIMD3(-0.4, -0.6, 0.7)
        let tones: [(String, Hatching?)] = [
            ("edges alone", nil),
            ("uniform", Hatching(angle: .pi / 4, spacing: 0.08)),
            ("uniform at its own step", Hatching(angle: 0.3, spacing: 0.1, step: 0.03)),
            ("Turing's pattern", Hatching(angle: .pi / 6, spacing: 0.05, tone: .turing(scale: 0.4, seed: 3))),
            ("Turing's pattern in four widths",
             Hatching(angle: .pi / 6, spacing: 0.035, tone: .turing(scale: 0.18, seed: 3), widthSteps: 4)),
            ("half-Lambert", Hatching(angle: .pi / 4, spacing: 0.08, tone: .halfLambert(light: light))),
            ("half-Lambert in four widths",
             Hatching(angle: .pi / 4, spacing: 0.08, tone: .halfLambert(light: light), widthSteps: 4)),
        ]
        let renderer = try MetalRenderer()
        let options = BakeOptions(resolution: 600)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("kurven-polyhedra-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        // Every tone on the two solids that are not convex, one each on
        // the Platonic solids.
        for (index, (kind, solid)) in solids.enumerated() {
            let convex = index < Solid.Platonic.allCases.count
            for (label, hatching) in convex ? [tones[index % tones.count]] : tones {
                var specs = [LayerSpec(name: "edges", role: .outline, source: .edges, width: 0.6,
                                       heightPolicy: .surface)]
                var layers = [Layer(spec: specs[0], paths: solid.edgePaths)]
                if let hatching {
                    specs.append(LayerSpec(name: "hatching", role: .scaffold,
                                           source: .hatching(hatching, faces: .solid), width: 0.3,
                                           heightPolicy: .surface))
                    layers.append(Layer(spec: specs[1], paths: solid.hatchPaths(hatching)))
                }
                let direct = SVG.render(try renderer.bake(
                    Scene(solid: solid, layers: layers, camera: .oblique(SurfaceRequest.projection)),
                    options: options).strokes)
                let url = directory.appendingPathComponent("\(kind) \(label).kurven")
                try PolyhedronBundle(url: url, solid: solid, layers: specs, presets: [SurfaceRequest.preset],
                                     function: kind).write(to: url)
                let text = try String(contentsOf: url.appendingPathComponent("manifest.json"), encoding: .utf8)
                let read = try PolyhedronBundle.read(at: url)
                let fromBundle = SVG.render(try renderer.bake(read.scene(preset: read.manifest.presets[0]),
                                                              options: options).strokes)
                let strokes = direct.components(separatedBy: "<polyline").count - 1
                Check.expect(read.solid == solid && read.manifest.layers == specs
                             && read.manifest.canonicalJSON == text && strokes > 3 && fromBundle == direct,
                             "\(kind), \(label): saved, reopened and baked, the bundle bakes the solid's SVG "
                             + "byte for byte", "\(strokes) strokes")
            }
        }
    }

    Check.suite("preview: a polyhedron at rest draws what its bake draws, and nothing it hides") {
        // The window's frame of the gallery's frame: its edges and its
        // hatching in four widths, cut at rest in closed form as the bake
        // cuts them, then drawn. Every baked pixel is drawn, and every drawn
        // pixel is within two of a baked one, so no hidden edge and no hatch
        // line on a face turned away shows.
        let bundle = try PolyhedronBundle.read(at: Fixtures.url("contract/polyhedron_frame.kurven"))
        let preset = bundle.manifest.presets[0]
        let viewport = Viewport(width: 1200, height: 900)
        let scene = bundle.scene(preset: preset)
        guard let bounds = scene.quickBounds() else { Check.expect(false, "the frame has bounds"); return }
        let navigator = Navigator(orbit: Orbit(matching: preset.projection), framing: .fitting(bounds, in: viewport))
        let renderer = try MetalRenderer()
        let target = try renderer.makePreviewTarget(viewport)
        let frame = navigator.framing.frame(viewport)
        let baked = try renderer.bake(scene.looking(navigator.camera), options: BakeOptions(resolution: 3000))
        var bake = [Bool](repeating: false, count: viewport.width * viewport.height)
        for layer in baked.strokes.layers {
            for path in layer.paths.paths {
                for (a, b) in zip(path, path.dropFirst()) {
                    let steps = max(Int(simd_length(b.v - a.v) / frame.unitsPerPixel.max() * 4), 1)
                    for j in 0...steps {
                        let t = Double(j) / Double(steps)
                        let i = frame.index(of: P2<ViewSpace>(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t))
                        guard i.x >= 0, i.x < viewport.height, i.y >= 0, i.y < viewport.width else { continue }
                        bake[i.x * viewport.width + i.y] = true
                    }
                }
            }
        }
        func near(_ marks: [Bool], _ row: Int, _ col: Int, _ reach: Int) -> Bool {
            for dr in -reach...reach {
                for dc in -reach...reach {
                    let r = row + dr, c = col + dc
                    guard r >= 0, r < viewport.height, c >= 0, c < viewport.width else { continue }
                    if marks[r * viewport.width + c] { return true }
                }
            }
            return false
        }
        let bakedPixels = bake.indices.filter { bake[$0] }
        // A stroke is drawn a pixel and a half wide, so a drawn pixel is
        // held to within two of the bake's.
        func compared(draft: Bool, slopeScale: Float = 0) throws -> (recalled: Int, stray: Int, drawn: Int) {
            try renderer.renderPreview(scene, navigator: navigator, viewport: viewport,
                                       options: PreviewOptions(slopeScale: slopeScale, draft: draft), into: target)
            let bgra = try PNG.bgra(target)
            let drawn = (0..<(viewport.width * viewport.height)).map { k -> Bool in
                let r = Int(bgra[4 * k + 2]), g = Int(bgra[4 * k + 1]), b = Int(bgra[4 * k])
                return 299 * r + 587 * g + 114 * b < 224_000
            }
            let drawnPixels = drawn.indices.filter { drawn[$0] }
            return (bakedPixels.filter { near(drawn, $0 / viewport.width, $0 % viewport.width, 1) }.count,
                    drawnPixels.filter { !near(bake, $0 / viewport.width, $0 % viewport.width, 2) }.count,
                    drawnPixels.count)
        }
        // At rest, at the bake's predicate (no slope term) and the app's.
        for slopeScale in [0, PreviewOptions.defaultSlopeScale] {
            let rest = try compared(draft: false, slopeScale: slopeScale)
            Check.expect(bakedPixels.count > 5000 && rest.recalled == bakedPixels.count && rest.stray == 0,
                         "at rest, at slope scale \(slopeScale), the preview draws every pixel of the bake's edges "
                         + "and hatching, and nothing the bake hides",
                         "\(rest.recalled) of \(bakedPixels.count), \(rest.stray) of \(rest.drawn) drawn pixels astray")
        }
        // Moving, the strokes as carried are left to the depth pass: at the
        // app's slope scale they are all there and nothing hidden shows; at
        // the bake's predicate, lying on their faces, they break, which is
        // what laying them out at rest is for.
        let appDraft = try compared(draft: true, slopeScale: PreviewOptions.defaultSlopeScale)
        let draft = try compared(draft: true)
        Check.expect(appDraft.recalled == bakedPixels.count && appDraft.stray == 0,
                     "while the camera moves, at the app's slope scale, the draft draws them too",
                     "\(appDraft.recalled) of \(bakedPixels.count), \(appDraft.stray) astray")
        Check.expect(draft.recalled < bakedPixels.count || draft.stray > 0,
                     "and with no slope term, tested per fragment against the depth pass, does not",
                     "it misses \(bakedPixels.count - draft.recalled) and strays at \(draft.stray)")
    }

    Check.suite("bundles: a polyhedron restyled lays the hatching that changed, and nothing else") {
        let opened = try PolyhedronBundle.read(at: Fixtures.url("contract/polyhedron_octahedron.kurven"))
        var layers = opened.manifest.layers
        let crossed = layers.firstIndex { $0.name == "crossed" }!
        guard case .hatching(var hatching, let faces) = layers[crossed].source else {
            Check.expect(false, "the octahedron's fixture is crossed"); return
        }
        hatching.spacing = 0.2
        layers[crossed].source = .hatching(hatching, faces: faces)
        let restyled = try opened.restyled(layers)
        func paths(_ b: PolyhedronBundle, _ name: String) -> PolylineSet<WorldSpace>? {
            b.layers.first { $0.spec.name == name }?.paths
        }
        Check.expect(restyled.solid == opened.solid && paths(restyled, "edges") == paths(opened, "edges")
                     && paths(restyled, "hatching") == paths(opened, "hatching"),
                     "the solid, the edges and the hatching not restyled are kept as drawn")
        Check.expect(paths(restyled, "crossed") == opened.solid.hatchPaths(hatching)
                     && paths(restyled, "crossed") != paths(opened, "crossed"),
                     "the restyled hatching is laid again at its new spacing")
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("kurven-restyled-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        try restyled.write(to: directory)
        let reopened = try PolyhedronBundle.read(at: directory)
        Check.expect(reopened.manifest.layers == layers && reopened.layers.map(\.paths) == restyled.layers.map(\.paths),
                     "saved and reopened, the restyled bundle draws what was on screen")
        hatching.spacing = 1e-5
        layers[crossed].source = .hatching(hatching, faces: faces)
        Check.expectThrows("a restyle past what a polyhedron's hatchings lay is refused") {
            _ = try opened.restyled(layers)
        }
    }

    Check.suite("bundles: a malformed polyhedron's manifest is refused, saying why") {
        let good = try Fixtures.json("contract/polyhedron_frame.kurven/manifest.json")
        Check.expect((try? PolyhedronManifest(json: .object(good))) != nil, "the fixture itself is read")
        func mutated(_ change: (inout [String: JSONValue]) -> Void) -> JSONValue {
            var d = good; change(&d); return .object(d)
        }
        func polyhedron(vertices: [[Double]]? = nil, faces: JSONValue? = nil) -> JSONValue {
            mutated { d in
                var p = (try? d["polyhedron"]!.object("p")) ?? [:]
                if let vertices { p["vertices"] = .array(vertices.map { .array($0.map(JSONValue.double)) }) }
                if let faces { p["faces"] = faces }
                d["polyhedron"] = .object(p)
            }
        }
        let solid = Solid.frame(outer: 4, inner: 2, depth: 0.8)
        let points = solid.vertices.map { [$0.x, $0.y, $0.z] }
        func faces(_ loops: [[[Int]]]) -> JSONValue {
            .array(loops.map { .array($0.map { .array($0.map(JSONValue.int)) }) })
        }
        func layers(_ specs: [LayerSpec]) -> JSONValue { mutated { $0["layers"] = .array(specs.map(\.json)) } }
        func layer(_ source: LayerSource, width: Double = 0.3) -> JSONValue {
            layers([LayerSpec(name: "x", role: .scaffold, source: source, width: width, heightPolicy: .surface)])
        }
        func refuses(_ what: String, saying words: String, _ json: JSONValue) {
            let clock = ContinuousClock()
            let start = clock.now
            do {
                _ = try BundleManifest(json: json)
                Check.expect(false, what, "decoded")
            } catch {
                Check.expect("\(error)".contains(words) && clock.now - start < .seconds(1), what, "\(error)")
            }
        }
        var loops = solid.loops
        loops[0][0][2] = 16
        refuses("refuses a vertex index past the last", saying: "names vertex 16, and there are 16",
                polyhedron(faces: faces(loops)))
        loops = solid.loops
        loops[2] = [[0, 1]]
        refuses("refuses a loop of two vertices", saying: "at least three", polyhedron(faces: faces(loops)))
        loops = solid.loops
        loops.removeLast()
        refuses("refuses an open surface", saying: "not closed", polyhedron(faces: faces(loops)))
        var bent = points
        bent[0][2] = 0.1
        refuses("refuses a face that is not flat", saying: "not flat", polyhedron(vertices: bent))
        var far = points
        far[3][0] = 1e39
        refuses("refuses a coordinate past what the depth pass holds", saying: "not a finite point",
                polyhedron(vertices: far))
        do {
            let text = try String(contentsOf: Fixtures.url("contract/polyhedron_frame.kurven/manifest.json"),
                                  encoding: .utf8)
            refuses("refuses a coordinate JSON reads as infinite", saying: "not a finite point",
                    try JSONValue.parse(text.replacingOccurrences(of: "[2.0,2.0,0.8]", with: "[2.0,-1e400,0.8]")))
            refuses("refuses a hatching at an angle that is not a number", saying: "not a number",
                    try JSONValue.parse(text.replacingOccurrences(of: "\"angle\":0.5235987755982988",
                                                                  with: "\"angle\":-1e400")))
        }
        refuses("refuses a vertex of two coordinates", saying: "vertices[1] has shape [2]",
                polyhedron(vertices: [points[0], [1, 1]] + points.dropFirst(2)))
        refuses("refuses a face that is not a list of loops", saying: "faces[0]",
                polyhedron(faces: .array([.int(1)])))
        refuses("refuses more vertices than a polyhedron holds", saying: "4097 vertices, more than the 4096",
                polyhedron(vertices: Array(repeating: [0, 0, 0], count: 4097)))
        refuses("refuses a face of more corners than a face holds", saying: "more corners than a polyhedron holds",
                polyhedron(faces: faces([[Array(repeating: 0, count: 1025)]])))
        refuses("refuses more corners in all than a polyhedron holds", saying: "more corners than a polyhedron holds",
                polyhedron(faces: faces(Array(repeating: [Array(repeating: 0, count: 1000)], count: 17))))
        let edges = LayerSpec(name: "edges", role: .outline, source: .edges, width: 0.6, heightPolicy: .surface)
        refuses("refuses more layers than a polyhedron holds", saying: "65 layers, more than the 64",
                layers((0..<65).map { var e = edges; e.name = "edges \($0)"; return e }))
        refuses("refuses two layers of one name", saying: "two layers named 'edges'", layers([edges, edges]))
        refuses("refuses a contour on its faces", saying: "a contour on a polyhedron's faces",
                layer(.contour(field: .u, levels: [0], restriction: .all, faces: .solid)))
        refuses("refuses a dumped layer", saying: "a dumped layer on a polyhedron",
                layer(.file(vertices: "v.npy", offsets: "o.npy")))
        refuses("refuses a layer of another geometry", saying: "a coordinateCurves layer on a polyhedron",
                layer(.coordinateCurves(u: 4, v: 4)))
        refuses("refuses hatching on a landscape's faces", saying: "hatching on the plateaus of a polyhedron",
                layer(.plateauHatching(axis: .real, spacing: 0.1)))
        refuses("refuses a negative width", saying: "a width must be a number not below 0",
                layer(.edges, width: -1))
        let turing = Hatching(angle: 0.5, spacing: 0.05, tone: .turing(scale: 0.2, seed: 1))
        var wide = turing
        wide.widthSteps = 17
        refuses("refuses more widths than a hatch line has", saying: "17 widths, more than the 16",
                layer(.hatching(wide, faces: .solid)))
        refuses("refuses Turing's pattern finer than a billionth of a face", saying: "too fine for its lines",
                layer(.hatching(Hatching(angle: 0.5, spacing: 0.05, tone: .turing(scale: 1e-12, seed: 1)),
                                faces: .solid)))
        // Review's cube: within every count read off the lines and the
        // readings, but with the pattern far finer than a reading's step it
        // changes width at nearly every reading, each change bisected and a
        // stroke of its own: 216 s and 7.27 million strokes before the
        // changes were counted.
        do {
            let cube = Solid.platonic(.cube, circumradius: 2).resting(onFace: 0)
            var manifest = try PolyhedronManifest(json: .object(good))
            manifest.solid = cube
            let fine = Hatching(angle: .pi / 4, spacing: 0.003266, step: 0.01921, tone: .turing(scale: 1e-5, seed: 1),
                                widthSteps: 16)
            manifest.layers = [edges, LayerSpec(name: "hatching", role: .scaffold,
                                                source: .hatching(fine, faces: .solid), width: 0.3,
                                                heightPolicy: .surface)]
            refuses("refuses Turing's pattern too fine for its readings, quickly", saying: "change width",
                    manifest.json)
        }
        refuses("refuses hatch lines past what its hatchings lay", saying: "lines, more than the 65536",
                layer(.hatching(Hatching(angle: 0.5, spacing: 1e-5), faces: .solid)))
        refuses("refuses tones read past what its hatchings read", saying: "times, more than the 1048576",
                layer(.hatching(Hatching(angle: 0.5, spacing: 0.05, step: 1e-7), faces: .solid)))
        refuses("refuses its hatchings' work in all, not one at a time", saying: "lines, more than the 65536",
                layers((0..<4).map {
                    LayerSpec(name: "h\($0)", role: .scaffold, source: .hatching(Hatching(angle: 0.5, spacing: 0.0005),
                                                                                 faces: .solid),
                              width: 0.3, heightPolicy: .surface)
                }))
        // A preset's buffer is a depth buffer's resolution, in any kind of
        // manifest: none at all aborted on Metal, and a million exhausted
        // memory.
        func buffer(_ json: [String: JSONValue], _ n: Int) -> JSONValue {
            var d = json
            if case .array(var presets) = d["presets"], var first = try? presets[0].object("p") {
                first["buffer"] = .int(n)
                presets[0] = .object(first)
                d["presets"] = .array(presets)
            }
            return .object(d)
        }
        for (kind, json) in [("a polyhedron's", good),
                             ("a parametric surface's", try Fixtures.json("contract/surface.kurven/manifest.json")),
                             ("a landscape's", try Fixtures.json("contract/landscape.kurven/manifest.json"))] {
            for n in [0, -5, 20_001, 1_000_000] {
                refuses("refuses \(kind) preset with a buffer of \(n)", saying: "buffer must be 1 to 20000 pixels",
                        buffer(json, n))
            }
            Check.expect((try? BundleManifest(json: buffer(json, 20_000))) != nil,
                         "\(kind) preset with a buffer of 20000 is read")
        }
        refuses("refuses a landscape's key beside a polyhedron", saying: "a polyhedron with a landscape's domain",
                mutated { $0["domain"] = .object([:]) })
        refuses("refuses a parametric surface beside a polyhedron", saying: "a polyhedron with a parametric surface",
                mutated { $0["surface"] = .object([:]) })
        do {
            _ = try Manifest(json: .object(good))
            Check.expect(false, "a landscape's reader refuses a polyhedron, naming its reader", "decoded")
        } catch {
            Check.expect("\(error)".contains("PolyhedronManifest"),
                         "a landscape's reader refuses a polyhedron, naming its reader", "\(error)")
        }
        let surface = try Fixtures.json("contract/surface.kurven/manifest.json")
        do {
            _ = try SurfaceManifest(json: .object(surface.merging(["polyhedron": good["polyhedron"]!]) { $1 }))
            Check.expect(false, "a parametric surface's reader refuses a polyhedron beside it", "decoded")
        } catch {
            Check.expect("\(error)".contains("a parametric surface with a polyhedron"),
                         "a parametric surface's reader refuses a polyhedron beside it", "\(error)")
        }

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("kurven-unviewed-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try mutated { $0["presets"] = .array([]) }.canonical
            .write(to: directory.appendingPathComponent("manifest.json"), atomically: true, encoding: .utf8)
        Check.expectThrows("a polyhedron with no camera preset is refused when it is read") {
            _ = try PolyhedronBundle.read(at: directory)
        }
    }
}
