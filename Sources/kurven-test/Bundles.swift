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
        let restyled = try opened.restyled(layers)
        let again = directory.appendingPathComponent("restyled.kurven")
        try opened.write(to: again, layers: layers)
        let reopened = try SurfaceBundle.read(at: again)

        func paths(_ b: SurfaceBundle, _ name: String) -> PolylineSet<WorldSpace>? {
            b.drawing.layers.first { $0.spec.name == name }?.paths
        }
        Check.expect(restyled.drawing.content == opened.drawing.content
                     && restyled.drawing.surface.positions == opened.drawing.surface.positions,
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
                     && reopened.drawing.layers.map(\.spec) == restyled.drawing.layers.map(\.spec)
                     && reopened.drawing.layers.map(\.paths) == restyled.drawing.layers.map(\.paths),
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
        refuses("refuses a radius that is not positive", saying: "major radius must be a positive number, not -1.0",
                shape(sn.merging(["major": .double(-1)]) { $1 }))
        refuses("refuses a grid too coarse to span its rectangle", saying: "grid must be at least 8 samples, not 4",
                surface { $0["grid"] = .int(4) })
        refuses("refuses a shape it does not know, naming the ones it does", saying: "'klein' is not one of",
                shape(["kind": .string("klein")]))
        refuses("refuses a forced torus revolved on one state component twice", saying: "both state component 1",
                shape(forced))
        refuses("refuses a period that is not positive", saying: "real period must be a positive number",
                shape(["kind": .string("periodic"), "expression": .string("sn(z, 0.5)"),
                       "periods": .array([.double(0), .double(1)]), "major": .double(2), "minor": .double(1),
                       "grid": .int(120)]))
        refuses("refuses an empty expression", saying: "expression is empty",
                shape(["kind": .string("periodic"), "expression": .string(" "),
                       "periods": .array([.double(1), .double(1)]), "major": .double(2), "minor": .double(1),
                       "grid": .int(120)]))
        refuses("refuses a relief repeated less than once", saying: "repeats must be whole periods",
                shape(relief))
        refuses("refuses a negative count of coordinate curves", saying: "coordinate curves must be counts",
                surface { $0["lines"] = .object(["u": .int(-1), "v": .int(6)]) })
        refuses("refuses a linear curve of no turns", saying: "at least one turn of one strand",
                surface { $0["linearCurves"] = .object(["slope": .double(0.4), "turns": .int(0), "count": .int(1)]) })
        refuses("refuses a negative stroke width", saying: "folds width must be a number not below 0",
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
