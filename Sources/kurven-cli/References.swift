import Foundation
import simd
import KurvenCore
import KurvenBake
import KurvenLandscape
import KurvenMetal

/// `kurven-cli references <fixtures> <dir>` -- the drawings the web port is
/// held to, baked here so the port's check never needs Swift.
///
/// The port (`crates/`, `HANDOFF-web-port.md`) compares its drawings with
/// these per layer, by Hausdorff distance and total length, with
/// `kurven-cli compare` in Rust. Each is drawn by the path the command named
/// in its `how` draws it: a bundle written by `landscape` or `surface ...
/// --save` and drawn by `bake`, the golden slope lines as `kurven-cli
/// fixtures` draws them, and the composite as `scene` draws it. Nothing here
/// changes how anything is drawn.
///
/// Written into `<dir>`: one `<name>.svg` per drawing and `index.json`, which
/// lists each drawing with how it was made and the names of its stroke
/// layers in the order the SVG's groups stand (an SVG leaves out an empty
/// layer, and so does the list). `<fixtures>` is `tests/fixtures`, which
/// holds the bundles some of them are drawn from.
enum ReferenceDrawings {
    /// One drawing: its file stem, the commands it stands for, and how to
    /// draw it into a scratch directory.
    struct Reference {
        var name: String
        var how: String
        var draw: (_ scratch: URL) throws -> (strokes: Strokes, names: [String])
    }

    /// A bundle's drawing as `bake` draws it: under `preset` or its first,
    /// at that preset's depth resolution and tolerance, its contours placed
    /// by its function unless not `refine`.
    static func baked(_ bundle: URL, preset name: String? = nil, refine: Bool = true) throws
        -> (strokes: Strokes, names: [String])
    {
        var words = ["bake", bundle.path]
        if let name { words += ["--preset", name] }
        if !refine { words.append("--no-refine") }
        let (_, _, preset, scene) = try loadDrawing(Args(words))
        let options = BakeOptions(resolution: preset.buffer, tiles: nil,
                                  depthTolerance: preset.depthTolerance, outline: nil)
        let result = try MetalRenderer().bake(scene, options: options)
        let names = result.strokes.layers.indices.map { $0 < result.specs.count ? result.specs[$0].name : "outline" }
        return (result.strokes, names)
    }

    /// `surface ... --save B; bake B`.
    static func surface(_ words: [String], _ scratch: URL, _ stem: String) throws -> (strokes: Strokes, names: [String]) {
        let bundle = scratch.appendingPathComponent("\(stem).kurven")
        try surfaceCommand(Args(["surface"] + words + ["--save", bundle.path]))
        return try baked(bundle)
    }

    // MARK: the gallery, as docs/gallery/make.py sets it up

    /// The plates hatch the plateaus, walls and risers this much closer than
    /// a landscape does by default (make.py's `HATCH`).
    static let galleryHatch = 0.75

    /// `landscape` as make.py calls it, then the manifest edited as make.py
    /// edits it: one preset per camera, each the first preset with its name
    /// and its projection's fields replaced, and the plateaus', walls' and
    /// risers' hatching `galleryHatch` closer.
    static func galleryLandscape(_ scratch: URL, _ stem: String, _ words: [String],
                                 cameras: [(String, [String: JSONValue])]) throws -> URL {
        let bundle = scratch.appendingPathComponent("\(stem).kurven")
        try landscapeCommand(Args(["landscape"] + words + ["-o", bundle.path]))
        let path = bundle.appendingPathComponent("manifest.json")
        var m = try JSONValue.parse(Data(contentsOf: path)).object("manifest")
        guard case .array(let presets)? = m["presets"], case .object(let base)? = presets.first,
              case .object(let projection)? = base["projection"] else {
            throw CLIError("\(stem): the landscape wrote no preset to start from")
        }
        m["presets"] = .array(cameras.map { name, fields in
            var preset = base
            preset["name"] = .string(name)
            preset["projection"] = .object(projection.merging(fields) { $1 })
            return .object(preset)
        })
        guard case .array(let layers)? = m["layers"] else { throw CLIError("\(stem): no layers") }
        m["layers"] = .array(layers.map { layer in
            guard case .object(var l) = layer, case .object(var source)? = l["source"],
                  source["kind"] == .string("hatching"), case .object(let faces)? = source["faces"],
                  case .string(let kind)? = faces["kind"], ["plateaus", "walls", "risers"].contains(kind) else {
                return layer
            }
            switch source["spacing"] {
            case .double(let s)?: source["spacing"] = .double(s * galleryHatch)
            case .int(let s)?: source["spacing"] = .double(Double(s) * galleryHatch)
            default: return layer
            }
            l["source"] = .object(source)
            return .object(l)
        })
        try Data(JSONValue.object(m).canonical.utf8).write(to: path)
        return bundle
    }

    /// A bundle sampled by the first drawing that needs it, for the rest.
    final class Sampled {
        var bundle: URL?
    }

    static func camera(shear: Double, xAngle: Double, zAngle: Double, flipX: Bool) -> [String: JSONValue] {
        ["shear": .double(shear), "xAngle": .double(xAngle), "zAngle": .double(zAngle), "flipX": .bool(flipX)]
    }

    /// The arithmetic-geometric mean, as make.py computes it for cn's quarter
    /// periods.
    static func agm(_ a0: Double, _ b0: Double) -> Double {
        var (a, b) = (a0, b0)
        while abs(a - b) > 1e-15 * a { (a, b) = ((a + b) / 2, (a * b).squareRoot()) }
        return a
    }

    static func gallery() -> [Reference] {
        let gammaWords = ["gamma(z)", "--re", "-4.5,4.5", "--im", "-2.5,2.5", "--res", "1000",
                          "--bands", "-3.5:2.5,-2.5:3.0,-1.5:4.0,0.5:5.5", "--beyond", "4.0"]
        let m = 0.64
        let K = Double.pi / (2 * agm(1, (1 - m).squareRoot()))
        let Kp = Double.pi / (2 * agm(1, m.squareRoot()))
        let cnWords = ["cn(z, \(m))", "--re", "\(-K),\(5 * K)", "--im", "0,\(3 * Kp)", "--res", "1200",
                       "--truncation", "4"]
        let cubicWords = ["1/(z^3 - 1)", "--re", "-1.3,1.6", "--im", "-1.2,1.5", "--res", "300", "--truncation", "5"]
        let corners: [(String, [String: JSONValue])] = [(-90, 0.5), (-90, -0.5), (90, 0.5), (90, -0.5)].map { z, s in
            ("t\(z)s\(s)".replacingOccurrences(of: "-", with: "m"),
             camera(shear: s, xAngle: -55, zAngle: Double(z), flipX: true))
        }
        let frame = ["solid", "--kind", "frame", "--size", "2", "--hole", "2", "--depth", "0.8",
                     "--x-angle", "-55", "--z-angle", "-60", "--shear", "0", "--width", "0.6"]
        let turing = ["--hatching", "turing", "--angle", "30", "--spacing", "0.035", "--width-steps", "4",
                      "--scale", "0.18", "--seed", "3", "--hatch-width", "0.5"]
        var references = [
            Reference(name: "gallery-gamma",
                      how: "docs/gallery/make.py: landscape \(gammaWords.joined(separator: " ")), "
                           + "hatching x0.75, preset plate (shear -0.5, x -55, z -90); bake --preset plate") { scratch in
                let bundle = try galleryLandscape(scratch, "gamma", gammaWords, cameras: [
                    ("plate", camera(shear: -0.5, xAngle: -55, zAngle: -90, flipX: false))])
                return try baked(bundle, preset: "plate")
            },
            Reference(name: "gallery-cn",
                      how: "docs/gallery/make.py: landscape \(cnWords.joined(separator: " ")), "
                           + "hatching x0.75, preset plate (shear 0.51, x -63, z -90, flipX); bake --preset plate") { scratch in
                let bundle = try galleryLandscape(scratch, "cn", cnWords, cameras: [
                    ("plate", camera(shear: 0.51, xAngle: -63, zAngle: -90, flipX: true))])
                return try baked(bundle, preset: "plate")
            },
        ]
        // make.py samples the cubic once and bakes each corner's preset from
        // that one bundle; so does this.
        let cubic = Sampled()
        for (key, _) in corners {
            references.append(Reference(name: "gallery-cubic-\(key)",
                                        how: "docs/gallery/make.py: landscape \(cubicWords.joined(separator: " ")), "
                                             + "hatching x0.75, the four corner presets; bake --preset \(key)") { scratch in
                let bundle = try cubic.bundle ?? galleryLandscape(scratch, "cubic", cubicWords, cameras: corners)
                cubic.bundle = bundle
                return try baked(bundle, preset: key)
            })
        }
        references.append(Reference(name: "gallery-frame-edges",
                                    how: "docs/gallery/make.py: surface \(frame.joined(separator: " ")) --save B; bake B") {
            try surface(frame, $0, "frame-edges")
        })
        references.append(Reference(name: "gallery-frame-turing",
                                    how: "docs/gallery/make.py: surface \((frame + turing).joined(separator: " ")) --save B; bake B") {
            try surface(frame + turing, $0, "frame-turing")
        })
        return references
    }

    // MARK: the rest of the list

    static func all(fixtures: URL) -> [Reference] {
        var references = gallery()
        // The native landscapes of expr/landscapes.json, drawn on their function.
        for stem in ["gamma", "cubic", "typed", "tower"] {
            let bundle = fixtures.appendingPathComponent("expr/landscape_\(stem).kurven")
            references.append(Reference(name: "landscape-\(stem)",
                                        how: "bake tests/fixtures/expr/landscape_\(stem).kurven") { _ in try baked(bundle) })
        }
        let surfaces: [(String, [String])] = [
            ("sn", ["sn"]),
            ("sn-relief", ["sn", "--relief", "0.4"]),
            ("torus-linear", ["torus", "--linear-curves", "2/5"]),
            ("periodic", ["periodic", "--expression", "sn(z, 0.5)", "--periods", "7.416298709205487,3.7081493546027433"]),
        ]
        for (name, words) in surfaces {
            references.append(Reference(name: name, how: "surface \(words.joined(separator: " ")) --save B; bake B") {
                try surface(words, $0, name)
            })
        }
        // The slope lines of |f| golden (PR #60), drawn as `kurven-cli
        // fixtures` draws it: tests/fixtures/fields/landscape_slope_lines.svg,
        // byte for byte.
        let slopes = fixtures.appendingPathComponent("fields/landscape_slope_lines.kurven")
        references.append(Reference(name: "slope-lines",
                                    how: "kurven-cli fixtures: tests/fixtures/fields/landscape_slope_lines.svg") { _ in
            let (_, _, _, scene) = try loadDrawing(Args(["bake", slopes.path]))
            let layers = scene.visibleLayers()
            return (Strokes(layers: layers.map { layer, seen in (Style(layer.spec), HiddenLine.pass(seen)) }),
                    layers.map { $0.0.spec.name })
        })
        for kind in ["tetrahedron", "cube", "octahedron", "dodecahedron", "icosahedron", "csaszar", "frame"] {
            references.append(Reference(name: "solid-\(kind)", how: "surface solid --kind \(kind) --save B; bake B") {
                try surface(["solid", "--kind", kind], $0, "solid-\(kind)")
            })
        }
        // The cubic landscape on its grid with the contract's cube and torus
        // standing on it: tests/fixtures/composite/three.svg, byte for byte.
        let parts: [(String, Placement)] = [
            ("expr/landscape_cubic.kurven", .identity),
            ("contract/polyhedron_cube.kurven", Placement(scale: 0.35, offset: SIMD3(-1.2, 0.1, 0.3))),
            ("contract/surface_torus.kurven", Placement(scale: 0.35, offset: SIMD3(1, 0, 1.6))),
        ]
        references.append(Reference(name: "composite-three",
                                    how: "scene expr/landscape_cubic.kurven contract/polyhedron_cube.kurven --at -1.2,0.1,0.3,0.35 "
                                         + "contract/surface_torus.kurven --at 1,0,1.6,0.35 --no-refine") { _ in
            let drawing = try compositeDrawing(parts.map { (fixtures.appendingPathComponent($0.0).path, $0.1) },
                                               preset: nil, refine: false)
            return (drawing.strokes, drawing.layers.map { $0.0.spec.name })
        })
        return references
    }

    static func write(fixtures: URL, to dir: URL) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("kurven-references-\(ProcessInfo.processInfo.processIdentifier)")
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        var entries: [JSONValue] = []
        for reference in all(fixtures: fixtures) {
            let (strokes, names) = try reference.draw(scratch)
            guard names.count == strokes.layers.count else {
                throw CLIError("\(reference.name): \(names.count) names for \(strokes.layers.count) layers")
            }
            try Data(SVG.render(strokes).utf8).write(to: dir.appendingPathComponent("\(reference.name).svg"))
            // As the SVG has them: the layers with a stroke, in order.
            let drawn = strokes.layers.indices.filter { !strokes.layers[$0].paths.isEmpty }
            entries.append(.object([
                "name": .string(reference.name),
                "how": .string(reference.how),
                "layers": .array(drawn.map { .string(names[$0]) }),
                "paths": .array(drawn.map { .int(strokes.layers[$0].paths.count) }),
            ]))
            FileHandle.standardError.write(Data("  \(reference.name)\n".utf8))
        }
        try Data((JSONValue.object(["drawings": .array(entries)]).pretty(ascii: false) + "\n").utf8)
            .write(to: dir.appendingPathComponent("index.json"))
    }
}

func referencesCommand(_ args: Args) throws {
    try args.only("references", [])
    let words = args.positional.dropFirst()
    guard words.count == 2, let fixtures = words.first, let dir = words.last else {
        throw CLIError("references wants <fixtures> <dir>: tests/fixtures and where to write the drawings")
    }
    let clock = ContinuousClock()
    let elapsed = try clock.measure {
        try ReferenceDrawings.write(fixtures: URL(fileURLWithPath: fixtures), to: URL(fileURLWithPath: dir))
    }
    print("wrote the reference drawings under \(dir) in \(elapsed)")
}
