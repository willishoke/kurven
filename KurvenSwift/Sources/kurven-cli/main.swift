import Foundation
import KurvenCore
import KurvenMetal
import KurvenBake

/// `kurven-cli` -- the headless half of the frontend.
///
/// The app is not the product of Phase 1; this is. Everything the interactive
/// shell will do to a scene, this does to the same `Scene` value, so "the UI
/// bakes the same picture as the CLI" is true because they are one function
/// called twice, and the picture can be checked against the Python plates
/// before any window exists.
///
///     kurven-cli bake recip.kurven --preset recip -o recip.svg
///     kurven-cli inspect recip.kurven
///     kurven-cli contract tests/fixtures/contract
///     kurven-cli depth recip.kurven --preset recip -o depth.npy

struct CLIError: Error, CustomStringConvertible {
    let description: String
    init(_ d: String) { description = d }
}

struct Args {
    var positional: [String] = []
    var flags: [String: String] = [:]
    var switches: Set<String> = []

    init(_ argv: [String]) {
        var i = 0
        while i < argv.count {
            let a = argv[i]
            if a.hasPrefix("--") {
                let name = String(a.dropFirst(2))
                if i + 1 < argv.count, !argv[i + 1].hasPrefix("--") {
                    flags[name] = argv[i + 1]; i += 2
                } else {
                    switches.insert(name); i += 1
                }
            } else if a == "-o", i + 1 < argv.count {
                flags["output"] = argv[i + 1]; i += 2
            } else {
                positional.append(a); i += 1
            }
        }
    }

    func string(_ name: String) throws -> String {
        guard let v = flags[name] else { throw CLIError("missing --\(name)") }
        return v
    }
    func int(_ name: String, _ fallback: Int) throws -> Int {
        guard let v = flags[name] else { return fallback }
        guard let n = Int(v) else { throw CLIError("--\(name) wants an integer, got '\(v)'") }
        return n
    }
    func double(_ name: String) throws -> Double? {
        guard let v = flags[name] else { return nil }
        guard let d = Double(v) else { throw CLIError("--\(name) wants a number, got '\(v)'") }
        return d
    }
}

let usage = """
usage: kurven-cli <command> [options]

  bake <bundle> [--preset NAME] [--resolution N] [--tiles N] [--margin M] -o out.svg
        Render a bundle's plate to SVG through the depth-tested hidden-line
        pipeline. Defaults to the bundle's first preset and that preset's own
        depth resolution and clip margin -- the settings the published plate
        was made with.

  depth <bundle> [--preset NAME] [--resolution N] -o depth.npy
        Dump the depth buffer as float32 .npy, for comparison against the
        Python Z-buffer. This is what stands in for a GPU frame capture.

  inspect <bundle>
        Print the manifest: domain, caps, occluder, layers, presets, provenance.

  contract <dir>
        Decode every .kurven bundle in <dir>, re-encode its manifest, and
        compare canonical JSON against the file. The schema test.
"""

func loadScene(_ args: Args) throws -> (KurvenBundle, CameraPreset, Scene) {
    guard let path = args.positional.dropFirst().first else {
        throw CLIError("which bundle?")
    }
    let bundle = try KurvenBundle.read(at: URL(fileURLWithPath: path))
    let preset: CameraPreset
    if let name = args.flags["preset"] {
        preset = try bundle.manifest.preset(name)
    } else if let first = bundle.manifest.presets.first {
        preset = first
    } else {
        throw CLIError("bundle has no presets; pass --preset once one exists")
    }
    return (bundle, preset, Scene(bundle: bundle, preset: preset))
}

func bake(_ args: Args) throws {
    let (bundle, preset, scene) = try loadScene(args)
    let output = URL(fileURLWithPath: try args.string("output"))
    let options = BakeOptions(
        resolution: try args.int("resolution", preset.buffer),
        tiles: args.flags["tiles"].flatMap(Int.init),
        margin: try args.double("margin") ?? preset.margin)

    if !bundle.manifest.provenance.isReproducible {
        FileHandle.standardError.write(Data("""
            note: this bundle was contoured with \
            \(bundle.manifest.provenance.cpuCount) chunks, so its contours were \
            stitched in thread-completion order and are not reproducible.\n
            """.utf8))
    }

    let renderer = try MetalRenderer()
    let clock = ContinuousClock()
    var result: Bake!
    let elapsed = try clock.measure { result = try renderer.bake(scene, options: options) }
    try SVG.render(result.strokes).write(to: output, atomically: true, encoding: .utf8)

    print("""
        \(bundle.url.lastPathComponent) -> \(output.lastPathComponent)  [\(preset.name)]
          depth      \(result.frame.rows)x\(result.frame.cols) \
        in \(result.depths.count) tile\(result.depths.count == 1 ? "" : "s")
          strokes    \(result.strokes.pathCount) paths, \
        ink \(String(format: "%.1f", result.strokes.inkLength))
          took       \(elapsed)
        """)
    for (index, entry) in result.strokes.layers.enumerated() {
        let spec = scene.layers[index].spec
        print("    \(pad(spec.name, 14)) \(pad(String(entry.paths.count), 7)) paths"
              + "  lw \(entry.style.width)"
              + (spec.clipped ? "" : "  unclipped"))
    }
}

func depth(_ args: Args) throws {
    let (_, preset, scene) = try loadScene(args)
    let output = URL(fileURLWithPath: try args.string("output"))
    let resolution = try args.int("resolution", preset.buffer)
    guard let bounds = scene.viewBounds() else { throw CLIError("the scene is empty") }
    let frame = DepthFrame(covering: bounds, resolution: resolution)
    let image = try MetalRenderer().renderDepth(scene, frame: frame)
    // -infinity does not survive a float32 npy round trip through numpy's
    // comparisons cleanly, but it is exactly what the Python fill is, so it
    // goes out as-is and the reader sees the same value the clipper saw.
    try NPY.write(image.values, shape: [frame.rows, frame.cols], to: output)
    let drawn = image.values.reduce(0) { $1.isFinite ? $0 + 1 : $0 }
    print("""
        depth \(frame.rows)x\(frame.cols) -> \(output.lastPathComponent)
          axis0  [\(frame.axis0.lo), \(frame.axis0.hi)]
          axis1  [\(frame.axis1.lo), \(frame.axis1.hi)]
          filled \(drawn) of \(image.values.count) pixels
        """)
}

func inspect(_ args: Args) throws {
    let (bundle, _, _) = try loadScene(args)
    let m = bundle.manifest
    print("\(bundle.url.lastPathComponent)  schema \(m.schema)  axes \(m.axes.joined(separator: ", "))")
    print("  function   \(m.provenance.function)  git \(m.provenance.gitSha.prefix(8))"
          + (m.provenance.isReproducible ? "" : "  (chunks: \(m.provenance.cpuCount), NOT reproducible)"))
    print("  domain     real [\(m.domain.real.lo), \(m.domain.real.hi)]  "
          + "imag [\(m.domain.imag.lo), \(m.domain.imag.hi)]")
    print("  height     \(m.height.shape.ny)x\(m.height.shape.nx) \(m.height.dtype.rawValue)"
          + (m.phase == nil ? "  (no phase)" : "  + phase"))
    print("  caps       \(m.caps)")
    print("  occluder   step \(m.occluder.step)  \(m.occluder.tiles.count) tile(s)  "
          + "region \(regionName(m.occluder.region))  walls \(wallsName(m.occluder.walls))")
    for l in m.layers {
        let paths = (try? bundle.layer(l.name).paths.count) ?? 0
        print("    " + pad(l.name, 12) + " " + pad(l.role.rawValue, 10) + " "
              + pad(String(paths), 7) + " paths  lw \(l.width)"
              + (l.clipped ? "" : "  unclipped"))
    }
    for p in m.presets {
        print("  preset     \(pad(p.name, 10)) shear \(p.plate.shear)  "
              + "x \(p.plate.xAngle)  z \(p.plate.zAngle)  "
              + "flipX \(p.plate.flipX)  yScale \(p.plate.yScale.map { String($0) } ?? "-")  "
              + "margin \(p.margin)  buffer \(p.buffer)")
    }
}

func contract(_ args: Args) throws {
    guard let dir = args.positional.dropFirst().first else { throw CLIError("which directory?") }
    let url = URL(fileURLWithPath: dir)
    let bundles = try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "kurven" }.sorted { $0.path < $1.path }
    guard !bundles.isEmpty else { throw CLIError("no .kurven bundles in \(dir)") }
    var failures = 0
    for b in bundles {
        let text = try String(contentsOf: b.appendingPathComponent("manifest.json"), encoding: .utf8)
        let manifest = try Manifest(json: JSONValue.parse(text))
        let again = manifest.canonicalJSON
        if again == text {
            print("  ok    \(b.lastPathComponent)")
        } else {
            failures += 1
            print("  FAIL  \(b.lastPathComponent)")
            print(firstDifference(text, again))
        }
    }
    if failures > 0 { throw CLIError("\(failures) bundle(s) did not round trip") }
    print("all green")
}

func firstDifference(_ a: String, _ b: String) -> String {
    let x = Array(a), y = Array(b)
    var i = 0
    while i < min(x.count, y.count), x[i] == y[i] { i += 1 }
    let from = max(0, i - 40), to = min(min(x.count, y.count), i + 40)
    return "        at \(i):\n          python: \(String(x[from..<min(x.count, to)]))\n"
         + "          swift : \(String(y[from..<min(y.count, to)]))"
}

func pad(_ s: String, _ n: Int) -> String {
    s.count >= n ? s : s + String(repeating: " ", count: n - s.count)
}

func regionName(_ r: Region) -> String {
    switch r {
    case .full: "full"
    case .inside(let p): "inside(\(p.edges.count) edges)"
    }
}

func wallsName(_ w: Walls) -> String {
    switch w {
    case .none: "none"
    case .mesh: "mesh"
    case .perimeter(let p, _): "perimeter(\(p.edges.count) edges)"
    }
}

// MARK: - entry

let args = Args(Array(CommandLine.arguments.dropFirst()))
do {
    switch args.positional.first {
    case "bake": try bake(args)
    case "depth": try depth(args)
    case "inspect": try inspect(args)
    case "contract": try contract(args)
    default:
        print(usage)
        exit(args.positional.isEmpty ? 0 : 1)
    }
} catch {
    FileHandle.standardError.write(Data("kurven-cli: \(error)\n".utf8))
    exit(1)
}
