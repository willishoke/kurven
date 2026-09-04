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

  bake <bundle> [--preset NAME] [--resolution N] [--tiles N] [--margin M]
       [--dump PREFIX] -o out.svg
        Render a bundle's plate to SVG through the depth-tested hidden-line
        pipeline. Defaults to the bundle's first preset and that preset's own
        depth resolution and clip margin -- the settings the published plate
        was made with. --dump also writes each layer's strokes as
        PREFIX.<layer>.npy plus CSR offsets, which is what
        tests/compare_bake.py reads to check the result against the Python
        plate stroke for stroke.

  depth <bundle> [--preset NAME] [--resolution N] -o depth.npy
        Dump the depth buffer as float32 .npy, for comparison against the
        Python Z-buffer. This is what stands in for a GPU frame capture.

  bench <bundle> [--preset NAME] [--resolution N] [--frames N]
        Render the depth pass repeatedly from a moving camera and report the
        frame time. This is the Phase 2 question -- whether navigation can hold
        60 fps -- asked without a window: the preview does exactly this work
        per frame, plus a line pass that costs a fraction of it.

  preview <bundle> [--preset NAME] [--width N] [--height N] [--mode M]
          [--orbit "AZ,EL"] [--zoom F] -o out.png
        Render one preview frame offscreen and write it as a PNG. Modes:
        plate (the default), shaded, depth. --orbit turns the preset camera by
        that many degrees before drawing. This is how the preview is checked
        against the plate without a window in the way.

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

    if let prefix = args.flags["dump"] {
        // Strokes as arrays, so the comparison against the Python plate is on
        // geometry rather than on rasterized pixels. An SVG diff would measure
        // the renderer; this measures the drawing.
        for (index, entry) in result.strokes.layers.enumerated() {
            let name = scene.layers[index].spec.name
            let flat = entry.paths.vertices.flatMap { [Float($0.x), Float($0.y)] }
            try NPY.write(flat, shape: [entry.paths.vertices.count, 2],
                          to: URL(fileURLWithPath: "\(prefix).\(name).npy"))
            try NPY.write(entry.paths.offsets.map(Float.init),
                          shape: [entry.paths.offsets.count],
                          to: URL(fileURLWithPath: "\(prefix).\(name).idx.npy"))
        }
        let f = result.depth.frame
        let meta = JSONValue.object([
            "axis0": .array([.double(f.axis0.lo), .double(f.axis0.hi)]),
            "axis1": .array([.double(f.axis1.lo), .double(f.axis1.hi)]),
            "shape": .array([.int(f.rows), .int(f.cols)]),
            "margin": .double(options.margin ?? scene.margin),
            "layers": .array(scene.layers.map { .string($0.spec.name) }),
        ])
        try (meta.canonical + "\n").write(to: URL(fileURLWithPath: "\(prefix).frame.json"),
                                          atomically: true, encoding: .utf8)
    }

    print("""
        \(bundle.url.lastPathComponent) -> \(output.lastPathComponent)  [\(preset.name)]
          depth      \(result.depth.frame.rows)x\(result.depth.frame.cols) \
        in \(result.tiles * result.tiles) pass\(result.tiles == 1 ? "" : "es")
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

func bench(_ args: Args) throws {
    let (bundle, preset, base) = try loadScene(args)
    let resolution = try args.int("resolution", 1024)
    let frames = try args.int("frames", 60)
    let viewport = Viewport(width: resolution, height: resolution)
    guard let bounds = base.viewBounds() else { throw CLIError("the scene is empty") }
    let frame = DepthFrame(covering: bounds, resolution: resolution)
    let renderer = try MetalRenderer()
    let clock = ContinuousClock()

    // A camera that moves: the plate's own projection, orbited. Every frame
    // shares the scene's `content`, so this measures what navigation costs
    // once the resources are built -- which is the whole reason they are
    // separated from the camera.
    func camera(_ i: Int) -> Camera {
        var plate = preset.plate
        plate.zAngle = preset.plate.zAngle + Double(i) * 0.25
        return .plate(plate)
    }

    // Two things are worth timing separately: the depth pass alone, which is
    // what a bake pays per tile, and the whole preview, which is what a drag
    // pays per frame. Reporting only the first would flatter the app.
    let modeName = args.flags["mode"] ?? "plate"
    let mode: PreviewMode = modeName == "shaded" ? .shaded(Lighting())
        : modeName == "depth" ? .depth : .plate
    let target = try renderer.makePreviewTarget(viewport)
    var navigator = Navigator(orbit: Orbit(matching: preset.plate),
                              framing: Framing(center: P2(0, 0), unitsPerPixel: 1))
    if let bounds = base.looking(camera(0)).quickBounds() {
        navigator.framing = .fitting(bounds, in: viewport)
    }

    let warmup = try clock.measure {
        _ = try renderer.renderDepth(base.looking(camera(0)), frame: frame)
        try renderer.renderPreview(base, navigator: navigator, viewport: viewport,
                                   options: PreviewOptions(mode: mode), into: target)
    }

    func measure(_ body: (Int) throws -> Void) rethrows -> [Double] {
        var times: [Double] = []
        times.reserveCapacity(frames)
        for i in 1...frames {
            let d = try clock.measure { try body(i) }
            times.append(Double(d.components.attoseconds) / 1e18
                         + Double(d.components.seconds))
        }
        times.sort()
        return times
    }

    let depthTimes = try measure { i in
        _ = try renderer.renderDepth(base.looking(camera(i)), frame: frame)
    }
    var times = try measure { i in
        navigator.orbit = Orbit(matching: preset.plate)
        navigator.orbit.azimuth = Angle(degrees: preset.plate.zAngle + Double(i) * 0.25)
        try renderer.renderPreview(base, navigator: navigator, viewport: viewport,
                                   options: PreviewOptions(mode: mode), into: target)
    }
    times.sort()
    let median = times[times.count / 2]
    let p95 = times[min(times.count - 1, Int(Double(times.count) * 0.95))]

    func ms(_ t: Double) -> String { String(format: "%.2f ms", t * 1000) }
    let depthMedian = depthTimes[depthTimes.count / 2]
    print("\(bundle.url.lastPathComponent)  [\(preset.name), \(modeName)]  "
          + "\(frame.rows)x\(frame.cols)")
    print("  first frame  \(warmup)  (builds the resources)")
    print("  depth+read   median \(ms(depthMedian))   (what one bake tile costs)")
    print("  full preview median \(ms(median))  p95 \(ms(p95))  "
          + "min \(ms(times[0]))  max \(ms(times[times.count - 1]))")
    print("  that is      \(String(format: "%.0f", 1 / median)) fps at the median"
          + (renderer.hasLinearReadback ? "" : "   (readback is NOT linear here)"))
}

func preview(_ args: Args) throws {
    let (bundle, preset, base) = try loadScene(args)
    let output = URL(fileURLWithPath: try args.string("output"))
    let viewport = Viewport(width: try args.int("width", 1600),
                            height: try args.int("height", 1000))

    let mode: PreviewMode
    switch args.flags["mode"] ?? "plate" {
    case "plate": mode = .plate
    case "shaded": mode = .shaded(Lighting())
    case "depth": mode = .depth
    case let other: throw CLIError("unknown --mode '\(other)'; try plate, shaded or depth")
    }

    var navigator = Navigator(orbit: Orbit(matching: preset.plate),
                              framing: Framing(center: P2(0, 0), unitsPerPixel: 1))
    var scene = base
    scene.camera = navigator.camera
    guard let bounds = scene.quickBounds() else { throw CLIError("the scene is empty") }
    navigator.framing = .fitting(bounds, in: viewport)

    if let spec = args.flags["orbit"] {
        let parts = spec.split(separator: ",").compactMap { Double($0) }
        guard parts.count == 2 else { throw CLIError("--orbit wants \"azimuth,elevation\" in degrees") }
        navigator.orbit.azimuth = Angle(degrees: navigator.orbit.azimuth.degrees + parts[0])
        navigator.orbit.elevation = Angle(degrees: navigator.orbit.elevation.degrees + parts[1])
        scene.camera = navigator.camera
        if let b = scene.quickBounds() { navigator.framing = .fitting(b, in: viewport) }
    }
    if let zoom = try args.double("zoom") {
        navigator = navigator.applying(
            .zoom(factor: zoom, at: SIMD2(Double(viewport.width) / 2,
                                          Double(viewport.height) / 2)),
            in: viewport)
    }

    let renderer = try MetalRenderer()
    let target = try renderer.makePreviewTarget(viewport)

    let clock = ContinuousClock()
    let elapsed = try clock.measure {
        try renderer.renderPreview(base, navigator: navigator, viewport: viewport,
                                   options: PreviewOptions(mode: mode), into: target)
    }
    try PNG.write(target, to: output)

    // The view-space rectangle the picture covers, so another renderer can draw
    // the same strokes into the same frame and the two can be compared as
    // images. Without it "the preview looks like the plate" is an impression.
    let f = navigator.framing.frame(viewport)
    let meta = JSONValue.object([
        // In screen order axis0 runs down the image (view y) and axis1 across
        // it (view x), so this is (left, right) and (top, bottom).
        "viewX": .array([.double(f.axis1.lo), .double(f.axis1.hi)]),
        "viewY": .array([.double(f.axis0.lo), .double(f.axis0.hi)]),
        "width": .int(viewport.width), "height": .int(viewport.height),
        "azimuth": .double(navigator.orbit.azimuth.degrees),
        "elevation": .double(navigator.orbit.elevation.degrees),
        "margin": .double(base.margin),
        "plate": preset.plate.json,
    ])
    try (meta.canonical + "\n").write(
        to: output.deletingPathExtension().appendingPathExtension("frame.json"),
        atomically: true, encoding: .utf8)

    print("""
        \(bundle.url.lastPathComponent) -> \(output.lastPathComponent)          [\(preset.name), \(args.flags["mode"] ?? "plate")]
          \(viewport.width)x\(viewport.height)          azimuth \(String(format: "%.1f", navigator.orbit.azimuth.degrees))          elevation \(String(format: "%.1f", navigator.orbit.elevation.degrees))
          took \(elapsed)
          frame \(output.deletingPathExtension().lastPathComponent).frame.json
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
    case "bench": try bench(args)
    case "preview": try preview(args)
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
