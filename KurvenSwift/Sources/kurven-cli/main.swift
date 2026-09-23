import Foundation
import KurvenCore
import KurvenMetal
import KurvenBake
import KurvenService
import KurvenLandscape
import KurvenMath

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
    /// Options that may be given more than once, keeping every value.
    var repeated: [String: [String]] = [:]

    init(_ argv: [String]) {
        var i = 0
        while i < argv.count {
            let a = argv[i]
            if a.hasPrefix("--") {
                let name = String(a.dropFirst(2))
                if i + 1 < argv.count, Args.isValue(argv[i + 1]) {
                    flags[name] = argv[i + 1]
                    repeated[name, default: []].append(argv[i + 1])
                    i += 2
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

    /// Whether a token is a value rather than the next option.
    ///
    /// Anything beginning with `-` is an option, *except* a negative number:
    /// `--margin -0.5` means a margin and `--derived -o out.svg` does not mean
    /// a derived called "-o". Getting this wrong let a switch swallow the flag
    /// after it, which showed up as "missing --output" three arguments later.
    static func isValue(_ token: String) -> Bool {
        if !token.hasPrefix("-") { return true }
        // A negative *interval* too, since `--re -6,8` is how a landscape's
        // window is written and its left end is usually negative. Without this
        // the token falls through to a positional, and `landscape --re -6,8`
        // sampled rgamma over the wrong window while reporting a parse error
        // about the comma in an expression nobody typed.
        return token.split(separator: ",").allSatisfy { Double($0) != nil }
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
       [--silhouette W] [--dump PREFIX] -o out.svg
        Render a bundle's plate to SVG through the depth-tested hidden-line
        pipeline. Defaults to the bundle's first preset and that preset's own
        depth resolution and clip margin -- the settings the published plate
        was made with. --silhouette traces the outline of the drawn region off
        the depth buffer and adds it as a stroke of that width. --dump also
        writes each layer's strokes as
        PREFIX.<layer>.npy plus CSR offsets, which is what
        tests/compare_bake.py reads to check the result against the Python
        plate stroke for stroke.

  surface torus [--lines U,V] [--samples N]
  surface sn [--modulus M] [--grid N]
  surface periodic --expression E --periods P1,P2 [--grid N]
          [--major R] [--minor r]
          [--x-angle DEG] [--z-angle DEG] [--shear S] [--resolution N]
          [--tiles N] [--width W] [--folds W] -o out.svg
        Bake a surface to SVG, hidden lines decided by where on the surface
        each vertex lies rather than by depth. torus draws its parameter
        lines. sn and periodic draw a doubly periodic function's |f| and
        arg f contours on the torus its period rectangle glues into: sn(z, M)
        on [0, 4K) x [0, 2K'), or any expression over [0, P1) x [0, P2).
        The camera is a plate camera: --x-angle tilts, --z-angle turns, --shear
        is the oblique foreshortening the published plates use. The fold
        lines -- outline and inner silhouettes -- are drawn at --folds (twice
        --width by default; 0 leaves them out).

  depth <bundle> [--preset NAME] [--resolution N] -o depth.npy
        Dump the depth buffer as float32 .npy, for comparison against the
        Python Z-buffer. This is what stands in for a GPU frame capture.

  bench <bundle> [--preset NAME] [--resolution N] [--frames N]
        Render the depth pass repeatedly from a moving camera and report the
        frame time. This is the Phase 2 question -- whether navigation can hold
        60 fps -- asked without a window: the preview does exactly this work
        per frame, plus a line pass that costs a fraction of it.

  preview <bundle> [--preset NAME] [--width N] [--height N] [--mode M]
          [--orbit "AZ,EL"] [--zoom F] [--levels N] [--fov DEGREES]
          [--margin M] [--slope K] [--ink-width W] -o out.png
        Render one preview frame offscreen and write it as a PNG. Modes:
        plate (the default), shaded, depth. --orbit turns the preset camera by
        that many degrees before drawing. --levels redraws every *described*
        layer at N evenly spaced levels over its own range, which only a
        bundle exported with --derived can do. --fov switches to a perspective
        camera, which previews but does not bake. --margin and --slope set the
        ink test: the hidden-line margin, and how many pixels' worth of the
        surface's depth change it also allows (0 is the bake's predicate).
        --ink-width is the widest layer's stroke in pixels (default 1.5); the
        others are drawn in proportion to their plate widths. This is how the
        preview is checked against the plate without a window in the way.

  flicker <bundle> [preview's view options] [--frames N] [--step DEGREES]
          [--margin M] [--slope K] [--ink-width W] [--dump DIR] [--max-flip F]
        Turn the preview camera --step degrees a frame (default 0.02) for
        --frames frames (default 24), and report the ink that comes and goes:
        toggled (changed between frames, motion included) and flipped (a
        stroke that stayed put while the depth test changed its mind about
        it). Flicker is a motion artifact; this is it as a number. --max-flip
        fails the command when the flip rate is over F.

  inspect <bundle>
        Print the manifest: domain, caps, occluder, layers, presets, provenance.

  describe [--python PATH] [--repo PATH]
        Ask the Python service what functions it can sample and what options
        each takes. The service is found by walking up from the working
        directory for kurven/serve.py; KURVEN_REPO and KURVEN_PYTHON override.

  catalog
        Print the functions this program can sample and the language they are
        written in. This is the menu the app's function picker is built from,
        and kurven-test holds it to the Python side's.

  landscape [EXPRESSION | --function NAME] [--re LO,HI] [--im LO,HI]
            [--res N] [--cap Z] [--spacing S] [--service] -o out.kurven
        Sample a function over a rectangle and write its bundle -- natively,
        with no Python in the loop, or by the service with --service, which
        is the comparison path. Every layer is a description, so the result's
        cap, levels and hatch spacing are editable afterwards -- in the app,
        or by `bake --levels`.

  refine [EXPRESSION | --function NAME] [--re LO,HI] [--im LO,HI] [--res N]
         [--tolerance CELLS] [--depth N]
        Benchmark the contour refinement: each contour layer derived from the
        grid, then placed by f, with both timed and both measured against f.

  resample <example> [--set NAME=VALUE ...] [--derived] -o out.kurven
        Ask the service to rebuild one of the published plates. That is what
        Python is still for: the four plates are Python programs, and a
        frozen bundle of one cannot change its own domain or resolution.

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
        margin: try args.double("margin") ?? preset.margin,
        silhouette: try args.double("silhouette"))

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
        // The silhouette, when asked for, is one more stroke layer than the
        // scene has ink layers -- it is traced from the depth buffer, not
        // carried by the bundle.
        let spec = index < scene.layers.count ? scene.layers[index].spec : nil
        print("    \(pad(spec?.name ?? "silhouette", 14)) "
              + "\(pad(String(entry.paths.count), 7)) paths"
              + "  lw \(entry.style.width)"
              + (spec.map { $0.clipped ? "" : "  unclipped" } ?? "  from the depth buffer"))
    }
}

func surfaceCommand(_ args: Args) throws {
    let shape = args.positional.dropFirst().first ?? "torus"
    let R = try args.double("major") ?? 2
    let r = try args.double("minor") ?? 1
    let width = try args.double("width") ?? 0.3
    let surface: ParametricSurface
    var layers: [Layer]
    switch shape {
    case "torus":
        let samples = try args.int("samples", 1024)
        surface = ParametricSurface.torus(major: R, minor: r,
                                          samples: (samples, max(samples / 2, 8)))
        let counts = (args.flags["lines"] ?? "36,18").split(separator: ",").compactMap { Int($0) }
        guard counts.count == 2 else { throw CLIError("--lines wants U,V, as in 36,18") }
        let ink = surface.parameterLines(counts: (counts[0], counts[1]), resolution: 4 * samples)
        layers = [Layer(spec: LayerSpec(name: "lines", role: .scaffold,
                                        source: .parameterLines(u: counts[0], v: counts[1]),
                                        width: width, heightPolicy: .surface),
                        paths: ink)]
    case "sn", "periodic":
        let plate: PeriodicTorus.Plate
        if shape == "sn" {
            plate = try PeriodicTorus.jacobiSN(modulus: try args.double("modulus") ?? 0.64,
                                               major: R, minor: r,
                                               resolution: try args.int("grid", 600))
        } else {
            let periods = (args.flags["periods"] ?? "").split(separator: ",").compactMap { Double($0) }
            guard periods.count == 2 else {
                throw CLIError("--periods wants the real and imaginary periods, as in 6.4,4.1")
            }
            plate = try PeriodicTorus.plate(try args.string("expression"),
                                            periods: (periods[0], periods[1]),
                                            major: R, minor: r,
                                            resolution: try args.int("grid", 600))
        }
        surface = plate.surface
        // Their own folds come with them; --folds below restyles or drops them.
        layers = plate.layers.filter { if case .foldLines = $0.spec.source { false } else { true } }
    default:
        throw CLIError("unknown surface '\(shape)'; the catalog has: torus, sn, periodic")
    }
    let camera = Camera.plate(PlateProjection(
        shear: try args.double("shear") ?? 0,
        xAngle: try args.double("x-angle") ?? -55,
        zAngle: try args.double("z-angle") ?? 30,
        flipX: false, yScale: nil))
    // The outline and inner silhouettes, derived for the camera; --folds 0
    // leaves them out.
    let foldWidth = try args.double("folds") ?? 2 * width
    let folds = Layer(spec: LayerSpec(name: "folds", role: .outline, source: .foldLines,
                                      width: foldWidth, heightPolicy: .surface),
                      paths: .empty)
    if foldWidth > 0 { layers.append(folds) }
    let scene = Scene(surface: surface, layers: layers, camera: camera, margin: 0)
    let output = URL(fileURLWithPath: try args.string("output"))

    let options = BakeOptions(resolution: try args.int("resolution", 3000),
                              tiles: args.flags["tiles"].flatMap(Int.init))

    let renderer = try MetalRenderer()
    let clock = ContinuousClock()
    var result: Bake!
    let elapsed = try clock.measure { result = try renderer.bake(scene, options: options) }
    try SVG.render(result.strokes).write(to: output, atomically: true, encoding: .utf8)
    print("""
        \(shape) R=\(fmt(R)) r=\(fmt(r)) -> \(output.lastPathComponent)
          lattice    \(surface.u.samples)x\(surface.v.samples)\
        \(surface.outward == nil ? ", bounds no solid, back faces kept" : ", closed")
          depth      \(result.depth.frame.rows)x\(result.depth.frame.cols) \
        in \(result.tiles * result.tiles) pass\(result.tiles == 1 ? "" : "es")
          strokes    \(result.strokes.pathCount) paths, \
        ink \(String(format: "%.1f", result.strokes.inkLength))
          took       \(elapsed)
        """)
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

    // Dragging a level slider: re-derive the described layers, rebuild the line
    // buffer, redraw. The whole point of keeping the ink's identity separate
    // from the geometry's is that this does not touch the height texture.
    if args.switches.contains("editing") {
        // One slider, one layer: dragging a level control changes one level set
        // and the others are reused, which is what the app does.
        guard let edited = bundle.manifest.layers.enumerated().first(where: {
            KurvenBundle.levels(of: $0.element) != nil
        }) else { throw CLIError("this bundle has no described layers; export with --derived") }
        var current = bundle.layers
        let counts = (0..<frames).map { 4 + ($0 % 40) }
        var editTimes: [Double] = []
        for n in counts {
            let d = try clock.measure {
                current[edited.offset] = relaid(bundle, edited.element, levelCount: n)
                let scene = base.drawing(current)
                try renderer.renderPreview(scene, navigator: navigator,
                                           viewport: viewport,
                                           options: PreviewOptions(mode: mode),
                                           into: target)
            }
            editTimes.append(Double(d.components.attoseconds) / 1e18
                             + Double(d.components.seconds))
        }
        editTimes.sort()
        let m = editTimes[editTimes.count / 2]
        print("\(bundle.url.lastPathComponent)  editing '\(edited.element.name)', "
              + "\(viewport.width)x\(viewport.height)")
        print("  re-derive + redraw  median \(String(format: "%.2f ms", m * 1000))  "
              + "p95 \(String(format: "%.2f ms", editTimes[min(editTimes.count - 1, Int(Double(editTimes.count) * 0.95))] * 1000))")
        print("  that is             \(String(format: "%.0f", 1 / m)) fps while dragging a level slider")
        return
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

/// Every layer, with the described ones re-derived at `count` evenly spaced
/// levels over their own range. A dumped layer has no question behind it and
/// comes back unchanged, which is the difference `--derived` makes, in one
/// function.
func relaid(_ bundle: KurvenBundle, levelCount count: Int) -> [Layer] {
    bundle.manifest.layers.map { relaid(bundle, $0, levelCount: count) }
}

/// One layer, re-derived at `count` evenly spaced levels over its own range.
func relaid(_ bundle: KurvenBundle, _ spec: LayerSpec, levelCount count: Int) -> Layer {
    guard count >= 1, let levels = KurvenBundle.levels(of: spec),
          levels.count >= 2, let lo = levels.min(), let hi = levels.max() else {
        return (try? bundle.layer(spec.name)) ?? Layer(spec: spec, paths: .empty)
    }
    var redone: [Double] = []
    if count == 1 {
        redone = [(lo + hi) / 2]
    } else {
        for i in 0..<count { redone.append(lo + (hi - lo) * Double(i) / Double(count - 1)) }
    }
    return bundle.layer(spec, levels: redone)
}

/// The scene and camera a preview frame is drawn from: the preset's own view,
/// turned by `--orbit`, put in perspective by `--fov`, zoomed by `--zoom`, and
/// framed to fit the viewport. Shared by `preview` and `flicker`, so the two
/// look at the same picture from the same arguments.
func previewView(_ args: Args, viewport: Viewport) throws
    -> (bundle: KurvenBundle, preset: CameraPreset, scene: Scene, navigator: Navigator)
{
    let (bundle, preset, loaded) = try loadScene(args)
    var base = loaded
    if let n = args.flags["levels"].flatMap({ Int($0) }) {
        base = base.drawing(relaid(bundle, levelCount: n))
        if !bundle.manifest.layers.contains(where: { KurvenBundle.levels(of: $0) != nil }) {
            FileHandle.standardError.write(Data(
                ("note: no described layers here, so --levels changes nothing; "
                 + "export the bundle with --derived\n").utf8))
        }
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
    if let fov = try args.double("fov") {
        if let bounds = scene.quickBounds() {
            navigator = navigator.applying(
                .project(fieldOfView: Angle(degrees: fov), bounds), in: viewport)
            scene.camera = navigator.camera
        }
    }
    if let zoom = try args.double("zoom") {
        navigator = navigator.applying(
            .zoom(factor: zoom, at: SIMD2(Double(viewport.width) / 2,
                                          Double(viewport.height) / 2)),
            in: viewport)
    }
    // The inspector's Margin slider, from the command line.
    if let margin = try args.double("margin") { base.margin = margin }
    return (bundle, preset, base, navigator)
}

func previewMode(_ args: Args) throws -> PreviewMode {
    switch args.flags["mode"] ?? "plate" {
    case "plate": .plate
    case "shaded": .shaded(Lighting())
    case "depth": .depth
    case let other: throw CLIError("unknown --mode '\(other)'; try plate, shaded or depth")
    }
}

/// `--slope K`: the ink test's slope allowance, so a run can be compared with
/// the bake's predicate (`--slope 0`) at the same camera. `--ink-width W`: the
/// widest stroke in pixels, which is what a Retina window doubles.
func previewOptions(_ args: Args, mode: PreviewMode) throws -> PreviewOptions {
    var options = PreviewOptions(mode: mode)
    if let k = try args.double("slope") { options.slopeScale = Float(k) }
    if let w = try args.double("ink-width") { options.inkWidth = Float(w) }
    return options
}

func preview(_ args: Args) throws {
    let output = URL(fileURLWithPath: try args.string("output"))
    let viewport = Viewport(width: try args.int("width", 1600),
                            height: try args.int("height", 1000))
    let mode = try previewMode(args)
    let (bundle, preset, base, navigator) = try previewView(args, viewport: viewport)

    let renderer = try MetalRenderer()
    let target = try renderer.makePreviewTarget(viewport)

    let clock = ContinuousClock()
    let elapsed = try clock.measure {
        try renderer.renderPreview(base, navigator: navigator, viewport: viewport,
                                   options: try previewOptions(args, mode: mode),
                                   into: target)
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

/// Ink that comes and goes while the camera turns, as a number.
///
/// Flicker is a motion artifact, so no single frame shows it. This renders a
/// short orbit -- the camera turning `--step` degrees a frame about its target,
/// the framing held still -- and draws every frame twice: once as the preview
/// does, and once with the depth test switched off. A pixel that has a stroke
/// in both frames of a pair *without* the test, but ink in only one of them
/// *with* it, is a pixel where nothing moved and the visibility test changed
/// its mind. That is the flicker, and it is counted apart from the motion:
///
/// - *toggled*: ink in one frame and not the next. Honest motion contributes,
///   so this only compares runs at the same step.
/// - *flipped*: the test changing its mind under a stroke that stayed put.
///   Some of it is honest too -- a stroke sliding behind a ridge flips at the
///   ridge -- but that is a pixel or so per stroke, not a dash pattern.
///
/// Both are fractions of the mean ink per frame, so a picture with more ink is
/// not counted as flickering more for it.
func flicker(_ args: Args) throws {
    let viewport = Viewport(width: try args.int("width", 1200),
                            height: try args.int("height", 800))
    let (bundle, preset, base, start) = try previewView(args, viewport: viewport)
    let frames = max(try args.int("frames", 24), 2)
    let step = try args.double("step") ?? 0.02
    let dump = args.flags["dump"].map { URL(fileURLWithPath: $0) }
    if let dump {
        try FileManager.default.createDirectory(at: dump, withIntermediateDirectories: true)
    }
    var unclipped = base
    unclipped.margin = .infinity
    let options = try previewOptions(args, mode: .plate)

    let renderer = try MetalRenderer()
    let target = try renderer.makePreviewTarget(viewport)
    let pixels = viewport.width * viewport.height
    // Ink is any visible mark, the test `compare_preview.py` applies to both
    // of its pictures: a stroke narrower than a pixel is drawn lighter, not
    // narrower, so mid-grey would miss the plate's thinnest layers.
    func ink(_ scene: Scene, _ navigator: Navigator) throws -> [Bool] {
        try renderer.renderPreview(scene, navigator: navigator, viewport: viewport,
                                   options: options, into: target)
        let bgra = try PNG.bgra(target)
        return (0..<pixels).map { k in
            let b = Int(bgra[4 * k]), g = Int(bgra[4 * k + 1]), r = Int(bgra[4 * k + 2])
            return 299 * r + 587 * g + 114 * b < 224 * 1000
        }
    }

    var kept: [Bool] = [], everything: [Bool] = []
    var inkTotal = 0, toggled = 0, flipped = 0
    for i in 0..<frames {
        var navigator = start
        navigator.orbit.azimuth = Angle(degrees: start.orbit.azimuth.degrees
                                        + Double(i) * step)
        let k = try ink(base, navigator)
        if let dump {
            try PNG.write(target, to: dump.appendingPathComponent(
                String(format: "frame%03d.png", i)))
        }
        let e = try ink(unclipped, navigator)
        if i > 0 {
            for p in 0..<pixels where k[p] != kept[p] {
                toggled += 1
                if e[p] && everything[p] { flipped += 1 }
            }
        }
        inkTotal += k.reduce(0) { $0 + ($1 ? 1 : 0) }
        kept = k; everything = e
    }
    let perFrame = max(Double(inkTotal) / Double(frames), 1)
    let toggledRate = Double(toggled) / Double(frames - 1) / perFrame
    let flippedRate = Double(flipped) / Double(frames - 1) / perFrame

    func percent(_ x: Double) -> String { String(format: "%.2f%%", x * 100) }
    let from = start.orbit.azimuth.degrees
    print("""
        \(bundle.url.lastPathComponent)  [\(preset.name)]  \(frames) frames \
        \(step)° apart, margin \(base.margin), slope \(options.slopeScale)
          \(viewport.width)x\(viewport.height)  azimuth \(String(format: "%.2f", from)) \
        to \(String(format: "%.2f", from + Double(frames - 1) * step))  \
        elevation \(String(format: "%.1f", start.orbit.elevation.degrees))
          ink      \(Int(perFrame.rounded())) px a frame
          toggled  \(percent(toggledRate)) of it a frame   (motion included)
          flipped  \(percent(flippedRate)) of it a frame   (the stroke stayed; the test changed its mind)
        """)
    if let limit = try args.double("max-flip"), flippedRate > limit {
        throw CLIError("flipped \(percent(flippedRate)) a frame, over the \(percent(limit)) allowed")
    }
}

func service(_ args: Args) throws -> Service {
    let command: Service.Command
    if let python = args.flags["python"] {
        command = Service.Command(
            executable: URL(fileURLWithPath: python),
            arguments: ["-m", "kurven.serve"],
            directory: args.flags["repo"].map { URL(fileURLWithPath: $0) })
    } else {
        let near = args.flags["repo"].map { URL(fileURLWithPath: $0) }
            ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        guard let found = Service.Command.autodetect(near: near) else {
            throw CLIError("""
                could not find kurven/serve.py above \(near.path).                 Pass --repo, or set KURVEN_REPO.
                """)
        }
        command = found
    }
    return try Service(command: command)
}

func describe(_ args: Args) throws {
    let service = try service(args)
    defer { service.stop() }
    print("service: \(service.command.display)")
    let description = try blocking { try await service.describe() }
    print("  protocol \(description.protocolVersion)")
    for example in description.examples {
        guard example.available else {
            print("  \(example.name): unavailable — \(example.reason ?? "?")")
            continue
        }
        print("  \(example.name)")
        for a in example.arguments {
            let value = a.defaultText.map { " = \($0)" } ?? ""
            print("    \(pad(a.name, 16)) \(pad(a.kind.rawValue, 7))\(value)")
        }
    }
}

func resample(_ args: Args) throws {
    guard let example = args.positional.dropFirst().first else {
        throw CLIError("which example? try 'describe'")
    }
    let output = URL(fileURLWithPath: try args.string("output"))
    let settings: [String: String] = try (args.repeated["set"] ?? [])
        .reduce(into: [:]) { out, setting in
            guard let equals = setting.firstIndex(of: "=") else {
                throw CLIError("--set wants NAME=VALUE, got '\(setting)'")
            }
            out[String(setting[setting.startIndex..<equals])] =
                String(setting[setting.index(after: equals)...])
        }

    let service = try service(args)
    defer { service.stop() }
    let clock = ContinuousClock()
    var result: ExportResult!
    let elapsed = try clock.measure {
        result = try blocking {
            try await service.export(example: example, to: output,
                                     arguments: settings,
                                     derived: args.switches.contains("derived"))
        }
    }
    print("""
        \(example) -> \(result.url.lastPathComponent)          (\(String(format: "%.1f", Double(result.bytes) / 1e6)) MB in \(elapsed))
          \(result.manifest.layers.count) layers,         \(result.manifest.occluder.tiles.count) tile(s),         caps \(result.manifest.caps)
        """)
    for spec in result.manifest.layers {
        let what = spec.files == nil ? "derived" : "dumped "
        print("    \(pad(spec.name, 14)) \(what)  lw \(spec.width)")
    }
}

func catalogCommand(_ args: Args) throws {
    let catalog = Catalog.native
    print("  \(catalog.presets.count) presets, default resolution \(catalog.defaultResolution)")
    for preset in catalog.presets {
        let d = preset.domain
        print("""
              \(pad(preset.name, 10)) \(pad(preset.expression, 22)) \
            re [\(fmt(d.real.lo)), \(fmt(d.real.hi))]  \
            im [\(fmt(d.imag.lo)), \(fmt(d.imag.hi))]  \
            cap \(preset.cap.map(fmt) ?? "none")
            """)
    }
    print("  language: \(catalog.functions.count) functions, "
          + "constants \(catalog.constants.joined(separator: ", "))")
    print("    " + catalog.functions.map(\.name).joined(separator: " "))
}

/// `LO,HI` as an interval; the one place a pair of numbers is spelled on the
/// command line, so a landscape's window reads the way it is written down.
func interval(_ text: String, _ what: String) throws -> Interval {
    let parts = text.split(separator: ",").map { Double($0.trimmingCharacters(in: .whitespaces)) }
    guard parts.count == 2, let lo = parts[0], let hi = parts[1], lo < hi else {
        throw CLIError("--\(what) wants LO,HI with LO < HI, got '\(text)'")
    }
    return Interval(lo: lo, hi: hi)
}

func landscapeCommand(_ args: Args) throws {
    let output = URL(fileURLWithPath: try args.string("output"))
    let catalog = Catalog.native

    // The preset is the starting point and every flag is an override of it, so
    // `landscape --function gamma --res 1200` means what it looks like.
    let name = args.flags["function"] ?? (args.positional.count > 1 ? "" : "rgamma")
    var request: LandscapeRequest
    if let preset = catalog.preset(name) {
        request = LandscapeRequest(preset: preset, resolution: catalog.defaultResolution)
    } else if let expression = args.positional.dropFirst().first {
        request = LandscapeRequest(expression: expression,
                                   domain: Domain(real: Interval(lo: -4, hi: 4),
                                                  imag: Interval(lo: -2.5, hi: 2.5)),
                                   resolution: catalog.defaultResolution)
    } else {
        throw CLIError("no function \(name.isEmpty ? "" : "'\(name)' ")given; "
                       + "try 'catalog', or pass an expression")
    }
    if let re = args.flags["re"] { request.domain.real = try interval(re, "re") }
    if let im = args.flags["im"] { request.domain.imag = try interval(im, "im") }
    if let res = args.flags["res"], let n = Int(res) { request.resolution = n }
    if let cap = args.flags["cap"], let z = Double(cap) { request.caps = .uniform(z) }
    if args.switches.contains("no-cap") { request.caps = Caps.none }
    if let s = args.flags["spacing"], let v = Double(s) { request.spacing = v }

    let clock = ContinuousClock()
    var bundle: KurvenBundle!
    let sampled = try clock.measure {
        bundle = try args.switches.contains("service")
            ? sampledByService(request, to: output, args)
            : NativeLandscape.build(request)
    }
    if !args.switches.contains("service") { try bundle.write(to: output) }
    let m = bundle.manifest
    print("""
        \(m.provenance.function) -> \(output.lastPathComponent) \
        (\(args.switches.contains("service") ? "by the service" : "natively") in \(sampled))
          \(m.height.shape.nx)x\(m.height.shape.ny) samples, caps \(m.caps)
        """)
    for spec in m.layers {
        print("    \(pad(spec.name, 14)) \(sourceName(spec.source))")
    }
}

/// `refine`: the contour refinement, benchmarked. Every contour layer of the
/// landscape is derived from the grid and then refined against f, and both
/// are timed and measured against f -- the residual over the gradient, in
/// cells, at every vertex and at every chord's midpoint.
func refineCommand(_ args: Args) throws {
    let catalog = Catalog.native
    let name = args.flags["function"] ?? (args.positional.count > 1 ? "" : "gamma")
    var request: LandscapeRequest
    if let preset = catalog.preset(name) {
        request = LandscapeRequest(preset: preset, resolution: catalog.defaultResolution)
    } else if let expression = args.positional.dropFirst().first {
        request = LandscapeRequest(expression: expression,
                                   domain: Domain(real: Interval(lo: -4, hi: 4),
                                                  imag: Interval(lo: -2.5, hi: 2.5)),
                                   resolution: catalog.defaultResolution)
    } else {
        throw CLIError("no function given; try 'catalog', or pass an expression")
    }
    if let re = args.flags["re"] { request.domain.real = try interval(re, "re") }
    if let im = args.flags["im"] { request.domain.imag = try interval(im, "im") }
    if let res = args.flags["res"], let n = Int(res) { request.resolution = n }
    if let cap = args.flags["cap"], let z = Double(cap) { request.caps = .uniform(z) }
    let tolerance = args.flags["tolerance"].flatMap(Double.init) ?? 0.02
    let depth = args.flags["depth"].flatMap(Int.init) ?? 5

    let (bundle, reports) = try NativeLandscape.benchmarkRefinement(
        request, tolerance: tolerance, maxDepth: depth)
    let shape = bundle.manifest.height.shape
    print("\(bundle.manifest.provenance.function)  \(shape.nx)x\(shape.ny) samples, "
          + "tolerance \(fmt(tolerance)) cells, depth \(depth)")
    print("  errors are distances from f's level set, in cells; "
          + "vertex = at the vertices, chord = at the midpoints between them")
    print("  " + pad("layer", 11) + pad("levels", 7) + pad("vertices", 17)
          + pad("time ms", 16) + pad("vertex mean/p95/max", 30) + "chord mean/p95/max")
    func three(_ t: (mean: Double, p95: Double, max: Double)) -> String {
        String(format: "%.4f/%.4f/%.3f", t.mean, t.p95, t.max)
    }
    for r in reports {
        print("  " + pad(r.layer, 11) + pad(String(r.levels), 7)
              + pad("\(r.gridVertices) -> \(r.refinedVertices)"
                    + (r.wrapVertices > 0 ? " (\(r.wrapVertices) on wraps)" : ""), 17)
              + pad(String(format: "%.1f -> +%.1f", r.gridSeconds * 1e3, r.refineSeconds * 1e3), 16)
              + pad(three(r.gridVertex) + " -> ", 30) + three(r.gridChord))
        print("  " + pad("", 11) + pad("", 7) + pad("", 17) + pad("", 16)
              + pad(three(r.refinedVertex), 30) + three(r.refinedChord))
    }
}

/// The same request answered by the Python service: the comparison path,
/// kept so a native landscape can be checked against the one Python would
/// have written for it.
func sampledByService(_ request: LandscapeRequest, to output: URL, _ args: Args) throws
    -> KurvenBundle {
    let service = try service(args)
    defer { service.stop() }
    let result = try blocking { try await service.landscape(request, to: output) }
    return try KurvenBundle.read(at: result.url)
}

func fmt(_ x: Double) -> String { String(format: "%g", x) }

func sourceName(_ source: LayerSource) -> String {
    switch source {
    case .file: "dumped"
    case .contour(let field, let levels, _, _): "\(levels.count) levels of \(field.rawValue)"
    case .wallHatch(let edges, let spacing, _, _, _, _):
        "wall hatch, \(edges.count) edges every \(fmt(spacing))"
    case .wallOutline(let edges, _, _): "wall outline, \(edges.count) edges"
    case .capHatch(let axis, let spacing, _):
        "cap hatch along \(axis.rawValue) every \(fmt(spacing))"
    case .capOutline: "cap outline"
    case .parameterLines(let u, let v): "\(u) + \(v) parameter lines"
    case .foldLines: "fold lines"
    }
}

/// Run an async call from this synchronous program.
///
/// The CLI is one command and then it exits, so there is nothing for a
/// concurrency runtime to overlap; a semaphore is the honest shape.
func blocking<T: Sendable>(_ body: @escaping @Sendable () async throws -> T) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var outcome: Result<T, Error>!
    Task {
        do { outcome = .success(try await body()) }
        catch { outcome = .failure(error) }
        semaphore.signal()
    }
    semaphore.wait()
    return try outcome.get()
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
    case "surface": try surfaceCommand(args)
    case "bench": try bench(args)
    case "preview": try preview(args)
    case "flicker": try flicker(args)
    case "describe": try describe(args)
    case "catalog": try catalogCommand(args)
    case "landscape": try landscapeCommand(args)
    case "refine": try refineCommand(args)
    case "resample": try resample(args)
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
