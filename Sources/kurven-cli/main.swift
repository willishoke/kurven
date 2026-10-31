import Foundation
import simd
import KurvenCore
import KurvenMetal
import KurvenBake
import KurvenLandscape
import KurvenMath
import KurvenDynamics

/// `kurven-cli` -- the headless half of the frontend.
///
/// The app is not the product of Phase 1; this is. Everything the interactive
/// shell will do to a scene, this does to the same `Scene` value, so "the UI
/// bakes the same picture as the CLI" is true because they are one function
/// called twice, and the picture can be checked before any window exists.
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
    /// `--shear -0.5` means a shear and `--derived -o out.svg` does not mean
    /// a derived called "-o". Getting this wrong let a switch swallow the flag
    /// after it, which showed up as "missing --output" three arguments later.
    static func isValue(_ token: String) -> Bool {
        if !token.hasPrefix("-") { return true }
        // A negative *interval* too, since `--re -6,8` is how a landscape's
        // window is written and its left end is usually negative. Without this
        // the token falls through to a positional, and `landscape --re -6,8`
        // sampled rgamma over the wrong window while reporting a parse error
        // about the comma in an expression nobody typed. And a list of
        // number pairs, `--bands -3.5:2.5,0.5:5.5`, for the same reason.
        return token.split(separator: ",").allSatisfy {
            $0.split(separator: ":", omittingEmptySubsequences: false).allSatisfy { Double($0) != nil }
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
    /// A number that must be positive and finite, as a length is.
    func positive(_ name: String) throws -> Double? {
        guard let d = try double(name) else { return nil }
        guard d > 0, d.isFinite else { throw CLIError("--\(name) wants a positive number, got '\(flags[name]!)'") }
        return d
    }
}

let usage = """
usage: kurven-cli <command> [options]

  bake <bundle> [--preset NAME] [--resolution N] [--tiles N] [--depth-tolerance T]
       [--outline W] [--dump PREFIX] -o out.svg
        Render a bundle's drawing to SVG through the depth-tested hidden-line
        pipeline. Defaults to the bundle's first preset and that preset's own
        depth resolution and depth tolerance -- the settings the published drawing
        was made with. --outline traces the outline of the drawn region off
        the depth buffer and adds it as a stroke of that width. --dump also
        writes each layer's strokes as PREFIX.<layer>.npy plus CSR offsets,
        and the frame as PREFIX.frame.json, for checking a bake stroke for
        stroke outside this program.

  surface torus [--coordinate-curves U,V] [--samples N]
  surface sn [--modulus M] [--grid N]
  surface sn --relief A [--repeat N[,M]] [--turn T] [--floor B] [--soft] [--samples S]
  surface periodic --expression E --periods P1,P2 [--grid N]
  surface forced [--system jerk] [--turns N] [--trajectory-width W]
          [--harmonics M] [--fit T] [--radial I] [--axial J] [--coordinate-curves U,V]
          [--major R] [--minor r] [--linear-curves SLOPE[,TURNS[,COUNT]]]
          [--x-angle DEG] [--z-angle DEG] [--shear S] [--resolution N]
          [--tiles N] [--width W] [--folds W] -o out.svg
          [--mode M] [--width-px N] [--height-px N] -o out.png
        Bake a surface to SVG, hidden lines decided by where on the surface
        each vertex lies rather than by depth. torus draws its coordinate
        curves. sn and periodic draw a doubly periodic function's |f| and
        arg f contours on the torus its period rectangle glues into: sn(z, M)
        on [0, 4K) x [0, 2K'), or any expression over [0, P1) x [0, P2).
        sn --relief A puts the landscape in relief: the torus displaced
        along its normal by A times the truncated |f|, over --repeat
        N periods round the ring and M round the tube (N near M keeps the
        tiling isotropic; a pole's plateau is a round disc only then). --turn
        starts the rectangle T imaginary periods up the plane, turning the
        function round the tube: 0 puts the zeros on the outer equator,
        1/4 the poles on top. --floor B displaces only |f| above B, so each
        pole stands alone on the bare torus; --soft truncates smoothly, as
        c tanh(|f|/c), instead of with a crease. --samples is the grid round the ring in all.
        forced finds a forced system's invariant torus in the spectrum of a
        long run, fits it as a Fourier series in the forcing phase and the
        system's own, and draws --turns turns of the trajectory on it, the
        forcing phase as the angle of revolution: the trajectory is the
        linear curve at the rotation number Ω/ω, and it cuts the torus most
        evenly at the denominators of that number's convergents, which are
        printed. Any surface draws --linear-curves: COUNT straight lines through
        the parameter rectangle at SLOPE turns of v per turn of u (2/5, or a
        decimal), TURNS turns long, or closed when the slope is a fraction.
        The camera is a oblique camera: --x-angle tilts, --z-angle turns, --shear
        is the oblique foreshortening the published plates use. The fold
        curves -- the outline and inner fold curves -- are drawn at --folds (twice
        --width by default; 0 leaves them out). An output ending in .png
        renders one frame of the realtime preview instead of baking.

  surface solid [--kind K] [--size R] [--hole H] [--depth D] [--width W]
          [--x-angle DEG] [--z-angle DEG] [--shear S] [--resolution N] [--color C]
          [--hatching uniform|turing|half-lambert [--angle DEG] [--spacing S] [--step S]
          [--width-steps N] [--seed N] [--scale S] [--hatch-width W]
          [--hatch-color C]] -o out.svg
        Bake a polyhedron to SVG, its edges drawn. K is one of tetrahedron,
        cube, octahedron, dodecahedron, icosahedron -- built from their vertex
        coordinates by a convex hull, set down on a face -- csaszar, the
        seven-vertex torus, or frame, a picture frame (2R across, a hole
        --hole wide, --depth thick). An edge shows where one of its two faces
        faces the eye and nothing stands in front, and the second half is
        decided by occlusion against every piece of the solid, in closed
        form: no pixel, the same drawing at any --resolution, which only the
        depth output sees. --size is the circumradius (2 by
        default). --hatching hatches every face with lines at --angle (45°)
        every --spacing, their width set by a tone: uniform is the whole
        face, turing is Turing's dappled pattern at --scale under --seed,
        half-lambert is the darkness under half-Lambert shading by a fixed
        light. With --width-steps 1 the lines are drawn where the tone is
        positive; with more their width follows the tone in that many steps,
        up to --hatch-width. --step is how often the tone is read along a
        line (the spacing by default).
        --color is the edges' stroke colour and --hatch-color the
        hatching's (any SVG colour; black by default).

  surface-costs [--repeat N] [--only torus|sn|forced]
        Time what each of the window's surface controls costs: the drawing
        rebuilt from the one before it, and the first preview frame after,
        as the window would pay for them -- at full resolution, and at the
        draft grid a drag uses. The numbers SurfaceSection's doc records.

  landscape-costs [--only NAME] [--resolution N]
        Time what each of the window's landscape controls costs, per catalog
        preset, as the window pays for it off the main actor: a resample in
        full and as a drag's draft, a truncation step and the step after it, a hatch
        spacing step, a layer's levels placed on f and as a drag's draft, and turning
        refinement on. The numbers Landscape.swift's doc records.

  depth <bundle> [--preset NAME] [--resolution N] -o depth.npy
        Dump the depth buffer as float32 .npy, for comparison against a
        reference. This is what stands in for a GPU frame capture.

  bench <bundle> [--preset NAME] [--resolution N] [--frames N] [--visibility]
        Render the depth pass repeatedly from a moving camera and report the
        frame time. This is the Phase 2 question -- whether navigation can hold
        60 fps -- asked without a window: the preview does exactly this work
        per frame, plus a line pass that costs a fraction of it.

  preview <bundle> [--preset NAME] [--width N] [--height N] [--mode M]
          [--orbit "AZ,EL"] [--center RE,IM,Z] [--zoom F] [--levels N] [--fov DEGREES]
          [--depth-tolerance T] [--slope K] [--stroke-width W] [--no-refine] [--draft] -o out.png
        Render one preview frame offscreen and write it as a PNG. Modes:
        drawing (the default), shaded, depth. --orbit turns the preset camera by
        that many degrees before drawing; --center puts that world point at
        the middle of the picture, and --zoom then zooms about it. --levels redraws every *described*
        layer at N evenly spaced levels over its own range, which only a
        bundle exported with --derived can do. --fov switches to a perspective
        camera, which previews but does not bake. --depth-tolerance and --slope
        set the visibility test: the hidden-line depth tolerance, and how many
        pixels' worth of the surface's depth change it also allows (0 is the
        bake's predicate).
        --stroke-width is the widest layer's stroke in pixels (default 1.5); the
        others are drawn in proportion to their widths in the bake. A bundle that
        names its function has its contours placed by it, as the window does;
        --no-refine draws the grid's contours instead, and bake, depth and
        flicker take it too. This is how the preview is checked against the
        drawing without a window in the way.

  flicker <bundle> [preview's view options] [--frames N] [--step DEGREES]
          [--depth-tolerance T] [--slope K] [--stroke-width W] [--dump DIR] [--max-flip F]
        Turn the preview camera --step degrees a frame (default 0.02) for
        --frames frames (default 24), and report the curves that comes and goes:
        toggled (changed between frames, motion included) and flipped (a
        stroke that stayed put while the depth test changed its mind about
        it). Flicker is a motion artifact; this is it as a number. --max-flip
        fails the command when the flip rate is over F.

  inspect <bundle>
        Print the manifest: domain, truncation, occluder, layers, presets, provenance.

  catalog
        Print the functions this program can sample and the language they are
        written in. This is the menu the app's function picker is built from.

  landscape [EXPRESSION | --function NAME] [--re LO,HI] [--im LO,HI]
            [--res N] [--truncation H | --bands BELOW:HEIGHT,... --beyond H]
            [--spacing S] -o out.kurven
        Sample a function over a rectangle and write its bundle. Every layer
        is a description, so the result's truncation, levels and hatch spacing are
        editable afterwards -- in the app, or by `bake --levels`.
        --bands truncates at a height per band of Re z, as the gamma drawing
        does: each BELOW:HEIGHT is the height where Re z < BELOW, the first band
        that matches wins, and --beyond is the height past the last. The
        contour levels are chosen up to the highest of them, so set the
        truncation here rather than in the manifest afterwards.

  refine [EXPRESSION | --function NAME] [--re LO,HI] [--im LO,HI] [--res N]
         [--tolerance CELLS] [--depth N]
        Benchmark the contour refinement: each contour layer derived from the
        grid, then placed by f, with both timed and both measured against f.

  contract <dir>
        Decode every .kurven bundle in <dir>, re-encode its manifest, and
        compare canonical JSON against the file. The schema test.

  fixtures <dir>
        Write the golden fixture sets under <dir> (tests/fixtures): contract,
        camera, clip, hatch and the three landscape bundles in expr. The
        frozen inputs among them (the contract bundles' noise layers, the
        camera's points, the clip fixture's buffer and polylines) are read
        from <dir>, not written. scripts/check.sh runs this and fails on any
        difference from the commit.
"""

func loadScene(_ args: Args) throws -> (KurvenBundle, CameraPreset, Scene) {
    guard let path = args.positional.dropFirst().first else {
        throw CLIError("which bundle?")
    }
    var bundle = try KurvenBundle.read(at: URL(fileURLWithPath: path))
    // The contours placed by the function, as the window draws them
    // (`Document.refineContours`), whenever the bundle names a function to
    // place them by. `--no-refine` draws the grid's contours instead, which
    // is the stage the fixtures pin.
    if !args.switches.contains("no-refine"), let refiner = NativeLandscape.refiner(for: bundle) {
        bundle = bundle.refined(by: refiner.refine)
    }
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
        depthTolerance: try args.double("depth-tolerance") ?? preset.depthTolerance,
        outline: try args.double("outline"))

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
            "depthTolerance": .double(options.depthTolerance ?? scene.depthTolerance),
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
        curves \(String(format: "%.1f", result.strokes.totalLength))
          took       \(elapsed)
        """)
    for (index, entry) in result.strokes.layers.enumerated() {
        // The outline, when asked for, is one more stroke layer than the
        // scene has layers -- it is traced from the depth buffer, not
        // carried by the bundle.
        let spec = index < scene.layers.count ? scene.layers[index].spec : nil
        print("    \(pad(spec?.name ?? "outline", 14)) "
              + "\(pad(String(entry.paths.count), 7)) paths"
              + "  lw \(entry.style.width)"
              + (spec.map { $0.clipped ? "" : "  unclipped" } ?? "  from the depth buffer"))
    }
}

func surfaceCommand(_ args: Args) throws {
    let shape = args.positional.dropFirst().first ?? "torus"
    if shape == "solid" { return try solidCommand(args) }
    let R = try args.double("major") ?? 2
    let r = try args.double("minor") ?? 1
    let width = try args.double("width") ?? 0.3
    // `surface sn --relief A`: the sn landscape in relief on its torus, the
    // surface displaced along its normal by A times |f| truncated at the
    // truncation, baked straight to SVG. A proof of concept: no window, no preview.
    if shape == "sn", let amplitude = try args.double("relief") {
        let m = try args.double("modulus") ?? 0.64
        // `--repeat N[,M]`: N periods round the ring and M round the tube,
        // the curves computed once and carried round by the periods.
        let repeats = (args.flags["repeat"] ?? "1").split(separator: ",").compactMap { Int($0) }
        guard (1...2).contains(repeats.count), repeats.allSatisfy({ $0 >= 1 }) else {
            throw CLIError("--repeat wants N or N,M whole periods, as in 8 or 8,2")
        }
        // `--samples`: the grid round the ring in all, 1024 a period at most.
        let grid = try args.int("samples", 8192)
        let clock = ContinuousClock()
        var drawing: PeriodicTorus.Drawing!
        let built = try clock.measure {
            var base = try PeriodicTorus.jacobiSN(modulus: m, major: R, minor: r,
                                                  resolution: try args.int("grid", 600),
                                                  grid: min(grid, 1024), truncated: true,
                                                  turn: try args.double("turn") ?? 0)
                .repeated(real: repeats[0], imag: repeats.count > 1 ? repeats[1] : 1, grid: grid)
            // `--soft`: the truncation as a smooth saturation rather than a crease,
            // c · tanh(|f| / c), an experiment on what the crease costs.
            if args.switches.contains("soft"), let height = base.height {
                let c = base.ceiling
                base = PeriodicTorus.Drawing(surface: base.surface, layers: base.layers,
                                           height: { p in c * tanh(height(p) / c) },
                                           ceiling: c, radii: base.radii)
            }
            // `--floor b`: only |f| above b displaces, so each pole stands
            // alone on the bare torus instead of on a raised ground.
            if let floor = try args.double("floor"), let height = base.height {
                base = PeriodicTorus.Drawing(surface: base.surface, layers: base.layers,
                                           height: { p in max(height(p) - floor, 0) },
                                           ceiling: base.ceiling - floor, radii: base.radii)
            }
            drawing = base.relief(amplitude: amplitude, reach: min(r, R - r))
        }
        var projection = SurfaceRequest.projection
        projection.shear = try args.double("shear") ?? projection.shear
        projection.xAngle = try args.double("x-angle") ?? projection.xAngle
        projection.zAngle = try args.double("z-angle") ?? projection.zAngle
        let scene = Scene(surface: drawing.surface, layers: drawing.layers,
                          camera: Camera.oblique(projection), depthTolerance: 0)
        let output = URL(fileURLWithPath: try args.string("output"))
        let renderer = try MetalRenderer()
        var result: Bake!
        let baked = try clock.measure {
            result = try renderer.bake(scene, options: BakeOptions(resolution: try args.int("resolution", 3000),
                                                                   tiles: args.flags["tiles"].flatMap(Int.init)))
        }
        try SVG.render(result.strokes).write(to: output, atomically: true, encoding: .utf8)
        print("""
            sn in relief, amplitude \(fmt(amplitude)) on R=\(fmt(R)) r=\(fmt(r)) -> \(output.lastPathComponent)
              reach      \(fmt(min(r, R - r))); displacement up to \(fmt(amplitude * drawing.ceiling))\
            \(drawing.surface.outward == nil ? ", NOT embedded: back faces kept" : ", embedded")
              strokes    \(result.strokes.pathCount) paths, curves \(String(format: "%.1f", result.strokes.totalLength))
              took       \(built) to build, \(baked) to bake
            """)
        return
    }
    func counts(_ text: String) throws -> SIMD2<Int> {
        let n = text.split(separator: ",").compactMap { Int($0) }
        guard n.count == 2 else { throw CLIError("--lines wants U,V, as in 36,18") }
        return SIMD2(n[0], n[1])
    }
    // The flags, parsed into the request the window keeps; the drawing is
    // built from it by the same function either way.
    var request: SurfaceRequest
    switch shape {
    case "torus":
        request = SurfaceRequest(.torus(major: R, minor: r),
                                 lines: try counts(args.flags["coordinate-curves"] ?? "36,18"))
    case "sn":
        request = SurfaceRequest(.sn(modulus: try args.double("modulus") ?? 0.64, major: R, minor: r,
                                     grid: try args.int("grid", 600)))
    case "periodic":
        let periods = (args.flags["periods"] ?? "").split(separator: ",").compactMap { Double($0) }
        guard periods.count == 2 else {
            throw CLIError("--periods wants the real and imaginary periods, as in 6.4,4.1")
        }
        request = SurfaceRequest(.periodic(expression: try args.string("expression"),
                                           periods: SIMD2(periods[0], periods[1]),
                                           major: R, minor: r, grid: try args.int("grid", 600)))
    case "forced":
        // A forced system's invariant torus, found in the spectrum of one
        // long run and fitted; drawn by revolution, with the trajectory on it.
        request = SurfaceRequest(.forced(system: args.flags["system"] ?? "jerk",
                                         fit: try args.double("fit") ?? 8000,
                                         harmonics: try args.int("harmonics", 24),
                                         radial: try args.int("radial", 0),
                                         axial: try args.int("axial", 1),
                                         radius: try args.double("radius") ?? 2.5,
                                         turns: try args.int("turns", 344)),
                                 lines: try args.flags["coordinate-curves"].map(counts))
    default:
        throw CLIError("unknown surface '\(shape)'; the catalog has: torus, sn, periodic, forced")
    }
    request.grid = try args.int("samples", 1024)
    // A linear curve: `--linear-curves 2/5`, `--linear-curves 0.618,24` for that many turns,
    // `--linear-curves 1/3,3,4` for four strands of it.
    if let text = args.flags["linear-curves"] {
        let parts = text.split(separator: ",").map(String.init)
        func slope(_ s: String) -> Double? {
            let f = s.split(separator: "/").compactMap { Double($0) }
            return f.count == 2 && f[1] != 0 ? f[0] / f[1] : (f.count == 1 ? f[0] : nil)
        }
        guard (1...3).contains(parts.count), let slope = slope(parts[0]),
              let turns = parts.count > 1 ? Int(parts[1]) : 24,
              let count = parts.count > 2 ? Int(parts[2]) : 1, turns >= 1, count >= 1 else {
            throw CLIError("--linear curve wants SLOPE[,TURNS[,COUNT]], as in 2/5 or 0.618,24,3")
        }
        request.linearCurves = SurfaceRequest.LinearCurves(slope: slope, turns: turns, count: count)
    }
    // The outline and inner fold curves, derived for the camera; --folds 0
    // leaves them out.
    request.style = SurfaceRequest.Style(lines: width, trajectory: try args.double("trajectory-width") ?? 0.08,
                                         folds: try args.double("folds") ?? 2 * width,
                                         linearCurves: width)
    let clock = ContinuousClock()
    var drawing: SurfaceDrawing!
    let built = try clock.measure { drawing = try SurfaceDrawing.build(request) }
    let surface = drawing.surface
    if let torus = drawing.torus {
        let rho = torus.rotationNumber
        let convergents = ContinuedFraction.convergents(of: rho, maxDenominator: 2000)
            .dropFirst().map { "\($0.p)/\($0.q)" }.joined(separator: ", ")
        print(String(format: """
            \(torus.system.name): forced at ω = %.6f, own frequency Ω = %.10f  (built in %@)
              fit       %d harmonics, misses its trajectory by %.2e (%.1e of its size)
              winds     at Ω/ω = %.9f = %@
                        nearly %@: drawn for those turns, it cuts the torus evenly
              drawn     by revolution, %@
            """, torus.forcing, torus.internalFrequency, "\(built)", torus.fit.harmonics,
            torus.residual, torus.residual / torus.extent, rho,
            ContinuedFraction.describe(ContinuedFraction.quotients(of: rho, count: 8), ellipsis: true),
            convergents,
            drawing.embedded ? "embedded: back faces are hidden"
                           : "NOT embedded -- a slice crosses itself; back faces are kept"))
    }
    var projection = SurfaceRequest.projection
    projection.shear = try args.double("shear") ?? projection.shear
    projection.xAngle = try args.double("x-angle") ?? projection.xAngle
    projection.zAngle = try args.double("z-angle") ?? projection.zAngle
    let camera = Camera.oblique(projection)
    let scene = drawing.scene(camera: camera)
    let output = URL(fileURLWithPath: try args.string("output"))

    if args.switches.contains("preview") || output.pathExtension.lowercased() == "png" {
        // The realtime preview, one frame, offscreen: what the app would show.
        let viewport = Viewport(width: try args.int("width-px", 1600),
                                height: try args.int("height-px", 1200))
        let orbit = Orbit(matching: projection)
        guard let bounds = scene.quickBounds() else { throw CLIError("the surface is empty") }
        let navigator = Navigator(orbit: orbit, framing: .fitting(bounds, in: viewport))
        let renderer = try MetalRenderer()
        let target = try renderer.makePreviewTarget(viewport)
        let elapsed = try clock.measure {
            try renderer.renderPreview(scene, navigator: navigator, viewport: viewport,
                                       options: try previewOptions(args, mode: try previewMode(args)),
                                       into: target)
        }
        try PNG.write(target, to: output)
        print("\(shape) -> \(output.lastPathComponent)  preview \(viewport.width)x\(viewport.height)"
              + "  took \(elapsed)")
        return
    }
    let options = BakeOptions(resolution: try args.int("resolution", 3000),
                              tiles: args.flags["tiles"].flatMap(Int.init))

    let renderer = try MetalRenderer()
    var result: Bake!
    let elapsed = try clock.measure { result = try renderer.bake(scene, options: options) }
    try SVG.render(result.strokes).write(to: output, atomically: true, encoding: .utf8)
    print("""
        \(shape)\(shape == "forced" ? "" : " R=\(fmt(R)) r=\(fmt(r))") -> \(output.lastPathComponent)
          grid    \(surface.u.samples)x\(surface.v.samples)\
        \(surface.outward == nil ? ", bounds no solid, back faces kept" : ", closed")
          depth      \(result.depth.frame.rows)x\(result.depth.frame.cols) \
        in \(result.tiles * result.tiles) pass\(result.tiles == 1 ? "" : "es")
          strokes    \(result.strokes.pathCount) paths, \
        curves \(String(format: "%.1f", result.strokes.totalLength))
          took       \(elapsed)
        """)
}

/// `surface solid`: a polyhedron, baked straight to SVG. No bundle: a
/// solid is a table and a rule, and the rule is exact.
func solidCommand(_ args: Args) throws {
    let kind = args.flags["kind"] ?? "dodecahedron"
    let size = try args.positive("size") ?? 2
    let width = try args.double("width") ?? 0.6
    let solid: Solid
    if kind == "csaszar" {
        // Császár's own coordinates stand on the table already; scaled so
        // its base is about the size of the others.
        let c = Solid.csaszar
        let r = c.vertices.map { simd_length(SIMD2($0.x, $0.y)) }.max() ?? 1
        solid = c.transformed(rotation: matrix_identity_double3x3 * (size / r))
    } else if kind == "frame" {
        let hole = try args.positive("hole") ?? size
        guard hole < 2 * size else {
            throw CLIError("--hole wants less than the frame's width, 2 × --size = \(2 * size), got \(hole)")
        }
        solid = Solid.frame(outer: 2 * size, inner: hole, depth: try args.positive("depth") ?? size / 2)
    } else if let platonic = Solid.Platonic(rawValue: kind) {
        solid = Solid.platonic(platonic, circumradius: size).resting(onFace: 0)
    } else {
        throw CLIError("unknown solid '\(kind)'; the table has: "
                       + Solid.Platonic.allCases.map(\.rawValue).joined(separator: ", ")
                       + ", csaszar, frame")
    }
    var projection = SurfaceRequest.projection
    projection.shear = try args.double("shear") ?? projection.shear
    projection.xAngle = try args.double("x-angle") ?? projection.xAngle
    projection.zAngle = try args.double("z-angle") ?? projection.zAngle
    let camera = Camera.oblique(projection)
    var layers = [Layer(spec: LayerSpec(name: "edges", role: .outline, source: .edges,
                                        width: width, heightPolicy: .surface,
                                        color: args.flags["color"] ?? "#000000"),
                        paths: solid.edgePaths)]
    if let kind = args.flags["hatching"] {
        let tone: Tone
        switch kind {
        case "uniform": tone = .uniform
        case "turing":
            let seed = try args.int("seed", 1)
            guard seed >= 0 else { throw CLIError("--seed wants a whole number, 0 or more, got \(seed)") }
            tone = .turing(scale: try args.positive("scale") ?? size / 4, seed: UInt64(seed))
        case "half-lambert": tone = .halfLambert(light: SIMD3(-0.4, -0.6, 0.7))
        case let other: throw CLIError("unknown tone '\(other)'; the tones are uniform, turing, half-lambert")
        }
        let spacing = try args.positive("spacing") ?? size / 25
        let step = try args.positive("step")
        if let step, step < Hatching.finestStep * spacing {
            throw CLIError("--step wants at least a millionth of --spacing, got \(step)")
        }
        let widthSteps = try args.int("width-steps", 1)
        guard widthSteps >= 1 else { throw CLIError("--width-steps wants 1 or more, got \(widthSteps)") }
        let hatching = Hatching(angle: (try args.double("angle") ?? 45) * .pi / 180, spacing: spacing,
                                step: step, tone: tone, widthSteps: widthSteps)
        layers.append(Layer(spec: LayerSpec(name: "hatching", role: .scaffold, source: .hatching(hatching, faces: .solid),
                                            width: try args.double("hatch-width") ?? width / 2,
                                            heightPolicy: .surface,
                                            color: args.flags["hatch-color"] ?? args.flags["color"] ?? "#000000"),
                            paths: solid.hatchPaths(hatching)))
    }
    let scene = Scene(solid: solid, layers: layers, camera: camera)
    let output = URL(fileURLWithPath: try args.string("output"))
    let options = BakeOptions(resolution: try args.int("resolution", 3000),
                              tiles: args.flags["tiles"].flatMap(Int.init))
    let renderer = try MetalRenderer()
    let clock = ContinuousClock()
    var result: Bake!
    let elapsed = try clock.measure { result = try renderer.bake(scene, options: options) }
    try SVG.render(result.strokes).write(to: output, atomically: true, encoding: .utf8)
    let drawn = result.strokes.layers[0].paths
    let corners = solid.vertices.map { camera.view($0).v }
    let whole = (0..<drawn.count).filter { i in
        let p = drawn[path: i]
        return solid.edges.contains { e in
            simd_length(p.first!.v - corners[e.a]) < 1e-9
                && simd_length(p.last!.v - corners[e.b]) < 1e-9
        }
    }.count
    let diagonals = solid.isDiagonal.filter { $0 }.count
    let hatchLength = result.strokes.layers.dropFirst().reduce(0.0) { $0 + $1.paths.totalLength }
    let hatchLines = result.strokes.layers.dropFirst().reduce(0) { $0 + $1.paths.count }
    let widths = result.strokes.layers.count - 1
    print("""
        \(kind) -> \(output.lastPathComponent)
          solid      \(solid.vertices.count) vertices, \(solid.edges.count - diagonals) edges, \
        \(solid.faces.count) faces\(diagonals > 0 ? " in \(solid.pieces.count) convex pieces" : ""), \
        genus \(1 - solid.eulerCharacteristic / 2), \
        \(solid.isConvex ? "convex: the sign test alone decides" : "not convex: edges are tested for occlusion")
          edges      \(drawn.count) spans drawn, \(whole) of them whole edges, \
        length \(String(format: "%.2f", drawn.totalLength))
          hatching   \(layers.count > 1 ? "\(hatchLines) lines at \(widths) width\(widths == 1 ? "" : "s"), length \(String(format: "%.2f", hatchLength))" : "none")
          depth      \(result.depth.frame.rows)x\(result.depth.frame.cols), \
        which visibility does not consult
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

    // A camera that moves: the drawing's own projection, orbited. Every frame
    // shares the scene's `content`, so this measures what navigation costs
    // once the resources are built -- which is the whole reason they are
    // separated from the camera.
    func camera(_ i: Int) -> Camera {
        var projection = preset.projection
        projection.zAngle = preset.projection.zAngle + Double(i) * 0.25
        return .oblique(projection)
    }

    // Two things are worth timing separately: the depth pass alone, which is
    // what a bake pays per tile, and the whole preview, which is what a drag
    // pays per frame. Reporting only the first would flatter the app.
    let modeName = args.flags["mode"] ?? "drawing"
    let mode: PreviewMode = modeName == "shaded" ? .shaded(Lighting())
        : modeName == "depth" ? .depth : .drawing
    let target = try renderer.makePreviewTarget(viewport)
    var navigator = Navigator(orbit: Orbit(matching: preset.projection),
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
    // buffer, redraw. The whole point of keeping the curves' identity separate
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
        navigator.orbit = Orbit(matching: preset.projection)
        navigator.orbit.azimuth = Angle(degrees: preset.projection.zAngle + Double(i) * 0.25)
        try renderer.renderPreview(base, navigator: navigator, viewport: viewport,
                                   options: PreviewOptions(mode: mode), into: target)
    }
    times.sort()
    let median = times[times.count / 2]
    let p95 = times[min(times.count - 1, Int(Double(times.count) * 0.95))]
    // The draft: what a frame costs while the camera is moving.
    let drafts = try measure { i in
        navigator.orbit = Orbit(matching: preset.projection)
        navigator.orbit.azimuth = Angle(degrees: preset.projection.zAngle + Double(i) * 0.25)
        try renderer.renderPreview(base, navigator: navigator, viewport: viewport,
                                   options: PreviewOptions(mode: mode, draft: true), into: target)
    }

    func ms(_ t: Double) -> String { String(format: "%.2f ms", t * 1000) }
    let depthMedian = depthTimes[depthTimes.count / 2]
    print("\(bundle.url.lastPathComponent)  [\(preset.name), \(modeName)]  "
          + "\(frame.rows)x\(frame.cols)")
    print("  first frame  \(warmup)  (builds the resources)")
    print("  depth+read   median \(ms(depthMedian))   (what one bake tile costs)")
    print("  full preview median \(ms(median))  p95 \(ms(p95))  "
          + "min \(ms(times[0]))  max \(ms(times[times.count - 1]))   (the camera at rest)")
    print("  draft frame  median \(ms(drafts[drafts.count / 2]))  "
          + "p95 \(ms(drafts[min(drafts.count - 1, Int(Double(drafts.count) * 0.95))]))"
          + "   (the camera moving)")
    print("  that is      \(String(format: "%.0f", 1 / median)) fps at rest, "
          + "\(String(format: "%.0f", 1 / drafts[drafts.count / 2])) fps moving"
          + (renderer.hasLinearReadback ? "" : "   (readback is NOT linear here)"))

    // --visibility: where a heightfield frame's CPU time goes. Fold curves and
    // wall curves are the camera's, so the preview derives and tests them per
    // camera change; each step is timed on its own, on f (what the preview
    // does) and on the grid without the function (what a draft could do).
    guard args.switches.contains("visibility"), let h = base.heightfield else { return }
    let view = camera(1).view
    let grid = Heightfield(surface: h.surface, occluder: h.occluder, tiles: h.tiles,
                              region: h.region, step: h.step, refine: nil)
    func seconds(_ d: Duration) -> Double {
        Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
    }
    func time(_ label: String, _ body: () throws -> Int) rethrows {
        var ts: [Double] = []
        var count = 0
        for _ in 0..<5 { ts.append(seconds(try clock.measure { count = try body() })) }
        ts.sort()
        print("  \(label.padding(toLength: 44, withPad: " ", startingAt: 0))"
              + "\(ms(ts[ts.count / 2]).padding(toLength: 12, withPad: " ", startingAt: 0))"
              + "\(count) vertices")
    }
    // What the window tests its curves to (`Document.framedScene`).
    let pixel = 0.25 * navigator.framing.unitsPerPixel
    print("visibility, per camera change (median of 5)")
    if h.refine != nil {
        time("folds traced on f") { h.foldCurves(view: view, refined: false).vertices.count }
        time("folds traced and tested on f") {
            h.visibleFolds(view: view, depthTolerance: base.depthTolerance, refined: false).vertices.count
        }
        time("folds tested on f, ends to a quarter pixel") {
            h.visibleFolds(view: view, depthTolerance: base.depthTolerance, refined: false,
                           resolution: pixel).vertices.count
        }
        time("folds refined onto f") { h.foldCurves(view: view).vertices.count }
        time("folds refined and tested, ends to a quarter pixel") {
            h.visibleFolds(view: view, depthTolerance: base.depthTolerance, resolution: pixel).vertices.count
        }
    }
    time("folds traced on the grid") { grid.foldCurves(view: view, refined: false).vertices.count }
    time("folds traced and tested on the grid") {
        grid.visibleFolds(view: view, depthTolerance: base.depthTolerance, refined: false).vertices.count
    }
    for layer in base.layers where layer.spec.source.liesOnWall && layer.spec.clipped {
        if h.refine != nil {
            time("\(layer.spec.name) tested on f") {
                h.visibleWallCurves(layer.paths, view: view, depthTolerance: base.depthTolerance).vertices.count
            }
            time("\(layer.spec.name) tested on f, to a quarter pixel") {
                h.visibleWallCurves(layer.paths, view: view, depthTolerance: base.depthTolerance,
                                 resolution: pixel).vertices.count
            }
        }
        time("\(layer.spec.name) tested on the grid") {
            grid.visibleWallCurves(layer.paths, view: view, depthTolerance: base.depthTolerance).vertices.count
        }
    }
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

    var navigator = Navigator(orbit: Orbit(matching: preset.projection),
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
    if let spec = args.flags["center"] {
        let parts = spec.split(separator: ",").compactMap { Double($0) }
        guard parts.count == 3 else { throw CLIError("--center wants \"re,im,z\" in world units") }
        let p = scene.camera.view(P3<WorldSpace>(parts[0], parts[1], parts[2]))
        navigator.framing.center = P2(p.x, p.y)
    }
    if let zoom = try args.double("zoom") {
        navigator = navigator.applying(
            .zoom(factor: zoom, at: SIMD2(Double(viewport.width) / 2,
                                          Double(viewport.height) / 2)),
            in: viewport)
    }
    // The inspector's Depth tolerance slider, from the command line.
    if let depthTolerance = try args.double("depth-tolerance") { base.depthTolerance = depthTolerance }
    return (bundle, preset, base, navigator)
}

/// The surface controls' costs, as the window pays them: `SurfaceDrawing.build`
/// from the drawing before, then one preview frame of the result -- through
/// `Scene.drawing` when the edit kept the surface, as the document does, so
/// a curve edit re-uploads nothing.
func surfaceCosts(_ args: Args) throws {
    let repeats = try args.int("repeat", 3)
    let renderer = try MetalRenderer()
    let viewport = Viewport(width: 1600, height: 1200)
    let target = try renderer.makePreviewTarget(viewport)
    let orbit = Orbit(matching: SurfaceRequest.projection)
    func seconds(_ d: Duration) -> Double {
        Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
    }
    func ms(_ x: Double) -> String { x < 1 ? String(format: "%.1f ms", x * 1e3) : String(format: "%.2f s", x) }

    func measure(_ label: String, _ base: SurfaceRequest, _ edit: (inout SurfaceRequest) -> Void,
                 draft: Bool = false, from start: SurfaceDrawing? = nil) throws {
        let before = try start ?? SurfaceDrawing.build(base)
        let scene = before.scene(camera: orbit.camera)
        guard let bounds = scene.quickBounds() else { throw CLIError("\(label): empty") }
        let navigator = Navigator(orbit: orbit, framing: .fitting(bounds, in: viewport))
        let options = PreviewOptions(slopeScale: 0)
        var request = before.request
        edit(&request)
        var builds: [Double] = [], frames: [Double] = []
        var kept = false
        for _ in 0..<repeats {
            // The window as it stands before the edit: its textures, its
            // curves and its normals already made.
            try renderer.renderPreview(scene, navigator: navigator, viewport: viewport,
                                       options: options, into: target)
            let clock = ContinuousClock()
            var after: SurfaceDrawing!
            builds.append(seconds(try clock.measure {
                after = try SurfaceDrawing.build(request, reusing: before, draft: draft)
            }))
            kept = after.content == before.content
            let next = kept ? scene.drawing(after.layers) : after.scene(camera: orbit.camera)
            frames.append(seconds(try clock.measure {
                try renderer.renderPreview(next, navigator: navigator, viewport: viewport,
                                           options: options, into: target)
            }))
        }
        let b = builds.sorted()[builds.count / 2], f = frames.sorted()[frames.count / 2]
        print("  \(label.padding(toLength: 40, withPad: " ", startingAt: 0))"
              + "\(ms(b).padding(toLength: 10, withPad: " ", startingAt: 0))"
              + "\(ms(f).padding(toLength: 10, withPad: " ", startingAt: 0))"
              + "\(ms(b + f).padding(toLength: 10, withPad: " ", startingAt: 0))"
              + (kept ? "curves only" : "new surface"))
    }
    /// A step of a drag: the drawing a drag's first step built, from the full
    /// one, edited again at the draft -- what each frame of a drag costs.
    func drag(_ label: String, _ base: SurfaceRequest,
              _ edit: @escaping (inout SurfaceRequest) -> Void) throws {
        let full = try SurfaceDrawing.build(base)
        let first = try SurfaceDrawing.build(base.drafted(), reusing: full, draft: true)
        try measure(label + ", dragging", base.drafted(), { r in
            var target = base; edit(&target); r = target.drafted()
        }, draft: true, from: first)
    }

    let presets = Dictionary(uniqueKeysWithValues: SurfacePreset.catalog.map { ($0.name, $0.request) })
    print("  \("control".padding(toLength: 40, withPad: " ", startingAt: 0))build     frame     total")

    let only = args.flags["only"]
    func wanted(_ section: String) -> Bool { only == nil || only == section }

    if wanted("torus") {
    print("torus")
    let torus = presets["torus"]!
    try measure("radii", torus) { $0.shape = .torus(major: 2.3, minor: 0.9) }
    try measure("line counts", torus) { $0.lines = SIMD2(48, 24) }
    try measure("line width", torus) { $0.style.lines = 0.5 }
    var wound = torus; wound.linearCurves = SurfaceRequest.LinearCurves(slope: 0.4, turns: 24, count: 3)
    try measure("linear curve on", torus) { $0.linearCurves = wound.linearCurves }
    try measure("linear curve slope", wound) { $0.linearCurves?.slope = 0.618 }
    }

    if wanted("sn") {
    print("sn")
    let sn = presets["sn"]!
    try measure("radii", sn) { $0.shape = .sn(modulus: 0.64, major: 2.3, minor: 0.9, grid: 600) }
    try measure("modulus", sn) { $0.shape = .sn(modulus: 0.7, major: 2, minor: 1, grid: 600) }
    try measure("grid", sn) { $0.shape = .sn(modulus: 0.64, major: 2, minor: 1, grid: 400) }
    try drag("radii", sn) { $0.shape = .sn(modulus: 0.64, major: 2.3, minor: 0.9, grid: 600) }
    try drag("modulus", sn) { $0.shape = .sn(modulus: 0.7, major: 2, minor: 1, grid: 600) }
    try drag("grid", sn) { $0.shape = .sn(modulus: 0.64, major: 2, minor: 1, grid: 400) }
    }

    guard wanted("forced") else { return }
    print("forced")
    let forced = presets["forced"]!
    let base = try SurfaceDrawing.build(forced)
    func forcedEdit(_ label: String, draft: Bool = false, grid: Int? = nil,
                    from start: SurfaceDrawing? = nil,
                    _ change: @escaping (inout String, inout Double, inout Int, inout Int, inout Int,
                                         inout Double, inout Int) -> Void) throws {
        try measure(label, forced, { r in
            if let grid { r.grid = grid }
            guard case .forced(var s, var f, var h, var ra, var ax, var rad, var t) = r.shape
            else { return }
            change(&s, &f, &h, &ra, &ax, &rad, &t)
            r.shape = .forced(system: s, fit: f, harmonics: h, radial: ra, axial: ax, radius: rad,
                              turns: t)
        }, draft: draft, from: start ?? base)
    }
    try forcedEdit("revolution radius") { _, _, _, _, _, rad, _ in rad = 3 }
    try forcedEdit("axial component") { _, _, _, _, ax, _, _ in ax = 2 }
    try forcedEdit("turns, fewer") { _, _, _, _, _, _, t in t = 223 }
    try forcedEdit("turns, more") { _, _, _, _, _, _, t in t = 567 }
    try measure("coordinate curves on", forced, { $0.lines = SIMD2(36, 18) }, from: base)
    try measure("linear curve on", forced, {
        $0.linearCurves = SurfaceRequest.LinearCurves(slope: 0.4, turns: 24, count: 3)
    }, from: base)
    var forcedWound = forced
    forcedWound.linearCurves = SurfaceRequest.LinearCurves(slope: 0.4, turns: 24, count: 3)
    try measure("linear curve slope", forcedWound) { $0.linearCurves?.slope = 0.618 }
    try forcedEdit("harmonics") { _, _, h, _, _, _, _ in h = 20 }
    try forcedEdit("fit length") { _, f, _, _, _, _, _ in f = 6000 }
    func forcedShape(_ r: inout SurfaceRequest, radius: Double? = nil) {
        guard case .forced(let s, let f, let h, let ra, let ax, let rad, let t) = r.shape
        else { return }
        r.shape = .forced(system: s, fit: f, harmonics: h, radial: ra, axial: ax,
                          radius: radius ?? rad, turns: t)
    }
    try drag("revolution radius", forced) { forcedShape(&$0, radius: 3) }
    // The release: the full request, from the last draft, with the full
    // grid's values and the full run still kept.
    let dragged = try SurfaceDrawing.build(forced.drafted(), reusing: base, draft: true)
    try measure("revolution radius, released", forced.drafted(), { r in
        var target = forced; forcedShape(&target, radius: 3); r = target
    }, from: dragged)
}

func landscapeCosts(_ args: Args) throws {
    let resolution = try args.int("resolution", NativeLandscape.defaultResolution)
    let draftResolution = 240
    let clock = ContinuousClock()
    func ms(_ d: Duration) -> String {
        let x = Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
        return x < 1 ? String(format: "%.0f ms", x * 1e3) : String(format: "%.2f s", x)
    }
    func column(_ s: String, _ width: Int = 11) -> String {
        s.padding(toLength: width, withPad: " ", startingAt: 0)
    }
    /// `Document.setTruncation`: the truncation, and the magnitude levels chosen under it.
    func truncated(_ manifest: Manifest, _ truncation: Truncation) -> Manifest {
        var m = manifest
        m.truncation = truncation
        let levels = LandscapeStyle.levels(under: truncation)
        for i in m.layers.indices {
            guard case .contour(let field, _, let restriction, let tiled) = m.layers[i].source else { continue }
            switch m.layers[i].name {
            case "mag_major": m.layers[i].source = .contour(field: field, levels: levels.major,
                                                            restriction: restriction, tiled: tiled)
            case "mag_minor": m.layers[i].source = .contour(field: field, levels: levels.minor,
                                                            restriction: restriction, tiled: tiled)
            default: break
            }
        }
        return m
    }

    print(column("preset", 10) + column("sample") + column("dragging") + column("truncation")
          + column("next truncation") + column("hatch") + column("levels") + column("dragging")
          + "refine on")
    for preset in Catalog.native.presets where args.flags["only"].map({ $0 == preset.name }) ?? true {
        var request = LandscapeRequest(preset: preset, resolution: resolution)
        request.spacing = nil
        var draft = request
        draft.resolution = min(resolution, draftResolution)
        var bundle: KurvenBundle!
        let sample = clock.measure { bundle = try? NativeLandscape.build(request) }
        let dragging = clock.measure { _ = try? NativeLandscape.build(draft, refine: false) }
        guard let bundle else { throw CLIError("\(preset.name): did not build") }

        var cells = [column(preset.name, 10), column(ms(sample)), column(ms(dragging))]
        if let ceiling = LandscapeStyle.ceiling(of: bundle.manifest.truncation) {
            var once: KurvenBundle!
            let truncationHeight = clock.measure { once = bundle.restyled(truncated(bundle.manifest, .uniform(ceiling * 0.9))) }
            let next = clock.measure { _ = once.restyled(truncated(once.manifest, .uniform(ceiling * 0.85))) }
            cells += [column(ms(truncationHeight)), column(ms(next))]
        } else {
            cells += [column("-"), column("-")]
        }
        if let i = bundle.manifest.layers.firstIndex(where: {
            if case .hatching(_, .solid) = $0.source { false } else if case .hatching = $0.source { true } else { false }
        }) {
            var m = bundle.manifest
            if case .hatching(var hatching, let faces) = m.layers[i].source {
                hatching.spacing *= 1.1
                m.layers[i].source = .hatching(hatching, faces: faces)
            }
            cells.append(column(ms(clock.measure { _ = bundle.restyled(m) })))
        } else {
            cells.append(column("-"))
        }
        // The level slider on the layer it costs most on: the finer phase.
        if let spec = bundle.manifest.layers.first(where: { $0.name == "ang_minor" })
            ?? bundle.manifest.layers.first(where: { KurvenBundle.levels(of: $0) != nil }),
           let base = KurvenBundle.levels(of: spec), base.count >= 2 {
            let n = base.count + 1, lo = base.min()!, hi = base.max()!
            let levels = (0..<n).map { lo + (hi - lo) * Double($0) / Double(n - 1) }
            let grid = KurvenBundle(url: bundle.url, manifest: bundle.manifest, surface: bundle.surface,
                                    layers: bundle.layers, wallMesh: bundle.wallMesh, refine: nil)
            cells.append(column(ms(clock.measure { _ = bundle.layer(spec, levels: levels) })))
            cells.append(column(ms(clock.measure { _ = grid.layer(spec, levels: levels) })))
        } else {
            cells += [column("-"), column("-")]
        }
        let plain = try NativeLandscape.build(request, refine: false)
        cells.append(ms(clock.measure {
            _ = plain.refined(by: NativeLandscape.refiner(for: plain)?.refine)
        }))
        print(cells.joined())
    }
}

func previewMode(_ args: Args) throws -> PreviewMode {
    switch args.flags["mode"] ?? "drawing" {
    case "drawing": .drawing
    case "shaded": .shaded(Lighting())
    case "depth": .depth
    case let other: throw CLIError("unknown --mode '\(other)'; try drawing, shaded or depth")
    }
}

/// `--slope K`: the visibility test's slope allowance, so a run can be compared with
/// the bake's predicate (`--slope 0`) at the same camera. `--stroke-width W`: the
/// widest stroke in pixels, which is what a Retina window doubles.
func previewOptions(_ args: Args, mode: PreviewMode) throws -> PreviewOptions {
    var options = PreviewOptions(mode: mode)
    options.draft = args.switches.contains("draft")
    if let k = try args.double("slope") { options.slopeScale = Float(k) }
    if let w = try args.double("stroke-width") { options.strokeWidth = Float(w) }
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
    // images. Without it "the preview looks like the drawing" is an impression.
    let f = navigator.framing.frame(viewport)
    let meta = JSONValue.object([
        // In screen order axis0 runs down the image (view y) and axis1 across
        // it (view x), so this is (left, right) and (top, bottom).
        "viewX": .array([.double(f.axis1.lo), .double(f.axis1.hi)]),
        "viewY": .array([.double(f.axis0.lo), .double(f.axis0.hi)]),
        "width": .int(viewport.width), "height": .int(viewport.height),
        "azimuth": .double(navigator.orbit.azimuth.degrees),
        "elevation": .double(navigator.orbit.elevation.degrees),
        "depthTolerance": .double(base.depthTolerance),
        "projection": preset.projection.json,
    ])
    try (meta.canonical + "\n").write(
        to: output.deletingPathExtension().appendingPathExtension("frame.json"),
        atomically: true, encoding: .utf8)

    print("""
        \(bundle.url.lastPathComponent) -> \(output.lastPathComponent)          [\(preset.name), \(args.flags["mode"] ?? "drawing")]
          \(viewport.width)x\(viewport.height)          azimuth \(String(format: "%.1f", navigator.orbit.azimuth.degrees))          elevation \(String(format: "%.1f", navigator.orbit.elevation.degrees))
          took \(elapsed)
          frame \(output.deletingPathExtension().lastPathComponent).frame.json
        """)
}

/// Strokes that come and go while the camera turns, as a number.
///
/// Flicker is a motion artifact, so no single frame shows it. This renders a
/// short orbit -- the camera turning `--step` degrees a frame about its target,
/// the framing held still -- and draws every frame twice: once as the preview
/// does, and once with the depth test switched off. A pixel that has a stroke
/// in both frames of a pair *without* the test, but curves in only one of them
/// *with* it, is a pixel where nothing moved and the visibility test changed
/// its mind. That is the flicker, and it is counted apart from the motion:
///
/// - *toggled*: curves in one frame and not the next. Honest motion contributes,
///   so this only compares runs at the same step.
/// - *flipped*: the test changing its mind under a stroke that stayed put.
///   Some of it is honest too -- a stroke sliding behind a ridge flips at the
///   ridge -- but that is a pixel or so per stroke, not a dash pattern.
///
/// Both are fractions of the mean curves per frame, so a picture with more curves are
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
    unclipped.depthTolerance = .infinity
    let options = try previewOptions(args, mode: .drawing)

    let renderer = try MetalRenderer()
    let target = try renderer.makePreviewTarget(viewport)
    let pixels = viewport.width * viewport.height
    // Curves are any visible mark: a stroke narrower than a pixel is drawn
    // lighter, not narrower, so mid-grey would miss the drawing's thinnest
    // layers.
    func curves(_ scene: Scene, _ navigator: Navigator) throws -> [Bool] {
        try renderer.renderPreview(scene, navigator: navigator, viewport: viewport,
                                   options: options, into: target)
        let bgra = try PNG.bgra(target)
        return (0..<pixels).map { k in
            let b = Int(bgra[4 * k]), g = Int(bgra[4 * k + 1]), r = Int(bgra[4 * k + 2])
            return 299 * r + 587 * g + 114 * b < 224 * 1000
        }
    }

    var kept: [Bool] = [], everything: [Bool] = []
    var strokedTotal = 0, toggled = 0, flipped = 0
    for i in 0..<frames {
        var navigator = start
        navigator.orbit.azimuth = Angle(degrees: start.orbit.azimuth.degrees
                                        + Double(i) * step)
        let k = try curves(base, navigator)
        if let dump {
            try PNG.write(target, to: dump.appendingPathComponent(
                String(format: "frame%03d.png", i)))
        }
        let e = try curves(unclipped, navigator)
        if i > 0 {
            for p in 0..<pixels where k[p] != kept[p] {
                toggled += 1
                if e[p] && everything[p] { flipped += 1 }
            }
        }
        strokedTotal += k.reduce(0) { $0 + ($1 ? 1 : 0) }
        kept = k; everything = e
    }
    let perFrame = max(Double(strokedTotal) / Double(frames), 1)
    let toggledRate = Double(toggled) / Double(frames - 1) / perFrame
    let flippedRate = Double(flipped) / Double(frames - 1) / perFrame

    func percent(_ x: Double) -> String { String(format: "%.2f%%", x * 100) }
    let from = start.orbit.azimuth.degrees
    print("""
        \(bundle.url.lastPathComponent)  [\(preset.name)]  \(frames) frames \
        \(step)° apart, depth tolerance \(base.depthTolerance), slope \(options.slopeScale)
          \(viewport.width)x\(viewport.height)  azimuth \(String(format: "%.2f", from)) \
        to \(String(format: "%.2f", from + Double(frames - 1) * step))  \
        elevation \(String(format: "%.1f", start.orbit.elevation.degrees))
          curves      \(Int(perFrame.rounded())) px a frame
          toggled  \(percent(toggledRate)) of it a frame   (motion included)
          flipped  \(percent(flippedRate)) of it a frame   (the stroke stayed; the test changed its mind)
        """)
    if let limit = try args.double("max-flip"), flippedRate > limit {
        throw CLIError("flipped \(percent(flippedRate)) a frame, over the \(percent(limit)) allowed")
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
            truncation \(preset.truncationHeight.map(fmt) ?? "none")
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

/// `--bands BELOW:HEIGHT,... --beyond H` as a truncation: a height per band of Re z.
func realBands(_ text: String, beyond: String?) throws -> Truncation {
    var bands: [RealBand] = []
    for part in text.split(separator: ",") {
        let pair = part.split(separator: ":").map { Double($0.trimmingCharacters(in: .whitespaces)) }
        guard pair.count == 2, let below = pair[0], let truncationHeight = pair[1], truncationHeight > 0 else {
            throw CLIError("--bands wants BELOW:HEIGHT,... with each HEIGHT > 0, got '\(part)'")
        }
        bands.append(RealBand(below: below, height: truncationHeight))
    }
    guard !bands.isEmpty, zip(bands, bands.dropFirst()).allSatisfy({ $0.below < $1.below }) else {
        throw CLIError("--bands wants its bands in increasing BELOW, got '\(text)'")
    }
    guard let beyond, let z = Double(beyond), z > 0 else {
        throw CLIError("--bands needs --beyond Z, the height past the last band")
    }
    return .realBands(bands, beyond: z)
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
    if let truncationHeight = args.flags["truncation"], let z = Double(truncationHeight) { request.truncation = .uniform(z) }
    if let bands = args.flags["bands"] { request.truncation = try realBands(bands, beyond: args.flags["beyond"]) }
    if args.switches.contains("no-truncation") { request.truncation = Truncation.none }
    if let s = args.flags["spacing"], let v = Double(s) { request.spacing = v }

    let clock = ContinuousClock()
    var bundle: KurvenBundle!
    // --repeat N builds it N times over, which is how a profiler gets a long
    // enough look at one build.
    for _ in 1..<(try args.int("repeat", 1)) { _ = try NativeLandscape.build(request) }
    let sampled = try clock.measure { bundle = try NativeLandscape.build(request) }
    try bundle.write(to: output)
    let m = bundle.manifest
    print("""
        \(m.provenance.function) -> \(output.lastPathComponent) (in \(sampled))
          \(m.height.shape.nx)x\(m.height.shape.ny) samples, truncation \(m.truncation)
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
    if let truncationHeight = args.flags["truncation"], let z = Double(truncationHeight) { request.truncation = .uniform(z) }
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

func fmt(_ x: Double) -> String { String(format: "%g", x) }

func sourceName(_ source: LayerSource) -> String {
    switch source {
    case .file: "dumped"
    case .contour(let field, let levels, _, _): "\(levels.count) levels of \(field.rawValue)"
    case .hatching(let h, .walls(let edges, _, _, _)):
        "wall hatching, \(edges.count) walls every \(fmt(h.spacing))"
    case .hatching(let h, .plateaus):
        "plateau hatching at \(fmt(h.angle * 180 / .pi))° every \(fmt(h.spacing))"
    case .hatching(let h, .solid):
        "hatching every \(fmt(h.spacing)) at \(fmt(h.angle * 180 / .pi))°, "
            + "\(h.widthSteps) width\(h.widthSteps == 1 ? "" : "s")"
    case .wallEdges(let edges, _, _): "edges of \(edges.count) walls"
    case .truncationEdge: "truncation edge"
    case .coordinateCurves(let u, let v): "\(u) + \(v) coordinate curves"
    case .linearCurves(let slope, let turns, let count):
        "\(count) linear curve\(count == 1 ? "" : "s") at slope \(fmt(slope)), \(turns) turns"
    case .foldCurves: "fold curves"
    case .trajectory: "a trajectory"
    case .edges: "the edges of a solid"
    }
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
    print("  truncation        \(m.truncation)")
    print("  occluder   step \(m.occluder.step)  \(m.occluder.tiles.count) tile(s)  "
          + "region \(regionName(m.occluder.region))  walls \(wallsName(m.occluder.walls))")
    for l in m.layers {
        let paths = (try? bundle.layer(l.name).paths.count) ?? 0
        print("    " + pad(l.name, 12) + " " + pad(l.role.rawValue, 10) + " "
              + pad(String(paths), 7) + " paths  lw \(l.width)"
              + (l.clipped ? "" : "  unclipped"))
    }
    for p in m.presets {
        print("  preset     \(pad(p.name, 10)) shear \(p.projection.shear)  "
              + "x \(p.projection.xAngle)  z \(p.projection.zAngle)  "
              + "flipX \(p.projection.flipX)  yScale \(p.projection.yScale.map { String($0) } ?? "-")  "
              + "depth tolerance \(p.depthTolerance)  buffer \(p.buffer)")
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
    return "        at \(i):\n          file  : \(String(x[from..<min(x.count, to)]))\n"
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
    case "surface-costs": try surfaceCosts(args)
    case "landscape-costs": try landscapeCosts(args)
    case "bench": try bench(args)
    case "preview": try preview(args)
    case "flicker": try flicker(args)
    case "catalog": try catalogCommand(args)
    case "landscape": try landscapeCommand(args)
    case "refine": try refineCommand(args)
    case "inspect": try inspect(args)
    case "contract": try contract(args)
    case "fixtures": try fixturesCommand(args)
    default:
        print(usage)
        exit(args.positional.isEmpty ? 0 : 1)
    }
} catch {
    FileHandle.standardError.write(Data("kurven-cli: \(error)\n".utf8))
    exit(1)
}
