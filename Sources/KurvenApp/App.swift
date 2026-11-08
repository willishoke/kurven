import SwiftUI
import AppKit
import UniformTypeIdentifiers
import KurvenCore
import KurvenMetal
import KurvenBake
import KurvenDynamics
import KurvenLandscape

/// The window.
///
/// Not a `DocumentGroup`. SwiftUI's document types read a file into memory as
/// `Data` or a `FileWrapper`, and a `.kurven` bundle is a directory of npy
/// arrays that can run to hundreds of megabytes -- reading it that way to hand
/// the loader a path it already had is the wrong shape. So the app opens URLs:
/// an open panel, the Finder (through the document type
/// `scripts/bundle-app.sh` declares), and the command line.
@main
struct KurvenApplication: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate

    // `SwiftUI.Scene`, spelled out: KurvenCore has a `Scene` too, and it is
    // the more important of the two here.
    var body: some SwiftUI.Scene {
        WindowGroup {
            DocumentWindow(document: delegate.document, thumbnails: delegate.thumbnails,
                           open: delegate.openPanel)
                .frame(minWidth: 900, minHeight: 600)
        }
        .commands {
            CommandGroup(replacing: .newItem) {
                // A landscape or a surface is the new document here: the app no
                // longer needs a bundle someone made earlier to show anything.
                Button("New…") { delegate.browse() }
                    .keyboardShortcut("n")
                Button("Open…") { delegate.openPanel() }
                    .keyboardShortcut("o")
            }
            CommandGroup(after: .saveItem) {
                Button("Save Bundle…") { delegate.savePanel() }
                    .keyboardShortcut("s")
                    .disabled(!delegate.document.isOpen)
            }
        }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let document = Document()
    /// One picture per catalog entry, drawn once and kept. App-level like the
    /// document, because the picker outlives whatever is open.
    let thumbnails = Thumbnails()

    // Run as a bare executable (`swift run KurvenApp`) there is no Info.plist,
    // so AppKit starts the process as a background app: no Dock icon, no menu
    // bar, and its window behind the terminal that launched it. Ask for a
    // regular app and bring it forward; inside a bundle this changes nothing.
    func applicationWillFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.activate(ignoringOtherApps: true)
        // A path on the command line opens it, so the app can be driven from a
        // shell the same way the CLI is.
        let args = CommandLine.arguments.dropFirst()
        // `--open PATH` as well as a bare path, and the difference is not
        // cosmetic. AppKit reads a bare argument as "this process was launched
        // to open that file", and SwiftUI's WindowGroup then declines to make
        // its default window -- so `Kurven recip.kurven` shows nothing at all,
        // while `Kurven --open recip.kurven` shows the landscape. A
        // dash-prefixed argument is not read that way. The Finder and
        // `open -a` are unaffected: they deliver the file through
        // `application(_:open:)` rather than through argv.
        let flagged = args.firstIndex(of: "--open").flatMap {
            $0 + 1 < args.endIndex ? String(args[$0 + 1]) : nil
        }
        let bare = args.first { !$0.hasPrefix("-") && $0.hasSuffix(".kurven") }
        let bundlePath = flagged ?? bare

        // `--screenshot PATH` renders one frame through the app's own state --
        // its Document, its Navigator, its preview options -- and exits. Without
        // it the only way to know the window draws the right thing is to look at
        // it, and a test that requires someone to look at it is not one.
        // `--landscape NAME|EXPRESSION [--screenshot PATH]` samples a landscape
        // through the window's own Document -- the same path the
        // picker takes -- so "the function picker works" is checkable by
        // comparing a PNG rather than by clicking.
        if let i = args.firstIndex(of: "--landscape"), i + 1 < args.endIndex {
            let what = args[i + 1]
            let shot = args.firstIndex(of: "--screenshot").flatMap {
                $0 + 1 < args.endIndex ? URL(fileURLWithPath: args[$0 + 1]) : nil
            }
            let save = args.firstIndex(of: "--save").flatMap {
                $0 + 1 < args.endIndex ? URL(fileURLWithPath: args[$0 + 1]) : nil
            }
            let resolution = args.firstIndex(of: "--resolution")
                .flatMap { $0 + 1 < args.endIndex ? Int(args[$0 + 1]) : nil }
            let truncationHeight = args.firstIndex(of: "--truncation")
                .flatMap { $0 + 1 < args.endIndex ? Double(args[$0 + 1]) : nil }
            // `--drag truncation|levels STEPS`: step that control as a drag does and
            // report the main actor's longest stall -- the beachball, measured.
            let drag = args.firstIndex(of: "--drag").flatMap { i -> (String, Int)? in
                guard i + 2 < args.endIndex, let n = Int(args[i + 2]) else { return nil }
                return (args[i + 1], n)
            }
            Task { await headlessLandscape(what, resolution: resolution, truncationHeight: truncationHeight, drag: drag,
                                           screenshot: shot, save: save) }
            return
        }
        func path(_ flag: String) -> URL? {
            args.firstIndex(of: flag).flatMap {
                $0 + 1 < args.endIndex ? URL(fileURLWithPath: args[$0 + 1]) : nil
            }
        }
        /// Every `FLAG KEY=NUMBER` pair, in order.
        func pairs(_ flag: String) -> [(String, Double)] {
            zip(args.indices, args).compactMap { k, f in
                guard f == flag, k + 1 < args.endIndex else { return nil }
                let pair = args[k + 1].split(separator: "=", maxSplits: 1)
                guard pair.count == 2, let v = Double(pair[1]) else { return nil }
                return (String(pair[0]), v)
            }
        }
        // Painting, with no window: `--pick PX,PY` and then edits as the
        // "Curves on" section makes them, in the order given, each one undo
        // step -- `--function EXPR --periods P1,P2`, `--add NAME` (an Add
        // menu entry, by the name of the layer it adds), `--remove LAYER`,
        // `--levels LAYER=COUNT` and `--strokes LAYER=COUNT` (each a drag of
        // its slider), `--axis real|imag` -- and `--undo`, `--redo`,
        // `--save PATH`, `--bake PATH`, `--screenshot PATH`, `--inspector
        // PATH` (the section, drawn) and `--dump LAYER=PREFIX` (the layer's
        // curves, as `kurven-cli fixtures` writes a hatching's) among them.
        let painting: Set<String> = ["--add", "--remove", "--function", "--undo", "--redo", "--axis",
                                     "--dump", "--inspector"]
        if args.contains(where: painting.contains), let bundlePath {
            Task { await headlessPaint(URL(fileURLWithPath: bundlePath), args: Array(args)) }
            return
        }
        // `--pick PX,PY [--screenshot PATH]` opens a bundle with no window,
        // draws a frame, and picks at that pixel as a click does: the depth
        // the frame left there, taken back through the camera, and the face
        // under it. Repeated, the clicks come in order, so one off the
        // drawing clears the pick before it. Each prints what it picked, and
        // `--screenshot` draws the frame again with the pick's boundary over
        // it. With `--at-once` the clicks come without waiting for one's
        // boundary before the next, as fast clicks do, and what is left once
        // the last lands is printed.
        let picks = zip(args.indices, args).compactMap { k, flag -> [Double]? in
            flag == "--pick" && k + 1 < args.endIndex ? args[k + 1].split(separator: ",").compactMap { Double($0) } : nil
        }
        if !picks.isEmpty, let bundlePath {
            guard picks.allSatisfy({ $0.count == 2 }) else {
                FileHandle.standardError.write(Data("Kurven: --pick takes a pixel, PX,PY\n".utf8))
                exit(2)
            }
            Task { await headlessPick(URL(fileURLWithPath: bundlePath), at: picks.map { SIMD2($0[0], $0[1]) },
                                      atOnce: args.contains("--at-once"), screenshot: path("--screenshot")) }
            return
        }
        // `--surface NAME [--screenshot PATH] [--bake PATH] [--save PATH]`
        // opens a surface from the catalog through the window's own Document
        // -- the gallery's path -- so the window can be held to `kurven-cli
        // surface NAME`: the same preview frame, pixel for pixel, the same
        // bake, and the same bundle. With no output it stays open, on that
        // surface. `--set FIELD=VALUE`, repeated, then edits it as the
        // controls do -- one edit at a time, each built from the drawing
        // before it.
        if let i = args.firstIndex(of: "--surface"), i + 1 < args.endIndex {
            let name = args[i + 1]
            let resolution = args.firstIndex(of: "--resolution")
                .flatMap { $0 + 1 < args.endIndex ? Int(args[$0 + 1]) : nil }
            Task { await headlessSurface(name, edits: pairs("--set"), screenshot: path("--screenshot"),
                                         bake: path("--bake"), save: path("--save"), resolution: resolution) }
            return
        }
        // `--thumbnails` draws every catalog entry into the picker's cache and
        // exits, so a first look at the gallery is instant and so that "the
        // picker draws what the app draws" is a set of files to look at rather
        // than a claim.
        if args.contains("--thumbnails") {
            Task { await warmThumbnails() }
            return
        }
        if args.contains("--resample"), let path = bundlePath {
            var settings: [String: String] = [:]
            var index = args.startIndex
            while index < args.endIndex {
                if args[index] == "--resample", index + 1 < args.endIndex,
                   let equals = args[index + 1].firstIndex(of: "=") {
                    settings[String(args[index + 1][..<equals])] =
                        String(args[index + 1][args[index + 1].index(after: equals)...])
                }
                index += 1
            }
            let shot = args.firstIndex(of: "--screenshot").flatMap {
                $0 + 1 < args.endIndex ? URL(fileURLWithPath: args[$0 + 1]) : nil
            }
            Task { await resample(bundle: URL(fileURLWithPath: path),
                                  settings: settings, screenshot: shot) }
            return
        }
        if let i = args.firstIndex(of: "--screenshot"), i + 1 < args.endIndex,
           let path = bundlePath {
            let output = URL(fileURLWithPath: args[i + 1])
            Task { await screenshot(bundle: URL(fileURLWithPath: path), to: output); exit(0) }
            return
        }
        // `--bake PATH` runs the window's own bake -- the same Document, the
        // same Scene value, the same BakeOptions -- and exits, so "the app bakes
        // what the CLI bakes" can be checked by comparing two files rather than
        // by reading two code paths and believing they agree. `--save PATH`
        // writes the bundle as ⌘S does. Before either, a layer's levels are
        // set by `--levels LAYER=COUNT` as its slider sets them, a hatching's
        // strokes by `--strokes LAYER=COUNT` as its slider sets them, and then
        // a surface's request edited by `--set FIELD=VALUE` as its controls
        // edit it.
        if path("--bake") != nil || path("--save") != nil, let bundlePath {
            let resolution = args.firstIndex(of: "--resolution")
                .flatMap { $0 + 1 < args.endIndex ? Int(args[$0 + 1]) : nil }
            Task { await headlessBundle(URL(fileURLWithPath: bundlePath), edits: pairs("--set"),
                                        levels: pairs("--levels"), strokes: pairs("--strokes"),
                                        save: path("--save"), bake: path("--bake"), resolution: resolution) }
            return
        }
        if let path = bundlePath { document.open(URL(fileURLWithPath: path)) }
        if flagged == nil, bare != nil {
            FileHandle.standardError.write(Data("""
                Kurven: macOS does not open a window for a process launched \
                with a bare file argument. Use --open \(bare!), or open the \
                bundle from the Finder.\n
                """.utf8))
        }
        fillScreen()
    }

    /// Open filling the screen.
    ///
    /// A landscape is worth the whole display: the sidebar is fixed at 320
    /// points, so every point the window gains is a point of drawing. Zoomed
    /// rather than fullscreen -- this takes the screen minus the menu bar and
    /// the Dock, and leaves the green button meaning what it usually means.
    ///
    /// Deferred by one turn of the run loop because SwiftUI has not created the
    /// window yet when this delegate method runs, and polled a few times after
    /// that because "not yet" is the normal answer on the first turn. Whatever
    /// frame was restored from the last run is overwritten, which is the point.
    private func fillScreen(attempt: Int = 0) {
        // Spaced in time, not merely deferred: twenty turns of the run loop go
        // by in under a millisecond, which is before SwiftUI has made the
        // window, so a chain of `async` retries is twenty ways of asking too
        // early. Two seconds of 50 ms polls is a wait.
        DispatchQueue.main.asyncAfter(deadline: .now() + (attempt == 0 ? 0 : 0.05)) { [weak self] in
            guard let window = NSApp.windows.first(where: { $0.isVisible }) else {
                if attempt < 40 { self?.fillScreen(attempt: attempt + 1) }
                return
            }
            guard let screen = window.screen ?? NSScreen.main else { return }
            window.setFrame(screen.visibleFrame, display: true)
        }
    }

    /// Make the landscape and report it. The window is never shown; everything
    /// it would do is done through the same model.
    private func headlessLandscape(_ what: String, resolution: Int?, truncationHeight: Double?,
                                   drag: (control: String, steps: Int)? = nil,
                                   screenshot: URL?, save: URL?) async {
        let catalog = document.catalog
        if let preset = catalog.preset(what) {
            document.create(preset)
        } else {
            // Not a name in the catalog, so it is an expression: the field and
            // the flag accept the same language.
            document.create(catalog.presets[0])
            while document.sampling { try? await Task.sleep(for: .milliseconds(25)) }
            document.landscape?.expression = what
            document.landscape?.name = ""
            document.landscapeEdited(draft: false)
        }
        if let resolution {
            while document.sampling { try? await Task.sleep(for: .milliseconds(25)) }
            document.landscape?.resolution = resolution
            document.landscapeEdited(draft: false)
        }
        while document.sampling { try? await Task.sleep(for: .milliseconds(25)) }
        // Truncating is the other kind of edit: no request, no sampling, just
        // the curves derived again from grids that are already here. Driving it
        // from here is how that path is checked without a pair of hands.
        if let truncationHeight { document.setTruncation(.uniform(truncationHeight)) }
        while document.drawing { try? await Task.sleep(for: .milliseconds(25)) }
        if let drag { await headlessDrag(drag.control, steps: drag.steps) }
        guard document.scene != nil else {
            let why = document.landscapeStatus ?? "no reason given"
            FileHandle.standardError.write(
                Data("Kurven: could not sample \(what) — \(why)\n".utf8))
            exit(1)
        }
        print("Kurven: \(document.title) — \(document.landscapeStatus ?? "")"
              + ", \(document.layers.count) layers, "
              + "\(document.layers.reduce(0) { $0 + $1.paths.count }) paths")
        if let save { document.saveBundle(to: save) }
        if let screenshot { renderScreenshot(to: screenshot, what: document.title) }
        exit(0)
    }

    /// Step a control as a drag does, a step a frame, and say how long the
    /// main actor was ever kept from the next one: a slider whose work is on
    /// the main actor stalls by that work, and past a second or two macOS
    /// shows the beachball. The time the last step's curves took to land is
    /// what the picture lags the hand by.
    private func headlessDrag(_ control: String, steps: Int) async {
        let clock = ContinuousClock()
        var longest: Duration = .zero
        var last = clock.now
        func tick() {
            let now = clock.now
            longest = max(longest, now - last)
            last = now
        }
        let ceiling = LandscapeStyle.ceiling(of: document.truncation) ?? 5
        let layer = document.layers.firstIndex { $0.spec.name == "ang_minor" }
            ?? document.layers.indices.first { document.isDerived(layer: $0) }
        let base = layer.flatMap { document.levels(forLayer: $0)?.count } ?? 0
        for k in 1...max(steps, 1) {
            try? await Task.sleep(for: .milliseconds(16))
            tick()
            switch control {
            case "truncation":
                document.setTruncation(.uniform(ceiling * (1 - 0.3 * Double(k) / Double(steps))))
            case "levels":
                guard let layer else { break }
                document.setLevelCount(base + k, forLayer: layer, draft: true)
            default:
                FileHandle.standardError.write(Data("Kurven: no control '\(control)' to drag\n".utf8))
                exit(2)
            }
            tick()
        }
        let released = clock.now
        if control == "levels" { document.rebuildCurves() }
        while document.drawing {
            try? await Task.sleep(for: .milliseconds(16))
            tick()
        }
        print("Kurven: dragged \(control) \(steps) steps; the main actor's longest stall "
              + "\(longest.formatted(.units(allowed: [.milliseconds]))), the curves landed "
              + "\((clock.now - released).formatted(.units(allowed: [.milliseconds]))) "
              + "after the release")
    }

    /// Open a surface and report it, then draw or bake it and exit -- or, with
    /// nothing to write, show it.
    private func headlessSurface(_ name: String, edits: [(String, Double)] = [],
                                 screenshot: URL?, bake: URL?, save: URL?, resolution: Int?) async {
        guard let preset = SurfacePreset.named(name) else {
            FileHandle.standardError.write(Data(("Kurven: no surface '\(name)'; the catalog has: "
                + SurfacePreset.catalog.map(\.name).joined(separator: ", ") + "\n").utf8))
            exit(1)
        }
        document.create(surface: preset)
        while document.building { try? await Task.sleep(for: .milliseconds(25)) }
        await headlessEdit(edits)
        guard document.surfaceDrawing != nil else {
            FileHandle.standardError.write(Data(
                "Kurven: could not build \(name) — \(document.surfaceStatus ?? "no reason given")\n".utf8))
            exit(1)
        }
        print("Kurven: \(document.title) — \(document.surfaceStatus ?? ""), "
              + "\(document.layers.count) layers, "
              + "\(document.layers.reduce(0) { $0 + $1.paths.count }) paths")
        guard screenshot != nil || bake != nil || save != nil else { fillScreen(); return }
        if let save { headlessSave(to: save) }
        if let screenshot { renderScreenshot(to: screenshot, what: document.title) }
        if let bake { await headlessBake(to: bake, resolution: resolution) }
        exit(0)
    }

    /// Edit the surface's request as its controls do, one field at a time,
    /// each built from the drawing before it.
    private func headlessEdit(_ edits: [(String, Double)]) async {
        for (name, value) in edits {
            guard let field = SurfaceRequest.Field(rawValue: name),
                  document.surface?[field] != nil else {
                FileHandle.standardError.write(Data(("Kurven: \(document.title) has no field '\(name)'; "
                    + "it has: " + (document.surface?.fields.map(\.rawValue) ?? [])
                        .joined(separator: ", ") + "\n").utf8))
                exit(1)
            }
            document.surface?[field] = value
            document.surfaceEdited(draft: false)
            while document.building { try? await Task.sleep(for: .milliseconds(25)) }
            print("Kurven: \(name) = \(value) — \(document.surfaceStatus ?? "")")
        }
    }

    /// Write the bundle as ⌘S does, or exit saying why not.
    private func headlessSave(to url: URL) {
        document.saveBundle(to: url)
        let status = (document.surfaceBundle != nil ? document.surfaceStatus
                      : document.polyhedronBundle != nil ? document.polyhedronStatus : document.landscapeStatus) ?? ""
        print("Kurven: \(status)")
        if !status.hasPrefix("saved") { exit(1) }
    }

    /// The window's bake at `resolution`, or the bundle's own, or exit saying why not.
    private func headlessBake(to output: URL, resolution: Int?) async {
        if let resolution { document.bakeResolution = resolution }
        document.bake(to: output)
        while document.baking { try? await Task.sleep(for: .milliseconds(20)) }
        print("Kurven: \(document.bakeStatus ?? "bake produced no status")")
        if document.bakeStatus?.hasPrefix("bake failed") == true { exit(1) }
    }

    /// Show the picker. Not "make a landscape at once": with fourteen
    /// functions in the catalog, the choice is the interesting part.
    private func warmThumbnails() async {
        let entries = document.catalog.presets.map(GalleryEntry.function)
            + SurfacePreset.catalog.map(GalleryEntry.surface)
        let clock = ContinuousClock()
        let started = clock.now
        thumbnails.warm(entries)
        while !entries.allSatisfy({ thumbnails.images[$0.id] != nil
                                    || thumbnails.isFailed($0) }) {
            try? await Task.sleep(for: .milliseconds(50))
        }
        for entry in entries {
            let mark = thumbnails.images[entry.id] != nil ? "drew" : "FAILED"
            print("  \(mark)  \(entry.id)")
        }
        print("Kurven: \(thumbnails.images.count) of \(entries.count) "
              + "thumbnails in \(clock.now - started)")
        exit(0)
    }

    func browse() {
        document.browsing = true
    }

    func savePanel() {
        let panel = NSSavePanel()
        let name = document.surface?.name ?? document.landscape?.name
            ?? document.polyhedronBundle?.manifest.provenance.function ?? ""
        let kind = document.surfaceBundle != nil ? "surface"
            : document.polyhedronBundle != nil ? "polyhedron" : "landscape"
        panel.nameFieldStringValue = (name.isEmpty ? kind : name) + ".kurven"
        panel.message = "Save this \(kind) as a .kurven bundle"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        document.saveBundle(to: url)
    }

    /// `--resample KEY=VALUE ... --screenshot PATH` resamples a landscape
    /// bundle the way an edit in the window does: the same Document, the
    /// bundle's own request with the overrides applied (`res`, `rMin`,
    /// `rMax`, `iMin`, `iMax`, `truncation`; `LandscapeRequest.applying`),
    /// the same native build. Without it the only way to know a resample
    /// lands is to move a control.
    private func resample(bundle: URL, settings: [String: String],
                          screenshot: URL?) async {
        await document.load(bundle)
        guard let manifest = document.bundle?.manifest else {
            var why = "it did not open"
            if case .failed(_, let message) = document.state { why = message }
            FileHandle.standardError.write(Data("Kurven: cannot resample — \(why)\n".utf8))
            exit(1)
        }
        var overrides: [String: Double] = [:]
        for (key, value) in settings {
            guard let number = Double(value) else {
                FileHandle.standardError.write(Data("Kurven: \(key)=\(value) is not a number\n".utf8))
                exit(1)
            }
            overrides[key] = number
        }
        guard let request = document.landscape?.applying(overrides: overrides) else {
            FileHandle.standardError.write(Data(("Kurven: cannot resample — this bundle's "
                + "provenance is '\(manifest.provenance.example)', not a landscape "
                + "`kurven-cli landscape` or the window wrote, so it opens read-only\n").utf8))
            exit(1)
        }
        document.landscape = request
        document.landscapeEdited(draft: false)
        while document.sampling { try? await Task.sleep(for: .milliseconds(25)) }
        guard document.isLandscape, document.scene != nil else {
            let why = document.landscapeStatus ?? "unknown"
            FileHandle.standardError.write(Data("Kurven: resample failed — \(why)\n".utf8))
            exit(1)
        }
        print("Kurven: rebuilt \(bundle.lastPathComponent) natively — "
              + "\(document.landscapeStatus ?? ""), \(document.layers.count) layers, "
              + "\(document.layers.reduce(0) { $0 + $1.paths.count }) paths")
        if let screenshot { renderScreenshot(to: screenshot, what: document.title) }
        exit(0)
    }

    /// Open a bundle, edit it as the window's controls do, and save or bake
    /// it: the document's own transitions, from open to save, with no window.
    private func headlessBundle(_ bundle: URL, edits: [(String, Double)], levels: [(String, Double)],
                                strokes: [(String, Double)] = [], save: URL?, bake: URL?, resolution: Int?) async {
        await document.load(bundle)
        guard document.scene != nil else {
            var why = "it did not open"
            if case .failed(_, let message) = document.state { why = message }
            FileHandle.standardError.write(Data("Kurven: could not open \(bundle.path) — \(why)\n".utf8))
            exit(1)
        }
        for (name, count) in levels {
            guard let index = document.layers.firstIndex(where: { $0.spec.name == name }),
                  document.isDerived(layer: index) else {
                FileHandle.standardError.write(Data("Kurven: no layer '\(name)' with levels to set\n".utf8))
                exit(1)
            }
            document.setLevelCount(Int(count), forLayer: index)
            while document.drawing { try? await Task.sleep(for: .milliseconds(20)) }
        }
        for (name, count) in strokes {
            guard let index = document.layers.firstIndex(where: { $0.spec.name == name }),
                  document.hatchCount(ofLayer: index) != nil else {
                FileHandle.standardError.write(Data("Kurven: no hatching '\(name)' with strokes to set\n".utf8))
                exit(1)
            }
            document.setHatchCount(Int(count), ofLayer: index)
            while document.drawing { try? await Task.sleep(for: .milliseconds(20)) }
        }
        await headlessEdit(edits)
        while document.drawing { try? await Task.sleep(for: .milliseconds(20)) }
        print("Kurven: \(document.title) — \(document.layers.count) layers, "
              + "\(document.layers.reduce(0) { $0 + $1.paths.count }) paths")
        if let save { headlessSave(to: save) }
        if let bake { await headlessBake(to: bake, resolution: resolution) }
        exit(0)
    }

    /// Open a bundle, draw a frame, and pick at each pixel as a click on it
    /// does; say what was picked, and draw the last pick if asked.
    private func headlessPick(_ bundle: URL, at pixels: [SIMD2<Double>], atOnce: Bool, screenshot: URL?) async {
        await document.load(bundle)
        guard let scene = document.framedScene, let navigator = document.navigator else {
            var why = "it did not open"
            if case .failed(_, let message) = document.state { why = message }
            FileHandle.standardError.write(Data("Kurven: could not open \(bundle.path) — \(why)\n".utf8))
            exit(1)
        }
        do {
            let renderer = try MetalRenderer()
            let target = try renderer.makePreviewTarget(document.viewport)
            try renderer.renderPreview(scene, navigator: navigator, viewport: document.viewport,
                                       options: document.previewOptions, into: target)
            for pixel in pixels {
                document.pick(at: renderer.worldPoint(atPixel: pixel, navigator: navigator, viewport: document.viewport))
                if atOnce { continue }
                while document.picking { try? await Task.sleep(for: .milliseconds(20)) }
                print("Kurven: \(document.pickStatus ?? "no pick")")
            }
            if atOnce {
                while document.picking { try? await Task.sleep(for: .milliseconds(20)) }
                // Long enough for a boundary a later click left behind to land.
                try? await Task.sleep(for: .milliseconds(500))
                print("Kurven: \(document.pickStatus ?? "no pick")")
            }
        } catch {
            FileHandle.standardError.write(Data("Kurven: \(error)\n".utf8))
            exit(1)
        }
        if let screenshot { renderScreenshot(to: screenshot, what: bundle.lastPathComponent) }
        exit(0)
    }

    /// Open a bundle and paint it as the "Curves on" section does, a step
    /// at a time in the order given, each edit one undo step of the
    /// document's own undo manager.
    private func headlessPaint(_ bundle: URL, args: [String]) async {
        func fail(_ why: String) -> Never {
            FileHandle.standardError.write(Data("Kurven: \(why)\n".utf8))
            exit(1)
        }
        await document.load(bundle)
        guard document.scene != nil else {
            var why = "it did not open"
            if case .failed(_, let message) = document.state { why = message }
            fail("could not open \(bundle.path) — \(why)")
        }
        let undo = UndoManager()
        undo.groupsByEvent = false
        document.undoManager = undo
        func settle() async {
            repeat { try? await Task.sleep(for: .milliseconds(20)) }
            while document.building || document.drawing || document.picking || document.sampling
        }
        /// One edit, one undo step.
        func edit(_ body: () throws -> Void) async {
            undo.beginUndoGrouping()
            do { try body() } catch {
                undo.endUndoGrouping()
                fail("refused: \(error)")
            }
            undo.endUndoGrouping()
            await settle()
        }
        func pair(_ text: String) -> (String, String) {
            let parts = text.split(separator: "=", maxSplits: 1).map(String.init)
            guard parts.count == 2 else { fail("'\(text)' wants NAME=VALUE") }
            return (parts[0], parts[1])
        }
        func picked() -> FacePick {
            guard let pick = document.picked else { fail("nothing is picked to paint on: --pick first") }
            return pick
        }
        let resolution = args.firstIndex(of: "--resolution").flatMap { $0 + 1 < args.count ? Int(args[$0 + 1]) : nil }
        var renderer: MetalRenderer?
        var k = 0
        while k < args.count {
            let flag = args[k]
            guard flag.hasPrefix("--") else { k += 1; continue }
            if flag == "--undo" || flag == "--redo" {
                if flag == "--undo" { undo.undo() } else { undo.redo() }
                await settle()
                print("Kurven: \(flag.dropFirst(2)) — \(document.layers.count) layers")
                k += 1
                continue
            }
            guard k + 1 < args.count else { fail("\(flag) wants a value") }
            let value = args[k + 1]
            k += 2
            switch flag {
            case "--pick":
                let p = value.split(separator: ",").compactMap { Double($0) }
                guard p.count == 2, let scene = document.framedScene, let navigator = document.navigator else {
                    fail("--pick takes a pixel, PX,PY, of an open drawing")
                }
                do {
                    let r = try renderer ?? MetalRenderer()
                    renderer = r
                    let target = try r.makePreviewTarget(document.viewport)
                    try r.renderPreview(scene, navigator: navigator, viewport: document.viewport,
                                        options: document.previewOptions, into: target)
                    document.pick(at: r.worldPoint(atPixel: SIMD2(p[0], p[1]), navigator: navigator,
                                                   viewport: document.viewport))
                } catch { fail("\(error)") }
                await settle()
                print("Kurven: \(document.pickStatus ?? "no pick")")
            case "--function":
                guard k + 1 < args.count, args[k] == "--periods" else { fail("--function wants --periods P1,P2 after it") }
                let p = args[k + 1].split(separator: ",").compactMap { Double($0) }
                k += 2
                guard p.count == 2 else { fail("--periods wants the real and imaginary periods, as in 6.4,4.1") }
                await edit { try document.addFunction(value, periods: SIMD2(p[0], p[1])) }
                print("Kurven: function \(value) — \(document.surfaceStatus ?? "")")
            case "--add":
                let pick = picked()
                let choices = document.paintChoices(on: pick)
                guard let choice = choices.first(where: { $0.id == value }) else {
                    fail("nothing called '\(value)' to add on \(document.paintTitle(pick)); the Add menu has: "
                         + choices.map(\.id).joined(separator: ", "))
                }
                await edit { document.add(choice) }
                print("Kurven: added \(value) — \(document.layers.count) layers")
            case "--remove":
                let pick = picked()
                let on = document.paintedLayers(on: pick)
                guard let index = on.first(where: { document.paint?.layers[$0].name == value }) else {
                    fail("no layer '\(value)' on \(document.paintTitle(pick)); it has: "
                         + on.compactMap { document.paint?.layers[$0].name }.joined(separator: ", "))
                }
                await edit { try document.removeLayer(at: index) }
                print("Kurven: removed \(value) — \(document.layers.count) layers")
            case "--levels", "--strokes":
                let (name, text) = pair(value)
                guard let count = Int(text) else { fail("\(flag) wants LAYER=COUNT") }
                // A drag: taken hold of, moved, let go.
                await edit {
                    document.beginPaintDrag()
                    if flag == "--levels" {
                        document.paintLevels(count, ofLayer: name, draft: true)
                        document.rebuildCurves()
                        document.endPaintDrag("Levels of \(name)")
                    } else {
                        document.paintStrokes(count, ofLayer: name)
                        document.endPaintDrag("Strokes of \(name)")
                    }
                }
                print("Kurven: \(flag.dropFirst(2)) of \(name) = \(count)")
            case "--axis":
                guard let axis = DomainAxis(rawValue: value) else { fail("--axis wants real or imag") }
                await edit { document.paintAxis(axis) }
            case "--save":
                headlessSave(to: URL(fileURLWithPath: value))
            case "--bake":
                await headlessBake(to: URL(fileURLWithPath: value), resolution: resolution)
            case "--screenshot":
                renderScreenshot(to: URL(fileURLWithPath: value), what: bundle.lastPathComponent)
            case "--inspector":
                await renderPaintSection(to: URL(fileURLWithPath: value))
            case "--dump":
                let (name, prefix) = pair(value)
                guard let layer = document.layers.first(where: { $0.spec.name == name }) else {
                    fail("no layer '\(name)' to dump")
                }
                let paths = layer.paths
                do {
                    try NPY.write(paths.vertices.flatMap { [$0.x, $0.y, $0.z] }, shape: [paths.vertices.count, 3],
                                  to: URL(fileURLWithPath: prefix + ".npy"))
                    try NPY.write(paths.offsets, shape: [paths.offsets.count],
                                  to: URL(fileURLWithPath: prefix + ".idx.npy"))
                } catch { fail("\(error)") }
                print("Kurven: dumped \(name), \(paths.count) paths")
            case "--resolution":
                break
            default:
                fail("\(flag) is not a painting step")
            }
        }
        exit(0)
    }

    /// The "Curves on" section as the sidebar draws it, into a PNG.
    private func renderPaintSection(to url: URL) async {
        guard let pick = document.picked else {
            FileHandle.standardError.write(Data("Kurven: nothing is picked\n".utf8))
            exit(1)
        }
        let view = Form {
            Section("Curves on \(document.paintTitle(pick))") { PaintSection(document: document, pick: pick) }
        }
        .formStyle(.grouped)
        .frame(width: 320, height: 640)
        .environment(\.colorScheme, .light)
        let host = NSHostingView(rootView: view)
        let window = NSWindow(contentRect: NSRect(x: -4000, y: -4000, width: 320, height: 640),
                              styleMask: [.borderless], backing: .buffered, defer: false)
        window.appearance = NSAppearance(named: .aqua)
        window.contentView = host
        // Never ordered in, so never closed: the last window closing would
        // end the run (`applicationShouldTerminateAfterLastWindowClosed`).
        host.layoutSubtreeIfNeeded()
        try? await Task.sleep(for: .milliseconds(400))
        guard let rep = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
            FileHandle.standardError.write(Data("Kurven: no bitmap for the section\n".utf8))
            exit(1)
        }
        host.cacheDisplay(in: host.bounds, to: rep)
        do {
            try rep.representation(using: .png, properties: [:])?.write(to: url)
        } catch {
            FileHandle.standardError.write(Data("Kurven: \(error)\n".utf8))
            exit(1)
        }
        print("Kurven: the section -> \(url.lastPathComponent)")
    }

    private func screenshot(bundle: URL, to output: URL) async {
        await document.load(bundle)
        renderScreenshot(to: output, what: bundle.lastPathComponent)
    }

    /// One frame of whatever the document currently holds.
    ///
    /// Separate from loading it, because a landscape has no file to re-open:
    /// the bundle it was sampled into is scratch and is deleted the moment it
    /// has been read. What is on screen is the value in memory.
    private func renderScreenshot(to output: URL, what: String) {
        guard let scene = document.shownScene, let navigator = document.navigator else {
            FileHandle.standardError.write(Data("Kurven: nothing to draw\n".utf8))
            exit(1)
        }
        do {
            let renderer = try MetalRenderer()
            let target = try renderer.makePreviewTarget(document.viewport)
            try renderer.renderPreview(scene, navigator: navigator,
                                       viewport: document.viewport,
                                       options: document.shownOptions, into: target)
            try PNG.write(target, to: output)
            print("Kurven: \(what) -> \(output.lastPathComponent) "
                  + "(\(document.viewport.width)x\(document.viewport.height), "
                  + "\(document.layers.count) layers)")
        } catch {
            FileHandle.standardError.write(Data("Kurven: \(error)\n".utf8))
            exit(1)
        }
    }

    func application(_ application: NSApplication, open urls: [URL]) {
        if let url = urls.first { document.open(url) }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func openPanel() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.treatsFilePackagesAsDirectories = false
        panel.message = "Choose a .kurven bundle"
        if panel.runModal() == .OK, let url = panel.url { document.open(url) }
    }
}

struct DocumentWindow: View {
    @Bindable var document: Document
    let thumbnails: Thumbnails
    @Environment(\.undoManager) private var undoManager
    // Handed in rather than found through `NSApp.delegate`: under
    // `@NSApplicationDelegateAdaptor` that is SwiftUI's delegate, not ours.
    let open: () -> Void

    var body: some View {
        HSplitView {
            ZStack {
                MetalView(document: document)
                overlay
            }
            .frame(minWidth: 480)
            Inspector(document: document)
                .frame(width: 320)
        }
        .navigationTitle(document.title)
        .sheet(isPresented: $document.browsing) {
            Gallery(functions: document.catalog.presets, surfaces: SurfacePreset.catalog,
                    thumbnails: thumbnails, choose: document.create(_:),
                    dismiss: { document.browsing = false })
        }
        .toolbar {
            ToolbarItem(placement: .status) { StatusView(document: document) }
        }
        // Painting registers its edits with the window's undo manager, as
        // the sections do.
        .onAppear { document.undoManager = undoManager }
        .onChange(of: undoManager) { document.undoManager = undoManager }
    }

    @ViewBuilder
    private var overlay: some View {
        switch document.state {
        case .empty where document.building:
            // A first surface takes a moment -- the forced torus is fitted
            // from a long run -- and the gallery staying up would read as the
            // click having missed.
            VStack(spacing: 8) {
                ProgressView()
                Text(document.surfaceStatus ?? "building…")
                    .font(.caption).foregroundStyle(.secondary)
            }
        case .empty:
            VStack(spacing: 0) {
                Gallery(functions: document.catalog.presets, surfaces: SurfacePreset.catalog,
                        thumbnails: thumbnails, choose: document.create(_:))
                Divider()
                HStack {
                    Text("…or open a bundle someone already made.")
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button("Open…", action: open)
                }
                .padding(.horizontal, 20).padding(.vertical, 10)
            }
            .background(.background)
        case .loading:
            ProgressView()
        case .failed(let url, let message):
            VStack(spacing: 6) {
                Label("Could not open \(url.lastPathComponent)",
                      systemImage: "exclamationmark.triangle")
                Text(message).font(.caption).foregroundStyle(.secondary)
                    .frame(maxWidth: 420)
            }
        case .ready:
            EmptyView()
        }
    }
}

/// The toolbar's status: what is open, where the camera is, and how long
/// frames take.
///
/// Scoped apart from `DocumentWindow` because it reads `navigator`, which
/// every drag writes, and the window's body holds the Metal view and the
/// sidebar -- neither of which a new azimuth changes.
struct StatusView: View {
    let document: Document

    var body: some View {
        // The toolbar draws its own capsule tight around whatever this
        // returns, so the padding has to come from in here: without it the
        // first glyph of "az" and the last of "13 ms" sit on the rounded edge.
        Group {
            switch document.state {
            case .empty:
                Text("No bundle open").foregroundStyle(.secondary)
            case .loading(let url):
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("Reading \(url.lastPathComponent)…")
                }
            case .ready:
                HStack(spacing: 14) {
                    if document.building {
                        HStack(spacing: 6) {
                            ProgressView().controlSize(.small)
                            Text(document.surfaceStatus ?? "building…")
                        }
                    }
                    if let n = document.navigator {
                        Text(String(format: "az %.1f°  el %.1f°", n.orbit.azimuth.degrees,
                                    n.orbit.elevation.degrees))
                    }
                    if let frames = document.frames.summary { Text(frames) }
                }
                .monospacedDigit().foregroundStyle(.secondary)
            case .failed(_, let message):
                Label(message, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red).lineLimit(1)
            }
        }
        .padding(.horizontal, 10)
        .fixedSize()
    }
}
