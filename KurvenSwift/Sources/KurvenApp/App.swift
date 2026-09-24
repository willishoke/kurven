import SwiftUI
import AppKit
import UniformTypeIdentifiers
import KurvenCore
import KurvenMetal
import KurvenBake
import KurvenDynamics

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
                // A landscape is the new document here: the app no longer needs
                // a bundle someone made earlier in order to show anything.
                Button("New Landscape…") { delegate.newLandscape() }
                    .keyboardShortcut("n")
                Button("Open…") { delegate.openPanel() }
                    .keyboardShortcut("o")
            }
            CommandGroup(after: .saveItem) {
                Button("Save Bundle…") { delegate.savePanel() }
                    .keyboardShortcut("s")
                    .disabled(delegate.document.bundle == nil)
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
        // The service is what makes a landscape possible, and with no bundle
        // open there is nothing else to trigger the search for it.
        document.connectService()
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
        // through the window's own Document and Service -- the same path the
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
            let cap = args.firstIndex(of: "--cap")
                .flatMap { $0 + 1 < args.endIndex ? Double(args[$0 + 1]) : nil }
            Task { await headlessLandscape(what, resolution: resolution, cap: cap,
                                           screenshot: shot, save: save) }
            return
        }
        // `--surface NAME [--screenshot PATH] [--bake PATH]` opens a surface
        // from the catalog through the window's own Document -- the gallery's
        // path -- so the window can be held to `kurven-cli surface NAME`:
        // the same preview frame, pixel for pixel, and the same bake. With
        // neither output it stays open, on that surface.
        if let i = args.firstIndex(of: "--surface"), i + 1 < args.endIndex {
            let name = args[i + 1]
            func path(_ flag: String) -> URL? {
                args.firstIndex(of: flag).flatMap {
                    $0 + 1 < args.endIndex ? URL(fileURLWithPath: args[$0 + 1]) : nil
                }
            }
            let resolution = args.firstIndex(of: "--resolution")
                .flatMap { $0 + 1 < args.endIndex ? Int(args[$0 + 1]) : nil }
            Task { await headlessSurface(name, screenshot: path("--screenshot"),
                                         bake: path("--bake"), resolution: resolution) }
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
        // by reading two code paths and believing they agree.
        if let i = args.firstIndex(of: "--bake"), i + 1 < args.endIndex,
           let path = bundlePath {
            let output = URL(fileURLWithPath: args[i + 1])
            let resolution = args.firstIndex(of: "--resolution")
                .flatMap { $0 + 1 < args.endIndex ? Int(args[$0 + 1]) : nil }
            Task { await headlessBake(bundle: URL(fileURLWithPath: path), to: output,
                                      resolution: resolution) }
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
    /// points, so every point the window gains is a point of plate. Zoomed
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
    private func headlessLandscape(_ what: String, resolution: Int?, cap: Double?,
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
        // the ink derived again from grids that are already here. Driving it
        // from here is how that path is checked without a pair of hands.
        if let cap { document.setCaps(.uniform(cap)) }
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

    /// Open a surface and report it, then draw or bake it and exit -- or, with
    /// nothing to write, show it.
    private func headlessSurface(_ name: String, screenshot: URL?, bake: URL?,
                                 resolution: Int?) async {
        guard let preset = SurfacePreset.named(name) else {
            FileHandle.standardError.write(Data(("Kurven: no surface '\(name)'; the catalog has: "
                + SurfacePreset.catalog.map(\.name).joined(separator: ", ") + "\n").utf8))
            exit(1)
        }
        document.create(surface: preset)
        while document.building { try? await Task.sleep(for: .milliseconds(25)) }
        guard document.plate != nil else {
            FileHandle.standardError.write(Data(
                "Kurven: could not build \(name) — \(document.surfaceStatus ?? "no reason given")\n".utf8))
            exit(1)
        }
        print("Kurven: \(document.title) — \(document.surfaceStatus ?? ""), "
              + "\(document.layers.count) layers, "
              + "\(document.layers.reduce(0) { $0 + $1.paths.count }) paths")
        guard screenshot != nil || bake != nil else { fillScreen(); return }
        if let screenshot { renderScreenshot(to: screenshot, what: document.title) }
        if let bake {
            if let resolution { document.bakeResolution = resolution }
            document.bake(to: bake)
            while document.baking { try? await Task.sleep(for: .milliseconds(20)) }
            print("Kurven: \(document.bakeStatus ?? "bake produced no status")")
            if document.bakeStatus?.hasPrefix("bake failed") == true { exit(1) }
        }
        exit(0)
    }

    /// Show the picker. Not "make a landscape at once": with fourteen
    /// functions in the catalog, the choice is the interesting part.
    private func warmThumbnails() async {
        let catalog = document.catalog
        let clock = ContinuousClock()
        let started = clock.now
        thumbnails.warm(catalog.presets)
        while thumbnails.images.count < catalog.presets.count {
            if catalog.presets.allSatisfy({ thumbnails.images[$0.name] != nil
                                            || thumbnails.isFailed($0) }) { break }
            try? await Task.sleep(for: .milliseconds(50))
        }
        for preset in catalog.presets {
            let mark = thumbnails.images[preset.name] != nil ? "drew" : "FAILED"
            print("  \(mark)  \(preset.name)")
        }
        print("Kurven: \(thumbnails.images.count) of \(catalog.presets.count) "
              + "thumbnails in \(clock.now - started)")
        exit(0)
    }

    func newLandscape() {
        document.browsing = true
    }

    func savePanel() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = (document.landscape?.name.isEmpty == false
                                      ? document.landscape!.name : "landscape") + ".kurven"
        panel.message = "Save this landscape as a .kurven bundle"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        document.saveBundle(to: url)
    }

    /// `--resample NAME=VALUE ... --screenshot PATH` drives the window's own
    /// resample: the same Document, the same Service, the same reload. Without
    /// it the only way to know the Rebuild button works is to press it.
    private func resample(bundle: URL, settings: [String: String],
                          screenshot: URL?) async {
        await document.load(bundle)
        document.connectService()
        // The description arrives on its own task; wait for it or for the
        // failure that explains why it will not.
        for _ in 0..<200 where document.serviceDescription == nil
            && document.serviceStatus == nil {
            try? await Task.sleep(for: .milliseconds(25))
        }
        guard document.example != nil else {
            let why = document.serviceStatus
                ?? (document.serviceDescription == nil
                    ? "the service never answered"
                    : "this bundle records no example to rebuild "
                      + "(provenance.example is empty)")
            FileHandle.standardError.write(Data("Kurven: cannot resample — \(why)\n".utf8))
            exit(1)
        }
        for (key, value) in settings { document.arguments[key] = value }
        let before = document.url
        document.resample()
        while document.resampling { try? await Task.sleep(for: .milliseconds(25)) }
        guard document.url != before, document.scene != nil else {
            var why = document.serviceStatus ?? "unknown"
            if case .failed(_, let message) = document.state { why = message }
            FileHandle.standardError.write(Data("Kurven: resample failed — \(why)\n".utf8))
            exit(1)
        }
        print("Kurven: resampled -> \(document.url?.lastPathComponent ?? "?") "
              + "(\(document.layers.count) layers)")
        if let screenshot { await self.screenshot(bundle: document.url!, to: screenshot) }
        exit(0)
    }

    private func headlessBake(bundle: URL, to output: URL, resolution: Int?) async {
        await document.load(bundle)
        guard document.scene != nil else {
            FileHandle.standardError.write(Data("Kurven: could not open \(bundle.path)\n".utf8))
            exit(1)
        }
        if let resolution { document.bakeResolution = resolution }
        document.bake(to: output)
        while document.baking { try? await Task.sleep(for: .milliseconds(20)) }
        print("Kurven: \(document.bakeStatus ?? "bake produced no status")")
        exit(document.bakeStatus?.hasPrefix("bake failed") == true ? 1 : 0)
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
        guard let scene = document.framedScene, let navigator = document.navigator else {
            FileHandle.standardError.write(Data("Kurven: nothing to draw\n".utf8))
            exit(1)
        }
        do {
            let renderer = try MetalRenderer()
            let target = try renderer.makePreviewTarget(document.viewport)
            try renderer.renderPreview(scene, navigator: navigator,
                                       viewport: document.viewport,
                                       options: document.previewOptions, into: target)
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
            Gallery(presets: document.catalog.presets, thumbnails: thumbnails,
                    choose: { document.create($0) },
                    dismiss: { document.browsing = false })
        }
        .toolbar {
            ToolbarItem(placement: .status) { StatusView(document: document) }
        }
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
                Gallery(presets: document.catalog.presets, thumbnails: thumbnails,
                        choose: { document.create($0) })
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
        case .ready, .surface:
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
            case .ready, .surface:
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
