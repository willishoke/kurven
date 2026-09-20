import SwiftUI
import AppKit
import UniformTypeIdentifiers
import KurvenCore
import KurvenMetal
import KurvenBake

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
            DocumentWindow(document: delegate.document, open: delegate.openPanel)
                .frame(minWidth: 900, minHeight: 600)
        }
        .commands {
            CommandGroup(replacing: .newItem) {
                // A landscape is the new document here: the app no longer needs
                // a bundle someone made earlier in order to show anything.
                Button("New Landscape") { delegate.newLandscape() }
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
        let bundlePath = args.first { !$0.hasPrefix("-") && $0.hasSuffix(".kurven") }

        // `--screenshot PATH` renders one frame through the app's own state --
        // its Document, its Navigator, its preview options -- and exits. Without
        // it the only way to know the window draws the right thing is to look at
        // it, and a test that requires someone to look at it is not one.
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
            Task { await headlessLandscape(what, resolution: resolution,
                                           screenshot: shot, save: save) }
            return
        }
        if let path = bundlePath { document.open(URL(fileURLWithPath: path)) }
    }

    /// Wait for the catalog, make the landscape, and report it. The window is
    /// never shown; everything it would do is done through the same model.
    private func headlessLandscape(_ what: String, resolution: Int?,
                                   screenshot: URL?, save: URL?) async {
        for _ in 0..<400 where document.catalog == nil && document.serviceStatus == nil {
            try? await Task.sleep(for: .milliseconds(25))
        }
        guard let catalog = document.catalog else {
            let why = document.serviceStatus ?? "it never answered"
            FileHandle.standardError.write(Data("Kurven: no service — \(why)\n".utf8))
            exit(1)
        }
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

    func newLandscape() {
        document.connectService()
        Task {
            for _ in 0..<400 where document.catalog == nil && document.serviceStatus == nil {
                try? await Task.sleep(for: .milliseconds(25))
            }
            if let first = document.catalog?.presets.first { document.create(first) }
        }
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
    // Handed in rather than found through `NSApp.delegate`: under
    // `@NSApplicationDelegateAdaptor` that is SwiftUI's delegate, not ours.
    let open: () -> Void
    @State private var redraws = 0

    var body: some View {
        HSplitView {
            ZStack {
                MetalView(document: document)
                    .id(redraws == Int.max ? 1 : 0)   // never re-created
                overlay
            }
            .frame(minWidth: 480)
            Inspector(document: document) { redraws &+= 1 }
                .frame(width: 320)
        }
        .navigationTitle(document.title)
        .toolbar {
            ToolbarItem(placement: .status) { StatusView(document: document) }
        }
    }

    @ViewBuilder
    private var overlay: some View {
        switch document.state {
        case .empty:
            VStack(spacing: 8) {
                Text("kurven").font(.largeTitle)
                Text("Pick a function in the sidebar, or open a .kurven bundle.")
                    .foregroundStyle(.secondary)
                HStack {
                    Button("New Landscape") {
                        if let first = document.catalog?.presets.first {
                            document.create(first)
                        }
                    }
                    .disabled(document.catalog == nil)
                    Button("Open…", action: open)
                }
                if document.catalog == nil, let status = document.serviceStatus {
                    Text(status).font(.caption).foregroundStyle(.secondary)
                        .frame(maxWidth: 420)
                }
            }
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
        switch document.state {
        case .empty:
            Text("No bundle open").foregroundStyle(.secondary)
        case .loading(let url):
            HStack {
                ProgressView().controlSize(.small)
                Text("Reading \(url.lastPathComponent)…")
            }
        case .ready:
            HStack(spacing: 14) {
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
}
