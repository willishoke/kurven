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
            DocumentWindow(document: delegate.document)
                .frame(minWidth: 900, minHeight: 600)
        }
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Open…") { delegate.openPanel() }
                    .keyboardShortcut("o")
            }
        }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let document = Document()

    func applicationDidFinishLaunching(_ notification: Notification) {
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
        if let path = bundlePath { document.open(URL(fileURLWithPath: path)) }
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
        guard let scene = document.scene, let navigator = document.navigator else {
            FileHandle.standardError.write(Data("Kurven: could not open \(bundle.path)\n".utf8))
            exit(1)
        }
        do {
            let renderer = try MetalRenderer()
            let target = try renderer.makePreviewTarget(document.viewport)
            try renderer.renderPreview(scene, navigator: navigator,
                                       viewport: document.viewport,
                                       options: document.previewOptions, into: target)
            try PNG.write(target, to: output)
            print("Kurven: \(bundle.lastPathComponent) -> \(output.lastPathComponent) "
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
            ToolbarItem(placement: .status) { status }
        }
    }

    @ViewBuilder
    private var status: some View {
        switch document.state {
        case .empty:
            Text("No bundle open").foregroundStyle(.secondary)
        case .loading(let url):
            HStack {
                ProgressView().controlSize(.small)
                Text("Reading \(url.lastPathComponent)…")
            }
        case .ready:
            if let n = document.navigator {
                Text(String(format: "az %.1f°  el %.1f°", n.orbit.azimuth.degrees,
                            n.orbit.elevation.degrees))
                    .monospacedDigit().foregroundStyle(.secondary)
            }
        case .failed(_, let message):
            Label(message, systemImage: "exclamationmark.triangle")
                .foregroundStyle(.red).lineLimit(1)
        }
    }

    @ViewBuilder
    private var overlay: some View {
        switch document.state {
        case .empty:
            VStack(spacing: 8) {
                Text("kurven").font(.largeTitle)
                Text("Open a .kurven bundle to navigate its landscape.")
                    .foregroundStyle(.secondary)
                Button("Open…") { (NSApp.delegate as? AppDelegate)?.openPanel() }
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
