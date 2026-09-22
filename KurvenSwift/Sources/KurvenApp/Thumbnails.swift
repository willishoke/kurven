import AppKit
import Foundation
import Observation
import KurvenCore
import KurvenMetal
import KurvenBake
import KurvenLandscape

/// A picture of each function in the catalog, drawn by the app that draws the
/// real thing.
///
/// A dropdown of fourteen names asks you to already know what ζ looks like
/// against tan. The alternative is not an icon set someone drew: it is the
/// landscape itself, sampled small and rendered through the same preview path
/// the window uses, so a thumbnail cannot show something the app would not.
///
/// Three things make it cheap enough to do on the way into a picker:
///
///   - **small samples.** 220 along the long side is a tenth of the work of the
///     real thing and, at 200 points across, indistinguishable from it.
///   - **coarser hatching.** The spacing rule puts about seventy strokes along
///     an edge, which is right for a four-thousand-pixel bake and solid black
///     at this size. A thumbnail asks for about twenty.
///   - **a disk cache.** The second look at the picker costs nothing, and the
///     key includes the expression, the window and the cap, so a catalog that
///     changes redraws itself rather than going stale.
@MainActor
@Observable
final class Thumbnails {
    private(set) var images: [String: NSImage] = [:]
    /// Which preset is being drawn, for the placeholder to say so.
    private(set) var drawing: String?
    private var failed: Set<String> = []
    private var running = false

    /// The size they are drawn at. Twice what the grid shows, so the ink is
    /// halved rather than aliased -- the preview draws one-pixel lines, and a
    /// one-pixel line shown at half scale is what a thumbnail is allowed to do
    /// about antialiasing without touching the renderer.
    static let size = Viewport(width: 440, height: 300)
    static let samples = 220
    /// Bumped when the *styling* a landscape is given by default changes --
    /// which levels, what is drawn on a truncated top -- since that changes the
    /// picture without changing anything else the key is made of.
    static let style = 2

    /// Draw whatever is missing, one at a time, in catalog order.
    ///
    /// One at a time so that a person's own click is never behind fourteen
    /// thumbnails' worth of sampling.
    func warm(_ presets: [FunctionPreset]) {
        guard !running else { return }
        running = true
        Task {
            for preset in presets where images[preset.name] == nil
                && !failed.contains(preset.name) {
                if let cached = NSImage(contentsOf: Thumbnails.cache(for: preset)) {
                    images[preset.name] = cached
                    continue
                }
                drawing = preset.name
                do {
                    images[preset.name] = try await draw(preset)
                } catch {
                    // A function that will not sample is not worth a retry loop
                    // on every appearance of the picker; the cell says so.
                    failed.insert(preset.name)
                }
                drawing = nil
            }
            running = false
        }
    }

    func isFailed(_ preset: FunctionPreset) -> Bool { failed.contains(preset.name) }

    private func draw(_ preset: FunctionPreset) async throws -> NSImage? {
        var request = LandscapeRequest(preset: preset, resolution: Thumbnails.samples)
        // About twenty strokes along the longest edge. The default is right for
        // a bake and a black rectangle here.
        request.spacing = max(abs(preset.domain.real.length),
                              abs(preset.domain.imag.length)) / 22
        let destination = Thumbnails.cache(for: preset)
        let viewport = Thumbnails.size
        try await Task.detached(priority: .utility) {
            let bundle = try NativeLandscape.build(request)
            guard let plate = bundle.manifest.presets.first else { return }
            var scene = Scene(bundle: bundle, preset: plate)
            var navigator = Navigator(orbit: Orbit(matching: plate.plate),
                                      framing: Framing(center: P2(0, 0), unitsPerPixel: 1))
            scene.camera = navigator.camera
            // The exact bounds, not the subsampled ones the window fits with:
            // a pole spire is a few samples wide, so a coarse subsample can
            // miss its tip and frame the picture with the spires cut off. At
            // this size the exact fold is over thirteen thousand points.
            if let bounds = scene.viewBounds() ?? scene.quickBounds() {
                navigator.framing = .fitting(bounds, in: viewport)
                scene.camera = navigator.camera
            }
            let renderer = try MetalRenderer()
            let target = try renderer.makePreviewTarget(viewport)
            try renderer.renderPreview(scene, navigator: navigator, viewport: viewport,
                                       options: PreviewOptions(
                                        mode: .plate,
                                        visibleLayers: Set(scene.layers.indices)),
                                       into: target)
            try FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try PNG.write(target, to: destination)
        }.value
        return NSImage(contentsOf: destination)
    }

    /// Where a preset's picture lives, keyed by everything that would change it.
    ///
    /// Not `hashValue`: Swift seeds it per process, so a cache keyed on one is
    /// a cache that misses every launch and grows forever.
    private static func cache(for preset: FunctionPreset) -> URL {
        let d = preset.domain
        let cap: String = preset.cap.map { "\($0)" } ?? "none"
        let window = "\(d.real.lo),\(d.real.hi),\(d.imag.lo),\(d.imag.hi)"
        let shape = "\(size.width)x\(size.height)@\(samples)"
        let key = preset.expression + "|" + window + "|" + cap + "|" + shape
            + "|style\(style)"
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in key.utf8 {
            hash = (hash ^ UInt64(byte)) &* 0x100000001b3
        }
        let root = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return root.appendingPathComponent("world.kurven/thumbnails", isDirectory: true)
            .appendingPathComponent(String(format: "%@-%016llx.png", preset.name, hash))
    }
}
