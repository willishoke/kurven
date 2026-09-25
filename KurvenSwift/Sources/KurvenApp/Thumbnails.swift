import AppKit
import Foundation
import Observation
import KurvenCore
import KurvenMetal
import KurvenBake
import KurvenLandscape
import KurvenDynamics

/// A picture of each function and surface in the catalogs, drawn by the app
/// that draws the real thing.
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
    nonisolated static let size = Viewport(width: 440, height: 300)
    static let samples = 220
    /// Bumped when the *styling* a landscape is given by default changes --
    /// which levels, what is drawn on a truncated top -- since that changes the
    /// picture without changing anything else the key is made of.
    static let style = 2

    /// Draw whatever is missing, one at a time, in catalog order.
    ///
    /// One at a time so that a person's own click is never behind fourteen
    /// thumbnails' worth of sampling.
    func warm(_ entries: [GalleryEntry]) {
        guard !running else { return }
        running = true
        Task {
            for entry in entries where images[entry.id] == nil && !failed.contains(entry.id) {
                if let cached = NSImage(contentsOf: Thumbnails.cache(for: entry)) {
                    images[entry.id] = cached
                    continue
                }
                drawing = entry.name
                do {
                    switch entry {
                    case .function(let preset): images[entry.id] = try await draw(preset)
                    case .surface(let preset): images[entry.id] = try await draw(preset)
                    }
                } catch {
                    // A plate that will not build is not worth a retry loop on
                    // every appearance of the picker; the cell says so.
                    failed.insert(entry.id)
                }
                drawing = nil
            }
            running = false
        }
    }

    func isFailed(_ entry: GalleryEntry) -> Bool { failed.contains(entry.id) }

    /// A surface, built from its own request at a coarser lattice -- a
    /// thumbnail is a tenth the width of the window -- and drawn from the
    /// camera it opens at. The forced torus is still fitted in full, which
    /// is most of its half second; that is why this stays off the main actor.
    private func draw(_ preset: SurfacePreset) async throws -> NSImage? {
        let request = Thumbnails.request(for: preset)
        let destination = Thumbnails.cache(for: .surface(preset))
        let viewport = Thumbnails.size
        try await Task.detached(priority: .utility) {
            let plate = try SurfacePlate.build(request)
            let orbit = Orbit(matching: SurfaceRequest.projection)
            let scene = plate.scene(camera: orbit.camera)
            guard let bounds = scene.quickBounds() else { return }
            let navigator = Navigator(orbit: orbit, framing: .fitting(bounds, in: viewport))
            try Thumbnails.render(scene, navigator: navigator, to: destination)
        }.value
        return NSImage(contentsOf: destination)
    }

    /// What a surface's thumbnail is built from: its own request, coarser.
    nonisolated static func request(for preset: SurfacePreset) -> SurfaceRequest {
        var request = preset.request
        request.lattice = Thumbnails.lattice
        // A trajectory as long as the window's, at a quarter the width, is
        // grey moiré: a quarter of it keeps its windings as far apart.
        if case .forced(let system, let fit, let harmonics, let radial, let axial, let radius,
                        let turns) = request.shape {
            request.shape = .forced(system: system, fit: fit, harmonics: harmonics,
                                    radial: radial, axial: axial, radius: radius,
                                    turns: max(turns / 4, 1))
        }
        return request
    }

    /// Lattice points around a surface's `u` for its thumbnail.
    nonisolated static let lattice = 256

    /// One preview frame of `scene`, written where the cache keeps it.
    nonisolated private static func render(_ scene: Scene, navigator: Navigator,
                                           to destination: URL) throws {
        let renderer = try MetalRenderer()
        let target = try renderer.makePreviewTarget(size)
        try renderer.renderPreview(scene, navigator: navigator, viewport: size,
                                   options: PreviewOptions(
                                    mode: .plate, visibleLayers: Set(scene.layers.indices)),
                                   into: target)
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try PNG.write(target, to: destination)
    }

    private func draw(_ preset: FunctionPreset) async throws -> NSImage? {
        var request = LandscapeRequest(preset: preset, resolution: Thumbnails.samples)
        // About twenty strokes along the longest edge. The default is right for
        // a bake and a black rectangle here.
        request.spacing = max(abs(preset.domain.real.length),
                              abs(preset.domain.imag.length)) / 22
        let destination = Thumbnails.cache(for: .function(preset))
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
            try Thumbnails.render(scene, navigator: navigator, to: destination)
        }.value
        return NSImage(contentsOf: destination)
    }

    /// Where an entry's picture lives, keyed by everything that would change it.
    ///
    /// Not `hashValue`: Swift seeds it per process, so a cache keyed on one is
    /// a cache that misses every launch and grows forever.
    private static func cache(for entry: GalleryEntry) -> URL {
        let key: String
        switch entry {
        case .function(let preset):
            let d = preset.domain
            let cap: String = preset.cap.map { "\($0)" } ?? "none"
            let window = "\(d.real.lo),\(d.real.hi),\(d.imag.lo),\(d.imag.hi)"
            let shape = "\(size.width)x\(size.height)@\(samples)"
            key = preset.expression + "|" + window + "|" + cap + "|" + shape
                + "|style\(style)"
        case .surface(let preset):
            // The request spelled out is every number the plate is built from.
            key = "surface|\(request(for: preset))|\(size.width)x\(size.height)|style\(style)"
        }
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in key.utf8 {
            hash = (hash ^ UInt64(byte)) &* 0x100000001b3
        }
        let root = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return root.appendingPathComponent("world.kurven/thumbnails", isDirectory: true)
            .appendingPathComponent(String(format: "%@-%016llx.png",
                                               entry.id.replacingOccurrences(of: ":", with: "-"),
                                               hash))
    }
}
