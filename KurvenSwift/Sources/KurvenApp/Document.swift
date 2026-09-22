import Foundation
import Observation
import KurvenCore
import KurvenMetal
import KurvenBake
import KurvenService
import KurvenLandscape

/// One open bundle, and everything the window knows about it.
///
/// Every mutation is a whole new value assigned into a stored property, so the
/// model holds no derived state that can go stale. In particular `scene` keeps
/// its `ContentID` across a camera change, which is what lets the renderer's
/// GPU resources survive navigation; `navigator` is the only thing a drag
/// touches.
@MainActor
@Observable
final class Document {
    enum State {
        case empty
        case loading(URL)
        case ready(KurvenBundle)
        case failed(URL, String)
    }

    private(set) var state: State = .empty

    /// The landscape and its ink. Its camera and margin are whatever they were
    /// when it was stored, and nothing reads them: `framedScene` supplies both.
    var scene: Scene?
    var navigator: Navigator?
    var viewport = Viewport(width: 1200, height: 800)

    /// Frame timing, for the status bar. Not state of the document, but the
    /// window's views already hold one of these and nothing else.
    let frames = FrameClock()

    /// The scene as the window shows it: the stored landscape, through the
    /// navigator's camera, clipped at the current margin.
    ///
    /// Assembled on read so a drag writes one property rather than two. The
    /// sidebar lists `scene`'s layers, so writing `scene` on every mouse event
    /// re-evaluated the whole sidebar on every mouse event, to show the same
    /// list of layers.
    var framedScene: Scene? {
        guard var scene, let navigator else { return nil }
        scene.camera = navigator.camera
        scene.margin = margin
        return scene
    }

    var mode: PreviewMode = .plate
    /// Layer indices that are drawn. Absent means all of them.
    var hiddenLayers: Set<Int> = []
    var margin: Double = 0.02

    /// Per-layer level-count overrides, for the layers a bundle *described*
    /// rather than dumped. Absent means the bundle's own levels.
    var levelCounts: [Int: Int] = [:]

    // MARK: - the landscape, when the document is one
    //
    // The state `Landscape.swift` works on. A landscape is a choice rather than
    // a file, so what a document holds is the request that produced it, the
    // request the controls have been edited to, and which of those is on screen.

    /// The menu of functions. Native: the same list the Python side reports,
    /// held to it by `kurven-test`, so a landscape needs no service at all.
    let catalog = Catalog.native
    /// What the controls say. Nil when this document is a bundle someone made
    /// earlier rather than a landscape being chosen.
    var landscape: LandscapeRequest?
    /// The request the open bundle actually came from.
    var shown: LandscapeRequest?
    /// The latest edit, waiting for the request in flight to finish, and
    /// whether it should re-frame when it arrives.
    var wanted: (request: LandscapeRequest, framing: Bool)?
    var sampling = false
    var landscapeStatus: String?
    /// True while the picker is up. On the document because the menu command
    /// and the window are on opposite sides of SwiftUI's scene boundary, and
    /// this is the one thing they both know about.
    var browsing = false
    /// The resolution a landscape is sampled at while a control is being
    /// dragged. Low enough that the picture keeps up with the hand; the full
    /// resolution follows when the drag ends.
    var draftResolution = 240

    var bakeResolution: Int = 4000
    var bakeStatus: String?
    private(set) var baking = false

    // MARK: - the service

    /// The Python half, when one could be found. Absent is a normal state: a
    /// bundle is a complete document without it, and everything but resampling
    /// works on a machine that has no repository checked out.
    private(set) var service: Service?
    private(set) var serviceDescription: Description?
    private(set) var serviceStatus: String?
    /// Argument values for the resample form, as text; absent means "the
    /// example's own default".
    var arguments: [String: String] = [:]
    private(set) var resampling = false

    var example: ExampleSpec? {
        guard let name = bundle?.manifest.provenance.example, !name.isEmpty else { return nil }
        return serviceDescription?.example(name)
    }

    var url: URL? {
        switch state {
        case .empty: nil
        case .loading(let u): u
        case .ready(let b): b.url
        case .failed(let u, _): u
        }
    }

    var title: String {
        if let landscape, isLandscape {
            return catalog.preset(landscape.name)?.label ?? landscape.expression
        }
        return url?.deletingPathExtension().lastPathComponent ?? "Kurven"
    }

    var bundle: KurvenBundle? {
        if case .ready(let b) = state { return b }
        return nil
    }

    var layers: [Layer] { scene?.layers ?? [] }

    var previewOptions: PreviewOptions {
        let visible = Set(layers.indices).subtracting(hiddenLayers)
        return PreviewOptions(mode: mode, visibleLayers: visible)
    }

    // MARK: - loading

    func open(_ url: URL) { Task { await load(url) } }

    /// The awaitable form, so a headless run can wait for the bundle rather
    /// than poll for it.
    func load(_ url: URL) async {
        state = .loading(url)
        scene = nil
        navigator = nil
        // Reading is megabytes of npy; it does not belong on the main actor
        // even though it is fast, because a stalled window during an open is
        // the difference between an app and a script with a window.
        let result = await Task.detached(priority: .userInitiated) { () -> Result<KurvenBundle, Error> in
            do { return .success(try KurvenBundle.read(at: url)) }
            catch { return .failure(error) }
        }.value
        switch result {
        case .success(let bundle): adopt(bundle)
        case .failure(let error): state = .failed(url, "\(error)")
        }
    }

    /// Take a freshly read bundle as the document.
    ///
    /// `keepingCamera` is for a resample: the landscape under the camera has
    /// changed, the camera has not. Re-fitting on every change would mean a
    /// window that jumps while a domain slider is dragged, which makes the
    /// slider useless for the thing it is for -- watching one feature as the
    /// window moves over it.
    /// Swap in a bundle derived from the same grids -- a restyling, not a new
    /// landscape (`Landscape.swift`). The only way `state` is written from
    /// outside this file, so "what is open" still changes in one place.
    func replace(bundle: KurvenBundle) { state = .ready(bundle) }

    /// Whether a landscape's contours are placed by its function rather than
    /// by its grid (`ContourRefiner`). A bundle read from disk records its
    /// expression, so it can be refined the same way a freshly sampled one is.
    var refineContours = true

    func adopt(_ bundle: KurvenBundle, keepingCamera: Bool = false) {
        var bundle = bundle
        if refineContours, bundle.refine == nil, let refiner = NativeLandscape.refiner(for: bundle) {
            bundle = bundle.refined(by: refiner.refine)
        }
        state = .ready(bundle)
        guard let preset = bundle.manifest.presets.first else {
            state = .failed(bundle.url, "the bundle declares no camera presets")
            return
        }
        var scene = Scene(bundle: bundle, preset: preset)
        derivedInk = [:]
        if keepingCamera, let navigator {
            scene.camera = navigator.camera
            self.scene = scene
            adoptLandscape(bundle)
            return
        }
        margin = preset.margin
        bakeResolution = preset.buffer
        hiddenLayers = []
        levelCounts = [:]

        var navigator = Navigator(orbit: Orbit(matching: preset.plate),
                                  framing: Framing(center: P2(0, 0), unitsPerPixel: 1))
        scene.camera = navigator.camera
        if let bounds = scene.quickBounds() {
            navigator.framing = .fitting(bounds, in: viewport)
        }
        self.scene = scene
        self.navigator = navigator
        adoptLandscape(bundle)
        connectService()
    }

    // MARK: - navigation

    func apply(_ gesture: Gesture) {
        guard let navigator else { return }
        self.navigator = navigator.applying(gesture, in: viewport)
    }

    func fit() {
        guard var scene, let navigator else { return }
        scene.camera = navigator.camera
        guard let bounds = scene.quickBounds() else { return }
        apply(.fit(bounds))
    }

    func use(preset: CameraPreset) {
        guard var scene, let navigator else { return }
        scene.camera = Orbit(matching: preset.plate, target: navigator.orbit.target).camera
        guard let bounds = scene.quickBounds() else { return }
        margin = preset.margin
        bakeResolution = preset.buffer
        apply(.preset(preset.plate, bounds))
    }

    func retarget(to world: P3<WorldSpace>) { apply(.retarget(world)) }

    func setPerspective(_ on: Bool) {
        guard var scene, let navigator else { return }
        scene.camera = navigator.camera
        guard let bounds = scene.quickBounds() else { return }
        apply(.project(fieldOfView: on ? Angle(degrees: 50) : nil, bounds))
    }

    /// The levels a described layer is currently drawn at.
    ///
    /// An override resamples the bundle's own level range uniformly. Which is a
    /// blunt instrument -- recip's magnitude levels are not an arithmetic
    /// sequence -- but it is the honest one for a slider: it says "more or
    /// fewer contours over the same range" and nothing about which particular
    /// values were interesting.
    func levels(forLayer index: Int) -> [Double]? {
        guard let bundle, index < bundle.manifest.layers.count,
              let base = KurvenBundle.levels(of: bundle.manifest.layers[index]),
              base.count >= 2 else { return nil }
        guard let n = levelCounts[index], n != base.count else { return base }
        let lo = base.min()!, hi = base.max()!
        guard n >= 2 else { return [(lo + hi) / 2] }
        return (0..<n).map { lo + (hi - lo) * Double($0) / Double(n - 1) }
    }

    func isDerived(layer index: Int) -> Bool {
        guard let bundle, index < bundle.manifest.layers.count else { return false }
        return KurvenBundle.levels(of: bundle.manifest.layers[index]) != nil
    }

    /// Re-derive the described layers at their current levels.
    ///
    /// `Scene.drawing` keeps the geometry's identity, so the renderer rebuilds
    /// the line buffer and leaves the height texture uploaded -- which is the
    /// difference between a slider that redraws in milliseconds and one that
    /// re-uploads a hundred megabytes per frame.
    func rebuildInk() {
        guard let bundle, let scene else { return }
        let layers = bundle.manifest.layers.enumerated().map { index, spec -> Layer in
            guard let levels = levels(forLayer: index) else {
                return (try? bundle.layer(spec.name)) ?? Layer(spec: spec, paths: .empty)
            }
            // Only the layer whose slider moved is re-derived. Dragging one
            // changes one level set; re-contouring all four costs four times as
            // much to produce three identical answers.
            if let cached = derivedInk[index], cached.levels == levels {
                return cached.layer
            }
            let layer = bundle.layer(spec, levels: levels)
            derivedInk[index] = (levels, layer)
            return layer
        }
        self.scene = scene.drawing(layers)
    }

    /// The last derivation of each described layer, keyed by the levels that
    /// produced it. Cleared whenever the landscape underneath it changes.
    var derivedInk: [Int: (levels: [Double], layer: Layer)] = [:]

    func setLevelCount(_ n: Int, forLayer index: Int) {
        levelCounts[index] = max(n, 1)
        rebuildInk()
    }

    func resetLevels(forLayer index: Int) {
        levelCounts.removeValue(forKey: index)
        rebuildInk()
    }

    func toggle(layer index: Int) {
        if hiddenLayers.contains(index) { hiddenLayers.remove(index) }
        else { hiddenLayers.insert(index) }
    }

    func connectService() {
        guard service == nil else { return }
        guard let command = Service.Command.autodetect(near: url) else {
            serviceStatus = "no kurven/serve.py found above this bundle; "
                + "set KURVEN_REPO to resample"
            return
        }
        do {
            let service = try Service(command: command)
            self.service = service
            Task {
                do {
                    serviceDescription = try await service.describe()
                    if let example, arguments.isEmpty {
                        // Start from what the bundle was actually made with, so
                        // the first resample is a change to one field rather
                        // than a different landscape.
                        for spec in example.arguments {
                            let recorded = bundle?.manifest.provenance.params[camel(spec.name)]
                            if let value = text(recorded) ?? spec.defaultText {
                                arguments[spec.name] = value
                            }
                        }
                    }
                    serviceStatus = nil
                } catch {
                    serviceStatus = "\(error)"
                }
            }
        } catch {
            serviceStatus = "\(error)"
        }
    }

    /// The manifest records provenance keys in camelCase (`rMin`); the service
    /// speaks argparse's snake_case (`r_min`). One place converts.
    private func camel(_ snake: String) -> String {
        let parts = snake.split(separator: "_")
        guard let first = parts.first else { return snake }
        return ([String(first)] + parts.dropFirst().map(\.capitalized)).joined()
    }

    private func text(_ value: JSONValue?) -> String? {
        switch value {
        case .some(.string(let s)): s
        case .some(.int(let i)): String(i)
        case .some(.double(let d)): String(d)
        case .some(.bool(let b)): b ? "true" : "false"
        default: nil
        }
    }

    /// Rebuild the landscape at the current arguments.
    ///
    /// The result is a new bundle written to the app's caches and opened in
    /// place of this one. Going through a file rather than streaming arrays
    /// back is the same choice the whole design makes: the answer is a value
    /// with a path, and everything downstream already knows how to read one.
    func resample() {
        guard let service, let name = bundle?.manifest.provenance.example,
              !name.isEmpty, !resampling else { return }
        let derived = bundle?.manifest.layers.contains { $0.files == nil } ?? false
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(name)-\(UInt32.random(in: 0...UInt32.max)).kurven")
        let settings = arguments
        resampling = true
        serviceStatus = "resampling…"
        Task {
            do {
                let result = try await service.export(example: name, to: output,
                                                      arguments: settings,
                                                      derived: derived)
                // Cleared only once the new bundle is open: anyone waiting on
                // `resampling` is waiting for a landscape, not for a file.
                await load(result.url)
                resampling = false
                serviceStatus = "\(String(format: "%.1f", Double(result.bytes) / 1e6)) MB"
            } catch {
                resampling = false
                serviceStatus = "\(error)"
            }
        }
    }

    // MARK: - baking

    /// Bake in the background, over the `Scene` value the window is showing.
    ///
    /// The scene is `Sendable` and the renderer is not, so the task builds its
    /// own -- which is also what keeps the bake off the queue the preview is
    /// drawing on. The result is the same function `kurven-cli bake` calls, on
    /// the same value, so the two produce the same picture by construction
    /// rather than by keeping two paths in step.
    var canBake: Bool { !(navigator?.orbit.isPerspective ?? false) }

    func bake(to url: URL) {
        guard let scene = framedScene, !baking else { return }
        guard canBake else {
            bakeStatus = "switch off perspective to bake: the plates are "
                + "orthographic, and a perspective bake has no oracle"
            return
        }
        let options = BakeOptions(resolution: bakeResolution, margin: margin)
        baking = true
        bakeStatus = "baking \(bakeResolution)²…"
        Task {
            let started = ContinuousClock().now
            let result = await Task.detached(priority: .userInitiated) { () -> Result<Int, Error> in
                do {
                    let renderer = try MetalRenderer()
                    let bake = try renderer.bake(scene, options: options)
                    try SVG.render(bake.strokes).write(to: url, atomically: true,
                                                       encoding: .utf8)
                    return .success(bake.strokes.pathCount)
                } catch { return .failure(error) }
            }.value
            baking = false
            switch result {
            case .success(let paths):
                bakeStatus = "\(paths) paths → \(url.lastPathComponent) "
                    + "in \(ContinuousClock().now - started)"
            case .failure(let error):
                bakeStatus = "bake failed: \(error)"
            }
        }
    }
}
