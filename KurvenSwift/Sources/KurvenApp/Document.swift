import Foundation
import Observation
import KurvenCore
import KurvenMetal
import KurvenBake

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

    var scene: Scene?
    var navigator: Navigator?
    var viewport = Viewport(width: 1200, height: 800)

    var mode: PreviewMode = .plate
    /// Layer indices that are drawn. Absent means all of them.
    var hiddenLayers: Set<Int> = []
    var margin: Double = 0.02

    var bakeResolution: Int = 4000
    var bakeStatus: String?
    private(set) var baking = false

    var url: URL? {
        switch state {
        case .empty: nil
        case .loading(let u): u
        case .ready(let b): b.url
        case .failed(let u, _): u
        }
    }

    var title: String { url?.deletingPathExtension().lastPathComponent ?? "Kurven" }

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

    private func adopt(_ bundle: KurvenBundle) {
        state = .ready(bundle)
        guard let preset = bundle.manifest.presets.first else {
            state = .failed(bundle.url, "the bundle declares no camera presets")
            return
        }
        var scene = Scene(bundle: bundle, preset: preset)
        margin = preset.margin
        bakeResolution = preset.buffer
        hiddenLayers = []

        var navigator = Navigator(orbit: Orbit(matching: preset.plate),
                                  framing: Framing(center: P2(0, 0), unitsPerPixel: 1))
        scene.camera = navigator.camera
        if let bounds = scene.quickBounds() {
            navigator.framing = .fitting(bounds, in: viewport)
        }
        self.scene = scene
        self.navigator = navigator
    }

    // MARK: - navigation

    func apply(_ gesture: Gesture) {
        guard var navigator, var scene else { return }
        navigator = navigator.applying(gesture, in: viewport)
        scene.camera = navigator.camera
        scene.margin = margin
        self.navigator = navigator
        self.scene = scene
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

    /// Re-derive the camera after a change that is not a gesture (the margin
    /// slider, a viewport resize).
    func refreshScene() {
        guard var scene, let navigator else { return }
        scene.camera = navigator.camera
        scene.margin = margin
        self.scene = scene
    }

    func toggle(layer index: Int) {
        if hiddenLayers.contains(index) { hiddenLayers.remove(index) }
        else { hiddenLayers.insert(index) }
    }

    // MARK: - baking

    /// Bake in the background, over the `Scene` value the window is showing.
    ///
    /// The scene is `Sendable` and the renderer is not, so the task builds its
    /// own -- which is also what keeps the bake off the queue the preview is
    /// drawing on. The result is the same function `kurven-cli bake` calls, on
    /// the same value, so the two produce the same picture by construction
    /// rather than by keeping two paths in step.
    func bake(to url: URL) {
        guard let scene, !baking else { return }
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
