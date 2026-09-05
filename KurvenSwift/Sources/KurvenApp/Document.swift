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

    /// Per-layer level-count overrides, for the layers a bundle *described*
    /// rather than dumped. Absent means the bundle's own levels.
    var levelCounts: [Int: Int] = [:]

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
    /// produced it.
    private var derivedInk: [Int: (levels: [Double], layer: Layer)] = [:]

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
