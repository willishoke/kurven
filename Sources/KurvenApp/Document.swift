import Foundation
import Observation
import KurvenCore
import KurvenMetal
import KurvenBake
import KurvenLandscape
import KurvenDynamics

/// One open bundle or surface drawing, and everything the window knows about it.
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
        /// A parametric surface. Not a bundle, so every control that edits
        /// or saves one stays away from it.
        case surface(SurfaceDrawing)
        case failed(URL, String)
    }

    private(set) var state: State = .empty {
        didSet { version &+= 1 }
    }
    /// Bumped whenever `state` is written, so curves derived on another thread
    /// from one bundle is dropped if a different one is open when it lands.
    private(set) var version = 0

    /// The landscape and its curves. Its camera and depth tolerance are whatever they were
    /// when it was stored, and nothing reads them: `framedScene` supplies both.
    var scene: Scene?
    var navigator: Navigator?
    var viewport = Viewport(width: 1200, height: 800)

    /// Frame timing, for the status bar. Not state of the document, but the
    /// window's views already hold one of these and nothing else.
    let frames = FrameClock()

    /// The scene as the window shows it: the stored landscape, through the
    /// navigator's camera, clipped at the current depth tolerance.
    ///
    /// Assembled on read so a drag writes one property rather than two. The
    /// sidebar lists `scene`'s layers, so writing `scene` on every mouse event
    /// re-evaluated the whole sidebar on every mouse event, to show the same
    /// list of layers.
    var framedScene: Scene? {
        guard var scene, let navigator else { return nil }
        scene.camera = navigator.camera
        scene.depthTolerance = depthTolerance
        // A quarter of a drawable pixel: finer than any end of a curve can show.
        scene.resolution = 0.25 * navigator.framing.unitsPerPixel
        return scene
    }

    var mode: PreviewMode = .drawing
    /// Layer indices that are drawn. Absent means all of them.
    var hiddenLayers: Set<Int> = []
    var depthTolerance: Double = 0.02

    /// Per-layer level-count overrides, for the layers a bundle *described*
    /// rather than dumped. Absent means the bundle's own levels.
    var levelCounts: [Int: Int] = [:]

    // MARK: - the landscape, when the document is one
    //
    // The state `Landscape.swift` works on. A landscape is a choice rather than
    // a file, so what a document holds is the request that produced it, the
    // request the controls have been edited to, and which of those is on screen.

    /// The menu of functions, native: a landscape is sampled in this process.
    let catalog = Catalog.native
    /// What the controls say. Nil when this document is a bundle someone made
    /// earlier rather than a landscape being chosen.
    var landscape: LandscapeRequest?
    /// The request the open bundle actually came from.
    var shown: LandscapeRequest?
    /// The latest edit, waiting for the request in flight to finish, and
    /// whether it should re-frame when it arrives.
    var wanted: (request: LandscapeRequest, framing: Bool, draft: Bool)?
    /// Whether the bundle on screen is a draft: its contours are the grid's,
    /// so the same samples asked for in full are still a landscape to build,
    /// and no restyle places them on f in the meantime.
    var shownDraft = false
    var sampling = false
    var landscapeStatus: String?
    /// True while the picker is up. On the document because the menu command
    /// and the window are on opposite sides of SwiftUI's scene boundary, and
    /// this is the one thing they both know about.
    var browsing = false
    /// The resolution a landscape is sampled at while a control is being
    /// dragged. Low enough that the picture keeps up with the hand; the full
    /// resolution follows when the drag ends. A draft's contours are the
    /// grid's, too: placing them on f costs by their length, not by the
    /// resolution, and is most of a build (zeta: 2.7 s of a 240² draft).
    var draftResolution = 240

    /// The styling the controls have asked for, ahead of the bundle that will
    /// carry it: a restyle is derived on another thread (`Landscape.swift`),
    /// and a slider reads its value from here so its thumb does not spring
    /// back while the curves are being drawn. Nil when the bundle is up to date.
    var styled: Styling?
    /// Curves being derived on another thread: a restyle, a change of
    /// refinement, or a layer's levels.
    var drawing = false
    /// A layer's levels changed, and whether the change was a draft.
    var curvesWanted: Bool?
    /// Shown in the status line when drawing the curves takes long enough to
    /// notice, which a slider that restyles within a frame never does.
    var drawingStatus: String?

    // MARK: - the surface, when the document is one
    //
    // The state `Surfaces.swift` works on, parallel to the landscape's: the
    // request the controls say, the one on screen, and the latest edit
    // waiting behind the build in flight.

    var surface: SurfaceRequest?
    var surfaceShown: SurfaceRequest?
    var surfaceWanted: (request: SurfaceRequest, framing: Bool, draft: Bool)?
    var building = false
    var surfaceStatus: String?
    /// Bumped whenever the document becomes a different thing -- a new
    /// landscape or a new surface -- so a build still in flight for the old
    /// one is dropped when it lands rather than replacing the new one.
    var generation = 0

    var bakeResolution: Int = 4000
    var bakeStatus: String?
    private(set) var baking = false

    var url: URL? {
        switch state {
        case .empty: nil
        case .loading(let u): u
        case .ready(let b): b.url
        case .surface: nil
        case .failed(let u, _): u
        }
    }

    var title: String {
        if case .surface(let surfaceDrawing) = state {
            return SurfacePreset.named(surfaceDrawing.request.name)?.label ?? "Surface"
        }
        if let landscape, isLandscape {
            return catalog.preset(landscape.name)?.label ?? landscape.expression
        }
        return url?.deletingPathExtension().lastPathComponent ?? "Kurven"
    }

    var bundle: KurvenBundle? {
        if case .ready(let b) = state { return b }
        return nil
    }

    var surfaceDrawing: SurfaceDrawing? {
        if case .surface(let p) = state { return p }
        return nil
    }

    /// Whether anything is on screen: a bundle or a surface.
    var isOpen: Bool { bundle != nil || surfaceDrawing != nil }

    var layers: [Layer] { scene?.layers ?? [] }

    var previewOptions: PreviewOptions {
        let visible = Set(layers.indices).subtracting(hiddenLayers)
        return PreviewOptions(mode: mode, visibleLayers: visible)
    }

    // MARK: - loading

    func open(_ url: URL) { Task { await load(url) } }

    /// A bundle read from disk, its contours placed by f when it names a
    /// function and refinement is on -- here, off the main actor, because on
    /// zeta that is seconds.
    nonisolated static func read(_ url: URL, refining: Bool) throws -> KurvenBundle {
        let bundle = try KurvenBundle.read(at: url)
        guard refining, bundle.refine == nil,
              let refiner = NativeLandscape.refiner(for: bundle) else { return bundle }
        return bundle.refined(by: refiner.refine)
    }

    /// The awaitable form, so a headless run can wait for the bundle rather
    /// than poll for it.
    func load(_ url: URL) async {
        state = .loading(url)
        scene = nil
        navigator = nil
        generation += 1
        // Reading is megabytes of npy; it does not belong on the main actor
        // even though it is fast, because a stalled window during an open is
        // the difference between an app and a script with a window.
        let refining = refineContours
        let result = await Task.detached(priority: .userInitiated) { () -> Result<KurvenBundle, Error> in
            do { return .success(try Self.read(url, refining: refining)) }
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

    ///
    /// Nothing here derives curves: the bundle arrives with its layers drawn,
    /// refined or not, from whichever thread read or sampled it.
    func adopt(_ bundle: KurvenBundle, keepingCamera: Bool = false, draft: Bool = false) {
        state = .ready(bundle)
        surface = nil
        surfaceShown = nil
        shownDraft = draft
        curvesWanted = nil
        if !keepingCamera { styled = nil }
        // Styling asked for while this bundle was being sampled, and not in
        // the request it was sampled from, is drawn on it next.
        defer { redraw() }
        guard let preset = bundle.manifest.presets.first else {
            state = .failed(bundle.url, "the bundle declares no camera presets")
            return
        }
        var scene = Scene(bundle: bundle, preset: preset)
        derivedCurves = [:]
        if keepingCamera, let navigator {
            scene.camera = navigator.camera
            self.scene = scene
            adoptLandscape(bundle)
            return
        }
        depthTolerance = preset.depthTolerance
        bakeResolution = preset.buffer
        hiddenLayers = []
        levelCounts = [:]

        var navigator = Navigator(orbit: Orbit(matching: preset.projection),
                                  framing: Framing(center: P2(0, 0), unitsPerPixel: 1))
        scene.camera = navigator.camera
        if let bounds = scene.quickBounds() {
            navigator.framing = .fitting(bounds, in: viewport)
        }
        self.scene = scene
        self.navigator = navigator
        adoptLandscape(bundle)
    }

    /// Take a built drawing as the document. A new drawing opens at the default
    /// oblique projection, fitted to the window; an edit keeps the camera.
    func adopt(_ surfaceDrawing: SurfaceDrawing, keepingCamera: Bool = false) {
        let before = self.surfaceDrawing
        state = .surface(surfaceDrawing)
        landscape = nil
        shown = nil
        derivedCurves = [:]
        if keepingCamera, let navigator, let scene {
            // Hidden layers are hidden by name: an edit that adds the lines
            // puts a layer before the folds, and a hidden fold stays hidden.
            let hidden = Set(hiddenLayers.compactMap { scene.layers[safe: $0]?.spec.name })
            hiddenLayers = Set(surfaceDrawing.layers.indices.filter { hidden.contains(surfaceDrawing.layers[$0].spec.name) })
            // The same surface keeps its identity, so the renderer keeps its
            // textures and its normals and lays out only the curves.
            self.scene = before?.content == surfaceDrawing.content
                ? scene.drawing(surfaceDrawing.layers) : surfaceDrawing.scene(camera: navigator.camera)
            return
        }
        let preset = SurfaceRequest.preset
        depthTolerance = preset.depthTolerance
        bakeResolution = preset.buffer
        hiddenLayers = []
        levelCounts = [:]
        var navigator = Navigator(orbit: Orbit(matching: preset.projection),
                                  framing: Framing(center: P2(0, 0), unitsPerPixel: 1))
        let scene = surfaceDrawing.scene(camera: navigator.camera)
        if let bounds = scene.quickBounds() {
            navigator.framing = .fitting(bounds, in: viewport)
        }
        self.scene = scene
        self.navigator = navigator
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
        scene.camera = Orbit(matching: preset.projection, target: navigator.orbit.target).camera
        guard let bounds = scene.quickBounds() else { return }
        depthTolerance = preset.depthTolerance
        bakeResolution = preset.buffer
        apply(.preset(preset.projection, bounds))
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
    ///
    /// Derived on another thread (`redraw`): a level set the refiner has not
    /// placed before is seconds on zeta. A `draft`, while the slider is
    /// dragged, is the grid's contours, and the release places them on f.
    func rebuildCurves(draft: Bool = false) {
        // A full redraw asked for stays asked for, whatever drafts follow it.
        curvesWanted = draft && (curvesWanted ?? true)
        redraw()
    }

    /// The described layers at their current levels, from `cached` where the
    /// levels are the same and the cached curves are as refined as asked.
    nonisolated static func stroked(_ bundle: KurvenBundle,
                                    levels: [[Double]?],
                                    cached: [Int: DerivedCurves],
                                    draft: Bool) -> (layers: [Layer], derived: [Int: DerivedCurves]) {
        // The grid's contours, for a draft: the same bundle without its refiner.
        let grid = KurvenBundle(url: bundle.url, manifest: bundle.manifest, surface: bundle.surface,
                                layers: bundle.layers, wallMesh: bundle.wallMesh, refine: nil)
        let refined = bundle.refine != nil && !draft
        var derived = cached
        let layers = bundle.manifest.layers.enumerated().map { index, spec -> Layer in
            guard let levels = levels[index] else {
                return (try? bundle.layer(spec.name)) ?? Layer(spec: spec, paths: .empty)
            }
            // Only the layer whose slider moved is re-derived. Dragging one
            // changes one level set; re-contouring all four costs four times as
            // much to produce three identical answers.
            if let c = cached[index], c.levels == levels, c.refined || !refined {
                return c.layer
            }
            let layer = (refined ? bundle : grid).layer(spec, levels: levels)
            derived[index] = DerivedCurves(levels: levels, layer: layer, refined: refined)
            return layer
        }
        return (layers, derived)
    }

    /// The last derivation of each described layer, keyed by the levels that
    /// produced it. Cleared whenever the landscape underneath it changes.
    var derivedCurves: [Int: DerivedCurves] = [:]

    struct DerivedCurves: Sendable {
        var levels: [Double]
        var layer: Layer
        /// Placed on f, or the grid's: a draft's.
        var refined: Bool
    }

    func setLevelCount(_ n: Int, forLayer index: Int, draft: Bool = false) {
        levelCounts[index] = max(n, 1)
        rebuildCurves(draft: draft)
    }

    func resetLevels(forLayer index: Int) {
        levelCounts.removeValue(forKey: index)
        rebuildCurves()
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
    var canBake: Bool { !(navigator?.orbit.isPerspective ?? false) }

    func bake(to url: URL) {
        guard let scene = framedScene, !baking else { return }
        guard canBake else {
            bakeStatus = "switch off perspective to bake: the plates are "
                + "orthographic, and a perspective bake has no oracle"
            return
        }
        let options = BakeOptions(resolution: bakeResolution, depthTolerance: depthTolerance)
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
