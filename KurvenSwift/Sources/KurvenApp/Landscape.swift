import Foundation
import Observation
import KurvenCore
import KurvenLandscape

/// The half of a document that is a *choice* rather than a file.
///
/// Opening a bundle gives you one landscape to look at. This gives you the
/// function, the window and the truncation as controls, and the split between
/// what that costs is the whole design:
///
///   - **the function, the window, the resolution** need f evaluated again, so
///     they are a new landscape (`NativeLandscape.build`) and a new bundle --
///     sampled in this process, since the evaluator is native.
///   - **the cap, the contour levels, the hatch spacing** need nothing but the
///     grids that are already here, because every layer of a landscape's bundle
///     is a description rather than dumped ink. They are `restyle`, which
///     re-derives from the grids and never leaves the process.
///
/// Neither runs on the main actor. Both coalesce -- one job in flight, the
/// latest edit wins -- so a slow function makes the picture lag the hand and
/// never stops the window. A resample drops to a draft resolution, with the
/// grid's contours, until the drag ends. A restyle is fast because the
/// refiner remembers the curves it has placed (`ContourRefine.remembering`):
/// moving the cap places the rim and the rings at the new levels, and every
/// phase line is the one already drawn.
///
/// What each costs, measured by `kurven-cli landscape-costs` on an Apple M5
/// Max at 600², off the main actor; `KurvenApp --landscape NAME --drag cap 30`
/// measures the main actor's longest stall during a drag, 21 ms or less on
/// every preset:
///
///                   sample    dragging   cap step   hatch   levels    dragging
///     zeta          4.5 s      10 ms      82 ms     12 ms   3.1 s      2 ms
///     rgamma        467 ms      2 ms      30 ms      1 ms   220 ms     1 ms
///     cn            206 ms      4 ms      38 ms      9 ms   123 ms     2 ms
///
/// Before the refiner remembered, a cap step on zeta was 4.5 s on the main
/// actor. A layer's levels placed on f are still seconds there: a new count
/// spreads every level anew, so none is a curve already placed, and the
/// release pays for all of them while the window carries on.
@MainActor
extension Document {
    var isLandscape: Bool {
        bundle?.manifest.provenance.example == "function"
    }

    /// Start a landscape from a catalog preset, replacing whatever is open.
    func create(_ preset: FunctionPreset) {
        var request = LandscapeRequest(preset: preset,
                                       resolution: catalog.defaultResolution)
        request.spacing = nil          // let the rules derive it for this window
        landscape = request
        generation += 1
        surfaceWanted = nil
        // A different function is a different landscape, not an edit of this
        // one: its styling is the new function's defaults, so nothing of the
        // old plate's cap or levels is carried across.
        send(request, framing: true, draft: false)
    }

    /// The controls changed. `draft` is true while one is being dragged.
    func landscapeEdited(draft: Bool) {
        guard var request = landscape else { return }
        if draft {
            request.resolution = min(request.resolution, draftResolution)
        }
        // The styling travels with the request, so a change of window keeps the
        // truncation and the hatching the user chose rather than re-deriving
        // them for the new rectangle -- the styling asked for, which may not
        // be drawn yet.
        if let styling, isLandscape {
            request.caps = styling.caps
            request.layers = styling.layers
        }
        send(request, framing: false, draft: draft)
    }

    /// One request in flight, the latest edit queued behind it.
    ///
    /// A slider produces edits far faster than a landscape can be sampled, and
    /// every intermediate one is worthless the moment the next arrives. Queueing
    /// them would make the picture lag the control by however long the queue is;
    /// dropping all but the last makes it lag by one sample.
    private func send(_ request: LandscapeRequest, framing: Bool, draft: Bool) {
        wanted = (request, framing, draft)
        pump()
    }

    private func pump() {
        guard !sampling, let (request, framing, draft) = wanted else { return }
        if let shown, shown.samples(as: request), !framing, !shownDraft || draft {
            // The samples on screen are already the ones asked for; only the
            // styling moved, and that is not this path's business.
            wanted = nil
            return
        }
        wanted = nil
        sampling = true
        landscapeStatus = "sampling…"
        let asked = generation
        Task {
            let clock = ContinuousClock()
            let started = clock.now
            do {
                // Sampled here, in this process: the evaluator is native, and
                // the bundle is a value rather than a directory to read back.
                let refining = refineContours && !draft
                let bundle = try await Task.detached(priority: .userInitiated) {
                    try NativeLandscape.build(request, refine: refining)
                }.value
                // A surface chosen meanwhile is the document now.
                if generation == asked {
                    adopt(bundle, keepingCamera: !framing && scene != nil, draft: draft)
                    shown = request
                }
                landscapeStatus = "\(request.resolution)² in \(clock.now - started)"
                    + (draft ? " (draft)" : "")
            } catch {
                landscapeStatus = "\(error)"
            }
            sampling = false
            pump()
        }
    }

    /// Note the landscape a newly opened bundle is, so its controls start where
    /// it is rather than where the catalog's defaults are. A bundle written by
    /// `landscape` records its own request in `provenance.params`.
    ///
    /// A bundle from one of the retired Python plates records the plate's
    /// arguments instead, under the function it sampled. When that function is
    /// a catalog preset, the controls start from those arguments
    /// (`request(fromPlate:)`), and the first edit -- or Rebuild -- is a native
    /// landscape over the same window. A bundle whose provenance names nothing
    /// in the catalog opens read-only.
    func adoptLandscape(_ bundle: KurvenBundle) {
        guard bundle.manifest.provenance.example == "function" else {
            landscape = Self.request(fromPlate: bundle.manifest, catalog: catalog)
            // Nothing on screen came from this request, so even an unchanged
            // one is a landscape to build rather than one already shown.
            shown = nil
            return
        }
        let params = bundle.manifest.provenance.params
        func number(_ key: String) -> Double? {
            switch params[key] {
            case .some(.double(let d)): d
            case .some(.int(let i)): Double(i)
            default: nil
            }
        }
        var request = LandscapeRequest(
            expression: {
                if case .some(.string(let s)) = params["expression"] { return s }
                return bundle.manifest.provenance.function
            }(),
            domain: bundle.manifest.domain,
            resolution: Int(number("resolution") ?? 600),
            caps: bundle.manifest.caps,
            layers: bundle.manifest.layers,
            name: {
                if case .some(.string(let s)) = params["name"] { return s }
                return ""
            }())
        request.spacing = nil
        landscape = request
        shown = request
    }

    /// The request a plate bundle's provenance describes: the catalog preset it
    /// names, over the window and at the resolution and truncation its
    /// arguments record (`plateArguments`). Missing arguments fall back to the
    /// manifest, which records the same window and caps. Nil when the function
    /// is not in the catalog: then there is no native landscape to rebuild.
    static func request(fromPlate manifest: Manifest, catalog: Catalog) -> LandscapeRequest? {
        guard let preset = catalog.preset(manifest.provenance.function) else { return nil }
        var request = LandscapeRequest(preset: preset, resolution: catalog.defaultResolution)
        request.domain = manifest.domain
        request.caps = manifest.caps
        var arguments: [String: Double] = [:]
        for (key, value) in manifest.provenance.params {
            switch value {
            case .double(let d): arguments[key] = d
            case .int(let i): arguments[key] = Double(i)
            default: break
            }
        }
        return request.applying(plateArguments: arguments)
    }

    // MARK: - restyling

    /// The styling the controls show: the latest asked for, else the bundle's.
    var styling: Styling? { styled ?? bundle.map { Styling($0.manifest) } }

    /// Edit the styling and re-derive the ink from the grids already in
    /// memory. The landscape is unchanged -- the samples are the samples -- so
    /// the camera, the framing and the window all stay exactly as they are.
    ///
    /// The edit lands in `styled` at once, where the controls read it, and
    /// the ink follows from another thread (`redraw`).
    func restyle(_ change: (inout Manifest) -> Void) {
        guard let bundle, navigator != nil, let before = styling else { return }
        var manifest = bundle.manifest
        before.apply(to: &manifest)
        change(&manifest)
        let after = Styling(manifest)
        guard after != before else { return }
        styled = after
        // A plate bundle being rebuilt keeps the cap it was given here, too.
        if landscape != nil { landscape?.caps = after.caps }
        redraw()
    }

    var caps: Caps { styling?.caps ?? .none }

    /// Switch the contour refinement on or off for the open landscape: the
    /// same grids, the ink derived again, the camera where it was.
    func setRefinement(_ on: Bool) {
        refineContours = on
        redraw()
    }

    /// The one job that derives ink off the main actor, coalesced like the
    /// resample: one in flight, and when it lands, whatever was asked for
    /// meanwhile. A restyle or a change of refinement comes first, since it
    /// draws every layer; a layer's levels after, on the bundle that results.
    ///
    /// A result for a bundle that is no longer open -- a resample landed
    /// while it was being drawn -- is dropped, and the styling still asked
    /// for is drawn on the new one.
    func redraw() {
        guard !drawing, let bundle, let target = styling else { return }
        let current = Styling(bundle.manifest)
        if styled == current { styled = nil }
        // A draft's contours are the grid's on purpose; the full build that
        // follows the drag places them.
        let refine = refineContours && isLandscape && !shownDraft
        let refinementMoves = isLandscape && !shownDraft && (bundle.refine != nil) != refine
        let job: Job
        if target != current || refinementMoves {
            job = .restyle(target, refine: refinementMoves ? refine : nil)
        } else if let draft = inkWanted, scene != nil {
            job = .levels(draft: draft)
        } else {
            return
        }
        inkWanted = job.isLevels ? nil : inkWanted
        drawing = true
        let asked = version
        let levelSets = bundle.manifest.layers.indices.map { levels(forLayer: $0) }
        let cached = derivedInk
        // Said only when it is long enough to see: a cap that redraws within
        // a frame should not flicker the status line on every step.
        let slow = Task {
            try? await Task.sleep(for: .milliseconds(150))
            if !Task.isCancelled { drawingStatus = job.describing }
        }
        Task {
            let clock = ContinuousClock()
            let started = clock.now
            let result = await Task.detached(priority: .userInitiated) { () -> Drawn in
                switch job {
                case .restyle(let styling, let refine):
                    var b = bundle
                    if let refine {
                        b = KurvenBundle(url: b.url, manifest: b.manifest, surface: b.surface,
                                         layers: b.layers, wallMesh: b.wallMesh,
                                         refine: refine ? NativeLandscape.refiner(for: b)?.refine : nil)
                    }
                    var manifest = b.manifest
                    styling.apply(to: &manifest)
                    return .restyled(b.restyled(manifest))
                case .levels(let draft):
                    let (layers, derived) = Document.inked(bundle, levels: levelSets,
                                                           cached: cached, draft: draft)
                    return .inked(layers, derived)
                }
            }.value
            slow.cancel()
            drawing = false
            drawingStatus = nil
            if version == asked {
                install(result)
                if case .restyle(_, let refine?) = job, refine, bundle.refine == nil,
                   self.bundle?.refine == nil {
                    // Asked to refine a landscape whose function would not
                    // compile again: say so on the toggle, rather than ask forever.
                    refineContours = false
                }
                if clock.now - started > .milliseconds(150) {
                    landscapeStatus = "\(job.describing.dropLast()) in \(clock.now - started)"
                }
            }
            redraw()
        }
    }

    private func install(_ drawn: Drawn) {
        switch drawn {
        case .restyled(let restyled):
            replace(bundle: restyled)
            derivedInk = [:]
            if styled == Styling(restyled.manifest) { styled = nil }
            guard let navigator, let preset = restyled.manifest.presets.first else { return }
            var scene = Scene(bundle: restyled, preset: preset)
            scene.camera = navigator.camera
            self.scene = scene
            // A layer drawn at levels of its own is drawn at them again.
            if !levelCounts.isEmpty { inkWanted = inkWanted ?? false }
        case .inked(let layers, let derived):
            derivedInk = derived
            if let scene { self.scene = scene.drawing(layers) }
        }
    }

    private enum Job: Sendable {
        /// The bundle under this styling; with `refine`, refined or not first.
        case restyle(Styling, refine: Bool?)
        case levels(draft: Bool)

        var isLevels: Bool { if case .levels = self { true } else { false } }

        var describing: String {
            switch self {
            case .restyle(_, let refine?): refine ? "placing contours on f…" : "drawing from the grid…"
            case .restyle: "restyling…"
            case .levels(let draft): draft ? "contouring…" : "placing contours on f…"
            }
        }
    }

    private enum Drawn: Sendable {
        case restyled(KurvenBundle)
        case inked([Layer], [Int: DerivedInk])
    }

    /// Truncate the landscape differently.
    ///
    /// Everything follows: the heightfield the occluder meshes, the crest each
    /// wall stroke rises to, where the plateaus are and therefore every cap
    /// stroke and rim, and which contours are cut off by `Keep.belowCap`.
    /// All of it is derived, so all of it is this one assignment.
    func setCaps(_ caps: Caps) {
        // Overrides are of the levels the cap chose; a new cap chooses again.
        levelCounts = [:]
        restyle { manifest in
            manifest.caps = caps
            // The levels are absolute, so a cap raised past them would leave the
            // top of the landscape blank. `LandscapeStyle` is the rule the
            // build used to choose them in the first place, applied again to the new
            // ceiling -- by name, because in a landscape's bundle those two
            // names are what the two magnitude families are called.
            guard isLandscape, LandscapeStyle.ceiling(of: caps) != nil else { return }
            let levels = LandscapeStyle.levels(under: caps)
            for i in manifest.layers.indices {
                guard case .contour(let field, _, let keep, let tiled)
                        = manifest.layers[i].source else { continue }
                let chosen: [Double]
                switch manifest.layers[i].name {
                case "mag_major": chosen = levels.major
                case "mag_minor": chosen = levels.minor
                default: continue
                }
                manifest.layers[i].source = .contour(field: field, levels: chosen,
                                                     keep: keep, tiled: tiled)
            }
        }
    }

    /// The hatch spacing of one described hatching layer, in world units.
    func spacing(ofLayer index: Int) -> Double? {
        guard let spec = styling?.layers[safe: index] else { return nil }
        switch spec.source {
        case .wallHatch(_, let spacing, _, _, _, _): return spacing
        case .capHatch(_, let spacing, _): return spacing
        default: return nil
        }
    }

    /// How many gaps a hatching layer rules along the domain's longer side,
    /// the side `LandscapeStyle.hatchSpacing` counts its seventy strokes on.
    ///
    /// Every face rounds its own count from the spacing and spreads its strokes
    /// to fit, so the ruling a person sees is a count, not a length. Setting
    /// the spacing from a whole count makes that count exact on the longer
    /// faces and the nearest whole one on the others.
    func hatchCount(ofLayer index: Int) -> Int? {
        guard let spacing = spacing(ofLayer: index), let reach = hatchReach,
              spacing > 0 else { return nil }
        return max(1, Int((reach / spacing).rounded()))
    }

    func setHatchCount(_ count: Int, ofLayer index: Int) {
        guard let reach = hatchReach, count > 0 else { return }
        setSpacing(reach / Double(count), ofLayer: index)
    }

    private var hatchReach: Double? {
        guard let domain = bundle?.manifest.domain else { return nil }
        let reach = max(abs(domain.real.length), abs(domain.imag.length))
        return reach > 0 ? reach : nil
    }

    func setSpacing(_ value: Double, ofLayer index: Int) {
        restyle { manifest in
            guard index < manifest.layers.count else { return }
            switch manifest.layers[index].source {
            case .wallHatch(let edges, _, let pitch, let trim, let base, let top):
                manifest.layers[index].source = .wallHatch(
                    edges: edges, spacing: value, pitch: pitch, trim: trim,
                    base: base, topOffset: top)
            case .capHatch(let axis, _, let tiled):
                manifest.layers[index].source = .capHatch(axis: axis, spacing: value,
                                                          tiled: tiled)
            default: break
            }
        }
    }

    /// Which way the strokes on the truncated tops run.
    func capHatchAxis() -> KeepAxis? {
        for spec in styling?.layers ?? [] {
            if case .capHatch(let axis, _, _) = spec.source { return axis }
        }
        return nil
    }

    func setCapHatchAxis(_ axis: KeepAxis) {
        restyle { manifest in
            for i in manifest.layers.indices {
                if case .capHatch(_, let spacing, let tiled) = manifest.layers[i].source {
                    manifest.layers[i].source = .capHatch(axis: axis, spacing: spacing,
                                                          tiled: tiled)
                }
            }
        }
    }

    // MARK: - keeping one

    /// Write the landscape as a `.kurven` bundle.
    ///
    /// Everything that would go in it is already in memory, and half of it --
    /// the cap, the levels, the spacing -- exists only here. A described bundle
    /// is a manifest and two grids, which is why this is short.
    func saveBundle(to url: URL) {
        guard let bundle else { return }
        do {
            try bundle.write(to: url)
            landscapeStatus = "saved \(url.lastPathComponent)"
        } catch {
            landscapeStatus = "could not save: \(error)"
        }
    }
}

/// What a restyle changes: the cap and the layers' descriptions. Everything
/// else in a manifest is the landscape's, and a restyle keeps it.
struct Styling: Equatable, Sendable {
    var caps: Caps
    var layers: [LayerSpec]

    init(_ manifest: Manifest) {
        caps = manifest.caps
        layers = manifest.layers
    }

    func apply(to manifest: inout Manifest) {
        manifest.caps = caps
        manifest.layers = layers
    }
}

extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

extension LandscapeRequest {
    /// This request under a Python plate's arguments, as its provenance
    /// recorded them: `rMin`, `rMax`, `iMin`, `iMax` the window, `res` the
    /// resolution, `zLimit` a uniform cap. Other keys are not the landscape's
    /// business (the plate's camera, its buffer) and are ignored.
    func applying(plateArguments a: [String: Double]) -> LandscapeRequest {
        var request = self
        request.domain = Domain(
            real: Interval(lo: a["rMin"] ?? domain.real.lo, hi: a["rMax"] ?? domain.real.hi),
            imag: Interval(lo: a["iMin"] ?? domain.imag.lo, hi: a["iMax"] ?? domain.imag.hi))
        if let res = a["res"] { request.resolution = Int(res.rounded()) }
        if let z = a["zLimit"] { request.caps = .uniform(z) }
        return request
    }
}
