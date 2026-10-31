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
///   - **the truncation, the contour levels, the hatch spacing** need nothing but the
///     grids that are already here, because every layer of a landscape's bundle
///     is a description rather than dumped curves. They are `restyle`, which
///     re-derives from the grids and never leaves the process.
///
/// Neither runs on the main actor. Both coalesce -- one job in flight, the
/// latest edit wins -- so a slow function makes the picture lag the hand and
/// never stops the window. A resample drops to a draft resolution, with the
/// grid's contours, until the drag ends. A restyle is fast because the
/// refiner remembers the curves it has placed (`ContourRefine.remembering`):
/// moving the truncation places the truncation edge and the rings at the new levels, and every
/// phase line is the one already drawn.
///
/// What each costs, measured by `kurven-cli landscape-costs` on an Apple M5
/// Max at 600², off the main actor; `KurvenApp --landscape NAME --drag truncation 30`
/// measures the main actor's longest stall during a drag, 22 ms or less on
/// every preset:
///
///                   sample    dragging   truncation step   hatch   levels    dragging
///     zeta           50 ms      3 ms      22 ms      9 ms    14 ms     2 ms
///     cn             70 ms      4 ms      44 ms      9 ms    14 ms     2 ms
///     exp_inv        52 ms      3 ms      14 ms      1 ms    27 ms     2 ms
///
/// Every preset samples in 70 ms or less. zeta took 4.5 s, on the main actor
/// for a truncation step: the refiner placed every phase line over the truncation, where
/// its curves are cut away (`ContourRefine.Contours`' ceiling), and every
/// evaluation of ζ queued for one lock (`Zeta.weights`).
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
        // old drawing's truncation or levels is carried across.
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
            request.truncation = styling.truncation
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
    /// Any other bundle -- a fixture, or one written by hand -- records no
    /// request, and opens read-only.
    func adoptLandscape(_ bundle: KurvenBundle) {
        guard bundle.manifest.provenance.example == "function" else {
            landscape = nil
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
            truncation: bundle.manifest.truncation,
            layers: bundle.manifest.layers,
            name: {
                if case .some(.string(let s)) = params["name"] { return s }
                return ""
            }())
        request.spacing = nil
        landscape = request
        shown = request
    }

    // MARK: - restyling

    /// The styling the controls show: the latest asked for, else the bundle's.
    var styling: Styling? { styled ?? bundle.map { Styling($0.manifest) } }

    /// Edit the styling and re-derive the curves from the grids already in
    /// memory. The landscape is unchanged -- the samples are the samples -- so
    /// the camera, the framing and the window all stay exactly as they are.
    ///
    /// The edit lands in `styled` at once, where the controls read it, and
    /// the curves follow from another thread (`redraw`).
    func restyle(_ change: (inout Manifest) -> Void) {
        guard let bundle, navigator != nil, let before = styling else { return }
        var manifest = bundle.manifest
        before.apply(to: &manifest)
        change(&manifest)
        let after = Styling(manifest)
        guard after != before else { return }
        styled = after
        // The next resample keeps the truncation given here.
        if landscape != nil { landscape?.truncation = after.truncation }
        redraw()
    }

    var truncation: Truncation { styling?.truncation ?? .none }

    /// Switch the contour refinement on or off for the open landscape: the
    /// same grids, the curves derived again, the camera where it was.
    func setRefinement(_ on: Bool) {
        refineContours = on
        redraw()
    }

    /// The one job that derives curves off the main actor, coalesced like the
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
        } else if let draft = curvesWanted, scene != nil {
            job = .levels(draft: draft)
        } else {
            return
        }
        curvesWanted = job.isLevels ? nil : curvesWanted
        drawing = true
        let asked = version
        let levelSets = bundle.manifest.layers.indices.map { levels(forLayer: $0) }
        let cached = derivedCurves
        // Said only when it is long enough to see: a truncation that redraws within
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
                    let (layers, derived) = Document.stroked(bundle, levels: levelSets,
                                                             cached: cached, draft: draft)
                    return .stroked(layers, derived)
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
            derivedCurves = [:]
            if styled == Styling(restyled.manifest) { styled = nil }
            guard let navigator, let preset = restyled.manifest.presets.first else { return }
            var scene = Scene(bundle: restyled, preset: preset)
            scene.camera = navigator.camera
            self.scene = scene
            // A layer drawn at levels of its own is drawn at them again.
            if !levelCounts.isEmpty { curvesWanted = curvesWanted ?? false }
        case .stroked(let layers, let derived):
            derivedCurves = derived
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
        case stroked([Layer], [Int: DerivedCurves])
    }

    /// Truncate the landscape differently.
    ///
    /// Everything follows: the heightfield the occluder meshes, the top edge each
    /// wall stroke rises to, where the plateaus are and therefore every truncation
    /// stroke and truncation edge, and which contours are cut off by `Restriction.belowTruncation`.
    /// All of it is derived, so all of it is this one assignment.
    func setTruncation(_ truncation: Truncation) {
        // Overrides are of the levels the truncation chose; a new truncation chooses again.
        levelCounts = [:]
        restyle { manifest in
            manifest.truncation = truncation
            // The levels are absolute, so a truncation raised past them would leave the
            // top of the landscape blank. `LandscapeStyle` is the rule the
            // build used to choose them in the first place, applied again to the new
            // ceiling -- by name, because in a landscape's bundle those two
            // names are what the two magnitude families are called.
            guard isLandscape, LandscapeStyle.ceiling(of: truncation) != nil else { return }
            let levels = LandscapeStyle.levels(under: truncation)
            for i in manifest.layers.indices {
                guard case .contour(let field, _, let restriction, let tiled)
                        = manifest.layers[i].source else { continue }
                let chosen: [Double]
                switch manifest.layers[i].name {
                case "mag_major": chosen = levels.major
                case "mag_minor": chosen = levels.minor
                default: continue
                }
                manifest.layers[i].source = .contour(field: field, levels: chosen,
                                                     restriction: restriction, tiled: tiled)
            }
        }
    }

    /// The hatch spacing of one described hatching layer, in world units.
    func spacing(ofLayer index: Int) -> Double? {
        guard let spec = styling?.layers[safe: index] else { return nil }
        guard case .hatching(let hatching, _) = spec.source else { return nil }
        return hatching.spacing
    }

    /// How many gaps a hatching layer rules along the domain's longer side,
    /// the side `LandscapeStyle.hatchSpacing` counts its seventy strokes on.
    ///
    /// Every face rounds its own count from the spacing and spreads its strokes
    /// to fit, so the hatching a person sees is a count, not a length. Setting
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
            guard case .hatching(var hatching, let faces) = manifest.layers[index].source else { return }
            hatching.spacing = value
            manifest.layers[index].source = .hatching(hatching, faces: faces)
        }
    }

    /// Which way the strokes on the plateaus run.
    func plateauHatchAxis() -> DomainAxis? {
        for spec in styling?.layers ?? [] {
            if let axis = spec.source.plateauHatchAxis { return axis }
        }
        return nil
    }

    func setPlateauHatchAxis(_ axis: DomainAxis) {
        restyle { manifest in
            for i in manifest.layers.indices {
                if case .hatching(var hatching, .plateaus(let tiled)) = manifest.layers[i].source {
                    hatching.angle = axis.angle
                    manifest.layers[i].source = .hatching(hatching, faces: .plateaus(tiled: tiled))
                }
            }
        }
    }

    // MARK: - keeping one

    /// Write the landscape as a `.kurven` bundle.
    ///
    /// Everything that would go in it is already in memory, and half of it --
    /// the truncation, the levels, the spacing -- exists only here. A described bundle
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

/// What a restyle changes: the truncation and the layers' descriptions. Everything
/// else in a manifest is the landscape's, and a restyle keeps it.
struct Styling: Equatable, Sendable {
    var truncation: Truncation
    var layers: [LayerSpec]

    init(_ manifest: Manifest) {
        truncation = manifest.truncation
        layers = manifest.layers
    }

    func apply(to manifest: inout Manifest) {
        manifest.truncation = truncation
        manifest.layers = layers
    }
}

extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

extension LandscapeRequest {
    /// This request with `KurvenApp --resample`'s overrides: `rMin`, `rMax`,
    /// `iMin`, `iMax` the window, as a bundle's provenance records it, `res`
    /// the resolution and `truncation` a uniform truncation height, as
    /// `kurven-cli landscape` spells them. Other keys are ignored.
    func applying(overrides a: [String: Double]) -> LandscapeRequest {
        var request = self
        request.domain = Domain(
            real: Interval(lo: a["rMin"] ?? domain.real.lo, hi: a["rMax"] ?? domain.real.hi),
            imag: Interval(lo: a["iMin"] ?? domain.imag.lo, hi: a["iMax"] ?? domain.imag.hi))
        if let res = a["res"] { request.resolution = Int(res.rounded()) }
        if let z = a["truncation"] { request.truncation = .uniform(z) }
        return request
    }
}
