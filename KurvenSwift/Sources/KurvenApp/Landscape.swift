import Foundation
import Observation
import KurvenCore
import KurvenService
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
///     re-derives in milliseconds and never leaves the process.
///
/// Dragging a slider therefore does one of two quite different things, and the
/// difference is visible: a restyle redraws within a frame, a resample coalesces
/// (one request in flight, the latest edit wins) and drops to a draft
/// resolution until the drag ends.
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
        // A different function is a different landscape, not an edit of this
        // one: its styling is the new function's defaults, so nothing of the
        // old plate's cap or levels is carried across.
        send(request, framing: true)
    }

    /// The controls changed. `draft` is true while one is being dragged.
    func landscapeEdited(draft: Bool) {
        guard var request = landscape else { return }
        if draft {
            request.resolution = min(request.resolution, draftResolution)
        }
        // The styling travels with the request, so a change of window keeps the
        // truncation and the hatching the user chose rather than re-deriving
        // them for the new rectangle.
        if let manifest = bundle?.manifest, isLandscape {
            request.caps = manifest.caps
            request.layers = manifest.layers
        }
        send(request, framing: false)
    }

    /// One request in flight, the latest edit queued behind it.
    ///
    /// A slider produces edits far faster than a landscape can be sampled, and
    /// every intermediate one is worthless the moment the next arrives. Queueing
    /// them would make the picture lag the control by however long the queue is;
    /// dropping all but the last makes it lag by one sample.
    private func send(_ request: LandscapeRequest, framing: Bool) {
        wanted = (request, framing)
        pump()
    }

    private func pump() {
        guard !sampling, let (request, framing) = wanted else { return }
        if let shown, shown.samples(as: request), !framing {
            // The samples on screen are already the ones asked for; only the
            // styling moved, and that is not this path's business.
            wanted = nil
            return
        }
        wanted = nil
        sampling = true
        landscapeStatus = "sampling…"
        Task {
            let clock = ContinuousClock()
            let started = clock.now
            do {
                // Sampled here, in this process: the evaluator is native, and
                // the bundle is a value rather than a directory to read back.
                let bundle = try await Task.detached(priority: .userInitiated) {
                    try NativeLandscape.build(request)
                }.value
                adopt(bundle, keepingCamera: !framing && scene != nil)
                shown = request
                landscapeStatus = "\(request.resolution)² in \(clock.now - started)"
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

    // MARK: - restyling, which costs nothing

    /// Edit the manifest and re-derive the ink from the grids already in
    /// memory. The landscape is unchanged -- the samples are the samples -- so
    /// the camera, the framing and the window all stay exactly as they are.
    func restyle(_ change: (inout Manifest) -> Void) {
        guard let bundle, let navigator else { return }
        var manifest = bundle.manifest
        change(&manifest)
        guard manifest != bundle.manifest else { return }
        let restyled = bundle.restyled(manifest)
        replace(bundle: restyled)
        derivedInk = [:]
        guard let preset = restyled.manifest.presets.first else { return }
        var scene = Scene(bundle: restyled, preset: preset)
        scene.camera = navigator.camera
        self.scene = scene
        if isLandscape { landscape?.caps = manifest.caps }
    }

    var caps: Caps { bundle?.manifest.caps ?? .none }

    /// Truncate the landscape differently.
    ///
    /// Everything follows: the heightfield the occluder meshes, the crest each
    /// wall stroke rises to, where the plateaus are and therefore every cap
    /// stroke and rim, and which contours are cut off by `Keep.belowCap`.
    /// All of it is derived, so all of it is this one assignment.
    func setCaps(_ caps: Caps) {
        restyle { manifest in
            manifest.caps = caps
            // The levels are absolute, so a cap raised past them would leave the
            // top of the landscape blank. `LandscapeStyle` is the rule Python
            // used to choose them in the first place, applied again to the new
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
        levelCounts = [:]
    }

    /// The hatch spacing of one described hatching layer, in world units.
    func spacing(ofLayer index: Int) -> Double? {
        guard let spec = bundle?.manifest.layers[safe: index] else { return nil }
        switch spec.source {
        case .wallHatch(_, let spacing, _, _, _, _): return spacing
        case .capHatch(_, let spacing, _): return spacing
        default: return nil
        }
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
        for spec in bundle?.manifest.layers ?? [] {
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
    /// Written here rather than asked of Python, because everything that would
    /// go in it is already in memory and half of it -- the cap, the levels, the
    /// spacing -- exists only here: the service was last asked for a landscape
    /// several edits ago. A described bundle is a manifest and two grids, which
    /// is why this is short.
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

extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
