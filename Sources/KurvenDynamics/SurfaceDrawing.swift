import Foundation
import KurvenCore
import KurvenMath
import KurvenLandscape

/// A surface drawing, built: the surface, its curves, and what is worth keeping
/// between edits.
///
/// What an edit costs is decided here, by comparing the new request with the
/// drawing it replaces, and only what the edit touches is made again:
///
///   - **the curves alone** -- line counts, a linear curve's slope, stroke widths, a
///     forced trajectory's turns, sn's sampling grid -- keeps the surface,
///     and with
///     it `content`, so the window redraws the curves and keeps the surface's
///     textures;
///   - **the radii of a torus** re-place what is drawn on it: a periodic
///     function's contours are the function's, and move with the torus from
///     their coordinates;
///   - **a forced torus's revolution** re-places its values on the grid,
///     kept in state space, and the trajectory's states, without evaluating
///     the fitted series or integrating again;
///   - **its fit** -- system, fit length, harmonics -- fits again.
///
/// Everything reused is exactly what building from nothing would produce, so
/// a drawing depends on its request alone -- the window's is the CLI's. The one
/// `draft` names a request made while a control is dragged; nothing is
/// approximated for it now, and it is kept for the drag to say so.
///
/// A drawing restyled (`restyled(_:)`) draws other layers on the same
/// surface, as a bundle that kept them asks; what an edit reuses is still
/// only what its request drew.
public struct SurfaceDrawing: Sendable {
    public let request: SurfaceRequest
    public let surface: ParametricSurface
    public let layers: [Layer]
    /// The layers the request draws, which are all an edit reuses, whatever
    /// a restyle has drawn instead.
    let own: [Layer]
    /// The surface's identity: kept by an edit that leaves the surface alone,
    /// new whenever it moves.
    public let content: ContentID
    /// A forced drawing's fitted torus.
    public var torus: InvariantTorus? { forced?.torus }
    public let revolution: Revolution?
    /// False when a forced torus's revolution passes through itself, so its
    /// back faces are kept rather than hidden.
    public let embedded: Bool
    /// What a forced drawing keeps between edits.
    let forced: Forced?

    /// A fitted torus and what has been worked out from it, in state space:
    /// its values on each grid asked for.
    struct Forced: Sendable {
        let fit: FitKey
        let torus: InvariantTorus
        var values: [Int: [SIMD3<Double>]]
    }
    struct FitKey: Equatable, Sendable { let system: String; let fit: Double; let harmonics: Int }

    /// Whether building `request` from this drawing fits a torus again -- the
    /// one edit that takes a noticeable time.
    public func refits(for request: SurfaceRequest) -> Bool {
        guard case .forced(let s, let f, let h, _, _, _, _) = request.shape else { return false }
        return forced?.fit != FitKey(system: s, fit: f, harmonics: h)
    }

    /// The scene this drawing draws, from `camera`.
    public func scene(camera: Camera) -> Scene {
        Scene(surface: surface, layers: layers, camera: camera, depthTolerance: 0)
    }

    /// Everything the surface itself -- not what is drawn on it -- is built
    /// from: two requests with the same geometry have the same surface.
    static func geometry(_ r: SurfaceRequest) -> SurfaceRequest {
        var g = SurfaceRequest(r.shape, grid: r.grid)
        switch r.shape {
        case .torus: break
        case .sn(let m, let R, let rr, _): g.shape = .sn(modulus: m, major: R, minor: rr, grid: 0)
        case .periodic(_, let periods, let R, let rr, _):
            g.shape = .periodic(expression: "", periods: periods, major: R, minor: rr, grid: 0)
        case .forced(let s, let f, let h, let radial, let axial, let radius, _):
            g.shape = .forced(system: s, fit: f, harmonics: h, radial: radial, axial: axial,
                              radius: radius, turns: 0)
        case .relief:
            // Its contours are placed through the relief, so the whole
            // shape is its geometry, and a kept surface keeps them too.
            break
        }
        return g
    }

    /// Everything a periodic drawing's contours are built from: the function,
    /// its periods and its sampling, and not the torus they are drawn on.
    static func contours(_ r: SurfaceRequest) -> SurfaceRequest.Shape? {
        switch r.shape {
        case .sn(let m, _, _, let grid): .sn(modulus: m, major: 0, minor: 0, grid: grid)
        case .periodic(let e, let p, _, _, let grid):
            .periodic(expression: e, periods: p, major: 0, minor: 0, grid: grid)
        default: nil
        }
    }

    /// Build a drawing. `previous`, when given, lends whatever of it the new
    /// request has not changed. `draft` says the request is a drag's.
    public static func build(_ request: SurfaceRequest, reusing previous: SurfaceDrawing? = nil,
                             draft: Bool = false) throws -> SurfaceDrawing {
        let style = request.style
        let kept = previous.flatMap { geometry($0.request) == geometry(request) ? $0 : nil }
        func scaffold(_ name: String, _ source: LayerSource, _ width: Double,
                      _ paths: PolylineSet<WorldSpace>) -> Layer {
            Layer(spec: LayerSpec(name: name, role: .scaffold, source: source, width: width,
                                  heightPolicy: .surface),
                  paths: paths)
        }
        /// The lines, reused when neither the surface nor their counts moved.
        func lines(on surface: ParametricSurface, resolution: Int) -> Layer? {
            request.lines.map { n in
                let old = kept.flatMap { k in
                    k.request.lines == n ? k.own.first { $0.spec.name == "lines" } : nil
                }
                return scaffold("lines", .coordinateCurves(u: n.x, v: n.y), style.lines,
                                old?.paths ?? surface.coordinateCurves(counts: (n.x, n.y),
                                                                       resolution: resolution))
            }
        }
        /// The linear curve, reused when neither the surface nor it moved. Placed
        /// from the grid, so a new slope costs its own vertices and
        /// nothing of the surface's.
        func linearCurves(on surface: ParametricSurface) -> Layer? {
            request.linearCurves.map { w in
                let old = kept.flatMap { k in
                    k.request.linearCurves == w ? k.own.first { $0.spec.name == "linearCurves" } : nil
                }
                return scaffold("linearCurves", .linearCurves(slope: w.slope, turns: w.turns, count: w.count),
                                style.linearCurves,
                                old?.paths ?? surface.linearCurves(slope: w.slope, turns: w.turns,
                                                                   count: w.count))
            }
        }

        let surface: ParametricSurface
        var layers: [Layer]
        var revolution: Revolution?, embedded = true, forced: Forced?
        switch request.shape {
        case .torus(let R, let r):
            let n = request.grid
            surface = kept?.surface
                ?? ParametricSurface.torus(major: R, minor: r, samples: (n, max(n / 2, 8)))
            layers = lines(on: surface, resolution: lineResolution(request)).map { [$0] } ?? []

        case .sn(_, let R, let r, _), .periodic(_, _, let R, let r, _):
            let drawing: PeriodicTorus.Drawing
            if let previous, let key = contours(request), contours(previous.request) == key {
                // The same contours: as they are, or re-placed on new radii.
                let old = PeriodicTorus.Drawing(surface: previous.surface, layers: previous.own)
                drawing = kept != nil ? old : old.placed(major: R, minor: r, grid: request.grid)
            } else {
                drawing = try periodic(request)
            }
            surface = drawing.surface
            // Their own folds come with them; the style's replace them below.
            layers = drawing.layers.filter(\.isContour)

        case .relief:
            // The same relief keeps its contours as they are; any other
            // change is a new relief.
            if let kept {
                surface = kept.surface
                layers = kept.own.filter(\.isContour)
            } else {
                let drawing = try periodic(request)
                surface = drawing.surface
                layers = drawing.layers.filter(\.isContour)
            }

        case .forced(let name, let fit, let harmonics, let radial, let axial, let radius,
                     let turns):
            guard let system = ForcedSystem.catalog[name] else {
                throw DynamicsError.unknownSystem(name, known: ForcedSystem.catalog.keys.sorted())
            }
            let key = FitKey(system: name, fit: fit, harmonics: harmonics)
            var state: Forced
            if let old = previous?.forced, old.fit == key {
                state = old
            } else {
                state = Forced(fit: key, torus: try InvariantTorus.find(system, duration: fit,
                                                                        harmonics: harmonics),
                               values: [:])
            }
            let torus = state.torus
            let e = Revolution.fitting(torus, radial: radial, axial: axial, radius: radius)
            revolution = e
            let grid = (request.grid, request.grid / 2)
            if let kept {
                surface = kept.surface
                embedded = kept.embedded
            } else {
                let values = state.values[grid.0] ?? torus.values(grid: grid)
                // The grid asked for now, and the largest before it: the
                // one a drag's drafts return to.
                let largest = state.values.keys.max().map { max($0, grid.0) } ?? grid.0
                state.values = state.values.filter { $0.key == largest }
                state.values[grid.0] = values
                let placed = torus.surface(e, grid: grid, values: values)
                surface = placed.surface
                embedded = placed.embedded
            }

            forced = state

            // The trajectory is the linear curve at the rotation number, placed
            // from the grid: kept when the surface and its turns are.
            var sameCurves = false
            if kept != nil, case .forced(_, _, _, _, _, _, turns)? = previous?.request.shape {
                sameCurves = true
            }
            let old = sameCurves ? kept?.own.first { $0.spec.name == "trajectory" } : nil
            layers = [scaffold("trajectory", .trajectory, style.trajectory,
                               old?.paths ?? torus.trajectory(on: surface, turns: turns))]
            if let l = lines(on: surface, resolution: lineResolution(request)) { layers.append(l) }
        }
        if let w = linearCurves(on: surface) { layers.append(w) }
        // The outline and inner fold curves, derived per camera.
        if style.folds > 0 {
            layers.append(Layer(spec: LayerSpec(name: "folds", role: .outline, source: .foldCurves,
                                                width: style.folds, heightPolicy: .surface),
                                paths: .empty))
        }
        return SurfaceDrawing(request: request, surface: surface, layers: layers, own: layers,
                              content: kept?.content ?? ContentID(), revolution: revolution,
                              embedded: embedded, forced: forced)
    }

    /// How many vertices a coordinate curve has along its axis: four to a
    /// grid cell, two on a fitted torus, where each is the Fourier series.
    static func lineResolution(_ request: SurfaceRequest) -> Int {
        if case .forced = request.shape { return 2 * request.grid }
        return 4 * request.grid
    }

    /// The drawing a request's function makes on its torus, with `contours`
    /// of |f| and arg f, or the defaults: sn, any periodic expression, or
    /// sn in relief.
    static func periodic(_ request: SurfaceRequest,
                         contours: [LayerSpec]? = nil) throws -> PeriodicTorus.Drawing {
        switch request.shape {
        case .torus, .forced:
            preconditionFailure("a torus of revolution or a forced torus paints no function")
        case .sn(let m, let R, let r, let grid):
            return try PeriodicTorus.jacobiSN(modulus: m, major: R, minor: r, resolution: grid,
                                              grid: request.grid, contours: contours)
        case .periodic(let expression, let periods, let R, let r, let grid):
            return try PeriodicTorus.drawing(expression, periods: (periods.x, periods.y), major: R, minor: r,
                                             resolution: grid, grid: request.grid, contours: contours)
        case .relief(let m, let R, let r, let amplitude, let repeats, let grid, let turn, let floor, let soft):
            // `repeats` periods round the ring and the tube, the curves
            // computed once and carried round by the periods; the grid round
            // the ring in all, 1024 a period at most.
            var base = try PeriodicTorus.jacobiSN(modulus: m, major: R, minor: r, resolution: grid,
                                                  grid: min(request.grid, 1024), truncated: true, turn: turn,
                                                  contours: contours)
                .repeated(real: repeats.x, imag: repeats.y, grid: request.grid)
            // The truncation as a smooth saturation rather than a crease,
            // c · tanh(|f| / c), an experiment on what the crease costs.
            if soft, let height = base.height {
                let c = base.ceiling
                base = PeriodicTorus.Drawing(surface: base.surface, layers: base.layers,
                                             height: { p in c * tanh(height(p) / c) },
                                             ceiling: c, radii: base.radii)
            }
            // Only |f| above the floor displaces, so each pole stands alone
            // on the bare torus instead of on a raised ground.
            if let floor, let height = base.height {
                base = PeriodicTorus.Drawing(surface: base.surface, layers: base.layers,
                                             height: { p in max(height(p) - floor, 0) },
                                             ceiling: base.ceiling - floor, radii: base.radii)
            }
            return base.relief(amplitude: amplitude, reach: min(r, R - r))
        }
    }

    /// The same surface drawn with `specs`, as a bundle that kept them asks.
    ///
    /// A layer whose source the drawing draws already keeps its curves, under
    /// the new name and style. Any other is drawn on the surface: a contour of
    /// a coordinate or the height by the plain routine (`Face.contours(of:_:on:)`),
    /// of |f| or arg f by the request's function, sampled again for all of
    /// them at once; coordinate curves, linear curves, the trajectory. The
    /// surface and its identity are kept, so a restyle redraws curves alone.
    public func restyled(_ specs: [LayerSpec]) throws -> SurfaceDrawing {
        func drawn(_ spec: LayerSpec) -> Layer? { layers.first { $0.spec.source == spec.source } }
        let ofF = specs.filter { spec in
            if drawn(spec) == nil, case .contour(let field, _, _, _) = spec.source { field.isFieldOfF } else { false }
        }
        // On a surface with no function they are refused below, one by one.
        let placed = ofF.isEmpty || !request.paintsFunction ? [] : try Self.periodic(request, contours: ofF).layers
        let restyled = try specs.map { spec -> Layer in
            if let layer = drawn(spec) ?? placed.first(where: { $0.spec.source == spec.source }) {
                return Layer(spec: spec, paths: layer.paths)
            }
            return Layer(spec: spec, paths: try curves(spec))
        }
        return SurfaceDrawing(request: request, surface: surface, layers: restyled, own: own, content: content,
                              revolution: revolution, embedded: embedded, forced: forced)
    }

    /// The curves of a layer the request does not draw, on this surface.
    private func curves(_ spec: LayerSpec) throws -> PolylineSet<WorldSpace> {
        if let why = SurfaceManifest.unsupported(spec.source, on: request) {
            throw ManifestError.unsupported("layer '\(spec.name)': \(why)")
        }
        switch spec.source {
        case .coordinateCurves(let u, let v):
            return surface.coordinateCurves(counts: (u, v), resolution: Self.lineResolution(request))
        case .linearCurves(let slope, let turns, let count):
            return surface.linearCurves(slope: slope, turns: turns, count: count)
        case .foldCurves:
            return .empty
        case .trajectory:
            guard let torus, case .forced(_, _, _, _, _, _, let turns) = request.shape else { return .empty }
            return torus.trajectory(on: surface, turns: turns)
        case .contour(let field, let levels, _, .surface) where !field.isFieldOfF:
            let face = surface.face, map = surface.map
            let value: @Sendable (P2<ParamSpace>) -> Double = switch field {
            case .u: { $0.x }
            case .v: { $0.y }
            default: { map($0).position.z }
            }
            let period = field == .u ? surface.u.period : field == .v ? surface.v.period : nil
            let coords = face.contours(of: ScalarField(period: period, value: value), levels, on: face.copies[0])
                .flatMap(\.paths)
            return PolylineSet(paths: coords.map { $0.map { P3(map($0).position) } }, coords: coords)
        default:
            throw ManifestError.unsupported("layer '\(spec.name)': a \(spec.source) on a parametric surface")
        }
    }
}

extension Layer {
    /// Whether the layer is a contour: what a periodic drawing keeps of its own.
    var isContour: Bool { if case .contour = spec.source { true } else { false } }
}

/// A surface the gallery offers: a name to ask for it by, a label, a formula
/// and a note to show, and the request it opens as.
public struct SurfacePreset: Sendable, Equatable, Identifiable {
    public let name: String
    public let label: String
    public let formula: String
    public let notes: String
    public let request: SurfaceRequest
    public var id: String { name }

    public init(name: String, label: String, formula: String, notes: String,
                request: SurfaceRequest) {
        self.name = name; self.label = label; self.formula = formula; self.notes = notes
        var request = request
        request.name = name
        self.request = request
    }
}

public extension SurfacePreset {
    /// Named as `kurven-cli surface` names them, and built with its defaults,
    /// so a preset and the CLI's drawing of the same name are the same drawing.
    static let catalog: [SurfacePreset] = [
        SurfacePreset(name: "torus", label: "Torus", formula: "R = 2, r = 1",
                      notes: "The torus of revolution, ruled by its two families of circles.",
                      request: SurfaceRequest(.torus(major: 2, minor: 1), lines: SIMD2(36, 18))),
        SurfacePreset(name: "sn", label: "sn on its torus", formula: "sn(z, 0.64)",
                      notes: "Jacobi's sn is doubly periodic: its rectangle of periods, rolled "
                          + "up both ways, is a torus, and its contours close on it.",
                      request: SurfaceRequest(.sn(modulus: 0.64, major: 2, minor: 1, grid: 600))),
        SurfacePreset(name: "forced", label: "Forced oscillator",
                      formula: "x‴ = −αx′x″ − (ω₀² + 3βx²)x′ + A sin ωt",
                      notes: "A forced jerk oscillator's invariant torus, found in the spectrum "
                          + "of one long run and fitted, with the trajectory linear curve round it.",
                      // 344 turns: a convergent denominator of the rotation
                      // number, where the trajectory cuts the torus evenly.
                      request: SurfaceRequest(.forced(system: "jerk", fit: 8000, harmonics: 24,
                                                      radial: 0, axial: 1, radius: 2.5,
                                                      turns: 344))),
    ]

    static func named(_ name: String) -> SurfacePreset? { catalog.first { $0.name == name } }
}
