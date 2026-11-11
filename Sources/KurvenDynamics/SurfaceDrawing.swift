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
/// A layer painted on the surface, which the request does not draw, goes by
/// its field's class (`FieldClass`, `keeping(_:from:)`): a coordinate's
/// contours are carried while the parameter rectangle keeps its samples, the
/// height's while the surface is kept, and |f| and arg f's with the function.
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
    /// Contours of the request's function at other levels, from its samples
    /// as they were taken once (`PeriodicTorus.Drawing.contours`); nil for a
    /// surface that paints none.
    let contours: (@Sendable ([LayerSpec]) -> [Layer])?
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
        var contours: (@Sendable ([LayerSpec]) -> [Layer])?
        switch request.shape {
        case .torus(let R, let r):
            let n = request.grid
            surface = kept?.surface
                ?? ParametricSurface.torus(major: R, minor: r, samples: (n, max(n / 2, 8)))
            layers = lines(on: surface, resolution: request.lineResolution).map { [$0] } ?? []

        case .sn(_, let R, let r, _), .periodic(_, _, let R, let r, _):
            let drawing: PeriodicTorus.Drawing
            if let previous, let key = Self.contours(request), Self.contours(previous.request) == key {
                // The same contours: as they are, or re-placed on new radii.
                let old = PeriodicTorus.Drawing(surface: previous.surface, layers: previous.own,
                                                contours: previous.contours)
                drawing = kept != nil ? old : old.placed(major: R, minor: r, grid: request.grid)
            } else {
                drawing = try periodic(request)
            }
            surface = drawing.surface
            contours = drawing.contours
            // Their own folds come with them; the style's replace them below.
            layers = drawing.layers.filter(\.isContour)
            if let l = lines(on: surface, resolution: request.lineResolution) { layers.append(l) }

        case .relief:
            // The same relief keeps its contours as they are; any other
            // change is a new relief.
            if let kept {
                surface = kept.surface
                layers = kept.own.filter(\.isContour)
                contours = kept.contours
            } else {
                let drawing = try periodic(request)
                surface = drawing.surface
                layers = drawing.layers.filter(\.isContour)
                contours = drawing.contours
            }
            if let l = lines(on: surface, resolution: request.lineResolution) { layers.append(l) }

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
            if let l = lines(on: surface, resolution: request.lineResolution) { layers.append(l) }
        }
        if let w = linearCurves(on: surface) { layers.append(w) }
        // The outline and inner fold curves, derived per camera.
        if style.folds > 0 {
            layers.append(Layer(spec: LayerSpec(name: "folds", role: .outline, source: .foldCurves,
                                                width: style.folds, heightPolicy: .surface),
                                paths: .empty))
        }
        return SurfaceDrawing(request: request, surface: surface, layers: layers, own: layers,
                              contours: contours, content: kept?.content ?? ContentID(),
                              revolution: revolution, embedded: embedded, forced: forced)
    }

    /// The drawing a request's function makes on its torus: sn, any periodic
    /// expression, or sn in relief.
    static func periodic(_ request: SurfaceRequest) throws -> PeriodicTorus.Drawing {
        switch request.shape {
        case .torus, .forced:
            preconditionFailure("a torus of revolution or a forced torus paints no function")
        case .sn(let m, let R, let r, let grid):
            return try PeriodicTorus.jacobiSN(modulus: m, major: R, minor: r, resolution: grid,
                                              grid: request.grid)
        case .periodic(let expression, let periods, let R, let r, let grid):
            return try PeriodicTorus.drawing(expression, periods: (periods.x, periods.y), major: R, minor: r,
                                             resolution: grid, grid: request.grid)
        case .relief(let m, let R, let r, let amplitude, let repeats, let grid, let turn, let floor, let soft):
            // `repeats` periods round the ring and the tube, the curves
            // computed once and carried round by the periods; the grid round
            // the ring in all, 1024 a period at most.
            var base = try PeriodicTorus.jacobiSN(modulus: m, major: R, minor: r, resolution: grid,
                                                  grid: min(request.grid, 1024),
                                                  ceiling: SurfaceRequest.reliefCeiling, truncated: true, turn: turn)
                .repeated(real: repeats.x, imag: repeats.y, grid: request.grid)
            // The truncation as a smooth saturation rather than a crease,
            // c · tanh(|f| / c), an experiment on what the crease costs.
            if soft, let height = base.height {
                let c = base.ceiling
                base = PeriodicTorus.Drawing(surface: base.surface, layers: base.layers,
                                             height: { p in c * tanh(height(p) / c) },
                                             ceiling: c, radii: base.radii, contours: base.contours)
            }
            // Only |f| above the floor displaces, so each pole stands alone
            // on the bare torus instead of on a raised ground.
            if let floor, let height = base.height {
                base = PeriodicTorus.Drawing(surface: base.surface, layers: base.layers,
                                             height: { p in max(height(p) - floor, 0) },
                                             ceiling: base.ceiling - floor, radii: base.radii,
                                             contours: base.contours)
            }
            return base.relief(amplitude: amplitude, reach: min(r, R - r))
        }
    }

    /// The same surface drawn with `specs`, as a bundle that kept them asks.
    ///
    /// A layer whose source the drawing draws already keeps its curves, under
    /// the new name and style. Any other is drawn on the surface: a contour of
    /// a coordinate or the height by the plain routine (`Face.contours(of:_:on:)`),
    /// of |f| or arg f from the request's function as it was sampled once,
    /// all of them in one pass (`contours`); coordinate curves, linear
    /// curves, the trajectory. The surface and its identity are kept, so a
    /// restyle redraws curves alone.
    public func restyled(_ specs: [LayerSpec]) throws -> SurfaceDrawing {
        try restyled(specs, carrying: nil)
    }

    /// `restyled(_:)`, with the contours `previous` drew carried over from
    /// it by their field's class (`FieldClass`): a coordinate's where this
    /// drawing's parameter rectangle and its samples are the same -- found
    /// in parameter coordinates, they are the same curves there, and only
    /// their place in space is new -- and the height's where the surface is
    /// the same, since it is read off the map. |f| and arg f come with the
    /// drawing, from its function.
    private func restyled(_ specs: [LayerSpec], carrying previous: SurfaceDrawing?) throws -> SurfaceDrawing {
        func drawn(_ spec: LayerSpec) -> Layer? { layers.first { $0.spec.source == spec.source } }
        let sameRectangle = previous.map { $0.surface.u == surface.u && $0.surface.v == surface.v } ?? false
        let sameSurface = previous?.content == content
        func carried(_ spec: LayerSpec) -> PolylineSet<WorldSpace>? {
            guard case .contour(let field, _, _, _) = spec.source,
                  field.fieldClass == .coordinate ? sameRectangle : field.fieldClass == .embedding && sameSurface,
                  let old = previous?.layers.first(where: { $0.spec.source == spec.source }),
                  let coords = old.paths.coords else { return nil }
            let map = surface.map
            return PolylineSet(vertices: coords.map { P3(map($0).position) }, offsets: old.paths.offsets,
                               coords: coords)
        }
        let ofF = specs.filter { spec in
            if drawn(spec) == nil, case .contour(let field, _, _, _) = spec.source { field.isFieldOfF } else { false }
        }
        // On a surface with no function they are refused below, one by one.
        let placed = ofF.isEmpty ? [] : contours?(ofF) ?? []
        let restyled = try specs.map { spec -> Layer in
            if let layer = drawn(spec) ?? placed.first(where: { $0.spec.source == spec.source }) {
                return Layer(spec: spec, paths: layer.paths)
            }
            return Layer(spec: spec, paths: try carried(spec) ?? curves(spec))
        }
        return SurfaceDrawing(request: request, surface: surface, layers: restyled, own: own, contours: contours,
                              content: content, revolution: revolution, embedded: embedded, forced: forced)
    }

    /// This drawing, of an edited request, with the layers the drawing
    /// before it kept: each stays as it was styled -- its levels, its width,
    /// a contour the request does not draw -- but for the curves the request
    /// itself describes, its coordinate curves, linear curve, trajectory and
    /// folds, which are this request's: where it still draws them, and
    /// added where it now does. `previous`, the drawing the edit was made
    /// from, lends the contours of a coordinate it drew on the same
    /// parameter rectangle (`restyled(_:carrying:)`).
    public func keeping(_ kept: [LayerSpec], from previous: SurfaceDrawing? = nil) throws -> SurfaceDrawing {
        // The request's own curves as it draws them, whatever a restyle drew.
        let built = own.map(\.spec)
        var specs = kept.compactMap { spec in
            Self.describesRequest(spec) ? built.first { $0.name == spec.name && Self.describesRequest($0) } : spec
        }
        specs += built.filter { spec in Self.describesRequest(spec) && !specs.contains { $0.name == spec.name } }
        return try restyled(specs, carrying: previous)
    }

    /// Whether a layer is one of the curves a request describes in itself
    /// rather than as a layer: its coordinate curves, linear curve,
    /// trajectory or folds, as `build` names them.
    public static func describesRequest(_ spec: LayerSpec) -> Bool {
        switch (spec.name, spec.source) {
        case ("lines", .coordinateCurves), ("linearCurves", .linearCurves), ("trajectory", .trajectory),
             ("folds", .foldCurves): true
        default: false
        }
    }

    /// The curves of a layer the request does not draw, on this surface.
    private func curves(_ spec: LayerSpec) throws -> PolylineSet<WorldSpace> {
        if let why = SurfaceManifest.unsupported(spec.source, on: request) {
            throw ManifestError.unsupported("layer '\(spec.name)': \(why)")
        }
        switch spec.source {
        case .coordinateCurves(let u, let v):
            return surface.coordinateCurves(counts: (u, v), resolution: request.lineResolution)
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

/// Why a function cannot be painted on a surface, in words a field can show.
public struct FunctionRefusal: Error, CustomStringConvertible, Equatable {
    public let description: String
    public init(_ description: String) { self.description = description }
}

public extension SurfaceRequest {
    /// Whether a function can be painted on the surface: a round torus,
    /// plain or carrying one already. A forced torus is fitted to a flow,
    /// and sn in relief is built from sn.
    var takesFunction: Bool {
        switch shape {
        case .torus, .sn, .periodic: true
        case .forced, .relief: false
        }
    }

    /// The function's period rectangle, when the surface carries one; the
    /// plain torus's parameter rectangle, a whole turn each way, when not.
    var periods: SIMD2<Double>? {
        switch shape {
        case .torus: SIMD2(2 * .pi, 2 * .pi)
        case .sn(let m, _, _, _): SIMD2(4 * Jacobi.quarterPeriod(m), 2 * Jacobi.quarterPeriod(1 - m))
        case .periodic(_, let p, _, _, _): p
        case .forced, .relief: nil
        }
    }

    /// The rectangle of periods of an expression whose periods the catalog
    /// knows: sn(z, m) for a number 0 < m < 1, whose rectangle is
    /// [0, 4K) × [0, 2K'), as `surface sn` draws it. Nil for any other.
    static func periods(of expression: String) -> SIMD2<Double>? {
        guard case .call("sn", let arguments)? = try? KurvenMath.Expression.parse(expression),
              arguments.count == 2, arguments[0] == .variable,
              case .number(let m) = arguments[1], m > 0, m < 1 else { return nil }
        return SIMD2(4 * Jacobi.quarterPeriod(m), 2 * Jacobi.quarterPeriod(1 - m))
    }

    /// This request with `expression` painted on its torus over the period
    /// rectangle [0, periods.x) × [0, periods.y), as `kurven-cli surface
    /// periodic --expression E --periods P1,P2` asks for it: the radii and
    /// the grid kept, f sampled as the catalog's sn is, unless a function
    /// was sampled before, and then as that one was. The surface then has
    /// |f| and arg f to contour. Refused, saying why: a surface that takes
    /// no function, an expression the language does not parse, and what
    /// `check()` refuses, a period that is not positive among it.
    func painting(_ expression: String, periods: SIMD2<Double>) throws -> SurfaceRequest {
        let catalog: Int = if case .sn(_, _, _, let g)? = SurfacePreset.named("sn")?.request.shape { g } else { 600 }
        let (R, r, samples): (Double, Double, Int)
        switch shape {
        case .torus(let major, let minor): (R, r, samples) = (major, minor, catalog)
        case .sn(_, let major, let minor, let g), .periodic(_, _, let major, let minor, let g): (R, r, samples) = (major, minor, g)
        case .forced:
            throw FunctionRefusal("a function is painted on a round torus; a forced torus is fitted to its flow")
        case .relief:
            throw FunctionRefusal("a function is painted on a round torus; sn in relief is built from sn")
        }
        let text = expression.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { throw FunctionRefusal("the expression is empty") }
        do {
            _ = try KurvenMath.Expression.compile(text)
        } catch let error as ExpressionError {
            throw FunctionRefusal("\(text): \(error)")
        }
        var next = self
        next.shape = .periodic(expression: text, periods: periods, major: R, minor: r, grid: samples)
        next.name = ""
        let words = ["shape.periods": "a period", "shape.expression": "the expression",
                     "shape.grid": "the function's samples", "grid": "the grid"]
        do {
            try next.check { words[$0] ?? $0 }
        } catch ManifestError.invalid(let why) {
            throw FunctionRefusal(why)
        }
        return next
    }
}
