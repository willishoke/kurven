import Foundation
import KurvenCore
import KurvenMath
import KurvenLandscape

/// A surface plate, described: which surface, at what lattice, with what ink.
///
/// The value the CLI parses its flags into and the app keeps in its
/// document, so both build a plate with one function -- `SurfacePlate.build`
/// -- as a bake and a landscape are each built by one function called twice.
public struct SurfaceRequest: Sendable, Equatable {
    public enum Shape: Sendable, Equatable {
        /// The torus of revolution about the z axis.
        case torus(major: Double, minor: Double)
        /// sn(z, m) on its fundamental rectangle, glued into a torus. Its own
        /// case rather than a `periodic` expression, because the modulus
        /// decides the periods as well as the function.
        case sn(modulus: Double, major: Double, minor: Double, grid: Int)
        /// Any function with a rectangle of periods, glued into a torus.
        case periodic(expression: String, periods: SIMD2<Double>, major: Double, minor: Double,
                      grid: Int)
        /// A forced system's invariant torus, fitted from `fit` time units of
        /// one run at `harmonics`, drawn by revolution on a ring of `radius`
        /// with state components `radial` and `axial`; its trajectory drawn
        /// on it for `duration`, sampled `every`.
        case forced(system: String, fit: Double, harmonics: Int, radial: Int, axial: Int,
                    radius: Double, duration: Double, every: Double)
    }

    /// Stroke widths. A fold width of zero leaves the folds out.
    public struct Style: Sendable, Equatable {
        public var lines: Double
        public var trajectory: Double
        public var folds: Double
        public init(lines: Double = 0.3, trajectory: Double = 0.08, folds: Double = 0.6) {
            self.lines = lines; self.trajectory = trajectory; self.folds = folds
        }
    }

    public var shape: Shape
    /// Lattice points around `u`; `v` gets half as many (a periodic plate
    /// sizes `v` to its rectangle instead).
    public var lattice: Int
    /// Lines of constant `u` and `v`, when drawn. A plain torus has nothing
    /// else to draw; a forced torus draws them only when asked.
    public var lines: SIMD2<Int>?
    public var style: Style
    /// The catalog entry this came from, for provenance and the title.
    public var name: String

    public init(_ shape: Shape, lattice: Int = 1024, lines: SIMD2<Int>? = nil,
                style: Style = Style(), name: String = "") {
        self.shape = shape; self.lattice = lattice; self.lines = lines
        self.style = style; self.name = name
    }

    /// The projection a surface plate opens at.
    public static let projection = PlateProjection(shear: 0, xAngle: -55, zAngle: 30,
                                                   flipX: false, yScale: nil)

    /// The request a drag builds while it lasts: a quarter of the lattice
    /// around, a periodic function sampled at most 240 across, and a forced
    /// trajectory sampled a quarter as often -- its quarter-million vertices
    /// are what the first frame after an edit spends its time on. The full
    /// request follows when the drag ends.
    public func drafted() -> SurfaceRequest {
        var r = self
        r.lattice = min(lattice, 256)
        switch shape {
        case .torus: break
        case .sn(let m, let R, let rr, let grid):
            r.shape = .sn(modulus: m, major: R, minor: rr, grid: min(grid, 240))
        case .periodic(let e, let p, let R, let rr, let grid):
            r.shape = .periodic(expression: e, periods: p, major: R, minor: rr, grid: min(grid, 240))
        case .forced(let s, let f, let h, let radial, let axial, let radius, let d, let every):
            r.shape = .forced(system: s, fit: f, harmonics: h, radial: radial, axial: axial,
                              radius: radius, duration: d, every: 4 * every)
        }
        return r
    }

    /// The same, as a camera preset: no margin and the CLI's bake resolution,
    /// so the window and `kurven-cli surface` bake the same picture.
    public static let preset = CameraPreset(name: "plate", plate: projection, margin: 0,
                                            buffer: 3000)
}

public extension SurfaceRequest {
    /// One number of a request, to edit by name: what a control moves.
    enum Field: String, CaseIterable, Sendable {
        case major, minor, modulus, grid, radius, duration, every, harmonics, fit, radial, axial
    }

    /// The fields this request's shape has, in the order a panel lists them.
    var fields: [Field] {
        switch shape {
        case .torus: [.major, .minor]
        case .sn: [.modulus, .major, .minor, .grid]
        case .periodic: [.major, .minor, .grid]
        case .forced: [.radius, .radial, .axial, .duration, .every, .harmonics, .fit]
        }
    }

    /// A field's value, or nil where the shape has no such field; setting
    /// one the shape does not have changes nothing.
    subscript(field: Field) -> Double? {
        get {
            switch (shape, field) {
            case (.torus(let R, _), .major), (.sn(_, let R, _, _), .major),
                 (.periodic(_, _, let R, _, _), .major): R
            case (.torus(_, let r), .minor), (.sn(_, _, let r, _), .minor),
                 (.periodic(_, _, _, let r, _), .minor): r
            case (.sn(let m, _, _, _), .modulus): m
            case (.sn(_, _, _, let g), .grid), (.periodic(_, _, _, _, let g), .grid): Double(g)
            case (.forced(_, _, _, _, _, let x, _, _), .radius): x
            case (.forced(_, _, _, _, _, _, let x, _), .duration): x
            case (.forced(_, _, _, _, _, _, _, let x), .every): x
            case (.forced(_, _, let x, _, _, _, _, _), .harmonics): Double(x)
            case (.forced(_, let x, _, _, _, _, _, _), .fit): x
            case (.forced(_, _, _, let x, _, _, _, _), .radial): Double(x)
            case (.forced(_, _, _, _, let x, _, _, _), .axial): Double(x)
            default: nil
            }
        }
        set {
            guard let x = newValue, x.isFinite else { return }
            let n = Int(x.rounded())
            switch shape {
            case .torus(var R, var r):
                if field == .major { R = x } else if field == .minor { r = x } else { return }
                shape = .torus(major: R, minor: r)
            case .sn(var m, var R, var r, var g):
                switch field {
                case .modulus: m = x
                case .major: R = x
                case .minor: r = x
                case .grid: g = n
                default: return
                }
                shape = .sn(modulus: m, major: R, minor: r, grid: g)
            case .periodic(let e, let p, var R, var r, var g):
                switch field {
                case .major: R = x
                case .minor: r = x
                case .grid: g = n
                default: return
                }
                shape = .periodic(expression: e, periods: p, major: R, minor: r, grid: g)
            case .forced(let s, var f, var h, var ra, var ax, var rad, var d, var e):
                switch field {
                case .radius: rad = x
                case .duration: d = x
                case .every: e = x
                case .harmonics: h = n
                case .fit: f = x
                case .radial: ra = n
                case .axial: ax = n
                default: return
                }
                shape = .forced(system: s, fit: f, harmonics: h, radial: ra, axial: ax,
                                radius: rad, duration: d, every: e)
            }
        }
    }
}

/// A surface plate, built: the surface, its ink, and what is worth keeping
/// between edits.
///
/// What an edit costs is decided here, by comparing the new request with the
/// plate it replaces, and only what the edit touches is made again:
///
///   - **the ink alone** -- line counts, stroke widths, a forced trajectory's
///     length and sampling, sn's sampling grid -- keeps the surface, and with
///     it `content`, so the window redraws the ink and keeps the surface's
///     textures;
///   - **the radii of a torus** re-place what is drawn on it: a periodic
///     function's contours are the function's, and move with the torus from
///     their coordinates;
///   - **a forced torus's revolution** re-places its values on the lattice,
///     kept in state space, and the trajectory's states, without evaluating
///     the fitted series or integrating again;
///   - **its fit** -- system, fit length, harmonics -- fits again.
///
/// Everything reused is exactly what building from nothing would produce, so
/// a plate depends on its request alone -- the window's is the CLI's. The one
/// exception is asked for by name: a `draft` may read a shorter trajectory
/// off a longer run already integrated.
public struct SurfacePlate: Sendable {
    public let request: SurfaceRequest
    public let surface: ParametricSurface
    public let layers: [Layer]
    /// The surface's identity: kept by an edit that leaves the surface alone,
    /// new whenever it moves.
    public let content: ContentID
    /// A forced plate's fitted torus.
    public var torus: InvariantTorus? { forced?.torus }
    public let revolution: Revolution?
    /// False when a forced torus's revolution passes through itself, so its
    /// back faces are kept rather than hidden.
    public let embedded: Bool
    /// What a forced plate keeps between edits.
    let forced: Forced?

    /// A fitted torus and what has been worked out from it, in state space:
    /// its values on each lattice asked for, and each run of its trajectory.
    struct Forced: Sendable {
        let fit: FitKey
        let torus: InvariantTorus
        var values: [Int: [SIMD3<Double>]]
        var runs: [RunKey: InvariantTorus.Run]
    }
    struct FitKey: Equatable, Sendable { let system: String; let fit: Double; let harmonics: Int }
    struct RunKey: Hashable, Sendable { let duration: Double; let every: Double }

    /// Whether building `request` from this plate fits a torus again -- the
    /// one edit that takes a noticeable time.
    public func refits(for request: SurfaceRequest) -> Bool {
        guard case .forced(let s, let f, let h, _, _, _, _, _) = request.shape else { return false }
        return forced?.fit != FitKey(system: s, fit: f, harmonics: h)
    }

    /// The scene this plate draws, from `camera`.
    public func scene(camera: Camera) -> Scene {
        Scene(surface: surface, layers: layers, camera: camera, margin: 0)
    }

    /// Everything the surface itself -- not what is drawn on it -- is built
    /// from: two requests with the same geometry have the same surface.
    static func geometry(_ r: SurfaceRequest) -> SurfaceRequest {
        var g = SurfaceRequest(r.shape, lattice: r.lattice)
        switch r.shape {
        case .torus: break
        case .sn(let m, let R, let rr, _): g.shape = .sn(modulus: m, major: R, minor: rr, grid: 0)
        case .periodic(_, let periods, let R, let rr, _):
            g.shape = .periodic(expression: "", periods: periods, major: R, minor: rr, grid: 0)
        case .forced(let s, let f, let h, let radial, let axial, let radius, _, _):
            g.shape = .forced(system: s, fit: f, harmonics: h, radial: radial, axial: axial,
                              radius: radius, duration: 0, every: 0)
        }
        return g
    }

    /// Everything a periodic plate's contours are built from: the function,
    /// its periods and its sampling, and not the torus they are drawn on.
    static func contours(_ r: SurfaceRequest) -> SurfaceRequest.Shape? {
        switch r.shape {
        case .sn(let m, _, _, let grid): .sn(modulus: m, major: 0, minor: 0, grid: grid)
        case .periodic(let e, let p, _, _, let grid):
            .periodic(expression: e, periods: p, major: 0, minor: 0, grid: grid)
        default: nil
        }
    }

    /// Build a plate. `previous`, when given, lends whatever of it the new
    /// request has not changed. `draft` allows the one approximation above.
    public static func build(_ request: SurfaceRequest, reusing previous: SurfacePlate? = nil,
                             draft: Bool = false) throws -> SurfacePlate {
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
                    k.request.lines == n ? k.layers.first { $0.spec.name == "lines" } : nil
                }
                return scaffold("lines", .parameterLines(u: n.x, v: n.y), style.lines,
                                old?.paths ?? surface.parameterLines(counts: (n.x, n.y),
                                                                     resolution: resolution))
            }
        }

        let surface: ParametricSurface
        var layers: [Layer]
        var revolution: Revolution?, embedded = true, forced: Forced?
        switch request.shape {
        case .torus(let R, let r):
            let n = request.lattice
            surface = kept?.surface
                ?? ParametricSurface.torus(major: R, minor: r, samples: (n, max(n / 2, 8)))
            layers = lines(on: surface, resolution: 4 * n).map { [$0] } ?? []

        case .sn(_, let R, let r, _), .periodic(_, _, let R, let r, _):
            let plate: PeriodicTorus.Plate
            if let previous, let key = contours(request), contours(previous.request) == key {
                // The same contours: as they are, or re-placed on new radii.
                let old = PeriodicTorus.Plate(surface: previous.surface, layers: previous.layers)
                plate = kept != nil ? old : old.placed(major: R, minor: r, lattice: request.lattice)
            } else if case .sn(let m, _, _, let grid) = request.shape {
                plate = try PeriodicTorus.jacobiSN(modulus: m, major: R, minor: r,
                                                   resolution: grid, lattice: request.lattice)
            } else if case .periodic(let expression, let periods, _, _, let grid) = request.shape {
                plate = try PeriodicTorus.plate(expression, periods: (periods.x, periods.y),
                                                major: R, minor: r, resolution: grid,
                                                lattice: request.lattice)
            } else { fatalError("unreachable") }
            surface = plate.surface
            // Their own folds come with them; the style's replace them below.
            layers = plate.layers.filter { if case .contour = $0.spec.source { true } else { false } }

        case .forced(let name, let fit, let harmonics, let radial, let axial, let radius,
                     let duration, let every):
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
                               values: [:], runs: [:])
            }
            let torus = state.torus
            let e = Revolution.fitting(torus, radial: radial, axial: axial, radius: radius)
            revolution = e
            let lattice = (request.lattice, request.lattice / 2)
            if let kept {
                surface = kept.surface
                embedded = kept.embedded
            } else {
                let values = state.values[lattice.0] ?? torus.values(lattice: lattice)
                // The lattice asked for now, and the largest before it: the
                // one a drag's drafts return to.
                let largest = state.values.keys.max().map { max($0, lattice.0) } ?? lattice.0
                state.values = state.values.filter { $0.key == largest }
                state.values[lattice.0] = values
                let placed = torus.surface(e, lattice: lattice, values: values)
                surface = placed.surface
                embedded = placed.embedded
            }

            let runKey = RunKey(duration: duration, every: every)
            let count = Int((duration / every).rounded()) + 1
            let run: InvariantTorus.Run
            var exact = true
            if let r = state.runs[runKey] {
                run = r
            } else if draft, let longer = state.runs.first(where: {
                $0.key.every == every && $0.value.states.count >= count })?.value {
                run = longer.prefix(count)
                exact = false
            } else {
                run = try torus.run(duration: duration, every: every)
            }
            // Kept: this run when it is exact, and the longest exact run at
            // each spacing -- the one a shorter draft is read from, and the
            // full-resolution one a drag's release returns to.
            var runs: [RunKey: InvariantTorus.Run] = [:]
            for (k, r) in state.runs where (runs.first { $0.key.every == k.every }?.value.states.count
                                               ?? -1) < r.states.count {
                runs = runs.filter { $0.key.every != k.every }
                runs[k] = r
            }
            if exact { runs[runKey] = run }
            state.runs = runs
            forced = state

            // The trajectory on screen is this one when the revolution and
            // the run are both unchanged -- and that run was exact.
            var sameInk = false
            if kept != nil, case .forced(_, _, _, _, _, _, duration, every)? = previous?.request.shape,
               previous?.forced?.runs[runKey] != nil {
                sameInk = true
            }
            let old = sameInk ? kept?.layers.first { $0.spec.name == "trajectory" } : nil
            layers = [scaffold("trajectory", .trajectory, style.trajectory,
                               old?.paths ?? torus.trajectory(e, run))]
            if let l = lines(on: surface, resolution: 2 * request.lattice) { layers.append(l) }
        }
        // The outline and inner silhouettes, derived per camera.
        if style.folds > 0 {
            layers.append(Layer(spec: LayerSpec(name: "folds", role: .outline, source: .foldLines,
                                                width: style.folds, heightPolicy: .surface),
                                paths: .empty))
        }
        return SurfacePlate(request: request, surface: surface, layers: layers,
                            content: kept?.content ?? ContentID(), revolution: revolution,
                            embedded: embedded, forced: forced)
    }
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
    /// so a preset and the CLI's plate of the same name are the same plate.
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
                          + "of one long run and fitted, with the trajectory winding round it.",
                      request: SurfaceRequest(.forced(system: "jerk", fit: 8000, harmonics: 24,
                                                      radial: 0, axial: 1, radius: 2.5,
                                                      duration: 2300, every: 0.01))),
    ]

    static func named(_ name: String) -> SurfacePreset? { catalog.first { $0.name == name } }
}
