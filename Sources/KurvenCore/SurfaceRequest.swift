import Foundation

/// A surface drawing, described: which surface, at what grid, with what curves.
///
/// The value the CLI parses its flags into, the app keeps in its document,
/// and a parametric surface's bundle stores (`SurfaceManifest`), so each
/// builds a drawing with one function -- `SurfaceDrawing.build` -- as a bake
/// and a landscape are each built by one function called twice. The map is
/// code, so a bundle keeps the request that builds it rather than samples of
/// it, as a landscape's provenance keeps its request.
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
        /// on it for `turns` turns of the forcing -- as the linear curve at the
        /// rotation number, which is what the trajectory is on the fitted
        /// torus. The turns where it cuts the torus most evenly are the
        /// denominators of the rotation number's convergents.
        case forced(system: String, fit: Double, harmonics: Int, radial: Int, axial: Int,
                    radius: Double, turns: Int)
        /// sn's landscape in relief on its torus: the torus displaced along
        /// its normal by `amplitude` times |sn| truncated at its ceiling,
        /// over `repeats` periods round the ring and round the tube, the
        /// rectangle begun `turn` imaginary periods up the plane. Its phase
        /// lines end on the truncation edge. `floor`, when given, displaces
        /// only |sn| above it; `soft` saturates the truncation smoothly,
        /// c · tanh(|sn| / c), instead of creasing it.
        case relief(modulus: Double, major: Double, minor: Double, amplitude: Double,
                    repeats: SIMD2<Int>, grid: Int, turn: Double, floor: Double?, soft: Bool)

        /// The shape's name: its `kind` in a manifest.
        public var kind: String {
            switch self {
            case .torus: "torus"
            case .sn: "sn"
            case .periodic: "periodic"
            case .forced: "forced"
            case .relief: "relief"
            }
        }
    }

    /// Stroke widths. A fold width of zero leaves the folds out.
    public struct Style: Sendable, Equatable {
        public var lines: Double
        public var trajectory: Double
        public var folds: Double
        public var linearCurves: Double
        public init(lines: Double = 0.3, trajectory: Double = 0.08, folds: Double = 0.6,
                    linearCurves: Double = 0.3) {
            self.lines = lines; self.trajectory = trajectory; self.folds = folds
            self.linearCurves = linearCurves
        }
    }

    /// A linear curve drawn on the torus: `count` straight lines through its
    /// parameter rectangle at `slope` turns of `v` per turn of `u`, each
    /// `turns` turns long -- or closed, when the slope is a fraction whose
    /// denominator is within the turns. See `ParametricSurface.linearCurves`.
    public struct LinearCurves: Sendable, Equatable {
        public var slope: Double
        public var turns: Int
        public var count: Int
        public init(slope: Double, turns: Int = 24, count: Int = 1) {
            self.slope = slope; self.turns = turns; self.count = count
        }

        /// The turns after which a strand closes, when it does within
        /// `turns`: the denominator of a rational slope.
        public var closes: Int? { ParametricSurface.linearCurveCloses(slope: slope, within: turns) }
    }

    public var shape: Shape
    /// Grid points around `u`; `v` gets half as many (a periodic drawing
    /// sizes `v` to its rectangle instead).
    public var grid: Int
    /// Lines of constant `u` and `v`, when drawn. A plain torus has nothing
    /// else to draw; a forced torus draws them only when asked.
    public var lines: SIMD2<Int>?
    /// A linear curve, when drawn. Curves alone: the surface does not know it is
    /// there, so its slope can move under a slider.
    public var linearCurves: LinearCurves?
    public var style: Style
    /// The catalog entry this came from, for provenance and the title.
    public var name: String

    public init(_ shape: Shape, grid: Int = 1024, lines: SIMD2<Int>? = nil,
                linearCurves: LinearCurves? = nil, style: Style = Style(), name: String = "") {
        self.shape = shape; self.grid = grid; self.lines = lines
        self.linearCurves = linearCurves; self.style = style; self.name = name
    }

    /// The projection a surface drawing opens at.
    public static let projection = ObliqueProjection(shear: 0, xAngle: -55, zAngle: 30,
                                                     flipX: false, yScale: nil)

    /// The request a drag builds while it lasts: a quarter of the grid
    /// around, and a periodic function sampled at most 240 across. The full
    /// request follows when the drag ends. A forced trajectory needs no
    /// draft: it is a linear curve, a vertex per cell of whatever grid.
    public func drafted() -> SurfaceRequest {
        var r = self
        r.grid = min(grid, 256)
        switch shape {
        case .torus, .forced: break
        case .sn(let m, let R, let rr, let grid):
            r.shape = .sn(modulus: m, major: R, minor: rr, grid: min(grid, 240))
        case .periodic(let e, let p, let R, let rr, let grid):
            r.shape = .periodic(expression: e, periods: p, major: R, minor: rr, grid: min(grid, 240))
        case .relief(let m, let R, let rr, let a, let n, let grid, let turn, let floor, let soft):
            r.shape = .relief(modulus: m, major: R, minor: rr, amplitude: a, repeats: n,
                              grid: min(grid, 240), turn: turn, floor: floor, soft: soft)
        }
        return r
    }

    /// The same, as a camera preset: no depth tolerance and the CLI's bake resolution,
    /// so the window and `kurven-cli surface` bake the same picture.
    public static let preset = CameraPreset(name: "oblique", projection: projection, depthTolerance: 0,
                                            buffer: 3000)

    /// Whether the surface carries a function, whose |f| and arg f a contour
    /// can be taken of: sn and any periodic expression, in relief or not.
    public var paintsFunction: Bool {
        switch shape {
        case .sn, .periodic, .relief: true
        case .torus, .forced: false
        }
    }

    /// Whether the function is truncated, so a contour of it can be kept
    /// below the truncation: sn in relief, whose phase lines end on the
    /// truncation edge.
    public var truncatesFunction: Bool {
        if case .relief = shape { return true }
        return false
    }
}

public extension SurfaceRequest {
    /// One number of a request, to edit by name: what a control moves.
    enum Field: String, CaseIterable, Sendable {
        case major, minor, modulus, grid, radius, harmonics, fit, radial, axial
        /// A forced trajectory's length, in turns of the forcing.
        case trajectoryTurns
        /// The linear curve's, present when one is drawn: its slope, its length
        /// in turns, and how many strands.
        case slope, turns, strands
        /// A relief's amplitude: how far along the normal the torus is
        /// displaced per unit of the truncated |f|. `--relief` on the CLI.
        case relief
    }

    /// The fields this request's shape has, in the order a panel lists them.
    /// The linear curve's are listed by `linearCurveFields` when there is one.
    var fields: [Field] {
        switch shape {
        case .torus: [.major, .minor]
        case .sn: [.modulus, .major, .minor, .grid]
        case .relief: [.modulus, .major, .minor, .grid, .relief]
        case .periodic: [.major, .minor, .grid]
        case .forced: [.radius, .radial, .axial, .trajectoryTurns, .harmonics, .fit]
        }
    }

    static let linearCurveFields: [Field] = [.slope, .turns, .strands]

    /// The fields that move curves alone -- placed from the grid, nothing of
    /// the surface remade -- and so want no draft while dragged.
    static let curveFields: [Field] = linearCurveFields + [.trajectoryTurns]

    /// A field's value, or nil where the shape has no such field; setting
    /// one the shape does not have changes nothing.
    subscript(field: Field) -> Double? {
        get {
            switch (shape, field) {
            case (_, .slope): linearCurves?.slope
            case (_, .turns): linearCurves.map { Double($0.turns) }
            case (_, .strands): linearCurves.map { Double($0.count) }
            case (.torus(let R, _), .major), (.sn(_, let R, _, _), .major),
                 (.periodic(_, _, let R, _, _), .major), (.relief(_, let R, _, _, _, _, _, _, _), .major): R
            case (.torus(_, let r), .minor), (.sn(_, _, let r, _), .minor),
                 (.periodic(_, _, _, let r, _), .minor), (.relief(_, _, let r, _, _, _, _, _, _), .minor): r
            case (.sn(let m, _, _, _), .modulus), (.relief(let m, _, _, _, _, _, _, _, _), .modulus): m
            case (.relief(_, _, _, let a, _, _, _, _, _), .relief): a
            case (.sn(_, _, _, let g), .grid), (.periodic(_, _, _, _, let g), .grid),
                 (.relief(_, _, _, _, _, let g, _, _, _), .grid): Double(g)
            case (.forced(_, _, _, _, _, let x, _), .radius): x
            case (.forced(_, _, _, _, _, _, let x), .trajectoryTurns): Double(x)
            case (.forced(_, _, let x, _, _, _, _), .harmonics): Double(x)
            case (.forced(_, let x, _, _, _, _, _), .fit): x
            case (.forced(_, _, _, let x, _, _, _), .radial): Double(x)
            case (.forced(_, _, _, _, let x, _, _), .axial): Double(x)
            default: nil
            }
        }
        set {
            guard let x = newValue, x.isFinite else { return }
            let n = Int(x.rounded())
            switch field {
            case .slope: linearCurves?.slope = x; return
            case .turns: linearCurves?.turns = max(n, 1); return
            case .strands: linearCurves?.count = max(n, 1); return
            default: break
            }
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
            case .forced(let s, var f, var h, var ra, var ax, var rad, var t):
                switch field {
                case .radius: rad = x
                case .trajectoryTurns: t = max(n, 1)
                case .harmonics: h = n
                case .fit: f = x
                case .radial: ra = n
                case .axial: ax = n
                default: return
                }
                shape = .forced(system: s, fit: f, harmonics: h, radial: ra, axial: ax,
                                radius: rad, turns: t)
            case .relief(var m, var R, var r, var a, let k, var g, let turn, let floor, let soft):
                switch field {
                case .modulus: m = x
                case .major: R = x
                case .minor: r = x
                case .grid: g = n
                case .relief: a = x
                default: return
                }
                shape = .relief(modulus: m, major: R, minor: r, amplitude: a, repeats: k, grid: g,
                                turn: turn, floor: floor, soft: soft)
            }
        }
    }
}

// MARK: - the schema

public extension SurfaceRequest {
    /// Read from a manifest's `surface` (`SurfaceManifest`), and checked
    /// (`check()`): a request that could not build a drawing is refused here,
    /// with the number that is wrong, rather than when it is built.
    init(json: JSONValue) throws {
        let o = try json.object("SurfaceRequest")
        let s = try o.value("shape", "SurfaceRequest").object("SurfaceRequest.shape")
        let what = "SurfaceRequest.shape"
        let shape: Shape
        switch try s.string("kind", what) {
        case "torus":
            shape = .torus(major: try s.double("major", what), minor: try s.double("minor", what))
        case "sn":
            shape = .sn(modulus: try s.double("modulus", what), major: try s.double("major", what),
                        minor: try s.double("minor", what), grid: try s.int("grid", what))
        case "periodic":
            let p = try s.doubles("periods", what)
            guard p.count == 2 else { throw ManifestError.badShape("SurfaceRequest periods", [p.count]) }
            shape = .periodic(expression: try s.string("expression", what), periods: SIMD2(p[0], p[1]),
                              major: try s.double("major", what), minor: try s.double("minor", what),
                              grid: try s.int("grid", what))
        case "forced":
            shape = .forced(system: try s.string("system", what), fit: try s.double("fit", what),
                            harmonics: try s.int("harmonics", what), radial: try s.int("radial", what),
                            axial: try s.int("axial", what), radius: try s.double("radius", what),
                            turns: try s.int("turns", what))
        case "relief":
            let k = try s.ints("repeats", what)
            guard k.count == 2 else { throw ManifestError.badShape("SurfaceRequest repeats", [k.count]) }
            shape = .relief(modulus: try s.double("modulus", what), major: try s.double("major", what),
                            minor: try s.double("minor", what), amplitude: try s.double("amplitude", what),
                            repeats: SIMD2(k[0], k[1]), grid: try s.int("grid", what),
                            turn: try s.double("turn", what),
                            floor: s["floor"] == .null ? nil : try s.double("floor", what),
                            soft: try s.bool("soft", what))
        case let other:
            throw ManifestError.unknownKind(other, of: "SurfaceRequest.shape",
                                            known: ["torus", "sn", "periodic", "forced", "relief"])
        }
        let w = "SurfaceRequest.linearCurves", st = "SurfaceRequest.style"
        let lines: SIMD2<Int>? = o["lines"] == .null ? nil : try {
            let l = try o.value("lines", "SurfaceRequest").object("SurfaceRequest.lines")
            return SIMD2(try l.int("u", "SurfaceRequest.lines"), try l.int("v", "SurfaceRequest.lines"))
        }()
        let linear: LinearCurves? = o["linearCurves"] == .null ? nil : try {
            let l = try o.value("linearCurves", "SurfaceRequest").object(w)
            return LinearCurves(slope: try l.double("slope", w), turns: try l.int("turns", w),
                                count: try l.int("count", w))
        }()
        let style = try o.value("style", "SurfaceRequest").object(st)
        self.init(shape, grid: try o.int("grid", "SurfaceRequest"), lines: lines, linearCurves: linear,
                  style: Style(lines: try style.double("lines", st), trajectory: try style.double("trajectory", st),
                               folds: try style.double("folds", st),
                               linearCurves: try style.double("linearCurves", st)),
                  name: try o.string("name", "SurfaceRequest"))
        try check()
    }

    var json: JSONValue {
        var shape: [String: JSONValue]
        switch self.shape {
        case .torus(let R, let r):
            shape = ["major": .double(R), "minor": .double(r)]
        case .sn(let m, let R, let r, let grid):
            shape = ["modulus": .double(m), "major": .double(R), "minor": .double(r),
                     "grid": .int(grid)]
        case .periodic(let e, let p, let R, let r, let grid):
            shape = ["expression": .string(e),
                     "periods": .array([.double(p.x), .double(p.y)]),
                     "major": .double(R), "minor": .double(r), "grid": .int(grid)]
        case .forced(let system, let fit, let harmonics, let radial, let axial, let radius, let turns):
            shape = ["system": .string(system), "fit": .double(fit),
                     "harmonics": .int(harmonics), "radial": .int(radial), "axial": .int(axial),
                     "radius": .double(radius), "turns": .int(turns)]
        case .relief(let m, let R, let r, let a, let k, let grid, let turn, let floor, let soft):
            shape = ["modulus": .double(m), "major": .double(R), "minor": .double(r),
                     "amplitude": .double(a), "repeats": .array([.int(k.x), .int(k.y)]), "grid": .int(grid),
                     "turn": .double(turn), "floor": floor.map(JSONValue.double) ?? .null, "soft": .bool(soft)]
        }
        shape["kind"] = .string(self.shape.kind)
        return .object([
            "shape": .object(shape), "grid": .int(grid),
            "lines": lines.map { .object(["u": .int($0.x), "v": .int($0.y)]) } ?? .null,
            "linearCurves": linearCurves.map {
                .object(["slope": .double($0.slope), "turns": .int($0.turns), "count": .int($0.count)])
            } ?? .null,
            "style": .object(["lines": .double(style.lines), "trajectory": .double(style.trajectory),
                              "folds": .double(style.folds), "linearCurves": .double(style.linearCurves)]),
            "name": .string(name),
        ])
    }

    /// Refuse a request no drawing can be built from, or one whose drawing
    /// would cost without bound, naming the number: a radius that is not
    /// positive, a modulus outside (0, 1), a relief outside [0, 1], a grid
    /// too coarse to span its rectangle or too fine to hold (`maxSamples`),
    /// curves of more vertices than a layer holds (`maxVertices`), a forced
    /// torus revolved on one state component twice. What only building can find -- a system the
    /// catalog does not have, an expression that does not parse --
    /// `SurfaceDrawing.build` refuses. `name` spells a field, given its key
    /// in the request (`shape.modulus`), as the refusal names it: by its key
    /// in a manifest's `surface`, or as the CLI's flag.
    func check(naming name: (String) -> String = { "surface." + $0 }) throws {
        func refuse(_ key: String, _ why: String) -> ManifestError { .invalid("\(name(key)) \(why)") }
        func positive(_ x: Double, _ key: String) throws {
            guard x.isFinite, x > 0 else { throw refuse(key, "must be a positive number, not \(x)") }
        }
        func finite(_ x: Double, _ key: String) throws {
            guard x.isFinite else { throw refuse(key, "must be a finite number, not \(x)") }
        }
        func within(_ n: Int, _ range: ClosedRange<Int>, _ key: String, _ what: String) throws {
            guard range.contains(n) else {
                throw refuse(key, "must be \(range.lowerBound) to \(range.upperBound) \(what), not \(n)")
            }
        }
        func modulus(_ m: Double) throws {
            guard m.isFinite, m > 0, m < 1 else {
                throw refuse("shape.modulus", "must lie strictly between 0 and 1, not \(m)")
            }
        }
        func radii(_ R: Double, _ r: Double) throws {
            try positive(R, "shape.major"); try positive(r, "shape.minor")
        }
        try within(grid, 8...Self.maxGrid, "grid", "samples")
        switch shape {
        case .torus(let R, let r):
            try radii(R, r)
        case .sn(let m, let R, let r, let g):
            try modulus(m); try radii(R, r)
            try within(g, 8...Self.maxFunctionGrid, "shape.grid", "samples")
        case .periodic(let e, let p, let R, let r, let g):
            guard !e.trimmingCharacters(in: .whitespaces).isEmpty else { throw refuse("shape.expression", "is empty") }
            try positive(p.x, "shape.periods"); try positive(p.y, "shape.periods")
            try radii(R, r)
            try within(g, 8...Self.maxFunctionGrid, "shape.grid", "samples")
        case .forced(let system, let fit, let harmonics, let radial, let axial, let radius, let turns):
            guard !system.isEmpty else { throw refuse("shape.system", "is unnamed") }
            try positive(fit, "shape.fit"); try positive(radius, "shape.radius")
            guard fit <= Self.maxFit else {
                throw refuse("shape.fit", "must be at most \(Int(Self.maxFit)) time units, not \(fit)")
            }
            try within(harmonics, 1...Self.maxHarmonics, "shape.harmonics", "harmonics")
            try within(radial, 0...2, "shape.radial", "(x, x′ or x″)")
            try within(axial, 0...2, "shape.axial", "(x, x′ or x″)")
            guard radial != axial else {
                throw refuse("shape.radial", "and \(name("shape.axial")) are both state component \(radial): "
                             + "one component as both is a curve, not a torus")
            }
            try within(turns, 1...Self.maxVertices, "shape.turns", "turns")
            let vertices = Double(turns) * Double(sampleBound.u)
            guard vertices <= Double(Self.maxVertices) else {
                throw refuse("shape.turns", "would take " + String(format: "%.3g", vertices)
                             + " vertices for the trajectory, more than the \(Self.maxVertices) a layer holds")
            }
        case .relief(let m, let R, let r, let a, let k, let g, let turn, let floor, _):
            try modulus(m); try radii(R, r)
            guard Self.reliefRange.contains(a) else {
                throw refuse("shape.amplitude", "must be from 0 to 1, not \(a)")
            }
            try finite(turn, "shape.turn")
            if let floor { try finite(floor, "shape.floor") }
            try within(k.x, 1...Self.maxRepeats, "shape.repeats", "whole periods each way")
            try within(k.y, 1...Self.maxRepeats, "shape.repeats", "whole periods each way")
            try within(g, 8...Self.maxFunctionGrid, "shape.grid", "samples")
        }
        let samples = sampleBound
        guard Double(samples.u) * Double(samples.v) <= Double(Self.maxSamples) else {
            throw refuse("grid", "and the radii make a grid of \(samples.u) × \(samples.v) samples, more than "
                         + "the \(Self.maxSamples) a surface holds")
        }
        if let lines {
            try within(lines.x, 0...Self.maxVertices, "lines", "coordinate curves")
            try within(lines.y, 0...Self.maxVertices, "lines", "coordinate curves")
            if let why = Self.vertices(coordinateCurves: lines, resolution: lineResolution) {
                throw refuse("lines", why)
            }
        }
        if let w = linearCurves {
            if let why = vertices(linearCurves: w) { throw refuse("linearCurves", why) }
        }
        for (width, key) in [(style.lines, "style.lines"), (style.trajectory, "style.trajectory"),
                             (style.folds, "style.folds"), (style.linearCurves, "style.linearCurves")] {
            guard width.isFinite, width >= 0 else { throw refuse(key, "must be a number not below 0, not \(width)") }
        }
    }

    /// The most samples a surface's grid holds: 8192 round the ring by 4096
    /// round the tube.
    static let maxSamples = 1 << 25
    /// The most vertices one layer's curves hold.
    static let maxVertices = 1 << 24
    /// The most samples round the ring, and across a function's rectangle.
    static let maxGrid = 1 << 16, maxFunctionGrid = 4096
    /// The longest run a forced torus is fitted from, and its most harmonics.
    static let maxFit = 100_000.0, maxHarmonics = 64
    /// The most periods a relief repeats each way.
    static let maxRepeats = 16
    /// A relief's amplitudes. Not below 0: displaced inward past its
    /// reach, the torus would pass through itself while the relief still
    /// claimed to bound a solid (`PeriodicTorus.Drawing.relief`). At most 1,
    /// a displacement of at most the truncation height, 3. A relief past its
    /// reach is drawn with its back faces kept, so the bound is on cost, not
    /// on embedding.
    static let reliefRange = 0.0...1.0

    /// At most how many samples the surface's grid has round the ring and
    /// round the tube: exactly, but for sn, whose rectangle's aspect its
    /// modulus sets, where this takes the aspect at most 1.
    var sampleBound: (u: Int, v: Int) {
        func clamped(_ x: Double) -> Int { Int(min(max(x, 0), Double(Int32.max))) }
        // A periodic torus sizes its tube to keep the cells square
        // (`PeriodicTorus.torus(over:major:minor:grid:)`).
        func tube(_ grid: Int, aspect: Double, _ R: Double, _ r: Double) -> Int {
            max(clamped(Double(grid) * min(aspect, 1) * r / R), 32)
        }
        switch shape {
        case .torus: return (grid, max(grid / 2, 8))
        case .forced: return (grid, grid / 2)
        case .sn(_, let R, let r, _): return (grid, tube(grid, aspect: 1, R, r))
        case .periodic(_, let p, let R, let r, _): return (grid, tube(grid, aspect: p.y / p.x, R, r))
        case .relief(_, let R, let r, _, let k, _, _, _, _):
            let base = min(grid, 1024)
            return (min(clamped(Double(base) * Double(k.x)), grid),
                    max(clamped(Double(tube(base, aspect: 1, R, r)) * Double(k.y)), 32))
        }
    }

    /// How many vertices a coordinate curve has along its axis: four to a
    /// grid cell, two on a fitted torus, where each is the Fourier series.
    var lineResolution: Int {
        if case .forced = shape { return 2 * grid }
        return 4 * grid
    }

    /// Why coordinate curves at `counts` would hold more vertices than a
    /// layer does, or nil.
    static func vertices(coordinateCurves counts: SIMD2<Int>, resolution: Int) -> String? {
        let vertices = (Double(counts.x) + Double(counts.y)) * Double(resolution + 1)
        guard vertices > Double(maxVertices) else { return nil }
        return "would take " + String(format: "%.3g", vertices) + " vertices for \(counts.x) + \(counts.y) "
            + "coordinate curves, more than the \(maxVertices) a layer holds"
    }

    /// Why a linear curve cannot be drawn on this surface, or nil: a slope
    /// that is not a number, no turn or no strand, or more vertices than a
    /// layer holds, at a vertex per grid cell along the steeper axis
    /// (`ParametricSurface.linearCurves`).
    func vertices(linearCurves w: LinearCurves) -> String? {
        guard w.slope.isFinite else { return "wants a finite slope, not \(w.slope)" }
        guard (1...Self.maxVertices).contains(w.turns), (1...Self.maxVertices).contains(w.count) else {
            return "wants at least one turn of one strand, and at most \(Self.maxVertices), "
                + "not \(w.turns) turns of \(w.count)"
        }
        let samples = sampleBound
        let steps = max(Double(samples.u), (abs(w.slope) * Double(samples.v)).rounded(.up))
        let vertices = Double(w.count) * (Double(w.closes ?? w.turns) * steps + 1)
        guard vertices > Double(Self.maxVertices) else { return nil }
        return "would take " + String(format: "%.3g", vertices) + " vertices, more than the "
            + "\(Self.maxVertices) a layer holds"
    }
}
