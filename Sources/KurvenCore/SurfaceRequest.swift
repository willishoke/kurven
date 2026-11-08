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
    }

    /// The fields this request's shape has, in the order a panel lists them.
    /// The linear curve's are listed by `linearCurveFields` when there is one.
    var fields: [Field] {
        switch shape {
        case .torus: [.major, .minor]
        case .sn, .relief: [.modulus, .major, .minor, .grid]
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
            case .relief(var m, var R, var r, let a, let k, var g, let turn, let floor, let soft):
                switch field {
                case .modulus: m = x
                case .major: R = x
                case .minor: r = x
                case .grid: g = n
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
        let shape: [String: JSONValue]
        switch self.shape {
        case .torus(let R, let r):
            shape = ["kind": .string("torus"), "major": .double(R), "minor": .double(r)]
        case .sn(let m, let R, let r, let grid):
            shape = ["kind": .string("sn"), "modulus": .double(m), "major": .double(R), "minor": .double(r),
                     "grid": .int(grid)]
        case .periodic(let e, let p, let R, let r, let grid):
            shape = ["kind": .string("periodic"), "expression": .string(e),
                     "periods": .array([.double(p.x), .double(p.y)]),
                     "major": .double(R), "minor": .double(r), "grid": .int(grid)]
        case .forced(let system, let fit, let harmonics, let radial, let axial, let radius, let turns):
            shape = ["kind": .string("forced"), "system": .string(system), "fit": .double(fit),
                     "harmonics": .int(harmonics), "radial": .int(radial), "axial": .int(axial),
                     "radius": .double(radius), "turns": .int(turns)]
        case .relief(let m, let R, let r, let a, let k, let grid, let turn, let floor, let soft):
            shape = ["kind": .string("relief"), "modulus": .double(m), "major": .double(R), "minor": .double(r),
                     "amplitude": .double(a), "repeats": .array([.int(k.x), .int(k.y)]), "grid": .int(grid),
                     "turn": .double(turn), "floor": floor.map(JSONValue.double) ?? .null, "soft": .bool(soft)]
        }
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

    /// Refuse a request no drawing can be built from, naming the number:
    /// a radius that is not positive, a modulus outside (0, 1), a grid too
    /// coarse to span its rectangle, a forced torus revolved on one state
    /// component twice. What only building can find -- a system the catalog
    /// does not have, an expression that does not parse -- `SurfaceDrawing.build`
    /// refuses.
    func check() throws {
        func refuse(_ why: String) -> ManifestError { .invalid("a surface's \(why)") }
        func positive(_ x: Double, _ name: String) throws {
            guard x.isFinite, x > 0 else { throw refuse("\(name) must be a positive number, not \(x)") }
        }
        func finite(_ x: Double, _ name: String) throws {
            guard x.isFinite else { throw refuse("\(name) must be a finite number, not \(x)") }
        }
        func modulus(_ m: Double) throws {
            guard m.isFinite, m > 0, m < 1 else { throw refuse("modulus must lie strictly between 0 and 1, not \(m)") }
        }
        func samples(_ n: Int, _ name: String) throws {
            guard n >= 8 else { throw refuse("\(name) must be at least 8 samples, not \(n)") }
        }
        try samples(grid, "grid")
        switch shape {
        case .torus(let R, let r):
            try positive(R, "major radius"); try positive(r, "minor radius")
        case .sn(let m, let R, let r, let g):
            try modulus(m); try positive(R, "major radius"); try positive(r, "minor radius")
            try samples(g, "sn's grid")
        case .periodic(let e, let p, let R, let r, let g):
            guard !e.trimmingCharacters(in: .whitespaces).isEmpty else { throw refuse("expression is empty") }
            try positive(p.x, "real period"); try positive(p.y, "imaginary period")
            try positive(R, "major radius"); try positive(r, "minor radius")
            try samples(g, "function's grid")
        case .forced(let system, let fit, let harmonics, let radial, let axial, let radius, let turns):
            guard !system.isEmpty else { throw refuse("forced system is unnamed") }
            try positive(fit, "fit length"); try positive(radius, "ring radius")
            guard harmonics >= 1 else { throw refuse("harmonics must be at least 1, not \(harmonics)") }
            guard (0...2).contains(radial), (0...2).contains(axial) else {
                throw refuse("state components are x, x′ and x″, numbered 0 to 2, not \(radial) and \(axial)")
            }
            guard radial != axial else {
                throw refuse("radius and height are both state component \(radial): one component as both is a curve, not a torus")
            }
            guard turns >= 1 else { throw refuse("trajectory must run at least one turn, not \(turns)") }
        case .relief(let m, let R, let r, let a, let k, let g, let turn, let floor, _):
            try modulus(m); try positive(R, "major radius"); try positive(r, "minor radius")
            try finite(a, "relief amplitude"); try finite(turn, "turn")
            if let floor { try finite(floor, "relief floor") }
            guard k.x >= 1, k.y >= 1 else { throw refuse("repeats must be whole periods, at least 1 each way, not \(k.x) and \(k.y)") }
            try samples(g, "sn's grid")
        }
        if let lines, lines.x < 0 || lines.y < 0 {
            throw refuse("coordinate curves must be counts, not \(lines.x) and \(lines.y)")
        }
        if let w = linearCurves {
            try finite(w.slope, "linear curve's slope")
            guard w.turns >= 1, w.count >= 1 else {
                throw refuse("linear curve wants at least one turn of one strand, not \(w.turns) of \(w.count)")
            }
        }
        for (width, name) in [(style.lines, "lines"), (style.trajectory, "trajectory"), (style.folds, "folds"),
                              (style.linearCurves, "linear curves")] {
            guard width.isFinite, width >= 0 else { throw refuse("\(name) width must be a number not below 0, not \(width)") }
        }
    }
}
