import Foundation

/// The Swift mirror of `kurven/bundle.py`.
///
/// One source of truth for the schema (the Python dataclasses) and one mirror
/// kept honest by a fixture, not by codegen: `tests/fixtures/contract/*.kurven`
/// are decoded here and re-encoded, and the canonical JSON must come back byte
/// for byte. A mirror that has drifted fails that test on the first run rather
/// than producing a landscape that is subtly wrong.
///
/// Every sum type here is an `enum` with an exhaustive `switch` at each use
/// site, so "a cap kind nobody handled" is a compile error rather than a
/// silently ignored branch.
public enum ManifestError: Error, CustomStringConvertible, Equatable {
    case unsupportedSchema(Int, supported: Int)
    case unexpectedAxes([String], expected: [String])
    case unknownKind(String, of: String, known: [String])
    case badShape(String, [Int])
    case noSuchPreset(String, have: [String])
    case noSuchLayer(String, have: [String])

    public var description: String {
        switch self {
        case .unsupportedSchema(let s, let k):
            "manifest: schema \(s) is not the supported \(k)"
        case .unexpectedAxes(let a, let e):
            "manifest: axes \(a) are not \(e); this bundle is in a different coordinate order"
        case .unknownKind(let k, let of, let known):
            "manifest: \(of) kind '\(k)' is not one of \(known)"
        case .badShape(let what, let s): "manifest: \(what) has shape \(s)"
        case .noSuchPreset(let n, let have): "manifest: no preset '\(n)' (have \(have))"
        case .noSuchLayer(let n, let have): "manifest: no layer '\(n)' (have \(have))"
        }
    }
}

public let kurvenSchema = 1
/// The axis convention every bundle states about itself. A bundle that says
/// anything else is not readable by this code, and saying so up front is the
/// whole point of writing it into the file.
public let kurvenAxes = ["real", "imag", "magnitude"]

// MARK: - leaves

public struct Interval: Sendable, Equatable {
    public var lo: Double
    public var hi: Double
    public init(lo: Double, hi: Double) { self.lo = lo; self.hi = hi }
    public var length: Double { hi - lo }

    init(json: JSONValue) throws {
        let o = try json.object("Interval")
        self.init(lo: try o.double("lo", "Interval"), hi: try o.double("hi", "Interval"))
    }
    var json: JSONValue { .object(["lo": .double(lo), "hi": .double(hi)]) }
}

public struct Domain: Sendable, Equatable {
    public var real: Interval
    public var imag: Interval
    public init(real: Interval, imag: Interval) { self.real = real; self.imag = imag }

    init(json: JSONValue) throws {
        let o = try json.object("Domain")
        self.init(real: try Interval(json: o.value("real", "Domain")),
                  imag: try Interval(json: o.value("imag", "Domain")))
    }
    var json: JSONValue { .object(["real": real.json, "imag": imag.json]) }
}

public struct GridRef: Sendable, Equatable {
    public var file: String
    /// `(ny, nx)` -- rows index imag, columns index real.
    public var shape: (ny: Int, nx: Int)
    public var dtype: NPYDType

    public init(file: String, shape: (ny: Int, nx: Int), dtype: NPYDType) {
        self.file = file; self.shape = shape; self.dtype = dtype
    }
    public static func == (a: GridRef, b: GridRef) -> Bool {
        a.file == b.file && a.shape == b.shape && a.dtype == b.dtype
    }

    init(json: JSONValue) throws {
        let o = try json.object("GridRef")
        let s = try o.ints("shape", "GridRef")
        guard s.count == 2 else { throw ManifestError.badShape("GridRef", s) }
        let d = try o.string("dtype", "GridRef")
        guard let dtype = NPYDType(rawValue: d) else {
            throw NPYError.unsupportedDType(d, expected: NPYDType.allCases.map(\.rawValue))
        }
        self.init(file: try o.string("file", "GridRef"), shape: (s[0], s[1]), dtype: dtype)
    }
    var json: JSONValue {
        .object(["file": .string(file), "dtype": .string(dtype.rawValue),
                 "shape": .array([.int(shape.ny), .int(shape.nx)])])
    }
}

/// A 2x3 affine map on the world (x, y) plane. Elliptic's eighteen reflected
/// copies are eighteen of these, and the base tile is the identity.
public struct Affine2: Sendable, Equatable {
    public var a, b, tx, c, d, ty: Double

    public init(a: Double, b: Double, tx: Double, c: Double, d: Double, ty: Double) {
        self.a = a; self.b = b; self.tx = tx; self.c = c; self.d = d; self.ty = ty
    }
    public static let identity = Affine2(a: 1, b: 0, tx: 0, c: 0, d: 1, ty: 0)

    public func callAsFunction(_ p: P3<WorldSpace>) -> P3<WorldSpace> {
        P3(a * p.x + b * p.y + tx, c * p.x + d * p.y + ty, p.z)
    }

    /// The map back. `nil` for a degenerate tile, which a manifest should not
    /// contain and which this refuses to guess at.
    public var inverse: Affine2? {
        let det = a * d - b * c
        guard det != 0, det.isFinite else { return nil }
        let (ia, ib) = (d / det, -b / det)
        let (ic, id) = (-c / det, a / det)
        return Affine2(a: ia, b: ib, tx: -(ia * tx + ib * ty),
                       c: ic, d: id, ty: -(ic * tx + id * ty))
    }

    /// As a `Transform`, for folding into a camera or handing to a shader.
    public var transform: Transform<WorldSpace, WorldSpace> {
        Transform(rows: (SIMD3(a, b, 0), SIMD3(c, d, 0), SIMD3(0, 0, 1)),
                  translation: SIMD3(tx, ty, 0))
    }

    init(json: JSONValue) throws {
        let o = try json.object("Affine2")
        let m = try o.doubles("m", "Affine2")
        guard m.count == 6 else { throw ManifestError.badShape("Affine2", [m.count]) }
        self.init(a: m[0], b: m[1], tx: m[2], c: m[3], d: m[4], ty: m[5])
    }
    var json: JSONValue {
        .object(["m": .array([a, b, tx, c, d, ty].map(JSONValue.double))])
    }
}

public struct PerimeterEdge: Sendable, Equatable {
    public var start: P2<WorldSpace>
    public var end: P2<WorldSpace>
    public var density: Int

    public init(start: P2<WorldSpace>, end: P2<WorldSpace>, density: Int) {
        self.start = start; self.end = end; self.density = density
    }

    init(json: JSONValue) throws {
        let o = try json.object("Edge")
        let s = try o.doubles("start", "Edge"), e = try o.doubles("end", "Edge")
        guard s.count == 2, e.count == 2 else {
            throw ManifestError.badShape("Edge", [s.count, e.count])
        }
        self.init(start: P2(s[0], s[1]), end: P2(e[0], e[1]),
                  density: try o.int("density", "Edge"))
    }
    var json: JSONValue {
        .object(["start": .array([.double(start.x), .double(start.y)]),
                 "end": .array([.double(end.x), .double(end.y)]),
                 "density": .int(density)])
    }
}

public struct BoundaryPerimeter: Sendable, Equatable {
    public var edges: [PerimeterEdge]
    public init(edges: [PerimeterEdge]) { self.edges = edges }

    init(json: JSONValue) throws {
        let o = try json.object("Perimeter")
        self.init(edges: try o.array("edges", "Perimeter").map(PerimeterEdge.init(json:)))
    }
    var json: JSONValue { .object(["edges": .array(edges.map(\.json))]) }
}

// MARK: - caps

public struct RealBand: Sendable, Equatable {
    public var below: Double
    public var cap: Double
    public init(below: Double, cap: Double) { self.below = below; self.cap = cap }

    init(json: JSONValue) throws {
        let o = try json.object("RealBand")
        self.init(below: try o.double("below", "RealBand"), cap: try o.double("cap", "RealBand"))
    }
    var json: JSONValue { .object(["below": .double(below), "cap": .double(cap)]) }
}

/// How the surface is truncated from above.
///
/// A cap is a property of the model, not the camera: it is what turns a pole
/// into a visible plateau. The Python `Projection` conflated the two (zeta put
/// its clamp in the camera); here they are separate types and cannot be.
public enum Caps: Sendable, Equatable {
    case none
    case uniform(Double)
    case realBands([RealBand])

    /// The cap at world x (= Re z); `.infinity` where there is none.
    public func height(atX x: Double) -> Double {
        switch self {
        case .none: .infinity
        case .uniform(let z): z
        case .realBands(let bands):
            bands.first(where: { x < $0.below })?.cap ?? .infinity
        }
    }

    init(json: JSONValue) throws {
        let o = try json.object("Caps")
        switch try o.string("kind", "Caps") {
        case "none": self = .none
        case "uniform": self = .uniform(try o.double("z", "Caps.uniform"))
        case "realBands":
            self = .realBands(try o.array("bands", "Caps.realBands").map(RealBand.init(json:)))
        case let other:
            throw ManifestError.unknownKind(other, of: "Caps",
                                            known: ["none", "uniform", "realBands"])
        }
    }
    var json: JSONValue {
        switch self {
        case .none: .object(["kind": .string("none")])
        case .uniform(let z): .object(["kind": .string("uniform"), "z": .double(z)])
        case .realBands(let b):
            .object(["kind": .string("realBands"), "bands": .array(b.map(\.json))])
        }
    }
}

// MARK: - occluder

/// The cut-face curtains: dumped as a mesh, or described as a perimeter.
///
/// A dumped mesh has its crests at the *analytic* surface height, which is what
/// the Python plates draw. A derived one re-evaluates the crest by bilinear
/// lookup on `height.npy`, so it differs by the grid's interpolation error --
/// a difference worth measuring rather than assuming away.
public enum Walls: Sendable, Equatable {
    case none
    case mesh(vertices: String, triangles: String)
    case perimeter(BoundaryPerimeter, base: Double)

    init(json: JSONValue) throws {
        let o = try json.object("Walls")
        switch try o.string("kind", "Walls") {
        case "none": self = .none
        case "mesh":
            self = .mesh(vertices: try o.string("vertices", "Walls.mesh"),
                         triangles: try o.string("triangles", "Walls.mesh"))
        case "perimeter":
            self = .perimeter(try BoundaryPerimeter(json: o.value("perimeter", "Walls.perimeter")),
                              base: try o.double("base", "Walls.perimeter"))
        case let other:
            throw ManifestError.unknownKind(other, of: "Walls",
                                            known: ["none", "mesh", "perimeter"])
        }
    }
    var json: JSONValue {
        switch self {
        case .none: .object(["kind": .string("none")])
        case .mesh(let v, let t):
            .object(["kind": .string("mesh"), "vertices": .string(v), "triangles": .string(t)])
        case .perimeter(let p, let base):
            .object(["kind": .string("perimeter"), "perimeter": p.json, "base": .double(base)])
        }
    }
}

/// The footprint the heightfield is rasterized over.
///
/// zeta's landscape is a staircase, not a rectangle: the notch in front of the
/// s = 1 pole is where the plate cuts away to show the cross-section. It is an
/// absence of geometry, not a mask applied afterwards, so a cell contributes
/// triangles only when all four of its corners are inside -- no bridging, and
/// nothing to draw over.
public enum Region: Sendable, Equatable {
    case full
    case inside(BoundaryPerimeter)

    /// Whether a world point is inside. `.full` admits everything.
    public func contains(_ p: P2<WorldSpace>) -> Bool {
        switch self {
        case .full: return true
        case .inside(let perimeter):
            return pointInPolygon(p, perimeter.edges.map(\.start))
        }
    }

    init(json: JSONValue) throws {
        let o = try json.object("Region")
        switch try o.string("kind", "Region") {
        case "full": self = .full
        case "inside":
            self = .inside(try BoundaryPerimeter(json: o.value("perimeter", "Region.inside")))
        case let other:
            throw ManifestError.unknownKind(other, of: "Region", known: ["full", "inside"])
        }
    }
    var json: JSONValue {
        switch self {
        case .full: .object(["kind": .string("full")])
        case .inside(let p): .object(["kind": .string("inside"), "perimeter": p.json])
        }
    }
}

/// Even-odd ray crossing, scanning along world x (= real).
///
/// The scan axis is part of the contract, not an implementation detail: a test
/// that rays along the other axis agrees everywhere except on the boundary, and
/// the boundary is exactly where a staircase cutout lives. `kurven.bundle`'s
/// `point_in_polygon` is this function, and both `Perimeter` types on the Python
/// side delegate to it for the same reason.
public func pointInPolygon(_ p: P2<WorldSpace>, _ corners: [P2<WorldSpace>]) -> Bool {
    var inside = false
    for i in corners.indices {
        let a = corners[i], b = corners[(i + 1) % corners.count]
        if a.x == b.x { continue }
        let crossing = (b.y - a.y) * (p.x - a.x) / (b.x - a.x) + a.y
        if (a.x > p.x) != (b.x > p.x), p.y < crossing { inside.toggle() }
    }
    return inside
}

/// What blocks sight lines.
///
/// The heightfield is never triangles on disk: it is `height.npy` plus a `step`,
/// instanced once per entry in `tiles`. A renderer rasterizes it from the
/// texture at whatever detail it is drawing at, which is the only way the
/// elliptic plate's nineteen tiles and gamma's 20000-square bake are affordable.
///
/// `tiles` always contains the identity as its first entry, and `region` is the
/// footprint each tile is clipped to.
public struct Occluder: Sendable, Equatable {
    public var step: Int
    public var tiles: [Affine2]
    public var walls: Walls
    public var region: Region
    public var base: Double

    public init(step: Int, tiles: [Affine2], walls: Walls,
                region: Region = .full, base: Double) {
        self.step = step; self.tiles = tiles; self.walls = walls
        self.region = region; self.base = base
    }

    init(json: JSONValue) throws {
        let o = try json.object("Occluder")
        self.init(step: try o.int("step", "Occluder"),
                  tiles: try o.array("tiles", "Occluder").map(Affine2.init(json:)),
                  walls: try Walls(json: o.value("walls", "Occluder")),
                  region: try Region(json: o.value("region", "Occluder")),
                  base: o.optionalDouble("base") ?? 0)
    }
    var json: JSONValue {
        .object(["step": .int(step), "tiles": .array(tiles.map(\.json)),
                 "walls": walls.json, "region": region.json, "base": .double(base)])
    }
}

// MARK: - layers

public enum LayerRole: String, Sendable, CaseIterable {
    case magnitude, phase, scaffold, outline
}

/// How a layer's z was chosen. Recorded so a consumer that re-contours the grid
/// itself can reproduce the lift rather than guess it.
public enum HeightPolicy: String, Sendable, CaseIterable {
    case surface, level, magnitude
}

/// Which grid a derived layer contours.
public enum ContourField: String, Sendable, CaseIterable {
    case magnitude, phase
}

public enum KeepAxis: String, Sendable, CaseIterable {
    case real, imag
}

/// Which vertices of a derived contour survive.
///
/// A small closed vocabulary, not an expression language. Everything the plates
/// ask for is here -- stay inside the occluder's footprint, stay under the cap,
/// stay within a band of one axis -- and it can be evaluated without an
/// interpreter. Ink that needs more than this is ink that should be dumped, and
/// `LayerSource.file` is how the schema says so.
public indirect enum Keep: Sendable, Equatable {
    case all
    /// Inside `occluder.region`: the same polygon the heightfield is cut to,
    /// referenced rather than repeated.
    case region
    case belowCap
    case band(axis: KeepAxis, lo: Double, hi: Double)
    case every([Keep])

    init(json: JSONValue) throws {
        let o = try json.object("Keep")
        switch try o.string("kind", "Keep") {
        case "all": self = .all
        case "region": self = .region
        case "belowCap": self = .belowCap
        case "band":
            let axisText = try o.string("axis", "Keep.band")
            guard let axis = KeepAxis(rawValue: axisText) else {
                throw ManifestError.unknownKind(axisText, of: "Keep.band axis",
                                                known: KeepAxis.allCases.map(\.rawValue))
            }
            self = .band(axis: axis, lo: try o.double("lo", "Keep.band"),
                         hi: try o.double("hi", "Keep.band"))
        case "every":
            self = .every(try o.array("of", "Keep.every").map(Keep.init(json:)))
        case let other:
            throw ManifestError.unknownKind(other, of: "Keep",
                                            known: ["all", "region", "belowCap",
                                                    "band", "every"])
        }
    }
    var json: JSONValue {
        switch self {
        case .all: .object(["kind": .string("all")])
        case .region: .object(["kind": .string("region")])
        case .belowCap: .object(["kind": .string("belowCap")])
        case .band(let axis, let lo, let hi):
            .object(["kind": .string("band"), "axis": .string(axis.rawValue),
                     "lo": .double(lo), "hi": .double(hi)])
        case .every(let of):
            .object(["kind": .string("every"), "of": .array(of.map(\.json))])
        }
    }
}

/// Where a layer's geometry comes from: a file, or a description.
///
/// A dumped layer is the answer; a described one is the question. Describing it
/// is what lets a consumer move the levels and get a new answer, and it is what
/// shrinks a bundle from megabytes of vertices to a list of numbers.
public enum LayerSource: Sendable, Equatable {
    case file(vertices: String, offsets: String)
    /// Iso-lines of `field` at `levels`, lifted by the layer's height policy,
    /// filtered by `keep`, and replicated once per occluder tile when `tiled`.
    case contour(field: ContourField, levels: [Double], keep: Keep, tiled: Bool)

    init(json: JSONValue) throws {
        let o = try json.object("LayerSource")
        switch try o.string("kind", "LayerSource") {
        case "file":
            self = .file(vertices: try o.string("vertices", "LayerSource.file"),
                         offsets: try o.string("offsets", "LayerSource.file"))
        case "contour":
            let fieldText = try o.string("field", "LayerSource.contour")
            guard let field = ContourField(rawValue: fieldText) else {
                throw ManifestError.unknownKind(fieldText, of: "ContourField",
                                                known: ContourField.allCases.map(\.rawValue))
            }
            self = .contour(field: field,
                            levels: try o.doubles("levels", "LayerSource.contour"),
                            keep: try Keep(json: o.value("keep", "LayerSource.contour")),
                            tiled: try o.bool("tiled", "LayerSource.contour", default: false))
        case let other:
            throw ManifestError.unknownKind(other, of: "LayerSource",
                                            known: ["file", "contour"])
        }
    }
    var json: JSONValue {
        switch self {
        case .file(let v, let o):
            .object(["kind": .string("file"), "vertices": .string(v), "offsets": .string(o)])
        case .contour(let field, let levels, let keep, let tiled):
            .object(["kind": .string("contour"), "field": .string(field.rawValue),
                     "levels": .array(levels.map(JSONValue.double)),
                     "keep": keep.json, "tiled": .bool(tiled)])
        }
    }
}

public struct LayerSpec: Sendable, Equatable {
    public var name: String
    public var role: LayerRole
    public var source: LayerSource
    /// A matplotlib line width in points, carried into the SVG stroke width.
    public var width: Double
    public var heightPolicy: HeightPolicy
    public var color: String
    /// False for ink that lies *in* the occluding geometry -- a cut-face hatch,
    /// which a depth test would half erase.
    public var clipped: Bool

    public init(name: String, role: LayerRole, source: LayerSource,
                width: Double, heightPolicy: HeightPolicy,
                color: String = "#000000", clipped: Bool = true) {
        self.name = name; self.role = role; self.source = source
        self.width = width; self.heightPolicy = heightPolicy
        self.color = color; self.clipped = clipped
    }

    /// `(vertices, offsets)` when this layer is dumped, else nil.
    public var files: (vertices: String, offsets: String)? {
        if case .file(let v, let o) = source { return (v, o) }
        return nil
    }

    init(json: JSONValue) throws {
        let o = try json.object("LayerSpec")
        let roleText = try o.string("role", "LayerSpec")
        guard let role = LayerRole(rawValue: roleText) else {
            throw ManifestError.unknownKind(roleText, of: "LayerRole",
                                            known: LayerRole.allCases.map(\.rawValue))
        }
        let policyText = try o.string("heightPolicy", "LayerSpec")
        guard let policy = HeightPolicy(rawValue: policyText) else {
            throw ManifestError.unknownKind(policyText, of: "HeightPolicy",
                                            known: HeightPolicy.allCases.map(\.rawValue))
        }
        self.init(name: try o.string("name", "LayerSpec"), role: role,
                  source: try LayerSource(json: o.value("source", "LayerSpec")),
                  width: try o.double("width", "LayerSpec"),
                  heightPolicy: policy,
                  color: (try? o.string("color", "LayerSpec")) ?? "#000000",
                  clipped: try o.bool("clipped", "LayerSpec", default: true))
    }
    var json: JSONValue {
        .object(["name": .string(name), "role": .string(role.rawValue),
                 "source": source.json,
                 "width": .double(width), "heightPolicy": .string(heightPolicy.rawValue),
                 "color": .string(color), "clipped": .bool(clipped)])
    }
}

// MARK: - camera

/// The Python `Projection` parameter set, verbatim.
///
/// Column 0 of the array `Projection.apply` consumes is imag and column 1 is
/// real, so `flipX` flips imag and `yScale` scales real. `Camera.plate` folds
/// the axis exchange into one matrix; these numbers cross unchanged so the two
/// cameras can be compared numerically instead of by inspection.
public struct PlateProjection: Sendable, Equatable {
    public var shear: Double
    public var xAngle: Double
    public var zAngle: Double
    public var flipX: Bool
    public var yScale: Double?

    public init(shear: Double, xAngle: Double, zAngle: Double,
                flipX: Bool, yScale: Double?) {
        self.shear = shear; self.xAngle = xAngle; self.zAngle = zAngle
        self.flipX = flipX; self.yScale = yScale
    }

    public init(json: JSONValue) throws {
        let o = try json.object("PlateProjection")
        self.init(shear: try o.double("shear", "PlateProjection"),
                  xAngle: try o.double("xAngle", "PlateProjection"),
                  zAngle: try o.double("zAngle", "PlateProjection"),
                  flipX: try o.bool("flipX", "PlateProjection"),
                  yScale: o.optionalDouble("yScale"))
    }
    public var json: JSONValue {
        .object(["shear": .double(shear), "xAngle": .double(xAngle),
                 "zAngle": .double(zAngle), "flipX": .bool(flipX),
                 "yScale": yScale.map(JSONValue.double) ?? .null])
    }
}

public struct CameraPreset: Sendable, Equatable {
    public var name: String
    public var plate: PlateProjection
    public var margin: Double
    public var buffer: Int

    public init(name: String, plate: PlateProjection, margin: Double, buffer: Int) {
        self.name = name; self.plate = plate; self.margin = margin; self.buffer = buffer
    }

    init(json: JSONValue) throws {
        let o = try json.object("CameraPreset")
        self.init(name: try o.string("name", "CameraPreset"),
                  plate: try PlateProjection(json: o.value("plate", "CameraPreset")),
                  margin: try o.double("margin", "CameraPreset"),
                  buffer: try o.int("buffer", "CameraPreset"))
    }
    var json: JSONValue {
        .object(["name": .string(name), "plate": plate.json,
                 "margin": .double(margin), "buffer": .int(buffer)])
    }
}

public struct Provenance: Sendable, Equatable {
    public var function: String
    /// The `kurven.export` example that produced this, when one did. `function`
    /// names the mathematics; this names the recipe, and it is what lets a
    /// consumer ask the service for the same landscape at another resolution.
    public var example: String
    public var params: [String: JSONValue]
    /// The contourpy chunk count the bundle was written with. Anything but 1
    /// means the contours were stitched in thread-completion order and the
    /// bundle is not reproducible -- worth knowing before blaming a diff on
    /// anything else.
    public var cpuCount: Int
    public var gitSha: String

    init(json: JSONValue) throws {
        let o = try json.object("Provenance")
        self.function = try o.string("function", "Provenance")
        self.example = (try? o.string("example", "Provenance")) ?? ""
        self.params = try o.value("params", "Provenance").object("Provenance.params")
        self.cpuCount = try o.int("cpuCount", "Provenance")
        self.gitSha = try o.string("gitSha", "Provenance")
    }
    var json: JSONValue {
        .object(["function": .string(function), "example": .string(example),
                 "params": .object(params),
                 "cpuCount": .int(cpuCount), "gitSha": .string(gitSha)])
    }

    public var isReproducible: Bool { cpuCount == 1 }
}

// MARK: - the manifest

public struct Manifest: Sendable, Equatable {
    public var schema: Int
    public var axes: [String]
    public var domain: Domain
    public var height: GridRef
    public var phase: GridRef?
    public var caps: Caps
    public var occluder: Occluder
    public var layers: [LayerSpec]
    public var presets: [CameraPreset]
    public var provenance: Provenance

    public init(json: JSONValue) throws {
        let o = try json.object("Manifest")
        schema = try o.int("schema", "Manifest")
        guard schema == kurvenSchema else {
            throw ManifestError.unsupportedSchema(schema, supported: kurvenSchema)
        }
        axes = try o.strings("axes", "Manifest")
        guard axes == kurvenAxes else {
            throw ManifestError.unexpectedAxes(axes, expected: kurvenAxes)
        }
        domain = try Domain(json: o.value("domain", "Manifest"))
        height = try GridRef(json: o.value("height", "Manifest"))
        phase = if case .some(.object) = o["phase"] {
            try GridRef(json: o.value("phase", "Manifest"))
        } else { nil }
        caps = try Caps(json: o.value("caps", "Manifest"))
        occluder = try Occluder(json: o.value("occluder", "Manifest"))
        layers = try o.array("layers", "Manifest").map(LayerSpec.init(json:))
        presets = try o.array("presets", "Manifest").map(CameraPreset.init(json:))
        provenance = try Provenance(json: o.value("provenance", "Manifest"))
    }

    public static func read(contentsOf url: URL) throws -> Manifest {
        try Manifest(json: JSONValue.parse(Data(contentsOf: url)))
    }

    public var json: JSONValue {
        .object(["schema": .int(schema), "axes": .array(axes.map(JSONValue.string)),
                 "domain": domain.json, "height": height.json,
                 "phase": phase.map(\.json) ?? .null, "caps": caps.json,
                 "occluder": occluder.json,
                 "layers": .array(layers.map(\.json)),
                 "presets": .array(presets.map(\.json)),
                 "provenance": provenance.json])
    }

    /// Canonical JSON with the trailing newline `Manifest.to_json` writes, so
    /// the contract test compares whole files.
    public var canonicalJSON: String { json.canonical + "\n" }

    public func preset(_ name: String) throws -> CameraPreset {
        guard let p = presets.first(where: { $0.name == name }) else {
            throw ManifestError.noSuchPreset(name, have: presets.map(\.name))
        }
        return p
    }

    public func layer(_ name: String) throws -> LayerSpec {
        guard let l = layers.first(where: { $0.name == name }) else {
            throw ManifestError.noSuchLayer(name, have: layers.map(\.name))
        }
        return l
    }
}
