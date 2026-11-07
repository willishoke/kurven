import Foundation
import simd

/// The bundle schema: the one definition of it.
///
/// It began as the mirror of the Python package's dataclasses and is now the
/// only copy. `tests/fixtures/contract/*.kurven` are decoded here and
/// re-encoded, and the canonical JSON must come back byte for byte, so a
/// change to the schema is a change to those golden files, made on purpose.
///
/// Every sum type here is an `enum` with an exhaustive `switch` at each use
/// site, so "a truncation kind nobody handled" is a compile error rather than a
/// silently ignored branch.
public enum ManifestError: Error, CustomStringConvertible, Equatable {
    case unsupportedSchema(Int, supported: Int)
    case unexpectedAxes([String], expected: [String])
    case unknownKind(String, of: String, known: [String])
    case badShape(String, [Int])
    case noSuchPreset(String, have: [String])
    case noSuchLayer(String, have: [String])
    case unsupported(String)

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
        case .unsupported(let what): "manifest: \(what)"
        }
    }
}

/// Schema 3 names the face a contour or a hatching lies on (`FaceSelector`),
/// risers among them, and contours a face's coordinates and height as well
/// as |f| and arg f. Schema 2 took the standard names (truncation,
/// restriction, step, coordinate curves, ...) and one hatching kind for
/// every face. Older schemas are refused, not translated: there are no
/// bundles outside this repository.
public let kurvenSchema = 3
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

    public init(json: JSONValue) throws {
        let o = try json.object("Interval")
        self.init(lo: try o.double("lo", "Interval"), hi: try o.double("hi", "Interval"))
    }
    public var json: JSONValue { .object(["lo": .double(lo), "hi": .double(hi)]) }
}

public struct Domain: Sendable, Equatable {
    public var real: Interval
    public var imag: Interval
    public init(real: Interval, imag: Interval) { self.real = real; self.imag = imag }

    public init(json: JSONValue) throws {
        let o = try json.object("Domain")
        self.init(real: try Interval(json: o.value("real", "Domain")),
                  imag: try Interval(json: o.value("imag", "Domain")))
    }
    public var json: JSONValue { .object(["real": real.json, "imag": imag.json]) }
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

    public init(json: JSONValue) throws {
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
    public var edges: [PerimeterEdge] { didSet { signedArea = Self.signedArea(of: edges) } }
    /// The area the edges' starts enclose, by the shoelace formula, signed:
    /// positive when the perimeter runs counter-clockwise seen from above,
    /// negative clockwise. Kept with the edges, so `inward(_:)` costs one
    /// edge.
    public private(set) var signedArea: Double
    public init(edges: [PerimeterEdge]) { self.edges = edges; signedArea = Self.signedArea(of: edges) }

    public init(json: JSONValue) throws {
        let o = try json.object("Perimeter")
        self.init(edges: try o.array("edges", "Perimeter").map(PerimeterEdge.init(json:)))
    }
    public var json: JSONValue { .object(["edges": .array(edges.map(\.json))]) }

    private static func signedArea(of edges: [PerimeterEdge]) -> Double {
        let corners = edges.map(\.start.v)
        return 0.5 * corners.indices.reduce(0.0) { a, k in
            let p = corners[k], q = corners[(k + 1) % corners.count]
            return a + p.x * q.y - q.x * p.y
        }
    }

    /// Edge `k`'s unit normal into the footprint the perimeter bounds, in
    /// the ground plane: to the left of the edge's direction when the
    /// perimeter runs counter-clockwise, to the right when it runs
    /// clockwise. Zero for an edge of no length.
    public func inward(_ k: Int) -> SIMD2<Double> {
        let d = edges[k].end.v - edges[k].start.v, length = simd_length(d)
        guard length > 0 else { return .zero }
        let left = SIMD2(-d.y, d.x) / length
        return signedArea < 0 ? -left : left
    }
}

// MARK: - truncation

public struct RealBand: Sendable, Equatable {
    /// The band is `x < below`, less the bands before it.
    public var below: Double
    public var height: Double
    public init(below: Double, height: Double) { self.below = below; self.height = height }

    init(json: JSONValue) throws {
        let o = try json.object("RealBand")
        self.init(below: try o.double("below", "RealBand"), height: try o.double("height", "RealBand"))
    }
    var json: JSONValue { .object(["below": .double(below), "height": .double(height)]) }
}

/// The truncation of the surface from above: the drawn surface is
/// `min(|f|, h(x))` for the truncation height `h`.
///
/// A truncation is a property of the model, not the camera: it is what turns
/// a pole into a visible plateau.
public enum Truncation: Sendable, Equatable {
    case none
    case uniform(Double)
    /// Gamma's per-spire truncation: a height per band of Re z. `beyond` is
    /// the height where no band matches -- a separate field rather than a
    /// final band with a huge threshold, because "everything else" is what
    /// it means.
    case realBands([RealBand], beyond: Double)

    /// The truncation height at world x (= Re z); `.infinity` where there is
    /// none. On a band boundary, the band that starts there: the truncation
    /// is right-continuous, and this is its limit from the right.
    public func height(atX x: Double) -> Double {
        height(atX: x, from: .right)
    }

    /// Which side of a point along x a one-sided limit approaches it from.
    public enum Side: Sendable, Equatable {
        /// From lesser x.
        case left
        /// From greater x.
        case right
    }

    /// The one-sided limit of the truncation height at x from `side`: the
    /// band just left of x or just right of it. The two differ only on a
    /// band boundary, where the band from the left is the one that ends
    /// there.
    public func height(atX x: Double, from side: Side) -> Double {
        switch self {
        case .none: .infinity
        case .uniform(let h): h
        case .realBands(let bands, let beyond):
            (side == .right ? bands.first { x < $0.below } : bands.first { x <= $0.below })?.height ?? beyond
        }
    }

    /// The band world x lies in: the index of the first listed band it is
    /// under, the count of them beyond the last, and 0 for a uniform
    /// truncation's one band. Nil where there is no truncation. On a band
    /// boundary, the band that starts there.
    public func band(atX x: Double) -> Int? {
        band(atX: x, from: .right)
    }

    /// The band x lies in as a one-sided limit from `side`, as
    /// `height(atX:from:)` reads it.
    public func band(atX x: Double, from side: Side) -> Int? {
        switch self {
        case .none: nil
        case .uniform: 0
        case .realBands(let bands, _):
            (side == .right ? bands.firstIndex { x < $0.below } : bands.firstIndex { x <= $0.below }) ?? bands.count
        }
    }

    /// One band's part of a stretch of Re z (`bands(from:to:)`).
    public struct BandSpan: Sendable, Equatable {
        /// The band, as `band(atX:)` numbers it.
        public let band: Int
        public let height: Double
        /// Where it holds the stretch: from `lo` up to, not including, `hi`.
        public let lo: Double, hi: Double
    }

    /// The bands that meet the stretch of Re z from `lo` to `hi`, in order
    /// along it, each with the part of the stretch it holds. The truncation is
    /// constant on each. Nothing where there is no truncation.
    public func bands(from lo: Double, to hi: Double) -> [BandSpan] {
        switch self {
        case .none:
            return []
        case .uniform(let h):
            return lo < hi ? [BandSpan(band: 0, height: h, lo: lo, hi: hi)] : []
        case .realBands(let bands, let beyond):
            // A band is x < below, less the bands before it: from the
            // greatest boundary before it to its own.
            var out: [BandSpan] = []
            var start = -Double.infinity
            for k in 0...bands.count {
                let end = k < bands.count ? bands[k].below : .infinity
                let a = max(start, lo), b = min(end, hi)
                if a < b { out.append(BandSpan(band: k, height: k < bands.count ? bands[k].height : beyond, lo: a, hi: b)) }
                if k < bands.count { start = max(start, end) }
            }
            return out.sorted { $0.lo < $1.lo }
        }
    }

    /// The greatest truncation height anywhere; `.infinity` where there is
    /// none.
    public var highest: Double {
        switch self {
        case .none: .infinity
        case .uniform(let h): h
        case .realBands(let bands, let beyond): bands.map(\.height).reduce(beyond, max)
        }
    }

    public init(json: JSONValue) throws {
        let o = try json.object("Truncation")
        switch try o.string("kind", "Truncation") {
        case "none": self = .none
        case "uniform": self = .uniform(try o.double("height", "Truncation.uniform"))
        case "realBands":
            self = .realBands(try o.array("bands", "Truncation.realBands").map(RealBand.init(json:)),
                              beyond: o.optionalDouble("beyond") ?? .infinity)
        case let other:
            throw ManifestError.unknownKind(other, of: "Truncation",
                                            known: ["none", "uniform", "realBands"])
        }
    }
    public var json: JSONValue {
        switch self {
        case .none: .object(["kind": .string("none")])
        case .uniform(let h): .object(["kind": .string("uniform"), "height": .double(h)])
        case .realBands(let b, let beyond):
            beyond.isFinite
                ? .object(["kind": .string("realBands"), "bands": .array(b.map(\.json)),
                           "beyond": .double(beyond)])
                : .object(["kind": .string("realBands"), "bands": .array(b.map(\.json))])
        }
    }
}

// MARK: - occluder

/// The cut-face curtains: dumped as a mesh, or described as a perimeter.
///
/// A dumped mesh has its top edges at the *analytic* surface height, which is what
/// the Python plates draw. A derived one re-evaluates the top edge by bilinear
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
/// s = 1 pole is where the drawing cuts away to show the cross-section. It is an
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

/// What blocks lines of sight.
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

/// Which field of its faces a contour layer is a level set of: |f| or arg f
/// on the graph, or on any face a coordinate of its parameter rectangle, `u`
/// and `v`, or its height. On the graph and a plateau u and v are Re z and
/// Im z, and the height |f| and h(x); on a wall and a riser, (s, z), the
/// distance along the line it stands on and the height, which is z again.
/// A parametric surface and a polyhedron face expose their own.
public enum ContourField: String, Sendable, CaseIterable {
    case magnitude, phase, u, v, height

    /// Whether this is one of f's fields, placed by the refiner.
    public var isFieldOfF: Bool { self == .magnitude || self == .phase }
}

/// An axis of the domain: the real or the imaginary.
public enum DomainAxis: String, Sendable, CaseIterable {
    case real, imag

    /// The angle of a line along the axis, in radians from the real axis.
    public var angle: Double { self == .real ? 0 : .pi / 2 }

    /// The axis a line at `angle` runs along, to within
    /// `Hatching.angleTolerance`, or nil when it runs along neither.
    public init?(angle: Double) {
        guard let axis = DomainAxis.allCases.first(where: {
            abs(angle - $0.angle) <= Hatching.angleTolerance
        }) else { return nil }
        self = axis
    }
}

/// The part of the domain a contour is restricted to.
///
/// A small closed vocabulary, not an expression language: inside the
/// occluder's footprint, below the truncation, within a band of one axis,
/// and intersections of these. It can be evaluated without an interpreter.
/// Curves that need more than this should be stored, and `LayerSource.file`
/// is how the schema says so.
public indirect enum Restriction: Sendable, Equatable {
    case all
    /// Inside `occluder.region`: the same polygon the heightfield is cut to,
    /// referenced rather than repeated.
    case footprint
    /// Where |f| is below the truncation: membership in the graph, off the
    /// plateaus. It also carries a contour of |f| or arg f that leaves over
    /// the truncation to the truncation edge (`Surface.derive`).
    case belowTruncation
    case band(axis: DomainAxis, lo: Double, hi: Double)
    case intersection([Restriction])

    /// Whether lying below the truncation is among the conditions.
    public var requiresBelowTruncation: Bool {
        switch self {
        case .belowTruncation: true
        case .intersection(let all): all.contains { $0.requiresBelowTruncation }
        case .all, .footprint, .band: false
        }
    }

    /// Whether lying in the footprint is among the conditions.
    public var requiresFootprint: Bool {
        switch self {
        case .footprint: true
        case .intersection(let all): all.contains { $0.requiresFootprint }
        case .all, .belowTruncation, .band: false
        }
    }

    init(json: JSONValue) throws {
        let o = try json.object("Restriction")
        switch try o.string("kind", "Restriction") {
        case "all": self = .all
        case "footprint": self = .footprint
        case "belowTruncation": self = .belowTruncation
        case "band":
            let axisText = try o.string("axis", "Restriction.band")
            guard let axis = DomainAxis(rawValue: axisText) else {
                throw ManifestError.unknownKind(axisText, of: "Restriction.band axis",
                                                known: DomainAxis.allCases.map(\.rawValue))
            }
            self = .band(axis: axis, lo: try o.double("lo", "Restriction.band"),
                         hi: try o.double("hi", "Restriction.band"))
        case "intersection":
            self = .intersection(try o.array("of", "Restriction.intersection").map(Restriction.init(json:)))
        case let other:
            throw ManifestError.unknownKind(other, of: "Restriction",
                                            known: ["all", "footprint", "belowTruncation",
                                                    "band", "intersection"])
        }
    }
    public var json: JSONValue {
        switch self {
        case .all: .object(["kind": .string("all")])
        case .footprint: .object(["kind": .string("footprint")])
        case .belowTruncation: .object(["kind": .string("belowTruncation")])
        case .band(let axis, let lo, let hi):
            .object(["kind": .string("band"), "axis": .string(axis.rawValue),
                     "lo": .double(lo), "hi": .double(hi)])
        case .intersection(let of):
            .object(["kind": .string("intersection"), "of": .array(of.map(\.json))])
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
    /// Level sets of `field` at `levels` on `faces`, restricted by
    /// `restriction`, and replicated once per occluder tile where the faces
    /// are `tiled`. |f| and arg f lie on the graph and are lifted by the
    /// layer's height policy, over the truncation onto a plateau unless the
    /// restriction keeps them below it; a coordinate or the height lies on
    /// its face, where the face puts it
    /// (`Surface.contours(of:_:on:restriction:in:refine:)`).
    case contour(field: ContourField, levels: [Double], restriction: Restriction, faces: FaceSelector)
    /// Hatching (`Hatching`) laid on `faces`: a landscape's walls, risers
    /// or plateaus, or every face of a solid, which a scene carries and a
    /// manifest refuses: a manifest is a landscape's. The landscape's hatchings are
    /// descriptions rather than stored curves for the same reason contours
    /// are: dragging the truncation moves every plateau, and a plateau that
    /// moves is a different set of hatch lines. `Hatch.swift` derives them,
    /// and `tests/fixtures/hatch` pins them.
    case hatching(Hatching, faces: FaceSelector)
    /// The edges of the listed walls: each one's top and bottom edges, and a
    /// vertical edge at each distinct corner, its two ends.
    case wallEdges(edges: [Int], base: Double)
    /// The truncation edge: where the graph meets the plateaus, the level
    /// set |f| = h(x).
    case truncationEdge(tiled: Bool)
    /// The coordinate curves of a parametric surface: `u` curves along which
    /// u is constant and `v` along which v is, evenly spaced. Each vertex
    /// carries its surface coordinate, so visibility is tested where on the
    /// surface it lies.
    case coordinateCurves(u: Int, v: Int)
    /// Linear curves on a torus: `count` straight lines through its
    /// parameter rectangle at `slope` turns of `v` per turn of `u`, each
    /// drawn `turns` turns round -- or until it closes, when the slope is
    /// rational and it is a (p, q) torus curve. Each vertex carries its
    /// surface coordinate.
    case linearCurves(slope: Double, turns: Int, count: Int)
    /// The fold curves of a parametric surface, where it turns edge-on to
    /// the projection direction; their image is the outline. They depend on
    /// the camera and are derived for it.
    case foldCurves
    /// A trajectory of a flow, integrated and carried with its coordinates
    /// on the invariant surface it winds round.
    case trajectory
    /// The edges of a solid, tested per edge by its two faces and cut
    /// exactly where something else hides them (`Solid.visibleEdges`).
    case edges

    public init(json: JSONValue) throws {
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
            let faces = try LayerSource.landscapeFaces(o, "LayerSource.contour")
            if let why = faces.unsupported(field) { throw ManifestError.unsupported(why) }
            self = .contour(field: field,
                            levels: try o.doubles("levels", "LayerSource.contour"),
                            restriction: try Restriction(json: o.value("restriction", "LayerSource.contour")),
                            faces: faces)
        case "hatching":
            let faces = try LayerSource.landscapeFaces(o, "LayerSource.hatching")
            if faces.kind == .wall || faces.kind == .riser, o["step"] != nil { throw LayerSource.wallStep }
            let hatching = try Hatching(fields: o)
            if let why = faces.unsupported(hatching) { throw ManifestError.unsupported(why) }
            self = .hatching(hatching, faces: faces)
        case "wallEdges":
            if o["step"] != nil { throw LayerSource.wallStep }
            self = .wallEdges(
                edges: try o.ints("edges", "LayerSource.wallEdges"),
                base: (try? o.double("base", "LayerSource.wallEdges")) ?? 0)
        case "truncationEdge":
            self = .truncationEdge(tiled: try o.bool("tiled", "LayerSource.truncationEdge",
                                                     default: false))
        case "foldCurves":
            self = .foldCurves
        case "trajectory":
            self = .trajectory
        case "edges":
            self = .edges
        case "coordinateCurves":
            self = .coordinateCurves(u: try o.int("u", "LayerSource.coordinateCurves"),
                                     v: try o.int("v", "LayerSource.coordinateCurves"))
        case "linearCurves":
            self = .linearCurves(slope: try o.double("slope", "LayerSource.linearCurves"),
                                 turns: try o.int("turns", "LayerSource.linearCurves"),
                                 count: try o.int("count", "LayerSource.linearCurves"))
        case let other:
            throw ManifestError.unknownKind(other, of: "LayerSource",
                                            known: ["file", "contour", "hatching", "wallEdges",
                                                    "truncationEdge", "coordinateCurves",
                                                    "linearCurves", "foldCurves", "trajectory",
                                                    "edges"])
        }
    }

    /// The faces a contour or a hatching in a manifest names: a landscape's.
    private static func landscapeFaces(_ o: [String: JSONValue], _ what: String) throws -> FaceSelector {
        let faces = try FaceSelector(json: o.value("faces", what))
        switch faces {
        case .graph, .plateaus, .walls, .risers:
            return faces
        case .surface, .solid:
            throw ManifestError.unsupported("a layer on a \(faces.name)'s faces: a manifest holds a landscape, "
                                            + "with a graph, plateaus, walls and risers")
        }
    }

    /// A wall's or a riser's line is its two ends, so there is no step along
    /// it to write (schema 3).
    private static let wallStep = ManifestError.unsupported(
        "a step on walls or risers: a line there is its two ends, and schema 3 writes none")

    public var json: JSONValue {
        switch self {
        case .file(let v, let o):
            .object(["kind": .string("file"), "vertices": .string(v), "offsets": .string(o)])
        case .contour(let field, let levels, let restriction, let faces):
            .object(["kind": .string("contour"), "field": .string(field.rawValue),
                     "levels": .array(levels.map(JSONValue.double)),
                     "restriction": restriction.json, "faces": faces.json])
        case .hatching(let hatching, let faces):
            .object(hatching.fields.merging(["kind": .string("hatching"), "faces": faces.json]) { $1 }
                .filter { key, _ in
                    switch faces {
                    case .walls, .risers: key != "step"
                    case .graph, .plateaus, .surface, .solid: true
                    }
                })
        case .wallEdges(let edges, let base):
            .object(["kind": .string("wallEdges"),
                     "edges": .array(edges.map { .int($0) }), "base": .double(base)])
        case .truncationEdge(let tiled):
            .object(["kind": .string("truncationEdge"), "tiled": .bool(tiled)])
        case .coordinateCurves(let u, let v):
            .object(["kind": .string("coordinateCurves"), "u": .int(u), "v": .int(v)])
        case .linearCurves(let slope, let turns, let count):
            .object(["kind": .string("linearCurves"), "slope": .double(slope), "turns": .int(turns),
                     "count": .int(count)])
        case .foldCurves:
            .object(["kind": .string("foldCurves")])
        case .trajectory:
            .object(["kind": .string("trajectory")])
        case .edges:
            .object(["kind": .string("edges")])
        }
    }

    /// Vertical hatching on the walls along `edges`, a line every `spacing`
    /// along each wall, each line its two ends.
    public static func wallHatching(edges: [Int], spacing: Double, endLines: Bool = false,
                                    base: Double = 0, topOffset: Double = 0) -> LayerSource {
        .hatching(Hatching(angle: FaceSelector.wallAngle, spacing: spacing),
                  faces: .walls(edges: edges, endLines: endLines, base: base, topOffset: topOffset))
    }

    /// Vertical hatching on the risers, as on the walls: a line every
    /// `spacing` along each, from its bottom edge to its top.
    public static func riserHatching(spacing: Double, tiled: Bool = false, endLines: Bool = false) -> LayerSource {
        .hatching(Hatching(angle: FaceSelector.wallAngle, spacing: spacing),
                  faces: .risers(tiled: tiled, endLines: endLines))
    }

    /// Hatching on the plateaus along one axis of the domain, a line every
    /// `spacing` across the other.
    public static func plateauHatching(axis: DomainAxis, spacing: Double, tiled: Bool = false) -> LayerSource {
        .hatching(Hatching(angle: axis.angle, spacing: spacing),
                  faces: .plateaus(tiled: tiled))
    }

    /// The axis plateau hatching runs along, or nil for any other source.
    public var plateauHatchAxis: DomainAxis? {
        guard case .hatching(let h, .plateaus) = self else { return nil }
        return DomainAxis(angle: h.angle)
    }

    /// True when the geometry is a rule rather than a file -- the layers whose
    /// parameters a consumer can edit and re-derive.
    public var isDerived: Bool {
        if case .file = self { return false }
        return true
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
    /// False for curves that lie *in* the occluding geometry -- a cut-face hatch,
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

    public init(json: JSONValue) throws {
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
    public var json: JSONValue {
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
/// real, so `flipX` flips imag and `yScale` scales real. `Camera.oblique` folds
/// the axis exchange into one matrix; these numbers cross unchanged so the two
/// cameras can be compared numerically instead of by inspection.
public struct ObliqueProjection: Sendable, Equatable {
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
        let o = try json.object("ObliqueProjection")
        self.init(shear: try o.double("shear", "ObliqueProjection"),
                  xAngle: try o.double("xAngle", "ObliqueProjection"),
                  zAngle: try o.double("zAngle", "ObliqueProjection"),
                  flipX: try o.bool("flipX", "ObliqueProjection"),
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
    public var projection: ObliqueProjection
    /// How far in view depth a curve may lie behind the surface and still be
    /// drawn: the hidden-line test's tolerance.
    public var depthTolerance: Double
    /// The depth buffer's resolution, in pixels along the longer side.
    public var buffer: Int

    public init(name: String, projection: ObliqueProjection, depthTolerance: Double, buffer: Int) {
        self.name = name; self.projection = projection; self.depthTolerance = depthTolerance
        self.buffer = buffer
    }

    init(json: JSONValue) throws {
        let o = try json.object("CameraPreset")
        self.init(name: try o.string("name", "CameraPreset"),
                  projection: try ObliqueProjection(json: o.value("projection", "CameraPreset")),
                  depthTolerance: try o.double("depthTolerance", "CameraPreset"),
                  buffer: try o.int("buffer", "CameraPreset"))
    }
    var json: JSONValue {
        .object(["name": .string(name), "projection": projection.json,
                 "depthTolerance": .double(depthTolerance), "buffer": .int(buffer)])
    }
}

public struct Provenance: Sendable, Equatable {
    public var function: String
    /// The recipe that produced this: "function" for a native landscape,
    /// whose request `params` records, so it can be sampled again; "fixture"
    /// for a golden fixture. `function` names the mathematics.
    public var example: String
    public var params: [String: JSONValue]
    /// The contourpy chunk count the bundle was written with. Anything but 1
    /// means the contours were stitched in thread-completion order and the
    /// bundle is not reproducible -- worth knowing before blaming a diff on
    /// anything else.
    public var cpuCount: Int
    public var gitSha: String

    public init(function: String, example: String, params: [String: JSONValue],
                cpuCount: Int, gitSha: String) {
        self.function = function; self.example = example; self.params = params
        self.cpuCount = cpuCount; self.gitSha = gitSha
    }

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
    public var truncation: Truncation
    public var occluder: Occluder
    public var layers: [LayerSpec]
    public var presets: [CameraPreset]
    public var provenance: Provenance

    public init(domain: Domain, height: GridRef, phase: GridRef?, truncation: Truncation,
                occluder: Occluder, layers: [LayerSpec], presets: [CameraPreset],
                provenance: Provenance) {
        self.schema = kurvenSchema; self.axes = kurvenAxes
        self.domain = domain; self.height = height; self.phase = phase; self.truncation = truncation
        self.occluder = occluder; self.layers = layers; self.presets = presets
        self.provenance = provenance
    }

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
        truncation = try Truncation(json: o.value("truncation", "Manifest"))
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
                 "phase": phase.map(\.json) ?? .null, "truncation": truncation.json,
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
