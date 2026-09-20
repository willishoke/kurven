import Foundation
import KurvenCore

/// Asking for a landscape of a function nobody wrote a program for.
///
/// `export` asks the service to run a published plate; this asks it to sample an
/// expression over a rectangle. The difference is what the answer can be edited
/// into afterwards: a landscape's bundle describes every one of its layers, so
/// the cap, the levels and the hatch spacing are the consumer's to change, and
/// only the function, the window and the resolution need Python again.
///
/// The catalog is data from the server, not a list here. A preset added to
/// `kurven.landscape` appears in this client without a line of Swift changing,
/// which is the same arrangement `describe` already has with argparse.
public struct FunctionPreset: Sendable, Equatable, Identifiable {
    public var name: String
    public var label: String
    public var expression: String
    /// The rectangle worth looking at.
    public var domain: Domain
    /// How far the sliders reach around it.
    public var window: Domain
    /// Where to truncate, when the function has a pole worth truncating.
    public var cap: Double?
    public var notes: String

    public var id: String { name }

    init(json: JSONValue) throws {
        let o = try json.object("FunctionPreset")
        name = try o.string("name", "FunctionPreset")
        label = try o.string("label", "FunctionPreset")
        expression = try o.string("expression", "FunctionPreset")
        domain = try Domain(json: o.value("domain", "FunctionPreset"))
        window = try Domain(json: o.value("window", "FunctionPreset"))
        cap = o.optionalDouble("cap")
        notes = (try? o.string("notes", "FunctionPreset")) ?? ""
    }

    public var caps: Caps { cap.map { Caps.uniform($0) } ?? .none }
}

/// One function of the expression language, as the server describes it.
public struct LanguageFunction: Sendable, Equatable, Identifiable {
    public var name: String
    public var arity: Int
    public var aliases: [String]
    public var help: String

    public var id: String { name }

    init(json: JSONValue) throws {
        let o = try json.object("LanguageFunction")
        name = try o.string("name", "LanguageFunction")
        arity = (try? o.int("arity", "LanguageFunction")) ?? 1
        aliases = (try? o.strings("aliases", "LanguageFunction")) ?? []
        help = (try? o.string("help", "LanguageFunction")) ?? ""
    }
}

public struct Catalog: Sendable, Equatable {
    public var presets: [FunctionPreset]
    public var functions: [LanguageFunction]
    public var constants: [String]
    public var defaultResolution: Int
    public var defaultBuffer: Int

    public func preset(_ name: String) -> FunctionPreset? {
        presets.first { $0.name == name }
    }
}

/// What to sample, over what, how finely -- and, once the user has edited it,
/// how it is drawn.
///
/// `caps` and `layers` are absent until the client has changed them, and absent
/// means "whatever the rules derive". Sending them back is what keeps a
/// truncation and a hatch spacing across a change of window: the landscape is
/// resampled, the styling is not re-decided.
public struct LandscapeRequest: Sendable, Equatable {
    public var expression: String
    public var domain: Domain
    public var resolution: Int
    public var caps: Caps?
    public var spacing: Double?
    public var layers: [LayerSpec]?
    /// The catalog entry this came from, for provenance. Empty when the
    /// expression was typed.
    public var name: String

    public init(expression: String, domain: Domain, resolution: Int,
                caps: Caps? = nil, spacing: Double? = nil,
                layers: [LayerSpec]? = nil, name: String = "") {
        self.expression = expression; self.domain = domain
        self.resolution = resolution; self.caps = caps; self.spacing = spacing
        self.layers = layers; self.name = name
    }

    public init(preset: FunctionPreset, resolution: Int) {
        self.init(expression: preset.expression, domain: preset.domain,
                  resolution: resolution, caps: preset.caps, name: preset.name)
    }

    /// True when the two ask for the same *samples*. Everything else is
    /// styling, which the consumer applies itself rather than asking for.
    public func samples(as other: LandscapeRequest) -> Bool {
        expression == other.expression && domain == other.domain
            && resolution == other.resolution
    }

    func json(output: URL) -> JSONValue {
        var o: [String: JSONValue] = [
            "expression": .string(expression),
            "domain": domain.json,
            "resolution": .int(resolution),
            "name": .string(name),
            "output": .string(output.path),
        ]
        if let caps { o["caps"] = caps.json }
        if let spacing { o["spacing"] = .double(spacing) }
        if let layers { o["layers"] = .array(layers.map(\.json)) }
        return .object(o)
    }
}

public extension Service {
    /// The menu of functions, and the language they are written in.
    func catalog() async throws -> Catalog {
        let o = try await request("catalog").object("catalog")
        let language = try o.value("language", "catalog").object("catalog.language")
        let defaults = (try? o.value("defaults", "catalog").object("catalog.defaults")) ?? [:]
        return Catalog(
            presets: try o.array("presets", "catalog").map(FunctionPreset.init(json:)),
            functions: try language.array("functions", "catalog.language")
                .map(LanguageFunction.init(json:)),
            constants: try language.array("constants", "catalog.language").map {
                try $0.object("constant").string("name", "constant")
            },
            defaultResolution: (try? defaults.int("resolution", "catalog.defaults")) ?? 600,
            defaultBuffer: (try? defaults.int("buffer", "catalog.defaults")) ?? 4000)
    }

    /// Parse an expression without sampling it: the canonical spelling back, or
    /// a `badExpression` failure carrying where it went wrong. A field being
    /// typed into can afford this and cannot afford a landscape.
    func validate(_ expression: String) async throws -> String {
        try await request("validate", .object(["expression": .string(expression)]))
            .object("validate").string("expression", "validate")
    }

    /// Sample a landscape and say where its bundle is.
    func landscape(_ landscape: LandscapeRequest, to output: URL) async throws
        -> ExportResult {
        let o = try await request("landscape", landscape.json(output: output))
            .object("landscape")
        return ExportResult(
            url: URL(fileURLWithPath: try o.string("path", "landscape")),
            bytes: try o.int("bytes", "landscape"),
            manifest: try Manifest(json: o.value("manifest", "landscape")))
    }
}

// MARK: - the derived numbers, on this side

/// The rules `kurven.landscape` derives a landscape's styling by, as far as the
/// consumer needs them.
///
/// Python decides these when it builds a landscape; this side needs the same
/// two when the *cap* moves, because the contour levels are stated in absolute
/// terms and a plate capped at 10 with levels drawn to 5 is half a plate. They
/// are pure arithmetic over one number, they are pinned against the server's own
/// answer in `kurven-test`, and having them here is the difference between a cap
/// slider that redraws in milliseconds and one that is a round trip.
public enum LandscapeStyle {
    /// `x` rounded to a number a person would have chosen: 1, 2, 2.5 or 5 times
    /// a power of ten. Down, for a spacing, where rounding up thins the ruling.
    public static func nice(_ x: Double, down: Bool = false) -> Double {
        guard x.isFinite, x > 0 else { return 1 }
        let scale = pow(10, (log10(x)).rounded(.down))
        let m = x / scale
        let steps: [Double] = [1, 2, 2.5, 5, 10]
        if down {
            return scale * (steps.filter { $0 <= m * (1 + 1e-12) }.max() ?? 1)
        }
        return scale * (steps.filter { $0 >= m * (1 - 1e-12) }.min() ?? 10)
    }

    /// Major and minor |f| levels under a ceiling: about ten majors, five
    /// minors to a major. Multiples of the step rather than a spread between two
    /// endpoints, so raising the cap adds contours at the top instead of moving
    /// every contour already drawn.
    public static func magnitudeLevels(upTo ceiling: Double)
        -> (major: [Double], minor: [Double]) {
        guard ceiling.isFinite, ceiling > 0 else { return ([], []) }
        let step = nice(ceiling / 10)
        let minorStep = step / 5
        let major = (1...max(Int(ceiling / step), 1)).map {
            (step * Double($0) * 1e10).rounded() / 1e10
        }
        let minor = (1...max(Int(ceiling / minorStep), 1)).map {
            (minorStep * Double($0) * 1e10).rounded() / 1e10
        }.filter { m in !major.contains { abs($0 - m) < 1e-9 } }
        return (major, minor)
    }

    /// The levels a landscape is actually given under these caps: the rule
    /// above, less any level sitting exactly on the cap.
    ///
    /// That contour *is* the rim of the plateau, and the rim has its own layer
    /// drawn at its own weight, so keeping both draws one line twice.
    /// `kurven.landscape.default_layers` drops it for the same reason, and the
    /// service round trip in `kurven-test` compares the two.
    public static func levels(under caps: Caps)
        -> (major: [Double], minor: [Double]) {
        guard let ceiling = ceiling(of: caps) else { return ([], []) }
        let all = magnitudeLevels(upTo: ceiling)
        return (all.major.filter { $0 < ceiling - 1e-9 },
                all.minor.filter { $0 < ceiling - 1e-9 })
    }

    /// The highest |f| worth a contour: the cap, when there is one.
    public static func ceiling(of caps: Caps) -> Double? {
        switch caps {
        case .none: return nil
        case .uniform(let z): return z
        case .realBands(let bands, let beyond):
            let tops = bands.map(\.cap) + (beyond.isFinite ? [beyond] : [])
            return tops.max()
        }
    }

    /// About seventy strokes along the longest side, rounded to a ruler.
    public static func hatchSpacing(_ domain: Domain) -> Double {
        nice(max(abs(domain.real.length), abs(domain.imag.length)) / 70, down: true)
    }
}
