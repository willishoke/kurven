import Foundation
import KurvenCore
import KurvenMath

/// A function worth looking at, and the window it is worth looking at in.
///
/// `window` is how far the domain sliders reach, not what they start at: the
/// interesting rectangle is `domain`, and the window is the room around it
/// where the picture still means something.
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

    public init(name: String, label: String, expression: String, domain: Domain,
                window: Domain, cap: Double?, notes: String = "") {
        self.name = name; self.label = label; self.expression = expression
        self.domain = domain; self.window = window; self.cap = cap; self.notes = notes
    }

    public init(json: JSONValue) throws {
        let o = try json.object("FunctionPreset")
        name = try o.string("name", "FunctionPreset")
        label = try o.string("label", "FunctionPreset")
        expression = try o.string("expression", "FunctionPreset")
        domain = try Domain(json: o.value("domain", "FunctionPreset"))
        window = try Domain(json: o.value("window", "FunctionPreset"))
        cap = o.optionalDouble("cap")
        notes = (try? o.string("notes", "FunctionPreset")) ?? ""
    }

    public var json: JSONValue {
        .object(["name": .string(name), "label": .string(label),
                 "expression": .string(expression), "domain": domain.json,
                 "window": window.json, "cap": cap.map(JSONValue.double) ?? .null,
                 "notes": .string(notes)])
    }

    public var caps: Caps { cap.map { Caps.uniform($0) } ?? .none }
}

/// One function of the expression language, as a picker describes it.
public struct LanguageFunction: Sendable, Equatable, Identifiable {
    public var name: String
    public var arity: Int
    public var aliases: [String]
    public var help: String

    public var id: String { name }

    public init(name: String, arity: Int, aliases: [String], help: String) {
        self.name = name; self.arity = arity; self.aliases = aliases; self.help = help
    }

    public init(json: JSONValue) throws {
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

    public init(presets: [FunctionPreset], functions: [LanguageFunction],
                constants: [String], defaultResolution: Int, defaultBuffer: Int) {
        self.presets = presets; self.functions = functions; self.constants = constants
        self.defaultResolution = defaultResolution; self.defaultBuffer = defaultBuffer
    }

    public func preset(_ name: String) -> FunctionPreset? {
        presets.first { $0.name == name }
    }

    /// The menu this side carries: `kurven.landscape.CATALOG`, and the language
    /// `KurvenMath` implements. `tests/fixtures/expr` compares it with the
    /// Python one entry for entry, so a preset added on either side is a test
    /// failure on the other until it is added there too.
    public static let native = Catalog(
        presets: NativeCatalog.presets,
        functions: Language.functions.map {
            LanguageFunction(name: $0.name, arity: $0.arity, aliases: $0.aliases, help: $0.help)
        },
        constants: Language.constants.map(\.name),
        defaultResolution: NativeLandscape.defaultResolution,
        defaultBuffer: NativeLandscape.plateBuffer)
}

/// What to sample, over what, how finely -- and, once the user has edited it,
/// how it is drawn.
///
/// `caps` and `layers` are absent until the client has changed them, and absent
/// means "whatever the rules derive". Keeping them across a change of window is
/// what makes a truncation and a hatch spacing survive a drag: the landscape is
/// resampled, the styling is not re-decided. This is `kurven.landscape.Spec`.
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

    /// `Spec.to_dict`, as a request or a fixture spells it.
    public init(json: JSONValue) throws {
        let o = try json.object("Spec")
        expression = try o.string("expression", "Spec")
        domain = try Domain(json: o.value("domain", "Spec"))
        resolution = (try? o.int("resolution", "Spec")) ?? NativeLandscape.defaultResolution
        caps = if case .some(.object) = o["caps"] { try Caps(json: o.value("caps", "Spec")) }
               else { nil }
        spacing = o.optionalDouble("spacing")
        layers = if case .some(.array(let list)) = o["layers"] {
            try list.map(LayerSpec.init(json:))
        } else { nil }
        name = (try? o.string("name", "Spec")) ?? ""
    }

    /// True when the two ask for the same *samples*. Everything else is
    /// styling, which the consumer applies itself rather than asking for.
    public func samples(as other: LandscapeRequest) -> Bool {
        expression == other.expression && domain == other.domain
            && resolution == other.resolution
    }

    public var json: JSONValue {
        var o: [String: JSONValue] = [
            "expression": .string(expression),
            "domain": domain.json,
            "resolution": .int(resolution),
            "name": .string(name),
        ]
        if let caps { o["caps"] = caps.json }
        if let spacing { o["spacing"] = .double(spacing) }
        if let layers { o["layers"] = .array(layers.map(\.json)) }
        return .object(o)
    }
}

/// `kurven.landscape.CATALOG`, entry for entry.
enum NativeCatalog {
    static func domain(_ r0: Double, _ r1: Double, _ i0: Double, _ i1: Double) -> Domain {
        Domain(real: Interval(lo: r0, hi: r1), imag: Interval(lo: i0, hi: i1))
    }

    static let presets: [FunctionPreset] = [
        FunctionPreset(
            name: "rgamma", label: "1/Γ(z) — reciprocal factorial", expression: "1/gamma(z)",
            domain: domain(-5.5, 4.0, 0.0, 2.5), window: domain(-12.0, 8.0, -6.0, 6.0), cap: 5.0,
            notes: "Entire, with zeros at the non-positive integers: the ripple along "
                + "the negative real axis. Jahnke-Emde's Fig. 6."),
        FunctionPreset(
            name: "gamma", label: "Γ(z) — gamma", expression: "gamma(z)",
            domain: domain(-4.5, 4.5, 0.0, 2.5), window: domain(-10.0, 10.0, -6.0, 6.0), cap: 5.0,
            notes: "A pole spire at each non-positive integer. Truncating per band "
                + "cuts each spire at its own height, as the published plate does."),
        FunctionPreset(
            name: "zeta", label: "ζ(z) — Riemann zeta", expression: "zeta(z)",
            domain: domain(-6.0, 8.0, 0.0, 30.0), window: domain(-20.0, 20.0, -40.0, 40.0), cap: 6.0,
            notes: "The pole at s = 1 and the zeros on the critical line. Sampled "
                + "here rather than loaded from a cache."),
        FunctionPreset(
            name: "cn", label: "cn(z, 0.64) — Jacobi elliptic", expression: "cn(z, 0.64)",
            domain: domain(-4.0, 4.0, -3.5, 3.5), window: domain(-12.0, 12.0, -12.0, 12.0), cap: 4.0,
            notes: "Doubly periodic: a lattice of poles, one per quarter-period cell."),
        FunctionPreset(
            name: "tan", label: "tan z", expression: "tan(z)",
            domain: domain(-4.8, 4.8, -1.6, 1.6), window: domain(-12.0, 12.0, -4.0, 4.0), cap: 5.0,
            notes: "A pole at every half-period along the real axis, and flat plateaus "
                + "of |tan| -> 1 away from it."),
        FunctionPreset(
            name: "sin", label: "sin z", expression: "sin(z)",
            domain: domain(-5.0, 5.0, -1.5, 1.5), window: domain(-12.0, 12.0, -5.0, 5.0), cap: 2.5,
            notes: "Entire: zeros on the real axis, growing exponentially away from it."),
        FunctionPreset(
            name: "exp_inv", label: "exp(1/z) — essential singularity", expression: "exp(1/z)",
            domain: domain(-1.2, 1.2, -1.2, 1.2), window: domain(-4.0, 4.0, -4.0, 4.0), cap: 5.0,
            notes: "Every value, infinitely often, in every neighbourhood of the "
                + "origin. The landscape shows why."),
        FunctionPreset(
            name: "cubic", label: "1/(z³ - 1)", expression: "1/(z^3 - 1)",
            domain: domain(-2.0, 2.0, -2.0, 2.0), window: domain(-6.0, 6.0, -6.0, 6.0), cap: 5.0,
            notes: "Three simple poles at the cube roots of unity, evenly spaced on "
                + "the unit circle. The polynomial itself is not a landscape at any "
                + "setting -- |z³| runs to 16 in the corners of a window this small, "
                + "so it reads as a tower or, capped, as a table."),
        FunctionPreset(
            name: "moebius", label: "(z² - 1)/(z² + 1)", expression: "(z^2 - 1)/(z^2 + 1)",
            domain: domain(-2.5, 2.5, -2.0, 2.0), window: domain(-6.0, 6.0, -6.0, 6.0), cap: 4.0,
            notes: "Zeros at ±1, poles at ±i: two spires and two pits."),
        FunctionPreset(
            name: "sqrt", label: "√z — a branch cut", expression: "sqrt(z)",
            domain: domain(-2.5, 2.5, -2.0, 2.0), window: domain(-6.0, 6.0, -6.0, 6.0), cap: nil,
            notes: "The principal branch: the phase contours end on the cut along the "
                + "negative real axis, where the surface is continuous and arg is not."),
        FunctionPreset(
            name: "log", label: "log z", expression: "log(z)",
            domain: domain(-2.5, 2.5, -2.0, 2.0), window: domain(-6.0, 6.0, -6.0, 6.0), cap: 3.0,
            notes: "A logarithmic pole at the origin: the one spire that grows slowly "
                + "enough to see the shape of."),
        FunctionPreset(
            name: "erf", label: "erf z", expression: "erf(z)",
            domain: domain(-3.0, 3.0, -2.5, 2.5), window: domain(-8.0, 8.0, -8.0, 8.0), cap: 5.0,
            notes: "Entire, but it grows like exp(z²) off the real axis."),
        FunctionPreset(
            name: "besselj", label: "J₀(z)", expression: "besselj(0, z)",
            domain: domain(-11.0, 11.0, -3.0, 3.0), window: domain(-25.0, 25.0, -8.0, 8.0), cap: 3.0,
            notes: "Oscillating and decaying along the real axis, growing off it."),
        FunctionPreset(
            name: "digamma", label: "ψ(z) — digamma", expression: "digamma(z)",
            domain: domain(-4.5, 4.5, 0.0, 2.5), window: domain(-10.0, 10.0, -6.0, 6.0), cap: 5.0,
            notes: "Γ's logarithmic derivative: simple poles at the non-positive "
                + "integers, and no zeros in sight."),
    ]
}
