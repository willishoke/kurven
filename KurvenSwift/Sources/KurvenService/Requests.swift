import Foundation
import KurvenCore

/// What an example accepts, as the server reported it.
///
/// A client builds its form from this rather than from a hand-written list, so
/// adding an option to a Python example makes it appear in the app without
/// anyone editing Swift. That is the whole reason the examples were given a
/// `parser()` separate from `main()`.
public struct ArgumentSpec: Sendable, Equatable, Identifiable {
    public enum Kind: String, Sendable { case int, float, string, flag }

    public var name: String
    public var kind: Kind
    public var help: String
    public var choices: [String]?
    /// The example's own default, as text.
    public var defaultText: String?

    public var id: String { name }

    init(json: JSONValue) throws {
        let o = try json.object("ArgumentSpec")
        name = try o.string("name", "ArgumentSpec")
        kind = Kind(rawValue: try o.string("kind", "ArgumentSpec")) ?? .string
        help = (try? o.string("help", "ArgumentSpec")) ?? ""
        choices = try? o.strings("choices", "ArgumentSpec")
        switch o["default"] {
        case .some(.string(let s)): defaultText = s
        case .some(.int(let i)): defaultText = String(i)
        case .some(.double(let d)): defaultText = String(d)
        case .some(.bool(let b)): defaultText = b ? "true" : "false"
        default: defaultText = nil
        }
    }
}

public struct ExampleSpec: Sendable, Equatable, Identifiable {
    public var name: String
    public var available: Bool
    public var reason: String?
    public var arguments: [ArgumentSpec]

    public var id: String { name }

    init(json: JSONValue) throws {
        let o = try json.object("ExampleSpec")
        name = try o.string("name", "ExampleSpec")
        available = try o.bool("available", "ExampleSpec", default: true)
        reason = try? o.string("reason", "ExampleSpec")
        arguments = available
            ? try o.array("arguments", "ExampleSpec").map(ArgumentSpec.init(json:))
            : []
    }
}

public struct Description: Sendable, Equatable {
    public var protocolVersion: Int
    public var examples: [ExampleSpec]

    public func example(_ name: String) -> ExampleSpec? {
        examples.first { $0.name == name }
    }
}

public struct ExportResult: Sendable {
    public var url: URL
    public var bytes: Int
    public var manifest: Manifest
}

public extension Service {
    /// The version of the protocol this client speaks. A server that answers
    /// with a different one is a server that was updated separately, and saying
    /// so beats decoding its replies hopefully.
    static let protocolVersion = 1

    func describe() async throws -> Description {
        let o = try await request("describe").object("describe")
        let version = try o.int("protocol", "describe")
        guard version == Service.protocolVersion else {
            throw Failure.remote(kind: "protocolMismatch",
                                 message: "the server speaks protocol \(version), "
                                        + "this client speaks \(Service.protocolVersion)")
        }
        return Description(protocolVersion: version,
                           examples: try o.array("examples", "describe")
                               .map(ExampleSpec.init(json:)))
    }

    /// Build a bundle and say where it is.
    func export(example: String, to output: URL, arguments: [String: String] = [:],
                derived: Bool = false, chunkCount: Int = 1) async throws -> ExportResult {
        var typed: [String: JSONValue] = [:]
        for (key, value) in arguments {
            // An empty field means "leave it alone", not "pass the empty
            // string". An example whose option has no default -- zeta's cache
            // path, everyone's chunk count -- would otherwise be handed one.
            guard !value.trimmingCharacters(in: .whitespaces).isEmpty else { continue }
            // Numbers go as numbers so the server's argparse sees what the
            // command line would; everything else is text, and "true"/"false"
            // are how a flag is spelled.
            if value == "true" || value == "false" {
                typed[key] = .bool(value == "true")
            } else if let i = Int(value) {
                typed[key] = .int(i)
            } else if let d = Double(value) {
                typed[key] = .double(d)
            } else {
                typed[key] = .string(value)
            }
        }
        let params = JSONValue.object([
            "example": .string(example),
            "output": .string(output.path),
            "arguments": .object(typed),
            "derived": .bool(derived),
            "chunkCount": .int(chunkCount),
        ])
        let o = try await request("export", params).object("export")
        return ExportResult(
            url: URL(fileURLWithPath: try o.string("path", "export")),
            bytes: try o.int("bytes", "export"),
            manifest: try Manifest(json: o.value("manifest", "export")))
    }
}
