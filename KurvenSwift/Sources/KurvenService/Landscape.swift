import Foundation
import KurvenCore
import KurvenLandscape

/// Asking the service for a landscape.
///
/// A landscape is built natively now (`NativeLandscape`), so this is the
/// comparison path rather than the only path: the same request, answered by
/// the Python pipeline, which `kurven-test` holds the native answer to. The
/// types are `KurvenLandscape`'s, so a bundle from either side is the same
/// value to everything downstream.
public extension Service {
    /// The menu of functions, and the language they are written in, as the
    /// server reports them.
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
    /// a `badExpression` failure carrying where it went wrong.
    func validate(_ expression: String) async throws -> String {
        try await request("validate", .object(["expression": .string(expression)]))
            .object("validate").string("expression", "validate")
    }

    /// Sample a landscape and say where its bundle is.
    func landscape(_ landscape: LandscapeRequest, to output: URL) async throws
        -> ExportResult {
        guard case .object(var o) = landscape.json else { fatalError("request is an object") }
        o["output"] = .string(output.path)
        let result = try await request("landscape", .object(o)).object("landscape")
        return ExportResult(
            url: URL(fileURLWithPath: try result.string("path", "landscape")),
            bytes: try result.int("bytes", "landscape"),
            manifest: try Manifest(json: result.value("manifest", "landscape")))
    }
}
