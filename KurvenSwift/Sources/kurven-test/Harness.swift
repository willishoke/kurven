import Foundation

/// A test harness in forty lines.
///
/// Command Line Tools ships `Testing.framework` without a `.swiftmodule`, so
/// `import Testing` does not resolve and `swift test` cannot build a bundle
/// without Xcode (see `Package.swift`). A dependency-free runner keeps the whole
/// package buildable and testable from a terminal, which is the property the
/// design commits to; it also matches the Python lane, which runs
/// `tests/check_bundle.py` rather than pytest.
enum Check {
    nonisolated(unsafe) static var failures: [String] = []
    nonisolated(unsafe) static var passed = 0
    nonisolated(unsafe) static var current = ""

    static func suite(_ name: String, _ body: () throws -> Void) {
        print("\n\(name)")
        do { try body() }
        catch {
            failures.append("\(name): threw \(error)")
            print("  FAIL  \(name) threw \(error)")
        }
    }

    static func expect(_ ok: Bool, _ what: String,
                       _ detail: @autoclosure () -> String = "") {
        let extra = detail()
        if ok {
            passed += 1
            print("  ok    \(what)\(extra.isEmpty ? "" : "  " + extra)")
        } else {
            failures.append(what)
            print("  FAIL  \(what)\(extra.isEmpty ? "" : "  " + extra)")
        }
    }

    static func expectThrows(_ what: String, _ body: () throws -> Void) {
        do {
            try body()
            expect(false, what, "did not throw")
        } catch {
            expect(true, what, "\(type(of: error))")
        }
    }

    static func summary() -> Int32 {
        print()
        if failures.isEmpty {
            print("all green (\(passed) checks)")
            return 0
        }
        print("\(failures.count) FAILED of \(passed + failures.count):")
        for f in failures { print("  - \(f)") }
        return 1
    }
}
