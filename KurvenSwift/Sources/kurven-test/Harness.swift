import Foundation
import Metal
import KurvenMetal

/// A test harness in sixty lines.
///
/// Command Line Tools ships `Testing.framework` without a `.swiftmodule`, so
/// `import Testing` does not resolve and `swift test` cannot build a bundle
/// without Xcode (see `Package.swift`). A dependency-free runner keeps the whole
/// package buildable and testable from a terminal, which is the property the
/// design commits to; it also matches the Python lane, which runs
/// `tests/check_bundle.py` rather than pytest.
///
/// A check passes, fails, or is *skipped*: the thing it needs is not on this
/// machine, and it says so. Skips are listed in the summary so a green run on a
/// machine with nothing on it reads as what it is. One kind is held to a
/// stricter rule: a suite skipped for want of a Metal device fails the run
/// unless `KURVEN_TEST_SKIP_GPU=1` says this machine is known to have none,
/// because the GPU suites are the ones that compare the preview and the bake
/// to the oracle, and a run that silently never drew anything is not green.
enum Check {
    nonisolated(unsafe) static var failures: [String] = []
    nonisolated(unsafe) static var skipped: [String] = []
    nonisolated(unsafe) static var gpuSkipped = 0
    nonisolated(unsafe) static var passed = 0
    nonisolated(unsafe) static var current = ""

    static func suite(_ name: String, _ body: () throws -> Void) {
        print("\n\(name)")
        current = name
        do { try body() }
        catch RendererError.noDevice {
            // `MetalRenderer()` with no device to give it: the suite could not
            // start, and that is the skip `gpu()` records explicitly.
            skipGPU(name)
        }
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

    /// Record that `what` could not run here, and why.
    static func skip(_ what: String, _ why: String) {
        skipped.append("\(what): \(why)")
        print("  --    \(what); skipped: \(why)")
    }

    /// The Metal device, or a recorded GPU skip of the current suite.
    static func gpu() -> MTLDevice? {
        if let device = MTLCreateSystemDefaultDevice() { return device }
        skipGPU(current)
        return nil
    }

    private static func skipGPU(_ what: String) {
        gpuSkipped += 1
        skip(what, "no Metal device")
    }

    static func summary() -> Int32 {
        print()
        if !skipped.isEmpty {
            print("\(skipped.count) skipped:")
            for s in skipped { print("  - \(s)") }
        }
        if !failures.isEmpty {
            print("\(failures.count) FAILED of \(passed + failures.count):")
            for f in failures { print("  - \(f)") }
            return 1
        }
        let allowed = ProcessInfo.processInfo.environment["KURVEN_TEST_SKIP_GPU"] == "1"
        if gpuSkipped > 0 && !allowed {
            print("\(gpuSkipped) suite(s) skipped for want of a Metal device. "
                  + "That fails the run: the GPU suites are the ones that draw. "
                  + "Set KURVEN_TEST_SKIP_GPU=1 if this machine is known to have no GPU.")
            return 1
        }
        print("all green (\(passed) checks\(skipped.isEmpty ? "" : ", \(skipped.count) skipped"))")
        return 0
    }
}
