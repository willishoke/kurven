// swift-tools-version: 6.0
import PackageDescription

// Zero third-party dependencies, by design: everything the frontend needs is
// in the platform (Foundation, simd, Metal, MetalKit, SwiftUI). The package is
// pure SwiftPM and stays that way -- there is no .metal build product and no
// .xcodeproj, so a terminal build and any future Xcode build cannot diverge.
// Shaders compile at runtime from a string, and a test compiles them, which is
// the check a `metal` build step would otherwise provide.
let package = Package(
    name: "KurvenSwift",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "KurvenCore", targets: ["KurvenCore"]),
        .library(name: "KurvenMetal", targets: ["KurvenMetal"]),
        .library(name: "KurvenBake", targets: ["KurvenBake"]),
        .executable(name: "kurven-cli", targets: ["kurven-cli"]),
        .executable(name: "kurven-test", targets: ["kurven-test"]),
        .executable(name: "KurvenApp", targets: ["KurvenApp"]),
    ],
    targets: [
        // Pure values. No Metal, no AppKit; its tests run without a GPU.
        .target(name: "KurvenCore"),

        // The uniform and vertex structs, defined once in C and shared by
        // Swift (as a module) and MSL (prepended to the source), so CPU/GPU
        // layout agreement is by construction rather than by matching two
        // hand-written declarations.
        //
        // SwiftPM does not track a C header as a dependency of the Swift
        // targets that import it, so an incremental build after editing this
        // header can leave a Swift target compiled against the old layout. The
        // symptom is a corrupted uniform and a segfault, not a compile error.
        // After editing ShaderTypes.h, build clean. `kurven-test` asks the GPU
        // what it actually sees, which catches it either way.
        .target(name: "KurvenShaderTypes"),

        // The one place with reference semantics: device, pipelines, textures.
        .target(name: "KurvenMetal", dependencies: ["KurvenCore", "KurvenShaderTypes"]),

        .target(name: "KurvenBake", dependencies: ["KurvenCore", "KurvenMetal"]),
        .executableTarget(name: "kurven-cli", dependencies: ["KurvenBake"]),

        // The test suite is an executable, not a `.testTarget`.
        //
        // Command Line Tools ships `Testing.framework` and
        // `_Testing_Foundation.framework` as binaries with no `.swiftmodule`,
        // so `import Testing` does not resolve and `swift test` cannot build a
        // test bundle without Xcode. Rather than make the test suite the one
        // thing in this package that needs Xcode -- and rather than pretend the
        // tests exist while they cannot run -- the suite is a program:
        // `swift run kurven-test`, exit code 0 or 1. This is also what the
        // Python side does (`tests/check_bundle.py`), for the same reason
        // (pytest is not installed either), so both lanes run the same way.
        .executableTarget(name: "kurven-test", dependencies: ["KurvenBake"]),

        // The window. A bare SwiftPM executable has no bundle, so
        // scripts/bundle-app.sh assembles Kurven.app around this binary with a
        // hand-written Info.plist and an ad hoc signature -- all of which
        // Command Line Tools can do.
        .executableTarget(name: "KurvenApp", dependencies: ["KurvenBake"]),
    ]
)
