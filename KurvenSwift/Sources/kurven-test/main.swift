import Foundation
import simd
import KurvenCore
import KurvenMetal
import KurvenBake

/// The Swift lane of the cross-language tests.
///
/// Every fixture read here was written by `tests/make_fixtures.py` from the
/// Python pipeline, and `tests/check_bundle.py` asserts the same things about
/// the same files. That is what makes them a contract: not two implementations
/// agreeing in principle, but two implementations agreeing on one set of bytes.
/// Cheapest question first -- does the schema round trip, does the array reader
/// accept what it should and refuse what it should not, does the camera land on
/// the same points, does the clipper split the same runs -- so a failure names
/// the earliest stage that broke rather than the last one to notice.
///
///     swift run kurven-test [path/to/tests/fixtures]

enum Fixtures {
    /// The repository root, found from this file's own path. The alternative is
    /// copying the fixtures into the package, and then there are two of them.
    nonisolated(unsafe) static var dir: URL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // kurven-test
        .deletingLastPathComponent()   // Sources
        .deletingLastPathComponent()   // KurvenSwift
        .deletingLastPathComponent()   // repo root
        .appendingPathComponent("tests/fixtures")

    static func url(_ path: String) -> URL { dir.appendingPathComponent(path) }

    static func json(_ path: String) throws -> [String: JSONValue] {
        try JSONValue.parse(Data(contentsOf: url(path))).object(path)
    }

    static var contractBundles: [URL] {
        (try? FileManager.default.contentsOfDirectory(
            at: url("contract"), includingPropertiesForKeys: nil))?
            .filter { $0.pathExtension == "kurven" }
            .sorted { $0.path < $1.path } ?? []
    }
}

// MARK: - 1. the schema contract

func contractTests() {
    Check.suite("contract: the schema is one definition and one mirror") {
        let bundles = Fixtures.contractBundles
        Check.expect(bundles.count == 3, "three fixture bundles are present",
                     "found \(bundles.count)")
        for b in bundles {
            let text = try String(contentsOf: b.appendingPathComponent("manifest.json"),
                                  encoding: .utf8)
            let manifest = try Manifest(json: JSONValue.parse(text))
            Check.expect(manifest.canonicalJSON == text,
                         "\(b.lastPathComponent) round trips byte for byte")
        }
    }

    Check.suite("contract: bundles decode into consistent values") {
        for url in Fixtures.contractBundles {
            let bundle = try KurvenBundle.read(at: url)
            let m = bundle.manifest
            var ok = bundle.surface.height.width == m.height.shape.nx
                && bundle.surface.height.height == m.height.shape.ny
                && (bundle.surface.phase == nil) == (m.phase == nil)
                && bundle.layers.count == m.layers.count
            for layer in bundle.layers {
                ok = ok && layer.paths.offsets.last == layer.paths.vertices.count
                for i in 0..<layer.paths.count { ok = ok && layer.paths[path: i].count >= 2 }
            }
            if case .mesh = m.occluder.walls {
                let walls = bundle.walls()
                ok = ok && !walls.isEmpty
                    && walls.triangles.allSatisfy { Int($0.max()) < walls.vertices.count }
            }
            Check.expect(ok, "\(url.lastPathComponent) decodes consistently")
        }
    }

    Check.suite("contract: what a reader cannot represent is refused, not guessed at") {
        let good = try Fixtures.json("contract/empty.kurven/manifest.json")
        func mutated(_ change: (inout [String: JSONValue]) -> Void) -> JSONValue {
            var d = good; change(&d); return .object(d)
        }
        func occluder(_ key: String, _ v: JSONValue) -> JSONValue {
            mutated { d in
                var o = (try? d["occluder"]!.object("o")) ?? [:]
                o[key] = v
                d["occluder"] = .object(o)
            }
        }
        Check.expectThrows("rejects an unsupported schema") {
            _ = try Manifest(json: mutated { $0["schema"] = .int(999) })
        }
        Check.expectThrows("rejects a bundle in another axis order") {
            _ = try Manifest(json: mutated {
                $0["axes"] = .array(["imag", "real", "magnitude"].map(JSONValue.string)) })
        }
        Check.expectThrows("rejects an unknown cap kind") {
            _ = try Manifest(json: mutated { $0["caps"] = .object(["kind": .string("logarithmic")]) })
        }
        Check.expectThrows("rejects an unknown wall kind") {
            _ = try Manifest(json: occluder("walls", .object(["kind": .string("spline")])))
        }
        Check.expectThrows("rejects an unknown region kind") {
            _ = try Manifest(json: occluder("region", .object(["kind": .string("mask")])))
        }
        Check.expectThrows("rejects a manifest with no domain") {
            _ = try Manifest(json: mutated { $0["domain"] = nil })
        }
    }

    Check.suite("contract: canonical JSON writes numbers the way Python's json does") {
        // The reason Codable is not used: JSONEncoder renders 5.0 as "5", and a
        // manifest that round trips to a different document is not a contract.
        Check.expect(JSONValue.double(5).canonical == "5.0", "5.0 keeps its point")
        Check.expect(JSONValue.double(-79.5).canonical == "-79.5", "-79.5")
        Check.expect(JSONValue.double(0.51).canonical == "0.51", "0.51 is shortest-round-trip")
        Check.expect(JSONValue.int(5).canonical == "5", "an integer stays an integer")
        Check.expect(JSONValue.object(["b": .int(1), "a": .int(2)]).canonical
                     == #"{"a":2,"b":1}"#, "keys are sorted, separators tight")
    }
}

// MARK: - 2. npy

func npyTests() {
    Check.suite("npy: every dtype the bundle declares reads back") {
        let f4 = try NPY.read(contentsOf: Fixtures.url("npy/f4.npy"))
        let f4Count = try f4.floats().count
        Check.expect(f4.dtype == .float32 && f4.shape == [3, 5] && f4Count == 15, "<f4 (3, 5)")
        let f8 = try NPY.read(contentsOf: Fixtures.url("npy/f8.npy"))
        Check.expect(f8.dtype == .float64 && f8.shape == [4, 2], "<f8 (4, 2)")
        let i8 = try NPY.read(contentsOf: Fixtures.url("npy/i8.npy"))
        let i8Count = try i8.ints().count
        Check.expect(i8.dtype == .int64 && i8Count == 6, "<i8 (6,)")
        let c16 = try NPY.read(contentsOf: Fixtures.url("npy/c16.npy"))
        Check.expect(c16.dtype == .complex128 && c16.shape == [2, 3], "<c16 (2, 3)")
        let oneD = try NPY.read(contentsOf: Fixtures.url("npy/f8_1d.npy"))
        let oneDValues = try oneD.doubles()
        Check.expect(oneD.shape == [7] && oneDValues == (0..<7).map(Double.init),
                     "a 1-d array parses its trailing-comma shape")
    }

    Check.suite("npy: what the reader cannot represent is an error, not a transposed landscape") {
        Check.expectThrows("Fortran order is refused") {
            _ = try NPY.read(contentsOf: Fixtures.url("npy/reject_fortran.npy"))
        }
        Check.expectThrows("an unsupported dtype is refused") {
            _ = try NPY.read(contentsOf: Fixtures.url("npy/reject_dtype.npy"))
        }
        Check.expectThrows("a file that is not npy at all is refused") {
            _ = try NPY.parse(Data("not an npy file".utf8))
        }
    }

    Check.suite("npy: float32 round trips through the writer") {
        let values: [Float] = (0..<24).map { Float($0) * 0.5 - 3 }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("kurven-npy-\(UUID().uuidString).npy")
        defer { try? FileManager.default.removeItem(at: url) }
        try NPY.write(values, shape: [4, 6], to: url)
        let back = try NPY.read(contentsOf: url)
        let read = try back.floats()
        Check.expect(back.shape == [4, 6] && read == values,
                     "a written array reads back identically")
    }
}

// MARK: - 3. the camera

func cameraTests() {
    // The test that makes "the bake reproduces the plate" credible before any
    // GPU code exists. If this passes, the only thing between Swift and the
    // Python plate is rasterization.
    Check.suite("camera: each plate preset lands on the Python projection") {
        for name in ["recip", "elliptic", "zeta"] {
            let spec = try Fixtures.json("camera/\(name).json")
            let plate = try PlateProjection(json: spec.value("plate", "fixture"))
            let world = try NPY.read(contentsOf: Fixtures.url("camera/\(name).points.npy")).rows3()
            let want = try NPY.read(contentsOf: Fixtures.url("camera/\(name).projected.npy")).rows3()
            let camera = Camera.plate(plate)
            var worst = 0.0
            for (w, expected) in zip(world, want) {
                worst = max(worst, simd_reduce_max(abs(camera.view(P3<WorldSpace>(w)).v - expected)))
            }
            Check.expect(world.count == want.count && worst < 1e-9,
                         "\(name) matches over \(world.count) points",
                         "worst |Δ| = \(worst)")
        }
    }

    Check.suite("camera: the plate matrix is the composition it claims to be") {
        let plate = PlateProjection(shear: 0.5, xAngle: -55, zAngle: -90,
                                    flipX: true, yScale: 0.75)
        let camera = Camera.plate(plate)
        let p = P3<WorldSpace>(1.5, -2.25, 3)
        // The same thing spelled out in the library's own column order.
        var legacy = SIMD3(p.y, p.x, p.z)      // (imag, real, z)
        legacy.y *= 0.75                        // yScale scales real
        legacy.x *= -1                          // flipX flips imag
        legacy.y -= 0.5 * legacy.x              // shear, on the flipped column
        let rotated = rotationX(Angle(degrees: -55)) * (rotationZ(Angle(degrees: -90)) * legacy)
        Check.expect(simd_reduce_max(abs(camera.view(p).v - rotated)) < 1e-12,
                     "folding the axis exchange into one matrix changes nothing")

        guard case .orthographic(let oblique) = camera.projection else {
            Check.expect(false, "a plate camera is orthographic"); return
        }
        Check.expect(oblique?.shear == 0.5, "the oblique shear survives as a camera property")
    }
}

// MARK: - 4. hidden-line clipping

func clipTests() {
    // `clip` is pure and takes a `DepthImage` value, so this runs the real
    // hidden-line stage against a Python-dumped buffer with no GPU in the loop.
    // Exact equality, not a tolerance: same integer pixel lookup, same run
    // splitting, same dropped singletons.
    Check.suite("clip: reproduces clip_hidden_lines exactly") {
        let meta = try Fixtures.json("clip/meta.json")
        let axis0 = try meta.doubles("axis0", "clip meta")
        let axis1 = try meta.doubles("axis1", "clip meta")
        let shape = try meta.ints("shape", "clip meta")
        let margin = try meta.double("margin", "clip meta")
        let frame = DepthFrame(axis0: Interval(lo: axis0[0], hi: axis0[1]),
                               axis1: Interval(lo: axis1[0], hi: axis1[1]),
                               rows: shape[0], cols: shape[1],
                               nudge: try meta.double("nudge", "clip meta"))
        let depth = DepthImage(
            frame: frame,
            values: try NPY.read(contentsOf: Fixtures.url("clip/depth.npy")).floats())

        let view = try NPY.read(contentsOf: Fixtures.url("clip/view.npy")).rows3()
            .map { P3<ViewSpace>($0) }
        let offsets = try NPY.read(contentsOf: Fixtures.url("clip/view.idx.npy")).ints()
        let got = HiddenLine.clip(PolylineSet(vertices: view, offsets: offsets),
                                  against: depth, margin: margin)

        let wantVerts = try NPY.read(contentsOf: Fixtures.url("clip/expected.npy"))
        let wantOffsets = try NPY.read(contentsOf: Fixtures.url("clip/expected.idx.npy")).ints()
        Check.expect(got.offsets == wantOffsets,
                     "the same runs, split at the same vertices",
                     "\(got.count) segments vs \(wantOffsets.count - 1)")

        let flat = try wantVerts.doubles()
        var exact = wantVerts.shape == [got.vertices.count, 2]
        if exact {
            for (i, v) in got.vertices.enumerated()
            where v.x != flat[2 * i] || v.y != flat[2 * i + 1] { exact = false; break }
        }
        Check.expect(exact, "every surviving vertex is bit-identical")
    }

    Check.suite("clip: out-of-frame vertices read as occluded, as they do in Python") {
        let frame = DepthFrame(axis0: Interval(lo: 0, hi: 1), axis1: Interval(lo: 0, hi: 1),
                               rows: 4, cols: 4)
        let depth = DepthImage(frame: frame, values: [Float](repeating: -.infinity, count: 16))
        Check.expect(depth.depth(under: P2(0.5, 0.5)) == -.infinity, "inside and empty is -inf")
        Check.expect(depth.depth(under: P2(-5, 0.5)) == .infinity, "left of the frame is +inf")
        Check.expect(depth.depth(under: P2(0.5, 9)) == .infinity, "past the frame is +inf")
    }
}

// MARK: - core values

func coreTests() {
    Check.suite("core: the depth frame's pixel mapping is ZBuffer's") {
        let frame = DepthFrame(axis0: Interval(lo: -2, hi: 3), axis1: Interval(lo: 10, hi: 14),
                               rows: 6, cols: 5)
        var ok = true
        for r in 0..<frame.rows {
            for c in 0..<frame.cols
            where frame.index(of: frame.coordinate(row: r, col: c)) != SIMD2(r, c) { ok = false }
        }
        Check.expect(ok, "every lattice vertex round trips to its own pixel")
        Check.expect(frame.index(of: P2(-2, 10)) == SIMD2(0, 0),
                     "the low corner lands on pixel 0, not -1 (the 0.01 nudge)")
    }

    Check.suite("core: the NDC transform puts each sample point on its lattice vertex") {
        let frame = DepthFrame(axis0: Interval(lo: -2, hi: 3), axis1: Interval(lo: 10, hi: 14),
                               rows: 6, cols: 5)
        let (scale, offset) = frame.metalNDC
        var worst = 0.0
        for r in 0..<frame.rows {
            for c in 0..<frame.cols {
                let p = frame.coordinate(row: r, col: c)
                // Metal samples pixel (col, row) at NDC ((2c+1)/cols - 1,
                // 1 - (2r+1)/rows): y counted from the top of the target.
                worst = max(worst, abs(scale.x * p.y + offset.x
                                       - (Double(2 * c + 1) / Double(frame.cols) - 1)))
                worst = max(worst, abs(scale.y * p.x + offset.y
                                       - (1 - Double(2 * r + 1) / Double(frame.rows))))
            }
        }
        Check.expect(worst < 1e-12, "no half-pixel shift between renderer and clipper",
                     "worst |Δ| = \(worst)")
    }

    Check.suite("core: bake tiles partition the plate's lattice exactly") {
        let frame = DepthFrame(axis0: Interval(lo: -3, hi: 5), axis1: Interval(lo: 0, hi: 7),
                               rows: 240, cols: 240)
        for n in [1, 2, 3, 7] {
            var covered = Set<SIMD2<Int>>()
            var overlapped = false
            for i in 0..<n {
                for j in 0..<n {
                    let sub = frame.tile(i, j, of: n)
                    for r in 0..<sub.rows {
                        for c in 0..<sub.cols {
                            let whole = frame.index(of: sub.coordinate(row: r, col: c))
                            if !covered.insert(whole).inserted { overlapped = true }
                        }
                    }
                }
            }
            Check.expect(!overlapped && covered.count == frame.rows * frame.cols,
                         "n = \(n): tiles cover every pixel exactly once",
                         "\(covered.count) of \(frame.rows * frame.cols)")
        }
    }

    Check.suite("core: point-in-polygon scans along x, matching kurven.bundle") {
        // zeta's staircase, in world (x = real, y = imag).
        let corners: [P2<WorldSpace>] = [
            P2(-6, 28), P2(-6, 0), P2(-2, 0), P2(-2, -5), P2(-0.5, -5),
            P2(-0.5, -14), P2(0.5, -14), P2(0.5, -28), P2(8, -28), P2(8, 28),
        ]
        Check.expect(pointInPolygon(P2(4, 0), corners), "deep inside")
        Check.expect(!pointInPolygon(P2(-4, -20), corners), "in the notch")
        Check.expect(pointInPolygon(P2(-4, 10), corners), "in the pole arm")
        Check.expect(!pointInPolygon(P2(-9, 0), corners), "left of everything")
        Check.expect(!pointInPolygon(P2(0, 40), corners), "past the top")
    }

    Check.suite("core: caps are a property of the model, and bands read in order") {
        Check.expect(Caps.none.height(atX: 3) == .infinity, "no cap is no cap")
        Check.expect(Caps.uniform(5).height(atX: -100) == 5, "a uniform cap ignores x")
        let bands = Caps.realBands([RealBand(below: -3.5, cap: 8),
                                    RealBand(below: -2.5, cap: 6),
                                    RealBand(below: -1.5, cap: 4)])
        Check.expect(bands.height(atX: -4) == 8 && bands.height(atX: -3) == 6
                     && bands.height(atX: -2) == 4 && bands.height(atX: 0) == .infinity,
                     "the first matching band wins, and past the last there is none")
    }

    Check.suite("core: CSR path identity cannot weld unrelated runs") {
        let a = (0..<5).map { P3<WorldSpace>(Double($0), 0, 0) }
        let b = (0..<3).map { P3<WorldSpace>(Double($0), 1, 0) }
        let set = PolylineSet(paths: [a, b, [P3(9, 9, 9)]])   // the singleton draws nothing
        Check.expect(set.count == 2 && set.offsets == [0, 5, 8],
                     "two paths, offsets [0, 5, 8], the one-vertex path dropped")
    }

    Check.suite("core: transforms compose left to right") {
        let a = Transform<WorldSpace, ViewSpace>(
            rows: (SIMD3(2, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1)),
            translation: SIMD3(1, 0, 0))
        let b = Transform<ViewSpace, PlateSpace>(
            rows: (SIMD3(1, 0, 0), SIMD3(0, 3, 0), SIMD3(0, 0, 1)))
        let p = P3<WorldSpace>(1, 2, 3)
        Check.expect((a >>> b)(p) == b(a(p)) && (a >>> b)(p) == P3<PlateSpace>(3, 6, 3),
                     "a >>> b is 'apply a, then b'")
    }

    Check.suite("core: decimation keeps the samples grid_mesh's stride keeps") {
        let grid = Grid2D(width: 7, height: 5,
                          domain: Domain(real: Interval(lo: 0, hi: 6),
                                         imag: Interval(lo: 0, hi: 4)),
                          values: (0..<35).map { Float($0) })
        let d = grid.decimated(by: 3)
        // Python: values[::3, ::3] -- columns 0, 3, 6 and rows 0, 3.
        Check.expect(d.width == 3 && d.height == 2 && d.values == [0, 3, 6, 21, 24, 27],
                     "decimated(by: 3) is [::3, ::3]")
        Check.expect(d.domain.real.hi == 6 && d.domain.imag.hi == 3,
                     "the domain narrows to the samples that survived")
    }
}

// MARK: - 5. the shader

func shaderTests() {
    // The check the absent `metal` compiler would otherwise provide: a shader
    // error fails the test run rather than the first launch.
    Check.suite("metal: the shader source compiles on this device") {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Check.expect(false, "a Metal device exists"); return
        }
        let library = try MetalRenderer.makeLibrary(device: device)
        for name in ["kv_height_vertex", "kv_mesh_vertex", "kv_depth_fragment"] {
            Check.expect(library.makeFunction(name: name) != nil,
                         "the library exposes \(name)")
        }
        _ = try MetalRenderer(device: device)
        Check.expect(true, "both depth pipelines build with a MAX blend on r32Float")
    }
}

// MARK: - entry

import Metal

if let override = CommandLine.arguments.dropFirst().first {
    Fixtures.dir = URL(fileURLWithPath: override)
}
print("kurven-test  fixtures: \(Fixtures.dir.path)")
contractTests()
npyTests()
cameraTests()
clipTests()
coreTests()
shaderTests()
exit(Check.summary())
