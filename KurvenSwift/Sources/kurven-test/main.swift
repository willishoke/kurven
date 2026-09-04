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

    Check.suite("clip: the GPU's empty sentinel and Python's -infinity decide alike") {
        // Carrying the sentinel instead of rewriting it saves a pass over every
        // pixel of the buffer. It is only sound because the visibility test
        // cannot tell the two apart, which is what this asserts.
        let frame = DepthFrame(axis0: Interval(lo: 0, hi: 1), axis1: Interval(lo: 0, hi: 1),
                               rows: 2, cols: 2)
        let sentinel: Float = -1e30
        let heights: [Float] = [-1e6, -1, 0, 1e6]
        let python = DepthImage(frame: frame,
                                values: [Float](repeating: -.infinity, count: 4))
        let metal = DepthImage(frame: frame,
                               values: [Float](repeating: sentinel, count: 4),
                               empty: sentinel)
        var alike = true
        for z in heights.map(Double.init) {
            for margin in [0.0, 0.02, 0.2] {
                let a = z + margin > python.depth(under: P2(0.25, 0.25))
                let b = z + margin > metal.depth(under: P2(0.25, 0.25))
                if a != b { alike = false }
            }
        }
        Check.expect(alike, "every visibility test agrees over both conventions")
        Check.expect(!python.isCovered(row: 0, col: 0) && !metal.isCovered(row: 0, col: 0),
                     "and coverage still reads as empty under both")
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
        // Both raster orders, and the screen order's descending row axis, have
        // to land on the same lattice the clipper indexes -- a half-pixel shift
        // here shows up as ink leaking through the surface along silhouettes
        // and nowhere else.
        for (name, frame) in [
            ("buffer order", DepthFrame(axis0: Interval(lo: -2, hi: 3),
                                        axis1: Interval(lo: 10, hi: 14),
                                        rows: 6, cols: 5, order: .buffer)),
            ("screen order", DepthFrame(axis0: Interval(lo: 3, hi: -2),
                                        axis1: Interval(lo: 10, hi: 14),
                                        rows: 6, cols: 5, order: .screen)),
        ] {
            let (linear, offset) = frame.metalNDC
            var worst = 0.0
            var roundTrips = true
            for r in 0..<frame.rows {
                for c in 0..<frame.cols {
                    let p = frame.coordinate(row: r, col: c)
                    if frame.index(of: p) != SIMD2(r, c) { roundTrips = false }
                    // Metal samples pixel (col, row) at NDC
                    // ((2c+1)/cols - 1, 1 - (2r+1)/rows): y from the top.
                    let ndc = linear * SIMD2(p.x, p.y) + offset
                    worst = max(worst, abs(ndc.x - (Double(2 * c + 1) / Double(frame.cols) - 1)))
                    worst = max(worst, abs(ndc.y - (1 - Double(2 * r + 1) / Double(frame.rows))))
                }
            }
            Check.expect(roundTrips, "\(name): every lattice vertex indexes its own pixel")
            Check.expect(worst < 1e-12, "\(name): no half-pixel shift", "worst |Δ| = \(worst)")
        }
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

// MARK: - 7. navigation

func navigationTests() {
    let plates: [(String, PlateProjection)] = [
        ("recip", PlateProjection(shear: 0.5, xAngle: -55, zAngle: -90, flipX: true, yScale: nil)),
        ("elliptic", PlateProjection(shear: 0.51, xAngle: -63, zAngle: -90, flipX: true, yScale: nil)),
        ("zeta", PlateProjection(shear: -0.18, xAngle: -79.5, zAngle: -90, flipX: true, yScale: 0.75)),
    ]

    // The bridge between "a plate" and "a thing you can orbit". If this fails,
    // loading a preset jumps rather than starting where the plate is.
    Check.suite("navigation: an orbit matching a preset IS the plate camera") {
        let rng = SplitMix(seed: 4)
        for (name, plate) in plates {
            let a = Camera.plate(plate).view
            let b = Orbit(matching: plate).camera.view
            var worst = 0.0
            for _ in 0..<200 {
                let p = P3<WorldSpace>(rng.next(-8, 8), rng.next(-30, 30), rng.next(0, 6))
                worst = max(worst, simd_reduce_max(abs(a(p).v - b(p).v)))
            }
            Check.expect(worst == 0, "\(name)", "worst |Δ| = \(worst)")
        }
    }

    Check.suite("navigation: orbiting and orbiting back is the identity") {
        let v = Viewport(width: 1200, height: 800)
        let rng = SplitMix(seed: 9)
        var worst = 0.0
        for (_, plate) in plates {
            var n = Navigator(orbit: Orbit(matching: plate),
                              framing: Framing(center: P2(0, 0), unitsPerPixel: 0.01))
            let start = n
            var back: [Gesture] = []
            for _ in 0..<12 {
                // Stay clear of the elevation clamp, which is deliberately not
                // invertible -- past vertical the landscape turns inside out.
                let d = SIMD2(rng.next(-40, 40), rng.next(-6, 6))
                n = n.applying(.orbit(d), in: v)
                back.append(.orbit(-d))
            }
            n = n.applying(back.reversed(), in: v)
            worst = max(worst, abs(n.orbit.azimuth.degrees - start.orbit.azimuth.degrees))
            worst = max(worst, abs(n.orbit.elevation.degrees - start.orbit.elevation.degrees))
        }
        Check.expect(worst < 1e-9, "azimuth and elevation return exactly",
                     "worst |Δ| = \(worst) degrees")
    }

    Check.suite("navigation: zoom holds the point under the cursor") {
        let v = Viewport(width: 1440, height: 900)
        let rng = SplitMix(seed: 17)
        var worst = 0.0
        for _ in 0..<200 {
            let n = Navigator(orbit: Orbit(matching: plates[0].1),
                              framing: Framing(center: P2(rng.next(-5, 5), rng.next(-5, 5)),
                                               unitsPerPixel: rng.next(0.001, 0.05)))
            let at = SIMD2(rng.next(0, 1440), rng.next(0, 900))
            let before = n.framing.viewPoint(atPixel: at, in: v)
            let after = n.applying(.zoom(factor: rng.next(0.2, 5), at: at), in: v)
                .framing.viewPoint(atPixel: at, in: v)
            worst = max(worst, max(abs(after.x - before.x), abs(after.y - before.y)))
        }
        Check.expect(worst < 1e-9, "the anchor does not drift", "worst |Δ| = \(worst)")
    }

    Check.suite("navigation: pan is invertible and zoom-independent") {
        let v = Viewport(width: 800, height: 600)
        let n = Navigator(orbit: Orbit(matching: plates[1].1),
                          framing: Framing(center: P2(1.5, -2), unitsPerPixel: 0.02))
        let d = SIMD2(37.0, -19.0)
        let there = n.applying(.pan(d), in: v).applying(.pan(-d), in: v)
        Check.expect(there.framing == n.framing, "panning back lands where it started")

        // A pixel drag moves the picture by that many pixels whatever the zoom.
        let zoomed = n.applying(.zoom(factor: 4, at: SIMD2(400, 300)), in: v)
        let moved = zoomed.applying(.pan(SIMD2(100, 0)), in: v)
        let dx = (moved.framing.center.x - zoomed.framing.center.x) / zoomed.framing.unitsPerPixel
        Check.expect(abs(dx + 100) < 1e-9, "100 px of drag is 100 px of travel",
                     "moved \(-dx) px")
    }

    Check.suite("navigation: pixels are square and round trip") {
        let v = Viewport(width: 1237, height: 811)
        let f = Framing(center: P2(-3, 7), unitsPerPixel: 0.013)
        let frame = f.frame(v)
        let upp = frame.unitsPerPixel
        Check.expect(abs(upp.x - upp.y) < 1e-12 && abs(upp.x - 0.013) < 1e-12,
                     "the raster has square pixels of the size asked for",
                     "\(upp.x) x \(upp.y)")
        Check.expect(frame.rows == 811 && frame.cols == 1237 && frame.order == .screen,
                     "one sample per pixel, in screen order")

        var worst = 0.0
        let rng = SplitMix(seed: 23)
        for _ in 0..<500 {
            let px = SIMD2(rng.next(0, 1237), rng.next(0, 811))
            let back = f.pixel(of: f.viewPoint(atPixel: px, in: v), in: v)
            worst = max(worst, simd_reduce_max(abs(back - px)))
        }
        Check.expect(worst < 1e-9, "pixel -> view -> pixel is the identity",
                     "worst |Δ| = \(worst)")

        // The top-left pixel really is the top left: view y decreasing downward.
        let topLeft = f.viewPoint(atPixel: SIMD2(0, 0), in: v)
        let bottomRight = f.viewPoint(atPixel: SIMD2(1236, 810), in: v)
        Check.expect(topLeft.x < bottomRight.x && topLeft.y > bottomRight.y,
                     "the picture is the right way up")
    }

    Check.suite("navigation: re-targeting turns about a new point without moving it") {
        let v = Viewport(width: 1000, height: 700)
        let n = Navigator(orbit: Orbit(matching: plates[2].1),
                          framing: Framing(center: P2(0.4, -1.1), unitsPerPixel: 0.03))
        let p = P3<WorldSpace>(2.5, -7, 3.25)
        let before = n.framing.pixel(of: n.camera.view(p).xy, in: v)
        let after = n.applying(.retarget(p), in: v)
        let moved = after.framing.pixel(of: after.camera.view(p).xy, in: v)
        Check.expect(simd_reduce_max(abs(moved - before)) < 1e-9,
                     "the clicked point stays under the cursor")
        Check.expect(after.orbit.target == p, "and becomes the centre of rotation")

        // Orbiting now leaves it alone, which is the point of re-targeting.
        let turned = after.applying(.orbit(SIMD2(120, 25)), in: v)
        let still = turned.framing.pixel(of: turned.camera.view(p).xy, in: v)
        Check.expect(simd_reduce_max(abs(still - before)) < 1e-6,
                     "and stays put through a turn", "moved \(simd_reduce_max(abs(still - before))) px")
    }

    Check.suite("navigation: fitting shows everything, with a margin") {
        let v = Viewport(width: 900, height: 600)
        let bounds = AABB<ViewSpace>(lo: SIMD3(-4, -2, 0), hi: SIMD3(6, 3, 1))
        let f = Framing.fitting(bounds, in: v)
        let corners = [SIMD2(-4.0, -2.0), SIMD2(6, -2), SIMD2(-4, 3), SIMD2(6, 3)]
        var inside = true
        for c in corners {
            let px = f.pixel(of: P2<ViewSpace>(c.x, c.y), in: v)
            if px.x < 0 || px.x > 900 || px.y < 0 || px.y > 600 { inside = false }
        }
        Check.expect(inside, "every corner of the bounds lands on screen")
        Check.expect(abs(f.center.x - 1) < 1e-12 && abs(f.center.y - 0.5) < 1e-12,
                     "centred on the bounds")
    }
}

/// A deterministic generator. The property tests want the same hundred cases
/// every run, so a failure is reproducible rather than a story about last
/// Tuesday.
final class SplitMix {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    func nextBits() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    func next(_ lo: Double, _ hi: Double) -> Double {
        lo + (hi - lo) * Double(nextBits() >> 11) * (1.0 / 9007199254740992.0)
    }
}

// MARK: - 6. the bake

func bakeTests() {
    // Tiling splits the render, never the clip. Clipping each pass separately
    // breaks a path at the seam -- the segment joining the last vertex on one
    // side to the first on the other is drawn by neither -- so the passes are
    // stitched into one depth image and the clip runs once. The property that
    // buys is this one: however the pass was divided, the drawing is the same.
    Check.suite("bake: a tiled bake draws what a single-pass bake draws") {
        guard let url = Fixtures.contractBundles.first(where: {
            $0.lastPathComponent == "uniform_mesh.kurven"
        }) else { Check.expect(false, "the uniform_mesh fixture is present"); return }

        let bundle = try KurvenBundle.read(at: url)
        let scene = Scene(bundle: bundle, preset: try bundle.manifest.preset("recip"))
        let renderer = try MetalRenderer()

        let whole = try renderer.bake(scene, options: BakeOptions(resolution: 240, tiles: 1))
        for n in [2, 3, 5] {
            let split = try renderer.bake(scene, options: BakeOptions(resolution: 240, tiles: n))
            // Neither the coverage nor the values are bit-identical, and
            // neither can be: each pass derives its NDC mapping from its own
            // sub-frame in float32, so a triangle edge landing on a pixel
            // centre can round to either side of it and the interpolated depth
            // across a triangle rounds slightly differently. Both effects are
            // confined to triangle edges and to the last few bits of a float.
            // What must hold is the consequence -- that no visibility decision
            // changes -- and the stroke check below asserts exactly that.
            var uncovered = 0
            var worst: Float = 0
            var span: Float = 0
            for i in whole.depth.values.indices {
                let a = whole.depth.values[i], b = split.depth.values[i]
                // "Covered" is `> empty`, not `isFinite`: the GPU clear value is
                // a large negative sentinel, because Metal cannot clear a float
                // attachment to infinity.
                let ca = a > whole.depth.empty, cb = b > split.depth.empty
                if ca != cb { uncovered += 1; continue }
                if ca { worst = max(worst, abs(a - b)); span = max(span, abs(a)) }
            }
            let coverage = Double(uncovered) / Double(whole.depth.values.count)
            Check.expect(coverage < 0.001,
                         "\(n)x\(n) passes cover the same pixels bar triangle edges",
                         "\(uncovered) of \(whole.depth.values.count) differ")
            Check.expect(worst <= 1e-5 * max(span, 1),
                         "\(n)x\(n) passes agree on depth to float32 rounding",
                         "worst |Δ| \(worst) over a span of \(span)")

            var strokesMatch = split.strokes.layers.count == whole.strokes.layers.count
            for (a, b) in zip(whole.strokes.layers, split.strokes.layers) {
                strokesMatch = strokesMatch && a.paths == b.paths
            }
            Check.expect(strokesMatch, "\(n)x\(n) passes produce the same strokes")
        }
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
navigationTests()
shaderTests()
bakeTests()
exit(Check.summary())
