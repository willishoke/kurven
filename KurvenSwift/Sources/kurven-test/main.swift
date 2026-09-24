import Foundation
import simd
import KurvenCore
import KurvenMetal
import KurvenBake
import KurvenService
import KurvenLandscape

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
        Check.expect(bundles.count == 4, "four fixture bundles are present",
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
            let clip = frame.metalClip
            var worst = 0.0
            var roundTrips = true
            for r in 0..<frame.rows {
                for c in 0..<frame.cols {
                    let p = frame.coordinate(row: r, col: c)
                    if frame.index(of: p) != SIMD2(r, c) { roundTrips = false }
                    // Metal samples pixel (col, row) at NDC
                    // ((2c+1)/cols - 1, 1 - (2r+1)/rows): y from the top.
                    let h = clip * SIMD4<Double>(p.x, p.y, 0, 1)
                    worst = max(worst, abs(h.x / h.w - (Double(2 * c + 1) / Double(frame.cols) - 1)))
                    worst = max(worst, abs(h.y / h.w - (1 - Double(2 * r + 1) / Double(frame.rows))))
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
                                    RealBand(below: -1.5, cap: 4)], beyond: .infinity)
        Check.expect(bands.height(atX: -4) == 8 && bands.height(atX: -3) == 6
                     && bands.height(atX: -2) == 4 && bands.height(atX: 0) == .infinity,
                     "the first matching band wins, and past the last there is none")
        // Gamma's calm region: past the spires there is still a cap.
        let calm = Caps.realBands([RealBand(below: -3.5, cap: 2.5),
                                   RealBand(below: 0.5, cap: 5.5)], beyond: 3.0)
        Check.expect(calm.height(atX: -4) == 2.5 && calm.height(atX: 0) == 5.5
                     && calm.height(atX: 2) == 3.0,
                     "and `beyond` is the cap where no band matches")
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
        for name in ["kv_height_vertex", "kv_mesh_vertex", "kv_depth_fragment",
                     "kv_param_vertex", "kv_param_depth_fragment", "kv_coord_fragment"] {
            Check.expect(library.makeFunction(name: name) != nil,
                         "the library exposes \(name)")
        }
        _ = try MetalRenderer(device: device)
        Check.expect(true, "both depth pipelines build with a MAX blend on r32Float")
    }

    // The claim `ShaderTypes.h` exists to make is that CPU and GPU agree on the
    // uniform layout by construction. Nothing at build time enforces it here --
    // there is no `metal` compiler, and SwiftPM does not track a C header as a
    // dependency of the Swift targets that import it, so an incremental build
    // after editing the header can leave the two sides disagreeing. That failure
    // mode is a corrupted uniform and a segfault, not a compile error, which is
    // exactly the kind that deserves a test.
    Check.suite("metal: the shader sees the uniform struct the CPU sent") {
        guard let device = MTLCreateSystemDefaultDevice() else {
            Check.expect(false, "a Metal device exists"); return
        }
        // Distinct values everywhere, so a field that lands on its neighbour's
        // bytes is visible rather than plausible.
        var v = matrix_identity_float4x4
        for c in 0..<4 { for r in 0..<4 { v[c][r] = Float(1 + c * 4 + r) } }
        var clipProbe = matrix_identity_float4x4
        for c in 0..<4 { for r in 0..<4 { clipProbe[c][r] = Float(101 + c * 4 + r) } }
        let sent = KVUniforms(
            view: v,
            clip: clipProbe,
            domainLo: SIMD2(301, 302),
            domainSize: SIMD2(401, 402),
            lattice: SIMD2(501, 502),
            gridSize: SIMD2(601, 602),
            step: 701,
            cap: 801,
            regionCount: 901,
            empty: -1001)
        // KVShading too. Its `simd_float3` is sixteen bytes with three used;
        // spelling it `packed_float3` in the shader is twelve, which shifts
        // every field after it. That mismatch compiled cleanly, rendered a
        // black landscape, and is the reason this struct is probed as well.
        let shading = KVShading(
            color: SIMD4(11, 12, 13, 14),
            margin: 21,
            empty: 22,
            slopeScale: 23,
            lightDirection: SIMD3(31, 32, 33),
            ambient: 41,
            strokeWidth: 42,
            depthRange: SIMD2(51, 52),
            viewport: SIMD2(61, 62))
        let surface = KVSurface(samples: SIMD2(71, 72), cells: SIMD2(73, 74),
                                lo: SIMD2(75, 76), spacing: SIMD2(77, 78),
                                depthRange: SIMD2(79, 80))
        let inkVertex = KVInkVertex(position: SIMD3(81, 82, 83), normal: SIMD3(84, 85, 86),
                                    coord: SIMD2(87, 88))
        let inkCamera = KVInk(sight: SIMD3(89, 90, 91), eye: SIMD3(92, 93, 94),
                              perspective: 95, onFolds: 96)

        var want: [Float] = []
        for c in 0..<4 { for r in 0..<4 { want.append(v[c][r]) } }
        for c in 0..<4 { for r in 0..<4 { want.append(clipProbe[c][r]) } }
        want += [301, 302, 401, 402, 501, 502, 601, 602, 701, 801, 901, -1001]
        want += [11, 12, 13, 14, 21, 22, 23, 31, 32, 33, 41, 42, 51, 52, 61, 62]
        want += [71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
        want += Array(81...96).map(Float.init)

        let probe = try MetalRenderer.probeUniformLayout(device: device,
                                                         sending: sent, and: shading,
                                                         and: surface,
                                                         and: (inkVertex, inkCamera))
        var mismatch: Int?
        for i in want.indices where i < probe.fields.count && probe.fields[i] != want[i] {
            mismatch = i; break
        }
        Check.expect(probe.fields.count == want.count && mismatch == nil,
                     "every field of all five structs arrives with the value it was given",
                     mismatch.map { "field \($0): sent \(want[$0]), saw \(probe.fields[$0])" } ?? "")
        let cpu = [MemoryLayout<KVUniforms>.size, MemoryLayout<KVShading>.size,
                   MemoryLayout<KVSurface>.size, MemoryLayout<KVInkVertex>.size,
                   MemoryLayout<KVInk>.size]
        Check.expect(probe.sizes == cpu, "and the two agree on all five sizes",
                     "GPU \(probe.sizes), CPU \(cpu)")
    }
}

// MARK: - 8. contouring

func contourTests() {
    // Comparing two sets of polylines directly means comparing decisions the
    // shape does not depend on: where a closed loop starts, which way round it
    // goes, whether an open run was traced from one end or the other. The
    // canonical form is the multiset of *segment midpoints*, which is invariant
    // to all three and still distinguishes any two different curve sets.
    func midpoints(_ paths: [[P2<DomainSpace>]]) -> [SIMD2<Double>] {
        var out: [SIMD2<Double>] = []
        for path in paths {
            for (a, b) in zip(path, path.dropFirst()) {
                out.append(SIMD2((a.x + b.x) / 2, (a.y + b.y) / 2))
            }
        }
        return out
    }

    /// How many midpoints of one set have no counterpart in the other, and how
    /// far the worst stray is. Sorting and zipping would be cheaper and wrong:
    /// crossings on vertical grid edges share an x exactly, so a difference in
    /// the last bit permutes the order and then compares unrelated points.
    func separation(_ a: [SIMD2<Double>], _ b: [SIMD2<Double>])
        -> (unmatched: Int, worst: Double) {
        func directed(_ from: [SIMD2<Double>], _ to: [SIMD2<Double>]) -> (Int, Double) {
            var count = 0, worst = 0.0
            for p in from {
                var best = Double.infinity
                for q in to { best = min(best, ((p - q) * (p - q)).sum()) }
                let d = best.squareRoot()
                if d > 1e-9 { count += 1; worst = max(worst, d) }
            }
            return (count, worst)
        }
        let (ca, wa) = directed(a, b)
        let (cb, wb) = directed(b, a)
        return (ca + cb, max(wa, wb))
    }

    Check.suite("contour: the native marching squares reproduces contourpy") {
        let index = try Fixtures.json("contour/index.json")
        let grid = try index.value("grid", "contour index").object("grid")
        let nx = try grid.int("nx", "grid"), ny = try grid.int("ny", "grid")
        let realBounds = try grid.doubles("real", "grid")
        let imagBounds = try grid.doubles("imag", "grid")
        let domain = Domain(real: Interval(lo: realBounds[0], hi: realBounds[1]),
                            imag: Interval(lo: imagBounds[0], hi: imagBounds[1]))
        let fields = try index.value("fields", "contour index").object("fields")

        for (name, entries) in fields.sorted(by: { $0.key < $1.key }) {
            let values = try NPY.read(contentsOf: Fixtures.url("contour/\(name).npy")).floats()
            let g = Grid2D(width: nx, height: ny, domain: domain, values: values)
            guard case .array(let list) = entries else {
                Check.expect(false, "\(name): the index lists levels"); continue
            }
            for entry in list {
                let e = try entry.object("contour entry")
                let level = try e.double("level", "contour entry")
                let tag = try e.string("tag", "contour entry")
                let wantPaths = try e.int("paths", "contour entry")

                // The fixture is in contourpy's (imag, real) column order; the
                // grid, and everything downstream of the bundle, is world order.
                let flat = try NPY.read(contentsOf: Fixtures.url("contour/\(tag).npy")).doubles()
                let offsets = try NPY.read(contentsOf: Fixtures.url("contour/\(tag).idx.npy")).ints()
                var want: [[P2<DomainSpace>]] = []
                for i in 0..<max(offsets.count - 1, 0) {
                    want.append((offsets[i]..<offsets[i + 1]).map {
                        P2<DomainSpace>(flat[2 * $0 + 1], flat[2 * $0])
                    })
                }

                let got = Contour.lines(of: g, level: level)
                let a = midpoints(got), b = midpoints(want)
                let (unmatched, worst) = separation(a, b)
                Check.expect(got.count == wantPaths && a.count == b.count && unmatched == 0,
                             "\(tag): \(got.count) paths, \(a.count) segments",
                             unmatched == 0 ? "exact"
                                : "vs \(wantPaths)/\(b.count); \(unmatched) of "
                                  + "\(a.count + b.count) midpoints unmatched, worst \(worst)")
            }
        }
    }

    Check.suite("contour: closed loops close and open runs reach the edge") {
        // A cone: one nested closed loop per level, none of them touching the
        // grid boundary.
        let n = 81
        var values = [Float](repeating: 0, count: n * n)
        for y in 0..<n {
            for x in 0..<n {
                let dx = Double(x - n / 2) / Double(n / 2)
                let dy = Double(y - n / 2) / Double(n / 2)
                values[y * n + x] = Float(1 - (dx * dx + dy * dy).squareRoot())
            }
        }
        let g = Grid2D(width: n, height: n,
                       domain: Domain(real: Interval(lo: -1, hi: 1),
                                      imag: Interval(lo: -1, hi: 1)),
                       values: values)
        let loops = Contour.lines(of: g, level: 0.5)
        Check.expect(loops.count == 1, "one level set, one loop", "\(loops.count)")
        if let loop = loops.first {
            let closed = loop.first == loop.last
            Check.expect(closed, "and it closes")
            // Radius 0.5 of a unit cone: circumference pi.
            var length = 0.0
            for (a, b) in zip(loop, loop.dropFirst()) {
                length += (SIMD2(b.x - a.x, b.y - a.y) * SIMD2(b.x - a.x, b.y - a.y)).sum().squareRoot()
            }
            Check.expect(abs(length - .pi / 2 * 2) < 0.02,
                         "with the circumference of a circle of radius 1/2",
                         "\(length) vs \(Double.pi)")
        }

        // A ramp: one open run per level, ending on the grid boundary.
        var ramp = [Float](repeating: 0, count: n * n)
        for y in 0..<n { for x in 0..<n { ramp[y * n + x] = Float(x) / Float(n - 1) } }
        let r = Grid2D(width: n, height: n, domain: g.domain, values: ramp)
        let runs = Contour.lines(of: r, level: 0.5)
        Check.expect(runs.count == 1 && runs[0].first != runs[0].last,
                     "a ramp gives one open run, not a loop")
        if let run = runs.first {
            let straight = run.allSatisfy { abs($0.x - 0.0) < 1e-9 }
            Check.expect(straight, "at exactly the level's coordinate")
        }
    }
}

// MARK: - the hatching

/// The four hatching sources, against `kurven.hatch`, which defines them.
///
/// Every height in the fixture was read out of the grid the fixture ships, by
/// the transcription of `Grid2D.sample` that `kurven.hatch.grid_sampler` is. So
/// this is not a tolerance test of two approximations: it is the same
/// arithmetic on the same float32 samples, and the answers are expected to
/// agree to the last few bits.
func hatchTests() {
    let index: [String: JSONValue]
    let grid: Grid2D<Float>
    let perimeter: BoundaryPerimeter
    do {
        index = try Fixtures.json("hatch/index.json")
        let domain = try Domain(json: index.value("domain", "hatch"))
        let shape = try index.value("shape", "hatch").object("hatch.shape")
        let values = try NPY.read(contentsOf: Fixtures.url("hatch/height.npy")).floats()
        grid = Grid2D(width: try shape.int("nx", "hatch.shape"),
                      height: try shape.int("ny", "hatch.shape"),
                      domain: domain, values: values)
        perimeter = try BoundaryPerimeter(json: index.value("perimeter", "hatch"))
    } catch {
        Check.suite("hatch: the fixture loads") {
            Check.expect(false, "hatch fixture loads", "\(error)")
        }
        return
    }

    Check.suite("hatch: every source reproduces kurven.hatch stroke for stroke") {
        let caps = try index.value("caps", "hatch").object("hatch.caps")
        for case .object(let c) in try index.array("cases", "hatch") {
            let name = try c.string("name", "hatch.case")
            let source = try LayerSource(json: c.value("source", "hatch.case"))
            let surface = Surface(
                height: grid, phase: nil,
                caps: try Caps(json: caps.value(try c.string("caps", "hatch.case"),
                                                "hatch.caps")))
            let context = Surface.HatchContext(perimeter: perimeter)
            guard let derived = surface.hatch(source, in: context) else {
                Check.expect(false, "\(name) is a hatching", "\(source)")
                continue
            }
            let verts = try NPY.read(contentsOf:
                Fixtures.url("hatch/" + c.string("vertices", "hatch.case"))).rows3()
            let offsets = try NPY.read(contentsOf:
                Fixtures.url("hatch/" + c.string("offsets", "hatch.case"))).ints()
            let expected = PolylineSet<WorldSpace>(vertices: verts.map { P3($0) },
                                                   offsets: offsets)
            guard derived.count == expected.count else {
                Check.expect(false, "\(name): \(expected.count) strokes",
                             "derived \(derived.count)")
                continue
            }
            // A stroke is a stroke, vertex for vertex -- both sides walked the
            // same samples in the same order. A *rim* is a marching-squares
            // contour, and a closed loop has no first vertex: contourpy and
            // this walk can start the same curve at different points on it. So
            // the rim is compared the way `contourTests` compares a contour, as
            // the set of segments it draws.
            if case .capOutline = source {
                let a = segmentMidpoints(derived), b = segmentMidpoints(expected)
                let (unmatched, worst) = strayed(a, b)
                Check.expect(a.count == b.count && unmatched == 0,
                             "\(name): \(derived.count) loops, segment for segment",
                             unmatched == 0 ? "exact"
                                : "\(unmatched) of \(a.count + b.count) segments "
                                  + "unmatched, worst \(worst)")
                continue
            }
            var worst = 0.0
            var shapeOK = true
            for i in 0..<derived.count {
                let a = derived[path: i], b = expected[path: i]
                guard a.count == b.count else { shapeOK = false; break }
                for (p, q) in zip(a, b) {
                    worst = max(worst, simd_reduce_max(simd_abs(p.v - q.v)))
                }
            }
            Check.expect(shapeOK && worst < 1e-9,
                         "\(name): \(derived.count) strokes, vertex for vertex",
                         shapeOK ? "max |Δ| = \(worst)" : "a stroke has a different length")
        }
    }

    // The rule for choosing contour levels under a cap lives on both sides:
    // Python picks them when it builds a landscape, and the app picks them
    // again whenever the cap slider moves, because a level list is absolute and
    // a raised cap would otherwise leave the top of the plate blank. Two
    // implementations of one rule, pinned to each other here.
    Check.suite("landscape: the derived numbers agree with kurven.landscape") {
        let style = try index.value("style", "hatch").object("hatch.style")
        var niceOK = true
        for case .object(let n) in try style.array("nice", "style") {
            let x = try n.double("x", "style.nice")
            let up = try n.double("up", "style.nice")
            let down = try n.double("down", "style.nice")
            niceOK = niceOK && LandscapeStyle.nice(x) == up
                && LandscapeStyle.nice(x, down: true) == down
        }
        Check.expect(niceOK, "nice() rounds the way Python's does")

        var levelsOK = true
        var detail = ""
        for case .object(let l) in try style.array("levels", "style") {
            let ceiling = try l.double("ceiling", "style.levels")
            let (major, minor) = LandscapeStyle.magnitudeLevels(upTo: ceiling)
            let wantMajor = try l.doubles("major", "style.levels")
            let wantMinor = try l.doubles("minor", "style.levels")
            let same = major.count == wantMajor.count && minor.count == wantMinor.count
                && zip(major, wantMajor).allSatisfy { abs($0 - $1) < 1e-9 }
                && zip(minor, wantMinor).allSatisfy { abs($0 - $1) < 1e-9 }
            if !same, detail.isEmpty {
                detail = "at ceiling \(ceiling): \(major.count)/\(minor.count) "
                    + "vs \(wantMajor.count)/\(wantMinor.count)"
            }
            levelsOK = levelsOK && same
        }
        Check.expect(levelsOK, "magnitudeLevels() chooses the levels Python chose", detail)
    }
}

/// Each segment's midpoint, which is where two walks of the same curve agree
/// even when they disagree about which vertex is first.
func segmentMidpoints(_ paths: PolylineSet<WorldSpace>) -> [SIMD3<Double>] {
    var out: [SIMD3<Double>] = []
    for i in 0..<paths.count {
        let path = paths[path: i]
        for (a, b) in zip(path, path.dropFirst()) { out.append((a.v + b.v) / 2) }
    }
    return out
}

/// How many midpoints of one set have no counterpart in the other. Sorting and
/// zipping would be cheaper and wrong, for the reason `separation` gives.
func strayed(_ a: [SIMD3<Double>], _ b: [SIMD3<Double>]) -> (unmatched: Int, worst: Double) {
    func directed(_ from: [SIMD3<Double>], _ to: [SIMD3<Double>]) -> (Int, Double) {
        var count = 0, worst = 0.0
        for p in from {
            var best = Double.infinity
            for q in to { best = min(best, simd_length_squared(p - q)) }
            let d = best.squareRoot()
            if d > 1e-9 { count += 1; worst = max(worst, d) }
        }
        return (count, worst)
    }
    let (ca, wa) = directed(a, b)
    let (cb, wb) = directed(b, a)
    return (ca + cb, max(wa, wb))
}

/// Restyling: the edit path a cap slider takes.
func restyleTests() {
    Check.suite("restyle: a new cap re-derives the ink and keeps the landscape") {
        let url = Fixtures.url("contract/landscape.kurven")
        guard FileManager.default.fileExists(atPath: url.path) else {
            Check.expect(false, "the landscape fixture bundle is present"); return
        }
        let bundle = try KurvenBundle.read(at: url)
        var manifest = bundle.manifest
        manifest.caps = .uniform(0.9)
        let restyled = bundle.restyled(manifest)

        Check.expect(restyled.surface.height.values == bundle.surface.height.values,
                     "the samples are untouched -- this is a restyling, not a resampling")
        Check.expect(restyled.surface.caps == .uniform(0.9),
                     "the cap is the new one")
        // Lowering the cap lowers every wall crest, so the wall hatch is
        // strictly shorter; the rim moves outward, so the cap layers change too.
        func height(_ b: KurvenBundle, _ name: String) -> Double {
            ((try? b.layer(name))?.paths.vertices.map(\.z).max()) ?? 0
        }
        Check.expect(height(restyled, "wall_hatch") <= 0.9 + 1e-9,
                     "the wall hatch stops at the new cap",
                     "\(height(restyled, "wall_hatch"))")
        Check.expect(height(bundle, "wall_hatch") > 0.9,
                     "and reached higher before it",
                     "\(height(bundle, "wall_hatch"))")
        Check.expect(restyled.walls().vertices.map(\.z).max() ?? 0 <= 0.9 + 1e-9,
                     "and so do the walls it hatches")

        // A dumped layer cannot be re-derived, so restyling must leave it be
        // rather than quietly emptying it.
        let dumped = try KurvenBundle.read(at: Fixtures.url("contract/uniform_mesh.kurven"))
        var other = dumped.manifest
        other.caps = .uniform(1.0)
        let keptA = try dumped.layer("mag").paths.count
        let keptB = try dumped.restyled(other).layer("mag").paths.count
        Check.expect(keptA == keptB && keptA > 0,
                     "a dumped layer survives a restyling unchanged", "\(keptA) paths")
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

    Check.suite("navigation: perspective is a way of looking, not a plate style") {
        let v = Viewport(width: 1200, height: 800)
        let plate = plates[1].1
        let bounds = AABB<ViewSpace>(lo: SIMD3(-4, -2, 0), hi: SIMD3(6, 3, 4))
        let ortho = Navigator(orbit: Orbit(matching: plate),
                              framing: Framing(center: P2(0, 0), unitsPerPixel: 0.01))
        Check.expect(!ortho.orbit.isPerspective && !ortho.camera.isPerspective,
                     "a preset camera is orthographic")

        let persp = ortho.applying(.project(fieldOfView: Angle(degrees: 50), bounds), in: v)
        Check.expect(persp.camera.isPerspective && persp.orbit.distance != nil,
                     "and .project makes one that is not")

        // The oblique shear is a property of an orthographic drawing. The ADT
        // makes "shear under perspective" unrepresentable; this checks the
        // camera actually takes that route rather than carrying it along.
        if case .perspective = persp.camera.projection {
            Check.expect(true, "with no oblique shear to carry")
        } else {
            Check.expect(false, "with no oblique shear to carry")
        }

        // What it is looking at ends up in the middle.
        let centre = persp.camera.view(persp.orbit.target)
        Check.expect(abs(centre.x) < 1e-9 && abs(centre.y) < 1e-9,
                     "the target lands at the view origin",
                     "(\(centre.x), \(centre.y))")
        Check.expect(centre.z < 0, "with the landscape in front of the eye",
                     "z = \(centre.z)")

        // Nearer things are bigger. That is the entire visible difference, and
        // it is what the orthographic camera cannot do.
        let clip = Camera.perspectiveClip(fovY: persp.orbit.fieldOfView, aspect: v.aspect)
        func widthOnScreen(atDepth z: Double) -> Double {
            let h = clip * SIMD4<Double>(1, 0, z, 1)
            return abs(h.x / h.w)
        }
        Check.expect(widthOnScreen(atDepth: -5) > widthOnScreen(atDepth: -50),
                     "the same span is wider when it is nearer",
                     "\(widthOnScreen(atDepth: -5)) vs \(widthOnScreen(atDepth: -50))")

        // And going back to orthographic restores the framing.
        let back = persp.applying(.project(fieldOfView: nil, bounds), in: v)
        Check.expect(!back.camera.isPerspective, "and .project(nil) goes back")

        // Zoom means coming closer, not shrinking a rectangle.
        let closer = persp.applying(.zoom(factor: 2, at: SIMD2(600, 400)), in: v)
        Check.expect(closer.orbit.distance! < persp.orbit.distance!
                     && closer.framing == persp.framing,
                     "zooming moves the eye and leaves the framing alone",
                     "\(persp.orbit.distance!) -> \(closer.orbit.distance!)")
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
    // The plates are orthographic and oblique; perspective is for navigating.
    // A bake indexes its depth buffer by an affine map from view coordinates to
    // pixels, which under perspective depends on depth -- and the fix would
    // produce the one artifact this design has no Python oracle for. Refusing
    // is the honest form of that, and it is refused rather than wrong.
    Check.suite("bake: a perspective camera is refused, with the reason") {
        guard let url = Fixtures.contractBundles.first(where: {
            $0.lastPathComponent == "uniform_mesh.kurven"
        }) else { Check.expect(false, "the fixture is present"); return }
        let bundle = try KurvenBundle.read(at: url)
        var scene = Scene(bundle: bundle, preset: try bundle.manifest.preset("recip"))
        let renderer = try MetalRenderer()

        _ = try renderer.bake(scene, options: BakeOptions(resolution: 64))
        Check.expect(true, "an orthographic scene bakes")

        var orbit = Orbit(matching: try bundle.manifest.preset("recip").plate)
        orbit.distance = 10
        scene.camera = orbit.camera
        do {
            _ = try renderer.bake(scene, options: BakeOptions(resolution: 64))
            Check.expect(false, "a perspective scene does not")
        } catch let error as BakeError {
            Check.expect("\(error)".contains("orthographic"),
                         "a perspective scene does not, and says why",
                         "\(error)".split(separator: "\n").first.map(String.init) ?? "")
        }
    }

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

// MARK: - the preview

func previewTests() {
    // A plane seen nearly edge-on: z = -x from forty degrees below the plate's
    // horizon, so its view depth changes by about eleven units per unit of
    // screen height. The triangulation of a plane is the plane, so the answer
    // is known at every pixel -- ink drawn on it is visible, ink a little under
    // it is hidden -- which is what a real landscape cannot offer.
    //
    // The preview compares a line's depth where it crosses a pixel with the
    // surface's depth at the pixel's centre. Here half a pixel of that is
    // several times the margin, so the bake's predicate alone discards much of
    // the ink on the plane, and while orbiting what it discards crawls. With
    // the slope allowance all of it must survive, and the allowance must still
    // fall far short of hiding depth: the ink under the plane stays hidden.
    Check.suite("preview: ink on a surface steep in view is kept, and ink under it is not") {
        let n = 65
        let domain = Domain(real: Interval(lo: -1, hi: 1), imag: Interval(lo: -1, hi: 1))
        var heights: [Float] = []
        for _ in 0..<n {
            for x in 0..<n { heights.append(Float(1 - 2 * Double(x) / Double(n - 1))) }
        }
        let surface = Surface(
            height: Grid2D(width: n, height: n, domain: domain, values: heights),
            phase: nil, caps: .none)
        // Slightly slanted on screen, so each stroke crosses a few rows and is
        // tested at every sub-pixel offset rather than at one.
        func strokes(under depth: Double) -> Layer {
            let paths = stride(from: -0.6, through: 0.45, by: 0.15).map { x0 -> [P3<WorldSpace>] in
                [P3(x0, -0.9, -x0 - depth), P3(x0 + 0.1, 0.9, -(x0 + 0.1) - depth)]
            }
            return Layer(spec: LayerSpec(name: "ink", role: .magnitude,
                                         source: .file(vertices: "", offsets: ""),
                                         width: 0.4, heightPolicy: .surface),
                         paths: PolylineSet(paths: paths))
        }
        let style = PlateStyle(shear: 0, flipX: false, yScale: nil)
        let orbit = Orbit(target: P3(0, 0, 0), azimuth: Angle(degrees: 0),
                          elevation: Angle(degrees: -40), style: style)
        let viewport = Viewport(width: 512, height: 128)
        func scene(_ layer: Layer, margin: Double) -> Scene {
            Scene(surface: surface, occluder: Mesh(vertices: [], triangles: []),
                  tiles: [.identity], step: 1, layers: [layer], camera: orbit.camera,
                  margin: margin)
        }
        guard let bounds = scene(strokes(under: 0), margin: 0).quickBounds() else {
            Check.expect(false, "the plane has bounds"); return
        }
        let navigator = Navigator(orbit: orbit, framing: .fitting(bounds, in: viewport))

        let renderer = try MetalRenderer()
        let target = try renderer.makePreviewTarget(viewport)
        func ink(_ layer: Layer, margin: Double, slope: Float) throws -> Int {
            try renderer.renderPreview(scene(layer, margin: margin), navigator: navigator,
                                       viewport: viewport,
                                       options: PreviewOptions(slopeScale: slope),
                                       into: target)
            let bgra = try PNG.bgra(target)
            return stride(from: 0, to: bgra.count, by: 4).reduce(0) { count, k in
                let luma = 299 * Int(bgra[k + 2]) + 587 * Int(bgra[k + 1]) + 114 * Int(bgra[k])
                // Any visible mark, as `compare_preview.py` counts ink.
                return count + (luma < 224 * 1000 ? 1 : 0)
            }
        }

        let margin = 0.005
        let on = strokes(under: 0), under = strokes(under: 0.05)
        let drawn = try ink(on, margin: .infinity, slope: 0)
        let bake = try ink(on, margin: margin, slope: 0)
        let kept = try ink(on, margin: margin, slope: PreviewOptions.defaultSlopeScale)
        Check.expect(drawn > 1000 && Double(bake) < 0.9 * Double(drawn),
                     "the plane is steep enough that the bake's predicate loses ink on it",
                     "\(bake) of \(drawn) px")
        Check.expect(Double(kept) >= 0.99 * Double(drawn),
                     "with the slope allowance, the ink on it is kept",
                     "\(kept) of \(drawn) px")

        let behind = try ink(under, margin: .infinity, slope: 0)
        let shown = try ink(under, margin: margin, slope: PreviewOptions.defaultSlopeScale)
        Check.expect(behind > 1000 && Double(shown) <= 0.01 * Double(behind),
                     "and ink just under it is still hidden",
                     "\(shown) of \(behind) px show")
    }
}

// MARK: - 9. the service

func serviceTests() {
    // The service is optional infrastructure: a bundle is a complete document
    // without it, and a machine with no checkout has nothing to connect to. So
    // a missing one is reported and skipped rather than failed -- the tests
    // that matter are the ones about the file format, and they do not need it.
    Check.suite("service: round trip through the protocol") {
        let repo = Fixtures.dir.deletingLastPathComponent().deletingLastPathComponent()
        guard let command = Service.Command.autodetect(near: repo) else {
            print("  --    no kurven/serve.py near \(repo.path); skipped")
            return
        }
        let service = try Service(command: command)
        defer { service.stop() }

        let description = try blocking { try await service.describe() }
        Check.expect(description.protocolVersion == Service.protocolVersion,
                     "the server speaks the protocol this client does",
                     "\(description.protocolVersion)")
        Check.expect(description.examples.contains { $0.name == "recip" },
                     "it offers the examples",
                     description.examples.map(\.name).joined(separator: ", "))

        guard let recip = description.example("recip"), recip.available else {
            Check.expect(false, "recip is available"); return
        }
        Check.expect(recip.arguments.contains { $0.name == "res" && $0.kind == .int },
                     "and reports their options with types",
                     "\(recip.arguments.count) options")

        // A small bundle, described rather than dumped, built and read back.
        // This is the whole loop: ask for a landscape, get a path, open it.
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("kurven-service-\(UUID().uuidString).kurven")
        defer { try? FileManager.default.removeItem(at: output) }
        let result = try blocking {
            try await service.export(example: "recip", to: output,
                                     arguments: ["res": "120", "occluder_res": "60"],
                                     derived: true)
        }
        Check.expect(result.manifest.provenance.example == "recip",
                     "the bundle records which example made it")
        Check.expect(result.manifest.layers.contains { $0.files == nil },
                     "and --derived describes its contour layers")

        let bundle = try KurvenBundle.read(at: result.url)
        Check.expect(bundle.surface.height.width == 120,
                     "the resampled grid is the size asked for",
                     "\(bundle.surface.height.width)")
        Check.expect(bundle.layers.contains { !$0.paths.isEmpty },
                     "and its described layers derive to actual ink")

        // The other half of the protocol: not "run this published plate" but
        // "sample this function". A landscape is the one thing a frozen bundle
        // cannot be, and the one thing the file format alone cannot test.
        let catalog = try blocking { try await service.catalog() }
        Check.expect(catalog.presets.count >= 10 && catalog.functions.count >= 20,
                     "the catalog offers presets and the language behind them",
                     "\(catalog.presets.count) presets, \(catalog.functions.count) functions")
        Check.expect(catalog.presets.allSatisfy {
                         $0.domain.real.length > 0 && $0.window.real.length
                             >= $0.domain.real.length
                     },
                     "every preset's window has room around its domain")

        Check.expect((try? blocking { try await service.validate("1/Γ(z)") }) == "1/gamma(z)",
                     "an expression validates to its canonical spelling")
        do {
            _ = try blocking { try await service.validate("gamma(z") }
            Check.expect(false, "an unclosed call is refused")
        } catch let error as Service.Failure {
            if case .remote(let kind, let message) = error {
                Check.expect(kind == "badExpression",
                             "an unclosed call is refused, by kind", message)
            } else {
                Check.expect(false, "an unclosed call is refused", "\(error)")
            }
        }

        let landscapeOutput = FileManager.default.temporaryDirectory
            .appendingPathComponent("kurven-landscape-\(UUID().uuidString).kurven")
        defer { try? FileManager.default.removeItem(at: landscapeOutput) }
        guard let preset = catalog.preset("gamma") else {
            Check.expect(false, "the catalog has gamma"); return
        }
        var request = LandscapeRequest(preset: preset, resolution: 200)
        request.caps = .realBands([RealBand(below: -2.5, cap: 3), RealBand(below: 0.5, cap: 4)],
                                  beyond: 5)
        let wanted = request
        let sampled = try blocking { try await service.landscape(wanted, to: landscapeOutput) }
        let landscape = try KurvenBundle.read(at: sampled.url)
        Check.expect(landscape.manifest.caps == wanted.caps,
                     "the landscape is truncated the way it was asked to be")
        Check.expect(landscape.manifest.layers.allSatisfy { $0.files == nil },
                     "every layer of a landscape is a description",
                     "\(landscape.manifest.layers.count) layers")
        let kinds = landscape.manifest.layers.map(\.source)
        Check.expect(kinds.contains { if case .capHatch = $0 { true } else { false } }
                     && kinds.contains { if case .wallHatch = $0 { true } else { false } }
                     && kinds.contains { if case .capOutline = $0 { true } else { false } }
                     && kinds.contains { if case .wallOutline = $0 { true } else { false } },
                     "including the hatching of its sides and its tops")
        Check.expect(landscape.layers.allSatisfy { !$0.paths.isEmpty },
                     "and every one of them derives to actual ink",
                     landscape.layers.map { "\($0.spec.name) \($0.paths.count)" }
                         .joined(separator: " "))
        // The levels Python chose for this cap are the ones this side would
        // choose for it: the rule the cap slider re-applies is the same rule.
        if let major = landscape.manifest.layers.first(where: { $0.name == "mag_major" }),
           case .contour(_, let levels, _, _) = major.source {
            let mine = LandscapeStyle.levels(under: wanted.caps!).major
            Check.expect(levels.count == mine.count
                         && zip(levels, mine).allSatisfy { abs($0 - $1) < 1e-9 },
                         "and its contour levels are the ones this side derives",
                         "\(levels.count) levels up to \(levels.last ?? 0)")
        }

        // Errors arrive typed, not as a string to parse.
        do {
            _ = try blocking {
                try await service.export(example: "recip", to: output,
                                         arguments: ["resolution": "120"])
            }
            Check.expect(false, "an unknown option is refused")
        } catch let error as Service.Failure {
            if case .remote(let kind, _) = error {
                Check.expect(kind == "unknownArgument",
                             "an unknown option is refused, by kind", kind)
            } else {
                Check.expect(false, "an unknown option is refused", "\(error)")
            }
        }
    }
}

/// Run an async call from this synchronous program.
func blocking<T: Sendable>(_ body: @escaping @Sendable () async throws -> T) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var outcome: Result<T, Error>!
    Task {
        do { outcome = .success(try await body()) }
        catch { outcome = .failure(error) }
        semaphore.signal()
    }
    semaphore.wait()
    return try outcome.get()
}

// MARK: - entry

import Metal
import Dispatch
import KurvenShaderTypes

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
contourTests()
hatchTests()
restyleTests()
shaderTests()
previewTests()
bakeTests()
serviceTests()
expressionTests()
literalTests()
landscapeTests()
refinementTests()
surfaceTests()
surfaceGPUTests()
surfaceBakeTests()
periodicTorusTests()
surfacePreviewTests()
foldTests()
odeTests()
frequencyTests()
exit(Check.summary())
