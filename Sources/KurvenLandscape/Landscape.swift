import Foundation
import Dispatch
import KurvenCore
import KurvenMath

/// A landscape sampled here: `kurven.landscape`, natively.
///
/// The Python module's contract is that a landscape is *derived* -- an
/// expression, a rectangle, a resolution and a truncation are the whole of
/// what it is, and every layer of its bundle is a description of curves rather
/// than the curves. That is what let the frontend restyle without asking Python.
/// This is the other half of the same idea: with the evaluator in
/// `KurvenMath`, the sampling does not need Python either, so a landscape is a
/// value this process builds from a `LandscapeRequest` in the time it takes to
/// evaluate f on a grid.
///
/// The rules were ported line for line, and when the Python builder was
/// retired the bundles in `tests/fixtures/expr` were byte-identical from both
/// (but for the git sha); they are golden now. Where Python rounds (`round()`
/// is half-to-even), this rounds the same way; where numpy's `linspace`
/// computes `start + i * step`, so does this.
public enum NativeLandscape {
    /// Samples along the longer side of the domain.
    public static let defaultResolution = 600
    /// The occluder's heightfield is decimated to about this many samples a side.
    public static let occluderResolution = 800
    /// Where a non-finite sample is put instead: past any truncation, and still a number.
    public static let huge = 1e12

    public static let projection = ObliqueProjection(shear: 0.5, xAngle: -55.0, zAngle: -90.0,
                                              flipX: true, yScale: nil)
    public static let defaultDepthTolerance = 0.02
    public static let defaultBuffer = 4000

    /// Phase contours: every 90 degrees major, every 30 minor.
    public static let phaseMajor: [Double] = linspace(-.pi, .pi, 5)
    public static let phaseMinor: [Double] = linspace(-.pi, .pi, 13).filter { v in
        !phaseMajor.contains { abs(v - $0) < 1e-12 }
    }

    /// numpy's `linspace`: `start + i * step`, with the last sample exactly `stop`.
    public static func linspace(_ lo: Double, _ hi: Double, _ n: Int) -> [Double] {
        guard n > 1 else { return n == 1 ? [lo] : [] }
        let step = (hi - lo) / Double(n - 1)
        return (0..<n).map { $0 == n - 1 ? hi : lo + Double($0) * step }
    }

    /// Python's `round()`: half to even.
    static func pyRound(_ x: Double) -> Int { Int(x.rounded(.toNearestOrEven)) }

    // MARK: the derived numbers

    /// `(nReal, nImag)` for a resolution given along the longer side, with
    /// square cells.
    public static func gridShape(_ domain: Domain, resolution: Int) -> (nReal: Int, nImag: Int) {
        let dr = abs(domain.real.hi - domain.real.lo)
        let di = abs(domain.imag.hi - domain.imag.lo)
        let longest = max(dr, di)
        if longest <= 0 { return (max(resolution, 2), max(resolution, 2)) }
        return (max(pyRound(Double(resolution) * dr / longest), 8),
                max(pyRound(Double(resolution) * di / longest), 8))
    }

    static func cellSize(_ domain: Domain, _ shape: (nReal: Int, nImag: Int)) -> (Double, Double) {
        (abs(domain.real.hi - domain.real.lo) / Double(max(shape.nReal - 1, 1)),
         abs(domain.imag.hi - domain.imag.lo) / Double(max(shape.nImag - 1, 1)))
    }

    /// numpy's default quantile: linear interpolation between order statistics.
    static func quantile(_ sorted: [Double], _ q: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        let position = q * Double(sorted.count - 1)
        let lo = Int(position.rounded(.down))
        let hi = min(lo + 1, sorted.count - 1)
        let t = position - Double(lo)
        return sorted[lo] + (sorted[hi] - sorted[lo]) * t
    }

    /// How far up to cut, read off the samples: a spire is a top more than
    /// three times the 99th percentile, cut at a round number near it.
    public static func defaultTruncation(_ magnitudes: [Double]) -> Truncation {
        let finite = magnitudes.filter(\.isFinite).sorted()
        guard let top = finite.last else { return .none }
        let q = quantile(finite, 0.99)
        if q <= 0 || top <= 3 * q { return .none }
        return .uniform(LandscapeStyle.nice(q))
    }

    /// The highest |f| worth a contour: the truncation when there is one, else a
    /// robust maximum of the samples.
    public static func ceiling(of truncation: Truncation, magnitudes: [Double]) -> Double {
        switch truncation {
        case .uniform(let z): return z
        case .realBands(let bands, let beyond):
            let tops = bands.map(\.height) + (beyond.isFinite ? [beyond] : [])
            return tops.max() ?? 1
        case .none:
            let finite = magnitudes.filter(\.isFinite).sorted()
            return finite.isEmpty ? 1 : quantile(finite, 0.999)
        }
    }

    /// The domain's boundary as a closed traversal: front, right, back, left,
    /// so edge 0 is the real axis at the near side.
    public static func rectangle(_ domain: Domain, density: [Int]) -> BoundaryPerimeter {
        let r0 = domain.real.lo, r1 = domain.real.hi
        let i0 = domain.imag.lo, i1 = domain.imag.hi
        let corners: [(P2<WorldSpace>, P2<WorldSpace>)] = [
            (P2(r0, i0), P2(r1, i0)), (P2(r1, i0), P2(r1, i1)),
            (P2(r1, i1), P2(r0, i1)), (P2(r0, i1), P2(r0, i0)),
        ]
        return BoundaryPerimeter(edges: zip(corners, density).map {
            PerimeterEdge(start: $0.0, end: $0.1, density: $1)
        })
    }

    /// Curtain samples per edge: one per grid cell along that edge.
    public static func wallDensity(_ domain: Domain, _ shape: (nReal: Int, nImag: Int)) -> [Int] {
        let (dx, dy) = cellSize(domain, shape)
        let alongReal = max(pyRound(abs(domain.real.hi - domain.real.lo) / max(dx, 1e-12)) + 1, 2)
        let alongImag = max(pyRound(abs(domain.imag.hi - domain.imag.lo) / max(dy, 1e-12)) + 1, 2)
        return [min(alongReal, 4096), min(alongImag, 4096), min(alongReal, 4096), min(alongImag, 4096)]
    }

    /// The whole drawing, as descriptions: `kurven.landscape.default_layers`.
    public static func defaultLayers(_ domain: Domain, shape: (nReal: Int, nImag: Int),
                                     truncation: Truncation, ceiling: Double, spacing: Double?,
                                     phase: Bool = true) -> [LayerSpec] {
        let spacing = spacing ?? LandscapeStyle.hatchSpacing(domain)
        let (dx, dy) = cellSize(domain, shape)
        let step = max(dx, dy)
        var (major, minor) = LandscapeStyle.magnitudeLevels(upTo: ceiling)
        if case .none = truncation {} else {
            major = major.filter { $0 < ceiling - 1e-9 }
            minor = minor.filter { $0 < ceiling - 1e-9 }
        }
        let edges = [0, 1, 2, 3]
        var layers = [
            LayerSpec(name: "mag_major", role: .magnitude,
                      source: .contour(field: .magnitude, levels: major, restriction: .belowTruncation, tiled: false),
                      width: 0.4, heightPolicy: .level),
        ]
        if phase {
            layers.append(LayerSpec(name: "ang_major", role: .phase,
                                    source: .contour(field: .phase, levels: phaseMajor,
                                                     restriction: .belowTruncation, tiled: false),
                                    width: 0.4, heightPolicy: .magnitude))
        }
        layers.append(LayerSpec(name: "mag_minor", role: .magnitude,
                                source: .contour(field: .magnitude, levels: minor,
                                                 restriction: .belowTruncation, tiled: false),
                                width: 0.15, heightPolicy: .level))
        if phase {
            layers.append(LayerSpec(name: "ang_minor", role: .phase,
                                    source: .contour(field: .phase, levels: phaseMinor,
                                                     restriction: .belowTruncation, tiled: false),
                                    width: 0.15, heightPolicy: .magnitude))
        }
        layers += [
            LayerSpec(name: "truncation_edge", role: .scaffold, source: .truncationEdge(tiled: false),
                      width: 0.4, heightPolicy: .level),
            LayerSpec(name: "plateau_hatch", role: .scaffold,
                      source: .plateauHatching(axis: .real, spacing: spacing, tiled: false),
                      width: 0.2, heightPolicy: .level),
            LayerSpec(name: "wall_edges", role: .scaffold,
                      source: .wallEdges(edges: edges, step: step, base: 0),
                      width: 0.3, heightPolicy: .surface),
            LayerSpec(name: "wall_hatch", role: .scaffold,
                      source: .wallHatching(edges: edges, spacing: spacing, step: step,
                                         trim: true, base: 0, topOffset: 0),
                      width: 0.25, heightPolicy: .surface),
            // The outline and inner fold curves, derived per camera: the
            // bounding curves along the curved edges, as the wall outline is
            // along the straight ones.
            LayerSpec(name: "folds", role: .outline, source: .foldCurves,
                      width: 0.4, heightPolicy: .surface),
        ]
        return layers
    }

    // MARK: sampling

    /// The samples: |f| and arg f on the grid, with every non-finite value
    /// pushed to `huge` and every value past it scaled back to it, as
    /// `kurven.landscape.evaluator` does. Rows index imag, columns real.
    public struct Samples: Sendable {
        public var magnitude: [Double]
        public var phase: [Double]
        public var shape: (nReal: Int, nImag: Int)
    }

    public static func sample(_ compiled: KurvenMath.Expression.Compiled, domain: Domain,
                              resolution: Int) -> Samples {
        sample(compiled, domain: domain, shape: gridShape(domain, resolution: resolution))
    }

    /// The same at a shape given outright, for a window that must line up
    /// sample for sample with another: `PeriodicTorus.drawing` samples the
    /// plane a few cells past a period rectangle on every side, on the
    /// rectangle's own grid extended.
    public static func sample(_ compiled: KurvenMath.Expression.Compiled, domain: Domain,
                              shape: (nReal: Int, nImag: Int)) -> Samples {
        let real = linspace(domain.real.lo, domain.real.hi, shape.nReal)
        let imag = linspace(domain.imag.lo, domain.imag.hi, shape.nImag)
        let count = shape.nReal * shape.nImag
        var magnitude = [Double](repeating: 0, count: count)
        var phase = [Double](repeating: 0, count: count)
        magnitude.withUnsafeMutableBufferPointer { mag in
            phase.withUnsafeMutableBufferPointer { ph in
                // Each row writes only its own range, so sharing is safe.
                nonisolated(unsafe) let magBase = mag.baseAddress!
                nonisolated(unsafe) let phBase = ph.baseAddress!
                DispatchQueue.concurrentPerform(iterations: shape.nImag) { y in
                    let row = y * shape.nReal
                    for x in 0..<shape.nReal {
                        var v = compiled(Complex(real[x], imag[y]))
                        if !v.isFinite {
                            v = Complex(huge)
                        } else if v.magnitude > huge {
                            let a = v.argument
                            v = Complex(huge * cos(a), huge * sin(a))
                        }
                        magBase[row + x] = v.magnitude
                        phBase[row + x] = v.argument
                    }
                }
            }
        }
        return Samples(magnitude: magnitude, phase: phase, shape: shape)
    }

    /// `kurven.export.quantize`: to float32, clamped into the float64 range so
    /// rounding never invents a value the samples did not have -- a phase grid
    /// with a value above π would contour at π where the drawing has nothing.
    public static func quantize(_ values: [Double]) -> [Float] {
        var lo = Double.infinity, hi = -Double.infinity
        for v in values where !v.isNaN { lo = min(lo, v); hi = max(hi, v) }
        guard lo <= hi else { return values.map { Float($0) } }
        var lo32 = Float(lo)
        if Double(lo32) < lo { lo32 = lo32.nextUp }
        var hi32 = Float(hi)
        if Double(hi32) > hi { hi32 = hi32.nextDown }
        return values.map { min(max(Float($0), lo32), hi32) }
    }

    // MARK: the landscape

    /// The bundle for a request: `build_scene` with the geometry left described
    /// and `export --derived`, in memory.
    ///
    /// With `refine`, the contour layers are placed by the function rather
    /// than by the grid (`ContourRefiner`), and the bundle carries the refiner
    /// so every restyle keeps doing so. Off, the curves are exactly what the
    /// Python side derives from the same grids, which is what the fixtures
    /// compare.
    public static func build(_ request: LandscapeRequest, gitSha: String = "native",
                             refine: Bool = true) throws -> KurvenBundle {
        let compiled = try KurvenMath.Expression.compile(request.expression)
        let canonical = compiled.expression
        let samples = sample(compiled, domain: request.domain, resolution: request.resolution)
        let shape = samples.shape
        let domain = request.domain
        let truncation = request.truncation ?? defaultTruncation(samples.magnitude)
        let density = wallDensity(domain, shape)
        let perimeter = rectangle(domain, density: density)
        let layers = request.layers ?? defaultLayers(
            domain, shape: shape, truncation: truncation,
            ceiling: ceiling(of: truncation, magnitudes: samples.magnitude),
            spacing: request.spacing)
        let step = max(1, max(shape.nReal, shape.nImag) / occluderResolution)
        let gridShape = (ny: shape.nImag, nx: shape.nReal)
        let manifest = Manifest(
            domain: domain,
            height: GridRef(file: "height.npy", shape: gridShape, dtype: .float32),
            phase: GridRef(file: "phase.npy", shape: gridShape, dtype: .float32),
            truncation: truncation,
            occluder: Occluder(step: step, tiles: [.identity],
                               walls: .perimeter(perimeter, base: 0), region: .full, base: 0),
            layers: layers,
            presets: [CameraPreset(name: "oblique", projection: projection, depthTolerance: defaultDepthTolerance,
                                   buffer: defaultBuffer)],
            provenance: Provenance(
                function: canonical, example: "function",
                params: ["expression": .string(canonical), "name": .string(request.name),
                         "resolution": .int(request.resolution),
                         "rMin": .double(domain.real.lo), "rMax": .double(domain.real.hi),
                         "iMin": .double(domain.imag.lo), "iMax": .double(domain.imag.hi),
                         "nReal": .int(shape.nReal), "nImag": .int(shape.nImag)],
                cpuCount: 1, gitSha: gitSha))
        let height = Grid2D(width: shape.nReal, height: shape.nImag, domain: domain,
                            values: quantize(samples.magnitude))
        let phase = Grid2D(width: shape.nReal, height: shape.nImag, domain: domain,
                           values: quantize(samples.phase))
        let surface = Surface(height: height, phase: phase, truncation: truncation, cached: false)
        let name = request.name.isEmpty ? "landscape" : request.name
        let refiner = refine ? ContourRefiner(compiled, domain: domain,
                                              width: shape.nReal, height: shape.nImag) : nil
        return KurvenBundle(url: URL(fileURLWithPath: "/native/\(name).kurven"),
                            manifest: manifest, surface: surface, refine: refiner?.refine)
    }
}

/// The rules `kurven.landscape` derives a landscape's styling by, as far as
/// the consumer needs them when the *truncation* moves: pure arithmetic over one
/// number, pinned against the Python answers in `kurven-test`.
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
    /// minors to a major, as multiples of the step.
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

    /// The levels a landscape is actually given under these truncations: the rule
    /// above, less any level sitting exactly on the plateau, which is the truncation edge.
    public static func levels(under truncation: Truncation)
        -> (major: [Double], minor: [Double]) {
        guard let ceiling = ceiling(of: truncation) else { return ([], []) }
        let all = magnitudeLevels(upTo: ceiling)
        return (all.major.filter { $0 < ceiling - 1e-9 },
                all.minor.filter { $0 < ceiling - 1e-9 })
    }

    /// The highest |f| worth a contour: the truncation, when there is one.
    public static func ceiling(of truncation: Truncation) -> Double? {
        switch truncation {
        case .none: return nil
        case .uniform(let z): return z
        case .realBands(let bands, let beyond):
            let tops = bands.map(\.height) + (beyond.isFinite ? [beyond] : [])
            return tops.max()
        }
    }

    /// About seventy strokes along the longest side, rounded to a ruler.
    public static func hatchSpacing(_ domain: Domain) -> Double {
        nice(max(abs(domain.real.length), abs(domain.imag.length)) / 70, down: true)
    }
}
