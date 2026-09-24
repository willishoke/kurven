import Foundation
import KurvenCore
import KurvenMath
import KurvenLandscape

/// A surface plate, described: which surface, at what lattice, with what ink.
///
/// The value the CLI parses its flags into and the app keeps in its
/// document, so both build a plate with one function -- `SurfacePlate.build`
/// -- as a bake and a landscape are each built by one function called twice.
public struct SurfaceRequest: Sendable, Equatable {
    public enum Shape: Sendable, Equatable {
        /// The torus of revolution about the z axis.
        case torus(major: Double, minor: Double)
        /// sn(z, m) on its fundamental rectangle, glued into a torus. Its own
        /// case rather than a `periodic` expression, because the modulus
        /// decides the periods as well as the function.
        case sn(modulus: Double, major: Double, minor: Double, grid: Int)
        /// Any function with a rectangle of periods, glued into a torus.
        case periodic(expression: String, periods: SIMD2<Double>, major: Double, minor: Double,
                      grid: Int)
        /// A forced system's invariant torus, fitted from `fit` time units of
        /// one run at `harmonics`, drawn by revolution on a ring of `radius`
        /// with state components `radial` and `axial`; its trajectory drawn
        /// on it for `duration`, sampled `every`.
        case forced(system: String, fit: Double, harmonics: Int, radial: Int, axial: Int,
                    radius: Double, duration: Double, every: Double)
    }

    /// Stroke widths. A fold width of zero leaves the folds out.
    public struct Style: Sendable, Equatable {
        public var lines: Double
        public var trajectory: Double
        public var folds: Double
        public init(lines: Double = 0.3, trajectory: Double = 0.08, folds: Double = 0.6) {
            self.lines = lines; self.trajectory = trajectory; self.folds = folds
        }
    }

    public var shape: Shape
    /// Lattice points around `u`; `v` gets half as many (a periodic plate
    /// sizes `v` to its rectangle instead).
    public var lattice: Int
    /// Lines of constant `u` and `v`, when drawn. A plain torus has nothing
    /// else to draw; a forced torus draws them only when asked.
    public var lines: SIMD2<Int>?
    public var style: Style
    /// The catalog entry this came from, for provenance and the title.
    public var name: String

    public init(_ shape: Shape, lattice: Int = 1024, lines: SIMD2<Int>? = nil,
                style: Style = Style(), name: String = "") {
        self.shape = shape; self.lattice = lattice; self.lines = lines
        self.style = style; self.name = name
    }

    /// The projection a surface plate opens at.
    public static let projection = PlateProjection(shear: 0, xAngle: -55, zAngle: 30,
                                                   flipX: false, yScale: nil)

    /// The same, as a camera preset: no margin and the CLI's bake resolution,
    /// so the window and `kurven-cli surface` bake the same picture.
    public static let preset = CameraPreset(name: "plate", plate: projection, margin: 0,
                                            buffer: 3000)
}

/// A surface plate, built: the surface, its ink, and what is worth keeping
/// between edits.
public struct SurfacePlate: Sendable {
    public let request: SurfaceRequest
    public let surface: ParametricSurface
    public let layers: [Layer]
    /// A forced plate's fitted torus, kept so an edit that leaves the fit
    /// alone does not refit.
    public let torus: InvariantTorus?
    public let revolution: Revolution?
    /// False when a forced torus's revolution passes through itself, so its
    /// back faces are kept rather than hidden.
    public let embedded: Bool

    /// The scene this plate draws, from `camera`.
    public func scene(camera: Camera) -> Scene {
        Scene(surface: surface, layers: layers, camera: camera, margin: 0)
    }

    /// Build a plate. `previous`, when given, lends whatever of it the new
    /// request has not changed: today, a forced system's fitted torus.
    public static func build(_ request: SurfaceRequest,
                             reusing previous: SurfacePlate? = nil) throws -> SurfacePlate {
        let style = request.style
        func scaffold(_ name: String, _ source: LayerSource, _ width: Double,
                      _ paths: PolylineSet<WorldSpace>) -> Layer {
            Layer(spec: LayerSpec(name: name, role: .scaffold, source: source, width: width,
                                  heightPolicy: .surface),
                  paths: paths)
        }
        func lines(on surface: ParametricSurface, resolution: Int) -> Layer? {
            request.lines.map { n in
                scaffold("lines", .parameterLines(u: n.x, v: n.y), style.lines,
                         surface.parameterLines(counts: (n.x, n.y), resolution: resolution))
            }
        }

        let surface: ParametricSurface
        var layers: [Layer]
        var torus: InvariantTorus?, revolution: Revolution?, embedded = true
        switch request.shape {
        case .torus(let R, let r):
            let n = request.lattice
            surface = ParametricSurface.torus(major: R, minor: r, samples: (n, max(n / 2, 8)))
            layers = lines(on: surface, resolution: 4 * n).map { [$0] } ?? []
        case .sn, .periodic:
            let plate: PeriodicTorus.Plate
            if case .sn(let m, let R, let r, let grid) = request.shape {
                plate = try PeriodicTorus.jacobiSN(modulus: m, major: R, minor: r,
                                                   resolution: grid, lattice: request.lattice)
            } else if case .periodic(let expression, let periods, let R, let r, let grid)
                        = request.shape {
                plate = try PeriodicTorus.plate(expression, periods: (periods.x, periods.y),
                                                major: R, minor: r, resolution: grid,
                                                lattice: request.lattice)
            } else { fatalError("unreachable") }
            surface = plate.surface
            // Their own folds come with them; the style's replace them below.
            layers = plate.layers.filter {
                if case .foldLines = $0.spec.source { false } else { true }
            }
        case .forced(let name, let fit, let harmonics, let radial, let axial, let radius,
                     let duration, let every):
            guard let system = ForcedSystem.catalog[name] else {
                throw DynamicsError.unknownSystem(name, known: ForcedSystem.catalog.keys.sorted())
            }
            if let kept = previous?.torus, case .forced(name, fit, harmonics, _, _, _, _, _)
                = previous?.request.shape {
                torus = kept
            } else {
                torus = try InvariantTorus.find(system, duration: fit, harmonics: harmonics)
            }
            let e = Revolution.fitting(torus!, radial: radial, axial: axial, radius: radius)
            let placed = torus!.surface(e, lattice: (request.lattice, request.lattice / 2))
            surface = placed.surface
            revolution = e
            embedded = placed.embedded
            layers = [scaffold("trajectory", .trajectory, style.trajectory,
                               try torus!.trajectory(e, duration: duration, every: every))]
            if let l = lines(on: surface, resolution: 2 * request.lattice) { layers.append(l) }
        }
        // The outline and inner silhouettes, derived per camera.
        if style.folds > 0 {
            layers.append(Layer(spec: LayerSpec(name: "folds", role: .outline, source: .foldLines,
                                                width: style.folds, heightPolicy: .surface),
                                paths: .empty))
        }
        return SurfacePlate(request: request, surface: surface, layers: layers, torus: torus,
                            revolution: revolution, embedded: embedded)
    }
}

/// A surface the gallery offers: a name to ask for it by, a label to show,
/// and the request it opens as.
public struct SurfacePreset: Sendable, Equatable, Identifiable {
    public let name: String
    public let label: String
    public let request: SurfaceRequest
    public var id: String { name }

    public init(name: String, label: String, request: SurfaceRequest) {
        self.name = name; self.label = label
        var request = request
        request.name = name
        self.request = request
    }
}

public extension SurfacePreset {
    /// Named as `kurven-cli surface` names them, and built with its defaults,
    /// so a preset and the CLI's plate of the same name are the same plate.
    static let catalog: [SurfacePreset] = [
        SurfacePreset(name: "torus", label: "Torus",
                      request: SurfaceRequest(.torus(major: 2, minor: 1), lines: SIMD2(36, 18))),
        SurfacePreset(name: "sn", label: "sn(z, 0.64) on its torus",
                      request: SurfaceRequest(.sn(modulus: 0.64, major: 2, minor: 1, grid: 600))),
        SurfacePreset(name: "forced", label: "Forced jerk oscillator",
                      request: SurfaceRequest(.forced(system: "jerk", fit: 8000, harmonics: 24,
                                                      radial: 0, axial: 1, radius: 2.5,
                                                      duration: 2300, every: 0.01))),
    ]

    static func named(_ name: String) -> SurfacePreset? { catalog.first { $0.name == name } }
}
