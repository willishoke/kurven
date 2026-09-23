import Foundation
import KurvenCore
import KurvenMath

/// A doubly periodic function drawn on the torus its periods glue into.
///
/// A function with periods ω₁ and ω₂ lives on the rectangle they span with
/// opposite edges identified, and that rectangle, rolled up both ways, is a
/// torus. The elliptic plate reflects one tile across a lattice to show the
/// same thing flat; here the contours close up on themselves instead.
///
/// The contours are the landscape's own: the function sampled over the
/// rectangle -- whose far edges are its near ones, so the grid meets itself --
/// marching squares on |f| and arg f, the phase wraps dropped, and every
/// vertex placed on the function's level set by `ContourRefiner`. Each vertex
/// keeps its point of the rectangle as its surface coordinate, which is where
/// it lies on the torus and what its visibility is judged by.
public enum PeriodicTorus {
    public struct Plate: Sendable {
        public let surface: ParametricSurface
        public let layers: [Layer]
    }

    /// Only rectangular period lattices: a real period along `real` and an
    /// imaginary one along `imag`, which is what sn, and cn and dn on
    /// suitable rectangles, have.
    public static func plate(_ expression: String, origin: Complex = Complex(0),
                             periods: (real: Double, imag: Double),
                             major R: Double = 2, minor r: Double = 1,
                             resolution: Int = 600, ceiling: Double = 3,
                             lattice: Int = 1024) throws -> Plate {
        let compiled = try KurvenMath.Expression.compile(expression)
        let domain = Domain(real: Interval(lo: origin.re, hi: origin.re + periods.real),
                            imag: Interval(lo: origin.im, hi: origin.im + periods.imag))
        let aspect = periods.imag / periods.real
        let surface = ParametricSurface.torus(
            major: R, minor: r,
            u: ParamAxis(domain.real, periodic: true, samples: lattice),
            v: ParamAxis(domain.imag, periodic: true,
                         samples: max(Int(Double(lattice) * min(aspect, 1) * r / R), 32)))

        let samples = NativeLandscape.sample(compiled, domain: domain, resolution: resolution)
        let shape = samples.shape
        let height = Grid2D(width: shape.nReal, height: shape.nImag, domain: domain,
                            values: NativeLandscape.quantize(samples.magnitude))
        let phase = Grid2D(width: shape.nReal, height: shape.nImag, domain: domain,
                           values: NativeLandscape.quantize(samples.phase))
        let field = Surface(height: height, phase: phase, caps: .uniform(ceiling))
        let refiner = ContourRefiner(compiled, domain: domain,
                                     width: shape.nReal, height: shape.nImag)

        let magnitude = LandscapeStyle.magnitudeLevels(upTo: ceiling)
        let families: [(String, LayerRole, ContourField, [Double], Double)] = [
            ("mag_major", .magnitude, .magnitude, magnitude.major, 0.35),
            ("mag_minor", .magnitude, .magnitude, magnitude.minor, 0.15),
            ("ang_major", .phase, .phase, NativeLandscape.phaseMajor, 0.35),
            ("ang_minor", .phase, .phase, NativeLandscape.phaseMinor, 0.15),
        ]
        var layers: [Layer] = families.map { name, role, kind, levels, width in
            let source = LayerSource.contour(field: kind, levels: levels, keep: .all, tiled: false)
            let flat = field.derive(source, policy: .surface, region: .full, tiles: [.identity],
                                    refine: refiner.refine)
            // (Re z, Im z) is the point of the rectangle, and so the surface
            // coordinate; the torus says where that is.
            let coords = flat.vertices.map { P2<ParamSpace>($0.x, $0.y) }
            let placed = PolylineSet<WorldSpace>(vertices: coords.map { P3(surface.map($0).position) },
                                     offsets: flat.offsets, coords: coords)
            return Layer(spec: LayerSpec(name: name, role: role, source: source, width: width,
                                         heightPolicy: .surface),
                         paths: placed)
        }
        layers.append(Layer(spec: LayerSpec(name: "folds", role: .outline, source: .foldLines,
                                            width: 0.6, heightPolicy: .surface),
                            paths: .empty))
        return Plate(surface: surface, layers: layers)
    }

    /// sn(z, m) on its fundamental rectangle, [0, 4K) × [0, 2K').
    public static func jacobiSN(modulus m: Double, major R: Double = 2, minor r: Double = 1,
                                resolution: Int = 600) throws -> Plate {
        let K = Jacobi.quarterPeriod(m), Kp = Jacobi.quarterPeriod(1 - m)
        return try plate("sn(z, \(m))", periods: (4 * K, 2 * Kp), major: R, minor: r,
                         resolution: resolution)
    }
}
