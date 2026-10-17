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
/// plane, marching squares on |f| and arg f, the phase wraps dropped, every
/// vertex placed on the function's level set by `ContourRefiner` and every
/// phase line carried to the zero or pole it ends at. The window sampled is
/// the period rectangle and a margin of cells past it on every side: the
/// torus is the plane modulo the periods (`PeriodLattice`), the rectangle one
/// fundamental domain of it, and a zero on the rectangle's edge -- sn has two
/// on the real axis and a pole on the imaginary one -- is a point of the
/// plane's interior like any other, with the whole fan of lines round it to
/// be placed. Each curve is then cut to the rectangle and carried into it by
/// whole periods, and that point of the rectangle is each vertex's surface
/// coordinate: where it lies on the torus and what its visibility is judged
/// by. Where a curve crosses the rectangle's edge it ends on the edge, and
/// its continuation, a copy of the same curve one period over, begins on the
/// opposite edge at the same point of the torus.
public enum PeriodicTorus {
    public struct Plate: Sendable {
        public let surface: ParametricSurface
        public let layers: [Layer]
        /// The landscape's height at a point of the rectangle, |f| truncated
        /// at the ceiling, for `relief`; nil for a plate built without f.
        public let height: (@Sendable (P2<ParamSpace>) -> Double)?
        public let ceiling: Double

        public init(surface: ParametricSurface, layers: [Layer],
                    height: (@Sendable (P2<ParamSpace>) -> Double)? = nil,
                    ceiling: Double = .infinity) {
            self.surface = surface; self.layers = layers
            self.height = height; self.ceiling = ceiling
        }

        /// The landscape in relief on the torus: the surface displaced along
        /// its normal by `amplitude` times the truncated |f|, and every
        /// layer placed again through the displaced map, since the ink's
        /// coordinates are the rectangle's whatever the embedding. The
        /// relief bounds a solid while the displacement stays under the
        /// torus's reach, `reach`; past it the surface passes through itself
        /// and its back faces are kept.
        public func relief(amplitude: Double, reach: Double) -> Plate {
            guard let height, amplitude != 0 else { return self }
            let encloses = surface.outward != nil && amplitude * ceiling < reach
            let displaced = surface.displaced(by: height, amplitude: amplitude, encloses: encloses)
            let layers = self.layers.map { layer -> Layer in
                guard let coords = layer.paths.coords else { return layer }
                return Layer(spec: layer.spec,
                             paths: PolylineSet(vertices: coords.map { P3(displaced.map($0).position) },
                                                offsets: layer.paths.offsets, coords: coords))
            }
            return Plate(surface: displaced, layers: layers, height: height, ceiling: ceiling)
        }

        /// The same function on a torus of other radii. The contours are the
        /// function's, which the radii do not touch: only where each vertex
        /// sits in space moves, read from its coordinate on the new torus.
        public func placed(major R: Double, minor r: Double, lattice: Int) -> Plate {
            let domain = Domain(real: surface.u.range, imag: surface.v.range)
            let torus = PeriodicTorus.torus(over: domain, major: R, minor: r, lattice: lattice)
            let layers = self.layers.map { layer -> Layer in
                guard let coords = layer.paths.coords else { return layer }
                return Layer(spec: layer.spec,
                             paths: PolylineSet(vertices: coords.map { P3(torus.map($0).position) },
                                                offsets: layer.paths.offsets, coords: coords))
            }
            return Plate(surface: torus, layers: layers, height: height, ceiling: ceiling)
        }
    }

    /// How far past the rectangle the plane is sampled, in cells. More than
    /// the refiner looks for a zero or pole from a run's end
    /// (`ContourRefiner.reach`) and the two cells it carries one over, so a
    /// singular point on the rectangle's edge wants nothing outside the window.
    public static let margin = 8

    /// Only rectangular period lattices: a real period along `real` and an
    /// imaginary one along `imag`, which is what sn, and cn and dn on
    /// suitable rectangles, have.
    /// `capped`: the phase lines are cut at the ceiling, as a landscape's
    /// are at its cap, and the rim |f| = ceiling is drawn; for a plate to be
    /// put in relief, where the cap is a plateau. Otherwise every phase line
    /// runs through its pole.
    public static func plate(_ expression: String, origin: Complex = Complex(0),
                             periods: (real: Double, imag: Double),
                             major R: Double = 2, minor r: Double = 1,
                             resolution: Int = 600, ceiling: Double = 3,
                             lattice: Int = 1024, capped: Bool = false) throws -> Plate {
        let compiled = try KurvenMath.Expression.compile(expression)
        let domain = Domain(real: Interval(lo: origin.re, hi: origin.re + periods.real),
                            imag: Interval(lo: origin.im, hi: origin.im + periods.imag))
        let surface = torus(over: domain, major: R, minor: r, lattice: lattice)
        let group = PeriodLattice(cell: domain)

        // The window: the rectangle's lattice, extended `margin` cells each
        // way and shifted by half a cell on both axes. The samples past the
        // edge are the function's own, since it is periodic, and a curve and
        // its copy one period over are computed from the same sample
        // positions, so they agree to rounding where the clip joins them.
        // The half cell keeps every sample off the rectangle's edges and off
        // its half-period lines: sn is real on the real axis and imaginary
        // on the imaginary one, so its lines of phase 0, π and ±π/2 lie
        // along them, and a level set lying along a row of samples is the
        // one configuration marching squares cannot resolve, each sample a
        // tie decided by rounding and the two copies of a curve chained
        // differently. Off the row, the same lines cross the cells.
        let shape = NativeLandscape.gridShape(domain, resolution: resolution)
        let dx = domain.real.length / Double(shape.nReal - 1)
        let dy = domain.imag.length / Double(shape.nImag - 1)
        let m = Double(margin)
        let cover = Domain(real: Interval(lo: domain.real.lo - (m + 0.5) * dx,
                                          hi: domain.real.hi + (m - 0.5) * dx),
                           imag: Interval(lo: domain.imag.lo - (m + 0.5) * dy,
                                          hi: domain.imag.hi + (m - 0.5) * dy))
        let coverShape = (nReal: shape.nReal + 2 * margin, nImag: shape.nImag + 2 * margin)

        let samples = NativeLandscape.sample(compiled, domain: cover, shape: coverShape)
        let height = Grid2D(width: coverShape.nReal, height: coverShape.nImag, domain: cover,
                            values: NativeLandscape.quantize(samples.magnitude))
        let phase = Grid2D(width: coverShape.nReal, height: coverShape.nImag, domain: cover,
                           values: NativeLandscape.quantize(samples.phase))
        let field = Surface(height: height, phase: phase, caps: .uniform(ceiling))
        // Not lifted: the ink lies on the torus, placed by its coordinate,
        // and |f| is no height of it.
        let refiner = ContourRefiner(compiled, domain: cover,
                                     width: coverShape.nReal, height: coverShape.nImag,
                                     lifted: false)
        // The rounding the clip allows at the rectangle's edges: wider than
        // the error the function is evaluated to, so that a zero found on
        // the far edge by Newton is on one side of it, not sometimes on
        // either, which would leave a stub of the line along that edge. A
        // crossing is put on the edge itself whatever the slack.
        let slack = 1e-6 * max(dx, dy)

        let magnitude = LandscapeStyle.magnitudeLevels(upTo: ceiling)
        var families: [(String, LayerRole, ContourField, [Double], Double)] = [
            ("mag_major", .magnitude, .magnitude, magnitude.major, 0.35),
            ("mag_minor", .magnitude, .magnitude, magnitude.minor, 0.15),
            ("ang_major", .phase, .phase, NativeLandscape.phaseMajor, 0.35),
            ("ang_minor", .phase, .phase, NativeLandscape.phaseMinor, 0.15),
        ]
        if capped { families.append(("rim", .outline, .magnitude, [ceiling], 0.6)) }
        var layers: [Layer] = families.map { name, role, kind, levels, width in
            // Capped, a phase line is lifted to |f| and kept under the
            // ceiling, ending on the rim where it crosses it.
            let cut = capped && kind == .phase
            let source = LayerSource.contour(field: kind, levels: levels,
                                             keep: cut ? .belowCap : .all, tiled: false)
            let flat = field.derive(source, policy: cut ? .magnitude : .surface, region: .full,
                                    tiles: [.identity], refine: refiner.refine)
            // Where a curve crosses the rectangle's edge, the crossing is
            // solved on f along the edge, so that it lies on the level set
            // like every other vertex: the level is the one the inside
            // vertex is on, and the crossing is looked for within half a
            // cell of the chord's, which is within the refiner's tolerance
            // of it. A phase near ±π is solved on the turned phase, as the
            // cut is placed.
            let onEdge: (P2<DomainSpace>, P2<DomainSpace>, P2<DomainSpace>) -> P2<DomainSpace> = {
                a, _, c in
                let g: @Sendable (Complex) -> Double
                switch kind {
                case .magnitude: g = refiner.magnitude
                case .phase:
                    g = abs(refiner.phase(Complex(a.x, a.y))) > .pi / 2
                        ? refiner.turnedPhase : refiner.phase
                }
                let level = g(Complex(a.x, a.y))
                guard level.isFinite else { return c }
                let vertical = min(abs(c.x - domain.real.lo), abs(c.x - domain.real.hi)) <= 2 * slack
                func at(_ t: Double) -> P2<DomainSpace> { vertical ? P2(c.x, t) : P2(t, c.y) }
                func residual(_ t: Double) -> Double {
                    let r = g(Complex(at(t).x, at(t).y)) - level
                    return kind == .phase ? ContourRefiner.wrapped(r) : r
                }
                // No bracket, no solve; and none either where the level set
                // lies along the edge itself, as sn's lines of phase 0 and
                // π do along the real axis: the residual is then zero the
                // whole way and any point would do. There the chord's
                // crossing, a vertex of a curve crossing the line, is right.
                let t0 = vertical ? c.y : c.x, h = 0.5 * refiner.cell
                let ra = residual(t0 - h), rb = residual(t0 + h)
                let flat = 1e-9 * (kind == .phase ? 1 : max(abs(level), 1))
                guard ra.isFinite, rb.isFinite, abs(ra) > flat, abs(rb) > flat, ra.sign != rb.sign,
                      let t = Root.brent(residual, t0 - h, t0 + h, fa: ra, fb: rb,
                                         xtol: 1e-12 * refiner.cell)
                else { return c }
                return at(t)
            }
            // Each curve of the plane, cut to the rectangle and carried into
            // it: (Re z, Im z) brought in by whole periods is the surface
            // coordinate, and the torus says where that is.
            var placed: [[P3<WorldSpace>]] = []
            var coords: [[P2<ParamSpace>]] = []
            for i in 0..<(flat.offsets.count - 1) {
                let path = flat.vertices[flat.offsets[i]..<flat.offsets[i + 1]]
                    .map { P2<DomainSpace>($0.x, $0.y) }
                for run in group.clipped(path, slack: slack, onEdge: onEdge) {
                    let c = group.canonical(run)
                    coords.append(c)
                    placed.append(c.map { P3(surface.map($0).position) })
                }
            }
            return Layer(spec: LayerSpec(name: name, role: role, source: source, width: width,
                                         heightPolicy: .surface),
                         paths: PolylineSet(paths: placed, coords: coords))
        }
        layers.append(Layer(spec: LayerSpec(name: "folds", role: .outline, source: .foldLines,
                                            width: 0.6, heightPolicy: .surface),
                            paths: .empty))
        // The landscape's height over the rectangle, for a relief.
        let relief: @Sendable (P2<ParamSpace>) -> Double = { c in
            let v = compiled(Complex(c.x, c.y))
            return v.isFinite ? min(v.magnitude, ceiling) : ceiling
        }
        return Plate(surface: surface, layers: layers, height: relief, ceiling: ceiling)
    }

    /// The torus a rectangle of periods glues into, its lattice `lattice`
    /// points around the real period and enough around the imaginary one to
    /// keep the cells square on the tube.
    static func torus(over domain: Domain, major R: Double, minor r: Double,
                      lattice: Int) -> ParametricSurface {
        let aspect = domain.imag.length / domain.real.length
        return ParametricSurface.torus(
            major: R, minor: r,
            u: ParamAxis(domain.real, periodic: true, samples: lattice),
            v: ParamAxis(domain.imag, periodic: true,
                         samples: max(Int(Double(lattice) * min(aspect, 1) * r / R), 32)))
    }

    /// sn(z, m) on its fundamental rectangle, [0, 4K) × [0, 2K').
    public static func jacobiSN(modulus m: Double, major R: Double = 2, minor r: Double = 1,
                                resolution: Int = 600, lattice: Int = 1024,
                                capped: Bool = false) throws -> Plate {
        let K = Jacobi.quarterPeriod(m), Kp = Jacobi.quarterPeriod(1 - m)
        return try plate("sn(z, \(m))", periods: (4 * K, 2 * Kp), major: R, minor: r,
                         resolution: resolution, lattice: lattice, capped: capped)
    }
}
