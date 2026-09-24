import Foundation
import simd
import KurvenCore
import KurvenMath
import KurvenMetal
import KurvenBake
import KurvenDynamics
import KurvenLandscape

// MARK: - a forced oscillator's invariant torus, found, fitted and drawn
//
// The system is `toroidal_manifolds.ipynb`'s. Its second frequency was found
// independently in a scipy prototype (DOP853 at rtol 1e-11, NAFF by
// `minimize_scalar`) as 1.3362213283 -- that is, ω + 0.3962213283 -- and the
// generator here is taken in (0, ω/2], so the two must agree to that
// prototype's precision.

func forcedTorusTests() {
    let found: InvariantTorus
    do { found = try InvariantTorus.find(.jerkTorus) } catch {
        Check.expect(false, "the notebook's oscillator has a torus", "\(error)"); return
    }

    Check.suite("dynamics: the notebook's forced oscillator winds round a torus") {
        Check.expect(abs(found.internalFrequency - 0.396_221_328_3) < 1e-8,
                     "its second frequency is the prototype's",
                     String(format: "Ω = %.10f", found.internalFrequency))
        Check.expect(found.residual < 1e-4 * found.extent,
                     "and the fitted torus holds the trajectory it came from",
                     String(format: "worst miss %.2e, %.1e of the attractor's size",
                            found.residual, found.residual / found.extent))
    }

    Check.suite("dynamics: drawn by revolution, it is embedded and the trajectory is on it") {
        let square: [SIMD2<Double>] = [SIMD2(0, 0), SIMD2(1, 0), SIMD2(1, 1), SIMD2(0, 1)]
        let bow: [SIMD2<Double>] = [SIMD2(0, 0), SIMD2(1, 1), SIMD2(1, 0), SIMD2(0, 1)]
        Check.expect(Revolution.isSimpleClosed(square) && !Revolution.isSimpleClosed(bow),
                     "a square is a simple closed curve and a bow tie is not")

        let e = Revolution.fitting(found)
        let (surface, embedded) = found.surface(e, lattice: (512, 256))
        Check.expect(embedded && surface.outward != nil,
                     "every slice is a simple closed curve clear of the axis, so it bounds a solid")

        let ink = try found.trajectory(e, duration: 500, every: 0.02)
        // At a fixed forcing phase the embedding is an isometry of (radius,
        // height), scaled: the ink is at most `scale` times the fit's own
        // miss from the surface.
        var worst = 0.0
        for (k, c) in ink.coords!.enumerated() {
            worst = max(worst, simd_length(ink.vertices[k].v - surface.map(c).position))
        }
        let bound = e.scale * found.residual
        Check.expect(ink.count > 1 && worst <= bound * (1 + 1e-9) + 1e-12,
                     "the trajectory lies on the drawn torus, within the fit's own miss",
                     String(format: "worst %.2e, bound %.2e, over %d vertices in %d paths",
                            worst, bound, ink.vertices.count, ink.count))
    }

    Check.suite("dynamics: the plate bakes, and its front is the torus's outside") {
        let e = Revolution.fitting(found)
        let (surface, _) = found.surface(e, lattice: (512, 256))
        let trajectory = Layer(spec: LayerSpec(name: "trajectory", role: .scaffold,
                                               source: .trajectory, width: 0.08,
                                               heightPolicy: .surface),
                               paths: try found.trajectory(e, duration: 500, every: 0.02))
        let folds = Layer(spec: LayerSpec(name: "folds", role: .outline, source: .foldLines,
                                          width: 0.5, heightPolicy: .surface),
                          paths: .empty)
        let camera = Camera.plate(PlateProjection(shear: 0, xAngle: -58, zAngle: 35,
                                                  flipX: false, yScale: nil))
        let scene = Scene(surface: surface, layers: [trajectory, folds], camera: camera, margin: 0)
        let baked = try MetalRenderer().bake(scene, options: BakeOptions(resolution: 2000))
        // On an embedded surface with its outside computed rightly, what is
        // front-most at any pixel faces the eye; only pixels along the
        // outline, where the surface is edge-on, may not.
        let image = baked.surface!
        let sight = camera.view.sightLine
        var covered = 0, away = 0
        for k in image.coords.indices where !image.coords[k].x.isNaN {
            covered += 1
            let c = P2<ParamSpace>(Double(image.coords[k].x), Double(image.coords[k].y))
            if (surface.facing(c, sight: sight) ?? 1) <= 0 { away += 1 }
        }
        Check.expect(covered > 100_000 && Double(away) < 0.002 * Double(covered),
                     "every front-most pixel faces the eye, but along the edge",
                     "\(away) of \(covered) face away")
        Check.expect(baked.strokes.layers[0].paths.inkLength > 0
                     && baked.strokes.layers[1].paths.inkLength > 0,
                     "and it draws both the trajectory and the folds",
                     "\(baked.strokes.layers[0].paths.count) trajectory runs, "
                     + "\(baked.strokes.layers[1].paths.count) fold runs")
    }

    latticeFoldTests(found)
}

// MARK: - the preview's folds, read off the lattice
//
// The preview re-derives the folds whenever the camera moves, so they have to
// cost a dot product per lattice point and not a map evaluation per lattice
// point: the forced torus's map is a 1,225-term Fourier series. Held to two
// things: they are where the exact folds are, to a pixel of the preview, and
// an orbit frame of the forced torus is fast enough to follow the pointer.

/// The worst distance from any vertex of either set to the other set's
/// segments, in view units: the symmetric Hausdorff distance of the two as
/// drawn, to the vertices' own spacing.
func hausdorff(_ a: PolylineSet<ViewSpace>, _ b: PolylineSet<ViewSpace>) -> Double {
    func segments(_ s: PolylineSet<ViewSpace>) -> [(SIMD2<Double>, SIMD2<Double>)] {
        var out: [(SIMD2<Double>, SIMD2<Double>)] = []
        for p in 0..<s.count {
            for i in s.offsets[p]..<(s.offsets[p + 1] - 1) {
                out.append((s.vertices[i].xy.v, s.vertices[i + 1].xy.v))
            }
        }
        return out
    }
    func directed(_ from: PolylineSet<ViewSpace>, _ to: [(SIMD2<Double>, SIMD2<Double>)]) -> Double {
        var worst = 0.0
        for v in from.vertices {
            let q = v.xy.v
            var best = Double.infinity
            for (p0, p1) in to {
                let d = p1 - p0, l2 = simd_length_squared(d)
                let t = l2 > 0 ? min(max(simd_dot(q - p0, d) / l2, 0), 1) : 0
                best = min(best, simd_length_squared(q - (p0 + t * d)))
            }
            worst = max(worst, best)
        }
        return worst.squareRoot()
    }
    return max(directed(a, segments(b)), directed(b, segments(a)))
}

func latticeFoldTests(_ found: InvariantTorus) {
    let viewport = Viewport(width: 1600, height: 1200)
    let e = Revolution.fitting(found)
    let forced = found.surface(e, lattice: (1024, 512)).surface
    let sn: ParametricSurface
    do { sn = try PeriodicTorus.jacobiSN(modulus: 0.64).surface } catch {
        Check.expect(false, "the sn torus builds", "\(error)"); return
    }

    Check.suite("surfaces: the preview's lattice folds are the exact folds, to a pixel") {
        for (name, surface) in [("forced", forced), ("sn", sn)] {
            let normals = surface.latticeNormals()
            for plate in [PlateProjection(shear: 0, xAngle: -55, zAngle: 30, flipX: false, yScale: nil),
                          PlateProjection(shear: 0.5, xAngle: -30, zAngle: 110, flipX: false,
                                          yScale: nil)] {
                let camera = Camera.plate(plate)
                let sight = camera.view.sightLine
                let exact = surface.foldLines(sight: sight)
                let lattice = surface.latticeFoldLines(normals: normals, sight: sight)
                let scene = Scene(surface: surface, layers: [], camera: camera, margin: 0)
                guard let bounds = scene.quickBounds() else {
                    Check.expect(false, "\(name) has bounds"); continue
                }
                let px = Framing.fitting(bounds, in: viewport).unitsPerPixel
                let d = hausdorff(exact.mapped(camera.view), lattice.mapped(camera.view)) / px
                Check.expect(exact.count > 0 && lattice.count > 0 && d < 1,
                             "\(name), x \(Int(plate.xAngle))° z \(Int(plate.zAngle))°: "
                             + "within a pixel at \(viewport.width)x\(viewport.height)",
                             String(format: "Hausdorff %.3f px; %d exact runs, %d lattice runs",
                                    d, exact.count, lattice.count))
            }
        }
    }

    Check.suite("surfaces: an orbit frame of the forced torus, folds and all, keeps up") {
        let renderer = try MetalRenderer()
        let trajectory = Layer(spec: LayerSpec(name: "trajectory", role: .scaffold,
                                               source: .trajectory, width: 0.08,
                                               heightPolicy: .surface),
                               paths: try found.trajectory(e, duration: 500, every: 0.02))
        let folds = Layer(spec: LayerSpec(name: "folds", role: .outline, source: .foldLines,
                                          width: 0.6, heightPolicy: .surface),
                          paths: .empty)
        let orbit = Orbit(matching: PlateProjection(shear: 0, xAngle: -55, zAngle: 30,
                                                    flipX: false, yScale: nil))
        let scene = Scene(surface: forced, layers: [trajectory, folds], camera: orbit.camera,
                          margin: 0)
        guard let bounds = scene.quickBounds() else {
            Check.expect(false, "the forced torus has bounds"); return
        }
        var navigator = Navigator(orbit: orbit, framing: .fitting(bounds, in: viewport))
        let target = try renderer.makePreviewTarget(viewport)
        let options = PreviewOptions(slopeScale: 0)
        // The first frame lays out the ink and the normals; the orbit after
        // it pays only for what the camera changes.
        try renderer.renderPreview(scene, navigator: navigator, viewport: viewport,
                                   options: options, into: target)
        let clock = ContinuousClock()
        var frames: [Double] = []
        for _ in 0..<7 {
            navigator = navigator.applying(.orbit(SIMD2(12, 3)), in: viewport)
            let t = try clock.measure {
                try renderer.renderPreview(scene.looking(navigator.camera), navigator: navigator,
                                           viewport: viewport, options: options, into: target)
            }
            frames.append(Double(t.components.attoseconds) / 1e15 + Double(t.components.seconds) * 1e3)
        }
        let median = frames.sorted()[frames.count / 2]
        Check.expect(median < 25, "a frame at a new camera takes under 25 ms",
                     String(format: "median %.1f ms of %@", median,
                            frames.map { String(format: "%.1f", $0) }.joined(separator: ", ")))
    }
}

// MARK: - a surface plate, described and built

func surfacePlateTests() {
    Check.suite("plates: a surface request builds its plate, and an edit keeps what it can") {
        func forced(radius: Double) -> SurfaceRequest {
            SurfaceRequest(.forced(system: "jerk", fit: 8000, harmonics: 24, radial: 0, axial: 1,
                                   radius: radius, duration: 200, every: 0.02),
                           lattice: 256)
        }
        let clock = ContinuousClock()
        var first: SurfacePlate!, second: SurfacePlate!
        let fitted = try clock.measure { first = try SurfacePlate.build(forced(radius: 2.5)) }
        let kept = try clock.measure {
            second = try SurfacePlate.build(forced(radius: 3), reusing: first)
        }
        Check.expect(first.torus != nil && first.embedded
                     && first.layers.map(\.spec.name) == ["trajectory", "folds"],
                     "a forced plate is its trajectory and its folds, on an embedded torus")
        Check.expect(second.revolution?.radius == 3 && kept < fitted / 4,
                     "moving the revolution re-places the torus without refitting it",
                     "\(fitted) to fit, \(kept) to re-place")

        let torus = try SurfacePlate.build(SurfaceRequest(.torus(major: 2, minor: 1),
                                                          lattice: 128, lines: SIMD2(12, 6),
                                                          style: .init(folds: 0)))
        Check.expect(torus.layers.map(\.spec.name) == ["lines"] && torus.surface.outward != nil,
                     "a fold width of zero leaves the folds out")
        let spindle = try SurfacePlate.build(SurfaceRequest(.torus(major: 1, minor: 1.2),
                                                            lattice: 128))
        Check.expect(spindle.surface.outward == nil,
                     "a torus with R ≤ r bounds nothing, so hides nothing")
        Check.expectThrows("an unknown system is refused by name") {
            _ = try SurfacePlate.build(SurfaceRequest(.forced(system: "lorenz", fit: 100,
                                                              harmonics: 4, radial: 0, axial: 1,
                                                              radius: 2.5, duration: 1,
                                                              every: 0.1)))
        }
    }
}
