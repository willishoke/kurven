import Foundation
import simd
import KurvenCore
import KurvenMath
import KurvenMetal
import KurvenBake
import KurvenDynamics

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
}
