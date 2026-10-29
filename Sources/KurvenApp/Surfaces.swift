import Foundation
import Observation
import KurvenCore
import KurvenMath
import KurvenDynamics

/// The half of a document that is a parametric surface: a torus of
/// revolution, a doubly periodic function on the torus its periods glue
/// into, or a forced system's invariant torus.
///
/// Built as a landscape is sampled -- one build in flight on a detached task,
/// the latest edit queued behind it -- through `SurfacePlate.build`, the
/// function `kurven-cli surface` calls, so the window and the CLI draw the
/// same plate. A surface is not a bundle: `bundle` is nil for one, so the
/// controls that restyle, resample or save a bundle stay away from it.
@MainActor
extension Document {
    /// Open what the gallery chose.
    func create(_ entry: GalleryEntry) {
        switch entry {
        case .function(let preset): create(preset)
        case .surface(let preset): create(surface: preset)
        }
    }

    /// Start a surface from the catalog, replacing whatever is open.
    func create(surface preset: SurfacePreset) {
        generation += 1
        wanted = nil
        surface = preset.request
        send(surface: preset.request, framing: true)
    }

    /// The controls changed `surface`. `draft` is true while one is being
    /// dragged: the plate is built from `SurfaceRequest.drafted()` until the
    /// drag ends, and at full resolution after.
    func surfaceEdited(draft: Bool) {
        guard let request = surface else { return }
        send(surface: draft ? request.drafted() : request, framing: false, draft: draft)
    }

    private func send(surface request: SurfaceRequest, framing: Bool, draft: Bool = false) {
        surfaceWanted = (request, framing, draft)
        pumpSurface()
    }

    private func pumpSurface() {
        guard !building, let (request, framing, draft) = surfaceWanted else { return }
        surfaceWanted = nil
        if !framing, request == surfaceShown { return }
        // Whatever the plate on screen can lend the next one -- a fitted
        // torus, its values on the lattice, the trajectory's run, the
        // contours -- goes with the request.
        let previous = framing ? nil : plate
        building = true
        surfaceStatus = (previous?.refits(for: request) ?? true)
            && { if case .forced = request.shape { true } else { false } }()
            ? "fitting the torus…" : "building…"
        let asked = generation
        Task {
            let clock = ContinuousClock()
            let started = clock.now
            do {
                let built = try await Task.detached(priority: .userInitiated) {
                    try SurfacePlate.build(request, reusing: previous, draft: draft)
                }.value
                // A landscape or another surface chosen meanwhile is the
                // document now.
                if generation == asked {
                    adopt(built, keepingCamera: !framing && scene != nil)
                    surfaceShown = request
                    surfaceStatus = status(of: built, took: clock.now - started)
                        + (draft ? " (draft)" : "")
                }
            } catch {
                if generation == asked { surfaceStatus = "\(error)" }
            }
            building = false
            pumpSurface()
        }
    }

    /// What the section's status line says about a plate: how long it took,
    /// and anything about it that changes what the drawing means.
    private func status(of plate: SurfacePlate, took: Duration) -> String {
        let seconds = Double(took.components.seconds)
            + Double(took.components.attoseconds) / 1e18
        var text = String(format: "built in %.2f s", seconds)
        if let torus = plate.torus {
            text += String(format: "; misses its trajectory by %.1e of its size",
                           torus.residual / torus.extent)
            let rho = torus.rotationNumber
            let nearly = ContinuedFraction.convergents(of: rho, maxDenominator: 600)
                .dropFirst().map { "\($0.p)/\($0.q)" }.joined(separator: ", ")
            text += String(format: "; the trajectory winds at Ω/ω = %.6f = %@, nearly %@",
                           rho, ContinuedFraction.describe(ContinuedFraction.quotients(of: rho, count: 7),
                                                           ellipsis: true), nearly)
        }
        if !plate.embedded {
            text += " — the revolution passes through itself, so back faces are kept"
        } else if plate.surface.outward == nil {
            text += " — the torus reaches its axis and bounds nothing, so back faces are kept"
        }
        return text
    }
}
