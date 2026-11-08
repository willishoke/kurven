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
/// the latest edit queued behind it -- through `SurfaceDrawing.build`, the
/// function `kurven-cli surface` calls, so the window and the CLI draw the
/// same drawing. The drawing on screen is a bundle (`SurfaceBundle`), the
/// request that builds it and its layers, so it is saved, restyled and baked
/// as a landscape is: a request edit here is the surface's resample, and a
/// layer's levels its restyle (`redrawSurface`).
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
        surfaceStyled = nil
        forgetEdits()
        surface = preset.request
        send(surface: preset.request, framing: true)
    }

    /// The controls changed `surface`. `draft` is true while one is being
    /// dragged: the drawing is built from `SurfaceRequest.drafted()` until the
    /// drag ends, and at full resolution after.
    func surfaceEdited(draft: Bool) {
        guard let request = surface else { return }
        send(surface: draft ? request.drafted() : request, framing: false, draft: draft)
    }

    private func send(surface request: SurfaceRequest, framing: Bool, draft: Bool = false) {
        surfaceWanted = (request, framing, draft)
        pumpSurface()
    }

    /// Paint layers on the surface: the request's own curves follow it, and
    /// any other layer is drawn on the surface as it is (`redrawSurface`),
    /// or on the one a request edit in flight will build.
    func restyleSurface(_ layers: [LayerSpec]) {
        surfaceStyled = layers
        redrawSurface()
    }

    private func pumpSurface() {
        guard !building, let (request, framing, draft) = surfaceWanted else { return }
        surfaceWanted = nil
        if !framing, request == surfaceShown { return }
        // Whatever the drawing on screen can lend the next one -- a fitted
        // torus, its values on the grid, the trajectory's run, the
        // contours -- goes with the request.
        let previous = framing ? nil : surfaceDrawing
        // The layers the bundle keeps -- its levels, a contour the request
        // does not draw -- survive the edit (`SurfaceDrawing.keeping(_:)`),
        // or the layers asked for since, which the edit then draws.
        let kept = framing ? nil : surfaceStyled ?? surfaceBundle?.manifest.layers
        let provenance = surfaceBundle?.manifest.provenance.gitSha
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
                    let built = try SurfaceDrawing.build(request, reusing: previous, draft: draft)
                    return try kept.map { try built.keeping($0, from: previous) } ?? built
                }.value
                // A landscape or another surface chosen meanwhile is the
                // document now.
                if generation == asked {
                    adopt(built, keepingCamera: !framing && scene != nil, gitSha: framing ? nil : provenance)
                    if surfaceStyled == built.layers.map(\.spec) { surfaceStyled = nil }
                    surfaceShown = request
                    surfaceStatus = status(of: built, took: clock.now - started)
                        + (draft ? " (draft)" : "")
                }
            } catch {
                if generation == asked { surfaceStatus = "\(error)" }
            }
            building = false
            pumpSurface()
            // Layers or levels asked for while it built are drawn on it.
            if !building { redrawSurface() }
        }
    }

    /// A parametric surface's layers at the levels their sliders set, derived
    /// off the main actor and coalesced as a landscape's are (`redraw`): each
    /// layer at levels of its own is drawn again on the surface
    /// (`SurfaceDrawing.restyled(_:)`), the rest kept. A draft is drawn in
    /// full: |f| and arg f are placed by the function either way.
    func redrawSurface() {
        guard !drawing, let surfaceBundle, scene != nil else { return }
        if let layers = surfaceStyled {
            // A request edit in flight draws them on what it builds.
            if building { return }
            guard layers != surfaceBundle.manifest.layers else {
                surfaceStyled = nil
                return redrawSurface()
            }
            drawing = true
            let asked = version
            Task {
                let restyled = await Task.detached(priority: .userInitiated) {
                    Result { try surfaceBundle.restyled(layers) }
                }.value
                drawing = false
                if version == asked {
                    switch restyled {
                    case .success(let bundle):
                        adoptRestyled(bundle)
                        if surfaceStyled == layers { surfaceStyled = nil }
                    case .failure(let error):
                        surfaceStatus = "\(error)"
                        if surfaceStyled == layers { surfaceStyled = nil }
                    }
                }
                redrawSurface()
            }
            return
        }
        guard curvesWanted != nil else { return }
        curvesWanted = nil
        drawing = true
        let asked = version
        let layers = drawnLayers
        Task {
            let restyled = await Task.detached(priority: .userInitiated) {
                Result { try surfaceBundle.drawing.restyled(layers) }
            }.value
            drawing = false
            if version == asked {
                switch restyled {
                case .success(let drawn): if let scene { self.scene = scene.drawing(drawn.layers) }
                case .failure(let error): surfaceStatus = "\(error)"
                }
            }
            redrawSurface()
        }
    }

    /// What the section's status line says about a drawing: how long it took,
    /// and anything about it that changes what the drawing means.
    private func status(of surfaceDrawing: SurfaceDrawing, took: Duration) -> String {
        let seconds = Double(took.components.seconds)
            + Double(took.components.attoseconds) / 1e18
        var text = String(format: "built in %.2f s", seconds)
        if let torus = surfaceDrawing.torus {
            text += String(format: "; misses its trajectory by %.1e of its size",
                           torus.residual / torus.extent)
            let rho = torus.rotationNumber
            let nearly = ContinuedFraction.convergents(of: rho, maxDenominator: 600)
                .dropFirst().map { "\($0.p)/\($0.q)" }.joined(separator: ", ")
            text += String(format: "; the trajectory winds at Ω/ω = %.6f = %@, nearly %@",
                           rho, ContinuedFraction.describe(ContinuedFraction.quotients(of: rho, count: 7),
                                                           ellipsis: true), nearly)
        }
        if !surfaceDrawing.embedded {
            text += " — the revolution passes through itself, so back faces are kept"
        } else if surfaceDrawing.surface.outward == nil {
            text += " — the torus reaches its axis and bounds nothing, so back faces are kept"
        }
        return text
    }
}
