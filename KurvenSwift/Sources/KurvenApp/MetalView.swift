import SwiftUI
import MetalKit
import KurvenCore
import KurvenMetal

/// SwiftUI has a `Gesture` protocol of its own, so the navigation one is named
/// here. The collision is worth living with: in `KurvenCore`, where there is no
/// SwiftUI, `Gesture` is exactly the right name for the thing.
private typealias Nav = KurvenCore.Gesture

/// The drawing surface, and the only place `NSEvent`s exist.
///
/// Input handling produces `Gesture` values and hands them to the model, which
/// folds them with a pure function. Nothing here knows what orbiting means; it
/// knows that a left-drag is one. That is what makes the navigation testable
/// without a window, and it is why the coordinator is thirty lines rather than
/// three hundred.
struct MetalView: NSViewRepresentable {
    let document: Document

    func makeCoordinator() -> Coordinator { Coordinator(document: document) }

    func makeNSView(context: Context) -> MTKView {
        let view = GestureView(frame: .zero, device: MTLCreateSystemDefaultDevice())
        view.coordinator = context.coordinator
        view.delegate = context.coordinator
        view.colorPixelFormat = .bgra8Unorm
        view.framebufferOnly = false
        // Draw on demand. A landscape that is not moving does not need sixty
        // frames a second of it, and a laptop notices.
        view.enableSetNeedsDisplay = true
        view.isPaused = true
        view.autoResizeDrawable = true
        context.coordinator.view = view
        return view
    }

    func updateNSView(_ view: MTKView, context: Context) {
        context.coordinator.document = document
        view.needsDisplay = true
    }

    @MainActor
    final class Coordinator: NSObject, MTKViewDelegate {
        var document: Document
        weak var view: MTKView?
        private var renderer: MetalRenderer?
        private(set) var lastError: String?

        init(document: Document) { self.document = document }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            document.viewport = Viewport(width: Int(size.width), height: Int(size.height))
            document.refreshScene()
        }

        func draw(in view: MTKView) {
            guard let scene = document.scene,
                  let navigator = document.navigator,
                  let drawable = view.currentDrawable else { return }
            do {
                let renderer = try self.renderer ?? MetalRenderer()
                self.renderer = renderer
                let viewport = Viewport(width: drawable.texture.width,
                                        height: drawable.texture.height)
                document.viewport = viewport
                guard let commands = renderer.commandQueue.makeCommandBuffer() else { return }
                try renderer.renderPreview(scene, navigator: navigator, viewport: viewport,
                                           options: document.previewOptions,
                                           into: drawable.texture,
                                           commandBuffer: commands)
                commands.present(drawable)
                commands.commit()
                lastError = nil
            } catch {
                lastError = "\(error)"
            }
        }

        /// The world point under a pixel, for double-click-to-retarget.
        func worldPoint(atPixel p: SIMD2<Double>) -> P3<WorldSpace>? {
            guard let renderer, let navigator = document.navigator else { return nil }
            return renderer.worldPoint(atPixel: p, navigator: navigator,
                                       viewport: document.viewport)
        }
    }
}

/// An `MTKView` that turns events into gestures.
///
/// Defaults follow SketchUp, which is what most people who draw landscapes
/// already have in their hands: left-drag orbits, shift or middle drag pans,
/// scroll zooms toward the cursor, and double-click re-targets.
final class GestureView: MTKView {
    weak var coordinator: MetalView.Coordinator?
    private var lastDrag: CGPoint?

    override var acceptsFirstResponder: Bool { true }

    /// Backing-store pixels, y from the top, which is what the framing wants.
    private func pixel(_ event: NSEvent) -> SIMD2<Double> {
        let local = convert(event.locationInWindow, from: nil)
        let scale = window?.backingScaleFactor ?? 1
        return SIMD2(Double(local.x) * Double(scale),
                     Double(bounds.height - local.y) * Double(scale))
    }

    private func delta(_ event: NSEvent) -> SIMD2<Double> {
        let scale = Double(window?.backingScaleFactor ?? 1)
        return SIMD2(Double(event.deltaX) * scale, Double(event.deltaY) * scale)
    }

    private func send(_ gesture: Nav) {
        MainActor.assumeIsolated {
            coordinator?.document.apply(gesture)
            needsDisplay = true
        }
    }

    override func mouseDown(with event: NSEvent) {
        if event.clickCount == 2 {
            MainActor.assumeIsolated {
                if let p = coordinator?.worldPoint(atPixel: pixel(event)) {
                    coordinator?.document.retarget(to: p)
                    needsDisplay = true
                }
            }
        }
    }

    override func mouseDragged(with event: NSEvent) {
        let d = delta(event)
        // Shift or option pans, matching the modifier most CAD tools use for
        // "move the paper rather than the camera".
        if event.modifierFlags.contains(.shift) || event.modifierFlags.contains(.option) {
            send(.pan(SIMD2(-d.x, d.y)))
        } else {
            send(.orbit(d))
        }
    }

    override func otherMouseDragged(with event: NSEvent) {
        let d = delta(event)
        send(.pan(SIMD2(-d.x, d.y)))
    }

    override func rightMouseDragged(with event: NSEvent) {
        let d = delta(event)
        send(.pan(SIMD2(-d.x, d.y)))
    }

    override func scrollWheel(with event: NSEvent) {
        let scale = Double(window?.backingScaleFactor ?? 1)
        let dx = Double(event.scrollingDeltaX) * scale
        let dy = Double(event.scrollingDeltaY) * scale
        if event.modifierFlags.contains(.shift) {
            send(.pan(SIMD2(dx, -dy)))
            return
        }
        // Toward the cursor, not the centre: under an orthographic camera that
        // is exactly "hold the point under the pointer still", and it is the
        // difference between zooming and hunting.
        let steps = dy * (event.hasPreciseScrollingDeltas ? 0.01 : 0.1) / scale
        guard steps != 0 else { return }
        send(.zoom(factor: exp(steps), at: pixel(event)))
    }

    override func magnify(with event: NSEvent) {
        guard event.magnification != 0 else { return }
        send(.zoom(factor: 1 + Double(event.magnification), at: pixel(event)))
    }

    override func rotate(with event: NSEvent) {
        guard event.rotation != 0 else { return }
        send(.roll(Double(event.rotation) * .pi / 180))
    }

    override func keyDown(with event: NSEvent) {
        MainActor.assumeIsolated {
            switch event.charactersIgnoringModifiers {
            case "f": coordinator?.document.fit(); needsDisplay = true
            case "1": coordinator?.document.mode = .plate; needsDisplay = true
            case "2": coordinator?.document.mode = .shaded(Lighting()); needsDisplay = true
            case "3": coordinator?.document.mode = .depth; needsDisplay = true
            default: super.keyDown(with: event)
            }
        }
    }
}
