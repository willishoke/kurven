import Foundation
import Observation
import QuartzCore

/// Frame timing, summarized twice a second.
///
/// Three numbers, because "it feels slow" has three different causes: the CPU
/// taking long to encode a frame, the GPU taking long to draw it, and frames
/// arriving unevenly even when both are quick. The last is what jitter is, so
/// alongside the typical interval it reports the worst one.
///
/// Samples go into unobserved storage and only `summary` is published, on a
/// half-second cadence: an observable written every frame is exactly what made
/// the window re-evaluate its sidebar on every mouse event, and a timer that
/// did that would be measuring itself.
@MainActor
@Observable
final class FrameClock {
    private(set) var summary: String?

    @ObservationIgnored private var cpu: [Double] = []
    @ObservationIgnored private var gpu: [Double] = []
    @ObservationIgnored private var intervals: [Double] = []
    @ObservationIgnored private var lastFrame: CFTimeInterval?
    @ObservationIgnored private var lastPublish = CACurrentMediaTime()

    /// A frame was encoded, taking `seconds` of CPU on the main thread.
    func frame(cpu seconds: Double) {
        let now = CACurrentMediaTime()
        // An interval is only a frame interval while drawing is continuous; the
        // gap between the last frame of one drag and the first of the next is
        // how long the user waited, not how long a frame took.
        if let lastFrame, now - lastFrame < 0.1 { intervals.append(now - lastFrame) }
        lastFrame = now
        cpu.append(seconds)
        publishIfDue(now)
    }

    /// The GPU finished a frame. Called back from Metal's completion thread,
    /// hence the hop.
    nonisolated func gpu(seconds: Double) {
        Task { @MainActor in self.gpu.append(seconds) }
    }

    private func publishIfDue(_ now: CFTimeInterval) {
        guard now - lastPublish >= 0.5 else { return }
        lastPublish = now
        func ms(_ xs: [Double]) -> String {
            xs.isEmpty ? "–" : String(format: "%.1f", median(xs) * 1000)
        }
        var text = "cpu \(ms(cpu)) · gpu \(ms(gpu)) ms"
        if !intervals.isEmpty {
            text += String(format: " · %.0f fps, worst %.0f ms",
                           1 / median(intervals), intervals.max()! * 1000)
        }
        summary = text
        cpu.removeAll(keepingCapacity: true)
        gpu.removeAll(keepingCapacity: true)
        intervals.removeAll(keepingCapacity: true)
    }

    private func median(_ xs: [Double]) -> Double { xs.sorted()[xs.count / 2] }
}
