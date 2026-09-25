import SwiftUI
import KurvenCore
import KurvenMath
import KurvenDynamics

/// The numbers a surface is made of, as controls.
///
/// Laid out as `LandscapeSection`'s literal rows are -- label, slider, field,
/// stepper -- and like them, one undo step per drag, named for the control.
/// Every control writes one field of the document's `SurfaceRequest` and the
/// plate is built again from the one on screen (`SurfacePlate.build`), which
/// redoes only what the edit touches. While a slider is dragged the plate is
/// built from `SurfaceRequest.drafted()`; the full request follows the release.
///
/// What each costs, measured by `kurven-cli surface-costs` on an Apple M5
/// Max at 1600×1200: the build from the plate before, plus the first preview
/// frame after it -- which lays out the ink, and uploads a surface only when
/// the surface moved.
///
///     torus    radii                    13 ms      new surface
///              line counts              11 ms      ink only
///              winding slope            11 ms      ink only
///     sn       radii                     6 ms      contours re-placed, not resampled
///              modulus                 100 ms      resampled; 38 ms while dragging
///              samples                  70 ms      resampled; ink only
///     forced   ring radius              39 ms      lattice re-placed from state space;
///                                                  7 ms while dragging
///              radius from, height from 32 ms      re-placed
///              turns                 17–36 ms      the trajectory is a winding, placed
///                                                  from the lattice; no draft needed
///              parameter lines          38 ms      the map at every line vertex
///              winding slope            15 ms      ink only; the trajectory's layout
///                                                  is kept
///              harmonics, fit length   390 ms      fitted again
///
/// So the forced torus's fit is the one control that is not interactive, and
/// it gets a stepper and a field rather than a slider. An ink edit lays out
/// only the layer it touched -- the preview keeps every other layer's
/// vertices from the frame before -- and ink normals are read off the
/// lattice, not asked of the map, so a new surface costs its lattice and
/// little else.
struct SurfaceSection: View {
    @Bindable var document: Document

    @State private var dragging: SurfaceRequest.Field?
    /// The request as it was when the current drag began: one undo step per
    /// drag, not one per pixel.
    @State private var beforeDrag: SurfaceRequest?
    @Environment(\.undoManager) private var undoManager

    var body: some View {
        if let request = document.surface {
            if let preset = SurfacePreset.named(request.name) {
                Text(preset.notes).font(.caption).foregroundStyle(.secondary)
            }
            ForEach(request.fields, id: \.self) { field in
                row(field, request)
            }
            lines(request)
            winding(request)
        }
        HStack {
            Button("Browse…") { document.browsing = true }.controlSize(.small)
            Spacer()
        }
        SurfaceStatus(document: document)
    }

    // MARK: - one number

    /// How a field is shown and how far it reaches.
    private struct Control {
        let label: String
        let range: ClosedRange<Double>
        let step: Double
        /// A slider, or only a field and a stepper: a refit is half a second,
        /// and a slider that stalls under the hand is worse than none.
        var slides = true
        var digits = 3
        var help = ""
    }

    private func control(_ field: SurfaceRequest.Field) -> Control {
        switch field {
        case .major:
            Control(label: "Major radius", range: 0.5...5, step: 0.05,
                    help: "The radius of the ring the tube runs round.")
        case .minor:
            Control(label: "Minor radius", range: 0.1...3, step: 0.05,
                    help: "The tube's radius. At or above the major radius the torus reaches "
                        + "its axis, bounds nothing, and hides nothing.")
        case .modulus:
            Control(label: "Modulus", range: 0.01...0.99, step: 0.01,
                    help: "m in sn(z, m). It sets the periods, so the rectangle the torus is "
                        + "rolled from changes with it, and the function is sampled again.")
        case .grid:
            Control(label: "Samples", range: 120...1200, step: 20, digits: 0,
                    help: "Samples across the period rectangle that the contours are traced "
                        + "on. While you drag, at most 240.")
        case .radius:
            Control(label: "Ring radius", range: 0.5...5, step: 0.05,
                    help: "The radius the torus is revolved at. Below about 1 a slice "
                        + "reaches the axis and the torus passes through itself.")
        case .trajectoryTurns:
            Control(label: "Turns", range: 1...600, step: 1, digits: 0,
                    help: "Turns of the forcing the trajectory is drawn for. The trajectory is "
                        + "the winding at Ω/ω, and the slider rests at the denominators of "
                        + "Ω/ω's convergents, where it cuts the torus most evenly; between "
                        + "them it bands.")
        case .harmonics:
            Control(label: "Harmonics", range: 4...40, step: 1, slides: false, digits: 0,
                    help: "The order of the Fourier fit. Too few and the fit misses the "
                        + "trajectory by more than it allows; the status line says so.")
        case .fit:
            Control(label: "Fit length", range: 1000...20000, step: 500, slides: false, digits: 0,
                    help: "Time units of the run the torus is fitted from.")
        case .radial, .axial:
            Control(label: field == .radial ? "Radius from" : "Height from", range: 0...2,
                    step: 1, digits: 0)
        case .slope:
            Control(label: "Slope", range: 0...2, step: 0.001,
                    help: "Turns through the tube per turn round the ring. The slider rests "
                        + "at fractions, where the winding closes, and on the forced torus at "
                        + "the trajectory's own Ω/ω.")
        case .turns:
            Control(label: "Turns", range: 1...64, step: 1, digits: 0,
                    help: "How many turns round the ring a strand runs, unless it closes first.")
        case .strands:
            Control(label: "Strands", range: 1...12, step: 1, digits: 0,
                    help: "Parallel windings, evenly spaced through the tube.")
        }
    }

    /// Whether a drag of this control builds drafts. A winding, and the
    /// trajectory, are ink alone, placed from the lattice, so their drags
    /// build the full plate each step: the surface is kept and only their
    /// vertices are made again.
    private func drafts(_ field: SurfaceRequest.Field) -> Bool {
        !SurfaceRequest.inkFields.contains(field)
    }

    @ViewBuilder
    private func row(_ field: SurfaceRequest.Field, _ request: SurfaceRequest) -> some View {
        let c = control(field)
        if field == .radial || field == .axial {
            componentPicker(field, request, label: c.label)
        } else if let current = request[field] {
            let value = Binding<Double>(
                get: { current },
                set: { set(field, to: min(max($0, c.range.lowerBound), c.range.upperBound),
                           draft: false, label: c.label) })
            // A slider with detents rests at them: a value within a detent's
            // reach lands on it exactly, so a fraction is a fraction and the
            // winding closes, and a convergent's turns are those turns.
            let detents = self.detents(field)
            LabeledContent(c.label) {
                HStack(spacing: 8) {
                    if c.slides {
                        Slider(value: Binding(
                            get: { min(max(current, c.range.lowerBound), c.range.upperBound) },
                            set: { set(field, to: quantize($0, step: c.step, detents: detents,
                                                           reach: Self.detentReach(field)),
                                       draft: dragging == field && drafts(field), label: c.label) }),
                               in: c.range, onEditingChanged: { editing in
                                   dragged(field, editing, label: c.label)
                               })
                            .accessibilityLabel(c.label)
                    } else {
                        Spacer()
                    }
                    TextField("", value: value,
                              format: .number.precision(.fractionLength(c.digits)))
                        .labelsHidden().multilineTextAlignment(.trailing).monospacedDigit()
                        .frame(width: 64)
                        .accessibilityLabel(c.label)
                    Stepper("", value: value, step: c.step).labelsHidden()
                        .accessibilityLabel(c.label)
                }
            }
            .help(c.help)
        }
    }

    /// Which state component is drawn as the ring's radius, and which as its
    /// height. The two must differ -- one component as both is a curve, not
    /// a torus -- so choosing the other's swaps them.
    @ViewBuilder
    private func componentPicker(_ field: SurfaceRequest.Field, _ request: SurfaceRequest,
                                 label: String) -> some View {
        let other: SurfaceRequest.Field = field == .radial ? .axial : .radial
        Picker(label, selection: Binding(
            get: { Int(request[field] ?? 0) },
            set: { chosen in
                var next = request
                if Int(request[other] ?? -1) == chosen { next[other] = request[field] }
                next[field] = Double(chosen)
                commit(next, label: label)
            })) {
            Text("x").tag(0)
            Text("x′").tag(1)
            Text("x″").tag(2)
        }
        .pickerStyle(.segmented)
        .help("The state component drawn as the ring's \(field == .radial ? "radius" : "height"). "
              + "Drawn by revolution, each slice of the torus gets a half-plane to itself.")
    }

    // MARK: - lines

    /// A plain torus is its lines; any other surface can have them too.
    @ViewBuilder
    private func lines(_ request: SurfaceRequest) -> some View {
        let isTorus = { if case .torus = request.shape { true } else { false } }()
        if !isTorus {
            Toggle("Parameter lines", isOn: Binding(
                get: { request.lines != nil },
                set: { on in
                    var next = request
                    next.lines = on ? SIMD2(36, 18) : nil
                    commit(next, label: "Parameter Lines")
                }))
                .help("Lines of constant forcing phase and constant internal phase: the "
                      + "invariant circles and the curves across them.")
        }
        if let n = request.lines {
            LabeledContent("Lines") {
                HStack(spacing: 6) {
                    count(n.x, label: "Lines Around") { var m = n; m.x = $0; return m }
                    Text("×").foregroundStyle(.secondary)
                    count(n.y, label: "Lines Across") { var m = n; m.y = $0; return m }
                }
            }
        }
    }

    @ViewBuilder
    private func count(_ value: Int, label: String,
                       _ change: @escaping (Int) -> SIMD2<Int>) -> some View {
        let binding = Binding<Int>(get: { value }, set: { v in
            guard var next = document.surface else { return }
            next.lines = change(min(max(v, 1), 200))
            commit(next, label: label)
        })
        TextField("", value: binding, format: .number)
            .labelsHidden().multilineTextAlignment(.trailing).monospacedDigit().frame(width: 40)
        Stepper("", value: binding, in: 1...200, step: 2).labelsHidden()
    }

    // MARK: - the winding

    /// A winding: a straight line on the flat torus, drawn on this one. Its
    /// slope is the control that moves in real time -- the surface never
    /// sees the winding, so a new slope is its own vertices and nothing else.
    @ViewBuilder
    private func winding(_ request: SurfaceRequest) -> some View {
        Toggle("Winding", isOn: Binding(
            get: { request.winding != nil },
            set: { on in
                var next = request
                // On, it starts at the trajectory's own winding when there is
                // one to match, else at 2/5, which closes.
                next.winding = on ? SurfaceRequest.Winding(slope: ownSlope ?? 0.4) : nil
                commit(next, label: "Winding")
            }))
            .help("A straight line through the parameter rectangle, so many turns through "
                  + "the tube per turn round the ring, drawn on the torus. A fraction closes; "
                  + "anything else winds on. On the forced torus the trajectory is the winding "
                  + "at the system's own Ω/ω.")
        if let w = request.winding {
            ForEach(SurfaceRequest.windingFields, id: \.self) { field in
                row(field, request)
            }
            Text(describe(w)).font(.caption).foregroundStyle(.secondary)
        }
    }

    /// How close to a detent a slider value lands on it: a pixel or two of
    /// a sidebar slider. The slope's range is two, the turns' six hundred.
    private static func detentReach(_ field: SurfaceRequest.Field) -> Double {
        field == .trajectoryTurns ? 4 : 0.015
    }

    /// A slider value to its step, or to the detent within reach of it.
    private func quantize(_ x: Double, step: Double, detents: [Double], reach: Double) -> Double {
        let stepped = (x / step).rounded() * step
        guard let d = detents.min(by: { abs($0 - x) < abs($1 - x) }),
              abs(d - x) <= reach else { return stepped }
        return d
    }

    /// Where a slider rests.
    ///
    /// The slope's: on a forced torus, the rotation number Ω/ω and its
    /// convergents -- the closed windings the trajectory is nearly, which
    /// dragging to from Ω/ω shows the orbit locking to; on any other torus,
    /// every fraction with a denominator up to eight. The trajectory's turns:
    /// the convergents' denominators, where the trajectory cuts the torus
    /// most evenly.
    private func detents(_ field: SurfaceRequest.Field) -> [Double] {
        switch field {
        case .slope:
            if let own = ownSlope {
                return [own] + convergents.map(\.value)
            }
            var out: [Double] = []
            for q in 1...8 {
                for p in 0...(2 * q) where q == 1 || p % q != 0 {
                    let x = Double(p) / Double(q)
                    if !out.contains(where: { abs($0 - x) < 1e-12 }) { out.append(x) }
                }
            }
            return out
        case .trajectoryTurns:
            return convergents.map { Double($0.q) }.filter { $0 >= 2 }
        default:
            return []
        }
    }

    /// The trajectory's own winding, when the plate on screen is a fitted
    /// torus: its rotation number, Ω/ω.
    private var ownSlope: Double? { document.plate?.torus?.rotationNumber }

    /// The rotation number's convergents within the turns slider's range.
    private var convergents: [ContinuedFraction.Convergent] {
        ownSlope.map { ContinuedFraction.convergents(of: $0, maxDenominator: 600) } ?? []
    }

    private func describe(_ w: SurfaceRequest.Winding) -> String {
        let strands = w.count == 1 ? "One strand" : "\(w.count) strands"
        if let own = ownSlope, abs(w.slope - own) < 1e-12 {
            return "\(strands) at the trajectory's own Ω/ω, "
                + String(format: "%.6f: open, %d turns.", own, w.turns)
        }
        if let q = w.closes {
            let p = Int((w.slope * Double(q)).rounded())
            let convergent = convergents.contains { $0.p == p && $0.q == q }
            return "\(strands) at \(p)/\(q)" + (convergent ? ", a convergent of Ω/ω" : "") + ": "
                + (q == 1 ? "closes after one turn." : "closes after \(q) turns.")
        }
        return "\(strands), open: \(w.turns) turns."
    }

    // MARK: - editing

    private func set(_ field: SurfaceRequest.Field, to value: Double, draft: Bool, label: String) {
        guard var next = document.surface, next[field] != value else { return }
        next[field] = value
        if draft {
            document.surface = next
            document.surfaceEdited(draft: true)
        } else {
            commit(next, label: label)
        }
    }

    /// A drag starts: remember where from. It ends: one undo step, and the
    /// full request.
    private func dragged(_ field: SurfaceRequest.Field, _ editing: Bool, label: String) {
        if editing {
            dragging = field
            beforeDrag = document.surface
        } else {
            dragging = nil
            if let before = beforeDrag, let after = document.surface, before != after {
                registerUndo(from: before, to: after, label)
            }
            beforeDrag = nil
            document.surfaceEdited(draft: false)
        }
    }

    /// One edit, one undo step.
    private func commit(_ next: SurfaceRequest, label: String) {
        guard let before = document.surface, before != next else { return }
        registerUndo(from: before, to: next, label)
        document.surface = next
        document.surfaceEdited(draft: false)
    }

    /// ⌘Z puts the request back, and ⌘⇧Z moves it again; the Edit menu
    /// names the control: "Undo Ring Radius".
    private func registerUndo(from before: SurfaceRequest, to after: SurfaceRequest,
                              _ label: String) {
        guard let undoManager else { return }
        undoManager.registerUndo(withTarget: document) { document in
            MainActor.assumeIsolated {
                registerUndo(from: after, to: before, label)
                document.surface = before
                document.surfaceEdited(draft: false)
            }
        }
        undoManager.setActionName(label)
    }
}
