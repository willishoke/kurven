import SwiftUI
import KurvenCore
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
///     sn       radii                     6 ms      contours re-placed, not resampled
///              modulus                 100 ms      resampled; 38 ms while dragging
///              samples                  70 ms      resampled; ink only
///     forced   ring radius              78 ms      lattice re-placed from state space;
///                                                  20 ms while dragging
///              radius from, height from 26 ms      re-placed (this one not embedded, so
///                                                  its ink needs no normals)
///              duration, sample spacing 60–110 ms  integrated again; 18 ms shortening
///                                                  while dragging, read off the run
///              parameter lines         112 ms      the map at every line vertex
///              harmonics, fit length   440 ms      fitted again
///
/// So the forced torus's fit is the one control that is not interactive, and
/// it gets a stepper and a field rather than a slider.
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
        case .duration:
            Control(label: "Duration", range: 100...8000, step: 50, digits: 0,
                    help: "How long a run of the trajectory is drawn. Shortening it while you "
                        + "drag reads the run already integrated.")
        case .every:
            Control(label: "Sample spacing", range: 0.002...0.05, step: 0.001, digits: 3,
                    help: "Time between the trajectory's vertices. Four times coarser while "
                        + "you drag any control.")
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
        }
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
            LabeledContent(c.label) {
                HStack(spacing: 8) {
                    if c.slides {
                        Slider(value: Binding(
                            get: { min(max(current, c.range.lowerBound), c.range.upperBound) },
                            set: { set(field, to: ($0 / c.step).rounded() * c.step,
                                       draft: dragging == field, label: c.label) }),
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
