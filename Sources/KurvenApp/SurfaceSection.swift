import SwiftUI
import KurvenCore
import KurvenMath
import KurvenDynamics

/// The numbers a surface is made of, as controls.
///
/// Laid out as `LandscapeSection`'s literal rows are -- label, slider, field,
/// stepper -- and like them, one undo step per drag, named for the control.
/// Every control writes one field of the document's `SurfaceRequest` and the
/// drawing is built again from the one on screen (`SurfaceDrawing.build`), which
/// redoes only what the edit touches. While a slider is dragged the drawing is
/// built from `SurfaceRequest.drafted()`; the full request follows the release.
///
/// What each costs is decided by the classes of fields it touches
/// (`FieldClass`), whatever is painted on the surface: a contour of a
/// coordinate is kept in parameter coordinates while the parameter rectangle
/// keeps its samples, and only carried into space again when the surface
/// moves; a contour of the height is kept while the surface is, and found
/// again when it moves; |f| and arg f are carried with the surface, and
/// sampled again when the function or its sampling changes
/// (`SurfaceDrawing.keeping(_:from:)`). Measured by `kurven-cli
/// surface-costs` on an Apple M5 Max at 1600×1200, with nothing painted: the
/// build from the drawing before, plus the first preview frame after it --
/// which lays out the curves, and uploads a surface only when the surface
/// moved.
///
///                                          coordinates  height   |f|, arg f
///     torus    radii                 13 ms  carried      again
///              line counts           11 ms  kept         kept
///              linear curve slope    11 ms  kept         kept
///     sn       radii                  6 ms  again        again    carried
///              modulus              115 ms  again        again    sampled again; 65 ms dragging
///              samples              100 ms  kept         kept     sampled again
///     forced   ring radius           39 ms  carried      again    7 ms dragging
///              radius from,          32 ms  carried      again
///              height from
///              turns              17–36 ms  kept         kept     the trajectory is placed from
///                                                                 the grid; no draft needed
///              coordinate curves     38 ms  kept         kept     the map at every line vertex
///              linear curve slope    15 ms  kept         kept
///              harmonics, fit       390 ms  carried      again    fitted again
///
/// sn sizes its tube's samples by its radii, so its radii move its
/// rectangle's samples and its coordinates are contoured again. The forced
/// torus's fit is the one control that is not interactive, and it gets a
/// stepper and a field rather than a slider. A curve edit lays out only the
/// layer it touched -- the preview keeps every other layer's vertices from
/// the frame before -- and curve normals are read off the grid, not asked
/// of the map, so a new surface costs its grid and little else.
struct SurfaceSection: View {
    @Bindable var document: Document


    var body: some View {
        if let request = document.surface {
            if let preset = SurfacePreset.named(request.name) {
                Text(preset.notes).font(.caption).foregroundStyle(.secondary)
            }
            ForEach(request.fields, id: \.self) { field in
                row(field, request)
            }
            lines(request)
            linearCurves(request)
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
                        + "the linear curve at Ω/ω, and the slider rests at the denominators of "
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
                        + "at fractions, where the linear curve closes, and on the forced torus at "
                        + "the trajectory's own Ω/ω.")
        case .turns:
            Control(label: "Turns", range: 1...64, step: 1, digits: 0,
                    help: "How many turns round the ring a strand runs, unless it closes first.")
        case .strands:
            Control(label: "Strands", range: 1...12, step: 1, digits: 0,
                    help: "Parallel linear curves, evenly spaced through the tube.")
        case .relief:
            Control(label: "Relief", range: SurfaceRequest.reliefRange, step: 0.01,
                    help: "How far the torus is displaced along its normal per unit of |sn|, "
                        + "truncated at 3. Past the torus's reach the relief passes through "
                        + "itself, and its back faces are kept.")
        }
    }

    @ViewBuilder
    private func row(_ field: SurfaceRequest.Field, _ request: SurfaceRequest) -> some View {
        let c = control(field)
        if field == .radial || field == .axial {
            componentPicker(field, request, label: c.label)
        } else if let current = request[field] {
            let slider = SurfaceFieldSlider(document: document, field: field, label: c.label)
            let value = Binding<Double>(
                get: { current },
                set: { slider.set(min(max($0, c.range.lowerBound), c.range.upperBound)) })
            // A slider with detents rests at them: a value within a detent's
            // reach lands on it exactly, so a fraction is a fraction and the
            // linear curve closes, and a convergent's turns are those turns.
            let detents = self.detents(field)
            LabeledContent(c.label) {
                HStack(spacing: 8) {
                    if c.slides {
                        Slider(value: Binding(
                            get: { min(max(current, c.range.lowerBound), c.range.upperBound) },
                            set: { slider.set(quantize($0, step: c.step, detents: detents,
                                                       reach: Self.detentReach(field))) }),
                               in: c.range, onEditingChanged: { slider.editingChanged($0) })
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
            Toggle("Coordinate curves", isOn: Binding(
                get: { request.lines != nil },
                set: { on in
                    var next = request
                    next.lines = on ? SIMD2(36, 18) : nil
                    commit(next, label: "Coordinate Curves")
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

    // MARK: - the linear curve

    /// A linear curve: a straight line on the flat torus, drawn on this one. Its
    /// slope is the control that moves in real time -- the surface never
    /// sees the linear curve, so a new slope is its own vertices and nothing else.
    @ViewBuilder
    private func linearCurves(_ request: SurfaceRequest) -> some View {
        Toggle("Linear curve", isOn: Binding(
            get: { request.linearCurves != nil },
            set: { on in
                var next = request
                // On, it starts at the trajectory's own linear curve when there is
                // one to match, else at 2/5, which closes.
                next.linearCurves = on ? SurfaceRequest.LinearCurves(slope: ownSlope ?? 0.4) : nil
                commit(next, label: "Linear curve")
            }))
            .help("A straight line through the parameter rectangle, so many turns through "
                  + "the tube per turn round the ring, drawn on the torus. A fraction closes; "
                  + "anything else winds on. On the forced torus the trajectory is the linear curve "
                  + "at the system's own Ω/ω.")
        if let w = request.linearCurves {
            ForEach(SurfaceRequest.linearCurveFields, id: \.self) { field in
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
    /// convergents -- the closed linear curves the trajectory is nearly, which
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

    /// The trajectory's own linear curve, when the drawing on screen is a fitted
    /// torus: its rotation number, Ω/ω.
    private var ownSlope: Double? { document.surfaceDrawing?.torus?.rotationNumber }

    /// The rotation number's convergents within the turns slider's range.
    private var convergents: [ContinuedFraction.Convergent] {
        ownSlope.map { ContinuedFraction.convergents(of: $0, maxDenominator: 600) } ?? []
    }

    private func describe(_ w: SurfaceRequest.LinearCurves) -> String {
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

    /// One edit, one undo step, which puts back the request and the layers
    /// together (`Document.edit`). The Edit menu names the control: "Undo
    /// Ring Radius".
    private func commit(_ next: SurfaceRequest, label: String) {
        document.editSurface(next, label)
    }
}

/// One number of a surface's request as its slider, field and stepper set
/// it, and a headless drag does: a drag is taken hold of, moved, let go,
/// one undo step; a value typed or stepped is an edit of its own.
@MainActor
struct SurfaceFieldSlider {
    let document: Document
    let field: SurfaceRequest.Field
    let label: String

    /// Whether a drag builds drafts. A linear curve, and the trajectory,
    /// are curves alone, placed from the grid, so their drags build the full
    /// drawing each step: the surface is kept and only their vertices are
    /// made again.
    var drafts: Bool { !SurfaceRequest.curveFields.contains(field) }

    func editingChanged(_ editing: Bool) {
        if editing {
            document.beginDrag()
        } else {
            // Let go first, so a drag back to where it began records the
            // bundle it began on (`Document.endDrag`); then the full request.
            document.endDrag(label)
            document.surfaceEdited(draft: false)
        }
    }

    /// A value the request refuses -- not a number, or out of its range --
    /// is not taken, a step of a drag included, and the status says why: the
    /// drawing and what a save writes stay at the last value taken.
    func set(_ value: Double) {
        guard let current = document.surface, current[field] != value else { return }
        let next: SurfaceRequest
        do { next = try current.setting(field, to: value) } catch {
            document.paintStatus = "\(error)"
            return
        }
        if document.editedBeforeDrag != nil {
            document.dragSurface(next, draft: drafts)
        } else {
            document.editSurface(next, label)
        }
    }
}
