import SwiftUI
import KurvenCore
import KurvenService
import KurvenLandscape
import KurvenMath

/// The controls for a landscape that is being chosen rather than opened.
///
/// Scoped as its own view, like `CameraSection`, because it reads `landscape`
/// and `sampling`, which a dragged slider writes several times a second; inline,
/// that would re-evaluate the layer list and the bake panel on every frame of a
/// drag to redraw four numbers.
///
/// Every control here says which side of the seam it is on. The function, the
/// window and the resolution resample (Python evaluates f again); the cap and
/// the hatch spacing restyle (the grids are already here). The status line
/// reports the first and never the second, because the second has no duration
/// worth reporting.
struct LandscapeSection: View {
    @Bindable var document: Document

    @State private var typed: String = ""
    @State private var editingExpression = false
    /// Each literal slider's reach, by the literal's index in the expression.
    /// Pinned when a drag starts, so a slider does not move its own scale,
    /// and forgotten when the expression is typed over.
    @State private var reaches: [Int: ClosedRange<Double>] = [:]
    /// The expression the reaches belong to: a change that did not come from
    /// a slider is a new set of literals.
    @State private var reachSource: String = ""
    @State private var draggingLiteral: Int?
    /// The expression as it was when the current drag began: one undo step
    /// per drag, not one per pixel.
    @State private var beforeDrag: String?
    @Environment(\.undoManager) private var undoManager

    private var landscape: LandscapeRequest? { document.landscape }

    var body: some View {
        let catalog = document.catalog
        presetPicker(catalog)
        if let landscape {
            expressionField(landscape)
            literalSliders(landscape)
            domainControls(landscape, window: window(catalog, landscape))
            resolutionControl(landscape)
            Toggle("Place contours by f, not by the grid",
                   isOn: Binding(get: { document.refineContours },
                                 set: { document.setRefinement($0) }))
                .font(.caption)
        }
        status
    }

    // MARK: - what

    @ViewBuilder
    private func presetPicker(_ catalog: Catalog) -> some View {
        Picker("Function", selection: Binding(
            get: { document.landscape?.name ?? "" },
            set: { name in
                guard let preset = catalog.preset(name) else { return }
                document.create(preset)
                typed = preset.expression
            })) {
            if document.landscape?.name.isEmpty ?? true {
                Text(document.landscape.map { "\($0.expression) (typed)" } ?? "—")
                    .tag("")
            }
            ForEach(catalog.presets) { preset in
                Text(preset.label).tag(preset.name)
            }
        }
        if let preset = catalog.preset(document.landscape?.name ?? ""), !preset.notes.isEmpty {
            Text(preset.notes).font(.caption).foregroundStyle(.secondary)
        }
        HStack {
            // The dropdown is faster for someone who knows the name; the
            // gallery answers the question a name cannot.
            Button("Browse…") { document.browsing = true }
                .controlSize(.small)
            Spacer()
        }
    }

    /// The whole point of the expression language being a real parser: this
    /// field takes anything the language accepts, not a name from a list. A
    /// dropdown is a set of starting points for it.
    @ViewBuilder
    private func expressionField(_ landscape: LandscapeRequest) -> some View {
        LabeledContent("f(z) =") {
            TextField("", text: Binding(
                get: { editingExpression ? typed : landscape.expression },
                set: { typed = $0; editingExpression = true }))
                .labelsHidden()
                .font(.system(.body, design: .monospaced))
                .onSubmit {
                    editingExpression = false
                    guard typed != landscape.expression, !typed.isEmpty else { return }
                    document.landscape?.expression = typed
                    // A typed expression is no longer the preset it started
                    // from, and saying so is what keeps the picker honest.
                    document.landscape?.name = ""
                    document.landscapeEdited(draft: false)
                }
        }
        .help("Press return to sample it. Anything the language parses: "
              + "gamma(z), 1/Γ(z), zeta(z), exp(1/z), z^3 - 1, sin(z)cn(z, 0.5)")
    }

    // MARK: - the numbers in it

    /// One control per number written in the expression: the modulus of
    /// `cn(z, 0.64)`, the order of `besselj(2, z)`, the exponent of `z^3`.
    ///
    /// Laid out as the Format inspector in Apple's own apps lays out a
    /// numeric property: a label that names the property, a slider, and a
    /// field with a stepper for the exact value. The slider rewrites the
    /// literal's characters in the expression and resamples, so there is no
    /// second place a parameter lives: the field reads `cn(z, 0.71)` because
    /// that is what the landscape is now. A moved number is no longer the
    /// preset it came from, as a typed one is not.
    @ViewBuilder
    private func literalSliders(_ landscape: LandscapeRequest) -> some View {
        let expression = landscape.expression
        let literals = KurvenMath.Expression.literals(in: expression)
        let labels = Self.labels(for: literals, in: expression)
        ForEach(Array(literals.enumerated()), id: \.offset) { index, literal in
            literalRow(index, literal, label: labels[index], in: expression)
        }
        .onChange(of: expression) { _, now in
            if now != reachSource { reaches = [:]; reachSource = now }
        }
    }

    /// "Modulus", "Order", "Exponent" -- and "Exponent 2" when the expression
    /// has two, so every row says which number it is.
    static func labels(for literals: [KurvenMath.Expression.Literal], in expression: String)
        -> [String]
    {
        let roles = literals.map { KurvenMath.Expression.role(of: $0, in: expression).label }
        var seen: [String: Int] = [:]
        for r in roles { seen[r, default: 0] += 1 }
        var counter: [String: Int] = [:]
        return roles.map { role in
            guard seen[role, default: 0] > 1 else { return role }
            counter[role, default: 0] += 1
            return "\(role) \(counter[role]!)"
        }
    }

    @ViewBuilder
    private func literalRow(_ index: Int, _ literal: KurvenMath.Expression.Literal,
                            label: String, in expression: String) -> some View {
        let reach = reaches[index] ?? Self.derivedReach(literal.value)
        let step = LandscapeStyle.nice((reach.upperBound - reach.lowerBound) / 100, down: true)
        let value = Binding<Double>(
            get: { literal.value },
            set: { value in
                guard value.isFinite else { return }
                // A typed or stepped value can be anywhere; the slider follows it.
                if !reach.contains(value) {
                    reaches[index] = min(reach.lowerBound, value)...max(reach.upperBound, value)
                }
                move(index, literal, to: value, in: expression, draft: false, label: label)
            })
        LabeledContent(label) {
            HStack(spacing: 8) {
                Slider(value: Binding(
                    get: { min(max(literal.value, reach.lowerBound), reach.upperBound) },
                    set: { value in
                        let quantized = (value / step).rounded() * step
                        move(index, literal, to: quantized, in: expression,
                             draft: draggingLiteral == index, label: label)
                    }), in: reach, onEditingChanged: { dragging in
                        if dragging {
                            // Pin the reach: derived from the value, it would
                            // otherwise stretch under the thumb as it moves.
                            reaches[index] = reach
                            draggingLiteral = index
                            beforeDrag = expression
                        } else {
                            draggingLiteral = nil
                            if let before = beforeDrag, let after = document.landscape?.expression,
                               before != after {
                                registerUndo(from: before, to: after, label)
                            }
                            beforeDrag = nil
                            document.landscapeEdited(draft: false)
                        }
                    })
                    .accessibilityLabel(label)
                    .accessibilityValue(KurvenMath.Expression.formatNumber(literal.value,
                                                                           significant: 6))
                TextField("", value: value,
                          format: .number.precision(.significantDigits(1...6)))
                    .labelsHidden()
                    .multilineTextAlignment(.trailing)
                    .monospacedDigit()
                    .frame(width: 64)
                    .accessibilityLabel(label)
                Stepper("", value: value, step: step)
                    .labelsHidden()
                    .accessibilityLabel(label)
            }
        }
        .help("\(label) in \(expression): the number \(literal.text). "
              + "Drag the slider, type a value, or step it; the expression and the "
              + "landscape follow.")
    }

    /// Rewrite one literal and resample. The expression is the model: the
    /// field, the provenance and the next set of literals all read from it.
    private func move(_ index: Int, _ literal: KurvenMath.Expression.Literal, to value: Double,
                      in expression: String, draft: Bool, label: String = "") {
        let next = KurvenMath.Expression.replacing(literal, with: value, significant: 6,
                                                   in: expression)
        guard next != expression else { return }
        // A drag registers its undo when it ends; a typed or stepped value is
        // one edit, registered here.
        if draggingLiteral != index { registerUndo(from: expression, to: next, label) }
        apply(next, draft: draft)
    }

    private func apply(_ next: String, draft: Bool) {
        reachSource = next
        editingExpression = false
        typed = next
        document.landscape?.expression = next
        document.landscape?.name = ""
        document.landscapeEdited(draft: draft)
    }

    /// ⌘Z puts the number back, and ⌘⇧Z moves it again. The action name is
    /// what the Edit menu shows: "Undo Modulus".
    private func registerUndo(from before: String, to after: String, _ label: String) {
        guard let undoManager else { return }
        undoManager.registerUndo(withTarget: document) { document in
            MainActor.assumeIsolated {
                // Re-registered the other way round, so the same edit can be
                // redone; SwiftUI has no handle on this view from here, so
                // the document's request is what moves.
                registerUndo(from: after, to: before, label)
                document.landscape?.expression = before
                document.landscape?.name = ""
                document.landscapeEdited(draft: false)
            }
        }
        undoManager.setActionName(label)
    }

    /// Room around a number, in the direction it already points: `0.64`
    /// reaches `0...1.28`, `-2` reaches `-4...0`, and `0` a unit each way.
    static func derivedReach(_ value: Double) -> ClosedRange<Double> {
        guard value.isFinite, value != 0 else { return -1...1 }
        return value < 0 ? (2 * value)...0 : 0...(2 * value)
    }

    // MARK: - where

    /// How far the sliders reach. A preset says (ζ wants sixty units of imag
    /// and tan wants three); a typed expression gets room around wherever it
    /// currently is.
    private func window(_ catalog: Catalog, _ landscape: LandscapeRequest) -> Domain {
        if let preset = catalog.preset(landscape.name) { return preset.window }
        let d = landscape.domain
        return Domain(real: Interval(lo: d.real.lo - d.real.length,
                                     hi: d.real.hi + d.real.length),
                      imag: Interval(lo: d.imag.lo - d.imag.length,
                                     hi: d.imag.hi + d.imag.length))
    }

    @ViewBuilder
    private func domainControls(_ landscape: LandscapeRequest, window: Domain) -> some View {
        intervalControl("Re", value: Binding(
            get: { document.landscape?.domain.real ?? landscape.domain.real },
            set: { document.landscape?.domain.real = $0 }), within: window.real)
        intervalControl("Im", value: Binding(
            get: { document.landscape?.domain.imag ?? landscape.domain.imag },
            set: { document.landscape?.domain.imag = $0 }), within: window.imag)
    }

    /// One axis of the window, as two thumbs that cannot cross.
    ///
    /// The pair is the control, not two controls: an interval whose ends have
    /// swapped is not a window, and a landscape sampled over it is a picture of
    /// nothing. The ends are kept a fiftieth of the slider's reach apart, which
    /// is close enough to zoom right in and far enough that the grid is never
    /// asked for a rectangle with no area.
    @ViewBuilder
    private func intervalControl(_ label: String, value: Binding<Interval>,
                                 within reach: Interval) -> some View {
        let gap = reach.length / 50
        LabeledContent(label) {
            HStack(spacing: 6) {
                number(Binding(
                    get: { value.wrappedValue.lo },
                    set: { value.wrappedValue.lo = min($0, value.wrappedValue.hi - gap) }))
                Slider(value: Binding(
                    get: { value.wrappedValue.lo },
                    set: { value.wrappedValue.lo = min($0, value.wrappedValue.hi - gap) }),
                       in: reach.lo...reach.hi, onEditingChanged: edited)
                Slider(value: Binding(
                    get: { value.wrappedValue.hi },
                    set: { value.wrappedValue.hi = max($0, value.wrappedValue.lo + gap) }),
                       in: reach.lo...reach.hi, onEditingChanged: edited)
                number(Binding(
                    get: { value.wrappedValue.hi },
                    set: { value.wrappedValue.hi = max($0, value.wrappedValue.lo + gap) }))
            }
        }
    }

    @ViewBuilder
    private func number(_ value: Binding<Double>) -> some View {
        TextField("", value: Binding(get: { value.wrappedValue }, set: {
            value.wrappedValue = $0
            document.landscapeEdited(draft: false)
        }), format: .number.precision(.fractionLength(1)))
            .labelsHidden().monospacedDigit().frame(width: 52)
    }

    // MARK: - how finely

    @ViewBuilder
    private func resolutionControl(_ landscape: LandscapeRequest) -> some View {
        let value = Binding(
            get: { Double(document.landscape?.resolution ?? landscape.resolution) },
            set: { document.landscape?.resolution = Int($0.rounded()) })
        LabeledContent("Samples") {
            HStack(spacing: 6) {
                Slider(value: value, in: 120...2000, step: 20, onEditingChanged: edited)
                Text("\(landscape.resolution)")
                    .font(.caption).monospacedDigit().frame(width: 34, alignment: .trailing)
            }
        }
        .help("Samples along the longer side of the window. While you drag any "
              + "of these controls the landscape is sampled at "
              + "\(document.draftResolution) and redrawn at this when you let go.")
    }

    /// A drag samples at the draft resolution while it lasts and at the real one
    /// when it ends -- so the picture keeps up with the hand, and is right when
    /// the hand stops.
    private func edited(_ isDragging: Bool) {
        document.landscapeEdited(draft: isDragging)
    }

    @ViewBuilder
    private var status: some View {
        if let text = document.landscapeStatus {
            HStack(spacing: 6) {
                if document.sampling { ProgressView().controlSize(.small) }
                Text(text).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            }
        }
    }
}

/// Where the landscape is cut off, and what that does to everything else.
///
/// Truncation is a property of the model, not of the camera: it is what turns a
/// pole into a plateau you can see the top of. Because the bundle describes its
/// layers rather than dumping them, moving it is a local re-derivation -- the
/// heightfield, the wall crests, every cap stroke, the rim, and which contours
/// survive, all from grids that never leave memory.
struct TruncationSection: View {
    @Bindable var document: Document

    var body: some View {
        Picker("Truncate", selection: Binding(get: { Kind(document.caps) },
                                              set: { apply($0) })) {
            Text("None").tag(Kind.none)
            Text("At a height").tag(Kind.uniform)
            Text("Per band of Re").tag(Kind.bands)
        }
        .pickerStyle(.segmented)

        switch document.caps {
        case .none:
            Text("The surface is drawn at |f|, however far up that goes. A "
                 + "function with a pole wants a cap; one without does not.")
                .font(.caption).foregroundStyle(.secondary)
        case .uniform(let z):
            capSlider("Cap", value: Binding(get: { z }, set: {
                document.setCaps(.uniform($0))
            }))
        case .realBands(let bands, let beyond):
            bandRows(bands, beyond)
        }
    }

    private var ceiling: Double {
        // Room to raise the cap past where the landscape currently ends, without
        // a slider whose whole useful range is its first pixel on a function
        // that reaches 1e12 at a pole.
        max((LandscapeStyle.ceiling(of: document.caps) ?? 5) * 3, 10)
    }

    @ViewBuilder
    private func capSlider(_ label: String, value live: Binding<Double>) -> some View {
        LabeledContent(label) {
            HStack(spacing: 6) {
                TextField("", value: live, format: .number.precision(.fractionLength(2)))
                    .labelsHidden().monospacedDigit().frame(width: 60)
                Slider(value: live, in: 0.1...ceiling)
            }
        }
    }

    /// Gamma's per-spire truncation, as rows: each band caps everything to the
    /// left of its threshold, and the last row is everything else. Tested in
    /// order, first match wins, which is exactly how the rows read top to
    /// bottom.
    @ViewBuilder
    private func bandRows(_ bands: [RealBand], _ beyond: Double) -> some View {
        ForEach(Array(bands.enumerated()), id: \.offset) { index, band in
            HStack(spacing: 6) {
                Text("Re <").font(.caption).foregroundStyle(.secondary)
                TextField("", value: Binding(
                    get: { band.below },
                    set: { value in
                        var next = bands
                        next[index].below = value
                        document.setCaps(.realBands(next, beyond: beyond))
                    }),
                    format: .number.precision(.fractionLength(1)))
                    .labelsHidden().monospacedDigit().frame(width: 52)
                Text("cap").font(.caption).foregroundStyle(.secondary)
                Slider(value: Binding(get: { band.cap }, set: {
                    var next = bands
                    next[index].cap = $0
                    document.setCaps(.realBands(next, beyond: beyond))
                }), in: 0.1...ceiling)
                Text(String(format: "%.2f", band.cap))
                    .font(.caption).monospacedDigit().frame(width: 40, alignment: .trailing)
                Button {
                    var next = bands
                    next.remove(at: index)
                    document.setCaps(.realBands(next, beyond: beyond))
                } label: { Image(systemName: "minus.circle") }
                    .buttonStyle(.borderless).controlSize(.small)
            }
        }
        capSlider("Beyond", value: Binding(
            get: { beyond.isFinite ? beyond : ceiling },
            set: {
                document.setCaps(.realBands(bands, beyond: $0))
            }))
        Button("Add a band") {
            let below = (bands.last?.below ?? -1) + 1
            document.setCaps(.realBands(bands + [RealBand(below: below,
                                                          cap: beyond.isFinite ? beyond : 5)],
                                        beyond: beyond))
        }
        .controlSize(.small)
    }

    private func apply(_ kind: Kind) {
        let current = LandscapeStyle.ceiling(of: document.caps) ?? 5
        switch kind {
        case .none: document.setCaps(.none)
        case .uniform: document.setCaps(.uniform(current))
        case .bands:
            // Start from the cap that is already there, as one band and a
            // beyond: the first edit should be a change to this landscape, not
            // a different one.
            let domain = document.bundle?.manifest.domain
            let middle = domain.map { ($0.real.lo + $0.real.hi) / 2 } ?? 0
            document.setCaps(.realBands([RealBand(below: middle, cap: current)],
                                        beyond: current))
        }
    }

    enum Kind: Hashable {
        case none, uniform, bands
        init(_ caps: Caps) {
            switch caps {
            case .none: self = .none
            case .uniform: self = .uniform
            case .realBands: self = .bands
            }
        }
    }
}
