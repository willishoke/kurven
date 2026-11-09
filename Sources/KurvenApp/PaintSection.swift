import SwiftUI
import KurvenCore
import KurvenDynamics

/// "Curves on *face*": the layers on the picked face, each with its own
/// controls and a way to take it off, and an Add menu of the curves the
/// face's fields admit (`Document.paintChoices(on:)`); on a round torus,
/// "Function…", which paints a function whose |f| and arg f can then be
/// contoured. Every edit is one undo step.
struct PaintSection: View {
    @Bindable var document: Document
    let pick: FacePick
    @State private var askingFunction = false

    var body: some View {
        Text(document.paintCaption(pick))
            .font(.caption).foregroundStyle(.secondary)
        let indices = document.paintedLayers(on: pick)
        if indices.isEmpty {
            Text("Nothing is drawn on it yet.").font(.caption).foregroundStyle(.tertiary)
        }
        ForEach(indices, id: \.self) { index in
            if let spec = document.edited?.layers[safe: index] {
                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Text(spec.name)
                        Spacer()
                        Text(document.layers.first { $0.spec.name == spec.name }.map { "\($0.paths.count)" } ?? "…")
                            .monospacedDigit().foregroundStyle(.secondary)
                        Button {
                            do { try document.removeLayer(at: index) } catch { document.paintStatus = "\(error)" }
                        } label: { Image(systemName: "minus.circle") }
                            .buttonStyle(.borderless)
                            .help("Take \(spec.name) off")
                    }
                    LayerControls(document: document, name: spec.name)
                }
            }
        }
        let choices = document.paintChoices(on: pick)
        HStack {
            Menu {
                ForEach(choices) { choice in
                    Button(choice.label) { document.add(choice) }
                }
            } label: { Label("Add", systemImage: "plus") }
                .disabled(choices.isEmpty)
                .help(document.isCurrent ? "Curves the face's fields admit"
                      : "Offered once the drawing on screen is the one asked for: its levels are read off it")
                .fixedSize()
            if pick.face == .surface, document.edited?.request?.takesFunction == true {
                Button("Function…") { askingFunction = true }
                    .help("Paint a function on the torus, over a rectangle of its periods, "
                          + "as `kurven-cli surface periodic` takes one: its |f| and arg f can then be contoured.")
            }
            Spacer()
        }
        .sheet(isPresented: $askingFunction) {
            FunctionSheet(document: document, dismiss: { askingFunction = false })
        }
        if let status = document.paintStatus {
            Text(status).font(.caption).foregroundStyle(.orange).lineLimit(3)
        }
    }
}

/// An expression and the rectangle of its periods, painted on the torus.
struct FunctionSheet: View {
    let document: Document
    let dismiss: () -> Void
    @State private var expression = ""
    @State private var real: Double
    @State private var imag: Double
    @State private var refusal: String?

    init(document: Document, dismiss: @escaping () -> Void) {
        self.document = document
        self.dismiss = dismiss
        var expression = ""
        if case .periodic(let e, _, _, _, _)? = document.edited?.request?.shape { expression = e }
        let periods = document.functionPeriods(for: expression)
        _expression = State(initialValue: expression)
        _real = State(initialValue: periods.x)
        _imag = State(initialValue: periods.y)
    }

    var body: some View {
        Form {
            TextField("f(z)", text: $expression, prompt: Text("sn(z, 0.64)"))
                .font(.body.monospaced())
                .onChange(of: expression) {
                    // A function whose periods the catalog knows brings its
                    // rectangle with it.
                    if let periods = SurfaceRequest.periods(of: expression) { (real, imag) = (periods.x, periods.y) }
                }
            LabeledContent("Periods") {
                HStack {
                    TextField("", value: $real, format: .number.precision(.significantDigits(1...17)))
                        .monospacedDigit().frame(width: 140)
                    Text("×").foregroundStyle(.secondary)
                    TextField("", value: $imag, format: .number.precision(.significantDigits(1...17)))
                        .monospacedDigit().frame(width: 140)
                    Text("i").foregroundStyle(.secondary)
                }
            }
            Text("f is drawn over [0, real) × [0, imag) of the plane, which the torus glues "
                 + "into itself: for the drawing to close, they are periods of f.")
                .font(.caption).foregroundStyle(.secondary)
            if let refusal {
                Text(refusal).font(.caption).foregroundStyle(.red).lineLimit(4)
            }
            HStack {
                Spacer()
                Button("Cancel", action: dismiss).keyboardShortcut(.cancelAction)
                Button("Paint") {
                    do {
                        try document.addFunction(expression, periods: SIMD2(real, imag))
                        dismiss()
                    } catch {
                        refusal = "\(error)"
                    }
                }
                .keyboardShortcut(.defaultAction)
            }
        }
        .formStyle(.grouped)
        .frame(width: 440)
    }
}

/// A layer's level slider, as the window's slider calls it and a headless
/// drag does: taken hold of, moved, let go. A drag is one undo step, taken
/// when it is let go; while it lasts the curves are the grid's, and placed on
/// f when it ends.
@MainActor
struct LevelSlider {
    let document: Document
    let name: String

    func editingChanged(_ editing: Bool) {
        if editing {
            document.beginDrag()
        } else {
            document.rebuildCurves()
            document.endDrag("Levels of \(name)")
        }
    }

    func set(_ value: Double) {
        document.paintLevels(Int(value.rounded()), ofLayer: name, draft: document.editedBeforeDrag != nil)
    }
}

/// A hatching's strokes slider, as the window's slider calls it and a
/// headless drag does.
@MainActor
struct StrokeSlider {
    let document: Document
    let name: String

    func editingChanged(_ editing: Bool) {
        if editing { document.beginDrag() } else { document.endDrag("Strokes of \(name)") }
    }

    func set(_ value: Double) {
        document.paintStrokes(Int(value.rounded()), ofLayer: name)
    }
}

/// A layer's own controls: its levels, its strokes, which way the plateaus'
/// hatching runs. A drag is one undo step, taken when the slider is let go.
struct LayerControls: View {
    @Bindable var document: Document
    let name: String

    var body: some View {
        // A described layer carries its question, so its level set can be
        // moved. A dumped one carries only the answer, and there is nothing
        // here to drag -- which is the difference between a bundle exported
        // with --derived and one without, made visible.
        if let index = document.layerSpecs.firstIndex(where: { $0.name == name }),
           document.isDerived(layer: index), let levels = document.levels(forLayer: index) {
            levelSlider(levels)
        }
        // A hatching carries its spacing the way a contour family carries its
        // levels, and for the same reason: it is a description, so the number
        // in it is a control.
        if let spec = document.edited?.layers.first(where: { $0.name == name }),
           let index = document.edited?.layers.firstIndex(of: spec) {
            if let count = document.hatchCount(ofLayer: index) { strokeSlider(count) }
            if let axis = spec.source.plateauHatchAxis { axisPicker(axis) }
        }
    }

    @ViewBuilder
    private func levelSlider(_ levels: [Double]) -> some View {
        let slider = LevelSlider(document: document, name: name)
        HStack(spacing: 6) {
            Text("levels").font(.caption).foregroundStyle(.secondary)
            Slider(value: Binding(get: { Double(levels.count) }, set: { slider.set($0) }), in: 2...80, step: 1,
                   onEditingChanged: { slider.editingChanged($0) })
            Text("\(levels.count)")
                .font(.caption).monospacedDigit().frame(width: 24, alignment: .trailing)
            Button {
                document.paintResetLevels(ofLayer: name)
            } label: { Image(systemName: "arrow.uturn.backward") }
                .buttonStyle(.borderless).controlSize(.small)
        }
        .padding(.leading, 20)
    }

    /// Hatch strokes along the domain's longer side, in whole steps: each face
    /// divides itself evenly, so a hatching is a count, and a drag steps through
    /// counts rather than sliding a hatching past the edges. Restyling, not
    /// resampling: the strokes are re-derived from grids that are already
    /// here, so this redraws within a frame however large the landscape is.
    @ViewBuilder
    private func strokeSlider(_ count: Int) -> some View {
        let slider = StrokeSlider(document: document, name: name)
        HStack(spacing: 6) {
            Text("strokes").font(.caption).foregroundStyle(.secondary)
            Slider(value: Binding(get: { Double(count) }, set: { slider.set($0) }), in: 2...400, step: 1,
                   onEditingChanged: { slider.editingChanged($0) })
            Text("\(count)")
                .font(.caption).monospacedDigit().frame(width: 28, alignment: .trailing)
        }
        .padding(.leading, 20)
    }

    /// Which way the strokes on the plateaus run. The plates rule them
    /// along the real axis; a landscape whose plateaus are long in the other
    /// direction reads better ruled the other way.
    @ViewBuilder
    private func axisPicker(_ axis: DomainAxis) -> some View {
        Picker("", selection: Binding(get: { axis }, set: { document.paintAxis($0) })) {
            Text("along Re").tag(DomainAxis.real)
            Text("along Im").tag(DomainAxis.imag)
        }
        .pickerStyle(.segmented).labelsHidden().controlSize(.small)
        .padding(.leading, 20)
    }
}
