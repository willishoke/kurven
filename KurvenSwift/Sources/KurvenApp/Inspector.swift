import SwiftUI
import KurvenCore
import KurvenService
import KurvenLandscape
import KurvenDynamics

/// The sidebar: what is drawn, from where, and how it bakes.
///
/// Every control writes one field of the `Document` and the picture follows.
/// Nothing here computes anything about the landscape -- the numbers it shows
/// are the model's, and the numbers it sets are the model's -- so there is no
/// second copy of the camera to keep in step with the first.
struct Inspector: View {
    @Bindable var document: Document

    var body: some View {
        Form {
            // The function comes first, and shows even with nothing open: a
            // landscape is now something you can start, not only something you
            // can be handed.
            Section("Landscape") {
                LandscapeSection(document: document)
            }
            if document.plate != nil || document.building {
                Section("Surface") { SurfaceStatus(document: document) }
            }
            if document.plate != nil {
                // A surface is not a bundle: no truncation, margin, resample
                // or provenance. Its camera has the one plate projection it
                // opened at, so "reset" still has somewhere to go.
                Section("Camera") {
                    CameraSection(document: document, presets: [SurfaceRequest.preset])
                }
                Section("Mode") { modePicker }
                Section("Layers") { layerList }
                Section("Bake") { BakeSection(document: document) }
            } else if let bundle = document.bundle {
                Section("Camera") {
                    CameraSection(document: document, presets: bundle.manifest.presets)
                }
                Section("Mode") { modePicker }
                Section("Truncation") {
                    TruncationSection(document: document)
                }
                Section("Layers") { layerList }
                Section("Ink") { marginField }
                Section("Resample") { resamplePanel }
                Section("Bake") { BakeSection(document: document) }
                Section("Bundle") { provenance(bundle) }
            } else {
                Text("Open a .kurven bundle.")
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .frame(minWidth: 280)
    }

    // MARK: - mode

    @ViewBuilder
    private var modePicker: some View {
        Picker("Preview", selection: Binding(
            get: { ModeChoice(document.mode) },
            set: { document.mode = $0.mode })) {
            Text("Plate").tag(ModeChoice.plate)
            Text("Shaded").tag(ModeChoice.shaded)
            Text("Depth").tag(ModeChoice.depth)
        }
        .pickerStyle(.segmented)
    }

    // MARK: - layers

    @ViewBuilder
    private var layerList: some View {
        ForEach(Array(document.layers.enumerated()), id: \.offset) { index, layer in
            VStack(alignment: .leading, spacing: 2) {
                Toggle(isOn: Binding(
                    get: { !document.hiddenLayers.contains(index) },
                    set: { _ in document.toggle(layer: index) })) {
                    HStack {
                        Text(layer.spec.name)
                        Spacer()
                        Text("\(layer.paths.count)")
                            .monospacedDigit().foregroundStyle(.secondary)
                        Text(layer.spec.clipped ? "" : "unclipped")
                            .font(.caption).foregroundStyle(.tertiary)
                    }
                }
                // A described layer carries its question, so its level set can
                // be moved. A dumped one carries only the answer, and there is
                // nothing here to drag -- which is the difference between a
                // bundle exported with --derived and one without, made visible.
                if document.isDerived(layer: index),
                   let levels = document.levels(forLayer: index) {
                    levelSlider(index: index, levels: levels)
                }
                // A hatching carries its spacing the way a contour family
                // carries its levels, and for the same reason: it is a
                // description, so the number in it is a control.
                if let spacing = document.spacing(ofLayer: index) {
                    spacingSlider(index: index, spacing: spacing)
                }
                if case .capHatch(let axis, _, _) = layer.spec.source {
                    axisPicker(axis)
                }
            }
        }
    }

    @ViewBuilder
    private func levelSlider(index: Int, levels: [Double]) -> some View {
        let count = Binding(
            get: { Double(levels.count) },
            set: { document.setLevelCount(Int($0.rounded()), forLayer: index) })
        HStack(spacing: 6) {
            Text("levels").font(.caption).foregroundStyle(.secondary)
            Slider(value: count, in: 1...80, step: 1)
            Text("\(levels.count)")
                .font(.caption).monospacedDigit().frame(width: 24, alignment: .trailing)
            Button {
                document.resetLevels(forLayer: index)
            } label: { Image(systemName: "arrow.uturn.backward") }
                .buttonStyle(.borderless).controlSize(.small)
        }
        .padding(.leading, 20)
    }

    /// World units between hatch strokes. Restyling, not resampling: the
    /// strokes are re-derived from grids that are already here, so this redraws
    /// within a frame however large the landscape is.
    @ViewBuilder
    private func spacingSlider(index: Int, spacing: Double) -> some View {
        let live = Binding(get: { spacing }, set: {
            document.setSpacing($0, ofLayer: index)
        })
        HStack(spacing: 6) {
            Text("spacing").font(.caption).foregroundStyle(.secondary)
            Slider(value: live, in: max(spacing / 20, 0.002)...max(spacing * 8, 0.05))
            Text(String(format: "%.3g", spacing))
                .font(.caption).monospacedDigit().frame(width: 34, alignment: .trailing)
        }
        .padding(.leading, 20)
    }

    /// Which way the strokes on the truncated tops run. The plates rule them
    /// along the real axis; a landscape whose plateaus are long in the other
    /// direction reads better ruled the other way.
    @ViewBuilder
    private func axisPicker(_ axis: KeepAxis) -> some View {
        Picker("", selection: Binding(get: { axis }, set: {
            document.setCapHatchAxis($0)
        })) {
            Text("along Re").tag(KeepAxis.real)
            Text("along Im").tag(KeepAxis.imag)
        }
        .pickerStyle(.segmented).labelsHidden().controlSize(.small)
        .padding(.leading, 20)
    }

    // MARK: - ink

    @ViewBuilder
    private var marginField: some View {
        let live = Binding(get: { document.margin }, set: {
            document.margin = $0
        })
        LabeledContent("Margin") {
            HStack {
                TextField("", value: live, format: .number.precision(.fractionLength(3)))
                    .labelsHidden().monospacedDigit().frame(width: 70)
                Slider(value: live, in: 0...0.5)
            }
        }
    }

    // MARK: - resample

    /// The only thing a frozen bundle cannot do for itself.
    ///
    /// The form is built from what the service says the example accepts, not
    /// from a list written here, so an option added to a Python example appears
    /// in this window without anyone editing Swift.
    @ViewBuilder
    private var resamplePanel: some View {
        if let example = document.example {
            ForEach(example.arguments) { spec in
                argumentField(spec)
            }
            HStack {
                Button(document.resampling ? "Resampling…" : "Rebuild") {
                    document.resample()
                }
                .disabled(document.resampling)
                Spacer()
            }
        } else if document.serviceDescription != nil {
            Text("This bundle does not record which example made it, so there "
                 + "is nothing to ask the service for.")
                .font(.caption).foregroundStyle(.secondary)
        } else {
            Text("Looking for the Python service…")
                .font(.caption).foregroundStyle(.secondary)
        }
        if let status = document.serviceStatus {
            Text(status).font(.caption).foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
    }

    @ViewBuilder
    private func argumentField(_ spec: ArgumentSpec) -> some View {
        let value = Binding(
            get: { document.arguments[spec.name] ?? spec.defaultText ?? "" },
            set: { document.arguments[spec.name] = $0 })
        LabeledContent(spec.name.replacingOccurrences(of: "_", with: " ")) {
            switch spec.kind {
            case .flag:
                Toggle("", isOn: Binding(get: { value.wrappedValue == "true" },
                                         set: { value.wrappedValue = $0 ? "true" : "false" }))
                    .labelsHidden()
            default:
                TextField("", text: value)
                    .labelsHidden().monospacedDigit().frame(width: 110)
            }
        }
        .help(spec.help)
    }

    // MARK: - provenance

    @ViewBuilder
    private func provenance(_ bundle: KurvenBundle) -> some View {
        let m = bundle.manifest
        LabeledContent("Function", value: m.provenance.function)
        LabeledContent("Grid", value: "\(m.height.shape.nx) × \(m.height.shape.ny)")
        LabeledContent("Occluder", value: "step \(m.occluder.step), "
                       + "\(m.occluder.tiles.count) tile\(m.occluder.tiles.count == 1 ? "" : "s")")
        if !m.provenance.isReproducible {
            // Contoured with more than one chunk means the seams were stitched
            // in thread-completion order. Worth saying before anyone blames a
            // difference on anything else.
            Label("contoured with \(m.provenance.cpuCount) chunks — not reproducible",
                  systemImage: "exclamationmark.triangle")
                .font(.caption).foregroundStyle(.orange)
        }
    }
}

/// What the surface on screen is, and how building it went. Its own view,
/// so the status a build writes does not re-evaluate the whole sidebar.
struct SurfaceStatus: View {
    let document: Document

    var body: some View {
        if let text = document.surfaceStatus {
            HStack(spacing: 6) {
                if document.building { ProgressView().controlSize(.small) }
                Text(text).font(.caption).foregroundStyle(.secondary).lineLimit(3)
            }
        }
    }
}

/// The camera: presets, the projection, and the orbit as numbers.
///
/// Its own view, and so its own observation scope, because it is the part of
/// the sidebar that reads `navigator` -- which every drag writes. Inline, that
/// read made the whole sidebar (layers, sliders, the resample form) re-evaluate
/// on every mouse event to redraw four numbers.
struct CameraSection: View {
    @Bindable var document: Document
    let presets: [CameraPreset]


    var body: some View {
        presetRow
        if document.navigator != nil { orbitFields }
    }

    @ViewBuilder
    private var presetRow: some View {
        HStack {
            projectionToggle
            Spacer()
        }
        HStack {
            ForEach(presets, id: \.name) { preset in
                Button(preset.name) {
                    document.use(preset: preset)
                }
            }
            Spacer()
            Button("Fit") { document.fit() }
                .keyboardShortcut("f", modifiers: [])
        }
    }

    /// Perspective is a way of navigating, not a plate style: the bake refuses
    /// it, and says so. Offering it next to the presets is the honest placement
    /// -- it belongs with "where am I looking from", not with "what kind of
    /// drawing is this".
    @ViewBuilder
    private var projectionToggle: some View {
        Toggle("Perspective", isOn: Binding(
            get: { document.navigator?.orbit.isPerspective ?? false },
            set: { document.setPerspective($0) }))
            .toggleStyle(.switch)
            .controlSize(.small)
            .help("Navigate in perspective. Baking still requires an "
                  + "orthographic camera — the plates are orthographic, and a "
                  + "perspective bake would have no Python oracle to check it "
                  + "against.")
    }

    @ViewBuilder
    private var orbitFields: some View {
        // Degrees, because that is what the plate parameters are stated in and
        // every conversion is a chance to lose a factor of pi.
        degreeField("Azimuth", value: Binding(
            get: { document.navigator?.orbit.azimuth.degrees ?? 0 },
            set: { document.navigator?.orbit.azimuth = Angle(degrees: $0) }))
        degreeField("Elevation", value: Binding(
            get: { document.navigator?.orbit.elevation.degrees ?? 0 },
            set: {
                let limit = Orbit.elevationLimit.degrees
                document.navigator?.orbit.elevation =
                    Angle(degrees: min(max($0, -limit), limit))
            }))
        LabeledContent("Scale") {
            Text(String(format: "%.5g units/px", document.navigator?.framing.unitsPerPixel ?? 0))
                .monospacedDigit().foregroundStyle(.secondary)
        }
        LabeledContent("Target") {
            Text(target(document.navigator?.orbit.target))
                .monospacedDigit().foregroundStyle(.secondary)
        }
    }

    private func target(_ p: P3<WorldSpace>?) -> String {
        guard let p else { return "-" }
        return String(format: "%.3g, %.3g, %.3g", p.x, p.y, p.z)
    }

    @ViewBuilder
    private func degreeField(_ label: String, value: Binding<Double>) -> some View {
        // One binding for both controls, so the field and the slider cannot
        // disagree about what the camera is.
        let live = Binding(get: { value.wrappedValue }, set: {
            value.wrappedValue = $0
        })
        LabeledContent(label) {
            HStack {
                TextField("", value: live, format: .number.precision(.fractionLength(1)))
                    .labelsHidden()
                    .monospacedDigit()
                    .frame(width: 70)
                Text("°").foregroundStyle(.secondary)
                Slider(value: live, in: -180...180)
            }
        }
    }
}

/// Baking, which reads `navigator` too (a perspective camera cannot bake), so
/// it is scoped apart for the same reason as `CameraSection`.
struct BakeSection: View {
    @Bindable var document: Document


    var body: some View {
        LabeledContent("Resolution") {
            TextField("", value: $document.bakeResolution, format: .number)
                .labelsHidden().monospacedDigit().frame(width: 90)
        }
        HStack {
            Button(document.baking ? "Baking…" : "Bake to SVG…") { runBake() }
                .disabled(document.baking || !document.canBake)
            Spacer()
        }
        if let status = document.bakeStatus {
            Text(status).font(.caption).foregroundStyle(.secondary)
        }
    }

    private func runBake() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.svg]
        panel.nameFieldStringValue = document.title + ".svg"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        document.bake(to: url)
    }
}

/// `PreviewMode` carries a `Lighting` payload, which a `Picker` cannot select
/// on. This is the selectable shadow of it.
enum ModeChoice: Hashable {
    case plate, shaded, depth

    init(_ mode: PreviewMode) {
        switch mode {
        case .plate: self = .plate
        case .shaded: self = .shaded
        case .depth: self = .depth
        }
    }

    var mode: PreviewMode {
        switch self {
        case .plate: .plate
        case .shaded: .shaded(Lighting())
        case .depth: .depth
        }
    }
}
