import SwiftUI
import KurvenCore

/// The sidebar: what is drawn, from where, and how it bakes.
///
/// Every control writes one field of the `Document` and the picture follows.
/// Nothing here computes anything about the landscape -- the numbers it shows
/// are the model's, and the numbers it sets are the model's -- so there is no
/// second copy of the camera to keep in step with the first.
struct Inspector: View {
    @Bindable var document: Document
    var onChange: () -> Void

    var body: some View {
        Form {
            if let bundle = document.bundle {
                Section("Camera") {
                    presets(bundle.manifest.presets)
                    if document.navigator != nil { orbitFields }
                }
                Section("Mode") { modePicker }
                Section("Layers") { layerList }
                Section("Ink") { marginField }
                Section("Bake") { bakePanel }
                Section("Bundle") { provenance(bundle) }
            } else {
                Text("Open a .kurven bundle.")
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .frame(minWidth: 280)
    }

    // MARK: - camera

    @ViewBuilder
    private func presets(_ presets: [CameraPreset]) -> some View {
        HStack {
            ForEach(presets, id: \.name) { preset in
                Button(preset.name) {
                    document.use(preset: preset)
                    onChange()
                }
            }
            Spacer()
            Button("Fit") { document.fit(); onChange() }
                .keyboardShortcut("f", modifiers: [])
        }
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
            document.refreshScene()
            onChange()
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

    // MARK: - mode

    @ViewBuilder
    private var modePicker: some View {
        Picker("Preview", selection: Binding(
            get: { ModeChoice(document.mode) },
            set: { document.mode = $0.mode; onChange() })) {
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
            Toggle(isOn: Binding(
                get: { !document.hiddenLayers.contains(index) },
                set: { _ in document.toggle(layer: index); onChange() })) {
                HStack {
                    Text(layer.spec.name)
                    Spacer()
                    Text("\(layer.paths.count)")
                        .monospacedDigit().foregroundStyle(.secondary)
                    Text(layer.spec.clipped ? "" : "unclipped")
                        .font(.caption).foregroundStyle(.tertiary)
                }
            }
        }
    }

    // MARK: - ink

    @ViewBuilder
    private var marginField: some View {
        let live = Binding(get: { document.margin }, set: {
            document.margin = $0
            document.refreshScene()
            onChange()
        })
        LabeledContent("Margin") {
            HStack {
                TextField("", value: live, format: .number.precision(.fractionLength(3)))
                    .labelsHidden().monospacedDigit().frame(width: 70)
                Slider(value: live, in: 0...0.5)
            }
        }
    }

    // MARK: - bake

    @ViewBuilder
    private var bakePanel: some View {
        LabeledContent("Resolution") {
            TextField("", value: $document.bakeResolution, format: .number)
                .labelsHidden().monospacedDigit().frame(width: 90)
        }
        HStack {
            Button(document.baking ? "Baking…" : "Bake to SVG…") { runBake() }
                .disabled(document.baking)
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
