import SwiftUI
import KurvenCore
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
            // A surface has its own section and a way back to the gallery;
            // the function picker would only say "—" beside it.
            if document.surfaceDrawing == nil && document.polyhedronBundle == nil && !document.building {
                Section("Landscape") {
                    LandscapeSection(document: document)
                }
            }
            if document.surfaceDrawing != nil || document.building {
                Section("Surface") { SurfaceSection(document: document) }
            }
            if let polyhedronBundle = document.polyhedronBundle {
                // A polyhedron's solid is its bundle's, and its visibility
                // exact, so it has no controls of its own but its layers'.
                Section("Polyhedron") { polyhedron(polyhedronBundle) }
                Section("Camera") {
                    CameraSection(document: document, presets: polyhedronBundle.manifest.presets)
                }
                Section("Mode") { modePicker }
                paintSection
                Section("Layers") { layerList }
                Section("Bake") { BakeSection(document: document) }
            } else if let surfaceBundle = document.surfaceBundle {
                // A surface has no truncation, and its visibility is exact
                // or read off its coordinates, so it has no depth tolerance
                // to set. Its camera has the presets its bundle keeps.
                Section("Camera") {
                    CameraSection(document: document, presets: surfaceBundle.manifest.presets)
                }
                Section("Mode") { modePicker }
                paintSection
                Section("Layers") { layerList }
                Section("Bake") { BakeSection(document: document) }
                Section("Bundle") { provenance(surfaceBundle) }
            } else if let bundle = document.bundle {
                Section("Camera") {
                    CameraSection(document: document, presets: bundle.manifest.presets)
                }
                Section("Mode") { modePicker }
                Section("Truncation") {
                    TruncationSection(document: document)
                }
                paintSection
                Section("Layers") { layerList }
                Section("Visibility") { depthToleranceField }
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
            Text("Drawing").tag(ModeChoice.drawing)
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
                LayerControls(document: document, name: layer.spec.name)
            }
        }
    }

    // MARK: - painting

    /// The curves on the picked face, when one is picked.
    @ViewBuilder
    private var paintSection: some View {
        if let pick = document.picked {
            Section("Curves on \(document.paintTitle(pick))") { PaintSection(document: document, pick: pick) }
        }
    }

    // MARK: - curves

    @ViewBuilder
    private var depthToleranceField: some View {
        let live = Binding(get: { document.depthTolerance }, set: {
            document.depthTolerance = $0
        })
        LabeledContent("Depth tolerance") {
            HStack {
                TextField("", value: live, format: .number.precision(.fractionLength(3)))
                    .labelsHidden().monospacedDigit().frame(width: 70)
                Slider(value: live, in: 0...0.5)
            }
        }
    }

    // MARK: - provenance

    @ViewBuilder
    private func polyhedron(_ bundle: PolyhedronBundle) -> some View {
        let solid = bundle.solid
        let edges = solid.isDiagonal.filter { !$0 }.count
        LabeledContent("Solid", value: bundle.manifest.provenance.function)
        LabeledContent("Boundary", value: "\(solid.vertices.count) vertices, \(edges) edges, \(solid.faces.count) faces")
        if let genus = solid.genus {
            LabeledContent("Genus", value: "\(genus)" + (solid.isConvex ? ", convex" : ", not convex"))
        } else {
            LabeledContent("Shells", value: "\(solid.shellCount), χ = \(solid.eulerCharacteristic)")
        }
        if let status = document.polyhedronStatus {
            Text(status).font(.caption).foregroundStyle(.secondary).lineLimit(3)
        }
    }

    @ViewBuilder
    private func provenance(_ bundle: SurfaceBundle) -> some View {
        let request = bundle.manifest.surface
        LabeledContent("Function", value: bundle.manifest.provenance.function)
        LabeledContent("Grid", value: "\(request.grid) around")
    }

    @ViewBuilder
    private func provenance(_ bundle: KurvenBundle) -> some View {
        let m = bundle.manifest
        LabeledContent("Function", value: m.provenance.function)
        LabeledContent("Grid", value: "\(m.height.shape.nx) × \(m.height.shape.ny)")
        LabeledContent("Occluder", value: "step \(m.occluder.step), "
                       + "\(m.occluder.tiles.count) tile\(m.occluder.tiles.count == 1 ? "" : "s")")
        if !m.provenance.isReproducible {
            // Contoured with more than one chunk means the chunk boundaries were stitched
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
/// read made the whole sidebar (layers, sliders, the landscape controls) re-evaluate
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

    /// Perspective is a way of navigating, not a drawing convention: the bake refuses
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
                  + "perspective bake would have nothing to check it "
                  + "against.")
    }

    @ViewBuilder
    private var orbitFields: some View {
        // Degrees, because that is what the drawing parameters are stated in and
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
    case drawing, shaded, depth

    init(_ mode: PreviewMode) {
        switch mode {
        case .drawing: self = .drawing
        case .shaded: self = .shaded
        case .depth: self = .depth
        }
    }

    var mode: PreviewMode {
        switch self {
        case .drawing: .drawing
        case .shaded: .shaded(Lighting())
        case .depth: .depth
        }
    }
}
