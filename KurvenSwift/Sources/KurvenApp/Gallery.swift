import SwiftUI
import KurvenCore
import KurvenLandscape
import KurvenDynamics

/// One cell of the gallery: a function to draw as a landscape, or a surface.
enum GalleryEntry: Identifiable, Sendable {
    case function(FunctionPreset)
    case surface(SurfacePreset)

    /// Unique across both kinds: a function and a surface may share a name.
    var id: String {
        switch self {
        case .function(let p): p.name
        case .surface(let p): "surface:" + p.name
        }
    }
    var name: String {
        switch self {
        case .function(let p): p.name
        case .surface(let p): p.name
        }
    }
    var label: String {
        switch self {
        case .function(let p): p.label
        case .surface(let p): p.label
        }
    }
    /// The line under the label: the expression, or the surface's formula.
    var formula: String {
        switch self {
        case .function(let p): p.expression
        case .surface(let p): p.formula
        }
    }
    var notes: String {
        switch self {
        case .function(let p): p.notes
        case .surface(let p): p.notes
        }
    }
}

/// The catalogs as pictures: pick a landscape, or a surface, by looking at it.
///
/// The sidebar's dropdown is the same list and will always be faster for
/// someone who knows what they want. This is for everyone else, and for the
/// question a name cannot answer -- what does ψ look like next to ζ -- which is
/// the whole reason these plates were drawn in 1909 rather than tabulated.
///
/// Each cell is the landscape or surface it names, drawn by the renderer that
/// will draw it full size (`Thumbnails`). Nothing here is an illustration of
/// the app. The surfaces are a group of their own, after the functions: they
/// are a different kind of plate, not three more functions.
struct Gallery: View {
    let functions: [FunctionPreset]
    let surfaces: [SurfacePreset]
    let thumbnails: Thumbnails
    var choose: (GalleryEntry) -> Void
    var dismiss: (() -> Void)?

    @State private var search = ""
    @State private var hovered: String?

    private var entries: [GalleryEntry] {
        functions.map(GalleryEntry.function) + surfaces.map(GalleryEntry.surface)
    }

    private func shown(_ entries: [GalleryEntry]) -> [GalleryEntry] {
        let text = search.trimmingCharacters(in: .whitespaces).lowercased()
        guard !text.isEmpty else { return entries }
        return entries.filter {
            $0.name.lowercased().contains(text) || $0.label.lowercased().contains(text)
                || $0.formula.lowercased().contains(text)
                || $0.notes.lowercased().contains(text)
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 210), spacing: 20)],
                          spacing: 20) {
                    group("Functions", shown(functions.map(GalleryEntry.function)))
                    group("Surfaces", shown(surfaces.map(GalleryEntry.surface)))
                }
                .padding(20)
            }
        }
        .frame(minWidth: 560, minHeight: 420)
        // Only when the picker is actually looked at: fourteen landscapes is a
        // second of sampling, and a window that opens a bundle never needs any
        // of them.
        .task { thumbnails.warm(entries) }
    }

    @ViewBuilder
    private func group(_ title: String, _ entries: [GalleryEntry]) -> some View {
        if !entries.isEmpty {
            Section {
                ForEach(entries) { entry in cell(entry) }
            } header: {
                Text(title).font(.title3.weight(.semibold))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.top, 4)
            }
        }
    }

    @ViewBuilder
    private var header: some View {
        HStack(spacing: 10) {
            Text("Choose a function or a surface").font(.headline)
            if let drawing = thumbnails.drawing {
                ProgressView().controlSize(.small)
                Text("drawing \(drawing)…").font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            HStack(spacing: 4) {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Filter", text: $search)
                    .textFieldStyle(.plain).frame(width: 140)
            }
            if let dismiss {
                Button("Cancel", action: dismiss).keyboardShortcut(.cancelAction)
            }
        }
        .padding(.horizontal, 20).padding(.vertical, 12)
    }

    @ViewBuilder
    private func cell(_ entry: GalleryEntry) -> some View {
        let isHovered = hovered == entry.id
        VStack(alignment: .leading, spacing: 6) {
            ZStack {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color(nsColor: .textBackgroundColor))
                if let image = thumbnails.images[entry.id] {
                    Image(nsImage: image)
                        .resizable().scaledToFit().padding(4)
                } else if thumbnails.isFailed(entry) {
                    Label("could not draw", systemImage: "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    ProgressView().controlSize(.small)
                }
            }
            .aspectRatio(CGSize(width: Thumbnails.size.width, height: Thumbnails.size.height),
                         contentMode: .fit)
            .overlay(RoundedRectangle(cornerRadius: 8)
                .strokeBorder(isHovered ? Color.accentColor : Color.secondary.opacity(0.3),
                              lineWidth: isHovered ? 2 : 1))
            Text(entry.label).lineLimit(1)
            Text(entry.formula)
                .font(.caption).monospaced().foregroundStyle(.secondary).lineLimit(1)
        }
        .padding(6)
        .background(RoundedRectangle(cornerRadius: 10)
            .fill(isHovered ? Color.accentColor.opacity(0.10) : .clear))
        .contentShape(Rectangle())
        .onHover { hovered = $0 ? entry.id : (hovered == entry.id ? nil : hovered) }
        .onTapGesture {
            choose(entry)
            dismiss?()
        }
        .help(entry.notes)
        .accessibilityLabel("\(entry.label). \(entry.notes)")
    }
}
