import SwiftUI
import KurvenCore
import KurvenService

/// The catalog as pictures: pick a landscape by looking at it.
///
/// The sidebar's dropdown is the same list and will always be faster for
/// someone who knows what they want. This is for everyone else, and for the
/// question a name cannot answer -- what does ψ look like next to ζ -- which is
/// the whole reason these plates were drawn in 1909 rather than tabulated.
///
/// Each cell is the landscape it names, drawn by the renderer that will draw it
/// full size (`Thumbnails`). Nothing here is an illustration of the app.
struct Gallery: View {
    let presets: [FunctionPreset]
    let thumbnails: Thumbnails
    var service: Service?
    var choose: (FunctionPreset) -> Void
    var dismiss: (() -> Void)?

    @State private var search = ""
    @State private var hovered: String?

    private var shown: [FunctionPreset] {
        let text = search.trimmingCharacters(in: .whitespaces).lowercased()
        guard !text.isEmpty else { return presets }
        return presets.filter {
            $0.name.lowercased().contains(text) || $0.label.lowercased().contains(text)
                || $0.expression.lowercased().contains(text)
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
                    ForEach(shown) { preset in cell(preset) }
                }
                .padding(20)
            }
        }
        .frame(minWidth: 560, minHeight: 420)
        // Only when the picker is actually looked at: fourteen landscapes is a
        // second or two of the service, and a window that opens a bundle never
        // needs any of them.
        .task { thumbnails.warm(presets, service: service) }
    }

    @ViewBuilder
    private var header: some View {
        HStack(spacing: 10) {
            Text("Choose a function").font(.headline)
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
    private func cell(_ preset: FunctionPreset) -> some View {
        let isHovered = hovered == preset.name
        VStack(alignment: .leading, spacing: 6) {
            ZStack {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color(nsColor: .textBackgroundColor))
                if let image = thumbnails.images[preset.name] {
                    Image(nsImage: image)
                        .resizable().scaledToFit().padding(4)
                } else if thumbnails.isFailed(preset) {
                    Label("could not sample", systemImage: "exclamationmark.triangle")
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
            Text(preset.label).lineLimit(1)
            Text(preset.expression)
                .font(.caption).monospaced().foregroundStyle(.secondary).lineLimit(1)
        }
        .padding(6)
        .background(RoundedRectangle(cornerRadius: 10)
            .fill(isHovered ? Color.accentColor.opacity(0.10) : .clear))
        .contentShape(Rectangle())
        .onHover { hovered = $0 ? preset.name : (hovered == preset.name ? nil : hovered) }
        .onTapGesture {
            choose(preset)
            dismiss?()
        }
        .help(preset.notes)
        .accessibilityLabel("\(preset.label). \(preset.notes)")
    }
}
