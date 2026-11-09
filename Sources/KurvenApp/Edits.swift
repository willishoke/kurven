import Foundation
import KurvenCore
import KurvenLandscape
import KurvenDynamics

/// What an edit changes and a save writes: the layers, a parametric
/// surface's request, which draws some of them and carries the function, a
/// landscape's request and truncation, the level counts the sliders set, and
/// the git sha the bundle records. One value, so every edit -- painting, a
/// truncation, a surface's numbers, a landscape's window -- is one step to
/// undo, the steps unwind in order, and undoing one puts back exactly what a
/// save wrote before it.
struct Edited: Equatable {
    var layers: [LayerSpec]
    var request: SurfaceRequest?
    /// A landscape's request: its function, window and resolution. Nil for a
    /// landscape opened read-only, and for every other kind of bundle.
    var landscape: LandscapeRequest?
    var truncation: Truncation?
    var levelCounts: [String: Int]
    /// The opened bundle's, until a request edit makes the drawing again in
    /// this process, which records its own.
    var gitSha: String
}

/// The git sha a drawing made again in this process records.
let rebuiltGitSha = "native"

@MainActor
extension Document {
    // MARK: - the state

    /// What is edited now: the layers last asked for, which a restyle in
    /// flight is still drawing, and the requests the controls say.
    var edited: Edited? {
        switch state {
        case .ready(.landscape(let bundle)):
            // Sampled again when the request asks for other samples than the
            // bundle's, and recorded then as this process's.
            let resampled = landscape.map { !(shown?.samples(as: $0) ?? true) } ?? false
            return Edited(layers: styling?.layers ?? [], request: nil, landscape: landscape,
                          truncation: styling?.truncation, levelCounts: levelCounts,
                          gitSha: resampled ? landscapeGitSha ?? rebuiltGitSha : bundle.manifest.provenance.gitSha)
        case .ready(.surface(let bundle)):
            let request = surface ?? bundle.manifest.surface
            return Edited(layers: surfaceStyled ?? bundle.manifest.layers, request: request, landscape: nil,
                          truncation: nil, levelCounts: levelCounts,
                          gitSha: request == bundle.manifest.surface ? bundle.manifest.provenance.gitSha
                              : surfaceGitSha ?? rebuiltGitSha)
        case .ready(.polyhedron(let bundle)):
            return Edited(layers: polyhedronLayers ?? [], request: nil, landscape: nil, truncation: nil,
                          levelCounts: levelCounts, gitSha: bundle.manifest.provenance.gitSha)
        case .empty, .loading, .failed:
            return nil
        }
    }

    /// Make `next` what is edited: the layers restyled from what is built, a
    /// request edit built or sampled as the controls build one, the levels
    /// drawn again. A surface's own curves follow its request whatever order
    /// the builds land in (`SurfaceDrawing.keeping(_:from:)`).
    func setEdited(_ next: Edited) {
        guard let current = edited, next != current else { return }
        let levelsMoved = next.levelCounts != current.levelCounts
        levelCounts = next.levelCounts
        switch state {
        case .ready(.landscape):
            if next.landscape != current.landscape { landscape = next.landscape }
            if next.layers != current.layers || next.truncation != current.truncation {
                restyle {
                    $0.layers = next.layers
                    if let truncation = next.truncation { $0.truncation = truncation }
                }
            }
            if let request = next.landscape, let before = current.landscape, !request.samples(as: before) {
                landscapeGitSha = next.gitSha
                landscapeEdited(draft: false)
            }
        case .ready(.surface):
            if let request = next.request, request != current.request {
                surfaceStyled = next.layers
                surfaceGitSha = next.gitSha
                surface = request
                surfaceEdited(draft: false)
            } else if next.layers != current.layers {
                restyleSurface(next.layers)
            }
        case .ready(.polyhedron):
            if next.layers != current.layers { restylePolyhedron { $0 = next.layers } }
        case .empty, .loading, .failed:
            return
        }
        if levelsMoved { rebuildCurves() }
    }

    /// One edit, one undo step, named in the Edit menu. An edit the document
    /// cannot take -- more hatch lines than a polyhedron holds, a layer a
    /// surface cannot draw -- is refused, saying why, and leaves no step.
    func edit(_ name: String, _ change: (inout Edited) throws -> Void) rethrows {
        guard let before = edited else { return }
        var after = before
        try change(&after)
        if after.request != before.request || !Self.samplesAlike(after.landscape, before.landscape) {
            after.gitSha = rebuiltGitSha
        }
        guard after != before else { return }
        if let why = refusal(of: after) {
            paintStatus = why
            return
        }
        registerUndo(from: before, to: after, name)
        setEdited(after)
    }

    private static func samplesAlike(_ a: LandscapeRequest?, _ b: LandscapeRequest?) -> Bool {
        guard let a, let b else { return a == nil && b == nil }
        return a.samples(as: b)
    }

    /// A slider taken hold of: what to undo back to when it is let go.
    func beginDrag() {
        editedBeforeDrag = edited
    }

    /// A slider let go: the whole drag is one undo step, unless nothing it
    /// asked for was taken.
    func endDrag(_ name: String) {
        defer { editedBeforeDrag = nil }
        guard let before = editedBeforeDrag, let after = edited, before != after, refusal(of: after) == nil else { return }
        registerUndo(from: before, to: after, name)
    }

    /// `body` as one undo step: a drag of one step, for an edit made by
    /// calls a drag also makes. Inside a drag, a step of it.
    func editing(_ name: String, _ body: () -> Void) {
        guard editedBeforeDrag == nil else { return body() }
        beginDrag()
        body()
        endDrag(name)
    }

    /// Why the document cannot take `next`, or nil: what its manifest's
    /// reader would refuse.
    func refusal(of next: Edited) -> String? {
        switch state {
        case .ready(.polyhedron(let bundle)):
            return PolyhedronManifest.unsupported(next.layers, on: bundle.solid)
        case .ready(.surface):
            guard let request = next.request else { return nil }
            do { try request.check() } catch { return "\(error)" }
            var names = Set<String>()
            for layer in next.layers {
                if let why = SurfaceManifest.unsupported(layer.source, on: request) { return "layer '\(layer.name)': \(why)" }
                guard names.insert(layer.name).inserted else { return "two layers named '\(layer.name)'" }
            }
            return nil
        case .ready(.landscape), .empty, .loading, .failed:
            return nil
        }
    }

    /// ⌘Z puts `before` back, and ⌘⇧Z `after`.
    private func registerUndo(from before: Edited, to after: Edited, _ name: String) {
        guard let undoManager else { return }
        // A headless run's events open their group as the window's do: at
        // the first registration, so an event that registers nothing leaves
        // no step (`undoGroupOpened`).
        if opensUndoGroups, undoManager.groupingLevel == 0, !undoManager.isUndoing, !undoManager.isRedoing {
            undoManager.beginUndoGrouping()
            undoGroupOpened = true
        }
        undoManager.registerUndo(withTarget: self) { document in
            MainActor.assumeIsolated {
                document.registerUndo(from: after, to: before, name)
                document.setEdited(before)
            }
        }
        undoManager.setActionName(name)
    }

    /// A different document: nothing done to the last one can be undone on
    /// it, and nothing of it is picked.
    func forgetEdits() {
        undoManager?.removeAllActions(withTarget: self)
        editedBeforeDrag = nil
        paintStatus = nil
        picked = nil
        selection = nil
        surfaceGitSha = nil
        landscapeGitSha = nil
    }

    // MARK: - the controls' edits

    /// A surface's request as its controls set it, one undo step.
    func editSurface(_ next: SurfaceRequest, _ name: String) {
        edit(name) { $0.request = next }
    }

    /// A step of a drag of a surface's control: built as a draft, or in full.
    func dragSurface(_ next: SurfaceRequest, draft: Bool) {
        surfaceGitSha = nil
        surface = next
        surfaceEdited(draft: draft)
    }

    /// A landscape's request as its controls set it, one undo step.
    func editLandscape(_ name: String, _ change: (inout LandscapeRequest) -> Void) {
        edit(name) { edited in
            guard var request = edited.landscape else { return }
            change(&request)
            edited.landscape = request
        }
    }

    /// A step of a drag of a landscape's control: sampled at the draft
    /// resolution, or in full.
    func dragLandscape(draft: Bool) {
        landscapeGitSha = nil
        landscapeEdited(draft: draft)
    }

    /// A truncation typed, chosen or stepped: one undo step, with what it
    /// does to the levels and the risers' hatching (`setTruncation(_:)`).
    func editTruncation(_ truncation: Truncation, _ name: String = "Truncation") {
        editing(name) { setTruncation(truncation) }
    }
}
