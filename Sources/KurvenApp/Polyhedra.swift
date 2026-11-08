import Foundation
import Observation
import KurvenCore

/// The half of a document that is a polyhedron, opened from its bundle
/// (`PolyhedronBundle`): its vertices and loops, built into a solid once, and
/// its edges and hatching drawn on it.
///
/// The solid is the bundle's, so what the controls edit is its layers: a
/// hatching's spacing, from the stroke slider a landscape's hatching has. A
/// restyle is laid off the main actor and coalesced as a landscape's -- one
/// in flight, the latest edit queued behind it -- and lays the hatching that
/// changed and nothing else: the solid, its pieces and every other layer are
/// kept as drawn (`PolyhedronBundle.restyled(_:)`), and the renderer keeps
/// the solid's mesh. A restyle a polyhedron cannot take, more hatch lines
/// than it holds, is refused in the status line and the slider goes back.
@MainActor
extension Document {
    /// The polyhedron's bundle, when the document is one.
    var polyhedronBundle: PolyhedronBundle? {
        if case .ready(.polyhedron(let b)) = state { return b }
        return nil
    }

    /// The layers the controls show: the latest asked for, else the bundle's.
    var polyhedronLayers: [LayerSpec]? { polyhedronStyled ?? polyhedronBundle?.manifest.layers }

    /// Edit the layers, and lay what changed from the solid already built.
    func restylePolyhedron(_ change: (inout [LayerSpec]) -> Void) {
        guard var layers = polyhedronLayers else { return }
        let before = layers
        change(&layers)
        guard layers != before else { return }
        polyhedronStyled = layers
        redrawPolyhedron()
    }

    /// The one job that lays a polyhedron's layers off the main actor: the
    /// layers last asked for, when they are not the ones on screen.
    func redrawPolyhedron() {
        guard !drawing, let bundle = polyhedronBundle, let layers = polyhedronStyled else { return }
        guard layers != bundle.manifest.layers else {
            polyhedronStyled = nil
            return
        }
        drawing = true
        let asked = version
        Task {
            let result = await Task.detached(priority: .userInitiated) {
                Result { try bundle.restyled(layers) }
            }.value
            drawing = false
            if version == asked {
                switch result {
                case .success(let restyled):
                    adopt(restyled, keepingCamera: true)
                    polyhedronStatus = nil
                case .failure(let error):
                    polyhedronStatus = "\(error)"
                    if polyhedronStyled == layers { polyhedronStyled = nil }
                }
            }
            redrawPolyhedron()
        }
    }
}
