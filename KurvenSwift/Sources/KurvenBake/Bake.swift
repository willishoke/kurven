import Foundation
import KurvenCore
import KurvenMetal

/// Turning a scene into strokes.
///
/// The realtime preview and the bake are the same computation at different
/// resolutions. The preview depth-tests line fragments in a shader; the bake
/// reads the depth texture back and clips line vertices against it with exactly
/// the semantics of `outline.clip_hidden_lines`. Preview approximates bake by
/// construction rather than by resemblance, and the bake is exact because the
/// stage that decides visibility is a pure function of a `DepthImage`.
public struct BakeOptions: Sendable {
    /// Depth resolution, in pixels along each axis of the whole plate.
    public var resolution: Int
    /// How many tiles to split the depth *pass* into per axis. `nil` picks the
    /// smallest number that fits Metal's texture limit.
    public var tiles: Int?
    /// Hidden-line margin; `nil` takes the scene's.
    public var margin: Double?
    /// Trace the outline of the drawn region and add it as a final stroke
    /// layer. Off by default: the plates are bounded by their own wall ink, and
    /// an outline on top of that is a second edge.
    public var silhouette: Double?

    public init(resolution: Int, tiles: Int? = nil, margin: Double? = nil,
                silhouette: Double? = nil) {
        self.resolution = resolution; self.tiles = tiles; self.margin = margin
        self.silhouette = silhouette
    }

    func tileCount(for resolution: Int) -> Int {
        if let tiles { return max(tiles, 1) }
        return max(1, (resolution + metalTextureLimit - 1) / metalTextureLimit)
    }
}

public enum BakeError: Error, CustomStringConvertible {
    case emptyScene

    public var description: String {
        switch self {
        case .emptyScene: "bake: the scene projects to nothing (no geometry, no ink)"
        }
    }
}

public struct Bake: Sendable {
    public let strokes: Strokes
    /// The whole plate's depth, stitched from however many passes it took.
    public let depth: DepthImage
    /// How many passes that was, per axis.
    public let tiles: Int
}

public extension MetalRenderer {
    /// Bake a scene to strokes.
    ///
    /// Tiling splits the *render*, never the clip. A path that crosses a tile
    /// boundary has vertices decided by two different passes, and clipping each
    /// pass separately breaks the path at the seam: the segment joining the last
    /// vertex on one side to the first on the other is drawn by neither, and a
    /// single stranded vertex is dropped as a run shorter than two. So the
    /// passes are stitched into one `DepthImage` first and the clip runs once
    /// over the whole plate. That costs the full buffer in memory -- 1.6 GB for
    /// gamma's 20000-square bake, held while the tiles are still being drawn --
    /// and buys a bake that does not depend on how it was split. A single pass
    /// skips the stitch entirely and is handed straight to the clip.
    ///
    /// Tiles meet on pixel boundaries -- `DepthFrame.tile` splits the lattice,
    /// not the coordinate range -- so a coordinate lands on the same pixel
    /// however the pass was divided, and the stitched buffer has no seam.
    ///
    /// It is not bit-identical to a single pass, and cannot be: each pass
    /// derives its NDC mapping from its own sub-frame in float32, so a triangle
    /// edge landing on a pixel centre may round to either side and interpolated
    /// depths differ in their last bits. Both effects are confined to triangle
    /// edges. Measured on the fixture bundle, a 5x5 split moves depth by at
    /// most 3e-6 over a span of 0.57 and changes coverage on well under a tenth
    /// of a percent of pixels -- and changes no stroke at all, which is the
    /// property that matters and the one `kurven-test` asserts.
    func bake(_ scene: Scene, options: BakeOptions) throws -> Bake {
        guard let bounds = scene.viewBounds() else { throw BakeError.emptyScene }
        let frame = DepthFrame(covering: bounds, resolution: options.resolution)
        let n = options.tileCount(for: options.resolution)
        let margin = options.margin ?? scene.margin

        let depth: DepthImage
        if n == 1 {
            // One pass is the whole plate. Copying it into a second buffer of
            // the same size to call it "stitched" was 390 ms of a 1.6 s bake at
            // 16000 square, to produce a bit-identical array.
            depth = try renderDepth(scene, frame: frame)
        } else {
            var values: [Float] = []
            var empty: Float = -.infinity
            for ti in 0..<n {
                for tj in 0..<n {
                    let sub = frame.tile(ti, tj, of: n)
                    let tile = try renderDepth(scene, frame: sub)
                    if values.isEmpty {
                        empty = tile.empty
                        values = [Float](repeating: empty, count: frame.rows * frame.cols)
                    }
                    let row0 = frame.rows * ti / n
                    let col0 = frame.cols * tj / n
                    let rowBytes = sub.cols * MemoryLayout<Float>.stride
                    values.withUnsafeMutableBufferPointer { dst in
                        tile.values.withUnsafeBufferPointer { src in
                            for r in 0..<sub.rows {
                                memcpy(dst.baseAddress! + (row0 + r) * frame.cols + col0,
                                       src.baseAddress! + r * sub.cols, rowBytes)
                            }
                        }
                    }
                }
            }
            depth = DepthImage(frame: frame, values: values, empty: empty)
        }

        var layers: [(style: Style, paths: PolylineSet<PlateSpace>)] = []
        for (layer, projected) in scene.projectedLayers() {
            let clipped = layer.spec.clipped
                ? HiddenLine.clip(projected, against: depth, margin: margin)
                : HiddenLine.pass(projected)
            layers.append((Style(layer.spec), clipped))
        }
        if let width = options.silhouette {
            layers.append((Style(color: "#000000", width: width), depth.silhouette()))
        }
        return Bake(strokes: Strokes(layers: layers), depth: depth, tiles: n)
    }
}
