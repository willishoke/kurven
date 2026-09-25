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
    case perspective

    public var description: String {
        switch self {
        case .emptyScene: "bake: the scene projects to nothing (no geometry, no ink)"
        case .perspective:
            """
            bake: this camera is perspective, and only orthographic cameras bake.
            A bake clips per vertex against a depth buffer it indexes by an             affine map from view coordinates to pixels; under perspective that             map depends on depth. It could be fixed by projecting first -- but             there is no perspective Projection on the Python side, so the result             would be the one artifact with no oracle to check it against.             Perspective is for navigating; the plates are orthographic.
            """
        }
    }
}

public struct Bake: Sendable {
    public let strokes: Strokes
    /// The whole plate's depth, stitched from however many passes it took.
    public let depth: DepthImage
    /// The whole plate's front-most surface coordinates, for a parametric
    /// scene; nil for a heightfield.
    public let surface: SurfaceImage?
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
        guard !scene.camera.isPerspective else { throw BakeError.perspective }
        guard let bounds = scene.viewBounds() else { throw BakeError.emptyScene }
        let frame = DepthFrame(covering: bounds, resolution: options.resolution)
        let n = options.tileCount(for: options.resolution)
        let margin = options.margin ?? scene.margin

        let depth: DepthImage
        let image: SurfaceImage?
        switch scene.geometry {
        case .heightfield:
            image = nil
            if n == 1 {
                // One pass is the whole plate. Copying it into a second buffer
                // of the same size to call it "stitched" was 390 ms of a 1.6 s
                // bake at 16000 square, to produce a bit-identical array.
                depth = try renderDepth(scene, frame: frame)
            } else {
                var empty: Float = -.infinity
                let values = try stitch(frame, n) { sub in
                    let tile = try renderDepth(scene, frame: sub)
                    empty = tile.empty
                    return tile.values
                }
                depth = DepthImage(frame: frame, values: values, empty: empty)
            }
        case .parametric:
            // Depth and coordinates are stitched by the same code over the
            // same sub-frames, so a coordinate sits on the pixel its depth does
            // however the plate was split.
            let whole: SurfaceImage
            if n == 1 {
                whole = try renderSurface(scene, frame: frame)
            } else {
                var empty: Float = -.infinity
                // `stitch` visits the tiles in one fixed order, so the second
                // call takes the coordinates back in the order the first
                // rendered them.
                var coordTiles: [[SIMD2<Float>]] = []
                let values = try stitch(frame, n) { sub in
                    let tile = try renderSurface(scene, frame: sub)
                    empty = tile.depth.empty
                    coordTiles.append(tile.coords)
                    return tile.depth.values
                }
                var next = 0
                let coords = try stitch(frame, n) { _ in
                    defer { next += 1 }
                    return coordTiles[next]
                }
                whole = SurfaceImage(depth: DepthImage(frame: frame, values: values, empty: empty),
                                     coords: coords)
            }
            depth = whole.depth
            image = whole
        }

        let visibility = scene.parametric.flatMap { s in
            image.map { SurfaceVisibility(surface: s, image: $0, view: scene.camera.view) }
        }
        let onHeightfield = scene.heightfield.map { h in
            HeightfieldVisibility(heightfield: h, depth: depth, margin: margin,
                                  view: scene.camera.view)
        }
        var layers: [(style: Style, paths: PolylineSet<PlateSpace>)] = []
        for (layer, projected) in scene.projectedLayers() {
            let clipped: PolylineSet<PlateSpace>
            if !layer.spec.clipped {
                clipped = HiddenLine.pass(projected)
            } else if let visibility, projected.coords != nil {
                // Ink that knows where on the surface it lies is judged by
                // that, with no margin in it.
                let onFolds: Bool
                if case .foldLines = layer.spec.source { onFolds = true } else { onFolds = false }
                clipped = HiddenLine.clip(projected, on: visibility, onFolds: onFolds)
            } else if let onHeightfield, layer.spec.liesOnSurface {
                // Ink on the heightfield itself is hidden where the surface
                // faces away, and depth-tested where it does not.
                clipped = HiddenLine.clip(projected, world: layer.paths, on: onHeightfield)
            } else {
                clipped = HiddenLine.clip(projected, against: depth, margin: margin)
            }
            layers.append((Style(layer.spec), clipped))
        }
        if let width = options.silhouette {
            layers.append((Style(color: "#000000", width: width), depth.silhouette()))
        }
        return Bake(strokes: Strokes(layers: layers), depth: depth, surface: image, tiles: n)
    }

    /// Render the plate as `n x n` sub-frames and copy each into its place in
    /// one row-major array.
    ///
    /// Tiles meet on pixel boundaries -- `DepthFrame.tile` splits the lattice,
    /// not the coordinate range -- so a pixel lands in the same place however
    /// the plate was divided.
    private func stitch<T>(_ frame: DepthFrame, _ n: Int,
                           _ render: (DepthFrame) throws -> [T]) throws -> [T] {
        var out: [T]?
        for ti in 0..<n {
            for tj in 0..<n {
                let sub = frame.tile(ti, tj, of: n)
                let tile = try render(sub)
                if out == nil {
                    out = [T](repeating: tile[0], count: frame.rows * frame.cols)
                }
                let row0 = frame.rows * ti / n
                let col0 = frame.cols * tj / n
                let rowBytes = sub.cols * MemoryLayout<T>.stride
                out!.withUnsafeMutableBytes { dst in
                    tile.withUnsafeBytes { src in
                        for r in 0..<sub.rows {
                            memcpy(dst.baseAddress! + ((row0 + r) * frame.cols + col0)
                                       * MemoryLayout<T>.stride,
                                   src.baseAddress! + r * rowBytes, rowBytes)
                        }
                    }
                }
            }
        }
        return out ?? []
    }
}
