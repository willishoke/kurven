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
    /// How many tiles to split the depth pass into per axis. `nil` picks the
    /// smallest number that fits Metal's texture limit.
    public var tiles: Int?
    /// Hidden-line margin; `nil` takes the scene's.
    public var margin: Double?

    public init(resolution: Int, tiles: Int? = nil, margin: Double? = nil) {
        self.resolution = resolution; self.tiles = tiles; self.margin = margin
    }

    func tileCount(for resolution: Int) -> Int {
        if let tiles { return max(tiles, 1) }
        return (resolution + metalTextureLimit - 1) / metalTextureLimit
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
    public let frame: DepthFrame
    /// One depth image per tile, in row-major tile order. Kept so the CLI can
    /// dump them and a test can compare them against the Python oracle.
    public let depths: [DepthImage]
}

public extension MetalRenderer {
    /// Bake a scene to strokes.
    ///
    /// Tiling is a pure operation on the depth frame: each tile is the same
    /// scene rendered into a sub-rectangle of the same lattice, so a clipped
    /// vertex gets the same answer whichever tile it lands in. Only the *depth*
    /// pass tiles; the clip is done once per tile over the layers whose vertices
    /// fall inside it, and the runs are reassembled in path order.
    func bake(_ scene: Scene, options: BakeOptions) throws -> Bake {
        guard let bounds = scene.viewBounds() else { throw BakeError.emptyScene }
        let frame = DepthFrame(covering: bounds, resolution: options.resolution)
        let n = options.tileCount(for: options.resolution)
        let margin = options.margin ?? scene.margin

        let projected = scene.projectedLayers()

        var depths: [DepthImage] = []
        // Per layer, per path, the visible runs found so far. Collected across
        // tiles and concatenated at the end so paths stay in declaration order
        // however the depth pass was split.
        var runs: [[[[P3<PlateSpace>]]]] = projected.map {
            Array(repeating: [], count: $0.1.count)
        }

        for ti in 0..<n {
            for tj in 0..<n {
                let sub = frame.tile(ti, tj, of: n)
                let depth = try renderDepth(scene, frame: sub)
                depths.append(depth)
                for (li, (layer, paths)) in projected.enumerated() where layer.spec.clipped {
                    for pi in 0..<paths.count {
                        let path = paths[path: pi]
                        // Only vertices this tile actually covers can be
                        // decided here; the rest belong to another tile.
                        guard path.contains(where: { sub.contains($0.xy) }) else { continue }
                        var run: [P3<PlateSpace>] = []
                        for v in path {
                            guard sub.contains(v.xy) else {
                                if run.count >= 2 { runs[li][pi].append(run) }
                                run = []
                                continue
                            }
                            if v.z + margin > depth.depth(under: v.xy) {
                                run.append(P3(v.x, v.y, v.z))
                            } else {
                                if run.count >= 2 { runs[li][pi].append(run) }
                                run = []
                            }
                        }
                        if run.count >= 2 { runs[li][pi].append(run) }
                    }
                }
            }
        }

        var layers: [(style: Style, paths: PolylineSet<PlateSpace>)] = []
        for (li, (layer, paths)) in projected.enumerated() {
            let set: PolylineSet<PlateSpace>
            if layer.spec.clipped {
                set = PolylineSet(paths: runs[li].flatMap { $0 })
            } else {
                set = HiddenLine.pass(paths)
            }
            layers.append((Style(layer.spec), set))
        }
        return Bake(strokes: Strokes(layers: layers), frame: frame, depths: depths)
    }
}
