import Foundation
import Metal
import KurvenCore
import KurvenShaderTypes

/// The GPU-side of a scene's *content*: everything that does not change when
/// the camera moves.
///
/// A camera change must touch one uniform buffer and nothing else. Without this
/// it touches the entire heightfield: `renderDepth` was re-deriving the lattice,
/// re-clamping the grid and re-uploading the texture on every call, which is
/// 56 ms per frame on zeta's hundred-megabyte grid -- eighteen frames a second
/// before drawing anything, and four redundant uploads in a tiled bake.
///
/// Built once per `Scene.content`. Nothing here reads the camera.
final class SceneResources {
    let content: ContentID

    let heights: MTLTexture
    /// True when the cap is already folded into `heights` (band caps depend on
    /// x, so they are baked; a uniform cap stays a live uniform).
    let capBaked: Bool

    /// The decimated lattice: dimensions and domain only. The heights come from
    /// the texture, so the samples themselves are never on the CPU.
    let latticeWidth: Int
    let latticeHeight: Int
    let latticeDomain: Domain
    let gridWidth: Int
    let gridHeight: Int

    let tiles: MTLBuffer
    let tileCount: Int
    let region: MTLBuffer
    let regionCount: Int
    let walls: MTLBuffer?
    let wallVertexCount: Int

    var cells: Int { max(latticeWidth - 1, 0) * max(latticeHeight - 1, 0) }

    init(scene: Scene, device: MTLDevice) throws {
        content = scene.content

        let grid: Grid2D<Float>
        switch scene.surface.caps {
        case .none, .uniform: grid = scene.surface.height; capBaked = false
        case .realBands: grid = scene.surface.clamped(); capBaked = true
        }
        gridWidth = grid.width
        gridHeight = grid.height

        let extent = grid.decimatedExtent(by: scene.step)
        latticeWidth = extent.width
        latticeHeight = extent.height
        latticeDomain = extent.domain

        heights = try SceneResources.upload(grid, device: device)

        let tileValues = scene.tiles.map {
            KVTile(linear: SIMD4<Float>(Float($0.a), Float($0.b), Float($0.c), Float($0.d)),
                   offset: SIMD2<Float>(Float($0.tx), Float($0.ty)))
        }
        tiles = try SceneResources.buffer(tileValues, "tile transforms", device)
        tileCount = tileValues.count

        let corners: [SIMD2<Float>]
        if case .inside(let p) = scene.region {
            corners = p.edges.map { SIMD2(Float($0.start.x), Float($0.start.y)) }
        } else {
            corners = []
        }
        regionCount = corners.count
        // Bound even when empty: Metal validation requires every declared buffer
        // argument to have a binding, whether or not the shader reads it.
        region = try SceneResources.buffer(corners.isEmpty ? [SIMD2<Float>.zero] : corners,
                                           "the region polygon", device)

        // Walls are expanded to a flat vertex list once. They are a few thousand
        // triangles, so an index buffer would save little and cost a binding.
        let wallVertices = scene.occluder.triangles.flatMap { t in
            t.indices.map { i in
                KVVertex(position: SIMD3<Float>(scene.occluder.vertices[Int(t[i])].v))
            }
        }
        wallVertexCount = wallVertices.count
        walls = wallVertices.isEmpty
            ? nil : try SceneResources.buffer(wallVertices, "wall vertices", device)
    }

    private static func upload(_ grid: Grid2D<Float>, device: MTLDevice) throws -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float, width: grid.width, height: grid.height,
            mipmapped: false)
        d.usage = .shaderRead
        d.storageMode = .shared
        guard let t = device.makeTexture(descriptor: d) else {
            throw RendererError.allocation("a \(grid.width)x\(grid.height) height texture")
        }
        grid.values.withUnsafeBytes { bytes in
            t.replace(region: MTLRegionMake2D(0, 0, grid.width, grid.height),
                      mipmapLevel: 0, withBytes: bytes.baseAddress!,
                      bytesPerRow: grid.width * MemoryLayout<Float>.stride)
        }
        return t
    }

    private static func buffer<T>(_ values: [T], _ what: String,
                                  _ device: MTLDevice) throws -> MTLBuffer {
        guard let b = values.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count,
                              options: .storageModeShared)
        }) else { throw RendererError.allocation(what) }
        return b
    }
}

/// The render target, and how its pixels get back to the CPU.
///
/// A texture created from an `MTLBuffer` is *linear*: its rows are contiguous in
/// memory the CPU can already address, so reading it back is a memcpy. An
/// ordinary texture is stored in a tiled, swizzled layout and `getBytes` has to
/// detile it -- measured at about 1 GB/s against the ~100 GB/s the hardware has,
/// which made readback 1.0 s of a 1.6 s bake at 16000 square.
///
/// Not every device allows a linear texture as a render target, so this falls
/// back to an ordinary one and says which it got.
final class DepthTarget {
    let texture: MTLTexture
    let rows: Int
    let cols: Int
    /// Non-nil when the texture is linear and its memory is directly readable.
    let linear: (buffer: MTLBuffer, bytesPerRow: Int)?

    var isLinear: Bool { linear != nil }

    init(rows: Int, cols: Int, device: MTLDevice) throws {
        self.rows = rows; self.cols = cols
        let d = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float, width: cols, height: rows, mipmapped: false)
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .shared

        let rowBytes = cols * MemoryLayout<Float>.stride
        let align = max(device.minimumLinearTextureAlignment(for: .r32Float), 1)
        let stride = ((rowBytes + align - 1) / align) * align
        if let buffer = device.makeBuffer(length: stride * rows, options: .storageModeShared),
           let t = buffer.makeTexture(descriptor: d, offset: 0, bytesPerRow: stride) {
            texture = t
            linear = (buffer, stride)
            return
        }
        guard let t = device.makeTexture(descriptor: d) else {
            throw RendererError.allocation("a \(cols)x\(rows) depth target")
        }
        texture = t
        linear = nil
    }

    /// The rendered pixels, as a value.
    ///
    /// The clear sentinel is carried through as `DepthImage.empty` rather than
    /// rewritten to `-.infinity`: the two decide every visibility test
    /// identically, and rewriting means a scalar pass over every pixel -- 400 ms
    /// of a 1.6 s bake, to change nothing anyone reads.
    func read(frame: DepthFrame, empty: Float) -> DepthImage {
        let count = rows * cols
        let rowBytes = cols * MemoryLayout<Float>.stride
        let values = [Float](unsafeUninitializedCapacity: count) { out, initialized in
            if let (buffer, stride) = linear {
                let base = buffer.contents()
                if stride == rowBytes {
                    memcpy(out.baseAddress!, base, count * MemoryLayout<Float>.stride)
                } else {
                    for r in 0..<rows {
                        memcpy(out.baseAddress! + r * cols, base + r * stride, rowBytes)
                    }
                }
            } else {
                texture.getBytes(out.baseAddress!, bytesPerRow: rowBytes,
                                 from: MTLRegionMake2D(0, 0, cols, rows), mipmapLevel: 0)
            }
            initialized = count
        }
        return DepthImage(frame: frame, values: values, empty: empty)
    }
}
