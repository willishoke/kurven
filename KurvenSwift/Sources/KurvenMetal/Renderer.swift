import Foundation
import Metal
import KurvenCore
import KurvenShaderTypes

public enum RendererError: Error, CustomStringConvertible {
    case noDevice
    case shaderCompilation(String)
    case missingFunction(String)
    case pipeline(String)
    case allocation(String)
    case textureTooLarge(Int, limit: Int)

    public var description: String {
        switch self {
        case .noDevice: "metal: no device (this build needs a GPU)"
        case .shaderCompilation(let m): "metal: shader compilation failed\n\(m)"
        case .missingFunction(let n): "metal: the library has no function '\(n)'"
        case .pipeline(let m): "metal: pipeline state failed (\(m))"
        case .allocation(let w): "metal: could not allocate \(w)"
        case .textureTooLarge(let n, let l): "metal: \(n) exceeds the \(l) texture limit"
        }
    }
}

/// Metal's 2D texture side limit on every Apple GPU family this targets. A bake
/// past this tiles (`KurvenBake`), which is a pure camera operation.
public let metalTextureLimit = 16384

/// The offscreen depth renderer.
///
/// This is the one place in the frontend with reference semantics: a device,
/// pipelines, textures. Everything it consumes is a value and everything it
/// produces is a value; `render` takes a `Scene` and a `DepthFrame` and returns
/// a `DepthImage`.
///
/// The depth buffer is a MAX-blended `r32Float` color attachment, not a depth
/// attachment. That is exactly `GL_MAX` over view z -- what the Python Z-buffer
/// holds, in the float32 the readback and the clipper compare -- so "the GPU
/// preview and the CPU oracle agree" is a statement about one number rather
/// than about two encodings of it.
public final class MetalRenderer {
    public let device: MTLDevice
    let queue: MTLCommandQueue
    let heightPipeline: MTLRenderPipelineState
    let meshPipeline: MTLRenderPipelineState

    /// Written where nothing was drawn. Metal cannot clear to -infinity, so the
    /// buffer clears to this and `readback` maps it back -- the same trick, and
    /// the same sentinel magnitude, as the Python moderngl path.
    static let emptySentinel: Float = -1e30

    public init(device: MTLDevice? = nil) throws {
        guard let device = device ?? MTLCreateSystemDefaultDevice() else {
            throw RendererError.noDevice
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw RendererError.allocation("a command queue")
        }
        self.queue = queue

        let library = try Self.makeLibrary(device: device)
        func function(_ name: String) throws -> MTLFunction {
            guard let f = library.makeFunction(name: name) else {
                throw RendererError.missingFunction(name)
            }
            return f
        }
        let fragment = try function("kv_depth_fragment")

        func pipeline(vertex: String) throws -> MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = try function(vertex)
            d.fragmentFunction = fragment
            let a = d.colorAttachments[0]!
            a.pixelFormat = .r32Float
            a.isBlendingEnabled = true
            a.rgbBlendOperation = .max
            a.alphaBlendOperation = .max
            a.sourceRGBBlendFactor = .one
            a.destinationRGBBlendFactor = .one
            a.sourceAlphaBlendFactor = .one
            a.destinationAlphaBlendFactor = .one
            do { return try device.makeRenderPipelineState(descriptor: d) }
            catch { throw RendererError.pipeline("\(vertex): \(error)") }
        }
        self.heightPipeline = try pipeline(vertex: "kv_height_vertex")
        self.meshPipeline = try pipeline(vertex: "kv_mesh_vertex")
    }

    /// Compile `Shaders.source`. Exposed so a test can fail `swift test` on a
    /// shader error rather than leaving it for the first launch -- the check the
    /// absent `metal` compiler would otherwise provide.
    public static func makeLibrary(device: MTLDevice) throws -> MTLLibrary {
        let options = MTLCompileOptions()
        options.mathMode = .fast
        do { return try device.makeLibrary(source: Shaders.source, options: options) }
        catch { throw RendererError.shaderCompilation("\(error)") }
    }

    // MARK: - rendering

    /// Rasterize a scene's occluding geometry into a depth image.
    public func renderDepth(_ scene: Scene, frame: DepthFrame) throws -> DepthImage {
        guard frame.cols <= metalTextureLimit, frame.rows <= metalTextureLimit else {
            throw RendererError.textureTooLarge(max(frame.rows, frame.cols),
                                                limit: metalTextureLimit)
        }

        let lattice = scene.surface.clamped().decimated(by: scene.step)
        let (heights, capped) = try uploadHeights(scene)
        let target = try makeTarget(frame)

        var uniforms = self.uniforms(scene, frame: frame, lattice: lattice,
                                     capped: capped)
        let tiles = scene.tiles.map {
            KVTile(linear: SIMD4<Float>(Float($0.a), Float($0.b), Float($0.c), Float($0.d)),
                   offset: SIMD2<Float>(Float($0.tx), Float($0.ty)))
        }
        let region: [SIMD2<Float>] = {
            guard case .inside(let p) = scene.region else { return [] }
            return p.edges.map { SIMD2(Float($0.start.x), Float($0.start.y)) }
        }()
        let wallVertices = scene.occluder.triangles.flatMap { t in
            t.indices.map { i in
                KVVertex(position: SIMD3<Float>(scene.occluder.vertices[Int(t[i])].v))
            }
        }

        let descriptor = MTLRenderPassDescriptor()
        descriptor.colorAttachments[0].texture = target
        descriptor.colorAttachments[0].loadAction = .clear
        descriptor.colorAttachments[0].storeAction = .store
        descriptor.colorAttachments[0].clearColor =
            MTLClearColor(red: Double(Self.emptySentinel), green: 0, blue: 0, alpha: 0)

        guard let commands = queue.makeCommandBuffer(),
              let encoder = commands.makeRenderCommandEncoder(descriptor: descriptor) else {
            throw RendererError.allocation("a render encoder")
        }

        encoder.setVertexBytes(&uniforms, length: MemoryLayout<KVUniforms>.stride, index: 0)

        // The heightfield: no vertex or index buffer, six vertices per cell,
        // one instance per tile.
        let cells = max(lattice.width - 1, 0) * max(lattice.height - 1, 0)
        if cells > 0 {
            encoder.setRenderPipelineState(heightPipeline)
            encoder.setVertexBuffer(try buffer(tiles, "tile transforms"), offset: 0, index: 1)
            if !region.isEmpty {
                encoder.setVertexBuffer(try buffer(region, "the region polygon"),
                                        offset: 0, index: 2)
            } else {
                // Bound anyway: Metal validation requires every declared buffer
                // argument to have a binding even when the shader never reads it.
                encoder.setVertexBuffer(try buffer([SIMD2<Float>.zero], "a region placeholder"),
                                        offset: 0, index: 2)
            }
            encoder.setVertexTexture(heights, index: 0)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0,
                                   vertexCount: cells * 6, instanceCount: tiles.count)
        }

        if !wallVertices.isEmpty {
            encoder.setRenderPipelineState(meshPipeline)
            encoder.setVertexBuffer(try buffer(wallVertices, "wall vertices"),
                                    offset: 0, index: 3)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0,
                                   vertexCount: wallVertices.count)
        }

        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw RendererError.pipeline("\(error)") }

        return DepthReadback.read(target, frame: frame, empty: Self.emptySentinel)
    }

    // MARK: - resources

    private func uniforms(_ scene: Scene, frame: DepthFrame,
                          lattice: Grid2D<Float>, capped: Bool) -> KVUniforms {
        let ndc = frame.metalNDC
        let d = lattice.domain
        return KVUniforms(
            view: scene.camera.view.float4x4,
            ndcScale: SIMD2<Float>(Float(ndc.scale.x), Float(ndc.scale.y)),
            ndcOffset: SIMD2<Float>(Float(ndc.offset.x), Float(ndc.offset.y)),
            domainLo: SIMD2<Float>(Float(d.real.lo), Float(d.imag.lo)),
            domainSize: SIMD2<Float>(Float(d.real.length), Float(d.imag.length)),
            lattice: SIMD2<UInt32>(UInt32(lattice.width), UInt32(lattice.height)),
            gridSize: SIMD2<UInt32>(UInt32(scene.surface.height.width),
                                    UInt32(scene.surface.height.height)),
            step: UInt32(scene.step),
            cap: capped ? .infinity : Float(scene.surface.caps.height(atX: 0)),
            regionCount: {
                if case .inside(let p) = scene.region { return UInt32(p.edges.count) }
                return 0
            }(),
            empty: Self.emptySentinel)
    }

    /// Upload the height grid. A uniform cap stays a live knob applied in the
    /// shader; band caps depend on x and are baked in here, which is the whole
    /// difference between the two arms and the reason `capped` comes back.
    private func uploadHeights(_ scene: Scene) throws -> (MTLTexture, capped: Bool) {
        let grid: Grid2D<Float>
        let capped: Bool
        switch scene.surface.caps {
        case .none, .uniform: grid = scene.surface.height; capped = false
        case .realBands: grid = scene.surface.clamped(); capped = true
        }
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
        return (t, capped)
    }

    private func makeTarget(_ frame: DepthFrame) throws -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float, width: frame.cols, height: frame.rows,
            mipmapped: false)
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .shared
        guard let t = device.makeTexture(descriptor: d) else {
            throw RendererError.allocation("a \(frame.cols)x\(frame.rows) depth target")
        }
        return t
    }

    private func buffer<T>(_ values: [T], _ what: String) throws -> MTLBuffer {
        guard let b = values.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count,
                              options: .storageModeShared)
        }) else { throw RendererError.allocation(what) }
        return b
    }
}
