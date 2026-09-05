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
    /// The queue this renderer submits on. A bake makes its own renderer, and
    /// therefore its own queue, so it never contends with a preview.
    public var commandQueue: MTLCommandQueue { queue }
    let heightPipeline: MTLRenderPipelineState
    let meshPipeline: MTLRenderPipelineState
    // The preview's second pass. Same geometry, different fragment work.
    let paperSurfacePipeline: MTLRenderPipelineState
    let paperWallPipeline: MTLRenderPipelineState
    let shadedSurfacePipeline: MTLRenderPipelineState
    let shadedWallPipeline: MTLRenderPipelineState
    let linePipeline: MTLRenderPipelineState
    let depthViewPipeline: MTLRenderPipelineState
    /// The pixel format the preview draws into; the depth pass is always
    /// r32Float.
    public let previewFormat: MTLPixelFormat

    /// Written where nothing was drawn. Metal cannot clear to -infinity, so the
    /// buffer clears to this and `readback` maps it back -- the same trick, and
    /// the same sentinel magnitude, as the Python moderngl path.
    static let emptySentinel: Float = -1e30

    public init(device: MTLDevice? = nil,
                previewFormat: MTLPixelFormat = .bgra8Unorm) throws {
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

        // The colour pass has no depth attachment. It does not need one: pass 1
        // already holds the front-most view depth at every pixel, and both the
        // shaded surface and the ink decide visibility by reading it. That is
        // one depth semantic in the program rather than two, and it is the same
        // one `clip_hidden_lines` uses.
        func colorPipeline(_ vertex: String, _ fragment: String,
                           primitive: MTLPrimitiveTopologyClass = .triangle) throws
            -> MTLRenderPipelineState
        {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = try function(vertex)
            d.fragmentFunction = try function(fragment)
            d.colorAttachments[0].pixelFormat = previewFormat
            d.inputPrimitiveTopology = primitive
            do { return try device.makeRenderPipelineState(descriptor: d) }
            catch { throw RendererError.pipeline("\(vertex)/\(fragment): \(error)") }
        }
        self.previewFormat = previewFormat
        self.paperSurfacePipeline = try colorPipeline("kv_surface_vertex", "kv_paper_fragment")
        self.paperWallPipeline = try colorPipeline("kv_wall_vertex", "kv_paper_fragment")
        self.shadedSurfacePipeline = try colorPipeline("kv_surface_vertex", "kv_shaded_fragment")
        self.shadedWallPipeline = try colorPipeline("kv_wall_vertex", "kv_shaded_fragment")
        self.linePipeline = try colorPipeline("kv_line_vertex", "kv_line_fragment",
                                              primitive: .line)
        self.depthViewPipeline = try colorPipeline("kv_fullscreen_vertex",
                                                   "kv_depth_view_fragment")
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

    /// Resources for the scene being drawn.
    ///
    /// A single-entry memo. The renderer belongs to one document -- one window,
    /// or one bake -- so it draws one scene's content over and over from
    /// different places, which is exactly the shape a single entry serves. Two
    /// bundles means two renderers.
    private var cachedResources: SceneResources?
    private var cachedLines: (ink: ContentID, geometry: LineGeometry)?
    private var target: DepthTarget?
    private var previewDepth: MTLTexture?
    /// The depth attachment the last preview frame wrote, for pixel queries.
    var lastPreviewDepth: MTLTexture? { previewDepth }

    func resources(for scene: Scene) throws -> SceneResources {
        if let r = cachedResources, r.content == scene.content { return r }
        let r = try SceneResources(scene: scene, device: device)
        cachedResources = r
        return r
    }

    /// Line geometry, cached on the scene's *ink* identity rather than its
    /// content, so moving a level set rebuilds the strokes and leaves the
    /// heightfield texture alone.
    func lineGeometry(for scene: Scene) throws -> LineGeometry {
        if let c = cachedLines, c.ink == scene.ink { return c.geometry }
        let g = try LineGeometry(scene: scene, device: device)
        cachedLines = (scene.ink, g)
        return g
    }

    /// The preview's depth attachment.
    ///
    /// Shared rather than private, because double-clicking to re-target reads
    /// one pixel of it: the depth under the cursor, unprojected through the
    /// camera's inverse, is the world point you clicked on. One pixel is not
    /// worth a blit, and on unified memory `shared` costs nothing.
    func depthTexture(rows: Int, cols: Int) throws -> MTLTexture {
        if let t = previewDepth, t.width == cols, t.height == rows { return t }
        let d = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float, width: cols, height: rows, mipmapped: false)
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .shared
        guard let t = device.makeTexture(descriptor: d) else {
            throw RendererError.allocation("a \(cols)x\(rows) preview depth texture")
        }
        previewDepth = t
        return t
    }

    /// Draw the occluding geometry: the implicit heightfield, then the walls.
    /// Shared by the depth pass and the preview's surface pass, so the two
    /// cannot draw different things.
    func encodeGeometry(_ encoder: MTLRenderCommandEncoder, res: SceneResources,
                        height: MTLRenderPipelineState, mesh: MTLRenderPipelineState) {
        if res.cells > 0 {
            encoder.setRenderPipelineState(height)
            encoder.setVertexBuffer(res.tiles, offset: 0, index: 1)
            encoder.setVertexBuffer(res.region, offset: 0, index: 2)
            encoder.setVertexTexture(res.heights, index: 0)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0,
                                   vertexCount: res.cells * 6,
                                   instanceCount: res.tileCount)
        }
        if let walls = res.walls {
            encoder.setRenderPipelineState(mesh)
            encoder.setVertexBuffer(walls, offset: 0, index: 3)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0,
                                   vertexCount: res.wallVertexCount)
        }
    }

    func previewUniforms(_ scene: Scene, frame: DepthFrame,
                         resources res: SceneResources) -> KVUniforms {
        uniforms(scene, frame: frame, resources: res)
    }

    private func target(rows: Int, cols: Int) throws -> DepthTarget {
        if let t = target, t.rows == rows, t.cols == cols { return t }
        let t = try DepthTarget(rows: rows, cols: cols, device: device)
        target = t
        return t
    }

    /// Whether the depth target's memory is directly readable, or has to be
    /// detiled on the way out. Reported by `kurven-cli` so a slow bake on some
    /// future device has an obvious first suspect.
    public var hasLinearReadback: Bool { target?.isLinear ?? false }

    /// Rasterize a scene's occluding geometry into a depth image.
    public func renderDepth(_ scene: Scene, frame: DepthFrame) throws -> DepthImage {
        guard frame.cols <= metalTextureLimit, frame.rows <= metalTextureLimit else {
            throw RendererError.textureTooLarge(max(frame.rows, frame.cols),
                                                limit: metalTextureLimit)
        }

        let res = try resources(for: scene)
        let out = try target(rows: frame.rows, cols: frame.cols)
        var uniforms = self.uniforms(scene, frame: frame, resources: res)

        let descriptor = MTLRenderPassDescriptor()
        descriptor.colorAttachments[0].texture = out.texture
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
        encodeGeometry(encoder, res: res, height: heightPipeline, mesh: meshPipeline)
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw RendererError.pipeline("\(error)") }

        return out.read(frame: frame, empty: Self.emptySentinel)
    }

    // MARK: - uniforms

    func uniforms(_ scene: Scene, frame: DepthFrame,
                  resources res: SceneResources) -> KVUniforms {
        let ndc = frame.metalNDC
        let d = res.latticeDomain
        return KVUniforms(
            view: scene.camera.view.float4x4,
            ndcLinear: simd_float2x2(SIMD2<Float>(ndc.linear.columns.0),
                                     SIMD2<Float>(ndc.linear.columns.1)),
            ndcOffset: SIMD2<Float>(Float(ndc.offset.x), Float(ndc.offset.y)),
            domainLo: SIMD2<Float>(Float(d.real.lo), Float(d.imag.lo)),
            domainSize: SIMD2<Float>(Float(d.real.length), Float(d.imag.length)),
            lattice: SIMD2<UInt32>(UInt32(res.latticeWidth), UInt32(res.latticeHeight)),
            gridSize: SIMD2<UInt32>(UInt32(res.gridWidth), UInt32(res.gridHeight)),
            step: UInt32(scene.step),
            cap: res.capBaked ? .infinity : Float(scene.surface.caps.height(atX: 0)),
            regionCount: UInt32(res.regionCount),
            empty: Self.emptySentinel)
    }
}

// MARK: - layout agreement

public extension MetalRenderer {
    /// What the shader sees when the CPU hands it a `KVUniforms`.
    ///
    /// The header is the single declaration of the struct, but nothing at build
    /// time enforces that the shader gets the same one: there is no `metal`
    /// compiler under Command Line Tools, and SwiftPM does not treat a C header
    /// as a dependency of the Swift targets that import it -- so an incremental
    /// build after editing the header can leave the two sides disagreeing, which
    /// shows up as a corrupted uniform and a crash rather than as a compile
    /// error. Asking the GPU what it sees turns that into a test.
    static func probeUniformLayout(device: MTLDevice, sending uniforms: KVUniforms,
                                   and shading: KVShading)
        throws -> (fields: [Float], uniformSize: Int, shadingSize: Int)
    {
        let library = try makeLibrary(device: device)
        guard let function = library.makeFunction(name: "kv_layout_probe") else {
            throw RendererError.missingFunction("kv_layout_probe")
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        let n = Shaders.layoutProbeCount
        guard let queue = device.makeCommandQueue(),
              let out = device.makeBuffer(length: n * MemoryLayout<Float>.stride,
                                          options: .storageModeShared),
              let commands = queue.makeCommandBuffer(),
              let encoder = commands.makeComputeCommandEncoder() else {
            throw RendererError.allocation("the layout probe")
        }
        var u = uniforms
        var sh = shading
        encoder.setComputePipelineState(pipeline)
        encoder.setBytes(&u, length: MemoryLayout<KVUniforms>.stride, index: 0)
        encoder.setBuffer(out, offset: 0, index: 1)
        encoder.setBytes(&sh, length: MemoryLayout<KVShading>.stride, index: 2)
        encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw RendererError.pipeline("\(error)") }

        let p = out.contents().bindMemory(to: Float.self, capacity: n)
        let fields = Array(UnsafeBufferPointer(start: p, count: n))
        return (Array(fields.dropLast(2)), Int(fields[n - 2]), Int(fields[n - 1]))
    }
}
