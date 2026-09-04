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

    /// Resources for the scene being drawn.
    ///
    /// A single-entry memo. The renderer belongs to one document -- one window,
    /// or one bake -- so it draws one scene's content over and over from
    /// different places, which is exactly the shape a single entry serves. Two
    /// bundles means two renderers.
    private var resources: SceneResources?
    private var target: DepthTarget?

    private func resources(for scene: Scene) throws -> SceneResources {
        if let r = resources, r.content == scene.content { return r }
        let r = try SceneResources(scene: scene, device: device)
        resources = r
        return r
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
        if res.cells > 0 {
            encoder.setRenderPipelineState(heightPipeline)
            encoder.setVertexBuffer(res.tiles, offset: 0, index: 1)
            encoder.setVertexBuffer(res.region, offset: 0, index: 2)
            encoder.setVertexTexture(res.heights, index: 0)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0,
                                   vertexCount: res.cells * 6,
                                   instanceCount: res.tileCount)
        }

        if let walls = res.walls {
            encoder.setRenderPipelineState(meshPipeline)
            encoder.setVertexBuffer(walls, offset: 0, index: 3)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0,
                                   vertexCount: res.wallVertexCount)
        }

        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw RendererError.pipeline("\(error)") }

        return out.read(frame: frame, empty: Self.emptySentinel)
    }

    // MARK: - uniforms

    private func uniforms(_ scene: Scene, frame: DepthFrame,
                          resources res: SceneResources) -> KVUniforms {
        let ndc = frame.metalNDC
        let d = res.latticeDomain
        return KVUniforms(
            view: scene.camera.view.float4x4,
            ndcScale: SIMD2<Float>(Float(ndc.scale.x), Float(ndc.scale.y)),
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
