import Foundation
import Metal
import simd
import KurvenCore
import KurvenShaderTypes

/// A parametric surface on the GPU: its lattice as a position texture.
///
/// Like the heightfield, no mesh: `kv_param_vertex` turns a vertex id into a
/// cell, a corner and a texel. The texel holds the whole point rather than a
/// height, which is the one thing a torus needs that a landscape did not.
final class SurfaceResources {
    let positions: MTLTexture
    let lattice: KVSurface
    let cells: Int
    /// The view-z range, per camera; the lattice is fixed.
    let viewDepth: @Sendable (Transform<WorldSpace, ViewSpace>) -> (lo: Double, hi: Double)

    init(_ s: ParametricSurface, device: MTLDevice) throws {
        let d = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: s.u.samples, height: s.v.samples,
            mipmapped: false)
        d.usage = .shaderRead
        d.storageMode = .shared
        guard let t = device.makeTexture(descriptor: d) else {
            throw RendererError.allocation("a \(s.u.samples)x\(s.v.samples) position texture")
        }
        let texels = s.positions.map { SIMD4<Float>(Float($0.x), Float($0.y), Float($0.z), 1) }
        texels.withUnsafeBytes { bytes in
            t.replace(region: MTLRegionMake2D(0, 0, s.u.samples, s.v.samples),
                      mipmapLevel: 0, withBytes: bytes.baseAddress!,
                      bytesPerRow: s.u.samples * MemoryLayout<SIMD4<Float>>.stride)
        }
        positions = t
        lattice = KVSurface(
            samples: SIMD2(UInt32(s.u.samples), UInt32(s.v.samples)),
            cells: SIMD2(UInt32(s.u.cells), UInt32(s.v.cells)),
            lo: SIMD2(Float(s.u.range.lo), Float(s.v.range.lo)),
            spacing: SIMD2(Float(s.u.spacing), Float(s.v.spacing)),
            depthRange: .zero)
        cells = s.u.cells * s.v.cells
        let points = s.positions
        viewDepth = { view in
            var lo = Double.infinity, hi = -Double.infinity
            for p in points { let z = view(p).z; lo = min(lo, z); hi = max(hi, z) }
            return (lo, hi)
        }
    }
}

public extension MetalRenderer {
    /// Rasterize a parametric surface into a depth image and, beside it, the
    /// front-most surface coordinate at every pixel.
    ///
    /// Two passes over the same triangles. The first is the depth pass as the
    /// heightfield has it -- view z, MAX-blended -- so everything that reads a
    /// `DepthImage` reads this one the same way. The second keeps the
    /// front-most fragment's coordinate behind a depth test of its own, on
    /// view z normalized over the surface's own range. Where two sheets are
    /// within float precision of each other in depth the two passes may name
    /// different winners; both are then front-most to that precision.
    ///
    /// Orthographic cameras only, as the bake is; the frame is the lattice
    /// the image is sampled on.
    func renderSurface(_ surface: ParametricSurface, view: Transform<WorldSpace, ViewSpace>,
                       frame: DepthFrame) throws -> SurfaceImage {
        guard frame.cols <= metalTextureLimit, frame.rows <= metalTextureLimit else {
            throw RendererError.textureTooLarge(max(frame.rows, frame.cols),
                                                limit: metalTextureLimit)
        }
        let res = try SurfaceResources(surface, device: device)
        return try renderSurface(res, view: view, frame: frame)
    }

    /// A parametric scene's surface image, from its camera. Orthographic only.
    func renderSurface(_ scene: Scene, frame: DepthFrame) throws -> SurfaceImage {
        guard frame.cols <= metalTextureLimit, frame.rows <= metalTextureLimit else {
            throw RendererError.textureTooLarge(max(frame.rows, frame.cols),
                                                limit: metalTextureLimit)
        }
        return try renderSurface(try surfaceResources(for: scene), view: scene.camera.view,
                                 frame: frame)
    }

    internal func renderSurface(_ res: SurfaceResources, view: Transform<WorldSpace, ViewSpace>,
                                frame: DepthFrame) throws -> SurfaceImage {
        let sentinel = Self.emptySentinel
        let m = frame.metalClip
        var uniforms = KVUniforms(
            view: view.float4x4,
            clip: simd_float4x4(SIMD4<Float>(m.columns.0), SIMD4<Float>(m.columns.1),
                                SIMD4<Float>(m.columns.2), SIMD4<Float>(m.columns.3)),
            domainLo: .zero, domainSize: .zero, lattice: .zero, gridSize: .zero,
            step: 1, cap: .infinity, regionCount: 0, empty: sentinel)
        // Padded, so the farthest point lands short of the cleared 1.0 and the
        // LESS test admits it.
        let (lo, hi) = res.viewDepth(view)
        let pad = 1e-3 * (hi - lo) + 1e-6
        var lattice = res.lattice
        lattice.depthRange = SIMD2(Float(lo - pad), Float(hi + pad))

        let depthTarget = try DepthTarget(rows: frame.rows, cols: frame.cols, device: device)
        let coordTarget = try DepthTarget(rows: frame.rows, cols: frame.cols, device: device,
                                          format: .rg32Float, bytesPerPixel: 8)
        let z = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: frame.cols, height: frame.rows, mipmapped: false)
        z.usage = .renderTarget
        z.storageMode = .private
        guard let zTexture = device.makeTexture(descriptor: z) else {
            throw RendererError.allocation("a \(frame.cols)x\(frame.rows) depth attachment")
        }
        guard let commands = queue.makeCommandBuffer() else {
            throw RendererError.allocation("a command buffer")
        }

        func draw(_ encoder: MTLRenderCommandEncoder, _ pipeline: MTLRenderPipelineState) {
            encoder.setRenderPipelineState(pipeline)
            encoder.setVertexBytes(&uniforms, length: MemoryLayout<KVUniforms>.stride, index: 0)
            encoder.setVertexBytes(&lattice, length: MemoryLayout<KVSurface>.stride, index: 6)
            encoder.setVertexTexture(res.positions, index: 0)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: res.cells * 6)
            encoder.endEncoding()
        }

        let depthPass = MTLRenderPassDescriptor()
        depthPass.colorAttachments[0].texture = depthTarget.texture
        depthPass.colorAttachments[0].loadAction = .clear
        depthPass.colorAttachments[0].storeAction = .store
        depthPass.colorAttachments[0].clearColor =
            MTLClearColor(red: Double(sentinel), green: 0, blue: 0, alpha: 0)
        guard let first = commands.makeRenderCommandEncoder(descriptor: depthPass) else {
            throw RendererError.allocation("a render encoder")
        }
        draw(first, paramDepthPipeline)

        let coordPass = MTLRenderPassDescriptor()
        coordPass.colorAttachments[0].texture = coordTarget.texture
        coordPass.colorAttachments[0].loadAction = .clear
        coordPass.colorAttachments[0].storeAction = .store
        coordPass.colorAttachments[0].clearColor =
            MTLClearColor(red: Double(sentinel), green: Double(sentinel), blue: 0, alpha: 0)
        coordPass.depthAttachment.texture = zTexture
        coordPass.depthAttachment.loadAction = .clear
        coordPass.depthAttachment.storeAction = .dontCare
        coordPass.depthAttachment.clearDepth = 1
        guard let second = commands.makeRenderCommandEncoder(descriptor: coordPass) else {
            throw RendererError.allocation("a render encoder")
        }
        second.setDepthStencilState(coordDepthState)
        draw(second, coordPipeline)

        commands.commit()
        commands.waitUntilCompleted()
        if let error = commands.error { throw RendererError.pipeline("\(error)") }

        let depth = depthTarget.read(frame: frame, empty: sentinel)
        let nothing = SIMD2<Float>(.nan, .nan)
        let coords = coordTarget.pixels(SIMD2<Float>.self).map { $0.x <= sentinel ? nothing : $0 }
        return SurfaceImage(depth: depth, coords: coords)
    }
}
