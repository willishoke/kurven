import Foundation
import Metal
import KurvenCore
import KurvenShaderTypes

/// Line geometry for the preview, laid out once per scene content.
///
/// One buffer holds every layer's segments end to end, and a layer is a range in
/// it. Toggling a layer is then draw-call selection rather than a re-upload,
/// which is the difference between a visibility checkbox that is instant and one
/// that stutters on zeta's six hundred thousand vertices.
struct LineGeometry {
    let buffer: MTLBuffer?
    /// `(first vertex, count)` per layer, in the scene's layer order.
    let ranges: [(first: Int, count: Int)]

    init(scene: Scene, device: MTLDevice) throws {
        var vertices: [KVVertex] = []
        var ranges: [(first: Int, count: Int)] = []
        for layer in scene.layers {
            let first = vertices.count
            for i in 0..<layer.paths.count {
                let path = layer.paths[path: i]
                // `.line` primitives, not a strip: a strip would join the end of
                // one path to the start of the next, which is precisely the
                // welding the CSR representation exists to prevent.
                for (a, b) in zip(path, path.dropFirst()) {
                    vertices.append(KVVertex(position: SIMD3<Float>(a.v)))
                    vertices.append(KVVertex(position: SIMD3<Float>(b.v)))
                }
            }
            ranges.append((first, vertices.count - first))
        }
        self.ranges = ranges
        if vertices.isEmpty {
            buffer = nil
        } else {
            guard let b = vertices.withUnsafeBytes({ bytes in
                device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count,
                                  options: .storageModeShared)
            }) else { throw RendererError.allocation("line vertices") }
            buffer = b
        }
    }

    var vertexCount: Int { ranges.reduce(0) { $0 + $1.count } }
}

/// What a preview frame draws.
public struct PreviewOptions: Sendable {
    public var mode: PreviewMode
    /// Which layers to draw, by index into `Scene.layers`. `nil` draws them all.
    public var visibleLayers: Set<Int>?
    /// Paper colour. The plate's surface is white because the page is.
    public var background: SIMD4<Float>

    public init(mode: PreviewMode = .plate, visibleLayers: Set<Int>? = nil,
                background: SIMD4<Float> = SIMD4(1, 1, 1, 1)) {
        self.mode = mode; self.visibleLayers = visibleLayers; self.background = background
    }
}

public extension MetalRenderer {
    /// Draw one preview frame into `target`.
    ///
    /// Two passes over the geometry the bake uses. The first is the depth pass,
    /// unchanged and shared. The second draws the surface (opaque, so ink behind
    /// it does not show through the silhouette) and then the ink, deciding
    /// visibility by reading that depth texture at each fragment's own pixel.
    ///
    /// The bake tests per vertex and this tests per fragment, so the two can
    /// disagree on runs shorter than a pixel. That is the only difference
    /// between them, it is bounded by construction, and it is the right way
    /// round: the preview is for navigating and the bake is the artifact.
    func renderPreview(_ scene: Scene, navigator: Navigator, viewport: Viewport,
                       options: PreviewOptions = PreviewOptions(),
                       into target: MTLTexture,
                       commandBuffer: MTLCommandBuffer? = nil) throws {
        var scene = scene
        scene.camera = navigator.camera
        let frame = navigator.framing.frame(viewport)

        let res = try resources(for: scene)
        let lines = try lineGeometry(for: scene)
        let depth = try depthTexture(rows: viewport.height, cols: viewport.width)
        var uniforms = self.previewUniforms(scene, frame: frame, resources: res)

        guard let commands = commandBuffer ?? queue.makeCommandBuffer() else {
            throw RendererError.allocation("a command buffer")
        }

        // Pass 1: depth only. Same geometry, same MAX blend, same numbers the
        // bake clips against.
        let depthPass = MTLRenderPassDescriptor()
        depthPass.colorAttachments[0].texture = depth
        depthPass.colorAttachments[0].loadAction = .clear
        depthPass.colorAttachments[0].storeAction = .store
        depthPass.colorAttachments[0].clearColor =
            MTLClearColor(red: Double(Self.emptySentinel), green: 0, blue: 0, alpha: 0)
        guard let e1 = commands.makeRenderCommandEncoder(descriptor: depthPass) else {
            throw RendererError.allocation("the depth encoder")
        }
        e1.setVertexBytes(&uniforms, length: MemoryLayout<KVUniforms>.stride, index: 0)
        encodeGeometry(e1, res: res, height: heightPipeline, mesh: meshPipeline)
        e1.endEncoding()

        // Pass 2: the picture.
        // The depth view needs a range to map to black and white. The cheap
        // box bound is right for it: it is a debug picture, and scanning six
        // million samples to normalize one would cost more than the frame.
        let box = scene.quickBounds()
        var shading = KVShading(
            color: SIMD4(0, 0, 0, 1),
            margin: Float(scene.margin),
            empty: Self.emptySentinel,
            lightDirection: SIMD3(0.4, -0.6, 0.7),
            ambient: 0.25,
            depthRange: SIMD2(Float(box?.lo.z ?? 0), Float(box?.hi.z ?? 1)))
        if case .shaded(let lighting) = options.mode {
            shading.lightDirection = lighting.direction
            shading.ambient = lighting.ambient
        }

        let colorPass = MTLRenderPassDescriptor()
        colorPass.colorAttachments[0].texture = target
        colorPass.colorAttachments[0].loadAction = .clear
        colorPass.colorAttachments[0].storeAction = .store
        colorPass.colorAttachments[0].clearColor = MTLClearColor(
            red: Double(options.background.x), green: Double(options.background.y),
            blue: Double(options.background.z), alpha: Double(options.background.w))
        guard let e2 = commands.makeRenderCommandEncoder(descriptor: colorPass) else {
            throw RendererError.allocation("the colour encoder")
        }
        e2.setVertexBytes(&uniforms, length: MemoryLayout<KVUniforms>.stride, index: 0)
        e2.setFragmentBytes(&shading, length: MemoryLayout<KVShading>.stride, index: 1)

        switch options.mode {
        case .plate:
            encodeGeometry(e2, res: res, height: paperSurfacePipeline, mesh: paperWallPipeline)
        case .shaded:
            e2.setFragmentTexture(depth, index: 1)
            encodeGeometry(e2, res: res, height: shadedSurfacePipeline, mesh: shadedWallPipeline)
        case .depth:
            // The depth attachment *is* the picture: a full-screen triangle maps
            // it to grey. This is the substitute for the frame-capture viewer
            // Command Line Tools does not ship, and it is most of what such a
            // viewer gets used for.
            e2.setRenderPipelineState(depthViewPipeline)
            e2.setFragmentTexture(depth, index: 1)
            e2.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        }

        if options.mode != .depth, let buffer = lines.buffer {
            e2.setRenderPipelineState(linePipeline)
            e2.setVertexBuffer(buffer, offset: 0, index: 4)
            e2.setFragmentTexture(depth, index: 1)
            for (i, layer) in scene.layers.enumerated() {
                if let visible = options.visibleLayers, !visible.contains(i) { continue }
                let range = lines.ranges[i]
                guard range.count > 0 else { continue }
                // Unclipped ink is drawn with an unreachable margin rather than
                // a second pipeline: a cut-face hatch lies *in* the wall it
                // hatches, and a depth test would erase about half of it.
                shading.margin = layer.spec.clipped ? Float(scene.margin) : .infinity
                shading.color = Self.color(layer.spec.color)
                e2.setFragmentBytes(&shading, length: MemoryLayout<KVShading>.stride, index: 1)
                e2.drawPrimitives(type: .line, vertexStart: range.first,
                                  vertexCount: range.count)
            }
        }
        e2.endEncoding()

        if commandBuffer == nil {
            commands.commit()
            commands.waitUntilCompleted()
            if let error = commands.error { throw RendererError.pipeline("\(error)") }
        }
    }

    /// An offscreen target the preview can draw into and `PNG` can read back.
    /// Offered here so a caller does not need to import Metal to ask for a
    /// picture -- the layering is `Core <- Metal <- Bake <- App`, and the CLI
    /// sits on Bake.
    func makePreviewTarget(_ viewport: Viewport) throws -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: previewFormat, width: viewport.width,
            height: viewport.height, mipmapped: false)
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .shared
        guard let t = device.makeTexture(descriptor: d) else {
            throw RendererError.allocation(
                "a \(viewport.width)x\(viewport.height) preview target")
        }
        return t
    }

    /// `#rrggbb` to linear-ish float. Preview only; the SVG carries the string.
    static func color(_ hex: String) -> SIMD4<Float> {
        var v = hex.hasPrefix("#") ? String(hex.dropFirst()) : hex
        if v.count == 3 { v = v.map { "\($0)\($0)" }.joined() }
        guard v.count == 6, let n = UInt32(v, radix: 16) else { return SIMD4(0, 0, 0, 1) }
        return SIMD4(Float((n >> 16) & 0xFF) / 255, Float((n >> 8) & 0xFF) / 255,
                     Float(n & 0xFF) / 255, 1)
    }
}
