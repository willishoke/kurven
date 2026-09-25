import Foundation
import Dispatch
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
                // Separate segments, not a strip: a strip would join the end of
                // one path to the start of the next, which is precisely the
                // welding the CSR representation exists to prevent. Each pair is
                // one instance of the stroke quad.
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
    /// How many pixels' worth of the surface's own depth change the ink test
    /// allows on top of the margin. Without it, ink on a surface that is steep
    /// in view breaks into dashes that crawl as the camera moves. Zero is the
    /// bake's predicate exactly; see `ink_visible` in the shader for the rest.
    public var slopeScale: Float

    /// The widest layer's stroke, in pixels. Every other layer is drawn in
    /// proportion to its plate width, so the plate's hierarchy -- major lines
    /// over minor ones -- survives at any zoom.
    ///
    /// Constant on screen rather than scaled with zoom: the plate's widths are
    /// points on a sixteen-inch page, a fraction of a pixel on a window
    /// showing all of it, and a stroke that thins as you pull back reads as
    /// the picture fading rather than as the picture getting smaller. The app
    /// multiplies this by the display's backing scale, so a stroke is as wide
    /// on a Retina screen as on any other.
    public var inkWidth: Float

    /// A pixel's worth covers the half-pixel between where a line crosses a
    /// pixel and the pixel's centre, with room for the surface to bend in
    /// between. Measured with `kurven-cli flicker`, not guessed.
    public static let defaultSlopeScale: Float = 1
    public static let defaultInkWidth: Float = 1.5

    public init(mode: PreviewMode = .plate, visibleLayers: Set<Int>? = nil,
                background: SIMD4<Float> = SIMD4(1, 1, 1, 1),
                slopeScale: Float = PreviewOptions.defaultSlopeScale,
                inkWidth: Float = PreviewOptions.defaultInkWidth) {
        self.mode = mode; self.visibleLayers = visibleLayers; self.background = background
        self.slopeScale = slopeScale; self.inkWidth = inkWidth
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
    /// disagree on runs shorter than a pixel. The per-fragment test also allows
    /// `slopeScale` pixels' worth of the surface's depth change on top of the
    /// margin, because a fragment compares the line's depth where it crosses a
    /// pixel with the surface's depth at the pixel's centre, and on a steep
    /// surface those differ by more than the margin (see `ink_visible`). Those
    /// two are the only differences between them. Both are bounded by a pixel,
    /// and both are the right way round: the preview is for navigating and the
    /// bake is the artifact.
    func renderPreview(_ scene: Scene, navigator: Navigator, viewport: Viewport,
                       options: PreviewOptions = PreviewOptions(),
                       into target: MTLTexture,
                       commandBuffer: MTLCommandBuffer? = nil) throws {
        if let surface = scene.parametric {
            return try renderParametricPreview(scene, surface: surface, navigator: navigator,
                                               viewport: viewport, options: options,
                                               into: target, commandBuffer: commandBuffer)
        }
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
        // million samples to normalize one would cost more than the frame. Only
        // the depth view: it is still twenty thousand samples through the
        // camera, and the other modes were paying that every frame for nothing.
        let box = options.mode == .depth ? scene.quickBounds() : nil
        var shading = KVShading(
            color: SIMD4(0, 0, 0, 1),
            margin: Float(scene.margin),
            empty: Self.emptySentinel,
            slopeScale: options.slopeScale,
            lightDirection: SIMD3(0.4, -0.6, 0.7),
            ambient: 0.25,
            strokeWidth: options.inkWidth,
            depthRange: SIMD2(Float(box?.lo.z ?? 0), Float(box?.hi.z ?? 1)),
            viewport: SIMD2(Float(viewport.width), Float(viewport.height)))
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
            e2.setRenderPipelineState(strokePipeline)
            e2.setVertexBuffer(buffer, offset: 0, index: 4)
            e2.setFragmentTexture(depth, index: 1)
            // Every layer, visible or not, so hiding the widest one does not
            // make the rest jump wider.
            let widest = scene.layers.map(\.spec.width).max() ?? 0
            for (i, layer) in scene.layers.enumerated() {
                if let visible = options.visibleLayers, !visible.contains(i) { continue }
                let range = lines.ranges[i]
                guard range.count > 0 else { continue }
                // Unclipped ink is drawn with an unreachable margin rather than
                // a second pipeline: a cut-face hatch lies *in* the wall it
                // hatches, and a depth test would erase about half of it.
                shading.margin = layer.spec.clipped ? Float(scene.margin) : .infinity
                shading.color = Self.color(layer.spec.color)
                shading.strokeWidth = options.inkWidth
                    * Float(widest > 0 ? layer.spec.width / widest : 1)
                e2.setVertexBytes(&shading, length: MemoryLayout<KVShading>.stride, index: 1)
                e2.setFragmentBytes(&shading, length: MemoryLayout<KVShading>.stride, index: 1)
                // Four vertices a segment, one instance per segment, starting at
                // this layer's first. Passed rather than left to `baseInstance`
                // so what `instance_id` counts from is not a question.
                var first = UInt32(range.first / 2)
                e2.setVertexBytes(&first, length: MemoryLayout<UInt32>.stride, index: 5)
                e2.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4,
                                  instanceCount: range.count / 2)
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

public extension MetalRenderer {
    /// The view depth the last preview left at a pixel, or nil where nothing was
    /// drawn.
    func previewDepth(atPixel p: SIMD2<Int>) -> Double? {
        guard let texture = lastPreviewDepth,
              p.x >= 0, p.y >= 0, p.x < texture.width, p.y < texture.height else { return nil }
        var value: Float = 0
        withUnsafeMutableBytes(of: &value) { bytes in
            texture.getBytes(bytes.baseAddress!, bytesPerRow: MemoryLayout<Float>.stride,
                             from: MTLRegionMake2D(p.x, p.y, 1, 1), mipmapLevel: 0)
        }
        return value > Self.emptySentinel ? Double(value) : nil
    }

    /// The world point under a pixel: the depth the preview drew there, taken
    /// back through the camera.
    ///
    /// Under an orthographic camera the view coordinates of a pixel are known
    /// exactly from the framing, so the only unknown is depth -- and the depth
    /// buffer the preview already drew is holding it. Nothing is re-rendered and
    /// nothing is intersected.
    func worldPoint(atPixel p: SIMD2<Double>, navigator: Navigator,
                    viewport: Viewport) -> P3<WorldSpace>? {
        guard let z = previewDepth(atPixel: SIMD2(Int(p.x.rounded(.down)),
                                                  Int(p.y.rounded(.down)))) else { return nil }
        let v = navigator.framing.viewPoint(atPixel: p, in: viewport)
        return navigator.camera.view.inverse(P3<ViewSpace>(v.x, v.y, z))
    }
}

// MARK: - a parametric surface

/// Ink on a parametric surface, laid out for the surface stroke shader: each
/// vertex with its outward normal and its surface coordinate. Like
/// `LineGeometry`, one buffer and a range per layer; a layer without
/// coordinates has an empty range here and is drawn from `LineGeometry`.
struct SurfaceInkGeometry {
    let buffer: MTLBuffer?
    let ranges: [(first: Int, count: Int)]
    /// Each layer's vertices as laid out, kept so a later layout can take a
    /// layer whose ink did not change as it is -- the normals are the
    /// expensive part, and an edit to one layer's ink leaves the others'.
    let layers: [[KVInkVertex]]

    init(_ layers: [PolylineSet<WorldSpace>?], surface: ParametricSurface,
         onFolds: [Bool], device: MTLDevice) throws {
        try self.init(layers: zip(layers, onFolds).map { paths, folds in
            paths.map { Self.layout($0, surface: surface, onFolds: folds) } ?? []
        }, device: device)
    }

    /// One layer's ink, laid out: a segment per vertex pair, each end with
    /// its outward normal and its coordinate. Empty for ink without
    /// coordinates, which `LineGeometry` draws.
    static func layout(_ paths: PolylineSet<WorldSpace>, surface: ParametricSurface,
                       onFolds folds: Bool) -> [KVInkVertex] {
        guard let coords = paths.coords else { return [] }
        // Facing is not asked of fold ink, nor of a surface with no
        // outside; a zero normal says so to the shader. Otherwise the
        // map's own normal, once per vertex and in parallel: a forced
        // torus's map is a Fourier series of 1,225 terms, and its
        // trajectory has a quarter of a million vertices.
        var normals = [SIMD3<Float>](repeating: .zero, count: coords.count)
        if !folds, let outward = surface.outward {
            let chunk = 4096, n = coords.count
            normals.withUnsafeMutableBufferPointer { out in
                nonisolated(unsafe) let base = out.baseAddress!
                DispatchQueue.concurrentPerform(iterations: (n + chunk - 1) / chunk) { b in
                    for k in (b * chunk)..<min((b + 1) * chunk, n) {
                        base[k] = SIMD3<Float>(surface.map(coords[k]).normal * outward)
                    }
                }
            }
        }
        func vertex(_ k: Int) -> KVInkVertex {
            let c = coords[k]
            return KVInkVertex(position: SIMD3<Float>(paths.vertices[k].v),
                               normal: normals[k],
                               coord: SIMD2<Float>(Float(c.x), Float(c.y)))
        }
        var vertices: [KVInkVertex] = []
        vertices.reserveCapacity(2 * max(coords.count - paths.count, 0))
        for i in 0..<paths.count {
            let lo = paths.offsets[i], hi = paths.offsets[i + 1]
            for k in lo..<(hi - 1) {
                vertices.append(vertex(k))
                vertices.append(vertex(k + 1))
            }
        }
        return vertices
    }

    /// The layers' laid-out vertices, end to end in one buffer.
    init(layers: [[KVInkVertex]], device: MTLDevice) throws {
        self.layers = layers
        var vertices: [KVInkVertex] = []
        vertices.reserveCapacity(layers.reduce(0) { $0 + $1.count })
        var ranges: [(first: Int, count: Int)] = []
        for layer in layers {
            ranges.append((vertices.count, layer.count))
            vertices.append(contentsOf: layer)
        }
        self.ranges = ranges
        if vertices.isEmpty {
            buffer = nil
        } else {
            guard let b = vertices.withUnsafeBytes({ bytes in
                device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count,
                                  options: .storageModeShared)
            }) else { throw RendererError.allocation("surface ink vertices") }
            buffer = b
        }
    }
}

/// A parametric scene's ink for the preview: the layers that do not depend on
/// the camera, laid out once per ink, and the fold lines, laid out again
/// whenever the camera moves.
struct SurfaceInk {
    let statics: SurfaceInkGeometry
    let folds: SurfaceInkGeometry
    let onFolds: [Bool]

    /// Where layer `i` is drawn from, or nil when it has no coordinates.
    func draw(_ i: Int) -> (MTLBuffer, (first: Int, count: Int))? {
        let g = onFolds[i] ? folds : statics
        guard let buffer = g.buffer, g.ranges[i].count > 0 else { return nil }
        return (buffer, g.ranges[i])
    }
}

extension MetalRenderer {
    /// The preview of a parametric scene: `renderPreview` for a surface that
    /// is not a heightfield.
    ///
    /// The depth pass and the coordinate pass are the bake's, over the same
    /// position texture. The paper is the same lattice again. Ink that carries
    /// coordinates is drawn by the surface stroke shader, which asks the
    /// bake's two questions per fragment; the fold lines are derived for this
    /// camera every time it moves, because that is what they depend on.
    ///
    /// Under a perspective camera the folds are the orthographic ones for the
    /// view axis -- a preview approximation, like the rest of perspective
    /// here; the bake refuses perspective outright.
    func renderParametricPreview(_ scene: Scene, surface: ParametricSurface,
                                 navigator: Navigator, viewport: Viewport,
                                 options: PreviewOptions, into target: MTLTexture,
                                 commandBuffer: MTLCommandBuffer?) throws {
        var scene = scene
        scene.camera = navigator.camera
        let frame = navigator.framing.frame(viewport)
        let res = try surfaceResources(for: scene)
        let lines = try lineGeometry(for: scene)
        let depth = try depthTexture(rows: viewport.height, cols: viewport.width)
        let (coordTexture, zTexture) = try coordinateTextures(rows: viewport.height,
                                                              cols: viewport.width)

        let clip = clipMatrix(scene, frame: frame)
        var uniforms = KVUniforms(
            view: scene.camera.view.float4x4, clip: clip,
            domainLo: .zero, domainSize: .zero, lattice: .zero, gridSize: .zero,
            step: 1, cap: .infinity, regionCount: 0, empty: Self.emptySentinel)
        let bounds = scene.quickBounds()
        let (zlo, zhi) = (bounds?.lo.z ?? -1, bounds?.hi.z ?? 1)
        let pad = 0.05 * (zhi - zlo) + 1e-6
        var lattice = res.lattice
        lattice.depthRange = SIMD2(Float(zlo - pad), Float(zhi + pad))

        guard let commands = commandBuffer ?? queue.makeCommandBuffer() else {
            throw RendererError.allocation("a command buffer")
        }
        func drawSurface(_ e: MTLRenderCommandEncoder, _ pipeline: MTLRenderPipelineState) {
            e.setRenderPipelineState(pipeline)
            e.setVertexBytes(&uniforms, length: MemoryLayout<KVUniforms>.stride, index: 0)
            e.setVertexBytes(&lattice, length: MemoryLayout<KVSurface>.stride, index: 6)
            e.setVertexTexture(res.positions, index: 0)
            e.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: res.cells * 6)
        }

        // Pass 1: depth, MAX-blended, as the bake draws it.
        let depthPass = MTLRenderPassDescriptor()
        depthPass.colorAttachments[0].texture = depth
        depthPass.colorAttachments[0].loadAction = .clear
        depthPass.colorAttachments[0].storeAction = .store
        depthPass.colorAttachments[0].clearColor =
            MTLClearColor(red: Double(Self.emptySentinel), green: 0, blue: 0, alpha: 0)
        guard let e1 = commands.makeRenderCommandEncoder(descriptor: depthPass) else {
            throw RendererError.allocation("the depth encoder")
        }
        drawSurface(e1, paramDepthPipeline)
        e1.endEncoding()

        // Pass 1b: the front-most coordinate, behind its own depth test.
        let coordPass = MTLRenderPassDescriptor()
        coordPass.colorAttachments[0].texture = coordTexture
        coordPass.colorAttachments[0].loadAction = .clear
        coordPass.colorAttachments[0].storeAction = .store
        coordPass.colorAttachments[0].clearColor = MTLClearColor(
            red: Double(Self.emptySentinel), green: Double(Self.emptySentinel), blue: 0, alpha: 0)
        coordPass.depthAttachment.texture = zTexture
        coordPass.depthAttachment.loadAction = .clear
        coordPass.depthAttachment.storeAction = .dontCare
        coordPass.depthAttachment.clearDepth = 1
        guard let e1b = commands.makeRenderCommandEncoder(descriptor: coordPass) else {
            throw RendererError.allocation("the coordinate encoder")
        }
        e1b.setDepthStencilState(coordDepthState)
        drawSurface(e1b, coordPipeline)
        e1b.endEncoding()

        // Pass 2: the picture.
        let box = options.mode == .depth ? bounds : nil
        var shading = KVShading(
            color: SIMD4(0, 0, 0, 1), margin: Float(scene.margin), empty: Self.emptySentinel,
            slopeScale: options.slopeScale, lightDirection: SIMD3(0.4, -0.6, 0.7),
            ambient: 0.25, strokeWidth: options.inkWidth,
            depthRange: SIMD2(Float(box?.lo.z ?? 0), Float(box?.hi.z ?? 1)),
            viewport: SIMD2(Float(viewport.width), Float(viewport.height)))
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
        e2.setFragmentBytes(&shading, length: MemoryLayout<KVShading>.stride, index: 1)
        switch options.mode {
        case .plate:
            drawSurface(e2, paramPaperPipeline)
        case .shaded:
            e2.setFragmentTexture(depth, index: 1)
            drawSurface(e2, paramShadedPipeline)
        case .depth:
            e2.setRenderPipelineState(depthViewPipeline)
            e2.setFragmentTexture(depth, index: 1)
            e2.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        }

        if options.mode != .depth {
            let ink = try surfaceInk(for: scene, surface: surface)
            let isPerspective = scene.camera.isPerspective
            let eye = scene.camera.view.inverse(P3<ViewSpace>(0, 0, 0))
            e2.setVertexBytes(&uniforms, length: MemoryLayout<KVUniforms>.stride, index: 0)
            e2.setFragmentBytes(&lattice, length: MemoryLayout<KVSurface>.stride, index: 6)
            e2.setFragmentTexture(depth, index: 1)
            e2.setFragmentTexture(coordTexture, index: 2)
            let widest = scene.layers.map(\.spec.width).max() ?? 0
            for (i, layer) in scene.layers.enumerated() {
                if let visible = options.visibleLayers, !visible.contains(i) { continue }
                shading.color = Self.color(layer.spec.color)
                shading.strokeWidth = options.inkWidth
                    * Float(widest > 0 ? layer.spec.width / widest : 1)
                shading.margin = layer.spec.clipped ? Float(scene.margin) : .infinity
                if layer.spec.clipped, let (buffer, judged) = ink.draw(i) {
                    var camera = KVInk(sight: SIMD3<Float>(scene.camera.view.sightLine),
                                       eye: SIMD3<Float>(eye.v),
                                       perspective: isPerspective ? 1 : 0,
                                       onFolds: ink.onFolds[i] ? 1 : 0)
                    e2.setRenderPipelineState(surfaceStrokePipeline)
                    e2.setVertexBuffer(buffer, offset: 0, index: 4)
                    e2.setVertexBytes(&shading, length: MemoryLayout<KVShading>.stride, index: 1)
                    e2.setFragmentBytes(&shading, length: MemoryLayout<KVShading>.stride, index: 1)
                    e2.setVertexBytes(&camera, length: MemoryLayout<KVInk>.stride, index: 7)
                    e2.setFragmentBytes(&camera, length: MemoryLayout<KVInk>.stride, index: 7)
                    var first = UInt32(judged.first / 2)
                    e2.setVertexBytes(&first, length: MemoryLayout<UInt32>.stride, index: 5)
                    e2.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4,
                                      instanceCount: judged.count / 2)
                } else if let buffer = lines.buffer, lines.ranges[i].count > 0 {
                    // Ink without coordinates, judged by depth as anywhere else.
                    let range = lines.ranges[i]
                    e2.setRenderPipelineState(strokePipeline)
                    e2.setVertexBuffer(buffer, offset: 0, index: 4)
                    e2.setVertexBytes(&shading, length: MemoryLayout<KVShading>.stride, index: 1)
                    e2.setFragmentBytes(&shading, length: MemoryLayout<KVShading>.stride, index: 1)
                    var first = UInt32(range.first / 2)
                    e2.setVertexBytes(&first, length: MemoryLayout<UInt32>.stride, index: 5)
                    e2.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4,
                                      instanceCount: range.count / 2)
                }
            }
        }
        e2.endEncoding()

        if commandBuffer == nil {
            commands.commit()
            commands.waitUntilCompleted()
            if let error = commands.error { throw RendererError.pipeline("\(error)") }
        }
    }
}
