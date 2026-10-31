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

    init(scene: Scene, device: MTLDevice, draft: Bool = false) throws {
        let laid = Self.layOut(scene: scene, draft: draft, reusing: nil)
        try self.init(laid, device: device)
    }

    /// Whether a layer's line geometry is the camera's: a heightfield's fold
    /// lines, or the ink in its walls; or, on a parametric surface that
    /// carries its solid, its fold curves and every clipped layer that knows
    /// where on the surface it lies. Sphere tracing sights along parallel lines,
    /// so under perspective a surface's ink stays with the stroke shader.
    static func isTestedExactly(_ layer: Layer, in scene: Scene) -> Bool {
        if let s = scene.parametric {
            guard s.neighbourhood != nil, !scene.camera.isPerspective else { return false }
            if case .foldCurves = layer.spec.source { return true }
            return layer.spec.clipped && layer.paths.coords != nil
        }
        guard scene.heightfield != nil else { return false }
        if case .foldCurves = layer.spec.source { return true }
        return layer.spec.source.isWallInk && layer.spec.clipped
    }

    /// Whether the scene has ink that is tested for the camera: a
    /// heightfield's fold curves and the ink in its walls, or a sphere-traced
    /// surface's ink. Such a scene's line geometry is the camera's, and is
    /// laid out again when it moves.
    static func isTestedExactly(_ scene: Scene) -> Bool {
        scene.layers.contains { isTestedExactly($0, in: scene) }
    }

    /// Every layer's segments, end to end, with a layer's range into them.
    /// The CPU side of the geometry, apart from the renderer so it can be
    /// done on another thread; `init(_:device:)` uploads it.
    ///
    /// A heightfield's fold curves are the camera's, traced here for this
    /// one -- on f and refined onto the exact fold, as the bake's are --
    /// and tested by the solid here rather than per fragment:
    /// the surface is edge-on at a fold, which is where a depth test is
    /// worth nothing. Wall ink likewise: tested by the wall it lies in and
    /// the solid. Both are drawn unclipped.
    ///
    /// A `draft` is what a frame can afford while the camera moves: the
    /// folds traced on the grid, not on f, and left to the depth test
    /// with the wall ink, which is drawn as stored. On a landscape the
    /// tested layout is hundreds of milliseconds and the draft is one, so
    /// a drag draws drafts and the tested layout follows when the camera
    /// rests.
    ///
    /// A layer that is not the camera's is taken from `reusing` -- the same
    /// ink's last layout -- as it is: a contour's normals are read off f,
    /// four evaluations a vertex, and the camera does not move them.
    static func layOut(scene: Scene, draft: Bool, reusing previous: VisibleLines?) -> VisibleLines {
        var layers: [[KVLineVertex]] = []
        layers.reserveCapacity(scene.layers.count)
        for (i, layer) in scene.layers.enumerated() {
            if !isTestedExactly(layer, in: scene), let previous, i < previous.layers.count {
                layers.append(previous.layers[i])
                continue
            }
            var vertices: [KVLineVertex] = []
            var paths = layer.paths
            if let s = scene.parametric, isTestedExactly(layer, in: scene) {
                // A sphere-traced surface's ink at rest is the bake's: the folds
                // on the exact fold, and every vertex tested by sphere tracing
                // against the solid. Per fragment against the coordinate
                // image, the folds and the ink running along them drew as
                // dashes wherever the front-most coordinate at a pixel
                // flipped between the near sheet and the one behind it,
                // which at a fold is everywhere. The draft is drawn from the
                // surface ink, not from here.
                paths = draft ? PolylineSet(paths: [])
                    : Self.sphereTraced(layer, on: s, view: scene.camera.view, creases: scene.creases)
            } else if case .foldCurves = layer.spec.source, let h = scene.heightfield {
                // At rest the folds are the bake's: every vertex moved onto
                // the exact fold of f before sphere tracing tests it against f.
                // Traced on the grid and left there, a vertex lies up to
                // a cell from the outline, which on a steep graph is far
                // more than sphere tracing's depth tolerance on f, so vertices fell
                // alternately in front of the graph and behind it and the
                // fold drew as dashes.
                paths = draft
                    ? h.onGrid.foldCurves(view: scene.camera.view, refined: false)
                    : h.visibleFolds(view: scene.camera.view, depthTolerance: scene.depthTolerance,
                                     resolution: scene.resolution)
            } else if layer.spec.source.isWallInk, layer.spec.clipped, !draft,
                      let h = scene.heightfield {
                paths = h.visibleWallInk(layer.paths, view: scene.camera.view, depthTolerance: scene.depthTolerance,
                                         resolution: scene.resolution)
            }
            // Ink on a heightfield's surface carries the surface's outward
            // normal, so the stroke shader can hide it where the surface
            // faces away; a zero normal is ink tested by depth alone. Once
            // per vertex, in parallel: zeta's contours have six hundred
            // thousand, and each normal is four height reads.
            var normals: [SIMD3<Float>]?
            if let h = scene.heightfield, layer.spec.liesOnSurface {
                let points = paths.vertices
                var out = [SIMD3<Float>](repeating: .zero, count: points.count)
                let chunk = 4096, n = points.count
                out.withUnsafeMutableBufferPointer { buffer in
                    nonisolated(unsafe) let base = buffer.baseAddress!
                    DispatchQueue.concurrentPerform(iterations: (n + chunk - 1) / chunk) { b in
                        for k in (b * chunk)..<min((b + 1) * chunk, n) {
                            base[k] = SIMD3<Float>(h.normal(at: P2(points[k].x, points[k].y)))
                        }
                    }
                }
                normals = out
            }
            for i in 0..<paths.count {
                let lo = paths.offsets[i], hi = paths.offsets[i + 1]
                // Separate segments, not a strip: a strip would join the end of
                // one path to the start of the next, which is precisely the
                // welding the CSR representation exists to prevent. Each pair is
                // one instance of the stroke quad.
                for k in lo..<(hi - 1) {
                    for end in [k, k + 1] {
                        vertices.append(KVLineVertex(
                            position: SIMD3<Float>(paths.vertices[end].v),
                            normal: normals?[end] ?? .zero))
                    }
                }
            }
            layers.append(vertices)
        }
        // Sphere tracing has no pixel in it, so a sphere-traced surface's layout serves
        // every zoom.
        let resolution = draft ? .infinity : scene.parametric != nil ? 0 : scene.resolution
        return VisibleLines(layers: layers, resolution: resolution)
    }

    /// A layer of a sphere-traced surface's ink, as the bake tests it: what of it
    /// the solid leaves in sight, in world space for the stroke shader.
    static func sphereTraced(_ layer: Layer, on surface: ParametricSurface,
                        view: Transform<WorldSpace, ViewSpace>,
                        creases: [[P2<ParamSpace>]] = []) -> PolylineSet<WorldSpace> {
        let paths: PolylineSet<WorldSpace>, folds: Bool
        if case .foldCurves = layer.spec.source {
            paths = surface.foldCurves(sight: view.projectionDirection, creases: creases)
            folds = true
        } else {
            paths = layer.paths
            folds = false
        }
        // An outline lies on a fold or a crease, where the normal says
        // nothing: sphere tracing alone tests it, as the bake does.
        let visibility = SurfaceVisibility(sphereTracing: surface, view: view)
        let seen = HiddenLine.clip(paths.mapped(view), on: visibility,
                                   onFolds: layer.spec.role == .outline,
                                   onCrease: layer.spec.role == .outline && !folds)
        return PolylineSet(vertices: seen.vertices.map { view.inverse(P3<ViewSpace>($0.v)) },
                           offsets: seen.offsets)
    }

    /// `layOut`'s result, uploaded: every layer's segments end to end.
    init(_ laid: VisibleLines, device: MTLDevice) throws {
        var ranges: [(first: Int, count: Int)] = []
        var vertices: [KVLineVertex] = []
        vertices.reserveCapacity(laid.layers.reduce(0) { $0 + $1.count })
        for layer in laid.layers {
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
            }) else { throw RendererError.allocation("line vertices") }
            buffer = b
        }
    }

    var vertexCount: Int { ranges.reduce(0) { $0 + $1.count } }
}

/// Line geometry laid out and not yet uploaded, a layer at a time: the part
/// of a frame's ink that can be made away from the renderer, and kept for
/// the layers the next camera leaves alone.
public struct VisibleLines: Sendable {
    let layers: [[KVLineVertex]]
    /// The `Scene.resolution` the ink was tested to: a layout serves a scene
    /// that asks no finer, so zooming out keeps it and zooming in tests again.
    let resolution: Double

    func serves(_ scene: Scene) -> Bool { resolution <= scene.resolution }
}

public extension MetalRenderer {
    /// The tested line geometry of a scene, laid out: what `renderPreview`
    /// would make for the camera when `draft` is off, made anywhere. A
    /// window that draws drafts while the camera moves lays this out on
    /// another thread once it rests, and `install`s it.
    static func visibleLines(for scene: Scene, reusing previous: VisibleLines?) -> VisibleLines {
        LineGeometry.layOut(scene: scene, draft: false, reusing: previous)
    }

    /// The scene's ink as last laid out, for `visibleLines` to keep what the
    /// camera leaves alone; nil when the last layout was another's.
    func laidLines(for scene: Scene) -> VisibleLines? {
        guard let c = cachedLines, c.ink == scene.ink else { return nil }
        return c.laid
    }

    /// Keep a tested layout for the next frame of its scene.
    func install(_ lines: VisibleLines, for scene: Scene) throws {
        let key: Camera? = LineGeometry.isTestedExactly(scene) ? scene.camera : nil
        cachedLines = (scene.ink, key, false, lines, try LineGeometry(lines, device: device))
    }

    /// Whether the next frame of this scene can be drawn tested without
    /// laying its ink out first: the layout is in hand, or nothing in the
    /// scene is tested for the camera.
    func hasJudgedLines(for scene: Scene) -> Bool {
        guard LineGeometry.isTestedExactly(scene) else { return true }
        guard let c = cachedLines, c.ink == scene.ink, c.camera == scene.camera else { return false }
        return !c.draft && c.laid.serves(scene)
    }
}

/// What a preview frame draws.
public struct PreviewOptions: Sendable {
    public var mode: PreviewMode
    /// Which layers to draw, by index into `Scene.layers`. `nil` draws them all.
    public var visibleLayers: Set<Int>?
    /// Paper colour. The plate's surface is white because the page is.
    public var background: SIMD4<Float>
    /// How many pixels' worth of the surface's own depth change the ink test
    /// allows on top of the depth tolerance. Without it, ink on a surface that is steep
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

    /// Draw the frame a moving camera can afford: a heightfield's fold
    /// lines traced on the grid and left to the depth test, its wall ink
    /// as stored. Off, they are tested by the solid for this camera, which
    /// on a landscape is hundreds of milliseconds of CPU per camera; see
    /// `LineGeometry.layOut`.
    public var draft: Bool

    /// A pixel's worth covers the half-pixel between where a line crosses a
    /// pixel and the pixel's centre, with room for the surface to bend in
    /// between. Measured with `kurven-cli flicker`, not guessed.
    public static let defaultSlopeScale: Float = 1
    public static let defaultInkWidth: Float = 1.5

    public init(mode: PreviewMode = .plate, visibleLayers: Set<Int>? = nil,
                background: SIMD4<Float> = SIMD4(1, 1, 1, 1),
                slopeScale: Float = PreviewOptions.defaultSlopeScale,
                inkWidth: Float = PreviewOptions.defaultInkWidth,
                draft: Bool = false) {
        self.mode = mode; self.visibleLayers = visibleLayers; self.background = background
        self.slopeScale = slopeScale; self.inkWidth = inkWidth; self.draft = draft
    }
}

public extension MetalRenderer {
    /// Draw one preview frame into `target`.
    ///
    /// Two passes over the geometry the bake uses. The first is the depth pass,
    /// unchanged and shared. The second draws the surface (opaque, so ink behind
    /// it does not show through the outline) and then the ink, deciding
    /// visibility by reading that depth texture at each fragment's own pixel.
    ///
    /// The bake tests per vertex and this tests per fragment, so the two can
    /// disagree on runs shorter than a pixel. The per-fragment test also allows
    /// `slopeScale` pixels' worth of the surface's depth change on top of the
    /// depth tolerance, because a fragment compares the line's depth where it crosses a
    /// pixel with the surface's depth at the pixel's centre, and on a steep
    /// surface those differ by more than the depth tolerance (see `ink_visible`). Those
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
        let lines = try lineGeometry(for: scene, draft: options.draft)
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
            depthTolerance: Float(scene.depthTolerance),
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
            // Which way the ink's surface faces is asked of the camera: ink
            // on the heightfield carries the normal, and the shader hides
            // what faces away.
            var camera = KVInk(sight: SIMD3<Float>(scene.camera.view.projectionDirection),
                               eye: SIMD3<Float>(scene.camera.view.inverse(P3<ViewSpace>(0, 0, 0)).v),
                               perspective: scene.camera.isPerspective ? 1 : 0, onFolds: 0)
            e2.setVertexBytes(&camera, length: MemoryLayout<KVInk>.stride, index: 7)
            // Every layer, visible or not, so hiding the widest one does not
            // make the rest jump wider.
            let widest = scene.layers.map(\.spec.width).max() ?? 0
            for (i, layer) in scene.layers.enumerated() {
                if let visible = options.visibleLayers, !visible.contains(i) { continue }
                let range = lines.ranges[i]
                guard range.count > 0 else { continue }
                // Unclipped ink is drawn with an unreachable depth tolerance rather than
                // a second pipeline: a cut-face hatch lies *in* the wall it
                // hatches, and a depth test would erase about half of it. A
                // heightfield's folds and wall ink were tested on the way in,
                // unless this frame is a draft, which leaves them to the test.
                var visible = scene.heightfield != nil && layer.spec.source.isWallInk
                if case .foldCurves = layer.spec.source { visible = true }
                if options.draft { visible = false }
                shading.depthTolerance = layer.spec.clipped && !visible ? Float(scene.depthTolerance) : .infinity
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
    ///
    /// The normals are read off `grid` -- the surface's grid normals,
    /// interpolated -- when given, and asked of the map otherwise. The
    /// grid's are within a pixel of the map's at a drawing grid and
    /// cost four reads where a forced torus's map costs a Fourier series of
    /// 1,225 terms: the difference between a trajectory of three hundred
    /// thousand vertices laid out in a hundred milliseconds and in ten.
    static func layout(_ paths: PolylineSet<WorldSpace>, surface: ParametricSurface,
                       onFolds folds: Bool, grid: [SIMD3<Float>]? = nil) -> [KVInkVertex] {
        guard let coords = paths.coords else { return [] }
        // Facing is not asked of fold ink, nor of a surface with no
        // outside; a zero normal says so to the shader. Otherwise a normal
        // once per vertex, in parallel.
        var normals = [SIMD3<Float>](repeating: .zero, count: coords.count)
        if !folds, let outward = surface.outward {
            let chunk = 4096, n = coords.count, sign = Float(outward)
            normals.withUnsafeMutableBufferPointer { out in
                nonisolated(unsafe) let base = out.baseAddress!
                DispatchQueue.concurrentPerform(iterations: (n + chunk - 1) / chunk) { b in
                    for k in (b * chunk)..<min((b + 1) * chunk, n) {
                        base[k] = grid.map { surface.interpolateNormal(coords[k], normals: $0) * sign }
                            ?? SIMD3<Float>(surface.map(coords[k]).normal * outward)
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
/// the camera, laid out once per ink, and the fold curves, laid out again
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
    /// position texture. The paper is the same grid again. Ink that carries
    /// coordinates is drawn by the surface stroke shader, which asks the
    /// bake's two questions per fragment; the fold curves are derived for this
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
        let lines = try lineGeometry(for: scene, draft: options.draft)
        let depth = try depthTexture(rows: viewport.height, cols: viewport.width)
        let (coordTexture, zTexture) = try coordinateTextures(rows: viewport.height,
                                                              cols: viewport.width)

        let clip = clipMatrix(scene, frame: frame)
        var uniforms = KVUniforms(
            view: scene.camera.view.float4x4, clip: clip,
            domainLo: .zero, domainSize: .zero, grid: .zero, gridSize: .zero,
            step: 1, truncationHeight: .infinity, regionCount: 0, empty: Self.emptySentinel)
        let bounds = scene.quickBounds()
        let (zlo, zhi) = (bounds?.lo.z ?? -1, bounds?.hi.z ?? 1)
        let pad = 0.05 * (zhi - zlo) + 1e-6
        var grid = res.grid
        grid.depthRange = SIMD2(Float(zlo - pad), Float(zhi + pad))

        guard let commands = commandBuffer ?? queue.makeCommandBuffer() else {
            throw RendererError.allocation("a command buffer")
        }
        func drawSurface(_ e: MTLRenderCommandEncoder, _ pipeline: MTLRenderPipelineState) {
            e.setRenderPipelineState(pipeline)
            e.setVertexBytes(&uniforms, length: MemoryLayout<KVUniforms>.stride, index: 0)
            e.setVertexBytes(&grid, length: MemoryLayout<KVSurface>.stride, index: 6)
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
            color: SIMD4(0, 0, 0, 1), depthTolerance: Float(scene.depthTolerance), empty: Self.emptySentinel,
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
            e2.setFragmentBytes(&grid, length: MemoryLayout<KVSurface>.stride, index: 6)
            e2.setFragmentTexture(depth, index: 1)
            e2.setFragmentTexture(coordTexture, index: 2)
            let widest = scene.layers.map(\.spec.width).max() ?? 0
            for (i, layer) in scene.layers.enumerated() {
                if let visible = options.visibleLayers, !visible.contains(i) { continue }
                shading.color = Self.color(layer.spec.color)
                shading.strokeWidth = options.inkWidth
                    * Float(widest > 0 ? layer.spec.width / widest : 1)
                shading.depthTolerance = layer.spec.clipped ? Float(scene.depthTolerance) : .infinity
                // At rest a sphere-traced surface's ink comes tested by the solid,
                // and is drawn as it comes.
                let sphereTraced = !options.draft && LineGeometry.isTestedExactly(layer, in: scene)
                if sphereTraced { shading.depthTolerance = .infinity }
                if layer.spec.clipped, !sphereTraced, let (buffer, visible) = ink.draw(i) {
                    var camera = KVInk(sight: SIMD3<Float>(scene.camera.view.projectionDirection),
                                       eye: SIMD3<Float>(eye.v),
                                       perspective: isPerspective ? 1 : 0,
                                       onFolds: ink.onFolds[i] ? 1 : 0)
                    e2.setRenderPipelineState(surfaceStrokePipeline)
                    e2.setVertexBuffer(buffer, offset: 0, index: 4)
                    e2.setVertexBytes(&shading, length: MemoryLayout<KVShading>.stride, index: 1)
                    e2.setFragmentBytes(&shading, length: MemoryLayout<KVShading>.stride, index: 1)
                    e2.setVertexBytes(&camera, length: MemoryLayout<KVInk>.stride, index: 7)
                    e2.setFragmentBytes(&camera, length: MemoryLayout<KVInk>.stride, index: 7)
                    var first = UInt32(visible.first / 2)
                    e2.setVertexBytes(&first, length: MemoryLayout<UInt32>.stride, index: 5)
                    e2.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4,
                                      instanceCount: visible.count / 2)
                } else if let buffer = lines.buffer, lines.ranges[i].count > 0 {
                    // Ink without coordinates, tested by depth as anywhere
                    // else; or tested already.
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
