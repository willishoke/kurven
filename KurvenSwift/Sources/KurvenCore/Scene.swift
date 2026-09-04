import Foundation
import simd

/// An immutable snapshot of everything a frame needs.
///
/// The renderer's signature is `draw(scene:viewport:)`: a frame is a function of
/// a value, not of accumulated state. Moving the camera means producing a new
/// `Scene` that differs in one field; the GPU buffers behind it are a memo keyed
/// by the bundle's identity, not something the scene mutates.
public struct Scene: Sendable {
    public var surface: Surface
    /// Wall curtains, in world coordinates.
    public var occluder: Mesh<WorldSpace>
    /// Heightfield instances; always at least the identity.
    public var tiles: [Affine2]
    /// The footprint each instance is clipped to.
    public var region: Region
    /// Subsampling step for the heightfield when it is rasterized.
    public var step: Int
    public var layers: [Layer]
    public var camera: Camera
    public var mode: PreviewMode
    /// Hidden-line margin, in world units of depth.
    public var margin: Double

    public init(surface: Surface, occluder: Mesh<WorldSpace>, tiles: [Affine2],
                region: Region = .full, step: Int, layers: [Layer], camera: Camera,
                mode: PreviewMode = .plate, margin: Double) {
        self.surface = surface; self.occluder = occluder; self.tiles = tiles
        self.region = region; self.step = step; self.layers = layers
        self.camera = camera; self.mode = mode; self.margin = margin
    }

    /// The scene a bundle describes under one of its presets.
    public init(bundle: KurvenBundle, preset: CameraPreset) {
        self.init(surface: bundle.surface,
                  occluder: bundle.walls(),
                  tiles: bundle.manifest.occluder.tiles,
                  region: bundle.manifest.occluder.region,
                  step: bundle.manifest.occluder.step,
                  layers: bundle.layers,
                  camera: .plate(preset.plate),
                  margin: preset.margin)
    }

    /// Every layer's vertices in view space, in declaration (draw) order.
    public func projectedLayers() -> [(Layer, PolylineSet<ViewSpace>)] {
        layers.map { ($0, $0.paths.mapped(camera.view)) }
    }

    /// The view-space extent the depth buffer covers: every heightfield sample,
    /// the walls, and every *clipped* layer.
    ///
    /// This is `ZBuffer(xs.min(), xs.max(), ys.min(), ys.max(), ...)` on the
    /// Python side, and it has to be the same extent or the two rasterize onto
    /// different pixel lattices and every comparison downstream is measuring the
    /// framing rather than the drawing. Two consequences of matching it exactly:
    ///
    /// - The heightfield is scanned sample by sample, not bounded by its box.
    ///   The box of a rotated landscape is looser than the hull of its samples,
    ///   and the difference is visible at bake resolution.
    /// - Masked-out samples still count. `build_occluder` emits every lattice
    ///   vertex and drops only *triangles*, so `occ_rot` includes the vertices
    ///   inside zeta's cutout even though nothing references them. Excluding
    ///   them here would be more principled and would not be the same picture.
    ///
    /// Unclipped ink is excluded, as it is in Python: it is drawn but never
    /// looked up, so letting it stretch the frame would spend depth resolution
    /// on nothing and would make the buffer depend on decoration.
    public func viewBounds() -> AABB<ViewSpace>? {
        var lo = SIMD3<Double>(repeating: .infinity)
        var hi = SIMD3<Double>(repeating: -.infinity)
        var any = false
        func add(_ p: P3<ViewSpace>) {
            lo = simd_min(lo, p.v); hi = simd_max(hi, p.v); any = true
        }
        for p in occluder.vertices { add(camera.view(p)) }
        forEachHeightfieldSample { add(camera.view($0)) }
        for (layer, projected) in projectedLayers() where layer.spec.clipped {
            for v in projected.vertices { add(v) }
        }
        return any ? AABB(lo: lo, hi: hi) : nil
    }

    /// Every vertex of the decimated, capped heightfield, once per tile.
    ///
    /// The lattice is `clamped().decimated(by: step)`, which is exactly
    /// `Surface.grid_mesh`'s `self.clamped[::step, ::step]`. Not materialized:
    /// elliptic's is three million points and the caller only ever folds over
    /// them.
    public func forEachHeightfieldSample(_ body: (P3<WorldSpace>) -> Void) {
        let lattice = surface.clamped().decimated(by: step)
        for tile in tiles {
            for y in 0..<lattice.height {
                for x in 0..<lattice.width {
                    let p = lattice.position(x: x, y: y)
                    body(tile(P3(p.x, p.y, Double(lattice[x, y]))))
                }
            }
        }
    }
}

/// What a preview draws.
public enum PreviewMode: Sendable, Equatable {
    /// The plate: white occluding surface, black hidden-line ink.
    case plate
    /// A lit heightfield, for orientation.
    case shaded(Lighting)
    /// The depth attachment itself -- the substitute for a frame-capture
    /// viewer, and the reason not having Xcode costs nothing here.
    case depth
}

public struct Lighting: Sendable, Equatable {
    public var direction: SIMD3<Float>
    public var ambient: Float
    public init(direction: SIMD3<Float> = SIMD3(0.4, -0.6, 0.7), ambient: Float = 0.25) {
        self.direction = simd_normalize(direction); self.ambient = ambient
    }
}
