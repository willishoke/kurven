import Foundation

/// The Metal source, as a string.
///
/// Runtime compilation is the single, permanent path. There is no `.metal`
/// build product, so a terminal build and any future Xcode build cannot
/// diverge, and `ShaderTypes.h` is prepended here so the structs the shader
/// reads are literally the structs Swift writes. `MetalRenderer` compiles this
/// once per process and a test compiles it on the default device, which is the
/// check the absent `metal` compiler would otherwise provide.
public enum Shaders {
    /// `ShaderTypes.h` verbatim, minus its include guard and its `#include`,
    /// which MSL supplies itself.
    static let shaderTypes = """
    typedef struct {
        float4x4 view;
        float2x2 ndcLinear;
        float2 ndcOffset;
        float2 domainLo;
        float2 domainSize;
        uint2  lattice;
        uint2  gridSize;
        uint   step;
        float  cap;
        uint   regionCount;
        float  empty;
    } KVUniforms;

    typedef struct { float4 linear; float2 offset; } KVTile;
    typedef struct { float3 position; } KVVertex;

    typedef struct {
        float4 color;
        float  margin;
        float  empty;
        // float3, not packed_float3: C's simd_float3 is sixteen bytes with
        // three used, and packing it here shifts everything after it. That
        // mismatch compiled cleanly and rendered a black landscape.
        float3 lightDirection;
        float  ambient;
        float2 depthRange;
    } KVShading;
    """

    public static let source = """
    #include <metal_stdlib>
    using namespace metal;

    """ + shaderTypes + """


    // ---------------------------------------------------------------------
    // shared
    // ---------------------------------------------------------------------

    struct DepthOut {
        float4 position [[position]];
        float  depth;
    };

    // Even-odd ray crossing scanning along world x, the same predicate as
    // `kurven.bundle.point_in_polygon` and `KurvenCore.pointInPolygon`. The
    // scan axis is part of the contract: a test raying along the other axis
    // agrees everywhere except on the boundary, which is where a staircase
    // cutout lives.
    static bool inside_region(float2 p, constant float2 *corners, uint n) {
        if (n == 0) { return true; }
        bool inside = false;
        for (uint i = 0; i < n; ++i) {
            float2 a = corners[i];
            float2 b = corners[(i + 1) % n];
            if (a.x == b.x) { continue; }
            float crossing = (b.y - a.y) * (p.x - a.x) / (b.x - a.x) + a.y;
            if (((a.x > p.x) != (b.x > p.x)) && (p.y < crossing)) { inside = !inside; }
        }
        return inside;
    }

    static float4 to_ndc(constant KVUniforms &u, float3 world) {
        float3 v = (u.view * float4(world, 1.0)).xyz;
        // Depth is carried in the color attachment and MAX-blended, so the
        // position's own z is only there to pass the clip test.
        float2 ndc = u.ndcLinear * v.xy + u.ndcOffset;
        return float4(ndc, 0.5, 1.0);
    }

    static float view_depth(constant KVUniforms &u, float3 world) {
        return (u.view * float4(world, 1.0)).z;
    }

    // ---------------------------------------------------------------------
    // heightfield: an implicit mesh, six vertices per lattice cell
    // ---------------------------------------------------------------------
    //
    // No vertex buffer and no index buffer exist for the surface. The vertex id
    // names a cell and a corner; the height comes from a texel read. An explicit
    // mesh for the elliptic plate would be six million triangles and for
    // gamma's bake two hundred million, and none of them would say anything the
    // texture does not already say.

    vertex DepthOut kv_height_vertex(uint vid [[vertex_id]],
                                     uint iid [[instance_id]],
                                     constant KVUniforms &u [[buffer(0)]],
                                     constant KVTile *tiles [[buffer(1)]],
                                     constant float2 *region [[buffer(2)]],
                                     texture2d<float, access::read> heights [[texture(0)]])
    {
        uint cells_x = u.lattice.x - 1;
        uint cell = vid / 6u;
        uint corner = vid % 6u;
        uint cx = cell % cells_x;
        uint cy = cell / cells_x;

        // Two triangles: (00, 01, 10) and (01, 11, 10), matching
        // `zbuffer.surface_grid_mesh`'s winding.
        const uint2 corners[6] = { uint2(0,0), uint2(1,0), uint2(0,1),
                                   uint2(1,0), uint2(1,1), uint2(0,1) };
        uint2 c = corners[corner];

        KVTile t = tiles[iid];

        // A cell survives only when all four of its corners are in the region.
        // Same rule as the CPU mask: the notch is an absence of geometry, not
        // something drawn and then covered up.
        bool kept = true;
        if (u.regionCount > 0) {
            for (uint k = 0; k < 4u && kept; ++k) {
                uint2 q = uint2(k & 1u, k >> 1u);
                float2 d = u.domainLo + u.domainSize
                    * float2(float(cx + q.x) / float(u.lattice.x - 1),
                             float(cy + q.y) / float(u.lattice.y - 1));
                float2 w = float2(t.linear.x * d.x + t.linear.y * d.y + t.offset.x,
                                  t.linear.z * d.x + t.linear.w * d.y + t.offset.y);
                kept = inside_region(w, region, u.regionCount);
            }
        }

        uint ix = min((cx + c.x) * u.step, u.gridSize.x - 1u);
        uint iy = min((cy + c.y) * u.step, u.gridSize.y - 1u);
        float h = min(heights.read(uint2(ix, iy)).r, u.cap);

        float2 d = u.domainLo + u.domainSize
            * float2(float(cx + c.x) / float(u.lattice.x - 1),
                     float(cy + c.y) / float(u.lattice.y - 1));
        float3 world = float3(t.linear.x * d.x + t.linear.y * d.y + t.offset.x,
                              t.linear.z * d.x + t.linear.w * d.y + t.offset.y,
                              h);

        DepthOut out;
        out.position = to_ndc(u, world);
        out.depth = view_depth(u, world);
        if (!kept) {
            // Off-screen in w: the whole triangle is culled, and no fragment of
            // it ever reaches the blend.
            out.position = float4(0.0, 0.0, 0.0, 0.0);
        }
        return out;
    }

    // ---------------------------------------------------------------------
    // walls: a small explicit mesh
    // ---------------------------------------------------------------------

    vertex DepthOut kv_mesh_vertex(uint vid [[vertex_id]],
                                   constant KVUniforms &u [[buffer(0)]],
                                   constant KVVertex *verts [[buffer(3)]])
    {
        float3 world = verts[vid].position;
        DepthOut out;
        out.position = to_ndc(u, world);
        out.depth = view_depth(u, world);
        return out;
    }

    // ---------------------------------------------------------------------
    // the depth fragment: write view z, MAX-blended
    // ---------------------------------------------------------------------
    //
    // A color attachment with a MAX blend rather than a depth attachment: it is
    // exactly `GL_MAX` over view z, which is what the Python Z-buffer holds, in
    // the same float32 the readback and the clipper compare. A depth attachment
    // would work too and would put the numbers in a different space for no gain.

    fragment float kv_depth_fragment(DepthOut in [[stage_in]]) {
        return in.depth;
    }

    // ---------------------------------------------------------------------
    // preview
    // ---------------------------------------------------------------------
    //
    // Two passes over the same geometry the bake uses. The first is the depth
    // pass, unchanged. The second draws into the visible target and decides
    // visibility by sampling that depth texture at the fragment's own pixel --
    // which is why there is one depth semantic in this program and not two, and
    // why "the preview approximates the bake" is a statement about where the
    // test is evaluated rather than about how the picture is made.

    struct PreviewOut {
        float4 position [[position]];
        float  depth;
        float3 world;
    };

    vertex PreviewOut kv_surface_vertex(uint vid [[vertex_id]],
                                        uint iid [[instance_id]],
                                        constant KVUniforms &u [[buffer(0)]],
                                        constant KVTile *tiles [[buffer(1)]],
                                        constant float2 *region [[buffer(2)]],
                                        texture2d<float, access::read> heights [[texture(0)]])
    {
        uint cells_x = u.lattice.x - 1;
        uint cell = vid / 6u;
        uint corner = vid % 6u;
        uint cx = cell % cells_x;
        uint cy = cell / cells_x;
        const uint2 corners[6] = { uint2(0,0), uint2(1,0), uint2(0,1),
                                   uint2(1,0), uint2(1,1), uint2(0,1) };
        uint2 c = corners[corner];
        KVTile t = tiles[iid];

        bool kept = true;
        if (u.regionCount > 0) {
            for (uint k = 0; k < 4u && kept; ++k) {
                uint2 q = uint2(k & 1u, k >> 1u);
                float2 d = u.domainLo + u.domainSize
                    * float2(float(cx + q.x) / float(u.lattice.x - 1),
                             float(cy + q.y) / float(u.lattice.y - 1));
                float2 w = float2(t.linear.x * d.x + t.linear.y * d.y + t.offset.x,
                                  t.linear.z * d.x + t.linear.w * d.y + t.offset.y);
                kept = inside_region(w, region, u.regionCount);
            }
        }

        uint ix = min((cx + c.x) * u.step, u.gridSize.x - 1u);
        uint iy = min((cy + c.y) * u.step, u.gridSize.y - 1u);
        float h = min(heights.read(uint2(ix, iy)).r, u.cap);
        float2 d = u.domainLo + u.domainSize
            * float2(float(cx + c.x) / float(u.lattice.x - 1),
                     float(cy + c.y) / float(u.lattice.y - 1));
        float3 world = float3(t.linear.x * d.x + t.linear.y * d.y + t.offset.x,
                              t.linear.z * d.x + t.linear.w * d.y + t.offset.y, h);

        PreviewOut out;
        out.position = kept ? to_ndc(u, world) : float4(0.0, 0.0, 0.0, 0.0);
        out.depth = view_depth(u, world);
        out.world = world;
        return out;
    }

    vertex PreviewOut kv_wall_vertex(uint vid [[vertex_id]],
                                     constant KVUniforms &u [[buffer(0)]],
                                     constant KVVertex *verts [[buffer(3)]])
    {
        float3 world = verts[vid].position;
        PreviewOut out;
        out.position = to_ndc(u, world);
        out.depth = view_depth(u, world);
        out.world = world;
        return out;
    }

    // Paper. The plate's surface is not drawn -- it is white, and the page is
    // white -- but it must still be *opaque*, or the ink behind it shows
    // through the silhouette. So the plate mode paints it flat.
    fragment float4 kv_paper_fragment(PreviewOut in [[stage_in]]) {
        return float4(1.0, 1.0, 1.0, 1.0);
    }

    // A lit heightfield, for orientation rather than for the plate. The normal
    // comes from the screen-space derivatives of the world position, so it needs
    // no normal buffer and no second texture read.
    //
    // There is no depth attachment on this pass. It does not need one: the depth
    // texture already holds the front-most view depth at every pixel, so a
    // fragment is the visible surface exactly when its own depth is that one.
    // Reusing the buffer is one depth semantic in the program instead of two.
    fragment float4 kv_shaded_fragment(PreviewOut in [[stage_in]],
                                       constant KVShading &s [[buffer(1)]],
                                       texture2d<float, access::read> depth [[texture(1)]])
    {
        // Derivatives first: they must be evaluated before any discard.
        float3 n = normalize(cross(dfdx(in.world), dfdy(in.world)));
        float behind = depth.read(uint2(in.position.xy)).r;
        if (in.depth < behind - 1e-4 * max(1.0, abs(behind))) { discard_fragment(); }
        float lambert = abs(dot(n, normalize(s.lightDirection)));
        float v = s.ambient + (1.0 - s.ambient) * lambert;
        return float4(float3(v), 1.0);
    }

    // The depth attachment, drawn directly. This is the substitute for the GPU
    // frame-capture viewer Command Line Tools does not ship, and it is most of
    // what such a viewer actually gets used for.
    vertex float4 kv_fullscreen_vertex(uint vid [[vertex_id]]) {
        const float2 p[3] = { float2(-1.0, -3.0), float2(-1.0, 1.0), float2(3.0, 1.0) };
        return float4(p[vid], 0.0, 1.0);
    }

    fragment float4 kv_depth_view_fragment(float4 position [[position]],
                                           constant KVShading &s [[buffer(1)]],
                                           texture2d<float, access::read> depth [[texture(1)]])
    {
        float z = depth.read(uint2(position.xy)).r;
        if (!(z > s.empty)) { return float4(0.10, 0.11, 0.13, 1.0); }
        float t = saturate((z - s.depthRange.x)
                           / max(s.depthRange.y - s.depthRange.x, 1e-6));
        return float4(float3(t), 1.0);
    }

    // Ink. Lines carry their own view depth and compare it against the surface
    // the first pass left, at their own pixel: `z + margin > buffer`, the same
    // predicate `clip_hidden_lines` applies per vertex.
    struct LineOut {
        float4 position [[position]];
        float  depth;
    };

    vertex LineOut kv_line_vertex(uint vid [[vertex_id]],
                                  constant KVUniforms &u [[buffer(0)]],
                                  constant KVVertex *verts [[buffer(4)]])
    {
        float3 world = verts[vid].position;
        LineOut out;
        out.position = to_ndc(u, world);
        out.depth = view_depth(u, world);
        return out;
    }

    fragment float4 kv_line_fragment(LineOut in [[stage_in]],
                                     constant KVShading &s [[buffer(1)]],
                                     texture2d<float, access::read> depth [[texture(1)]])
    {
        uint2 p = uint2(in.position.xy);
        float behind = depth.read(p).r;
        if (in.depth + s.margin <= behind) { discard_fragment(); }
        return s.color;
    }

    // ---------------------------------------------------------------------
    // layout probe
    // ---------------------------------------------------------------------
    //
    // Reads every field of a KVUniforms the CPU filled with known values and
    // writes back what it saw. If the two sides disagree about the layout --
    // because the header changed and a stale build did not propagate it, or
    // because this copy of the struct drifted from the header -- the values
    // come back wrong and `kurven-test` says which field.
    //
    // The header is the single declaration, but nothing at build time enforces
    // that MSL gets the same one: there is no `metal` compiler here and
    // SwiftPM does not track a C header as a dependency of a Swift target. So
    // the agreement is checked rather than assumed, which is the same trade the
    // rest of this design makes.

    kernel void kv_layout_probe(constant KVUniforms &u [[buffer(0)]],
                                device float *out [[buffer(1)]],
                                constant KVShading &sh [[buffer(2)]],
                                uint tid [[thread_position_in_grid]])
    {
        if (tid != 0) { return; }
        uint k = 0;
        for (uint c = 0; c < 4; ++c) {
            for (uint r = 0; r < 4; ++r) { out[k++] = u.view[c][r]; }
        }
        out[k++] = u.ndcLinear[0][0];
        out[k++] = u.ndcLinear[0][1];
        out[k++] = u.ndcLinear[1][0];
        out[k++] = u.ndcLinear[1][1];
        out[k++] = u.ndcOffset.x;
        out[k++] = u.ndcOffset.y;
        out[k++] = u.domainLo.x;
        out[k++] = u.domainLo.y;
        out[k++] = u.domainSize.x;
        out[k++] = u.domainSize.y;
        out[k++] = float(u.lattice.x);
        out[k++] = float(u.lattice.y);
        out[k++] = float(u.gridSize.x);
        out[k++] = float(u.gridSize.y);
        out[k++] = float(u.step);
        out[k++] = u.cap;
        out[k++] = float(u.regionCount);
        out[k++] = u.empty;
        out[k++] = sh.color.x;
        out[k++] = sh.color.y;
        out[k++] = sh.color.z;
        out[k++] = sh.color.w;
        out[k++] = sh.margin;
        out[k++] = sh.empty;
        out[k++] = sh.lightDirection.x;
        out[k++] = sh.lightDirection.y;
        out[k++] = sh.lightDirection.z;
        out[k++] = sh.ambient;
        out[k++] = sh.depthRange.x;
        out[k++] = sh.depthRange.y;
        out[k++] = float(sizeof(KVUniforms));
        out[k++] = float(sizeof(KVShading));
    }
    """

    /// How many floats `kv_layout_probe` writes: every field of both structs,
    /// then both sizes.
    public static let uniformFieldCount = 16 + 4 + 2 + 2 + 2 + 2 + 2 + 1 + 1 + 1 + 1
    public static let shadingFieldCount = 4 + 1 + 1 + 3 + 1 + 2
    public static let layoutProbeCount = uniformFieldCount + shadingFieldCount + 2
}
