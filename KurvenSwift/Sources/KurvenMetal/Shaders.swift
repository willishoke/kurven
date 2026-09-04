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
    """
}
