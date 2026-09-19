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
        float4x4 clip;
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
        float  slopeScale;
        // float3, not packed_float3: C's simd_float3 is sixteen bytes with
        // three used, and packing it here shifts everything after it. That
        // mismatch compiled cleanly and rendered a black landscape.
        float3 lightDirection;
        float  ambient;
        float  strokeWidth;
        float2 depthRange;
        float2 viewport;
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
        // Depth is carried in the color attachment and MAX-blended, so the
        // position's own z is only there to pass the clip test; the clip
        // matrix writes a constant into it. Under perspective the divide is the
        // hardware's, which is the only reason a 4x4 is needed at all.
        return u.clip * (u.view * float4(world, 1.0));
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

    // How far the surface's depth moves across one pixel at `p`: per axis, the
    // smaller of the two one-sided differences, and the larger of the two axes.
    //
    // The smaller, not the larger, so that a silhouette on one side -- a jump
    // to a far surface, or to nothing -- is not mistaken for slope. Were it
    // counted, ink hidden behind a ridge would show through at the ridge.
    // Neighbours off the texture or holding `empty` do not count at all.
    static float axis_slope(texture2d<float, access::read> depth, int2 p, int2 d,
                            float here, float empty)
    {
        int2 size = int2(depth.get_width(), depth.get_height());
        int2 a = p - d, b = p + d;
        bool hasA = all(a >= 0) && all(a < size);
        bool hasB = all(b >= 0) && all(b < size);
        float za = hasA ? depth.read(uint2(a)).r : empty;
        float zb = hasB ? depth.read(uint2(b)).r : empty;
        hasA = hasA && za > empty;
        hasB = hasB && zb > empty;
        float da = abs(here - za), db = abs(zb - here);
        if (hasA && hasB) { return min(da, db); }
        if (hasA) { return da; }
        if (hasB) { return db; }
        return 0.0;
    }

    static float depth_slope(texture2d<float, access::read> depth, uint2 p,
                             float here, float empty)
    {
        return max(axis_slope(depth, int2(p), int2(1, 0), here, empty),
                   axis_slope(depth, int2(p), int2(0, 1), here, empty));
    }

    // Whether ink at view depth `z` is in front of the surface at pixel `p`.
    //
    // `z + margin > buffer` is `clip_hidden_lines`' predicate, which the bake
    // applies per vertex. Here it is applied per fragment, and that moves the
    // two things being compared apart: `z` is the line's depth where it crosses
    // the pixel, up to half a pixel from the centre, and `buffer` is the
    // surface's depth *at* the centre. On a surface that is steep in view, half
    // a pixel is more depth than the margin, and visible ink is discarded in a
    // pattern that moves whenever the camera does -- which is the flicker.
    //
    // So the preview adds `slopeScale` pixels' worth of the surface's own depth
    // change to the margin: the depth bias shadow maps use, for the same
    // reason. It is a preview-only relaxation. The bake does not do this, and
    // with `slopeScale` zero this is the bake's predicate exactly.
    static bool ink_visible(texture2d<float, access::read> depth, uint2 p, float z,
                            constant KVShading &s)
    {
        float behind = depth.read(p).r;
        if (!(behind > s.empty)) { return true; }
        float slack = s.margin + s.slopeScale * depth_slope(depth, p, behind, s.empty);
        return z + slack > behind;
    }

    // ---------------------------------------------------------------------
    // ink: every segment its own screen-space quad
    // ---------------------------------------------------------------------
    //
    // `.line` primitives are one pixel wide whatever the stroke, with no
    // coverage and no antialiasing. So each segment is drawn as a quad of its
    // own, expanded in screen space to the stroke's width and a pixel beyond,
    // and the fragment works out how much of its pixel the stroke covers from
    // its distance to the centreline. One instance per segment, reading both
    // ends from the line buffer: segment i is vertices 2i and 2i+1.
    //
    // Visibility is decided at the *centreline*, not at the fragment: every
    // fragment across a stroke asks whether the point of the centreline beside
    // it is in front of the surface. A stroke is visible where its line is,
    // which is the question the bake asks too, and a wide stroke is not eaten
    // from its edges where the surface is steep in view -- the centreline point
    // is always within half a pixel of the pixel it is tested at, which is what
    // `slopeScale` is sized for.

    struct StrokeOut {
        float4 position [[position]];
        // View depth, along the segment and constant across it: the depth of
        // the centreline beside this fragment.
        float  depth;
        // That centreline point, in pixels, and this fragment's signed distance
        // from it. Screen-space quantities, so not perspective-corrected.
        float2 center [[center_no_perspective]];
        float  across [[center_no_perspective]];
    };

    // Clip a segment to the part the hardware would keep, `z <= w`, before
    // anything divides by w. An orthographic camera keeps all of it; under
    // perspective this is the part in front of the eye. False when there is
    // none.
    static bool clip_near(thread float4 &a, thread float4 &b,
                          thread float &da, thread float &db)
    {
        const float keep = 1e-4;
        float ea = a.w - a.z, eb = b.w - b.z;
        if (ea < keep && eb < keep) { return false; }
        if (ea < keep) {
            float t = (keep - ea) / (eb - ea);
            a = mix(a, b, t); da = mix(da, db, t);
        } else if (eb < keep) {
            float t = (keep - eb) / (ea - eb);
            b = mix(b, a, t); db = mix(db, da, t);
        }
        return true;
    }

    // Clip space to pixels, y down: the frame `[[position]]` arrives in.
    static float2 to_pixels(float4 c, float2 viewport) {
        float2 ndc = c.xy / c.w;
        return float2(ndc.x + 1.0, 1.0 - ndc.y) * 0.5 * viewport;
    }

    // Back, keeping the endpoint's own z and w, so the rasterizer interpolates
    // depth along the segment exactly as it would have along the line.
    static float4 from_pixels(float2 p, float4 c, float2 viewport) {
        float2 ndc = float2(2.0 * p.x / viewport.x - 1.0, 1.0 - 2.0 * p.y / viewport.y);
        return float4(ndc * c.w, c.z, c.w);
    }

    // Four vertices, as a strip: (a, -), (a, +), (b, -), (b, +).
    vertex StrokeOut kv_stroke_vertex(uint vid [[vertex_id]],
                                      uint iid [[instance_id]],
                                      constant KVUniforms &u [[buffer(0)]],
                                      constant KVShading &s [[buffer(1)]],
                                      constant KVVertex *verts [[buffer(4)]],
                                      constant uint &first [[buffer(5)]])
    {
        uint segment = first + iid;
        float3 wa = verts[2u * segment].position;
        float3 wb = verts[2u * segment + 1u].position;
        float4 ca = to_ndc(u, wa), cb = to_ndc(u, wb);
        float da = view_depth(u, wa), db = view_depth(u, wb);

        StrokeOut out;
        out.depth = 0.0;
        out.center = float2(0.0);
        out.across = 0.0;
        if (!clip_near(ca, cb, da, db)) {
            // Off-screen in w, as the heightfield culls its masked cells.
            out.position = float4(0.0, 0.0, 0.0, 0.0);
            return out;
        }

        float2 pa = to_pixels(ca, s.viewport), pb = to_pixels(cb, s.viewport);
        float2 along = pb - pa;
        float len = length(along);
        // A segment seen end-on has no direction; its quad has no length
        // either, so any normal will do and nothing is drawn.
        float2 dir = len > 1e-6 ? along / len : float2(1.0, 0.0);
        float2 normal = float2(-dir.y, dir.x);
        // Half the stroke, and a pixel beyond it for the edge to fall off in.
        float reach = 0.5 * s.strokeWidth + 1.0;
        bool atB = vid >= 2u;
        float side = (vid & 1u) != 0u ? 1.0 : -1.0;
        float2 end = atB ? pb : pa;

        out.position = from_pixels(end + normal * (side * reach), atB ? cb : ca, s.viewport);
        out.depth = atB ? db : da;
        out.center = end;
        out.across = side * reach;
        return out;
    }

    fragment float4 kv_stroke_fragment(StrokeOut in [[stage_in]],
                                       constant KVShading &s [[buffer(1)]],
                                       texture2d<float, access::read> depth [[texture(1)]])
    {
        // How much of the pixel the stroke covers, across its width: the
        // overlap of the stroke's cross-section with the pixel's, which is a
        // box filter. A stroke narrower than a pixel covers less of it rather
        // than drawing narrower, so a 0.15 layer reads lighter than a 0.4 one,
        // as it does on paper.
        float d = abs(in.across);
        float half_width = 0.5 * s.strokeWidth;
        float coverage = min(d + half_width, 0.5) - max(d - half_width, -0.5);
        if (coverage <= 0.0) { discard_fragment(); }

        float2 last = float2(depth.get_width() - 1u, depth.get_height() - 1u);
        uint2 p = uint2(clamp(in.center, float2(0.0), last));
        if (!ink_visible(depth, p, in.depth, s)) { discard_fragment(); }
        return float4(s.color.rgb, s.color.a * min(coverage, 1.0));
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
        for (uint c = 0; c < 4; ++c) {
            for (uint r = 0; r < 4; ++r) { out[k++] = u.clip[c][r]; }
        }
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
        out[k++] = sh.slopeScale;
        out[k++] = sh.lightDirection.x;
        out[k++] = sh.lightDirection.y;
        out[k++] = sh.lightDirection.z;
        out[k++] = sh.ambient;
        out[k++] = sh.strokeWidth;
        out[k++] = sh.depthRange.x;
        out[k++] = sh.depthRange.y;
        out[k++] = sh.viewport.x;
        out[k++] = sh.viewport.y;
        out[k++] = float(sizeof(KVUniforms));
        out[k++] = float(sizeof(KVShading));
    }
    """

    /// How many floats `kv_layout_probe` writes: every field of both structs,
    /// then both sizes.
    public static let uniformFieldCount = 16 + 16 + 2 + 2 + 2 + 2 + 1 + 1 + 1 + 1
    public static let shadingFieldCount = 4 + 1 + 1 + 1 + 3 + 1 + 1 + 2 + 2
    public static let layoutProbeCount = uniformFieldCount + shadingFieldCount + 2
}
