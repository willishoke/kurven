// Structs shared by Swift and Metal Shading Language.
//
// This header is imported by Swift as a module and *textually prepended* to the
// MSL source before it is compiled at runtime, so both sides get their layout
// from the same declaration. Without Xcode there is no .metal build product and
// therefore no compiler to catch a mismatch between two hand-written structs;
// with one header there is nothing to mismatch.
//
// Everything here is float, not double: the GPU has no doubles. The camera is
// built and composed in double on the CPU and demoted once, at upload.

#ifndef KurvenShaderTypes_h
#define KurvenShaderTypes_h

#include <simd/simd.h>
#include <stdint.h>

typedef struct {
    // World -> view. View z increases toward the viewer, matching the Python
    // rotated z and the MAX depth blend.
    simd_float4x4 view;

    // View -> Metal clip space, with the perspective divide left to the
    // hardware. A full 4x4 rather than a 2x2 and an offset because an
    // orthographic camera's map is affine and a perspective one's is not, and
    // one matrix covers both. Which view component drives which screen axis is
    // folded in here too: the bake rasterizes in ZBuffer's order (rows index
    // view x) so it can be compared against Python, the preview in screen
    // order (view x across) so it looks like the plate.
    simd_float4x4 clip;

    // The heightfield's domain rectangle and its lattice size after decimation.
    simd_float2 domainLo;      // (real.lo, imag.lo)
    simd_float2 domainSize;    // (real.length, imag.length)
    simd_uint2  lattice;       // samples along (real, imag)
    simd_uint2  gridSize;      // the full texture size, before decimation
    uint32_t    step;          // texel stride: lattice sample i reads texel i*step

    // A uniform cap stays a live knob (min() in the shader). Band caps are
    // baked into the texture before upload, and this is +INFINITY for them.
    float cap;

    // Number of polygon corners bounding the rasterized region; 0 means the
    // whole rectangle.
    uint32_t regionCount;

    // Value written where nothing is drawn. Metal has no -inf clear, so the
    // buffer clears to this and the reader maps it back.
    float empty;
} KVUniforms;

// A heightfield instance: the 2x3 affine of one tile.
typedef struct {
    simd_float4 linear;   // (a, b, c, d)
    simd_float2 offset;   // (tx, ty)
} KVTile;

// A wall-mesh vertex, in world coordinates.
typedef struct {
    simd_float3 position;
} KVVertex;

// What the preview's second pass needs on top of KVUniforms: how to decide
// whether a line fragment is hidden, and what to paint and how wide.
typedef struct {
    simd_float4 color;
    // Added to a line's view depth before it is compared with the surface --
    // `outline.clip_hidden_lines`' margin, applied per fragment instead of per
    // vertex. That, and `slopeScale` below, are the whole gap between preview
    // and bake.
    float margin;
    // The value the depth texture holds where nothing was drawn.
    float empty;
    // How many pixels' worth of the surface's own depth change to add to the
    // margin, per fragment. Preview only, and not in the bake: the preview
    // compares a line's depth where it crosses a pixel with the surface's depth
    // at the pixel's centre, and on a surface that is steep in view those
    // differ by more than any constant margin. Zero is the bake's predicate.
    float slopeScale;
    // Key light direction in view space, and how much light reaches the parts
    // it does not.
    simd_float3 lightDirection;
    float ambient;
    // The stroke's width in pixels. Ink is drawn as screen-space quads this
    // wide, plus a pixel for the edge to fall off in.
    float strokeWidth;
    // View-depth range for the `.depth` inspection mode: what to map to black
    // and to white.
    simd_float2 depthRange;
    // The target's size in pixels, which is what a stroke's width is measured
    // against when it is expanded to a quad.
    simd_float2 viewport;
} KVShading;

// A parametric surface's lattice: how a vertex id becomes a texel of the
// position texture and a surface coordinate.
typedef struct {
    // Texels along (u, v).
    simd_uint2 samples;
    // Cells along (u, v): `samples` on a periodic axis, whose last cell closes
    // back onto texel 0, and `samples - 1` on a bounded one.
    simd_uint2 cells;
    // The coordinate of lattice index 0, and the coordinate per index. Index
    // `samples` on a periodic axis is one period past index 0, so a cell that
    // closes the seam interpolates across it rather than back through the
    // whole range.
    simd_float2 lo;
    simd_float2 spacing;
    // The view-z range the coordinate pass's depth attachment spans: the
    // nearest (y) maps to 0 and the farthest (x) to 1, so a LESS test keeps
    // the front-most fragment, as the MAX blend does.
    simd_float2 depthRange;
} KVSurface;

// A vertex of ink drawn on a parametric surface: where it is, which way the
// surface faces there, and where on the surface it lies.
typedef struct {
    simd_float3 position;
    // The *outward* normal, not normalized: the parametrization's normal times
    // the surface's orientation. Zero for a surface that bounds no solid, and
    // for ink on the folds themselves, where facing is not asked.
    simd_float3 normal;
    simd_float2 coord;
} KVInkVertex;

// What the ink test needs of the camera to ask which way a surface faces.
typedef struct {
    // World direction sight lines travel, for an orthographic camera.
    simd_float3 sight;
    // World position of the eye, for a perspective one.
    simd_float3 eye;
    uint32_t perspective;
    // Nonzero for fold ink: judged by the front-most test alone.
    uint32_t onFolds;
} KVInk;

#endif /* KurvenShaderTypes_h */
