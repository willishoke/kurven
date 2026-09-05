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
// whether a line fragment is hidden, and what to paint.
typedef struct {
    simd_float4 color;
    // Added to a line's view depth before it is compared with the surface --
    // `outline.clip_hidden_lines`' margin, applied per fragment instead of per
    // vertex. That difference is the whole gap between preview and bake.
    float margin;
    // The value the depth texture holds where nothing was drawn.
    float empty;
    // Key light direction in view space, and how much light reaches the parts
    // it does not.
    simd_float3 lightDirection;
    float ambient;
    // View-depth range for the `.depth` inspection mode: what to map to black
    // and to white.
    simd_float2 depthRange;
} KVShading;

#endif /* KurvenShaderTypes_h */
