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

    // View -> Metal NDC. ndc.x is built from view.y and ndc.y from view.x,
    // because Python's ZBuffer indexes rows by the first view component; see
    // DepthFrame.metalNDC for the derivation and the row-order flip.
    simd_float2 ndcScale;
    simd_float2 ndcOffset;

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

#endif /* KurvenShaderTypes_h */
