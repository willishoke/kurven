// The WGSL source, as a string: the port of Sources/KurvenMetal/Shaders.swift.
//
// One module, compiled once per device, every pipeline an entry point of
// it, as Swift's one MTLLibrary. The structs are ShaderTypes.h's, laid out
// by WGSL's rules rather than C's (a vec3f is twelve bytes here, sixteen in
// MSL), and layout.ts writes them at exactly these offsets; the layout
// probe at the end reads every field back, so the two sides cannot drift
// apart silently (layout.ts, `probeLayout`, and the browser lane).
//
// What differs from Metal, each for a reason WGSL gives:
//
// - Swift passes an infinite depth tolerance and truncation height. WGSL
//   lets an implementation assume no infinity reaches a shader, so the page
//   writes the largest float, `UNREACHABLE` below, which every comparison
//   here treats as Metal treats infinity.
// - Metal reads `[[vertex_id]]` and `[[instance_id]]`; WGSL's
//   `vertex_index` and `instance_index`. A stroke layer is a buffer of its
//   own here (the page keeps each layer's segments under the engine's key,
//   so a frame re-uploads only what changed), so Swift's `first` offset into
//   one shared buffer is always zero and is not passed.
// - `[[center_no_perspective]]` is `@interpolate(linear)`: the same
//   screen-space interpolation.
// - The per-draw `setVertexBytes` of a layer's `KVShading` and `KVStroke`
//   is one `KVLayer` slot per layer in a uniform buffer, bound at its
//   offset.

/** The largest finite float: Swift's infinity, where WGSL may not hold one. */
export const UNREACHABLE = 3.4028234663852886e38;

export const SHADERS = /* wgsl */ `
const UNREACHABLE: f32 = 3.4028234663852886e38;

struct KVUniforms {
  view: mat4x4f,
  clip: mat4x4f,
  domainLo: vec2f,
  domainSize: vec2f,
  grid: vec2u,
  gridSize: vec2u,
  step: u32,
  truncationHeight: f32,
  regionCount: u32,
  empty: f32,
}

struct KVTile { linear: vec4f, offset: vec2f }
struct KVVertex { position: vec3f }
struct KVLineVertex { position: vec3f, normal: vec3f }

struct KVShading {
  color: vec4f,
  depthTolerance: f32,
  empty: f32,
  slopeScale: f32,
  lightDirection: vec3f,
  ambient: f32,
  strokeWidth: f32,
  depthRange: vec2f,
  viewport: vec2f,
}

struct KVSurface {
  samples: vec2u,
  cells: vec2u,
  lo: vec2f,
  spacing: vec2f,
  depthRange: vec2f,
}

struct KVStrokeVertex { position: vec3f, normal: vec3f, coord: vec2f }

struct KVStroke {
  sight: vec3f,
  eye: vec3f,
  perspective: u32,
  onFolds: u32,
}

// One layer's draw: what Swift sets with setVertexBytes before each.
struct KVLayer { shading: KVShading, stroke: KVStroke }

// ---------------------------------------------------------------------------
// bindings
// ---------------------------------------------------------------------------
//
// Group 0 is the frame's: the scene's resources, keyed on its content, and
// the two textures the depth and coordinate passes leave. Group 1 is a
// layer's: its slot and its segments. Each pipeline's bind group holds the
// bindings its entry points read (renderer.ts lists them).

@group(0) @binding(0) var<uniform> u: KVUniforms;
@group(0) @binding(1) var<storage, read> tiles: array<KVTile>;
@group(0) @binding(2) var<storage, read> region: array<vec2f>;
@group(0) @binding(3) var heights: texture_2d<f32>;
@group(0) @binding(4) var<storage, read> walls: array<KVVertex>;
@group(0) @binding(5) var<uniform> sf: KVSurface;
@group(0) @binding(6) var positions: texture_2d<f32>;
@group(0) @binding(7) var depth: texture_2d<f32>;
@group(0) @binding(8) var coords: texture_2d<f32>;
@group(0) @binding(9) var<uniform> frameShading: KVShading;

@group(1) @binding(0) var<uniform> layer: KVLayer;
@group(1) @binding(1) var<storage, read> lineVerts: array<KVLineVertex>;
@group(1) @binding(2) var<storage, read> strokeVerts: array<KVStrokeVertex>;

// ---------------------------------------------------------------------------
// shared
// ---------------------------------------------------------------------------

struct DepthOut {
  @builtin(position) position: vec4f,
  @location(0) depth: f32,
}

// Even-odd ray crossing scanning along world x, the same predicate as
// \`kurven.bundle.point_in_polygon\` and \`KurvenCore.pointInPolygon\`. The
// scan axis is part of the contract: a test raying along the other axis
// agrees everywhere except on the boundary, which is where a staircase
// cutout lives.
fn inside_region(p: vec2f, n: u32) -> bool {
  if (n == 0u) { return true; }
  var inside = false;
  for (var i = 0u; i < n; i = i + 1u) {
    let a = region[i];
    let b = region[(i + 1u) % n];
    if (a.x == b.x) { continue; }
    let crossing = (b.y - a.y) * (p.x - a.x) / (b.x - a.x) + a.y;
    if (((a.x > p.x) != (b.x > p.x)) && (p.y < crossing)) { inside = !inside; }
  }
  return inside;
}

// Depth is carried in the color attachment and MAX-blended, so the
// position's own z is only there to pass the clip test; the clip matrix
// writes a constant into it. Under perspective the divide is the
// hardware's, which is the only reason a 4x4 is needed at all.
fn to_ndc(world: vec3f) -> vec4f {
  return u.clip * (u.view * vec4f(world, 1.0));
}

fn view_depth(world: vec3f) -> f32 {
  return (u.view * vec4f(world, 1.0)).z;
}

const CORNERS = array<vec2u, 6>(vec2u(0u, 0u), vec2u(1u, 0u), vec2u(0u, 1u),
                                vec2u(1u, 0u), vec2u(1u, 1u), vec2u(0u, 1u));

// ---------------------------------------------------------------------------
// heightfield: an implicit mesh, six vertices per grid cell
// ---------------------------------------------------------------------------
//
// No vertex buffer and no index buffer exist for the surface. The vertex id
// names a cell and a corner; the height comes from a texel read. An explicit
// mesh for the elliptic plate would be six million triangles and for
// gamma's bake two hundred million, and none of them would say anything the
// texture does not already say.

struct HeightVertex { world: vec3f, depth: f32, kept: bool }

fn height_vertex(vid: u32, iid: u32) -> HeightVertex {
  let cells_x = u.grid.x - 1u;
  let cell = vid / 6u;
  let corner = vid % 6u;
  let cx = cell % cells_x;
  let cy = cell / cells_x;
  // Two triangles: (00, 10, 01) and (10, 11, 01), matching
  // \`zbuffer.surface_grid_mesh\`.
  var corners = CORNERS;
  let c = corners[corner];
  let t = tiles[iid];

  // A cell survives only when all four of its corners are in the region.
  // Same rule as the CPU mask: the notch is an absence of geometry, not
  // something drawn and then covered up.
  var kept = true;
  if (u.regionCount > 0u) {
    for (var k = 0u; k < 4u && kept; k = k + 1u) {
      let q = vec2u(k & 1u, k >> 1u);
      let d = u.domainLo + u.domainSize
        * vec2f(f32(cx + q.x) / f32(u.grid.x - 1u), f32(cy + q.y) / f32(u.grid.y - 1u));
      let w = vec2f(t.linear.x * d.x + t.linear.y * d.y + t.offset.x,
                    t.linear.z * d.x + t.linear.w * d.y + t.offset.y);
      kept = inside_region(w, u.regionCount);
    }
  }

  let ix = min((cx + c.x) * u.step, u.gridSize.x - 1u);
  let iy = min((cy + c.y) * u.step, u.gridSize.y - 1u);
  let h = min(textureLoad(heights, vec2u(ix, iy), 0).r, u.truncationHeight);
  let d = u.domainLo + u.domainSize
    * vec2f(f32(cx + c.x) / f32(u.grid.x - 1u), f32(cy + c.y) / f32(u.grid.y - 1u));
  let world = vec3f(t.linear.x * d.x + t.linear.y * d.y + t.offset.x,
                    t.linear.z * d.x + t.linear.w * d.y + t.offset.y, h);
  return HeightVertex(world, view_depth(world), kept);
}

@vertex
fn kv_height_vertex(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> DepthOut {
  let v = height_vertex(vid, iid);
  var out: DepthOut;
  out.position = to_ndc(v.world);
  out.depth = v.depth;
  if (!v.kept) {
    // Off-screen in w: the whole triangle is culled, and no fragment of it
    // ever reaches the blend.
    out.position = vec4f(0.0, 0.0, 0.0, 0.0);
  }
  return out;
}

// ---------------------------------------------------------------------------
// walls: a small explicit mesh
// ---------------------------------------------------------------------------

@vertex
fn kv_mesh_vertex(@builtin(vertex_index) vid: u32) -> DepthOut {
  let world = walls[vid].position;
  var out: DepthOut;
  out.position = to_ndc(world);
  out.depth = view_depth(world);
  return out;
}

// ---------------------------------------------------------------------------
// the depth fragment: write view z, MAX-blended
// ---------------------------------------------------------------------------
//
// A color attachment with a MAX blend rather than a depth attachment: it is
// exactly \`GL_MAX\` over view z, which is what the Python Z-buffer holds, in
// the same float32 the readback and the clipper compare. (WebGPU blends an
// r32float target only with the \`float32-blendable\` feature, which the page
// asks for.)

@fragment
fn kv_depth_fragment(in: DepthOut) -> @location(0) f32 {
  return in.depth;
}

// ---------------------------------------------------------------------------
// a parametric surface: an implicit mesh over a position texture
// ---------------------------------------------------------------------------
//
// The heightfield's scheme with the texel holding the whole point rather
// than its height: the vertex id names a cell and a corner, the position
// comes from a texel read, and a periodic axis wraps its index so the last
// cell closes onto the first. One vertex function serves both passes, so
// the depth pass and the coordinate pass rasterize the same triangles --
// and the same ones, corner for corner, as \`SurfaceImage.rasterize\`.

struct SurfaceOut {
  @builtin(position) position: vec4f,
  @location(0) depth: f32,
  @location(1) coord: vec2f,
}

fn param_index(vid: u32) -> vec2u {
  let cell = vid / 6u;
  let corner = vid % 6u;
  var corners = CORNERS;
  return vec2u(cell % sf.cells.x, cell / sf.cells.x) + corners[corner];
}

fn param_world(vid: u32) -> vec3f {
  let index = param_index(vid);
  return textureLoad(positions, index % sf.samples, 0).xyz;
}

@vertex
fn kv_param_vertex(@builtin(vertex_index) vid: u32) -> SurfaceOut {
  let index = param_index(vid);
  let world = textureLoad(positions, index % sf.samples, 0).xyz;

  var out: SurfaceOut;
  out.position = to_ndc(world);
  out.depth = view_depth(world);
  out.coord = sf.lo + sf.spacing * vec2f(index);
  // Only the coordinate pass has a depth attachment; the depth pass
  // ignores this. Nearest is 0, so LESS keeps what MAX keeps.
  let span = max(sf.depthRange.y - sf.depthRange.x, 1e-30);
  out.position.z = saturate((sf.depthRange.y - out.depth) / span) * out.position.w;
  return out;
}

@fragment
fn kv_param_depth_fragment(in: SurfaceOut) -> @location(0) f32 {
  return in.depth;
}

@fragment
fn kv_coord_fragment(in: SurfaceOut) -> @location(0) vec2f {
  return in.coord;
}

// ---------------------------------------------------------------------------
// preview
// ---------------------------------------------------------------------------
//
// Two passes over the same geometry the bake uses. The first is the depth
// pass, unchanged. The second draws into the visible target and decides
// visibility by sampling that depth texture at the fragment's own pixel --
// which is why there is one depth semantic in this program and not two, and
// why "the preview approximates the bake" is a statement about where the
// test is evaluated rather than about how the picture is made.

struct PreviewOut {
  @builtin(position) position: vec4f,
  @location(0) depth: f32,
  @location(1) world: vec3f,
}

// A parametric surface as paper, or shaded: the grid of \`kv_param_vertex\`.
@vertex
fn kv_param_surface_vertex(@builtin(vertex_index) vid: u32) -> PreviewOut {
  let world = param_world(vid);
  var out: PreviewOut;
  out.position = to_ndc(world);
  out.depth = view_depth(world);
  out.world = world;
  return out;
}

@vertex
fn kv_surface_vertex(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> PreviewOut {
  let v = height_vertex(vid, iid);
  var out: PreviewOut;
  out.position = select(vec4f(0.0, 0.0, 0.0, 0.0), to_ndc(v.world), v.kept);
  out.depth = v.depth;
  out.world = v.world;
  return out;
}

@vertex
fn kv_wall_vertex(@builtin(vertex_index) vid: u32) -> PreviewOut {
  let world = walls[vid].position;
  var out: PreviewOut;
  out.position = to_ndc(world);
  out.depth = view_depth(world);
  out.world = world;
  return out;
}

// Paper. The drawing's surface is not drawn -- it is white, and the page is
// white -- but it must still be *opaque*, or the strokes behind it show
// through the outline. So the drawing mode paints it flat.
@fragment
fn kv_paper_fragment(in: PreviewOut) -> @location(0) vec4f {
  return vec4f(1.0, 1.0, 1.0, 1.0);
}

// A lit heightfield, for orientation rather than for the drawing. The normal
// comes from the screen-space derivatives of the world position, so it needs
// no normal buffer and no second texture read.
//
// There is no depth attachment on this pass. It does not need one: the depth
// texture already holds the front-most view depth at every pixel, so a
// fragment is the visible surface exactly when its own depth is that one.
// Reusing the buffer is one depth semantic in the program instead of two.
@fragment
fn kv_shaded_fragment(in: PreviewOut) -> @location(0) vec4f {
  // Derivatives first: they must be evaluated before any discard.
  let n = normalize(cross(dpdx(in.world), dpdy(in.world)));
  let behind = textureLoad(depth, vec2u(in.position.xy), 0).r;
  if (in.depth < behind - 1e-4 * max(1.0, abs(behind))) { discard; }
  let lambert = abs(dot(n, normalize(frameShading.lightDirection)));
  let v = frameShading.ambient + (1.0 - frameShading.ambient) * lambert;
  return vec4f(vec3f(v), 1.0);
}

// The depth attachment, drawn directly. This is the substitute for the GPU
// frame-capture viewer, and it is most of what such a viewer actually gets
// used for.
@vertex
fn kv_fullscreen_vertex(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
  var p = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(-1.0, 1.0), vec2f(3.0, 1.0));
  return vec4f(p[vid], 0.0, 1.0);
}

@fragment
fn kv_depth_view_fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let z = textureLoad(depth, vec2u(position.xy), 0).r;
  if (!(z > frameShading.empty)) { return vec4f(0.10, 0.11, 0.13, 1.0); }
  let t = saturate((z - frameShading.depthRange.x)
                   / max(frameShading.depthRange.y - frameShading.depthRange.x, 1e-6));
  return vec4f(vec3f(t), 1.0);
}

// How far the surface's depth moves across one pixel at \`p\`: per axis, the
// smaller of the two one-sided differences, and the larger of the two axes.
//
// The smaller, not the larger, so that an outline on one side -- a jump
// to a far surface, or to nothing -- is not mistaken for slope. Were it
// counted, curves hidden behind a ridge would show through at the ridge.
// Neighbours off the texture or holding \`empty\` do not count at all.
fn axis_slope(p: vec2i, d: vec2i, here: f32, empty: f32) -> f32 {
  let size = vec2i(textureDimensions(depth));
  let a = p - d;
  let b = p + d;
  var hasA = all(a >= vec2i(0)) && all(a < size);
  var hasB = all(b >= vec2i(0)) && all(b < size);
  var za = empty;
  var zb = empty;
  if (hasA) { za = textureLoad(depth, vec2u(a), 0).r; }
  if (hasB) { zb = textureLoad(depth, vec2u(b), 0).r; }
  hasA = hasA && za > empty;
  hasB = hasB && zb > empty;
  let da = abs(here - za);
  let db = abs(zb - here);
  if (hasA && hasB) { return min(da, db); }
  if (hasA) { return da; }
  if (hasB) { return db; }
  return 0.0;
}

fn depth_slope(p: vec2u, here: f32, empty: f32) -> f32 {
  return max(axis_slope(vec2i(p), vec2i(1, 0), here, empty),
             axis_slope(vec2i(p), vec2i(0, 1), here, empty));
}

// Whether curves at view depth \`z\` is in front of the surface at pixel \`p\`.
//
// \`z + depthTolerance > buffer\` is \`clip_hidden_lines\`' predicate, which the
// bake applies per vertex. Here it is applied per fragment, and that moves
// the two things being compared apart: \`z\` is the line's depth where it
// crosses the pixel, up to half a pixel from the centre, and \`buffer\` is the
// surface's depth *at* the centre. On a surface that is steep in view, half
// a pixel is more depth than the depth tolerance, and visible strokes are
// discarded in a pattern that moves whenever the camera does -- which is the
// flicker.
//
// So the preview adds \`slopeScale\` pixels' worth of the surface's own depth
// change to the depthTolerance: the depth bias shadow maps use, for the same
// reason. It is a preview-only relaxation. The bake does not do this, and
// with \`slopeScale\` zero this is the bake's predicate exactly.
fn stroke_visible(p: vec2u, z: f32, s: KVShading) -> bool {
  let behind = textureLoad(depth, p, 0).r;
  if (!(behind > s.empty)) { return true; }
  // An unreachable tolerance passes everything, as Metal's infinity does,
  // without adding to it.
  if (s.depthTolerance >= UNREACHABLE) { return true; }
  let slack = s.depthTolerance + s.slopeScale * depth_slope(p, behind, s.empty);
  return z + slack > behind;
}

// ---------------------------------------------------------------------------
// curves: every segment its own screen-space quad
// ---------------------------------------------------------------------------
//
// Line primitives are one pixel wide whatever the stroke, with no coverage
// and no antialiasing. So each segment is drawn as a quad of its own,
// expanded in screen space to the stroke's width and a pixel beyond, and the
// fragment works out how much of its pixel the stroke covers from its
// distance to the centreline. One instance per segment, reading both ends
// from the layer's buffer: segment i is vertices 2i and 2i+1.
//
// Visibility is decided at the *centreline*, not at the fragment: every
// fragment across a stroke asks whether the point of the centreline beside
// it is in front of the surface. A stroke is visible where its line is,
// which is the question the bake asks too, and a wide stroke is not eaten
// from its edges where the surface is steep in view -- the centreline point
// is always within half a pixel of the pixel it is tested at, which is what
// \`slopeScale\` is sized for.

struct StrokeOut {
  @builtin(position) position: vec4f,
  // View depth, along the segment and constant across it: the depth of the
  // centreline beside this fragment.
  @location(0) depth: f32,
  // That centreline point, in pixels, and this fragment's signed distance
  // from it. Screen-space quantities, so not perspective-corrected.
  @location(1) @interpolate(linear) center: vec2f,
  @location(2) @interpolate(linear) across: f32,
  // How squarely the surface under the curves faces the eye, for curves on
  // a heightfield: negative is hidden. Zero where it is not asked.
  @location(3) @interpolate(linear) facing: f32,
  @location(4) @interpolate(linear) coord: vec2f,
}

struct Clipped { a: vec4f, b: vec4f, da: f32, db: f32, kept: bool }

// Clip a segment to the part the hardware would keep, \`z <= w\`, before
// anything divides by w. An orthographic camera keeps all of it; under
// perspective this is the part in front of the eye. Not kept when there is
// none.
fn clip_near(a0: vec4f, b0: vec4f, da0: f32, db0: f32) -> Clipped {
  let keep = 1e-4;
  var a = a0;
  var b = b0;
  var da = da0;
  var db = db0;
  let ea = a.w - a.z;
  let eb = b.w - b.z;
  if (ea < keep && eb < keep) { return Clipped(a, b, da, db, false); }
  if (ea < keep) {
    let t = (keep - ea) / (eb - ea);
    a = mix(a, b, t); da = mix(da, db, t);
  } else if (eb < keep) {
    let t = (keep - eb) / (ea - eb);
    b = mix(b, a, t); db = mix(db, da, t);
  }
  return Clipped(a, b, da, db, true);
}

// Clip space to pixels, y down: the frame \`@builtin(position)\` arrives in.
fn to_pixels(c: vec4f, viewport: vec2f) -> vec2f {
  let ndc = c.xy / c.w;
  return vec2f(ndc.x + 1.0, 1.0 - ndc.y) * 0.5 * viewport;
}

// Back, keeping the endpoint's own z and w, so the rasterizer interpolates
// depth along the segment exactly as it would have along the line.
fn from_pixels(p: vec2f, c: vec4f, viewport: vec2f) -> vec4f {
  let ndc = vec2f(2.0 * p.x / viewport.x - 1.0, 1.0 - 2.0 * p.y / viewport.y);
  return vec4f(ndc * c.w, c.z, c.w);
}

// Four vertices, as a strip: (a, -), (a, +), (b, -), (b, +). The quad corner
// \`vid\` of the segment from world \`wa\` to world \`wb\`, shared by both kinds
// of curves so they are the same strokes.
fn stroke_corner(vid: u32, wa: vec3f, wb: vec3f, s: KVShading) -> StrokeOut {
  let clipped = clip_near(to_ndc(wa), to_ndc(wb), view_depth(wa), view_depth(wb));
  var out: StrokeOut;
  out.depth = 0.0;
  out.center = vec2f(0.0);
  out.across = 0.0;
  out.facing = 0.0;
  out.coord = vec2f(0.0);
  if (!clipped.kept) {
    // Off-screen in w, as the heightfield culls its masked cells.
    out.position = vec4f(0.0, 0.0, 0.0, 0.0);
    return out;
  }
  let pa = to_pixels(clipped.a, s.viewport);
  let pb = to_pixels(clipped.b, s.viewport);
  let along = pb - pa;
  let len = length(along);
  // A segment seen end-on has no direction; its quad has no length either,
  // so any normal will do and nothing is drawn.
  let dir = select(vec2f(1.0, 0.0), along / len, len > 1e-6);
  let normal = vec2f(-dir.y, dir.x);
  // Half the stroke, and a pixel beyond it for the edge to fall off in.
  let reach = 0.5 * s.strokeWidth + 1.0;
  let atB = vid >= 2u;
  let side = select(-1.0, 1.0, (vid & 1u) != 0u);
  let end = select(pa, pb, atB);
  out.position = from_pixels(end + normal * (side * reach), select(clipped.a, clipped.b, atB), s.viewport);
  out.depth = select(clipped.da, clipped.db, atB);
  out.center = end;
  out.across = side * reach;
  return out;
}

fn stroke_facing(p: vec3f, n: vec3f, k: KVStroke) -> f32 {
  let len = length(n);
  if (len == 0.0) { return 0.0; }
  let toward = select(-k.sight, normalize(k.eye - p), k.perspective != 0u);
  return dot(n, toward) / len;
}

// How much of the pixel the stroke covers, across its width: the overlap of
// the stroke's cross-section with the pixel's, which is a box filter. A
// stroke narrower than a pixel covers less of it rather than drawing
// narrower, so a 0.15 layer reads lighter than a 0.4 one, as it does on
// paper.
fn coverage(across: f32, strokeWidth: f32) -> f32 {
  let d = abs(across);
  let half_width = 0.5 * strokeWidth;
  return min(d + half_width, 0.5) - max(d - half_width, -0.5);
}

// Strokes on a heightfield carry the surface's normal, and are hidden where
// the surface faces away from the eye before the depth test is asked -- the
// test the depth tolerance cannot make, where a steep graph turns away and
// its back lies within the tolerance of its front (see
// \`HeightfieldVisibility\`). A zero normal asks nothing.
@vertex
fn kv_stroke_vertex(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> StrokeOut {
  let a = lineVerts[2u * iid];
  let b = lineVerts[2u * iid + 1u];
  var out = stroke_corner(vid, a.position, b.position, layer.shading);
  var end = a;
  if (vid >= 2u) { end = b; }
  out.facing = stroke_facing(end.position, end.normal, layer.stroke);
  return out;
}

@fragment
fn kv_stroke_fragment(in: StrokeOut) -> @location(0) vec4f {
  let s = layer.shading;
  let c = coverage(in.across, s.strokeWidth);
  if (c <= 0.0) { discard; }
  if (in.facing < 0.0) { discard; }
  let last = vec2f(textureDimensions(depth) - vec2u(1u));
  let p = vec2u(clamp(in.center, vec2f(0.0), last));
  if (!stroke_visible(p, in.depth, s)) { discard; }
  return vec4f(s.color.rgb, s.color.a * min(c, 1.0));
}

// ---------------------------------------------------------------------------
// curves on a parametric surface: visible by where on the surface it lies
// ---------------------------------------------------------------------------
//
// The bake's two tests, per fragment. Facing: on a surface that bounds a
// solid, curves whose outward normal faces away from the eye is hidden.
// Front-most: the surface in front of the stroke's centreline, at its pixel
// or one beside it, is the stroke's own neighbourhood of the surface -- a
// tolerance of two per-pixel footprints of the coordinate texture plus a
// grid cell, exactly \`SurfaceVisibility.isFrontMost\`. No depth tolerance,
// and no slope allowance: neither test is about depth.

@vertex
fn kv_surface_stroke_vertex(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> StrokeOut {
  let a = strokeVerts[2u * iid];
  let b = strokeVerts[2u * iid + 1u];
  var out = stroke_corner(vid, a.position, b.position, layer.shading);
  var end = a;
  if (vid >= 2u) { end = b; }
  out.coord = end.coord;
  out.facing = stroke_facing(end.position, end.normal, layer.stroke);
  return out;
}

// The difference of two coordinates the short way round a periodic axis.
fn coord_delta(d: vec2f) -> vec2f {
  let period = sf.spacing * vec2f(sf.cells);
  let wrapped = d - period * round(d / period);
  return select(d, wrapped, sf.cells == sf.samples);
}

struct CoordAt { found: bool, value: vec2f }

fn coord_at(p: vec2i, empty: f32) -> CoordAt {
  let size = vec2i(textureDimensions(coords));
  if (any(p < vec2i(0)) || any(p >= size)) { return CoordAt(false, vec2f(0.0)); }
  let v = textureLoad(coords, vec2u(p), 0).xy;
  if (!(v.x > empty)) { return CoordAt(false, vec2f(0.0)); }
  return CoordAt(true, v);
}

// \`SurfaceVisibility.footprint\`, one screen axis: the smaller one-sided
// difference, so a neighbour across an occlusion edge is not counted.
fn coord_rate(p: vec2i, here: vec2f, d: vec2i, empty: f32) -> vec2f {
  var best = vec2f(UNREACHABLE);
  var any_side = false;
  for (var side = -1; side <= 1; side = side + 2) {
    let there = coord_at(p + side * d, empty);
    if (!there.found) { continue; }
    best = min(best, abs(coord_delta(there.value - here)));
    any_side = true;
  }
  return select(vec2f(0.0), best, any_side);
}

fn front_most(p: vec2i, coord: vec2f, empty: f32) -> bool {
  var covered = false;
  for (var dy = -1; dy <= 1; dy = dy + 1) {
    for (var dx = -1; dx <= 1; dx = dx + 1) {
      let q = p + vec2i(dx, dy);
      let qc = coord_at(q, empty);
      if (!qc.found) { continue; }
      covered = true;
      let f = max(coord_rate(q, qc.value, vec2i(1, 0), empty),
                  coord_rate(q, qc.value, vec2i(0, 1), empty));
      let d = abs(coord_delta(qc.value - coord));
      if (all(d <= 2.0 * f + sf.spacing)) { return true; }
    }
  }
  return !covered;
}

@fragment
fn kv_surface_stroke_fragment(in: StrokeOut) -> @location(0) vec4f {
  let s = layer.shading;
  let c = coverage(in.across, s.strokeWidth);
  if (c <= 0.0) { discard; }
  if (layer.stroke.onFolds == 0u && in.facing < 0.0) { discard; }
  let last = vec2f(textureDimensions(coords) - vec2u(1u));
  let p = vec2i(clamp(in.center, vec2f(0.0), last));
  if (!front_most(p, in.coord, s.empty)) { discard; }
  return vec4f(s.color.rgb, s.color.a * min(c, 1.0));
}

// ---------------------------------------------------------------------------
// layout probe
// ---------------------------------------------------------------------------
//
// Reads every field of the structs the CPU filled with known values and
// writes back what it saw. If the two sides disagree about the layout --
// because this copy of a struct drifted from layout.ts's offsets -- the
// values come back wrong and the probe says which field. Nothing at build
// time checks that WGSL and TypeScript agree, so the agreement is checked
// rather than assumed, as Swift's is.

@group(0) @binding(0) var<uniform> probeU: KVUniforms;
@group(0) @binding(1) var<storage, read_write> probeOut: array<f32>;
@group(0) @binding(2) var<uniform> probeShading: KVShading;
@group(0) @binding(3) var<uniform> probeSurface: KVSurface;
@group(0) @binding(4) var<storage, read> probeVertex: array<KVStrokeVertex>;
@group(0) @binding(5) var<uniform> probeStroke: KVStroke;
@group(0) @binding(6) var<storage, read> probeLine: array<KVLineVertex>;
@group(0) @binding(7) var<storage, read> probeTile: array<KVTile>;

@compute @workgroup_size(1)
fn kv_layout_probe() {
  var k = 0u;
  for (var c = 0u; c < 4u; c = c + 1u) {
    for (var r = 0u; r < 4u; r = r + 1u) { probeOut[k] = probeU.view[c][r]; k = k + 1u; }
  }
  for (var c = 0u; c < 4u; c = c + 1u) {
    for (var r = 0u; r < 4u; r = r + 1u) { probeOut[k] = probeU.clip[c][r]; k = k + 1u; }
  }
  var fields = array<f32, 12>(
    probeU.domainLo.x, probeU.domainLo.y, probeU.domainSize.x, probeU.domainSize.y,
    f32(probeU.grid.x), f32(probeU.grid.y), f32(probeU.gridSize.x), f32(probeU.gridSize.y),
    f32(probeU.step), probeU.truncationHeight, f32(probeU.regionCount), probeU.empty);
  for (var i = 0u; i < 12u; i = i + 1u) { probeOut[k] = fields[i]; k = k + 1u; }
  let sh = probeShading;
  var shading = array<f32, 15>(
    sh.color.x, sh.color.y, sh.color.z, sh.color.w, sh.depthTolerance, sh.empty, sh.slopeScale,
    sh.lightDirection.x, sh.lightDirection.y, sh.lightDirection.z, sh.ambient, sh.strokeWidth,
    sh.depthRange.x, sh.depthRange.y, sh.viewport.x);
  for (var i = 0u; i < 15u; i = i + 1u) { probeOut[k] = shading[i]; k = k + 1u; }
  probeOut[k] = sh.viewport.y; k = k + 1u;
  let s = probeSurface;
  var surface = array<f32, 10>(
    f32(s.samples.x), f32(s.samples.y), f32(s.cells.x), f32(s.cells.y),
    s.lo.x, s.lo.y, s.spacing.x, s.spacing.y, s.depthRange.x, s.depthRange.y);
  for (var i = 0u; i < 10u; i = i + 1u) { probeOut[k] = surface[i]; k = k + 1u; }
  let v = probeVertex[1];
  var vertex = array<f32, 8>(v.position.x, v.position.y, v.position.z,
                             v.normal.x, v.normal.y, v.normal.z, v.coord.x, v.coord.y);
  for (var i = 0u; i < 8u; i = i + 1u) { probeOut[k] = vertex[i]; k = k + 1u; }
  let ik = probeStroke;
  var stroke = array<f32, 8>(ik.sight.x, ik.sight.y, ik.sight.z, ik.eye.x, ik.eye.y, ik.eye.z,
                             f32(ik.perspective), f32(ik.onFolds));
  for (var i = 0u; i < 8u; i = i + 1u) { probeOut[k] = stroke[i]; k = k + 1u; }
  let l = probeLine[1];
  var line = array<f32, 6>(l.position.x, l.position.y, l.position.z, l.normal.x, l.normal.y, l.normal.z);
  for (var i = 0u; i < 6u; i = i + 1u) { probeOut[k] = line[i]; k = k + 1u; }
  let t = probeTile[1];
  var tile = array<f32, 6>(t.linear.x, t.linear.y, t.linear.z, t.linear.w, t.offset.x, t.offset.y);
  for (var i = 0u; i < 6u; i = i + 1u) { probeOut[k] = tile[i]; k = k + 1u; }
}
`;
