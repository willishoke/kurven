// The structs of shaders.ts, written byte for byte where WGSL reads them:
// the TypeScript half of ShaderTypes.h. Each writer names its offsets once,
// by WGSL's layout rules (a vec3f aligns to 16 and takes 12, a struct
// rounds up to its alignment), and `probeLayout` has the GPU read every
// field back, as Swift's `kv_layout_probe` does, because nothing at build
// time checks that the two sides agree.

import { SHADERS, UNREACHABLE } from "./shaders.ts";
import { BufferUsage, MapMode } from "./usage.ts";

export const UNIFORMS_SIZE = 176;
export const SHADING_SIZE = 80;
export const SURFACE_SIZE = 48;
export const STROKE_SIZE = 48;
/** A layer's slot, `KVLayer`, at its own offset in one buffer: 256 bytes
 * apart, WebGPU's least uniform offset alignment. */
export const LAYER_SIZE = SHADING_SIZE + STROKE_SIZE;
export const LAYER_STRIDE = 256;
export const LINE_VERTEX_SIZE = 32;
export const STROKE_VERTEX_SIZE = 48;
export const TILE_SIZE = 32;
export const WALL_VERTEX_SIZE = 16;

/** Swift's infinity, as a shader may hold it (shaders.ts). */
export function finite(x: number | null): number {
  return x === null || !Number.isFinite(x) ? UNREACHABLE : x;
}

export interface KVUniforms {
  /** Column by column, as `Transform.float4x4`. */
  view: ArrayLike<number>;
  clip: ArrayLike<number>;
  domainLo: [number, number];
  domainSize: [number, number];
  grid: [number, number];
  gridSize: [number, number];
  step: number;
  truncationHeight: number;
  regionCount: number;
  empty: number;
}

export function writeUniforms(v: DataView, offset: number, u: KVUniforms): void {
  for (let i = 0; i < 16; i++) v.setFloat32(offset + 4 * i, u.view[i] ?? 0, true);
  for (let i = 0; i < 16; i++) v.setFloat32(offset + 64 + 4 * i, u.clip[i] ?? 0, true);
  v.setFloat32(offset + 128, u.domainLo[0], true);
  v.setFloat32(offset + 132, u.domainLo[1], true);
  v.setFloat32(offset + 136, u.domainSize[0], true);
  v.setFloat32(offset + 140, u.domainSize[1], true);
  v.setUint32(offset + 144, u.grid[0], true);
  v.setUint32(offset + 148, u.grid[1], true);
  v.setUint32(offset + 152, u.gridSize[0], true);
  v.setUint32(offset + 156, u.gridSize[1], true);
  v.setUint32(offset + 160, u.step, true);
  v.setFloat32(offset + 164, u.truncationHeight, true);
  v.setUint32(offset + 168, u.regionCount, true);
  v.setFloat32(offset + 172, u.empty, true);
}

export interface KVShading {
  color: ArrayLike<number>;
  depthTolerance: number;
  empty: number;
  slopeScale: number;
  lightDirection: ArrayLike<number>;
  ambient: number;
  strokeWidth: number;
  depthRange: [number, number];
  viewport: [number, number];
}

export function writeShading(v: DataView, offset: number, s: KVShading): void {
  for (let i = 0; i < 4; i++) v.setFloat32(offset + 4 * i, s.color[i] ?? 0, true);
  v.setFloat32(offset + 16, s.depthTolerance, true);
  v.setFloat32(offset + 20, s.empty, true);
  v.setFloat32(offset + 24, s.slopeScale, true);
  for (let i = 0; i < 3; i++) v.setFloat32(offset + 32 + 4 * i, s.lightDirection[i] ?? 0, true);
  v.setFloat32(offset + 44, s.ambient, true);
  v.setFloat32(offset + 48, s.strokeWidth, true);
  v.setFloat32(offset + 56, s.depthRange[0], true);
  v.setFloat32(offset + 60, s.depthRange[1], true);
  v.setFloat32(offset + 64, s.viewport[0], true);
  v.setFloat32(offset + 68, s.viewport[1], true);
}

export interface KVSurface {
  samples: [number, number];
  cells: [number, number];
  lo: [number, number];
  spacing: [number, number];
  depthRange: [number, number];
}

export function writeSurface(v: DataView, offset: number, s: KVSurface): void {
  v.setUint32(offset, s.samples[0], true);
  v.setUint32(offset + 4, s.samples[1], true);
  v.setUint32(offset + 8, s.cells[0], true);
  v.setUint32(offset + 12, s.cells[1], true);
  v.setFloat32(offset + 16, s.lo[0], true);
  v.setFloat32(offset + 20, s.lo[1], true);
  v.setFloat32(offset + 24, s.spacing[0], true);
  v.setFloat32(offset + 28, s.spacing[1], true);
  v.setFloat32(offset + 32, s.depthRange[0], true);
  v.setFloat32(offset + 36, s.depthRange[1], true);
}

export interface KVStroke {
  sight: ArrayLike<number>;
  eye: ArrayLike<number>;
  perspective: boolean;
  onFolds: boolean;
}

export function writeStroke(v: DataView, offset: number, k: KVStroke): void {
  for (let i = 0; i < 3; i++) v.setFloat32(offset + 4 * i, k.sight[i] ?? 0, true);
  for (let i = 0; i < 3; i++) v.setFloat32(offset + 16 + 4 * i, k.eye[i] ?? 0, true);
  v.setUint32(offset + 28, k.perspective ? 1 : 0, true);
  v.setUint32(offset + 32, k.onFolds ? 1 : 0, true);
}

/** Segments as `KVLineVertex`es: position and normal, three floats each. */
export function lineVertices(positions: Float32Array, normals: Float32Array): ArrayBuffer {
  const n = positions.length / 3;
  const out = new Float32Array((n * LINE_VERTEX_SIZE) / 4);
  for (let k = 0; k < n; k++) {
    const o = (k * LINE_VERTEX_SIZE) / 4;
    out.set(positions.subarray(3 * k, 3 * k + 3), o);
    out.set(normals.subarray(3 * k, 3 * k + 3), o + 4);
  }
  return out.buffer;
}

/** Segments as `KVStrokeVertex`es: position, normal and coordinate. */
export function strokeVertices(positions: Float32Array, normals: Float32Array, coords: Float32Array): ArrayBuffer {
  const n = positions.length / 3;
  const out = new Float32Array((n * STROKE_VERTEX_SIZE) / 4);
  for (let k = 0; k < n; k++) {
    const o = (k * STROKE_VERTEX_SIZE) / 4;
    out.set(positions.subarray(3 * k, 3 * k + 3), o);
    out.set(normals.subarray(3 * k, 3 * k + 3), o + 4);
    out.set(coords.subarray(2 * k, 2 * k + 2), o + 8);
  }
  return out.buffer;
}

/** `KVTile`s from six floats a tile, `(a, b, c, d, tx, ty)`. */
export function tileValues(tiles: Float32Array): ArrayBuffer {
  const n = tiles.length / 6;
  const out = new Float32Array((Math.max(n, 1) * TILE_SIZE) / 4);
  for (let k = 0; k < n; k++) out.set(tiles.subarray(6 * k, 6 * k + 6), (k * TILE_SIZE) / 4);
  return out.buffer;
}

/** `KVVertex`es from three floats a vertex. */
export function wallVertices(walls: Float32Array): ArrayBuffer {
  const n = walls.length / 3;
  const out = new Float32Array((Math.max(n, 1) * WALL_VERTEX_SIZE) / 4);
  for (let k = 0; k < n; k++) out.set(walls.subarray(3 * k, 3 * k + 3), (k * WALL_VERTEX_SIZE) / 4);
  return out.buffer;
}

/** What `probeLayout` sends, field by field, in the order the probe writes
 * them back: distinct values, so a field read at the wrong offset is seen. */
function probeValues() {
  const view = Array.from({ length: 16 }, (_, i) => 1 + i);
  const clip = Array.from({ length: 16 }, (_, i) => 101 + i);
  const u: KVUniforms = {
    view,
    clip,
    domainLo: [201, 202],
    domainSize: [203, 204],
    grid: [205, 206],
    gridSize: [207, 208],
    step: 209,
    truncationHeight: 210,
    regionCount: 211,
    empty: -1e30,
  };
  const sh: KVShading = {
    color: [301, 302, 303, 304],
    depthTolerance: 305,
    empty: 306,
    slopeScale: 307,
    lightDirection: [308, 309, 310],
    ambient: 311,
    strokeWidth: 312,
    depthRange: [313, 314],
    viewport: [315, 316],
  };
  const sf: KVSurface = { samples: [401, 402], cells: [403, 404], lo: [405, 406], spacing: [407, 408], depthRange: [409, 410] };
  const vertex = [501, 502, 503, 504, 505, 506, 507, 508];
  const k: KVStroke = { sight: [601, 602, 603], eye: [604, 605, 606], perspective: true, onFolds: false };
  const line = [701, 702, 703, 704, 705, 706];
  const tile = [801, 802, 803, 804, 805, 806];
  const expected = [
    ...view,
    ...clip,
    ...u.domainLo,
    ...u.domainSize,
    ...u.grid,
    ...u.gridSize,
    u.step,
    u.truncationHeight,
    u.regionCount,
    u.empty,
    ...Array.from(sh.color),
    sh.depthTolerance,
    sh.empty,
    sh.slopeScale,
    ...Array.from(sh.lightDirection),
    sh.ambient,
    sh.strokeWidth,
    ...sh.depthRange,
    ...sh.viewport,
    ...sf.samples,
    ...sf.cells,
    ...sf.lo,
    ...sf.spacing,
    ...sf.depthRange,
    ...vertex,
    ...Array.from(k.sight),
    ...Array.from(k.eye),
    1,
    0,
    ...line,
    ...tile,
  ];
  return { u, sh, sf, vertex, k, line, tile, expected };
}

/** Names of the probe's fields, for a message that says which disagrees. */
const PROBE_NAMES = [
  ...Array.from({ length: 16 }, (_, i) => `view[${i >> 2}][${i & 3}]`),
  ...Array.from({ length: 16 }, (_, i) => `clip[${i >> 2}][${i & 3}]`),
  "domainLo.x", "domainLo.y", "domainSize.x", "domainSize.y", "grid.x", "grid.y", "gridSize.x", "gridSize.y",
  "step", "truncationHeight", "regionCount", "empty",
  "color.r", "color.g", "color.b", "color.a", "depthTolerance", "shading.empty", "slopeScale",
  "lightDirection.x", "lightDirection.y", "lightDirection.z", "ambient", "strokeWidth",
  "depthRange.x", "depthRange.y", "viewport.x", "viewport.y",
  "samples.x", "samples.y", "cells.x", "cells.y", "lo.x", "lo.y", "spacing.x", "spacing.y",
  "surface.depthRange.x", "surface.depthRange.y",
  "vertex.position.x", "vertex.position.y", "vertex.position.z", "vertex.normal.x", "vertex.normal.y",
  "vertex.normal.z", "vertex.coord.x", "vertex.coord.y",
  "sight.x", "sight.y", "sight.z", "eye.x", "eye.y", "eye.z", "perspective", "onFolds",
  "line.position.x", "line.position.y", "line.position.z", "line.normal.x", "line.normal.y", "line.normal.z",
  "tile.a", "tile.b", "tile.c", "tile.d", "tile.tx", "tile.ty",
];

/** What the shader sees when the page hands it these structs: every field
 * it read wrong, by name, or none. */
export async function probeLayout(device: GPUDevice): Promise<string[]> {
  const { u, sh, sf, vertex, k, line, tile, expected } = probeValues();
  const module = device.createShaderModule({ code: SHADERS });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "kv_layout_probe" } });
  const uniform = (size: number, write: (v: DataView) => void) => {
    const bytes = new ArrayBuffer(size);
    write(new DataView(bytes));
    const b = device.createBuffer({ size, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, bytes);
    return b;
  };
  const storage = (bytes: ArrayBuffer) => {
    const b = device.createBuffer({ size: bytes.byteLength, usage: BufferUsage.STORAGE | BufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, bytes);
    return b;
  };
  const twice = (values: number[], pack: (a: Float32Array, b: Float32Array, c: Float32Array) => ArrayBuffer) => {
    // Two vertices, the probe reading the second: an element's stride is
    // checked with its fields.
    const p = new Float32Array([0, 0, 0, ...values.slice(0, 3)]);
    const n = new Float32Array([0, 0, 0, ...values.slice(3, 6)]);
    const c = new Float32Array([0, 0, ...values.slice(6, 8)]);
    return pack(p, n, c);
  };
  const out = device.createBuffer({
    size: 4 * expected.length,
    usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC,
  });
  const read = device.createBuffer({ size: 4 * expected.length, usage: BufferUsage.MAP_READ | BufferUsage.COPY_DST });
  const group = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform(UNIFORMS_SIZE, (v) => writeUniforms(v, 0, u)) } },
      { binding: 1, resource: { buffer: out } },
      { binding: 2, resource: { buffer: uniform(SHADING_SIZE, (v) => writeShading(v, 0, sh)) } },
      { binding: 3, resource: { buffer: uniform(SURFACE_SIZE, (v) => writeSurface(v, 0, sf)) } },
      { binding: 4, resource: { buffer: storage(twice(vertex, strokeVertices)) } },
      { binding: 5, resource: { buffer: uniform(STROKE_SIZE, (v) => writeStroke(v, 0, k)) } },
      { binding: 6, resource: { buffer: storage(twice(line, (p, n) => lineVertices(p, n))) } },
      { binding: 7, resource: { buffer: storage(tileValues(new Float32Array([0, 0, 0, 0, 0, 0, ...tile]))) } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.copyBufferToBuffer(out, 0, read, 0, 4 * expected.length);
  device.queue.submit([encoder.finish()]);
  await read.mapAsync(MapMode.READ);
  const seen = new Float32Array(read.getMappedRange().slice(0));
  read.unmap();
  const wrong: string[] = [];
  expected.forEach((want, i) => {
    if (seen[i] !== Math.fround(want)) wrong.push(`${PROBE_NAMES[i] ?? i}: sent ${want}, read ${seen[i]}`);
  });
  return wrong;
}
