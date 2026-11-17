// The preview on WebGPU: the drawing half of Sources/KurvenMetal
// (Renderer.swift, Preview.swift, Resources.swift, Surface.swift). The
// engine decides what a frame draws (kurven_session::preview); this draws
// it, holding only GPU copies of what it was sent, under the engine's keys:
//
// - the scene's resources (the height texture, tiles, region and walls; a
//   solid's faces; a parametric surface's position texture), a single-entry
//   memo on the scene's content, as Swift's `cachedResources`, so a camera
//   change uploads uniforms and nothing else;
// - each layer's segments, under its key, which says what they depend on
//   (the curves, and the camera for those tested for it), so a frame
//   re-uploads only the layers it changed, as Swift's `cachedLines`,
//   `cachedSurfaceStrokes` and `cachedFolds` keep the rest.
//
// The passes are Swift's: the depth pass, MAX-blended view z into an
// r32float target; for a parametric surface the coordinate pass beside it;
// then the picture, in one of the three modes, with every stroke tested per
// fragment against what the passes left.

import type { Arrays, LayerJSON, PreviewFrame, ResourcesJSON } from "./frame.ts";
import {
  finite,
  LAYER_STRIDE,
  LINE_VERTEX_SIZE,
  lineVertices,
  SHADING_SIZE,
  STROKE_VERTEX_SIZE,
  strokeVertices,
  SURFACE_SIZE,
  tileValues,
  UNIFORMS_SIZE,
  wallVertices,
  writeShading,
  writeStroke,
  writeSurface,
  writeUniforms,
} from "./layout.ts";
import { SHADERS } from "./shaders.ts";
import { BufferUsage, TextureUsage, MapMode } from "./usage.ts";

/** Written where nothing was drawn (`MetalRenderer.emptySentinel`): WebGPU,
 * as Metal, cannot clear to minus infinity. */
export const EMPTY = -1e30;

/** What a preview frame draws that is the page's to choose
 * (`PreviewOptions`), not the document's. */
export interface RenderOptions {
  /** The widest layer's stroke, in device pixels; every other layer in
   * proportion to its width. Constant on screen rather than scaled with
   * zoom: a stroke that thins as you pull back reads as the picture fading. */
  strokeWidth: number;
  /** How many pixels' worth of the surface's own depth change the
   * visibility test allows on top of the depth tolerance; zero is the
   * bake's predicate exactly. Measured in Swift with `kurven-cli flicker`. */
  slopeScale: number;
  background: [number, number, number, number];
}

export const DEFAULT_OPTIONS: RenderOptions = { strokeWidth: 1.5, slopeScale: 1, background: [1, 1, 1, 1] };

/** Why the page cannot draw the preview on this browser's GPU. */
export class Unsupported extends Error {}

/** The scene's resources on the GPU: built once per content. */
interface SceneResources {
  content: string;
  kind: ResourcesJSON["kind"];
  heights: GPUTexture;
  positions: GPUTexture;
  tiles: GPUBuffer;
  tileCount: number;
  region: GPUBuffer;
  regionCount: number;
  walls: GPUBuffer;
  wallVertexCount: number;
  /** Grid cells of the heightfield or the parametric surface. */
  cells: number;
  json: ResourcesJSON;
  bytes: number;
}

interface LayerGeometry {
  buffer: GPUBuffer;
  count: number;
  surface: boolean;
}

/** Which group-0 bindings each pipeline's entry points read (shaders.ts):
 * an automatic layout holds those and no others. */
const BINDINGS = {
  height: [0, 1, 2, 3],
  mesh: [0, 4],
  paramDepth: [0, 5, 6],
  coord: [0, 5, 6],
  paperSurface: [0, 1, 2, 3],
  paperWall: [0, 4],
  shadedSurface: [0, 1, 2, 3, 7, 9],
  shadedWall: [0, 4, 7, 9],
  stroke: [0, 7],
  depthView: [7, 9],
  paramPaper: [0, 5, 6],
  paramShaded: [0, 5, 6, 7, 9],
  surfaceStroke: [0, 5, 8],
} as const;

type PipelineName = keyof typeof BINDINGS;

export class PreviewRenderer {
  readonly device: GPUDevice;
  readonly format: GPUTextureFormat;
  private readonly pipelines: Record<PipelineName, GPURenderPipeline>;
  private resources: SceneResources | null = null;
  private readonly geometry = new Map<string, LayerGeometry>();
  private readonly uniforms: GPUBuffer;
  private readonly frameShading: GPUBuffer;
  private readonly surface: GPUBuffer;
  private layerSlots: GPUBuffer | null = null;
  private targets: { width: number; height: number; depth: GPUTexture; coords: GPUTexture; z: GPUTexture } | null =
    null;
  /** A one-texel stand-in for a texture a pipeline's layout names and this
   * scene has none of: WebGPU, as Metal's validation, wants every declared
   * binding bound. */
  private readonly blank: { r32: GPUTexture; rgba32: GPUTexture; storage: GPUBuffer };

  /** The device, asked for what the passes need: `float32-blendable`, so the
   * depth pass MAX-blends an r32float target as Metal's does, and the
   * adapter's largest textures and buffers. */
  static async create(format?: GPUTextureFormat): Promise<PreviewRenderer> {
    if (!("gpu" in navigator) || !navigator.gpu) throw new Unsupported("this browser has no WebGPU");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Unsupported("WebGPU has no adapter here");
    if (!adapter.features.has("float32-blendable")) {
      throw new Unsupported("the GPU cannot MAX-blend a float target (float32-blendable), which the depth pass is");
    }
    const device = await adapter.requestDevice({
      requiredFeatures: ["float32-blendable"],
      requiredLimits: {
        maxTextureDimension2D: adapter.limits.maxTextureDimension2D,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    return new PreviewRenderer(device, format ?? navigator.gpu.getPreferredCanvasFormat());
  }

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.format = format;
    void device.lost.then((info) => console.error(`kurven preview, WebGPU device lost (${info.reason}): ${info.message}`));
    device.addEventListener("uncapturederror", (e) => {
      console.error("kurven preview, WebGPU:", (e as GPUUncapturedErrorEvent).error.message);
    });
    const module = device.createShaderModule({ code: SHADERS });
    // The depth pass: view z into r32float, MAX-blended, exactly `GL_MAX`.
    const max: GPUBlendState = {
      color: { operation: "max", srcFactor: "one", dstFactor: "one" },
      alpha: { operation: "max", srcFactor: "one", dstFactor: "one" },
    };
    const depthPipeline = (vertex: string, fragment = "kv_depth_fragment") =>
      device.createRenderPipeline({
        layout: "auto",
        vertex: { module, entryPoint: vertex },
        fragment: { module, entryPoint: fragment, targets: [{ format: "r32float", blend: max }] },
        primitive: { topology: "triangle-list" },
      });
    // Straight alpha: the stroke's coverage, over what is there.
    const over: GPUBlendState = {
      color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
      alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
    };
    // The colour pass has no depth attachment. It does not need one: the
    // depth pass already holds the front-most view depth at every pixel, and
    // both the shaded surface and the strokes decide visibility by reading
    // it: one depth semantic in the program rather than two.
    const colorPipeline = (vertex: string, fragment: string, strip = false, blended = false) =>
      device.createRenderPipeline({
        layout: "auto",
        vertex: { module, entryPoint: vertex },
        fragment: { module, entryPoint: fragment, targets: [{ format, ...(blended ? { blend: over } : {}) }] },
        primitive: { topology: strip ? "triangle-strip" : "triangle-list" },
      });
    this.pipelines = {
      height: depthPipeline("kv_height_vertex"),
      mesh: depthPipeline("kv_mesh_vertex"),
      paramDepth: depthPipeline("kv_param_vertex", "kv_param_depth_fragment"),
      // The coordinate pass cannot MAX-blend: a blend keeps the greatest of
      // each channel separately, and a coordinate is not ordered by depth.
      // So it has a depth attachment of its own and keeps the fragment that
      // wins a LESS test on normalized view depth.
      coord: device.createRenderPipeline({
        layout: "auto",
        vertex: { module, entryPoint: "kv_param_vertex" },
        fragment: { module, entryPoint: "kv_coord_fragment", targets: [{ format: "rg32float" }] },
        primitive: { topology: "triangle-list" },
        depthStencil: { format: "depth32float", depthCompare: "less", depthWriteEnabled: true },
      }),
      paperSurface: colorPipeline("kv_surface_vertex", "kv_paper_fragment"),
      paperWall: colorPipeline("kv_wall_vertex", "kv_paper_fragment"),
      shadedSurface: colorPipeline("kv_surface_vertex", "kv_shaded_fragment"),
      shadedWall: colorPipeline("kv_wall_vertex", "kv_shaded_fragment"),
      stroke: colorPipeline("kv_stroke_vertex", "kv_stroke_fragment", true, true),
      depthView: colorPipeline("kv_fullscreen_vertex", "kv_depth_view_fragment"),
      paramPaper: colorPipeline("kv_param_surface_vertex", "kv_paper_fragment"),
      paramShaded: colorPipeline("kv_param_surface_vertex", "kv_shaded_fragment"),
      surfaceStroke: colorPipeline("kv_surface_stroke_vertex", "kv_surface_stroke_fragment", true, true),
    };
    const uniform = (size: number) =>
      device.createBuffer({ size, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    this.uniforms = uniform(UNIFORMS_SIZE);
    this.frameShading = uniform(SHADING_SIZE);
    this.surface = uniform(SURFACE_SIZE);
    const texture = (format: GPUTextureFormat) =>
      device.createTexture({ size: [1, 1], format, usage: TextureUsage.TEXTURE_BINDING });
    this.blank = {
      r32: texture("r32float"),
      rgba32: texture("rgba32float"),
      storage: device.createBuffer({ size: 64, usage: BufferUsage.STORAGE }),
    };
  }

  /** What the page holds, for the engine to leave out of the next frame. */
  holds(): { content?: string; layers: string[] } {
    return { ...(this.resources ? { content: this.resources.content } : {}), layers: [...this.geometry.keys()] };
  }

  /** Forget every resource and layer, as a new renderer would hold none. */
  forget(): void {
    this.dropResources();
    for (const g of this.geometry.values()) g.buffer.destroy();
    this.geometry.clear();
  }

  /** Bytes held on the GPU for the scene and its layers. */
  heldBytes(): number {
    let n = this.resources?.bytes ?? 0;
    for (const g of this.geometry.values()) n += g.buffer.size;
    return n;
  }

  private dropResources(): void {
    const r = this.resources;
    if (!r) return;
    for (const t of [r.heights, r.positions]) if (t !== this.blank.r32 && t !== this.blank.rgba32) t.destroy();
    for (const b of [r.tiles, r.region, r.walls]) if (b !== this.blank.storage) b.destroy();
    this.resources = null;
  }

  /** `SceneResources.init` and `SurfaceResources.init`, from the engine's
   * values. */
  private build(content: string, json: ResourcesJSON, arrays: Arrays): SceneResources {
    const device = this.device;
    const storage = (bytes: ArrayBuffer) => {
      if (bytes.byteLength === 0) return this.blank.storage;
      const b = device.createBuffer({ size: bytes.byteLength, usage: BufferUsage.STORAGE | BufferUsage.COPY_DST });
      device.queue.writeBuffer(b, 0, bytes);
      return b;
    };
    const f32 = (name: string) => (arrays[name] as Float32Array | undefined) ?? new Float32Array(0);
    const r: SceneResources = {
      content,
      kind: json.kind,
      json,
      heights: this.blank.r32,
      positions: this.blank.rgba32,
      tiles: this.blank.storage,
      tileCount: 0,
      region: this.blank.storage,
      regionCount: 0,
      walls: this.blank.storage,
      wallVertexCount: 0,
      cells: 0,
      bytes: 0,
    };
    const walls = f32("resource.walls");
    if (walls.length > 0) {
      // Walls are expanded to a flat vertex list once. They are a few
      // thousand triangles, so an index buffer would save little.
      r.walls = storage(wallVertices(walls));
      r.wallVertexCount = walls.length / 3;
      r.bytes += r.walls.size;
    }
    if (json.kind === "heightfield") {
      const [w, h] = json.gridSize;
      this.fits(w, h, "the height texture");
      r.heights = device.createTexture({
        size: [w, h],
        format: "r32float",
        usage: TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST,
      });
      device.queue.writeTexture({ texture: r.heights }, f32("resource.heights"), { bytesPerRow: 4 * w }, [w, h]);
      const tiles = f32("resource.tiles");
      r.tiles = storage(tileValues(tiles));
      r.tileCount = tiles.length / 6;
      // Bound even when empty, as every declared binding must be.
      const region = f32("resource.region");
      r.region = region.length > 0 ? storage(region.slice().buffer) : this.blank.storage;
      r.regionCount = region.length / 2;
      r.cells = Math.max(json.grid[0] - 1, 0) * Math.max(json.grid[1] - 1, 0);
      r.bytes += 4 * w * h + r.tiles.size;
    } else if (json.kind === "parametric") {
      const [w, h] = json.samples;
      this.fits(w, h, "the position texture");
      const p = f32("resource.positions");
      const texels = new Float32Array(4 * w * h);
      for (let k = 0; k < w * h; k++) {
        texels[4 * k] = p[3 * k] ?? 0;
        texels[4 * k + 1] = p[3 * k + 1] ?? 0;
        texels[4 * k + 2] = p[3 * k + 2] ?? 0;
        texels[4 * k + 3] = 1;
      }
      r.positions = device.createTexture({
        size: [w, h],
        format: "rgba32float",
        usage: TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST,
      });
      device.queue.writeTexture({ texture: r.positions }, texels, { bytesPerRow: 16 * w }, [w, h]);
      r.cells = json.cells[0] * json.cells[1];
      r.bytes += 16 * w * h;
    }
    return r;
  }

  private fits(w: number, h: number, what: string): void {
    const limit = this.device.limits.maxTextureDimension2D;
    if (w > limit || h > limit) throw new Unsupported(`${what} is ${w} by ${h}, past this GPU's ${limit}`);
  }

  /** The textures the passes write, at the viewport's size. */
  private targetsFor(width: number, height: number) {
    const t = this.targets;
    if (t && t.width === width && t.height === height) return t;
    if (t) for (const x of [t.depth, t.coords, t.z]) x.destroy();
    this.fits(width, height, "the viewport");
    const device = this.device;
    const make = (format: GPUTextureFormat, usage: number) => device.createTexture({ size: [width, height], format, usage });
    const read = TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_SRC;
    this.targets = {
      width,
      height,
      depth: make("r32float", read),
      coords: make("rg32float", read),
      z: make("depth32float", TextureUsage.RENDER_ATTACHMENT),
    };
    return this.targets;
  }

  /** Take in a frame's resources and segments: what the engine sent, kept
   * under its keys; a layer no longer drawn under a key is let go. */
  take(frame: PreviewFrame, arrays: Arrays): void {
    if (frame.resources) {
      this.dropResources();
      this.resources = this.build(frame.content, frame.resources, arrays);
    } else if (this.resources?.content !== frame.content) {
      throw new Error(`the frame is of content ${frame.content}, and the page holds ${this.resources?.content ?? "none"}`);
    }
    const keys = new Set(frame.layers.map((l) => l.key));
    for (const [key, g] of this.geometry) {
      if (!keys.has(key)) {
        g.buffer.destroy();
        this.geometry.delete(key);
      }
    }
    frame.layers.forEach((l, i) => {
      if (l.segments === null) return;
      const positions = arrays[`layer.${i}.positions`] as Float32Array | undefined;
      const normals = arrays[`layer.${i}.normals`] as Float32Array | undefined;
      if (!positions || !normals) throw new Error(`layer ${i} (${l.name}) came without its segments`);
      const surface = l.pipeline === "surface";
      const coords = (arrays[`layer.${i}.coords`] as Float32Array | undefined) ?? new Float32Array(0);
      const bytes = surface ? strokeVertices(positions, normals, coords) : lineVertices(positions, normals);
      this.geometry.get(l.key)?.buffer.destroy();
      const size = Math.max(bytes.byteLength, surface ? STROKE_VERTEX_SIZE : LINE_VERTEX_SIZE);
      const buffer = this.device.createBuffer({ size, usage: BufferUsage.STORAGE | BufferUsage.COPY_DST });
      if (bytes.byteLength > 0) this.device.queue.writeBuffer(buffer, 0, bytes);
      this.geometry.set(l.key, { buffer, count: l.segments, surface });
    });
  }

  /** Draw one preview frame into `target` (`renderPreview`): the depth pass,
   * the coordinate pass for a parametric surface, then the picture. */
  render(frame: PreviewFrame, target: GPUTexture, options: RenderOptions = DEFAULT_OPTIONS): void {
    const res = this.resources;
    if (!res || res.content !== frame.content) throw new Error("render: the frame's resources are not taken in");
    const device = this.device;
    const width = target.width;
    const height = target.height;
    const t = this.targetsFor(width, height);
    const u = frame.uniforms;
    const parametric = res.json.kind === "parametric";

    // KVUniforms: the camera, and the heightfield's decimated grid.
    const hf = res.json.kind === "heightfield" ? res.json : null;
    const uniformBytes = new ArrayBuffer(UNIFORMS_SIZE);
    writeUniforms(new DataView(uniformBytes), 0, {
      view: u.view,
      clip: u.clip,
      domainLo: hf ? hf.domainLo : [0, 0],
      domainSize: hf ? hf.domainSize : [0, 0],
      grid: hf ? hf.grid : [0, 0],
      gridSize: hf ? hf.gridSize : [0, 0],
      step: hf ? hf.step : 1,
      truncationHeight: finite(u.truncationHeight),
      regionCount: res.regionCount,
      empty: EMPTY,
    });
    device.queue.writeBuffer(this.uniforms, 0, uniformBytes);
    const shading = (color: ArrayLike<number>, tolerance: number | null, widthRatio: number) => ({
      color,
      depthTolerance: finite(tolerance),
      empty: EMPTY,
      slopeScale: options.slopeScale,
      lightDirection: u.lightDirection,
      ambient: u.ambient,
      strokeWidth: options.strokeWidth * widthRatio,
      depthRange: u.depthRange,
      viewport: [width, height] as [number, number],
    });
    const frameBytes = new ArrayBuffer(SHADING_SIZE);
    writeShading(new DataView(frameBytes), 0, shading([0, 0, 0, 1], u.depthTolerance, 1));
    device.queue.writeBuffer(this.frameShading, 0, frameBytes);
    if (res.json.kind === "parametric") {
      const surfaceBytes = new ArrayBuffer(SURFACE_SIZE);
      writeSurface(new DataView(surfaceBytes), 0, {
        samples: res.json.samples,
        cells: res.json.cells,
        lo: res.json.lo,
        spacing: res.json.spacing,
        depthRange: u.surfaceDepthRange ?? [-1, 1],
      });
      device.queue.writeBuffer(this.surface, 0, surfaceBytes);
    }

    const bind = (name: PipelineName) =>
      device.createBindGroup({
        layout: this.pipelines[name].getBindGroupLayout(0),
        entries: BINDINGS[name].map((binding) => ({ binding, resource: this.resource(binding, res, t) })),
      });
    const encoder = device.createCommandEncoder();

    // Pass 1: depth only, MAX-blended, the same numbers the bake clips
    // against.
    const depthPass = encoder.beginRenderPass({
      colorAttachments: [
        { view: t.depth.createView(), loadOp: "clear", storeOp: "store", clearValue: { r: EMPTY, g: 0, b: 0, a: 0 } },
      ],
    });
    if (parametric) {
      depthPass.setPipeline(this.pipelines.paramDepth);
      depthPass.setBindGroup(0, bind("paramDepth"));
      depthPass.draw(res.cells * 6);
    } else {
      this.encodeGeometry(depthPass, res, "height", "mesh", bind);
    }
    depthPass.end();

    // Pass 1b: the front-most coordinate, behind its own depth test.
    if (parametric) {
      const coordPass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: t.coords.createView(),
            loadOp: "clear",
            storeOp: "store",
            clearValue: { r: EMPTY, g: EMPTY, b: 0, a: 0 },
          },
        ],
        depthStencilAttachment: { view: t.z.createView(), depthLoadOp: "clear", depthStoreOp: "discard", depthClearValue: 1 },
      });
      coordPass.setPipeline(this.pipelines.coord);
      coordPass.setBindGroup(0, bind("coord"));
      coordPass.draw(res.cells * 6);
      coordPass.end();
    }

    // Pass 2: the picture.
    const [r, g, b, a] = options.background;
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: target.createView(), loadOp: "clear", storeOp: "store", clearValue: { r, g, b, a } }],
    });
    switch (frame.mode) {
      case "drawing":
        if (parametric) this.drawParametric(pass, res, "paramPaper", bind);
        else this.encodeGeometry(pass, res, "paperSurface", "paperWall", bind);
        break;
      case "shaded":
        if (parametric) this.drawParametric(pass, res, "paramShaded", bind);
        else this.encodeGeometry(pass, res, "shadedSurface", "shadedWall", bind);
        break;
      case "depth":
        // The depth attachment *is* the picture: a full-screen triangle maps
        // it to grey.
        pass.setPipeline(this.pipelines.depthView);
        pass.setBindGroup(0, bind("depthView"));
        pass.draw(3);
        break;
    }

    if (frame.mode !== "depth") {
      const drawn = frame.layers
        .map((l) => ({ l, g: this.geometry.get(l.key) }))
        .filter((d): d is { l: LayerJSON; g: LayerGeometry } => d.l.visible && d.g !== undefined && d.g.count > 0);
      if (drawn.length > 0) {
        const slots = this.slotsFor(drawn.length);
        const bytes = new ArrayBuffer(drawn.length * LAYER_STRIDE);
        const view = new DataView(bytes);
        drawn.forEach(({ l }, k) => {
          writeShading(view, k * LAYER_STRIDE, shading(l.color, l.depthTolerance, l.width));
          // Which way the strokes' surface faces is asked of the camera.
          writeStroke(view, k * LAYER_STRIDE + SHADING_SIZE, {
            sight: u.sight,
            eye: u.eye,
            perspective: u.perspective,
            onFolds: l.onFolds,
          });
        });
        device.queue.writeBuffer(slots, 0, bytes);
        const groups = { stroke: bind("stroke"), surfaceStroke: parametric ? bind("surfaceStroke") : null };
        drawn.forEach(({ l, g }, k) => {
          const name = g.surface ? "surfaceStroke" : "stroke";
          const pipeline = this.pipelines[name];
          pass.setPipeline(pipeline);
          const group0 = groups[name];
          if (!group0) throw new Error(`layer ${l.name}: the surface stroke pipeline without a surface`);
          pass.setBindGroup(0, group0);
          pass.setBindGroup(
            1,
            device.createBindGroup({
              layout: pipeline.getBindGroupLayout(1),
              entries: [
                { binding: 0, resource: { buffer: slots, offset: k * LAYER_STRIDE, size: SHADING_SIZE + 48 } },
                { binding: g.surface ? 2 : 1, resource: { buffer: g.buffer } },
              ],
            }),
          );
          // Four vertices a segment, one instance per segment.
          pass.draw(4, g.count);
        });
      }
    }
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  /** The one binding a pipeline's layout names, for this scene. */
  private resource(binding: number, res: SceneResources, t: { depth: GPUTexture; coords: GPUTexture }): GPUBindingResource {
    switch (binding) {
      case 0:
        return { buffer: this.uniforms };
      case 1:
        return { buffer: res.tiles };
      case 2:
        return { buffer: res.region };
      case 3:
        return res.heights.createView();
      case 4:
        return { buffer: res.walls };
      case 5:
        return { buffer: this.surface };
      case 6:
        return res.positions.createView();
      case 7:
        return t.depth.createView();
      case 8:
        return t.coords.createView();
      case 9:
        return { buffer: this.frameShading };
      default:
        throw new Error(`no binding ${binding}`);
    }
  }

  /** Draw the occluding geometry: the implicit heightfield, then the walls.
   * Shared by the depth pass and the picture's surface pass, so the two
   * cannot draw different things (`encodeGeometry`). */
  private encodeGeometry(
    pass: GPURenderPassEncoder,
    res: SceneResources,
    height: PipelineName,
    mesh: PipelineName,
    bind: (name: PipelineName) => GPUBindGroup,
  ): void {
    if (res.cells > 0 && res.tileCount > 0) {
      pass.setPipeline(this.pipelines[height]);
      pass.setBindGroup(0, bind(height));
      // No vertex or index buffer: six vertices per cell, one instance per
      // tile.
      pass.draw(res.cells * 6, res.tileCount);
    }
    if (res.wallVertexCount > 0) {
      pass.setPipeline(this.pipelines[mesh]);
      pass.setBindGroup(0, bind(mesh));
      pass.draw(res.wallVertexCount);
    }
  }

  private drawParametric(
    pass: GPURenderPassEncoder,
    res: SceneResources,
    name: PipelineName,
    bind: (name: PipelineName) => GPUBindGroup,
  ): void {
    pass.setPipeline(this.pipelines[name]);
    pass.setBindGroup(0, bind(name));
    pass.draw(res.cells * 6);
  }

  private slotsFor(n: number): GPUBuffer {
    const size = n * LAYER_STRIDE;
    if (this.layerSlots && this.layerSlots.size >= size) return this.layerSlots;
    this.layerSlots?.destroy();
    this.layerSlots = this.device.createBuffer({ size, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    return this.layerSlots;
  }

  /** A texture's pixels, row by row, four bytes each in its own order
   * (`bgra8unorm` reads B, G, R, A). */
  async readPixels(texture: GPUTexture): Promise<Uint8Array> {
    const { width, height } = texture;
    const row = Math.ceil((4 * width) / 256) * 256;
    const buffer = this.device.createBuffer({ size: row * height, usage: BufferUsage.MAP_READ | BufferUsage.COPY_DST });
    const encoder = this.device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: row }, [width, height]);
    this.device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(MapMode.READ);
    const mapped = new Uint8Array(buffer.getMappedRange());
    const out = new Uint8Array(4 * width * height);
    for (let y = 0; y < height; y++) out.set(mapped.subarray(y * row, y * row + 4 * width), 4 * width * y);
    buffer.unmap();
    buffer.destroy();
    return out;
  }

  /** The depth the last frame's depth pass left, row by row: the preview's
   * depth buffer, what a pick reads in Swift (`previewDepth`). */
  async readDepth(): Promise<Float32Array> {
    const t = this.targets;
    if (!t) return new Float32Array(0);
    const { width, height } = t;
    const row = Math.ceil((4 * width) / 256) * 256;
    const buffer = this.device.createBuffer({ size: row * height, usage: BufferUsage.MAP_READ | BufferUsage.COPY_DST });
    const encoder = this.device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture: t.depth }, { buffer, bytesPerRow: row }, [width, height]);
    this.device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(MapMode.READ);
    const mapped = new Float32Array(buffer.getMappedRange());
    const out = new Float32Array(width * height);
    for (let y = 0; y < height; y++) out.set(mapped.subarray((y * row) / 4, (y * row) / 4 + width), width * y);
    buffer.unmap();
    buffer.destroy();
    return out;
  }
}
