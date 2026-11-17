// The preview in the page: a canvas that draws what the engine's frames say
// (MetalView.swift's coordinator, without the window). While the camera
// moves it draws drafts; when it has rested a quarter second, the frame at
// rest, whose tested strokes the engine lays out for the camera. Without
// WebGPU it draws the judged strokes at rest and nothing in between.
//
// One frame is asked of the engine at a time, and what the page holds is
// read as the ask is sent, so the engine never leaves out a layer the page
// let go of meanwhile; a newer wish while one is in flight replaces any
// waiting one (Swift's `pump`: one job in flight, the latest wins).

import type { EngineClient } from "../engine/client.ts";
import type { Reply, TypedArray } from "../engine/protocol.ts";
import { drawJudged, type Judged } from "./fallback.ts";
import type { Arrays, PreviewFrame } from "./frame.ts";
import { DEFAULT_OPTIONS, type PreviewRenderer, type RenderOptions } from "./renderer.ts";
import { TextureUsage } from "./usage.ts";

/** How long one frame took: in the engine (the request answered, arrays
 * and all), and on the GPU (submitted to done), in milliseconds. */
export interface Timing {
  kind: "draft" | "rest" | "judged";
  engineMs: number;
  drawMs: number;
  totalMs: number;
  /** Bytes the reply carried. */
  bytes: number;
}

/** The pixels a frame left on the canvas, row by row, four bytes each. */
export interface Pixels {
  width: number;
  height: number;
  order: "bgra" | "rgba";
  data: Uint8Array;
}

type Wish = "draft" | "rest";

export class PreviewView {
  readonly canvas: HTMLCanvasElement;
  private readonly client: EngineClient;
  private readonly renderer: PreviewRenderer | null;
  private readonly gpu: GPUCanvasContext | null;
  private readonly ctx: CanvasRenderingContext2D | null;
  options: RenderOptions;
  /** The last frame drawn, and what it was. */
  last: { kind: Timing["kind"]; timing: Timing } | null = null;
  /** The last judged strokes drawn, for Export and for the browser lane. */
  judged: { judged: Judged; arrays: Record<string, TypedArray> } | null = null;
  /** The last preview frame drawn. */
  frame: PreviewFrame | null = null;
  readonly timings: Timing[] = [];
  private inFlight: Promise<void> | null = null;
  private waiting: Wish | null = null;
  private idle: ReturnType<typeof setTimeout> | null = null;
  private readonly listeners: ((t: Timing) => void)[] = [];

  constructor(client: EngineClient, canvas: HTMLCanvasElement, renderer: PreviewRenderer | null, options = DEFAULT_OPTIONS) {
    this.client = client;
    this.canvas = canvas;
    this.renderer = renderer;
    this.options = { ...options };
    if (renderer) {
      const gpu = canvas.getContext("webgpu") as GPUCanvasContext | null;
      if (!gpu) throw new Error("the canvas has no WebGPU context");
      gpu.configure({
        device: renderer.device,
        format: renderer.format,
        alphaMode: "opaque",
        usage: TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC,
      });
      this.gpu = gpu;
      this.ctx = null;
    } else {
      this.gpu = null;
      this.ctx = canvas.getContext("2d");
    }
  }

  get usesGPU(): boolean {
    return this.renderer !== null;
  }

  /** Call `listener` after every frame drawn. */
  onFrame(listener: (t: Timing) => void): void {
    this.listeners.push(listener);
  }

  /** The camera or a control moved: a draft now, and the frame at rest once
   * nothing has moved for a quarter second. */
  moved(): void {
    if (this.renderer) this.want("draft");
    if (this.idle) clearTimeout(this.idle);
    this.idle = setTimeout(() => {
      this.idle = null;
      this.want("rest");
    }, 250);
  }

  /** Something changed that is not motion -- a mode, a layer shown -- and
   * the frame at rest is wanted again. */
  changed(): void {
    this.want("rest");
  }

  private want(wish: Wish): void {
    // A rest wished while a draft waits keeps the draft first: the draft is
    // what the camera shows now, and the rest follows it.
    this.waiting = wish;
    if (!this.inFlight) this.pump();
  }

  private pump(): void {
    const wish = this.waiting;
    if (!wish) return;
    this.waiting = null;
    this.inFlight = this.draw(wish)
      .then(() => undefined)
      .catch((error: unknown) => console.error("kurven preview:", error))
      .finally(() => {
        this.inFlight = null;
        this.pump();
      });
  }

  /** Wait for every frame wished so far to be drawn. */
  async settled(): Promise<void> {
    while (this.inFlight || this.waiting || this.idle) {
      if (this.idle) {
        clearTimeout(this.idle);
        this.idle = null;
        this.want("rest");
      }
      if (this.inFlight) await this.inFlight;
      else if (this.waiting) this.pump();
    }
  }

  /** Draw a frame now, outside the wishes: what the browser lane asks for.
   * `read` reads the pixels it left. */
  async drawNow(wish: Wish, read = false): Promise<{ timing: Timing; pixels: Pixels | null }> {
    await this.settled();
    let pixels: Pixels | null = null;
    const timing = await this.draw(wish, (p) => {
      pixels = p;
    }, read);
    return { timing, pixels };
  }

  private async draw(wish: Wish, onPixels?: (p: Pixels) => void, read = false): Promise<Timing> {
    const started = performance.now();
    let timing: Timing;
    if (this.renderer && this.gpu) {
      const renderer = this.renderer;
      const have = renderer.holds();
      const answer = await this.client.request({ kind: "preview", draft: wish === "draft", have });
      if (answer.stale) throw new Error("a preview frame went stale");
      const { reply, arrays } = answer;
      if (!reply.ok) throw new Error(`preview: ${reply.error ?? "refused"}`);
      const engineMs = performance.now() - started;
      const frame = reply as unknown as Reply & PreviewFrame;
      const drawStarted = performance.now();
      renderer.take(frame, arrays as Arrays);
      const target = this.gpu.getCurrentTexture();
      renderer.render(frame, target, this.options);
      // Read in the same task, before the canvas texture is presented.
      const pixels = read ? renderer.readPixels(target) : null;
      await renderer.device.queue.onSubmittedWorkDone();
      const drawMs = performance.now() - drawStarted;
      if (pixels && onPixels) {
        onPixels({ width: target.width, height: target.height, order: orderOf(renderer.format), data: await pixels });
      }
      this.frame = frame;
      timing = { kind: wish, engineMs, drawMs, totalMs: performance.now() - started, bytes: bytesOf(arrays) };
    } else {
      // Judged strokes only, at rest; a draft draws nothing new.
      if (wish === "draft") {
        return { kind: "draft", engineMs: 0, drawMs: 0, totalMs: 0, bytes: 0 };
      }
      const answer = await this.client.request({ kind: "bake" }, { view: "judged" });
      if (answer.stale) throw new Error("the judged strokes went stale");
      const { reply, arrays } = answer;
      if (!reply.ok) throw new Error(`bake: ${reply.error ?? "refused"}`);
      const engineMs = performance.now() - started;
      const judged = reply as unknown as Reply & Judged;
      const drawStarted = performance.now();
      if (this.ctx) drawJudged(this.ctx, judged, arrays, this.options.strokeWidth);
      const drawMs = performance.now() - drawStarted;
      this.judged = { judged, arrays };
      if (read && onPixels && this.ctx) {
        const { width, height } = this.canvas;
        const data = this.ctx.getImageData(0, 0, width, height).data;
        onPixels({ width, height, order: "rgba", data: new Uint8Array(data.buffer.slice(0)) });
      }
      timing = { kind: "judged", engineMs, drawMs, totalMs: performance.now() - started, bytes: bytesOf(arrays) };
    }
    this.last = { kind: timing.kind, timing };
    this.timings.push(timing);
    for (const l of this.listeners) l(timing);
    return timing;
  }
}

function orderOf(format: GPUTextureFormat): Pixels["order"] {
  return format.startsWith("bgra") ? "bgra" : "rgba";
}

function bytesOf(arrays: Record<string, TypedArray>): number {
  let n = 0;
  for (const a of Object.values(arrays)) n += a.byteLength;
  return n;
}
