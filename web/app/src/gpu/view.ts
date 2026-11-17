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
  /** The frame being asked for and drawn: one at a time, wished or asked
   * for now, so no two asks of the engine carry the same holdings. */
  private inFlight: Promise<void> | null = null;
  /** Wished and not yet asked for: a draft first, then the frame at rest. */
  private waiting = { draft: false, rest: false };
  /** The client's restarts the renderer's holdings are of. */
  private restartsSeen = 0;
  private idle: ReturnType<typeof setTimeout> | null = null;
  private readonly listeners: ((t: Timing) => void)[] = [];

  /** Where an offscreen frame is drawn, when the canvas is not (`offscreen`). */
  private target: GPUTexture | null = null;
  readonly offscreen: boolean;

  /** `offscreen`: draw each frame into a texture of the canvas's size and
   * format and never configure the canvas, so nothing is presented. The
   * pixels are the ones the canvas would show: the same renderer, pipelines
   * and target format (the preferred canvas format the renderer's colour
   * pipelines are built for), the same size; the browser lane holds the two
   * to each other byte for byte where presentation works. For a GPU process
   * that can run WebGPU but not present it (SwiftShader on Linux, headless). */
  constructor(
    client: EngineClient,
    canvas: HTMLCanvasElement,
    renderer: PreviewRenderer | null,
    options = DEFAULT_OPTIONS,
    offscreen = false,
  ) {
    this.offscreen = offscreen;
    this.client = client;
    this.canvas = canvas;
    this.renderer = renderer;
    this.options = { ...options };
    if (renderer && offscreen) {
      this.gpu = null;
      this.ctx = null;
    } else if (renderer) {
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
    // what the camera shows now, and the rest follows it. A draft wished
    // while a rest waits supersedes it: the camera moved again, and its rest
    // comes when it stops (`moved`).
    if (wish === "draft") this.waiting = { draft: true, rest: false };
    else this.waiting.rest = true;
    this.pump();
  }

  private pump(): void {
    if (this.inFlight) return;
    const wish: Wish | null = this.waiting.draft ? "draft" : this.waiting.rest ? "rest" : null;
    if (!wish) return;
    this.waiting[wish] = false;
    void this.exclusive(() => this.draw(wish)).catch((error: unknown) => console.error("kurven preview:", error));
  }

  /** Run `job` when no other frame is in flight, as the one in flight. */
  private async exclusive<T>(job: () => Promise<T>): Promise<T> {
    while (this.inFlight) await this.inFlight;
    const running = job();
    this.inFlight = running.then(
      () => undefined,
      () => undefined,
    );
    try {
      return await running;
    } finally {
      this.inFlight = null;
      this.pump();
    }
  }

  /** The offscreen target at the canvas's size, made again when it changes. */
  private offscreenTarget(): GPUTexture {
    const renderer = this.renderer;
    if (!renderer) throw new Error("no renderer");
    const { width, height } = this.canvas;
    if (!this.target || this.target.width !== width || this.target.height !== height) {
      this.target?.destroy();
      this.target = renderer.device.createTexture({
        size: [width, height],
        format: renderer.format,
        usage: TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC,
      });
    }
    return this.target;
  }

  /** The last frame drawn again, into the offscreen target, and its pixels:
   * what the browser lane holds the canvas's pixels to. */
  async redrawOffscreen(): Promise<Pixels> {
    const renderer = this.renderer;
    const frame = this.frame;
    if (!renderer || !frame) throw new Error("no frame drawn on the GPU");
    return this.exclusive(async () => {
      const target = this.offscreenTarget();
      renderer.render(frame, target, this.options);
      const data = await renderer.readPixels(target);
      return { width: target.width, height: target.height, order: orderOf(renderer.format), data };
    });
  }

  /** Wait for every frame wished so far to be drawn. */
  async settled(): Promise<void> {
    while (this.inFlight || this.waiting.draft || this.waiting.rest || this.idle) {
      if (this.idle) {
        clearTimeout(this.idle);
        this.idle = null;
        this.want("rest");
      }
      if (this.inFlight) await this.inFlight;
      else this.pump();
    }
  }

  /** Draw a frame now, after what is wished, through the same one-at-a-time
   * queue: what the browser lane asks for. `read` reads the pixels it left. */
  async drawNow(wish: Wish, read = false): Promise<{ timing: Timing; pixels: Pixels | null }> {
    await this.settled();
    let pixels: Pixels | null = null;
    const timing = await this.exclusive(() =>
      this.draw(
        wish,
        (p) => {
          pixels = p;
        },
        read,
      ),
    );
    return { timing, pixels };
  }

  private async draw(wish: Wish, onPixels?: (p: Pixels) => void, read = false): Promise<Timing> {
    const started = performance.now();
    let timing: Timing;
    if (this.renderer) {
      const renderer = this.renderer;
      // A restarted engine names its scenes afresh (its keys start with its
      // own nonce, so none it sends can be mistaken for a held one); what the
      // renderer holds from the engine before is let go.
      if (this.client.restarts !== this.restartsSeen) {
        this.restartsSeen = this.client.restarts;
        renderer.forget();
      }
      const have = renderer.holds();
      const answer = await this.client.request({ kind: "preview", draft: wish === "draft", have });
      if (answer.stale) throw new Error("a preview frame went stale");
      const { reply, arrays } = answer;
      if (!reply.ok) throw new Error(`preview: ${reply.error ?? "refused"}`);
      const engineMs = performance.now() - started;
      const frame = reply as unknown as Reply & PreviewFrame;
      const drawStarted = performance.now();
      renderer.take(frame, arrays as Arrays);
      const target = this.gpu ? this.gpu.getCurrentTexture() : this.offscreenTarget();
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
