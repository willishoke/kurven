// The preview in the page: a canvas that draws what the engine's frames say
// (MetalView.swift's coordinator, without the window). While the camera
// moves it draws drafts; when it has rested a quarter second, the frame at
// rest, whose tested strokes the engine lays out for the camera. Without
// WebGPU it draws the judged strokes at rest and nothing in between.
//
// With a pool of engines (engine/pool.ts), as Swift's window does with its
// threads: the draft stays on the canvas at rest, and the frame at rest is
// laid out off the page's engine, by the pool's helpers, and swapped in when
// it is ready, unless the camera or a control has moved since (an epoch,
// bumped by every move, says so), so a draft is never kept waiting behind
// it. Without WebGPU the judged strokes at rest are baked by the pool the
// same way. Without a pool the page's engine lays out the frame at rest
// itself, and whatever is asked meanwhile waits for it.
//
// One frame is asked of the engine at a time, and what the page holds is
// read as the ask is sent, so the engine never leaves out a layer the page
// let go of meanwhile; a newer wish while one is in flight replaces any
// waiting one (Swift's `pump`: one job in flight, the latest wins).

import type { Answer, EngineClient } from "../engine/client.ts";
import type { EnginePool, RestLayout } from "../engine/pool.ts";
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

export type Wish = "draft" | "rest";

/** What the pool made for a frame at rest, for the page's engine to serve:
 * the layouts of its tested layers, or (without WebGPU) the judged strokes. */
type Made = { laid: RestLayout } | { judged: Extract<Answer, { stale: false }> };

/** A frame at rest whose making the camera or a control overtook. */
class Overtaken extends Error {}

/** What a frame at rest was made from, when it was wished, and whether it
 * is still the one wanted. */
type Given = { made: Made; started: number; current: () => boolean };

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
  private readonly workers: ((work: Wish | null) => void)[] = [];
  /** The frame being computed and drawn now, if any: what the page's
   * indicator says while a slow frame at rest blocks the engine. */
  working: Wish | null = null;
  /** The pool making the frame at rest, if the page has one. */
  readonly pool: EnginePool | null;
  /** Bumped by every move and change: a frame at rest made for an older
   * epoch is not drawn. */
  private epoch = 0;
  /** The frame at rest being made by the pool, and whether another is wished
   * once it is done. */
  private refining: Promise<void> | null = null;
  private refineAgain = false;
  /** When the frame at rest now being made by the pool was wished. */
  refiningSince: number | null = null;
  /** Frames at rest the page's engine laid out itself because it refused
   * the helpers' layouts for another reason than a move. */
  fallbacks = 0;

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
    pool: EnginePool | null = null,
  ) {
    this.pool = pool;
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

  /** Call `listener` when a frame starts being asked for and drawn, and with
   * null when it is done. */
  onWork(listener: (work: Wish | null) => void): void {
    this.workers.push(listener);
  }

  private setWorking(work: Wish | null): void {
    this.working = work;
    for (const l of this.workers) l(work);
  }

  /** The camera or a control moved: a draft now, and the frame at rest once
   * nothing has moved for a quarter second. */
  moved(): void {
    this.epoch++;
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
    this.epoch++;
    // With a pool the frame at rest comes later, off the page's engine: a
    // draft shows the change now.
    if (this.renderer && this.pool) this.want("draft");
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
    if (wish === "rest" && this.pool) {
      this.refine();
      return;
    }
    void this.exclusive(() => this.draw(wish)).catch((error: unknown) => console.error("kurven preview:", error));
  }

  /** The frame at rest, made by the pool in the background: one at a time,
   * a wish meanwhile made once the one in flight is done. */
  private refine(): void {
    if (this.refining) {
      this.refineAgain = true;
      return;
    }
    this.refiningSince = performance.now();
    this.setWorking(this.working);
    this.refining = this.making()
      .then(
        () => undefined,
        (error: unknown) => console.error("kurven preview:", error),
      )
      .finally(() => {
        this.refining = null;
        this.refiningSince = null;
        this.setWorking(this.working);
        if (this.refineAgain) {
          this.refineAgain = false;
          this.refine();
        }
      });
  }

  /** Make the frame at rest with the pool and draw it, if nothing has moved
   * since it was wished; the timing of the frame drawn, or null. */
  private async making(onPixels?: (p: Pixels) => void, read = false): Promise<Timing | null> {
    const pool = this.pool;
    if (!pool) throw new Error("no pool");
    const epoch = this.epoch;
    const current = () => this.epoch === epoch;
    const started = performance.now();
    let made: Made;
    if (this.renderer) {
      const laid = await pool.layOutRest(this.renderer.holds(), current);
      if (!laid) return null;
      made = { laid };
    } else {
      const answer = await pool.bake(false, current);
      if (answer.stale) return null;
      made = { judged: answer };
    }
    if (!current()) return null;
    try {
      return await this.exclusive(async () => {
        if (!current()) throw new Overtaken();
        return this.draw("rest", onPixels, read, { made, started, current });
      });
    } catch (error) {
      if (error instanceof Overtaken) return null;
      throw error;
    }
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
    while (this.inFlight || this.waiting.draft || this.waiting.rest || this.idle || this.refining) {
      if (this.idle) {
        clearTimeout(this.idle);
        this.idle = null;
        this.want("rest");
      }
      if (this.inFlight) await this.inFlight;
      else if (this.waiting.draft || this.waiting.rest) this.pump();
      else if (this.refining) await this.refining;
    }
  }

  /** Whether the pool is making a frame at rest. */
  get isRefining(): boolean {
    return this.refining !== null;
  }

  /** Draw a frame now, after what is wished, through the same one-at-a-time
   * queue: what the browser lane asks for. `read` reads the pixels it left. */
  async drawNow(wish: Wish, read = false): Promise<{ timing: Timing; pixels: Pixels | null }> {
    await this.settled();
    let pixels: Pixels | null = null;
    if (wish === "rest" && this.pool) {
      // Made by the pool, as at every rest; nothing moves meanwhile.
      const timing = await this.making((p) => {
        pixels = p;
      }, read);
      if (!timing) throw new Error("the frame at rest was overtaken");
      return { timing, pixels };
    }
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

  private async draw(
    wish: Wish,
    onPixels?: (p: Pixels) => void,
    read = false,
    given?: Given,
  ): Promise<Timing> {
    this.setWorking(wish);
    try {
      return await this.drawFrame(wish, onPixels, read, given);
    } finally {
      this.setWorking(null);
    }
  }

  /** Draw a frame: asked of the page's engine, or, at rest with a pool, from
   * what the pool made (`given`), timed from when it was wished. */
  private async drawFrame(
    wish: Wish,
    onPixels?: (p: Pixels) => void,
    read = false,
    given?: Given,
  ): Promise<Timing> {
    const started = given?.started ?? performance.now();
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
      const laid = given && "laid" in given.made ? given.made.laid : null;
      let answer = await this.client.request(
        {
          kind: "preview",
          draft: wish === "draft",
          have,
          ...(laid ? { laid: laid.laid, fingerprint: laid.fingerprint } : {}),
        },
        laid ? { inputs: laid.inputs } : {},
      );
      if (answer.stale) throw new Error("a preview frame went stale");
      if (!answer.reply.ok && laid) {
        // Overtaken by a move or a change since: its own frame at rest
        // follows, and this one is let go quietly.
        if (given && !given.current()) throw new Overtaken();
        // Refused for another reason (the page's engine holds another
        // document than the helpers laid out, for no move the page made):
        // said, and the frame at rest laid out by the page's engine itself.
        this.fallbacks++;
        console.warn(`kurven: the helpers' frame at rest was refused (${answer.reply.error ?? "refused"}); laid out here`);
        answer = await this.client.request({ kind: "preview", draft: false, have });
        if (answer.stale) throw new Error("a preview frame went stale");
      }
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
      const answer =
        given && "judged" in given.made ? given.made.judged : await this.client.request({ kind: "bake" }, { view: "judged" });
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
