// The page: a thin view of the engine. All state lives in the engine
// (crates/kurven-session, run in a Worker through crates/kurven-wasm); the
// page renders state and sends edits (HANDOFF-web-port.md, Architecture).
//
// Phase 7's page is the preview and enough around it to see it and test it:
// a canvas, orbit (drag), pan (shift-drag), zoom toward the cursor (wheel),
// fit (f), and the three modes (1, 2, 3), as the Swift app's GestureView
// maps them. What it shows comes from the query:
//
//   ?landscape=cn        a catalog landscape (`create`)
//   ?surface=sn          a catalog surface (`createSurface`)
//   &gpu=0               the page without WebGPU: judged strokes at rest
//   &size=900x700        the canvas in device pixels, rather than the window
//   &slope=0             the preview's slope scale (default 1)
//
// A bundle is opened by the browser lane through `window.kurven.open`. The
// formula field, the controls and the rest are Phase 8's.

import { browserWorker, EngineClient } from "./engine/client.ts";
import type { Bundle, DocumentState, Gesture, Request } from "./engine/protocol.ts";
import { DEFAULT_OPTIONS, PreviewRenderer, Unsupported } from "./gpu/renderer.ts";
import { probeLayout } from "./gpu/layout.ts";
import { PreviewView } from "./gpu/view.ts";

const query = new URLSearchParams(location.search);
const app = document.querySelector<HTMLElement>("#app");
if (!app) throw new Error("index.html has no #app");

const canvas = document.createElement("canvas");
canvas.id = "preview";
canvas.style.display = "block";
canvas.style.touchAction = "none";
const status = document.createElement("p");
status.id = "status";
status.style.font = "12px ui-monospace, monospace";
status.style.margin = "4px 8px";
app.replaceChildren(canvas, status);

const scale = window.devicePixelRatio || 1;
const size = query.get("size")?.match(/^(\d+)x(\d+)$/);
const fixed = size ? { width: Number(size[1]), height: Number(size[2]) } : null;
function measure(): { width: number; height: number } {
  if (fixed) return fixed;
  return {
    width: Math.max(1, Math.round(window.innerWidth * scale)),
    height: Math.max(1, Math.round((window.innerHeight - 24) * scale)),
  };
}
const first = measure();
canvas.width = first.width;
canvas.height = first.height;
canvas.style.width = `${first.width / scale}px`;
canvas.style.height = `${first.height / scale}px`;

const client = new EngineClient(browserWorker, `${import.meta.env.BASE_URL}kurven_wasm.wasm`);
let state: DocumentState | null = null;

async function send(request: Request): Promise<DocumentState | null> {
  const answer = await client.request(request);
  if (answer.stale) return state;
  if (!answer.reply.ok) status.textContent = answer.reply.error ?? "refused";
  state = answer.reply.state ?? state;
  return state;
}

/** Why the page draws without WebGPU, if it does. */
let fallbackReason: string | null = null;
async function makeRenderer(): Promise<PreviewRenderer | null> {
  if (query.get("gpu") === "0") {
    fallbackReason = "asked for (gpu=0)";
    return null;
  }
  try {
    return await PreviewRenderer.create();
  } catch (error) {
    if (!(error instanceof Unsupported)) throw error;
    fallbackReason = error.message;
    return null;
  }
}

const renderer = await makeRenderer();
const slope = query.get("slope");
const view = new PreviewView(client, canvas, renderer, {
  ...DEFAULT_OPTIONS,
  // The widest stroke, in device pixels: a stroke as wide on a dense
  // screen as on any other.
  strokeWidth: DEFAULT_OPTIONS.strokeWidth * scale,
  slopeScale: slope !== null ? Number(slope) : DEFAULT_OPTIONS.slopeScale,
});
view.onFrame((t) => {
  status.textContent =
    `${state?.title ?? ""} · ${view.usesGPU ? "WebGPU" : `judged strokes only (${fallbackReason})`} · ` +
    `${t.kind} ${t.engineMs.toFixed(0)} ms engine, ${t.drawMs.toFixed(1)} ms drawing`;
});

await send({ kind: "viewport", width: canvas.width, height: canvas.height });
const landscape = query.get("landscape");
const surface = query.get("surface");
let opened: DocumentState | null = null;
if (landscape) opened = await send({ kind: "create", preset: landscape });
else if (surface) opened = await send({ kind: "createSurface", preset: surface });
if (opened?.kind === "landscape" || opened?.kind === "surface") view.changed();

// MARK: - input, as the Swift app's GestureView maps it

function gesture(g: Gesture): void {
  void send({ kind: "gesture", gesture: g }).then(() => view.moved());
}

let drag: { x: number; y: number } | null = null;
canvas.addEventListener("pointerdown", (e) => {
  drag = { x: e.clientX, y: e.clientY };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const d: [number, number] = [(e.clientX - drag.x) * scale, (e.clientY - drag.y) * scale];
  drag = { x: e.clientX, y: e.clientY };
  if (d[0] === 0 && d[1] === 0) return;
  // Shift or option pans, matching the modifier most CAD tools use for
  // "move the paper rather than the camera".
  if (e.shiftKey || e.altKey) gesture({ kind: "pan", delta: [-d[0], d[1]] });
  else gesture({ kind: "orbit", delta: d });
});
// A drag ends however the pointer is let go of: up, cancelled by the
// browser, or its capture lost.
for (const ended of ["pointerup", "pointercancel", "lostpointercapture"] as const) {
  canvas.addEventListener(ended, () => {
    drag = null;
  });
}
canvas.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    // Toward the cursor, not the centre: under an orthographic camera that
    // is exactly "hold the point under the pointer still".
    const rect = canvas.getBoundingClientRect();
    const at: [number, number] = [(e.clientX - rect.left) * scale, (e.clientY - rect.top) * scale];
    const steps = (-e.deltaY * (e.deltaMode === 0 ? 0.01 : 0.1)) / scale;
    if (steps !== 0) gesture({ kind: "zoom", factor: Math.exp(steps), at });
  },
  { passive: false },
);
window.addEventListener("keydown", (e) => {
  const act = async (r: Request) => {
    await send(r);
    view.changed();
  };
  switch (e.key) {
    case "f":
      void act({ kind: "fit" });
      break;
    case "1":
      void act({ kind: "mode", mode: "drawing" });
      break;
    case "2":
      void act({ kind: "mode", mode: "shaded" });
      break;
    case "3":
      void act({ kind: "mode", mode: "depth" });
      break;
  }
});
window.addEventListener("resize", () => {
  if (fixed) return;
  const next = measure();
  canvas.width = next.width;
  canvas.height = next.height;
  canvas.style.width = `${next.width / scale}px`;
  canvas.style.height = `${next.height / scale}px`;
  void send({ kind: "viewport", width: next.width, height: next.height }).then(() => view.changed());
});

// MARK: - the browser lane's handle (web/app/test/browser)

declare global {
  interface Window {
    kurven: unknown;
  }
}

window.kurven = {
  view,
  usesGPU: view.usesGPU,
  fallbackReason,
  state: () => state,
  send,
  /** A request, its reply and arrays as they come. */
  request: (r: Request) => client.request(r),
  /** Open a bundle given as its files. */
  open: async (bundle: Bundle) => {
    await client.stage(bundle);
    const s = await send({ kind: "open", name: bundle.name });
    view.changed();
    return s;
  },
  /** Draw a frame now, and read the pixels it left. */
  draw: (wish: "draft" | "rest", read = false) => view.drawNow(wish, read),
  /** The shown surface's coordinate image, by the engine's CPU rasterizer
   * and by the GPU's passes, over the same frame. */
  surfaceImage: async (resolution: number) => {
    if (!renderer) throw new Error("no WebGPU");
    const answer = await client.request({ kind: "surfaceImage", resolution });
    if (answer.stale || !answer.reply.ok) throw new Error(answer.stale ? "stale" : answer.reply.error);
    const r = answer.reply as unknown as {
      frame: { rows: number; cols: number };
      view: number[];
      clip: number[];
      surfaceDepthRange: [number, number];
      span: number;
    };
    // The resources the GPU renders from are the frame's: draw one first.
    await view.drawNow("draft");
    const gpu = await renderer.renderSurfaceImage(r.view, r.clip, r.surfaceDepthRange, r.frame.rows, r.frame.cols);
    return { ...r, cpu: { depth: answer.arrays.depth, coords: answer.arrays.coords }, gpu };
  },
  probe: () => (renderer ? probeLayout(renderer.device) : Promise.resolve(["no WebGPU"])),
  adapter: () => (renderer ? renderer.device.adapterInfo : null),
  held: () => (renderer ? { ...renderer.holds(), bytes: renderer.heldBytes() } : null),
  settled: () => view.settled(),
};
document.body.dataset.ready = "true";
