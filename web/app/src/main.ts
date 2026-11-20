// The page: a thin view of the engine. All state lives in the engine
// (crates/kurven-session, run in a Worker through crates/kurven-wasm); the
// page renders state and sends edits (HANDOFF-web-port.md, Architecture).
//
// Phase 8, steps 1 to 3: the formula with its errors at their place, the
// catalog, the canvas (orbit, pan, zoom toward the cursor, fit), the window
// as two thumbs per axis, the resolution, the truncation height, the layers
// and their level counts, the camera's presets, the three modes, Export SVG
// (the judged strokes, the native bake's bytes), and the whole state in the
// URL's fragment (page/link.ts).
//
// The engine is one Worker: a frame at rest (cn's folds and walls, about
// 2.2 s) holds it, and whatever the page asks meanwhile waits behind it.
// The page says so (the indicator over the canvas) and never waits itself:
// controls and gestures are coalesced, so a hand that keeps moving costs one
// request behind, never a queue (crates/FINDINGS.md, Phase 8).
//
// The query, for the browser lane and for looking:
//
//   ?landscape=cn        a catalog landscape (`create`), where no link is
//   ?surface=sn          a catalog surface (`createSurface`)
//   &gpu=0               the page without WebGPU: judged strokes at rest
//   &size=900x700        the canvas in device pixels, rather than the stage
//   &slope=0             the preview's slope scale (default 1)
//   &offscreen=1         draw into a texture, never presented (gpu/view.ts)

import "./page/style.css";

import { browserWorker, Cancelled, EngineClient, VIEWS } from "./engine/client.ts";
import type { Bundle, Controls, DocumentState, Edited, Preset, Request } from "./engine/protocol.ts";
import { DEFAULT_OPTIONS, PreviewRenderer, Unsupported } from "./gpu/renderer.ts";
import { probeLayout } from "./gpu/layout.ts";
import { PreviewView, type Timing } from "./gpu/view.ts";
import { CanvasInput } from "./page/canvas.ts";
import { Panel } from "./page/controls.ts";
import { h } from "./page/dom.ts";
import { Engine, Latest, quietly } from "./page/engine.ts";
import { decode, encode, hides, linkOf, restoring } from "./page/link.ts";
import { type Entry, Picker, Thumbnails } from "./page/picker.ts";

const query = new URLSearchParams(location.search);
const app = document.querySelector<HTMLElement>("#app");
if (!app) throw new Error("index.html has no #app");

const scale = window.devicePixelRatio || 1;
const size = query.get("size")?.match(/^(\d+)x(\d+)$/);
const fixed = size ? { width: Number(size[1]), height: Number(size[2]) } : null;

const canvas = h("canvas", { id: "preview", "aria-label": "The drawing. Drag to orbit; arrows orbit from the keyboard." });
const indicatorText = h("span", { id: "indicator-text" });
const indicator = h("div", { class: "indicator", id: "indicator", role: "status", "aria-live": "polite" }, h("span", { class: "dot" }), indicatorText);
const stage = h("main", { class: fixed ? "stage fixed" : "stage" }, canvas, indicator);

const wasmURL = `${import.meta.env.BASE_URL}kurven_wasm.wasm`;
const client = new EngineClient(browserWorker, wasmURL);
const engine = new Engine(client);

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
const view = new PreviewView(
  client,
  canvas,
  renderer,
  {
    ...DEFAULT_OPTIONS,
    // The widest stroke, in device pixels: as wide on a dense screen as on
    // any other.
    strokeWidth: DEFAULT_OPTIONS.strokeWidth * scale,
    slopeScale: slope !== null ? Number(slope) : DEFAULT_OPTIONS.slopeScale,
  },
  query.get("offscreen") === "1",
);

// The picker's pictures: an engine of their own, started when the picker
// is first opened, drawn on the window's GPU device (page/picker.ts).
const thumbnails = new Thumbnails(wasmURL, renderer?.device ?? null);
const picker = new Picker(thumbnails, (entry: Entry) => {
  panel.formula.editing = false;
  panel.formula.clearError();
  const request: Request =
    entry.kind === "surface" ? { kind: "createSurface", preset: entry.name } : { kind: "create", preset: entry.name };
  engine.edit(request, () => view.changed());
});

const panel = new Panel({
  engine,
  changed: () => view.changed(),
  moved: () => view.moved(),
  exportSVG: () => void exportSVG().catch(quietly),
  cancelExport: () => void client.cancel("the export was cancelled").catch(quietly),
  browse: () => picker.open(),
});
app.replaceChildren(stage, panel.root, picker.dialog);
if (!view.usesGPU) {
  // Said where the drawing is: no drafts here, and each frame at rest is the
  // bake (the browser lane reads it as #status, Phase 7's name).
  stage.append(h("p", { class: "hint", id: "status" }, `judged strokes only: ${fallbackReason ?? "no WebGPU"}`));
}

// MARK: - the canvas's size

/** The canvas in device pixels: the stage's inside, whole pixels down. */
function measure(): { width: number; height: number } {
  if (fixed) return fixed;
  return {
    width: Math.max(1, Math.floor(stage.clientWidth * scale)),
    height: Math.max(1, Math.floor(stage.clientHeight * scale)),
  };
}

/** Size the canvas to the stage. Its CSS size is its device size over the
 * device's scale exactly, so a pointer's pixel is the engine's pixel. */
function resize(): boolean {
  const next = measure();
  if (canvas.width === next.width && canvas.height === next.height) return false;
  canvas.width = next.width;
  canvas.height = next.height;
  canvas.style.width = `${next.width / scale}px`;
  canvas.style.height = `${next.height / scale}px`;
  return true;
}
resize();

const viewports = new Latest<[number, number]>(async ([width, height]) => {
  await engine.send({ kind: "viewport", width, height });
  view.changed();
});
if (!fixed) {
  new ResizeObserver(() => {
    if (resize()) viewports.push([canvas.width, canvas.height]);
  }).observe(stage);
}

// MARK: - what the engine is doing

let exporting: { started: number } | null = null;
let lastFrame: Timing | null = null;
let restStarted = 0;

function seconds(ms: number): string {
  return ms < 1000 ? `${ms.toFixed(0)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

/** The indicator over the canvas: what holds the engine, and for how long. */
function indicate(): void {
  const now = performance.now();
  let text: string;
  let busy = true;
  if (exporting) text = `exporting: baking the judged strokes… ${seconds(now - exporting.started)}`;
  else if (engine.busy.length) text = `${engine.busy[engine.busy.length - 1]}…`;
  else if (view.working === "rest") {
    text = view.usesGPU
      ? `drawing at rest: testing the folds and walls… ${seconds(now - restStarted)}`
      : `drawing the judged strokes… ${seconds(now - restStarted)}`;
  } else if (view.working === "draft") text = "draft";
  else {
    busy = false;
    text = lastFrame ? `${lastFrame.kind === "draft" ? "draft" : "at rest"} · ${seconds(lastFrame.engineMs)}` : "";
  }
  indicatorText.textContent = text;
  indicator.dataset.busy = String(busy);
  document.body.dataset.busy = String(busy);
}
engine.onBusy(indicate);
view.onWork((work) => {
  if (work === "rest") restStarted = performance.now();
  indicate();
});
view.onFrame((t) => {
  lastFrame = t;
  indicate();
});
setInterval(() => {
  if (indicator.dataset.busy === "true") indicate();
}, 200);

// MARK: - the state, shown, and written to the URL

let linkTimer: ReturnType<typeof setTimeout> | null = null;
let linking: Promise<void> = Promise.resolve();
let restoringLink = false;
/** The fragment the page last wrote or opened: one that differs is a link
 * pasted in, which `hashchange` opens and nothing overwrites first. */
let lastLinked = "";

function scheduleLink(): void {
  if (linkTimer) clearTimeout(linkTimer);
  linkTimer = setTimeout(() => {
    linkTimer = null;
    linking = linking.then(writeLink, writeLink);
  }, 400);
}

async function writeLink(): Promise<void> {
  try {
    await writingLink();
  } catch (error) {
    quietly(error);
  }
}

async function writingLink(): Promise<void> {
  const state = engine.state;
  if (!state || restoringLink) return;
  const reply = await engine.send({ kind: "edited" });
  const edited = (reply?.edited as Edited | null | undefined) ?? null;
  const link = linkOf(engine.state ?? state, edited);
  if (!link) return;
  if (location.hash !== lastLinked && location.hash.startsWith("#k1")) return; // a new link, not yet opened
  const hash = await encode(link);
  if (hash !== location.hash) history.replaceState(null, "", `${location.pathname}${location.search}${hash}`);
  lastLinked = hash;
}

engine.onState((state: DocumentState, request: Request) => {
  panel.render(state);
  if (!VIEWS.has(request.kind)) scheduleLink();
});

// MARK: - Export SVG: the judged strokes, never the GPU's draft

let lastExport: Uint8Array | null = null;

async function exportSVG(): Promise<void> {
  if (exporting) return;
  exporting = { started: performance.now() };
  panel.exporting(true);
  panel.exportStatus.textContent = "";
  indicate();
  try {
    const answer = await engine.call({ kind: "bake", svg: true });
    if (!answer) return;
    const { reply, arrays } = answer;
    if (!reply.ok) {
      panel.exportStatus.textContent = reply.error ?? "the bake was refused";
      return;
    }
    const svg = arrays.svg;
    if (!(svg instanceof Uint8Array)) throw new Error("the bake sent no SVG");
    lastExport = svg;
    const name = `${(engine.state?.title ?? "kurven").replace(/[\\/:*?"<>|\s]+/g, "_")}.svg`;
    const url = URL.createObjectURL(new Blob([svg as BlobPart], { type: "image/svg+xml" }));
    const a = h("a", { href: url, download: name });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
    const took = seconds(performance.now() - exporting.started);
    panel.exportStatus.textContent = `${engine.state?.status.bake ?? ""} → ${name}, ${svg.length.toLocaleString()} bytes, in ${took}`;
  } catch (error) {
    if (error instanceof Cancelled) {
      panel.exportStatus.textContent = "cancelled: the engine was restarted and the document replayed";
      view.changed();
    } else {
      panel.exportStatus.textContent = `export failed: ${error instanceof Error ? error.message : String(error)}`;
    }
  } finally {
    exporting = null;
    panel.exporting(false);
    indicate();
  }
}

// MARK: - input

const input = new CanvasInput(engine, canvas, scale, () => view.moved());

/** Inputs that take typed text, whose keys are theirs: a slider, a check
 * box or a radio button takes no letters, digits or ⌘Z of its own. */
const TYPED = new Set(["text", "number", "search", "email", "url", "tel", "password"]);

function takesText(el: Element | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  if (el.isContentEditable || el instanceof HTMLTextAreaElement) return true;
  // A select answers letters by choosing the option they start.
  if (el instanceof HTMLSelectElement) return true;
  return el instanceof HTMLInputElement && TYPED.has(el.type);
}

window.addEventListener("keydown", (e) => {
  if (takesText(e.target as Element | null)) return;
  const mod = e.metaKey || e.ctrlKey;
  if (mod && (e.key === "z" || e.key === "Z")) {
    e.preventDefault();
    panel.request({ kind: e.shiftKey ? "redo" : "undo" });
    return;
  }
  if (mod && e.key === "y") {
    e.preventDefault();
    panel.request({ kind: "redo" });
    return;
  }
  if (mod || e.altKey) return;
  const keyed: Record<string, Request> = {
    f: { kind: "fit" },
    "1": { kind: "mode", mode: "drawing" },
    "2": { kind: "mode", mode: "shaded" },
    "3": { kind: "mode", mode: "depth" },
  };
  const r = keyed[e.key];
  if (r) {
    e.preventDefault();
    panel.request(r);
  }
});

// MARK: - opening: a link, the query, or Γ

const catalog = await engine.send({ kind: "catalog" });
if (!catalog?.ok || !catalog.controls) throw new Error(`the engine gave no catalog: ${catalog?.error ?? "no reply"}`);
panel.setCatalog((catalog.presets as Preset[] | undefined) ?? [], catalog.controls as Controls);
picker.setEntries((catalog.entries as Entry[] | undefined) ?? []);

/** Open the link a fragment holds: "none" when it holds none, "refused"
 * when it cannot be read or the engine refuses what it asks for (the page
 * then says why, and the document is the one before), "opened". */
async function openLink(hash: string): Promise<"none" | "refused" | "opened"> {
  const refused = (why: string): "refused" => {
    panel.notice(`This link was refused: ${why}`);
    return "refused";
  };
  let link;
  try {
    link = await decode(hash);
  } catch (error) {
    return refused(error instanceof Error ? error.message : String(error));
  }
  if (!link) return "none";
  if (linkTimer) {
    clearTimeout(linkTimer);
    linkTimer = null;
  }
  restoringLink = true;
  const wasRefine = engine.state?.refine ?? true;
  const wasMode = engine.state?.mode ?? "drawing";
  // What was sent before the landscape, put back if anything is refused:
  // the engine's `restore` changes nothing it refuses, so neither does the
  // page.
  const putBack = async () => {
    if (link.refine !== wasRefine) await engine.send({ kind: "refine", on: wasRefine });
    if (link.mode !== wasMode) await engine.send({ kind: "mode", mode: wasMode });
  };
  try {
    const { before, after } = restoring(link, [canvas.width, canvas.height]);
    // A setting the page already has is not sent again (a refinement
    // switched on redraws every layer).
    const sending = before.filter(
      (r) => !(r.kind === "refine" && r.on === wasRefine) && !(r.kind === "mode" && r.mode === wasMode),
    );
    // Every reply read: any refusal refuses the link.
    for (const r of sending) {
      const reply = await engine.send(r);
      if (!reply?.ok) {
        await putBack();
        return refused(reply?.error ?? `the engine did not answer its ${r.kind}`);
      }
    }
    const restored = engine.state;
    const rest = [...(restored ? hides(link, restored) : []), ...after];
    const partly: string[] = [];
    for (const r of rest) {
      const reply = await engine.send(r);
      if (!reply?.ok) partly.push(reply?.error ?? `no answer to its ${r.kind}`);
    }
    panel.notice(partly.length ? `This link was opened, but not all of it: ${partly.join("; ")}` : null);
    lastLinked = hash;
    return "opened";
  } catch (error) {
    // A cancel or a failure mid-way: said, and the settings put back as far
    // as the engine will take them.
    await putBack().catch(quietly);
    return refused(error instanceof Error ? error.message : String(error));
  } finally {
    restoringLink = false;
  }
}

// A link pasted into this tab: opened, not overwritten. One refused leaves
// the document as it was, and the URL is put back to it.
window.addEventListener("hashchange", () => {
  const hash = location.hash;
  if (hash === lastLinked) return;
  void openLink(hash)
    .then((opened) => {
      if (opened === "opened") view.changed();
      else if (opened === "refused") {
        lastLinked = hash;
        scheduleLink();
      }
    })
    .catch(quietly);
});

await engine.send({ kind: "viewport", width: canvas.width, height: canvas.height });
const first = await openLink(location.hash);
// A link refused is overwritten by the document opened in its place.
if (first === "refused") lastLinked = location.hash;
if (first !== "opened") {
  const landscape = query.get("landscape");
  const surface = query.get("surface");
  if (surface) await engine.send({ kind: "createSurface", preset: surface });
  else if (landscape !== "") await engine.send({ kind: "create", preset: landscape ?? "gamma" });
}
if (engine.state) panel.render(engine.state);
view.changed();
indicate();

// MARK: - the browser lane's handle (web/app/test/browser)

declare global {
  interface Window {
    kurven: unknown;
  }
}

window.kurven = {
  view,
  engine,
  client,
  input,
  usesGPU: view.usesGPU,
  fallbackReason,
  state: () => engine.state,
  send: async (r: Request) => {
    await engine.send(r);
    return engine.state;
  },
  /** A request, its reply and arrays as they come. */
  request: (r: Request) => client.request(r),
  /** Open a bundle given as its files. */
  open: async (bundle: Bundle) => {
    await client.stage(bundle);
    await engine.send({ kind: "open", name: bundle.name });
    view.changed();
    return engine.state;
  },
  /** Draw a frame now, and read the pixels it left. */
  draw: (wish: "draft" | "rest", read = false) => view.drawNow(wish, read),
  /** The last frame drawn again offscreen, read back. */
  redrawOffscreen: () => view.redrawOffscreen(),
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
  /** Every frame wished drawn, every gesture and control sent. */
  settled: async () => {
    for (;;) {
      await input.settled();
      await viewports.flush();
      await engine.settled();
      await view.settled();
      // A frame or a reply can have led to another: settled when none did.
      if (!engine.busy.length && view.working === null) {
        await engine.settled();
        if (view.working === null) return;
      }
    }
  },
  /** The URL once every change so far is written into it. */
  linked: async () => {
    await input.settled();
    if (linkTimer) {
      clearTimeout(linkTimer);
      linkTimer = null;
      linking = linking.then(writeLink, writeLink);
    }
    await linking;
    return location.href;
  },
  /** The requests a native session replays to hold this document. */
  journal: () => client.journaled,
  /** The picker: open it, and the pictures as they stand. */
  picker: {
    open: () => picker.open(),
    warmed: () => thumbnails.warm((catalog.entries as Entry[] | undefined) ?? []),
    counts: () => ({ ...thumbnails.counts, pictures: thumbnails.pictures.size, failed: [...thumbnails.failed] }),
  },
  /** The last SVG exported, as text. */
  lastExport: () => (lastExport ? new TextDecoder().decode(lastExport) : null),
};
document.body.dataset.ready = "true";
