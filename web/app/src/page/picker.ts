// The picker: the catalogs as pictures (Gallery.swift), each picture drawn
// by the engine that draws the real thing (Thumbnails.swift).
//
// The dropdown is the same list of functions and will always be faster for
// someone who knows what they want. This is for everyone else, and for the
// question a name cannot answer -- what does ψ look like next to ζ -- which is
// the whole reason the plates were drawn rather than tabulated.
//
// Each picture is the entry's scene built in the engine as Swift's
// Thumbnails builds it (kurven_session::thumbnails: a function sampled at
// 220, a surface on a coarser grid, the camera it opens at fitted to 440 by
// 300) and drawn by the page's preview renderer into a texture, or, without
// WebGPU, its judged strokes drawn on a 2D canvas. Shown at half size, so
// the curves are halved rather than aliased.
//
// The pictures are made by an engine of their own (a second Worker, started
// the first time the picker is opened: a page opened on a link never needs
// any of them), one at a time in catalog order, so a click on the document
// is never behind fourteen thumbnails' worth of sampling. They are cached in
// IndexedDB under the key the engine gives each (what would change the
// picture, hashed), so a second look costs nothing and a catalog that
// changes redraws itself rather than going stale.

import { browserWorker, EngineClient } from "../engine/client.ts";
import type { Reply, TypedArray } from "../engine/protocol.ts";
import { drawJudged, type Judged } from "../gpu/fallback.ts";
import type { Arrays, PreviewFrame } from "../gpu/frame.ts";
import { DEFAULT_OPTIONS, PreviewRenderer } from "../gpu/renderer.ts";
import { TextureUsage } from "../gpu/usage.ts";
import { h } from "./dom.ts";

/** A cell of the picker, as the `catalog` reply lists it. */
export interface Entry {
  id: string;
  kind: "function" | "surface";
  name: string;
  label: string;
  formula: string;
  notes: string;
  /** The key the picture is cached under. */
  thumbnail: string;
}

/** The size the engine draws a thumbnail at (thumbnails::SIZE). */
const WIDTH = 440;
const HEIGHT = 300;

/** The cache: IndexedDB, one store of PNGs by key. A browser without it, or
 * one that refuses (a private window), draws the pictures every time. */
class Cache {
  private db: Promise<IDBDatabase | null>;

  constructor() {
    this.db = new Promise((resolve) => {
      try {
        const open = indexedDB.open("kurven-thumbnails", 1);
        open.onupgradeneeded = () => open.result.createObjectStore("pictures");
        open.onsuccess = () => resolve(open.result);
        open.onerror = () => resolve(null);
        open.onblocked = () => resolve(null);
      } catch {
        resolve(null);
      }
    });
  }

  async get(key: string): Promise<Blob | null> {
    const db = await this.db;
    if (!db) return null;
    return new Promise((resolve) => {
      try {
        const get = db.transaction("pictures").objectStore("pictures").get(key);
        get.onsuccess = () => resolve(get.result instanceof Blob ? get.result : null);
        get.onerror = () => resolve(null);
      } catch {
        resolve(null);
      }
    });
  }

  async put(key: string, picture: Blob): Promise<void> {
    const db = await this.db;
    if (!db) return;
    await new Promise<void>((resolve) => {
      try {
        const t = db.transaction("pictures", "readwrite");
        t.objectStore("pictures").put(picture, key);
        t.oncomplete = () => resolve();
        t.onerror = () => resolve();
        t.onabort = () => resolve();
      } catch {
        resolve();
      }
    });
  }
}

/** The pictures: made by an engine of their own, one at a time, cached. */
export class Thumbnails {
  private readonly wasm: string;
  private readonly device: GPUDevice | null;
  private client: EngineClient | null = null;
  private renderer: PreviewRenderer | null = null;
  private readonly cache = new Cache();
  /** An object URL per entry's id, once drawn or read from the cache. */
  readonly pictures = new Map<string, string>();
  readonly failed = new Map<string, string>();
  /** What is being drawn now, for the header to say. */
  drawing: string | null = null;
  /** How many pictures were drawn by the engine, and how many read back. */
  readonly counts = { drawn: 0, cached: 0 };
  private running: Promise<void> | null = null;
  private readonly listeners: (() => void)[] = [];

  /** `device`: the page's WebGPU device, or null to draw judged strokes. */
  constructor(wasm: string, device: GPUDevice | null) {
    this.wasm = wasm;
    this.device = device;
  }

  /** The cache key's suffix: a picture drawn on the GPU and one of judged
   * strokes are different pictures of the same entry. */
  private get flavour(): string {
    return this.device ? "gpu" : "judged";
  }

  onChange(listener: () => void): void {
    this.listeners.push(listener);
  }

  private told(): void {
    for (const l of this.listeners) l();
  }

  /** Whether the pictures are wanted: the picker is open. A run of drawing
   * stops after the picture under way once they are not, and the engine
   * is let go of; wanted again before that, the run goes on. */
  private wanted = false;
  private entries: Entry[] = [];

  private done(): boolean {
    return this.entries.every((e) => this.pictures.has(e.id) || this.failed.has(e.id));
  }

  /** Draw whatever is missing, one at a time, in catalog order. Resolves
   * once every picture is drawn, cached or failed, or the pictures are no
   * longer wanted (`stop`). */
  async warm(entries: Entry[]): Promise<void> {
    this.wanted = true;
    this.entries = entries;
    // A run stopped and not yet over when the picker opens again has
    // returned or will: wait for it, and start another for what is left.
    for (;;) {
      this.running ??= this.warming().finally(() => {
        this.running = null;
        this.drawing = null;
        // Every picture drawn, cached or failed, or none wanted: the engine
        // that drew them is let go of, and started again if one is wanted
        // later.
        if (!this.wanted || this.done()) this.end();
        this.told();
      });
      await this.running;
      if (!this.wanted || this.done()) return;
    }
  }

  /** The picker closed: stop after the picture being drawn, and let the
   * engine go. The pictures drawn are kept; the rest wait for the next
   * opening. */
  async stop(): Promise<void> {
    this.wanted = false;
    if (this.running) await this.running.catch(() => {});
    if (!this.wanted && !this.running) this.end();
  }

  /** Terminate the thumbnails' Worker, if one is running. */
  private end(): void {
    this.client?.close();
    this.client = null;
  }

  /** Whether the thumbnails' Worker is running. */
  get engineRunning(): boolean {
    return this.client !== null;
  }

  private async warming(): Promise<void> {
    for (const entry of this.entries) {
      if (!this.wanted) return;
      if (this.pictures.has(entry.id) || this.failed.has(entry.id)) continue;
      const key = `${entry.thumbnail}.${this.flavour}`;
      const cached = await this.cache.get(key);
      if (cached) {
        this.pictures.set(entry.id, URL.createObjectURL(cached));
        this.counts.cached++;
        this.told();
        continue;
      }
      this.drawing = entry.name;
      this.told();
      try {
        const picture = await this.draw(entry);
        await this.cache.put(key, picture);
        this.pictures.set(entry.id, URL.createObjectURL(picture));
        this.counts.drawn++;
      } catch (error) {
        // A drawing that will not build is not worth a retry loop on every
        // opening of the picker; the cell says so.
        this.failed.set(entry.id, error instanceof Error ? error.message : String(error));
      }
      this.drawing = null;
      this.told();
    }
  }

  /** One entry's picture, as a PNG. */
  private async draw(entry: Entry): Promise<Blob> {
    this.client ??= new EngineClient(browserWorker, this.wasm);
    const answer = await this.client.request(
      { kind: "thumbnail", entry: entry.id, ...(this.device ? {} : { judged: true }) },
      { view: "thumbnail" },
    );
    if (answer.stale) throw new Error("the thumbnail went stale");
    const { reply, arrays } = answer;
    if (!reply.ok) throw new Error(reply.error ?? "refused");
    const canvas = new OffscreenCanvas(WIDTH, HEIGHT);
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("no 2D canvas");
    if (this.device) {
      const pixels = await this.render(reply as unknown as Reply & PreviewFrame, arrays as Arrays);
      ctx.putImageData(new ImageData(new Uint8ClampedArray(pixels.buffer as ArrayBuffer), WIDTH, HEIGHT), 0, 0);
    } else {
      drawJudged(ctx as unknown as CanvasRenderingContext2D, reply as unknown as Judged, arrays as Record<string, TypedArray>, DEFAULT_OPTIONS.strokeWidth);
    }
    return canvas.convertToBlob({ type: "image/png" });
  }

  /** A frame drawn into a texture of the thumbnail's size, read back as
   * RGBA: the window's renderer, a second instance on the same device, so
   * the window's held scene is not let go. */
  private async render(frame: PreviewFrame, arrays: Arrays): Promise<Uint8Array> {
    const device = this.device!;
    this.renderer ??= new PreviewRenderer(device, "rgba8unorm");
    const renderer = this.renderer;
    renderer.forget();
    renderer.take(frame, arrays);
    const target = device.createTexture({
      size: [WIDTH, HEIGHT],
      format: "rgba8unorm",
      usage: TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC,
    });
    try {
      renderer.render(frame, target, DEFAULT_OPTIONS);
      return await renderer.readPixels(target);
    } finally {
      target.destroy();
      renderer.forget();
    }
  }
}

/** The picker as a dialog: a filter, then the functions and the surfaces,
 * each cell its picture, label and formula; a click chooses. */
export class Picker {
  readonly dialog: HTMLDialogElement;
  private readonly grid: HTMLElement;
  private readonly filter: HTMLInputElement;
  private readonly progress: HTMLElement;
  private entries: Entry[] = [];
  private readonly thumbnails: Thumbnails;
  private readonly choose: (entry: Entry) => void;

  constructor(thumbnails: Thumbnails, choose: (entry: Entry) => void) {
    this.thumbnails = thumbnails;
    this.choose = choose;
    this.filter = h("input", { type: "search", placeholder: "Filter", "aria-label": "Filter the catalog", id: "picker-filter" });
    this.filter.addEventListener("input", () => this.layOut());
    this.progress = h("span", { class: "picker-progress", id: "picker-progress", "aria-live": "polite" });
    const cancel = h("button", { type: "button", id: "picker-cancel" }, "Cancel");
    cancel.addEventListener("click", () => this.dialog.close());
    this.grid = h("div", { class: "picker-grid", id: "picker-grid" });
    this.dialog = h(
      "dialog",
      { class: "picker", id: "picker", "aria-labelledby": "picker-title" },
      h(
        "header",
        {},
        h("h2", { id: "picker-title" }, "Choose a function or a surface"),
        this.progress,
        this.filter,
        cancel,
      ),
      this.grid,
    );
    // Closed, the pictures' engine is let go of once the one being drawn is.
    this.dialog.addEventListener("close", () => void this.thumbnails.stop());
    // A click on the backdrop dismisses, as a sheet's Cancel does.
    this.dialog.addEventListener("click", (e) => {
      if (e.target === this.dialog) this.dialog.close();
    });
    thumbnails.onChange(() => this.refresh());
  }

  setEntries(entries: Entry[]): void {
    this.entries = entries;
    this.layOut();
  }

  open(): void {
    if (!this.dialog.open) this.dialog.showModal();
    this.filter.focus();
    void this.thumbnails.warm(this.entries);
  }

  private shown(): Entry[] {
    const text = this.filter.value.trim().toLowerCase();
    if (!text) return this.entries;
    return this.entries.filter((e) =>
      [e.name, e.label, e.formula, e.notes].some((s) => s.toLowerCase().includes(text)),
    );
  }

  private layOut(): void {
    const shown = this.shown();
    const group = (title: string, kind: Entry["kind"]) => {
      const cells = shown.filter((e) => e.kind === kind).map((e) => this.cell(e));
      if (!cells.length) return [];
      return [h("h3", {}, title), ...cells];
    };
    this.grid.replaceChildren(...group("Functions", "function"), ...group("Surfaces", "surface"));
    this.refresh();
  }

  private cell(entry: Entry): HTMLElement {
    const button = h(
      "button",
      { type: "button", class: "cell", "data-entry": entry.id, title: entry.notes, "aria-label": `${entry.label}. ${entry.notes}` },
      h("span", { class: "picture" }),
      h("span", { class: "label" }, entry.label),
      h("code", { class: "formula" }, entry.formula),
    );
    button.addEventListener("click", () => {
      this.dialog.close();
      this.choose(entry);
    });
    return button;
  }

  /** The pictures and the header as the thumbnails stand. */
  private refresh(): void {
    const t = this.thumbnails;
    this.progress.textContent = t.drawing ? `drawing ${t.drawing}…` : "";
    for (const cell of this.grid.querySelectorAll<HTMLElement>(".cell")) {
      const id = cell.dataset.entry ?? "";
      const frame = cell.querySelector<HTMLElement>(".picture");
      if (!frame) continue;
      const url = t.pictures.get(id);
      const failed = t.failed.get(id);
      const state = url ? "drawn" : failed ? "failed" : "waiting";
      if (frame.dataset.state === state) continue;
      frame.dataset.state = state;
      if (url) frame.replaceChildren(h("img", { src: url, alt: "", width: WIDTH / 2, height: HEIGHT / 2 }));
      else if (failed) frame.replaceChildren(h("span", { class: "failed", title: failed }, "could not draw"));
      else frame.replaceChildren(h("span", { class: "spinner", "aria-hidden": "true" }));
    }
  }
}
