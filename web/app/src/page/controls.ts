// The panel: the controls the Swift app's sidebar has for a landscape
// (Inspector.swift, LandscapeSection.swift, PaintSection.swift's level
// slider), each sending an edit and showing the state the engine sends back.
// Nothing here is the model: a control's value is the latest state's, and
// a control being moved shows where the hand is until the engine answers.

import type { Controls, DocumentState, Domain, LayerState, Preset, Request, Truncation } from "../engine/protocol.ts";
import { h, shown } from "./dom.ts";
import { type Engine, quietly, SliderDrag } from "./engine.ts";
import { Formula } from "./formula.ts";
import { BandRows } from "./bands.ts";
import { controlsKey, LAYER_WORDS, LayerControls } from "./layers.ts";
import { PaintSection } from "./paint.ts";
import { SurfaceSection } from "./surface.ts";

/** What the panel asks of the page around it. */
export interface Host {
  engine: Engine;
  /** Something changed that wants the frame at rest. */
  changed(): void;
  /** Something moved: a draft now, the rest when it stops. */
  moved(): void;
  /** Export the judged strokes as SVG; cancel one under way. */
  exportSVG(): void;
  cancelExport(): void;
  /** Open the picker: the catalogs as pictures. */
  browse(): void;
}

type Interval = { lo: number; hi: number };

/** Where a landscape's truncation tops out: its height, or its highest band. */
function ceilingOf(t: Truncation | null): number | null {
  if (!t) return null;
  if (t.kind === "uniform") return t.height;
  if (t.kind === "realBands") {
    const heights = t.bands.map((b) => b.height);
    if (t.beyond !== undefined && Number.isFinite(t.beyond)) heights.push(t.beyond);
    return heights.length ? Math.max(...heights) : null;
  }
  return null;
}

/** A range input that is dragged as one of the engine's sliders (taken
 * hold of, moved, let go: one undo step) and stepped from the keyboard as
 * an edit of its own, as SwiftUI's Slider is. */
export function sliderInput(
  input: HTMLInputElement,
  drag: () => SliderDrag,
  move: (value: number, drag: SliderDrag) => void,
  keyed: (value: number) => void,
): { dragging: () => boolean } {
  let active: SliderDrag | null = null;
  let byKey = false;
  input.addEventListener("pointerdown", () => {
    // One drag of a slider at a time: a second pointer on it while the
    // first holds it is not a second drag (its begin would have no end).
    if (active) return;
    active = drag();
    active.begin();
    const up = () => {
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
      const d = active;
      active = null;
      if (d) void d.end().catch(quietly);
    };
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
  });
  input.addEventListener("keydown", (e) => {
    if (e.key.startsWith("Arrow") || e.key === "Home" || e.key === "End" || e.key.startsWith("Page")) byKey = true;
  });
  input.addEventListener("input", () => {
    const value = Number(input.value);
    if (active) move(value, active);
    else if (byKey) {
      byKey = false;
      keyed(value);
    }
  });
  return { dragging: () => active !== null };
}

/** One axis of the window as two thumbs that cannot cross, with a field for
 * each end (LandscapeSection.swift's `intervalControl`). The ends are kept a
 * fiftieth of the reach apart. */
class IntervalControl {
  readonly root: HTMLElement;
  private readonly lo: HTMLInputElement;
  private readonly hi: HTMLInputElement;
  private readonly loField: HTMLInputElement;
  private readonly hiField: HTMLInputElement;
  private readonly span: HTMLElement;
  private reach: Interval = { lo: -1, hi: 1 };
  private current: Interval = { lo: 0, hi: 1 };
  private readonly loState: { dragging: () => boolean };
  private readonly hiState: { dragging: () => boolean };

  constructor(host: Host, label: string, keys: [string, string]) {
    const id = label.toLowerCase();
    this.lo = h("input", { type: "range", step: "any", "aria-label": `${label}, low end`, id: `${id}-lo` });
    this.hi = h("input", { type: "range", step: "any", "aria-label": `${label}, high end`, id: `${id}-hi` });
    this.loField = h("input", { type: "number", step: "0.1", "aria-label": `${label}, low end` });
    this.hiField = h("input", { type: "number", step: "0.1", "aria-label": `${label}, high end` });
    this.span = h("div", { class: "span" });
    this.root = h(
      "div",
      { class: "row" },
      h("span", { class: "label" }, label),
      h(
        "div",
        { class: "interval", role: "group", "aria-label": `${label}, the window's ${label === "Re" ? "real" : "imaginary"} axis` },
        this.loField,
        h("div", { class: "track" }, this.span, this.lo, this.hi),
        this.hiField,
      ),
    );
    const gap = () => (this.reach.hi - this.reach.lo) / 50;
    const clampLo = (x: number) => Math.min(x, this.current.hi - gap());
    const clampHi = (x: number) => Math.max(x, this.current.lo + gap());
    const set = (key: string, value: number) => {
      host.engine.edit({ kind: "set", key, value }, () => host.changed());
    };
    const [loKey, hiKey] = keys;
    const dragOf = (key: string) => () => new SliderDrag(host.engine, { slider: "landscape", label: key }, () => host.moved());
    // The thumbs show the value sent, held off each other, by pointer or
    // key: they do not cross where the window cannot.
    this.loState = sliderInput(
      this.lo,
      dragOf(loKey),
      (x, d) => {
        const v = clampLo(x);
        this.lo.value = String(v);
        this.current.lo = v;
        this.paint();
        d.move(v);
      },
      (x) => {
        const v = clampLo(x);
        this.lo.value = String(v);
        set(loKey, v);
      },
    );
    this.hiState = sliderInput(
      this.hi,
      dragOf(hiKey),
      (x, d) => {
        const v = clampHi(x);
        this.hi.value = String(v);
        this.current.hi = v;
        this.paint();
        d.move(v);
      },
      (x) => {
        const v = clampHi(x);
        this.hi.value = String(v);
        set(hiKey, v);
      },
    );
    this.loField.addEventListener("change", () => {
      const x = Number(this.loField.value);
      if (Number.isFinite(x) && this.loField.value !== "") set(loKey, clampLo(x));
    });
    this.hiField.addEventListener("change", () => {
      const x = Number(this.hiField.value);
      if (Number.isFinite(x) && this.hiField.value !== "") set(hiKey, clampHi(x));
    });
  }

  private dragging(): boolean {
    return this.loState.dragging() || this.hiState.dragging();
  }

  update(current: Interval, reach: Interval): void {
    if (this.dragging()) return; // the reach is pinned, and the hand says where
    this.current = { ...current };
    this.reach = { lo: Math.min(reach.lo, current.lo), hi: Math.max(reach.hi, current.hi) };
    for (const r of [this.lo, this.hi]) {
      r.min = String(this.reach.lo);
      r.max = String(this.reach.hi);
    }
    this.lo.value = String(current.lo);
    this.hi.value = String(current.hi);
    if (document.activeElement !== this.loField) this.loField.value = shown(current.lo);
    if (document.activeElement !== this.hiField) this.hiField.value = shown(current.hi);
    this.paint();
  }

  private paint(): void {
    const w = this.reach.hi - this.reach.lo || 1;
    const a = (this.current.lo - this.reach.lo) / w;
    const b = (this.current.hi - this.reach.lo) / w;
    this.span.style.left = `${a * 100}%`;
    this.span.style.width = `${Math.max(0, b - a) * 100}%`;
  }
}

export class Panel {
  readonly root: HTMLElement;
  readonly formula: Formula;
  private readonly host: Host;
  private presets: Preset[] = [];
  /** How far the sliders reach: the engine's, from the catalog reply. */
  private reach: Controls | null = null;
  private readonly title: HTMLElement;
  private readonly fn: HTMLSelectElement;
  private readonly notes: HTMLElement;
  private readonly re: IntervalControl;
  private readonly im: IntervalControl;
  private readonly samples: HTMLInputElement;
  private readonly samplesOut: HTMLOutputElement;
  private readonly samplesState: { dragging: () => boolean };
  private readonly landscapeStatus: HTMLElement;
  private readonly truncationKinds: Record<Truncation["kind"], HTMLInputElement>;
  private readonly heightRow: HTMLElement;
  private readonly height: HTMLInputElement;
  private readonly heightField: HTMLInputElement;
  private readonly heightState: { dragging: () => boolean };
  private readonly bands: BandRows;
  private readonly truncationNote: HTMLElement;
  private readonly presetButtons: HTMLElement;
  private readonly modes: Record<DocumentState["mode"], HTMLInputElement>;
  private readonly layers: HTMLUListElement;
  private layerKey = "";
  private readonly layerControls = new Map<string, LayerControls>();
  private readonly exportButton: HTMLButtonElement;
  private readonly cancelButton: HTMLButtonElement;
  readonly exportStatus: HTMLElement;
  private readonly noticeLine: HTMLElement;
  private readonly landscapeSections: HTMLElement[];
  private readonly landscapeSection: HTMLElement;
  readonly surface: SurfaceSection;
  readonly paint: PaintSection;
  private readonly polyhedron: HTMLElement;
  private readonly polyhedronFacts: HTMLElement;
  private readonly undoButton: HTMLButtonElement;
  private readonly redoButton: HTMLButtonElement;
  private isExporting = false;

  constructor(host: Host) {
    this.host = host;
    const engine = host.engine;
    this.title = h("span", { class: "title", id: "title" });

    // MARK: the function
    this.fn = h("select", { id: "function", "aria-label": "Function: a catalog landscape" });
    this.fn.addEventListener("change", () => {
      const preset = this.fn.value;
      if (!preset) return;
      this.formula.editing = false;
      this.formula.clearError();
      engine.edit({ kind: "create", preset }, () => host.changed());
    });
    this.notes = h("p", { class: "notes" });
    this.formula = new Formula((text) => this.submit(text));
    this.re = new IntervalControl(host, "Re", ["rMin", "rMax"]);
    this.im = new IntervalControl(host, "Im", ["iMin", "iMax"]);
    this.samples = h("input", { type: "range", id: "samples", "aria-label": "Samples along the window's longer side" });
    this.samplesOut = h("output", { for: "samples" });
    this.samplesState = sliderInput(
      this.samples,
      () => new SliderDrag(engine, { slider: "landscape", label: "res" }, () => host.moved()),
      (x, d) => {
        this.samplesOut.value = String(x);
        d.move(x);
      },
      (x) => engine.edit({ kind: "set", key: "res", value: x }, () => host.changed()),
    );
    this.landscapeStatus = h("p", { class: "status", id: "landscape-status", "aria-live": "polite" });
    const browse = h("button", { type: "button", class: "browse", id: "browse", title: "The catalog as pictures" }, "Browse…");
    browse.addEventListener("click", () => host.browse());
    const landscape = h(
      "section",
      { "aria-labelledby": "h-landscape", id: "landscape-section" },
      h("h2", { id: "h-landscape" }, "Landscape"),
      h("div", { class: "row function" }, h("label", { class: "label", for: "function" }, "Function"), h("div", { class: "pick" }, this.fn, browse)),
      this.notes,
      this.formula.root,
      this.re.root,
      this.im.root,
      h(
        "div",
        { class: "row", title: "Samples along the longer side of the window. While a window thumb is dragged the landscape is sampled at the draft resolution, and at this when it is let go." },
        h("label", { class: "label", for: "samples" }, "Samples"),
        h("div", { class: "slider" }, this.samples, this.samplesOut),
      ),
      this.landscapeStatus,
    );

    // MARK: the truncation
    const kind = (value: Truncation["kind"], label: string) =>
      h("input", { type: "radio", name: "truncation", value, "aria-label": label });
    this.truncationKinds = {
      none: kind("none", "None"),
      uniform: kind("uniform", "At a height"),
      realBands: kind("realBands", "Per band of Re"),
    };
    const kinds = h(
      "fieldset",
      { class: "segmented", id: "truncation-kind" },
      h("legend", {}, "Truncate"),
      h("label", {}, this.truncationKinds.none, "None"),
      h("label", {}, this.truncationKinds.uniform, "At a height"),
      h("label", {}, this.truncationKinds.realBands, "Per band of Re"),
    );
    for (const input of Object.values(this.truncationKinds)) {
      input.addEventListener("change", () => this.truncate(input.value as Truncation["kind"]));
    }
    this.height = h("input", { type: "range", id: "truncation-height", step: "any", "aria-label": "Truncation height" });
    this.heightField = h("input", { type: "number", step: "0.1", min: 0, "aria-label": "Truncation height" });
    this.heightState = sliderInput(
      this.height,
      () => new SliderDrag(engine, { slider: "truncation" }, () => host.moved()),
      (x, d) => {
        this.heightField.value = shown(x, 3);
        d.move(x);
      },
      (x) => this.sendTruncation({ kind: "uniform", height: x }),
    );
    this.heightField.addEventListener("change", () => {
      const x = Number(this.heightField.value);
      if (this.heightField.value !== "" && Number.isFinite(x) && x > 0) this.sendTruncation({ kind: "uniform", height: x });
    });
    this.heightRow = h(
      "div",
      { class: "row" },
      h("label", { class: "label", for: "truncation-height" }, "Height"),
      h("div", { class: "slider" }, this.height, this.heightField),
    );
    this.bands = new BandRows(host, () => this.reach?.truncation.min ?? 0.1);
    this.truncationNote = h("p", { class: "notes" });
    const truncation = h(
      "section",
      { "aria-labelledby": "h-truncation" },
      h("h2", { id: "h-truncation" }, "Truncation"),
      kinds,
      this.heightRow,
      this.bands.root,
      this.truncationNote,
    );

    // MARK: the camera and the mode
    this.presetButtons = h("div", { class: "buttons", id: "presets" });
    const fit = h("button", { type: "button", id: "fit", title: "Fit the drawing to the canvas (f)" }, "Fit");
    fit.addEventListener("click", () => engine.edit({ kind: "fit" }, () => host.changed()));
    const mode = (value: DocumentState["mode"]) => h("input", { type: "radio", name: "mode", value });
    this.modes = { drawing: mode("drawing"), shaded: mode("shaded"), depth: mode("depth") };
    const modes = h(
      "fieldset",
      { class: "segmented", id: "mode" },
      h("legend", {}, "Preview"),
      h("label", { title: "1" }, this.modes.drawing, "Drawing"),
      h("label", { title: "2" }, this.modes.shaded, "Shaded"),
      h("label", { title: "3" }, this.modes.depth, "Depth"),
    );
    for (const input of Object.values(this.modes)) {
      input.addEventListener("change", () => this.setMode(input.value as DocumentState["mode"]));
    }
    const camera = h(
      "section",
      { "aria-labelledby": "h-camera" },
      h("h2", { id: "h-camera" }, "Camera"),
      h("div", { class: "buttons" }, this.presetButtons, fit),
      h("div", { class: "row" }, h("span", { class: "label" }, "Preview"), modes),
    );

    // MARK: the layers
    this.layers = h("ul", { class: "layers", id: "layers" });
    const layers = h("section", { "aria-labelledby": "h-layers" }, h("h2", { id: "h-layers" }, "Layers"), this.layers);

    // MARK: export
    this.exportButton = h(
      "button",
      { type: "button", class: "primary", id: "export", title: "The judged strokes, as the native bake writes them" },
      "Export SVG",
    );
    this.exportButton.addEventListener("click", () => host.exportSVG());
    this.cancelButton = h("button", { type: "button", id: "cancel-export", hidden: true }, "Cancel");
    this.cancelButton.addEventListener("click", () => host.cancelExport());
    this.exportStatus = h("p", { class: "status", id: "export-status", "aria-live": "polite" });
    const exporting = h(
      "section",
      { "aria-labelledby": "h-export" },
      h("h2", { id: "h-export" }, "Export"),
      h("div", { class: "export" }, this.exportButton, this.cancelButton),
      this.exportStatus,
      h(
        "p",
        { class: "keys" },
        "Drag to orbit, ",
        h("kbd", {}, "shift"),
        "-drag to pan, scroll to zoom. ",
        h("kbd", {}, "f"),
        " fits, ",
        h("kbd", {}, "1"),
        h("kbd", {}, "2"),
        h("kbd", {}, "3"),
        " switch the preview, ",
        h("kbd", {}, "⌘Z"),
        " undoes.",
      ),
    );

    this.noticeLine = h("p", { class: "notice", id: "notice", role: "alert", hidden: true });
    this.landscapeSections = [truncation];
    this.landscapeSection = landscape;
    this.surface = new SurfaceSection(host);
    this.paint = new PaintSection(host);
    // A polyhedron's solid is its bundle's and its visibility exact, so it
    // has no controls of its own but its layers' (Inspector.swift).
    this.polyhedronFacts = h("dl", { class: "facts", id: "polyhedron-facts" });
    this.polyhedron = h(
      "section",
      { "aria-labelledby": "h-polyhedron", id: "polyhedron-section" },
      h("h2", { id: "h-polyhedron" }, "Polyhedron"),
      this.polyhedronFacts,
      h("p", { class: "status", id: "polyhedron-status", "aria-live": "polite" }),
    );
    // The Edit menu's Undo and Redo, named for the step (⌘Z, ⇧⌘Z).
    this.undoButton = h("button", { type: "button", id: "undo", title: "⌘Z" }, "Undo");
    this.redoButton = h("button", { type: "button", id: "redo", title: "⇧⌘Z" }, "Redo");
    this.undoButton.addEventListener("click", () => this.request({ kind: "undo" }));
    this.redoButton.addEventListener("click", () => this.request({ kind: "redo" }));
    this.root = h(
      "aside",
      { class: "panel", "aria-label": "Controls" },
      h("header", {}, h("h1", {}, "kurven"), this.title),
      h("div", { class: "buttons edit", role: "group", "aria-label": "Edit" }, this.undoButton, this.redoButton),
      this.noticeLine,
      landscape,
      this.surface.root,
      this.polyhedron,
      truncation,
      camera,
      this.paint.root,
      layers,
      exporting,
    );
  }

  /** Say something about the page itself (a link refused), or nothing. */
  notice(text: string | null): void {
    this.noticeLine.textContent = text ?? "";
    this.noticeLine.hidden = !text;
  }

  /** The catalog's presets, and the reach of the sliders, which every
   * slider is built to: the engine holds a link to the same. */
  setCatalog(presets: Preset[], reach: Controls): void {
    this.presets = presets;
    this.reach = reach;
    this.samples.min = String(reach.samples.min);
    this.samples.max = String(reach.samples.max);
    this.samples.step = String(reach.samples.step);
    this.height.min = String(reach.truncation.min);
    this.layerKey = ""; // the level sliders are made again to it
    this.paint.setReach(reach);
  }

  private submit(text: string): void {
    const engine = this.host.engine;
    const now = engine.state?.landscape?.expression;
    if (text === "" && now === undefined) return;
    engine.edit({ kind: "expression", expression: text }, (reply) => {
      if (reply.ok) this.host.changed();
      // Typed on since return was pressed: what is in the field is newer
      // than this reply, which neither takes it over nor marks it.
      if (this.formula.input.value !== text) return;
      if (reply.ok) {
        this.formula.editing = false;
        this.formula.clearError();
        const now = engine.state?.landscape?.expression;
        if (typeof now === "string") this.formula.show(now);
      } else {
        const at = (reply.at as { message: string; position: number | null; length: number } | undefined) ?? null;
        this.formula.showError(text, at, at ? `${at.message}${at.position !== null ? ` (at character ${at.position + 1})` : ""}` : (reply.error ?? "refused"));
      }
    });
  }

  private sendTruncation(t: Truncation): void {
    this.host.engine.edit({ kind: "truncation", truncation: t }, () => this.host.changed());
  }

  /** TruncationSection.swift's `apply`: a kind chosen. */
  private truncate(kind: Truncation["kind"]): void {
    const current = ceilingOf(this.host.engine.state?.truncation ?? null) ?? 5;
    if (kind === "none") this.sendTruncation({ kind: "none" });
    else if (kind === "uniform") this.sendTruncation({ kind: "uniform", height: current });
    else {
      // Start from the truncation that is already there, as one band and a
      // beyond: the first edit should be a change to this landscape, not a
      // different one.
      const d = (this.host.engine.state?.landscape as { domain?: Domain } | null)?.domain;
      const middle = d ? (d.real.lo + d.real.hi) / 2 : 0;
      this.sendTruncation({ kind: "realBands", bands: [{ below: middle, height: current }], beyond: current });
    }
  }

  private setMode(mode: DocumentState["mode"]): void {
    this.host.engine.edit({ kind: "mode", mode }, () => this.host.changed());
  }

  /** Show what the engine says. */
  render(state: DocumentState): void {
    this.title.textContent = state.title;
    const isLandscape = state.kind === "landscape" && state.landscape !== null;
    for (const s of this.landscapeSections) s.hidden = !isLandscape;
    // A surface has its own section and a way back to the picker; a
    // polyhedron has no controls of its own but its layers' (Inspector.swift).
    this.landscapeSection.hidden = state.surface !== null || state.kind === "polyhedron";
    this.surface.render(state);
    this.paint.render(state);
    this.polyhedron.hidden = state.polyhedron === null;
    if (state.polyhedron) {
      const p = state.polyhedron;
      const facts: [string, string][] = [
        ["Solid", p.solid],
        ["Boundary", p.boundary],
        [p.shapeLabel, p.shape],
      ];
      const key = facts.flat().join("|");
      if (this.polyhedronFacts.dataset.key !== key) {
        this.polyhedronFacts.dataset.key = key;
        this.polyhedronFacts.replaceChildren(...facts.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)]));
      }
      const status = this.polyhedron.querySelector<HTMLElement>("#polyhedron-status");
      if (status) status.textContent = state.status.polyhedron ?? "";
    }
    this.undoButton.disabled = !state.undo.can;
    this.undoButton.textContent = state.undo.name ? `Undo ${state.undo.name}` : "Undo";
    this.redoButton.disabled = !state.redo.can;
    this.redoButton.textContent = state.redo.name ? `Redo ${state.redo.name}` : "Redo";
    const request = state.landscape as { name?: string; expression?: string; domain?: Domain; resolution?: number } | null;

    // The function.
    const name = request?.name ?? "";
    const options: HTMLOptionElement[] = [];
    if (!name) options.push(h("option", { value: "" }, request?.expression ? `${request.expression} (typed)` : "—"));
    for (const p of this.presets) options.push(h("option", { value: p.name }, p.label));
    const key = options.map((o) => `${o.value}:${o.textContent}`).join("|");
    if (this.fn.dataset.key !== key) {
      this.fn.replaceChildren(...options);
      this.fn.dataset.key = key;
    }
    this.fn.value = name;
    const preset = this.presets.find((p) => p.name === name);
    this.notes.textContent = preset?.notes ?? "";
    this.notes.hidden = !preset?.notes;
    if (request?.expression !== undefined) this.formula.show(request.expression);

    // The window: a preset's reach, or room around a typed one.
    if (request?.domain) {
      const d = request.domain;
      const window: Domain = preset?.window ?? {
        real: { lo: d.real.lo - (d.real.hi - d.real.lo), hi: d.real.hi + (d.real.hi - d.real.lo) },
        imag: { lo: d.imag.lo - (d.imag.hi - d.imag.lo), hi: d.imag.hi + (d.imag.hi - d.imag.lo) },
      };
      this.re.update(d.real, window.real);
      this.im.update(d.imag, window.imag);
    }
    if (request?.resolution !== undefined && !this.samplesState.dragging()) {
      this.samples.value = String(request.resolution);
      this.samplesOut.value = String(request.resolution);
    }
    this.landscapeStatus.textContent = state.status.landscape ?? (state.failure ? `failed: ${state.failure}` : "");

    // The truncation.
    const t = state.truncation;
    if (t) {
      this.truncationKinds[t.kind].checked = true;
      this.heightRow.hidden = t.kind !== "uniform";
      if (t.kind === "uniform" && !this.heightState.dragging()) {
        const ceiling = Math.max((ceilingOf(t) ?? 5) * 3, 10);
        this.height.max = String(ceiling);
        this.height.value = String(t.height);
        if (document.activeElement !== this.heightField) this.heightField.value = shown(t.height, 4);
      }
      this.bands.root.hidden = t.kind !== "realBands";
      if (t.kind === "realBands") this.bands.update(t, Math.max((ceilingOf(t) ?? 5) * 3, 10));
      this.truncationNote.textContent =
        t.kind === "none"
          ? "The surface is drawn at |f|, however far up that goes. A function with a pole wants a truncation; one without does not."
          : "";
      this.truncationNote.hidden = this.truncationNote.textContent === "";
    }

    // The camera's presets.
    const presetKey = state.presets.join("|");
    if (this.presetButtons.dataset.key !== presetKey) {
      this.presetButtons.dataset.key = presetKey;
      this.presetButtons.replaceChildren(
        ...state.presets.map((p, index) => {
          const b = h("button", { type: "button", "data-preset": index }, p);
          b.addEventListener("click", () => this.host.engine.edit({ kind: "preset", index }, () => this.host.changed()));
          return b;
        }),
      );
    }
    this.modes[state.mode].checked = true;

    this.renderLayers(state);
    this.exportButton.disabled = this.isExporting || !state.canBake || !(state.kind === "landscape" || state.kind === "surface" || state.kind === "polyhedron");
  }

  private renderLayers(state: DocumentState): void {
    const key = state.layers.map(controlsKey).join("|") + `|${this.reach !== null}`;
    if (key !== this.layerKey) {
      this.layerKey = key;
      this.layerControls.clear();
      this.layers.replaceChildren(...state.layers.map((l, i) => this.layerRow(l, i)));
    }
    state.layers.forEach((l) => {
      const li = this.layers.querySelector<HTMLLIElement>(`li[data-layer="${CSS.escape(l.name)}"]`);
      if (!li) return;
      const box = li.querySelector<HTMLInputElement>("input[type=checkbox]");
      if (box) box.checked = !l.hidden;
      const count = li.querySelector<HTMLElement>(".count");
      if (count) count.textContent = l.paths !== undefined ? String(l.paths) : "";
      const unclipped = li.querySelector<HTMLElement>(".unclipped");
      if (unclipped) unclipped.hidden = l.clipped !== false;
      this.layerControls.get(l.name)?.update(l);
    });
  }

  /** A layer of the Layers list (Inspector.swift's `layerList`): shown or
   * not, its name in the glossary's words with its manifest name, how many
   * paths it draws, "unclipped" when it is not tested for visibility, and
   * its own controls. */
  private layerRow(layer: LayerState, index: number): HTMLLIElement {
    const engine = this.host.engine;
    const id = `layer-${index}`;
    const box = h("input", { type: "checkbox", id });
    box.addEventListener("change", () => engine.edit({ kind: "hide", layer: index }, () => this.host.changed()));
    const controls = new LayerControls(this.host, this.reach, layer);
    this.layerControls.set(layer.name, controls);
    return h(
      "li",
      { "data-layer": layer.name },
      h(
        "label",
        { class: "name", for: id },
        box,
        LAYER_WORDS[layer.name] ?? layer.name,
        h("span", { class: "unclipped", hidden: true }, "unclipped"),
        h("small", {}, layer.name),
        h("span", { class: "count", title: "paths drawn" }),
      ),
      controls.root,
    );
  }

  /** The export button while the judged strokes are being made. */
  exporting(on: boolean): void {
    this.isExporting = on;
    this.exportButton.disabled = on;
    this.exportButton.textContent = on ? "Exporting…" : "Export SVG";
    this.cancelButton.hidden = !on;
  }

  /** Keys for the whole page: f, 1, 2, 3, undo and redo. */
  request(r: Request): void {
    this.host.engine.edit(r, () => this.host.changed());
  }
}
