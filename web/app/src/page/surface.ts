// The Surface section: the numbers a surface is made of, as controls
// (SurfaceSection.swift). Laid out as Swift lays it out -- label, slider,
// field (whose arrows are the stepper) -- and like it, one undo step per
// drag, named for the control. What each control shows and how far it
// reaches is the engine's (the state's `surfaceControls`,
// kurven_session::surface_controls): a slider's move is sent as it is and
// the engine rests it at its step or a detent within reach; a value typed
// or stepped is clamped to the reach by the engine, as Swift's binding
// clamps it.

import type { DocumentState, SurfaceControls, SurfaceRow } from "../engine/protocol.ts";
import { type Host, sliderInput } from "./controls.ts";
import { h } from "./dom.ts";
import { SliderDrag } from "./engine.ts";

/** A number as a field shows it: to the control's digits. */
function fixed(x: number, digits: number): string {
  return digits === 0 ? String(Math.round(x)) : String(Number(x.toFixed(digits)));
}

/** One row of a number: a slider (when it slides) and a field. */
class NumberRow {
  readonly root: HTMLElement;
  readonly field: string;
  private readonly range: HTMLInputElement | null;
  private readonly input: HTMLInputElement;
  private readonly state: { dragging: () => boolean } | null;
  private digits = 3;

  constructor(host: Host, row: SurfaceRow) {
    this.field = row.field;
    const id = `surface-${row.field}`;
    this.input = h("input", { type: "number", id: `${id}-field`, "aria-label": row.label, "data-field": row.field });
    this.input.addEventListener("change", () => {
      const x = Number(this.input.value);
      if (this.input.value === "" || !Number.isFinite(x)) return;
      host.engine.edit({ kind: "surfaceField", field: this.field, value: x }, () => host.changed());
    });
    if (row.slides) {
      this.range = h("input", { type: "range", id, "aria-label": row.label, "data-field": row.field });
      this.state = sliderInput(
        this.range,
        () =>
          new SliderDrag(
            host.engine,
            { slider: "surfaceField", field: this.field, label: row.label },
            () => host.moved(),
            true,
          ),
        (x, d) => d.move(x),
        (x) => host.engine.edit({ kind: "surfaceField", field: this.field, value: x }, () => host.changed()),
      );
    } else {
      this.range = null;
      this.state = null;
    }
    this.root = h(
      "div",
      { class: "row", title: row.help, "data-row": row.field },
      h("label", { class: "label", for: row.slides ? id : `${id}-field` }, row.label),
      row.slides ? h("div", { class: "slider" }, this.range, this.input) : h("div", { class: "slider" }, h("span", {}), this.input),
    );
    this.update(row);
  }

  update(row: SurfaceRow): void {
    this.digits = row.digits;
    for (const el of [this.range, this.input]) {
      if (!el) continue;
      el.min = String(row.min);
      el.max = String(row.max);
      el.step = String(row.step);
    }
    if (row.value === null) return;
    if (this.range && !this.state?.dragging()) this.range.value = String(row.value);
    if (document.activeElement !== this.input) this.input.value = fixed(row.value, this.digits);
  }
}

/** The forced torus's state component, drawn as the ring's radius or
 * height: x, x′ or x″, as a segmented choice. */
function componentRow(host: Host, field: "radial" | "axial", label: string): { root: HTMLElement; set: (v: number) => void } {
  const inputs = ["x", "x′", "x″"].map((name, value) => {
    const input = h("input", { type: "radio", name: `component-${field}`, value, "aria-label": `${label}: ${name}` });
    input.addEventListener("change", () => host.engine.edit({ kind: "component", field, value }, () => host.changed()));
    return { input, label: h("label", {}, input, name) };
  });
  const root = h(
    "div",
    {
      class: "row",
      title: `The state component drawn as the ring's ${field === "radial" ? "radius" : "height"}. Drawn by revolution, each slice of the torus gets a half-plane to itself.`,
    },
    h("span", { class: "label" }, label),
    h("fieldset", { class: "segmented", id: `component-${field}` }, h("legend", {}, label), ...inputs.map((i) => i.label)),
  );
  return {
    root,
    set: (v) => {
      const i = inputs[v];
      if (i) i.input.checked = true;
    },
  };
}

export class SurfaceSection {
  readonly root: HTMLElement;
  private readonly host: Host;
  private readonly notes: HTMLElement;
  private readonly rows: HTMLElement;
  private readonly radial: ReturnType<typeof componentRow>;
  private readonly axial: ReturnType<typeof componentRow>;
  private readonly components: HTMLElement;
  private readonly conformalRow: HTMLElement;
  private readonly conformal: HTMLInputElement;
  private readonly linesSwitchRow: HTMLElement;
  private readonly linesSwitch: HTMLInputElement;
  private readonly linesRow: HTMLElement;
  private readonly around: HTMLInputElement;
  private readonly across: HTMLInputElement;
  private readonly linear: HTMLInputElement;
  private readonly linearRows: HTMLElement;
  private readonly linearNote: HTMLElement;
  private readonly status: HTMLElement;
  private numberRows = new Map<string, NumberRow>();
  private rowsKey = "";
  private linearKey = "";
  private linearNumberRows = new Map<string, NumberRow>();

  constructor(host: Host) {
    this.host = host;
    const engine = host.engine;
    this.notes = h("p", { class: "notes", id: "surface-notes" });
    this.rows = h("div", { id: "surface-rows" });
    this.radial = componentRow(host, "radial", "Radius from");
    this.axial = componentRow(host, "axial", "Height from");
    this.components = h("div", {}, this.radial.root, this.axial.root);

    this.conformal = h("input", { type: "checkbox", id: "surface-conformal" });
    this.conformal.addEventListener("change", () =>
      engine.edit({ kind: "surfaceField", field: "conformal", value: this.conformal.checked ? 1 : 0 }, () => host.changed()),
    );
    this.conformalRow = h(
      "label",
      {
        class: "check",
        title:
          "Isothermal coordinates round the tube, on the torus whose minor radius the modulus pins: every |sn| contour then crosses every arg sn contour at a right angle on the surface.",
      },
      this.conformal,
      "Conformal",
    );

    this.linesSwitch = h("input", { type: "checkbox", id: "surface-lines" });
    this.linesSwitch.addEventListener("change", () =>
      engine.edit({ kind: "surfaceLines", on: this.linesSwitch.checked }, () => host.changed()),
    );
    this.linesSwitchRow = h(
      "label",
      { class: "check", title: "Lines of constant forcing phase and constant internal phase: the invariant circles and the curves across them." },
      this.linesSwitch,
      "Coordinate curves",
    );
    const count = (axis: "around" | "across") => {
      const input = h("input", { type: "number", id: `lines-${axis}`, "aria-label": `Lines ${axis}` });
      input.addEventListener("change", () => {
        const n = Number(input.value);
        if (input.value === "" || !Number.isFinite(n)) return;
        engine.edit({ kind: "surfaceLineCount", axis, count: Math.round(n) }, () => host.changed());
      });
      return input;
    };
    this.around = count("around");
    this.across = count("across");
    this.linesRow = h(
      "div",
      { class: "row" },
      h("span", { class: "label" }, "Lines"),
      h("div", { class: "pair" }, this.around, h("span", { class: "times" }, "×"), this.across),
    );

    this.linear = h("input", { type: "checkbox", id: "surface-linear" });
    this.linear.addEventListener("change", () => engine.edit({ kind: "linearCurve", on: this.linear.checked }, () => host.changed()));
    this.linearRows = h("div", { id: "linear-rows" });
    this.linearNote = h("p", { class: "notes", id: "linear-description" });
    const browse = h("button", { type: "button", class: "browse", id: "browse-surface" }, "Browse…");
    browse.addEventListener("click", () => host.browse());
    this.status = h("p", { class: "status", id: "surface-status", "aria-live": "polite" });
    this.root = h(
      "section",
      { "aria-labelledby": "h-surface", id: "surface-section" },
      h("h2", { id: "h-surface" }, "Surface"),
      this.notes,
      this.rows,
      this.components,
      this.conformalRow,
      this.linesSwitchRow,
      this.linesRow,
      h(
        "label",
        {
          class: "check",
          title:
            "A straight line through the parameter rectangle, so many turns through the tube per turn round the ring, drawn on the torus. A fraction closes; anything else winds on. On the forced torus the trajectory is the linear curve at the system's own Ω/ω.",
        },
        this.linear,
        "Linear curve",
      ),
      this.linearRows,
      this.linearNote,
      h("div", { class: "buttons" }, browse),
      this.status,
    );
  }

  /** Show what the engine says. */
  render(state: DocumentState): void {
    const c: SurfaceControls | null = state.surfaceControls;
    this.root.hidden = c === null;
    if (!c) return;
    this.notes.textContent = c.notes ?? "";
    this.notes.hidden = !c.notes;
    const key = c.rows.map((r) => `${r.field}:${r.slides}`).join("|");
    if (key !== this.rowsKey) {
      this.rowsKey = key;
      this.numberRows = new Map(c.rows.map((r) => [r.field, new NumberRow(this.host, r)]));
      this.rows.replaceChildren(...[...this.numberRows.values()].map((r) => r.root));
    }
    for (const r of c.rows) this.numberRows.get(r.field)?.update(r);

    this.components.hidden = c.components === null;
    if (c.components) {
      this.radial.set(c.components.radial);
      this.axial.set(c.components.axial);
    }
    this.conformalRow.hidden = c.conformal === null;
    this.conformal.checked = c.conformal === true;
    this.linesSwitchRow.hidden = !c.linesSwitch;
    this.linesSwitch.checked = c.lines !== null;
    this.linesRow.hidden = c.lines === null;
    for (const [input, n] of [
      [this.around, c.lines?.[0]],
      [this.across, c.lines?.[1]],
    ] as const) {
      input.min = String(c.lineCounts.min);
      input.max = String(c.lineCounts.max);
      input.step = String(c.lineCounts.step);
      if (n !== undefined && document.activeElement !== input) input.value = String(n);
    }

    const linear = c.linearCurve;
    this.linear.checked = linear !== null;
    const lkey = linear ? linear.rows.map((r) => r.field).join("|") : "";
    if (lkey !== this.linearKey) {
      this.linearKey = lkey;
      this.linearNumberRows = new Map((linear?.rows ?? []).map((r) => [r.field, new NumberRow(this.host, r)]));
      this.linearRows.replaceChildren(...[...this.linearNumberRows.values()].map((r) => r.root));
    }
    for (const r of linear?.rows ?? []) this.linearNumberRows.get(r.field)?.update(r);
    this.linearNote.textContent = linear?.description ?? "";
    this.linearNote.hidden = !linear;

    // The paint status says why a value was refused: Swift writes it and
    // shows it only in the Curves on section, when a face is picked; the
    // page says it here too, where the value was typed.
    const status = [state.status.surface, state.picked ? null : state.status.paint].filter(Boolean).join(" · ");
    this.status.textContent = status;
    this.status.dataset.building = String(c.building);
  }
}

