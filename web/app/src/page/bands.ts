// A banded truncation as rows (TruncationSection.swift's `bandRows`):
// gamma's per-spire truncation, each band truncating everything to the left
// of its threshold, and a last row for everything else. Tested in order,
// first match wins, which is how the rows read top to bottom. A height
// dragged is one undo step; a threshold typed, a band added or taken off is
// one of its own (`Document::edit_truncation`).

import type { Truncation } from "../engine/protocol.ts";
import type { Host } from "./controls.ts";
import { h, shown } from "./dom.ts";
import { quietly, SliderDrag } from "./engine.ts";

type Bands = Extract<Truncation, { kind: "realBands" }>;

/** The truncation with `beyond` as the manifest writes it: absent when
 * there is none (infinite). */
function banded(bands: Bands["bands"], beyond: number | undefined): Truncation {
  return beyond !== undefined && Number.isFinite(beyond) ? { kind: "realBands", bands, beyond } : { kind: "realBands", bands } as Truncation;
}

export class BandRows {
  readonly root: HTMLElement;
  private readonly host: Host;
  private readonly rows: HTMLElement;
  private readonly beyondRange: HTMLInputElement;
  private readonly beyondField: HTMLInputElement;
  private current: Bands | null = null;
  /** The rows' drags under way: the drawn rows are left alone meanwhile. */
  private dragging = 0;
  private key = "";
  /** The height sliders' least value: the engine's reach. */
  private readonly least: () => number;

  constructor(host: Host, least: () => number) {
    this.least = least;
    this.host = host;
    this.rows = h("div", { id: "band-rows" });
    this.beyondRange = h("input", { type: "range", step: "any", id: "band-beyond", "aria-label": "Beyond the last band: height" });
    this.beyondField = h("input", { type: "number", step: "0.01", "aria-label": "Beyond the last band: height", id: "band-beyond-field" });
    this.drag(this.beyondRange, (x) => {
      const t = this.current;
      return t ? banded(t.bands, x) : null;
    });
    this.beyondField.addEventListener("change", () => {
      const x = Number(this.beyondField.value);
      const t = this.current;
      if (t && this.beyondField.value !== "" && Number.isFinite(x)) this.send(banded(t.bands, x));
    });
    const add = h("button", { type: "button", id: "band-add" }, "Add a band");
    add.addEventListener("click", () => {
      const t = this.current;
      if (!t) return;
      const last = t.bands[t.bands.length - 1];
      const below = (last?.below ?? -1) + 1;
      const height = t.beyond !== undefined && Number.isFinite(t.beyond) ? t.beyond : 5;
      this.send(banded([...t.bands, { below, height }], t.beyond));
    });
    this.root = h(
      "div",
      { class: "bands", id: "bands" },
      this.rows,
      h("div", { class: "row" }, h("label", { class: "label", for: "band-beyond" }, "Beyond"), h("div", { class: "slider" }, this.beyondRange, this.beyondField)),
      h("div", { class: "buttons" }, add),
    );
  }

  private send(t: Truncation): void {
    this.host.engine.edit({ kind: "truncation", truncation: t }, () => this.host.changed());
  }

  /** A height slider dragged as the truncation slider: taken hold of, each
   * move the whole truncation, let go (one undo step, "Truncation"). */
  private drag(range: HTMLInputElement, next: (x: number) => Truncation | null): void {
    let active: SliderDrag | null = null;
    range.addEventListener("pointerdown", () => {
      if (active) return;
      active = new SliderDrag(this.host.engine, { slider: "truncation" }, () => this.host.moved());
      active.begin();
      this.dragging++;
      const up = () => {
        window.removeEventListener("pointerup", up);
        window.removeEventListener("pointercancel", up);
        const d = active;
        active = null;
        this.dragging--;
        if (d) void d.end().catch(quietly);
      };
      window.addEventListener("pointerup", up);
      window.addEventListener("pointercancel", up);
    });
    range.addEventListener("input", () => {
      const t = next(Number(range.value));
      if (!t) return;
      if (active) active.move(t);
      else this.send(t);
    });
  }

  /** Show the truncation the engine says, and `ceiling` as the sliders' top
   * (room to raise it past where it ends: max(3 h, 10)). */
  update(t: Bands, ceiling: number): void {
    this.current = t;
    if (this.dragging > 0) return;
    const key = `${t.bands.length}`;
    if (key !== this.key) {
      this.key = key;
      this.rows.replaceChildren(...t.bands.map((_, i) => this.row(i)));
    }
    t.bands.forEach((b, i) => {
      const row = this.rows.children[i] as HTMLElement | undefined;
      if (!row) return;
      const below = row.querySelector<HTMLInputElement>("input[type=number]")!;
      const range = row.querySelector<HTMLInputElement>("input[type=range]")!;
      const out = row.querySelector<HTMLOutputElement>("output")!;
      if (document.activeElement !== below) below.value = shown(b.below);
      range.min = String(this.least());
      range.max = String(ceiling);
      range.value = String(b.height);
      out.value = b.height.toFixed(2);
    });
    const beyond = t.beyond !== undefined && Number.isFinite(t.beyond) ? t.beyond : ceiling;
    this.beyondRange.min = String(this.least());
    this.beyondRange.max = String(ceiling);
    this.beyondRange.value = String(beyond);
    if (document.activeElement !== this.beyondField) this.beyondField.value = beyond.toFixed(2);
  }

  /** Band `i`: its threshold, its height, and a way to take it off. */
  private row(i: number): HTMLElement {
    const below = h("input", { type: "number", step: "0.1", "aria-label": `Band ${i + 1}: Re below`, "data-band": i });
    below.addEventListener("change", () => {
      const x = Number(below.value);
      const t = this.current;
      if (!t || below.value === "" || !Number.isFinite(x)) return;
      const bands = t.bands.map((b, k) => (k === i ? { ...b, below: x } : b));
      this.send(banded(bands, t.beyond));
    });
    const range = h("input", { type: "range", step: "any", "aria-label": `Band ${i + 1}: height`, "data-band": i });
    this.drag(range, (x) => {
      const t = this.current;
      if (!t) return null;
      return banded(
        t.bands.map((b, k) => (k === i ? { ...b, height: x } : b)),
        t.beyond,
      );
    });
    const out = h("output", {});
    range.addEventListener("input", () => {
      out.value = Number(range.value).toFixed(2);
    });
    const remove = h("button", { type: "button", class: "remove", "aria-label": `Take band ${i + 1} off`, "data-band-remove": i }, "−");
    remove.addEventListener("click", () => {
      const t = this.current;
      if (!t) return;
      this.send(banded(t.bands.filter((_, k) => k !== i), t.beyond));
    });
    return h("div", { class: "band" }, h("span", { class: "faint" }, "Re <"), below, h("span", { class: "faint" }, "height"), range, out, remove);
  }
}
