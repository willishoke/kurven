// "Curves on face" (PaintSection.swift): what the picked face is, the
// layers on it, each with its own controls and a way to take it off, and an
// Add menu of the curves the face's fields admit (the engine's
// `paint_choices`); on a surface that takes one, "Function…", which paints
// a function whose |f| and arg f can then be contoured. Every edit is one
// undo step. A face is picked by a click on the drawing, held still
// (page/canvas.ts), and the pick is the engine's.

import type { Controls, DocumentState, LayerState } from "../engine/protocol.ts";
import type { Host } from "./controls.ts";
import { h } from "./dom.ts";
import { controlsKey, LAYER_WORDS, LayerControls } from "./layers.ts";

/** An expression and the rectangle of its periods, painted on the torus
 * (FunctionSheet): f is drawn over [0, real) × [0, imag) of the plane. */
class FunctionSheet {
  readonly dialog: HTMLDialogElement;
  private readonly expression: HTMLInputElement;
  private readonly real: HTMLInputElement;
  private readonly imag: HTMLInputElement;
  private readonly refusal: HTMLElement;
  private readonly host: Host;

  constructor(host: Host) {
    this.host = host;
    this.expression = h("input", { type: "text", id: "function-expression", placeholder: "sn(z, 0.64)", "aria-label": "f(z)", spellcheck: false, autocomplete: "off" });
    this.real = h("input", { type: "number", id: "function-real", step: "any", "aria-label": "Real period" });
    this.imag = h("input", { type: "number", id: "function-imag", step: "any", "aria-label": "Imaginary period" });
    this.refusal = h("p", { class: "error", id: "function-refusal", role: "alert" });
    // A function whose periods the catalog knows brings its rectangle.
    this.expression.addEventListener("input", () => {
      const text = this.expression.value;
      void this.host.engine.look({ kind: "periods", expression: text }, "periods").then((reply) => {
        const known = reply?.known as [number, number] | null | undefined;
        if (known && this.expression.value === text) {
          this.real.value = String(known[0]);
          this.imag.value = String(known[1]);
        }
      });
    });
    const cancel = h("button", { type: "button", id: "function-cancel" }, "Cancel");
    cancel.addEventListener("click", () => this.dialog.close());
    const paint = h("button", { type: "submit", class: "primary", id: "function-paint" }, "Paint");
    const form = h(
      "form",
      { method: "dialog" },
      h("h2", {}, "Paint a function"),
      h("label", { class: "field" }, h("span", {}, "f(z)"), this.expression),
      h("div", { class: "field" }, h("span", {}, "Periods"), h("div", { class: "pair" }, this.real, h("span", { class: "times" }, "×"), this.imag, h("span", { class: "times" }, "i"))),
      h("p", { class: "notes" }, "f is drawn over [0, real) × [0, imag) of the plane, which the torus glues into itself: for the drawing to close, they are periods of f."),
      this.refusal,
      h("div", { class: "buttons end" }, cancel, paint),
    );
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      this.paint();
    });
    this.dialog = h("dialog", { class: "sheet", id: "function-sheet", "aria-label": "Paint a function" }, form);
  }

  /** Open on the function painted now, or none, with the periods the engine
   * offers for it. */
  async open(expression: string): Promise<void> {
    this.expression.value = expression;
    this.refusal.textContent = "";
    const reply = await this.host.engine.look({ kind: "periods", expression }, "periods");
    const offered = reply?.offered as [number, number] | undefined;
    if (offered) {
      this.real.value = String(offered[0]);
      this.imag.value = String(offered[1]);
    }
    if (!this.dialog.open) this.dialog.showModal();
    this.expression.focus();
  }

  private paint(): void {
    const expression = this.expression.value;
    const periods: [number, number] = [Number(this.real.value), Number(this.imag.value)];
    this.host.engine.edit({ kind: "function", expression, periods }, (reply) => {
      if (reply.ok) {
        this.dialog.close();
        this.host.changed();
      } else {
        this.refusal.textContent = reply.error ?? "refused";
      }
    });
  }
}

export class PaintSection {
  readonly root: HTMLElement;
  private readonly host: Host;
  private readonly heading: HTMLElement;
  private readonly caption: HTMLElement;
  private readonly list: HTMLUListElement;
  private readonly empty: HTMLElement;
  private readonly add: HTMLSelectElement;
  private readonly functionButton: HTMLButtonElement;
  private readonly status: HTMLElement;
  readonly sheet: FunctionSheet;
  private reach: Controls | null = null;
  private key = "";
  private readonly controls = new Map<string, LayerControls>();
  private expression = "";

  constructor(host: Host) {
    this.host = host;
    this.heading = h("h2", { id: "h-paint" }, "Curves on");
    this.caption = h("p", { class: "notes", id: "paint-caption" });
    this.list = h("ul", { class: "layers", id: "paint-layers" });
    this.empty = h("p", { class: "notes faint" }, "Nothing is drawn on it yet.");
    this.add = h("select", { id: "paint-add", "aria-label": "Add a curve to the picked face" });
    this.add.addEventListener("change", () => {
      const name = this.add.value;
      this.add.value = "";
      if (name) host.engine.edit({ kind: "add", name }, () => host.changed());
    });
    this.sheet = new FunctionSheet(host);
    this.functionButton = h(
      "button",
      {
        type: "button",
        id: "paint-function",
        title: "Paint a function on the torus, over a rectangle of its periods, as `kurven-cli surface periodic` takes one: its |f| and arg f can then be contoured.",
      },
      "Function…",
    );
    this.functionButton.addEventListener("click", () => void this.sheet.open(this.expression));
    this.status = h("p", { class: "status warn", id: "paint-status", "aria-live": "polite" });
    this.root = h(
      "section",
      { "aria-labelledby": "h-paint", id: "paint-section" },
      this.heading,
      this.caption,
      this.empty,
      this.list,
      h("div", { class: "buttons" }, this.add, this.functionButton),
      this.status,
    );
  }

  setReach(reach: Controls): void {
    this.reach = reach;
    this.key = "";
  }

  render(state: DocumentState): void {
    const pick = state.picked;
    this.root.hidden = pick === null;
    if (!pick) return;
    this.heading.textContent = `Curves on ${pick.title}`;
    this.caption.textContent = pick.caption;
    this.expression = pick.expression;
    const layers = pick.layers
      .map((name) => state.layers.find((l) => l.name === name))
      .filter((l): l is LayerState => l !== undefined);
    this.empty.hidden = layers.length > 0;
    const key = layers.map(controlsKey).join("|") + `|${this.reach !== null}`;
    if (key !== this.key) {
      this.key = key;
      this.controls.clear();
      this.list.replaceChildren(...layers.map((l) => this.row(l)));
    }
    for (const l of layers) {
      const li = this.list.querySelector<HTMLElement>(`li[data-layer="${CSS.escape(l.name)}"]`);
      const count = li?.querySelector<HTMLElement>(".count");
      if (count) count.textContent = l.paths !== undefined ? String(l.paths) : "…";
      this.controls.get(l.name)?.update(l);
    }
    // The Add menu: the curves the face's fields admit, offered once the
    // drawing on screen is the one asked for (its levels are read off it).
    const options = [h("option", { value: "" }, pick.choices.length ? "Add…" : "Add: nothing here"), ...pick.choices.map((c) => h("option", { value: c.id }, c.label))];
    const okey = pick.choices.map((c) => c.id).join("|");
    if (this.add.dataset.key !== okey) {
      this.add.dataset.key = okey;
      this.add.replaceChildren(...options);
    }
    this.add.disabled = pick.choices.length === 0;
    this.add.title = state.current ? "Curves the face's fields admit" : "Offered once the drawing on screen is the one asked for: its levels are read off it";
    this.functionButton.hidden = !pick.takesFunction;
    this.status.textContent = state.status.paint ?? "";
  }

  private row(layer: LayerState): HTMLLIElement {
    const remove = h("button", { type: "button", class: "remove", title: `Take ${layer.name} off`, "aria-label": `Take ${layer.name} off` }, "−");
    remove.addEventListener("click", () => this.host.engine.edit({ kind: "remove", layer: layer.name }, () => this.host.changed()));
    const controls = new LayerControls(this.host, this.reach, layer);
    this.controls.set(layer.name, controls);
    return h(
      "li",
      { "data-layer": layer.name },
      h("div", { class: "name" }, LAYER_WORDS[layer.name] ?? layer.name, h("small", {}, layer.name), h("span", { class: "count" }), remove),
      controls.root,
    );
  }
}
