// A layer's own controls (PaintSection.swift's LayerControls): its levels,
// its strokes, which way the plateaus' hatching runs, the light an isophote
// or a half-Lambert tone is read under. A drag is one undo step, taken when
// the slider is let go; a key step is an edit of its own. Shared by the
// Layers list and the Curves on section. The reach of the level and stroke
// sliders is the engine's (the `catalog` reply's `controls`).

import type { Controls, LayerState } from "../engine/protocol.ts";
import { type Host, sliderInput } from "./controls.ts";
import { h } from "./dom.ts";
import { quietly, SliderDrag } from "./engine.ts";

/** The glossary's words for the layers a landscape's catalog draws, by the
 * names its manifest gives them (docs/glossary.md). */
export const LAYER_WORDS: Record<string, string> = {
  mag_major: "|f| level curves",
  mag_minor: "|f| level curves, minor",
  ang_major: "arg f level curves",
  ang_minor: "arg f level curves, minor",
  truncation_edge: "truncation edge",
  plateau_hatch: "plateau hatching",
  riser_hatch: "riser hatching",
  wall_edges: "wall edges",
  wall_hatch: "wall hatching",
  folds: "fold curves",
  slope_lines: "slope lines",
  isophotes: "isophotes",
};

/** Which controls a layer has: the controls are made again when it changes. */
export function controlsKey(l: LayerState): string {
  return [l.name, l.levels !== undefined, l.strokes !== undefined, l.axis !== undefined, l.lightAngles !== undefined].join(":");
}

/** A labelled slider row of a layer's: the slider and its value. */
function sliderRow(label: string, input: HTMLInputElement, out: HTMLOutputElement, extra?: HTMLElement): HTMLElement {
  return h("div", { class: "levels" }, h("span", {}, label), input, out, extra ?? h("span", {}));
}

/** The controls of one layer, and how to show its state again. */
export class LayerControls {
  readonly root: HTMLElement;
  private readonly updates: ((l: LayerState) => void)[] = [];

  constructor(host: Host, reach: Controls | null, layer: LayerState) {
    const engine = host.engine;
    const name = layer.name;
    const words = LAYER_WORDS[name] ?? name;
    const rows: HTMLElement[] = [];

    // A described layer carries its question, so its level set can be moved.
    if (layer.levels !== undefined && reach) {
      const range = h("input", {
        type: "range",
        min: reach.levels.min,
        max: reach.levels.max,
        step: 1,
        "aria-label": `Levels of ${words}`,
        "data-levels": name,
      });
      const out = h("output", {});
      const reset = h("button", { type: "button", title: "The levels the landscape chose", "aria-label": `Reset the levels of ${name}` }, "↺");
      reset.addEventListener("click", () => engine.edit({ kind: "resetLevels", layer: name }, () => host.changed()));
      const state = sliderInput(
        range,
        () => new SliderDrag(engine, { slider: "levels", name }, () => host.moved()),
        (x, d) => {
          out.value = String(x);
          d.move(x);
        },
        (x) => engine.edit({ kind: "levels", layer: name, count: x }, () => host.changed()),
      );
      rows.push(sliderRow("levels", range, out, reset));
      this.updates.push((l) => {
        if (l.levels === undefined || state.dragging()) return;
        range.value = String(l.levels);
        out.value = String(l.levels);
      });
    }

    // A hatching carries its spacing as a contour family carries its
    // levels: a count of strokes along the longer side, in whole steps.
    if (layer.strokes !== undefined && reach) {
      const range = h("input", {
        type: "range",
        min: reach.strokes.min,
        max: reach.strokes.max,
        step: 1,
        "aria-label": `Strokes of ${words}`,
        "data-strokes": name,
      });
      const out = h("output", {});
      const state = sliderInput(
        range,
        () => new SliderDrag(engine, { slider: "strokes", name }, () => host.moved()),
        (x, d) => {
          out.value = String(x);
          d.move(x);
        },
        (x) => engine.edit({ kind: "strokes", layer: name, count: x }, () => host.changed()),
      );
      rows.push(sliderRow("strokes", range, out));
      this.updates.push((l) => {
        if (l.strokes === undefined || state.dragging()) return;
        range.value = String(l.strokes);
        out.value = String(l.strokes);
      });
    }

    // Which way the strokes on the plateaus run.
    if (layer.axis !== undefined) {
      const radio = (axis: "real" | "imag", label: string) => {
        const input = h("input", { type: "radio", name: `axis-${name}`, value: axis, "aria-label": `Hatching ${label}` });
        input.addEventListener("change", () => engine.edit({ kind: "axis", axis }, () => host.changed()));
        return { input, label: h("label", {}, input, label) };
      };
      const re = radio("real", "along Re");
      const im = radio("imag", "along Im");
      rows.push(h("fieldset", { class: "segmented axis", "data-axis": name }, h("legend", {}, "Hatching runs"), re.label, im.label));
      this.updates.push((l) => {
        if (l.axis === "real") re.input.checked = true;
        if (l.axis === "imag") im.input.checked = true;
      });
    }

    // The light, by its azimuth and its elevation, each its own slider.
    if (layer.lightAngles !== undefined) {
      const angles = { azimuth: layer.lightAngles[0], elevation: layer.lightAngles[1] };
      const sliders = (["azimuth", "elevation"] as const).map((which) => {
        const [min, max] = which === "azimuth" ? [-180, 180] : [-90, 90];
        const range = h("input", { type: "range", min, max, step: 1, "aria-label": `Light of ${words}: ${which}`, "data-light": `${name}:${which}` });
        const out = h("output", {});
        const value = (x: number): [number, number] =>
          which === "azimuth" ? [x, angles.elevation] : [angles.azimuth, x];
        const drag = () => new SliderDrag(engine, { slider: "light", name }, () => host.moved());
        const state = sliderInput(
          range,
          drag,
          (x, d) => {
            out.value = `${Math.round(x)}°`;
            d.move(value(x));
          },
          (x) => {
            // A key step: one undo step, as the slider set with no drag is.
            const d = drag();
            d.begin();
            d.move(value(x));
            void d.end().then(() => host.changed(), quietly);
          },
        );
        rows.push(sliderRow(which, range, out));
        return { range, out, state, which };
      });
      this.updates.push((l) => {
        if (!l.lightAngles) return;
        [angles.azimuth, angles.elevation] = l.lightAngles;
        for (const s of sliders) {
          if (s.state.dragging()) continue;
          const x = s.which === "azimuth" ? angles.azimuth : angles.elevation;
          s.range.value = String(x);
          s.out.value = `${Math.round(x)}°`;
        }
      });
    }

    this.root = h("div", { class: "layer-controls" }, ...rows);
    this.update(layer);
  }

  update(layer: LayerState): void {
    for (const u of this.updates) u(layer);
  }
}
