// A frame of the preview as the engine sends it (kurven_session::preview,
// the `preview` request): what the GPU draws, decided in the engine. The
// page holds the GPU's copies of what it was sent, under the engine's keys,
// and nothing else.

import type { TypedArray } from "../engine/protocol.ts";

/** A pixel grid (`DepthFrame`): row r, column c sample the view point
 * `axis0.lo + r (axis0.hi - axis0.lo) / (rows - 1)` down, and likewise across. */
export interface PixelFrame {
  axis0: [number, number];
  axis1: [number, number];
  rows: number;
  cols: number;
  nudge: number;
}

export type ResourcesJSON =
  | {
      kind: "heightfield";
      gridSize: [number, number];
      grid: [number, number];
      domainLo: [number, number];
      domainSize: [number, number];
      truncationBaked: boolean;
      step: number;
    }
  | { kind: "solid" }
  | { kind: "parametric"; samples: [number, number]; cells: [number, number]; lo: [number, number]; spacing: [number, number] };

export interface UniformsJSON {
  view: number[];
  clip: number[];
  /** Null: none applied in the shader (Swift's infinity). */
  truncationHeight: number | null;
  frame: PixelFrame;
  depthTolerance: number;
  lightDirection: [number, number, number];
  ambient: number;
  depthRange: [number, number];
  surfaceDepthRange: [number, number] | null;
  sight: [number, number, number];
  eye: [number, number, number];
  perspective: boolean;
}

export interface LayerJSON {
  name: string;
  key: string;
  visible: boolean;
  pipeline: "stroke" | "surface";
  onFolds: boolean;
  color: [number, number, number, number];
  /** Over the widest layer's width. */
  width: number;
  /** Null: drawn as it comes (Swift's infinite tolerance). */
  depthTolerance: number | null;
  /** Why the engine could not test the layer for the camera, which it then
   * sends as carried, depth-tested; null when it could. */
  refusal: string | null;
  /** The segments sent with this frame, or null when the page holds the key
   * (or the layer is hidden). */
  segments: number | null;
}

export interface PreviewFrame {
  content: string;
  resources: ResourcesJSON | null;
  uniforms: UniformsJSON;
  mode: "drawing" | "shaded" | "depth";
  layers: LayerJSON[];
}

export type Arrays = Record<string, TypedArray>;

/** The pixel a view point falls in (`DepthFrame::index`), `[row, col]`. */
export function pixelIndex(f: PixelFrame, x: number, y: number): [number, number] {
  // Screen order: rows index view y, columns view x.
  const row = Math.floor(f.nudge + ((f.rows - 1) * (y - f.axis0[0])) / (f.axis0[1] - f.axis0[0]));
  const col = Math.floor(f.nudge + ((f.cols - 1) * (x - f.axis1[0])) / (f.axis1[1] - f.axis1[0]));
  return [row, col];
}

/** A view point in canvas coordinates, pixel centres at half integers:
 * column c's centre samples the view x that `pixelIndex` puts at c. */
export function canvasPoint(f: PixelFrame, x: number, y: number): [number, number] {
  return [
    0.5 + ((f.cols - 1) * (x - f.axis1[0])) / (f.axis1[1] - f.axis1[0]),
    0.5 + ((f.rows - 1) * (y - f.axis0[0])) / (f.axis0[1] - f.axis0[0]),
  ];
}
