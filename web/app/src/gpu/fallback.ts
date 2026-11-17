// The preview without WebGPU: the judged strokes, what Export writes, drawn
// with Canvas 2D at rest. No depth pass and no draft: every stroke the
// engine's bake returns is already tested for the camera, so it is drawn
// as it comes, each layer at its width over the widest, as the preview draws
// its layers.

import type { TypedArray } from "../engine/protocol.ts";
import { canvasPoint, type PixelFrame } from "./frame.ts";

/** A bake as the engine replies to one: each layer's style, and the frame
 * that maps its page points (view x and y) to the viewport's pixels. */
export interface Judged {
  strokes: { width: number; color: string }[];
  frame: PixelFrame | null;
}

/** Draw judged strokes into a 2D context the size of the viewport, on white.
 * `strokeWidth` is the widest layer's width in device pixels. */
export function drawJudged(
  ctx: CanvasRenderingContext2D,
  judged: Judged,
  arrays: Record<string, TypedArray>,
  strokeWidth: number,
): void {
  const { width, height } = ctx.canvas;
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  const frame = judged.frame;
  if (!frame) {
    ctx.restore();
    return;
  }
  const widest = judged.strokes.reduce((m, s) => Math.max(m, s.width), 0);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  judged.strokes.forEach((style, i) => {
    const points = arrays[`stroke.${i}.points`] as Float64Array | undefined;
    const offsets = arrays[`stroke.${i}.offsets`] as Uint32Array | undefined;
    if (!points || !offsets) return;
    ctx.strokeStyle = style.color;
    ctx.lineWidth = strokeWidth * (widest > 0 ? style.width / widest : 1);
    ctx.beginPath();
    for (let p = 0; p + 1 < offsets.length; p++) {
      const lo = offsets[p] ?? 0;
      const hi = offsets[p + 1] ?? 0;
      for (let k = lo; k < hi; k++) {
        const [x, y] = canvasPoint(frame, points[2 * k] ?? 0, points[2 * k + 1] ?? 0);
        if (k === lo) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  });
  ctx.restore();
}
