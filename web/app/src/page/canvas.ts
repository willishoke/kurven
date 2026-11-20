// The canvas's input, as the Swift app's GestureView maps it: drag to orbit,
// shift- or option-drag to pan, the wheel (or a pinch) to zoom toward the
// pointer. From the keyboard, with the canvas focused: the arrows orbit,
// shift and the arrows pan, + and - zoom about the centre. Each is a
// `Gesture` value the engine folds into its navigator (navigation.rs).
//
// A click held still picks the face under it, or clears the pick off the
// drawing (GestureView's mouseUp): a click that orbits picks nothing. The
// engine reads the point under the pixel and names the face.
//
// Gestures are batched, never merged: while one request is with the
// engine, what the hand does next waits in order, and goes as one
// `gestures` request that applies each as a `gesture` request would. So a
// camera that moves faster than the engine answers -- as it does while a
// frame at rest holds the Worker -- costs one request behind, never a
// queue, and ends where the gestures one by one would put it, to the bit.
// (Summing them would not: the orbit's arithmetic does not associate in
// floating point and its elevation is clamped, and a zoom's anchor moves
// with every pan.)

import type { Gesture } from "../engine/protocol.ts";
import { type Engine, quietly } from "./engine.ts";

export class CanvasInput {
  private readonly engine: Engine;
  private readonly canvas: HTMLCanvasElement;
  private readonly scale: number;
  private readonly moved: () => void;
  /** Given and not yet sent, in order. */
  private waiting: Gesture[] = [];
  private sending: Promise<void> | null = null;
  /** Every gesture given, in order (the last 10,000): the browser lane
   * replays them one by one to check the batches. */
  readonly given: Gesture[] = [];
  private readonly pointers = new Map<number, { x: number; y: number }>();
  /** Whether the pointer down now has moved since it went down. */
  private dragged = false;

  constructor(engine: Engine, canvas: HTMLCanvasElement, scale: number, moved: () => void) {
    this.engine = engine;
    this.canvas = canvas;
    this.scale = scale;
    this.moved = moved;
    canvas.tabIndex = 0;
    canvas.setAttribute("role", "img");
    canvas.addEventListener("pointerdown", (e) => this.down(e));
    canvas.addEventListener("pointermove", (e) => this.move(e));
    canvas.addEventListener("pointerup", (e) => this.up(e));
    for (const ended of ["pointerup", "pointercancel", "lostpointercapture"] as const) {
      canvas.addEventListener(ended, (e) => this.pointers.delete(e.pointerId));
    }
    canvas.addEventListener("wheel", (e) => this.wheel(e), { passive: false });
    canvas.addEventListener("keydown", (e) => this.key(e));
  }

  /** Wait for every gesture given so far to reach the engine. */
  async settled(): Promise<void> {
    while (this.sending || this.waiting.length) {
      if (!this.sending) this.pump();
      await this.sending;
    }
  }

  private give(g: Gesture): void {
    this.waiting.push(g);
    this.given.push(g);
    if (this.given.length > 10_000) this.given.splice(0, this.given.length - 10_000);
    this.pump();
  }

  private pump(): void {
    if (this.sending || !this.waiting.length) return;
    const gestures = this.waiting;
    this.waiting = [];
    const [only] = gestures;
    const request = gestures.length === 1 && only ? { kind: "gesture" as const, gesture: only } : { kind: "gestures" as const, gestures };
    this.sending = this.engine
      .send(request)
      .then(() => this.moved())
      .catch(quietly)
      .finally(() => {
        this.sending = null;
        this.pump();
      });
  }

  /** A point of the canvas in device pixels, from a client point. */
  private at(x: number, y: number): [number, number] {
    const rect = this.canvas.getBoundingClientRect();
    return [(x - rect.left) * this.scale, (y - rect.top) * this.scale];
  }

  private down(e: PointerEvent): void {
    if (this.pointers.size === 0) this.dragged = false;
    this.canvas.focus({ preventScroll: true });
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    this.canvas.setPointerCapture(e.pointerId);
  }

  private move(e: PointerEvent): void {
    const last = this.pointers.get(e.pointerId);
    if (!last) return;
    if (this.pointers.size === 2) {
      this.dragged = true;
      // A pinch: the change in the two pointers' spread, about their middle.
      const [a, b] = [...this.pointers.entries()];
      if (!a || !b) return;
      const other = a[0] === e.pointerId ? b[1] : a[1];
      const before = Math.hypot(last.x - other.x, last.y - other.y);
      const after = Math.hypot(e.clientX - other.x, e.clientY - other.y);
      this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (before > 0 && after > 0 && after !== before) {
        this.give({ kind: "zoom", factor: after / before, at: this.at((e.clientX + other.x) / 2, (e.clientY + other.y) / 2) });
      }
      return;
    }
    const d: [number, number] = [(e.clientX - last.x) * this.scale, (e.clientY - last.y) * this.scale];
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (d[0] === 0 && d[1] === 0) return;
    this.dragged = true;
    // Shift or option pans: "move the paper rather than the camera".
    if (e.shiftKey || e.altKey) this.give({ kind: "pan", delta: [-d[0], d[1]] });
    else this.give({ kind: "orbit", delta: d });
  }

  /** A click, held still, with one pointer: a pick at its pixel, after the
   * gestures given before it. */
  private up(e: PointerEvent): void {
    if (!this.pointers.has(e.pointerId) || this.pointers.size !== 1 || this.dragged || e.button !== 0) return;
    if (e.detail > 1) return; // the second click of a double click
    const pixel = this.at(e.clientX, e.clientY);
    void this.settled().then(() => this.engine.edit({ kind: "pick", pixel }, () => this.moved()));
  }

  private wheel(e: WheelEvent): void {
    e.preventDefault();
    // Toward the cursor: under an orthographic camera that is exactly "hold
    // the point under the pointer still".
    const steps = (-e.deltaY * (e.deltaMode === 0 ? 0.01 : 0.1)) / this.scale;
    if (steps !== 0) this.give({ kind: "zoom", factor: Math.exp(steps), at: this.at(e.clientX, e.clientY) });
  }

  private key(e: KeyboardEvent): void {
    const step = 12 * this.scale;
    const arrows: Record<string, [number, number]> = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowUp: [0, -step],
      ArrowDown: [0, step],
    };
    const d = arrows[e.key];
    const centre: [number, number] = [this.canvas.width / 2, this.canvas.height / 2];
    if (d) {
      e.preventDefault();
      if (e.shiftKey) this.give({ kind: "pan", delta: [-d[0], d[1]] });
      else this.give({ kind: "orbit", delta: d });
    } else if (e.key === "+" || e.key === "=") {
      e.preventDefault();
      this.give({ kind: "zoom", factor: 1.25, at: centre });
    } else if (e.key === "-" || e.key === "_") {
      e.preventDefault();
      this.give({ kind: "zoom", factor: 0.8, at: centre });
    }
  }
}
