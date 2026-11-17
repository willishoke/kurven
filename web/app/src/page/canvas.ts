// The canvas's input, as the Swift app's GestureView maps it: drag to orbit,
// shift- or option-drag to pan, the wheel (or a pinch) to zoom toward the
// pointer. From the keyboard, with the canvas focused: the arrows orbit,
// shift and the arrows pan, + and - zoom about the centre. Each is a
// `Gesture` value the engine folds into its navigator (navigation.rs).
//
// Gestures are coalesced: while one is with the engine, what the hand does
// next is summed into one gesture of each kind, so a camera that moves
// faster than the engine answers -- as it does while a frame at rest holds
// the Worker -- costs one request behind, never a queue.

import type { Gesture } from "../engine/protocol.ts";
import type { Engine } from "./engine.ts";

interface Pending {
  orbit: [number, number];
  pan: [number, number];
  zoom: { factor: number; at: [number, number] } | null;
}

export class CanvasInput {
  private readonly engine: Engine;
  private readonly canvas: HTMLCanvasElement;
  private readonly scale: number;
  private readonly moved: () => void;
  private pending: Pending = { orbit: [0, 0], pan: [0, 0], zoom: null };
  private sending: Promise<void> | null = null;
  private readonly pointers = new Map<number, { x: number; y: number }>();

  constructor(engine: Engine, canvas: HTMLCanvasElement, scale: number, moved: () => void) {
    this.engine = engine;
    this.canvas = canvas;
    this.scale = scale;
    this.moved = moved;
    canvas.tabIndex = 0;
    canvas.setAttribute("role", "img");
    canvas.addEventListener("pointerdown", (e) => this.down(e));
    canvas.addEventListener("pointermove", (e) => this.move(e));
    for (const ended of ["pointerup", "pointercancel", "lostpointercapture"] as const) {
      canvas.addEventListener(ended, (e) => this.pointers.delete(e.pointerId));
    }
    canvas.addEventListener("wheel", (e) => this.wheel(e), { passive: false });
    canvas.addEventListener("keydown", (e) => this.key(e));
  }

  /** Wait for every gesture given so far to reach the engine. */
  async settled(): Promise<void> {
    while (this.sending || this.hasPending()) {
      if (!this.sending) this.pump();
      await this.sending;
    }
  }

  private hasPending(): boolean {
    const p = this.pending;
    return p.orbit[0] !== 0 || p.orbit[1] !== 0 || p.pan[0] !== 0 || p.pan[1] !== 0 || p.zoom !== null;
  }

  private give(g: Gesture): void {
    const p = this.pending;
    switch (g.kind) {
      case "orbit":
        p.orbit = [p.orbit[0] + g.delta[0], p.orbit[1] + g.delta[1]];
        break;
      case "pan":
        p.pan = [p.pan[0] + g.delta[0], p.pan[1] + g.delta[1]];
        break;
      case "zoom":
        p.zoom = { factor: (p.zoom?.factor ?? 1) * g.factor, at: g.at };
        break;
      default:
        break;
    }
    this.pump();
  }

  private pump(): void {
    if (this.sending || !this.hasPending()) return;
    const p = this.pending;
    this.pending = { orbit: [0, 0], pan: [0, 0], zoom: null };
    const gestures: Gesture[] = [];
    if (p.orbit[0] !== 0 || p.orbit[1] !== 0) gestures.push({ kind: "orbit", delta: p.orbit });
    if (p.pan[0] !== 0 || p.pan[1] !== 0) gestures.push({ kind: "pan", delta: p.pan });
    if (p.zoom) gestures.push({ kind: "zoom", factor: p.zoom.factor, at: p.zoom.at });
    this.sending = (async () => {
      for (const gesture of gestures) await this.engine.send({ kind: "gesture", gesture });
      this.moved();
    })()
      .catch((error: unknown) => console.error("kurven gesture:", error))
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
    this.canvas.focus({ preventScroll: true });
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    this.canvas.setPointerCapture(e.pointerId);
  }

  private move(e: PointerEvent): void {
    const last = this.pointers.get(e.pointerId);
    if (!last) return;
    if (this.pointers.size === 2) {
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
    // Shift or option pans: "move the paper rather than the camera".
    if (e.shiftKey || e.altKey) this.give({ kind: "pan", delta: [-d[0], d[1]] });
    else this.give({ kind: "orbit", delta: d });
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
