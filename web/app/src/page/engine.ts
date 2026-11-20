// The page's side of the engine: requests sent through the client, the
// state each reply carries, and what is in flight, which the indicator shows.
// The page keeps no state of its own: what it shows is the latest state the
// engine sent (HANDOFF-web-port.md, Architecture, rule 1).

import { type Answer, Cancelled, type EngineClient } from "../engine/client.ts";
import type { DocumentState, Reply, Request, Slider, Truncation, TypedArray } from "../engine/protocol.ts";

/** What a request makes the engine do, for the indicator. */
function doing(r: Request): string | null {
  switch (r.kind) {
    case "create":
    case "restore":
    case "expression":
      return "sampling";
    case "set":
      return "sampling";
    case "drag":
      if (r.control.slider === "landscape") return r.phase === "move" ? null : "sampling";
      if (r.control.slider === "surfaceField") return "building";
      return "restyling";
    case "truncation":
    case "levels":
    case "resetLevels":
      return "restyling";
    case "createSurface":
    case "surfaceField":
    case "surfaceLines":
    case "surfaceLineCount":
    case "linearCurve":
    case "component":
      return "building";
    case "undo":
    case "redo":
      return "undoing";
    case "pick":
      return "picking";
    case "add":
    case "remove":
    case "function":
    case "strokes":
    case "axis":
    case "light":
      return "painting";
    case "bake":
      return "baking the judged strokes";
    default:
      return null;
  }
}

export class Engine {
  readonly client: EngineClient;
  state: DocumentState | null = null;
  private readonly stateListeners: ((s: DocumentState, r: Request) => void)[] = [];
  private readonly busyListeners: ((what: string[]) => void)[] = [];
  private readonly inFlight = new Map<number, string>();
  private next = 0;
  /** Every request sent and not yet answered, whatever it is. */
  private readonly pending = new Set<Promise<unknown>>();

  constructor(client: EngineClient) {
    this.client = client;
  }

  /** What the engine is doing for the page, newest last. */
  get busy(): string[] {
    return [...this.inFlight.values()];
  }

  onState(listener: (s: DocumentState, r: Request) => void): void {
    this.stateListeners.push(listener);
  }

  onBusy(listener: (what: string[]) => void): void {
    this.busyListeners.push(listener);
  }

  /** Send a request; the reply, with the state it carries taken in. */
  async send(request: Request): Promise<Reply | null> {
    return (await this.call(request))?.reply ?? null;
  }

  /** Send an edit and, once answered, call `after` with the reply. A
   * request a cancel dropped (the export's Cancel restarts the Worker and
   * replays what was answered) is let go quietly: the document is the
   * replayed one, and the next state shows it. Anything else is logged. */
  edit(request: Request, after?: (reply: Reply) => void): void {
    this.send(request).then(
      (reply) => {
        if (reply && after) after(reply);
      },
      quietly,
    );
  }

  /** A look at the document under a view key: not journaled, and a newer
   * look under the same key supersedes it (null when it did). */
  async look(request: Request, view: string): Promise<Reply | null> {
    const answer = await this.client.request(request, { view });
    return answer.stale ? null : answer.reply;
  }

  /** Wait until every request sent so far, and any it led to, is answered. */
  async settled(): Promise<void> {
    while (this.pending.size) await Promise.allSettled([...this.pending]);
  }

  /** `send`, with the arrays the reply carries; null for a stale view. */
  call(request: Request): Promise<{ reply: Reply; arrays: Record<string, TypedArray> } | null> {
    const call = this.calling(request);
    this.pending.add(call);
    void call.then(
      () => this.pending.delete(call),
      () => this.pending.delete(call),
    );
    return call;
  }

  private async calling(request: Request): Promise<{ reply: Reply; arrays: Record<string, TypedArray> } | null> {
    const what = doing(request);
    const id = this.next++;
    if (what) {
      this.inFlight.set(id, what);
      this.told();
    }
    let answer: Answer;
    try {
      answer = await this.client.request(request);
    } finally {
      if (what) {
        this.inFlight.delete(id);
        this.told();
      }
    }
    if (answer.stale) return null;
    const reply = answer.reply;
    if (reply.state) {
      this.state = reply.state;
      for (const l of this.stateListeners) l(reply.state, request);
    }
    return { reply, arrays: answer.arrays };
  }

  private told(): void {
    const what = this.busy;
    for (const l of this.busyListeners) l(what);
  }
}

/** Run `job` for the latest value pushed, one at a time: a value pushed
 * while one runs waits, replacing any waiting before it, so a control that
 * moves faster than the engine answers costs one step behind, never a
 * queue (Swift's `pump`: one job in flight, the latest wins). */
export class Latest<T> {
  private running: Promise<void> | null = null;
  private waiting: { value: T } | null = null;
  private readonly job: (value: T) => Promise<unknown>;

  constructor(job: (value: T) => Promise<unknown>) {
    this.job = job;
  }

  push(value: T): void {
    this.waiting = { value };
    this.pump();
  }

  private pump(): void {
    if (this.running || !this.waiting) return;
    const { value } = this.waiting;
    this.waiting = null;
    this.running = this.job(value)
      .then(() => undefined, quietly)
      .finally(() => {
        this.running = null;
        this.pump();
      });
  }

  /** Wait until every value pushed has run. */
  async flush(): Promise<void> {
    while (this.running || this.waiting) {
      if (!this.running) this.pump();
      await this.running;
    }
  }
}

/** A slider of the engine's taken hold of, moved and let go: one undo step
 * (kurven_session::sliders). Moves are coalesced; the let-go waits for the
 * last move to be sent. */
export class SliderDrag {
  private readonly engine: Engine;
  private readonly control: Slider;
  private readonly moves: Latest<number | [number, number] | Truncation>;
  private readonly after: () => void;
  /** Whether the engine took hold of the slider: false when a cancel
   * dropped the begin, and then the drag sends nothing more. */
  private began: Promise<boolean> = Promise.resolve(false);
  active = false;

  /** `rest`: the moves are a surface field's slider values, which the
   * engine rests at the step or a detent within reach. */
  constructor(engine: Engine, control: Slider, after: () => void, rest = false) {
    this.engine = engine;
    this.control = control;
    this.after = after;
    this.moves = new Latest(async (value) => {
      if (!(await this.began)) return;
      // A banded truncation's height moves the whole truncation.
      const moved = typeof value === "object" && !Array.isArray(value) ? { truncation: value } : { value };
      await engine.send({ kind: "drag", control, phase: "move", ...moved, ...(rest ? { rest: true } : {}) });
      after();
    });
  }

  begin(): void {
    if (this.active) return;
    this.active = true;
    this.began = this.engine.send({ kind: "drag", control: this.control, phase: "begin" }).then(
      () => {
        this.after();
        return true;
      },
      (error: unknown) => {
        quietly(error);
        return false;
      },
    );
  }

  move(value: number | [number, number] | Truncation): void {
    if (!this.active) return;
    this.moves.push(value);
  }

  /** Let go. A begin the engine took is always ended: a let-go a cancel
   * drops is sent again once the Worker is back, so the replayed document
   * closes the drag and files its one undo step. */
  async end(): Promise<void> {
    if (!this.active) return;
    this.active = false;
    await this.moves.flush();
    if (!(await this.began)) return;
    for (let tries = 0; tries < 3; tries++) {
      try {
        await this.engine.send({ kind: "drag", control: this.control, phase: "end" });
        this.after();
        return;
      } catch (error) {
        if (!(error instanceof Cancelled)) {
          quietly(error);
          return;
        }
      }
    }
  }
}

/** A request's failure as the page takes it: one a cancel dropped is
 * expected (the Worker was restarted on purpose), anything else is logged. */
export function quietly(error: unknown): void {
  if (error instanceof Cancelled) return;
  console.error("kurven:", error);
}
