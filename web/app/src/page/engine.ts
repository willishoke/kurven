// The page's side of the engine: requests sent through the client, the
// state each reply carries, and what is in flight, which the indicator shows.
// The page keeps no state of its own: what it shows is the latest state the
// engine sent (HANDOFF-web-port.md, Architecture, rule 1).

import type { Answer, EngineClient } from "../engine/client.ts";
import type { DocumentState, Reply, Request, Slider, TypedArray } from "../engine/protocol.ts";

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
      return "restyling";
    case "truncation":
    case "levels":
    case "resetLevels":
      return "restyling";
    case "undo":
    case "redo":
      return "undoing";
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
      .then(
        () => undefined,
        (error: unknown) => console.error("kurven:", error),
      )
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
  private readonly moves: Latest<number | [number, number]>;
  private readonly after: () => void;
  active = false;

  constructor(engine: Engine, control: Slider, after: () => void) {
    this.engine = engine;
    this.control = control;
    this.after = after;
    this.moves = new Latest(async (value) => {
      await engine.send({ kind: "drag", control, phase: "move", value });
      after();
    });
  }

  begin(): void {
    if (this.active) return;
    this.active = true;
    void this.engine.send({ kind: "drag", control: this.control, phase: "begin" }).then(this.after);
  }

  move(value: number | [number, number]): void {
    if (!this.active) return;
    this.moves.push(value);
  }

  async end(): Promise<void> {
    if (!this.active) return;
    this.active = false;
    await this.moves.flush();
    await this.engine.send({ kind: "drag", control: this.control, phase: "end" });
    this.after();
  }
}
