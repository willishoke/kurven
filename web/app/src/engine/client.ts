// The page's side of the protocol (protocol.ts): a typed client of the
// engine's Worker. It numbers requests, keeps the document's latest state,
// coalesces views by key, drops stale ones, and cancels a request that hangs
// by restarting the Worker and replaying the journal.

import type {
  Bundle,
  DocumentState,
  FromWorker,
  Port,
  Reply,
  Request,
  ToWorker,
  TypedArray,
} from "./protocol.ts";

/** A Worker as the client drives it: a port it can terminate. */
export interface WorkerHandle extends Port {
  terminate(): void;
}

/** A reply as the caller gets it, or a stale view dropped. */
export type Answer =
  | { stale: false; reply: Reply; arrays: Record<string, TypedArray> }
  | { stale: true };

/** Why a request did not get its answer. */
export class Cancelled extends Error {
  constructor(why: string) {
    super(why);
    this.name = "Cancelled";
  }
}

export interface Options {
  /** A view's key: a newer ask under it supersedes this one (protocol.ts,
   * rule 2). Views are not journaled. */
  view?: string;
  /** Cancel, restart and replay if no answer comes within this. */
  timeoutMs?: number;
}

/** Requests that change nothing the journal must replay. */
const VIEWS = new Set(["state", "frame", "bake", "grids", "save", "headless"]);

interface Pending {
  id: number;
  request: Request;
  options: Options;
  resolve: (answer: Answer) => void;
  reject: (error: Error) => void;
  timer?: ReturnType<typeof setTimeout>;
}

type Journal = { kind: "stage"; bundle: Bundle } | { kind: "request"; request: Request };

export class EngineClient {
  private worker!: WorkerHandle;
  /** The Worker has loaded the engine. */
  private booted!: Promise<void>;
  /** ...and, after a restart, the journal is replayed into it: what a
   * request waits for, so none overtakes the replay. */
  private ready!: Promise<void>;
  private next = 1;
  private pending = new Map<number, Pending>();
  /** Per view key: the ask in flight, and the newest one waiting. */
  private views = new Map<string, { inFlight: number | null; waiting: Pending | null }>();
  private journal: Journal[] = [];
  /** The document as the latest reply left it. */
  state: DocumentState | null = null;
  /** How many times the Worker was restarted. */
  restarts = 0;

  private readonly spawn: () => WorkerHandle;
  private readonly wasm: ArrayBuffer | string;

  constructor(spawn: () => WorkerHandle, wasm: ArrayBuffer | string) {
    this.spawn = spawn;
    this.wasm = wasm;
    this.start();
  }

  private start(): void {
    this.worker = this.spawn();
    let resolveReady!: () => void;
    let rejectReady!: (e: Error) => void;
    this.booted = new Promise((resolve, reject) => {
      resolveReady = resolve;
      rejectReady = reject;
    });
    this.ready = this.booted;
    this.worker.onmessage = (event) => {
      const message = event.data as FromWorker;
      switch (message.type) {
        case "ready":
          resolveReady();
          return;
        case "staged":
          return;
        case "reply":
          this.answered(message.id, message.reply, message.arrays);
          return;
        case "failed":
          if (message.id === null) {
            rejectReady(new Error(message.message));
            return;
          }
          // A trap: the module is gone with whatever it held.
          void this.restart(new Cancelled(`the engine failed: ${message.message}`));
          return;
      }
    };
    // A copy of the module's bytes: a transfer would empty the client's.
    const wasm = typeof this.wasm === "string" ? this.wasm : this.wasm.slice(0);
    this.post({ type: "init", wasm }, typeof wasm === "string" ? [] : [wasm]);
  }

  private post(message: ToWorker, transfer: Transferable[] = []): void {
    this.worker.postMessage(message, transfer);
  }

  /** Stage a bundle's files for an `open` or a `headless` run. */
  async stage(bundle: Bundle): Promise<void> {
    await this.ready;
    this.stageNow(bundle);
  }

  private stageNow(bundle: Bundle): void {
    this.journal.push({ kind: "stage", bundle });
    // Copies: the journal keeps the bytes to stage again after a restart.
    const files = bundle.files.map((f) => ({ path: f.path, bytes: f.bytes.slice(0) }));
    this.post({ type: "stage", bundle: { name: bundle.name, files } }, files.map((f) => f.bytes));
  }

  /** Send a request; the answer comes in the order requests were sent. */
  async request(request: Request, options: Options = {}): Promise<Answer> {
    await this.ready;
    return this.requestNow(request, options);
  }

  private requestNow(request: Request, options: Options = {}): Promise<Answer> {
    return new Promise<Answer>((resolve, reject) => {
      const p: Pending = { id: this.next++, request, options, resolve, reject };
      if (options.view !== undefined) {
        const lane = this.views.get(options.view) ?? { inFlight: null, waiting: null };
        this.views.set(options.view, lane);
        if (lane.inFlight !== null) {
          // One in flight already: this waits, replacing any waiting ask,
          // which is stale before it is sent.
          lane.waiting?.resolve({ stale: true });
          lane.waiting = p;
          return;
        }
        lane.inFlight = p.id;
      }
      this.send(p);
    });
  }

  private send(p: Pending): void {
    this.pending.set(p.id, p);
    if (p.options.timeoutMs !== undefined) {
      p.timer = setTimeout(() => {
        void this.cancel(`no answer to ${p.request.kind} within ${p.options.timeoutMs} ms`);
      }, p.options.timeoutMs);
    }
    this.post({ type: "request", id: p.id, request: p.request });
  }

  private answered(id: number, reply: Reply, arrays: Record<string, TypedArray>): void {
    const p = this.pending.get(id);
    if (!p) return; // answered after a restart dropped it
    this.pending.delete(id);
    if (p.timer) clearTimeout(p.timer);
    if (reply.state) this.state = reply.state;
    if (!VIEWS.has(p.request.kind) && p.options.view === undefined && reply.ok) {
      this.journal.push({ kind: "request", request: p.request });
    }
    const view = p.options.view;
    if (view !== undefined) {
      const lane = this.views.get(view)!;
      lane.inFlight = null;
      const waiting = lane.waiting;
      lane.waiting = null;
      if (waiting) {
        // A newer ask came while this one was computed: this one is stale,
        // and the newer one goes now.
        p.resolve({ stale: true });
        lane.inFlight = waiting.id;
        this.send(waiting);
        return;
      }
    }
    p.resolve({ stale: false, reply, arrays });
  }

  /** Cancel whatever is pending: terminate the Worker, reject the pending
   * requests, start a new Worker and replay the journal into it. */
  async cancel(why = "cancelled"): Promise<void> {
    await this.restart(new Cancelled(why));
  }

  private async restart(error: Error): Promise<void> {
    this.worker.terminate();
    this.restarts++;
    const dropped = [...this.pending.values()];
    this.pending.clear();
    for (const lane of this.views.values()) {
      if (lane.waiting) dropped.push(lane.waiting);
      lane.inFlight = null;
      lane.waiting = null;
    }
    for (const p of dropped) {
      if (p.timer) clearTimeout(p.timer);
      p.reject(error);
    }
    this.start();
    const journal = this.journal;
    this.journal = [];
    this.ready = (async () => {
      await this.booted;
      for (const entry of journal) {
        if (entry.kind === "stage") this.stageNow(entry.bundle);
        else await this.requestNow(entry.request);
      }
    })();
    await this.ready;
  }

  /** Stop the Worker for good. */
  close(): void {
    this.worker.terminate();
  }
}

/** A browser Worker running worker-entry.ts. */
export function browserWorker(): WorkerHandle {
  return new Worker(new URL("./worker-entry.ts", import.meta.url), { type: "module" }) as unknown as WorkerHandle;
}
