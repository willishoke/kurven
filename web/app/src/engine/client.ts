// The page's side of the protocol (protocol.ts): a typed client of the
// engine's Worker. It numbers requests, keeps the document's latest state,
// coalesces views by key, drops stale ones, and cancels a request that hangs
// by restarting the Worker and replaying the journal.
//
// The journal is every staged bundle and every request that touched the
// document (the reply says so: `touched`, a failed open among them), kept
// compact so a replay is bounded by what the document is, not by how long
// it was edited:
//
// - An `open`, `create`, `createSurface` or `restore` that succeeds is a clean slate
//   (the engine forgets the document's edits and its camera, and adopts the
//   new one whole), so the requests before it are dropped, but for the last
//   of each that sets what an open keeps (`viewport`, `mode`,
//   `draftResolution`, `refine`). An open that fails is not: the engine, as
//   Swift's Document, keeps the landscape or surface request, the levels and
//   the styling it had, which a later edit acts on. Staged bundles stay: any
//   later open may read them, as the engine keeps them.
// - A run of gestures (one at a time or several in a `gestures` request) is
//   replaced by one `camera` request, the camera the
//   run left, to the bit (`camera.exact` in the state): gestures move the
//   navigator alone. They are not summed: the orbit's folds do not
//   associate in floating point, and its elevation is clamped.
// - A run of moves of one slider is its last move, for the sliders whose
//   moves set a value outright and are never refused: a landscape's window,
//   the truncation, a light, a layer's levels. A surface's field and a
//   hatching's strokes can be refused a step, and a refused step keeps the
//   one before, so their moves are kept.
// - A run of one setting outright (`viewport`, `mode`, the resolutions,
//   `refine`) is its last.
// - A bundle staged again under its name, every file of the earlier one
//   among the later's, replaces it, unless an open between reads it.
//
// A replay keeps the whole journal until it has run: a second cancel during
// a replay terminates that Worker too and replays the same journal into the
// next. An entry that traps the engine on two replays in a row is dropped
// with everything after it, which may depend on it, and the next call is
// told (`JournalTruncated`); the document is then the one up to that entry. The replayed document equals the last one reported but for its
// status lines, which are messages: a save's or a bake's is not replayed.
//
// Other engines can hold the same document (pool.ts): beside the journal the
// client keeps a log of what it recorded, in order, each entry marked when it
// took the place of the one before (a camera, a slider's move, a setting set
// outright: compaction's own claim that the new entry alone, where the old
// one was, leaves the same document as both). An engine that has applied the
// log up to some place catches up by applying the rest, each entry that a
// later one took the place of left out; one that cannot (the journal was cut
// since) starts again from a copy of the journal (`snapshot`).

import type {
  Bundle,
  DocumentState,
  ExactCamera,
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

/** A replay trapped the engine twice on one entry: the journal was cut
 * there, and the document is what the entries before it make. */
export class JournalTruncated extends Error {
  readonly request: Request | null;
  constructor(request: Request | null, why: string) {
    super(
      `the engine trapped twice replaying ${request ? request.kind : "a staged bundle"}, ` +
        `so it and what followed are dropped: ${why}`,
    );
    this.name = "JournalTruncated";
    this.request = request;
  }
}

export interface Options {
  /** A view's key: a newer ask under it supersedes this one (protocol.ts,
   * rule 2). Views are not journaled. */
  view?: string;
  /** Cancel, restart and replay if no answer comes within this. */
  timeoutMs?: number;
  /** Typed arrays given to this request alone, by name (`kv_give`): another
   * engine's answer, which the request reads back to the bit. Copied as
   * sent, so the caller keeps its own. */
  inputs?: Record<string, TypedArray>;
}

/** The document as the client's engine holds it at one moment, for another
 * engine to hold too: the journal then, and how far the log had come. */
export interface Snapshot {
  /** The log's length then. */
  at: number;
  /** How many times the journal had been cut then: a log from before a cut
   * cannot be caught up across it. */
  cuts: number;
  journal: readonly Entry[];
}

/** Requests that change nothing the journal must replay: views of the
 * document. The page reads it too, to tell an edit from a look. */
export const VIEWS: ReadonlySet<string> = new Set([
  "state",
  "catalog",
  "edited",
  "frame",
  "preview",
  "surfaceImage",
  "bake",
  "grids",
  "save",
  "headless",
  "bakePlan",
  "bakePiece",
  "bakeAssemble",
  "restLayers",
  "restLayout",
]);
/** Requests after which the document is a clean slate. */
const OPENING = new Set(["open", "create", "createSurface", "restore"]);
/** Settings an open keeps, so the last of each before it is replayed. */
const KEPT_BY_OPEN = new Set(["viewport", "mode", "draftResolution", "refine"]);
/** Settings set outright, so a run of one is its last. */
const OUTRIGHT = new Set(["viewport", "mode", "draftResolution", "bakeResolution", "refine", "depthTolerance"]);
/** Sliders whose moves set a value outright and are never refused. */
const OUTRIGHT_SLIDERS = new Set(["landscape", "truncation", "light", "levels"]);

/** An entry of the journal: a staged bundle, or a request that touched the
 * document. */
export type Entry =
  | { kind: "stage"; bundle: Bundle }
  | { kind: "request"; request: Request; ok: boolean; camera?: ExactCamera };

interface Pending {
  id: number;
  /** What it sends: a request, or a bundle to stage. */
  work: { kind: "request"; request: Request } | { kind: "stage"; bundle: Bundle };
  options: Options;
  /** Sent by a replay, of the journal's entry at this place: answered, it
   * is not journaled again. */
  replay: boolean;
  entry?: number;
  resolve: (answer: Answer) => void;
  reject: (error: Error) => void;
  timer?: ReturnType<typeof setTimeout>;
}

/** A path in a bundle as the engine reads it (kurven_core's normal_path). */
function normalPath(path: string): string {
  return path
    .split("/")
    .filter((c) => c !== "" && c !== ".")
    .join("/");
}

function same(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

export class EngineClient {
  private worker: WorkerHandle | null = null;
  /** The current Worker has loaded the engine. */
  private booted: Promise<void> = Promise.resolve();
  /** ...and the journal is replayed into it: what a request waits for, so
   * none overtakes a replay. Rejected while the engine would not load. */
  private ready: Promise<void> = Promise.resolve();
  /** Bumped by every start, so a Worker's messages, and a replay, that a
   * newer start has overtaken are ignored. */
  private epoch = 0;
  private bootFailed = false;
  /** Rejects the current boot: a Worker terminated before it loaded never
   * says so, and a replay waiting for it must give way. */
  private abandonBoot: (error: Error) => void = () => {};
  private closed = false;
  private next = 1;
  private pending = new Map<number, Pending>();
  /** Per view key: the ask in flight, and the newest one waiting. */
  private views = new Map<string, { inFlight: number | null; waiting: Pending | null }>();
  private journal: Entry[] = [];
  /** The journal entry the last replay trapped on, if the replay after it
   * has not run past it. */
  private trappedAt: number | null = null;
  /** Said to the next call: the journal was cut. */
  private truncated: JournalTruncated | null = null;
  /** The document as the latest reply left it. */
  state: DocumentState | null = null;
  /** How many times the Worker was restarted. */
  restarts = 0;
  /** What was recorded, in order, from `logBase` on (the header says why),
   * kept only once a pool asks for it (`keepLog`). */
  private log: { entry: Entry; replaces: boolean }[] = [];
  private logging = false;
  private logBase = 0;
  /** How many times the journal was cut (`JournalTruncated`). */
  private cuts = 0;
  /** The engine's linear memory, in bytes, after its last reply. */
  memory = 0;

  private readonly spawn: () => WorkerHandle;
  private readonly wasm: ArrayBuffer | string;

  /** `journal`: what the new engine replays before any request, the
   * journal of another client's engine, so it holds the same document. */
  constructor(spawn: () => WorkerHandle, wasm: ArrayBuffer | string, journal: readonly Entry[] = []) {
    this.spawn = spawn;
    this.wasm = wasm;
    this.journal = journal.slice();
    this.relaunch();
  }

  /** The document as this engine holds it now, for another to hold. */
  snapshot(): Snapshot {
    return { at: this.logBase + this.log.length, cuts: this.cuts, journal: this.journal.slice() };
  }

  /** What an engine that holds the document as of the log's place `from`
   * applies to hold it as of `to`, each entry a later one took the place of
   * left out; null when the log no longer holds `from` (it was let go of, or
   * the journal was cut since `cuts`). */
  catchUp(from: number, to: number, cuts: number): Entry[] | null {
    if (cuts !== this.cuts || from < this.logBase || to > this.logBase + this.log.length || from > to) return null;
    const out: Entry[] = [];
    for (let i = from; i < to; i++) {
      const next = i + 1 < to ? this.log[i + 1 - this.logBase] : undefined;
      if (next?.replaces) continue;
      out.push(this.log[i - this.logBase]!.entry);
    }
    return out;
  }

  /** Keep the log from now on, for a pool's helpers to catch up on: until
   * then, and on a helper's own client, nothing is logged. */
  keepLog(): void {
    if (this.logging) return;
    this.logging = true;
    this.log = [];
    this.logBase = 0;
  }

  /** Let go of the log before `at`: no engine will catch up from there. */
  forgetLog(at: number): void {
    const n = Math.min(Math.max(at - this.logBase, 0), this.log.length);
    this.log.splice(0, n);
    this.logBase += n;
  }

  /** Hold another document: the journal replaced by `journal`, the log
   * begun again, the Worker restarted and the journal replayed into it. */
  adopt(journal: readonly Entry[]): Promise<void> {
    this.journal = journal.slice();
    this.log = [];
    this.logBase = 0;
    this.cuts = 0;
    return this.restart(new Cancelled("the engine was given another document"));
  }

  /** How many entries a replay would send. */
  get journalLength(): number {
    return this.journal.length;
  }

  /** The journal's requests, in order, without its staged bundles: what a
   * native session replays to hold the same document (`kurven-cli
   * session`), as a restarted Worker replays it. */
  get journaled(): Request[] {
    return this.journal.flatMap((e) => (e.kind === "request" ? [e.request] : []));
  }

  // MARK: - the Worker

  /** A new Worker, loading the engine. */
  private start(): void {
    this.abandonBoot(new Cancelled("restarted before the engine loaded"));
    const epoch = ++this.epoch;
    const worker = this.spawn();
    this.worker = worker;
    let resolveBoot!: () => void;
    let rejectBoot!: (e: Error) => void;
    this.booted = new Promise((resolve, reject) => {
      resolveBoot = resolve;
      rejectBoot = reject;
    });
    this.booted.catch(() => {});
    this.abandonBoot = rejectBoot;
    worker.onmessage = (event) => {
      if (epoch !== this.epoch || this.closed) return;
      const message = event.data as FromWorker;
      switch (message.type) {
        case "ready":
          resolveBoot();
          return;
        case "staged":
          this.staged(message.id);
          return;
        case "reply":
          if (message.memory !== undefined) this.memory = message.memory;
          this.answered(message.id, message.reply, message.arrays);
          return;
        case "failed":
          if (message.id === null) {
            // The engine would not load: said to whoever waits, and tried
            // again by the next request.
            this.bootFailed = true;
            worker.terminate();
            rejectBoot(new Error(`the engine would not load: ${message.message}`));
          } else if (message.trap) {
            // A trap: the module is gone with whatever it held.
            this.trapped(message.id, message.message);
          } else {
            this.failed(message.id, new Error(message.message));
          }
          return;
      }
    };
    // A copy of the module's bytes: a transfer would empty the client's.
    const wasm = typeof this.wasm === "string" ? this.wasm : this.wasm.slice(0);
    this.post({ type: "init", wasm }, typeof wasm === "string" ? [] : [wasm]);
  }

  /** Start a Worker and replay the whole journal into it. The journal is
   * left as it is until the replay has run; a replay a newer start
   * overtakes gives way to the newer one, and its callers wait for that. */
  private relaunch(): Promise<void> {
    this.start();
    const epoch = this.epoch;
    const journal = this.journal.slice();
    const replay = async (): Promise<void> => {
      try {
        // A boot that fails rejects the replay: said to whoever waits.
        await this.booted;
      } catch (error) {
        if (epoch === this.epoch) throw error;
        return this.ready;
      }
      for (const [i, entry] of journal.entries()) {
        if (epoch !== this.epoch) break;
        try {
          if (entry.kind === "stage") await this.send({ kind: "stage", bundle: entry.bundle }, {}, true, i);
          else await this.send({ kind: "request", request: entry.request }, {}, true, i);
        } catch {
          // Overtaken by a newer start (checked above), or an entry the
          // engine took before and refuses now, which the rest goes on past.
        }
      }
      if (epoch !== this.epoch) return this.ready;
      this.trappedAt = null;
    };
    this.ready = replay();
    this.ready.catch(() => {});
    return this.ready;
  }

  private post(message: ToWorker, transfer: Transferable[] = []): void {
    this.worker?.postMessage(message, transfer);
  }

  /** Before a call: refused once closed; the engine loaded again if it
   * would not load before; then the replay waited for, again if a restart
   * replaced it meanwhile. Returns the epoch of the Worker it waited for,
   * which the caller checks is still the current one as it sends: a restart
   * between the two would post the call to a Worker not yet loaded and
   * replayed. */
  private async ensure(): Promise<number> {
    for (;;) {
      if (this.closed) throw new Cancelled("the client is closed");
      if (this.bootFailed) {
        this.bootFailed = false;
        this.relaunch();
      }
      const epoch = this.epoch;
      await this.ready;
      if (this.closed) throw new Cancelled("the client is closed");
      if (this.truncated) {
        const error = this.truncated;
        this.truncated = null;
        throw error;
      }
      if (epoch === this.epoch) return epoch;
    }
  }

  // MARK: - calls

  /** Stage a bundle's files for an `open` or a `headless` run. Resolves
   * once the engine holds them; rejects, staging nothing to replay, if it
   * refuses them. */
  async stage(bundle: Bundle): Promise<void> {
    while ((await this.ensure()) !== this.epoch);
    await this.send({ kind: "stage", bundle }, {}, false);
  }

  /** Send a request; the answer comes in the order requests were sent. */
  async request(request: Request, options: Options = {}): Promise<Answer> {
    while ((await this.ensure()) !== this.epoch);
    return this.send({ kind: "request", request }, options, false);
  }

  private send(work: Pending["work"], options: Options, replay: boolean, entry?: number): Promise<Answer> {
    return new Promise<Answer>((resolve, reject) => {
      const p: Pending = { id: this.next++, work, options, replay, entry, resolve, reject };
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
      this.dispatch(p);
    });
  }

  private dispatch(p: Pending): void {
    this.pending.set(p.id, p);
    if (p.options.timeoutMs !== undefined) {
      const kind = p.work.kind === "request" ? p.work.request.kind : "stage";
      p.timer = setTimeout(() => {
        this.cancel(`no answer to ${kind} within ${p.options.timeoutMs} ms`).catch(() => {});
      }, p.options.timeoutMs);
    }
    if (p.work.kind === "stage") {
      // Copies: the journal keeps the bytes to stage again after a restart.
      const files = p.work.bundle.files.map((f) => ({ path: f.path, bytes: f.bytes.slice(0) }));
      this.post({ type: "stage", id: p.id, bundle: { name: p.work.bundle.name, files } }, files.map((f) => f.bytes));
    } else {
      const given = Object.entries(p.options.inputs ?? {});
      const inputs = given.map(([name, a]) => ({
        name,
        bytes: (a.buffer as ArrayBuffer).slice(a.byteOffset, a.byteOffset + a.byteLength),
      }));
      this.post(
        { type: "request", id: p.id, request: p.work.request, ...(inputs.length ? { inputs } : {}) },
        inputs.map((i) => i.bytes),
      );
    }
  }

  private take(id: number): Pending | undefined {
    const p = this.pending.get(id);
    if (!p) return undefined; // answered after a restart dropped it
    this.pending.delete(id);
    if (p.timer) clearTimeout(p.timer);
    return p;
  }

  private staged(id: number): void {
    const p = this.take(id);
    if (!p || p.work.kind !== "stage") return;
    if (!p.replay) this.record({ kind: "stage", bundle: p.work.bundle });
    p.resolve({ stale: true });
  }

  private failed(id: number, error: Error): void {
    const p = this.take(id);
    if (!p) return;
    this.release(p);
    p.reject(error);
  }

  private answered(id: number, reply: Reply, arrays: Record<string, TypedArray>): void {
    const p = this.take(id);
    if (!p || p.work.kind !== "request") return;
    if (reply.state) this.state = reply.state;
    const request = p.work.request;
    if (!p.replay && !VIEWS.has(request.kind) && p.options.view === undefined && reply.touched !== false) {
      this.record({ kind: "request", request, ok: reply.ok, camera: reply.state?.camera?.exact });
    }
    if (this.release(p)) {
      // A newer ask came while this one was computed: this one is stale.
      p.resolve({ stale: true });
      return;
    }
    p.resolve({ stale: false, reply, arrays });
  }

  /** Free a view's lane, sending the ask waiting in it. True when there was
   * one, which makes this one stale. */
  private release(p: Pending): boolean {
    const view = p.options.view;
    if (view === undefined) return false;
    const lane = this.views.get(view)!;
    lane.inFlight = null;
    const waiting = lane.waiting;
    lane.waiting = null;
    if (!waiting) return false;
    lane.inFlight = waiting.id;
    this.dispatch(waiting);
    return true;
  }

  // MARK: - the journal

  private record(entry: Entry): void {
    const { kept, replaces } = this.compact(entry);
    if (!this.logging) return;
    this.log.push({ entry: kept, replaces });
    // Trimmed with the journal: an engine that would catch up on more of the
    // log than twice the journal (and a few) replays the journal instead
    // (pool.ts's `sync`), so the log keeps no more, staged bundles and all.
    const most = 2 * this.journal.length + 16;
    if (this.log.length > most) this.forgetLog(this.logBase + this.log.length - most);
  }

  /** Put `entry` in the journal by the header's rules: what went in, and
   * whether it took the place of the journal's last entry. */
  private compact(entry: Entry): { kept: Entry; replaces: boolean } {
    const j = this.journal;
    const last = j[j.length - 1];
    if (entry.kind === "stage") {
      const name = entry.bundle.name;
      const paths = new Set(entry.bundle.files.map((f) => normalPath(f.path)));
      const opens = (i: number) =>
        j.slice(i + 1).some((e) => e.kind === "request" && e.request.kind === "open" && e.request.name === name);
      this.journal = j.filter(
        (e, i) =>
          e.kind !== "stage" ||
          e.bundle.name !== name ||
          opens(i) ||
          !e.bundle.files.every((f) => paths.has(normalPath(f.path))),
      );
      this.journal.push(entry);
      return { kept: entry, replaces: false };
    }
    const request = entry.request;
    if (OPENING.has(request.kind) && entry.ok) {
      const lastOf = new Map<string, number>();
      j.forEach((e, i) => {
        if (e.kind === "request" && KEPT_BY_OPEN.has(e.request.kind)) lastOf.set(e.request.kind, i);
      });
      this.journal = j.filter((e, i) => e.kind === "stage" || lastOf.get(e.request.kind) === i);
      this.journal.push(entry);
      return { kept: entry, replaces: false };
    }
    const lastRequest = last?.kind === "request" ? last : undefined;
    const moving = (r: Request) => r.kind === "gesture" || r.kind === "gestures" || r.kind === "camera";
    if (moving(request) && entry.camera) {
      const camera: Entry = { kind: "request", request: { kind: "camera", ...entry.camera }, ok: true, camera: entry.camera };
      const run = lastRequest && moving(lastRequest.request);
      const replaces = Boolean(run && lastRequest.camera);
      if (replaces) j.pop();
      j.push(camera);
      return { kept: camera, replaces };
    }
    let replaces = false;
    if (
      request.kind === "drag" &&
      request.phase === "move" &&
      OUTRIGHT_SLIDERS.has(request.control.slider) &&
      lastRequest?.request.kind === "drag" &&
      lastRequest.request.phase === "move" &&
      same(lastRequest.request.control, request.control)
    ) {
      j.pop();
      replaces = true;
    } else if (OUTRIGHT.has(request.kind) && lastRequest?.request.kind === request.kind) {
      j.pop();
      replaces = true;
    }
    j.push(entry);
    return { kept: entry, replaces };
  }

  // MARK: - cancelling

  /** The engine trapped on a message: restart. A replayed entry it traps on
   * twice in a row is cut from the journal, with what follows it. */
  private trapped(id: number, why: string): void {
    const p = this.pending.get(id);
    if (p?.replay && p.entry !== undefined) {
      if (this.trappedAt === p.entry) {
        const entry = this.journal[p.entry];
        this.truncated = new JournalTruncated(entry?.kind === "request" ? entry.request : null, why);
        this.journal = this.journal.slice(0, p.entry);
        this.cuts++;
        this.trappedAt = null;
      } else {
        this.trappedAt = p.entry;
      }
    } else {
      this.trappedAt = null;
    }
    this.restart(new Cancelled(`the engine failed: ${why}`)).catch(() => {});
  }

  /** Cancel whatever is pending: terminate the Worker, reject the pending
   * requests, start a new Worker and replay the journal into it. Resolves
   * once the replay has run; a cancel during a replay starts it again. */
  async cancel(why = "cancelled"): Promise<void> {
    await this.restart(new Cancelled(why));
  }

  private restart(error: Error): Promise<void> {
    if (this.closed) return Promise.resolve();
    this.worker?.terminate();
    this.restarts++;
    this.dropPending(error);
    return this.relaunch();
  }

  /** Reject every request pending or waiting, with its timer cleared. */
  private dropPending(error: Error): void {
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
  }

  /** Stop for good: the Worker terminated, what is pending rejected, its
   * timers cleared, and every later call refused. */
  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.epoch++;
    this.abandonBoot(new Cancelled("the client is closed"));
    this.worker?.terminate();
    this.worker = null;
    this.dropPending(new Cancelled("the client is closed"));
  }
}

/** A browser Worker running worker-entry.ts. */
export function browserWorker(): WorkerHandle {
  return new Worker(new URL("./worker-entry.ts", import.meta.url), { type: "module" }) as unknown as WorkerHandle;
}
