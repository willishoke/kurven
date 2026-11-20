// A pool of engines holding the page's document. The primary is the page's
// own engine (client.ts): it takes every edit and draws every draft, so the
// hand never waits behind slow work. Helpers are engines of their own, each
// in a Worker of its own, holding the same document by replaying the
// primary's journal and catching up on its log (client.ts's header), as a
// restarted Worker does. The slow work is shared among the helpers along the
// seams kurven_session's pool.rs names, and put back together by the
// primary in the order Swift's threads put theirs, so what comes out is the
// same bytes whatever the pool's size and whichever helper finished first:
//
// - The judged strokes, what Export writes: the primary plans the bake
//   (`bakePlan`), the helpers bake its pieces (`bakePiece`) as each comes
//   free, the largest first, and the primary puts them together
//   (`bakeAssemble`), which answers as `bake` does.
// - The frame at rest: the primary names the tested layers a frame at rest
//   would lay out (`restLayers`), the helpers lay them out (`restLayout`),
//   and the primary serves the frame from them under its own keys
//   (`preview` given `laid`), refusing layouts of another document.
//
// One job runs at a time. A job asked whether it is still wanted
// (`current`) stops handing out shares once it is not, and lets the shares
// in flight finish; a cancel (`cancel`) restarts the helpers in flight, as
// the client cancels a request. A helper whose Worker dies is restarted by
// its client, which replays its journal, and the share it held is handed
// out again.
//
// Every share names the fingerprint of the document it was asked of, and a
// helper holding another refuses it as such (`another`). A helper that so
// refuses, or fails a share any other way, or whose replay was cut
// (`JournalTruncated`), may hold another document than its place in the log
// says; it is marked, and brought to the job's document from the primary's
// journal (its Worker restarted and the journal replayed) before it is
// asked anything again, the share it was refused handed out again.

import { type Answer, Cancelled, EngineClient, type Snapshot, type WorkerHandle } from "./client.ts";
import type { Laid, Piece, Request, TypedArray } from "./protocol.ts";

/** A helper: its client, and how far along the primary's log its document is. */
interface Helper {
  client: EngineClient;
  at: number;
  cuts: number;
  /** A share is in flight on it. */
  busy: boolean;
  /** Its document may not be the one its place in the log says: brought to
   * the next job's from the journal, not the log. */
  dirty: boolean;
}

/** A job of the pool, while it runs. */
interface Job {
  abandoned: boolean;
}

/** Layouts at rest the helpers laid out, for the primary to serve. */
export interface RestLayout {
  laid: Laid[];
  fingerprint: string;
  inputs: Record<string, TypedArray>;
}

/** A reply that answered. */
type Answered = Extract<Answer, { stale: false }>;

/** A share the engine refused: the first in the requests' order, which
 * stops the job. */
class Refused extends Error {
  readonly answer: Answered;
  constructor(answer: Answered) {
    super(answer.reply.error ?? "refused");
    this.answer = answer;
  }
}

/** A size for the pool on this machine: a helper for every core but the
 * page's own, at most eight (cn's bake gains little past eight helpers,
 * whose longest piece, the fold curves, does not split; and each helper
 * holds a copy of the document). */
export function poolSize(cores: number | undefined): number {
  // One core, or none said: no helpers, the page's engine alone.
  if (cores === undefined || !(cores > 1)) return 0;
  return Math.min(8, cores - 1);
}

export class EnginePool {
  readonly primary: EngineClient;
  private readonly spawn: () => WorkerHandle;
  private readonly wasm: ArrayBuffer | string;
  /** How many helpers a job may share its work among. */
  size: number;
  private readonly helpers: Helper[] = [];
  private chain: Promise<unknown> = Promise.resolve();
  /** The jobs running or waiting to. */
  private readonly jobs = new Set<Job>();
  private closed = false;
  /** Shares a helper refused as asked of another document than it held,
   * each followed by bringing it to the job's: none, unless the two came
   * apart. */
  mismatches = 0;
  /** How many times a share was handed out again after its helper died. */
  reissued = 0;

  constructor(primary: EngineClient, spawn: () => WorkerHandle, wasm: ArrayBuffer | string, size: number) {
    this.primary = primary;
    // The helpers catch up on the primary's log, which it keeps only for a pool.
    primary.keepLog();
    this.spawn = spawn;
    this.wasm = wasm;
    this.size = Math.max(1, Math.floor(size));
  }

  /** The helpers' clients, for the lanes that measure or break them. */
  get clients(): EngineClient[] {
    return this.helpers.map((h) => h.client);
  }

  /** Change how many helpers a job shares among; helpers past it are closed. */
  resize(size: number): void {
    this.size = Math.max(1, Math.floor(size));
    for (const h of this.helpers.splice(this.size)) h.client.close();
  }

  /** The judged strokes of the document as the primary holds it when this
   * is answered in its turn, as `bake` answers (with the SVG when `svg`).
   * Stale when `current` says it is no longer wanted; rejects with
   * `Cancelled` when cancelled. */
  bake(svg: boolean, current: () => boolean = () => true): Promise<Answer> {
    return this.run(async (job) => {
      // Answered after every request sent before it, so the document it
      // plans is the one the page has edited so far, and the snapshot taken
      // when it lands is that document.
      const planned = await this.primary.request({ kind: "bakePlan" });
      if (planned.stale || !planned.reply.ok) return planned;
      const snapshot = this.primary.snapshot();
      const fingerprint = planned.reply.fingerprint as string;
      const frame = planned.reply.frame;
      const pieces = planned.reply.pieces as (Piece & { weight: number })[];
      const plan: Piece[] = pieces.map(({ weight: _, ...piece }) => piece as Piece);
      let results: Answered[] | null;
      try {
        results = await this.share(
          job,
          snapshot,
          pieces.map((p) => p.weight),
          (k) => ({ kind: "bakePiece", piece: plan[k]!, fingerprint }),
          current,
        );
      } catch (error) {
        if (!(error instanceof Refused)) throw error;
        // Said by the primary as the bake's status, as `bake` says it.
        return this.primary.request({ kind: "bakeAssemble", failed: error.message });
      }
      if (!results) return { stale: true };
      const inputs: Record<string, TypedArray> = {};
      const colors: string[][] = [];
      results.forEach((r, k) => {
        colors.push(r.reply.colors as string[]);
        for (const [name, array] of Object.entries(r.arrays)) inputs[`${k}.${name.slice("piece.".length)}`] = array;
      });
      if (job.abandoned) throw new Cancelled("the pool's work was cancelled");
      if (!current()) return { stale: true };
      // Put together by the primary as of the plan's document: if it has
      // moved on since, the strokes are still the plan's, its frame theirs,
      // and the reply says `snapshot`.
      const assembled = await this.primary.request(
        { kind: "bakeAssemble", svg, plan, colors, fingerprint, frame },
        { inputs },
      );
      // Cancelled while it was put together: nothing is answered.
      if (job.abandoned) throw new Cancelled("the pool's work was cancelled");
      return assembled;
    });
  }

  /** The tested layers of a frame at rest, laid out by the helpers, for the
   * primary to serve holding `have`; null when no longer wanted, or when a
   * helper's layout is not of the primary's document. */
  layOutRest(have: { content?: string; layers: string[] }, current: () => boolean): Promise<RestLayout | null> {
    return this.run(async (job) => {
      const listed = await this.primary.request({ kind: "restLayers", have });
      if (listed.stale || !listed.reply.ok || !current()) return null;
      const layers = listed.reply.layers as number[];
      const weights = listed.reply.weights as number[];
      const fingerprint = listed.reply.fingerprint as string | null;
      if (!fingerprint) return null;
      if (layers.length === 0) return { laid: [], fingerprint, inputs: {} };
      const snapshot = this.primary.snapshot();
      let results: Answered[] | null;
      try {
        results = await this.share(
          job,
          snapshot,
          weights,
          (k) => ({ kind: "restLayout", layers: [layers[k]!], fingerprint }),
          current,
          true,
        );
      } catch (error) {
        if (!(error instanceof Refused)) throw error;
        // A layout the helper's document has no such layer for: the helper
        // was marked, and the next frame at rest brings it back.
        console.warn(`kurven: a frame at rest was refused by a helper (${error.message}); it stays a draft`);
        return null;
      }
      if (!results) return null;
      const laid: Laid[] = [];
      const inputs: Record<string, TypedArray> = {};
      for (const r of results) {
        laid.push(...(r.reply.laid as Laid[]));
        Object.assign(inputs, r.arrays);
      }
      return { laid, fingerprint, inputs };
    });
  }

  /** Stop the job running and every job waiting: no share more is handed
   * out, and the helpers with one in flight are restarted (their documents
   * replayed, as a cancelled request's are), so each ends now with
   * `Cancelled`. */
  cancel(why = "cancelled"): void {
    if (this.jobs.size === 0) return;
    for (const job of this.jobs) job.abandoned = true;
    for (const h of this.helpers) if (h.busy) void h.client.cancel(why).catch(() => {});
  }

  /** Close every helper; the primary is the page's to close. */
  close(): void {
    this.closed = true;
    this.cancel("the pool is closed");
    for (const h of this.helpers.splice(0)) h.client.close();
  }

  /** Wait until no job runs. */
  async settled(): Promise<void> {
    for (;;) {
      const chain = this.chain;
      await chain.catch(() => {});
      if (chain === this.chain) return;
    }
  }

  // MARK: - jobs

  /** Run `work` once every job before it has ended, as the job running. */
  private run<T>(work: (job: Job) => Promise<T>): Promise<T> {
    const job: Job = { abandoned: false };
    this.jobs.add(job);
    const running = this.chain.catch(() => {}).then(async () => {
      try {
        if (this.closed) throw new Cancelled("the pool is closed");
        if (job.abandoned) throw new Cancelled("the pool's work was cancelled");
        return await work(job);
      } finally {
        this.jobs.delete(job);
      }
    });
    this.chain = running.catch(() => {});
    return running;
  }

  /** Share `count = weights.length` requests among the helpers, each brought
   * to the document of `snapshot` first, the heaviest handed out first; the
   * answers in the requests' order. Null when `current` stopped it. A
   * refusal throws `Refused` with the first in the requests' order;
   * `marksRefused`: the helper that refused may hold another document. */
  private async share(
    job: Job,
    snapshot: Snapshot,
    weights: number[],
    request: (k: number) => Request,
    current: () => boolean,
    marksRefused = false,
  ): Promise<Answered[] | null> {
    const queue = weights.map((_, k) => k).sort((a, b) => weights[b]! - weights[a]! || a - b);
    const helpers = this.ready(Math.min(this.size, queue.length), snapshot);
    const results: Answered[] = new Array(weights.length);
    const tries = new Map<number, number>();
    let failure: unknown = null;
    // The first share refused in the requests' order, not in time: the one
    // the serial work would have stopped at. Shares after it are not handed
    // out; those before it still are, since one of them may be refused too.
    let refusedAt = Number.POSITIVE_INFINITY;
    const going = () => !job.abandoned && failure === null && current();
    const again = (k: number, error: unknown): void => {
      const n = (tries.get(k) ?? 0) + 1;
      if (n > 2) {
        failure ??= error;
        return;
      }
      tries.set(k, n);
      this.reissued++;
      queue.unshift(k);
    };
    const work = async (h: Helper): Promise<void> => {
      for (;;) {
        try {
          await this.sync(h, snapshot);
        } catch (error) {
          failure ??= error;
          return;
        }
        let k: number | undefined;
        while (going() && (k = queue.shift()) !== undefined) {
          if (k > refusedAt) continue;
          h.busy = true;
          try {
            const answer = await h.client.request(request(k));
            if (answer.stale) throw new Error("a share of the pool's work went stale");
            if (!answer.reply.ok && answer.reply.another === true) {
              // Asked of another document than it holds: brought to the
              // job's before it is asked again, the share handed out again.
              this.mismatches++;
              h.dirty = true;
              again(k, new Error(answer.reply.error ?? "a share of another document"));
              break;
            }
            results[k] = answer;
            if (!answer.reply.ok) {
              refusedAt = Math.min(refusedAt, k);
              // A layout refused is the helper's document's doing: marked.
              if (marksRefused) h.dirty = true;
            }
          } catch (error) {
            // The helper's Worker died, or was cancelled: its client restarted
            // it and replays its journal, which is the job's document, so the
            // share goes out again. Anything else (its replay cut short, a
            // failure of its own) leaves its document in doubt: marked.
            if (!(error instanceof Cancelled)) h.dirty = true;
            if (job.abandoned) return;
            again(k, error);
            if (h.dirty) break;
          } finally {
            h.busy = false;
          }
        }
        // Out of the loop for a marked helper with shares left: brought
        // back, and on with the queue.
        if (!(h.dirty && going() && queue.some((j) => j <= refusedAt))) return;
      }
    };
    await Promise.all(helpers.map(work));
    // No engine will need the log before the helpers' places again.
    this.primary.forgetLog(Math.min(...this.helpers.map((h) => (h.cuts === snapshot.cuts ? h.at : snapshot.at))));
    if (job.abandoned) throw new Cancelled("the pool's work was cancelled");
    if (failure !== null) throw failure;
    if (refusedAt !== Number.POSITIVE_INFINITY) throw new Refused(results[refusedAt]!);
    if (!current()) return null;
    return results;
  }

  /** `n` helpers, made where there are fewer, each a new engine replaying
   * the snapshot's journal. */
  private ready(n: number, snapshot: Snapshot): Helper[] {
    while (this.helpers.length < n) {
      const client = new EngineClient(this.spawn, this.wasm, snapshot.journal);
      this.helpers.push({ client, at: snapshot.at, cuts: snapshot.cuts, busy: false, dirty: false });
    }
    return this.helpers.slice(0, n);
  }

  /** Bring a helper to the snapshot's document: the log since its place,
   * each entry a later one took the place of left out; or, where the log
   * no longer reaches back so far or is longer than the journal, the
   * snapshot's journal replayed into a new Worker. */
  private async sync(h: Helper, snapshot: Snapshot): Promise<void> {
    if (!h.dirty && h.at === snapshot.at && h.cuts === snapshot.cuts) return;
    const entries = !h.dirty && h.at <= snapshot.at ? this.primary.catchUp(h.at, snapshot.at, h.cuts) : null;
    let adopted = entries === null || entries.length > snapshot.journal.length;
    if (!adopted) {
      try {
        for (const entry of entries!) {
          if (entry.kind === "stage") await h.client.stage(entry.bundle);
          else await h.client.request(entry.request);
        }
      } catch {
        // A Worker that died part way: its journal holds what it answered,
        // not what it did not; start it again from the snapshot's.
        adopted = true;
      }
    }
    if (adopted) await h.client.adopt(snapshot.journal);
    h.at = snapshot.at;
    h.cuts = snapshot.cuts;
    h.dirty = false;
  }
}
