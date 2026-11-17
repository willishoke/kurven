// The Worker protocol end to end under Node: the page's client
// (src/engine/client.ts) driving the wasm module in a worker_threads Worker
// (src/engine/worker.ts), no browser.
//
//   node web/app/test/worker.test.mjs <kurven_wasm.wasm> <crates/fixtures/swift/session> <native session run>
//
// It opens a landscape, edits it, saves and bakes it, and requires the
// manifest Swift's app saved for the same edit and the drawing the native
// session wrote, byte for byte; checks that a view asked for again
// supersedes the older ask; and cancels an edit that hangs, a Bessel order of
// 1e12 painted on a torus, by its timeout, after which the restarted Worker
// holds the document as it was before the edit and answers again.

import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { Worker } from "node:worker_threads";
import { Cancelled, EngineClient, JournalTruncated } from "../src/engine/client.ts";

const [wasmPath, session, native] = process.argv.slice(2);
if (!wasmPath || !session || !native) {
  console.error("usage: node worker.test.mjs <kurven_wasm.wasm> <session fixtures> <native session run>");
  process.exit(2);
}

function nodeWorker() {
  const worker = new Worker(new URL("./node-worker.mjs", import.meta.url));
  const handle = {
    postMessage: (message, transfer) => worker.postMessage(message, transfer),
    onmessage: null,
    terminate: () => void worker.terminate(),
  };
  worker.on("message", (data) => handle.onmessage?.({ data }));
  return handle;
}

function bundle(path) {
  const files = [];
  const walk = (dir) => {
    for (const name of readdirSync(dir)) {
      const p = join(dir, name);
      if (statSync(p).isDirectory()) walk(p);
      else {
        const bytes = readFileSync(p);
        files.push({ path: relative(path, p), bytes: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length) });
      }
    }
  };
  walk(path);
  return { name: path, files };
}

const bytes = readFileSync(wasmPath);
const client = new EngineClient(nodeWorker, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
const fresh = (answer) => {
  assert.equal(answer.stale, false);
  assert.equal(answer.reply.ok, true, answer.reply.error);
  return answer;
};
let checks = 0;

// A landscape opened, truncated, saved and baked: the gate's
// gamma-truncation script, through the Worker.
const gamma = join(session, "inputs", "gamma.kurven");
await client.stage(bundle(gamma));
fresh(await client.request({ kind: "open", name: gamma }));
assert.equal(client.state.kind, "landscape");
assert.equal(client.state.truncation.height, 5);
fresh(await client.request({ kind: "truncation", truncation: { kind: "uniform", height: 3 } }));
assert.deepEqual(client.state.truncation, { kind: "uniform", height: 3 });
assert.equal(client.state.undo.name, "Truncation");
checks += 4;

const saved = fresh(await client.request({ kind: "save", name: "saved.kurven" }));
const manifest = saved.arrays["file:manifest.json"];
assert.ok(manifest instanceof Uint8Array);
assert.deepEqual(
  Buffer.from(manifest),
  readFileSync(join(session, "gamma-truncation", "saved.kurven", "manifest.json")),
  "the manifest Swift's app saved",
);
const baked = fresh(await client.request({ kind: "bake", svg: true }, { view: "bake" }));
assert.deepEqual(Buffer.from(baked.arrays.svg), readFileSync(join(native, "gamma-truncation", "drawing.svg")));
const points = baked.arrays["stroke.0.points"];
assert.ok(points instanceof Float64Array && points.length > 0, "strokes come back as typed arrays");
checks += 3;

// Undo, as ⌘Z: the bundle's own truncation back.
fresh(await client.request({ kind: "undo" }));
assert.deepEqual(client.state.truncation, { kind: "uniform", height: 5 });
checks += 1;

// A view asked for three times at once: one in flight, the second replaced
// by the third while it waits, and the first stale when it lands, since a
// newer ask came meanwhile. Only the newest answer is delivered.
fresh(await client.request({ kind: "gesture", gesture: { kind: "orbit", delta: [40, 10] } }));
const frames = await Promise.all([1, 2, 3].map(() => client.request({ kind: "frame", draft: false }, { view: "frame" })));
assert.deepEqual(frames.map((f) => f.stale), [true, true, false]);
assert.ok(frames[2].reply.layers.length > 0 && frames[2].arrays["layer.0.vertices"] instanceof Float32Array);
checks += 2;

// An edit that hangs, cancelled by its timeout: the Worker is terminated
// and restarted, and the journal puts the document back.
const torus = join(session, "inputs", "torus.kurven");
await client.stage(bundle(torus));
fresh(await client.request({ kind: "open", name: torus }));
const before = client.state;
assert.equal(before.kind, "surface");
const started = Date.now();
await assert.rejects(
  client.request({ kind: "function", expression: "besseli(1e12, z)" }, { timeoutMs: 1500 }),
  (error) => error instanceof Cancelled,
);
assert.equal(client.restarts, 1);
const after = fresh(await client.request({ kind: "state" }));
// What a save or a bake said is a status, which no journal replays.
const unsaid = ({ status: _, ...rest }) => rest;
assert.deepEqual(unsaid(after.reply.state), unsaid(before), "the restarted Worker holds the document as it was");
fresh(await client.request({ kind: "set", key: "minor", value: 0.7 }));
assert.equal(client.state.surface.shape.minor, 0.7);
checks += 5;

// A cancel during a replay: the second terminates the Worker the first is
// replaying into and replays the whole journal again, nothing lost, and
// neither cancel's promise rejects (an unhandled rejection ends this run).
const journaled = client.journalLength;
const settled = client.state;
await Promise.all([client.cancel("first"), client.cancel("second during the first's replay")]);
const second = client.cancel("third");
await new Promise((resolve) => setTimeout(resolve, 40));
await Promise.all([second, client.cancel("fourth, later in a replay")]);
assert.equal(client.restarts, 5);
assert.equal(client.journalLength, journaled, "the journal is whole after the replays");
assert.deepEqual(unsaid(fresh(await client.request({ kind: "state" })).reply.state), unsaid(settled));
checks += 3;

// A failed open changes the document (it forgets its edits and is a
// failure), so it is journaled and replayed; an open of a name never staged
// touches nothing and is not.
await client.stage({ name: "broken.kurven", files: [{ path: "manifest.json", bytes: new TextEncoder().encode("nope").buffer }] });
let answer = await client.request({ kind: "open", name: "broken.kurven" });
assert.equal(answer.reply.ok, false);
assert.equal(answer.reply.touched, true);
assert.equal(client.state.kind, "failed");
const afterFailure = client.journalLength;
answer = await client.request({ kind: "open", name: "never-staged.kurven" });
assert.equal(answer.reply.touched, false);
assert.equal(client.journalLength, afterFailure);
await client.cancel("replay a failure");
assert.equal(fresh(await client.request({ kind: "state" })).reply.state.kind, "failed");
checks += 5;

// A failed open keeps what the document held: Swift's Document keeps the
// landscape's request (its controls still show it; the edits after find no
// bundle to act on). So it is no clean slate: the replay keeps the open
// before it, or the request it kept would be gone.
fresh(await client.request({ kind: "open", name: gamma }));
assert.equal((await client.request({ kind: "open", name: "broken.kurven" })).reply.touched, true);
fresh(await client.request({ kind: "set", key: "rMax", value: 5 }));
fresh(await client.request({ kind: "drag", control: { slider: "landscape", label: "rMax" }, phase: "begin" }));
fresh(await client.request({ kind: "drag", control: { slider: "landscape", label: "rMax" }, phase: "end" }));
const kept = client.state;
assert.equal(kept.kind, "failed");
assert.equal(kept.landscape.expression, "gamma(z)", "the request the failed open kept");
await client.cancel("replay past a failed open");
assert.deepEqual(unsaid(fresh(await client.request({ kind: "state" })).reply.state), unsaid(kept));
checks += 2;

// A request and a cancel in the same tick: the request waits for the new
// Worker's replay rather than going to it before it has loaded.
const sameTick = client.request({ kind: "state" });
const cancelled = client.cancel("in the same tick as a request");
await cancelled;
assert.deepEqual(unsaid(fresh(await sameTick).reply.state), unsaid(kept));
checks += 1;

// A long session replays bounded: from its last open, a run of gestures as
// one camera, a drag's moves as its last, a run of viewports as its last.
fresh(await client.request({ kind: "open", name: gamma }));
const opened = client.journalLength;
for (let k = 0; k < 300; k++) {
  fresh(await client.request({ kind: "gesture", gesture: { kind: k % 3 ? "orbit" : "zoom", delta: [3.1, -1.3], factor: 1.01, at: [600, 400] } }));
}
fresh(await client.request({ kind: "drag", control: { slider: "truncation" }, phase: "begin" }));
for (const h of [4.8, 4.5, 4.2, 3.9, 3.6]) {
  fresh(await client.request({ kind: "drag", control: { slider: "truncation" }, phase: "move", value: h }));
}
fresh(await client.request({ kind: "drag", control: { slider: "truncation" }, phase: "end" }));
for (let w = 0; w < 20; w++) fresh(await client.request({ kind: "viewport", width: 1000 + w, height: 700 }));
assert.equal(client.journalLength, opened + 5, "camera, drag begin, its last move, drag end, viewport");
const long = client.state;
await client.cancel("replay a long session");
const replayed = fresh(await client.request({ kind: "state" })).reply.state;
assert.deepEqual(unsaid(replayed), unsaid(long), "the bounded replay gives the same document");
assert.deepEqual(replayed.camera.exact, long.camera.exact);
checks += 3;

// A preview's keys name the engine instance: a restarted Worker counts its
// scenes from one again, so a key the page held from the last could name a
// scene of this one; each engine's nonce (module.ts, `seed`) starts its keys,
// and none of the new Worker's is one the old gave, for the same scene or a
// new one.
const viewKeys = (answer) => [answer.reply.content, ...answer.reply.layers.map((l) => l.key)];
const beforeRestart = viewKeys(fresh(await client.request({ kind: "preview", draft: true })));
await client.cancel("a new engine for the preview's keys");
const sameScene = viewKeys(fresh(await client.request({ kind: "preview", draft: true })));
fresh(await client.request({ kind: "open", name: torus }));
const newScene = viewKeys(fresh(await client.request({ kind: "preview", draft: true })));
for (const k of [...sameScene, ...newScene]) assert.ok(!beforeRestart.includes(k), `${k} was the old engine's`);
assert.notEqual(sameScene[0].split("~")[0], beforeRestart[0].split("~")[0], "a new nonce");
fresh(await client.request({ kind: "open", name: gamma }));
checks += 2;

// Closed: what is pending is rejected, its timeout disarmed, and nothing
// after is taken. The pending edit here would hang, and its timeout would
// restart the Worker if it were still armed.
const restarts = client.restarts;
const hanging = client.request({ kind: "expression", expression: "besseli(1e12, z)" }, { timeoutMs: 300 });
await new Promise((resolve) => setTimeout(resolve, 20));
client.close();
await assert.rejects(hanging, (error) => error instanceof Cancelled);
await new Promise((resolve) => setTimeout(resolve, 500));
assert.equal(client.restarts, restarts, "no timeout fired after close");
await assert.rejects(client.request({ kind: "state" }), (error) => error instanceof Cancelled);
await assert.rejects(client.stage(bundle(gamma)), (error) => error instanceof Cancelled);
checks += 4;

// The client's failures, against a Worker that fails on purpose: an engine
// that will not load is said to the request waiting for it, and the next
// request loads it again; a bundle the engine refuses rejects its stage and
// is not journaled.
function fakeWorker(script) {
  const handle = {
    onmessage: null,
    terminated: false,
    postMessage(message) {
      queueMicrotask(() => {
        if (handle.terminated) return;
        const send = (data) => handle.onmessage?.({ data });
        if (message.type === "init") {
          send(script.boots++ < script.failBoots ? { type: "failed", id: null, trap: false, message: "no engine" } : { type: "ready" });
        } else if (message.type === "stage") {
          send(message.bundle.name === script.refuse ? { type: "failed", id: message.id, trap: false, message: "refused" } : { type: "staged", id: message.id });
        } else if (message.type === "request") {
          handle.requests = (handle.requests ?? 0) + 1;
          script.onRequest?.(handle);
          if (handle.terminated) return;
          if (script.trap?.(message)) {
            send({ type: "failed", id: message.id, trap: true, message: "unreachable" });
            return;
          }
          send({ type: "reply", id: message.id, reply: { ok: true, touched: true, arrays: [] }, arrays: {} });
        }
      });
    },
    terminate() {
      handle.terminated = true;
    },
  };
  return handle;
}
const script = { boots: 0, failBoots: 1, refuse: "bad.kurven" };
const fake = new EngineClient(() => fakeWorker(script), new ArrayBuffer(8));
await assert.rejects(fake.request({ kind: "state" }), /would not load/);
assert.equal(fresh(await fake.request({ kind: "set", key: "major", value: 2 })).stale, false);
assert.equal(script.boots, 2, "loaded again for the next request");
await assert.rejects(fake.stage({ name: "bad.kurven", files: [] }), /refused/);
await fake.stage({ name: "good.kurven", files: [] });
assert.equal(fake.journalLength, 2, "the edit and the good stage, not the refused one");
checks += 5;

// A cancel exactly in the middle of a replay, on the replay's second
// request: the next Worker gets the whole journal, not what was left of it.
let workers = [];
const midway = { boots: 0, failBoots: 0 };
const counted = new EngineClient(() => {
  const w = fakeWorker(midway);
  workers.push(w);
  return w;
}, new ArrayBuffer(8));
for (const value of [1, 2, 3]) await counted.request({ kind: "set", key: "major", value });
await counted.request({ kind: "relief", value: 0.3 });
assert.equal(counted.journalLength, 4);
let interrupted = null;
midway.onRequest = (w) => {
  if (w === workers[1] && w.requests === 2 && !interrupted) interrupted = counted.cancel("in the middle of a replay");
};
await counted.cancel("start a replay");
await interrupted;
assert.ok(interrupted, "the second cancel came during the first replay");
assert.equal(workers[1].requests, 2, "the first replay stopped at its second request");
assert.equal(workers[2].requests, 4, "the next replay sent the whole journal");
assert.equal(counted.journalLength, 4);
counted.close();
fake.close();
checks += 4;

// An entry that traps the engine on every replay: after two traps on it,
// it and what follows are cut, the next call is told, and the document is
// the one up to it.
workers = [];
const poisoned = { boots: 0, failBoots: 0 };
const trapping = new EngineClient(() => {
  const w = fakeWorker(poisoned);
  workers.push(w);
  return w;
}, new ArrayBuffer(8));
for (const value of [1, 2]) await trapping.request({ kind: "set", key: "major", value });
await trapping.request({ kind: "relief", value: 0.3 });
await trapping.request({ kind: "set", key: "major", value: 3 });
assert.equal(trapping.journalLength, 4);
poisoned.trap = (message) => message.request.kind === "relief";
await trapping.cancel("replay into a trap");
assert.equal(workers.length, 4, "the first Worker, the cancel's, and one restart for each trap");
assert.equal(trapping.journalLength, 2, "cut at the entry that trapped twice");
await assert.rejects(trapping.request({ kind: "state" }), (error) => error instanceof JournalTruncated && error.request.kind === "relief");
assert.equal(fresh(await trapping.request({ kind: "state" })).stale, false, "told once, then answered");
trapping.close();
checks += 4;

console.log(`the Worker protocol under Node: ${checks} checks, one hang cancelled in ${Date.now() - started} ms`);
