// The pool of engines under Node (src/engine/pool.ts): helpers in
// worker_threads Workers holding the primary's document, no browser.
//
//   node web/app/test/pool.test.mjs <kurven_wasm.wasm> <crates/fixtures/swift/session> <native session run>
//
// The judged strokes baked by a pool of 1, 2 and 4 helpers are the bytes
// the page's engine bakes alone and the bytes the native session wrote for
// the same edit; helpers catch up on the primary's edits between bakes (a
// run of gestures, a drag's moves), a helper that dies mid-bake is restarted
// and its piece handed out again, a cancel ends a bake, a bake no longer
// wanted is stale; and a frame at rest served from the helpers' layouts is
// the frame the primary lays out itself.

import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { Worker } from "node:worker_threads";
import { Cancelled, EngineClient } from "../src/engine/client.ts";
import { EnginePool, poolSize } from "../src/engine/pool.ts";

const [wasmPath, session, native] = process.argv.slice(2);
if (!wasmPath || !session || !native) {
  console.error("usage: node pool.test.mjs <kurven_wasm.wasm> <session fixtures> <native session run>");
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
const wasm = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length);
const client = new EngineClient(nodeWorker, wasm);
const pool = new EnginePool(client, nodeWorker, wasm, 4);
const fresh = (answer) => {
  assert.equal(answer.stale, false);
  assert.equal(answer.reply.ok, true, answer.reply.error);
  return answer;
};
const svgOf = (answer) => Buffer.from(fresh(answer).arrays.svg);
let checks = 0;
const started = Date.now();

// The gate's gamma-truncation script: open, truncate, bake.
const gamma = join(session, "inputs", "gamma.kurven");
await client.stage(bundle(gamma));
fresh(await client.request({ kind: "open", name: gamma }));
fresh(await client.request({ kind: "truncation", truncation: { kind: "uniform", height: 3 } }));
const alone = fresh(await client.request({ kind: "bake", svg: true }));
const nativeSVG = readFileSync(join(native, "gamma-truncation", "drawing.svg"));
assert.deepEqual(Buffer.from(alone.arrays.svg), nativeSVG, "the page's engine alone bakes native's bytes");
const status = client.state.status.bake;
checks += 1;

// Pools of 1, 2 and 4: the same reply, every stroke and the SVG.
for (const size of [1, 2, 4]) {
  pool.resize(size);
  const t = Date.now();
  const together = fresh(await pool.bake(true));
  assert.deepEqual(Buffer.from(together.arrays.svg), nativeSVG, `a pool of ${size}: native's bytes`);
  for (const [name, array] of Object.entries(alone.arrays)) {
    assert.deepEqual(Buffer.from(together.arrays[name].buffer), Buffer.from(array.buffer), `a pool of ${size}: ${name}`);
  }
  assert.equal(together.reply.state.status.bake, status, "the bake's status, as bake says it");
  console.log(`  gamma, a pool of ${size}: ${Date.now() - t} ms`);
  checks += 3;
}
assert.equal(pool.clients.length, 4);

// The helpers catch up: three hundred gestures and a drag's moves later,
// the pool bakes what the page's engine bakes alone.
for (let k = 0; k < 60; k++) {
  fresh(await client.request({ kind: "gesture", gesture: { kind: k % 3 ? "orbit" : "zoom", delta: [3.1, -1.3], factor: 1.01, at: [600, 400] } }));
}
fresh(await client.request({ kind: "drag", control: { slider: "truncation" }, phase: "begin" }));
for (const h of [2.8, 2.5, 2.2]) fresh(await client.request({ kind: "drag", control: { slider: "truncation" }, phase: "move", value: h }));
fresh(await client.request({ kind: "drag", control: { slider: "truncation" }, phase: "end" }));
const moved = svgOf(await client.request({ kind: "bake", svg: true }));
assert.deepEqual(svgOf(await pool.bake(true)), moved, "the helpers caught up on the gestures and the drag");
checks += 1;

// A helper that dies mid-bake is restarted, its journal replayed, and its
// piece handed out again: the same bytes.
fresh(await client.request({ kind: "gesture", gesture: { kind: "orbit", delta: [-20, 4] } }));
const turned = svgOf(await client.request({ kind: "bake", svg: true }));
const victim = pool.clients[1];
const restarts = victim.restarts;
const asked = victim.request.bind(victim);
let killed = null;
victim.request = (request, options) => {
  const answer = asked(request, options);
  // Its Worker terminated while it bakes its first piece.
  if (!killed && request.kind === "bakePiece") {
    // Once the piece is posted to the Worker (the client's pending map,
    // read here from outside), not before, when it would wait for the new one.
    const posted = async () => {
      while (![...victim.pending.values()].some((p) => p.work.request?.kind === "bakePiece")) {
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
      await victim.cancel("killed mid-piece");
    };
    killed = posted();
  }
  return answer;
};
const dying = await pool.bake(true);
victim.request = asked;
await killed;
assert.equal(victim.restarts, restarts + 1);
assert.deepEqual(svgOf(dying), turned, "a helper killed mid-bake: the same bytes");
assert.ok(pool.reissued >= 1, "its piece was handed out again");
checks += 3;

// A cancel ends the bake with Cancelled; the pool bakes again after it.
const cancelled = pool.bake(true);
setTimeout(() => pool.cancel("cancelled on purpose"), 100);
await assert.rejects(cancelled, (error) => error instanceof Cancelled);
assert.deepEqual(svgOf(await pool.bake(true)), turned, "after a cancel");
checks += 2;

// A bake no longer wanted is stale, and the next one is not.
let wanted = true;
const unwanted = pool.bake(false, () => wanted);
setTimeout(() => (wanted = false), 50);
assert.equal((await unwanted).stale, true);
checks += 1;

// The frame at rest: laid out by the helpers, served by the primary, the
// frame the primary lays out itself, key for key and segment for segment.
const draft = fresh(await client.request({ kind: "preview", draft: true }));
const have = { content: draft.reply.content, layers: draft.reply.layers.map((l) => l.key) };
const laid = await pool.layOutRest(have, () => true);
assert.ok(laid && laid.laid.length > 0, "the tested layers, laid out");
const served = fresh(await client.request({ kind: "preview", draft: false, have, laid: laid.laid, fingerprint: laid.fingerprint }, { inputs: laid.inputs }));
const itself = fresh(await client.request({ kind: "preview", draft: false, have }));
const unsaid = ({ state: _, ...rest }) => rest;
assert.deepEqual(unsaid(served.reply), unsaid(itself.reply));
for (const [name, array] of Object.entries(itself.arrays)) {
  assert.deepEqual(Buffer.from(served.arrays[name].buffer), Buffer.from(array.buffer), name);
}
assert.equal(pool.mismatches, 0);
// Given to a primary whose document has moved on: refused.
fresh(await client.request({ kind: "gesture", gesture: { kind: "orbit", delta: [5, 0] } }));
const late = await client.request({ kind: "preview", draft: false, have, laid: laid.laid, fingerprint: laid.fingerprint }, { inputs: laid.inputs });
assert.equal(late.reply.ok, false);
assert.equal(late.reply.another, true, "refused as of another document");
checks += 4;

// A helper that drifts -- its document changed behind the pool's back --
// refuses the shares asked of the page's document as of another; the pool
// brings it back from the primary's journal and hands the shares out again,
// and the bake is the page's engine's bytes.
const drift = (c) => c.request({ kind: "truncation", truncation: { kind: "uniform", height: 1.7 } });
for (const c of pool.clients) fresh(await drift(c));
const mismatched = pool.mismatches;
const reference = svgOf(await client.request({ kind: "bake", svg: true }));
assert.deepEqual(svgOf(await pool.bake(true)), reference, "every helper drifted: brought back, the same bytes");
assert.ok(pool.mismatches >= mismatched + 4, "each refused a share as of another document");
fresh(await drift(pool.clients[2]));
assert.deepEqual(svgOf(await pool.bake(true)), reference, "one helper drifted");
// ...and the frame at rest: a drifted helper's layout is asked again of it,
// brought back, and the frame served is the primary's own.
for (const c of pool.clients) fresh(await drift(c));
const draft2 = fresh(await client.request({ kind: "preview", draft: true }));
const have2 = { content: draft2.reply.content, layers: draft2.reply.layers.map((l) => l.key) };
const laid2 = await pool.layOutRest(have2, () => true);
assert.ok(laid2 && laid2.laid.length > 0, "laid out, by helpers brought back");
const served2 = fresh(await client.request({ kind: "preview", draft: false, have: have2, laid: laid2.laid, fingerprint: laid2.fingerprint }, { inputs: laid2.inputs }));
const itself2 = fresh(await client.request({ kind: "preview", draft: false, have: have2 }));
assert.deepEqual(unsaid(served2.reply), unsaid(itself2.reply));
checks += 5;

// A bake refused in its pieces reports the refusal the serial bake would:
// the first in the plan's order, not the first to come back. Two pieces
// refused on purpose, the later one first.
const plan = fresh(await client.request({ kind: "bakePlan" })).reply.pieces;
assert.ok(plan.length > 8);
const same = (a, b) => JSON.stringify({ ...a, weight: 0 }) === JSON.stringify({ ...b, weight: 0 });
const asks = pool.clients.map((c) => c.request.bind(c));
pool.clients.forEach((c, i) => {
  c.request = async (request, options) => {
    if (request.kind === "bakePiece") {
      for (const [k, wait] of [[3, 300], [7, 0]]) {
        if (same(request.piece, plan[k])) {
          await new Promise((resolve) => setTimeout(resolve, wait));
          return { stale: false, reply: { ok: false, touched: false, error: `refused ${k}`, arrays: [] }, arrays: {} };
        }
      }
    }
    return asks[i](request, options);
  };
});
const refused = await pool.bake(true);
pool.clients.forEach((c, i) => (c.request = asks[i]));
assert.equal(refused.reply.ok, false);
assert.equal(refused.reply.error, "refused 3", "the first refused in the plan's order");
assert.equal(client.state.status.bake, "refused 3", "said as the bake's status");
checks += 3;

// A bake refused before its pieces is refused as the page's engine's is.
fresh(await client.request({ kind: "perspective", on: true }));
const aloneRefused = await client.request({ kind: "bake" });
const pooledRefused = await pool.bake(false);
assert.equal(pooledRefused.reply.ok, false);
assert.equal(pooledRefused.reply.error, aloneRefused.reply.error);
fresh(await client.request({ kind: "perspective", on: false }));
checks += 2;

// Cancelled while the primary puts the pieces together: nothing answered.
const assemble = client.request.bind(client);
client.request = (request, options) => {
  if (request.kind === "bakeAssemble") pool.cancel("cancelled while put together");
  return assemble(request, options);
};
await assert.rejects(pool.bake(true), (error) => error instanceof Cancelled);
client.request = assemble;
checks += 1;

// A layout a helper refuses (its document in doubt): it is brought back
// and asked again, and the frame at rest is laid out all the same.
{
  const victim = pool.clients[0];
  const asked = victim.request.bind(victim);
  let refusedOnce = false;
  victim.request = async (request, options) => {
    if (request.kind === "restLayout" && !refusedOnce) {
      refusedOnce = true;
      return { stale: false, reply: { ok: false, touched: false, error: "no layer shown", arrays: [] }, arrays: {} };
    }
    return asked(request, options);
  };
  fresh(await client.request({ kind: "gesture", gesture: { kind: "orbit", delta: [-3, 1] } }));
  const preview = fresh(await client.request({ kind: "preview", draft: true }));
  const holding = { content: preview.reply.content, layers: preview.reply.layers.map((l) => l.key) };
  const reissued = pool.reissued;
  const again = await pool.layOutRest(holding, () => true);
  victim.request = asked;
  assert.ok(refusedOnce && again && again.laid.length > 0, "laid out after a refusal");
  assert.ok(pool.reissued > reissued, "the refused layout asked again");
  checks += 2;
}

// A resize mid-bake: the helpers closed give back their shares, those left
// take them, and the bake is the same bytes.
{
  pool.resize(4);
  const reference = svgOf(await client.request({ kind: "bake", svg: true }));
  const warm = svgOf(await pool.bake(true));
  assert.deepEqual(warm, reference);
  const shrinking = pool.bake(true);
  await new Promise((resolve) => setTimeout(resolve, 60));
  pool.resize(1);
  assert.deepEqual(svgOf(await shrinking), reference, "resized to one mid-bake");
  assert.equal(pool.clients.length, 1);
  pool.resize(4);
  checks += 2;
}

// Export's cancel abandons the export alone: a frame at rest waiting
// behind it is laid out.
{
  const preview = fresh(await client.request({ kind: "preview", draft: true }));
  const holding = { content: preview.reply.content, layers: preview.reply.layers.map((l) => l.key) };
  const exporting = pool.bake(true);
  const resting = pool.layOutRest(holding, () => true);
  setTimeout(() => pool.cancel("cancel the export"), 30);
  await assert.rejects(exporting, (error) => error instanceof Cancelled);
  const laidOut = await resting;
  assert.ok(laidOut && laidOut.laid.length > 0, "the frame at rest went on");
  checks += 2;
}

// An export does not wait for a frame at rest no longer wanted: its shares
// in flight are cancelled and the export starts.
{
  const preview = fresh(await client.request({ kind: "preview", draft: true }));
  const holding = { content: preview.reply.content, layers: preview.reply.layers.map((l) => l.key) };
  // Each layout held back a second, so the frame at rest is in flight when
  // the export is asked for.
  const asks = pool.clients.map((c) => c.request.bind(c));
  pool.clients.forEach((c, i) => {
    c.request = async (request, options) => {
      if (request.kind === "restLayout") await new Promise((resolve) => setTimeout(resolve, 1000));
      return asks[i](request, options);
    };
  });
  let wantedRest = true;
  const resting = pool.layOutRest(holding, () => wantedRest);
  await new Promise((resolve) => setTimeout(resolve, 100));
  wantedRest = false;
  const reference = svgOf(await client.request({ kind: "bake", svg: true }));
  const asked = Date.now();
  const exported = svgOf(await pool.bake(true));
  const took = Date.now() - asked;
  pool.clients.forEach((c, i) => (c.request = asks[i]));
  assert.deepEqual(exported, reference);
  const rest = await resting.catch((error) => (error instanceof Cancelled ? "cancelled" : error));
  assert.ok(rest === null || rest === "cancelled", `the overtaken frame at rest: ${rest}`);
  console.log(`  an export behind a frame at rest no longer wanted: ${took} ms`);
  checks += 2;
}

// A plan of one piece -- a surface -- is baked by the primary, not shared.
{
  const torus = join(session, "inputs", "torus.kurven");
  await client.stage(bundle(torus));
  fresh(await client.request({ kind: "open", name: torus }));
  const alone = fresh(await client.request({ kind: "bake", svg: true }));
  const answer = fresh(await pool.bake(true));
  assert.equal(answer.reply.baked, true, "baked by the primary");
  assert.deepEqual(Buffer.from(answer.arrays.svg), Buffer.from(alone.arrays.svg));
  fresh(await client.request({ kind: "open", name: gamma }));
  checks += 2;
}

// Phase 8, steps 5 to 7: the scenes the page can now make, pooled. Each is
// baked by the pool and by the page's engine alone, to the same bytes (and
// the page's engine's are native's: check-web's requests step); a bundle
// opened from a file is staged under a name of its own and let go of after,
// as main.ts does, so the helpers follow the journal's unstage rules.
{
  let opened = 0;
  const open = async (name) => {
    const staged = { ...bundle(join(session, "inputs", `${name}.kurven`)), name: `opened/${++opened}/${name}.kurven` };
    await client.stage(staged);
    fresh(await client.request({ kind: "open", name: staged.name, untrusted: true }));
    fresh(await client.request({ kind: "unstage", name: staged.name }));
  };
  const scenes = [
    [
      "a torus, its radius, lines and linear curve",
      async () => {
        fresh(await client.request({ kind: "createSurface", preset: "torus" }));
        fresh(await client.request({ kind: "surfaceField", field: "major", value: 2.6 }));
        fresh(await client.request({ kind: "surfaceLineCount", axis: "around", count: 24 }));
        fresh(await client.request({ kind: "linearCurve", on: true }));
      },
    ],
    [
      "sn conformal, with its coordinate curves",
      async () => {
        fresh(await client.request({ kind: "createSurface", preset: "sn" }));
        fresh(await client.request({ kind: "surfaceLines", on: true }));
        fresh(await client.request({ kind: "surfaceField", field: "conformal", value: 1 }));
      },
    ],
    [
      "the forced torus, its components swapped",
      async () => {
        fresh(await client.request({ kind: "createSurface", preset: "forced" }));
        fresh(await client.request({ kind: "component", field: "radial", value: 1 }));
      },
    ],
    [
      "a torus painted with sn, a curve added",
      async () => {
        fresh(await client.request({ kind: "createSurface", preset: "torus" }));
        fresh(await client.request({ kind: "pick", point: [2, 0, 1] }));
        fresh(await client.request({ kind: "function", expression: "sn(z, 0.64)" }));
        fresh(await client.request({ kind: "add", name: "surface_u" }));
      },
    ],
    [
      "sn in relief, opened from a file",
      async () => {
        await open("relief");
        fresh(await client.request({ kind: "surfaceField", field: "relief", value: 0.3 }));
      },
    ],
    [
      "the picture frame, opened from a file, its strokes and a face hatched",
      async () => {
        await open("frame");
        fresh(await client.request({ kind: "strokes", layer: "hatching", count: 30 }));
        fresh(await client.request({ kind: "pick", point: [-1.5, 0, 1] }));
        fresh(await client.request({ kind: "add", name: "hatching" }));
      },
    ],
    [
      "gamma opened from a file, banded, its band dragged, its graph painted",
      async () => {
        await open("gamma");
        const bands = (h) => ({ kind: "realBands", bands: [{ below: 0, height: h }, { below: 1.5, height: 2 }], beyond: 4 });
        fresh(await client.request({ kind: "truncation", truncation: bands(3) }));
        fresh(await client.request({ kind: "drag", control: { slider: "truncation" }, phase: "begin" }));
        for (const h of [3.2, 3.5]) {
          fresh(await client.request({ kind: "drag", control: { slider: "truncation" }, phase: "move", truncation: bands(h) }));
        }
        fresh(await client.request({ kind: "drag", control: { slider: "truncation" }, phase: "end" }));
        fresh(await client.request({ kind: "pick", point: [2.5, 0, 1.3293403881791355] }));
        fresh(await client.request({ kind: "add", name: "graph_v" }));
      },
    ],
  ];
  for (const [what, make] of scenes) {
    await make();
    const t = Date.now();
    const alone = svgOf(await client.request({ kind: "bake", svg: true }));
    const together = svgOf(await pool.bake(true));
    assert.deepEqual(together, alone, `${what}: pooled, the bake alone`);
    console.log(`  ${what}: pooled as alone, ${Date.now() - t} ms`);
    checks += 1;
  }
  // Of the bundles opened from files, the journal holds the bytes of the
  // one the document is, alone (those staged by name and never let go of
  // stay, as the engine keeps them).
  const fromFiles = client.journal.filter((e) => e.kind === "stage" && e.bundle.name.startsWith("opened/"));
  assert.deepEqual(fromFiles.map((e) => e.bundle.name), [`opened/${opened}/gamma.kurven`]);
  checks += 1;
  fresh(await client.request({ kind: "open", name: gamma }));
}

// The log is the pool's: trimmed with the journal, and kept by no client
// that has no pool, a helper's own among them.
for (let k = 0; k < 120; k++) fresh(await client.request({ kind: "viewport", width: 900 + k, height: 700 }));
for (let k = 0; k < 40; k++) fresh(await client.request({ kind: "drag", control: { slider: "truncation" }, phase: k ? "move" : "begin", value: 2 + k / 100 }));
assert.ok(client.log.length <= 2 * client.journalLength + 16, `the log trimmed: ${client.log.length} for a journal of ${client.journalLength}`);
for (const c of pool.clients) assert.equal(c.log.length, 0, "a helper keeps no log");
const lone = new EngineClient(nodeWorker, wasm);
await lone.stage(bundle(gamma));
fresh(await lone.request({ kind: "open", name: gamma }));
for (let k = 0; k < 20; k++) fresh(await lone.request({ kind: "gesture", gesture: { kind: "orbit", delta: [2, 1] } }));
assert.equal(lone.log.length, 0, "no pool, no log");
lone.close();
assert.deepEqual([poolSize(undefined), poolSize(1), poolSize(2), poolSize(4), poolSize(18)], [0, 0, 1, 3, 8]);
checks += 4;

// What each engine holds: the module's linear memory after gamma.
const memory = [client.memory, ...pool.clients.map((c) => c.memory)];
console.log(`  linear memory per engine: ${memory.map((m) => (m / 2 ** 20).toFixed(0)).join(", ")} MiB`);

// A pool that keeps failing is switched off for the session, and said so;
// its callers bake on the primary (main.ts's `baked`, the view's `making`).
{
  const warnings = [];
  const warn = console.warn;
  console.warn = (...args) => warnings.push(args.join(" "));
  const broken = (c) => {
    c.request = (request, options) =>
      request.kind === "bakePiece" ? Promise.reject(new Error("a helper that will not bake")) : EngineClient.prototype.request.call(c, request, options);
  };
  for (let n = 1; n <= 3; n++) {
    for (const c of pool.clients) broken(c);
    await assert.rejects(pool.bake(true), /will not bake/);
    assert.equal(pool.disabled, n === 3, `after ${n} failures`);
  }
  console.warn = warn;
  assert.ok(warnings.some((w) => w.includes("switched off")), "said on the console");
  assert.equal(pool.clients.length, 0, "its helpers closed");
  checks += 4;
}

pool.close();
client.close();
console.log(`the pool of engines under Node: ${checks} checks in ${Date.now() - started} ms`);
