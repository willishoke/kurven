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
import { EnginePool } from "../src/engine/pool.ts";

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
checks += 3;

// What each engine holds: the module's linear memory after gamma.
const memory = [client.memory, ...pool.clients.map((c) => c.memory)];
console.log(`  linear memory per engine: ${memory.map((m) => (m / 2 ** 20).toFixed(0)).join(", ")} MiB`);

pool.close();
client.close();
console.log(`the pool of engines under Node: ${checks} checks in ${Date.now() - started} ms`);
