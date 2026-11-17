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
import { Cancelled, EngineClient } from "../src/engine/client.ts";

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

client.close();
console.log(`the Worker protocol under Node: ${checks} checks, one hang cancelled in ${Date.now() - started} ms`);
