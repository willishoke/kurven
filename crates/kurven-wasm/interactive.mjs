// Interactive timings in the wasm module under Node: the requests
// crates/kurven-session/tests/interactive.rs sends natively, through the
// same protocol, for cn and gamma at the app's default resolution.
//
//   node crates/kurven-wasm/interactive.mjs <kurven_wasm.wasm>

import { readFileSync } from "node:fs";

const [wasmPath] = process.argv.slice(2);
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const wasm = instance.exports;
const encoder = new TextEncoder();
const decoder = new TextDecoder();

function call(request) {
  const bytes = encoder.encode(JSON.stringify(request));
  const ptr = wasm.kv_alloc(bytes.length);
  new Uint8Array(wasm.memory.buffer, ptr, bytes.length).set(bytes);
  const n = wasm.kv_call(ptr, bytes.length);
  wasm.kv_free(ptr, bytes.length);
  const reply = JSON.parse(decoder.decode(new Uint8Array(wasm.memory.buffer, wasm.kv_reply_ptr(), n)));
  // The arrays, copied out as a Worker would before transferring them.
  for (let i = 0; i < wasm.kv_array_count(); i++) {
    const p = wasm.kv_array_ptr(i);
    wasm.memory.buffer.slice(p, p + wasm.kv_array_len(i));
  }
  if (!reply.ok) throw new Error(`${request.kind}: ${reply.error}`);
  return reply;
}

const drag = (phase, value) => ({ kind: "drag", control: { slider: "truncation" }, phase, value });
for (const [preset, lowered] of [["cn", 3.2], ["gamma", 4.0]]) {
  const script = [
    ["open: sample at 600, place the contours on f", { kind: "create", preset }],
    [null, { kind: "gesture", gesture: { kind: "orbit", delta: [30, 0] } }],
    ["camera change: draft frame (folds on the grid)", { kind: "frame", draft: true }],
    ["camera change: rest frame (the window's layout)", { kind: "frame", draft: false }],
    ["camera change: judged strokes (the bake)", { kind: "bake" }],
    [null, drag("begin")],
    ["truncation drag: a step (restyle from the grids)", drag("move", lowered)],
    ["truncation drag: draft frame", { kind: "frame", draft: true }],
    [null, drag("end")],
    ["truncation drag: rest frame (the window's layout)", { kind: "frame", draft: false }],
    ["truncation drag: judged strokes (the bake)", { kind: "bake" }],
  ];
  for (const [name, request] of script) {
    const started = performance.now();
    call(request);
    const ms = performance.now() - started;
    if (name) console.log(`${preset}\t${name}\t${ms.toFixed(1)} ms`);
  }
}
