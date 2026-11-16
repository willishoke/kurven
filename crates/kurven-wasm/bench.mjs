// Times the gallery plates baked by the WebAssembly module under Node.
//
//   node crates/kurven-wasm/bench.mjs <kurven_wasm.wasm>
//
// Each plate is built from its request, edited as docs/gallery/make.py edits
// it and baked in memory (`bake_gallery_plate`), once; the time is the
// call's wall clock. Not part of the check: a measure for crates/FINDINGS.md.

import { readFileSync } from "node:fs";

const [wasmPath] = process.argv.slice(2);
if (!wasmPath) {
  console.error("usage: node bench.mjs <kurven_wasm.wasm>");
  process.exit(2);
}
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
for (const [which, name] of [[0, "gallery-gamma"], [1, "gallery-cn"]]) {
  const started = performance.now();
  const strokes = instance.exports.bake_gallery_plate(which);
  const seconds = (performance.now() - started) / 1000;
  console.log(`${name}: ${strokes} strokes in ${seconds.toFixed(2)} s`);
}
