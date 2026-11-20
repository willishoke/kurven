// The page's requests run through the WebAssembly module under Node.
//
//   node crates/kurven-wasm/requests.mjs <kurven_wasm.wasm> <requests.jsonl> <dir> [--stage NAME BUNDLE]...
//
// Each line of the file is one request as the page sends it to its Worker
// (a line starting with `#` is a comment), run in order through the module's
// session, every reply written under <dir> as `kurven-cli session
// --replies` writes the native session's: NNNN.json and each array as
// NNNN.<name>.bin. scripts/check-web.sh requires the two trees to be
// byte-identical: what the page asks of the engine in steps 4 to 7 -- the
// picker's thumbnails, a surface's controls, a banded truncation dragged, a
// face picked and painted, a bundle opened from a file and saved -- answered
// the same in the browser as natively. Node's standard library only.

import { mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { join, relative } from "node:path";

const [wasmPath, requestsPath, outDir, ...rest] = process.argv.slice(2);
if (!wasmPath || !requestsPath || !outDir) {
  console.error("usage: node requests.mjs <kurven_wasm.wasm> <requests.jsonl> <dir> [--stage NAME BUNDLE]...");
  process.exit(2);
}

const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
const wasm = instance.exports;
const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8", { fatal: true });

function lend(bytes) {
  const ptr = wasm.kv_alloc(bytes.length);
  new Uint8Array(wasm.memory.buffer, ptr, bytes.length).set(bytes);
  return [ptr, bytes.length];
}

function stage(bundle, path, data) {
  const names = [encoder.encode(bundle), encoder.encode(path)].map(lend);
  const [ptr, len] = lend(data);
  const ok = wasm.kv_stage(...names.flat(), ptr, len);
  for (const [p, l] of names) wasm.kv_free(p, l);
  if (!ok) throw new Error(`could not stage ${bundle}/${path}`);
}

function files(dir, root = dir, out = []) {
  for (const name of readdirSync(dir).sort()) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) files(p, root, out);
    else out.push(relative(root, p));
  }
  return out;
}

for (let i = 0; i < rest.length; i += 3) {
  if (rest[i] !== "--stage" || !rest[i + 1] || !rest[i + 2]) throw new Error(`expected --stage NAME BUNDLE, not ${rest.slice(i).join(" ")}`);
  const [name, dir] = [rest[i + 1], rest[i + 2]];
  for (const path of files(dir)) stage(name, path, readFileSync(join(dir, path)));
}

mkdirSync(outDir, { recursive: true });
const lines = readFileSync(requestsPath, "utf8").split("\n");
let count = 0;
lines.forEach((line, k) => {
  if (!line.trim() || line.startsWith("#")) return;
  const [ptr, len] = lend(encoder.encode(line));
  const n = wasm.kv_call(ptr, len);
  wasm.kv_free(ptr, len);
  const text = decoder.decode(new Uint8Array(wasm.memory.buffer, wasm.kv_reply_ptr(), n));
  const number = String(k + 1).padStart(4, "0");
  writeFileSync(join(outDir, `${number}.json`), text);
  JSON.parse(text).arrays.forEach((a, i) => {
    const bytes = new Uint8Array(wasm.memory.buffer, wasm.kv_array_ptr(i), wasm.kv_array_len(i)).slice();
    writeFileSync(join(outDir, `${number}.${a.name.replace(/[^A-Za-z0-9._-]/g, "_")}.bin`), bytes);
  });
  count++;
});
console.log(`${requestsPath}: ${count} request(s) answered`);
