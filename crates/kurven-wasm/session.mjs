// The session's edit scripts run through the WebAssembly module under Node.
//
//   node crates/kurven-wasm/session.mjs <kurven_wasm.wasm> <scripts.txt> <inputs> <dir>
//
// Each line of scripts.txt (crates/fixtures/swift/session) is the app's
// command line; it runs through the module's session (`kv_call` with a
// `headless` request, the run `kurven-cli app` makes natively), its input
// bundle staged from <inputs>, and what it saves and bakes is written under
// <dir>/NAME, as the native run writes it. scripts/check-web.sh then
// requires the two trees to be byte-identical. Node's standard library only.

import { mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";

const [wasmPath, scriptsPath, inputs, outDir] = process.argv.slice(2);
if (!wasmPath || !scriptsPath || !inputs || !outDir) {
  console.error("usage: node session.mjs <kurven_wasm.wasm> <scripts.txt> <inputs> <dir>");
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
  const parts = [encoder.encode(bundle), encoder.encode(path), data].map(lend);
  const ok = wasm.kv_stage(...parts.flat());
  for (const [ptr, len] of parts) wasm.kv_free(ptr, len);
  if (!ok) throw new Error(`could not stage ${bundle}/${path}`);
}

function call(request) {
  const [ptr, len] = lend(encoder.encode(JSON.stringify(request)));
  const n = wasm.kv_call(ptr, len);
  wasm.kv_free(ptr, len);
  // Read the memory afresh: the call may have grown it.
  const reply = JSON.parse(decoder.decode(new Uint8Array(wasm.memory.buffer, wasm.kv_reply_ptr(), n)));
  const arrays = reply.arrays.map((a, i) => ({
    name: a.name,
    bytes: new Uint8Array(wasm.memory.buffer, wasm.kv_array_ptr(i), wasm.kv_array_len(i)).slice(),
  }));
  return { reply, arrays };
}

function files(dir, root = dir, out = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) files(p, root, out);
    else out.push(relative(root, p));
  }
  return out;
}

let runs = 0;
for (const line of readFileSync(scriptsPath, "utf8").split("\n")) {
  const words = line.trim().split(/\s+/);
  if (!words[0] || words[0].startsWith("#")) continue;
  const [name, , , input, ...rest] = words;
  const out = join(outDir, name);
  mkdirSync(out, { recursive: true });
  wasm.kv_unstage();
  const args = [];
  if (input !== "-") {
    const bundle = join(inputs, `${input}.kurven`);
    for (const f of files(bundle)) stage(bundle, f, readFileSync(join(bundle, f)));
    args.push(bundle);
  }
  for (const a of rest) args.push(a.startsWith("@") ? join(out, a.slice(1)) : a);
  const { reply, arrays } = call({ kind: "headless", args });
  if (!reply.ok || reply.exit !== 0) {
    console.error(`${name}: exit ${reply.exit}\n${(reply.stderr ?? [reply.error]).join("\n")}`);
    process.exit(1);
  }
  for (const { name: path, bytes } of arrays) {
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, bytes);
  }
  runs++;
}
console.log(`ran ${runs} edit script(s) through ${wasmPath} under ${outDir}`);
