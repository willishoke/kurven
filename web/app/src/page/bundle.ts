// A `.kurven` bundle as a file (Phase 8, step 7). A bundle is a directory --
// manifest.json and, for a landscape, its grids as .npy -- which a browser can
// neither download nor be handed as one file. So Save writes the bundle's
// files into a ZIP archive under the bundle's directory name
// (`gamma.kurven.zip`, holding `gamma.kurven/manifest.json`, ...: unzipped,
// the bundle Swift's app and kurven-cli open), and Open takes such an archive,
// or a bundle's directory chosen as a folder.
//
// What is opened is someone else's bytes, read within bounds before the
// engine sees them: the archive at most 160 MB, at most 64 entries, each at
// most 64 MB and all of them at most 128 MB unpacked (a 2000-sample
// landscape's two grids are 64 MB), a deflated entry inflated as a stream
// and stopped past its cap; no encryption, no ZIP64, no path that climbs out
// of the bundle. The engine then holds the manifest to the controls' reach
// before anything is built, and opens it in a document of its own, so a
// refused bundle leaves the open one and its undo history as they were
// (kurven_session::files, the protocol's untrusted `open`).

import type { Bundle } from "../engine/protocol.ts";

export const LIMITS = {
  archive: 160 * 1024 * 1024,
  entries: 64,
  entry: 64 * 1024 * 1024,
  total: 128 * 1024 * 1024,
};

/** Why a file could not be read as a bundle. */
export class Unreadable extends Error {}

// MARK: - CRC-32, as ZIP stores it

const CRC = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC[(c ^ bytes[i]!) & 0xff]! ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

// MARK: - writing

/** The bundle's files as a ZIP archive, stored (no compression), each under
 * `dir/`. Dated 1980-01-01, so the same bundle writes the same bytes. */
export function zip(dir: string, files: { path: string; bytes: Uint8Array }[]): Uint8Array {
  const encoder = new TextEncoder();
  const locals: Uint8Array[] = [];
  const centrals: Uint8Array[] = [];
  let offset = 0;
  for (const f of files) {
    const name = encoder.encode(`${dir}/${f.path}`);
    const crc = crc32(f.bytes);
    const local = new Uint8Array(30 + name.length);
    const l = new DataView(local.buffer);
    l.setUint32(0, 0x04034b50, true);
    l.setUint16(4, 20, true); // version needed
    l.setUint16(6, 0x0800, true); // UTF-8 names
    l.setUint16(8, 0, true); // stored
    l.setUint16(10, 0, true); // 00:00
    l.setUint16(12, 0x21, true); // 1980-01-01
    l.setUint32(14, crc, true);
    l.setUint32(18, f.bytes.length, true);
    l.setUint32(22, f.bytes.length, true);
    l.setUint16(26, name.length, true);
    l.setUint16(28, 0, true);
    local.set(name, 30);
    const central = new Uint8Array(46 + name.length);
    const c = new DataView(central.buffer);
    c.setUint32(0, 0x02014b50, true);
    c.setUint16(4, 20, true);
    c.setUint16(6, 20, true);
    c.setUint16(8, 0x0800, true);
    c.setUint16(10, 0, true);
    c.setUint16(12, 0, true);
    c.setUint16(14, 0x21, true);
    c.setUint32(16, crc, true);
    c.setUint32(20, f.bytes.length, true);
    c.setUint32(24, f.bytes.length, true);
    c.setUint16(28, name.length, true);
    c.setUint32(42, offset, true);
    central.set(name, 46);
    locals.push(local, f.bytes);
    centrals.push(central);
    offset += local.length + f.bytes.length;
  }
  const size = centrals.reduce((n, c) => n + c.length, 0);
  const end = new Uint8Array(22);
  const e = new DataView(end.buffer);
  e.setUint32(0, 0x06054b50, true);
  e.setUint16(8, files.length, true);
  e.setUint16(10, files.length, true);
  e.setUint32(12, size, true);
  e.setUint32(16, offset, true);
  const out = new Uint8Array(offset + size + 22);
  let at = 0;
  for (const part of [...locals, ...centrals, end]) {
    out.set(part, at);
    at += part.length;
  }
  return out;
}

// MARK: - reading

/** Inflate raw deflate, stopping past `cap` bytes. */
async function inflate(bytes: Uint8Array, cap: number): Promise<Uint8Array> {
  if (typeof DecompressionStream === "undefined") throw new Unreadable("this browser cannot inflate a compressed entry");
  const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
  const reader = stream.getReader();
  const parts: Uint8Array[] = [];
  let n = 0;
  for (;;) {
    let chunk;
    try {
      chunk = await reader.read();
    } catch {
      throw new Unreadable("a compressed entry is corrupt");
    }
    if (chunk.done) break;
    n += chunk.value.length;
    if (n > cap) {
      await reader.cancel().catch(() => {});
      throw new Unreadable(`an entry inflates past ${cap} bytes`);
    }
    parts.push(chunk.value);
  }
  const out = new Uint8Array(n);
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

/** A path inside a bundle, or a refusal: relative, no climbing out, no
 * backslashes, no empty or dot components. */
function safePath(name: string): string {
  if (name.includes("\\") || name.startsWith("/") || name.includes("\0")) throw new Unreadable(`an entry is named ${JSON.stringify(name)}`);
  const parts = name.split("/").filter((p) => p !== "" && p !== ".");
  if (parts.some((p) => p === "..")) throw new Unreadable(`an entry climbs out of the bundle: ${JSON.stringify(name)}`);
  return parts.join("/");
}

/** The files of a bundle: under a directory of their own (`x.kurven/`) or at
 * the top; directories and macOS's resource forks left out. */
function bundleFiles(entries: { path: string; bytes: Uint8Array }[]): { path: string; bytes: ArrayBuffer }[] {
  const files = entries.filter((e) => e.path && !e.path.startsWith("__MACOSX/") && !e.path.split("/").some((p) => p.startsWith("._")));
  let prefix = "";
  if (!files.some((f) => f.path === "manifest.json")) {
    const top = new Set(files.map((f) => f.path.split("/")[0]));
    const [only] = top;
    if (top.size === 1 && only && files.some((f) => f.path === `${only}/manifest.json`)) prefix = `${only}/`;
  }
  const out = files
    .filter((f) => f.path.startsWith(prefix) && f.path.length > prefix.length)
    .map((f) => ({ path: f.path.slice(prefix.length), bytes: f.bytes.slice().buffer as ArrayBuffer }));
  if (!out.some((f) => f.path === "manifest.json")) throw new Unreadable("it holds no manifest.json: it is not a .kurven bundle");
  return out;
}

/** A ZIP archive's entries, within the limits. */
export async function unzip(data: Uint8Array): Promise<{ path: string; bytes: ArrayBuffer }[]> {
  if (data.length > LIMITS.archive) throw new Unreadable(`it is ${data.length} bytes, more than ${LIMITS.archive}`);
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  // The end of central directory record, within the last 64 KB and 22 bytes.
  let end = -1;
  for (let i = data.length - 22; i >= Math.max(0, data.length - 22 - 65535); i--) {
    if (view.getUint32(i, true) === 0x06054b50) {
      end = i;
      break;
    }
  }
  if (end < 0) throw new Unreadable("it is not a ZIP archive");
  const count = view.getUint16(end + 10, true);
  const size = view.getUint32(end + 12, true);
  const start = view.getUint32(end + 16, true);
  if (count === 0xffff || size === 0xffffffff || start === 0xffffffff) throw new Unreadable("it is a ZIP64 archive, which a bundle never needs");
  if (count > LIMITS.entries) throw new Unreadable(`it holds ${count} entries, more than ${LIMITS.entries}`);
  if (start + size > end) throw new Unreadable("its directory runs past its end");
  const decoder = new TextDecoder("utf-8", { fatal: false });
  const entries: { path: string; bytes: Uint8Array }[] = [];
  let total = 0;
  let at = start;
  for (let k = 0; k < count; k++) {
    if (at + 46 > end || view.getUint32(at, true) !== 0x02014b50) throw new Unreadable("its directory is corrupt");
    const flags = view.getUint16(at + 8, true);
    const method = view.getUint16(at + 10, true);
    const crc = view.getUint32(at + 16, true);
    const packed = view.getUint32(at + 20, true);
    const unpacked = view.getUint32(at + 24, true);
    const nameLength = view.getUint16(at + 28, true);
    const extraLength = view.getUint16(at + 30, true);
    const commentLength = view.getUint16(at + 32, true);
    const local = view.getUint32(at + 42, true);
    const rawName = decoder.decode(data.subarray(at + 46, at + 46 + nameLength));
    at += 46 + nameLength + extraLength + commentLength;
    if (rawName.endsWith("/")) continue; // a directory
    const path = safePath(rawName);
    if (flags & 1) throw new Unreadable(`${path} is encrypted`);
    if (method !== 0 && method !== 8) throw new Unreadable(`${path} is compressed by method ${method}; a bundle is stored or deflated`);
    if (unpacked > LIMITS.entry) throw new Unreadable(`${path} is ${unpacked} bytes, more than ${LIMITS.entry}`);
    total += unpacked;
    if (total > LIMITS.total) throw new Unreadable(`its files come to more than ${LIMITS.total} bytes`);
    if (local + 30 > data.length || view.getUint32(local, true) !== 0x04034b50) throw new Unreadable(`${path}'s header is corrupt`);
    const dataStart = local + 30 + view.getUint16(local + 26, true) + view.getUint16(local + 28, true);
    if (dataStart + packed > data.length) throw new Unreadable(`${path} runs past the archive's end`);
    const raw = data.subarray(dataStart, dataStart + packed);
    const bytes = method === 0 ? raw : await inflate(raw, Math.min(unpacked, LIMITS.entry));
    if (bytes.length !== unpacked) throw new Unreadable(`${path} is not the size its directory says`);
    if (crc32(bytes) !== crc) throw new Unreadable(`${path} fails its checksum`);
    entries.push({ path, bytes });
  }
  return bundleFiles(entries);
}

/** The bundle a file holds: a ZIP archive of a bundle's files, or a bare
 * manifest.json (a surface's or a polyhedron's whole bundle). */
export async function readFile(file: File): Promise<Bundle> {
  if (file.size > LIMITS.archive) throw new Unreadable(`${file.name} is ${file.size} bytes, more than ${LIMITS.archive}`);
  const data = new Uint8Array(await file.arrayBuffer());
  const name = file.name.replace(/\.zip$/i, "") || "bundle.kurven";
  if (file.name.toLowerCase().endsWith(".json")) {
    return { name: name.replace(/\.json$/i, ".kurven"), files: [{ path: "manifest.json", bytes: data.buffer as ArrayBuffer }] };
  }
  return { name, files: await unzip(data) };
}

/** The bundle a folder holds, its files as the browser lists them (each
 * with its path under the folder chosen). */
export async function readFolder(files: File[]): Promise<Bundle> {
  if (files.length === 0) throw new Unreadable("the folder is empty");
  if (files.length > LIMITS.entries * 2) throw new Unreadable(`the folder holds ${files.length} files, more than a bundle`);
  const entries: { path: string; bytes: Uint8Array }[] = [];
  let total = 0;
  let top = "";
  for (const f of files) {
    const relative = (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name;
    const path = safePath(relative);
    top ||= path.split("/")[0] ?? "";
    if (f.size > LIMITS.entry) throw new Unreadable(`${path} is ${f.size} bytes, more than ${LIMITS.entry}`);
    total += f.size;
    if (total > LIMITS.total) throw new Unreadable(`its files come to more than ${LIMITS.total} bytes`);
    entries.push({ path, bytes: new Uint8Array(await f.arrayBuffer()) });
  }
  if (entries.length > LIMITS.entries) throw new Unreadable(`it holds ${entries.length} files, more than ${LIMITS.entries}`);
  return { name: top || "bundle.kurven", files: bundleFiles(entries) };
}
