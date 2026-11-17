// The whole state in the URL, so a link reproduces a drawing (Phase 8,
// step 3). The engine holds the state; a link holds what the engine says is
// edited (`edited`: the landscape's request, the styling it is drawn with,
// the level counts) and what the view is (the camera to the bit, the hidden
// layers, the mode, the depth tolerance, the bake's resolution, whether the
// contours are placed by f). Opening a link sends the engine the requests
// that make that document again (`restoring`), on whatever viewport the new
// page has: the camera carries its own framing.
//
// The fragment is `#k1.` and the JSON deflated and base64url'd, or `#k1j.`
// and the JSON base64url'd where the browser has no CompressionStream. JSON
// numbers are the shortest that read back to the same double in both
// directions (JavaScript's and the engine's canonical writer), so the camera
// and the window come back to the bit; a -0 comes back as 0, which JSON
// cannot say (crates/FINDINGS.md, review of PR #70).

import type { DocumentState, Edited, ExactCamera, Request } from "../engine/protocol.ts";

export interface Link {
  edited: Edited;
  hidden: string[];
  camera: ExactCamera;
  mode: DocumentState["mode"];
  depthTolerance: number;
  bakeResolution: number;
  refine: boolean;
}

/** The link for the document as `state` and `edited` say it, or null for
 * a document a link cannot hold yet (a surface, a polyhedron, a bundle opened
 * read-only: Phase 8's later steps). */
export function linkOf(state: DocumentState, edited: Edited | null): Link | null {
  if (state.kind !== "landscape" || !edited?.landscape || !state.camera) return null;
  return {
    edited,
    hidden: state.layers.filter((l) => l.hidden).map((l) => l.name),
    camera: state.camera.exact,
    mode: state.mode,
    depthTolerance: state.depthTolerance,
    bakeResolution: state.bakeResolution,
    refine: state.refine,
  };
}

/** The requests that make the linked document again, in order, but for the
 * hidden layers, which are toggled by their place once the restore says
 * where each is (`hides`). `viewport` is the new page's. */
export function restoring(link: Link, viewport: [number, number]): { before: Request[]; after: Request[] } {
  const before: Request[] = [{ kind: "viewport", width: viewport[0], height: viewport[1] }];
  // What an open keeps, set before it, outright: a page that had others
  // (a link pasted into a tab) takes the link's.
  before.push({ kind: "refine", on: link.refine });
  before.push({ kind: "mode", mode: link.mode });
  before.push({ kind: "restore", ...link.edited });
  // What an open sets from the bundle's preset, set after it.
  const after: Request[] = [
    { kind: "depthTolerance", value: link.depthTolerance },
    { kind: "bakeResolution", value: link.bakeResolution },
    { kind: "camera", ...link.camera },
  ];
  return { before, after };
}

/** The `hide` requests for a link's hidden layers, by their place in the
 * restored document's layers. */
export function hides(link: Link, state: DocumentState): Request[] {
  const out: Request[] = [];
  state.layers.forEach((l, i) => {
    if (link.hidden.includes(l.name) && !l.hidden) out.push({ kind: "hide", layer: i });
  });
  return out;
}

function base64url(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
}

function unbase64url(text: string): Uint8Array {
  const s = atob(text.replaceAll("-", "+").replaceAll("_", "/"));
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}

async function through(bytes: Uint8Array, stream: TransformStream<Uint8Array, Uint8Array>): Promise<Uint8Array> {
  const body = new Blob([bytes as BlobPart]).stream().pipeThrough(stream);
  return new Uint8Array(await new Response(body).arrayBuffer());
}

const compressing = typeof CompressionStream === "function" && typeof DecompressionStream === "function";

/** The fragment for a link, `#` included. */
export async function encode(link: Link): Promise<string> {
  const json = new TextEncoder().encode(JSON.stringify({ v: 1, ...link }));
  if (!compressing) return `#k1j.${base64url(json)}`;
  const packed = await through(json, new CompressionStream("deflate-raw") as TransformStream<Uint8Array, Uint8Array>);
  return `#k1.${base64url(packed)}`;
}

/** The longest fragment read: a link is about 1.3 KB deflated (a typed
 * expression's, the catalog's are shorter) and 7.5 KB plain, so 16 KB is
 * twice the plainest, and a fragment past it is not one of the page's. */
export const MOST_FRAGMENT = 16 * 1024;
/** The most JSON a fragment may inflate to: a link's is 5 to 6 KB (nine
 * layers, their levels listed), so 64 KB is ten times that, and an inflate
 * that runs past it (a zip bomb) is stopped there, before it is parsed. */
export const MOST_JSON = 64 * 1024;

/** Why a link is refused. */
export class LinkError extends Error {}

/** Inflate `bytes`, refusing once past `MOST_JSON` bytes out. */
async function inflate(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
  const reader = (stream as ReadableStream<Uint8Array>).getReader();
  const chunks: Uint8Array[] = [];
  let n = 0;
  for (;;) {
    let next;
    try {
      next = await reader.read();
    } catch {
      throw new LinkError("its compressed part is not deflate data");
    }
    if (next.done) break;
    n += next.value.length;
    if (n > MOST_JSON) {
      await reader.cancel().catch(() => {});
      throw new LinkError(`it inflates past ${MOST_JSON / 1024} KB, which no link of this page does`);
    }
    chunks.push(next.value);
  }
  const out = new Uint8Array(n);
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}

const isNumber = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x);
const isObject = (x: unknown): x is Record<string, unknown> => typeof x === "object" && x !== null && !Array.isArray(x);
const numbers = (x: unknown, n: number): x is number[] => Array.isArray(x) && x.length === n && x.every(isNumber);

/** `value` as a link, every field of the right type, or a LinkError naming
 * the first that is not. The engine holds what the numbers ask for to the
 * controls' reach (`restore`); this holds the shape, so nothing the page
 * does with a link reads a field that is not there. */
function checked(value: unknown): Link {
  const fail = (what: string): never => {
    throw new LinkError(`its ${what} is not what a link holds`);
  };
  if (!isObject(value) || value.v !== 1) throw new LinkError("it is not a link this page can read");
  const edited = value.edited;
  if (!isObject(edited)) return fail("landscape");
  const l = edited.landscape;
  if (!isObject(l) || typeof l.expression !== "string" || typeof l.name !== "string" || !Number.isInteger(l.resolution)) {
    return fail("landscape");
  }
  const d = l.domain;
  const interval = (i: unknown) => isObject(i) && isNumber(i.lo) && isNumber(i.hi);
  if (!isObject(d) || !interval(d.real) || !interval(d.imag)) return fail("window");
  if (!Array.isArray(edited.layers) || !edited.layers.every(isObject)) return fail("layers");
  if (!isObject(edited.truncation) || typeof edited.truncation.kind !== "string") return fail("truncation");
  const counts = edited.levelCounts;
  if (!isObject(counts) || !Object.values(counts).every((n) => Number.isInteger(n))) return fail("level counts");
  const hidden = value.hidden ?? [];
  if (!Array.isArray(hidden) || !hidden.every((h) => typeof h === "string")) return fail("hidden layers");
  const c = value.camera;
  if (
    !isObject(c) ||
    !isNumber(c.azimuth) ||
    !isNumber(c.elevation) ||
    !numbers(c.target, 3) ||
    !numbers(c.center, 2) ||
    !isNumber(c.unitsPerPixel) ||
    !(c.distance === null || c.distance === undefined || isNumber(c.distance))
  ) {
    return fail("camera");
  }
  const mode = value.mode ?? "drawing";
  if (mode !== "drawing" && mode !== "shaded" && mode !== "depth") return fail("mode");
  const depthTolerance = value.depthTolerance ?? 0.02;
  if (!isNumber(depthTolerance)) return fail("depth tolerance");
  const bakeResolution = value.bakeResolution ?? 4000;
  if (!Number.isInteger(bakeResolution)) return fail("bake resolution");
  const refine = value.refine ?? true;
  if (typeof refine !== "boolean") return fail("refinement");
  return {
    edited: edited as unknown as Edited,
    hidden: hidden as string[],
    camera: { ...(c as unknown as ExactCamera), distance: (c.distance as number | null | undefined) ?? null },
    mode,
    depthTolerance: depthTolerance as number,
    bakeResolution: bakeResolution as number,
    refine,
  };
}

/** The link a fragment holds, or null if it holds none. Throws a LinkError
 * saying why for one that is too long, will not inflate within bounds, is
 * not JSON, or is not a link's shape. */
export async function decode(hash: string): Promise<Link | null> {
  if (!hash.startsWith("#k1")) return null;
  if (hash.length > MOST_FRAGMENT) throw new LinkError(`it is longer than ${MOST_FRAGMENT / 1024} KB, which no link of this page is`);
  const m = hash.match(/^#k1(j?)\.([A-Za-z0-9_-]+)$/);
  if (!m) throw new LinkError("it is not written as this page writes links");
  let bytes: Uint8Array;
  try {
    bytes = unbase64url(m[2] ?? "");
  } catch {
    throw new LinkError("it is not base64url");
  }
  if (m[1] === "j") {
    if (bytes.length > MOST_JSON) throw new LinkError(`it holds more than ${MOST_JSON / 1024} KB`);
  } else {
    bytes = await inflate(bytes);
  }
  let value: unknown;
  try {
    value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch {
    throw new LinkError("it does not hold JSON");
  }
  return checked(value);
}
