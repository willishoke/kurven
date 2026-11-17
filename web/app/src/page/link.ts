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
  // What an open keeps, set before it.
  if (!link.refine) before.push({ kind: "refine", on: false });
  if (link.mode !== "drawing") before.push({ kind: "mode", mode: link.mode });
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

/** The link a fragment holds, or null if it holds none. Throws on one that
 * is malformed, saying so. */
export async function decode(hash: string): Promise<Link | null> {
  const m = hash.match(/^#k1(j?)\.([A-Za-z0-9_-]+)$/);
  if (!m) return null;
  let bytes = unbase64url(m[2] ?? "");
  if (m[1] !== "j") {
    bytes = await through(bytes, new DecompressionStream("deflate-raw") as TransformStream<Uint8Array, Uint8Array>);
  }
  const value = JSON.parse(new TextDecoder().decode(bytes)) as Partial<Link> & { v?: number };
  if (value.v !== 1 || !value.edited || !value.camera) throw new Error("not a kurven link this page can read");
  return {
    edited: value.edited,
    hidden: value.hidden ?? [],
    camera: value.camera,
    mode: value.mode ?? "drawing",
    depthTolerance: value.depthTolerance ?? 0.02,
    bakeResolution: value.bakeResolution ?? 4000,
    refine: value.refine ?? true,
  };
}
