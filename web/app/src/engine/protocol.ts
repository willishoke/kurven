// The message protocol between the page and the engine's Worker.
//
// The engine is kurven-session behind kurven-wasm's plain C ABI
// (crates/kurven-wasm/src/lib.rs); its requests and replies are
// kurven_session::protocol's, whose doc says what each field means. Every
// request is a JSON object with a `kind`; every reply carries the
// document's state as the request left it (but a `headless` run's), and
// the typed arrays it names travel beside it, transferred, not copied.
//
// Three rules hold the protocol together:
//
// 1. Requests are answered in the order they are sent: the Worker runs one
//    at a time, and each reply carries an `id` the client gave it.
// 2. A view of the document -- a frame, a bake, the grids -- is asked for
//    under a key, and a newer ask under the same key supersedes an older
//    one: an older reply that lands after a newer ask was sent is stale,
//    and the client drops it (its promise resolves `{ stale: true }`). At
//    most one ask per key is in flight; a newer one waits, replacing any
//    already waiting, so a camera that keeps moving costs one frame behind,
//    never a queue (Swift's `pump`: one job in flight, the latest wins).
// 3. A request can hang (`besseli(1e12, z)` loops about 1e12 times;
//    crates/FINDINGS.md), and a Worker cannot be interrupted, only
//    terminated. So the client can cancel: it terminates the Worker,
//    rejects whatever is pending with `Cancelled`, starts a new one, and
//    replays the requests that changed the document and were answered (the
//    journal), so the new Worker holds the document the old one held before
//    the request that hung. A timeout cancels the same way.

/** A request the engine answers: kurven_session::protocol's. */
export type Request =
  | { kind: "state" }
  /** The catalog's landscape presets, for the formula's dropdown. */
  | { kind: "catalog" }
  /** What is edited now, whole: what the page's URL holds. */
  | { kind: "edited" }
  /** A landscape as a link names it: `edited`'s fields, sampled afresh with
   * no edits to undo (Phase 8; kurven_session::Document::restore_landscape). */
  | ({ kind: "restore" } & Edited)
  /** The Visibility section's depth tolerance (Inspector.swift). */
  | { kind: "depthTolerance"; value: number }
  /** The engine instance's name, given by the Worker at load (module.ts). */
  | { kind: "seed"; nonce: string }
  | { kind: "open"; name: string }
  | { kind: "create"; preset: string }
  | { kind: "createSurface"; preset: string }
  | { kind: "expression"; expression: string }
  | { kind: "set"; key: string; value: number }
  | { kind: "drag"; control: Slider; phase: "begin" | "move" | "end"; value?: number | [number, number] }
  | { kind: "levels"; layer: string; count: number }
  | { kind: "resetLevels"; layer: string }
  | { kind: "strokes"; layer: string; count: number }
  | { kind: "relief"; value: number }
  | { kind: "truncation"; truncation: Truncation }
  | { kind: "pick"; pixel?: [number, number]; point?: [number, number, number] }
  | { kind: "add"; name: string }
  | { kind: "remove"; layer: string }
  | { kind: "function"; expression: string; periods?: [number, number] }
  | { kind: "undo" }
  | { kind: "redo" }
  | { kind: "axis"; axis: "real" | "imag" }
  | { kind: "light"; layer: string; direction: [number, number, number] }
  | { kind: "hide"; layer: number }
  | { kind: "gesture"; gesture: Gesture }
  /** Gestures in order, each applied as `gesture` applies it: the camera
   * they make one by one, in one request. */
  | { kind: "gestures"; gestures: Gesture[] }
  | ({ kind: "camera" } & ExactCamera)
  | { kind: "fit" }
  | { kind: "preset"; index: number }
  | { kind: "perspective"; on: boolean }
  | { kind: "viewport"; width: number; height: number }
  | { kind: "mode"; mode: "drawing" | "shaded" | "depth" }
  | { kind: "refine"; on: boolean }
  | { kind: "draftResolution"; value: number }
  | { kind: "bakeResolution"; value: number }
  | { kind: "frame"; draft: boolean }
  /** A frame of the preview for the page's GPU (kurven_session::preview,
   * src/gpu/frame.ts): what it holds under `have` is not sent again. */
  | {
      kind: "preview";
      draft: boolean;
      have?: { content?: string; layers?: string[] };
      /** Layouts at rest another engine of the pool laid out (`restLayout`),
       * their arrays given as the request's inputs, and the fingerprint of
       * the document they are of (kurven_session::pool). */
      laid?: Laid[];
      fingerprint?: string;
    }
  | { kind: "bake"; svg?: boolean }
  /** The judged strokes shared among a pool of engines (kurven_session::pool):
   * the pieces a bake splits into, one piece baked, and the pieces put
   * together, its arrays given as inputs, which answers as `bake` does (or
   * `failed`: a piece's refusal, said as the bake's status). */
  | { kind: "bakePlan" }
  /** `fingerprint`: the plan's document; a helper holding another refuses
   * the share as such (the reply's `another`). */
  | { kind: "bakePiece"; piece: Piece; fingerprint: string }
  /** `fingerprint` and `frame`: the plan's; put together after the document
   * moved on, the reply says `snapshot`. */
  | { kind: "bakeAssemble"; svg?: boolean; plan: Piece[]; colors: string[][]; fingerprint: string; frame: unknown }
  | { kind: "bakeAssemble"; failed: string }
  /** The frame at rest shared likewise: the layers it would lay out, holding
   * what `have` holds, and some of them laid out. */
  | { kind: "restLayers"; have?: { content?: string; layers?: string[] } }
  | { kind: "restLayout"; layers: number[]; fingerprint: string }
  /** The shown surface's coordinate image by the CPU rasterizer, with the
   * frame, view and clip the GPU renders the same image by. */
  | { kind: "surfaceImage"; resolution: number }
  | { kind: "grids" }
  | { kind: "save"; name: string }
  | { kind: "headless"; args: string[] };

/** A share of a bake (kurven_bake::pieces::Piece): the whole bake, a layer
 * whole, or paths `from..to` of a layer. */
export type Piece = { whole: true } | { layer: number } | { layer: number; from: number; to: number };

/** A layout at rest as `restLayout` describes it. */
export interface Laid {
  layer: number;
  refusal: string | null;
}

/** What is edited (kurven_session::edits::Edited): a landscape's request,
 * the styling it is drawn with -- its layers as a manifest describes them
 * and its truncation -- and the level counts the sliders set. */
export interface Edited {
  landscape: LandscapeRequest;
  layers: unknown[];
  truncation: Truncation;
  levelCounts: Record<string, number>;
}

/** A landscape's request as the engine writes it (`LandscapeRequest::json`). */
export interface LandscapeRequest {
  expression: string;
  name: string;
  resolution: number;
  domain: Domain;
  truncation?: Truncation;
  spacing?: number;
  layers?: unknown[];
}

/** A rectangle of the complex plane. */
export interface Domain {
  real: { lo: number; hi: number };
  imag: { lo: number; hi: number };
}

/** A catalog entry (`FunctionPreset::json`). */
export interface Preset {
  name: string;
  label: string;
  expression: string;
  domain: Domain;
  /** How far the window's sliders reach. */
  window: Domain;
  truncation: number | null;
  notes: string;
}

/** How far the window's controls reach: the engine's (`sliders::reach`),
 * which a link is held to, sent in the `catalog` reply. */
export interface Controls {
  samples: { min: number; max: number; step: number };
  levels: { min: number; max: number };
  strokes: { min: number; max: number };
  truncation: { min: number };
}

/** Where an expression went wrong, as the `expression` refusal says it: in
 * characters as the language counts them (Unicode scalar values). */
export interface ExpressionAt {
  message: string;
  position: number | null;
  length: number;
}

/** A truncation as a manifest writes it. */
export type Truncation =
  | { kind: "none" }
  | { kind: "uniform"; height: number }
  | { kind: "realBands"; bands: { below: number; height: number }[]; beyond: number };

/** A control named in a drag. */
export type Slider =
  | { slider: "landscape"; label: string }
  | { slider: "levels"; name: string }
  | { slider: "strokes"; name: string }
  | { slider: "light"; name: string }
  | { slider: "truncation" }
  | { slider: "surfaceField"; field: string; label?: string };

/** What the input handling produces (kurven_core::navigation::Gesture). */
export type Gesture =
  | { kind: "orbit"; delta: [number, number] }
  | { kind: "pan"; delta: [number, number] }
  | { kind: "zoom"; factor: number; at: [number, number] }
  | { kind: "roll"; radians: number }
  | { kind: "retarget"; point: [number, number, number] };

/** The camera to the bit, as the state reports it and a `camera` request
 * sets it: the angles in radians as the orbit holds them. */
export interface ExactCamera {
  azimuth: number;
  elevation: number;
  target: [number, number, number];
  distance: number | null;
  center: [number, number];
  unitsPerPixel: number;
}

/** One layer as the controls show it. */
export interface LayerState {
  name: string;
  kind: string;
  role: string;
  width: number;
  color: string;
  hidden: boolean;
  levels?: number;
  strokes?: number;
  light?: [number, number, number];
  spacing?: number;
}

/** The document as the page shows it: everything a control reads. */
export interface DocumentState {
  kind: "empty" | "failed" | "landscape" | "surface" | "polyhedron";
  failure?: string;
  title: string;
  status: Record<"landscape" | "surface" | "polyhedron" | "paint" | "pick" | "bake", string | null>;
  landscape: Record<string, unknown> | null;
  surface: Record<string, unknown> | null;
  truncation: Truncation | null;
  layers: LayerState[];
  levelCounts: Record<string, number>;
  camera?: {
    azimuth: number;
    elevation: number;
    target: [number, number, number];
    center: [number, number];
    unitsPerPixel: number;
    perspective: boolean;
    exact: ExactCamera;
  };
  viewport: [number, number];
  presets: string[];
  picked: {
    face: string;
    copy: number;
    title: string;
    choices: { id: string; label: string }[];
    layers: string[];
  } | null;
  undo: { can: boolean; name: string | null };
  redo: { can: boolean; name: string | null };
  current: boolean;
  depthTolerance: number;
  bakeResolution: number;
  draftResolution: number;
  canBake: boolean;
  refine: boolean;
  mode: "drawing" | "shaded" | "depth";
}

/** A typed array a reply carries. */
export type TypedArray = Float64Array | Float32Array | Uint32Array | Uint8Array;

/** A reply: its JSON, and its arrays by name. */
export interface Reply {
  ok: boolean;
  /** False only for a request refused before it reached the document. */
  touched: boolean;
  error?: string;
  state?: DocumentState;
  arrays: { name: string; type: "f64" | "f32" | "u32" | "u8" }[];
  [key: string]: unknown;
}

/** A bundle's files, by their path in it, for an `open` or a `headless` run. */
export interface Bundle {
  name: string;
  files: { path: string; bytes: ArrayBuffer }[];
}

/** Page to Worker. */
export type ToWorker =
  | { type: "init"; wasm: ArrayBuffer | string }
  | { type: "stage"; id: number; bundle: Bundle }
  | { type: "unstage" }
  /** `inputs`: typed arrays given to this request alone (`kv_give`), the
   * shares of a pool's work another Worker answered with. */
  | { type: "request"; id: number; request: Request; inputs?: { name: string; bytes: ArrayBuffer }[] };

/** Worker to page. */
export type FromWorker =
  | { type: "ready" }
  | { type: "staged"; id: number }
  /** `memory`: the module's linear memory after the request, in bytes. */
  | { type: "reply"; id: number; reply: Reply; arrays: Record<string, TypedArray>; memory?: number }
  /** A stage or a request that threw (`id`), or the engine that would not
   * load (`id` null). `trap`: the module trapped (it aborts on a panic) and
   * is gone with what it held; otherwise only that message failed. */
  | { type: "failed"; id: number | null; trap: boolean; message: string };

/** The end of a channel either side holds: a Worker, the Worker's own
 * global, or Node's worker_threads, adapted. */
export interface Port {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((event: { data: unknown }) => void) | null;
}
