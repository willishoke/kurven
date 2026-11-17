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
  | { kind: "fit" }
  | { kind: "preset"; index: number }
  | { kind: "perspective"; on: boolean }
  | { kind: "viewport"; width: number; height: number }
  | { kind: "mode"; mode: "drawing" | "shaded" | "depth" }
  | { kind: "refine"; on: boolean }
  | { kind: "draftResolution"; value: number }
  | { kind: "bakeResolution"; value: number }
  | { kind: "frame"; draft: boolean }
  | { kind: "bake"; svg?: boolean }
  | { kind: "grids" }
  | { kind: "save"; name: string }
  | { kind: "headless"; args: string[] };

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
}

/** A typed array a reply carries. */
export type TypedArray = Float64Array | Float32Array | Uint32Array | Uint8Array;

/** A reply: its JSON, and its arrays by name. */
export interface Reply {
  ok: boolean;
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
  | { type: "stage"; bundle: Bundle }
  | { type: "unstage" }
  | { type: "request"; id: number; request: Request };

/** Worker to page. */
export type FromWorker =
  | { type: "ready" }
  | { type: "staged"; name: string }
  | { type: "reply"; id: number; reply: Reply; arrays: Record<string, TypedArray> }
  | { type: "failed"; id: number | null; message: string };

/** The end of a channel either side holds: a Worker, the Worker's own
 * global, or Node's worker_threads, adapted. */
export interface Port {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((event: { data: unknown }) => void) | null;
}
