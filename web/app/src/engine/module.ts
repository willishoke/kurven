// kurven-wasm's plain C ABI, read with nothing but WebAssembly.instantiate
// (crates/kurven-wasm/src/lib.rs says what each export takes). The bytes a
// reply leaves in the module's memory are copied out once, into buffers of
// their own, which is what the Worker then transfers to the page.

import type { Reply, TypedArray } from "./protocol.ts";

interface Exports {
  memory: WebAssembly.Memory;
  kv_alloc(len: number): number;
  kv_free(ptr: number, len: number): void;
  kv_stage(b: number, bl: number, p: number, pl: number, d: number, dl: number): number;
  kv_unstage(): void;
  kv_call(ptr: number, len: number): number;
  kv_reply_ptr(): number;
  kv_array_count(): number;
  kv_array_ptr(i: number): number;
  kv_array_len(i: number): number;
}

const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8", { fatal: true });

export class Engine {
  private readonly wasm: Exports;

  private constructor(wasm: Exports) {
    this.wasm = wasm;
  }

  static async load(bytes: ArrayBuffer): Promise<Engine> {
    const { instance } = await WebAssembly.instantiate(bytes, {});
    return new Engine(instance.exports as unknown as Exports);
  }

  /** Bytes into memory the module lent, and the loan back afterwards. */
  private lend<T>(parts: Uint8Array[], body: (args: number[]) => T): T {
    const loans = parts.map((bytes) => {
      const ptr = this.wasm.kv_alloc(bytes.length);
      new Uint8Array(this.wasm.memory.buffer, ptr, bytes.length).set(bytes);
      return [ptr, bytes.length] as const;
    });
    try {
      return body(loans.flat());
    } finally {
      for (const [ptr, len] of loans) this.wasm.kv_free(ptr, len);
    }
  }

  stage(bundle: string, path: string, bytes: Uint8Array): void {
    const ok = this.lend([encoder.encode(bundle), encoder.encode(path), bytes], (a) =>
      this.wasm.kv_stage(a[0]!, a[1]!, a[2]!, a[3]!, a[4]!, a[5]!),
    );
    if (!ok) throw new Error(`could not stage ${bundle}/${path}`);
  }

  unstage(): void {
    this.wasm.kv_unstage();
  }

  /** One request: the reply, and its arrays copied out of the module. */
  call(request: unknown): { reply: Reply; arrays: Record<string, TypedArray> } {
    const n = this.lend([encoder.encode(JSON.stringify(request))], (a) => this.wasm.kv_call(a[0]!, a[1]!));
    // Read the memory afresh: the call may have grown it.
    const memory = this.wasm.memory.buffer;
    const reply = JSON.parse(decoder.decode(new Uint8Array(memory, this.wasm.kv_reply_ptr(), n))) as Reply;
    const arrays: Record<string, TypedArray> = {};
    reply.arrays.forEach(({ name, type }, i) => {
      const ptr = this.wasm.kv_array_ptr(i);
      const copy = memory.slice(ptr, ptr + this.wasm.kv_array_len(i));
      arrays[name] =
        type === "f64"
          ? new Float64Array(copy)
          : type === "f32"
            ? new Float32Array(copy)
            : type === "u32"
              ? new Uint32Array(copy)
              : new Uint8Array(copy);
    });
    return { reply, arrays };
  }
}
