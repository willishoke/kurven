// The Worker's side of the protocol (protocol.ts): it loads the engine,
// stages bundles, and answers requests one at a time, transferring each
// reply's arrays to the page. `serve` takes the port it listens on, so the
// same code runs in a browser's Worker (worker-entry.ts) and under Node's
// worker_threads (web/app/test/worker.test.mjs).

import { Engine } from "./module.ts";
import type { FromWorker, Port, ToWorker } from "./protocol.ts";

export function serve(port: Port): void {
  let engine: Engine | null = null;
  const send = (message: FromWorker, transfer: Transferable[] = []) => port.postMessage(message, transfer);
  port.onmessage = (event) => {
    void handle(event.data as ToWorker);
  };

  async function handle(message: ToWorker): Promise<void> {
    try {
      switch (message.type) {
        case "init": {
          const bytes =
            typeof message.wasm === "string" ? await (await fetch(message.wasm)).arrayBuffer() : message.wasm;
          engine = await Engine.load(bytes);
          send({ type: "ready" });
          return;
        }
        case "stage": {
          if (!engine) throw new Error("the engine is not loaded");
          for (const file of message.bundle.files) {
            engine.stage(message.bundle.name, file.path, new Uint8Array(file.bytes));
          }
          send({ type: "staged", id: message.id });
          return;
        }
        case "unstage":
          engine?.unstage();
          return;
        case "request": {
          if (!engine) throw new Error("the engine is not loaded");
          const { reply, arrays } = engine.call(message.request);
          const transfer = Object.values(arrays).map((a) => a.buffer as ArrayBuffer);
          send({ type: "reply", id: message.id, reply, arrays }, transfer);
          return;
        }
      }
    } catch (error) {
      const id = message.type === "request" || message.type === "stage" ? message.id : null;
      // A trap leaves the module unusable: the client restarts the Worker
      // on it. Anything else failed that message alone.
      const trap = error instanceof WebAssembly.RuntimeError;
      if (trap) engine = null;
      send({ type: "failed", id, trap, message: String(error) });
    }
  }
}
