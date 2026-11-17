// The engine's Worker under Node: worker.ts's `serve` on worker_threads'
// parent port, adapted to the protocol's Port.

import { parentPort } from "node:worker_threads";
import { serve } from "../src/engine/worker.ts";

const port = {
  postMessage: (message, transfer) => parentPort.postMessage(message, transfer),
  onmessage: null,
};
parentPort.on("message", (data) => port.onmessage?.({ data }));
serve(port);
