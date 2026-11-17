// The browser's Worker: `new Worker(new URL("./worker-entry.ts",
// import.meta.url), { type: "module" })`, then the client's `init`.

import type { Port } from "./protocol.ts";
import { serve } from "./worker.ts";

serve(self as unknown as Port);
