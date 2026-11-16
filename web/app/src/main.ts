// The page: a thin view of the engine. All state lives in the engine
// (crates/kurven-session, run in a Worker through crates/kurven-wasm); the
// page renders state and sends edits (HANDOFF-web-port.md, Architecture).
// Phase 0 has no engine to show, so this is the page's home and nothing
// more: Phase 8 builds it.

const app = document.querySelector<HTMLElement>("#app");
if (!app) {
  throw new Error("index.html has no #app");
}
const heading = document.createElement("h1");
heading.textContent = "kurven";
app.replaceChildren(heading);
