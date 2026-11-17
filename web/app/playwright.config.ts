// The browser lane (scripts/check-browser.sh): the preview in Chromium,
// against the judged strokes the engine bakes for the same frame. Separate
// from scripts/check-web.sh, which needs no browser.
//
// Chrome for Testing in its new headless mode (`channel: "chromium"`):
// the old headless shell has WebGPU but cannot present a WebGPU canvas
// (its device is lost at the first frame). Three projects:
//
// - swiftshader: WebGPU on SwiftShader, Dawn's CPU Vulkan, the same on every
//   machine; what CI runs.
// - gpu: WebGPU on the machine's own GPU (Metal on a Mac), where there is one.
// - fallback: Chromium with no GPU at all (`--disable-gpu`: WebGPU has no
//   adapter), the page drawing judged strokes at rest.

import { defineConfig } from "@playwright/test";

const port = Number(process.env.KURVEN_PORT ?? 4175);
// On Linux, Dawn reaches SwiftShader through Vulkan, which Chromium leaves
// off there unless asked, and a headless run has no surface to present a
// Vulkan swapchain to. Not yet observed on a CI runner
// (crates/FINDINGS.md, "The browser lane in CI").
const linux = process.platform === "linux" ? ["--enable-features=Vulkan", "--disable-vulkan-surface"] : [];

export default defineConfig({
  testDir: "test/browser",
  timeout: 600_000,
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: { baseURL: `http://localhost:${port}`, viewport: { width: 1000, height: 800 }, deviceScaleFactor: 1 },
  webServer: {
    command: `npx vite preview --port ${port} --strictPort`,
    url: `http://localhost:${port}/kurven_wasm.wasm`,
    reuseExistingServer: false,
    timeout: 60_000,
  },
  projects: [
    {
      name: "swiftshader",
      use: {
        channel: "chromium",
        launchOptions: { args: ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", ...linux] },
      },
    },
    { name: "gpu", use: { channel: "chromium", launchOptions: { args: ["--enable-unsafe-webgpu"] } } },
    { name: "fallback", use: { channel: "chromium", launchOptions: { args: ["--disable-gpu"] } } },
  ],
});
