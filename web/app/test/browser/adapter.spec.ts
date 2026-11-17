// What the browser's WebGPU is, first, so a lane that cannot draw says why in
// its log: the adapter (vendor, architecture, whether a fallback), its
// features and limits, a device asked for what the renderer asks for, a
// compute pass (the layout probe), and a canvas cleared and presented --
// each step's error or the device's loss, with its reason, printed. The
// page's own fallback reason is printed too.

import { expect, test } from "@playwright/test";

test("the adapter, the device and a presented canvas", async ({ page }, info) => {
  test.skip(info.project.name === "fallback", "no WebGPU by design");
  page.on("console", (m) => console.log(`console ${m.type()}: ${m.text()}`));
  page.on("pageerror", (e) => console.log(`page error: ${e.message}`));
  await page.goto("/?size=64x64");
  await page.waitForFunction(() => document.body.dataset.ready === "true", null, { timeout: 120_000 });
  const report = await page.evaluate(async () => {
    const out: Record<string, unknown> = { fallbackReason: (window as any).kurven.fallbackReason };
    const step = async (name: string, body: () => Promise<unknown>) => {
      try {
        out[name] = (await body()) ?? "ok";
      } catch (error) {
        out[name] = `FAILED: ${String(error)}`;
      }
    };
    if (!("gpu" in navigator) || !navigator.gpu) return { ...out, gpu: "no navigator.gpu" };
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { ...out, adapter: "requestAdapter answered null" };
    const a = adapter.info;
    out.adapter = { vendor: a.vendor, architecture: a.architecture, device: a.device, description: a.description, fallback: (a as any).isFallbackAdapter };
    out.features = [...adapter.features].sort();
    const l = adapter.limits;
    out.limits = { maxTextureDimension2D: l.maxTextureDimension2D, maxStorageBufferBindingSize: l.maxStorageBufferBindingSize, maxBufferSize: l.maxBufferSize };
    out.float32Blendable = adapter.features.has("float32-blendable");
    let device: GPUDevice | null = null;
    await step("device", async () => {
      device = await adapter.requestDevice({ requiredFeatures: adapter.features.has("float32-blendable") ? ["float32-blendable"] : [] });
      void device.lost.then((l) => {
        out.lost = `${l.reason}: ${l.message}`;
      });
      device.addEventListener("uncapturederror", (e) => {
        out.uncaptured = String((e as GPUUncapturedErrorEvent).error.message);
      });
    });
    await step("probe", () => (window as any).kurven.probe());
    if (device) {
      const d = device as GPUDevice;
      await step("canvas", async () => {
        const canvas = document.createElement("canvas");
        canvas.width = canvas.height = 16;
        document.body.append(canvas);
        const ctx = canvas.getContext("webgpu") as GPUCanvasContext;
        ctx.configure({ device: d, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: "opaque" });
        const encoder = d.createCommandEncoder();
        const pass = encoder.beginRenderPass({
          colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: "clear", storeOp: "store", clearValue: { r: 1, g: 0, b: 0, a: 1 } }],
        });
        pass.end();
        d.queue.submit([encoder.finish()]);
        await d.queue.onSubmittedWorkDone();
        return "presented";
      });
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
    return out;
  });
  const line = `${info.project.name} WebGPU: ${JSON.stringify(report)}`;
  console.log(line);
  info.annotations.push({ type: "adapter", description: line });
  expect(report.lost, "the device is not lost").toBeUndefined();
  expect(report.fallbackReason, "the page draws on the GPU").toBeNull();
});
