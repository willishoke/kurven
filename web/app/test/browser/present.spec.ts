// Presenting a canvas: sn's frame at rest drawn into the canvas and read
// back, then the same frame drawn again into the offscreen target the
// SwiftShader project's gate reads, held to be the same bytes. That is what
// lets the offscreen gate stand for the canvas: one renderer, one frame,
// the same pipelines and target format and size, and the same pixels.
//
// On Linux, SwiftShader's GPU process runs WebGPU but has lost the device the
// moment a canvas is presented (crates/FINDINGS.md, "The browser lane in
// CI"); there this test is allowed to fail, says how in its annotation, and
// is run under three sets of flags (playwright.config.ts: present,
// present-gl, present-angle) to learn whether any presents. On the Mac's
// own GPU (`gpu`) it must pass.

import { expect, test } from "@playwright/test";

test("a frame presented on the canvas is the frame drawn offscreen", async ({ page }, info) => {
  test.skip(info.project.name === "fallback", "no WebGPU by design");
  const allowed = process.platform === "linux" && info.project.name !== "gpu";
  page.on("console", (m) => {
    if (m.type() === "error") console.log(`console: ${m.text()}`);
  });
  try {
    await page.goto("/?size=640x480&slope=0&surface=sn");
    await page.waitForFunction(() => document.body.dataset.ready === "true", null, { timeout: 120_000 });
    const r = await page.evaluate(async () => {
      const k = (window as any).kurven;
      if (!k.usesGPU) throw new Error(`no WebGPU: ${k.fallbackReason}`);
      await k.settled();
      const { pixels } = await k.draw("rest", true);
      const again = await k.redrawOffscreen();
      let differ = 0;
      let dark = 0;
      for (let i = 0; i < pixels.data.length; i++) if (pixels.data[i] !== again.data[i]) differ++;
      for (let i = 0; i < pixels.data.length; i += 4) if (pixels.data[i] < 128) dark++;
      return { width: pixels.width, height: pixels.height, same: pixels.width === again.width && pixels.height === again.height, differ, dark };
    });
    const line = `${info.project.name}: presented ${r.width}x${r.height}, ${r.dark} dark pixels, ${r.differ} bytes differ from offscreen`;
    console.log(line);
    info.annotations.push({ type: "presentation", description: line });
    expect(r.same).toBe(true);
    expect(r.dark).toBeGreaterThan(1000);
    expect(r.differ, "the canvas's pixels are the offscreen target's").toBe(0);
  } catch (error) {
    if (!allowed) throw error;
    const line = `${info.project.name}: presenting failed on Linux, allowed: ${String(error).split("\n")[0]}`;
    console.log(line);
    info.annotations.push({ type: "allowed failure", description: line });
  }
});
