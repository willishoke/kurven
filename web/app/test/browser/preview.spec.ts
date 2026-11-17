// The preview in Chromium (handoff Phase 7's gate): for a landscape, a
// surface and a polyhedron, the preview at rest against the judged strokes
// the engine bakes for the same frame, by Swift's coverage measure
// (coverage.js); the WGSL structs read back as written; the page without
// WebGPU drawing the judged strokes; and the frame times, measured.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { expect, type Page, test } from "@playwright/test";

const coverage = fileURLToPath(new URL("./coverage.js", import.meta.url));
const inputs = fileURLToPath(new URL("../../../../crates/fixtures/swift/session/inputs/", import.meta.url));

/** The three scenes, each at the camera it opens with, at Swift's test's
 * viewport, with the slope allowance off (the bake's predicate exactly). */
const scenes = [
  { name: "gamma", query: "landscape=gamma" },
  { name: "cn", query: "landscape=cn" },
  { name: "sn", query: "surface=sn" },
  { name: "frame", query: "", bundle: "frame.kurven" },
] as const;

async function load(page: Page, query: string, bundle?: string): Promise<void> {
  page.on("console", (m) => {
    if (m.type() === "error") console.log(`console: ${m.text()}`);
  });
  page.on("pageerror", (e) => console.log(`page error: ${e.message}`));
  await page.goto(`/?size=900x700&slope=0${query ? `&${query}` : ""}`);
  await page.waitForFunction(() => document.body.dataset.ready === "true", null, { timeout: 120_000 });
  await page.addScriptTag({ path: coverage });
  if (bundle) {
    const manifest = readFileSync(`${inputs}${bundle}/manifest.json`, "utf8");
    await page.evaluate(
      async ([name, text]) => {
        const k = (window as any).kurven;
        const bytes = new TextEncoder().encode(text);
        await k.open({ name, files: [{ path: "manifest.json", bytes: bytes.buffer }] });
      },
      [bundle, manifest] as const,
    );
  }
}

interface Measured {
  usesGPU: boolean;
  fallbackReason: string | null;
  baked: number;
  recall: number;
  stroked: number;
  precision: number;
  control: { pixels: number; stray: number };
  rest: { engineMs: number; drawMs: number; kind: string };
}

/** Draw the frame at rest, read it, and hold it to the judged strokes. */
async function measure(page: Page): Promise<Measured> {
  return page.evaluate(async () => {
    const k = (window as any).kurven;
    const C = (globalThis as any).kurvenCoverage;
    await k.settled();
    const { timing, pixels } = await k.draw("rest", true);
    // Without WebGPU the frame at rest is the judged strokes: held to their
    // own rasterization, it tests the drawing, not the bake twice.
    const bake = k.view.judged
      ? { reply: k.view.judged.judged, arrays: k.view.judged.arrays }
      : await k.request({ kind: "bake" });
    if (bake.reply.ok === false) throw new Error(`bake: ${bake.reply.error}`);
    const frame = bake.reply.frame;
    const judged = C.judgedMask(frame, bake.reply, bake.arrays);
    const preview = C.strokeMask(pixels);
    const m = C.measure(preview, judged, pixels.width, pixels.height);
    // The control: every curve of the frame, drawn with no test at all.
    const drawn = await k.request({ kind: "frame", draft: true });
    const preview0 = await k.request({ kind: "preview", draft: true });
    const view = preview0.reply.uniforms.view;
    const control = C.stray(C.unclippedMask(frame, view, drawn.reply, drawn.arrays), judged, pixels.width, pixels.height);
    return { usesGPU: k.usesGPU, fallbackReason: k.fallbackReason, ...m, control, rest: timing };
  });
}

for (const scene of scenes) {
  test(`the preview at rest draws what the judged strokes draw: ${scene.name}`, async ({ page }, info) => {
    await load(page, scene.query, "bundle" in scene ? scene.bundle : undefined);
    const m = await measure(page);
    const line =
      `${info.project.name} ${scene.name}: recall ${(100 * m.recall).toFixed(2)}% of ${m.baked} judged pixels, ` +
      `precision ${(100 * m.precision).toFixed(2)}% of ${m.stroked} drawn; control ${(100 * m.control.stray).toFixed(1)}% ` +
      `of ${m.control.pixels} stray; ${m.rest.kind} ${m.rest.engineMs.toFixed(0)} ms engine, ${m.rest.drawMs.toFixed(1)} ms drawing`;
    console.log(line);
    info.annotations.push({ type: "coverage", description: line });
    await page.locator("#preview").screenshot({ path: info.outputPath(`${scene.name}.png`) });
    expect(m.usesGPU, `WebGPU or not, as the project says (the page's fallback: ${m.fallbackReason})`).toBe(
      info.project.name !== "fallback",
    );
    expect(m.baked).toBeGreaterThan(5000);
    expect(m.recall, "the curves the judged strokes keep, the preview draws").toBeGreaterThan(0.99);
    expect(m.precision, "and the curves the preview draws, the judged strokes keep").toBeGreaterThan(0.99);
    // The control: drawn with no test at all, the same curves fail the
    // precision the preview is held to, so the measure tells the two apart.
    // Swift asks a fifth of them to be stray on its torus; a landscape's
    // hidden curves run within two pixels of visible ones far more often
    // (crates/FINDINGS.md, "The browser lane's gate"), and sn and the frame
    // meet the fifth as well.
    expect(m.control.stray, "while the same curves drawn unclipped are not").toBeGreaterThan(1 - 0.99);
    if (scene.name === "sn" || scene.name === "frame") expect(m.control.stray).toBeGreaterThan(0.2);
  });
}

test("the structs the page writes are the ones the shaders read", async ({ page }, info) => {
  test.skip(info.project.name === "fallback", "no WebGPU");
  await load(page, "");
  const wrong = await page.evaluate(() => (window as any).kurven.probe());
  expect(wrong).toEqual([]);
});

test("without WebGPU the page draws the judged strokes", async ({ page }, info) => {
  test.skip(info.project.name !== "fallback", "the fallback's");
  await load(page, "landscape=gamma");
  const why = await page.evaluate(() => (window as any).kurven.fallbackReason);
  console.log(`fallback: ${why}`);
  expect(why).toContain("adapter");
  await page.evaluate(() => (window as any).kurven.settled());
  await expect(page.locator("#status")).toContainText("judged strokes only");
});

test("frame times: drafts while the camera moves, and the frame at rest", async ({ page }, info) => {
  test.skip(info.project.name === "fallback", "no WebGPU");
  for (const scene of ["landscape=cn", "surface=sn"]) {
    await load(page, scene);
    const t = await page.evaluate(async () => {
      const k = (window as any).kurven;
      await k.settled();
      const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] ?? 0;
      const drafts: { engineMs: number; drawMs: number; totalMs: number }[] = [];
      const rests: { engineMs: number; drawMs: number; totalMs: number }[] = [];
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 8; j++) {
          await k.send({ kind: "gesture", gesture: { kind: "orbit", delta: [6, 1] } });
          drafts.push((await k.draw("draft")).timing);
        }
        rests.push((await k.draw("rest")).timing);
      }
      const of = (xs: typeof drafts) => ({
        engineMs: median(xs.map((x) => x.engineMs)),
        drawMs: median(xs.map((x) => x.drawMs)),
        totalMs: median(xs.map((x) => x.totalMs)),
      });
      return { title: k.state().title, draft: of(drafts), rest: of(rests), held: k.held() };
    });
    const line =
      `${info.project.name} ${t.title}: draft ${t.draft.totalMs.toFixed(1)} ms (engine ${t.draft.engineMs.toFixed(1)}, ` +
      `GPU ${t.draft.drawMs.toFixed(1)}), rest ${t.rest.totalMs.toFixed(0)} ms (engine ${t.rest.engineMs.toFixed(0)}, ` +
      `GPU ${t.rest.drawMs.toFixed(1)}); ${(t.held.bytes / 1e6).toFixed(1)} MB held`;
    console.log(line);
    info.annotations.push({ type: "timing", description: line });
  }
});
