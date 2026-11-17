// Swift's two preview suites of a parametric surface (Sources/kurven-test/
// Surfaces.swift, `surfacePreviewTests` and `edgeOnPreviewTests`), which
// need a GPU and so run here rather than in kurven-surfaces' tests, with
// Swift's scenes, cameras and thresholds. Each scene is a surface bundle,
// its manifest the session fixtures' with Swift's request and camera put in.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { expect, type Page, test } from "@playwright/test";

const coverage = fileURLToPath(new URL("./coverage.js", import.meta.url));
const inputs = fileURLToPath(new URL("../../../../crates/fixtures/swift/session/inputs/", import.meta.url));

test.beforeEach(({}, info) => {
  test.skip(info.project.name === "fallback", "the GPU preview's");
});

async function open(page: Page, size: string, name: string, manifest: object, project: string): Promise<void> {
  page.on("console", (m) => {
    if (m.type() === "error") console.log(`console: ${m.text()}`);
  });
  // SwiftShader draws offscreen (preview.spec.ts says why).
  await page.goto(`/?size=${size}&slope=0${project === "swiftshader" ? "&offscreen=1" : ""}`);
  await page.waitForFunction(() => document.body.dataset.ready === "true", null, { timeout: 120_000 });
  await page.addScriptTag({ path: coverage });
  await page.evaluate(
    async ([name, text]) => {
      const k = (window as any).kurven;
      const bytes = new TextEncoder().encode(text);
      const state = await k.open({ name, files: [{ path: "manifest.json", bytes: bytes.buffer }] });
      if (state.kind !== "surface") throw new Error(`opened ${state.kind}: ${state.failure ?? ""}`);
      await k.settled();
    },
    [name, JSON.stringify(manifest)] as const,
  );
}

function fixture(name: string): any {
  return JSON.parse(readFileSync(`${inputs}${name}.kurven/manifest.json`, "utf8"));
}

/** "preview: a parametric surface draws what its bake draws": the round
 * torus (512 by 256) with 24 by 12 coordinate curves and its folds, under an
 * oblique camera, shear and all; at rest and in a draft, the judged strokes
 * rasterized into the preview's pixels against the preview's. */
test("a parametric surface's preview draws what its bake draws", async ({ page }, info) => {
  const m = fixture("torus");
  m.surface.grid = 512;
  m.surface.lines = { u: 24, v: 12 };
  m.layers[0].source = { kind: "coordinateCurves", u: 24, v: 12 };
  m.layers[1].width = 0.3;
  m.presets[0].projection = { flipX: false, shear: 0.5, xAngle: -50, yScale: null, zAngle: 25 };
  m.presets[0].depthTolerance = 0;
  await open(page, "900x700", "torus-lines.kurven", m, info.project.name);
  const result = await page.evaluate(async () => {
    const k = (window as any).kurven;
    const C = (globalThis as any).kurvenCoverage;
    const bake = await k.request({ kind: "bake" });
    const frame = bake.reply.frame;
    const judged = C.judgedMask(frame, bake.reply, bake.arrays);
    const out: Record<string, unknown> = {};
    for (const wish of ["rest", "draft"]) {
      const { pixels } = await k.draw(wish, true);
      out[wish] = C.measure(C.strokeMask(pixels), judged, pixels.width, pixels.height);
    }
    // The control: the same lines drawn without a visibility test.
    const drawn = await k.request({ kind: "frame", draft: true });
    const view = k.view.frame.uniforms.view;
    out.control = C.stray(C.unclippedMask(frame, view, drawn.reply, drawn.arrays), judged, frame.cols, frame.rows);
    return out as any;
  });
  for (const wish of ["rest", "draft"] as const) {
    const r = result[wish];
    const line = `${info.project.name} torus ${wish}: recall ${(100 * r.recall).toFixed(2)}% of ${r.baked}, precision ${(100 * r.precision).toFixed(2)}% of ${r.stroked}`;
    console.log(line);
    info.annotations.push({ type: "coverage", description: line });
    expect(r.baked).toBeGreaterThan(5000);
    expect(r.recall, `${wish}: the curves the bake keeps, the preview draws`).toBeGreaterThan(0.99);
    expect(r.precision, `${wish}: and the curves the preview draws, the bake keeps`).toBeGreaterThan(0.99);
  }
  console.log(`${info.project.name} torus control: ${(100 * result.control.stray).toFixed(1)}% of ${result.control.pixels} stray`);
  expect(result.control.stray, "while the same lines drawn unclipped visibly are not").toBeGreaterThan(0.2);
});

/** "preview: a torus nearly edge-on keeps its outline whole at rest": sn on
 * its torus seen from half a degree off its equatorial plane, the folds
 * alone. At rest the window draws what sphere tracing keeps, as the bake
 * does; the draft, tested per fragment against the coordinate image, is held
 * to showing the difference, or this proves nothing. */
test("a torus nearly edge-on keeps its outline whole at rest", async ({ page }, info) => {
  const m = fixture("sn");
  m.surface.grid = 600;
  m.surface.shape.grid = 200;
  m.presets[0].projection.xAngle = -89.5;
  m.presets[0].projection.zAngle = 118.2;
  await open(page, "1400x900", "sn-edge-on.kurven", m, info.project.name);
  const r = await page.evaluate(async () => {
    const k = (window as any).kurven;
    const C = (globalThis as any).kurvenCoverage;
    const layers = k.state().layers.map((l: { name: string }) => l.name);
    const folds = layers.indexOf("folds");
    for (let i = 0; i < layers.length; i++) if (i !== folds) await k.send({ kind: "hide", layer: i });
    const bake = await k.request({ kind: "bake" });
    const frame = bake.reply.frame;
    // A surface's strokes are its layers in order, the folds among them.
    const judged = C.judgedMask(frame, bake.reply, bake.arrays, folds);
    const recall: Record<string, number> = {};
    let baked = 0;
    for (const wish of ["rest", "draft"]) {
      const { pixels } = await k.draw(wish, true);
      const m = C.measure(C.strokeMask(pixels), judged, pixels.width, pixels.height);
      recall[wish] = m.recall;
      baked = m.baked;
    }
    return { baked, rest: recall.rest ?? 0, draft: recall.draft ?? 0 };
  });
  const line = `${info.project.name} sn edge-on: at rest ${(100 * r.rest).toFixed(3)}% of ${r.baked} baked pixels, the draft ${(100 * r.draft).toFixed(3)}%`;
  console.log(line);
  info.annotations.push({ type: "coverage", description: line });
  expect(r.baked).toBeGreaterThan(2000);
  expect(r.rest, "at rest the preview draws all of the outline the bake keeps").toBeGreaterThan(0.999);
  expect(r.draft, "while the draft, tested per fragment, breaks it").toBeLessThan(r.rest - 0.002);
});

/** "metal: the GPU coordinate pass is the CPU rasterizer's": the WebGPU
 * depth and coordinate passes held to the bake's CPU rasterizer
 * (`SurfaceImage::rasterize`), on Swift's two tori (R/r = 2 and 1.25,
 * 384 by 192) at 800 pixels, from two cameras, one sheared. Swift's cameras
 * are built in its test from an elevation, an azimuth and a shear; here they
 * are a bundle's oblique presets, which turn the same three ways. */
test("the GPU coordinate pass is the CPU rasterizer's", async ({ page }, info) => {
  // Not met on SwiftShader, and not loosened for it: its rasterizer
  // interpolates view depth up to 3e-3 of the span off the CPU's at about
  // 1.5% of the covered pixels, inside triangles, where Metal is within
  // 1e-4 at all but a few dozen edge pixels (crates/FINDINGS.md, "Swift's
  // GPU suites in the browser lane").
  test.skip(info.project.name === "swiftshader", "SwiftShader's interpolation misses Swift's 1e-4 of the span");
  for (const [major, minor] of [
    [2, 1],
    [1.25, 1],
  ]) {
    for (const [name, projection] of [
      ["elevation 60°", { flipX: false, shear: 0, xAngle: -60, yScale: null, zAngle: 30 }],
      ["sheared", { flipX: false, shear: 0.5, xAngle: -35, yScale: null, zAngle: -20 }],
    ] as const) {
      const m = fixture("torus");
      m.surface.grid = 384;
      m.surface.shape = { kind: "torus", major, minor };
      m.presets[0].projection = projection;
      await open(page, "900x700", `torus-${major}-${name}.kurven`, m, info.project.name);
      const r = await page.evaluate(async () => {
        const k = (window as any).kurven;
        const C = (globalThis as any).kurvenCoverage;
        const image = await k.surfaceImage(800);
        const surface = k.view.renderer.resources.json;
        const { rows, cols } = image.frame;
        return C.compareImages(image.gpu, image.cpu, rows, cols, image.span, surface);
      });
      const label = `R/r = ${major / minor}, ${name}`;
      const line =
        `${info.project.name} ${label}: ${r.strays} of ${r.covered} pixels differ, ${r.strays - r.onEdge} off an edge; ` +
        `elsewhere depth within ${r.worstDepth.toExponential(2)}; coordinates within ${r.worstCoord.toFixed(4)} cells ` +
        `at ${r.checked} pixels, ${r.coordWrong} past one`;
      console.log(line);
      info.annotations.push({ type: "coordinate pass", description: line });
      expect(r.strays, `${label}: coverage and depth differ only on edges`).toBe(r.onEdge);
      expect(r.strays * 200, `${label}: on under half a percent of the pixels`).toBeLessThan(r.covered);
      expect(r.coordWrong, `${label}: and the coordinates are the CPU's front-most point's`).toBe(0);
    }
  }
});
