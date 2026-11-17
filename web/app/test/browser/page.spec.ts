// The page (HANDOFF-web-port.md, Phase 8, steps 1 to 3), driven as a user
// drives it: typing a formula, a bad one marked where the engine says;
// picking from the catalog; orbiting, zooming toward the cursor, panning and
// fitting with the mouse and the keyboard; dragging a window thumb and the
// truncation height; hiding a layer and counting levels; switching modes and
// presets. Then Export: the SVG the page downloads is held byte for byte to
// `kurven-cli session` running the same requests natively and baking what
// they left; and the URL, opened in a fresh page at another size, gives the
// same document and the same SVG bytes.

import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { expect, type Page, type TestInfo, test } from "@playwright/test";

const root = fileURLToPath(new URL("../../../../", import.meta.url));
const cli = `${root}crates/target/release/kurven-cli`;
const language = JSON.parse(readFileSync(`${root}tests/fixtures/expr/language.json`, "utf8")) as {
  errors: { text: string; position: number; length: number; message: string }[];
};

/** The page's address for a project: SwiftShader draws offscreen
 * (crates/FINDINGS.md, "presenting does not"). */
function at(project: string, query: Record<string, string> = {}): string {
  const q = new URLSearchParams(query);
  if (project === "swiftshader") q.set("offscreen", "1");
  const text = q.toString();
  return text ? `/?${text}` : "/";
}

async function load(page: Page, url: string): Promise<void> {
  page.on("console", (m) => {
    if (m.type() === "error") console.log(`console: ${m.text()}`);
  });
  page.on("pageerror", (e) => console.log(`page error: ${e.message}`));
  await page.goto(url);
  await page.waitForFunction(() => document.body.dataset.ready === "true", null, { timeout: 180_000 });
  await idle(page);
}

/** Every gesture and edit answered, every frame wished drawn. */
async function idle(page: Page): Promise<void> {
  await page.evaluate(() => (window as any).kurven.settled());
  await page.waitForFunction(() => document.body.dataset.busy === "false", null, { timeout: 300_000 });
  await page.evaluate(() => (window as any).kurven.settled());
}

const state = (page: Page) => page.evaluate(() => (window as any).kurven.state());

/** Drag a range input's thumb from its value by `dx` pixels. */
async function dragThumb(page: Page, selector: string, dx: number, steps = 8): Promise<void> {
  const el = page.locator(selector);
  const box = (await el.boundingBox())!;
  const frac = await el.evaluate((i: HTMLInputElement) => (Number(i.value) - Number(i.min)) / (Number(i.max) - Number(i.min)));
  const thumb = 14; // the thumb's width, style.css
  const x = box.x + thumb / 2 + frac * (box.width - thumb);
  const y = box.y + box.height / 2;
  await page.mouse.move(x, y);
  await page.mouse.down();
  for (let k = 1; k <= steps; k++) await page.mouse.move(x + (dx * k) / steps, y);
  await page.mouse.up();
}

async function exported(page: Page): Promise<Buffer> {
  const [download] = await Promise.all([page.waitForEvent("download", { timeout: 300_000 }), page.click("#export")]);
  return readFileSync((await download.path())!);
}

/** What `kurven-cli session` bakes from the page's journal: the requests
 * that made its document, run natively. */
function native(journal: unknown[], info: TestInfo, name: string): Buffer {
  const requests = info.outputPath(`${name}.jsonl`);
  const svg = info.outputPath(`${name}.svg`);
  writeFileSync(requests, journal.map((r) => JSON.stringify(r)).join("\n") + "\n");
  execFileSync(cli, ["session", requests, "-o", svg], { stdio: ["ignore", "inherit", "inherit"] });
  return readFileSync(svg);
}

test.describe.configure({ mode: "serial" });

test("the formula: an error marked at the engine's position, an expression sampled", async ({ page }, info) => {
  await load(page, at(info.project.name));
  const field = page.locator("#formula");
  // Language.json's frozen positions: at the end, a name, a whole call.
  const cases = language.errors.filter((e) => ["sin(z", "foo(z)", "gamma(z, 2)", "z +"].includes(e.text));
  expect(cases.length).toBe(4);
  for (const c of cases) {
    await field.fill(c.text);
    await field.press("Enter");
    const marker = page.locator("#formula-marker");
    await expect(marker).toBeVisible();
    expect(Number(await marker.getAttribute("data-position"))).toBe(c.position);
    expect(Number(await marker.getAttribute("data-length"))).toBe(c.length);
    const chars = Array.from(c.text);
    const fault = chars.slice(c.position, c.position + c.length).join("") || " ";
    expect(await marker.textContent()).toBe(fault);
    await expect(page.locator("#formula-error")).toContainText(c.message);
    await expect(field).toHaveAttribute("aria-invalid", "true");
    // The mark sits over those characters: its left edge is where the
    // field's text puts the first of them.
    const where = await page.evaluate((prefix) => {
      const input = document.querySelector<HTMLInputElement>("#formula")!;
      const mark = document.querySelector("#formula-marker")!;
      const style = getComputedStyle(input);
      const ctx = document.createElement("canvas").getContext("2d")!;
      ctx.font = style.font;
      const origin = input.getBoundingClientRect().left + parseFloat(style.borderLeftWidth) + parseFloat(style.paddingLeft);
      return { mark: mark.getBoundingClientRect().left, text: origin + ctx.measureText(prefix).width };
    }, chars.slice(0, c.position).join(""));
    expect(Math.abs(where.mark - where.text)).toBeLessThan(1.5);
    // Refused before the document: nothing sampled, nothing to undo.
    expect((await state(page)).landscape.expression).toBe("gamma(z)");
  }
  // A good expression: sampled, drawn, and the dropdown says typed.
  await field.fill("z^3 - 1");
  await field.press("Enter");
  await idle(page);
  await expect(page.locator("#formula-marker")).toHaveCount(0);
  const s = await state(page);
  expect(s.landscape.expression).toBe("z^3 - 1");
  expect(s.landscape.name).toBe("");
  await expect(page.locator("#function option:checked")).toHaveText("z^3 - 1 (typed)");
  expect(s.undo).toEqual({ can: true, name: "Expression" });
});

test("the catalog, the canvas, the window, the truncation, layers, levels, modes and presets", async ({ page }, info) => {
  await load(page, at(info.project.name));
  const gpu = await page.evaluate(() => (window as any).kurven.usesGPU);

  // The catalog.
  await page.selectOption("#function", "sin");
  await idle(page);
  let s = await state(page);
  expect(s.landscape.name).toBe("sin");
  expect(await page.inputValue("#formula")).toBe(s.landscape.expression);
  const opened = s.camera.exact;

  // Orbit: a drag across the canvas turns the camera.
  const canvas = page.locator("#preview");
  const box = (await canvas.boundingBox())!;
  const mid = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
  await page.mouse.move(mid.x, mid.y);
  await page.mouse.down();
  for (let k = 1; k <= 6; k++) await page.mouse.move(mid.x + 10 * k, mid.y + 3 * k);
  await page.mouse.up();
  await idle(page);
  s = await state(page);
  expect(s.camera.exact.azimuth).not.toBe(opened.azimuth);
  expect(s.camera.exact.elevation).not.toBe(opened.elevation);
  const orbited = s.camera.exact;

  // Zoom toward the cursor: in at a corner, the centre moves toward it.
  // Whole CSS pixels: a mouse event's client point is one.
  const cursor = { x: Math.round(box.x + box.width * 0.8), y: Math.round(box.y + box.height * 0.5) };
  await page.mouse.move(cursor.x, cursor.y);
  await page.mouse.wheel(0, -200);
  await idle(page);
  s = await state(page);
  expect(s.camera.exact.unitsPerPixel).toBeLessThan(orbited.unitsPerPixel);
  expect(s.camera.exact.azimuth).toBe(orbited.azimuth);
  const zoomed = s.camera.exact;
  // ...the point under the cursor held: the page x of the cursor is the same
  // before and after (center + (px - w/2) · upp, per axis).
  const px = (cursor.x - box.x) * (await page.evaluate(() => window.devicePixelRatio));
  const w = s.viewport[0];
  const before = orbited.center[0] + (px - w / 2) * orbited.unitsPerPixel;
  const after = zoomed.center[0] + (px - w / 2) * zoomed.unitsPerPixel;
  expect(Math.abs(after - before)).toBeLessThan(1e-9 * Math.max(1, Math.abs(before)));

  // Pan: shift-drag moves the paper, not the camera.
  await page.keyboard.down("Shift");
  await page.mouse.move(mid.x, mid.y);
  await page.mouse.down();
  await page.mouse.move(mid.x - 40, mid.y + 20, { steps: 4 });
  await page.mouse.up();
  await page.keyboard.up("Shift");
  await idle(page);
  s = await state(page);
  expect(s.camera.exact.azimuth).toBe(zoomed.azimuth);
  expect(s.camera.exact.center).not.toEqual(zoomed.center);

  // From the keyboard: the canvas focused, an arrow orbits, f fits.
  await canvas.focus();
  await page.keyboard.press("ArrowLeft");
  await idle(page);
  expect((await state(page)).camera.exact.azimuth).not.toBe(zoomed.azimuth);
  await page.keyboard.press("f");
  await idle(page);
  const fitted = (await state(page)).camera.exact;
  await page.click("#fit");
  await idle(page);
  expect((await state(page)).camera.exact).toEqual(fitted);

  // A preset: back to the camera it opened at.
  await page.click('#presets button[data-preset="0"]');
  await idle(page);
  s = await state(page);
  expect(s.camera.exact.azimuth).toBe(opened.azimuth);
  expect(s.camera.exact.elevation).toBe(opened.elevation);

  // A window thumb dragged: the high end of Re comes in, sampled when let
  // go, one step to undo.
  const re = s.landscape.domain.real;
  const steps = s.undo.can ? 1 : 0;
  await dragThumb(page, "#re-hi", -12);
  await idle(page);
  s = await state(page);
  expect(s.landscape.domain.real.hi).toBeLessThan(re.hi);
  expect(s.landscape.domain.real.lo).toBe(re.lo);
  expect(s.landscape.domain.real.hi).toBeGreaterThan(re.lo);
  expect(s.undo.can).toBe(true);
  expect(Number(await page.inputValue("#re-hi"))).toBe(s.landscape.domain.real.hi);
  void steps;

  // A typed end: an edit of its own.
  const reHiField = page.locator('input[type="number"][aria-label="Re, high end"]');
  await reHiField.fill("1.5");
  await reHiField.press("Enter");
  await idle(page);
  expect((await state(page)).landscape.domain.real.hi).toBe(1.5);

  // The truncation height dragged: drafts while it moves (with WebGPU),
  // the truncation the hand left.
  s = await state(page);
  expect(s.truncation.kind).toBe("uniform");
  const height = s.truncation.height;
  const drafts = await page.evaluate(() => (window as any).kurven.view.timings.filter((t: any) => t.kind === "draft").length);
  await dragThumb(page, "#truncation-height", -40, 10);
  await idle(page);
  s = await state(page);
  expect(s.truncation.kind).toBe("uniform");
  expect(s.truncation.height).toBeLessThan(height);
  expect(Number(await page.inputValue('input[type="number"][aria-label="Truncation height"]'))).toBeCloseTo(s.truncation.height, 3);
  const draftsAfter = await page.evaluate(() => (window as any).kurven.view.timings.filter((t: any) => t.kind === "draft").length);
  if (gpu) expect(draftsAfter).toBeGreaterThan(drafts);
  expect(s.undo.name).toBe("Truncation");

  // None, then a height again.
  await page.locator("#truncation-kind label", { hasText: "None" }).click();
  await idle(page);
  expect((await state(page)).truncation.kind).toBe("none");
  await page.locator("#truncation-kind label", { hasText: "At a height" }).click();
  await idle(page);
  expect((await state(page)).truncation).toEqual({ kind: "uniform", height: 5 });

  // A layer hidden, and shown again.
  const folds = page.locator('li[data-layer="folds"] input[type="checkbox"]');
  await folds.uncheck();
  await idle(page);
  expect((await state(page)).layers.find((l: any) => l.name === "folds").hidden).toBe(true);
  await folds.check();
  await idle(page);
  expect((await state(page)).layers.find((l: any) => l.name === "folds").hidden).toBe(false);

  // Level counts: stepped from the keyboard, then dragged.
  const levels = page.locator('[data-levels="mag_major"]');
  const count = (await state(page)).layers.find((l: any) => l.name === "mag_major").levels;
  await levels.focus();
  await page.keyboard.press("ArrowRight");
  await idle(page);
  await page.keyboard.press("ArrowRight");
  await idle(page);
  s = await state(page);
  expect(s.layers.find((l: any) => l.name === "mag_major").levels).toBe(count + 2);
  expect(s.levelCounts.mag_major).toBe(count + 2);
  await dragThumb(page, '[data-levels="mag_major"]', 30);
  await idle(page);
  s = await state(page);
  expect(s.layers.find((l: any) => l.name === "mag_major").levels).toBeGreaterThan(count + 2);
  expect(s.undo.name).toBe("Levels of mag_major");

  // The three modes: by the panel and by the keys, each drawn.
  for (const [how, mode] of [
    ["Shaded", "shaded"],
    ["3", "depth"],
    ["Drawing", "drawing"],
  ] as const) {
    if (how.length === 1) {
      await canvas.focus();
      await page.keyboard.press(how);
    } else {
      await page.locator("#mode label", { hasText: how }).click();
    }
    await idle(page);
    expect((await state(page)).mode).toBe(mode);
    if (gpu) expect(await page.evaluate(() => (window as any).kurven.view.frame.mode)).toBe(mode);
  }

  // Undo from the keyboard: the levels drag, one step.
  await canvas.focus();
  await page.keyboard.press("ControlOrMeta+z");
  await idle(page);
  expect((await state(page)).layers.find((l: any) => l.name === "mag_major").levels).toBe(count + 2);
});

test("Export is the native bake's bytes, and the URL reproduces the drawing", async ({ page, context }, info) => {
  await load(page, at(info.project.name));
  // An edited document: a typed expression, a window end, the truncation,
  // a level count, a hidden layer, a mode, and the camera moved.
  const field = page.locator("#formula");
  await field.fill("gamma(z) + 1/z");
  await field.press("Enter");
  await idle(page);
  const reLo = page.locator('input[type="number"][aria-label="Re, low end"]');
  await reLo.fill("-3.25");
  await reLo.press("Enter");
  await idle(page);
  const heightField = page.locator('input[type="number"][aria-label="Truncation height"]');
  await heightField.fill("3.5");
  await heightField.press("Enter");
  await idle(page);
  await page.locator('[data-levels="mag_major"]').focus();
  await page.keyboard.press("ArrowLeft");
  await idle(page);
  await page.locator('li[data-layer="wall_hatch"] input[type="checkbox"]').uncheck();
  await idle(page);
  await page.locator("#mode label", { hasText: "Shaded" }).click();
  await idle(page);
  const canvas = page.locator("#preview");
  const box = (await canvas.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 45, box.y + box.height / 2 - 15, { steps: 5 });
  await page.mouse.up();
  await page.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.6);
  await page.mouse.wheel(0, -120);
  await idle(page);

  // Export: the judged strokes, the bytes kurven-cli writes natively for
  // the same requests.
  const svg = await exported(page);
  expect(svg.subarray(0, 5).toString()).toBe("<?xml");
  const journal = await page.evaluate(() => (window as any).kurven.journal());
  const reference = native(journal, info, "edited");
  expect(svg.length).toBe(reference.length);
  expect(svg.equals(reference)).toBe(true);
  // Hidden layers and the mode are the preview's: the bake draws every layer.
  await expect(page.locator("#export-status")).toContainText("paths");

  // The URL: the state, whole.
  const url = await page.evaluate(() => (window as any).kurven.linked());
  expect(url).toMatch(/#k1j?\./);
  const before = await page.evaluate(async () => {
    const k = (window as any).kurven;
    const edited = (await k.request({ kind: "edited" })).reply.edited;
    const s = k.state();
    return { edited, camera: s.camera.exact, layers: s.layers, mode: s.mode, depthTolerance: s.depthTolerance, bakeResolution: s.bakeResolution };
  });

  // A fresh page, at another size, from the URL alone.
  const fresh = await context.newPage();
  await fresh.setViewportSize({ width: 1180, height: 760 });
  await load(fresh, url);
  const after = await fresh.evaluate(async () => {
    const k = (window as any).kurven;
    const edited = (await k.request({ kind: "edited" })).reply.edited;
    const s = k.state();
    return { edited, camera: s.camera.exact, layers: s.layers, mode: s.mode, depthTolerance: s.depthTolerance, bakeResolution: s.bakeResolution };
  });
  expect(after).toEqual(before);
  expect(await fresh.inputValue("#formula")).toBe("gamma(z) + 1/z");
  const again = await exported(fresh);
  expect(again.equals(svg)).toBe(true);
  // ...and natively, from the requests the link made.
  const restored = await fresh.evaluate(() => (window as any).kurven.journal());
  expect(restored.some((r: any) => r.kind === "restore")).toBe(true);
  expect(native(restored, info, "restored").equals(svg)).toBe(true);
  await fresh.close();
});

test("a slow frame at rest: the page says so, stays live, and drafts follow it", async ({ page }, info) => {
  test.skip(info.project.name === "fallback", "without WebGPU the frame at rest is the bake; the export test covers it");
  await load(page, at(info.project.name, { landscape: "cn" }));
  // Move, then hold still: the frame at rest starts, and the indicator says so.
  const canvas = page.locator("#preview");
  const box = (await canvas.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 30, box.y + box.height / 2, { steps: 3 });
  await page.mouse.up();
  await page.waitForFunction(() => (window as any).kurven.view.working === "rest", null, { timeout: 60_000 });
  await expect(page.locator("#indicator")).toHaveAttribute("data-busy", "true");
  await expect(page.locator("#indicator-text")).toContainText("drawing at rest");
  // The page answers meanwhile: a control responds without the engine.
  const t0 = Date.now();
  await page.locator("#formula").fill("cn(z, 0.5)");
  expect(Date.now() - t0).toBeLessThan(1000);
  await page.locator("#formula").fill("");
  // A gesture given now is coalesced, and its draft is drawn once the rest
  // frame lets the engine go.
  const timings = await page.evaluate(() => (window as any).kurven.view.timings.length);
  const asked = Date.now();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 - 30, box.y + box.height / 2, { steps: 6 });
  await page.mouse.up();
  await page.waitForFunction(
    (n) => (window as any).kurven.view.timings.slice(n).some((t: any) => t.kind === "draft"),
    timings,
    { timeout: 60_000 },
  );
  console.log(`${info.project.name}: a draft asked during cn's frame at rest was drawn ${Date.now() - asked} ms later`);
  await idle(page);
  await expect(page.locator("#indicator-text")).toContainText("at rest");
});

test("at phone width: the drawing the full width, the controls under it, nothing to scroll sideways", async ({ page }, info) => {
  await page.setViewportSize({ width: 375, height: 740 });
  await load(page, at(info.project.name));
  const widths = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, client: document.documentElement.clientWidth }));
  expect(widths.scroll).toBeLessThanOrEqual(widths.client);
  const canvas = (await page.locator("#preview").boundingBox())!;
  const panel = (await page.locator(".panel").boundingBox())!;
  expect(canvas.width).toBeGreaterThan(370);
  expect(panel.y).toBeGreaterThanOrEqual(canvas.y + canvas.height - 1);
  for (const el of await page.locator(".panel input, .panel select, .panel button").all()) {
    const b = await el.boundingBox();
    if (b) expect(b.x + b.width).toBeLessThanOrEqual(375 + 0.5);
  }
  // The keyboard reaches the canvas and the controls in order.
  await page.locator("#preview").focus();
  await page.keyboard.press("Tab");
  expect(await page.evaluate(() => document.activeElement?.id)).toBe("function");
  await page.keyboard.press("Tab");
  expect(await page.evaluate(() => document.activeElement?.id)).toBe("formula");
});
