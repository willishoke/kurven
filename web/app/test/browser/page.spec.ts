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
  await spy(page);
  const start = await page.evaluate(() => ({
    camera: (window as any).kurven.state().camera.exact,
    given: (window as any).kurven.input.given.length,
  }));
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
  // The gestures given while the engine was held went in batches, and left
  // the camera each one alone would have, to the bit.
  const sent: string[] = await page.evaluate(() => (window as any).kurven.sent);
  expect(sent).toContain("gestures");
  const check = await page.evaluate(async (start) => {
    const k = (window as any).kurven;
    const ended = k.state().camera.exact;
    const given = k.input.given.slice(start.given);
    await k.request({ kind: "camera", ...start.camera });
    for (const gesture of given) await k.request({ kind: "gesture", gesture });
    const one = (await k.request({ kind: "state" })).reply.state.camera.exact;
    return { ended, one, count: given.length };
  }, start);
  expect(check.count).toBeGreaterThan(3);
  expect(check.one).toEqual(check.ended);
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
  // Browse..., under the dropdown as in LandscapeSection.swift (step 4).
  await page.keyboard.press("Tab");
  expect(await page.evaluate(() => document.activeElement?.id)).toBe("browse");
  await page.keyboard.press("Tab");
  expect(await page.evaluate(() => document.activeElement?.id)).toBe("formula");
});

/** A link's fragment for `link`, uncompressed (link.ts reads both). */
function fragment(link: unknown): string {
  return `#k1j.${Buffer.from(JSON.stringify({ v: 1, ...(link as object) })).toString("base64url")}`;
}

/** Record the kinds of request the page sends from now on. */
async function spy(page: Page): Promise<void> {
  await page.evaluate(() => {
    const k = (window as any).kurven;
    const sent: unknown[] = [];
    k.sent = sent;
    const request = k.client.request.bind(k.client);
    k.client.request = (r: any, o: any) => {
      sent.push(r.kind === "drag" ? `drag ${r.phase}` : r.kind);
      return request(r, o);
    };
  });
}

test("a link asking for more than the controls reach is refused, and the page says so", async ({ page }, info) => {
  // A document's link, made by the engine, then made to ask too much.
  await load(page, at(info.project.name));
  const good = await page.evaluate(async () => {
    const k = (window as any).kurven;
    const edited = (await k.request({ kind: "edited" })).reply.edited;
    return { edited, camera: k.state().camera.exact };
  });
  for (const [what, change] of [
    ["resolution", (l: any) => (l.edited.landscape.resolution = 1_000_000)],
    ["levels", (l: any) => (l.edited.levelCounts = { mag_major: 1_000_000_000 })],
  ] as const) {
    const link: any = { ...structuredClone(good), hidden: [], mode: "depth", depthTolerance: 0.02, bakeResolution: 4000, refine: false };
    change(link);
    const fresh = await page.context().newPage();
    await load(fresh, `${at(info.project.name)}${fragment(link)}`);
    await expect(fresh.locator("#notice")).toContainText("This link was refused");
    await expect(fresh.locator("#notice")).toContainText(what === "resolution" ? "Samples slider" : "level slider");
    // Nothing of it stays: the page opened its own landscape, at its own
    // settings, and the URL is that landscape's.
    const s = await state(fresh);
    expect(s.landscape.name).toBe("gamma");
    expect(s.mode).toBe("drawing");
    expect(s.refine).toBe(true);
    const url = await fresh.evaluate(() => (window as any).kurven.linked());
    expect(url).not.toContain(fragment(link));
    await fresh.close();
  }
});

test("a link pasted into the tab is opened, and not written over", async ({ page }, info) => {
  await load(page, at(info.project.name));
  await page.locator('[data-levels="mag_major"]').focus();
  await page.keyboard.press("ArrowLeft");
  await idle(page);
  const first = await page.evaluate(() => (window as any).kurven.linked());
  const edited = await page.evaluate(async () => (await (window as any).kurven.request({ kind: "edited" })).reply.edited);
  await page.selectOption("#function", "sin");
  await idle(page);
  const second = await page.evaluate(() => (window as any).kurven.linked());
  expect(second).not.toBe(first);
  // The first link, pasted.
  await page.evaluate((hash) => (location.hash = hash), new URL(first).hash);
  await page.waitForFunction(() => (window as any).kurven.state().landscape.name === "gamma", null, { timeout: 120_000 });
  await idle(page);
  const now = await page.evaluate(async () => (await (window as any).kurven.request({ kind: "edited" })).reply.edited);
  expect(now).toEqual(edited);
  expect(await page.evaluate(() => (window as any).kurven.linked())).toBe(first);
});

test("the formula keeps what is typed after return, and drops marks the text has left", async ({ page }, info) => {
  await load(page, at(info.project.name));
  const field = page.locator("#formula");
  // Typed on while the engine samples what return sent.
  await field.fill("z^2");
  await field.press("Enter");
  await page.keyboard.type(" + 1");
  await idle(page);
  // Every reply that renders the panel since (the link's `edited` among
  // them) has left the field alone.
  await page.evaluate(() => (window as any).kurven.linked());
  await page.evaluate(() => (window as any).kurven.send({ kind: "state" }));
  expect(await field.inputValue()).toBe("z^2 + 1");
  expect((await state(page)).landscape.expression).toBe("z^2");
  // A refusal's marks, then another landscape chosen: the field shows its
  // expression, and the marks of the refused text go with it.
  await field.fill("foo(z)");
  await field.press("Enter");
  await expect(page.locator("#formula-marker")).toBeVisible();
  await page.selectOption("#function", "sin");
  await idle(page);
  expect(await field.inputValue()).toBe((await state(page)).landscape.expression);
  await expect(page.locator("#formula-marker")).toHaveCount(0);
  await expect(page.locator("#formula-error")).toBeEmpty();
  await expect(field).not.toHaveAttribute("aria-invalid", "true");
});

test("the page's keys work wherever focus is but a text field", async ({ page }, info) => {
  await load(page, at(info.project.name));
  // On a layer's check box: 2 is the shaded preview.
  await page.locator('li[data-layer="folds"] input[type="checkbox"]').focus();
  await page.keyboard.press("2");
  await idle(page);
  expect((await state(page)).mode).toBe("shaded");
  // On a slider: its arrow is an edit, ⌘Z undoes it.
  const levels = page.locator('[data-levels="mag_major"]');
  const count = (await state(page)).layers.find((l: any) => l.name === "mag_major").levels;
  await levels.focus();
  await page.keyboard.press("ArrowRight");
  await idle(page);
  expect((await state(page)).layers.find((l: any) => l.name === "mag_major").levels).toBe(count + 1);
  await page.keyboard.press("ControlOrMeta+z");
  await idle(page);
  expect((await state(page)).layers.find((l: any) => l.name === "mag_major").levels).toBe(count);
  // On a radio button: 1 is the drawing.
  await page.locator('#mode input[value="depth"]').focus();
  await page.keyboard.press("1");
  await idle(page);
  expect((await state(page)).mode).toBe("drawing");
  // In the formula, the keys are the formula's.
  await page.locator("#formula").focus();
  await page.keyboard.press("3");
  await idle(page);
  expect((await state(page)).mode).toBe("drawing");
});

test("a slider is one drag at a time, and a window's thumbs do not cross", async ({ page }, info) => {
  await load(page, at(info.project.name));
  await spy(page);
  // A second pointer on a slider already held is no second drag.
  await page.evaluate(() => {
    const range = document.querySelector('[data-levels="mag_major"]')!;
    for (let id = 1; id <= 2; id++) range.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true, pointerId: id }));
    window.dispatchEvent(new PointerEvent("pointerup", { bubbles: true, pointerId: 1 }));
  });
  await idle(page);
  const sent: string[] = await page.evaluate(() => (window as any).kurven.sent);
  expect(sent.filter((k) => k === "drag begin").length).toBe(1);
  expect(sent.filter((k) => k === "drag end").length).toBe(1);
  // The low thumb of Re dragged far past the high one stops a fiftieth of
  // the reach short of it, and shows where it stopped.
  const before = (await state(page)).landscape.domain.real;
  await dragThumb(page, "#re-lo", 80, 10);
  await idle(page);
  const re = (await state(page)).landscape.domain.real;
  const reach = await page.evaluate(() => {
    const r = document.querySelector<HTMLInputElement>("#re-lo")!;
    return Number(r.max) - Number(r.min);
  });
  expect(re.hi).toBe(before.hi);
  expect(re.lo).toBeLessThan(re.hi);
  expect(re.hi - re.lo).toBeGreaterThanOrEqual(reach / 50 - 1e-12);
  expect(Number(await page.inputValue("#re-lo"))).toBe(re.lo);
  expect(Number(await page.inputValue("#re-lo"))).toBeLessThan(Number(await page.inputValue("#re-hi")));
});

/** Fail on any error the page throws or leaves unhandled. */
function strict(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  return errors;
}

/** gamma's link as the engine and the page write it, changed by `change`. */
async function gammaLink(page: Page, change: (link: any) => void): Promise<string> {
  const link = await page.evaluate(async () => {
    const k = (window as any).kurven;
    const edited = (await k.request({ kind: "edited" })).reply.edited;
    const s = k.state();
    return { edited, camera: s.camera.exact, hidden: [], mode: "drawing", depthTolerance: s.depthTolerance, bakeResolution: s.bakeResolution, refine: true };
  });
  change(link);
  return fragment(link);
}

test("malformed links are refused and said, and the page starts as it would", async ({ page }, info) => {
  await load(page, at(info.project.name));
  const { deflateRawSync } = await import("node:zlib");
  const bomb = `#k1.${deflateRawSync(Buffer.alloc(2_000_000, 0x20)).toString("base64url")}`;
  const cases: [string, string, string][] = [
    ["a zip bomb", bomb, "inflates past 64 KB"],
    ["a long fragment", `#k1j.${"A".repeat(20_000)}`, "longer than 16 KB"],
    ["not base64url", "#k1.%%%", "not written as this page writes links"],
    ["not JSON", `#k1j.${Buffer.from("{nope").toString("base64url")}`, "does not hold JSON"],
    ["hidden as a number", await gammaLink(page, (l) => (l.hidden = 5)), "hidden layers"],
    ["a camera without a target", await gammaLink(page, (l) => delete l.camera.target), "camera"],
    [
      "a layer kind a link does not carry",
      await gammaLink(page, (l) =>
        l.edited.layers.push({
          name: "stored",
          role: "magnitude",
          width: 0.2,
          color: "#000000",
          clipped: true,
          heightPolicy: "magnitude",
          source: { kind: "file", vertices: "a.npy", offsets: "b.npy" },
        }),
      ),
      "carries no file layer",
    ],
    [
      "wall edges past the walls",
      await gammaLink(page, (l) => {
        for (const x of l.edited.layers) if (x.name === "wall_edges") x.source.edges = [0, 9];
      }),
      "an edge no wall has",
    ],
  ];
  for (const [what, hash, says] of cases) {
    const started = Date.now();
    const fresh = await page.context().newPage();
    const errors = strict(fresh);
    await load(fresh, `${at(info.project.name)}${hash}`);
    await expect(fresh.locator("#notice"), what).toContainText("This link was refused");
    await expect(fresh.locator("#notice"), what).toContainText(says);
    expect((await state(fresh)).landscape.name, what).toBe("gamma");
    expect(errors, what).toEqual([]);
    await fresh.close();
    console.log(`${info.project.name}: ${what} refused in ${Date.now() - started} ms`);
  }
});

test("a link that cannot be made leaves the open document and what it can undo", async ({ page }, info) => {
  const errors = strict(page);
  await load(page, at(info.project.name));
  const height = page.locator('input[type="number"][aria-label="Truncation height"]');
  await height.fill("3.5");
  await height.press("Enter");
  await idle(page);
  const before = await page.evaluate(async () => (await (window as any).kurven.request({ kind: "edited" })).reply.edited);
  const undo = (await state(page)).undo;
  expect(undo).toEqual({ can: true, name: "Truncation" });
  // Each end of its window finite, the width not: it passes the bounds, and
  // cannot be sampled.
  const hash = await gammaLink(page, (l) => {
    l.edited.landscape.domain = { real: { lo: -1e308, hi: 1e308 }, imag: { lo: 0, hi: 2 } };
    l.edited.layers = l.edited.layers.filter((x: any) => x.source.kind !== "hatching");
  });
  await page.evaluate((h) => (location.hash = h), hash);
  await expect(page.locator("#notice")).toContainText("could not be made");
  await idle(page);
  expect(await page.evaluate(async () => (await (window as any).kurven.request({ kind: "edited" })).reply.edited)).toEqual(before);
  expect((await state(page)).undo).toEqual(undo);
  // ...and the URL is the open document's again.
  expect(await page.evaluate(() => (window as any).kurven.linked())).not.toContain(hash);
  // A link opened, but one of its settings refused: said.
  const partly = await gammaLink(page, (l) => (l.depthTolerance = -1));
  await page.evaluate((h) => (location.hash = h), partly);
  await expect(page.locator("#notice")).toContainText("opened, but not all of it");
  expect(errors).toEqual([]);
});

test("a cancel leaves no request unhandled, and a drag held through it is let go", async ({ page }, info) => {
  const errors = strict(page);
  await load(page, at(info.project.name));
  const levels = page.locator('[data-levels="mag_major"]');
  const count = (await state(page)).layers.find((l: any) => l.name === "mag_major").levels;
  // Take hold of a level slider; while it is held, an export starts and is
  // cancelled, and the hand moves and lets go.
  const box = (await levels.boundingBox())!;
  const frac = await levels.evaluate((i: HTMLInputElement) => (Number(i.value) - Number(i.min)) / (Number(i.max) - Number(i.min)));
  const x = box.x + 7 + frac * (box.width - 14);
  await page.mouse.move(x, box.y + box.height / 2);
  await page.mouse.down();
  await page.waitForFunction(() => (window as any).kurven.engine.busy.length === 0);
  await page.evaluate(() => document.querySelector<HTMLButtonElement>("#export")!.click());
  await page.mouse.move(x + 20, box.y + box.height / 2, { steps: 4 });
  await page.evaluate(() => document.querySelector<HTMLButtonElement>("#cancel-export")!.click());
  await page.mouse.up();
  await expect(page.locator("#export-status")).toContainText("cancelled");
  await idle(page);
  // The drag was closed: a step from the keyboard is an edit of its own.
  await levels.focus();
  await page.keyboard.press("ArrowRight");
  await idle(page);
  const s = await state(page);
  expect(s.undo.name).toBe("Levels of mag_major");
  const now = s.layers.find((l: any) => l.name === "mag_major").levels;
  await page.keyboard.press("ControlOrMeta+z");
  await idle(page);
  expect((await state(page)).layers.find((l: any) => l.name === "mag_major").levels).toBe(now - 1);
  expect(count).toBeGreaterThan(0);
  expect(errors).toEqual([]);
});

test("a formula typed and not sampled stays, said, until return or escape", async ({ page }, info) => {
  await load(page, at(info.project.name));
  const field = page.locator("#formula");
  await field.fill("z^5");
  // Focus elsewhere, and a reply renders the panel: the text stays.
  await page.locator('li[data-layer="folds"] input[type="checkbox"]').uncheck();
  await idle(page);
  expect(await field.inputValue()).toBe("z^5");
  await expect(page.locator("#formula-draft")).toBeVisible();
  await expect(field).toHaveAttribute("data-draft", "true");
  expect((await state(page)).landscape.expression).toBe("gamma(z)");
  // Escape puts back the landscape's.
  await field.focus();
  await page.keyboard.press("Escape");
  expect(await field.inputValue()).toBe("gamma(z)");
  await expect(page.locator("#formula-draft")).toBeHidden();
  // Return samples it.
  await field.fill("z^5");
  await field.press("Enter");
  await idle(page);
  expect((await state(page)).landscape.expression).toBe("z^5");
  await expect(page.locator("#formula-draft")).toBeHidden();
});

test("the sliders reach as far as the engine says", async ({ page }, info) => {
  await load(page, at(info.project.name));
  const controls = await page.evaluate(async () => (await (window as any).kurven.request({ kind: "catalog" })).reply.controls);
  const samples = page.locator("#samples");
  expect(Number(await samples.getAttribute("min"))).toBe(controls.samples.min);
  expect(Number(await samples.getAttribute("max"))).toBe(controls.samples.max);
  expect(Number(await samples.getAttribute("step"))).toBe(controls.samples.step);
  const levels = page.locator('[data-levels="mag_major"]');
  expect(Number(await levels.getAttribute("min"))).toBe(controls.levels.min);
  expect(Number(await levels.getAttribute("max"))).toBe(controls.levels.max);
  expect(Number(await page.locator("#truncation-height").getAttribute("min"))).toBe(controls.truncation.min);
});
