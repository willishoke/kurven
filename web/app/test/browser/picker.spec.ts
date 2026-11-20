// The picker (HANDOFF-web-port.md, Phase 8, step 4; Gallery.swift and
// Thumbnails.swift): the catalogs as pictures, each drawn by the engine in
// an engine of its own -- the open document untouched -- one at a time,
// and cached, so a second look at the picker draws nothing.

import { expect, type Page, test } from "@playwright/test";

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

async function idle(page: Page): Promise<void> {
  await page.evaluate(() => (window as any).kurven.settled());
  await page.waitForFunction(() => document.body.dataset.busy === "false", null, { timeout: 300_000 });
  await page.evaluate(() => (window as any).kurven.settled());
}

const state = (page: Page) => page.evaluate(() => (window as any).kurven.state());

/** How much of a picture is ink: the share of its pixels darker than paper. */
async function inkOf(page: Page, id: string): Promise<{ ink: number; width: number; height: number }> {
  return page.evaluate(async (id) => {
    const img = document.querySelector<HTMLImageElement>(`#picker [data-entry="${CSS.escape(id)}"] img`);
    if (!img) throw new Error(`no picture for ${id}`);
    await img.decode();
    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(img, 0, 0);
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let ink = 0;
    for (let k = 0; k < data.length; k += 4) {
      if (299 * data[k]! + 587 * data[k + 1]! + 114 * data[k + 2]! < 224_000) ink++;
    }
    return { ink: ink / (canvas.width * canvas.height), width: canvas.width, height: canvas.height };
  }, id);
}

test.describe.configure({ mode: "serial" });

test("the picker draws every entry, leaves the document alone, and chooses", async ({ page }, info) => {
  test.setTimeout(900_000);
  await load(page, at(info.project.name));
  const before = await state(page);
  const journal = await page.evaluate(() => (window as any).kurven.journal().length);

  await page.click("#browse");
  await expect(page.locator("#picker")).toBeVisible();
  const ids = await page.locator("#picker .cell").evaluateAll((cells) => cells.map((c) => (c as HTMLElement).dataset.entry));
  expect(ids).toContain("gamma");
  expect(ids).toContain("surface:sn");
  // The functions first, then the surfaces, under their headings.
  await expect(page.locator("#picker h3")).toHaveText(["Functions", "Surfaces"]);
  await page.evaluate(() => (window as any).kurven.picker.warmed());
  const counts = await page.evaluate(() => (window as any).kurven.picker.counts());
  expect(counts.failed).toEqual([]);
  expect(counts.pictures).toBe(ids.length);
  // Every picture drawn: the engine that drew them is let go of.
  expect(counts.engineRunning).toBe(false);
  console.log(`${info.project.name}: ${counts.drawn} drawn, ${counts.cached} from the cache`);
  // Every picture is the drawing: ink on paper, at the engine's 440 by 300.
  for (const id of ids) {
    const { ink, width, height } = await inkOf(page, id!);
    expect([width, height]).toEqual([440, 300]);
    expect(ink, `${id}'s picture has ink`).toBeGreaterThan(0.01);
    expect(ink, `${id}'s picture is mostly paper`).toBeLessThan(0.6);
  }
  // The open document was not touched, and nothing was journaled.
  const after = await state(page);
  expect(after.landscape).toEqual(before.landscape);
  expect(after.camera.exact).toEqual(before.camera.exact);
  expect(await page.evaluate(() => (window as any).kurven.journal().length)).toBe(journal);

  // The filter: by name, label, formula or notes.
  await page.fill("#picker-filter", "torus");
  const shown = await page.locator("#picker .cell").evaluateAll((cells) => cells.map((c) => (c as HTMLElement).dataset.entry));
  expect(shown).toContain("surface:torus");
  expect(shown).not.toContain("gamma");
  await page.fill("#picker-filter", "");

  // A surface chosen: the picker closes and the surface opens.
  await page.click('#picker [data-entry="surface:torus"]');
  await expect(page.locator("#picker")).toBeHidden();
  await idle(page);
  let s = await state(page);
  expect(s.kind).toBe("surface");
  expect(s.surface.name).toBe("torus");
  await expect(page.locator("#surface-section")).toBeVisible();
  await expect(page.locator("#landscape-section")).toBeHidden();
  // And back to a function, from the surface's own Browse.
  await page.click("#browse-surface");
  await page.click('#picker [data-entry="zeta"]');
  await idle(page);
  s = await state(page);
  expect(s.kind).toBe("landscape");
  expect(s.landscape.name).toBe("zeta");
  await expect(page.locator("#surface-section")).toBeHidden();

  // A second page: every picture from the cache, none drawn.
  const second = await page.context().newPage();
  await load(second, at(info.project.name));
  await second.evaluate(() => (window as any).kurven.picker.open());
  await second.evaluate(() => (window as any).kurven.picker.warmed());
  const again = await second.evaluate(() => (window as any).kurven.picker.counts());
  expect(again.drawn).toBe(0);
  expect(again.cached).toBe(ids.length);
  // All from the cache: no engine was started for them.
  expect(again.engineRunning).toBe(false);
  await second.close();
});
