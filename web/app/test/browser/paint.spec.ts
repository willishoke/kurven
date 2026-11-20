// Phase 8, step 6 (HANDOFF-web-port.md): a banded truncation edited by its
// rows; a face picked by a click and "Curves on" it -- a curve added, its
// levels counted, taken off; a function painted on a torus through its
// sheet; undo and redo by their buttons; a polyhedron opened, its hatching's
// strokes dragged and a face of it hatched. Each document's Export is held
// byte for byte to `kurven-cli session` running the page's journal natively.

import { execFileSync } from "node:child_process";
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { expect, type Page, type TestInfo, test } from "@playwright/test";

const root = fileURLToPath(new URL("../../../../", import.meta.url));
const cli = `${root}crates/target/release/kurven-cli`;
const inputs = `${root}crates/fixtures/swift/session/inputs/`;

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

async function dragThumb(page: Page, selector: string, dx: number, steps = 8): Promise<void> {
  const el = page.locator(selector);
  const box = (await el.boundingBox())!;
  const frac = await el.evaluate((i: HTMLInputElement) => (Number(i.value) - Number(i.min)) / (Number(i.max) - Number(i.min)));
  const thumb = 16;
  const x = box.x + thumb / 2 + frac * (box.width - thumb);
  const y = box.y + box.height / 2;
  await page.mouse.move(x, y);
  await page.mouse.down();
  for (let k = 1; k <= steps; k++) await page.mouse.move(x + (dx * k) / steps, y);
  await page.mouse.up();
}

async function type(page: Page, selector: string, value: string): Promise<void> {
  await page.fill(selector, value);
  await page.press(selector, "Enter");
  await page.locator(selector).blur();
}

async function exportMatches(page: Page, info: TestInfo, name: string, staged: [string, string][] = []): Promise<void> {
  const [download] = await Promise.all([page.waitForEvent("download", { timeout: 300_000 }), page.click("#export")]);
  const svg = readFileSync((await download.path())!);
  const journal = await page.evaluate(() => (window as any).kurven.journal());
  const requests = info.outputPath(`${name}.jsonl`);
  const out = info.outputPath(`${name}.svg`);
  writeFileSync(requests, journal.map((r: unknown) => JSON.stringify(r)).join("\n") + "\n");
  const stage = staged.flatMap(([n, dir]) => ["--stage", n, dir]);
  execFileSync(cli, ["session", requests, "-o", out, ...stage], { stdio: ["ignore", "inherit", "inherit"] });
  expect(svg.length).toBeGreaterThan(1000);
  expect(svg.equals(readFileSync(out)), `${name}: the page's Export is kurven-cli session's bytes`).toBe(true);
}

async function openBundle(page: Page, name: string): Promise<string> {
  const dir = `${inputs}${name}.kurven`;
  const files = readdirSync(dir).map((f) => ({ path: f, base64: readFileSync(`${dir}/${f}`).toString("base64") }));
  await page.evaluate(
    async ([name, files]) => {
      const decoded = files.map((f) => ({ path: f.path, bytes: Uint8Array.from(atob(f.base64), (c) => c.charCodeAt(0)).buffer }));
      await (window as any).kurven.open({ name, files: decoded });
    },
    [`${name}.kurven`, files] as const,
  );
  await idle(page);
  return dir;
}

/** A pixel of the canvas on the drawing: the engine's camera taken to the
 * world point given, then the viewport's pixel, by the frame's mapping. */
async function pixelOf(page: Page, world: [number, number, number]): Promise<[number, number]> {
  return page.evaluate(async (world) => {
    const k = (window as any).kurven;
    const r = await k.request({ kind: "preview", draft: true, have: k.held() ?? {} });
    const u = r.reply.uniforms;
    const v = u.view as number[]; // column-major 4x4
    const x = v[0]! * world[0] + v[4]! * world[1] + v[8]! * world[2] + v[12]!;
    const y = v[1]! * world[0] + v[5]! * world[1] + v[9]! * world[2] + v[13]!;
    const f = u.frame;
    const col = 0.5 + ((f.cols - 1) * (x - f.axis1[0])) / (f.axis1[1] - f.axis1[0]);
    const row = 0.5 + ((f.rows - 1) * (y - f.axis0[0])) / (f.axis0[1] - f.axis0[0]);
    return [col, row];
  }, world);
}

/** Click the canvas at a pixel of it, held still. */
async function click(page: Page, pixel: [number, number]): Promise<void> {
  const box = (await page.locator("#preview").boundingBox())!;
  await page.mouse.click(box.x + pixel[0], box.y + pixel[1]);
  await idle(page);
}

test.describe.configure({ mode: "serial" });

test("a banded truncation, by its rows", async ({ page }, info) => {
  await load(page, at(info.project.name, { landscape: "sin" }));
  await page.check('input[name="truncation"][value="realBands"]');
  await idle(page);
  let s = await state(page);
  expect(s.truncation.kind).toBe("realBands");
  expect(s.truncation.bands.length).toBe(1);
  await expect(page.locator("#bands")).toBeVisible();
  await page.click("#band-add");
  await idle(page);
  s = await state(page);
  expect(s.truncation.bands.length).toBe(2);
  expect(s.truncation.bands[1].below).toBe(s.truncation.bands[0].below + 1);
  // A band's height dragged: one undo step, the whole truncation each move.
  const before = s.truncation.bands[0].height;
  await dragThumb(page, '#band-rows input[type=range][data-band="0"]', 30);
  await idle(page);
  s = await state(page);
  expect(s.truncation.bands[0].height).toBeGreaterThan(before);
  expect(s.undo.name).toBe("Truncation");
  // A threshold typed, the beyond typed, a band taken off.
  await type(page, '#band-rows input[type=number][data-band="1"]', "2.5");
  await idle(page);
  s = await state(page);
  expect(s.truncation.bands[1].below).toBe(2.5);
  await type(page, "#band-beyond-field", "4");
  await idle(page);
  s = await state(page);
  expect(s.truncation.beyond).toBe(4);
  await page.click('[data-band-remove="0"]');
  await idle(page);
  s = await state(page);
  expect(s.truncation.bands.length).toBe(1);
  expect(s.truncation.bands[0].below).toBe(2.5);
  await exportMatches(page, info, "bands");
});

test("a face picked, curves on it, a function painted, undo and redo", async ({ page }, info) => {
  await load(page, at(info.project.name, { surface: "torus" }));
  await expect(page.locator("#paint-section")).toBeHidden();
  // A click on the tube picks the surface.
  await click(page, await pixelOf(page, [2, 0, 1]));
  let s = await state(page);
  expect(s.picked?.face).toBe("surface");
  await expect(page.locator("#paint-section")).toBeVisible();
  await expect(page.locator("#h-paint")).toHaveText("Curves on the surface");
  await expect(page.locator("#paint-caption")).toHaveText("Picked the surface, which is one face.");
  // A click that orbits picks nothing new; a click through the hole clears.
  const box = (await page.locator("#preview").boundingBox())!;
  await page.mouse.move(box.x + 100, box.y + 100);
  await page.mouse.down();
  await page.mouse.move(box.x + 140, box.y + 110);
  await page.mouse.up();
  await idle(page);
  expect((await state(page)).picked?.face).toBe("surface");
  await click(page, await pixelOf(page, [2, 0, 1]));
  // A double click picks once: its second click is not a pick.
  const pixel = await pixelOf(page, [2, 0, 1]);
  await page.evaluate(() => {
    const k = (window as any).kurven;
    k.picks = 0;
    const request = k.client.request.bind(k.client);
    k.client.request = (r: any, o: any) => {
      if (r.kind === "pick") k.picks++;
      return request(r, o);
    };
  });
  const canvasBox = (await page.locator("#preview").boundingBox())!;
  await page.mouse.dblclick(canvasBox.x + pixel[0], canvasBox.y + pixel[1]);
  await idle(page);
  expect(await page.evaluate(() => (window as any).kurven.picks)).toBe(1);

  // Function…: sn's periods come with it; painted, the surface is periodic.
  await page.click("#paint-function");
  await expect(page.locator("#function-sheet")).toBeVisible();
  await page.fill("#function-expression", "sn(z, 0.64)");
  const periods = await page.evaluate(() => (window as any).kurven.request({ kind: "periods", expression: "sn(z, 0.64)" }));
  await expect(page.locator("#function-real")).toHaveValue(String(periods.reply.known[0]));
  await page.click("#function-paint");
  await expect(page.locator("#function-sheet")).toBeHidden();
  await idle(page);
  s = await state(page);
  expect(s.surface.shape.kind).toBe("periodic");
  expect(s.undo.name).toBe("Function sn(z, 0.64)");
  // A bad function is refused in the sheet, and the sheet stays.
  await page.click("#paint-function");
  await page.fill("#function-expression", "sn(z");
  await page.click("#function-paint");
  await expect(page.locator("#function-refusal")).not.toBeEmpty();
  await page.click("#function-cancel");

  // A curve added from the Add menu, its levels counted, taken off.
  s = await state(page);
  expect(s.picked.choices.length).toBeGreaterThan(0);
  const choice = s.picked.choices.find((c: any) => /\|f\|/.test(c.label)) ?? s.picked.choices[0];
  await page.selectOption("#paint-add", choice.id);
  await idle(page);
  s = await state(page);
  expect(s.picked.layers).toContain(choice.id);
  const painted = page.locator(`#paint-layers li[data-layer="${choice.id}"]`);
  await expect(painted).toBeVisible();
  const levels = painted.locator("input[data-levels]");
  await expect(levels).toHaveCount(1);
  await levels.focus();
  await page.keyboard.press("ArrowRight");
  await idle(page);
  expect((await state(page)).undo.name).toBe(`Levels of ${choice.id}`);
  await painted.locator("button.remove").click();
  await idle(page);
  s = await state(page);
  expect(s.picked.layers).not.toContain(choice.id);

  // Undo and redo by their buttons, named for the step.
  await expect(page.locator("#undo")).toHaveText(`Undo ${s.undo.name}`);
  await page.click("#undo");
  await idle(page);
  s = await state(page);
  expect(s.picked.layers).toContain(choice.id);
  await expect(page.locator("#redo")).toBeEnabled();
  await page.click("#redo");
  await idle(page);
  s = await state(page);
  expect(s.picked.layers).not.toContain(choice.id);
  await page.click("#undo");
  await idle(page);
  await exportMatches(page, info, "painted");
});

test("a polyhedron: its facts, its hatching's strokes, a face hatched", async ({ page }, info) => {
  await load(page, at(info.project.name));
  const dir = await openBundle(page, "frame");
  let s = await state(page);
  expect(s.kind).toBe("polyhedron");
  await expect(page.locator("#polyhedron-section")).toBeVisible();
  await expect(page.locator("#polyhedron-facts")).toContainText("16 vertices, 24 edges, 10 faces");
  await expect(page.locator("#landscape-section")).toBeHidden();
  const hatching = s.layers.find((l: any) => l.strokes !== undefined);
  expect(hatching).toBeTruthy();
  const strokes = page.locator(`#layers input[data-strokes="${hatching.name}"]`);
  await expect(strokes).toHaveAttribute("max", "400");
  await dragThumb(page, `#layers input[data-strokes="${hatching.name}"]`, 25);
  await idle(page);
  s = await state(page);
  const after = s.layers.find((l: any) => l.name === hatching.name);
  expect(after.strokes).not.toBe(hatching.strokes);
  expect(s.undo.name).toBe(`Strokes of ${hatching.name}`);
  // A face of the solid picked by a click, and hatched from the Add menu.
  const centroid = await page.evaluate(async () => {
    const g = await (window as any).kurven.request({ kind: "grids" });
    const v = g.arrays.vertices as Float32Array;
    const t = g.arrays.triangles as Uint32Array;
    // The triangle nearest the eye along the camera's line of sight is
    // not known here; any face under the pixel will do, so take the
    // centroid of the largest triangle.
    let best = [0, 0, 0];
    let area = -1;
    for (let k = 0; k < t.length; k += 3) {
      const p = [0, 1, 2].map((j) => [v[3 * t[k + j]!]!, v[3 * t[k + j]! + 1]!, v[3 * t[k + j]! + 2]!]);
      const a = p[1]!.map((x, i) => x - p[0]![i]!);
      const b = p[2]!.map((x, i) => x - p[0]![i]!);
      const c = [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!];
      const s = Math.hypot(c[0]!, c[1]!, c[2]!);
      if (s > area) {
        area = s;
        best = [0, 1, 2].map((i) => (p[0]![i]! + p[1]![i]! + p[2]![i]!) / 3);
      }
    }
    return best as [number, number, number];
  });
  await click(page, await pixelOf(page, centroid));
  s = await state(page);
  expect(s.picked?.face).toMatch(/^face \d+$/);
  await expect(page.locator("#paint-caption")).toContainText("faces of the solid");
  const hatch = s.picked.choices[0];
  expect(hatch).toBeTruthy();
  await page.selectOption("#paint-add", hatch.id);
  await idle(page);
  s = await state(page);
  expect(s.picked.layers).toContain(hatch.id);
  expect(s.undo.name).toBeTruthy();
  await exportMatches(page, info, "polyhedron", [["frame.kurven", dir]]);
});
