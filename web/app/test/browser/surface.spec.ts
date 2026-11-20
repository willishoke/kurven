// The Surface section (HANDOFF-web-port.md, Phase 8, step 5;
// SurfaceSection.swift): a torus, sn, a periodic function and sn in relief,
// each edited by its controls as a user edits them -- a slider dragged, a
// value typed past its reach, a switch, a count -- every control's reach the
// engine's. Each document's Export is held byte for byte to `kurven-cli
// session` running the page's journal natively (a bundle the page opened
// staged natively too).

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

/** A control's row, as the state lists it. */
function row(s: any, field: string): any {
  const rows = [...s.surfaceControls.rows, ...(s.surfaceControls.linearCurve?.rows ?? [])];
  const r = rows.find((r: any) => r.field === field);
  if (!r) throw new Error(`no row ${field}`);
  return r;
}

/** Drag a range input's thumb from its value by `dx` pixels. */
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

async function exported(page: Page): Promise<Buffer> {
  const [download] = await Promise.all([page.waitForEvent("download", { timeout: 300_000 }), page.click("#export")]);
  return readFileSync((await download.path())!);
}

/** What `kurven-cli session` bakes from the page's journal, the bundles the
 * page opened staged as it staged them. */
function native(journal: unknown[], info: TestInfo, name: string, staged: [string, string][] = []): Buffer {
  const requests = info.outputPath(`${name}.jsonl`);
  const svg = info.outputPath(`${name}.svg`);
  writeFileSync(requests, journal.map((r) => JSON.stringify(r)).join("\n") + "\n");
  const stage = staged.flatMap(([n, dir]) => ["--stage", n, dir]);
  execFileSync(cli, ["session", requests, "-o", svg, ...stage], { stdio: ["ignore", "inherit", "inherit"] });
  return readFileSync(svg);
}

async function exportMatches(page: Page, info: TestInfo, name: string, staged: [string, string][] = []): Promise<void> {
  const svg = await exported(page);
  const journal = await page.evaluate(() => (window as any).kurven.journal());
  const reference = native(journal, info, name, staged);
  expect(svg.length).toBeGreaterThan(1000);
  expect(svg.equals(reference), `${name}: the page's Export is kurven-cli session's bytes`).toBe(true);
}

/** Open a bundle of the session fixtures as the page opens one. */
async function openBundle(page: Page, name: string): Promise<string> {
  const dir = `${inputs}${name}.kurven`;
  const files = readdirSync(dir).map((f) => ({ path: f, base64: readFileSync(`${dir}/${f}`).toString("base64") }));
  await page.evaluate(
    async ([name, files]) => {
      const k = (window as any).kurven;
      const decoded = files.map((f) => ({ path: f.path, bytes: Uint8Array.from(atob(f.base64), (c) => c.charCodeAt(0)).buffer }));
      await k.open({ name, files: decoded });
    },
    [`${name}.kurven`, files] as const,
  );
  await idle(page);
  return dir;
}

test.describe.configure({ mode: "serial" });

test("a torus: radii, its lines, the linear curve, undo; Export is the native bytes", async ({ page }, info) => {
  await load(page, at(info.project.name, { surface: "torus" }));
  let s = await state(page);
  expect(s.kind).toBe("surface");
  await expect(page.locator("#surface-section")).toBeVisible();
  await expect(page.locator("#truncation-kind")).toBeHidden();
  // The rows are the engine's, with its reach.
  const major = row(s, "major");
  expect(await page.locator("#surface-major").getAttribute("min")).toBe(String(major.min));
  expect(await page.locator("#surface-major").getAttribute("max")).toBe(String(major.max));
  // No switch for a plain torus's lines, and no Conformal.
  await expect(page.locator("#surface-lines")).toBeHidden();
  await expect(page.locator("#surface-conformal")).toBeHidden();

  // A radius dragged: one undo step, named for the control.
  const was = row(s, "major").value;
  await dragThumb(page, "#surface-major", 40);
  await idle(page);
  s = await state(page);
  expect(row(s, "major").value).toBeGreaterThan(was);
  expect(s.undo.name).toBe("Major radius");
  // Typed past its reach: clamped to it.
  await type(page, "#surface-minor-field", "9");
  await idle(page);
  s = await state(page);
  expect(row(s, "minor").value).toBe(row(s, "minor").max);
  // The lines' counts.
  await type(page, "#lines-around", "24");
  await idle(page);
  s = await state(page);
  expect(s.surfaceControls.lines).toEqual([24, 18]);
  expect(s.undo.name).toBe("Lines Around");

  // The linear curve: on at 2/5, its slope dragged to a fraction or a step.
  await page.check("#surface-linear");
  await idle(page);
  s = await state(page);
  expect(row(s, "slope").value).toBe(0.4);
  await expect(page.locator("#linear-description")).toHaveText("One strand at 2/5: closes after 5 turns.");
  await dragThumb(page, "#surface-slope", 23);
  await idle(page);
  s = await state(page);
  const slope = row(s, "slope");
  const rested = slope.detents.includes(slope.value) || Math.abs(slope.value * 1000 - Math.round(slope.value * 1000)) < 1e-9;
  expect(rested, `the slope ${slope.value} rests at a fraction or its step`).toBe(true);
  expect(s.undo.name).toBe("Slope");
  await type(page, "#surface-strands-field", "3");
  await idle(page);

  // Undo puts the strands back.
  await page.keyboard.press(process.platform === "darwin" ? "Meta+z" : "Control+z");
  await idle(page);
  s = await state(page);
  expect(row(s, "strands").value).toBe(1);

  await exportMatches(page, info, "torus");
});

test("sn: conformal, coordinate curves, the modulus dragged; a periodic function", async ({ page }, info) => {
  await load(page, at(info.project.name, { surface: "sn" }));
  await expect(page.locator("#surface-notes")).toContainText("Jacobi's sn");
  await page.check("#surface-lines");
  await idle(page);
  let s = await state(page);
  expect(s.surfaceControls.lines).toEqual([36, 18]);
  await dragThumb(page, "#surface-modulus", -30);
  await idle(page);
  s = await state(page);
  expect(row(s, "modulus").value).toBeLessThan(0.64);
  expect(s.undo.name).toBe("Modulus");
  await page.check("#surface-conformal");
  await idle(page);
  s = await state(page);
  expect(s.surfaceControls.conformal).toBe(true);
  // Conformal pins the minor radius: no row for it.
  await expect(page.locator('[data-row="minor"]')).toHaveCount(0);
  await exportMatches(page, info, "sn");

  // A function painted on a torus is a periodic surface, with its own rows.
  await page.evaluate(() => (window as any).kurven.send({ kind: "createSurface", preset: "torus" }));
  await idle(page);
  await page.evaluate(() => (window as any).kurven.send({ kind: "function", expression: "sin(z)" }));
  await idle(page);
  s = await state(page);
  expect(s.surfaceControls.kind).toBe("periodic");
  expect(s.surfaceControls.rows.map((r: any) => r.field)).toEqual(["major", "minor", "grid"]);
  await type(page, "#surface-grid-field", "200");
  await idle(page);
  s = await state(page);
  expect(row(s, "grid").value).toBe(200);
  await exportMatches(page, info, "periodic");
});

test("sn in relief, opened: its relief typed, clamped to its reach, undone", async ({ page }, info) => {
  await load(page, at(info.project.name));
  const dir = await openBundle(page, "relief");
  let s = await state(page);
  expect(s.surfaceControls.kind).toBe("relief");
  const relief = row(s, "relief");
  expect([relief.min, relief.max]).toEqual([0, 1]);
  await type(page, "#surface-relief-field", "0.3");
  await idle(page);
  s = await state(page);
  expect(row(s, "relief").value).toBe(0.3);
  expect(s.undo.name).toBe("Relief");
  // Typed past its reach, clamped to it; the relief then passes the tube's
  // reach, and the status says what that does to the drawing.
  await type(page, "#surface-relief-field", "1.5");
  await idle(page);
  s = await state(page);
  expect(row(s, "relief").value).toBe(1);
  await expect(page.locator("#surface-status")).toContainText("back faces are kept");
  await page.keyboard.press(process.platform === "darwin" ? "Meta+z" : "Control+z");
  await idle(page);
  s = await state(page);
  expect(row(s, "relief").value).toBe(0.3);
  await exportMatches(page, info, "relief", [["relief.kurven", dir]]);
});
