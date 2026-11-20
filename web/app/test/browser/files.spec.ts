// Phase 8, step 7 (HANDOFF-web-port.md): a `.kurven` bundle opened from a
// file and saved as one. A bundle's folder opened, edited, its Export held
// to `kurven-cli session` natively; Save bundle's archive holding the files
// a native save writes, byte for byte, and opened in a fresh page to the
// same drawing. Then untrusted input: files that are not bundles, archives
// that break the rules, manifests past the controls' reach, and one whose
// function never returns -- each refused and said, the document and its
// undo history left as they were.

import { execFileSync } from "node:child_process";
import { mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
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

async function exported(page: Page): Promise<Buffer> {
  const [download] = await Promise.all([page.waitForEvent("download", { timeout: 300_000 }), page.click("#export")]);
  return readFileSync((await download.path())!);
}

function native(journal: unknown[], info: TestInfo, name: string, extra: string[]): Buffer {
  const requests = info.outputPath(`${name}.jsonl`);
  const svg = info.outputPath(`${name}.svg`);
  writeFileSync(requests, journal.map((r) => JSON.stringify(r)).join("\n") + "\n");
  execFileSync(cli, ["session", requests, "-o", svg, ...extra], { stdio: ["ignore", "inherit", "inherit"] });
  return readFileSync(svg);
}

/** An archive's entries, by `unzip`'s listing and contents. */
function unzipped(archive: string, dir: string): Record<string, Buffer> {
  execFileSync("unzip", ["-q", "-o", archive, "-d", dir]);
  const out: Record<string, Buffer> = {};
  const walk = (d: string, prefix: string) => {
    for (const e of readdirSync(d, { withFileTypes: true })) {
      if (e.isDirectory()) walk(join(d, e.name), `${prefix}${e.name}/`);
      else out[`${prefix}${e.name}`] = readFileSync(join(d, e.name));
    }
  };
  walk(dir, "");
  return out;
}

/** What the document is, for "nothing changed": the state but its status
 * lines, which are messages. */
async function document(page: Page): Promise<unknown> {
  const s = await state(page);
  return { ...s, status: null };
}

test.describe.configure({ mode: "serial" });

test("a bundle's folder opened and edited; saved, the native files; reopened, the same drawing", async ({ page }, info) => {
  await load(page, at(info.project.name));
  const gamma = `${inputs}gamma.kurven`;
  await page.setInputFiles("#open-folder", gamma);
  await idle(page);
  let s = await state(page);
  expect(s.kind).toBe("landscape");
  await expect(page.locator("#notice")).toBeHidden();
  await expect(page.locator("#file-status")).toHaveText("opened gamma.kurven");
  // An edit, then Export: the native bytes, the folder staged natively.
  await page.check('input[name="truncation"][value="uniform"]');
  await idle(page);
  await page.fill("input[aria-label='Truncation height'][type=number]", "4");
  await page.press("input[aria-label='Truncation height'][type=number]", "Enter");
  await idle(page);
  s = await state(page);
  expect(s.truncation).toEqual({ kind: "uniform", height: 4 });
  const journal = await page.evaluate(() => (window as any).kurven.journal());
  const svg = await exported(page);
  expect(svg.equals(native(journal, info, "folder", ["--stage", "gamma.kurven", gamma]))).toBe(true);

  // Save bundle: an archive of the bundle's directory, its files the native save's.
  const [download] = await Promise.all([page.waitForEvent("download"), page.click("#save")]);
  expect(download.suggestedFilename()).toMatch(/\.kurven\.zip$/);
  const archive = (await download.path())!;
  const name = download.suggestedFilename().replace(/\.zip$/, "");
  const saved = unzipped(archive, mkdtempSync(join(info.outputPath(), "saved-")));
  const nativeDir = mkdtempSync(join(info.outputPath(), "native-"));
  native(journal, info, "save", ["--stage", "gamma.kurven", gamma, "--save", name, nativeDir]);
  const reference = Object.fromEntries(readdirSync(nativeDir).map((f) => [`${name}/${f}`, readFileSync(join(nativeDir, f))]));
  expect(Object.keys(saved).sort()).toEqual(Object.keys(reference).sort());
  for (const [path, bytes] of Object.entries(reference)) expect(saved[path]!.equals(bytes), path).toBe(true);
  expect(Object.keys(saved)).toContain(`${name}/manifest.json`);

  // The archive opened in a fresh page: the same drawing, byte for byte.
  const fresh = await page.context().newPage();
  await load(fresh, at(info.project.name));
  await fresh.setInputFiles("#open-file", archive);
  await idle(fresh);
  expect((await state(fresh)).truncation).toEqual({ kind: "uniform", height: 4 });
  expect((await exported(fresh)).equals(svg)).toBe(true);
  await fresh.close();
});

test("a surface and a polyhedron open from their manifests and archives", async ({ page }, info) => {
  await load(page, at(info.project.name));
  for (const [bundle, kind] of [
    ["relief", "surface"],
    ["cube", "polyhedron"],
  ] as const) {
    await page.setInputFiles("#open-file", `${inputs}${bundle}.kurven/manifest.json`);
    await idle(page);
    const s = await state(page);
    expect(s.kind, `${bundle}: ${await page.locator("#notice").textContent()}`).toBe(kind);
    const journal = await page.evaluate(() => (window as any).kurven.journal());
    const svg = await exported(page);
    // A bare manifest is staged as `<name>.kurven`.
    expect(svg.equals(native(journal, info, bundle, ["--stage", "manifest.kurven", `${inputs}${bundle}.kurven`]))).toBe(true);
  }
});

test("untrusted files are refused, and the document and its undo history stay", async ({ page }, info) => {
  await load(page, at(info.project.name, { landscape: "sin", openTimeout: "6000" }));
  await page.check('input[name="truncation"][value="none"]');
  await idle(page);
  const before = await document(page);
  const undo = (await state(page)).undo;
  expect(undo.can).toBe(true);
  const dir = mkdtempSync(join(info.outputPath(), "bad-"));
  const manifest = (name: string) => JSON.parse(readFileSync(`${inputs}${name}.kurven/manifest.json`, "utf8"));
  const archive = async (file: string, files: { path: string; text: string }[], wrap = "x.kurven") => {
    const bytes: number[] = await page.evaluate(([w, f]) => (window as any).kurven.zip(w, f), [wrap, files] as const);
    writeFileSync(join(dir, file), Buffer.from(bytes));
    return join(dir, file);
  };
  const levels = manifest("gamma");
  levels.layers[0].source.levels = Array.from({ length: 2000 }, (_, k) => k / 100);
  const ring = manifest("torus");
  ring.surface.grid = 60000;
  const hangs = manifest("torus");
  hangs.surface.shape = { kind: "periodic", expression: "besseli(1e12, z)", periods: [6.283185307179586, 6.283185307179586], major: 2, minor: 1, grid: 120 };
  writeFileSync(join(dir, "noise.zip"), Buffer.from(Array.from({ length: 4096 }, (_, k) => (k * 2654435761) & 0xff)));
  writeFileSync(join(dir, "garbage.json"), "{not json");
  const cases: [string, RegExp][] = [
    [join(dir, "noise.zip"), /not a ZIP archive/],
    [join(dir, "garbage.json"), /was not opened/],
    [await archive("nomanifest.zip", [{ path: "height.npy", text: "x" }]), /no manifest\.json/],
    [await archive("climbs.zip", [{ path: "../manifest.json", text: "{}" }], "x"), /climbs out/],
    [await archive("levels.zip", [{ path: "manifest.json", text: JSON.stringify(levels) }]), /2000 levels/],
    [await archive("ring.zip", [{ path: "manifest.json", text: JSON.stringify(ring) }]), /grid/],
  ];
  for (const [file, says] of cases) {
    await page.setInputFiles("#open-file", file);
    await idle(page);
    await expect(page.locator("#notice")).toBeVisible();
    await expect(page.locator("#notice")).toContainText(says);
    expect(await document(page), file).toEqual(before);
  }
  // A function that never returns: the opening times out, the engine is
  // restarted and the document replayed as it was.
  const restarts = await page.evaluate(() => (window as any).kurven.client.restarts);
  await page.setInputFiles("#open-file", await archive("hangs.zip", [{ path: "manifest.json", text: JSON.stringify(hangs) }]));
  await expect(page.locator("#notice")).toContainText("took more than 6 s", { timeout: 60_000 });
  await idle(page);
  expect(await page.evaluate(() => (window as any).kurven.client.restarts)).toBe(restarts + 1);
  expect(await document(page)).toEqual(before);
  // And the edit before is still the one undone.
  expect((await state(page)).undo).toEqual(undo);
  await page.click("#undo");
  await idle(page);
  expect((await state(page)).truncation.kind).not.toBe("none");
});
