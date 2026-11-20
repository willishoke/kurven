// The pool of engines in Chromium (src/engine/pool.ts): Export baked by
// pools of 1, 2 and 4 helpers is the bytes the page's engine bakes alone and
// the bytes `kurven-cli session` bakes natively from the page's journal; at
// rest the draft stays on the canvas while the helpers lay out the frame at
// rest, which is swapped in. With KURVEN_MEASURE=1, the pooled bake of cn and
// gamma is timed at 1, 2, 4 and 8 helpers (crates/FINDINGS.md).

import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { expect, type Page, type TestInfo, test } from "@playwright/test";

const root = fileURLToPath(new URL("../../../../", import.meta.url));
const cli = `${root}crates/target/release/kurven-cli`;

function at(project: string, query: Record<string, string> = {}): string {
  const q = new URLSearchParams(query);
  if (project === "swiftshader") q.set("offscreen", "1");
  return `/?${q.toString()}`;
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
  await page.waitForFunction(() => document.body.dataset.busy === "false", null, { timeout: 600_000 });
  await page.evaluate(() => (window as any).kurven.settled());
}

async function exported(page: Page): Promise<Buffer> {
  const [download] = await Promise.all([page.waitForEvent("download", { timeout: 600_000 }), page.click("#export")]);
  return readFileSync((await download.path())!);
}

function native(journal: unknown[], info: TestInfo, name: string): Buffer {
  const requests = info.outputPath(`${name}.jsonl`);
  const svg = info.outputPath(`${name}.svg`);
  writeFileSync(requests, journal.map((r) => JSON.stringify(r)).join("\n") + "\n");
  execFileSync(cli, ["session", requests, "-o", svg], { stdio: ["ignore", "inherit", "inherit"] });
  return readFileSync(svg);
}

test.describe.configure({ mode: "serial" });

test("Export through a pool of 1, 2 or 4 helpers is the page's engine's bytes and native's", async ({ page }, info) => {
  await load(page, at(info.project.name, { pool: "2" }));
  // Edited as a user edits: a typed expression, the truncation, an orbit.
  await page.locator("#formula").fill("gamma(z) + 1/z");
  await page.locator("#formula").press("Enter");
  await idle(page);
  await page.evaluate(async () => {
    const k = (window as any).kurven;
    await k.send({ kind: "truncation", truncation: { kind: "uniform", height: 3.5 } });
    await k.send({ kind: "gesture", gesture: { kind: "orbit", delta: [25, -8] } });
  });
  await idle(page);
  const two = await exported(page);
  await expect(page.locator("#export-status")).toContainText("paths");
  const sizes: Record<string, Buffer> = { 2: two };
  for (const size of [1, 4]) {
    await page.evaluate((n) => (window as any).kurven.pool.resize(n), size);
    sizes[size] = await exported(page);
  }
  const alone = await page.evaluate(async () => {
    const answer = await (window as any).kurven.request({ kind: "bake", svg: true });
    return Array.from(answer.arrays.svg as Uint8Array);
  });
  const journal = await page.evaluate(() => (window as any).kurven.journal());
  const natively = native(journal, info, "pooled");
  for (const [size, svg] of Object.entries(sizes)) {
    expect(svg.equals(natively), `a pool of ${size}: native's bytes`).toBe(true);
  }
  expect(Buffer.from(alone).equals(natively), "the page's engine alone: native's bytes").toBe(true);
  const helpers = await page.evaluate(() => (window as any).kurven.pool.clients.length);
  expect(helpers).toBe(4);
});

test("at rest the draft stays, and the helpers' frame at rest is swapped in", async ({ page }, info) => {
  test.skip(info.project.name === "fallback", "without WebGPU the frame at rest is the judged strokes");
  await load(page, at(info.project.name, { landscape: "cn" }));
  const t = await page.evaluate(async () => {
    const k = (window as any).kurven;
    const timings = k.view.timings as { kind: string; engineMs: number; totalMs: number }[];
    const results: { draft: number; rest: number; refining: number }[] = [];
    for (let i = 0; i < 3; i++) {
      const n = timings.length;
      await k.send({ kind: "gesture", gesture: { kind: "orbit", delta: [12, 2] } });
      k.view.moved();
      const moved = performance.now();
      // The draft: drawn at once, and kept on the canvas while at rest.
      while (!timings.slice(n).some((x) => x.kind === "draft")) await new Promise((r) => setTimeout(r, 5));
      const draft = performance.now() - moved;
      while (!k.view.isRefining) await new Promise((r) => setTimeout(r, 5));
      const refining = performance.now() - moved;
      await k.settled();
      const rest = timings.slice(n).find((x) => x.kind === "rest");
      results.push({ draft, refining, rest: rest ? rest.totalMs + refining : Number.NaN });
    }
    return { results, title: k.state().title, size: k.pool.size, kinds: timings.slice(-2).map((x) => x.kind) };
  });
  const line = t.results
    .map((r) => `draft ${r.draft.toFixed(0)} ms, the helpers start at ${r.refining.toFixed(0)} ms, rest at ${r.rest.toFixed(0)} ms`)
    .join("; ");
  console.log(`${info.project.name} ${t.title}, ${t.size} helpers: ${line}`);
  info.annotations.push({ type: "timing", description: line });
  expect(t.kinds[t.kinds.length - 1]).toBe("rest");
  for (const r of t.results) {
    expect(r.rest).toBeGreaterThan(r.draft);
    if (info.project.name === "gpu") expect(r.draft).toBeLessThan(1000);
  }
});

test("measure: pooled bakes of cn and gamma at 1, 2, 4 and 8 helpers", async ({ page }, info) => {
  test.skip(process.env.KURVEN_MEASURE !== "1", "a measurement: KURVEN_MEASURE=1");
  test.skip(info.project.name !== "gpu", "measured once, on the Mac's own Chromium");
  for (const preset of ["gamma", "cn"]) {
    await load(page, at(info.project.name, { landscape: preset, pool: "8" }));
    const m = await page.evaluate(async () => {
      const k = (window as any).kurven;
      const out: { size: number; ms: number; bytes: number }[] = [];
      let first: Uint8Array | null = null;
      let same = true;
      // Each size twice; the second taken, its helpers warm (caught up).
      for (const size of [1, 2, 4, 8]) {
        k.pool.resize(size);
        let ms = 0;
        let svg: Uint8Array | null = null;
        for (let run = 0; run < 2; run++) {
          const t = performance.now();
          const answer = await k.pool.bake(true);
          ms = performance.now() - t;
          svg = answer.arrays.svg;
        }
        if (!first) first = svg;
        else same &&= first!.length === svg!.length && first!.every((b: number, i: number) => b === svg![i]);
        out.push({ size, ms, bytes: svg!.length });
      }
      const t = performance.now();
      const alone = await k.request({ kind: "bake", svg: true });
      const aloneMs = performance.now() - t;
      same &&= first!.length === alone.arrays.svg.length && first!.every((b: number, i: number) => b === alone.arrays.svg[i]);
      return {
        out,
        aloneMs,
        same,
        cores: navigator.hardwareConcurrency,
        memory: [k.client.memory, ...k.pool.clients.map((c: any) => c.memory)],
      };
    });
    const line =
      `${preset}: ${m.out.map((o) => `${o.size} helpers ${(o.ms / 1000).toFixed(2)} s`).join(", ")}; ` +
      `the page's engine alone ${(m.aloneMs / 1000).toFixed(2)} s; ${m.out[0]!.bytes} bytes, ` +
      `all the same: ${m.same}; ${m.cores} cores; memory per engine ${m.memory.map((b: number) => (b / 2 ** 20).toFixed(0)).join(", ")} MiB`;
    console.log(line);
    info.annotations.push({ type: "measure", description: line });
    expect(m.same).toBe(true);
  }
});
