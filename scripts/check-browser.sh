#!/bin/bash
# The browser lane: the page's preview in Chromium, by Playwright. Separate
# from scripts/check-web.sh, which needs no browser.
#
#   scripts/check-browser.sh                   # swiftshader, present and fallback, and gpu on a Mac
#   scripts/check-browser.sh adapter --project=gpu   # Playwright's own arguments: a filter
#                                                    # goes before --project, which takes a list
#
# It builds the page's wasm module (as check-web.sh does, in its own build
# tree), builds the page, puts the module beside it, and runs
# web/app/test/browser against `vite preview` (web/app/playwright.config.ts
# says what each project is). The Chromium is the one @playwright/test pins
# (Chrome for Testing 153, Playwright's revision 1243), installed on first
# use by `npx playwright install chromium`, which needs the network once.
#
# The page (Phase 8, page.spec.ts): driven as a user drives it, its Export
# held byte for byte to `kurven-cli session` baking the same requests
# natively (crates/target/release/kurven-cli), and its URL to a fresh page.
#
# The gate (HANDOFF-web-port.md, Phase 7): for a landscape (cn), a surface
# (sn) and a polyhedron (the picture frame), the preview at rest against
# the judged strokes the engine bakes for the same frame, by Swift's
# coverage measure and thresholds (test/browser/coverage.js); and the page
# without WebGPU, drawing the judged strokes. WebGPU runs on SwiftShader,
# Dawn's CPU Vulkan, in the `swiftshader` project, so the lane needs no GPU;
# `gpu` adds the machine's own (Metal on a Mac).

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CRATES="$ROOT/crates"
APP="$ROOT/web/app"
PAGE_WASM="$CRATES/target/page/wasm32-unknown-unknown/web/kurven_wasm.wasm"

step() {
    printf '\n\033[1m== %s\033[0m\n' "$1"
}

step "the page's wasm module"
(cd "$CRATES" && CARGO_TARGET_DIR="$CRATES/target/page" cargo build --profile web \
    --target wasm32-unknown-unknown -p kurven-wasm --no-default-features) || exit 1

step "kurven-cli, natively: the reference the page's Export is held to"
(cd "$CRATES" && cargo build --release -p kurven-cli) || exit 1

step "the page"
if [ ! -f "$APP/node_modules/.package-lock.json" ] ||
    [ "$APP/package-lock.json" -nt "$APP/node_modules/.package-lock.json" ]; then
    npm --prefix "$APP" ci --no-audit --no-fund || exit 1
fi
npm --prefix "$APP" run --silent build || exit 1
cp "$PAGE_WASM" "$APP/dist/kurven_wasm.wasm" || exit 1

step "Chromium"
(cd "$APP" && npx playwright install chromium) || exit 1

step "the browser lane"
args=("$@")
if [ ${#args[@]} = 0 ]; then
    args=(--project swiftshader --project present --project fallback)
    [ "$(uname)" = Darwin ] && args+=(--project gpu)
fi
cd "$APP" && npx playwright test "${args[@]}"
