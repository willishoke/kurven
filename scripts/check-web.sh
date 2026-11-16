#!/bin/bash
# The web port's check: everything under crates/ and web/app, with no Swift
# and no browser.
#
#   scripts/check-web.sh
#
# The port is held to the Swift reference in three layers (HANDOFF-web-port.md,
# "The gate"): the oracle fixtures of tests/fixtures, read as the Swift suites
# read them; Swift's golden sets; and the reference drawings `kurven-cli
# references` baked into crates/fixtures/swift, compared per layer by
# `kurven-cli compare` to a tolerance. All three run as the native tests.
#
# One libm, everywhere: the engine reaches the transcendental functions only
# through kurven_math::real, which calls the libm crate on every target, so
# native and wasm compute the same bits and this check, run natively, stands
# for the browser. The grep below and clippy.toml refuse the std spellings,
# and the wasm module must write the port's golden sets byte for byte as
# native does. The port's goldens (crates/fixtures/rust) are regenerated in
# place and compared with the commit, as scripts/check.sh does for Swift's:
# a change to one is a regeneration committed on its own.
#
# Needs rustup, which installs the toolchain crates/rust-toolchain.toml pins
# (with rustfmt, clippy and the wasm32 target) on first use, and Node with npm
# (the page's type check and build install web/app's pinned packages on
# first use).

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CRATES="$ROOT/crates"
APP="$ROOT/web/app"
WORK="$CRATES/target/check-web"
CLI="$CRATES/target/release/kurven-cli"
WASM="$CRATES/target/wasm32-unknown-unknown/web/kurven_wasm.wasm"
# A cargo command on the workspace, wherever this is run from: run in
# crates/, so rustup takes the toolchain crates/rust-toolchain.toml pins.
workspace() {
    (cd "$CRATES" && cargo "$@")
}

started=$SECONDS
failures=0
step() {
    local name="$1"; shift
    printf '\n\033[1m== %s\033[0m\n' "$name"
    if "$@"; then
        return 0
    fi
    printf '\033[31m   FAILED: %s\033[0m\n' "$name"
    failures=$((failures + 1))
}

# The pinned toolchain, installed if it is not: a no-op, offline, when it is.
need() {
    command -v rustup >/dev/null || { echo "no rustup"; return 1; }
    (cd "$CRATES" && rustup toolchain install --no-self-update) || return 1
    (cd "$CRATES" && rustup target list --installed) | grep -qx wasm32-unknown-unknown ||
        { echo "the pinned toolchain has no wasm32-unknown-unknown target"; return 1; }
    command -v node >/dev/null && command -v npm >/dev/null || { echo "no node and npm"; return 1; }
    workspace --version && node --version
}
step "toolchain: the pinned Rust with the wasm target, node and npm" need

step "rustfmt" workspace fmt --all -- --check

step "clippy, native" workspace clippy --release --workspace --all-targets -- -D warnings
step "clippy, wasm" workspace clippy --release -p kurven-wasm --target wasm32-unknown-unknown -- -D warnings

# The std spellings of the transcendental functions, of fused multiply-add,
# and of the reads of a sign bit (a NaN's is platform noise; real::sign_minus,
# real::copysign and real::total_order read every NaN as plus), anywhere in
# the engine but kurven_math::real. Type-blind, so
# it also catches what clippy cannot see; a method of the port's own of one
# of these names would trip it too (crates/FINDINGS.md). Lines that are
# comments are not code.
libm() {
    local names='sin|cos|tan|sin_cos|asin|acos|atan|atan2|sinh|cosh|tanh|asinh|acosh|atanh|exp|exp2|exp_m1|ln|log|log2|log10|ln_1p|powf|powi|hypot|cbrt|mul_add|gamma|ln_gamma|is_sign_negative|is_sign_positive|total_cmp|copysign'
    local pattern="\\.($names)\\(|(^|[^A-Za-z0-9_])f(32|64)::($names)([^A-Za-z0-9_]|\$)"
    local hits
    hits=$(grep -rnE --include='*.rs' "$pattern" "$CRATES"/*/src "$CRATES"/*/tests 2>/dev/null |
        grep -v "^$CRATES/kurven-math/src/real.rs:" |
        grep -vE '^[^:]+:[0-9]+:[[:space:]]*//')
    [ -z "$hits" ] && return 0
    printf '%s\n' "$hits"
    echo "call kurven_math::real instead: one libm on native and wasm, and a NaN's sign read as plus"
    return 1
}
step "no std transcendental, fused multiply-add or sign-bit read outside kurven_math::real" libm

step "native tests" workspace test --release --workspace

# Regenerated in place from scratch, so a file the writer no longer writes
# shows as deleted, and compared with the commit, not the index, so a staged
# change is a change; a file the commit does not hold, staged or not, is one
# too. Nothing is written to the index.
golden() {
    local new failed=0
    workspace build --release -p kurven-cli || return 1
    rm -rf "$CRATES/fixtures/rust"
    "$CLI" fixtures "$CRATES/fixtures/rust" >/dev/null || return 1
    # A golden file .gitignore leaves out is never committed, and so never
    # compared: every file the generator writes must be one git keeps.
    local ignored
    ignored=$(cd "$ROOT" && find crates/fixtures/rust -type f -print0 | xargs -0 git check-ignore --no-index --)
    if [ -n "$ignored" ]; then
        printf 'ignored by .gitignore, so never compared: %s\n' $ignored
        failed=1
    fi
    if ! git -C "$ROOT" diff --quiet HEAD -- crates/fixtures/rust; then
        # The lines themselves, not only the files: on CI the tree is gone
        # once the job ends, and a golden that differs on another platform
        # can only be read here.
        git -C "$ROOT" diff --stat HEAD -- crates/fixtures/rust
        git -C "$ROOT" diff --no-color -U0 HEAD -- crates/fixtures/rust | head -80
        failed=1
    fi
    new=$(git -C "$ROOT" ls-files --others --exclude-standard -- crates/fixtures/rust)
    if [ -n "$new" ]; then
        printf 'not in the commit: %s\n' $new
        failed=1
    fi
    return $failed
}
step "the committed golden fixtures are what kurven-cli fixtures writes" golden

# The comparison, as a command: each reference against itself is the same
# at distance zero, and with one stroke taken out it is not.
compare() {
    local dir="$WORK/compare" svg name out
    rm -rf "$dir" && mkdir -p "$dir" || return 1
    for svg in "$CRATES"/fixtures/swift/*.svg; do
        name=$(basename "$svg" .svg)
        out=$("$CLI" compare "$svg" "$svg") || { printf '%s\n' "$out"; echo "$name is not the same as itself"; return 1; }
        printf '%s\n' "$out" | grep -q '^same within tolerance' || { printf '%s\n' "$out"; return 1; }
    done
    awk '/<polyline/ && !cut { cut = 1; next } { print }' "$CRATES/fixtures/swift/gallery-gamma.svg" > "$dir/cut.svg"
    out=$("$CLI" compare "$CRATES/fixtures/swift/gallery-gamma.svg" "$dir/cut.svg")
    [ $? = 1 ] && printf '%s\n' "$out" | grep -q '^DIFFERENT' ||
        { printf '%s\n' "$out"; echo "a drawing with a stroke taken out was the same"; return 1; }
    echo "$(ls "$CRATES"/fixtures/swift/*.svg | wc -l | tr -d ' ') references the same as themselves; one cut is different"
}
step "kurven-cli compare: a reference is itself, and not itself with a stroke taken out" compare

wasm() {
    workspace build --profile web --target wasm32-unknown-unknown -p kurven-wasm || return 1
    echo "kurven_wasm.wasm: $(wc -c < "$WASM" | tr -d ' ') bytes"
}
step "wasm build" wasm

# The same golden sets through the wasm module under Node, byte for byte.
identity() {
    local native="$WORK/native" browser="$WORK/wasm"
    rm -rf "$native" "$browser"
    "$CLI" fixtures "$native" >/dev/null &&
        node "$CRATES/kurven-wasm/goldens.mjs" "$WASM" "$browser" &&
        diff -r "$native" "$browser" &&
        echo "$(find "$native" -type f | wc -l | tr -d ' ') golden file(s) byte-identical, native and wasm"
}
step "wasm under Node writes the golden sets native writes, byte for byte" identity

# The page: its pinned packages installed when the lock is newer than the
# install, then a type check and a build.
page() {
    if [ ! -f "$APP/node_modules/.package-lock.json" ] ||
        [ "$APP/package-lock.json" -nt "$APP/node_modules/.package-lock.json" ]; then
        npm --prefix "$APP" ci --no-audit --no-fund || return 1
    fi
    npm --prefix "$APP" run --silent check && npm --prefix "$APP" run --silent build
}
step "the page type checks and builds" page

printf '\n%ds\n' $((SECONDS - started))
if [ "$failures" = 0 ]; then
    printf '\033[32mall green\033[0m\n'
    exit 0
fi
printf '\033[31m%d step(s) failed\033[0m\n' "$failures"
exit 1
