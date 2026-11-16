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
# Needs cargo with the wasm32-unknown-unknown target, and Node with npm (the
# page's type check and build install web/app's pinned packages on first use).

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CRATES="$ROOT/crates"
APP="$ROOT/web/app"
WORK="$CRATES/target/check-web"
CLI="$CRATES/target/release/kurven-cli"
WASM="$CRATES/target/wasm32-unknown-unknown/web/kurven_wasm.wasm"
# A cargo command on the workspace, wherever this is run from.
workspace() {
    local command="$1"; shift
    cargo "$command" --manifest-path "$CRATES/Cargo.toml" "$@"
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

need() {
    command -v cargo >/dev/null || { echo "no cargo"; return 1; }
    rustup target list --installed 2>/dev/null | grep -qx wasm32-unknown-unknown ||
        { echo "no wasm32-unknown-unknown target: rustup target add wasm32-unknown-unknown"; return 1; }
    command -v node >/dev/null && command -v npm >/dev/null || { echo "no node and npm"; return 1; }
    cargo --version && node --version
}
step "toolchain: cargo with the wasm target, node and npm" need

step "rustfmt" workspace fmt --all -- --check

step "clippy, native" workspace clippy --release --workspace --all-targets -- -D warnings
step "clippy, wasm" workspace clippy --release -p kurven-wasm --target wasm32-unknown-unknown -- -D warnings

# The std spellings of the transcendental functions, and of fused
# multiply-add, anywhere in the engine but kurven_math::real. Type-blind, so
# it also catches what clippy cannot see; a method of the port's own of one
# of these names would trip it too (crates/FINDINGS.md). Lines that are
# comments are not code.
libm() {
    local names='sin|cos|tan|sin_cos|asin|acos|atan|atan2|sinh|cosh|tanh|asinh|acosh|atanh|exp|exp2|exp_m1|ln|log|log2|log10|ln_1p|powf|powi|hypot|cbrt|mul_add|gamma|ln_gamma'
    local pattern="\\.($names)\\(|(^|[^A-Za-z0-9_])f(32|64)::($names)([^A-Za-z0-9_]|\$)"
    local hits
    hits=$(grep -rnE --include='*.rs' "$pattern" "$CRATES"/*/src "$CRATES"/*/tests 2>/dev/null |
        grep -v "^$CRATES/kurven-math/src/real.rs:" |
        grep -vE '^[^:]+:[0-9]+:[[:space:]]*//')
    [ -z "$hits" ] && return 0
    printf '%s\n' "$hits"
    echo "call kurven_math::real instead: one libm on native and wasm"
    return 1
}
step "no std transcendental or fused multiply-add outside kurven_math::real" libm

step "native tests" workspace test --release --workspace

# Regenerated in place from scratch, so a file the writer no longer writes
# shows as deleted, and compared with the commit; untracked files count.
golden() {
    workspace build --release -p kurven-cli || return 1
    rm -rf "$CRATES/fixtures/rust"
    "$CLI" fixtures "$CRATES/fixtures/rust" >/dev/null || return 1
    git -C "$ROOT" add --intent-to-add crates/fixtures/rust
    git -C "$ROOT" diff --exit-code --stat -- crates/fixtures/rust
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
