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
# `kurven-cli compare` to a tolerance. The first two run as the native
# tests; the drawings the port bakes so far are baked by the Rust
# `kurven-cli references` and compared in a step of their own, each under its
# line of crates/fixtures/references.txt.
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
# The page's module: the session alone, without the golden sets.
PAGE_WASM="$CRATES/target/page/wasm32-unknown-unknown/web/kurven_wasm.wasm"
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
step "clippy, the page's wasm" workspace clippy --release -p kurven-wasm --target wasm32-unknown-unknown --no-default-features -- -D warnings

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

# The drawings against Swift's: every reference drawing the port bakes,
# baked by `kurven-cli references` and held to Swift's by `kurven-cli
# compare` under its line of crates/fixtures/references.txt, which says
# where each tolerance came from and why a layer is skipped. The layers'
# names must be Swift's, in Swift's order (each index.json lists them), and a
# drawing with no line there fails.
references() {
    local dir="$WORK/references" table="$CRATES/fixtures/references.txt"
    local name distance length skip out failed=0 held=0 baked listed args
    rm -rf "$dir"
    "$CLI" references "$ROOT/tests/fixtures" "$dir" >/dev/null || return 1
    baked=$(node -e 'for (const d of JSON.parse(require("fs").readFileSync(process.argv[1])).drawings) console.log(d.name)' "$dir/index.json") || return 1
    listed=$(awk '!/^#/ && NF { print $1 }' "$table")
    for name in $baked; do
        printf '%s\n' "$listed" | grep -qx "$name" || { echo "$name is baked and has no line in crates/fixtures/references.txt"; failed=1; }
    done
    for name in $(printf '%s\n' "$listed" | sort | uniq -d); do
        echo "$name has more than one line in crates/fixtures/references.txt"
        failed=1
    done
    # The same layer names, in the same order, as Swift's index lists.
    node -e '
        const fs = require("fs");
        const read = (p) => Object.fromEntries(JSON.parse(fs.readFileSync(p)).drawings.map((d) => [d.name, d.layers.join(" ")]));
        const [rust, swift] = [read(process.argv[1]), read(process.argv[2])];
        let bad = 0;
        for (const name of Object.keys(rust)) {
            if (rust[name] !== swift[name]) { console.log(`${name}: layers [${rust[name]}] against Swift [${swift[name]}]`); bad = 1; }
        }
        process.exit(bad);' "$dir/index.json" "$CRATES/fixtures/swift/index.json" || failed=1
    # Every line is four fields; a last line without its newline is read
    # too, not dropped uncompared.
    local line fields
    while IFS= read -r line || [ -n "$line" ]; do
        read -r -a fields <<< "$line"
        case "${fields[0]:-}" in '' | '#'*) continue ;; esac
        if [ "${#fields[@]}" != 4 ]; then
            echo "crates/fixtures/references.txt: want name, distance, length and skip (- for none): $line"
            failed=1
            continue
        fi
        name=${fields[0]} distance=${fields[1]} length=${fields[2]} skip=${fields[3]}
        args=(--tolerance "$distance" --length-tolerance "$length")
        [ "$skip" != - ] && args+=(--skip "$skip")
        if ! out=$("$CLI" compare "$CRATES/fixtures/swift/$name.svg" "$dir/$name.svg" "${args[@]}"); then
            printf '%s\n%s\n' "$name" "$out"
            failed=1
        fi
        held=$((held + 1))
    done < "$table"
    [ "$failed" = 0 ] && echo "$held drawing(s) the same as Swift's within their tolerances"
    return $failed
}
step "the port's reference drawings are Swift's, within crates/fixtures/references.txt" references

# The session against Swift's app: every edit script of
# crates/fixtures/swift/session/scripts.txt replayed by `kurven-cli app`, each
# saved bundle's manifest byte for byte the one Swift's KurvenApp saved, and
# each baked drawing held to Swift's by `kurven-cli compare` at the line's
# tolerances. A file Swift wrote that the port did not, or the other way
# round (but a landscape's grids, which make.sh leaves out), fails.
session() {
    local here="$CRATES/fixtures/swift/session" dir="$WORK/session"
    local name distance length input rest out a f r failed=0 held=0 drawings=0
    rm -rf "$dir"
    while read -r name distance length input rest; do
        case "$name" in '' | '#'*) continue ;; esac
        out="$dir/$name"
        mkdir -p "$out" || return 1
        local args=()
        [ "$input" != - ] && args+=("$here/inputs/$input.kurven")
        for a in $rest; do
            case "$a" in @*) args+=("$out/${a#@}") ;; *) args+=("$a") ;; esac
        done
        if ! (cd "$out" && "$CLI" app "${args[@]}" > "$out/.stdout" 2>&1); then
            echo "$name: the port's app failed"
            cat "$out/.stdout"
            failed=1
            continue
        fi
        [ -d "$here/$name" ] || { echo "$name: Swift wrote nothing; run crates/fixtures/swift/session/make.sh $name"; failed=1; continue; }
        local theirs ours
        theirs=$(cd "$here/$name" && find . -type f | sort)
        ours=$(cd "$out" && find . -type f ! -name '*.npy' ! -name .stdout | sort)
        if [ "$theirs" != "$ours" ]; then
            echo "$name: Swift wrote [$theirs], the port [$ours]" | tr '\n' ' '
            echo
            failed=1
            continue
        fi
        for f in $theirs; do
            case "$f" in
                *manifest.json)
                    if cmp -s "$here/$name/$f" "$out/$f"; then held=$((held + 1)); else
                        echo "$name: $f is not Swift's"
                        diff "$here/$name/$f" "$out/$f" | head -c 2000
                        failed=1
                    fi
                    ;;
                *.svg)
                    if r=$("$CLI" compare "$here/$name/$f" "$out/$f" --tolerance "$distance" --length-tolerance "$length"); then
                        drawings=$((drawings + 1))
                    else
                        printf '%s: %s\n%s\n' "$name" "$f" "$r"
                        failed=1
                    fi
                    ;;
            esac
        done
    done < "$here/scripts.txt"
    [ "$failed" = 0 ] && echo "$held manifest(s) byte for byte Swift's, $drawings drawing(s) within their tolerances"
    return $failed
}
step "the session's edit scripts give Swift's bundles and drawings" session

# Two builds: the check's module, with the golden sets and the timing
# exports, and the page's, without them (`--no-default-features`), in a
# build tree of its own so neither overwrites the other.
wasm() {
    workspace build --profile web --target wasm32-unknown-unknown -p kurven-wasm || return 1
    echo "kurven_wasm.wasm: $(wc -c < "$WASM" | tr -d ' ') bytes, with the golden sets"
    (cd "$CRATES" && CARGO_TARGET_DIR="$CRATES/target/page" cargo build --profile web \
        --target wasm32-unknown-unknown -p kurven-wasm --no-default-features) || return 1
    echo "kurven_wasm.wasm: $(wc -c < "$PAGE_WASM" | tr -d ' ') bytes, the page's"
}
step "wasm build" wasm

# The same golden sets through the wasm module under Node, byte for byte.
identity() {
    local native="$WORK/native" browser="$WORK/wasm"
    rm -rf "$native" "$browser"
    "$CLI" fixtures "$native" >/dev/null &&
        node "$CRATES/kurven-wasm/goldens.mjs" "$WASM" "$browser" &&
        diff -r "$native" "$browser" &&
        echo "$(find "$native" -type f | wc -l | tr -d ' ') golden file(s) byte-identical, native and wasm" || return 1
    # The session's edit scripts through the module's session, against what
    # `kurven-cli app` wrote natively in the session step: every bundle file
    # and drawing, the grids of a resampled landscape among them.
    local here="$CRATES/fixtures/swift/session" ran="$WORK/session" through="$WORK/session-wasm"
    [ -d "$ran" ] || { echo "no native session run to compare: the session step did not run"; return 1; }
    # Through both modules: the check's, and the page's, which is what a
    # browser loads.
    local module label
    for module in "$WASM" "$PAGE_WASM"; do
        label="the check's"
        [ "$module" = "$PAGE_WASM" ] && label="the page's"
        rm -rf "$through"
        node "$CRATES/kurven-wasm/session.mjs" "$module" "$here/scripts.txt" "$here/inputs" "$through" &&
            diff -r -x .stdout "$ran" "$through" &&
            echo "$(find "$through" -type f | wc -l | tr -d ' ') session file(s) byte-identical, native and $label wasm" ||
            return 1
    done
}
step "wasm under Node writes the golden sets and the session's files native writes, byte for byte" identity

# Whether every file under the directories given (relative to the root) is
# the commit's: none ignored by .gitignore, none untracked, none changed
# since the commit. Says which, and fails, otherwise.
committed() {
    local ignored untracked failed=0
    ignored=$(git -C "$ROOT" ls-files --others --ignored --exclude-standard -- "$@")
    untracked=$(git -C "$ROOT" ls-files --others --exclude-standard -- "$@")
    if [ -n "$ignored" ]; then
        printf 'ignored by .gitignore, so never committed: %s\n' $ignored
        failed=1
    fi
    if [ -n "$untracked" ]; then
        printf 'not in the commit: %s\n' $untracked
        failed=1
    fi
    if ! git -C "$ROOT" diff --quiet HEAD -- "$@"; then
        git -C "$ROOT" diff --stat HEAD -- "$@"
        failed=1
    fi
    return $failed
}

# The page's requests of steps 4 to 7 (crates/fixtures/page/*.jsonl: the
# picker's thumbnails, a surface's controls, a banded truncation dragged, a
# face picked and painted, bundles opened from files and saved), every
# reply -- its JSON and its arrays -- through the page's module under Node,
# byte for byte what `kurven-cli session --replies` answers natively.
requests() {
    local here="$CRATES/fixtures/page" inputs="$CRATES/fixtures/swift/session/inputs" f name
    # What the step reads is the commit's: a fixture .gitignore leaves out,
    # one not added, or one changed since, is refused here rather than
    # missed on a machine that checks out the commit (PR #77's first CI run
    # had no levels.kurven).
    committed crates/fixtures/page crates/fixtures/swift/session/inputs || return 1
    local stages=(--stage frame.kurven "$inputs/frame.kurven" --stage gamma.kurven "$inputs/gamma.kurven"
        --stage relief.kurven "$inputs/relief.kurven" --stage cube.kurven "$inputs/cube.kurven"
        --stage levels.kurven "$here/levels.kurven")
    for f in "$here"/*.jsonl; do
        name="$(basename "$f" .jsonl)"
        rm -rf "$WORK/requests-native/$name" "$WORK/requests-wasm/$name"
        "$CLI" session "$f" "${stages[@]}" --replies "$WORK/requests-native/$name" &&
            node "$CRATES/kurven-wasm/requests.mjs" "$PAGE_WASM" "$f" "$WORK/requests-wasm/$name" "${stages[@]}" &&
            diff -r "$WORK/requests-native/$name" "$WORK/requests-wasm/$name" &&
            echo "$name: $(find "$WORK/requests-native/$name" -type f | wc -l | tr -d ' ') reply file(s) byte-identical, native and the page's wasm" ||
            return 1
    done
}
step "the page's requests (steps 4 to 7) answered the same natively and in wasm, byte for byte" requests

# The Worker protocol end to end under Node (worker_threads, no browser): the
# page's client (web/app/src/engine/client.ts) driving the wasm module in a
# Worker, its saved manifest Swift's and its drawing native's, a superseded
# view dropped, and a hung edit cancelled and the document replayed; against
# the check's module and the page's.
worker() {
    node "$APP/test/worker.test.mjs" "$WASM" "$CRATES/fixtures/swift/session" "$WORK/session" &&
        node "$APP/test/worker.test.mjs" "$PAGE_WASM" "$CRATES/fixtures/swift/session" "$WORK/session"
}
step "the Worker protocol under Node" worker

# The pool of engines under Node: helpers in worker_threads Workers holding
# the page's document (web/app/src/engine/pool.ts). The judged strokes a
# pool of 1, 2 and 4 bakes are the bytes the page's engine bakes alone and
# the native session wrote; helpers catch up on edits, one killed mid-bake
# is restarted and its piece handed out again, a cancel and a stale bake
# end; a frame at rest served from the helpers' layouts is the primary's
# own. Against the check's module and the page's.
pool() {
    node "$APP/test/pool.test.mjs" "$WASM" "$CRATES/fixtures/swift/session" "$WORK/session" &&
        node "$APP/test/pool.test.mjs" "$PAGE_WASM" "$CRATES/fixtures/swift/session" "$WORK/session"
}
step "the pool of engines under Node: pooled bakes are native's bytes" pool

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
