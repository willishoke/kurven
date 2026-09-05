#!/bin/bash
# Run everything, in both lanes.
#
#   scripts/check.sh [--quick] [--clean]
#
# --quick skips the end-to-end comparisons, which sample and contour the real
# grids and take a couple of minutes. --clean rebuilds the Swift package from
# scratch.
#
# Why --clean exists, and why this script forces a relink by default: SwiftPM's
# incremental build does not reliably rebuild or relink a target when a type's
# layout or a function's signature changes in a library it depends on. The
# symptom is not a compile error -- it is a segfault in unrelated code, or a
# missing symbol at link time. It has happened four times in this repository's
# short Swift history, so the cheap defence is on by default.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE="$ROOT/KurvenSwift"
PYTHON="$ROOT/.venv/bin/python"
QUICK=0
CLEAN=0
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=1 ;;
        --clean) CLEAN=1 ;;
        *) echo "usage: $0 [--quick] [--clean]" >&2; exit 2 ;;
    esac
done

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

if [ "$CLEAN" = 1 ]; then
    rm -rf "$PACKAGE/.build"
else
    # Force the executables to relink against whatever the libraries now are.
    touch "$PACKAGE"/Sources/kurven-test/*.swift \
          "$PACKAGE"/Sources/kurven-cli/*.swift \
          "$PACKAGE"/Sources/KurvenApp/*.swift 2>/dev/null
fi

step "swift build (release)" swift build -c release --package-path "$PACKAGE"
BIN="$PACKAGE/.build/release"

step "swift lane" "$BIN/kurven-test"
step "python lane" "$PYTHON" "$ROOT/tests/check_bundle.py"
step "schema round trip, cross-language" \
    "$BIN/kurven-cli" contract "$ROOT/tests/fixtures/contract"

if [ "$QUICK" = 0 ]; then
    for example in recip elliptic zeta; do
        step "bake vs plate: $example" \
            "$PYTHON" "$ROOT/tests/compare_bake.py" "$example"
        step "preview vs plate: $example" \
            "$PYTHON" "$ROOT/tests/compare_preview.py" "$example"
    done
    # gamma at its published settings is ten thousand squared and adaptive; a
    # bundle carries one uniform grid, so this is the --no-adaptive form of it
    # at a size that finishes. See examples/gamma.py: SCENE_CAVEATS.
    GAMMA=(--res 700 --no-adaptive --surface-res 700 --buffer 3000)
    step "bake vs plate: gamma" \
        "$PYTHON" "$ROOT/tests/compare_bake.py" gamma "${GAMMA[@]}"
    step "preview vs plate: gamma" \
        "$PYTHON" "$ROOT/tests/compare_preview.py" gamma "${GAMMA[@]}"
fi

printf '\n'
if [ "$failures" = 0 ]; then
    printf '\033[32mall green\033[0m\n'
    exit 0
fi
printf '\033[31m%d step(s) failed\033[0m\n' "$failures"
exit 1
