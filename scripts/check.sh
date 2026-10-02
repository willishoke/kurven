#!/bin/bash
# Run everything, in both lanes.
#
#   scripts/check.sh [--clean]
#
# --clean rebuilds the Swift package from scratch.
#
# The Swift lane is the one that draws: it holds the depth pass to an exact
# ray cast, the contours to the function they were placed by, and the
# surfaces to their analytic visibility. The Python lane checks the Python
# package against the same fixture files the Swift lane reads, which is what
# keeps the bundle schema one definition. Nothing here draws a plate twice
# and compares the two: that was the cutover's test, and the cutover is done.
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

# The Python lane's interpreter, in the same order of preference as
# `Service.Command.autodetect`: an explicit `KURVEN_PYTHON`; otherwise uv, which
# creates and syncs the project's virtualenv from the lockfile on first use, so
# a fresh clone or a git worktree needs no setup step; otherwise a virtualenv
# somebody made by hand.
if [ -n "${KURVEN_PYTHON:-}" ]; then
    PYTHON=("$KURVEN_PYTHON")
elif command -v uv >/dev/null 2>&1; then
    PYTHON=(uv run --project "$ROOT" --frozen --extra gpu python)
else
    PYTHON=("$ROOT/.venv/bin/python")
fi
CLEAN=0
for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN=1 ;;
        *) echo "usage: $0 [--clean]" >&2; exit 2 ;;
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
step "python lane" "${PYTHON[@]}" "$ROOT/tests/check_bundle.py"
step "expression language" "${PYTHON[@]}" "$ROOT/tests/check_expr.py"
step "schema round trip, cross-language" \
    "$BIN/kurven-cli" contract "$ROOT/tests/fixtures/contract"

printf '\n'
if [ "$failures" = 0 ]; then
    printf '\033[32mall green\033[0m\n'
    exit 0
fi
printf '\033[31m%d step(s) failed\033[0m\n' "$failures"
exit 1
