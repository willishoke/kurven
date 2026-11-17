#!/bin/bash
# The Swift app's answers to the session's edit scripts.
#
#   crates/fixtures/swift/session/make.sh [NAME ...]
#
# Runs Swift's KurvenApp (`swift build -c release` first) headlessly over every
# line of scripts.txt, or the lines named, each into a directory of its own
# here, and keeps what the gate compares: each saved bundle's manifest (a
# landscape's grids are left out: the gate holds manifests byte for byte, and
# a resampled grid is Swift's libm's) and each baked drawing. What the app
# printed goes to $TMPDIR, not here: its statuses carry times. The inputs
# under inputs/ were written by Swift's kurven-cli (see FINDINGS, Phase 6).

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
APP="$ROOT/.build/release/KurvenApp"
LOG="${TMPDIR:-/tmp}/kurven-session-swift"
[ -x "$APP" ] || { echo "no $APP: swift build -c release"; exit 1; }
mkdir -p "$LOG"

while read -r name distance length input rest; do
    case "$name" in '' | '#'*) continue ;; esac
    if [ $# -gt 0 ] && ! printf '%s\n' "$@" | grep -qx "$name"; then continue; fi
    out="$HERE/$name"
    rm -rf "$out" && mkdir -p "$out"
    args=()
    [ "$input" != - ] && args+=("$HERE/inputs/$input.kurven")
    for a in $rest; do
        case "$a" in @*) args+=("$out/${a#@}") ;; *) args+=("$a") ;; esac
    done
    # Run from the output directory, so nothing the app leaves lands elsewhere.
    if (cd "$out" && "$APP" "${args[@]}" > "$LOG/$name.out" 2> "$LOG/$name.err"); then
        status=0
    else
        status=$?
    fi
    find "$out" -name '*.npy' -delete
    echo "$name: exit $status, $(find "$out" -type f | wc -l | tr -d ' ') file(s)"
    [ "$status" = 0 ] || { cat "$LOG/$name.out" "$LOG/$name.err"; exit 1; }
done < "$HERE/scripts.txt"
