# Development

## Building

kurven is a pure SwiftPM package with no third-party dependencies and no Xcode
requirement: macOS 15 or later, the Command Line Tools, and a Metal device.
Shaders compile at runtime from a string, and the test suite is an executable
rather than a `.testTarget` (the Command Line Tools ship `Testing.framework`
without a `.swiftmodule`).

```bash
swift build -c release --package-path KurvenSwift
scripts/bundle-app.sh                     # assembles build/Kurven.app
```

Build release for anything larger than a smoke test: the readback and the
judge are tight scalar loops, and unoptimized Swift bounds-checks every element
of a several-hundred-megabyte buffer.

Swift is the only implementation. Python is in the repository for one
purpose: where kurven implements something a third-party library already
implements, that library is the oracle, and `tests/oracles.py` asks it for the
answers and writes them to `tests/fixtures`. Nothing in the build or the checks
runs Python.

## Checks

```bash
scripts/check.sh            # everything, both lanes
scripts/check.sh --clean    # the same, rebuilding the package from scratch
```

The Swift lane is the one that draws, and it is held to analytic references
rather than to a second implementation: the depth pass to an exact ray cast of
the heightfield, the refined contours to the function they were placed by,
every judged line to a brute-force cast through the drawn solid (the stress
suite, which also demands the drawing not change with resolution), and a
surface's visibility to its analytic answer. The end-to-end comparisons that
drew every plate in both languages were the cutover's test and are gone: they
had come to pin the grid contours the refiner moved past.

The fixtures in `tests/fixtures` are in git and are of two kinds.

- **Oracle** sets are a third-party library's answers. `expr/` holds every
  function of the language to numpy, scipy.special or mpmath (scipy's zeta
  and ellipj are real-only) on a grid that includes the real axis exactly,
  plus each special function on the window its landscape uses; `contour/`
  holds the native marching squares to contourpy on three awkward grids;
  `npy/` holds the reader to files numpy wrote. `tests/oracles.py` writes
  them, names the library behind each function, and refuses a language table
  with a function it has no oracle for. CI never runs it: regenerate by hand
  when an oracle or a case changes, and commit the result on its own.
- **Golden** sets are kurven's own answers, written by
  `kurven-cli fixtures tests/fixtures`: the schema round-trip bundles
  (`contract/`), the plate cameras (`camera/`), the hidden-line clip
  (`clip/`), every hatching kind (`hatch/`), and three native landscapes with
  their specs (`expr/landscape_*`, `expr/landscapes.json`). A golden file proves
  nothing about correctness; it proves that a change to the output was made on
  purpose. `scripts/check.sh` regenerates them in place and fails on any
  difference, so a change to one is a regeneration committed on its own, with
  the diff in the message. A few files in these sets are frozen *inputs* (the
  contract bundles' noise layers, the camera's points, the clip fixture's depth
  buffer and probes), which the writer reads rather than writes.

`expr/language.json` is neither: the canonical spellings of the ambiguous
readings, the error positions and the function table, frozen as the retired
Python implementation of the language reported them, and edited by hand.

Or a piece at a time:

```bash
swift run -c release --package-path KurvenSwift kurven-test        # the Swift lane
swift run -c release --package-path KurvenSwift kurven-cli fixtures tests/fixtures
uv run tests/oracles.py                                            # the oracle sets
```

Run `kurven-test` with no argument; it finds the fixtures itself. The oracles'
dependencies (numpy, scipy, mpmath, contourpy) are declared inline at the top
of `tests/oracles.py` and pinned in `tests/oracles.py.lock`; `uv` makes the
environment from that lock on first use.

**CI** (`.github/workflows/check.yml`) runs `scripts/check.sh` on every push
to main and every pull request, on an Apple Silicon runner so the Swift lane
draws with Metal. Run by hand with the `oracles` input, it also regenerates the
oracle sets under uv and diffs them, which is the check to run after a scipy or
mpmath bump.

A check that cannot run on a machine is *skipped* and listed in the summary
rather than failed, with one exception: a suite skipped for want of a Metal
device fails the run, because the GPU suites are the ones that draw.
`KURVEN_TEST_SKIP_GPU=1` says a machine is known to have none.

## Project structure

```
KurvenSwift/Sources/
  KurvenMath/      complex arithmetic, the special functions, the expression
                   language; depends on nothing
  KurvenCore/      pure values: spaces, camera, navigation, contours, the
                   heightfield and its judge, hatching, solids, npy, SVG
  KurvenLandscape/ a landscape sampled here: the catalog, the refiner, the
                   derived styling, the bundle
  KurvenDynamics/  parametric surfaces: tori, doubly periodic functions,
                   forced systems
  KurvenMetal/     the depth pass, the preview, the resource cache
  KurvenBake/      scene -> strokes; tiling; PNG
  KurvenApp/       the window, the picker, the landscape as a set of controls
  kurven-cli/      landscape, bake, surface, preview, bench, refine, catalog,
                   inspect, contract, fixtures, ...
  kurven-test/     the test suite
tests/             oracles.py, its lock, and the fixtures
scripts/           check.sh, bundle-app.sh
docs/gallery/      the README's figures and the script that bakes them
```

## Bundles from the Python plates

The four plates of the original atlas were Python programs, retired with the
rest of the Python package; their bundles still open, since reading is native.
A plate bundle records the plate's arguments in `provenance.params`, and when
its `provenance.function` is a catalog preset the app fills the landscape
controls from them (`rMin`, `rMax`, `iMin`, `iMax` the window, `res` the
resolution, `zLimit` the cap) and Rebuild samples it natively. One whose
function is not in the catalog opens read-only.

```bash
KurvenApp --open recip.kurven --resample res=800 --screenshot recip.png
```

## Things worth knowing

- **SwiftPM relinks.** SwiftPM's incremental build does not reliably rebuild
  or relink a target when a type's layout or a function's signature changes in
  a library it depends on. The symptom is not a compile error but a segfault
  in unrelated code, or a missing symbol at link time. `scripts/check.sh`
  forces the executables to relink by default; `--clean` rebuilds everything.
- **Launching the app from a script.** From a non-interactive shell (a CI job,
  an agent's subprocess), a bare file argument makes AppKit treat the launch
  as "opened with a file", and SwiftUI's `WindowGroup` then declines to create
  its default window. Use `--open PATH`. The Finder and `open -a` were never
  affected.
- **The README's figures** are baked by `python3 docs/gallery/make.py` from a
  release build of the CLI. Rerun it after a change to how plates are drawn.
