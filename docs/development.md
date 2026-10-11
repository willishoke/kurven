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

Swift is the one canonical implementation. Python remains for three things:
the four published plates (`examples/`), which are Python programs; the
fixture generator; and the Python lane of the tests, which uses scipy as the
reference for the special functions. It is not a second backend, and new
behaviour is not mirrored into it. `uv` finds or makes its environment from
the lockfile on first use, so a fresh clone or worktree needs no setup step;
`KURVEN_PYTHON` names an interpreter explicitly.

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

What remains of the cross-language contract is in files. `tests/make_fixtures.py`
writes `tests/fixtures/`; both lanes read the same files, and the files are in
git, the three landscape bundles included, so the Swift lane needs nothing but
a checkout.

- **The schema round trip.** A fixture manifest decoded by Swift and
  re-encoded must come back byte for byte. That is the only check that the two
  schema definitions agree, and it is why the Swift mirror needs no codegen.
- **The hatching.** `tests/fixtures/hatch` is every one of the four hatching
  kinds derived by `kurven/hatch.py` from one grid, one perimeter and two kinds
  of truncation (one height, and a height per band); the Swift lane derives them again from the same description and
  compares vertex for vertex.
- **The functions.** `tests/fixtures/expr` holds every function of the
  language to scipy on a grid that includes the real axis exactly.

Or a piece at a time:

```bash
swift run -c release --package-path KurvenSwift kurven-test   # the Swift lane
python tests/check_bundle.py           # python lane: schema, CSR, camera, clip
python tests/check_expr.py             # the expression language and its numerics
python tests/make_fixtures.py          # regenerate the fixtures
```

Run `kurven-test` with no argument; it finds the fixtures itself.

**CI** (`.github/workflows/check.yml`) runs `scripts/check.sh` on every push
to main and every pull request, on an Apple Silicon runner so the Swift lane
draws with Metal. Before it, CI regenerates the fixtures and compares them
with what is committed, with the git SHA each manifest records as the one
line allowed to differ.

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
  KurvenService/   the Python plates over a pipe
  KurvenApp/       the window, the picker, the landscape as a set of controls
  kurven-cli/      landscape, bake, surface, preview, bench, refine, catalog,
                   inspect, contract, ...
  kurven-test/     the Swift lane of the tests
kurven/            the Python library behind the published plates, the bundle
                   schema's Python side, and the hatching reference
examples/          the published plates as Python programs: gamma, elliptic,
                   zeta, recip_factorial; function.py for any landscape
tests/             the fixture generator and the Python lane
scripts/           check.sh, bundle-app.sh
docs/gallery/      the README's figures and the script that bakes them
```

## The published plates

The four plates of the original atlas are Python programs, and a frozen bundle
of one cannot change its own domain or resolution. `python -m kurven.export`
writes one as a bundle, and `kurven-cli resample` asks the service to rebuild
one; the app opens either.

```bash
python -m kurven.export recip -o recip.kurven --res 1600 --derived
python examples/gamma.py --gpu          # the full plate: ~1 min with a GPU, ~10 on CPU
```

`--derived` writes descriptions instead of arrays wherever it can. Not every
layer can be described: elliptic's phase contours are trimmed by a rule
written for that one plate, and those stay dumped, which the schema says
rather than hides.

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
- **zeta's plate samples its own grid.** `examples/zeta.py` was born from a
  precomputed ζ grid in the author's notes; where that file is absent it looks
  for `$KURVEN_ZETA_CACHE`, then `~/.cache/kurven/zeta_5000.npy`, and failing
  both samples the grid itself (three seconds) and saves it there.
- **The README's figures** are baked by `python3 docs/gallery/make.py` from a
  release build of the CLI. Rerun it after a change to how plates are drawn.
