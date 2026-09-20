# kurven

A Python library for rendering **analytic landscapes** — the 3D surface visualization of complex functions pioneered by Jahnke and Emde in their 1909 atlas *Tafeln höherer Funktionen*. The height of the surface at each point z is |f(z)|; magnitude and phase contour lines are projected onto the surface and clipped against a depth buffer to produce a hidden-line-removed vector graphic.

The published plates are reproductions — each one a program that knows its
function, its window and where to cut its poles. **Landscapes you choose** are
the general form of the same thing: pick a function (or write one), drag the
window and the truncation, and the plate follows. See *[Landscapes you
choose](#landscapes-you-choose)*.

## Gallery

### Γ(z) — Gamma function

![Gamma function analytic landscape](docs/gamma_hi_res.svg)

The upper half-plane Re(z) ∈ [−4.5, 4.5], Im(z) ∈ [0, 2.5]. Four pole spires rise at the non-positive integers; the surface is truncated at a consistent height per spire so each cap is visible from above.

---

### cn(z, m) — Jacobi elliptic function

![Jacobi elliptic cn analytic landscape](docs/elliptic_hi_res.svg)

The doubly periodic function cn(z, m = 0.64). One fundamental tile [−K, 0] × [−K′, 0] is sampled and reflected across the quarter-period lattice to fill a 3×6 plate; a spire rises at each tile corner where cn has a pole, every surface truncated at |cn| = 4 so the caps read from above. Magnitude and phase contours are draped over the landscape and hidden-line-clipped against a depth buffer meshed from the surface itself — the clamped |cn| heightfield plus the vertical cut-face walls — so the front-left cutout, which exposes the cross-section through one spire, occludes what lies behind it.

---

## Algorithm

The pipeline has six stages: **sample → contour → lift → project → depth-buffer → clip**.

### 1. Sample

Evaluate f(z) on a uniform grid of complex values. For functions with poles or rapid variation (e.g. Γ near the negative integers), an **adaptive sampler** (`kurven/sampling.py`) first probes a coarse grid to locate high-gradient regions via |∇ log|f||, then re-samples those zones at a finer density. This concentrates resolution where contours are densest without paying for it everywhere.

### 2. Contour

Extract iso-lines of |f(z)| and arg(f(z)) at a chosen set of levels using marching squares (via [contourpy](https://contourpy.readthedocs.io/)). Two families of curves are generated — magnitude and phase — each at major and minor spacings. The contourpy threaded backend parallelizes across levels; chunk-boundary seams are stitched by exact endpoint matching (`_stitch_chunk_seams`).

For functions with adaptive sampling, coarse contours that fall inside a fine zone are dropped; fine contours fill in. Where a coarse and fine path meet at a zone boundary they are welded together (`_stitch_paths`).

### 3. Lift to 3D

Each contour vertex (x, y) in the complex plane is lifted to (x, y, |f(x + iy)|). This turns the flat contour diagram into a network of curves draped over the magnitude surface.

### 4. Project

An isometric-style projection:

1. **Shear** — y′ = y + s·x (default s = 0.5) introduces foreshortening perpendicular to the viewer, matching the visual style of the original Jahnke-Emde plates.
2. **Rotate** — a Z rotation followed by an X rotation tilts the surface toward the viewer.

The result is a 3D point cloud in screen space.

### 5. Depth buffer

The magnitude surface is independently meshed as a triangulated grid and rasterized into a Z-buffer (`kurven/zbuffer.py`). Each pixel stores the maximum depth value seen so far. A GPU path via [moderngl](https://moderngl.readthedocs.io/) rasterizes with `GL_MAX` blending; the CPU fallback rasterizes triangle-by-triangle using barycentric interpolation.

### 6. Clip and outline

**Hidden-line removal** (`clip_hidden_lines`): each contour point is checked against the Z-buffer. Points behind the surface are invisible; the remaining runs are split into visible segments.

**Silhouette** (`extract_outline`): the Z-buffer is binarized (filled vs. empty) and the level-0 contour gives the outer boundary of the rendered shape — the "picture frame" silhouette.

Both are saved as SVG polylines via matplotlib.

---

## Landscapes you choose

A published plate is a program. `examples/gamma.py` knows that it is drawing Γ,
on [−4.5, 4.5] × [0, 2.5], with four pole spires cut at four different heights,
hatched down the front face at a density someone picked by eye. That is the
right way to reproduce a plate from 1909 and the wrong way to look at a
function nobody has drawn yet.

So the four facts that make a landscape — **which function, over what window,
truncated where, sampled how finely** — are a value (`kurven.landscape.Spec`),
and everything else is derived from them by rules stated once.

```bash
python examples/function.py --function gamma --gpu
python examples/function.py --expression "exp(1/z)" \
    --r-min -1 --r-max 1 --i-min -1 --i-max 1 --cap 4
kurven-cli landscape --function zeta --res 800 -o zeta.kurven   # via the service
```

### The expression

`kurven/expr.py` is a small complex expression language: its own tokenizer and
recursive-descent parser (no `eval`, no `ast`), over numpy and scipy.

```
gamma(z)      1/Γ(z)        zeta(z)        exp(1/z)      z^3 - 1
tan(z)        sin(z)cn(z, 0.64)            (z^2-1)/(z^2+1)          z!
```

Implicit multiplication (`2z`, `(z-1)(z+1)`), `^` for powers, postfix `!` for
Γ(z+1), unicode aliases (`Γ`, `ζ`, `ψ`, `π`), and 43 functions. Errors carry the
character they went wrong at, which is what lets the app's expression field say
where.

Two of those functions scipy has no complex form of, so they are implemented
here: **ζ(s)** (Borwein's alternating series, the functional equation below the
critical line, and Euler–Maclaurin at the points where Borwein's own
`1 − 2^(1−s)` factor vanishes — σ = 1, t ≈ 9.06, 18.13, …, which are inside the
ζ plate's own window), and the **Jacobi elliptics** sn, cn, dn. 200k samples of
ζ take 0.1 s, so the zeta landscape no longer needs the precomputed cache the
published plate loads.

The catalog (`kurven.landscape.CATALOG`) is fourteen presets over that language
— each an expression plus the window and truncation that make it read as a
landscape — and it is *data the server reports*, so a preset added in Python
appears in the app without a line of Swift changing.

### Truncation, and the hatching that goes with it

A pole makes a landscape unboundedly tall, and a picture of one is a needle and
a floor. `Caps` says where to cut: nowhere, at a height, or — gamma's case — at
a different height per band of Re, so each spire's plateau reads from above.

The cut faces and the truncated tops are then *shaded*, which is the part every
published plate wrote by hand, with hand-found spire radii and `while` loops
stepping outward until |f| fell under the limit. `kurven/hatch.py` is the
general form, and it is written into the bundle as four more described layer
kinds:

| kind | what it draws |
|---|---|
| `wallHatch` | vertical strokes up the cut faces, every `spacing` along each perimeter edge, from the ground to the capped surface |
| `wallOutline` | each face's crest and foot, and a post at each corner |
| `capHatch` | parallel strokes across each plateau, at the cap, ending exactly on the rim |
| `capOutline` | the rim itself: the zero contour of |f| − cap(x), which under band caps is not a contour of |f| at all |

### Two speeds of edit

Because those layers are *descriptions* rather than dumped strokes, changing
your mind costs one of two very different things:

- **the function, the window, the resolution** need f evaluated again. That is a
  request to Python (`kurven.serve`'s `landscape` method) and a new bundle:
  ~30 ms at 600², ~80 ms for ζ at 500².
- **the cap, the contour levels, the hatch spacing** need nothing but the grids
  already in memory. `KurvenBundle.restyled` re-derives the ink, the wall
  curtains and the plateaus, and the picture follows within a frame.

A cap slider therefore moves the heightfield the occluder meshes, the crest
every wall stroke rises to, every plateau and rim and cap stroke, and which
contours are cut off — all from one assignment, with no round trip.

## Project structure

```
kurven/
  expr.py        — the expression language: tokenizer, parser, and the
                   functions (including complex zeta and the Jacobi elliptics)
  landscape.py   — a landscape you choose: the catalog, the derived styling,
                   and the generic scene builder
  hatch.py       — the reference implementation of the four hatching kinds
  sampling.py    — uniform + adaptive grid evaluation, gradient-zone discovery
  contours.py    — marching-squares extraction, seam stitching, path lifting
  surface.py     — the sampled |f| landscape; contour lifting, heightfield mesh
  perimeter.py   — a boundary outline: walls, ground ink and mask from one definition
  occluder.py    — heightfield + wall curtains, tiled, as one mesh
  scaffold.py    — the drawn structural line-work (the ink twin of occluder.py)
  projection.py  — isometric shear + rotation
  zbuffer.py     — Z-buffer class, CPU and GPU triangle rasterizers
  outline.py     — hidden-line clipping, silhouette extraction
  scene.py       — Scene: the camera-independent half of a plate
  bundle.py      — the .kurven bundle: typed manifest + npy arrays
  export.py      — python -m kurven.export: a Scene, serialized
  pipeline.py    — thin convenience wrapper for the generic stages

examples/
  function.py    — any function's landscape: the plate that is not written in
                   advance, and the oracle for the derived hatching
  gamma.py       — Γ(z): faithful reproduction of the Jahnke-Emde gamma plate
                   (see SCENE_CAVEATS for what about it stays Python-only)
  elliptic.py    — cn(z, m): Jacobi elliptic function landscape
  zeta.py        — ζ(s): Riemann zeta function
  recip_factorial.py — 1/Γ(z): the reciprocal-factorial relief

KurvenSwift/     — the Swift/Metal frontend (see below)
  KurvenCore/    — pure values: spaces, camera, navigation, npy, clip, SVG,
                   and Hatch.swift, which derives the four hatchings
  KurvenMetal/   — the depth pass, the preview, the resource cache
  KurvenBake/    — scene -> strokes; tiling; PNG
  KurvenService/ — the Python half over a pipe: examples, and landscapes
  kurven-cli/    — bake, preview, depth, bench, inspect, contract, catalog,
                   landscape
  kurven-test/   — the Swift lane of the tests (an executable, not swift test)
  KurvenApp/     — the window, and the landscape as a set of controls
scripts/
  bundle-app.sh  — assembles Kurven.app (no Xcode required)
tests/
  make_fixtures.py  — writes tests/fixtures, the oracle both lanes are held to
  check_bundle.py   — the Python lane of the contract tests
  check_expr.py     — the expression language: parsing, safety, and the
                      numerics of zeta and the elliptics
  compare_bake.py   — end-to-end: the Swift bake against the Python plate
  verify_refactor.py — pixel-identical before/after diffing for refactors
```

## Installation

```bash
uv sync            # CPU only
uv sync --extra gpu  # + moderngl for GPU rasterization
```

## Running the examples

```bash
# Gamma (default: res=10000, buffer=20000 — takes ~10 min on CPU, ~1 min with GPU)
python examples/gamma.py --gpu

# Smoke-test at lower res
python examples/gamma.py --res 800 --buffer 1600

# Elliptic cn
python examples/elliptic.py --gpu

# Both write <prefix>_hi_res.svg (and gamma also writes <prefix>_raw.svg)
python examples/gamma.py --gpu --output-prefix out/my_gamma

# A landscape of your own, from the catalog or from an expression
python examples/function.py --function tan --gpu
python examples/function.py --expression "besselj(0, z)" --r-min -12 --r-max 12
python examples/function.py --function gamma \
    --bands "-3.5<3.1,-2.5<4.2,-1.5<5,0.5<5.6,5"     # a cap per spire
```

## The camera seam, and the Swift frontend

The pipeline divides cleanly in two, and not where you would expect. The seam
is not "library versus application" but **camera-independent** work (sample →
contour → lift) versus **camera-dependent** work (project → depth-buffer → clip
→ ink). The first half is the expensive one, needs scipy, and does not change
when you move the camera; the second half is cheap and must run again for every
new viewpoint.

Each example is split at that seam: `build_scene()` returns a
`kurven.scene.Scene` — everything a plate is before anyone decides how to look
at it — and `render_plate(scene, projection)` draws it. `main()` is their
composition, so the plates are unchanged.

A **`.kurven` bundle** is a `Scene`, serialized: a directory holding a typed
`manifest.json` and `.npy` arrays.

```bash
python -m kurven.export recip    -o recip.kurven --res 1600
python -m kurven.export elliptic -o elliptic.kurven --res 2000
python -m kurven.export zeta     -o zeta.kurven
python -m kurven.export gamma    -o gamma.kurven --no-adaptive --res 4000
python -m kurven.export recip    -o recip.kurven --derived   # describe, don't dump
python -m kurven.export function -o mine.kurven  --derived \
    --expression "zeta(z)" --res 800                        # or a landscape
```

`--derived` writes descriptions instead of arrays wherever it can. The walls
become a perimeter and a contour layer becomes the levels it is a contour of, so
the consumer regenerates both from `height.npy` and `phase.npy`. recip's bundle
then contains no contour vertices at all and elliptic's drops from 115 MB to
63 MB. Not every layer can be described — elliptic's phase contours are trimmed
by a rule written for that one plate, and those stay dumped, which the schema
says rather than hides.

Describing a layer is also what makes it editable: a bundle exported with
`--derived` gets a levels slider per contour layer in the app, a spacing slider
per hatching, and a truncation it can move; one exported without gets none of
them, because there is no question in it to ask again. A landscape's bundle is
described all the way through — no vertices at all, a manifest and two grids —
which is why it is a few hundred kilobytes and why everything about it except
the samples is a control.

Bundle arrays are in world order — `x = real`, `y = imag`, `z = |f|` — which is
*not* the `(imag, real, z)` column order the library carries internally. The
exchange happens in one function (`kurven.bundle.swap_to_world`) and the
convention is written into the manifest, because forgetting it is the single
most common bug in this codebase's history.

`KurvenSwift/` reads bundles and does the camera-dependent half in Swift and
Metal — realtime navigation means every camera-dependent stage runs per frame.
It is a pure SwiftPM package with no third-party dependencies and no Xcode
requirement: shaders compile at runtime from a string, and the test suite is an
executable rather than a `.testTarget` (Command Line Tools ships
`Testing.framework` without a `.swiftmodule`).

```bash
swift build -c release --package-path KurvenSwift
KurvenSwift/.build/release/kurven-cli bake recip.kurven -o recip.svg
KurvenSwift/.build/release/kurven-cli inspect recip.kurven
KurvenSwift/.build/release/kurven-cli bench recip.kurven   # per-frame depth cost
```

Build release for anything larger than a smoke test: the readback and the clip
are tight scalar loops, and unoptimized Swift bounds-checks every element of a
several-hundred-megabyte buffer.

GPU resources are keyed on `Scene.content`, which survives a camera change, so
navigation costs one uniform upload plus the depth pass — 1.6 ms for recip and
5.9 ms for zeta at 1024², against 33 ms of redundant heightfield upload per
frame if they were rebuilt. `kurven-cli bench` measures it.

A GPU depth test *is* hidden-line removal, so the realtime preview and the exact
bake are the same computation at two resolutions. The preview draws the depth
pass, then tests each line fragment against it; the bake reads the same depth
back and clips line vertices against it with exactly the semantics of
`outline.clip_hidden_lines`. The only difference is per-fragment versus
per-vertex, which can disagree on runs shorter than a pixel and nowhere else.

### The app

```bash
scripts/bundle-app.sh                    # assembles build/Kurven.app
open -a build/Kurven.app recip.kurven    # or double-click the bundle
swift run -c release KurvenApp --open ../recip.kurven    # from a terminal
```

`--open` is an alias for the bare path, and it matters in one place: launched
from a *non-interactive* shell — a CI job, an agent's subprocess — a bare file
argument makes AppKit treat the launch as "opened with a file", and SwiftUI's
`WindowGroup` then declines to create its default window, so the app comes up
with nothing on screen. A dash-prefixed argument is not read that way. From a
terminal the bare path is fine, and the Finder and `open -a` were never
affected, because they deliver the file through `application(_:open:)` rather
than through the command line.

Left-drag orbits, shift-drag pans, scroll zooms toward the cursor, double-click
re-targets the turn onto the point you clicked, `f` fits, `1`/`2`/`3` switch
between the plate, a shaded surface, and the raw depth buffer. The inspector
carries the camera as numbers, the plate presets, per-layer visibility and
levels, the hidden-line margin, and a bake panel.

⌘N starts a landscape rather than opening one, and what it opens is a picker:
the catalog as pictures, each cell the landscape it names, drawn small by the
same renderer that will draw it full size. A name cannot answer what ψ looks
like against ζ, which is the reason these plates were drawn in 1909 rather than
tabulated. They cost about 30 ms each — 220 samples, hatching coarsened to
about twenty strokes an edge, since the spacing that is right for a
four-thousand-pixel bake is solid black at 200 points across — and they are
cached under `~/Library/Caches/world.kurven`, keyed by everything that would
change them. `Kurven --thumbnails` draws them all (0.4 s for fourteen) so the
first look is instant. With nothing open, that picker *is* the window.

The sidebar's top section is the function (a catalog dropdown, `Browse…` for
the same gallery, and a field that takes any expression the language parses),
the window as two thumbs per axis, the resolution, and the truncation
— a height, or a staircase of bands in Re with a row per band. A layer that is
*described* carries its own control underneath it: a level count for a contour
family, a spacing for a hatching. Dragging a window slider samples at a draft
resolution and at the full one when you let go; dragging the cap does not
sample at all (see *[Two speeds of edit](#two-speeds-of-edit)*). ⌘S writes what
is on screen as a `.kurven` bundle, manifest and grids, from this side.

Navigation is a pure function: input handling produces `Gesture` values and
`Navigator.applying` folds them, so orbit, pan, zoom and re-target are tested
without a window (`swift run kurven-test`). The app is scriptable for the same
reason it is testable:

```bash
Kurven.app/Contents/MacOS/Kurven recip.kurven --screenshot out.png
Kurven.app/Contents/MacOS/Kurven recip.kurven --bake out.svg --resolution 4000
Kurven.app/Contents/MacOS/Kurven --landscape zeta --resolution 800 \
    --cap 4 --save zeta.kurven --screenshot zeta.png
Kurven.app/Contents/MacOS/Kurven --thumbnails      # fill the picker's cache
```

The third drives the function picker the way a pair of hands would — sample a
catalog preset, move the cap, keep the result — through the same `Document` the
window uses, so "the picker works" is a file to compare rather than a thing to
click.

The second is how "the app bakes what the CLI bakes" is checked — it is the
same `Scene` value through the same function, and the two SVGs are byte-identical.

Frame times, full preview at 3200² (`kurven-cli bench`): recip 2.2 ms, elliptic
6.8 ms, zeta 12.1 ms. The cost is vertex-bound rather than fill-bound, so it
barely moves between 1600² and 3200².

### Testing across the two lanes

```bash
scripts/check.sh            # everything, both lanes
scripts/check.sh --quick    # skip the end-to-end comparisons
```


Correctness is anchored on the Python pipeline as oracle. `tests/make_fixtures.py`
writes `tests/fixtures/`; both lanes read the same files.

Or a piece at a time:

```bash
python tests/make_fixtures.py          # regenerate the oracle
python tests/check_bundle.py           # python lane: schema, CSR, camera, clip
python tests/check_expr.py             # the expression language and its numerics
swift run --package-path KurvenSwift kurven-test     # swift lane, same fixtures
python tests/compare_bake.py recip     # end to end: swift bake vs python plate
python tests/compare_bake.py recip --derived   # and derived vs dumped
python tests/compare_preview.py recip  # and the preview, as pixels
python tests/compare_bake.py function --cpu    # a landscape nobody wrote a plate for
```

The cheapest test is the sharpest: a fixture manifest decoded by Swift and
re-encoded must come back byte for byte. That is the only check that the two
schema definitions agree, and it is why the Swift mirror needs no codegen.

The hatching is held to the same standard one level down. `tests/fixtures/hatch`
is every one of the four kinds derived by `kurven/hatch.py` from one grid, one
perimeter and two kinds of cap; the Swift lane derives them again from the same
description and compares vertex for vertex, and they are bit-identical (the
rim, being a marching-squares loop with no first vertex, is compared as its set
of segments). One level up, `compare_bake.py function --cpu` draws a generated
landscape both ways: against Python's CPU rasterizer — the definition, where
moderngl samples half a pixel off its own lattice — every layer agrees exactly,
0.0% ink difference and a Hausdorff distance of zero. `--derived` then measures
what is *meant* to differ: the wall crests, because Python evaluates f where the
consumer interpolates the grid (0.5% of the ink), and the contours on
near-vertical pole flanks, where a hair of depth decides visibility.

Two caveats worth knowing before blaming a change for them:

- **zeta's two end-to-end steps need a file this repository does not ship.**
  `examples/zeta.py` loads a precomputed ζ grid from a path in the author's
  notes; without it, `bake/preview vs plate: zeta` fail with `FileNotFoundError`.
  (A ζ *landscape* needs nothing: `kurven.expr` samples it directly.)
- a generated landscape full of pole spires is the worst case for comparing two
  rasterizers, which is why its end-to-end step uses `--cpu`. Against moderngl,
  21% of depth pixels differ on gamma's spires while every contour stays within
  0.03 of the plate's diagonal — the curves agree, the tie-breaking does not.

## References

- Jahnke, E. & Emde, F. (1909). *Tafeln höherer Funktionen*. Teubner.
- Needham, T. (1997). *Visual Complex Analysis*. Oxford University Press.
